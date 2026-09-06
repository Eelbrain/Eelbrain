# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from types import SimpleNamespace
from unittest.mock import patch

import mne
import pytest

from eelbrain._exceptions import ConfigurationError
from eelbrain._experiment.preprocessing import RawMaxwell, RawSource
from eelbrain.testing import requires_mne_head_pos, requires_mne_testing_data


def test_raw_source_rename_channels():
    "rename_channels renames the montage and builtin adjacency, not the data"
    rename = {'A1': 'Fp1', 'A2': 'Fz', 'A3': 'Cz'}
    raw = RawSource(montage='biosemi16', rename_channels=rename, adjacency='biosemi16')
    # Montage uses data names
    for data_name, montage_name in rename.items():
        assert data_name in raw.montage.ch_names
        assert montage_name not in raw.montage.ch_names
    # Builtin adjacency is resolved to an edge list with data names
    assert isinstance(raw.adjacency, list)
    adjacency_names = {name for pair in raw.adjacency for name in pair}
    assert 'A1' in adjacency_names
    assert 'Fp1' not in adjacency_names
    # Fp2 is not renamed and keeps its montage name
    assert 'Fp2' in adjacency_names
    # Renamed and original adjacency describe the same graph
    _, ch_names = mne.channels.read_ch_adjacency('biosemi16')
    reverse = {data_name: montage_name for data_name, montage_name in rename.items()}
    original = RawSource(montage='biosemi16', adjacency='biosemi16')
    assert original.adjacency == 'biosemi16'
    renamed_back = sorted(tuple(sorted((reverse.get(a, a), reverse.get(b, b)))) for a, b in raw.adjacency)
    coo = mne.channels.read_ch_adjacency('biosemi16')[0].tocoo()
    expected = sorted({tuple(sorted((ch_names[min(i, j)], ch_names[max(i, j)]))) for i, j in zip(coo.row, coo.col) if i != j})
    assert renamed_back == expected

    # rename_channels requires a montage
    with pytest.raises(ConfigurationError):
        RawSource(rename_channels=rename)
    # rename_channels values need to be in the montage
    with pytest.raises(ConfigurationError):
        RawSource(montage='biosemi16', rename_channels={'A1': 'NoSuchChannel'})


@requires_mne_head_pos
def test_maxwell_head_pos_semantic_dict():
    "head_pos is omitted from the fingerprint when unset, so caches predating it stay valid"
    maxwell = RawMaxwell('raw', st_duration=10.)
    assert maxwell.head_pos is False
    assert maxwell._as_dict() == {
        'type': 'RawMaxwell',
        'source': 'raw',
        'bad_condition': 'error',
        'kwargs': {'st_duration': 10.},
    }

    movecomp = RawMaxwell('raw', st_duration=10., head_pos=True)
    assert movecomp._as_dict()['head_pos'] is True
    assert movecomp != maxwell
    # head_pos configures the pipe, it is never forwarded to MNE
    assert movecomp.kwargs == {'st_duration': 10.}
    with pytest.raises(TypeError):
        RawMaxwell('raw', head_position=True)


def test_maxwell_head_pos_st_only():
    "Movement compensation happens in the SSS reconstruction, which st_only skips"
    with pytest.raises(ConfigurationError, match='st_only'):
        RawMaxwell('raw', st_duration=10., st_only=True, head_pos=True)


@requires_mne_head_pos
@requires_mne_testing_data
def test_maxwell_head_pos_filter_chpi():
    "cHPI signals and line noise are removed before Maxwell filtering with head_pos=True; the empty room gets the same line noise treatment"
    sss_dir = mne.datasets.testing.data_path(download=False) / 'SSS'
    raw = mne.io.read_raw_fif(sss_dir / 'test_move_anon_raw.fif', allow_maxshield='yes', verbose=False).crop(0, 2).load_data()
    head_pos = mne.chpi.read_head_pos(sss_dir / 'test_move_anon_raw.pos')
    path = SimpleNamespace(fpath='test_move_anon_raw.fif', find_empty_room=lambda: SimpleNamespace(fpath='test_move_anon_raw.fif'))
    pipe = RawMaxwell('raw', head_pos=True)
    heavy = {'find_bad_channels_maxwell': lambda raw, **kwargs: ([], []), 'maxwell_filter': lambda raw, **kwargs: raw, 'maxwell_filter_prepare_emptyroom': lambda raw_er, **kwargs: raw_er}
    with patch.multiple(mne.preprocessing, **heavy), patch.object(mne.chpi, 'filter_chpi') as filter_chpi:
        pipe._make(raw, path=path, head_pos=head_pos)
        filter_chpi.assert_called_once()
        assert filter_chpi.call_args.args[0] is raw
        assert filter_chpi.call_args.kwargs['allow_line_only'] is False

        # a single static sample means no compensation, but the line noise treatment stays the same across recordings
        filter_chpi.reset_mock()
        pipe._make(raw, path=path, head_pos=head_pos[:1])
        filter_chpi.assert_called_once()

        # the empty room only gets line noise removed, whether or not its own header lists the coil frequencies
        filter_chpi.reset_mock()
        raw_er = raw.copy()
        with raw_er.info._unlock():
            raw_er.info['hpi_meas'] = []
        pipe._make(raw_er, path=path, noise=True, reference=raw)
        filter_chpi.assert_called_once()
        assert filter_chpi.call_args.args[0] is raw_er
        assert filter_chpi.call_args.kwargs['allow_line_only'] is True

        # recordings without coil frequencies (CTF, KIT) cannot use filter_chpi, and neither can their empty room
        filter_chpi.reset_mock()
        with raw.info._unlock():
            raw.info['hpi_meas'] = []
        pipe._make(raw, path=path, head_pos=head_pos)
        pipe._make(raw_er, path=path, noise=True, reference=raw)
        filter_chpi.assert_not_called()
