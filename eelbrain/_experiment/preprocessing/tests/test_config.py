# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import pytest

from eelbrain._exceptions import ConfigurationError, DataError
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
    # filter_chpi follows head_pos
    assert movecomp.filter_chpi is True
    assert RawMaxwell('raw', st_duration=10., filter_chpi=True)._as_dict() == {**maxwell._as_dict(), 'filter_chpi': True}
    # h_freq at its default is omitted, too; it configures bad channel detection, not maxwell_filter
    assert maxwell.h_freq == 40.
    assert RawMaxwell('raw', st_duration=10., h_freq=40)._as_dict() == maxwell._as_dict()
    assert RawMaxwell('raw', st_duration=10., h_freq=30.)._as_dict() == {**maxwell._as_dict(), 'h_freq': 30.}
    assert RawMaxwell('raw', st_duration=10., h_freq=None)._as_dict() == {**maxwell._as_dict(), 'h_freq': None}
    # without the low-pass filter, cHPI signals would enter the bad channel detection
    with pytest.raises(ConfigurationError, match='h_freq=None'):
        RawMaxwell('raw', st_duration=10., h_freq=None, head_pos=True)
    with pytest.raises(ConfigurationError, match='h_freq=None'):
        RawMaxwell('raw', st_duration=10., h_freq=None, filter_chpi=True)


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
    heavy = {'find_bad_channels_maxwell': MagicMock(return_value=([], [])), 'maxwell_filter': lambda raw, **kwargs: raw, 'maxwell_filter_prepare_emptyroom': lambda raw_er, **kwargs: raw_er}
    with patch.multiple(mne.preprocessing, **heavy), patch.object(mne.chpi, 'filter_chpi') as filter_chpi:
        pipe._make(raw, path=path, head_pos=head_pos)
        filter_chpi.assert_called_once()
        assert filter_chpi.call_args.args[0] is raw
        assert filter_chpi.call_args.kwargs['allow_line_only'] is False
        # bad channels are detected on low-passed data, before filter_chpi runs
        assert heavy['find_bad_channels_maxwell'].call_args.kwargs['h_freq'] == 40.

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

        # line noise removal needs the power line frequency
        raw_er.info['line_freq'] = None
        with pytest.raises(DataError, match='PowerLineFrequency'):
            pipe._make(raw_er, path=path, noise=True, reference=raw)

        # filter_chpi=True removes cHPI signals without movement compensation
        filter_chpi.reset_mock()
        raw_er.info['line_freq'] = raw.info['line_freq']
        chpi_only = RawMaxwell('raw', st_duration=10., st_only=True, filter_chpi=True)
        chpi_only._make(raw, path=path)
        chpi_only._make(raw_er, path=path, noise=True, reference=raw)
        assert filter_chpi.call_count == 2
        assert [call.kwargs['allow_line_only'] for call in filter_chpi.call_args_list] == [False, True]

        # filter_chpi=False keeps the cHPI signals even with movement compensation
        filter_chpi.reset_mock()
        RawMaxwell('raw', head_pos=True, filter_chpi=False)._make(raw, path=path, head_pos=head_pos)
        filter_chpi.assert_not_called()

        # Neuromag headers list the coil frequencies even when the coils were off; nothing to remove then
        filter_chpi.reset_mock()
        with patch('eelbrain._experiment.preprocessing.config.find_chpi', return_value=None):
            pipe._make(raw, path=path, head_pos=head_pos[:1])
            pipe._make(raw_er, path=path, noise=True, reference=raw)
        filter_chpi.assert_not_called()

        # recordings without coil frequencies (CTF, KIT) cannot use filter_chpi, and neither can their empty room
        filter_chpi.reset_mock()
        with raw.info._unlock():
            raw.info['hpi_meas'] = []
        pipe._make(raw, path=path, head_pos=head_pos)
        pipe._make(raw_er, path=path, noise=True, reference=raw)
        filter_chpi.assert_not_called()


@requires_mne_head_pos
@requires_mne_testing_data
def test_maxwell_movement_annotations():
    "Segments with excessive movement are annotated on the compensated data"
    sss_dir = mne.datasets.testing.data_path(download=False) / 'SSS'
    raw = mne.io.read_raw_fif(sss_dir / 'test_move_anon_raw.fif', allow_maxshield='yes', verbose=False).crop(0, 2).load_data()
    head_pos = mne.chpi.read_head_pos(sss_dir / 'test_move_anon_raw.pos')
    path = SimpleNamespace(fpath='test_move_anon_raw.fif', find_empty_room=lambda: SimpleNamespace(fpath='test_move_anon_raw.fif'))
    heavy = {'find_bad_channels_maxwell': lambda raw, **kwargs: ([], []), 'maxwell_filter': lambda raw, **kwargs: raw}
    movement = mne.Annotations([raw.first_time + 0.5], [0.2], ['BAD_mov_dist'], orig_time=raw.annotations.orig_time)
    with patch.multiple(mne.preprocessing, **heavy), patch.object(mne.chpi, 'filter_chpi'), patch.object(mne.preprocessing, 'annotate_movement', return_value=(movement, [])) as annotate_movement:
        # without limits, nothing is annotated
        raw_sss = RawMaxwell('raw', head_pos=True)._make(raw, path=path, head_pos=head_pos)
        annotate_movement.assert_not_called()
        assert 'BAD_mov_dist' not in raw_sss.annotations.description

        pipe = RawMaxwell('raw', head_pos=True, rotation_velocity_limit=30., mean_distance_limit=0.01)
        raw_sss = pipe._make(raw, path=path, head_pos=head_pos)
        annotate_movement.assert_called_once()
        assert annotate_movement.call_args.args[1] is head_pos
        assert annotate_movement.call_args.kwargs['rotation_velocity_limit'] == 30.
        assert annotate_movement.call_args.kwargs['translation_velocity_limit'] is None
        assert annotate_movement.call_args.kwargs['mean_distance_limit'] == 0.01
        assert annotate_movement.call_args.kwargs['use_dev_head_trans'] == 'info'
        assert 'BAD_mov_dist' in raw_sss.annotations.description

        # a recording without usable cHPI is not compensated and gets no movement annotations
        annotate_movement.reset_mock()
        pipe._make(raw, path=path, head_pos=head_pos[:1])
        annotate_movement.assert_not_called()
