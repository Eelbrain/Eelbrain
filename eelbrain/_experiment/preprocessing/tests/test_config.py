# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import NDVar, UTS, filter_data
from eelbrain._exceptions import ConfigurationError, DataError
from eelbrain._experiment.preprocessing import RawFilter, RawMaxwell, RawSource
from eelbrain._experiment.preprocessing.config import _MNELogFilter
from eelbrain._experiment.trf import filter_predictor
from eelbrain.testing import requires_mne_testing_data


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
    # Montage names from before MNE 1.13 are rejected (stored fingerprints with them are migrated, see RawSourceInput.normalize_stored_fingerprint)
    with pytest.raises(ConfigurationError, match="'colin27_1020'"):
        RawSource(montage='standard_1020')


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
    assert RawMaxwell('raw', st_duration=10., flat_duration=0.03)._as_dict() == maxwell._as_dict()
    assert RawMaxwell('raw', st_duration=10., flat_duration=None)._as_dict() == {**maxwell._as_dict(), 'flat_duration': None}
    # without the low-pass filter, cHPI signals would enter the bad channel detection
    with pytest.raises(ConfigurationError, match='h_freq=None'):
        RawMaxwell('raw', st_duration=10., h_freq=None, head_pos=True)
    with pytest.raises(ConfigurationError, match='h_freq=None'):
        RawMaxwell('raw', st_duration=10., h_freq=None, filter_chpi=True)


def test_maxwell_head_pos_st_only():
    "Movement compensation happens in the SSS reconstruction, which st_only skips; the head positions still shape the temporal projection basis"
    with pytest.warns(UserWarning, match='st_only'):
        pipe = RawMaxwell('raw', st_duration=10., st_only=True, head_pos=True)
    assert pipe.head_pos is True
    assert pipe.kwargs['st_only'] is True


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


def test_maxwell_log_filter():
    "MNE's messages about the bad channel detection are forwarded to the pipeline log; a detection without intervals is a warning"
    log = MagicMock()
    mne_log = _MNELogFilter(log, 'raw')

    def filter_message(message, level=logging.INFO):
        return mne_log.filter(logging.LogRecord('mne', level, '', 0, message, None, None))

    assert not filter_message("    Scanning for bad channels in 3 intervals (5.0 s) ...")
    log.info.assert_called_once_with("Raw %s: %s", 'raw', "Scanning for bad channels in 3 intervals (5.0 s) ...")
    log.warning.assert_not_called()
    assert not filter_message("    Scanning for bad channels in 0 intervals (5.0 s) ...")
    log.warning.assert_called_once()
    assert "no stretch without BAD_ annotations" in log.warning.call_args.args[0]
    assert mne_log.messages == []  # not relevant for the ill-conditioned basis error
    assert not filter_message("    Using origin -1.5, 3.2, 44.9 mm in the head frame")
    assert mne_log.messages == ["Using origin -1.5, 3.2, 44.9 mm in the head frame"]
    # per-chunk progress is dropped, warnings pass through
    assert not filter_message("        Processing 4 data chunks")
    assert filter_message("All-flat segment detected", logging.WARNING)
    assert log.info.call_count == 2
    # excluded channels are tracked per interval; an unexpected format is kept as is
    assert not filter_message("        Interval   2:    5.000 -    9.999")
    assert not filter_message("            Bad:       MEG 046 12.3", logging.DEBUG)
    assert not filter_message("            Bad:       MEG0113", logging.DEBUG)
    assert mne_log.interval == "Interval 2: 5.000 - 9.999"
    assert mne_log.excluded == ["MEG 046 (z=12.3)", "MEG0113"]
    assert not filter_message("        Interval   3:   10.000 -   14.999")
    assert mne_log.excluded == []


@requires_mne_testing_data
def test_maxwell_flat_segments():
    "Segments in which MEG channels are flat are annotated as BAD_flat and skipped by bad channel detection and Maxwell filtering"
    sss_dir = mne.datasets.testing.data_path(download=False) / 'SSS'
    raw = mne.io.read_raw_fif(sss_dir / 'test_move_anon_raw.fif', allow_maxshield='yes', verbose=False).crop(0, 2).load_data()
    path = SimpleNamespace(fpath='test_move_anon_raw.fif', find_empty_room=lambda: SimpleNamespace(fpath='test_move_anon_raw.fif'))
    sfreq = raw.info['sfreq']
    # one channel saturates for 100 ms, another is dead
    clipped, dead = mne.pick_types(raw.info, meg=True)[[0, 1]]
    raw._data[clipped, int(0.5 * sfreq):int(0.6 * sfreq)] = raw._data[clipped, int(0.5 * sfreq)]
    raw._data[dead] = 0.
    heavy = {'find_bad_channels_maxwell': MagicMock(return_value=([], [])), 'maxwell_filter': MagicMock(side_effect=lambda raw, **kwargs: raw)}
    with patch.multiple(mne.preprocessing, **heavy):
        raw_sss = RawMaxwell('raw')._make(raw.copy(), path=path)
        detector_raw = heavy['find_bad_channels_maxwell'].call_args.args[0]
        assert detector_raw.info['bads'] == [raw.ch_names[dead]]
        flat = detector_raw.annotations[detector_raw.annotations.description == 'BAD_flat']
        assert len(flat) == 1
        assert flat.onset[0] - raw.first_time == pytest.approx(0.5, abs=2 / sfreq)
        assert flat.duration[0] == pytest.approx(0.1, abs=2 / sfreq)
        assert heavy['find_bad_channels_maxwell'].call_args.kwargs['skip_by_annotation'] == ('edge', 'bad_acq_skip', 'bad_flat')
        assert heavy['maxwell_filter'].call_args.kwargs['skip_by_annotation'] == ('edge', 'bad_acq_skip', 'bad_flat')
        # the annotations are stored with the filtered data
        assert 'BAD_flat' in raw_sss.annotations.description

        # a user-specified skip_by_annotation is extended
        RawMaxwell('raw', skip_by_annotation='bad_mov')._make(raw.copy(), path=path)
        assert heavy['maxwell_filter'].call_args.kwargs['skip_by_annotation'] == ('bad_mov', 'bad_flat')

        # flat_duration=None disables the annotation and leaves the dead channel to find_bad_channels_maxwell (mocked here)
        raw_sss = RawMaxwell('raw', flat_duration=None)._make(raw.copy(), path=path)
        detector_raw = heavy['find_bad_channels_maxwell'].call_args.args[0]
        assert detector_raw.info['bads'] == []
        assert 'BAD_flat' not in raw_sss.annotations.description
        assert 'skip_by_annotation' not in heavy['find_bad_channels_maxwell'].call_args.kwargs
        assert 'skip_by_annotation' not in heavy['maxwell_filter'].call_args.kwargs

        # a stretch shorter than flat_duration is not annotated
        raw_sss = RawMaxwell('raw', flat_duration=0.2)._make(raw.copy(), path=path)
        assert 'BAD_flat' not in raw_sss.annotations.description


@requires_mne_testing_data
def test_maxwell_ill_conditioned_context():
    "The error for an ill-conditioned SSS basis reports where it occurred"
    sss_dir = mne.datasets.testing.data_path(download=False) / 'SSS'
    raw = mne.io.read_raw_fif(sss_dir / 'test_move_anon_raw.fif', allow_maxshield='yes', verbose=False).crop(0, 2).load_data()
    path = SimpleNamespace(fpath='test_move_anon_raw.fif', find_empty_room=lambda: SimpleNamespace(fpath='test_move_anon_raw.fif'))
    error = RuntimeError("Matrix is badly conditioned: 5494 >= 1000")
    mne_logger = logging.getLogger('mne')

    def detector_log(*messages):
        # MNE's messages during bad channel detection; the channels excluded per iteration are logged at DEBUG level
        def detector(raw, **kwargs):
            mne_logger.info("    Bad MEG channels being reconstructed: ['MEG0111']")
            for message in messages:
                mne_logger.log(logging.DEBUG if message.startswith('Bad:') else logging.INFO, '            ' + message)
            raise error
        return detector

    heavy = {'find_bad_channels_maxwell': MagicMock(return_value=([], [])), 'maxwell_filter': MagicMock(side_effect=error)}
    with patch.multiple(mne.preprocessing, **heavy), mne.utils.use_log_level('debug'):
        # the filter proper
        with pytest.raises(DataError, match="in the Maxwell filter, after bad channel detection.*digitization") as exc_info:
            RawMaxwell('raw', flat_duration=None)._make(raw.copy(), path=path)
        assert 'Bad MEG channels' not in str(exc_info.value)

        # before any interval was processed
        heavy['find_bad_channels_maxwell'].side_effect = detector_log()
        with pytest.raises(DataError, match=r"during bad channel detection, while preparing the SSS basis .* before any interval was processed\. This usually indicates"):
            RawMaxwell('raw', flat_duration=None)._make(raw.copy(), path=path)

        # the first interval, with the initial sensor geometry
        heavy['find_bad_channels_maxwell'].side_effect = detector_log("Interval   1:    0.000 -    4.999")
        with pytest.raises(DataError, match=r"\(Matrix is badly conditioned: 5494 >= 1000; Bad MEG channels being reconstructed: \['MEG0111'\]\)\. The error occurred during bad channel detection in Interval 1: 0.000 - 4.999, with the initial sensor geometry\. This usually indicates an inaccurate head digitization"):
            RawMaxwell('raw', flat_duration=None)._make(raw.copy(), path=path)

        # a later interval, after excluding channels from it
        heavy['find_bad_channels_maxwell'].side_effect = detector_log("Interval   1:    0.000 -    4.999", "Flat ( 1): MEG0112", "Bad:       MEG0113 12.3", "Interval   2:    5.000 -    9.999", "Bad:       MEG0122 9.0", "Bad:       MEG0123 7.1")
        with pytest.raises(DataError, match=r"in Interval 2: 5.000 - 9.999, after excluding 2 noisy channels from this interval \(MEG0122 \(z=9.0\), MEG0123 \(z=7.1\)\); Flat \(1\): MEG0112\. Excluding many channels .* clipping.*Set flat_duration to annotate.* Otherwise, this usually indicates an inaccurate head digitization"):
            RawMaxwell('raw', flat_duration=None)._make(raw.copy(), path=path)
        with pytest.raises(DataError, match=r"Excluding many channels .* clipping.*flat are annotated and skipped automatically"):
            RawMaxwell('raw')._make(raw.copy(), path=path)

        # a misplaced sensor is reported even after channels were excluded from the interval
        raw_misplaced = raw.copy()
        raw_misplaced.info['chs'][mne.pick_types(raw.info, meg=True)[0]]['loc'][:3] = 0.
        with pytest.raises(DataError, match=r"after excluding 2 noisy channels .* 1 MEG sensor located inside the head sphere .*: MEG0113\. Mark") as exc_info:
            RawMaxwell('raw', flat_duration=None)._make(raw_misplaced, path=path)
        assert 'Excluding many channels' not in str(exc_info.value)


def test_raw_filter_predictor_nyquist():
    "A low-pass at or above the predictor's Nyquist frequency is skipped; the high-pass is kept"
    x = NDVar(np.random.default_rng(0).normal(size=500), UTS(0, 0.02, 500), {'sampling': 'continuous'}, 'x')  # 50 Hz
    raw = {'raw': RawSource(), '1-40': RawFilter('raw', 1, 40), '25': RawFilter('raw', None, 25), '20': RawFilter('raw', None, 20)}
    # high-pass only (the 40 Hz low-pass is above the 25 Hz Nyquist frequency)
    assert_array_equal(filter_predictor(x, raw, '1-40', True).x, filter_data(x, 1, None, pad='edge').x)
    # a low-pass at Nyquist is skipped too, leaving nothing to apply
    assert filter_predictor(x, raw, '25', True) is x
    # a low-pass below Nyquist is applied
    assert_array_equal(filter_predictor(x, raw, '20', True).x, filter_data(x, None, 20, pad='edge').x)
    # filter_x=False still skips everything
    assert filter_predictor(x, raw, '1-40', False) is x
    # a high-pass above Nyquist would remove everything
    raw['30-40'] = RawFilter('raw', 30, 40)
    with pytest.raises(ValueError):
        filter_predictor(x, raw, '30-40', True)
