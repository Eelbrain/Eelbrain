# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import logging
from types import SimpleNamespace
from unittest.mock import patch
from warnings import catch_warnings, filterwarnings

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath

from eelbrain._experiment.preprocessing.config import CHPI_MIN_ACTIVE_FRACTION
from eelbrain._experiment.preprocessing.nodes import CanonicalHeadPositionDerivative, RawHeadPositionDerivative, RawSourceInput, find_chpi
from eelbrain.testing import requires_mne_testing_data


def test_read_raw_applies_bids_channels(tmp_path):
    "RawSourceInput._apply_bids_channels applies channels.tsv channel metadata"
    bids_path = BIDSPath(
        root=tmp_path,
        subject='01',
        task='test',
        datatype='meg',
        suffix='meg',
        extension='.fif',
    )
    bids_path.fpath.parent.mkdir(parents=True)
    info = mne.create_info(['MEG 001', 'EOG 001'], 100, ['mag', 'misc'])
    data = np.zeros((2, 100))
    raw = mne.io.RawArray(data, info, verbose='error')
    raw.save(bids_path.fpath, overwrite=True, verbose='error')

    channels_path = bids_path.copy().update(suffix='channels', extension='.tsv').fpath
    channels = pd.DataFrame({
        'name': ['MEG 001', 'EOG 001'],
        'type': ['MEGMAG', 'EOG'],
        'units': ['T', 'V'],
        'status': ['good', 'bad'],
    })
    channels.to_csv(channels_path, sep='\t', index=False)

    raw_read = RawSourceInput._read_raw(bids_path.fpath, preload=False)
    RawSourceInput._apply_bids_channels(bids_path, raw_read)

    assert raw_read.get_channel_types(picks=['EOG 001']) == ['eog']
    assert raw_read.info['bads'] == []

    # the power line frequency comes from the JSON sidecar when the header lacks it
    assert raw_read.info['line_freq'] is None
    sidecar_path = bids_path.copy().update(extension='.json').fpath
    sidecar_path.write_text('{"PowerLineFrequency": "n/a"}')
    RawSourceInput._apply_bids_line_freq(bids_path, raw_read)
    assert raw_read.info['line_freq'] is None
    sidecar_path.write_text('{"PowerLineFrequency": 50}')
    RawSourceInput._apply_bids_line_freq(bids_path, raw_read)
    assert raw_read.info['line_freq'] == 50.
    # the header takes precedence
    raw_read.info['line_freq'] = 60.
    RawSourceInput._apply_bids_line_freq(bids_path, raw_read)
    assert raw_read.info['line_freq'] == 60.


def generate_head_positions(n: int) -> np.ndarray:
    """Synthetic (n, 10) head positions in MaxFilter format"""
    out = np.empty((n, 10))
    out[:, 0] = 42.0 + np.arange(n) * 0.01  # t
    out[:, 1:4] = np.column_stack([np.linspace(.010, .014, n), np.linspace(-.005, -.001, n), np.full(n, .002)])  # q1, q2, q3
    out[:, 4:7] = np.column_stack([np.linspace(.001, .003, n), np.full(n, -.002), np.linspace(.050, .053, n)])  # tx, ty, tz
    out[:, 7:] = np.column_stack([np.full(n, .99), np.full(n, .001), np.full(n, .01)])  # gof, err, v
    return out


@pytest.mark.parametrize('n', [1, 5])
def test_head_position_roundtrip(tmp_path, n):
    """RawHeadPositionDerivative save/load preserves the MaxFilter (n, 10) format"""
    node = RawHeadPositionDerivative('raw@raw')
    path = tmp_path / 'test.pos'
    positions = generate_head_positions(n)
    node.save(None, path, positions)
    loaded = node.load(None, path)
    assert loaded.shape == (n, 10)
    assert np.allclose(loaded, positions, atol=1e-5)
    # _check_pos requires strictly ascending times, which the 3-decimal .pos time format must not collapse
    assert (np.diff(loaded[:, 0]) > 0).all()


def test_head_position_none_roundtrip(tmp_path):
    """RawHeadPositionDerivative encodes None as an empty file"""
    node = RawHeadPositionDerivative('raw@raw')
    path = tmp_path / 'test.pos'
    node.save(None, path, None)
    assert path.stat().st_size == 0
    assert node.load(None, path) is None
    # a rebuild writes over the previous artifact in place, so None has to replace earlier positions
    node.save(None, path, generate_head_positions(5))
    node.save(None, path, None)
    assert node.load(None, path) is None


def build_canonical_head_position(raws: list, positions: list) -> mne.transforms.Transform | None:
    "Run CanonicalHeadPositionDerivative.build on recordings with head positions"
    artifacts = {}
    for i, (raw, head_pos) in enumerate(zip(raws, positions)):
        artifacts[f'task-{i}'] = head_pos
        artifacts[f'raw:task-{i}'] = raw
    ctx = SimpleNamespace(declared_dependencies=list(artifacts), load=artifacts.__getitem__)
    return CanonicalHeadPositionDerivative('raw', frozenset(), [], []).build(ctx)


def test_canonical_head_position():
    """CanonicalHeadPositionDerivative weights samples by the time they were held, or returns None without movement"""
    info = mne.create_info(['MEG 0111'], 100., 'mag')
    raws = [mne.io.RawArray(np.zeros((1, 1000)), info, first_samp=4200, verbose=False) for _ in range(2)]  # 10 s each, starting at t=42 s
    positions = generate_head_positions(5)  # t = 42.00, 42.01, ..., 42.04
    static = positions[:1]
    assert build_canonical_head_position(raws[:1], [positions]) is None  # a single recording keeps its own dev_head_t
    assert build_canonical_head_position(raws, [static, static]) is None

    # a tracked recording with movement and a static recording
    trans = build_canonical_head_position(raws, [positions, static])
    assert trans['from'] == mne.io.constants.FIFF.FIFFV_COORD_DEVICE
    assert trans['to'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD
    # each sample counts until the next sample or the end of its recording: 4 x 10 ms, then 9.96 s for the last tracked sample and 10 s for the static one
    weights = np.array([.01, .01, .01, .01, 9.96, 10.])
    samples = np.vstack([positions, static])
    assert np.allclose(trans['trans'][:3, 3], (weights / weights.sum()) @ samples[:, 4:7], atol=1e-6)
    # the rotation is a proper rotation, which an element-wise mean of rotation matrices would not be
    rot = trans['trans'][:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3))

    # BAD segments are excluded: with the static recording marked bad, only the tracked recording contributes
    raws[1].set_annotations(mne.Annotations(0., 10., 'BAD_all'))  # onset relative to the start of the data
    trans_bad = build_canonical_head_position(raws, [positions, static])
    trans_tracked = mne.preprocessing.compute_average_dev_head_t(raws[0], positions)
    assert np.allclose(trans_bad['trans'], trans_tracked['trans'])


def build_head_position(raw: mne.io.BaseRaw) -> np.ndarray | None:
    "Run RawHeadPositionDerivative.build on one recording"
    ctx = SimpleNamespace(load=lambda name: raw, state={'subject': 'test'}, registry=SimpleNamespace(log=logging.getLogger('test')))
    return RawHeadPositionDerivative('raw').build(ctx)


@requires_mne_testing_data
def test_find_chpi_active_fraction(caplog):
    """Neuromag recordings count as continuous HPI only if the coils were active for a substantial fraction of the recording"""
    raw = mne.io.read_raw_fif(mne.datasets.testing.data_path(download=False) / 'SSS' / 'test_move_anon_raw.fif', allow_maxshield='yes', verbose=False)
    assert find_chpi(raw) == 'freqs'
    n_times = len(raw.times)
    n_brief = int(n_times * CHPI_MIN_ACTIVE_FRACTION / 2)
    with patch.object(mne.chpi, 'get_active_chpi', return_value=np.zeros(n_times)):
        assert find_chpi(raw) is None
    brief = np.r_[np.full(n_brief, 5), np.zeros(n_times - n_brief)]
    with patch.object(mne.chpi, 'get_active_chpi', return_value=brief), caplog.at_level(logging.INFO):
        assert find_chpi(raw) is None
    assert 'active during only 5%' in caplog.text
    longer = np.r_[np.full(3 * n_brief, 5), np.zeros(n_times - 3 * n_brief)]
    with patch.object(mne.chpi, 'get_active_chpi', return_value=longer):
        assert find_chpi(raw) == 'freqs'


@requires_mne_testing_data
def test_head_position_ctf():
    """Continuous head localization from CTF HLC channels"""
    path = mne.datasets.testing.data_path(download=False) / 'CTF' / 'testdata_ctf_mc.ds'
    raw = mne.io.read_raw_ctf(path, verbose=False)
    assert find_chpi(raw) == 'ctf'
    with catch_warnings():
        filterwarnings('ignore', 'HPI.*is poor', RuntimeWarning)
        head_pos = build_head_position(raw)
    assert head_pos.shape[1] == 10
    assert len(head_pos) > 1
    # within a few mm of the positions computed by the CTF software, shipped with the data set
    reference = mne.chpi.read_head_pos(path.with_suffix('.pos'))
    for column in (4, 5, 6):
        assert np.allclose(head_pos[:, column], np.interp(head_pos[:, 0], reference[:, 0], reference[:, column]), atol=5e-3)


@requires_mne_testing_data
def test_head_position_kit():
    """cHPI from the KIT stim channel; KIT recordings without cHPI fall back to the static transform"""
    kit_dir = mne.datasets.testing.data_path(download=False) / 'KIT'
    raw = mne.io.read_raw_kit(kit_dir / 'MQKIT_125_2sec.con', kit_dir / 'MQKIT_125.mrk', kit_dir / 'MQKIT_125.elp', kit_dir / 'MQKIT_125.hsp', verbose=False)
    assert find_chpi(raw) == 'kit'
    assert build_head_position(raw).shape == (2, 10)

    raw_berlin = mne.io.read_raw_kit(kit_dir / 'data_berlin.con', verbose=False)
    assert find_chpi(raw_berlin) is None
    assert build_head_position(raw_berlin).shape == (1, 10)
