# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for ChannelModel, including long, variable-length epochs."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from eelbrain import Datalist, NDVar, Sensor, UTS
from eelbrain._meeg import ChannelModel


CH_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'T7', 'T8', 'Oz']
SFREQ = 100.


def _sensor():
    return Sensor.from_montage('standard_1020', channels=CH_NAMES)


def _block(n_times, seed, rank=8, noise=0.02):
    # spatially redundant data (rank < n_sensors) so each channel is
    # predictable from the others, plus a little sensor noise
    r = np.random.RandomState(seed)
    n_ch = len(CH_NAMES)
    x = r.randn(n_ch, rank) @ r.randn(rank, n_times)
    x += noise * r.randn(n_ch, n_times)
    return x * 1e-6


def _blocks(lengths=(200, 300, 400)):
    sensor = _sensor()
    return Datalist([NDVar(_block(n, i), (sensor, UTS(0, 1 / SFREQ, n)), 'eeg') for i, n in enumerate(lengths)])


def _inject(block, channel, start, stop, amplitude=100e-6):
    # add a transient on one channel over samples [start, stop)
    x = block.x.copy()
    ci = list(block.sensor.names).index(channel)
    x[ci, start:stop] += amplitude * np.sin(np.linspace(0, 4 * np.pi, stop - start))
    return NDVar(x, block.dims, block.name)


def test_channel_model_list_input():
    "ChannelModel.fit/predict/score accept a list of long epochs"
    blocks = _blocks()
    model = ChannelModel('ols')
    model.fit(blocks)
    assert len(model.estimators_) == len(CH_NAMES)
    assert model.sensor == blocks[0].sensor

    # predict returns one NDVar per epoch with matching time
    pred = model.predict(blocks)
    assert isinstance(pred, Datalist)
    assert len(pred) == len(blocks)
    for p, b in zip(pred, blocks):
        assert p.time == b.time
        # clean, redundant data is well predicted from the other channels
        corr = np.mean([np.corrcoef(p.x[i], b.x[i])[0, 1] for i in range(len(CH_NAMES))])
        assert corr > 0.7

    # score returns one per-sensor NDVar per epoch
    score = model.score(blocks)
    assert isinstance(score, Datalist)
    assert all(s.sensor == blocks[0].sensor for s in score)

    # not fit -> informative error
    with pytest.raises(RuntimeError):
        ChannelModel('ols').predict(blocks)


def test_find_bad_windows():
    "ChannelModel.find_bad_windows isolates a transient bad channel in time"
    blocks = _blocks()
    model = ChannelModel('ols')
    model.fit(blocks)

    # transient on Cz in epoch 1, samples 100:150 (1.00-1.50 s)
    data = list(blocks)
    data[1] = _inject(blocks[1], 'Cz', 100, 150)

    windows = model.find_bad_windows(data, threshold=20e-6, window=0.5, hop=0.25, min_duration=0.05)
    assert isinstance(windows, Datalist)
    # only epoch 1 has a window, only on Cz
    assert windows[0] == []
    assert windows[2] == []
    assert {w.channel for w in windows[1]} == {'Cz'}
    w = windows[1][0]
    # detected window brackets the true artifact (1.0-1.5 s) within window/hop
    assert w.tmin <= 1.0
    assert w.tmax >= 1.5
    assert w.tmin >= 1.0 - 0.5
    assert w.tmax <= 1.5 + 0.5


def test_find_bad_windows_whole_epoch():
    "A channel bad for the whole epoch yields a full-length window"
    blocks = _blocks(lengths=(300,))
    model = ChannelModel('ols')
    model.fit(blocks)

    data = [_inject(blocks[0], 'O1', 0, blocks[0].time.nsamples)]
    windows = model.find_bad_windows(data, threshold=20e-6, window=0.5, hop=0.25)
    assert {w.channel for w in windows[0]} == {'O1'}
    w = next(w for w in windows[0] if w.channel == 'O1')
    assert w.tmin == blocks[0].time.tmin
    assert w.tmax == pytest.approx(blocks[0].time.tstop)


def test_find_bad_windows_min_duration_and_merge():
    "min_duration drops blips; merge_gap coalesces nearby windows"
    sensor = _sensor()
    time = UTS(0, 1 / SFREQ, 400)
    error = np.zeros((len(CH_NAMES), 400))
    ci = CH_NAMES.index('Pz')
    error[ci, 10] = 1.  # 1-sample blip
    error[ci, 100:150] = 1.  # run A
    error[ci, 155:200] = 1.  # run B, 5-sample gap after A

    model = ChannelModel('ols')
    model.sensor = sensor

    # blip dropped (min_duration), runs A and B merged (merge_gap > gap)
    windows = model._windows_from_error(error, time, threshold=0.5, min_duration=0.05, merge_gap=0.1)
    assert len(windows) == 1
    assert windows[0].channel == 'Pz'
    assert windows[0].tmin == pytest.approx(1.0)
    assert windows[0].tmax == pytest.approx(2.0)

    # with a small merge_gap the two runs stay separate
    windows = model._windows_from_error(error, time, threshold=0.5, min_duration=0.05, merge_gap=0.01)
    assert len(windows) == 2
