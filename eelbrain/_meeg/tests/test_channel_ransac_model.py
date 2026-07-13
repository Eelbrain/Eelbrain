# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for ChannelRANSACModel, and for API parity with ChannelModel."""
import inspect

import numpy as np
import pytest

from eelbrain import Datalist, NDVar, Sensor, UTS
from eelbrain._meeg import ChannelModel
from eelbrain._meeg._channel_model import ChannelRANSACModel


CH_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'T7', 'T8', 'Oz']
SFREQ = 100.
RANSAC_KWARGS = dict(n_resamples=30, subset_size=0.3, random_seed=1)


def _sensor():
    return Sensor.from_montage('standard_1020', channels=CH_NAMES)


def _smooth_block(n_times, rng, n_sources=6, noise=0.01):
    # RANSAC reconstructs channels from the scalp positions of their
    # neighbors (spherical-spline interpolation), so, unlike ChannelModel's
    # regression, it only works on spatially smooth (physiologically
    # plausible) topographies. Build one by spreading a few random sources
    # over the scalp with a Gaussian kernel.
    sensor = _sensor()
    locs = sensor.locs
    n_ch = len(sensor)
    src_locs = locs[rng.choice(n_ch, n_sources, replace=False)]
    d2 = ((locs[:, None, :] - src_locs[None, :, :]) ** 2).sum(-1)
    weights = np.exp(-d2 / (2 * 0.06 ** 2))
    x = weights @ rng.standard_normal((n_sources, n_times))
    x += noise * rng.standard_normal((n_ch, n_times)) * np.abs(x).max()
    return x * 1e-6


def _block(n_times, rng, **kwargs):
    sensor = _sensor()
    return NDVar(_smooth_block(n_times, rng, **kwargs), (sensor, UTS(0, 1 / SFREQ, n_times)), 'eeg')


def _blocks(rng, lengths=(500, 600, 700)):
    return Datalist([_block(n, rng) for n in lengths])


def _inject(block, channel, start, stop, rng, amplitude_factor=4):
    # replace a channel's data over samples [start, stop) with uncorrelated noise
    x = block.x.copy()
    ci = list(block.sensor.names).index(channel)
    x[ci, start:stop] = rng.standard_normal(stop - start) * np.abs(x).max() * amplitude_factor
    return NDVar(x, block.dims, block.name)


def test_ransac_basic_fit_predict_score():
    "Basic fit/predict/score cycle, and bad-channel detection"
    rng = np.random.default_rng(0)
    block = _block(2000, rng)
    bad = _inject(block, 'Cz', 0, 2000, rng)
    model = ChannelRANSACModel(**RANSAC_KWARGS)
    model.fit(bad)
    assert model.sensor == block.sensor
    assert model.estimators_ is not None

    pred = model.predict(bad)
    assert isinstance(pred, NDVar)
    assert pred.sensor == block.sensor
    assert pred.time == block.time

    # default corr_threshold (no positional argument needed)
    score = model.score(bad)
    assert isinstance(score, NDVar)
    assert score.sensor == block.sensor
    ci = list(block.sensor.names).index('Cz')
    assert score.x[ci] == 1
    assert score.x.sum() == 1  # only Cz is bad


def test_ransac_not_fit_error():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        ChannelRANSACModel(**RANSAC_KWARGS).predict(_block(500, rng))
    with pytest.raises(RuntimeError):
        ChannelRANSACModel(**RANSAC_KWARGS).score(_block(500, rng))


def test_ransac_list_input():
    "fit/predict/score accept a list of long epochs"
    rng = np.random.default_rng(0)
    blocks = _blocks(rng)
    model = ChannelRANSACModel(**RANSAC_KWARGS)
    model.fit(blocks)
    assert model.sensor == blocks[0].sensor

    pred = model.predict(blocks)
    assert isinstance(pred, Datalist)
    assert len(pred) == len(blocks)
    for p, b in zip(pred, blocks):
        assert p.sensor == b.sensor

    score = model.score(blocks)
    assert isinstance(score, Datalist)
    assert len(score) == len(blocks)
    assert all(s.sensor == blocks[0].sensor for s in score)


def test_ransac_epoched_input():
    "fit/predict/score/find_bad_windows accept epoched (case x sensor x time) data"
    sensor = _sensor()
    n_cases, n_times = 4, 400
    rng = np.random.default_rng(0)
    x = np.stack([_smooth_block(n_times, rng) for _ in range(n_cases)])
    data = NDVar(x, ('case', sensor, UTS(0, 1 / SFREQ, n_times)), 'eeg')

    model = ChannelRANSACModel(**RANSAC_KWARGS)
    model.fit(data)

    pred = model.predict(data)
    assert pred.has_case and len(pred.get_dim('case')) == n_cases
    assert pred.sensor == sensor

    score = model.score(data)
    assert score.has_case and len(score.get_dim('case')) == n_cases
    assert score.sensor == sensor

    windows = model.find_bad_windows(data, window_len=0.5)
    assert isinstance(windows, Datalist)
    assert len(windows) == n_cases


def test_ransac_find_bad_windows():
    "find_bad_windows isolates a transient bad channel in time"
    rng = np.random.default_rng(0)
    block = _block(1000, rng)
    model = ChannelRANSACModel(**RANSAC_KWARGS)
    model.fit(block)

    # transient on Cz, samples 300:400 (3.0-4.0 s)
    bad = _inject(block, 'Cz', 300, 400, rng)
    windows = model.find_bad_windows(bad, window_len=1.0, corr_threshold=0.75, min_duration=0.05)
    assert {w.channel for w in windows} == {'Cz'}
    w = windows[0]
    assert w.tmin == pytest.approx(3.0)
    assert w.tmax == pytest.approx(4.0)


# --- API parity with ChannelModel -------------------------------------------------------------

def _make_models():
    return {
        'ChannelModel': ChannelModel('ols'),
        'ChannelRANSACModel': ChannelRANSACModel(**RANSAC_KWARGS),
    }


def test_api_generic_pipeline_interchangeable():
    "A single generic pipeline works unmodified for both model classes"
    def run_pipeline(model, data):
        model.fit(data)
        assert model.sensor is not None
        assert model.estimators_ is not None

        pred = model.predict(data)
        assert isinstance(pred, NDVar)
        assert pred.sensor == data.sensor
        assert pred.time == data.time

        score = model.score(data)
        assert isinstance(score, NDVar)
        assert score.has_dim('sensor')

        windows = model.find_bad_windows(data)
        assert isinstance(windows, list)
        assert all(w.channel for w in windows)
        return score

    ch_names = list(_sensor().names)
    ci = ch_names.index('Cz')

    # transient artifact (1 s out of 20 s), so ChannelModel's default fit()
    # threshold does not exclude the whole block
    rng = np.random.default_rng(0)
    ransac_block = _block(2000, rng)
    ransac_bad = _inject(ransac_block, 'Cz', 1000, 1100, rng)
    ransac_score = run_pipeline(ChannelRANSACModel(**RANSAC_KWARGS), ransac_bad)
    assert ransac_score.x[ci] == ransac_score.x.max()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(ch_names), 8)) @ rng.standard_normal((8, 2000))
    x += 0.02 * rng.standard_normal((len(ch_names), 2000))
    ols_block = NDVar(x * 1e-6, (_sensor(), UTS(0, 1 / SFREQ, 2000)), 'eeg')
    ols_bad = _inject(ols_block, 'Cz', 1000, 1100, rng, amplitude_factor=20)
    ols_score = run_pipeline(ChannelModel('ols'), ols_bad)
    assert ols_score.x[ci] == ols_score.x.max()


def test_api_estimators_state_before_and_after_fit():
    "both models expose the same not-fit / fit contract on .sensor / .estimators_"
    rng = np.random.default_rng(0)
    block = _block(500, rng)
    for model in _make_models().values():
        assert model.sensor is None
        assert model.estimators_ is None
        with pytest.raises(RuntimeError):
            model.predict(block)
        model.fit(block)
        assert model.sensor == block.sensor
        assert model.estimators_ is not None
