# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for clean_windows/fit_eeg_distribution."""
import mne
import numpy as np

from eelbrain._meeg._clean_windows import clean_windows, fit_eeg_distribution


SFREQ = 100.
CH_NAMES = [f'EEG{i:03d}' for i in range(8)]


def _raw(n_times, rng, artifact=None, artifact_channels=None):
    data = rng.standard_normal((len(CH_NAMES), n_times)) * 5e-6
    if artifact is not None:
        start, stop = artifact
        chs = range(len(CH_NAMES)) if artifact_channels is None else artifact_channels
        for ci in chs:
            data[ci, start:stop] += rng.standard_normal(stop - start) * 200e-6
    info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
    return mne.io.RawArray(data, info, verbose='error')


def test_fit_eeg_distribution():
    "fit_eeg_distribution recovers mean/std of a clean Gaussian sample"
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(loc=5e-6, scale=1e-6, size=2000))
    mu, sigma, alpha, beta = fit_eeg_distribution(x)
    assert 3e-6 < mu < 7e-6
    assert 0 < sigma < 3e-6


def test_clean_windows_detects_artifact():
    "clean_windows brackets an injected high-power segment, attributed to every channel"
    rng = np.random.default_rng(0)
    raw = _raw(6000, rng, artifact=(2000, 2500))  # 20-25 s, all channels
    intervals = clean_windows(raw)
    assert len(intervals) == len(CH_NAMES)
    assert {ch for ch, _, _ in intervals} == set(CH_NAMES)
    for ch, tmin, tmax in intervals:
        assert tmin <= 20.0
        assert tmax >= 25.0
        assert 0 <= tmin
        assert tmax <= 60.0


def test_clean_windows_channel_attribution():
    "clean_windows attributes bad windows to (at least) the channels that were actually bad"
    rng = np.random.default_rng(0)
    bad_channels = (0, 2, 4, 6)  # more than max_bad_channels(0.2 * 8 -> 2), so the window still counts as bad
    raw = _raw(6000, rng, artifact=(2000, 2500), artifact_channels=bad_channels)
    intervals = clean_windows(raw)
    assert intervals  # enough corrupted channels for the window to be flagged
    got_channels = {ch for ch, _, _ in intervals}
    # every actually-corrupted channel must be attributed; the robust distribution
    # fit can occasionally also flag an individual clean channel's window, so this
    # does not assert an exact match, only that attribution discriminates at all
    assert got_channels >= {CH_NAMES[i] for i in bad_channels}
    assert got_channels < set(CH_NAMES)


def test_clean_windows_clean_data():
    "clean_windows returns no intervals for clean data"
    rng = np.random.default_rng(1)
    raw = _raw(3000, rng)
    assert clean_windows(raw) == []


def test_clean_windows_raw_local_time():
    "clean_windows intervals are raw-local (0 at raw's own first sample)"
    rng = np.random.default_rng(0)
    raw = _raw(6000, rng, artifact=(2000, 2500))  # artifact at absolute 20-25 s
    cropped = raw.copy().crop(tmin=5.0)
    assert cropped.first_samp == 500

    # in the cropped raw's own (raw-local) time, the artifact is at 15-20 s,
    # not 20-25 s -- if the function used raw.first_samp/absolute time by
    # mistake, the detected interval would show up at 20-25 s instead
    intervals_cropped = clean_windows(cropped)
    assert len(intervals_cropped) == len(CH_NAMES)
    for ch, tmin, tmax in intervals_cropped:
        assert tmin <= 15.0
        assert tmax >= 20.0
        assert tmax <= 55.0  # cropped raw is only 55 s long
