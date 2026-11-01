# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for finding ICA components that follow the heartbeat of a reference component."""
import numpy as np
import pytest

from eelbrain._meeg.ica_cardiac import peak_locked_sources


TSTEP = 0.01


def _sources(duration: float = 200., seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic ICA sources ``(4, n_times)`` and heartbeat times [s]

    0: sharp heartbeat spikes (negative polarity, as ICA sign is arbitrary)
    1: slow, noisy response lagging each heartbeat by 0.25 s
    2: white noise
    3: 0.9 Hz oscillation, unrelated to the heartbeat
    """
    rng = np.random.RandomState(seed)
    n_times = int(duration / TSTEP)
    time = np.arange(n_times) * TSTEP
    # heartbeats with variable intervals
    beats = np.cumsum(rng.uniform(0.7, 0.9, int(duration)))
    beats = beats[beats < duration - 1]
    x = rng.normal(0, 1, (4, n_times))
    x[0] *= 0.2
    x[3] += 2 * np.sin(2 * np.pi * 0.9 * time)
    for beat in beats:
        i = int(round(beat / TSTEP))
        x[0, i] -= 10
        x[1, i + 25: i + 35] += 1.5 * np.hanning(10)
    return x, beats


def test_peak_locked_sources():
    x, beats = _sources()
    result = peak_locked_sources([x], 0, TSTEP)
    assert result.reference == 0
    assert result.n_beats == result.n_peaks == len(beats)
    assert 0.7 <= result.intervals.min() and result.intervals.max() <= 0.9
    assert result.time[0] == pytest.approx(-0.3)
    assert result.time[-1] == pytest.approx(0.6 - TSTEP)
    assert result.evoked.shape == result.sd.shape == (4, 90)
    assert (result.sd ** 2).mean(1) == pytest.approx(1)
    # the reference peak is at time 0 (in its original polarity)
    assert result.time[result.evoked[0].argmin()] == 0
    # the lagged response is found, but not noise or an unrelated rhythm
    assert result.score[0] > 100
    assert result.score[1] > 10
    assert result.score[2] < 3
    assert result.score[3] < 3
    assert 0.2 < result.time[result.evoked[1].argmax()] < 0.35
    # effect size: the reference consists of the heartbeat, the lagged response only partly
    assert result.explained[0] > 0.9
    assert 0.03 < result.explained[1] < 0.2
    assert result.explained[2] < 0.02
    assert result.explained[3] < 0.02

    # the same data split into segments: beats whose window crosses a boundary are dropped
    segments = np.split(x, 20, axis=1)
    result_segments = peak_locked_sources(segments, 0, TSTEP)
    assert result_segments.n_peaks == result.n_peaks
    assert result_segments.n_beats < result.n_beats
    # intervals are only measured within segments
    assert len(result_segments.intervals) < len(result.intervals)
    assert 0.7 <= result_segments.intervals.min() and result_segments.intervals.max() <= 0.9
    assert result_segments.score[1] > 10
    assert result_segments.score[2] < 3

    # polarity: a single large artifact of the opposite sign does not flip the reference
    y = -x
    y[0, 5000] = -50
    result_artifact = peak_locked_sources([y], 0, TSTEP)
    assert result_artifact.n_peaks == len(beats)
    assert result_artifact.time[result_artifact.evoked[0].argmax()] == 0
    assert result_artifact.score[1] > 10

    # errors
    with pytest.raises(ValueError, match="window must have positive length"):
        peak_locked_sources([x], 0, TSTEP, tstart=0.1, tstop=0.)
    with pytest.raises(ValueError, match="window must have positive length"):
        peak_locked_sources([x], 0, TSTEP, tstart=-0.001, tstop=0.001)
    with pytest.raises(ValueError, match="No peaks"):
        peak_locked_sources([x], 0, TSTEP, threshold=100)
    with pytest.raises(ValueError, match="shorten the window"):
        peak_locked_sources(np.split(x, 500, axis=1), 0, TSTEP, tstart=-1, tstop=1)
