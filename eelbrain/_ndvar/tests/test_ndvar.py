# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
from scipy import signal

from eelbrain import NDVar, Case, Scalar, UTS, datasets, concatenate, convolve, correlation_coefficient, cross_correlation, cwt_morlet, find_intervals, find_peaks, frequency_response, gaussian, normalize_in_cells, psd_welch, resample, set_time
from eelbrain.testing import assert_dataobj_equal, get_ndvar


def test_concatenate():
    """Test concatenate()

    Concatenation of SourceSpace is tested in .test_mne.test_source_estimate
    """
    ds = datasets.get_uts(True)

    v0 = ds[0, 'utsnd']
    v1 = ds[1, 'utsnd']
    vc = concatenate((v1, v0))
    assert_array_equal(vc.sub(time=(0, 1)).x, v1.x)
    assert_array_equal(vc.sub(time=(1, 2)).x, v0.x)
    assert_array_equal(vc.info, ds['utsnd'].info)

    # scalar
    psd = psd_welch(ds['utsnd'], n_fft=100)
    v0 = psd.sub(frequency=(None, 5))
    v1 = psd.sub(frequency=(45, None))
    conc = concatenate((v0, v1), 'frequency')
    assert_array_equal(conc.frequency.values[:5], psd.frequency.values[:5])
    assert_array_equal(conc.frequency.values[5:], psd.frequency.values[45:])
    conc_data = conc.get_data(v1.dimnames)
    assert_array_equal(conc_data[:, :, 5:], v1.x)

    # cat
    x = get_ndvar(2, frequency=0, cat=4)
    x_re = concatenate([x.sub(cat=(None, 'c')), x.sub(cat=('c', None))], 'cat')
    assert_dataobj_equal(x_re, x)


def test_convolve():
    # convolve is also tested in test_boosting.py
    ds = datasets._get_continuous()

    h1d = ds['h1']
    h2d = ds['h2']
    x1d = ds['x1']
    x2d = ds['x2']

    # 1d
    xc = convolve(h1d, x1d)
    xc_np = np.convolve(h1d.x, x1d.x)
    assert_array_equal(xc.x, xc_np[:100])

    # 2d
    xc = convolve(h2d, x2d)
    xc_np = np.convolve(h2d.x[0], x2d.x[0]) + np.convolve(h2d.x[1], x2d.x[1])
    assert_allclose(xc.x, xc_np[:100])

    # add dimension through kernel
    xc = convolve(h2d, x1d)
    xc_np = np.vstack((np.convolve(h2d.x[0], x1d.x), np.convolve(h2d.x[1], x1d.x)))
    assert_array_equal(xc.x, xc_np[:, :100])

    # add dimension through x
    xc = convolve(h1d, x2d)
    xc_np = np.vstack((np.convolve(h1d.x, x2d.x[0]), np.convolve(h1d.x, x2d.x[1])))
    assert_array_equal(xc.x, xc_np[:, :100])

    # 2 predictors
    xc = convolve([h1d, h2d], [x1d, x2d])
    xc_np = np.convolve(h1d.x, x1d.x) + np.convolve(h2d.x[0], x2d.x[0]) + np.convolve(h2d.x[1], x2d.x[1])
    assert_allclose(xc.x, xc_np[:100])


def test_correlation_coefficient():
    ds = datasets.get_uts()
    uts = ds['uts']
    uts2 = uts.copy()
    uts2.x += np.random.normal(0, 1, uts2.shape)

    assert correlation_coefficient(uts, uts2) == pytest.approx(
        np.corrcoef(uts.x.ravel(), uts2.x.ravel())[0, 1])
    assert_allclose(
        correlation_coefficient(uts[:10], uts2[:10], 'time').x,
        [np.corrcoef(uts.x[i], uts2.x[i])[0, 1] for i in range(10)])
    assert_allclose(
        correlation_coefficient(uts[:, :-.1], uts2[:, :-.1], 'case').x,
        [np.corrcoef(uts.x[:, i], uts2.x[:, i])[0, 1] for i in range(10)])


def test_cross_correlation():
    ds = datasets._get_continuous()
    x = ds['x1']

    assert cross_correlation(x, x).argmax() == 0
    assert cross_correlation(x[2:], x).argmax() == 0
    assert cross_correlation(x[:9], x).argmax() == 0
    assert cross_correlation(x, x[1:]).argmax() == 0
    assert cross_correlation(x, x[:8]).argmax() == 0
    assert cross_correlation(x[2:], x[:8]).argmax() == 0


def test_cwt():
    ds = datasets._get_continuous()
    # 1d
    y = cwt_morlet(ds['x1'], [2, 3, 4])
    assert y.ndim == 2
    # 2d
    y = cwt_morlet(ds['x2'], [2, 3, 4])
    assert y.ndim == 3


def test_diff():
    ds = datasets.get_uts()
    y = ds['uts']
    time_mask = np.logical_and(y.time.times > 0, y.time.times < 0.100).reshape((1, -1))
    mask = np.repeat(time_mask, len(y), 0)
    y_masked = y.mask(mask)
    # target for time-diff
    time_mask = np.logical_and(y.time.times > 0, y.time.times < 0.110).reshape((1, -1))
    target_mask = np.repeat(time_mask, len(y), 0)

    # 1d
    diff = y_masked[0].diff('time')
    assert_array_equal(diff.x.data, np.diff(y.x[0], 1, 0, y.x[0, 0]))
    assert_array_equal(diff.x.mask, target_mask[0])

    # 2d
    diff = y_masked.diff('case')
    assert_array_equal(diff.x.data, np.diff(y.x, 1, 0, y.x[:1]))
    assert_array_equal(diff.x.mask, mask)

    diff = y_masked.diff('time')
    assert_array_equal(diff.x.data, np.diff(y.x, 1, 1, y.x[:, :1]))
    assert_array_equal(diff.x.mask, target_mask)


def test_dot():
    ds = datasets.get_uts(True)

    # x subset of y
    index = ['3', '2']
    utsnd = ds['utsnd']
    topo = utsnd.mean(('case', 'time'))
    y1 = topo.sub(sensor=index).dot(utsnd.sub(sensor=index))
    assert_dataobj_equal(topo[index].dot(utsnd), y1)
    assert_dataobj_equal(topo.dot(utsnd.sub(sensor=index)), y1)


def test_find_intervals():
    time = UTS(-5, 1, 10)
    x = NDVar([0, 1, 0, 1, 1, 0, 1, 1, 1, 0], (time,))
    assert find_intervals(x) == ((-4, -3), (-2, 0), (1, 4))
    x = NDVar([0, 1, 0, 1, 1, 0, 1, 1, 1, 1], (time,))
    assert find_intervals(x) == ((-4, -3), (-2, 0), (1, 5))
    x = NDVar([1, 1, 0, 1, 1, 0, 1, 1, 1, 1], (time,))
    assert find_intervals(x) == ((-5, -3), (-2, 0), (1, 5))


def test_find_peaks():
    scalar = Scalar('scalar', range(9))
    time = UTS(0, .1, 12)
    v = NDVar.zeros((scalar, time))
    wsize = [0, 0, 1, 2, 3, 2, 1, 0, 0]
    for i, s in enumerate(wsize):
        if s:
            v.x[i, 5 - s: 5 + s] += np.hamming(2 * s)

    v = round(v, 5)  # numpy hamming window is not exactly symmetric
    peaks = find_peaks(v)
    assert_array_equal(np.where(peaks.x), ([4, 4], [4, 5]))


def test_frequency_response():
    b_array = signal.firwin(80, 0.5, window=('kaiser', 8))
    freqs_array, fresp_array = signal.freqz(b_array)
    hz_to_rad = 2 * np.pi * 0.01

    b = NDVar(b_array, (UTS(0, 0.01, 80),))
    fresp = frequency_response(b)
    assert_array_equal(fresp.x, fresp_array)
    assert_array_equal(fresp.frequency.values * hz_to_rad, freqs_array)

    b2d = concatenate((b, b), Case)
    fresp = frequency_response(b2d)
    assert_array_equal(fresp.x[0], fresp_array)
    assert_array_equal(fresp.x[1], fresp_array)
    assert_array_equal(fresp.frequency.values * hz_to_rad, freqs_array)


def test_gaussian():
    time = UTS(0, 0.1, 5)
    assert_array_equal(gaussian(0.1, 0.2, time).x, signal.windows.gaussian(7, 2)[2:])
    assert_array_equal(gaussian(0.2, 0.2, time).x, signal.windows.gaussian(5, 2))
    assert_array_equal(gaussian(0.3, 0.1, time).x, signal.windows.gaussian(7, 1)[:5])
    time = UTS(0, 0.1, 6)
    assert_array_equal(gaussian(0.1, 0.2, time).x, signal.windows.gaussian(9, 2)[3:])
    assert_array_equal(gaussian(0.2, 0.2, time).x, signal.windows.gaussian(9, 2)[2:8])
    assert_array_equal(gaussian(0.3, 0.2, time).x, signal.windows.gaussian(9, 2)[1:7])
    assert_array_equal(gaussian(0.4, 0.1, time).x, signal.windows.gaussian(9, 1)[:6])


def test_mask():
    ds = datasets.get_uts(True)

    x = NDVar([1, 2, 3], Case)
    assert x.mean() == 2.0
    y = x.mask([True, False, False])
    assert y.mean() == 2.5

    # multi-dimensional
    y = ds[:2, 'utsnd'].copy()
    mask_x = y.time.times >= 0.500
    mask_ndvar = NDVar(mask_x, y.time)
    y_masked = y.mask(mask_ndvar)
    assert_array_equal(y_masked.x.mask[:, :, 70:], True)
    assert_array_equal(y_masked.x.mask[:, :, :70], False)
    # mask that is smaller than array
    mask = mask_ndvar.sub(time=(0.100, None))
    with pytest.raises(ValueError):
        y.mask(mask)
    y_masked = y.mask(mask, missing=True)
    assert_array_equal(y_masked.x.mask[:, :, 70:], True)
    assert_array_equal(y_masked.x.mask[:, :, 30:70], False)
    assert_array_equal(y_masked.x.mask[:, :, :30], True)


def test_normalize_in_cells():
    ds = datasets.get_uts(True)
    ab = ds.eval("A % B")
    indices = [ab == cell for cell in ab.cells]
    # z-score
    ds['utsnd_n'] = normalize_in_cells('utsnd', 'sensor', data=ds)
    y_mean = ds['utsnd_n'].mean('case')
    assert_allclose(y_mean.mean('sensor'), 0, atol=1e-10)
    assert_allclose(y_mean.std('sensor'), 1)
    ds['utsnd_n'] = normalize_in_cells('utsnd', 'sensor', 'A % B', data=ds)
    for index in indices:
        y_mean = ds[index, 'utsnd_n'].mean('case')
        assert_allclose(y_mean.mean('sensor'), 0, atol=1e-10)
        assert_allclose(y_mean.std('sensor'), 1)
    # range
    ds['utsnd_n'] = normalize_in_cells('utsnd', 'sensor', data=ds, method='range')
    y_mean = ds['utsnd_n'].mean('case')
    assert_allclose(y_mean.min('sensor'), 0, atol=1e-10)
    assert_allclose(y_mean.max('sensor'), 1)
    ds['utsnd_n'] = normalize_in_cells('utsnd', 'sensor', 'A % B', data=ds, method='range')
    for index in indices:
        y_mean = ds[index, 'utsnd_n'].mean('case')
        assert_allclose(y_mean.min('sensor'), 0, atol=1e-10)
        assert_allclose(y_mean.max('sensor'), 1)


def test_resample():
    x = NDVar([0.0, 1.0, 1.4, 1.0, 0.0], UTS(0, 0.1, 5)).mask([True, False, False, False, True])
    y = resample(x, 20)
    assert_array_equal(y.x.mask, [True, False, False, False, False, False, False, False, True, True])
    y = resample(x, 20, npad=0)
    assert_array_equal(y.x.mask, [True, False, False, False, False, False, False, False, True, True])


def test_set_time():
    for x in [get_ndvar(2, 100, 0), get_ndvar(2, 100, 8)]:
        x_sub = x.sub(time=(0.000, None))
        assert x_sub.time.tmin == 0.000
        x_pad = set_time(x_sub, x)
        assert x_pad.time.tmin == -0.100
        assert x_pad.x.ravel()[0] == 0
        x_pad = set_time(x_sub, x, mode='edge')
        assert x_pad.time.tmin == -0.100
        assert x_pad.x.ravel()[0] == x_sub.x.ravel()[0]


def test_smoothing():
    x = get_ndvar(2)
    xt = NDVar(x.x.swapaxes(1, 2), [x.dims[i] for i in [0, 2, 1]], x.name, x.info)

    # smoothing across time
    ma = x.smooth('time', 0.2, 'blackman')
    assert_dataobj_equal(x.smooth('time', window='blackman', window_samples=20), ma)
    with pytest.raises(TypeError):
        x.smooth('time')
    with pytest.raises(TypeError):
        x.smooth('time', 0.2, 'blackman', window_samples=20)
    mas = xt.smooth('time', 0.2, 'blackman')
    assert_allclose(ma.x, mas.x.swapaxes(1, 2), 1e-10)
    ma_mean = x.mean('case').smooth('time', 0.2, 'blackman')
    assert_allclose(ma.mean('case').x, ma_mean.x)
    # against raw scipy.signal
    window = signal.get_window('blackman', 20, False)
    window /= window.sum()
    window.shape = (1, 20, 1)
    assert_array_equal(ma.x[:, 10:-10], signal.convolve(x.x, window, 'same')[:, 10:-10])
    # mode parameter
    full = signal.convolve(x.x, window, 'full')
    ma = x.smooth('time', 0.2, 'blackman', mode='left')
    assert_array_equal(ma.x[:], full[:, :100])
    ma = x.smooth('time', 0.2, 'blackman', mode='right')
    assert_array_equal(ma.x[:], full[:, 19:])

    # fix_edges: smooth with constant sum
    xs = x.smooth('frequency', window_samples=1, fix_edges=True)
    assert_dataobj_equal(xs.sum('frequency'), x.sum('frequency'))
    xs = x.smooth('frequency', window_samples=2, fix_edges=True)
    assert_dataobj_equal(xs.sum('frequency'), x.sum('frequency'), 14)
    xs = x.smooth('frequency', window_samples=3, fix_edges=True)
    assert_dataobj_equal(xs.sum('frequency'), x.sum('frequency'), 14)
    xs = x.smooth('frequency', window_samples=5, fix_edges=True)
    assert_dataobj_equal(xs.sum('frequency'), x.sum('frequency'), 14)
    xs = x.smooth('frequency', window_samples=4, fix_edges=True)
    assert_dataobj_equal(xs.sum('frequency'), x.sum('frequency'), 14)

    # gaussian
    x = get_ndvar(2, frequency=0, sensor=5)
    x.smooth('sensor', 0.1, 'gaussian')
    x = get_ndvar(2, sensor=5)
    x.smooth('sensor', 0.1, 'gaussian')
