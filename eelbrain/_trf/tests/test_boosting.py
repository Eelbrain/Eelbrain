# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from dataclasses import replace
from itertools import product
from math import floor
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pickle
import pytest
from pytest import approx
import scipy.io
import scipy.stats
from eelbrain import datasets, boosting, combine, convolve, correlation_coefficient, epoch_impulse_predictor, NDVar, UTS, Scalar

from eelbrain.testing import assert_dataobj_equal
from eelbrain._trf._boosting import Boosting, DeconvolutionData, Split, convolve_1d
from eelbrain._trf._boosting_opt import boosting_fit


DATA_DIR = Path(__file__).parent


def assert_res_equal(res1, res):
    assert_array_equal(res1.h, res.h)
    assert res1.r == res.r
    assert res1.r_rank == res.r_rank


def convolve_array(
        h: np.ndarray,  # (n_stims, h_n_samples)
        x: np.ndarray,  # (n_stims, n_samples)
        x_pads: np.ndarray,  # (n_stims,)
        h_i_start: int = 0,
) -> np.ndarray:
    n_x, n_times = x.shape
    segments = np.array(((0, n_times),), np.int64)
    out = np.empty(n_times)
    convolve_1d(h, x, x_pads, h_i_start, segments, out)
    return out


def test_boosting():
    "Test boosting NDVars"
    ds = datasets._get_continuous(ynd=True)
    y = ds['y']
    ynd = ds['ynd']
    x1 = ds['x1']
    x2 = ds['x2']
    y_mean = y.mean()
    x2_mean = x2.mean('time')

    # test values from running function, not verified independently
    res = boosting(y, x1 * 2000, 0, 1, scale_data=False, mindelta=0.0025, debug=True)
    assert repr(res) == '<BoostingResult y ~ x1, 0 - 1, scale_data=False, mindelta=0.0025>'
    assert_allclose(res.h.x, [0, 0, 0, 0.0025, 0, 0, 0, 0, 0, 0.001875], atol=1e-6)
    assert res.r == approx(0.75, abs=0.001)
    assert res.y_mean is None
    assert res.h.info['unit'] == 'V'
    assert res.h_scaled.info['unit'] == 'V'
    assert res.residual == approx(((y[.9:] - res.y_pred[.9:])**2).sum())
    with pytest.raises(NotImplementedError):
        _ = res.proportion_explained

    res = boosting(y, x1, 0, 1)
    assert repr(res) == '<BoostingResult y ~ x1, 0 - 1>'
    assert res.r == approx(0.83, abs=0.001)
    assert res.y_mean == y_mean
    assert res.y_scale == y.std()
    assert res.x_mean == x1.mean()
    assert res.x_scale == x1.std()
    assert res.h.name == 'x1'
    assert res.h.info['unit'] == 'normalized'
    assert res.h_scaled.name == 'x1'
    assert res.h_scaled.info['unit'] == 'V'
    assert res.proportion_explained == approx(0.506, abs=0.001)
    # inplace
    res_ip = boosting(y.copy(), x1.copy(), 0, 1, 'inplace')
    assert_res_equal(res_ip, res)
    # persistence
    res_p = pickle.loads(pickle.dumps(res, pickle.HIGHEST_PROTOCOL))
    assert_res_equal(res_p, res)

    # L1 error
    res = boosting(y, x1 * 2000, 0, 1, error='l1', debug=True)
    assert res.residual == ((y[.9:] - res.y_mean) / res.y_scale - res.y_pred[.9:]).abs().sum()
    res_ndb = boosting(y, x1 * 2000, 0, 1, error='l1')
    assert res_ndb.residual == res.residual

    # cross-validation
    res = boosting(y[:7.5], x1[:7.5], 0, 1, scale_data=False, partitions=3, debug=True)
    res_cv = boosting(y, x1, 0, 1, test=1, scale_data=False, partitions=4, debug=True, partition_results=True)
    assert correlation_coefficient(res.h, res_cv.h) == approx(.986, abs=.001)
    # using cross-prediction
    y_pred = res_cv.cross_predict(x1, scale='normalized')
    assert correlation_coefficient(y_pred, y, 'time') == pytest.approx(res_cv.r)
    # fit res_cv on same data as res
    fit = res_cv.fit
    fit.data.splits = replace(fit.data.splits, splits=[fit.data.splits.splits[i] for i in [9, 10, 11]])
    fit.fit(0, 1, 0, 'l2')
    res_cv = fit.evaluate_fit(debug=True)
    assert_dataobj_equal(res_cv.h, res.h)
    y_pred = convolve(res_cv.h_scaled, x1[7.5:])
    assert correlation_coefficient(y_pred, y[7.5:]) == approx(res_cv.r, abs=1e-8)

    res = boosting(y, x2, 0, 1)
    assert res.r == approx(0.601, abs=0.001)
    assert res.proportion_explained == approx(0.273, abs=0.001)

    res = boosting(y, x2, 0, 1, error='l1')
    assert res.r == approx(0.553, abs=0.001)
    assert res.y_mean == y.mean()
    assert res.y_scale == (y - y_mean).abs().mean()
    assert_array_equal(res.x_mean.x, x2_mean)
    assert_array_equal(res.x_scale, (x2 - x2_mean).abs().mean('time'))
    assert res.proportion_explained == approx(0.123, abs=0.001)

    # 2 predictors
    res = boosting(y, [x1, x2], 0, 1)
    assert res.r == approx(0.947, abs=0.001)
    # selective stopping
    res = boosting(y, [x1, x2], 0, 1, selective_stopping=1)
    assert res.r == approx(0.967, abs=0.001)
    res = boosting(y, [x1, x2], 0, 1, selective_stopping=2)
    assert res.r == approx(0.992, abs=0.001)

    # 2d-y
    res_cv = boosting('ynd', ['x1', 'x2'], 0, 1, data=ds, test=1, partitions=4, partition_results=True)
    y_pred = res_cv.cross_predict(data=ds)
    assert_dataobj_equal(correlation_coefficient(y_pred, ynd, 'time'), res_cv.r, 2, name=False)


@pytest.mark.parametrize('error', ['l1', 'l2'])
def test_boosting_cross_predict(error):
    """Test cross-predicting data using stored TRFs"""
    ds = datasets._get_continuous()

    # Without scaling
    trf = boosting('y', 'x1', 0, 1, data=ds, error=error, partitions=3, test=1, partition_results=True, debug=True, scale_data=False)
    y_pred = trf.cross_predict('x1', ds)
    assert_array_equal(y_pred, trf.y_pred)

    # With scaling: normalized
    trf = boosting('y', 'x1', 0, 1, data=ds, error=error, partitions=3, test=1, partition_results=True, debug=True)
    # Results
    assert getattr(trf, f'{error}_total') == pytest.approx(trf.n_samples, 1e-16)
    # Re-predict
    y_pred = trf.cross_predict('x1', ds, scale='normalized')
    assert_array_equal(y_pred, trf.y_pred)
    # Proportion explained
    y_normalized = ds['y'] - trf.y_mean
    y_normalized /= trf.y_scale
    y_residual = y_normalized - y_pred
    if error == 'l1':
        proportion_explained = 1 - (y_residual.abs().sum('time') / y_normalized.abs().sum('time'))
    else:
        proportion_explained = 1 - ((y_residual ** 2).sum('time') / (y_normalized ** 2).sum('time'))
    assert proportion_explained == pytest.approx(trf.proportion_explained, 1e-16)

    # With scaling: original scale
    y_pred = trf.cross_predict('x1', ds)
    assert_array_equal(y_pred, trf.y_pred * trf.y_scale + trf.y_mean)


def test_boosting_epochs():
    """Test boosting with epoched data"""
    ds = datasets.get_uts(True, vector3d=True)
    p1 = epoch_impulse_predictor('uts', 'A=="a1"', name='a1', data=ds)
    p0 = epoch_impulse_predictor('uts', 'A=="a0"', name='a0', data=ds)
    p1 = p1.smooth('time', .05, 'hamming')
    p0 = p0.smooth('time', .05, 'hamming')
    # 1d
    for tstart, basis in product((-0.1, 0.1, 0), (0, 0.05)):
        print(f"{tstart=}, {basis=}")
        res = boosting('uts', [p0, p1], tstart, 0.6, model='A', data=ds, basis=basis, partitions=3, debug=True)
        assert res.r == approx(0.238, abs=2e-3)
        y = convolve(res.h_scaled, [p0, p1], name='predicted')
        assert correlation_coefficient(y, res.y_pred) > .999
        assert y.name == 'predicted'
        r = correlation_coefficient(y, ds['uts'])
        assert res.r == approx(r, abs=1e-3)
        assert res.splits.n_partitions == 3
    # 2d
    res = boosting('utsnd', [p0, p1], 0, 0.6, model='A', data=ds, partitions=3)
    assert len(res.h) == 2
    assert res.h[0].shape == (5, 60)
    assert res.h[1].shape == (5, 60)
    y = convolve(res.h_scaled, [p0, p1])
    r = correlation_coefficient(y, ds['utsnd'], ('case', 'time'))
    assert_dataobj_equal(res.r, r, decimal=3, name=False)
    # cross-validation
    res_cv = boosting('utsnd', [p0, p1], 0, 0.6, error='l1', data=ds, partitions=3, test=1, partition_results=True, debug=True)
    y_pred = res_cv.cross_predict([p0, p1], scale='normalized')
    assert_array_equal(y_pred, res_cv.y_pred)
    # vector
    res = boosting('v3d', [p0, p1], 0, 0.6, error='l1', model='A', data=ds, partitions=3)
    assert res.residual.ndim == 0


def test_boosting_object():
    "Test Boosting object"
    ds = datasets._get_continuous(ynd=True)
    data = DeconvolutionData('y', 'x2', ds)
    data.apply_basis(0.2, 'hamming')
    data.normalize('l1')
    data.initialize_cross_validation(4, test=1)
    model = Boosting(data)
    model.fit(0, 1, selective_stopping=1, error='l1')
    res_oo = model.evaluate_fit()

    res_func = boosting('y', 'x2', 0, 1, data=ds, error='l1', basis=0.2, partitions=4, test=1, selective_stopping=1)
    assert res_oo.r == res_func.r

    res_part = model.evaluate_fit(partition_results=True)
    assert len(res_part.partition_results) == 4
    # h with basis
    h_part = combine([res.h for res in res_part.partition_results])
    assert len(h_part) == 4
    assert_dataobj_equal(h_part.mean('case'), res_oo.h, decimal=10)
    # raw h
    h_part = combine([res._h for res in res_part.partition_results])
    assert_dataobj_equal(h_part.mean('case'), res_oo._h)


def test_result():
    "Test boosting results"
    ds = datasets._get_continuous()
    x1 = ds['x1']

    # convolve function
    y = convolve([ds['h1'], ds['h2']], [ds['x1'], ds['x2']])
    assert_dataobj_equal(y, ds['y'], name=False)

    # test prediction with res.h and res.h_scaled
    res = boosting(ds['y'], ds['x1'], 0, 1)
    y1 = convolve(res.h_scaled, ds['x1'])
    x_scaled = ds['x1'] / res.x_scale
    y2 = convolve(res.h, x_scaled)
    y2 *= res.y_scale
    y2 += y1.mean() - y2.mean()  # mean can't be reconstructed
    assert_dataobj_equal(y1, y2, decimal=12)
    # reconstruction
    res = boosting(x1, y, -1, 0, debug=True)
    x1r = convolve(res.h_scaled, y)
    assert correlation_coefficient(res.y_pred, x1r) > .999
    assert correlation_coefficient(x1r[0.9:], x1[0.9:]) == approx(res.r, abs=1e-3)

    # test NaN checks  (modifies data)
    ds['x2'].x[1, 50] = np.nan
    with pytest.raises(ValueError):
        boosting(ds['y'], ds['x2'], 0, .5)
    with pytest.raises(ValueError):
        boosting(ds['y'], ds['x2'], 0, .5, False)
    ds['x2'].x[1, :] = 1
    with catch_warnings():
        filterwarnings('ignore', category=RuntimeWarning)
        with pytest.raises(ValueError):
            boosting(ds['y'], ds['x2'], 0, .5)
        ds['y'].x[50] = np.nan
        with pytest.raises(ValueError):
            boosting(ds['y'], ds['x1'], 0, .5)
        with pytest.raises(ValueError):
            boosting(ds['y'], ds['x1'], 0, .5, False)


def evaluate_kernel(y, y_pred, test_seg_len, n_skip):
    y = y[n_skip: test_seg_len]
    y_pred = y_pred[n_skip:]
    return np.corrcoef(y, y_pred)[0, 1], scipy.stats.spearmanr(y, y_pred)[0]


def test_boosting_fit():
    "Test boosting() against svdboostV4pred.m"
    # 1d-TRF
    mat = scipy.io.loadmat(DATA_DIR / 'test_boosting.mat')
    y = mat['signal'][0]
    x = mat['stim']
    x_pads = np.zeros(len(x))

    y_len = len(y)
    seg_len = int(y_len / 40)
    all_segments = np.array([[0, seg_len], [seg_len, y_len]], np.int64)
    split = Split(all_segments[1:], all_segments[:1])
    tstart = np.array([0], np.int64)
    tstop = np.array([10], np.int64)
    h, history = boosting_fit(y, x, x_pads, split.train, split.validate, split.train_and_validate, tstart, tstop, 0.005, 0.005, 2)
    test_sse_history = [step.e_test for step in history]
    test_seg_len = int(floor(x.shape[1] / 40))
    y_pred = convolve_array(h, x[:, :test_seg_len], x_pads)
    r, rr = evaluate_kernel(y, y_pred, test_seg_len, h.shape[1] - 1)

    assert_allclose(test_sse_history, mat['Str_testE'][0])
    assert_array_equal(h, mat['h'])
    assert r == approx(mat['crlt'][0, 0])
    assert rr == approx(mat['crlt'][1, 0])

    # 2d-TRF
    mat = scipy.io.loadmat(DATA_DIR / 'test_boosting_2d.mat')
    y = mat['signal'][0]
    x = mat['stim']
    x_pads = np.zeros(len(x))
    tstart = np.array([0, 0, 0], np.int64)
    tstop = np.array([10, 10, 10], np.int64)
    h, history = boosting_fit(y, x, x_pads, split.train, split.validate, split.train_and_validate, tstart, tstop, 0.005, 0.005, 2)
    test_sse_history = [step.e_test for step in history]
    test_seg_len = int(floor(x.shape[1] / 40))
    y_pred = convolve_array(h, x[:, :test_seg_len], x_pads)
    r, rr = evaluate_kernel(y, y_pred, test_seg_len, h.shape[1] - 1)

    # svdboostV4pred multiplies error by number of predictors
    assert_allclose(test_sse_history, mat['Str_testE'][0] / 3)
    assert_array_equal(h, mat['h'])
    assert r == approx(mat['crlt'][0, 0])
    assert rr == approx(mat['crlt'][1, 0])


def test_trf_len():
    # test vanilla boosting
    rng = np.random.RandomState(0)
    x = NDVar(rng.normal(0, 1, 1000), UTS(0, 0.1, 1000), name='x')
    k = NDVar(rng.randint(0, 10, 5) / 10, UTS(0, 0.1, 5), name='k')
    y = convolve(k, x, name='y')
    res = boosting(y, x, 0, 0.5, partitions=3)
    assert correlation_coefficient(res.h, k) > 0.99
    assert repr(res) == '<BoostingResult y ~ x, 0 - 0.5, partitions=3>'
    # split the predictor into two complementary time windows (should be identical)
    res2 = boosting(y, [x, x], [0.000, 0.250], [0.250, 0.500], partitions=3)
    assert_array_equal(res2.h[0] + res2.h[1], res.h)
    assert res2.r == res.r

    # test multiple tstart, tend
    x2 = NDVar(rng.normal(0, 1, 1000), UTS(0, 0.1, 1000), name='x2')
    k2 = NDVar(rng.randint(0, 10, 4) / 10, UTS(-0.1, 0.1, 4), name='k2')
    y2 = y + convolve(k2, x2)
    res = boosting(y2, [x, x2], [0, -0.1], [0.5, 0.3], partitions=3)
    assert correlation_coefficient(res.h[0].sub(time=(0, 0.5)), k) > 0.99
    assert correlation_coefficient(res.h[1].sub(time=(-0.1, 0.3)), k2) > 0.99
    assert repr(res) == '<BoostingResult y ~ x (0 - 0.5) + x2 (-0.1 - 0.3), partitions=3>'

    # test scalar res.tstart res.tstop
    res = boosting(y2, [x, x2], 0, 0.5, partitions=3)
    assert res.tstart == 0
    assert res.tstop == 0.5

    # test duplicate tstart, tend for multiple predictors
    k3 = NDVar(k2.x, UTS(0, 0.1, 4))
    y3 = convolve(k, x) + convolve(k3, x2)
    res = boosting(y3, [x, x2], 0, 0.5, partitions=3)
    assert correlation_coefficient(res.h[0], k) > 0.99
    assert correlation_coefficient(res.h[1].sub(time=(0, 0.4)), k3) > 0.99

    # test vanilla boosting with 2d predictor
    x4 = NDVar(rng.normal(0, 1, (2, 1000)), (Scalar('xdim', [1, 2]), UTS(0, 0.1, 1000)))
    k4 = NDVar(rng.randint(0, 10, (2, 5)) / 10, (Scalar('xdim', [1, 2]), UTS(0, 0.1, 5)))
    y4 = convolve(k4, x4)
    res = boosting(y4, x4, 0, 0.5, partitions=3)
    assert correlation_coefficient(res.h, k4) > 0.99

    # test multiple tstart, tstop with 1d, 2d predictors
    y5 = y4 + y2
    res = boosting(y5, [x, x2, x4], [0, -0.1, 0], [0.5, 0.3, 0.5], partitions=3)
    assert correlation_coefficient(res.h[0].sub(time=(0, 0.5)), k) > 0.99
    assert correlation_coefficient(res.h[1].sub(time=(-0.1, 0.3)), k2) > 0.99
    assert correlation_coefficient(res.h[2].sub(time=(0, 0.5)), k4) > 0.99

    # tests tstart/tstop for each time series (not implemented)
    # res2 = boosting(y5, [x, x2, x4], [0, -0.1, 0, 0], [0.5, 0.3, 0.5, 0.5])
    # assert_array_equal(res.h[0], res2.h[0])
    # assert_array_equal(res.h[1], res2.h[1])
    # assert_array_equal(res.h[2], res2.h[2])
