# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import Factor, NDVar, Scalar, datasets, epoch_impulse_predictor
from eelbrain._ndvar._convolve import convolve_1d
from eelbrain._trf.shared import DeconvolutionData, lag_matrix, split_data


def test_deconvolution_data_continuous():
    ds = datasets._get_continuous(ynd=True)

    # normalizing
    data = DeconvolutionData(ds['x1'][:8], ds['y'][:8])
    data.normalize('l1')
    data.flatten()
    assert data.x.flat[0].mean() == pytest.approx(0, abs=1e-16)
    assert abs(data.x.flat).mean() == pytest.approx(1, abs=1e-10)
    with pytest.raises(ValueError):
        data.apply_basis(0.050, 'hamming')
    # with basis
    data = DeconvolutionData(ds['x1'][:8], ds['y'][:8])
    data.apply_basis(0.500, 'hamming')
    data.normalize('l1')
    data.flatten()
    assert data.x.flat[0].mean() == pytest.approx(0, abs=1e-16)
    assert abs(data.x.flat).mean() == pytest.approx(1, abs=1e-10)

    # partitioning, no testing set
    data.initialize_cross_validation(4, test=0)
    assert len(data.splits.splits) == 4
    assert_array_equal(data.splits.splits[0].validate, [[0, 20]])
    assert_array_equal(data.splits.splits[0].train, [[20, 80]])
    assert_array_equal(data.splits.splits[1].validate, [[20, 40]])
    assert_array_equal(data.splits.splits[1].train, [[0, 20], [40, 80]])
    assert_array_equal(data.splits.splits[2].validate, [[40, 60]])
    assert_array_equal(data.splits.splits[2].train, [[0, 40], [60, 80]])
    assert_array_equal(data.splits.splits[3].validate, [[60, 80]])
    assert_array_equal(data.splits.splits[3].train, [[0, 60]])

    # partitioning, testing set
    data.initialize_cross_validation(4, test=1)
    assert len(data.splits.splits) == 12
    # 0/1
    assert_array_equal(data.splits.splits[0].test, [[0, 20]])
    assert_array_equal(data.splits.splits[0].validate, [[20, 40]])
    assert_array_equal(data.splits.splits[0].train, [[40, 80]])
    # 0/2
    assert_array_equal(data.splits.splits[1].test, [[0, 20]])
    assert_array_equal(data.splits.splits[1].validate, [[40, 60]])
    assert_array_equal(data.splits.splits[1].train, [[20, 40], [60, 80]])
    # 0/3
    assert_array_equal(data.splits.splits[2].test, [[0, 20]])
    assert_array_equal(data.splits.splits[2].validate, [[60, 80]])
    assert_array_equal(data.splits.splits[2].train, [[20, 60]])
    # 1/0
    assert_array_equal(data.splits.splits[3].test, [[20, 40]])
    assert_array_equal(data.splits.splits[3].validate, [[0, 20]])
    assert_array_equal(data.splits.splits[3].train, [[40, 80]])
    # 1/2
    assert_array_equal(data.splits.splits[4].test, [[20, 40]])
    assert_array_equal(data.splits.splits[4].validate, [[40, 60]])
    assert_array_equal(data.splits.splits[4].train, [[0, 20], [60, 80]])
    # 1/3, 2/0, 2/1, 2/3, 3/0, 3/1, 3/2
    assert_array_equal(data.splits.splits[11].test, [[60, 80]])
    assert_array_equal(data.splits.splits[11].validate, [[40, 60]])
    assert_array_equal(data.splits.splits[11].train, [[0, 40]])


def test_deconvolution_data_trials():
    ds = datasets.get_uts()
    n_times = len(ds['uts'].time)
    ds['imp'] = epoch_impulse_predictor('uts', data=ds)
    ds['imp_a'] = epoch_impulse_predictor('uts', "A == 'a1'", data=ds)

    data = DeconvolutionData('imp', 'uts', ds)
    data.normalize('l1')
    assert_array_equal(data.segments, [[i * n_times, (i + 1) * n_times] for i in range(60)])

    # partitioning
    data.initialize_cross_validation(3, test=0)
    assert len(data.splits.splits) == 3
    arange = np.arange(len(data.segments)) % 3
    for i, split in enumerate(data.splits.splits):
        validate_index = arange == i
        assert_array_equal(split.validate, data.segments[validate_index])
        assert_array_equal(split.train, data.segments[~validate_index])

    # continuoue model
    data.initialize_cross_validation(3, 'A', ds, test=0)
    assert len(data.splits.splits) == 3
    for i, split in enumerate(data.splits.splits):
        validate_index = arange == i
        assert_array_equal(split.validate, data.segments[validate_index])
        assert_array_equal(split.train, data.segments[~validate_index])

    # alternating model
    ds['C'] = Factor('abc', tile=20)
    data.initialize_cross_validation(3, 'C', ds, test=0)
    assert len(data.splits.splits) == 3
    arange = np.repeat(np.arange(20), 3) % 3
    for i, split in enumerate(data.splits.splits):
        validate_index = arange == i
        assert_array_equal(split.validate, data.segments[validate_index])
        assert_array_equal(split.train, data.segments[~validate_index])


def test_deconvolution_data_segments():
    "Segments are views into the input data; the flat arrays are built on demand"
    ds = datasets.get_uts(True)
    y = ds['utsnd']
    x = ds['uts']
    data = DeconvolutionData([x, 'uts'], y, ds)
    assert len(data.y.segments) == len(data.x.segments) == 60
    assert data.y.n_rows == 5 and data.x.n_rows == 2
    # y segments are views into the NDVar
    assert np.shares_memory(data.y.segments[0], y.x)
    assert_array_equal(data.y.segments[3], y.x[3])
    assert data.segment_slices(np.array([[0, 10], [95, 215]])) == [(0, 0, 10), (0, 95, 100), (1, 0, 100), (2, 0, 15)]
    with pytest.raises(RuntimeError):
        data.y.flat
    # modifying does not touch the input
    y_copy = y.x.copy()
    data.normalize('l2')
    assert_array_equal(y.x, y_copy)
    # the flat array holds the same data, and the segments become views into it
    data.flatten()
    assert data.y.flat.shape == (5, 6000)
    assert_array_equal(data.y.flat[:, 300:400], data.y.segments[3])
    assert np.shares_memory(data.y.flat, data.y.segments[3])
    data.y.segments[3][0, 0] = 99
    assert data.y.flat[0, 300] == 99
    # pickling drops the flat arrays, the segments keep the data
    data_2 = pickle.loads(pickle.dumps(data))
    assert data_2.y._flat is None
    data_2.flatten()
    assert_array_equal(data_2.y.flat, data.y.flat)
    assert_array_equal(data_2.x.mean, data.x.mean)

    # in-place normalization modifies the input
    x_copy = x.x.copy()
    data = DeconvolutionData(x, y, in_place=True)
    data.normalize('l1')
    assert_array_equal(data.x.segments[0][0], x.x[0])
    assert not np.array_equal(x.x, x_copy)
    # for a single predictor, the flat array is a view into the input (no copy)
    assert np.shares_memory(data.x.flat, x.x)
    assert_array_equal(data.x.flat[0, 100:200], x.x[1])
    x.x[:] = x_copy

    # without y (e.g., for prediction)
    data = DeconvolutionData([x, 'uts'], data=ds)
    assert data.y is None
    data.normalize('l1')
    data.check_data()
    data.flatten()
    assert data.x.flat.shape == (2, 6000)
    with pytest.raises(ValueError, match="same time dimension"):
        DeconvolutionData(x, y.sub(time=(0, 0.5)))

    # non-contiguous input: the flat array is a contiguous copy
    x_strided = NDVar(np.repeat(x.x, 2, 1)[:, ::2], x.dims, 'x_strided')
    data = DeconvolutionData(x_strided, y)
    assert not data.x.flat.flags.c_contiguous
    data.flatten()
    assert data.x.flat.flags.c_contiguous
    assert_array_equal(data.x.flat[0, 100:200], x.x[1])
    data.normalize('l1')
    assert_array_equal(x_strided.x, x.x)
    # predictor with time first
    x_t = NDVar(np.stack([x.x[0], x.x[1]], 1), (x.time, Scalar('s', [0, 1])), 'x_t')
    data = DeconvolutionData(x_t, y[0])
    assert not data.x._owned  # transposed view into the input
    data.flatten()
    assert data.x.flat.flags.c_contiguous
    assert_array_equal(data.x.flat, x.x[:2])
    # several cases with time first: the segments are transposed views, the flat array is still C-contiguous
    x_ct = NDVar(np.stack([x.x, x.x ** 2], 2), (x.dims[0], x.time, Scalar('s', [0, 1])), 'x_ct')
    data = DeconvolutionData(x_ct, y)
    assert not data.x._owned
    assert not data.x.segments[0].flags.c_contiguous
    data.flatten()
    assert data.x.flat.flags.c_contiguous
    assert_array_equal(data.x.flat[1, 100:200], x.x[1] ** 2)
    # predictor whose rows combine non-adjacent dimensions: get_data() has to copy, so the data is owned
    x_ab = NDVar(np.ascontiguousarray(x.x[:4].reshape((2, 2, -1)).transpose((0, 2, 1))), (Scalar('a', [0, 1]), x.time, Scalar('b', [0, 1])), 'x_ab')
    data = DeconvolutionData(x_ab, y[0])
    assert data.x._owned
    assert_array_equal(data.x.segments[0], x.x[:4])
    data.normalize('l1')
    assert_array_equal(x_ab.x[:, 0], x.x[:4].reshape((2, 2, -1))[:, :, 0])  # input unchanged


def test_deconvolution_data_check():
    "Flat and NaN data are detected across segments, regardless of DC offset"
    ds = datasets.get_uts(True)
    y = ds['utsnd'].copy()
    x = ds['uts'].copy()
    DeconvolutionData(x, y).check_data()
    y.x[:, 1, :] = 0.1
    with pytest.raises(ValueError, match="1 flat time series"):
        DeconvolutionData(x, y).check_data()
    y = ds['utsnd']
    x.x[3, 10] = np.nan
    with pytest.raises(ValueError, match="NaN in"):
        DeconvolutionData(x, y).check_data()


def test_deconvolution_data_normalize_x():
    "Normalization can be restricted to x, and applied with given values"
    ds = datasets._get_continuous(ynd=True)
    y, x = ds['y'], ds['x1']
    data = DeconvolutionData(x, y)
    data.x.normalize('l2')
    assert data.y.mean is None
    assert_array_equal(data.y.segments[0][0], y.x)
    x_mean, x_scale = x.mean('time'), x.std('time')
    assert data.x.mean[0] == pytest.approx(x_mean)
    assert data.x.scale[0] == pytest.approx(x_scale)
    assert data.x.pads[0] == pytest.approx(-x_mean / x_scale)
    # applying a second normalization composes with the first
    data.x.apply_normalization(1., 2.)
    assert data.x.mean[0] == pytest.approx(x_mean + x_scale)
    assert data.x.scale[0] == pytest.approx(x_scale * 2)
    data.flatten()
    np.testing.assert_allclose(data.x.flat[0], ((x.x - x_mean) / x_scale - 1) / 2)
    # the recorded values reproduce the normalized data
    data_2 = DeconvolutionData(x, y)
    data_2.x.apply_normalization(data.x.mean, data.x.scale)
    data_2.flatten()
    np.testing.assert_allclose(data_2.x.flat, data.x.flat)
    # the normalization is reported even though no error norm was used
    y_mean, y_scale, x_mean_2, x_scale_2 = data_2.data_scale_ndvars()
    assert y_mean is None and y_scale is None
    assert x_mean_2 == pytest.approx(x_mean + x_scale) and x_scale_2 == pytest.approx(x_scale * 2)


def test_lag_matrix():
    x = np.arange(10, dtype=float)
    lagged = lag_matrix(x, 0, 3, -1)
    assert lagged.shape == (10, 3)
    assert_array_equal(lagged[:, 0], x)
    assert_array_equal(lagged[:, 1], [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert_array_equal(lagged[:, 2], [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    # negative start: the kernel starts before the stimulus
    lagged = lag_matrix(x, -2, 4, -1)
    assert_array_equal(lagged[:, 0], [2, 3, 4, 5, 6, 7, 8, 9, -1, -1])
    assert_array_equal(lagged[:, 3], [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    # shifting the window corresponds to shifting the rows
    rng = np.random.RandomState(0)
    x = rng.normal(size=200)
    assert_array_equal(lag_matrix(x, -20, 41)[:-20], lag_matrix(x, 0, 41)[20:])
    # consistent with convolve_1d
    h = rng.normal(size=(1, 4))
    y_pred = np.empty(200)
    convolve_1d(h, x[np.newaxis], np.zeros(1), -2, np.array([[0, 200]]), y_pred)
    np.testing.assert_allclose(lag_matrix(x, -2, 4) @ h[0], y_pred)


def test_split_data_unequal_partitions():
    "Partitions that are not a multiple of the number of segments: equal sample counts, cut at segment boundaries"
    segments = np.array([[0, 100], [100, 200]])
    splits = split_data(segments, 3, validate=1, test=0)
    assert len(splits.splits) == 3
    assert_array_equal(splits.split_segments, [[0, 67], [67, 100], [100, 133], [133, 200]])
    assert_array_equal(splits.splits[0].validate, [[0, 67]])
    assert_array_equal(splits.splits[0].train, [[67, 100], [100, 200]])
    assert_array_equal(splits.splits[1].validate, [[67, 100], [100, 133]])
    assert_array_equal(splits.splits[1].train, [[0, 67], [133, 200]])
    assert_array_equal(splits.splits[2].validate, [[133, 200]])
    assert_array_equal(splits.splits[2].train, [[0, 100], [100, 133]])
    # a split point on a segment boundary is not a soft split
    splits = split_data(np.array([[0, 100], [100, 300]]), 3, validate=1, test=0)
    assert_array_equal(splits.split_segments, [[0, 100], [100, 200], [200, 300]])
    assert_array_equal(splits.splits[1].validate, [[100, 200]])
    assert_array_equal(splits.splits[1].train, [[0, 100], [200, 300]])
    assert_array_equal(splits.splits[0].train, [[100, 300]])


def test_splits_eq():
    segments = np.array([[0, 100]])
    splits = split_data(segments, 5)
    assert splits == split_data(segments, 5)
    assert splits != split_data(segments, 4)
    assert splits.splits[0] == pickle.loads(pickle.dumps(splits.splits[0]))
    assert splits.splits[0] != splits.splits[1]
    # with model
    segments = np.array([[i * 10, (i + 1) * 10] for i in range(6)])
    model = Factor('ab', repeat=3)
    splits = split_data(segments, 3, model)
    assert splits == split_data(segments, 3, model)
    assert splits == pickle.loads(pickle.dumps(splits))
    assert splits != split_data(segments, 3, Factor('ab', tile=3))
    assert splits != split_data(segments, 3)
    assert split_data(segments, 3) != splits
