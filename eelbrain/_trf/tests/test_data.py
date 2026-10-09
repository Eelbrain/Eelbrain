# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import Factor, datasets, epoch_impulse_predictor
from eelbrain._ndvar._convolve import convolve_1d
from eelbrain._trf.shared import DeconvolutionData, lag_matrix, split_data


def test_deconvolution_data_continuous():
    ds = datasets._get_continuous(ynd=True)

    # normalizing
    data = DeconvolutionData(ds['y'][:8], ds['x1'][:8])
    data.normalize('l1')
    assert data.x[0].mean() == pytest.approx(0, abs=1e-16)
    assert abs(data.x).mean() == pytest.approx(1, abs=1e-10)
    with pytest.raises(ValueError):
        data.apply_basis(0.050, 'hamming')
    # with basis
    data = DeconvolutionData(ds['y'][:8], ds['x1'][:8])
    data.apply_basis(0.500, 'hamming')
    data.normalize('l1')
    assert data.x[0].mean() == pytest.approx(0, abs=1e-16)
    assert abs(data.x).mean() == pytest.approx(1, abs=1e-10)

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

    data = DeconvolutionData('uts', 'imp', ds)
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
    data = DeconvolutionData(y, [x, 'uts'], ds)
    assert len(data.y_segments) == len(data.x_segments) == 60
    assert data.n_y == 5 and data.n_x == 2
    # y segments are views into the NDVar
    assert np.shares_memory(data.y_segments[0], y.x)
    assert_array_equal(data.y_segments[3], y.x[3])
    assert data.segment_slices(np.array([[0, 10], [95, 215]])) == [(0, 0, 10), (0, 95, 100), (1, 0, 100), (2, 0, 15)]
    # modifying does not touch the input
    y_copy = y.x.copy()
    data.normalize('l2')
    assert_array_equal(y.x, y_copy)
    # the flat array holds the same data, and the segments become views into it
    assert data.y.shape == (5, 6000)
    assert_array_equal(data.y[:, 300:400], data.y_segments[3])
    assert np.shares_memory(data.y, data.y_segments[3])
    data.y_segments[3][0, 0] = 99
    assert data.y[0, 300] == 99
    # pickling drops the flat arrays, the segments keep the data
    data_2 = pickle.loads(pickle.dumps(data))
    assert data_2._y_flat is None
    assert_array_equal(data_2.y, data.y)
    assert_array_equal(data_2.x_mean, data.x_mean)

    # in-place normalization modifies the input
    x_copy = x.x.copy()
    data = DeconvolutionData(y, x, in_place=True)
    data.normalize('l1')
    assert_array_equal(data.x_segments[0][0], x.x[0])
    assert not np.array_equal(x.x, x_copy)
    # for a single predictor, the flat array is a view into the input (no copy)
    assert np.shares_memory(data.x, x.x)
    assert_array_equal(data.x[0, 100:200], x.x[1])
    x.x[:] = x_copy


def test_deconvolution_data_check():
    "Flat and NaN data are detected across segments, regardless of DC offset"
    ds = datasets.get_uts(True)
    y = ds['utsnd'].copy()
    x = ds['uts'].copy()
    DeconvolutionData(y, x)._check_data()
    y.x[:, 1, :] = 0.1
    with pytest.raises(ValueError, match="1 flat time series"):
        DeconvolutionData(y, x)._check_data()
    y = ds['utsnd']
    x.x[3, 10] = np.nan
    with pytest.raises(ValueError, match="NaN in"):
        DeconvolutionData(y, x)._check_data()


def test_deconvolution_data_normalize_x():
    "Normalization can be restricted to x, and applied with given values"
    ds = datasets._get_continuous(ynd=True)
    y, x = ds['y'], ds['x1']
    data = DeconvolutionData(y, x)
    data.normalize('l2', y=False)
    assert data.y_mean is None
    assert_array_equal(data.y_segments[0][0], y.x)
    x_mean, x_scale = x.mean('time'), x.std('time')
    assert data.x_mean[0] == pytest.approx(x_mean)
    assert data.x_scale[0] == pytest.approx(x_scale)
    assert data.x_pads[0] == pytest.approx(-x_mean / x_scale)
    # applying a second normalization composes with the first
    data.apply_x_normalization(1., 2.)
    assert data.x_mean[0] == pytest.approx(x_mean + x_scale)
    assert data.x_scale[0] == pytest.approx(x_scale * 2)
    np.testing.assert_allclose(data.x[0], ((x.x - x_mean) / x_scale - 1) / 2)
    # the recorded values reproduce the normalized data
    data_2 = DeconvolutionData(y, x)
    data_2.apply_x_normalization(data.x_mean, data.x_scale)
    np.testing.assert_allclose(data_2.x, data.x)


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
