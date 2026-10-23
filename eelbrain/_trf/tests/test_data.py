# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import Factor, NDVar, Scalar, Space, datasets, epoch_impulse_predictor
from eelbrain._ndvar._convolve import convolve_1d
from eelbrain._trf.shared import DeconvolutionData, Split, Splits, lag_matrix, lag_window, package_values, split_data


def test_deconvolution_data_continuous():
    ds = datasets._get_continuous(ynd=True)

    # normalizing
    data = DeconvolutionData(ds['x1'][:8], ds['y'][:8])
    with pytest.raises(ValueError, match="need 'l1' or 'l2'"):
        data.normalize('L1')
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
    assert not any(split.is_test or split.splits for split in data.splits.splits)
    assert_array_equal(data.splits.splits[0].reserved, [[0, 20]])
    assert_array_equal(data.splits.splits[0].train, [[20, 80]])
    assert_array_equal(data.splits.splits[1].reserved, [[20, 40]])
    assert_array_equal(data.splits.splits[1].train, [[0, 20], [40, 80]])
    assert_array_equal(data.splits.splits[2].reserved, [[40, 60]])
    assert_array_equal(data.splits.splits[2].train, [[0, 40], [60, 80]])
    assert_array_equal(data.splits.splits[3].reserved, [[60, 80]])
    assert_array_equal(data.splits.splits[3].train, [[0, 60]])

    # partitioning, testing set
    data.initialize_cross_validation(4, test=1)
    assert len(data.splits.splits) == 4
    assert all(split.is_test and len(split.splits) == 3 for split in data.splits.splits)
    assert [split.i_reserved for split in data.splits.splits] == [0, 1, 2, 3]
    # test partition 0
    test_split = data.splits.splits[0]
    assert_array_equal(test_split.reserved, [[0, 20]])
    assert_array_equal(test_split.train, [[20, 80]])
    assert [split.i_reserved for split in test_split.splits] == [1, 2, 3]
    assert not any(split.is_test for split in test_split.splits)
    # 0/1
    assert_array_equal(test_split.splits[0].reserved, [[20, 40]])
    assert_array_equal(test_split.splits[0].train, [[40, 80]])
    # 0/2
    assert_array_equal(test_split.splits[1].reserved, [[40, 60]])
    assert_array_equal(test_split.splits[1].train, [[20, 40], [60, 80]])
    # 0/3
    assert_array_equal(test_split.splits[2].reserved, [[60, 80]])
    assert_array_equal(test_split.splits[2].train, [[20, 60]])
    # test partition 1
    test_split = data.splits.splits[1]
    assert_array_equal(test_split.reserved, [[20, 40]])
    assert_array_equal(test_split.train, [[0, 20], [40, 80]])
    assert [split.i_reserved for split in test_split.splits] == [0, 2, 3]
    # 1/0
    assert_array_equal(test_split.splits[0].reserved, [[0, 20]])
    assert_array_equal(test_split.splits[0].train, [[40, 80]])
    # 1/2
    assert_array_equal(test_split.splits[1].reserved, [[40, 60]])
    assert_array_equal(test_split.splits[1].train, [[0, 20], [60, 80]])
    # test partition 3, validation partition 2
    test_split = data.splits.splits[3]
    assert_array_equal(test_split.reserved, [[60, 80]])
    assert_array_equal(test_split.train, [[0, 60]])
    assert_array_equal(test_split.splits[2].reserved, [[40, 60]])
    assert_array_equal(test_split.splits[2].train, [[0, 40]])


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
        assert_array_equal(split.reserved, data.segments[validate_index])
        assert_array_equal(split.train, data.segments[~validate_index])

    # continuoue model
    data.initialize_cross_validation(3, 'A', ds, test=0)
    assert len(data.splits.splits) == 3
    for i, split in enumerate(data.splits.splits):
        validate_index = arange == i
        assert_array_equal(split.reserved, data.segments[validate_index])
        assert_array_equal(split.train, data.segments[~validate_index])

    # alternating model
    ds['C'] = Factor('abc', tile=20)
    data.initialize_cross_validation(3, 'C', ds, test=0)
    assert len(data.splits.splits) == 3
    arange = np.repeat(np.arange(20), 3) % 3
    for i, split in enumerate(data.splits.splits):
        validate_index = arange == i
        assert_array_equal(split.reserved, data.segments[validate_index])
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
    assert data.segment_n_times == [100] * 60
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
    with pytest.raises(ValueError, match="flat data in utsnd-1"):
        DeconvolutionData(x, y).check_data()
    y = ds['utsnd']
    x.x[3, 10] = np.nan
    with pytest.raises(ValueError, match="NaN in"):
        DeconvolutionData(x, y).check_data()
    # vector data: the scale has one entry per vector
    x = ds['uts']
    y = NDVar(np.random.RandomState(0).normal(size=(60, 2, 3, 100)), (x.dims[0], Scalar('s', [0, 1]), Space('RAI'), x.time), 'y')
    y.x[:, 1] = 0
    data = DeconvolutionData(x, y)
    # normalizing checks before modifying the data (also x when the check of y fails)
    with pytest.raises(ValueError, match="flat data in y-1$"):
        data.normalize('l2')
    assert data.y.mean is None and data.y.scale_norm is None
    assert_array_equal(data.y.segments[0], y.x[0].reshape((6, 100)))
    assert data.x.mean is None and data.x.scale_norm is None
    assert_array_equal(data.x.segments[0], x.x[:1])
    x.x[3, 10] = np.nan
    with pytest.raises(ValueError, match="NaN in uts"):
        DeconvolutionData(x, y).x.normalize(True, 'l1')
    # a scale supplied by the user does not vouch for the data
    data = DeconvolutionData(x, y)
    data.x.normalize(0., 1.)
    with pytest.raises(ValueError, match="NaN in uts"):
        data.check_data()
    x.x[:] = 1
    data = DeconvolutionData(x, y)
    data.x.normalize(0., 1.)
    with pytest.raises(ValueError, match="flat data in uts"):
        data.check_data()


def test_deconvolution_data_normalize_x():
    "Normalization can be restricted to x, and applied with given values"
    ds = datasets._get_continuous(ynd=True)
    y, x = ds['y'], ds['x1']
    data = DeconvolutionData(x, y)
    data.x.normalize(True, 'l2')
    assert data.y.mean is None
    assert_array_equal(data.y.segments[0][0], y.x)
    x_mean, x_scale = x.mean('time'), x.std('time')
    assert data.x.mean[0] == pytest.approx(x_mean)
    assert data.x.scale[0] == pytest.approx(x_scale)
    assert data.x.pads[0] == pytest.approx(-x_mean / x_scale)
    # applying a second normalization composes with the first
    assert data.x.scale_norm == 'l2'
    data.x.normalize(1., 2.)
    assert data.x.mean[0] == pytest.approx(x_mean + x_scale)
    assert data.x.scale[0] == pytest.approx(x_scale * 2)
    assert data.x.scale_norm is None
    data.flatten()
    np.testing.assert_allclose(data.x.flat[0], ((x.x - x_mean) / x_scale - 1) / 2)
    # the recorded values reproduce the normalized data
    data_2 = DeconvolutionData(x, y)
    data_2.x.normalize(data.x.mean, data.x.scale)
    data_2.flatten()
    np.testing.assert_allclose(data_2.x.flat, data.x.flat)
    # the normalization is reported even though no error norm was used
    y_mean, y_scale, x_mean_2, x_scale_2 = data_2.data_scale_ndvars()
    assert y_mean is None and y_scale is None
    assert x_mean_2 == pytest.approx(x_mean + x_scale) and x_scale_2 == pytest.approx(x_scale * 2)
    # scaling without centering
    data_3 = DeconvolutionData(x, y)
    data_3.x.normalize(False, 'l1')
    assert_array_equal(data_3.x.mean, [0])
    assert data_3.x.scale[0] == pytest.approx(abs(x.x).mean())
    assert data_3.x.pads[0] == 0
    np.testing.assert_allclose(data_3.x.segments[0][0], x.x / abs(x.x).mean())


def test_deconvolution_data_vector_only():
    "Packaging values for y whose only non-time dimension is a vector dimension"
    ds = datasets._get_continuous()
    x = ds['x1']
    rng = np.random.RandomState(0)
    space = Space('RA')
    y = NDVar(rng.normal(size=(2, len(x))), (space, x.time), 'y')
    data = DeconvolutionData(x, y)
    data.normalize('l2')
    y_mean, y_scale, _, _ = data.data_scale_ndvars()
    assert y_mean.dims == (space,)
    np.testing.assert_allclose(y_mean.x, y.mean('time').x)
    assert np.isscalar(y_scale)
    # per-row values keep the vector dimension, per-vector values are scalar
    value = data.y.package_values(np.arange(2.), 'value')[0]
    assert value.dims == (space,)
    assert_array_equal(value.x, [0, 1])
    assert data.y.package_values(np.arange(1.), 'value')[0] == 0
    # packaging from the blocks alone, without the array
    assert package_values(data.y.blocks, np.arange(1.), 'value', n_vector=2)[0] == 0
    assert_array_equal(package_values(data.y.blocks, np.arange(2.), 'value', n_vector=2)[0].x, [0, 1])


def test_row_block_equality():
    "Blocks identify rows by name and dimensions; the NDVar info does not matter"
    ds = datasets._get_continuous()
    x = ds['x1']
    data = DeconvolutionData([x, ds['x2']], ds['y'])
    x_other = NDVar(x.x, x.dims, x.name, {'unit': 'other'})
    data_other = DeconvolutionData([x_other, ds['x2']], ds['y'])
    assert data_other.x.blocks == data.x.blocks
    assert data_other.y.blocks == data.y.blocks
    assert DeconvolutionData([ds['x2'], x], ds['y']).x.blocks != data.x.blocks
    assert DeconvolutionData(x, ds['y']).x.blocks != data.x.blocks


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


def test_lag_window():
    "Lag windows in samples: tstop is exclusive, tmax inclusive"
    assert lag_window(0, 0.5, 0.01, 1) == ([0], [50])
    assert lag_window(0, None, 0.01, 1, tmax=0.49) == ([0], [50])
    assert lag_window(-0.1, [0.2, 0.3], 0.01, 2) == ([-10, -10], [20, 30])
    assert lag_window([0, 0.1], 0.2, 0.01, 2) == ([0, 10], [20, 20])
    # a window that ends between samples includes the last lag before tstop
    assert lag_window(0, 0.505, 0.01, 1) == ([0], [51])
    # floating point error does not add a lag
    assert lag_window(0, 0.1 + 0.02, 0.01, 1) == ([0], [12])
    assert lag_window(0, 0.3, 0.005, 1) == ([0], [60])
    with pytest.raises(ValueError, match="need one value per predictor"):
        lag_window([0], 0.2, 0.01, 2)
    with pytest.raises(ValueError, match="empty lag window"):
        lag_window(0.2, 0.2, 0.01, 1)
    with pytest.raises(TypeError):
        lag_window(0, 0.2, 0.01, 1, tmax=0.19)


def test_trf_module():
    "Public module for the shared components"
    from eelbrain import trf
    assert trf.DeconvolutionData is DeconvolutionData
    assert trf.lag_window is lag_window
    assert isinstance(trf.n_workers(), int)


def test_split_data_time():
    "split='time': equal sample counts, cut at segment boundaries"
    segments = np.array([[0, 100], [100, 200]])
    splits = split_data(segments, 3, validate=1, test=0, split='time')
    assert splits.split == 'time'
    assert len(splits.splits) == 3
    assert_array_equal(splits.split_segments, [[0, 67], [67, 100], [100, 133], [133, 200]])
    assert_array_equal(splits.splits[0].reserved, [[0, 67]])
    assert_array_equal(splits.splits[0].train, [[67, 100], [100, 200]])
    assert_array_equal(splits.splits[1].reserved, [[67, 100], [100, 133]])
    assert_array_equal(splits.splits[1].train, [[0, 67], [133, 200]])
    assert_array_equal(splits.splits[2].reserved, [[133, 200]])
    assert_array_equal(splits.splits[2].train, [[0, 100], [100, 133]])
    # a split point on a segment boundary is not a soft split
    splits = split_data(np.array([[0, 100], [100, 300]]), 3, validate=1, test=0, split='time')
    assert_array_equal(splits.split_segments, [[0, 100], [100, 200], [200, 300]])
    assert_array_equal(splits.splits[1].reserved, [[100, 200]])
    assert_array_equal(splits.splits[1].train, [[0, 100], [200, 300]])
    assert_array_equal(splits.splits[0].train, [[100, 300]])
    # with more segments than partitions: contiguous blocks of segments
    segments = np.array([[i * 10, (i + 1) * 10] for i in range(6)])
    splits = split_data(segments, 3, validate=1, test=0, split='time')
    assert_array_equal(splits.splits[0].reserved, [[0, 10], [10, 20]])
    # default for continuous data
    assert split_data(np.array([[0, 100]]), 3).split == 'time'
    # more partitions than samples
    with pytest.raises(ValueError, match="not enough samples"):
        split_data(np.array([[0, 2], [2, 4]]), 5, validate=1, test=0, split='time')
    with pytest.raises(ValueError, match="not enough samples"):  # continuous data
        split_data(np.array([[0, 3]]), 5, validate=1, test=0)


def test_split_data_cases():
    "split='cases' (default with segments) never cuts segments"
    segments = np.array([[0, 100], [100, 200]])
    with pytest.raises(ValueError, match="only 2 cases"):
        split_data(segments, 3, validate=1, test=0)
    with pytest.raises(ValueError, match="only 2 cases"):  # a multiple of the number of segments
        split_data(segments, 4, validate=1, test=0)
    with pytest.raises(TypeError, match="model can only be used"):
        split_data(segments, 2, Factor('ab'), validate=1, test=0, split='time')
    with pytest.raises(ValueError, match="split='trials'"):
        split_data(segments, 2, test=0, split='trials')


def test_split_data_no_validate_no_test():
    "Without validation and test set, a single split uses all data for training"
    segments = np.array([[0, 100], [100, 200]])
    splits = split_data(segments, 3, validate=0, test=0, split='time')
    assert len(splits.splits) == 1
    split = splits.splits[0]
    assert_array_equal(split.train, [[0, 100], [100, 200]])  # the segment boundary is hard
    assert split.reserved is None
    assert split.i_reserved is None
    assert split.splits == []


def test_split_data_within_cases():
    "Each segment is cut into ``partitions`` pieces, and partition i pools the i-th piece of every segment"
    segments = np.array([[0, 100], [100, 160]])
    splits = split_data(segments, 3, validate=1, test=0, split='within_cases')
    assert splits.split == 'within_cases'
    assert splits.n_partitions == 3
    assert_array_equal(splits.split_segments, [[0, 33], [33, 67], [67, 100], [100, 120], [120, 140], [140, 160]])
    assert_array_equal(splits.splits[0].reserved, [[0, 33], [100, 120]])
    assert_array_equal(splits.splits[0].train, [[33, 100], [120, 160]])
    assert_array_equal(splits.splits[1].reserved, [[33, 67], [120, 140]])
    assert_array_equal(splits.splits[1].train, [[0, 33], [67, 100], [100, 120], [140, 160]])
    assert_array_equal(splits.splits[2].reserved, [[67, 100], [140, 160]])
    assert_array_equal(splits.splits[2].train, [[0, 67], [100, 140]])
    assert "split='within_cases'" in repr(splits)
    assert splits != split_data(segments, 3, validate=1, test=0, split='time')
    assert splits == pickle.loads(pickle.dumps(splits))
    # default number of partitions
    assert split_data(segments, split='within_cases').n_partitions == 5
    # continuous data is unaffected
    assert split_data(segments[:1], 3, split='within_cases').splits == split_data(segments[:1], 3).splits
    # also with enough segments for whole-segment partitions
    segments = np.array([[i * 10, (i + 1) * 10] for i in range(6)])
    splits = split_data(segments, 2, validate=1, test=0, split='within_cases')
    assert_array_equal(splits.splits[0].reserved, [[0, 5], [10, 15], [20, 25], [30, 35], [40, 45], [50, 55]])
    with pytest.raises(ValueError, match="not enough samples"):
        split_data(np.array([[0, 2], [2, 4]]), 3, validate=1, test=0, split='within_cases')


def test_splits_legacy_format():
    "Load Splits pickled before nested splits (one flat Split per boosting run)"
    def flat_split(state: dict) -> Split:
        split = Split.__new__(Split)
        split.__setstate__(state)
        return split

    def legacy_splits(splits: Splits, flat: list) -> Splits:
        legacy = Splits.__new__(Splits)
        legacy.__setstate__({**splits.__getstate__(), 'splits': flat})
        return legacy

    segments = np.array([[0, 100], [100, 200]])
    # with test set
    splits = split_data(segments, 4, split='time')
    states = [{'train': split.train, 'validate': split.reserved, 'test': test_split.reserved, 'i_test': test_split.i_reserved, 'i_validate': split.i_reserved} for test_split in splits.splits for split in test_split.splits]
    assert len(states) == 12
    assert legacy_splits(splits, [flat_split(state) for state in states]) == splits
    # before BoostingResult version 16, i_validate was not stored
    assert legacy_splits(splits, [flat_split({k: v for k, v in state.items() if k != 'i_validate'}) for state in states]) == splits
    # without test set
    splits = split_data(segments, 4, test=0, split='time')
    states = [{'train': split.train, 'validate': split.reserved, 'test': None, 'i_test': 0, 'i_validate': split.i_reserved} for split in splits.splits]
    assert legacy_splits(splits, [flat_split(state) for state in states]) == splits
    assert legacy_splits(splits, [flat_split({k: v for k, v in state.items() if k != 'i_validate'}) for state in states]) == splits


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
