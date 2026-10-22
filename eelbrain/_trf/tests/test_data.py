# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import Factor, datasets, epoch_impulse_predictor
from eelbrain._trf.shared import DeconvolutionData


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
