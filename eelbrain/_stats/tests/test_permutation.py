# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import sys

import numpy as np
import pytest

from eelbrain import Factor, Var
from eelbrain._stats.permutation import (
    resample, permute_order, permute_sign_flip, rand_rotation_matrices)


def test_permutation():
    """Test permutation"""
    v = Var(np.arange(6))
    res = np.empty((5, 6))
    for i, y in enumerate(resample(v, samples=5)):
        res[i] = y.x

    # with unit
    s = Factor('abc', tile=2)
    for i, y in enumerate(resample(v, samples=5, unit=s)):
        res[i] = y.x

    # check we have only appropriate cells
    cols = [np.unique(res[:, i]) for i in range(res.shape[1])]
    for i in range(3):
        assert len(np.setdiff1d(cols[i], [i, i + 3])) == 0
    for i in range(3, 6):
        assert len(np.setdiff1d(cols[i], [i, i - 3])) == 0

    # check we have some variability
    assert max(map(len, cols)) == 2

    # make sure sequence is stable
    assert list(map(tuple, permute_order(4, 3))) == [(2, 3, 1, 0), (2, 1, 3, 0), (0, 2, 3, 1)]


def test_permutation_sign_flip():
    "Test permute_sign_flip()"
    res = np.empty((2 ** 6 - 1, 6), dtype=np.int8)
    i = 0
    for i, sign in enumerate(permute_sign_flip(6, samples=-1)):
        res[i] = sign
    assert i == 2 ** 6 - 2
    assert np.all(res.min(1) < 0)  # Check that all permutations have a sign flip
    for i, row in enumerate(res):
        assert not np.any(np.all(row == res[:i], 1))

    # n > 62
    res = list(map(tuple, permute_sign_flip(66, 2)))
    assert len(res[0]) == 66
    assert len(res) == 2
    assert res[0] != res[1]

    # make sure sequence is stable
    if sys.version_info[0] == 3:
        target = [(1, -1, -1, -1), (-1, -1, -1, 1), (-1, 1, -1, -1)]
    else:
        target = [(-1, 1, -1, -1), (-1, -1, 1, -1), (1, -1, -1, 1)]
    assert list(map(tuple, permute_sign_flip(4, 3))) == target


def test_rand_rotation_matrices():
    seed = 12345
    with pytest.raises(NotImplementedError):
        rand_rotation_matrices(10, seed, 4)
    with pytest.raises(ValueError):
        rand_rotation_matrices(10, seed, 1)
    for n in (2, 3):
        rot_mat = rand_rotation_matrices(10, seed, 2)
        np.testing.assert_allclose(np.linalg.det(rot_mat), 1)
