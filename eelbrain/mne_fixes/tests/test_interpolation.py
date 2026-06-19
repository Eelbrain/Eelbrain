# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal

from mne.channels.interpolation import _make_interpolation_matrix as mne_make_interpolation_matrix

from eelbrain.mne_fixes._interpolation import _make_interpolation_matrix


def test_make_interpolation_matrix():
    "Vendored spherical-spline kernel matches MNE-Python's implementation"
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((20, 3)) + [0, 0, 0.05]
    assert_array_equal(
        _make_interpolation_matrix(pos[:15], pos[15:]),
        mne_make_interpolation_matrix(pos[:15].copy(), pos[15:].copy()))
