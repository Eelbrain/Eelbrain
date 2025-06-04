# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
import scipy.stats
from numpy.testing import assert_allclose
from eelbrain import datasets
from eelbrain._stats import opt
from eelbrain._stats.permutation import permute_sign_flip


def test_t_1samp():
    "Test t_1samp functions"
    ds = datasets.get_uts()
    y = ds.eval("uts.x")
    n_cases = len(y)
    t = np.empty(y.shape[1])

    # t
    opt.t_1samp(y, t)
    t_sp, _ = scipy.stats.ttest_1samp(y, 0)
    assert_allclose(t, t_sp)

    # perm
    t_perm = np.empty_like(t)
    for sign in permute_sign_flip(n_cases, 2):
        opt.t_1samp_perm(y, t_perm, sign)
        opt.t_1samp(y * sign[:, None], t)
        assert_allclose(t_perm, t)
