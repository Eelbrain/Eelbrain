# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import scipy.stats

from eelbrain import datasets, Model, Var
from eelbrain._stats import stats
from eelbrain._stats.permutation import permute_order, rand_rotation_matrices
from eelbrain.testing.r_bridge import r


def test_corr():
    "Test stats.corr"
    ds = datasets.get_uts()
    y = ds.eval("uts.x[:,:3]")
    x = ds.eval('Y.x')
    n_cases = len(y)
    df = n_cases - 2

    corr = stats.corr(y, x)
    p = stats.rtest_p(corr, df)
    for i in range(len(corr)):
        r_sp, p_sp = scipy.stats.pearsonr(y[:, i], x)
        assert corr[i] == pytest.approx(r_sp)
        assert p[i] == pytest.approx(p_sp)

    # NaN
    with warnings.catch_warnings():  # divide by 0
        warnings.simplefilter("ignore")
        assert stats.corr(np.arange(10), np.zeros(10)) == 0

    # perm
    y_perm = np.empty_like(y)
    for perm in permute_order(n_cases, 2):
        y_perm[perm] = y
        stats.corr(y, x, corr, perm)
        for i in range(len(corr)):
            r_sp, _ = scipy.stats.pearsonr(y_perm[:, i], x)
            assert corr[i] == pytest.approx(r_sp)


def test_lm():
    "Test linear model function against scipy lstsq"
    ds = datasets.get_uts(True)
    uts = ds['uts']
    utsnd = ds['utsnd']
    x = ds.eval("A*B")
    p = x._parametrize()
    n = ds.n_cases

    # 1d betas
    betas = stats.lm_betas(uts.x, p)
    sp_betas = scipy.linalg.lstsq(p.x, uts.x.reshape((n, -1)))[0]
    assert_allclose(betas, sp_betas)

    # 2d betas
    betas = stats.lm_betas(utsnd.x, p)
    sp_betas = scipy.linalg.lstsq(p.x, utsnd.x.reshape((n, -1)))[0]
    sp_betas = sp_betas.reshape((x.df,) + utsnd.shape[1:])
    assert_allclose(betas, sp_betas)


def test_lm_r():
    "Test linear model agains R"
    ds = datasets.get_uv()
    ds['A2'] = Var(ds['A'] == 'a2').astype(float)
    ds['B2'] = Var(ds['B'] == 'b2').astype(float)
    ds.to_r('ds')
    y = ds['fltvar'].x[:, None]

    p = Model([ds['A2']])._parametrize('dummy')
    b, se, t = stats.lm_t(y, p)
    r_res = r("summary(lm(fltvar ~ A2, ds, type=2))")
    assert_allclose(b[:, 0], r_res[3][0:2])
    assert_allclose(se[:, 0], r_res[3][2:4])
    assert_allclose(t[:, 0], r_res[3][4:6])

    p = ds.eval("A2 + B2")._parametrize('dummy')
    b, se, t = stats.lm_t(y, p)
    r_res = r("summary(lm(fltvar ~ A2 + B2, ds, type=2))")
    assert_allclose(b[:, 0], r_res[3][0:3])
    assert_allclose(se[:, 0], r_res[3][3:6])
    assert_allclose(t[:, 0], r_res[3][6:9])

    p = ds.eval("A2 + B2 + intvar")._parametrize('dummy')
    b, se, t = stats.lm_t(y, p)
    r_res = r("summary(lm(fltvar ~ A2 + B2 + intvar, ds, type=2))")
    assert_allclose(b[:, 0], r_res[3][0:4])
    assert_allclose(se[:, 0], r_res[3][4:8])
    assert_allclose(t[:, 0], r_res[3][8:12])


def test_dispersion():
    "Test variability functions"
    ds = datasets.get_loftus_masson_1994()
    y = ds['n_recalled'].x.astype(np.float64)
    x = ds['exposure'].as_factor()
    match = ds['subject']

    sem = scipy.stats.sem(y, 0, 1)
    ci = sem * scipy.stats.t.isf(0.05 / 2., len(y) - 1)

    # invalid spec
    with pytest.raises(ValueError):
        stats.dispersion(y, 0, 0, '1mile')
    with pytest.raises(ValueError):
        stats.dispersion(y, 0, 0, 'ci7ci')

    # standard error
    assert stats.dispersion(y, None, None, 'sem') == sem
    assert stats.dispersion(y, None, None, '2sem') == 2 * sem
    # within subject standard-error
    target = scipy.stats.sem(stats.residuals(y[:, None], match), 0, len(match.cells))[0]
    assert stats.dispersion(y, None, match, 'sem') == pytest.approx(target)
    # one data point per match cell
    n = match.df + 1
    with pytest.raises(ValueError):
        stats.dispersion(y[:n], None, match[:n], 'sem')

    target = np.array([scipy.stats.sem(y[x == cell], 0, 1) for cell in x.cells])
    es = stats.dispersion(y, x, None, 'sem')
    assert_allclose(es, target)

    stats.dispersion(y, x, None, 'sem', pool=True)

    # confidence intervals
    assert stats.dispersion(y, None, None, '95%ci') == pytest.approx(ci)
    assert stats.dispersion(y, x, None, '95%ci', pool=True) == pytest.approx(3.86, abs=1e-2)  # L&M: 3.85
    assert stats.dispersion(y, x, match, '95%ci') == pytest.approx(0.52, abs=1e-2)

    assert_equal(
        stats.dispersion(y, x, None, '95%ci')[::-1],
        stats.dispersion(y, x, None, '95%ci', x.cells[::-1])
    )


def test_t_1samp():
    "Test 1-sample t-test"
    ds = datasets.get_uts(True)

    y = ds['uts'].x
    t = scipy.stats.ttest_1samp(y, 0, 0)[0]
    assert_allclose(stats.t_1samp(y), t, 10)

    y = ds['utsnd'].x
    t = scipy.stats.ttest_1samp(y, 0, 0)[0]
    assert_allclose(stats.t_1samp(y), t, 10)


def test_t_ind():
    "Test independent samples t-test"
    ds = datasets.get_uts(True)
    y = ds.eval("utsnd.x")
    n_cases = len(y)
    n = n_cases // 2
    groups = (np.arange(n_cases) < n)
    groups.dtype = np.int8

    t = stats.t_ind(y, groups)
    p = stats.ttest_p(t, n_cases - 2)
    t_sp, p_sp = scipy.stats.ttest_ind(y[:n], y[n:])
    assert_allclose(t, t_sp)
    assert_allclose(p, p_sp)
    assert_allclose(stats.ttest_t(p, n_cases - 2), np.abs(t))

    # permutation
    y_perm = np.empty_like(y)
    for perm in permute_order(n_cases, 2):
        stats.t_ind(y, groups, out=t, perm=perm)
        y_perm[perm] = y
        t_sp, _ = scipy.stats.ttest_ind(y_perm[:n], y_perm[n:])
        assert_allclose(t, t_sp)


def test_vector():
    rotation = rand_rotation_matrices(4, 0)
    zeros = np.zeros((4, 3))
    assert stats.t2_1samp(zeros) == 0
    assert stats.t2_1samp(zeros, rotation) == 0
