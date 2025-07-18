# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import repeat

import numpy as np
from numpy import newaxis
from numpy.testing import assert_allclose
import pytest

from eelbrain import datasets, load, test, testnd, Dataset, Factor, NDVar, Var
from eelbrain._data_obj import UTS
from eelbrain._exceptions import IncompleteModelError
from eelbrain._stats import glm
from eelbrain._stats.permutation import permute_order
from eelbrain._utils.r_bridge import r, r_require, r_warning_filter
from eelbrain.testing import assert_fmtxt_str_equals, requires_r_ez, file_path


def assert_f_test_equal(f_test, r_res, r_row, f_lmf, f_nd, r_kind='aov'):
    if r_kind in ('aov', 'rmaov'):
        r_res = {'df': r_res[0][r_row], 'SS': r_res[1][r_row],
                 'MS': r_res[2][r_row], 'F': r_res[3][r_row],
                 'p': r_res[4][r_row]}
    elif r_kind == 'ez':
        pass
    elif r_kind == 'Anova':
        r_res = {'df': r_res[1][r_row], 'SS': r_res[0][r_row],
                 'F': r_res[2][r_row], 'p': r_res[3][r_row]}
    else:
        raise ValueError(f"{r_kind=}")

    assert f_test.df == r_res['df']
    if 'SS' in r_res:
        assert f_test.ss == pytest.approx(r_res['SS'])
    if 'MS' in r_res:
        assert f_test.ms == pytest.approx(r_res['MS'])
    assert f_test.f == pytest.approx(r_res['F'])
    assert f_test.p == pytest.approx(r_res['p'])
    assert f_lmf == pytest.approx(r_res['F'])  # lm-fitter comparison"
    assert f_nd == pytest.approx(r_res['F'])  # nd-ANOVA comparison"


def assert_f_tests_equal(f_tests, r_res, fs, fnds, r_kind='aov'):
    if r_kind == 'ez':
        r_results = []
        for f_test in f_tests:
            f_test_name = set(f_test.name.split(' x '))
            for i, r_name in enumerate(r_res[0][0]):
                if set(r_name.split(':')) == f_test_name:
                    r_results.append({'df': r_res[0][1][i], 'F': r_res[0][3][i],
                                      'p': r_res[0][4][i]})
                    break
            else:
                raise RuntimeError("Effect %s not in ezANOVA" % f_test_name)
    elif r_kind == 'rmaov':
        r_results = [r_res[i][0] for i in range(len(f_tests))]
    else:
        r_results = repeat(r_res, len(f_tests))

    r_row = 0
    for i, r_res in enumerate(r_results):
        if r_kind != 'rmaov':
            r_row = i
        assert_f_test_equal(f_tests[i], r_res, r_row, fs[i], fnds[i], r_kind)


def run_on_lm_fitter(y, x, ds):
    y = ds.eval(y)
    y = y.x[:, newaxis]
    y = np.hstack((y, y))
    x = ds.eval(x)
    fitter = glm._nd_anova(x)
    fmaps = fitter.map(y)
    fs = fmaps[:, 0]
    return fs


def run_as_ndanova(y, x, ds):
    yt = ds.eval(y).x[:, None]
    y2 = np.concatenate((yt, yt * 2), 1)
    ndvar = NDVar(y2, ('case', UTS(0, 0.1, 2)))
    res = testnd.ANOVA(ndvar, x, data=ds)
    f1 = [fmap.x[0] for fmap in res.f]
    f2 = [fmap.x[1] for fmap in res.f]
    for f1_, f2_ in zip(f1, f2):
        assert f1_ == f2_
    return f1


def test_anova():
    "Test ANOVA"
    r_require('car')

    ds = datasets.get_uv(nrm=True)
    ds.to_r('ds')

    # fixed effects
    aov = test.ANOVA('fltvar', 'A*B', data=ds)
    assert_fmtxt_str_equals(aov, """
                    SS   df      MS          F        p
    ---------------------------------------------------
    A            28.69    1   28.69   25.69***   < .001
    B             0.04    1    0.04    0.03        .855
    A x B         1.16    1    1.16    1.04        .310
    Residuals    84.85   76    1.12
    ---------------------------------------------------
    Total       114.74   79
    """)
    fs = run_on_lm_fitter('fltvar', 'A*B', ds)
    fnds = run_as_ndanova('fltvar', 'A*B', ds)
    r_res = r("Anova(lm(fltvar ~ A * B, ds, type=2))")
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'Anova')

    # random effect
    aov = test.ANOVA('fltvar', 'A*B*rm', data=ds)
    assert_fmtxt_str_equals(aov, """
                SS   df      MS   MS(denom)   df(denom)          F        p
    -----------------------------------------------------------------------
    A        28.69    1   28.69        1.21          19   23.67***   < .001
    B         0.04    1    0.04        1.15          19    0.03        .859
    A x B     1.16    1    1.16        1.01          19    1.15        .297
    -----------------------------------------------------------------------
    Total   114.74   79
    """)
    fs = run_on_lm_fitter('fltvar', 'A*B*rm', ds)
    fnds = run_as_ndanova('fltvar', 'A*B*rm', ds)
    r('test.aov <- aov(fltvar ~ A * B + Error(rm / (A * B)), ds)')
    print(r('test.summary <- summary(test.aov)'))
    r_res = r['test.summary'][1:]
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'rmaov')

    # Factor.name = None
    with pytest.raises(ValueError):
        test.ANOVA('fltvar', ds['A'] * Factor(ds['B'] == 'b1'), data=ds)


@requires_r_ez
def test_anova_eq():
    "Test ANOVA against r-ez"
    r_require('ez')

    ds = datasets.get_uv(nrm=True)
    ds.to_r('ds')

    # nested random effect
    aov_explicit = test.ANOVA('fltvar', 'A + B + A%B + nrm(B) + A%nrm(B)', data=ds)
    aov = test.ANOVA('fltvar', 'A * B * nrm(B)', data=ds)
    assert str(aov_explicit) == str(aov)
    print(aov)
    fs = run_on_lm_fitter('fltvar', 'A * B * nrm(B)', ds)
    fnds = run_as_ndanova('fltvar', 'A * B * nrm(B)', ds)
    r_res = r('ezANOVA(ds, fltvar, nrm, A, between=B)')
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'ez')

    # sub parameter
    r1 = test.ANOVA('fltvar', 'B * rm', data=ds.sub('A == "a1"'))
    r2 = test.ANOVA('fltvar', 'B * rm', sub='A == "a1"', data=ds)
    assert str(r2) == str(r1)

    # not fully specified model with random effects
    with pytest.raises(IncompleteModelError):
        test.ANOVA('fltvar', 'A*rm', data=ds)

    # unequal group size, 1-way
    sds = ds.sub("A == 'a1'").sub("nrm.isnotin(('s037', 's038', 's039'))")
    sds.to_r('sds')
    aov = test.ANOVA('fltvar', 'B * nrm(B)', data=sds)
    print(aov)
    fs = run_on_lm_fitter('fltvar', 'B * nrm(B)', sds)
    fnds = run_as_ndanova('fltvar', 'B * nrm(B)', sds)
    with r_warning_filter:  # type argument to ezANOVA
        r_res = r('ezANOVA(sds, fltvar, nrm, between=B)')
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'ez')

    # unequal group size, 2-way
    sds = ds.sub("nrm.isnotin(('s037', 's038', 's039'))")
    sds.to_r('sds')
    aov = test.ANOVA('fltvar', 'A * B * nrm(B)', data=sds)
    print(aov)
    fs = run_on_lm_fitter('fltvar', 'A * B * nrm(B)', sds)
    fnds = run_as_ndanova('fltvar', 'A * B * nrm(B)', sds)
    with r_warning_filter:  # type argument to ezANOVA
        r_res = r('ezANOVA(sds, fltvar, nrm, A, between=B)')
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'ez')

    # empty cells
    dss = ds.sub("A%B != ('a2', 'b2')")
    with pytest.raises(NotImplementedError):
        test.ANOVA('fltvar', 'A*B', data=dss)
    with pytest.raises(NotImplementedError):
        run_on_lm_fitter('fltvar', 'A*B', ds=dss)
    dss = ds.sub("A%B != ('a1', 'b1')")
    with pytest.raises(NotImplementedError):
        test.ANOVA('fltvar', 'A*B', data=dss)
    with pytest.raises(NotImplementedError):
        run_on_lm_fitter('fltvar', 'A*B', ds=dss)


def test_ndanova():
    ds = datasets.get_uts(nrm=True)
    ds['An'] = ds['A'].as_var({'a0': 0, 'a1': 1})

    with pytest.raises(NotImplementedError):
        testnd.ANOVA('uts', 'An*B*rm', data=ds)

    # nested random effect
    res = testnd.ANOVA('uts', 'A + A%B + B * nrm(A)', data=ds, match='nrm', samples=100, pmin=0.05)
    assert len(res.find_clusters(0.05)) == 8


def test_anova_perm():
    "Test permutation argument for ANOVA"
    ds = datasets.get_uts()
    y = ds['uts'].x
    y_perm = np.empty_like(y)
    n_cases, n_tests = y.shape

    # balanced ANOVA
    aov = glm._BalancedFixedNDANOVA(ds.eval('A*B'))
    r1 = aov.preallocate(y.shape[1:])
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_allclose(r2, r1, 1e-6, 1e-6)

    # full repeated measures ANOVA
    aov = glm._BalancedMixedNDANOVA(ds.eval('A*B*rm'))
    r1 = aov.preallocate(y.shape[1:])
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_allclose(r2, r1, 1e-6, 1e-6)

    # incremental ANOVA
    ds = ds[1:]
    y = ds['uts'].x
    y_perm = np.empty_like(y)
    n_cases, n_tests = y.shape
    aov = glm._IncrementalNDANOVA(ds.eval('A*B'))
    r1 = aov.preallocate(y.shape[1:])
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_allclose(r2, r1, 1e-6, 1e-6)


@pytest.mark.skip('Rounding error on different platforms')
def test_anova_crawley():
    y = Var([2, 3, 3, 4, 3, 4, 5, 6,
             1, 2, 1, 2, 1, 1, 2, 2,
             2, 2, 2, 2, 1, 1, 2, 3], name="Growth Rate")
    genot = Factor(range(6), repeat=4, name="Genotype")
    hrs = Var([8, 12, 16, 24], tile=6, name="Hours")
    aov = test.ANOVA(y, hrs * genot)
    assert_fmtxt_str_equals(aov, """
                          SS   df     MS          F        p
    --------------------------------------------------------
    Hours               7.06    1   7.06   54.90***   < .001
    Genotype           27.88    5   5.58   43.36***   < .001
    Hours x Genotype    3.15    5   0.63    4.90*       .011
    Residuals           1.54   12   0.13
    --------------------------------------------------------
    Total              39.62   23
    """)


def test_anova_fox():
    data_path = file_path('fox-prestige')
    ds = load.txt.tsv(data_path, delimiter=' ', skipinitialspace=True)
    ds = ds.sub("type != 'NA'")
    aov = test.ANOVA('prestige', '(income + education) * type', data=ds)
    assert_fmtxt_str_equals(aov, """
                             SS   df        MS          F        p
    --------------------------------------------------------------
    income              1131.90    1   1131.90   28.35***   < .001
    education           1067.98    1   1067.98   26.75***   < .001
    type                 591.16    2    295.58    7.40**      .001
    income x type        951.77    2    475.89   11.92***   < .001
    education x type     238.40    2    119.20    2.99        .056
    Residuals           3552.86   89     39.92
    --------------------------------------------------------------
    Total              28346.88   97
    """)


def test_anova_rutherford():
    # ANOVA 1
    y = Var([7, 3, 6, 6, 5, 8, 6, 7,
             7, 11, 9, 11, 10, 10, 11, 11,
             8, 14, 10, 11, 12, 10, 11, 12],
            name='y')
    a = Factor('abc', repeat=8, name='A')
    subject = Factor(list(range(24)), name='subject', random=True)
    aov = test.ANOVA(y, a + subject(a))
    assert_fmtxt_str_equals(aov, """
                SS   df      MS   MS(denom)   df(denom)          F        p
    -----------------------------------------------------------------------
    A       112.00    2   56.00        2.48          21   22.62***   < .001
    -----------------------------------------------------------------------
    Total   164.00   23
    """)
    aov = test.ANOVA(y, a)
    assert_fmtxt_str_equals(aov, """
                    SS   df      MS          F        p
    ---------------------------------------------------
    A           112.00    2   56.00   22.62***   < .001
    Residuals    52.00   21    2.48
    ---------------------------------------------------
    Total       164.00   23
    """)
    subject = Factor(range(8), tile=3, name='subject', random=True)
    aov = test.ANOVA(y, a * subject)
    assert_fmtxt_str_equals(aov, """
                SS   df      MS   MS(denom)   df(denom)          F        p
    -----------------------------------------------------------------------
    A       112.00    2   56.00        2.71          14   20.63***   < .001
    -----------------------------------------------------------------------
    Total   164.00   23
    """)

    # ANCOVA
    y = Var([16, 7, 11, 9, 10, 11, 8, 8,
             16, 10, 13, 10, 10, 14, 11, 12,
             24, 29, 10, 22, 25, 28, 22, 24])
    cov = Var([9, 5, 6, 4, 6, 8, 3, 5,
               8, 5, 6, 5, 3, 6, 4, 6,
               5, 8, 3, 4, 6, 9, 4, 5], name='cov')
    a = Factor([1, 2, 3], repeat=8, name='A')
    aov = test.ANOVA(y, a + cov)
    assert_fmtxt_str_equals(aov, """
                     SS   df       MS          F        p
    -----------------------------------------------------
    A            807.82    2   403.91   62.88***   < .001
    cov          199.54    1   199.54   31.07***   < .001
    Residuals    128.46   20     6.42
    -----------------------------------------------------
    Total       1112.00   23
    """)
    aov = test.ANOVA(y, cov * a)
    assert_fmtxt_str_equals(aov, """
                     SS   df       MS          F        p
    -----------------------------------------------------
    cov          199.54    1   199.54   32.93***   < .001
    A            807.82    2   403.91   66.66***   < .001
    cov x A       19.39    2     9.70    1.60        .229
    Residuals    109.07   18     6.06
    -----------------------------------------------------
    Total       1112.00   23
    """)

    # ANOVA 2
    y = Var([7, 3, 6, 6, 5, 8, 6, 7,
             7, 11, 9, 11, 10, 10, 11, 11,
             8, 14, 10, 11, 12, 10, 11, 12,
             16, 7, 11, 9, 10, 11, 8, 8,
             16, 10, 13, 10, 10, 14, 11, 12,
             24, 29, 10, 22, 25, 28, 22, 24])
    a = Factor([1, 0], repeat=3 * 8, name='A')
    b = Factor(list(range(3)), tile=2, repeat=8, name='B')
    # Independent Measures:
    subject = Factor(list(range(8 * 6)), name='subject', random=True)
    aov = test.ANOVA(y, a * b + subject(a % b))
    assert_fmtxt_str_equals(aov, """
                 SS   df       MS   MS(denom)   df(denom)          F        p
    -------------------------------------------------------------------------
    A        432.00    1   432.00        9.05          42   47.75***   < .001
    B        672.00    2   336.00        9.05          42   37.14***   < .001
    A x B    224.00    2   112.00        9.05          42   12.38***   < .001
    -------------------------------------------------------------------------
    Total   1708.00   47
    """)
    subject = Factor(list(range(8)), tile=6, name='subject', random=True)
    aov = test.ANOVA(y, a * b * subject)
    assert_fmtxt_str_equals(aov, """
                 SS   df       MS   MS(denom)   df(denom)          F        p
    -------------------------------------------------------------------------
    A        432.00    1   432.00       10.76           7   40.14***   < .001
    B        672.00    2   336.00       11.50          14   29.22***   < .001
    A x B    224.00    2   112.00        6.55          14   17.11***   < .001
    -------------------------------------------------------------------------
    Total   1708.00   47
    """)


def test_anova_r_adler():
    """Test ANOVA accuracy by comparing with R (Adler dataset of car package)

    An unbalanced 3 by 2 independent measures design.
    """
    from rpy2.robjects import r

    # "Adler" dataset
    r_require('car')
    ds = Dataset.from_r('Adler')
    ds['rating'] = ds['rating'].astype(np.float64)

    # with balanced data
    dsb = ds.equalize_counts('expectation % instruction')
    dsb.to_r('AdlerB')
    aov = test.ANOVA('rating', 'instruction * expectation', data=dsb)
    fs = run_on_lm_fitter('rating', 'instruction * expectation', dsb)
    fnds = run_as_ndanova('rating', 'instruction * expectation', dsb)
    print(r('a.aov <- aov(rating ~ instruction * expectation, AdlerB)'))
    print(r('a.summary <- summary(a.aov)'))
    r_res = r['a.summary'][0]
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds)

    # with unbalanced data; for Type II SS use car package
    aov = test.ANOVA('rating', 'instruction * expectation', data=ds)
    fs = run_on_lm_fitter('rating', 'instruction * expectation', ds)
    fnds = run_as_ndanova('rating', 'instruction * expectation', ds)
    r_res = r("Anova(lm(rating ~ instruction * expectation, Adler, type=2))")
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'Anova')

    # single predictor
    aov = test.ANOVA('rating', 'instruction', data=ds)
    fs = run_on_lm_fitter('rating', 'instruction', ds)
    fnds = run_as_ndanova('rating', 'instruction', ds)
    r_res = r("Anova(lm(rating ~ instruction, Adler, type=2))")
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0], 'Anova')


def test_anova_r_sleep():
    "Test ANOVA accuracy by comparing with R (sleep dataset)"
    from rpy2.robjects import r

    # "sleep" dataset
    print(r('data(sleep)'))
    ds = Dataset.from_r('sleep')
    ds['ID'].random = True

    # independent measures
    aov = test.ANOVA('extra', 'group', data=ds)
    fs = run_on_lm_fitter('extra', 'group', ds)
    fnds = run_as_ndanova('extra', 'group', ds)
    print(r('sleep.aov <- aov(extra ~ group, sleep)'))
    print(r('sleep.summary <- summary(sleep.aov)'))
    r_res = r['sleep.summary'][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0])

    # repeated measures
    aov = test.ANOVA('extra', 'group * ID', data=ds)
    fs = run_on_lm_fitter('extra', 'group * ID', ds)
    fnds = run_as_ndanova('extra', 'group * ID', ds)
    print(r('sleep.aov <- aov(extra ~ group + Error(ID / group), sleep)'))
    print(r('sleep.summary <- summary(sleep.aov)'))
    r_res = r['sleep.summary'][1][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0])

    # unbalanced (independent measures)
    ds2 = ds[1:]
    print(r('sleep2 <- subset(sleep, (group == 2) | (ID != 1))'))
    aov = test.ANOVA('extra', 'group', data=ds2)
    fs = run_on_lm_fitter('extra', 'group', ds2)
    fnds = run_as_ndanova('extra', 'group', ds2)
    print(r('sleep2.aov <- aov(extra ~ group, sleep2)'))
    print(r('sleep2.summary <- summary(sleep2.aov)'))
    r_res = r['sleep2.summary'][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0])


def test_lmfitter():
    "Test the _nd_anova class"
    ds = datasets.get_uts()

    # independent, residuals vs. Hopkins
    y = ds['uts'].x

    x = ds.eval("A * B")
    lm = glm._nd_anova(x)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    x_full = ds.eval("A * B + ind(A%B)")
    lm_full = glm._nd_anova(x_full)
    assert isinstance(lm_full, glm._BalancedMixedNDANOVA)
    f_maps_full = lm_full.map(y)
    p_maps_full = lm_full.p_maps(f_maps)

    for f, f_full in zip(f_maps, f_maps_full):
        assert_allclose(f, f_full)
    for p, p_full in zip(p_maps, p_maps_full):
        assert_allclose(p, p_full)

    # repeated measures
    x = ds.eval("A * B * rm")
    lm = glm._nd_anova(x)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    aov = test.ANOVA(y[:, 0], x)
    for f_test, f_map, p_map in zip(aov.f_tests, f_maps, p_maps):
        assert f_map[0] == pytest.approx(f_test.f)
        assert p_map[0] == pytest.approx(f_test.p)
