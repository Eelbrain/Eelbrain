# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from numpy.testing import assert_array_equal
import numpy as np
import pingouin
import pytest
import scipy.stats

from eelbrain import datasets, test
from eelbrain._stats import test as _test
from eelbrain.fmtxt import asfmtext
from eelbrain.testing import assert_fmtxt_str_equals


def test_correlations():
    "Test test.correlations()"
    ds = datasets.get_uv()

    res = test.correlations('fltvar', 'fltvar2', ds=ds)
    print(res)
    assert str(res[2][0]).strip() == '.398'
    res = test.correlations('fltvar', 'fltvar2', ds=ds, asds=True)
    assert res[0, 'r'] == pytest.approx(.398, abs=1e-3)
    res = test.Correlation('fltvar', 'fltvar2', data=ds)
    assert res.r == pytest.approx(.398, abs=1e-3)

    res = test.correlations('fltvar', 'fltvar2', 'A', ds=ds)
    print(res)
    assert str(res[2][0]).strip() == 'a1'
    assert str(res[2][1]).strip() == '-.149'
    assert str(res[3][1]).strip() == '.740'
    res = test.correlations('fltvar', 'fltvar2', 'A', ds=ds, asds=True)
    assert res[0, 'A'] == 'a1'
    assert res[0, 'r'] == pytest.approx(-0.149, abs=1e-3)
    assert res[1, 'r'] == pytest.approx(.740, abs=1e-3)
    res = test.Correlation('fltvar', 'fltvar2', "A == 'a1'", ds)
    assert res.r == pytest.approx(-0.149, abs=1e-3)

    res = test.correlations('fltvar', 'fltvar2', 'A%B', ds=ds)
    print(res)
    assert str(res[2][2]).strip() == '-.276'
    res = test.correlations('fltvar', 'fltvar2', 'A%B', ds=ds, asds=True)
    assert res[0, 'r'] == pytest.approx(-0.276, abs=1e-3)

    res = test.correlations('fltvar', ('fltvar2', 'intvar'), 'A%B', ds=ds)
    print(res)
    assert str(res[2][1]).strip() == 'a1'
    assert str(res[2][2]).strip() == 'b1'
    assert str(res[2][3]).strip() == '-.276'
    res = test.correlations('fltvar', ('fltvar2', 'intvar'), 'A%B', ds=ds, asds=True)
    assert res[0, 'r'] == pytest.approx(-0.276, abs=1e-3)
    res = test.Correlation('fltvar', 'intvar', "(A=='a1')&(B=='b1')", ds)
    assert res.r == pytest.approx(0.315, abs=1e-3)

    # pairwise correlation
    doc = test.pairwise_correlations(['intvar', 'fltvar', 'fltvar2'], data=ds)
    assert_fmtxt_str_equals(doc, """
                    intvar                      fltvar                    fltvar2
    ------------------------------------------------------------------------------------------
    intvar                                r(78) = 0.10               r(78) = -0.08
                                              p = .383                   p = .500
    fltvar    r(78) = 0.10                                           r(78) = 0.40***
                  p = .383                                               p < .001
    fltvar2   r(78) = -0.08               r(78) = 0.40***
                  p = .500                    p < .001
    """)


def test_mann_whitney():
    ds = datasets.get_uv()

    ds_agg = ds.aggregate('A % rm', drop_bad=True)
    n = ds_agg.n_cases // 2
    a, b = ds_agg[:n, 'fltvar'], ds_agg[n:, 'fltvar']
    u, p = scipy.stats.mannwhitneyu(a.x, b.x, alternative='two-sided')

    res = test.MannWhitneyU('fltvar', 'A', 'a1', 'a2', 'rm', data=ds)
    assert res.u == u
    assert res.p == p

    res = test.MannWhitneyU(a, b)
    assert res.u == u
    assert res.p == p


def test_star():
    "Test the star function"
    assert_array_equal(_test.star([0.1, 0.04, 0.01], int), [0, 1, 2])
    assert_array_equal(_test.star([0.001, 0.04, 0.1], int), [3, 1, 0])


def test_ttest():
    """Test univariate t-test functions"""
    ds = datasets.get_uv()

    print(test.ttest('fltvar', data=ds))
    print(test.ttest('fltvar', 'A', data=ds))
    print(test.ttest('fltvar', 'A%B', data=ds))
    print(test.ttest('fltvar', 'A', match='rm', data=ds))
    print(test.ttest('fltvar', 'A', 'a1', match='rm', data=ds))
    print(test.ttest('fltvar', 'A%B', ('a1', 'b1'), match='rm', data=ds))

    # Prepare data for scipy
    a1_index = ds.eval("A == 'a1'")
    a2_index = ds.eval("A == 'a2'")
    b1_index = ds.eval("B == 'b1'")
    a1_in_b1_index = np.logical_and(a1_index, b1_index)
    a2_in_b1_index = np.logical_and(a2_index, b1_index)
    a1 = ds[a1_index, 'fltvar'].x
    a2 = ds[a2_index, 'fltvar'].x
    a1_in_b1 = ds[a1_in_b1_index, 'fltvar'].x
    a2_in_b1 = ds[a2_in_b1_index, 'fltvar'].x

    # TTest1Samp
    standard = pingouin.ttest(ds['fltvar'], 0)
    res = test.TTestOneSample('fltvar', data=ds)
    t, p = scipy.stats.ttest_1samp(ds['fltvar'], 0)
    assert res.t == pytest.approx(t, 10)
    assert res.p == pytest.approx(p, 10)
    assert res.d == pytest.approx(standard['cohen-d'][0], 10)
    assert str(res.full) == 'M = 0.40, SD = 1.20, t(79) = 2.96, p = .004'
    res = test.TTestOneSample('fltvar', data=ds, tail=1)
    assert res.t == pytest.approx(t, 10)
    assert res.p == pytest.approx(p / 2., 10)
    assert res.d == pytest.approx(standard['cohen-d'][0], 10)
    assert str(res.full) == 'M = 0.40, SD = 1.20, t(79) = 2.96, p = .002'

    # TTestIndependent
    cohens_d = pingouin.compute_effsize(a1, a2)
    res = test.TTestIndependent('fltvar', 'A', 'a1', 'a2', data=ds)
    t, p = scipy.stats.ttest_ind(ds[a1_index, 'fltvar'], ds[a2_index, 'fltvar'])
    assert res.t == pytest.approx(t, 10)
    assert res.p == pytest.approx(p, 10)
    assert str(res.full) == 'a1: M = 1.00, SD = 1.02; a2: M = -0.20, SD = 1.05; t(78) = 5.10, p < .001'
    assert res.d == cohens_d

    # TTestRelated
    res = test.TTestRelated('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds)
    standard = pingouin.ttest(a1_in_b1, a2_in_b1)
    difference = a1_in_b1 - a2_in_b1
    t, p = scipy.stats.ttest_rel(a1_in_b1, a2_in_b1)
    assert_array_equal(res.difference.x, difference)
    assert res.df == len(a1_in_b1) - 1
    assert res.tail == 0
    assert res.t == pytest.approx(t)
    assert res.p == pytest.approx(p)
    assert res.d == pytest.approx(standard['cohen-d'][0], 10)
    print(res)
    print(asfmtext(res))
    assert str(res.full) == 'a1: M = 0.90; a2: M = -0.06; difference: M = 0.96, SD = 1.65, t(19) = 2.53, p = .021'

    res = test.TTestRelated('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds, 1)
    assert_array_equal(res.difference.x, difference)
    assert res.df == len(a1_in_b1) - 1
    assert res.tail == 1
    assert res.t == pytest.approx(t)
    assert res.p == pytest.approx(p / 2)
    print(res)
    print(asfmtext(res))

    res = test.TTestRelated('fltvar', 'A', 'a2', 'a1', 'rm', "B=='b1'", ds, 1)
    assert_array_equal(res.difference.x, -difference)
    assert res.df == len(a1_in_b1) - 1
    assert res.tail == 1
    assert res.t == pytest.approx(-t)
    assert res.p == pytest.approx(1 - p / 2)

    res = test.TTestRelated('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds, -1)
    assert_array_equal(res.difference.x, difference)
    assert res.df == len(a1_in_b1) - 1
    assert res.tail == -1
    assert res.t == pytest.approx(t)
    assert res.p == pytest.approx(1 - p / 2)
    print(res)
    print(asfmtext(res))
    # alternative argspec
    a1_in_b1 = ds.eval("fltvar[(B == 'b1') & (A == 'a1')]")
    a2_in_b1 = ds.eval("fltvar[(B == 'b1') & (A == 'a2')]")
    res_alt = test.TTestRelated(a1_in_b1, a2_in_b1, tail=-1)
    print(res_alt)
    assert res_alt.t == res.t
    assert res_alt.p == res.p


def test_wilcoxon():
    ds = datasets.get_uv()

    ds_agg = ds.aggregate('A % rm', drop_bad=True)
    n = ds_agg.n_cases // 2
    w, p = scipy.stats.wilcoxon(ds_agg[:n, 'fltvar'].x, ds_agg[n:, 'fltvar'].x, alternative='two-sided')

    res = test.WilcoxonSignedRank('fltvar', 'A', 'a1', 'a2', 'rm', data=ds)
    assert res.w == w
    assert res.p == p
