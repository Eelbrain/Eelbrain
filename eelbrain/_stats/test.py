# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistical tests for univariate variables"""
from functools import cached_property
import math
from typing import Dict, Literal, Sequence, Union

import numpy as np
import scipy.stats

from .. import fmtxt
from .._celltable import Celltable
from .._data_obj import (
    CategorialArg, CellArg, IndexArg, VarArg, NumericArg,
    Dataset, Factor, Interaction, Var, NDVar,
    ascategorial, asnumeric, assub, asvar, asndvar,
    combine, cellname, dataobj_repr, nice_label,
)
from .._utils import deprecate_ds_arg
from . import stats


__test__ = False
DEFAULT_LEVELS = {.05: '*', .01: '**', .001: '***'}
DEFAULT_LEVELS_TREND = {.05: '*', .01: '**', .001: '***', .1: '`'}

MCCArg = Union[bool, Literal['hochberg', 'bonferroni', 'holm']]


def get_levels(
        levels: Union[bool, dict],
        trend: Union[bool, str] = False,
):
    if levels is True:
        if trend is True:
            return DEFAULT_LEVELS_TREND
        elif trend:
            return {**DEFAULT_LEVELS, .1: trend}
        else:
            return DEFAULT_LEVELS
    elif trend:
        raise TypeError(f"{trend=} only valid when levels=True")
    return levels


class Correlation:
    """Pearson product moment correlation between y and x

    Parameters
    ----------
    y : Var | NDVar
        First variable.
    x : Var | NDVar
        Second variable. Needs to have same type/shape as ``y``.
    sub : index
        Use only a subset of the data
    data : Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables.

    Attributes
    ----------
    r : float
        Pearson correlation coefficient.
    p : float
        Two-tailed p-value.
    df : int
        Degrees of freedom.
    """
    _statistic = 'r'

    @deprecate_ds_arg
    def __init__(
            self,
            y: NumericArg,
            x: NumericArg,
            sub: IndexArg = None,
            data: Dataset = None,
    ):
        sub, n = assub(sub, data, return_n=True)
        y, n = asnumeric(y, sub, data, n, return_n=True, array=True)
        x = asnumeric(x, sub, data, n, array=True)
        if type(y) is not type(x):
            raise TypeError(f"y and x must be same type; got type(y)={type(y)}, type(x)={type(x)}")
        elif isinstance(y, Var):
            x_y = y.x
            x_x = x.x
        elif isinstance(y, NDVar):
            if y.dims != x.dims:
                raise ValueError(f"y and x have different dimensions; y.dims={y.dims}, x.dims={x.dims}")
            x_y = y.x.ravel()
            x_x = x.x.ravel()
        else:
            if y.shape != x.shape:
                raise ValueError(f"Shape mismatch: y.shape={y.shape}, x.shape={x.shape}")
            x_y = y.ravel()
            x_x = x.ravel()
        self.r, self.p, self.df = self._corr(x_y, x_x)
        self._y = dataobj_repr(y)
        self._x = dataobj_repr(x)

    def _corr(self, y: np.ndarray, x: np.ndarray):
        return _corr(y, x)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self._y} ~ {self._x}; {self._asfmtext()}> "

    def _asfmtext(self, **_):
        return fmtxt.FMText([fmtxt.eq(self._statistic, self.r, self.df), ', ', fmtxt.peq(self.p)])

    @property
    def stars(self):
        return fmtxt.Stars.from_p(self.p)


class RankCorrelation(Correlation):
    """Spearman rank correlation between y and x

    Parameters
    ----------
    y : Var | NDVar
        First variable.
    x : Var | NDVar
        Second variable. Needs to have same type/shape as ``y``.
    sub : index
        Use only a subset of the data
    ds : Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables.

    Attributes
    ----------
    r : float
        Spearman rank correlation coefficient.
    p : float
        Two-tailed p-value.
    """

    def _corr(self, y: np.ndarray, x: np.ndarray):
        r, p = scipy.stats.spearmanr(y, x)
        return r, p, None

    def _asfmtext(self, **_):
        return fmtxt.FMText([fmtxt.eq(self._statistic, self.r), ', ', fmtxt.peq(self.p)])


def lilliefors(data, formatted=False, **kwargs):
    """Lilliefors' test for normal distribution

    The Lilliefors test is an adaptation of the Kolmogorov-Smirnov test. It
    is used to test the null hypothesis that data come from a normally
    distributed population, when the null hypothesis does not specify which
    normal distribution, i.e. does not specify the expected value and variance.

    Parameters
    ----------
    data : array_like
        Data to test.
    formatted : bool
        Return a single string with the results instead of the numbers.
    kwargs :
        All keyword arguments are forwarded to :func:`scipy.stats.kstest`.

    Returns
    -------
    D : float
        The D-value of the Kolmogorov-Smirnov Test
    p_estimate : float
        The approximate p value according to Dallal and Wilkinson (1986).
        Requires minimal sample size of 5. p is reasonably accurate only when
        it is <= .1 (cf. Dallal and Wilkens).

    Notes
    -----
    Uses the scipy.stats.kstest implementation of the Kolmogorov-Smirnov test.

    References
    ----------
    Dallal, G. E. and Wilkinson, L. (1986). An Analytic Approximation to the
            Distribution of Lilliefors's Test Statistic for Normality. The
            American Statistician, 40(4), 294--296.
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for Normality
            with Mean and Variance Unknown. Journal of the American
            Statistical Association, 62(318), 399--402.
    """
    # p values agree with R lillie.test (nortest package) on low values of p.
    # lillie.test adjusts something at p>.1
    # http://pbil.univ-lyon1.fr/library/nortest/R/nortest
    data = np.asarray(data)
    n = len(data)  # data.shape[-1] #axis]
    assert n >= 5, "sample size must be greater than 4"
    # perform Kolmogorov-Smirnov with estimated mean and std
    m = np.mean(data)
    s = np.std(data, ddof=1)
    d, ks_p = scipy.stats.kstest(data, 'norm', args=(m, s), **kwargs)
    # approximate p (Dallal)
    if n > 100:
        d *= (n / 100) ** .49
        n = 100
    p_estimate = np.exp(- 7.01256 * d ** 2 * (n + 2.78019)
                        + 2.99587 * d * (n + 2.78019) ** .5
                        - .122119
                        + .974598 / (n ** .5)
                        + 1.67997 / n)
    # approximate P (Molin & Abdi)
    l = d  # ???
    b2 = 0.08861783849346
    b1 = 1.30748185078790
    b0 = 0.37872256037043
    a = (-(b1 + n) + np.sqrt((b1 + n) ** 2 - 4 * b2 * (b0 - l ** -2))) / 2 * b2
    p = (-.37782822932809
         + 1.67819837908004 * a
         - 3.02959249450445 * a ** 2
         + 2.80015798142101 * a ** 3
         - 1.39874347510845 * a ** 4
         + 0.40466213484419 * a ** 5
         - 0.06353440854207 * a ** 6
         + 0.00287462087623 * a ** 7
         + 0.00069650013110 * a ** 8
         - 0.00011872227037 * a ** 9
         + 0.00000575586834 * a ** 10)
    if formatted:
        return f"D={d:.4f}, Dallal p={p_estimate:.4f}, Molin&Abdi p={p:.4f}"
    else:
        return d, p_estimate


def _hochberg_threshold(n, alpha=.05):
    j = np.arange(n)
    threshold = alpha / (n - j)
    return threshold


def mcp_adjust(ps, method='Hochberg'):
    """Adjust p-values for multiple comparison

    Parameters
    ----------
    ps : sequence of scalar
        P-values.
    method : 'hochberg' | 'bonferroni' | 'holm'
        Correction method. Default is 'hochberg'.

    Returns
    -------
    adjusted_ps : list of scalar
        Adjusted p-values.
    """
    if not method:
        return ps
    n = len(ps)
    if n <= 1:
        return ps
    method_ = method.lower()
    if method_ == 'bonferroni':
        ps_adjusted = [p * n for p in ps]
    elif method_ in ('hochberg', 'holm'):
        ascsort = np.argsort(ps)
        ps_asc = np.array(ps)[ascsort]
        iout_asc = np.arange(n)[ascsort]
        ps_adjusted = [-1] * n
        p_buffer = 1
        if method_ == 'holm':
            for i in range(n):
                p = ps_asc[i]
                p_adj = (n - i) * p
                p_buffer = max(p_buffer, p_adj)
                ps_adjusted[iout_asc[i]] = p_buffer
        elif method_ == 'hochberg':
            for i in range(1, n + 1):
                p = ps_asc[-i]
                p_adj = (i) * p
                p_buffer = min(p_adj, p_buffer)
                ps_adjusted[iout_asc[-i]] = p_buffer
    else:
        msg = ('%r is not a valid argument for multiple comparison correction '
               'method' % method)
        raise ValueError(msg)

    return ps_adjusted


def _get_correction_caption(corr, n):
    if corr == 'Hochberg':
        return "(* Corrected after Hochberg, 1988)"
    elif corr == 'Bonferroni':
        return "(* Bonferroni corrected for %i tests)" % n
    elif corr == 'Holm':
        return "(* Corrected after Holm)"
    else:
        return "(* Uncorrected)"


def _n_stars(p: float, levels: Sequence[float] = (0.05, 0.01, 0.001)):
    return sum(p <= level for level in levels)


def star(
        p_list: Union[float, Sequence[float]],
        out: type = str,
        levels: Union[bool, dict] = True,
        trend: Union[bool, str] = False,
        eq_strlen: bool = False,
):
    """Determine number of stars for p-value

    Parameters
    ----------
    p_list
        P-values.
    out : {str, int}
        Return string with stars ('**') or an integer indicating the number of
        stars.
    levels
        ``{p: str, ...}`` dictionary. The default is ``{.05 : '*',
        .01 : '**', .001: '***'}``; ``trend=True`` adds ``{.1: "'"}``.
    trend
        Add trend to default ``levels``.
    eq_strlen
        Equalize string lengths; when strings are returned, make sure they all
        have the same length (default ``False``).
    """
    levels = get_levels(levels, trend)

    levels_descending = sorted(levels.keys(), reverse=True)
    symbols_descending = [''] + [levels[l] for l in levels_descending]

    if out is int:
        int_out = True
    elif out is str:
        int_out = False
    else:
        raise TypeError("out=%r" % (out,))

    # allow input (p_list) to contain single p-value
    if not np.iterable(p_list):
        n = _n_stars(p_list, levels)
        if int_out:
            return n
        else:
            return symbols_descending[n]

    ns = [_n_stars(p, levels) for p in p_list]
    if int_out:
        return ns
    symbols = [symbols_descending[n] for n in ns]
    if eq_strlen:
        maxlen = max(map(len, symbols))
        return [s.ljust(maxlen) for s in symbols]
    else:
        return symbols


def star_factor(p, levels=True):
    """Create a factor with stars for a sequence of p-values

    Parameters
    ----------
    p : sequence of scalar
        Sequence of p-values.
    levels : dict {scalar: str}
        {value: star-mark} dictionary.

    Returns
    -------
    stars : Factor
        Factor with the appropriate star marking for each item in p.
    """
    levels = get_levels(levels)
    sorted_levels = sorted(levels, reverse=True)
    star_labels = {i: levels[v] for i, v in enumerate(sorted_levels, 1)}
    star_labels[0] = ''
    level_values = np.reshape(sorted_levels, (-1, 1))
    return Factor(np.sum(p <= level_values, 0), labels=star_labels)


def _independent_measures_args(y, x, c1, c0, match, ds, sub, nd_data=False):
    "Interpret parameters for independent measures tests (2 different argspecs)"
    if isinstance(x, str):
        x = ds.eval(x)

    if nd_data:
        two_y = isinstance(x, NDVar)
        coerce = asndvar
    else:
        two_y = isinstance(x, Var) and match is None
        coerce = asvar

    if two_y:
        assert match is None
        y1 = coerce(y, sub, ds)
        y0 = coerce(x, sub, ds)
        y = combine((y1, y0))
        c1_name = y1.name if c1 is None else c1
        c0_name = y0.name if c0 is None else c0
        x_name = y0.name
    else:
        ct = Celltable(y, x, match, sub, cat=(c1, c0), data=ds, coercion=coerce, dtype=np.float64)
        c1, c0 = ct.cat
        c1_name = c1
        c0_name = c0
        x_name = ct.x.name
        match = ct.match
        y = ct.y
        y1 = ct.data[c1]
        y0 = ct.data[c0]
    return y, y1, y0, c1, c0, match, x_name, c1_name, c0_name


def _related_measures_args(y, x, c1, c0, match, data, sub, nd_data=False):
    "Interpret parameters for related measures tests (2 different argspecs)"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x=} specified as str without specifying data")
        x = data.eval(x)

    if nd_data:
        two_y = isinstance(x, NDVar)
        coerce = asndvar
    else:
        two_y = isinstance(x, Var) and match is None
        coerce = asvar

    if two_y:
        assert match is None
        y1 = coerce(y, sub, data)
        n = len(y1)
        y0 = coerce(x, sub, data, n)
        c1_name = y1.name if c1 is None else c1
        c0_name = y0.name if c0 is None else c0
        x_name = y0.name
    elif match is None:
        raise TypeError("The `match` argument needs to be specified for related measures tests")
    else:
        ct = Celltable(y, x, match, sub, cat=(c1, c0), data=data, coercion=coerce, dtype=np.float64)
        c1, c0 = ct.cat
        c1_name = c1
        c0_name = c0
        if not ct.all_within:
            raise ValueError(f"conditions {c1!r} and {c0!r} do not have the same values on {dataobj_repr(ct.match)}")
        n = len(ct.y) // 2
        y1 = ct.y[:n]
        y0 = ct.y[n:]
        x_name = ct.x.name
        match = ct.match
    return y1, y0, c1, c0, match, n, x_name, c1_name, c0_name


@deprecate_ds_arg
def ttest(
        y: VarArg,
        x: CategorialArg = None,
        against: Union[float, CellArg] = 0,
        match: CategorialArg = None,
        sub: IndexArg = None,
        corr: MCCArg = 'Hochberg',
        title: str = '{desc}',
        data: Dataset = None,
        tail: Literal[-1, 0, 1] = 0,
) -> fmtxt.Table:
    """T-tests for one or more samples

    parameters
    ----------
    y
        Dependent variable
    x
        Perform tests separately for all categories in x.
    against
        Baseline against which to test (scalar or category in x).
    match
        Repeated measures factor.
    sub
        Only use part of the data.
    corr
        Method for multiple comparison correction (default 'hochberg').
    title
        Title for the table.
    data
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables
    tail
        Tailedness of the test.

    Returns
    -------
    table : FMText Table
        Table with results.
    """
    ct = Celltable(y, x, match, sub, data=data, coercion=asvar)

    par = True
    if par:
        infl = '' if x is None else 's'
        if ct.y.name is None:
            title_desc = f"T-test{infl} against {cellname(against)}"
        else:
            title_desc = f"T-test{infl} of {ct.y.name} against {cellname(against)}"
        statistic_name = 't'
    else:
        raise NotImplementedError

    names = []
    tests = []
    if isinstance(against, (str, tuple)):
        if against not in ct.cells:
            raise ValueError(f"{against=}: not a cell in x ({dataobj_repr(ct.x)})")
        for cell in ct.cells:
            if cell == against:
                continue
            names.append(cell)
            if match is not None and ct.within[cell, against]:
                res = TTestRelated(ct.data[cell], ct.data[against], tail=tail)
            else:
                res = TTestIndependent(ct.data[cell], ct.data[against], tail=tail)
            tests.append(res)
    elif np.isscalar(against):
        for cell in ct.cells:
            names.append(cellname(cell))
            res = TTestOneSample(ct.data[cell], tail=tail)
            tests.append(res)
    else:
        raise TypeError(f"{against=}")

    if len(tests) <= 1:
        corr = None

    ps = [res.p for res in tests]
    dfs = [res.df for res in tests]
    if corr:
        ps_adjusted = mcp_adjust(ps, corr)
    else:
        ps_adjusted = ps
    stars = star(ps_adjusted, out=str)
    if len(set(dfs)) == 1:
        df_in_header = True
    else:
        df_in_header = False

    # table
    table = fmtxt.Table('l' + 'r' * (4 - df_in_header + bool(corr)))
    table.title(title.format(desc=title_desc))
    if corr:
        table.caption(_get_correction_caption(corr, len(tests)))

    # header
    table.cell("Effect")
    table.cell("Difference")
    if df_in_header:
        table.cell(fmtxt.symbol(statistic_name, dfs[0]))
    else:
        table.cell(statistic_name, 'math')
        table.cell('df', 'math')
    table.cell('p', 'math')
    if corr:
        table.cell(fmtxt.symbol('p', corr))
    table.midrule()

    # body
    for name, res, p_adj, mark in zip(names, tests, ps_adjusted, stars):
        table.cell(name)
        table.cell(res._difference)
        table.cell(fmtxt.stat(res.t, stars=mark, of=3))
        if not df_in_header:
            table.cell(res.df)

        table.cell(fmtxt.p(res.p))
        if corr:
            table.cell(fmtxt.p(p_adj))
    return table


class TTest:
    _statistic = 't'

    def __init__(self, difference, t, df, tail, std):
        self._difference = difference
        self.t = t
        self.df = df
        self.p = stats.ttest_p(self.t, self.df, tail)
        self.tail = tail
        # effect-size
        self.d = difference / std

    @property
    def stars(self):
        return fmtxt.Stars.from_p(self.p)

    def _asfmtext(
            self,
            rasterize: bool = None,
            difference: bool = False,
            **_,
    ):
        out = [fmtxt.eq('t', self.t, self.df), ', ', fmtxt.peq(self.p)]
        if difference:
            out = [fmtxt.eq('difference', self._difference), ', '] + out
        return fmtxt.FMText(out)


class TTestOneSample(TTest):
    """One-sample t-test

    Parameters
    ----------
    y : Var
        Dependent variable.
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison).
    sub : index-array
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    popmean : float
        Population mean to test against (default 0).
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed, default);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).

    Attributes
    ----------
    mean : float
        Mean of ``y``.
    t : float
        T-value.
    p : float
        P-value.
    tail : 0 | 1 | -1
        Tailedness of the p value.
    df : int
        Degrees of freedom.
    d : float
        Cohen's *d*.
    full : FMText
        Full description of the test result.
    """
    @deprecate_ds_arg
    def __init__(
            self,
            y: VarArg,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            popmean: float = 0,
            tail: int = 0,
    ):
        ct = Celltable(y, None, match, sub, data=data, coercion=asvar)
        n = len(ct.y)
        if n <= 2:
            raise ValueError(f"Not enough observations for t-test ({n=})")

        self.mean = ct.y.mean()
        self.popmean = popmean
        self._y = ct.y
        self._y_name = dataobj_repr(ct.y)
        v = ct.y.x[:, None]
        if popmean:
            v = v - popmean
        t = stats.t_1samp(v)[0]
        TTest.__init__(self, v.mean(), t, n - 1, tail, ct.y.std(ddof=1))

    def __repr__(self):
        cmp = '=><'[self.tail]
        return f"<{self.__class__.__name__}: {self._y_name} {cmp} {self.popmean}; {self._asfmtext(difference=True)}>"

    @cached_property
    def full(self):
        return fmtxt.FMText([fmtxt.eq('M', self.mean), ', ', fmtxt.eq('SD', self._y.std()), ', ', self._asfmtext()])


class TTestIndependent(TTest):
    """Independent measures t-test

    The test data can be specified in two forms:

     - In long form, with ``y`` supplying the data, ``x`` specifying condition
       for each case.
     - With ``y`` and ``x`` supplying data for the two conditions.

    Parameters
    ----------
    y : Var
        Dependent variable.
    x : categorial
        Model containing the cells which should be compared.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Units within which measurements are related and should be averaged over
        (e.g. 'subject' in a between-group comparison).
    sub : index-array
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed, default);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).

    Attributes
    ----------
    t : float
        T-value.
    p : float
        P-value.
    tail : 0 | 1 | -1
        Tailedness of the p value.
    df : int
        Degrees of freedom.
    full : FMText
        Full description of the test result.
    """
    @deprecate_ds_arg
    def __init__(
            self,
            y: VarArg,
            x: CategorialArg,
            c1: CellArg = None,
            c0: CellArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
    ):
        y, y1, y0, c1, c0, match, x_name, c1_name, c0_name = _independent_measures_args(y, x, c1, c0, match, data, sub)

        n1 = len(y1)
        n0 = len(y0)
        n = n1 + n0
        df = n - 2
        if df < 1:
            raise ValueError(f"Not enough cases for t-test: {n1=}, {n0=}, {df=}")

        groups = np.arange(n) < n1
        c1_data, c0_data = y[groups], y[~groups]
        c1_mean, c0_mean = c1_data.mean(), c0_data.mean()
        pooled_variance = ((c1_data - c1_mean).var(ddof=1) * (n1 - 1) + (c0_data - c0_mean).var(ddof=1) * (n0 - 1)) / (n - 2)
        difference = c1_mean - c0_mean
        groups.dtype = np.int8
        t = stats.t_ind(y.x[:, None], groups)[0]

        self._y = dataobj_repr(y)
        self._x = x_name
        TTest.__init__(self, difference, t, df, tail, math.sqrt(pooled_variance))
        self.c1_name = c1_name
        self.c0_name = c0_name
        self._y1 = y1
        self._y0 = y0
        self._two_y = c1 is None

    def __repr__(self):
        cmp = '=><'[self.tail]
        if self._two_y:
            desc = f"{self.c1_name} {cmp} {self.c0_name}"
        else:
            desc = f"{self._y} ~ {self._x}, {self.c1_name} {cmp} {self.c0_name}"
        return f"<{self.__class__.__name__}: {desc}; {self._asfmtext(difference=True)}>"

    @cached_property
    def full(self):
        return fmtxt.FMText([
            self.c1_name, ': ', fmtxt.eq('M', self._y1.mean()), ', ', fmtxt.eq('SD', self._y1.std()), '; ',
            self.c0_name, ': ', fmtxt.eq('M', self._y0.mean()), ', ', fmtxt.eq('SD', self._y0.std()), '; ',
            self._asfmtext()])


class MannWhitneyU:
    """Mann-Whitney U-test (non-parametric independent measures test)

    The test data can be specified in two forms:

     - In long form, with ``y`` supplying the data, ``x`` specifying condition
       for each case.
     - With ``y`` and ``x`` supplying data for the two conditions.

    Parameters
    ----------
    y : Var
        Dependent variable. Alternatively, the first of two variables that are
        compared.
    x : categorial
        Model containing the cells which should be compared. Alternatively, the
        second of two varaibles that are compared.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison). If match is unspecified, it is assumed that
        ``y`` and ``x`` are two measurements with matched cases.
    sub : index-array
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed, default);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    continuity : bool
        Continuity correction (default ``True``).

    Attributes
    ----------
    u : float
        Mann-Whitney U statistic.
    p : float
        P-value.
    tail : 0 | 1 | -1
        Tailedness of the p value.

    See Also
    --------
    TTestRelated : parametric alternative

    Notes
    -----
    Based on :func:`scipy.stats.mannwhitneyu`.
    """
    _statistic = 'U'

    @deprecate_ds_arg
    def __init__(
            self,
            y: VarArg,
            x: Union[CategorialArg, VarArg],
            c1: CellArg = None,
            c0: CellArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
            continuity: bool = True,
    ):
        y, y1, y0, c1, c0, match, x_name, c1_name, c0_name = _independent_measures_args(y, x, c1, c0, match, data, sub)
        if tail == 0:
            alternative = 'two-sided'
        elif tail == 1:
            alternative = 'greater'
        elif tail == -1:
            alternative = 'less'
        else:
            raise ValueError(f"{tail=}")
        self.u, self.p = scipy.stats.mannwhitneyu(y1.x, y0.x, continuity, alternative)
        self._y = dataobj_repr(y1)
        self._x = x_name
        self._c1 = c1
        self._c0 = c0
        self.tail = tail
        self.continuity = continuity
        self._two_y = c1 is None

    def __repr__(self):
        cmp = '=><'[self.tail]
        if self._two_y:
            desc = f"{self._c1} {cmp} {self._c0}"
        else:
            desc = f"{self._y} ~ {self._x}, {self._c1} {cmp} {self._c0}"
        return f"<MannWhitneyU: {desc}; {self._asfmtext()}>"

    def _asfmtext(self, **_):
        return fmtxt.FMText([fmtxt.eq('U', self.u), ', ', fmtxt.peq(self.p)])

    @property
    def stars(self):
        return fmtxt.Stars.from_p(self.p)


class TTestRelated(TTest):
    """Related-measures t-test

    The test data can be specified in two forms:

     - In long form, with ``y`` supplying the data, ``x`` specifying condition
       for each case and ``match`` determining which cases are related.
     - In wide/repeated measures form, with ``y`` and ``x`` both supplying data
       with matching case order.

    Parameters
    ----------
    y : Var
        Dependent variable. Alternatively, the first of two variables that are
        compared.
    x : categorial
        Model containing the cells which should be compared. Alternatively, the
        second of two varaibles that are compared.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison). If match is unspecified, it is assumed that
        ``y`` and ``x`` are two measurements with matched cases.
    sub : index-array
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed, default);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).

    Attributes
    ----------
    t : float
        T-value.
    p : float
        P-value.
    tail : 0 | 1 | -1
        Tailedness of the p value.
    difference : Var
        Difference values.
    df : int
        Degrees of freedom.
    c1_mean : float
        Mean of condition ``c1``.
    c0_mean : float
        Mean of condition ``c0``.
    d : float
        Cohen's *d*.
    full : FMText
        Full description of the test result.

    See Also
    --------
    WilcoxonSignedRank : non-parametric alternative
    """
    @deprecate_ds_arg
    def __init__(
            self,
            y: VarArg,
            x: Union[CategorialArg, VarArg],
            c1: CellArg = None,
            c0: CellArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
    ):
        y1, y0, c1, c0, match, n, x_name, c1_name, c0_name = _related_measures_args(y, x, c1, c0, match, data, sub)
        if n <= 2:
            raise ValueError("Not enough observations for t-test (n=%i)" % n)
        self._y = dataobj_repr(y1)
        self._x = x_name
        self._c1 = c1
        self._c0 = c0
        self.c1_name = c1_name
        self.c0_name = c0_name
        self.c1_mean = y1.mean()
        self.c0_mean = y0.mean()
        self.difference = y1 - y0
        t = stats.t_1samp(self.difference.x[:, None])[0]
        TTest.__init__(self, self.difference.x.mean(), t, n - 1, tail, self.difference.std(ddof=1))
        self._match = dataobj_repr(match, True)

    def __repr__(self):
        cmp = '=><'[self.tail]
        if self._match is None:
            desc = f"{self._c1} {cmp} {self._c0}"
        else:
            desc = f"{self._y} ~ {self._x}, {self._c1} {cmp} {self._c0}"
        return f"<{self.__class__.__name__}: {desc}; {self._asfmtext(difference=True)}>"

    @cached_property
    def full(self):
        return fmtxt.FMText([
            self.c1_name, ': ', fmtxt.eq('M', self.c1_mean), '; ',
            self.c0_name, ': ', fmtxt.eq('M', self.c0_mean), '; ',
            'difference: ', fmtxt.eq('M', self.difference.mean()), ', ',
            fmtxt.eq('SD', self.difference.std()), ', ',
            self._asfmtext()])


class WilcoxonSignedRank:
    """Wilcoxon signed-rank test (non-parametric related measures test)

    The test data can be specified in two forms:

     - In long form, with ``y`` supplying the data, ``x`` specifying condition
       for each case and ``match`` determining which cases are related.
     - In wide/repeated measures form, with ``y`` and ``x`` both supplying data
       with matching case order.

    Parameters
    ----------
    y : Var
        Dependent variable. Alternatively, the first of two variables that are
        compared.
    x : categorial
        Model containing the cells which should be compared. Alternatively, the
        second of two varaibles that are compared.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison). If match is unspecified, it is assumed that
        ``y`` and ``x`` are two measurements with matched cases.
    sub : index-array
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed, default);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    zero_method : str
        How to handle zero differences (see :func:`scipy.stats.wilcoxon`).
    correction : bool
        Continuity correction (default ``False``).

    Attributes
    ----------
    w : float
        Rank sum statistic.
    p : float
        P-value.
    tail : 0 | 1 | -1
        Tailedness of the p value.
    difference : Var
        Difference values.
    c1_mean : float
        Mean of condition ``c1``.
    c0_mean : float
        Mean of condition ``c0``.

    See Also
    --------
    TTestRelated : parametric alternative

    Notes
    -----
    Based on :func:`scipy.stats.wilcoxon`.
    """
    _statistic = 'W'

    @deprecate_ds_arg
    def __init__(
            self,
            y: VarArg,
            x: Union[CategorialArg, VarArg],
            c1: CellArg = None,
            c0: CellArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
            zero_method: str = 'wilcox',
            correction: bool = False,
    ):
        y1, y0, c1, c0, match, n, x_name, c1_name, c0_name = _related_measures_args(y, x, c1, c0, match, data, sub)
        if tail == 0:
            alternative = 'two-sided'
        elif tail == 1:
            alternative = 'greater'
        elif tail == -1:
            alternative = 'less'
        else:
            raise ValueError(f"tail={tail!r}")
        self.w, self.p = scipy.stats.wilcoxon(y1.x, y0.x, zero_method, correction, alternative)
        self._y = dataobj_repr(y1)
        self._x = x_name
        self._c1 = c1
        self._c0 = c0
        self.c1_mean = y1.mean()
        self.c0_mean = y0.mean()
        self.difference = y1 - y0
        self._match = dataobj_repr(match, True)
        self.tail = tail
        self.zero_method = zero_method
        self.correction = correction

    def __repr__(self):
        cmp = '=><'[self.tail]
        if self._match is None:
            desc = f"{self._c1} {cmp} {self._c0}"
        else:
            desc = f"{self._y} ~ {self._x}, {self._c1} {cmp} {self._c0}"
        return f"<{self.__class__.__name__}: {desc}; {self._asfmtext()}>"

    def _asfmtext(self, **_):
        return fmtxt.FMText([fmtxt.eq('W', self.w), ', ', fmtxt.peq(self.p)])

    @property
    def stars(self):
        return fmtxt.Stars.from_p(self.p)


@deprecate_ds_arg
def pairwise(
        y: VarArg,
        x: CategorialArg,
        match: CategorialArg = None,
        sub: IndexArg = None,
        cells: Sequence[CellArg] = None,
        data: Dataset = None,
        par: bool = True,
        corr: MCCArg = 'Hochberg',
        trend: Union[bool, str] = False,
        title: Union[bool, str] = True,
        mirror: bool = False,
        labels: Dict[CellArg, str] = None,
):
    """Pairwise comparison table

    Parameters
    ----------
    y : Var
        Dependent measure.
    x : categorial
        Categories to compare.
    match : None | Factor
        Repeated measures factor.
    sub : index-array
        Perform tests with a subset of the data.
    cells
        Cells to include. All entries have to be cells of ``model``. Can be
        used to change the order of cells in the table.
    data : Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables.
    par : bool
        Use parametric test for pairwise comparisons (use non-parametric
        tests if False).
    corr : None | 'hochberg' | 'bonferroni' | 'holm'
        Method for multiple comparison correction.
    trend
        Marker for a trend in pairwise comparisons.
    title
        Title for the table.
    mirror
        Redundant table including all row/column combinations.
    labels
        Alternative labels for ``x`` as ``{cell: label}`` dictionary.

    Returns
    -------
    table : FMText Table
        Table with results.
    """
    ct = Celltable(y, x, match, sub, cells, data, coercion=asvar)
    tests = _pairwise(ct, par, corr, trend)

    # extract test results
    k = len(ct)
    symbols = tests['symbols']
    ps_adjusted = tests['ps']
    cellnames = ct.cellnames()
    if labels:
        cellnames = [labels.get(name, name) for name in cellnames]

    # create TABLE
    table = fmtxt.Table('l' + 'l' * (k - 1 + mirror))
    if title is True:
        table.title(f"Pairwise {tests['name']}")
    elif title:
        table.title(title)
    table.caption(tests['caption'])

    # headings
    table.cell()
    for name in cellnames[1 - mirror:]:
        table.cell(name)
    table.midrule()

    n_stars = 3 + bool(trend)
    for row in range(0, k - 1 + mirror):
        table.cell(cellnames[row])
        for col in range(1 - mirror, k):
            if row == col:
                table.cell()
            elif mirror or col > row:
                key = ct.cells[min(row, col)], ct.cells[max(row, col)]
                test = tests['tests'][key]
                kennwert = getattr(test, test._statistic.lower())
                df = getattr(test, 'df', None)
                content = [
                    fmtxt.eq(test._statistic, kennwert, df, stars=symbols[key], of=n_stars),
                    fmtxt.linebreak,
                    fmtxt.peq(test.p),
                ]
                if corr:
                    content.extend([fmtxt.linebreak, fmtxt.peq(ps_adjusted[key], 'c')])
                table.cell(content)
            else:
                table.cell()
    return table


def _pairwise(
        ct: Celltable,  # list of groups/treatments
        parametric: bool,
        corr: MCCArg,
        trend: Union[bool, str] = False,
        levels: Union[bool, dict] = True,
):
    """Pairwise tests

    Returns
    -------
    results : dict
        dictionary with results:
        'test': test name
        'caption': information about correction
        'statistic': abbreviation used for the staistic (e.g. 'Q')
        statistic: list of values
        'df': df
        'p': list of corresponding pa values
        'stars': list of n stars (ints)
        'pw_indexes': dict linking table index (i,j) to the list index for p etc.
    """
    # find test
    k = len(ct.x.cells)
    if k < 3:  # need no correction for single test
        corr = None

    if parametric:
        if ct.all_within:
            test_name = "T-Tests (paired samples)"
        elif ct.any_within:
            test_name = "T-Tests (mixed)"
        else:
            test_name = "T-Tests (independent samples)"
    else:
        if ct.all_within:
            test_name = "Wilcoxon Signed-Rank Test"
        elif ct.any_within:
            test_name = "Wilcoxon Signed-Rank/Mann-Whitney U Test"
        else:
            test_name = "Mann-Whitney U Test"

    # perform test
    tests = {}
    for (cell1, cell2), within in ct.within.items():
        if parametric:
            if within:
                test = TTestRelated(ct.y, ct.x, cell1, cell2, ct.match)
            else:
                test = TTestIndependent(ct.y, ct.x, cell1, cell2, ct.match)
        else:
            if within:
                test = WilcoxonSignedRank(ct.y, ct.x, cell1, cell2, ct.match)
            else:
                test = MannWhitneyU(ct.y, ct.x, cell1, cell2, ct.match)
        tests[cell1, cell2] = test

    # adjusted p-values
    ps = [test.p for test in tests.values()]
    if corr:
        ps_adjusted = mcp_adjust(ps, corr)
    else:
        ps_adjusted = ps
    p_dict = {key: p for key, p in zip(tests.keys(), ps_adjusted)}
    # prepare output
    return {
        'name': test_name,
        'caption': _get_correction_caption(corr, len(ps)),
        'tests': tests,
        'ps': p_dict,
        'stars': {key: star(p, int, levels, trend) for key, p, in p_dict.items()},
        'symbols': {key: star(p, str, levels, trend) for key, p, in p_dict.items()},
    }


@deprecate_ds_arg
def pairwise_correlations(
        xs: Sequence[Union[str, Var, NDVar]],
        sub: IndexArg = None,
        data: Dataset = None,
        labels: Dict[CellArg, str] = None,
):
    """Pairwise correlation table

    Parameters
    ----------
    xs : sequence of Var | NDVar
        Variables to correlate.
    sub : index
        Use only a subset of the data
    data : Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables.
    labels : {str: str} dict
        Alternative labels for ``xs`` as ``{x.name: label}`` dictionary.

    Returns
    -------
    pairwise_table : fmtxt.Table
        Table with pairwise correlations.
    """
    sub = assub(sub, data)
    xs_ = [asnumeric(x, sub, data) for x in xs]
    n_vars = len(xs_)
    x_labels = [nice_label(x, labels) for x in xs_]

    table = fmtxt.Table('l' + 'c' * n_vars)
    # header
    table.cells('', *x_labels)
    table.midrule()
    # body
    for i_row, x_row in enumerate(xs_):
        table.cell(x_labels[i_row])
        for x_col in xs_:
            if x_col is x_row:
                table.cell()
            else:
                corr = Correlation(x_row, x_col)
                cell = fmtxt.FMText([fmtxt.eq('r', corr.r, corr.df), fmtxt.Stars.from_p(corr.p), fmtxt.linebreak, fmtxt.peq(corr.p)])
                table.cell(cell)
        table.endline()
    return table


def correlations(y, x, cat=None, sub=None, ds=None, asds=False):
    """Correlation with one or more predictors

    Parameters
    ----------
    y : Var
        First variable
    x : Var | list of Var
        second variable (or list of variables).
    cat : categorial
        Show correlations separately for different groups in the data.
    sub : index
        Use only a subset of the data
    ds : Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables.
    asds : bool
        Return correlations in Dataset instead of Table.

    Returns
    -------
    correlations : Table | Dataset
        Table or Dataset (if ``asds=True``) with correlations.
    """
    sub = assub(sub, ds)
    y = asvar(y, sub, ds)
    if isinstance(x, (Var, str)):
        x = (x,)
        print_x_name = False
    else:
        print_x_name = True
    x = [asvar(x_, sub, ds) for x_ in x]
    if cat is None:
        cat_cells = [None]
        n_cat = 1
    else:
        cat = ascategorial(cat, sub, ds)
        cat_cells = cat.cells
        n_cat = len(cat_cells)

    # correlation Dataset, nested:
    # x -> cat
    ds = Dataset()
    if print_x_name:
        ds['x'] = Factor([dataobj_repr(x_) for x_ in x], repeat=n_cat)

    if n_cat > 1:
        if isinstance(cat, Interaction):
            cat_names = [dataobj_repr(c) for c in cat.base]
            for i, name in enumerate(cat_names):
                ds[name] = Factor([cell[i] for cell in cat.cells], tile=len(x))
        elif isinstance(cat, Factor):
            cat_names = (dataobj_repr(cat),)
            ds[dataobj_repr(cat)] = Factor(cat.cells, tile=len(x))
        else:
            raise TypeError(repr(cat))
    else:
        cat_names = ()

    rs = []
    dfs = []
    ps = []
    for x_ in x:
        for cell in cat_cells:
            if cell is None:
                r, p, df = _corr(y.x, x_.x)
            else:
                sub = cat == cell
                r, p, df = _corr(y[sub].x, x_[sub].x)
            rs.append(r)
            dfs.append(df)
            ps.append(p)
    ds['r'] = Var(rs)
    ds['df'] = Var(dfs)
    p = Var(ps)
    ds['sig'] = star_factor(p)
    ds['p'] = p
    if asds:
        return ds

    table = fmtxt.Table('l' * (4 + print_x_name + len(cat_names)),
                        title="Correlations with %s" % dataobj_repr(y))
    if print_x_name:
        table.cell('x')
    table.cells(*cat_names)
    table.cells('r', 'df', '*' 'p')
    table.midrule()
    last_x = None
    for case in ds.itercases():
        if print_x_name:
            if case['x'] == last_x:
                table.cell('')
            else:
                table.cell(case['x'])
                last_x = case['x']
        for name in cat_names:
            table.cell(case[name])
        table.cell(fmtxt.stat(case['r'], '%.3f', drop0=True))
        table.cell(case['df'])
        table.cell(case['sig'])
        table.cell(fmtxt.p(case['p']))
    return table


def _corr(y: np.ndarray, x: np.ndarray):
    n = len(y)
    assert len(x) == n
    df = n - 2
    r = np.corrcoef(y, x)[0, 1]
    if r == 1:
        return r, 0., df
    t = r / np.sqrt((1 - r ** 2) / df)
    p = scipy.stats.t.sf(np.abs(t), df) * 2
    return r, p, df
