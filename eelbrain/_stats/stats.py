# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistics functions that work on numpy arrays."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence, Union

import numpy as np
import scipy.stats
from scipy.linalg import inv

from .._data_obj import CategorialArg, CellArg, Dataset, FactorArg, Model, Parametrization, asarray, ascategorial, asfactor, asmodel
from . import opt
from . import vector


FLOAT64 = np.dtype('float64')


@dataclass
class DispersionSpec:
    multiplier: float = 1
    measure: Literal['SEM', 'CI', 'SD'] = 'SEM'

    @classmethod
    def from_string(cls, string: Union[str, 'DispersionSpec']):
        if isinstance(string, cls):
            return string
        m = re.match(r"^([.\d]*)(%?)(CI|SEM|SD)$", string.upper())
        if m is None:
            raise ValueError(f"{string!r}: invalid dispersion specification (available: CI, SEM, SD)")
        multiplier, perc, measure = m.groups()
        if multiplier:
            multiplier = float(multiplier)
            if perc:
                multiplier /= 100
        elif measure == 'CI':
            multiplier = .95
        else:
            multiplier = 1
        return cls(multiplier, measure)


def _as_float64(x):
    if x.dtype is FLOAT64:
        return x
    else:
        return x.astype(FLOAT64)


def lm_betas(
        y: np.ndarray,  # shape [n_cases, ...]
        p: Parametrization,  # model
) -> np.ndarray:
    """OLS regression coefficients

    Returns
    -------
    betas : array  [n_predictors, ...]
        Beta coefficients for the regression. The first beta is always the
        intercept (see Model).
    """
    y_ = y.reshape((len(y), -1))
    b = p.projector.dot(y_)
    return b.reshape((p.model.df, *y.shape[1:]))


def corr(y, x, out=None, perm=None):
    """Correlation parameter map

    Parameters
    ----------
    y : array_like, shape = (n_cases, ...)
        Dependent variable with case in the first axis and case mean zero.
    x : array_like, shape = (n_cases, )
        Covariate.

    Returns
    -------
    r : array, shape = (...)
        The correlation. Occurrence of NaN due to 0 variance in either y or x
        are replaced with 0.
    """
    if out is None and y.ndim > 1:
        out = np.empty(y.shape[1:])
    if perm is not None:
        x = x[perm]

    z_x = scipy.stats.zscore(x, ddof=1)
    z_x.shape = (len(x),) + (1,) * (y.ndim - 1)
    z_y = scipy.stats.zscore(y, ddof=1)
    z_y *= z_x
    out = z_y.sum(0, out=out)
    out /= len(x) - 1

    # replace NaN values
    isnan = np.isnan(out)
    if np.any(isnan):
        if np.isscalar(out):
            out = 0
        else:
            np.place(out, isnan, 0)
    return out


def lm_betas_se_1d(y, b, p):
    """Regression coefficient standard errors

    Parameters
    ----------
    y : array  [n_cases, n_tests]
        Dependent measure.
    b : array  [n_predictors, n_tests]
        Regression coefficients.
    p : Parametrization
        Parametrized model.
    """
    v = np.einsum('i...,i...', y, y)
    y_hat = b.T.dot(p.x.T).T
    v -= np.einsum('i...,i...', y_hat, y)
    v /= len(y) - p.x.shape[1]
    var_b = v * p.g.diagonal()[:, None]
    return np.sqrt(var_b, var_b)


def lm_t(
        y: np.ndarray,  # [n_cases, ...]
        p: Parametrization,  # Parametrized model
        out_t: np.ndarray = None,  # [n_betas, ...]
) -> (np.ndarray, np.ndarray, np.ndarray):  # [n_betas, ...]
    "Calculate t-values for regression coefficients"
    y_ = y.reshape((len(y), -1))
    b = p.projector.dot(y_)
    se = lm_betas_se_1d(y_, b, p)
    shape = (len(b), *y.shape[1:])
    b = b.reshape(shape)
    se = se.reshape((len(b), *y.shape[1:]))
    t = np.divide(b, se, out=out_t)
    return b, se, t


def residual_mean_square(y, x=None):
    """Mean square of the residuals

    Parameters
    ----------
    y : array [n, ...]
        Dependent measure.
    x : Model
        Model to account for a part of the variance.

    Returns
    -------
    mean_square : array [...]
        Estimate of the mean square within x.
    """
    n = len(y)
    assert n > 1, "Can't compute dispersion from a single measurement"
    out = np.empty(y.shape[1:])
    out_ = out.ravel()
    y_ = y.reshape((n, -1))
    if y_.dtype != float:
        y_ = y_.astype(float)
    if x is None:
        opt.ss(y_, out_)
        out_ /= n - 1
    else:
        x = asmodel(x)
        res = residuals(y_, x)
        opt.sum_square(res, out_)
        out_ /= x.df_error
    return out


def residuals(y, x):
    """Calculate residuals of y regressed on x (over the first axis)

    Parameters
    ----------
    y : array (n, ...)
        Data.
    x : Model
        Predictors.

    Returns
    -------
    residuals : array (...)
        Residuals.
    """
    n = len(y)
    x = asmodel(x)
    p = x._parametrize()
    res = np.empty(y.shape)
    y_ = y.reshape((n, -1))
    res_ = res.reshape((n, -1))
    opt.lm_res(y_, p.x, p.projector, res_)
    return res


def rms(a, axis=None):
    """Root mean square

    Parameters
    ----------
    a : array_like
        Data.
    axis : None | int | tuple of ints
        Axis or axes over which to calculate the RMS.
        The default (`axis` = `None`) is the RMS over all the dimensions of
        the input array.
    """
    square = np.square(a)
    out = square.mean(axis)
    if np.isscalar(out):
        return np.sqrt(out)
    else:
        return np.sqrt(out, out)


def rmssd(y):
    """Root mean square of successive differences

    Used for heart rate variance analysis.
    """
    assert np.ndim(y) == 1
    n = len(y)
    assert n > 1
    dy = np.diff(y)
    x = n / (n - 1) * np.sum(dy ** 2)
    return np.sqrt(x)


class Dispersion:

    def __init__(self, y, x=None, match=None):
        """Standard error of the mean (SEM)

        Parameters
        ----------
        y : array (n, ...)
            Data, first dimension reflecting cases.
        x : Categorial
            Categorial predictor for using pooled variance.
        match : Categorial
            Within-subject SEM.

        Notes
        -----
        See Loftus and Masson (1994).
        """
        if x is None or len(x.cells) == 1:
            n = len(y)
            if match is None or len(match) == len(match.cells):
                model = None
            else:
                model = match
        else:
            x = asfactor(x)
            n = x._cellsize()
            if isinstance(n, dict):
                raise NotImplementedError("SEM for unequal cell sizes")
            model = x if match is None else x + match
        sem = residual_mean_square(y, model)
        sem /= n
        np.sqrt(sem, sem)
        self.n = n
        self.model = model
        self.sem = sem

    def get(self, spec: Union[str, DispersionSpec]):
        spec = DispersionSpec.from_string(spec)
        if spec.measure == 'SEM':
            return self.sem * spec.multiplier
        elif spec.measure == 'CI':
            return self.ci(spec.multiplier)
        else:
            raise RuntimeError(spec)

    def ci(self, confidence):
        """Confidence interval based on the inverse t-test

        Parameters
        ----------
        confidence : scalar
            Confidence in the interval (i.e., .95 for 95% CI).

        Returns
        -------
        ci : array [...]
            Confidence interval (i.e., the mean of y lies within m +- ci with the
            specified confidence).

        Notes
        -----
        See `<http://en.wikipedia.org/wiki/Confidence_interval#Statistical_hypothesis_testing>`_
        """
        if self.model is None:
            df = self.n - 1
        elif isinstance(self.model, Model):
            df = self.model.df_error
        else:
            df = Model(self.model).df_error
        return self.sem * scipy.stats.t.isf((1 - confidence) / 2, df)


def t_1samp(y, out=None):
    "T-value for 1-sample t-test"
    n_cases = len(y)
    if out is None:
        out = np.empty(y.shape[1:])

    if out.ndim == 1:
        y_flat = y
        out_flat = out
    else:
        y_flat = y.reshape((n_cases, -1))
        out_flat = out.ravel()

    opt.t_1samp(y_flat, out_flat)
    return out


def t2_1samp(y, rotation=None, out=None):
    """T**2-value for 1-sample T**2-test

    Parameters
    ----------
    y : ndarray
        Vector field, shape ``(n_case, n_dims [, ...])``.
    rotation : ndarray
        Rotation matrix (for permutation).
    out : ndarray
        Container for result. Needs shape ``...``.
    """
    if y.ndim <= 1:
        raise ValueError(f'y with shape {y.shape}: T**2 statistic needs vector valued samples.')
    n_cases = y.shape[0]
    n_dims = y.shape[1]
    if out is None:
        out = np.empty(y.shape[2:])
    else:
        assert out.shape == y.shape[2:]

    if out.ndim == 1:
        y_flat = y
        out_flat = out
    else:
        y_flat = y.reshape((n_cases, n_dims, -1))
        out_flat = out.ravel()

    if rotation is None:
        vector.t2_stat(y_flat, out_flat)
    else:
        assert rotation.shape == (n_cases, 3, 3)
        vector.t2_stat_rotated(y_flat, rotation, out_flat)
    return out


def t_ind(y, group, out=None, perm=None):
    "T-value for independent samples t-test, assuming equal variance"
    n_cases = len(y)
    if out is None:
        out = np.empty(y.shape[1:])

    if perm is not None:
        group = group[perm]

    if out.ndim == 1:
        y_flat = y
        out_flat = out
    else:
        y_flat = y.reshape((n_cases, -1))
        out_flat = out.ravel()

    opt.t_ind(y_flat, out_flat, group)
    return out


def rtest_p(r, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    r = np.asanyarray(r)
    t = r * np.sqrt(df / (1 - r ** 2))
    p = ttest_p(t, df)
    return p


def rtest_r(p, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    p = np.asanyarray(p)
    t = ttest_t(p, df)
    r = t / np.sqrt(df + t ** 2)
    return r


def ttest_p(t, df, tail=0):
    """Two-tailed probability

    Parameters
    ----------
    t : array_like
        T values.
    df : int
        Degrees of freedom.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    """
    t = np.asanyarray(t)
    if tail == 0:
        t = np.abs(t)
    elif tail == -1:
        t = -t
    elif tail != 1:
        raise ValueError(f"{tail=}")
    p = scipy.stats.t.sf(t, df)
    if tail == 0:
        p *= 2
    return p


def ttest_t(p, df, tail=0):
    """Positive t value for a given probability

    Parameters
    ----------
    p : array_like
        Probability.
    df : int
        Degrees of freedom.
    tail : 0 | 1 | -1
        One- or two-tailed t-distribution (the return value is always positive):
        0: two-tailed;
        1 or -1: one-tailed).
    """
    p = np.asanyarray(p)
    if tail == 0:
        p = p / 2
    t = scipy.stats.t.isf(p, df)
    return t


def dispersion(
        y: np.ndarray,
        x: CategorialArg = None,
        match: FactorArg = None,
        spec: str = 'SEM',
        cells: Sequence[CellArg] = None,
        data: Dataset = None,
        pool: bool = None,
):
    """Calculate data dispersion measure

    Parameters
    ----------
    y
        Dependent measure.
    x
        Cells for pooling variance.
    match
        Calculate variability for related measures (Loftus & Masson 1994).
    spec
        The variability estimate. Contains an optional number, an optional
        percent-sign, and a kind ('ci' or 'sem'). Examples:
        ``sem``: Standard error of the mean;
        ``2sem``: 2 standard error of the mean;
        ``ci``: 95% confidence interval;
        ``99%ci``: 99% confidence interval.
    pool
        Pool the variability to create a single estimate (as opposed to one for
        each cell in x). Default ``True`` iff ``match`` is specified (for
        within-subject errors).
    cells
        Estimate variance in these cells of ``x`` (default ``x.cells``).

    Returns
    -------
    var : scalar | array
        Variability estimate. A single estimate if errors are pooled, otherwise
        an estimate for every cell in x.
    """
    spec_ = DispersionSpec.from_string(spec)
    y, n = asarray(y, data=data, return_n=True)
    if match is not None:
        match = asfactor(match, data=data, n=n)
    if pool is None:
        pool = match is not None
    if x is None:
        if match is not None and match.df == len(match) - 1:
            raise ValueError("Can't calculate within-subject error because the match predictor explains all variability")
    else:
        x = ascategorial(x, data=data, n=n)
        if not pool and cells is None:
            cells = x.cells

    y = np.asarray(y, np.float64)
    if spec_.measure == 'SD':
        if x is not None or match is not None:
            raise NotImplementedError(f"{spec!r} with x or match")
        out = y.std(0)
        if spec_.multiplier != 1:
            out *= spec_.multiplier
    elif pool or x is None:
        out = Dispersion(y, x, match).get(spec_)
    elif match is not None:
        raise NotImplementedError(f"{spec!r} unpooled with match")
    else:
        out = np.array([Dispersion(y[x == cell]).get(spec_) for cell in cells])

    # return scalars for 1d-arrays
    if out.ndim == 0:
        return out.item()
    else:
        return out


def xsinv(x):
    xt = x.T
    return inv(xt.dot(x)).dot(xt)
