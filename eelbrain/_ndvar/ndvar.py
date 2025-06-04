# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""NDVar operations

NDVar methods should work on all NDVars; This module defines operations that
depend on the presence of specific dimensions as functions, as well as
operations that operate on more than one NDVar.
"""
from collections import defaultdict
from copy import copy
from functools import reduce
from itertools import groupby, repeat, zip_longest
from math import floor
from numbers import Real
import operator
from typing import Any, Callable, Literal, Sequence, Union

import mne
import numpy as np
from numpy import newaxis
import numpy
from scipy import linalg, ndimage, signal, stats

from .. import _info, mne_fixes
from .._data_obj import NDVarArg, CategorialArg, Dataset, NDVar, Var, Factor, Case, Categorial, Dimension, Scalar, SourceSpace, SourceSpaceBase, UTS, ascategorial, asndvar, isnumeric, op_name
from .._exceptions import DimensionMismatchError
from .._external.colorednoise import powerlaw_psd_gaussian
from .._info import merge_info
from .._mne import complete_source_space
from .._stats.adjacency import Adjacency
from .._stats.adjacency import find_peaks as _find_peaks
from .._trf._fit_metrics import error_for_indexes
from .._utils import deprecate_ds_arg
from .._utils.numpy_utils import aslice
from ._convolve import convolve_2d


NDNumeric = Union[NDVar, Var, np.ndarray, float]
SequenceOfNDNumeric = Sequence[NDNumeric]


class Alignement:

    def __init__(self, y, x, last=None):
        shared = set(x.dimnames).intersection(y.dimnames)
        shared_dims = list(shared)
        if isinstance(last, str):
            if last not in shared:
                raise ValueError(f'Dimension {last!r} missing but required')
            shared_dims.remove(last)
            shared_dims.append(last)
        elif last is None:
            pass
        else:
            raise TypeError(f'last={last!r}')

        # determine dimensions
        n_shared = len(shared_dims)
        x_dims = x.get_dimnames(last=shared_dims)
        y_dims = y.get_dimnames(last=shared_dims)

        self.x_all = x_dims
        self.x_only = x_dims[:-n_shared]
        self.y_all = y_dims
        self.y_only = y_dims[:-n_shared]
        self.shared = shared_dims
        self.all = self.y_only + x_dims


def concatenate(
        ndvars: Union[NDVar, Sequence[NDVar]],
        dim: Union[str, Dimension] = 'time',
        name: str = None,
        tmin: Union[float, str] = 0,
        info: dict = None,
        ravel: str = None,
):
    """Concatenate multiple NDVars

    Parameters
    ----------
    ndvars
        NDVars to be concatenated. Can also be a single 2d NDVar to concatenate
        the levels of the dimension other than ``dim``.
    dim
        Either a string specifying an existsing dimension along which to
        concatenate, or a Dimension object to create a new dimension (default
        ``'time'``).
    name
        Name the NDVar holding the result.
    tmin : float | 'first'
        Time axis start, only applies when concatenating along time dimension
        (``dim='time'``); default is 0.
        Set ``tmin='first'`` to use ``tmin`` of ``ndvars[0]``.
    info
        Info for the returned ``ndvar``.
    ravel
        If ``ndvars`` is a single NDVar with more than 2 dimensions, ``ravel``
        specifies which dimension to unravel for concatenation.

    Returns
    -------
    ndvar : NDVar
        NDVar with concatenated data.
    """
    if isinstance(ndvars, NDVar):
        if ravel is not None:
            assert ndvars.has_dim(ravel)
        elif ndvars.ndim == 2:
            ravel = ndvars.get_dimnames(last=dim)[0]
        elif ndvars.has_case:
            ravel = 'case'
        else:
            raise ValueError(f"ndvars={ndvars!r}: parameters are ambiguous since more than one dimension could be raveled for concatenation; specify ravel parameter")
        ndvars = [ndvars.sub(**{ravel: v}) for v in ndvars.get_dim(ravel)]
    elif ravel is not None:
        raise TypeError(f'ravel={ravel!r}: parameter ony applies when ndvars is an NDVar')
    else:
        ndvars = list(ndvars)
    ndvar = ndvars[0]

    # Allow objects to implement concatenation: used by BoostingResult
    if hasattr(ndvar, '_eelbrain_concatenate'):
        return ndvar._eelbrain_concatenate(ndvars, dim)

    if info is None:
        if isinstance(ndvars, NDVar):
            info = ndvars.info
        else:
            info = merge_info(ndvars)

    if dim is Case or (isinstance(dim, str) and dim == 'case'):
        n = sum(1 if not v.has_case else len(v) for v in ndvars)
        dim = Case(n)

    if isinstance(dim, Dimension):
        dim_names = ndvar.dimnames[:ndvar.has_case] + (newaxis,) + ndvar.dimnames[ndvar.has_case:]
        x = np.concatenate([v.get_data(dim_names) for v in ndvars], int(ndvar.has_case))
        dims = ndvar.dims[:ndvar.has_case] + (dim,) + ndvar.dims[ndvar.has_case:]
    else:
        axis = ndvar.get_axis(dim)
        dim_names = ndvar.get_dimnames(first=[*repeat(None, axis), dim])
        dim_obj = ndvar.dims[axis]
        if isinstance(dim_obj, SourceSpace):
            out_dim = SourceSpace._concatenate([v.get_dim(dim) for v in ndvars])
            ndvars = [complete_source_space(v, to=out_dim, mask=False) for v in ndvars]
            x = reduce(np.add, [v.get_data(dim_names) for v in ndvars])
        else:
            out_dim = dim_obj._concatenate([v.get_dim(dim) for v in ndvars])
            if isinstance(out_dim, UTS):
                if isinstance(tmin, str):
                    if tmin != 'first':
                        raise ValueError(f"tmin={tmin!r}")
                else:
                    out_dim = set_tmin(out_dim, tmin)
            x = np.concatenate([v.get_data(dim_names) for v in ndvars], axis)
        dims = ndvar.get_dims(dim_names)
        dims = (*dims[:axis], out_dim, *dims[axis + 1:])
    return NDVar(x, dims, name or ndvar.name, info)


def _concatenate_values(
        values: Sequence[Any],
        dim: str,
        key: str,  # for error message
):
    if isinstance(values[0], (NDVar, np.ndarray)):
        if isinstance(values[0], NDVar) and values[0].has_dim(dim):
            return concatenate(values, dim)
        elif not all((v == values[0]).all() for v in values[1:]):
            raise ValueError(f'Inconsistent values for {key}: {values}')
    if isinstance(values[0], Dimension):
        if values[0].name == dim:
            return values[0]._concatenate(values)
        elif not all(dim_i == values[0] for dim_i in values[1:]):
            raise ValueError(f'Inconsistent values for {key}: {values}')
        else:
            return values[:1]
    elif isinstance(values[0], (tuple, list)):
        if isinstance(values[0][0], NDVar) and values[0][0].has_dim(dim):
            items = [concatenate(items, dim) for items in zip_longest(*values)]
            if isinstance(values[0], tuple):
                items = tuple(items)
            return items
        elif isinstance(values[0][0], Dimension):
            return [_concatenate_values(items, dim, f'{key}[{i}') for i, items in enumerate(zip_longest(*values))]

        equals = []
        for values_i in zip_longest(*values):
            value_i_0 = values_i[0]
            if isinstance(value_i_0, (NDVar, np.ndarray)):
                equal = ((value_i_j == value_i_0).all() for value_i_j in values_i[1:])
            else:
                equal = (value_i_j == value_i_0 for value_i_j in values_i[1:])
            equals.append(all(equal))

        # Error message
        if not all(equals):
            msg = [f'Inconsistent values for {key}:']
            for equal, values_i in zip(equals, zip_longest(*values)):
                prefix = '=' if equal else '≠'
                if isinstance(values_i[0], NDVar):
                    msg.append(f"{prefix}: {values_i[0].name}")
                    for v in values_i:
                        msg.append(f'  {v.x}')
                else:
                    desc = '  '.join(map(repr, values_i))
                    msg.append(f"{prefix}: {desc}")
            raise ValueError('\n'.join(msg))
    elif len(list(groupby(values))) > 1:
        raise ValueError(f'Inconsistent values for {key}: {values}')
    return values[0]


def convolve(h, x, ds=None, name=None):
    """Convolve ``h`` and ``x`` along the time dimension

    Parameters
    ----------
    h : NDVar | sequence of NDVar
        Kernel.
    x : NDVar | sequence of NDVar
        Data to convolve, corresponding to ``h``.
    ds : Dataset
        If provided, elements of ``x`` can be specified as :class:`str`.
    name : str
        Name for output variable.

    Returns
    -------
    y : NDVar
        Convolution, with same time dimension as ``x``.
    """
    if isinstance(x, str):
        x = asndvar(x, data=ds)
        is_single = True
    elif isinstance(x, NDVar):
        is_single = True
    else:
        x = [asndvar(xi, data=ds) for xi in x]
        is_single = False

    if isinstance(h, NDVar) != is_single:
        raise TypeError(f"{h=}: needs to match x")

    if not is_single:
        assert len(h) == len(x)
        out = None
        for h_, x_ in zip(h, x):
            y_i = convolve(h_, x_, name=name)
            if out is None:
                out = y_i
            else:
                out += y_i
        return out

    x_time = x.get_dim('time')
    h_time = h.get_dim('time')
    if x_time.tstep != h_time.tstep:
        raise ValueError(f"{h=}: incompatible time axis (unequel tstep; x: {x_time.tstep} h: {h_time.tstep})")

    # initialize output
    a = Alignement(h, x, 'time')
    # check alignment
    shared_dims = x.get_dims(a.shared[:-1])
    if shared_dims != h.get_dims(a.shared[:-1]):
        msg = "Incompatible dimensions"
        for dim1, dim2 in zip(shared_dims, h.get_dims(a.shared[:-1])):
            if dim1 != dim2:
                msg += f"\nh: {dim1}\nx: {dim2}"
        raise DimensionMismatchError(msg)
    # output shape
    x_only_shape = [len(d) for d in x.get_dims(a.x_only)]
    h_only_shape = [len(d) for d in h.get_dims(a.y_only)]
    shared_shape = [len(d) for d in shared_dims]
    out_shape = [*x_only_shape, *h_only_shape, x_time.nsamples]
    out = np.empty(out_shape)
    # flatten arrays
    n_x_only = reduce(operator.mul, x_only_shape, 1)
    n_h_only = reduce(operator.mul, h_only_shape, 1)
    n_shared = reduce(operator.mul, shared_shape, 1)
    x_flat = x.get_data(a.x_all).reshape((n_x_only, n_shared, x_time.nsamples)).astype(float, copy=False)
    h_flat = h.get_data(a.y_all).reshape((n_h_only, n_shared, len(h_time))).astype(float, copy=False)
    out_flat = out.reshape((n_x_only, n_h_only, x_time.nsamples))
    # convolve
    h_i_start = int(round(h_time.tmin / h_time.tstep))
    segments = np.array(((0, x_time.nsamples),), np.int64)
    x_pads = np.zeros((n_x_only, n_shared))
    convolve_2d(h_flat, x_flat, x_pads, h_i_start, segments, out_flat)
    dims = x.get_dims(a.x_only) + h.get_dims(a.y_only) + (x_time,)
    return NDVar(out, dims, *op_name(x, name=name))


def correlation_coefficient(x, y, dim=None, name=None):
    """Correlation between two NDVars along a specific dimension

    Parameters
    ----------
    x : NDVar
        First variable.
    y : NDVar
        Second variable.
    dim : str | tuple of str
        Dimension over which to compute correlation (by default all shared
        dimensions).
    name : str
        Name for output variable.

    Returns
    -------
    float | NDVar
        Correlation coefficient over ``dim``. Any other dimensions in ``x`` and
        ``y`` are retained in the output.
    """
    if dim is None:
        shared = set(x.dimnames).intersection(y.dimnames)
        if not shared:
            raise ValueError("%r and %r have no shared dimensions" % (x, y))
        dims = list(shared)
    elif isinstance(dim, str):
        dims = [dim]
    else:
        dims = list(dim)
    ndims = len(dims)

    # determine dimensions
    assert x.get_dims(dims) == y.get_dims(dims)
    x_dimnames = x.get_dimnames(last=dims)[:-ndims]
    y_dimnames = y.get_dimnames(last=dims)[:-ndims]
    shared_dims = [dim for dim in x_dimnames if dim in y_dimnames]
    assert x.get_dims(shared_dims) == y.get_dims(shared_dims)
    x_only = [dim for dim in x_dimnames if dim not in shared_dims]
    y_only = [dim for dim in y_dimnames if dim not in shared_dims]

    # align axes
    x_order = shared_dims + x_only + [newaxis] * len(y_only) + dims
    y_order = shared_dims + [newaxis] * len(x_only) + y_only + dims
    x_data = x.get_data(x_order)
    y_data = y.get_data(y_order)
    # ravel axes over which to aggregate
    x_data = x_data.reshape(x_data.shape[:-ndims] + (-1,))
    y_data = y_data.reshape(y_data.shape[:-ndims] + (-1,))

    # correlation coefficient
    z_x = stats.zscore(x_data, -1, 1)
    z_y = stats.zscore(y_data, -1, 1)
    xy = z_y * z_x
    out = xy.sum(-1)
    out /= xy.shape[-1] - 1

    if np.isscalar(out):
        return float(out)
    isnan = np.isnan(out)
    if np.any(isnan):
        np.place(out, isnan, 0)
    dims = x.get_dims(shared_dims + x_only) + y.get_dims(y_only)
    if name is None:
        name = x.name
    info = _info.for_stat_map('r', old=x.info)
    return NDVar(out, dims, name, info)


def cross_correlation(in1, in2, name=None):
    """Cross-correlation between two NDVars along the time axis

    Parameters
    ----------
    in1 : NDVar  (time,)
        First NDVar.
    in2 : NDVar  (time,)
        Second NDVar.
    name : str
        Name for the new NDVar.

    Returns
    -------
    NDVar  (time,)
        Cross-correlation between ``in1`` and ``in2``, with a time axis
        reflecting time shift.
    """
    x1 = in1.get_data(('time',))
    x2 = in2.get_data(('time',))
    in1_time = in1.get_dim('time')
    in2_time = in2.get_dim('time')
    tstep = in1_time.tstep
    if in2_time.tstep != tstep:
        raise ValueError(f"in1 and in2 need to have the same tstep, got {tstep} and {in2_time.tstep}")
    nsamples = in1_time.nsamples + in2_time.nsamples - 1
    in1_i0 = -(in1_time.tmin / tstep)
    in2_i0 = -(in2_time.tmin / tstep)
    in2_rel_i0 = in2_i0 - in2_time.nsamples
    out_i0 = in1_i0 - in2_rel_i0 - 1
    tmin = -out_i0 * tstep
    time = UTS(tmin, tstep, nsamples)
    x_corr = signal.correlate(x1, x2)
    return NDVar(x_corr, (time,), *op_name(in1, '*', in2, merge_info((in1, in2)), name))


def cwt_morlet(
        y: NDVar,
        frequencies: Union[Sequence[float], float],
        use_fft: bool = True,
        n_cycles: Union[float, Sequence[float]] = 3.0,
        zero_mean: bool = True,
        output: Literal['complex', 'power', 'phase', 'magnitude'] = 'magnitude',
        decim: int = 1,
        n_jobs: int = 1,
) -> NDVar:
    """Time frequency decomposition with Morlet wavelets (mne-python)

    Parameters
    ----------
    y
        Input signal.
    frequencies
        Frequencies of interest. For a scalar, the output will not contain a
        frequency dimension.
    use_fft
        Compute convolution with FFT or temporal convolution.
    n_cycles
        Number of cycles. Fixed number or one per frequency.
    zero_mean
        Make sure the wavelets are zero mean.
    output
        Format of the data in the returned NDVar. Default is the complex wavelet
        transform.
    decim
        Decimate the time axis by this factor.
    n_jobs
        Nomber of parallel threads.

    Returns
    -------
    NDVar
        Time frequency decompositions.
    """
    if output == 'magnitude':
        magnitude_out = True
        output = 'complex'
    elif output not in ('complex', 'phase', 'power'):
        raise ValueError(f"{output=}")
    else:
        magnitude_out = False
    dimnames = y.get_dimnames(last='time')
    data = y.get_data(dimnames)
    dims = y.get_dims(dimnames)
    shape_outer = 1 if y.ndim == 1 else reduce(operator.mul, data.shape[:-1])
    data_flat = data.reshape((1, shape_outer, data.shape[-1]))
    time_dim = dims[-1]
    sfreq = 1. / time_dim.tstep
    if np.isscalar(frequencies):
        frequencies = [frequencies]
        fdim = None
    else:
        fdim = Scalar("frequency", frequencies, 'Hz')
        frequencies = fdim.values

    x_flat: np.ndarray = mne.time_frequency.tfr_array_morlet(data_flat, sfreq, frequencies, n_cycles, zero_mean, use_fft, decim, output, n_jobs)

    out_shape = list(data.shape)
    out_dims = list(dims)
    if fdim is not None:
        out_shape.insert(-1, len(fdim))
        out_dims.insert(-1, fdim)
    if decim != 1:
        n_times = x_flat.shape[-1]
        out_shape[-1] = n_times
        out_dims[-1] = UTS(time_dim.tmin, time_dim.tstep * decim, n_times)
    x = x_flat.reshape(out_shape)
    if magnitude_out:
        x = abs(x)
    info = _info.default_info('A', y.info)
    return NDVar(x, out_dims, y.name, info)


def dss(ndvar) -> (NDVar, NDVar):
    """Denoising source separation (DSS)

    Parameters
    ----------
    ndvar : NDVar (case, dim, time)
        Data to decompose. DSS is performed over the case and time dimensions.

    Returns
    -------
    to_dss : NDVar (dss, dim)
        Transform data to DSS.
    from_dss : NDVar (dim, dss)
        Reconstruct data form DSS.

    Notes
    -----
    the method is described in  [1]_. This function uses the implementation from
    the `mne-sandbox <https://github.com/mne-tools/mne-sandbox>`_.

    References
    ----------
    .. [1] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial
        filtering. Journal of Neuroscience Methods, 171(2), 331–339.
        `10.1016/j.jneumeth.2008.03.015
        <https://doi.org/10.1016/j.jneumeth.2008.03.015>`_

    """
    dim_names = ndvar.get_dimnames(('case', None, 'time'))
    x = ndvar.get_data(dim_names)
    data_dim = ndvar.get_dim(dim_names[1])
    dss_mat = mne_fixes.dss(x, return_data=False)

    # if data is rank-deficient, dss_mat is not square
    a, b = dss_mat.shape
    if a != b:
        raise ValueError("Data for DSS is rank-deficient. Did you exclude all "
                         "bad channels?")

    n_comp = len(dss_mat)
    dss_dim = Scalar('dss', np.arange(n_comp))
    to_dss = NDVar(dss_mat, (dss_dim, data_dim), 'to dss')
    from_dss = NDVar(linalg.inv(dss_mat), (data_dim, dss_dim), 'from dss')
    return to_dss, from_dss


def erode(ndvar, dim):
    dim_obj = ndvar.get_dim(dim)
    if dim_obj._adjacency_type != 'grid':
        raise NotImplementedError(f"Erosion for {dim} with {dim_obj._adjacency_type!r} adjacency")
    ax = ndvar.get_axis(dim)
    struct = np.zeros((3,) * ndvar.ndim, bool)
    index = tuple(slice(None) if i == ax else 1 for i in range(ndvar.ndim))
    struct[index] = True
    x = ndimage.binary_erosion(ndvar.x, struct)
    return NDVar(x, ndvar.dims, ndvar.name, ndvar.info)


def filter_data(
        ndvar: NDVar,
        low: float = None,
        high: float = None,
        *,
        name: str = None,
        n_jobs: Union[str, int, None] = 1,  # overhead not worth it; tested for 5 min KIT MEG @ 1000 Hz
        **filter_kwargs,
) -> NDVar:
    """Apply :func:`mne.filter.filter_data` to an NDVar

    Parameters
    ----------
    ndvar
        Data to be filtered.
    low
        Low cutoff frequency (in Hz).
    high
        High cutoff frequency (in Hz).
    name
        Name for the output :class:`NDVar`.
    **
        Parameters for :func:`mne.filter.filter_data`.

    Returns
    -------
    NDVar
        Filtered data, same dimensions as ``ndvar``. The type is always floating
        point regardless of the type of ``ndvar``.
    """
    axis = ndvar.get_axis('time')
    if axis == ndvar.ndim:
        axis = None
        data = ndvar.x
    else:
        data = ndvar.x.swapaxes(axis, -1)
    sfreq = 1. / ndvar.time.tstep

    if data.dtype != float:
        data = data.astype(float)

    filter_kwargs.setdefault('copy', True)
    x = mne.filter.filter_data(data, sfreq, low, high, n_jobs=n_jobs, **filter_kwargs)

    if axis is not None:
        x = x.swapaxes(axis, -1)
    if name is None:
        name = ndvar.name
    return NDVar(x, ndvar.dims, name, ndvar.info)


def find_intervals(ndvar, interpolate=False):
    """Find intervals from a boolean NDVar

    Parameters
    ----------
    ndvar : boolean NDVar (time,)
        Data which to convert to intervals.
    interpolate : bool
        By default, ``start`` values reflect the first sample that is ``True``
        and ``stop`` values reflect the first sample that is ``False``. With
        ``interpolate=True``, time points are shifted half a sample to the
        left. This is desirable for example when marking regions in a plot.

    Returns
    -------
    intervals : iterator over tuples
        Intervals represented as ``(start, stop)`` tuples
    """
    if ndvar.dimnames != ('time',):
        raise DimensionMismatchError("Requires NDVar with time dimension only,"
                                     "got %r" % (ndvar,))
    # make sure we have a boolean NDVar
    if ndvar.x.dtype.kind != 'b':
        ndvar = ndvar != 0
    ndvar = ndvar.astype(int)

    diff = np.diff(ndvar.x)
    if ndvar.x[0]:
        diff = np.append(1, diff)
    else:
        diff = np.append(0, diff)

    onsets = ndvar.time.times[diff == 1]
    offsets = ndvar.time.times[diff == -1]
    if ndvar.x[-1]:
        offsets = np.append(offsets, ndvar.time.tstop)

    if interpolate:
        shift = -ndvar.time.tstep / 2.
        onsets += shift
        offsets += shift

    return tuple(zip(onsets, offsets))


def find_peaks(ndvar):
    """Find local maxima in an NDVar

    Parameters
    ----------
    ndvar : NDVar
        Data in which to find peaks.

    Returns
    -------
    peaks : NDVar of bool
        NDVar that is ``True`` at local maxima.
    """
    for custom_ax, dim in enumerate(ndvar.dims):
        if dim._adjacency_type == 'custom':
            break
    else:
        custom_ax = 0

    if custom_ax:
        x = ndvar.x.swapaxes(custom_ax, 0)
        dims = list(ndvar.dims)
        dims[custom_ax], dims[0] = dims[0], dims[custom_ax]
    else:
        x = ndvar.x
        dims = ndvar.dims

    adjacency = Adjacency(dims)
    peak_map = _find_peaks(x, adjacency)
    if custom_ax:
        peak_map = peak_map.swapaxes(custom_ax, 0)
    return NDVar(peak_map, ndvar.dims, ndvar.name)


def frequency_response(b, frequencies=None):
    """Frequency response for a FIR filter

    Parameters
    ----------
    b : NDVar  (..., time, ...)
        FIR Filter.
    frequencies : int | array_like
        Number of frequencies at which to compute the response, or array with
        exact frequencies in Hz.

    Returns
    -------
    frequency_response : NDVar  (..., frequency)
        Frequency response for each filter in ``b``.
    """
    time = b.get_dim('time')
    dimnames = b.get_dimnames(last='time')
    data = b.get_data(dimnames)
    if frequencies is None or isinstance(frequencies, int):
        wor_n = None
    else:
        if isinstance(frequencies, Scalar):
            freqs = frequencies.values
        else:
            freqs = np.asarray(frequencies)
        wor_n = freqs * (2 * np.pi * time.tstep)
    orig_shape = data.shape[:-1]
    data_flat = data.reshape((-1, len(time)))
    fresps = []
    for h in data_flat:
        freqs, fresp = signal.freqz(h, worN=wor_n)
        fresps.append(fresp)
    freqs_hz = freqs / (2 * np.pi * time.tstep)
    frequency = Scalar('frequency', freqs_hz, 'Hz')
    fresps = np.array(fresps).reshape(orig_shape + (-1,))
    dims = b.get_dims(dimnames[:-1]) + (frequency,)
    return NDVar(fresps, dims, b.name, b.info)


def gaussian(center: float, width: float, time: UTS):
    """Gaussian window :class:`NDVar`

    Parameters
    ----------
    center : scalar
        Center of the window (normalized to the closest sample on ``time``).
    width : scalar
        Standard deviation of the window.
    time : UTS
        Time dimension.

    Returns
    -------
    gaussian : NDVar
        Gaussian window on ``time``.
    """
    width_i = int(round(width / time.tstep))
    n_times = len(time)
    center_i = time._array_index(center)
    if center_i >= n_times / 2:
        start = None
        stop = n_times
        window_width = 2 * center_i + 1
    else:
        start = -n_times
        stop = None
        window_width = 2 * (n_times - center_i) - 1
    window_data = signal.windows.gaussian(window_width, width_i)[start: stop]
    return NDVar(window_data, (time,))


def label_operator(labels, operation='mean', exclude=None, weights=None,
                   dim_name='label', dim_values=None):
    """Convert labeled NDVar into a matrix operation to extract label values

    Parameters
    ----------
    labels : NDVar of int
        NDVar in which each label corresponds to a unique integer.
    operation : 'mean' | 'sum'
        Whether to extract the label mean or sum.
    exclude : array_like
        Values to exclude (i.e., use ``exclude=0`` to ignore the area where
        ``labels==0``.
    weights : NDVar
        NDVar with same dimension as ``labels`` to assign weights to label
        elements.
    dim_name : str
        Name for the dimension characterized by labels (default ``"label"``).
    dim_values : dict
        Dictionary mapping label ids (i.e., values in ``labels``) to values on
        the dimension characterized by labels. If values are strings the new
        dimension will be categorical, if values are scalar it will be Scalar.
        The default values are the integers in ``labels``.

    Returns
    -------
    m : NDVar
        Label operator, ``m.dot(data)`` extracts label mean/sum.
    """
    if operation not in ('mean', 'sum'):
        raise ValueError("operation=%r" % (operation,))
    dimname = labels.get_dimnames((None,))[0]
    dim = labels.get_dim(dimname)
    if weights is not None:
        if weights.get_dim(dimname) != dim:
            raise DimensionMismatchError("weights.{0} does not correspond to "
                                         "labels.{0}".format(dimname))
        weights = weights.get_data((dimname,))
    label_data = labels.get_data((dimname,))
    label_values = np.unique(label_data)
    if exclude is not None:
        label_values = np.setdiff1d(label_values, exclude)
    # out-dim
    if dim_values is None:
        label_dim = Scalar(dim_name, label_values)
    else:
        values = tuple(dim_values[i] for i in label_values)
        if all(isinstance(v, str) for v in values):
            label_dim = Categorial(dim_name, values)
        elif all(isinstance(v, Real) for v in values):
            label_dim = Scalar(dim_name, values)
        else:
            raise TypeError("If dim_values is specified, values mus be either "
                            "all strings or all real numbers; got %r" %
                            (dim_values,))
    # construct operator
    x = np.empty((len(label_values), len(dim)))
    for v, xs in zip(label_values, x):
        np.equal(label_data, v, xs)
        if weights is not None:
            xs *= weights
        if operation == 'mean':
            xs /= error_for_indexes(xs, np.array(((0, len(xs)),), np.int64), 1)
    return NDVar(x, (label_dim, dim), labels.name)


def _sequence_elementwise(items: SequenceOfNDNumeric, np_func: Callable, name: str):
    ndvars = [x for x in items if isinstance(x, NDVar)]
    vars_ = [x for x in items if isinstance(x, Var)]
    if ndvars or vars_:
        info = merge_info(ndvars + vars_)
    else:
        info = None

    if ndvars:
        ndvar = ndvars.pop(0)
        dims = ndvar.dims
        if any(x.dims != dims for x in ndvars):
            raise NotImplementedError("NDVars with mismatching dimensions")
    else:
        dims = None
    xs = [x.x if isnumeric(x) else x for x in items]
    x = reduce(np_func, xs)
    if info is None:
        return x
    elif dims is None:
        return Var(x, name, info)
    else:
        return NDVar(x, dims, name, info)


def maximum(ndvars: SequenceOfNDNumeric, name: str = None):
    "Element-wise maximum of multiple array-like objects"
    return _sequence_elementwise(ndvars, np.maximum, name)


def minimum(ndvars: SequenceOfNDNumeric, name: str = None):
    "Element-wise minimum of multiple array-like objects"
    return _sequence_elementwise(ndvars, np.minimum, name)


def neighbor_correlation(
        x: NDVar,
        dim: str = 'sensor',
        obs: str = 'time',
        func: Callable = np.mean,
        flat: float = 0,
        name: str = None,
) -> NDVar:
    """Calculate Neighbor correlation

    Parameters
    ----------
    x
        The data.
    dim
        Dimension over which to correlate neighbors (default 'sensor').
    obs
        Dimension which provides observations over which to compute the
        correlation (default 'time').
    func
        Numpy function to summarize the neighbors. Default :func:`numpy.mean`;
        for example, use :func:`numpy.max` to test whether any neighbor is
        correlated.
    flat
        Value to substitute for flat channels (default 0).
    name
        Name for the new NDVar.

    Returns
    -------
    NDVar
        Contains for each element the averaged correlation coefficient with its
        neighbors in ``dim`` .
    """
    x = asndvar(x)
    is_flat = x.std(obs).x < 1e-25
    dim_obj = x.get_dim(dim)
    if name is None:
        name = x.name

    # find neighbors
    neighbors = defaultdict(list)
    for a, b in dim_obj.adjacency():
        if is_flat[a] or is_flat[b]:
            continue
        neighbors[a].append(b)
        neighbors[b].append(a)

    # for each point, find the average correlation with its neighbors
    data = x.get_data((dim, obs))
    cc = numpy.corrcoef(data)
    ncs = numpy.zeros(len(dim_obj))
    for i in range(len(ncs)):
        if is_flat[i]:
            ncs[i] = flat
        elif i not in neighbors:
            raise ValueError("Some elements do not have any neighbors")
        else:
            ncs[i] = func(cc[i, neighbors[i]])
    info = _info.for_stat_map('r', old=x.info)
    return NDVar(ncs, (dim_obj,), name, info)


@deprecate_ds_arg
def normalize_in_cells(
        y: NDVarArg,
        for_dim: str,
        in_cells: CategorialArg = None,
        data: Dataset = None,
        method: str = 'z-score',
) -> NDVar:
    """Normalize data in cells to make it appropriate for ANOVA [1]_

    Parameters
    ----------
    y
        Dependent variable which should be normalized.
    for_dim
        Dimension which will be included as factor in the ANOVA (e.g.,
        ``'sensor'``).
    in_cells
        Model defining the cells within which to normalize (normally the factors
        that will be used as fixed effects in the ANOVA).
    data
        Dataset containing the data.
    method : 'z-score' | 'range'
        Method used for normalizing the data:
        ``z-score``: for the data in each cell, subtract the mean and divide by
        the standard deviation (mean and standard deviation are computed after
        averaging across cases in each cell)
        ``range``: for the data in each cell, subtract minimum and then divide
        by the maximum (i.e., change the range of the data to ``(0, 1)``; range
        is computed after averaging across cases in each cell).

    Notes
    -----
    This method normalizes data by *z*-scoring.
    A common example is a by sensor interaction effect in EEG data.
    ANOVA interaction effects assume additivity, but EEG topographies depend on
    source strength in a multiplicative fashion, which can lead to spurious
    interaction effects. Normalizing in each cell of the ANOVA model controls
    for this (see [1] for details).

    Examples
    --------
    See :ref:`exa-compare-topographies`.

    References
    ----------
    .. [1] McCarthy, G., & Wood, C. C. (1985). Scalp Distributions of Event-Related Potentials—An Ambiguity Associated with Analysis of Variance Models. Electroencephalography and Clinical Neurophysiology, 61, S226–S227. `10.1016/0013-4694(85)90858-2 <https://doi.org/10.1016/0013-4694(85)90858-2>`_
    """
    y, n = asndvar(y, data=data, return_n=True)
    if in_cells is None:
        cells = [slice(None)]
    else:
        x = ascategorial(in_cells, data=data, n=n)
        cells = [x == cell for cell in x.cells]

    y = y.copy()
    for cell in cells:
        y_cell_mean = y[cell].mean('case')
        if method == 'z-score':
            y[cell] -= y_cell_mean.mean(for_dim)
            y[cell] /= y_cell_mean.std(for_dim)
        elif method == 'range':
            y_min = y_cell_mean.min(for_dim)
            y_range = y_cell_mean.max(for_dim) - y_min
            y[cell] -= y_min
            y[cell] /= y_range
    return y


def powerlaw_noise(
        dims: Union[NDVar, Dimension, Sequence[Dimension]],
        exponent: float,
        seed: Union[int, np.random.RandomState] = None,
):
    """Gaussian :math:`(1/f)^{exponent}` noise.

    Parameters
    ----------
    dims
        Shape of the noise.
    exponent
        The power-spectrum of the generated noise is proportional to
        :math:`S(f) = (1 / f)^{exponent}`

        - flicker/pink noise: ``exponent=1``
        - brown noise: ``exponent=2``
    seed
        Seed for random number generator.

    Notes
    -----
    Based on `colorednoise <https://github.com/felixpatzelt/colorednoise>`_.
    """
    # randomg number generator
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    # dimensions
    if isinstance(dims, NDVar):
        dim_objs = dims.dims
    elif isinstance(dims, Dimension):
        dim_objs = [dims]
    else:
        dim_objs = dims
    shape = [len(dim) for dim in dim_objs]
    for time_ax, dim in enumerate(dim_objs):
        if isinstance(dim, UTS):
            break
    else:
        raise ValueError(f"dims={dims!r}: No time dimension")
    if time_ax < len(shape) - 1:
        shape.append(shape.pop(time_ax))
    x = powerlaw_psd_gaussian(exponent, shape, rng)
    if time_ax < len(shape) - 1:
        x = np.moveaxis(x, -1, time_ax)
    return NDVar(x, dim_objs, name=f'(1/f)^{exponent}')


def psd_welch(ndvar, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0, n_per_seg=None):
    """Power spectral density with Welch's method

    Parameters
    ----------
    ndvar : NDVar  (..., time, ...)
        Data with time dimension.
    fmin : scalar
        Lower bound of the frequencies of interest.
    fmax : scalar
        Upper bound of the frequencies of interest.
    n_fft : int
        Length of the FFT in samples (default 256).
    n_overlap : int
        Overlap between segments in samples (default 0).
    n_per_seg : int | None
        Length of each Welch segment in samples. Smaller ``n_per_seg`` result
        in smoother PSD estimates (default ``n_fft``).

    Returns
    -------
    psd : NDVar  (..., frequency)
        Power spectral density.

    Notes
    -----
    Uses :func:`mne.time_frequency.psd_array_welch` implementation.
    """
    time_ax = ndvar.get_axis('time')
    dims = list(ndvar.dims)
    del dims[time_ax]
    last_ax = ndvar.ndim - 1
    data = ndvar.x
    if time_ax != last_ax:
        data = data.rollaxis(time_ax, last_ax + 1)
    sfreq = 1. / ndvar.time.tstep
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq, fmin, fmax, n_fft=n_fft, n_overlap=n_overlap,
        n_per_seg=n_per_seg)
    dims.append(Scalar("frequency", freqs, 'Hz'))
    return NDVar(psds, dims, ndvar.name, ndvar.info)


def rename_dim(ndvar, old_name, new_name):
    """Rename an NDVar dimension

    Parameters
    ----------
    ndvar : NDVar
        NDVar on which a dimension should be renamed.
    old_name : str
        Current name of the dimension.
    new_name : str
        New name for the dimension.

    Returns
    -------
    ndvar_out : NDVar
        Shallow copy of ``ndvar`` with the dimension ``old_name`` renamed to
        ``new_name``.

    Notes
    -----
    This is needed when creating an NDVar with two instances of the same
    dimension (for example, point spread functions, mapping :class:`SourceSpace`
    to :class:`SourceSpace`).
    """
    axis = ndvar.get_axis(old_name)
    old_dim = ndvar.dims[axis]
    new_dim = copy(old_dim)
    dims = list(ndvar.dims)
    new_dim.name = new_name
    dims[axis] = new_dim
    return NDVar(ndvar.x, dims, ndvar.name, ndvar.info)


def resample(
        ndvar: NDVar,
        sfreq: float,
        npad: Union[int, Literal['auto']] = 'auto',
        window: Union[str, tuple] = None,
        pad: str = 'edge',
        name: str = None,
        n_jobs: Union[str, int] = 1,
) -> NDVar:
    """Resample an NDVar along the time dimension

    Parameters
    ----------
    ndvar
        Input data, with :class:`UTS` time dimension.
    sfreq
        New sampling frequency.
    npad
        Number of samples for padding at the beginning and end (default is
        determined automatically).
    window
        Window applied to the signal in the fourier domain (default is no
        window; see :func:`scipy.signal.resample`).
    pad
        Padding method (default ``'edge'``; see :func:`numpy.pad` ``mode``
        parameter).
    name
        Name for the new NDVar (default is ``ndvar.name``).
    n_jobs
        Parameter for :func:`mne.filter.resample`.

    Notes
    -----
    If padding is enabled, this function uses :func:`mne.filter.resample`. If
    not, :func:`scipy.signal.resample` is used directly.

    This function can be very slow when the number of time samples is uneven
    (see :func:`scipy.signal.resample`). Using ``npad='auto'`` (default) ensures
    an optimal number of samples.
    """
    if name is None:
        name = ndvar.name
    axis = ndvar.get_axis('time')
    time: UTS = ndvar.get_dim('time')
    new_tstep = 1. / sfreq
    if npad:
        old_sfreq = 1.0 / time.tstep
        x = ndvar.x if ndvar.x.dtype.kind == 'f' else ndvar.x.astype(float)
        x = mne.filter.resample(x, sfreq, old_sfreq, npad=npad, axis=axis, window=window, pad=pad)
        new_num = x.shape[axis]
        if isinstance(ndvar.x, np.ma.masked_array):
            mask = mne.filter.resample(ndvar.x.mask.astype(float), sfreq, old_sfreq, npad=npad, axis=axis, window=window, pad=pad, n_jobs=n_jobs)
            x = np.ma.masked_array(x, mask > 0.5)
    else:
        new_num = int(floor((time.tstop - time.tmin) / new_tstep))
        # crop input data
        new_duration = new_tstep * new_num
        old_num = int(round(new_duration / time.tstep))
        if old_num == time.nsamples:
            idx = None
        else:
            idx = (slice(None),) * axis + (slice(None, old_num),)
        x = ndvar.x if idx is None else ndvar.x[idx]
        # resample
        x = signal.resample(x, new_num, axis=axis, window=window)
        if isinstance(ndvar.x, np.ma.masked_array):
            mask = ndvar.x.mask if idx is None else ndvar.x.mask[idx]
            mask = signal.resample(mask.astype(float), new_num, axis=axis, window=window)
            x = np.ma.masked_array(x, mask > 0.5)
    time_dim = UTS(time.tmin, new_tstep, new_num)
    dims = (*ndvar.dims[:axis], time_dim, *ndvar.dims[axis + 1:])
    return NDVar(x, dims, name, ndvar.info)


class Filter:
    "Filter and downsample"
    def __init__(self, sfreq=None):
        self.sfreq = sfreq

    def __repr__(self):
        args = self._repr_args()
        if self.sfreq:
            args += ', sfreq=%i' % self.sfreq
        return '%s(%s)' % (self.__class__.__name__, args)

    def _repr_args(self):
        raise NotImplementedError

    def _get_b_a(self, tstep):
        raise NotImplementedError

    def __eq__(self, other):
        return self.sfreq == other.sfreq

    def filter(self, ndvar):
        """Filter an NDVar"""
        b, a = self._get_b_a(ndvar.time.tstep)
        if not np.all(np.abs(np.roots(a)) < 1):
            raise ValueError("Filter unstable")
        x = signal.lfilter(b, a, ndvar.x, ndvar.get_axis('time'))
        out = NDVar(x, ndvar.dims, ndvar.name, ndvar.info)
        if self.sfreq:
            return resample(out, self.sfreq)
        else:
            return out

    def filtfilt(self, ndvar):
        """Forward-backward filter an NDVar"""
        b, a = self._get_b_a(ndvar.time.tstep)
        if not np.all(np.abs(np.roots(a)) < 1):
            raise ValueError("Filter unstable")
        x = signal.filtfilt(b, a, ndvar.x, ndvar.get_axis('time'))
        out = NDVar(x, ndvar.dims, ndvar.name, ndvar.info)
        if self.sfreq:
            return resample(out, self.sfreq)
        else:
            return out


class Butterworth(Filter):
    """Butterworth filter

    Parameters
    ----------
    low : scalar
        Low cutoff frequency.
    high : scalar | None
        High cutoff frequency.
    order : int
        Filter order.
    sfreq : scalar
        Downsample filtered signal to this sampling frequency.

    Notes
    -----
    Uses :func:`scipy.signal.butter`.
    """
    def __init__(self, low, high, order, sfreq=None):
        if not low and not high:
            raise ValueError("Neither low nor high set")
        self.low = low
        self.high = high
        self.order = int(order)
        Filter.__init__(self, sfreq)

    def _repr_args(self):
        return '%s, %s, %i' % (self.low, self.high, self.order)

    def __eq__(self, other):
        return Filter.__eq__(self, other) and self.low == other.low and self.high == other.high and self.order == other.order

    def _get_b_a(self, tstep):
        nyq = 1. / tstep / 2.
        if self.low and self.high:
            return signal.butter(self.order, (self.low / nyq, self.high / nyq),
                                 'bandpass')
        elif self.low:
            return signal.butter(self.order, self.low / nyq, 'highpass')
        elif self.high:
            return signal.butter(self.order, self.high / nyq, 'lowpass')
        else:
            raise ValueError("Neither low nor high set")


def segment(
        continuous: NDVar,
        times: Sequence[float],
        tstart: float,
        tstop: float,
        decim: int = 1,
) -> NDVar:
    """Segment a continuous NDVar

    Parameters
    ----------
    continuous
        NDVar with a continuous time axis.
    times
        Times for which to extract segments.
    tstart
        Start time for segments.
    tstop
        Stop time for segments.
    decim
        Decimate data after segmenting by factor ``decim`` (the default is
        ``1``, i.e. no decimation).

    Returns
    -------
    segmented_data : NDVar
        NDVar with all data segments corresponding to ``times``, stacked along
        the ``case`` axis.
    """
    if continuous.has_case:
        raise ValueError("Continuous data can't have case dimension")
    axis = continuous.get_axis('time')
    tstep = None if decim == 1 else continuous.time.tstep * decim
    segments = [continuous.sub(time=(t + tstart, t + tstop, tstep)) for t in times]

    s0_time = segments[0].time
    uts = UTS(tstart, s0_time.tstep, s0_time.nsamples)
    dims = ('case', *continuous.dims[:axis], uts, *continuous.dims[axis + 1:])
    return NDVar(np.array(tuple(s.x for s in segments)), dims, continuous.name, continuous.info.copy())


def set_adjacency(
        data: Union[NDVar, Dimension],
        dim: str = None,
        adjacency: Union[str, Sequence] = 'none',
        name: str = None,
):
    """Change the adjacency of an NDVar or a Dimension

    Parameters
    ----------
    data
        NDVar for which to change the adjacency.
    dim
        Dimension for which to set the adjacency.
    adjacency
        Adjacency between elements. Set to ``"none"`` for no connections or
        ``"grid"`` to use adjacency in the sequence of elements as connection.
        Set to :class:`numpy.ndarray` to specify custom adjacency. The array
        should be of shape (n_edges, 2), and each row should specify one
        connection [i, j] with i < j, with rows sorted in ascending order. If
        the array's dtype is uint32, property checks are disabled to improve
        efficiency.
    name
        Name for the new NDVar (default is ``data.name``).

    Returns
    -------
    data_with_adjacency
        Shallow copy of ``data`` with the new adjacency.
    """
    if isinstance(data, NDVar):
        dimension = data.get_dim(dim)
    elif isinstance(data, Dimension):
        data, dimension = None, data
    else:
        raise TypeError(f"{data=}")
    new = copy(dimension)
    new._adjacency_type, new._adjacency = dimension._adjacency_arg(adjacency)
    if data is None:
        return new
    if name is None:
        name = data.name
    axis = data.get_axis(dim)
    dims = (*data.dims[:axis], new, *data.dims[axis + 1:])
    return NDVar(data.x, dims, name, data.info)


def set_parc(
        data: Union[NDVar, SourceSpace],
        parc: Union[str, Factor],
        dim: str = 'source',
        mask: bool = False,
        name: str = None,
) -> Union[NDVar, SourceSpace]:
    """Change the parcellation of an :class:`NDVar` or :class:`SourceSpace` dimension

    Parameters
    ----------
    data
        :class:`NDVar` or :class:`SourceSpace` for which to set the
        parcellation.
    parc
        New parcellation. Can be specified as :class:`Factor` assigning a label
        to each source vertex, or a string specifying a FreeSurfer parcellation
        (stored as ``*.annot`` files in the subject's ``label`` directory).
    dim
        Name of the dimension to operate on (usually ``'source'``, the default).
    mask
        Remove ``unknown-*`` vertices.
    name
        Name for the new NDVar.

    Returns
    -------
    data_with_parc
        Shallow copy of ``data`` with the source space parcellation set to
        ``parc``.
    """
    if isinstance(data, SourceSpaceBase):
        source = out = data._copy(parc=parc)
    elif isinstance(data, NDVar):
        if name is None:
            name = data.name
        axis = data.get_axis(dim)
        source = set_parc(data.dims[axis], parc, mask=False)
        dims = (*data.dims[:axis], source, *data.dims[axis + 1:])
        out = NDVar(data.x, dims, name, data.info)
    else:
        raise TypeError(data)
    if mask:
        index = source.parc.startswith('unknown-')
        if np.any(index):
            index = np.invert(index, out=index)
            if isinstance(out, SourceSpace):
                out = out[index]
            else:
                out = out.sub(source=index)
    return out


def set_tmin(
        data: Union[NDVar, UTS],
        tmin: float = 0.,
        name: str = None,
) -> Union[NDVar, UTS]:
    """Shift the time axis of an :class:`NDVar` relative to its data

    Parameters
    ----------
    data
        :class:`NDVar` or :class:`UTS` dimension on which to set ``tmin``.
    tmin
        New ``tmin`` value (default 0).
    name
        Name for the new NDVar.

    Returns
    -------
    out
        Shallow copy of ``data`` with updated time axis.

    See Also
    --------
    set_time : Pad/crop the :class:`NDVar`
    """
    if isinstance(data, UTS):
        return UTS(tmin, data.tstep, data.nsamples)
    elif not isinstance(data, NDVar):
        raise TypeError(f'{data=}')
    if name is None:
        name = data.name
    axis = data.get_axis('time')
    old: UTS = data.dims[axis]
    dims = list(data.dims)
    dims[axis] = UTS(tmin, old.tstep, old.nsamples)
    return NDVar(data.x, dims, name, data.info)


def set_time(
        ndvar: NDVar,
        time: Union[NDVar, UTS],
        mode: str = 'constant',
        name: str = None,
        **kwargs,
):
    """Crop and/or pad an :class:`NDVar` to match the time axis ``time``

    Parameters
    ----------
    ndvar : NDVar
        Input :class:`NDVar`.
    time : UTS | NDVar
        New time axis, or :class:`NDVar` with time axis to match.
    mode : str
        How to pad ``ndvar``, see :func:`numpy.pad`.
    name
        Name for the new NDVar.
    **
        See :func:`numpy.pad`.

    See Also
    --------
    set_tmin : Shift the :class:`NDVar` on the time axis
    """
    if isinstance(time, NDVar):
        time = time.get_dim('time')
    if name is None:
        name = ndvar.name
    axis = ndvar.get_axis('time')
    ndvar_time = ndvar.get_dim('time')
    if ndvar_time.tstep != time.tstep:
        raise ValueError(f"time={time}: wrong samplingrate (ndvar tstep={ndvar_time.tstep})")
    start_pad_float = (ndvar_time.tmin - time.tmin) / time.tstep
    if 0.001 < start_pad_float % 1 < 0.999:
        raise ValueError(f"time={time}: can't align to ndvar.time={ndvar_time}")
    start_pad = int(round(start_pad_float))
    end_pad = time.nsamples - (start_pad + ndvar_time.nsamples)
    # construct index into ndvar
    start = -start_pad if start_pad < 0 else None
    stop = end_pad if end_pad < 0 else None
    x = ndvar.x[aslice(axis, start, stop)]
    if start_pad > 0 or end_pad > 0:
        no_pad = (0, 0)
        pad = (max(0, start_pad), max(0, end_pad))
        pad_width = [*repeat(no_pad, axis), pad, *repeat(no_pad, ndvar.ndim - axis - 1)]
        x = np.pad(x, pad_width, mode, **kwargs)
    dims = [*ndvar.dims[:axis], time, *ndvar.dims[axis + 1:]]
    return NDVar(x, dims, name, ndvar.info)
