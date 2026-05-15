"""Predictors for continuous data"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import repeat
from collections.abc import Sequence

import numpy as np

from .._data_obj import NDVarArg, NDVar, Case, Dataset, UTS, asndvar, asarray
from .._utils import deprecate_ds_arg


@deprecate_ds_arg
def epoch_impulse_predictor(
        shape: NDVarArg | tuple[int, UTS],
        value: float | Sequence[float] | str = 1,
        latency: float | Sequence[float] | str = 0,
        name: str = None,
        data: Dataset = None,
) -> NDVar:
    """Time series with one impulse for each of ``n`` epochs

    Parameters
    ----------
    shape
        Shape of the output. Can be specified as the :class:`NDVar` with the
        data to predict, or an ``(n_cases, time_dimension)`` tuple.
    value
        Scalar or length ``n`` sequence of scalars specifying the value of each
        impulse (default 1).
    latency
        Scalar or length ``n`` sequence of scalars specifying the latency of
        each impulse (default 0).
    name
        Name for the output :class:`NDVar`.
    data
        If specified, input items (``shape``, ``value`` and ``latency``) can be
        strings to be evaluated in ``data``.

    See Also
    --------
    event_impulse_predictor : for continuous time series

    Examples
    --------
    See :ref:`exa-impulse` example.
    """
    if isinstance(shape, str):
        shape = asndvar(shape, data=data)
    if isinstance(value, str):
        value = asarray(value, data=data)
    if isinstance(latency, str):
        latency = asarray(latency, data=data)

    if isinstance(shape, NDVar):
        if not shape.has_case:
            raise ValueError(f'{shape=}: has no case dimension')
        n = len(shape)
        time = shape.get_dim('time')
    else:
        n, time = shape
        if not isinstance(time, UTS):
            raise TypeError(f'{shape=}: second item needs to be UTS instance')

    x = np.zeros((n, len(time)))
    t_index = time._array_index(latency)
    if np.isscalar(latency):
        x[:, t_index] = value
    else:
        x[np.arange(n), t_index] = value
    return NDVar(x, (Case, time), name)


@deprecate_ds_arg
def event_impulse_predictor(
        shape: NDVarArg | UTS,
        time: str | Sequence[float] = 'time',
        value: float | Sequence[float] | str = 1,
        latency: float | Sequence[float] | str = 0,
        name: str = None,
        data: Dataset = None,
) -> NDVar:
    """Time series with multiple impulses

    Parameters
    ----------
    shape
        Shape of the output. Can be specified as the :class:`NDVar` with the
        data to predict, or an ``(n_cases, time_dimension)`` tuple.
    time
        Time points at which impulses occur.
    value
        Magnitude of each impulse (default 1).
    latency
        Latency of each impulse relative to ``time`` (default 0).
    name
        Name for the output :class:`NDVar`.
    data
        If specified, input items (``time``, ``value`` and ``latency``) can be
        strings to be evaluated in ``data``.

    See Also
    --------
    epoch_impulse_predictor : for epoched data (with :class:`Case` dimension and a single impulse per epoch)
    """
    if isinstance(shape, NDVar):
        uts = shape.get_dim('time')
    elif isinstance(shape, UTS):
        uts = shape
    else:
        raise TypeError(f'{shape=}')

    time, n = asarray(time, data=data, return_n=True)
    dt = uts.tstep / 2
    index = (time > uts.tmin - dt) & (time < uts.tstop - dt)
    if index.all():
        index = None
    else:
        time = time[index]

    if isinstance(value, str) or not np.isscalar(value):
        value = asarray(value, sub=index, data=data, n=n)
    else:
        value = repeat(value)

    if isinstance(latency, str) or not np.isscalar(latency):
        latency = asarray(latency, sub=index, data=data, n=n)
    else:
        latency = repeat(latency)

    out = NDVar(np.zeros(len(uts)), uts, name)
    for t, l, v in zip(time, latency, value):
        t_ = t + l
        if uts.tmin <= t_ <= uts.tmax:
            out[t_] = v
    return out
