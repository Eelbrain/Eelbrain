# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import floor
from typing import Sequence

import numpy as np


FULL_SLICE = slice(None)
FULL_AXIS_SLICE = (FULL_SLICE,)
INT_TYPES = (int, np.integer)


def digitize_index(index: float, values: np.ndarray, tol: float = None):
    """Locate a scalar ``index`` on ``values``

    Parameters
    ----------
    index : scalar
        Index to locate on ``values``.
    values : array
        1-dimensional array on which to locate ``index``.
    tol : float
        Tolerance for suppressing an IndexError when index only falls near a
        value.
    """
    i = int(np.digitize(index, values, True))
    if index == values[i]:
        return i
    elif not tol:
        raise IndexError(f"{index}: does not match any value")
    elif i > 0 and values[i] - index > index - values[i - 1]:
        i -= 1
    diff = abs(index - values[i])
    if i == 0:
        if len(values) == 1:
            if diff <= abs(values[0]) * tol:
                return i
            else:
                raise IndexError("Index %r outside of tolerance from only "
                                 "value %r" % (index, values[0]))
        elif diff <= abs(values[1] - values[0]) * tol:
            return i
        else:
            raise IndexError("Index %r outside of tolerance" % (index,))
    elif diff <= abs(values[i] - values[i - 1]) * tol:
        return i
    else:
        raise IndexError("Index %r outside of tolerance" % (index,))


def digitize_slice_endpoint(index, values):
    """Locate a scalar slice endpoint on ``values``

    (Whenever index is between two values, move to the larger)

    Parameters
    ----------
    index : scalar
        Index to locate on ``values``.
    values : array
        1-dimensional array on which to locate ``index``.
    """
    return int(np.digitize(index, values, True))


def index(index, at: int):
    return FULL_AXIS_SLICE * at + (index,)


def index_to_bool_array(index, n):
    if isinstance(index, np.ndarray):
        if index.dtype.kind == 'b':
            return index
    out = np.zeros(n, bool)
    out[index] = True
    return out


def index_to_int_array(index, n):
    if isinstance(index, np.ndarray):
        if index.dtype.kind == 'i':
            return index
        elif index.dtype.kind == 'b':
            return np.flatnonzero(index)
    return np.arange(n)[index]


def index_length(index, n):
    "Length of an array index (number of selected elements)"
    if isinstance(index, slice):
        start = index.start or 0
        if start < 0:
            start = n - start
        stop = n if index.stop is None else index.stop
        if stop < 0:
            stop = n - stop
        step = index.step or 1
        return floor((stop - start) / step)

    if not isinstance(index, np.ndarray):
        index = np.asarray(index)
    if index.dtype.kind == 'b':
        if len(index) > n:
            index = index[:n]
        return index.sum()
    elif index.dtype.kind in 'iu':
        return len(index)
    else:
        raise TypeError("index %r" % (index,))


def apply_numpy_index(data, index):
    "Apply numpy index to non-numpy sequence"
    if isinstance(index, (int, slice)):
        return data[index]
    elif isinstance(index, list):
        if len(index) == 0:
            return ()
        elif isinstance(index[0], bool):
            return (item for i, item in zip(index, data) if i)
        else:
            return (data[i] for i in index)
    array = np.asarray(index)
    assert array.ndim == 1, "Index must be 1 dimensional, got %r" % (index,)
    if array.dtype.kind == 'i':
        return (data[i] for i in array)
    elif array.dtype.kind == 'b':
        assert len(array) == len(data), "Index must have same length as data"
        return (d for d, i in zip(data, array) if i)
    raise TypeError("Invalid numpy-like index: %r" % (index,))


def optimize_index(indexes: Sequence[int]):
    "If possible, turn integer-array index into slice"
    unique = np.unique(np.diff(indexes))
    if len(unique) == 1:
        step = unique[0]
        end = indexes[-1] + step
        if end < 0:
            end = None
        return slice(indexes[0], end, step)
    return indexes


def slice_to_arange(s, length):
    """Convert a slice into a numerical index

    Parameters
    ----------
    s : slice
        Slice.
    length : int
        Length of the target for the index (only needed if ``slice.stop`` is
        None).

    Returns
    -------
    arange : array of int
        Numerical index equivalent to slice.
    """
    if s.start is None:
        start = 0
    else:
        start = s.start

    if s.stop is None:
        stop = length
    else:
        stop = s.stop

    return np.arange(start, stop, s.step)


def aindex(axis, index):
    return FULL_AXIS_SLICE * axis + (index,)


def aslice(axis, start=None, stop=None, step=None):
    return FULL_AXIS_SLICE * axis + (slice(start, stop, step),)


def take_slice(x, axis, start=None, stop=None, step=None):
    """Like :func:`numpy.take` but with a slice"""
    return x[aslice(axis, start, stop, step)]


def deep_array(data, axis, ndim):
    "Array in which ``data`` lies along ``axis`` of ``ndim`` dimensions"
    if axis < 0:
        axis += ndim
        assert axis >= 0
    out = np.array(data)
    out.shape = (1,) * axis + out.shape + (1,) * (ndim - axis - out.ndim)
    return out
