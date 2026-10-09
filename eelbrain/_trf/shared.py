from dataclasses import dataclass, fields
from functools import cached_property, reduce
from itertools import product, zip_longest
from operator import mul
from collections.abc import Sequence

import numpy as np
from numpy import newaxis
import scipy.signal
from scipy.linalg import norm

from .. import _info
from .._data_obj import CategorialArg, NDVarArg, Datalist, Dataset, NDVar, Case, UTS, dataobj_repr, ascategorial, asndvar
from .._utils import PickleableDataClass, intervals


class EQMixIn:

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for field in fields(self):
            a, b = getattr(self, field.name), getattr(other, field.name)
            if isinstance(a, b.__class__):
                if isinstance(a, np.ndarray):
                    if not np.array_equal(a, b):
                        break
                elif a != b:
                    break
            else:
                break
        else:
            return True
        return False


@dataclass(eq=False)
class Split(PickleableDataClass, EQMixIn):
    train: np.ndarray  # (, 2) array of int, segment (start, stop)
    validate: np.ndarray = None
    test: np.ndarray = None
    i_test: int = 0  # Index (to group splits with the same test segment)
    i_validate: int = None  # Index of the validation segment

    @cached_property
    def train_and_validate(self):
        return np.vstack([self.train, self.validate])


def merge_segments(
        segments: np.ndarray,
        soft_splits: bool | np.ndarray = None,
):
    """Take a selection of input segments and remove soft boundaries"""
    # return out_segments
    if soft_splits is None or isinstance(soft_splits, np.ndarray) and len(soft_splits) == 0:
        return segments
    out_segments = list(segments)
    for i in range(len(out_segments) - 1, 0, -1):
        pre_seg, post_seg = out_segments[i - 1], out_segments[i]
        if pre_seg[1] >= post_seg[0]:
            if soft_splits is True or post_seg[0] in soft_splits:
                out_segments[i - 1] = [pre_seg[0], max(pre_seg[1], post_seg[1])]
                del out_segments[i]
    return np.vstack(out_segments)


@dataclass(eq=False)
class Splits(PickleableDataClass, EQMixIn):
    """The cross-validation scheme used by :func:`boosting` (:attr:`BoostingResult.splits`)"""
    splits: list[Split]
    partitions_arg: int | None
    n_partitions: int
    n_validate: int
    n_test: int
    model: CategorialArg = None
    segments: np.ndarray = None  # Original data segments
    split_segments: np.ndarray = None  # Data subdivision for splits

    def __repr__(self):
        if len(self.segments) == 1:
            desc = "continuous data"
        else:
            desc = f"{len(self.segments)} data segments"
        items = ['']
        if self.n_validate:
            items.append(f'n_validate={self.n_validate}')
        if self.n_test:
            items.append(f'n_test={self.n_test}')
        if self.model is not None:
            items.append(f'model={dataobj_repr(self.model)}')
        items = ', '.join(items)
        return f"<Splits: {desc} split into {len(self.split_segments)} sections{items}>"

    def plot(self, **kwargs):
        """Plot data splits (see :class:`plot.DataSplit` for parameters)"""
        from ..plot import DataSplit

        return DataSplit(self, **kwargs)


def split_data(
        segments: np.ndarray,  # (n, 2) array of [start, stop] indices
        partitions: int = None,  # Number of segments to split the data
        model: CategorialArg = None,  # sample evenly from cells
        data: Dataset = None,
        validate: int = 1,  # Number of segments in validation set
        test: int = 1,  # Number of segments in test set
):
    """Split data segments into train, validate and test segments"""
    if partitions and int(partitions) != partitions:
        raise TypeError(f"{partitions=}")
    if int(validate) != validate:
        raise TypeError(f"{validate=}")
    if int(test) != test:
        raise TypeError(f"{test=}")
    if partitions is not None and partitions <= validate + test:
        raise ValueError(f"{partitions=}: need at least {validate + test + 1} partitions with {validate=} and {test=}")
    partitions_arg = partitions
    assert validate >= 0
    if validate > 1:
        raise NotImplementedError
    assert test >= 0
    if test > 1:
        raise NotImplementedError
    if len(segments) == 1:
        if partitions is None:
            partitions = 5
        if model is not None:
            raise TypeError(f'model={dataobj_repr(model)!r}: model cannot be specified in unsegmented data')
        n_times = segments[0, 1] - segments[0, 0]
        split_points = np.round(np.linspace(0, n_times, partitions + 1)).astype(np.int64)
        soft_splits = split_points[1:-1]
        split_segments = np.vstack([split_points[i: i + 2] for i in range(partitions)])
        piece_partition = np.arange(partitions)
    else:
        n_segments = len(segments)
        # determine model cells
        if model is None:
            categories = [None]
            cell_size = n_segments
        else:
            model = ascategorial(model, data=data, n=n_segments)
            categories = [np.flatnonzero(model == cell) for cell in model.cells]
            cell_sizes = [len(i) for i in categories]
            cell_size = min(cell_sizes)
            cell_sizes_are_equal = len(set(cell_sizes)) == 1
            if partitions is None and not cell_sizes_are_equal:
                raise NotImplementedError(f'Automatic partition for variable cell size {dict(zip(model.cells, cell_sizes))}')
        # automatic selection of partitions
        if partitions is None:
            if 3 <= cell_size <= 10:
                partitions = cell_size
            else:
                raise NotImplementedError(f"Automatic partition for {cell_size} cases")
        # create segments, and assign each to a partition
        if cell_size >= partitions:
            # whole segments, interleaved within each cell
            soft_splits = None
            split_segments = segments
            piece_partition = np.empty(n_segments, np.int64)
            for cell_index in categories:
                if cell_index is None:
                    cell_index = np.arange(n_segments)
                piece_partition[cell_index] = np.arange(len(cell_index)) % partitions
        elif model is not None:
            raise NotImplementedError(f'{partitions=}: with model')
        elif partitions % cell_size == 0:
            # subdivide each segment into the same number of parts
            n_parts = partitions // cell_size
            split_segments = []
            soft_splits = []
            for start, stop in segments:
                split_points = np.round(np.linspace(start, stop, n_parts + 1)).astype(np.int64)
                soft_splits.append(split_points[1:-1])
                split_segments.extend(split_points[i: i + 2] for i in range(n_parts))
            soft_splits = np.concatenate(soft_splits)
            split_segments = np.vstack(split_segments)
            piece_partition = np.arange(partitions)
        else:
            # partitions with equal numbers of samples, cut additionally at segment boundaries
            split_points = np.round(np.linspace(segments[0, 0], segments[-1, 1], partitions + 1)).astype(np.int64)
            boundaries = segments[1:, 0]
            cuts = np.union1d(split_points, boundaries)
            split_segments = np.vstack([cuts[:-1], cuts[1:]]).T
            soft_splits = np.setdiff1d(split_points[1:-1], boundaries)
            piece_partition = np.searchsorted(split_points, split_segments[:, 0], side='right') - 1
    # create actual splits
    splits = []  # list of Split
    test_iter = range(partitions) if test else [None]
    validate_iter = range(partitions) if validate else [None]
    for i_test in test_iter:
        for i_validate in validate_iter:
            if i_test == i_validate:
                continue
            train_set = np.ones(len(split_segments), bool)
            # test set
            if i_test is None:
                test_segments = None
            else:
                test_set = piece_partition == i_test
                train_set ^= test_set
                test_segments = merge_segments(split_segments[test_set], soft_splits)
            # validation set
            if i_validate is None:
                validate_segments = None
            else:
                validate_set = piece_partition == i_validate
                train_set ^= validate_set
                validate_segments = merge_segments(split_segments[validate_set], soft_splits)
            # create split
            train_segments = merge_segments(split_segments[train_set], soft_splits)
            splits.append(Split(train_segments, validate_segments, test_segments, i_test, i_validate))
    return Splits(splits, partitions_arg, partitions, validate, test, model, segments, split_segments)


def _flatten(
        segments: list[np.ndarray],
        owned: bool,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    """Concatenate segments along time, and replace them with views into the result

    Parameters
    ----------
    segments
        Per-segment arrays with time as the last axis.
    owned
        Whether ``segments`` are owned by the caller (and may be modified in place).

    Returns
    -------
    flat
        Concatenated array.
    views
        Views into ``flat``, one per segment.
    owned
        Whether ``flat`` is owned by the caller.
    """
    if len(segments) == 1:
        return segments[0], segments, owned
    flat = np.concatenate(segments, axis=-1)
    return flat, _segment_views(flat, [seg.shape[-1] for seg in segments]), True


def _segment_views(
        flat: np.ndarray,
        n_times: Sequence[int],
) -> list[np.ndarray]:
    "Views into ``flat``, one per segment"
    stops = np.cumsum(n_times)
    return [flat[..., start:stop] for start, stop in zip(stops - n_times, stops)]


def _copy_segments(
        segments: list[np.ndarray],
        flat: np.ndarray | None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    "Copy segment data, keeping the segments as views into ``flat`` when it exists"
    if flat is None:
        return [seg.copy() for seg in segments], None
    flat = flat.copy()
    return _segment_views(flat, [seg.shape[-1] for seg in segments]), flat


def _case_segments(cases: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None, bool]:
    """Split a ``(n_cases, n_rows, n_times)`` array into per-case segments

    Returns
    -------
    segments
        One ``(n_rows, n_times)`` array per case.
    flat
        For a single row, the concatenated ``(1, n_cases * n_times)`` array,
        which is a reshape of ``cases`` (a view when possible); otherwise
        ``None`` (concatenating requires a transposed copy).
    owned
        Whether the data is a copy (and may be modified in place).
    """
    n_cases, n_rows, n_times = cases.shape
    if n_rows != 1:
        return list(cases), None, False
    flat = cases.reshape((1, -1))
    return _segment_views(flat, [n_times] * n_cases), flat, not np.may_share_memory(flat, cases)


def _segment_range(segments: list[np.ndarray]) -> np.ndarray:
    "Range (max - min) of each row along time, across segments (NaN for rows containing NaN)"
    return np.max([seg.max(-1) for seg in segments], 0) - np.min([seg.min(-1) for seg in segments], 0)


def _center_and_scale(
        segments: list[np.ndarray],
        error: str,
        n_vector: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize segments in place, each row across all segments

    Parameters
    ----------
    segments
        Per-segment arrays, ``(n_rows, n_times_i)``.
    error
        Scale by the mean absolute value (``'l1'``) or the standard deviation (``'l2'``).
    n_vector
        For vector data, the number of components in each vector; rows are then
        scaled by the vector norm, and the scale has ``n_rows / n_vector`` entries.

    Returns
    -------
    mean
        The mean of each row, which was subtracted.
    scale
        The scale of each row (or vector), by which the row was divided.
    """
    n = sum(seg.shape[-1] for seg in segments)
    mean = sum(seg.sum(-1) for seg in segments) / n
    for seg in segments:
        seg -= mean[:, newaxis]
    if n_vector:
        magnitudes = [norm(seg.reshape((-1, n_vector, seg.shape[-1])), axis=1) for seg in segments]
    else:
        magnitudes = segments
    if error == 'l1':
        scale = sum(np.abs(x).sum(-1) for x in magnitudes) / n
    elif error == 'l2':
        scale = (sum((x ** 2).sum(-1) for x in magnitudes) / n) ** 0.5
    else:
        raise RuntimeError(f"{error=}")
    row_scale = np.repeat(scale, n_vector) if n_vector else scale
    for seg in segments:
        seg /= row_scale[:, newaxis]
    return mean, scale


def lag_matrix(
        x: np.ndarray,
        i_start: int,
        n_lags: int,
        pad: float = 0,
) -> np.ndarray:
    """Matrix of lagged copies of a time series

    Parameters
    ----------
    x
        Time series, ``(n_times,)``.
    i_start
        Lag (in samples) of the first column.
    n_lags
        Number of lags (columns).
    pad
        Value representing ``x`` outside of its time axis.

    Returns
    -------
    lagged
        Array ``(n_times, n_lags)`` with ``lagged[t, j] = x[t - i_start - j]``, which
        is the convention of :func:`convolve`: the kernel sample ``j`` applies to
        ``x`` lagged by ``i_start + j`` samples.

    Notes
    -----
    The result is a view into a padded copy of ``x``, and is thus cheap to
    create; it should not be modified.
    """
    n_times = len(x)
    i_stop = i_start + n_lags - 1  # largest lag
    pad_head = max(i_stop, 0)
    pad_tail = max(-i_start, 0)
    padded = np.concatenate([np.full(pad_head, pad, x.dtype), x, np.full(pad_tail, pad, x.dtype)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, n_lags)
    # windows[k, m] = padded[k + m]; lagged[t, j] = padded[pad_head + t - i_start - j]
    k0 = pad_head - i_stop
    return windows[k0:k0 + n_times, ::-1]


class PredictorData:
    """Restructure model NDVars (like DeconvolutionData but for x only)

    Attributes
    ----------
    x_segments : list of array
        Predictor data for each segment, ``(n_predictors, n_times_i)``.
    data : array
        All segments concatenated along time, ``(n_predictors, n_times_flat)``;
        built from :attr:`x_segments` when first accessed, after which the
        segments are views into it.
    segments : array
        ``(n_segments, 2)`` array of segment ``[start, stop]`` indices on the
        concatenated time axis.
    """

    def __init__(
            self,
            x: NDVarArg | Sequence[NDVarArg],
            data: Dataset = None,
            copy: bool = False,
    ):
        if isinstance(x, (NDVar, Datalist, str)):
            multiple_x = False
            xs = [asndvar(x, data=data, ragged=True)]
            x_name = xs[0].name
        else:
            multiple_x = True
            xs = [asndvar(x_, data=data, ragged=True) for x_ in x]
            if len(xs) == 0:
                raise ValueError(f"{x=} of length 0")
            x_name = [x_.name for x_ in xs]
        is_ragged = not isinstance(xs[0], NDVar)
        if is_ragged:
            has_case = True
            n_cases = len(xs[0])
            if not all(len(xi) == n_cases for xi in xs[1:]):
                raise ValueError(f'x={xs}: different number of items')
            time_dim = [x0j.get_dim('time') for x0j in xs[0]]
            for xi in xs:
                if any(xij.get_dim('time') != time_x0j for xij, time_x0j in zip(xi, time_dim)):
                    raise ValueError("Not all NDVars in x have matching time dimensions")
            n_times = [len(uts) for uts in time_dim]
        else:
            time_dim = xs[0].get_dim('time')
            if any(xi.get_dim('time') != time_dim for xi in xs[1:]):
                raise ValueError("Not all NDVars in x have matching time dimensions")
            n_times = len(time_dim)

            # determine cases (used as segments)
            has_case = n_cases = None
            for xi in xs:
                # determine cases
                if n_cases is None:
                    has_case = xi.has_case
                    n_cases = len(xi) if xi.has_case else 0
                elif xi.has_case ^ has_case:
                    raise ValueError(f'x={xs}: some but not all x have case')
                elif has_case and len(xi) != n_cases:
                    raise ValueError(f'x={xs}: not all items have the same number of cases')
        case_to_segments = bool(has_case) and not is_ragged
        n_segments = n_cases if has_case else 1
        if is_ragged:
            segment_n_times = n_times
        else:
            segment_n_times = [n_times] * n_segments
        stops = np.cumsum(segment_n_times, dtype=np.int64)
        segments = np.hstack(((stops - segment_n_times)[:, newaxis], stops[:, newaxis]))

        # x_segments: list of (n_x, n_times_i) arrays
        x0s = [xi[0] for xi in xs] if is_ragged else xs
        x_dimnames = [xi.get_dimnames(first='case' if case_to_segments else None, last='time') for xi in x0s]
        n_leading = 1 if case_to_segments else 0
        x_dims = [xi.get_dims(dimnames[n_leading:-1]) for xi, dimnames in zip(x0s, x_dimnames)]
        x_ns = [reduce(mul, [len(dim) for dim in dims], 1) for dims in x_dims]
        x_indexes = [start if stop - start == 1 else slice(start, stop) for start, stop in intervals(np.cumsum(x_ns), first=0)]
        if is_ragged:
            x_arrays = [[xij.get_data(dimnames).reshape((n, -1)) for xij in xi] for xi, dimnames, n in zip(xs, x_dimnames, x_ns)]
        elif case_to_segments:
            x_arrays = [xi.get_data(dimnames).reshape((n_cases, n, n_times)) for xi, dimnames, n in zip(xs, x_dimnames, x_ns)]
        else:
            x_arrays = [[xi.get_data(dimnames).reshape((n, n_times))] for xi, dimnames, n in zip(xs, x_dimnames, x_ns)]
        x_flat = None
        if multiple_x:
            x_segments = [np.concatenate([arrays[i] for arrays in x_arrays]) for i in range(n_segments)]
            x_owned = True
        elif case_to_segments:
            x_segments, x_flat, x_owned = _case_segments(x_arrays[0])
        else:
            x_segments = list(x_arrays[0])
            x_owned = False

        # x_meta:  meta-information for x_data
        x_meta = []
        x_names = []
        for xi, dims, index in zip(x0s, x_dims, x_indexes):
            x_repr = dataobj_repr(xi)
            if len(dims) == 0:
                x_names.append(x_repr)
            else:
                for v in product(*dims):
                    x_names.append("-".join((x_repr, *map(str, v))))
            x_meta.append((xi.name, dims, index))

        self.is_ragged = is_ragged
        self.has_case = has_case
        self.n_cases = n_cases
        self.case_to_segments = case_to_segments
        self.time_dim = time_dim
        self.n_times = n_times
        self.n_times_flat = int(stops[-1])
        self.multiple_x = multiple_x
        self.x_name = x_name
        self.x_names = x_names
        self.x_meta = x_meta
        self.x_segments = x_segments
        self.x_owned = x_owned
        self._x_flat = x_flat
        self.segments = segments
        if copy:
            if self._x_flat is None:
                self._x_flat, self.x_segments, self.x_owned = _flatten(x_segments, x_owned)
            if not self.x_owned:
                self.x_segments, self._x_flat = _copy_segments(self.x_segments, self._x_flat)
                self.x_owned = True

    @property
    def data(self) -> np.ndarray:
        "Predictors concatenated along time, ``(n_predictors, n_times_flat)``"
        if self._x_flat is None:
            self._x_flat, self.x_segments, self.x_owned = _flatten(self.x_segments, self.x_owned)
        return self._x_flat


class DeconvolutionData:
    """Restructure input NDVars into arrays for deconvolution

    The data is stored as one array per segment (trial); segments are views into
    the input :class:`NDVar` data when possible. Arrays concatenated along time
    (:attr:`y`, :attr:`x`) are built when first accessed, after which the
    segments are views into them.

    Parameters
    ----------
    y
        Dependent variable.
    x
        Predictors.
    data
        Dataset in which to evaluate ``y`` and ``x`` if they are strings.
    in_place
        Modify the data of the input NDVars in place when normalizing (saves
        memory).

    Attributes
    ----------
    y_segments : list of array
        Dependent variable for each segment, ``(n_signals, n_times_i)``.
    x_segments : list of array
        Predictors for each segment, ``(n_predictors, n_times_i)``.
    y : array
        Dependent variable concatenated along time, ``(n_signals, n_times_flat)``.
    x : array
        Predictors concatenated along time, ``(n_predictors, n_times_flat)``.
    segments : np.ndarray
        ``(n_segments, 2)`` array of segment ``[start, stop]`` indices on the
        concatenated time axis. The segments delimit chunks of continuous data,
        such as trials.
    x_meta : list of tuple
        ``(name, dims, index)`` for each predictor variable: its name, its
        dimensions other than case and time, and the index of its rows in ``x``.
    x_mean, x_scale, y_mean, y_scale : array
        Normalization that was applied to the data (``None`` before :meth:`normalize`).
    splits : list of Split
        Cross-validation scheme.
    """
    # normalization
    x_mean = None
    x_scale = None
    y_mean = None
    y_scale = None
    scale_data: str = None
    # cross-validation
    splits: Splits = None

    def __init__(
            self,
            y: NDVarArg,
            x: NDVarArg | Sequence[NDVarArg],
            data: Dataset = None,
            in_place: bool = False,
    ):
        x_data = PredictorData(x, data)

        # check y
        if isinstance(y, (list, tuple)) and isinstance(y[0], str):
            raise TypeError(f"{y=}: need a single NDVar (or list with ragged trials) as dependent variable")
        y = asndvar(y, data=data, ragged=x_data.is_ragged)
        if x_data.is_ragged:
            n_cases = len(y)
            y0 = y[0]
            y_time_dim = [yi.get_dim('time') for yi in y]
        else:
            n_cases = len(y) if y.has_case else 0
            y0 = y
            y_time_dim = y.get_dim('time')
            if y.has_case ^ x_data.has_case:
                raise ValueError(f'{y=}: case dimension does not match x')
        if y_time_dim != x_data.time_dim:
            if isinstance(y_time_dim, list):
                desc = '\n'.join([f"{y_time}  {x_time}" for y_time, x_time in zip_longest(y_time_dim, x_data.time_dim)])
            else:
                desc = f"y_time={y_time_dim!r}; x_time={x_data.time_dim!r}"
            raise ValueError(f"y does not have the same time dimension as x:\n{desc}")
        elif n_cases != x_data.n_cases:
            raise ValueError(f'{y=}: different number of cases from x ({x_data.n_cases})')

        # vector dimension
        vector_dims = [dim.name for dim in y0.dims if dim._adjacency_type == 'vector']
        if not vector_dims:
            vector_dim = None
        elif len(vector_dims) == 1:
            vector_dim = y0.get_dim(vector_dims.pop())
        else:
            raise NotImplementedError(f"{y=}: more than one vector dimension ({', '.join(vector_dims)})")

        # y_data: flatten to ydim x time array
        last = ('time',)
        n_ydims = -1
        if x_data.case_to_segments:
            last = ('case', *last)
            n_ydims -= 1
        if vector_dim:
            last = (vector_dim.name, *last)
        y_dimnames = y0.get_dimnames(last=last)
        ydims = y0.get_dims(y_dimnames[:n_ydims])
        n_flat = reduce(mul, map(len, ydims), 1)
        y_flat = None
        y_owned = False
        if x_data.is_ragged:
            y_segments = [yi.get_data(y_dimnames).reshape((n_flat, -1)) for yi in y]
        elif x_data.case_to_segments:
            y_array = y.get_data(('case', *y_dimnames[:n_ydims], 'time')).reshape((n_cases, n_flat, x_data.n_times))
            y_segments, y_flat, y_owned = _case_segments(y_array)
        else:
            y_segments = [y.get_data(y_dimnames).reshape((n_flat, x_data.n_times))]
        self.time = x_data.time_dim[0] if x_data.is_ragged else x_data.time_dim
        self.segments = x_data.segments
        self.shortest_segment_n_times = np.min(np.diff(x_data.segments, axis=1))
        self.in_place = in_place
        # y
        self.y_segments = y_segments  # [(n_signals, n_times_i), ...]
        self._y_flat = y_flat
        self._y_owned = y_owned
        self.y_name = y.name
        self._y_repr = dataobj_repr(y)
        self.y_info = _info.copy(y0.info)
        self.ydims = ydims  # without case and time
        self.yshape = tuple(map(len, ydims))
        self.full_y_dims = None if x_data.is_ragged else y.get_dims(y_dimnames)
        self.vector_dim = vector_dim  # vector dimension
        # x
        self.x_segments = x_data.x_segments  # [(n_predictors, n_times_i), ...]
        self._x_flat = x_data._x_flat
        self._x_owned = x_data.x_owned
        self.x_name = x_data.x_name
        self.x_names = x_data.x_names
        self.x_meta = x_data.x_meta  # [(x.name, xdim, index), ...]; index is int or slice
        self.multiple_x = x_data.multiple_x
        # basis
        self.basis = 0
        self.basis_window = None

    def __getstate__(self) -> dict:
        # the segments hold all the data; a concatenated copy is rebuilt on demand
        state = {**self.__dict__, '_y_flat': None, '_x_flat': None, '_y_owned': True, '_x_owned': True}
        return state

    @property
    def y(self) -> np.ndarray:
        "Dependent variable concatenated along time, ``(n_signals, n_times_flat)``"
        if self._y_flat is None:
            self._y_flat, self.y_segments, self._y_owned = _flatten(self.y_segments, self._y_owned)
        return self._y_flat

    @property
    def x(self) -> np.ndarray:
        "Predictors concatenated along time, ``(n_predictors, n_times_flat)``"
        if self._x_flat is None:
            self._x_flat, self.x_segments, self._x_owned = _flatten(self.x_segments, self._x_owned)
        return self._x_flat

    @property
    def n_x(self) -> int:
        "Number of predictor time series (rows in ``x``)"
        return len(self.x_segments[0])

    @property
    def n_y(self) -> int:
        "Number of dependent time series (rows in ``y``)"
        return len(self.y_segments[0])

    def _copy_data(self, y=False):
        "Make sure the data is a copy before modifying"
        if self.in_place:
            return
        if not self._x_owned:
            self.x_segments, self._x_flat = _copy_segments(self.x_segments, self._x_flat)
            self._x_owned = True
        if y and not self._y_owned:
            self.y_segments, self._y_flat = _copy_segments(self.y_segments, self._y_flat)
            self._y_owned = True

    def segment_slices(self, ranges: np.ndarray) -> list[tuple[int, int, int]]:
        """Map ranges on the concatenated time axis to slices of the data segments

        Parameters
        ----------
        ranges
            ``(n, 2)`` array of ``[start, stop]`` indices on the concatenated
            time axis, e.g. :attr:`Split.train`.

        Returns
        -------
        slices
            ``(i_segment, start, stop)`` for each contiguous part of the ranges,
            with ``start`` and ``stop`` relative to the segment, such that
            ``y_segments[i_segment][:, start:stop]`` is the corresponding data.
        """
        seg_starts, seg_stops = self.segments.T
        out = []
        for start, stop in ranges:
            i_first = np.searchsorted(seg_stops, start, 'right')
            i_last = np.searchsorted(seg_starts, stop, 'left')
            for i in range(i_first, i_last):
                seg_start, seg_stop = self.segments[i]
                out.append((i, int(max(start, seg_start) - seg_start), int(min(stop, seg_stop) - seg_start)))
        return out

    def apply_basis(self, basis: float, basis_window: str):
        """Apply basis to x

        Notes
        -----
        Normalize after applying basis (basis can smooth out variance).
        The basis is applied to each segment separately.
        """
        if self.basis != 0:
            raise NotImplementedError("Applying basis more than once")
        elif not basis:
            return
        self._copy_data()
        n = int(round(basis / self.time.tstep))
        w = scipy.signal.get_window(basis_window, n, False)
        if len(w) <= 1:
            raise ValueError(f"{basis=}: Window is {len(w)} samples long")
        w /= w.sum()
        for seg in self.x_segments:
            seg[:] = scipy.signal.convolve(seg, w[newaxis], 'same')
        self.basis = basis
        self.basis_window = basis_window

    @property
    def x_pads(self) -> np.ndarray:
        "Value of each predictor outside the data (the normalized value of 0)"
        if self.x_mean is None:
            return np.zeros(self.n_x)
        return -self.x_mean / self.x_scale

    def _record_x_normalization(
            self,
            x_mean: np.ndarray,
            x_scale: np.ndarray,
    ):
        "Record a normalization step applied to the current ``x``, composing it with previous ones"
        if self.x_mean is None:
            self.x_mean = x_mean
            self.x_scale = x_scale
        else:
            self.x_mean = self.x_mean + x_mean * self.x_scale
            self.x_scale = self.x_scale * x_scale

    def normalize(self, error: str, y: bool = True):
        """Center and scale the data in place

        Parameters
        ----------
        error
            Scale by the mean absolute value (``'l1'``) or the standard
            deviation (``'l2'``).
        y
            Normalize ``y`` as well as ``x`` (set to ``False`` to only
            normalize the predictors).
        """
        self._copy_data(y=y)
        x_mean, x_scale = _center_and_scale(self.x_segments, error)
        self._record_x_normalization(x_mean, x_scale)
        if y:
            n_vector = len(self.vector_dim) if self.vector_dim else 0
            y_mean, y_scale = _center_and_scale(self.y_segments, error, n_vector)
            if self.y_mean is None:
                self.y_mean = y_mean
                self.y_scale = y_scale
            else:
                self.y_mean = self.y_mean + y_mean * np.repeat(self.y_scale, n_vector or 1)
                self.y_scale = self.y_scale * y_scale
        self.scale_data = error

    def apply_x_normalization(
            self,
            x_mean: np.ndarray | float,
            x_scale: np.ndarray | float,
    ):
        """Normalize the predictors in place with given values (e.g., from a fitted model)

        Parameters
        ----------
        x_mean
            Value to subtract from each predictor (``(n_predictors,)`` array or scalar).
        x_scale
            Value by which to divide each predictor.
        """
        x_mean = np.broadcast_to(x_mean, (self.n_x,)).astype(np.float64)
        x_scale = np.broadcast_to(x_scale, (self.n_x,)).astype(np.float64)
        self._copy_data()
        for seg in self.x_segments:
            seg -= x_mean[:, newaxis]
            seg /= x_scale[:, newaxis]
        self._record_x_normalization(x_mean, x_scale)

    def _check_data(self):
        x_check = _segment_range(self.x_segments) if self.x_scale is None else self.x_scale
        y_check = _segment_range(self.y_segments) if self.y_scale is None else self.y_scale
        # check for flat data
        zero_var = y_check == 0
        if np.any(zero_var):
            raise ValueError(f"y={self._y_repr}: contains {zero_var.sum()} flat time series")
        zero_var = x_check == 0
        if np.any(zero_var):
            names = [self.x_names[i] for i in np.flatnonzero(zero_var)]
            raise ValueError(f"x: flat data in {', '.join(names)}")
        # check for NaN
        has_nan = np.isnan(y_check.sum())
        if has_nan:
            raise ValueError(f"y={self._y_repr}: contains NaN")
        has_nan = np.isnan(x_check)
        if np.any(has_nan):
            names = [self.x_names[i] for i in np.flatnonzero(has_nan)]
            raise ValueError(f"x: NaN in {', '.join(names)}")

    def initialize_cross_validation(
            self,
            partitions: int = None,  # Number of segments to split the data
            model: CategorialArg = None,  # sample evenly from cells
            data: Dataset = None,
            validate: int = 1,  # Number of segments in validation set
            test: int = 1,  # Number of segments in test set
    ):
        """Initialize cross-validation scheme

        Notes
        -----
        General solution:

         - split data into even sized segments (hard and soft splits)
         - group segments by cell-index
         - create splits
         - merge segments at soft boundaries
        """
        self.splits = split_data(self.segments, partitions, model, data, validate, test)

    def data_scale_ndvars(self):
        if self.scale_data:
            # y
            if self.y_mean is None:
                y_mean = y_scale = None
            else:
                if self.yshape:
                    y_mean = NDVar(self.y_mean.reshape(self.yshape), self.ydims, self.y_name, self.y_info)
                else:
                    y_mean = self.y_mean[0]
                # scale does not include vector dim
                if self.vector_dim:
                    dims = self.ydims[:-1]
                    shape = self.yshape[:-1]
                else:
                    dims = self.ydims
                    shape = self.yshape
                if shape:
                    y_scale = NDVar(self.y_scale.reshape(shape), dims, self.y_name, self.y_info)
                else:
                    y_scale = self.y_scale[0]
            # x
            x_mean = []
            x_scale = []
            for name, xdims, index in self.x_meta:
                if xdims:
                    shape = [len(dim) for dim in xdims]
                    x_mean.append(NDVar(self.x_mean[index].reshape(shape), xdims, name))
                    x_scale.append(NDVar(self.x_scale[index].reshape(shape), xdims, name))
                else:
                    x_mean.append(self.x_mean[index])
                    x_scale.append(self.x_scale[index])
            if self.multiple_x:
                x_mean = tuple(x_mean)
                x_scale = tuple(x_scale)
            else:
                x_mean = x_mean[0]
                x_scale = x_scale[0]
        else:
            y_mean = y_scale = x_mean = x_scale = None
        return y_mean, y_scale, x_mean, x_scale

    def package_kernel(self, h, tstart):
        """Package kernel as NDVar

        Parameters
        ----------
        h : array  (n_y, n_x, n_times)
            Kernel data.
        """
        h_time = UTS(tstart, self.time.tstep, h.shape[-1], self.time.unit)
        hs = []
        if self.scale_data:
            info = _info.for_normalized_data(self.y_info, 'Response')
        else:
            info = self.y_info

        for name, xdims, index in self.x_meta:
            dims = (*self.ydims, *xdims, h_time)
            shape = [len(dim) for dim in dims]
            x = h[:, index, :].reshape(shape)
            hs.append(NDVar(x, dims, name, info))

        if self.multiple_x:
            return tuple(hs)
        else:
            return hs[0]

    def package_value(
            self,
            value: np.ndarray,  # data
            name: str,  # NDVar name
            info: dict = None,  # NDVar info
            meas: str = None,  # for NDVar info
    ):
        if not self.yshape:
            return value[0]

        # shape
        has_vector = value.shape[0] > self.yshape[0]
        if self.vector_dim and not has_vector:
            dims = self.ydims[:-1]
            shape = self.yshape[:-1]
        else:
            dims = self.ydims
            shape = self.yshape
        if not dims:
            return value[0]
        elif len(shape) > 1:
            value = value.reshape(shape)

        # info
        if meas:
            info = _info.for_stat_map(meas, old=info)
        elif info is None:
            info = self.y_info

        return NDVar(value, dims, name, info)

    def package_y_like(self, data, name):
        if not self.full_y_dims:
            raise NotImplementedError
        shape = tuple(map(len, self.full_y_dims))
        data = data.reshape(shape)
        # roll Case to first axis
        for axis, dim in enumerate(self.full_y_dims):
            if isinstance(dim, Case):
                data = np.rollaxis(data, axis)
                dims = list(self.full_y_dims)
                dims.insert(0, dims.pop(axis))
                break
        else:
            dims = self.full_y_dims
        return NDVar(data, dims, name)
