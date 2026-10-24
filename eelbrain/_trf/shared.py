from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import reduce
from itertools import product
from math import ceil
from operator import mul
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy import newaxis
import scipy.signal
from scipy.linalg import norm

from .. import _info
from .._data_obj import CategorialArg, NDVarArg, Datalist, Dataset, Dimension, Factor, Interaction, NDVar, Case, UTS, dataobj_repr, ascategorial, asndvar
from .._utils import PickleableDataClass

if TYPE_CHECKING:
    from ..plot import DataSplit


class EQMixIn:
    """Add __eq__() for dataclasses with array and categorial fields"""

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        for field_i in fields(self):
            a, b = getattr(self, field_i.name), getattr(other, field_i.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif isinstance(a, (Factor, Interaction)) or isinstance(b, (Factor, Interaction)):
                # element-wise comparison
                if not (isinstance(b, a.__class__) and len(a) == len(b) and np.all(a == b)):
                    return False
            elif a != b:
                return False
        return True


@dataclass(eq=False)
class Split(PickleableDataClass, EQMixIn):
    """One cross-validation split of the data (see :class:`Splits`)

    A split divides the data into training segments and (optionally) one held-out
    partition. The held-out partition is either a *test* set, which is never consulted
    while estimating the kernel, or a *validation* set, which is consulted repeatedly
    during fitting (e.g., for early stopping).
    A train/test split can in turn contain train/validation splits which subdivide
    its training data.
    Each split without inner splits (a "leaf") corresponds to one boosting run.

    Attributes
    ----------
    train
        ``(n, 2)`` array of ``[start, stop]`` indices of the training segments
        (on the concatenated time axis).
    reserved
        Held-out segments, in the same format as ``train`` (``None`` if all
        data is used for training).
    i_reserved
        Index of the held-out partition (``None`` if all data is used for
        training).
    is_test
        Whether the held-out segments are a test set (``True``) or a
        validation set (``False``).
    splits
        Inner splits, which divide ``train`` into training and validation
        segments (empty for a leaf split).
    """
    train: np.ndarray
    reserved: np.ndarray = None
    i_reserved: int = None
    is_test: bool = False
    splits: list[Split] = field(default_factory=list)

    def __post_init__(self):
        if self.reserved is not None and not self.is_test:
            assert not any(split.is_test for split in self.splits), "test split inside validation split"

    @property
    def train_and_reserved(self) -> np.ndarray:
        "Training and held-out segments combined, sorted by time (all data available to the split)"
        if self.reserved is None:
            return self.train
        segments = np.vstack([self.train, self.reserved])
        return segments[np.argsort(segments[:, 0])]

    @property
    def leaves(self) -> list[Split]:
        "Splits without inner splits, each corresponding to one boosting run (``[self]`` for a leaf)"
        if not self.splits:
            return [self]
        return [leaf for split in self.splits for leaf in split.leaves]

    def __setstate__(self, state: dict):
        if 'validate' in state:
            # backwards compatibility for flat format (before nested splits introduced in 0.43)
            train, validate, test, i_test = (state[key] for key in ['train', 'validate', 'test', 'i_test'])
            i_validate = state.get('i_validate')  # not stored before BoostingResult version 16 (restored by Splits.__setstate__)
            if validate is None:
                state = {'train': train, 'reserved': test, 'i_reserved': None if test is None else i_test, 'is_test': test is not None}
            elif test is None:
                state = {'train': train, 'reserved': validate, 'i_reserved': i_validate}
            else:
                validation_split = Split(train, validate, i_validate)
                state = {'train': validation_split.train_and_reserved, 'reserved': test, 'i_reserved': i_test, 'is_test': True, 'splits': [validation_split]}
        PickleableDataClass.__setstate__(self, state)


def merge_segments(
        segments: np.ndarray,
        soft_splits: bool | np.ndarray,
) -> np.ndarray:
    """Merge adjacent segments at soft boundaries

    Parameters
    ----------
    segments
        ``(n, 2)`` array of ``[start, stop]`` indices, sorted by time.
    soft_splits
        Boundaries at which adjacent segments are merged:
        ``True`` to merge at all adjacent segments,
         an array of boundary indices to merge only at those, or
        ``False`` to return ``segments`` unchanged.
    """
    if soft_splits is False or isinstance(soft_splits, np.ndarray) and len(soft_splits) == 0:
        return segments
    out_segments = list(segments)
    for i in range(len(out_segments) - 1, 0, -1):
        pre_seg, post_seg = out_segments[i - 1], out_segments[i]
        if pre_seg[1] >= post_seg[0]:
            if soft_splits is True or post_seg[0] in soft_splits:
                out_segments[i - 1] = [pre_seg[0], max(pre_seg[1], post_seg[1])]
                del out_segments[i]
    return np.vstack(out_segments)


def _restore_validation_indices(
        splits: list[Split],
        n_partitions: int,
        i_test: int | None = None,
):
    """Backwards compatibility

    Fill in validation partition indices missing from splits pickled before
    BoostingResult version 16 (created in partition order, skipping the test
    partition)
    """
    partitions = [p for p in range(n_partitions) if p != i_test]
    for p, split in zip(partitions, splits):
        if split.i_reserved is None and split.reserved is not None:
            split.i_reserved = p


@dataclass(eq=False)
class Splits(PickleableDataClass, EQMixIn):
    """The cross-validation scheme used by :func:`boosting` (:attr:`BoostingResult.splits`)

    With a test set (``test=1``), :attr:`splits` contains one test
    :class:`Split` per partition, and each of those contains the validation
    splits of its training data (:attr:`Split.splits`). Without test set,
    :attr:`splits` contains one validation :class:`Split` per partition; with
    neither validation nor test set, a single split that uses all data for
    training.

    Attributes
    ----------
    splits
        The top-level splits.
    partitions_arg
        The ``partitions`` argument used to create the splits (``None`` for
        automatic selection).
    n_partitions
        Number of partitions the data was divided into.
    n_validate
        Number of partitions in each validation set.
    n_test
        Number of partitions in each test set.
    model
        Model used to sample partitions evenly from cells.
    segments
        ``(n_segments, 2)`` array of ``[start, stop]`` indices of the original
        data segments (on the concatenated time axis).
    split_segments
        ``(n_sections, 2)`` array of the sections into which the data was
        subdivided to form partitions.
    """
    splits: list[Split]
    partitions_arg: int | None
    n_partitions: int
    n_validate: int
    n_test: int
    model: CategorialArg = None
    segments: np.ndarray = None
    split_segments: np.ndarray = None

    def __setstate__(self, state: dict):
        # Backwards compatibility for BoostingResult version < 17
        splits = state['splits']
        if splits:
            _restore_validation_indices(splits, state['n_partitions'])
        test_splits = [split for split in splits if split.is_test] if splits else []
        if len({split.i_reserved for split in test_splits}) < len(test_splits):
            soft_splits = np.setdiff1d(state['split_segments'][1:, 0], state['segments'][:, 0])
            merged = {}
            for split in splits:
                if split.i_reserved in merged:
                    merged[split.i_reserved].splits.extend(split.splits)
                else:
                    split.train = merge_segments(split.train, soft_splits)
                    merged[split.i_reserved] = split
            state['splits'] = list(merged.values())
            for split in state['splits']:
                _restore_validation_indices(split.splits, state['n_partitions'], split.i_reserved)
        PickleableDataClass.__setstate__(self, state)

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

    @property
    def leaves(self) -> list[tuple[Split, Split]]:
        "``(top-level split, leaf split)`` for each boosting run, grouped by top-level split (see :attr:`Split.leaves`)"
        return [(top_split, leaf) for top_split in self.splits for leaf in top_split.leaves]

    def plot(self, **kwargs) -> DataSplit:
        """Plot data splits (see :class:`plot.DataSplit` for parameters)"""
        from ..plot import DataSplit

        return DataSplit(self, **kwargs)


def split_data(
        segments: np.ndarray,
        partitions: int | None = None,
        model: CategorialArg | None = None,
        data: Dataset | None = None,
        validate: int = 1,
        test: int = 1,
) -> Splits:
    """Split data segments into train, validate and test segments

    Parameters
    ----------
    segments
        ``(n_segments, 2)`` array of ``[start, stop]`` indices of the data
        segments (on the concatenated time axis).
    partitions
        Number of partitions to divide the data into (by default, determined
        from the number of segments).
    model
        Sample partitions evenly from the cells of this model (one entry per
        segment).
    data
        Dataset in which to evaluate ``model`` if it is a string.
    validate
        Number of partitions in each validation set (0 or 1).
    test
        Number of partitions in each test set (0 or 1).

    Notes
    -----
    Splitting always follows the same four steps:

    1. Divide the data into *sections*. A section is either a whole segment or
       a piece of a segment. Boundaries between the original segments are
       *hard*; boundaries introduced by cutting a segment are *soft*.
    2. Assign each section to one of ``partitions`` partitions.
    3. Create the :class:`Split` tree. With ``test=1``, create one test split
       for each partition *i*: partition *i* is held out as test set and all
       remaining partitions form the training set. With ``validate=1``, each
       split created so far (or, with ``test=0``, the whole data) is further
       divided into validation splits, one for each partition *j* in its
       training set: partition *j* is held out as validation set and the
       remaining training partitions form the training set. With ``validate=1``
       and ``test=1`` this yields *k* test splits with ``k - 1`` validation
       splits each, i.e., ``k * (k - 1)`` boosting runs (for *k* =
       ``partitions``); with one of them set to 0, it yields *k* top-level
       splits without inner splits; with both set to 0, a single split that
       uses all data for training.
    4. Within each set of a split, merge adjacent sections at soft boundaries,
       so that pieces of a segment that end up in the same set form one
       continuous segment again. Hard boundaries are never merged.

    How the sections are formed and assigned to partitions (steps 1 and 2)
    depends on the data and on the arguments:

    **Continuous data** (a single segment): the segment is cut into
    ``partitions`` contiguous sections of equal length (default 5), and each
    section is one partition. ``model`` is not allowed.

    **Segmented data with at least as many segments as partitions** (``model``
    is ``None`` and ``n_segments >= partitions``, or each cell of ``model`` has
    at least ``partitions`` segments): whole segments are used as sections. The
    segments are assigned to partitions in an interleaved fashion, i.e.,
    partition *p* receives segments ``[p::partitions]``. With ``model``, this
    is done separately within each cell, so that every partition samples each
    cell evenly. The automatic default is ``partitions = n_segments`` (number
    of segments in the smallest cell with ``model``), which amounts to
    leave-one-out and is only available for 3 to 10 segments (per cell); with
    ``model`` it further requires all cells to have the same size.

    **Segmented data with more partitions than segments** (``model`` is
    ``None`` and ``partitions > n_segments``; not implemented with ``model``):
    segments are cut into pieces, and the pieces are used as sections.

     - If ``partitions`` is a multiple of ``n_segments``, each segment is cut
       into ``partitions / n_segments`` pieces of equal length, and each piece
       is one partition. Partitions are thus numbered consecutively along the
       time axis (the first pieces all come from the first segment), and
       partition sizes depend on the length of the segment they come from.
     - Otherwise, the concatenated time axis is divided into ``partitions``
       spans of equal length, and the data is cut at these span boundaries
       (soft) as well as at all segment boundaries (hard). Each section is
       assigned to the partition whose span contains its start, so partitions
       contain (approximately) equal numbers of samples, while a span that
       straddles a segment boundary yields two sections in the same partition.

    See :ref:`exa-data_split` for illustrations of these schemes, and use
    :meth:`Splits.plot` (or :func:`plot.preview_partitions`) to inspect a
    specific configuration.
    """
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
            soft_splits = False
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
    # with more partitions than samples, some partitions are empty
    partition_n_samples = np.bincount(piece_partition, np.diff(split_segments, axis=1)[:, 0], partitions)
    if np.any(partition_n_samples == 0):
        raise ValueError(f"{partitions=}: not enough samples ({segments[-1, 1] - segments[0, 0]}) to form {partitions} partitions")
    # create actual splits

    def make_split(
            train_partitions: list[int],
            i_reserved: int | None,
            is_test: bool,
            inner_validate: bool = False,
    ) -> Split:
        "Split training on ``train_partitions`` and holding out ``i_reserved``; ``inner_validate`` to add validation splits of the training partitions"
        train = merge_segments(split_segments[np.isin(piece_partition, train_partitions)], soft_splits)
        reserved = None if i_reserved is None else merge_segments(split_segments[piece_partition == i_reserved], soft_splits)
        if inner_validate:
            inner_splits = [make_split([p for p in train_partitions if p != i], i, False) for i in train_partitions]
        else:
            inner_splits = []
        return Split(train, reserved, i_reserved, is_test, inner_splits)

    all_partitions = list(range(partitions))
    if test:
        splits = [make_split([p for p in all_partitions if p != i], i, True, bool(validate)) for i in all_partitions]
    elif validate:
        splits = [make_split([p for p in all_partitions if p != i], i, False) for i in all_partitions]
    else:
        splits = [make_split(all_partitions, None, False)]
    return Splits(splits, partitions_arg, partitions, validate, test, model, segments, split_segments)


def _segment_views(
        flat: np.ndarray,
        n_times: Sequence[int],
) -> list[np.ndarray]:
    "Views into ``flat``, one per segment"
    stops = np.cumsum(n_times)
    return [flat[..., start:stop] for start, stop in zip(stops - n_times, stops)]


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


def _ceil_samples(x: float) -> int:
    "Round up, treating a value within floating point error of an integer as that integer"
    i = round(x)
    return i if abs(x - i) < 1e-6 else ceil(x)


def lag_window(
        tstart: float | Sequence[float],
        tstop: float | Sequence[float] | None,
        tstep: float,
        n_predictors: int,
        tmax: float | Sequence[float] | None = None,
) -> tuple[list[int], list[int]]:
    """Lag window of each predictor in samples

    Parameters
    ----------
    tstart
        Start of the window in seconds (the first lag). A scalar applies to
        all predictors; a sequence specifies one value per predictor.
    tstop
        End of the window in seconds, exclusive (like :attr:`UTS.tstop`); in
        the same format as ``tstart``.
    tstep
        Time step in seconds.
    n_predictors
        Number of predictors.
    tmax
        Alternative to ``tstop``: the last lag in seconds, inclusive (like
        :attr:`UTS.tmax`).

    Returns
    -------
    i_start
        First lag of each predictor in samples.
    i_stop
        End of each predictor's window in samples (exclusive): the kernel for
        predictor ``i`` has ``i_stop[i] - i_start[i]`` samples.

    Notes
    -----
    ``tstart`` and ``tmax`` are rounded to the nearest sample. ``tstop`` is
    rounded up (allowing for floating point error), so that the window includes
    every lag before ``tstop``.
    """
    if (tstop is None) == (tmax is None):
        raise TypeError("exactly one of tstop and tmax is required")

    def per_predictor(t: float | Sequence[float], name: str) -> list[float]:
        if isinstance(t, (Sequence, np.ndarray)):
            values = list(t)
            if len(values) != n_predictors:
                raise ValueError(f"{name}={t!r}: need one value per predictor ({n_predictors})")
            return values
        return [t] * n_predictors

    i_start = [int(round(t / tstep)) for t in per_predictor(tstart, 'tstart')]
    if tstop is None:
        i_stop = [int(round(t / tstep)) + 1 for t in per_predictor(tmax, 'tmax')]
    else:
        i_stop = [_ceil_samples(t / tstep) for t in per_predictor(tstop, 'tstop')]
    for i, (start, stop) in enumerate(zip(i_start, i_stop)):
        if stop <= start:
            desc = f'{tstart=}, {tmax=}' if tstop is None else f'{tstart=}, {tstop=}'
            raise ValueError(f"{desc}: empty lag window for predictor {i}")
    return i_start, i_stop


@dataclass
class RowBlock:
    """Rows of a :class:`SegmentedArray` that stem from one :class:`NDVar`

    Attributes
    ----------
    name
        Name of the NDVar.
    dims
        Dimensions of the NDVar other than case and time (the vector dimension
        last).
    index
        Rows of the block in the array (the flattened ``dims``).
    info
        Info dictionary of the NDVar (not considered for equality: two blocks
        are equal when they describe the same rows).
    """
    name: str | None
    dims: tuple[Dimension, ...]
    index: slice
    info: dict = field(compare=False)

    @property
    def shape(self) -> tuple[int, ...]:
        "Shape corresponding to :attr:`dims`"
        return tuple(len(dim) for dim in self.dims)

    @property
    def n_rows(self) -> int:
        "Number of rows in the block"
        return self.index.stop - self.index.start

    def names(self, per_vector: bool = False) -> list[str]:
        "Name of each row (or, with ``per_vector``, of each vector) for error messages"
        name = self.name or '<NDVar>'
        dims = self.dims[:-1] if per_vector else self.dims
        if not dims:
            return [name]
        return ["-".join((name, *map(str, values))) for values in product(*dims)]


def package_values(
        blocks: Sequence[RowBlock],
        value: np.ndarray,
        name: str | None = None,
        info: dict | None = None,
        dims: tuple[Dimension, ...] = (),
        n_vector: int = 0,
) -> list[NDVar | float]:
    """Package per-row values as one NDVar per block

    Parameters
    ----------
    blocks
        The blocks describing the rows of ``value`` (e.g., :attr:`SegmentedArray.blocks`).
    value
        One value per row, ``(n_rows, ...)``, or, for vector data, one
        value per vector, ``(n_rows / n_vector, ...)``, for values pooled
        over the vector dimension.
    name
        NDVar name (default: the name of the block's NDVar).
    info
        NDVar info (default: the info of the block's NDVar).
    dims
        Dimensions of any axes of ``value`` after the row axis.
    n_vector
        Number of components in each vector (0 for scalar data).

    Returns
    -------
    values
        For each block, an :class:`NDVar` with the block's dimensions
        (without the vector dimension for pooled values) followed by
        ``dims``, or a scalar for a block without dimensions.
    """
    per_vector = n_vector and len(value) != blocks[-1].index.stop
    out = []
    for block in blocks:
        if per_vector:
            block_dims = block.dims[:-1]
            block_value = value[block.index.start // n_vector:block.index.stop // n_vector]
        else:
            block_dims = block.dims
            block_value = value[block.index]
        out_dims = (*block_dims, *dims)
        if not out_dims:
            out.append(block_value[0])
        else:
            shape = tuple(len(dim) for dim in out_dims)
            out.append(NDVar(block_value.reshape(shape), out_dims, block.name if name is None else name, block.info if info is None else info))
    return out


class SegmentedArray:
    """Time series stored as one array per segment, with lazy concatenation and copy-on-write

    Parameters
    ----------
    segments
        One ``(n_rows, n_times_i)`` array per segment (trial).
    owned
        Whether the arrays may be modified in place (otherwise they are copied
        before the first modification).
    flat
        The segments concatenated along time, if already available (the
        segments are then views into it).
    blocks
        Origin of the rows, one block per input NDVar (see :class:`RowBlock`;
        see :meth:`from_ndvars` for building an array from NDVars).
    vector_dim
        For vector data, the vector dimension (the last dimension of each
        block); the scale is then one entry per vector rather than per row.
    name
        Description of the data for error messages.

    Attributes
    ----------
    segments : list of array
        Data for each segment, ``(n_rows, n_times_i)``; views into :attr:`flat`
        after :meth:`flatten`.
    n_vector : int
        Number of components in each vector (0 for scalar data).
    mean, scale : array
        Normalization that was applied to the data (``None`` before :meth:`normalize`).
    scale_norm : str
        The norm used to estimate ``scale`` in :meth:`normalize` (``None`` if
        ``scale`` was not estimated).
    """
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    scale_norm: str | None = None

    def __init__(
            self,
            segments: list[np.ndarray],
            owned: bool,
            flat: np.ndarray | None = None,
            blocks: Sequence[RowBlock] = (),
            vector_dim: Dimension | None = None,
            name: str = 'data',
    ):
        self.segments = segments
        self._owned = owned
        self._flat = flat
        self.blocks = list(blocks)
        self.vector_dim = vector_dim
        self.n_vector = len(vector_dim) if vector_dim else 0
        self.name = name

    @classmethod
    def from_ndvars(
            cls,
            ndvars: Sequence[NDVar | Datalist],
            case_to_segments: bool,
            vector: bool,
            name: str,
            in_place: bool,
    ) -> SegmentedArray:
        """Build an array from NDVars, with the rows of successive NDVars stacked

        Parameters
        ----------
        ndvars
            NDVars (or lists of NDVars for ragged trials) with the same time
            dimension and cases.
        case_to_segments
            Split the case dimension into segments.
        vector
            Treat a vector dimension as such (see :attr:`vector_dim`), rather
            than as a collection of independent rows.
        name
            Description of the data for error messages.
        in_place
            Allow modifying the data of the input NDVars in place.
        """
        blocks = []
        arrays = []  # (segments, flat, owned) for each NDVar
        vector_dim = None
        for v in ndvars:
            v0 = v if isinstance(v, NDVar) else v[0]
            if vector:
                v_vector_dims = [dim for dim in v0.dims if dim._adjacency_type == 'vector']
                if len(v_vector_dims) > 1:
                    raise NotImplementedError(f"{name}: more than one vector dimension ({', '.join(dim.name for dim in v_vector_dims)})")
                v_vector_dim = v_vector_dims[0] if v_vector_dims else None
                if blocks and v_vector_dim != vector_dim:
                    raise ValueError(f"{name}: not all NDVars have the same vector dimension")
                vector_dim = v_vector_dim
            last = (vector_dim.name, 'time') if vector_dim else ('time',)
            dimnames = v0.get_dimnames(first='case' if case_to_segments else None, last=last)
            dims = v0.get_dims(dimnames[1:-1] if case_to_segments else dimnames[:-1])
            n_rows = reduce(mul, map(len, dims), 1)
            start = blocks[-1].index.stop if blocks else 0
            blocks.append(RowBlock(v.name, dims, slice(start, start + n_rows), _info.copy(v0.info)))
            arrays.append(_ndvar_segments(v, dimnames, n_rows, case_to_segments))
        if len(arrays) == 1:
            segments, flat, owned = arrays[0]
            return cls(segments, owned or in_place, flat, blocks, vector_dim, name)
        # stack the rows of all NDVars into one concatenated array
        n_times = [seg.shape[-1] for seg in arrays[0][0]]
        stops = np.cumsum(n_times)
        flat = np.empty((blocks[-1].index.stop, stops[-1]))
        for (segments, _, _), block in zip(arrays, blocks):
            for seg, start, stop in zip(segments, stops - n_times, stops):
                flat[block.index, start:stop] = seg
        return cls(_segment_views(flat, n_times), True, flat, blocks, vector_dim, name)

    def __getstate__(self) -> dict:
        # the segments hold all the data; the concatenated array is rebuilt by flatten()
        return {**self.__dict__, '_flat': None, '_owned': True}

    @property
    def n_rows(self) -> int:
        "Number of time series (rows in each segment)"
        return len(self.segments[0])

    @property
    def names(self) -> list[str]:
        "Name of each row for error messages (empty without :attr:`blocks`)"
        return [name for block in self.blocks for name in block.names()]

    @property
    def _n_times(self) -> list[int]:
        "Number of time points in each segment"
        return [seg.shape[-1] for seg in self.segments]

    @property
    def flat(self) -> np.ndarray:
        "All segments concatenated along time, ``(n_rows, n_times_flat)``; available after :meth:`flatten`"
        if self._flat is None:
            raise RuntimeError("Concatenated data is only available after flatten()")
        return self._flat

    def flatten(self):
        """Build the concatenated array (:attr:`flat`)

        Afterwards, :attr:`flat` is C-contiguous (as required by the Cython
        routines) and :attr:`segments` are views into it, so that any reference
        to the previous segment arrays is stale. The data is not modified in
        place through :attr:`flat`; use :meth:`normalize`, which copies the
        data when it is not owned.
        """
        if self._flat is None:
            if len(self.segments) > 1:
                self._flat = self._concatenated()
                self.segments = _segment_views(self._flat, self._n_times)
                self._owned = True
                return
            self._flat = self.segments[0]
        if not self._flat.flags.c_contiguous:
            self._flat = np.ascontiguousarray(self._flat)
            self.segments = _segment_views(self._flat, self._n_times)
            self._owned = True

    def _concatenated(self) -> np.ndarray:
        "Segments concatenated along time, as a new C-contiguous array (np.concatenate() would keep the memory layout of transposed segments)"
        out = np.empty((self.n_rows, sum(self._n_times)), self.segments[0].dtype)
        return np.concatenate(self.segments, axis=-1, out=out)

    def _copy_data(self):
        "Make sure the data is a copy before modifying"
        if self._owned:
            return
        # multiple segments are copied into a concatenated array (so that a later flatten() does not need to copy again)
        if self._flat is None and len(self.segments) == 1:
            self.segments = [self.segments[0].copy()]
        else:
            if self._flat is None:
                self._flat = self._concatenated()
            else:
                self._flat = self._flat.copy()
            self.segments = _segment_views(self._flat, self._n_times)
        self._owned = True

    def _row_scale(self, scale: np.ndarray) -> np.ndarray:
        "Expand a per-vector scale to one entry per row"
        return np.repeat(scale, self.n_vector) if self.n_vector else scale

    def normalize(
            self,
            mean: bool | np.ndarray | float,
            scale: Literal['l1', 'l2'] | np.ndarray | float,
    ):
        """Center and scale each row in place

        Parameters
        ----------
        mean
            Value to subtract from each row (``(n_rows,)`` array or scalar);
            ``True`` to use the mean of each row, ``False`` to skip centering.
        scale
            Value by which to divide each row (``(n_rows,)`` array or scalar;
            one entry per vector for vector data); or estimate it from the
            centered data as the mean absolute value (``'l1'``) or the standard
            deviation (``'l2'``).

        Notes
        -----
        The data is checked for flat and NaN rows before it is modified.
        """
        self._apply_normalization(*self._normalization(mean, scale))

    def _normalization(
            self,
            mean: bool | np.ndarray | float,
            scale: Literal['l1', 'l2'] | np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        "``(mean, scale, scale_norm)`` for :meth:`normalize`, checking the data without modifying it"
        n = sum(self._n_times)
        center = mean is not False
        if mean is True:
            mean = sum(seg.sum(-1) for seg in self.segments) / n
        elif center:
            mean = np.broadcast_to(mean, (self.n_rows,)).astype(np.float64)
        else:
            mean = np.zeros(self.n_rows)
        if isinstance(scale, str):
            if scale not in ('l1', 'l2'):
                raise ValueError(f"{scale=}: need 'l1' or 'l2'")
            scale_norm = scale
            # estimate the scale from the centered data without modifying it, so that the data is intact if the check below fails
            total = 0
            for seg in self.segments:
                centered = seg - mean[:, newaxis] if center else seg
                if self.n_vector:
                    centered = norm(centered.reshape((-1, self.n_vector, seg.shape[-1])), axis=1)
                total = total + (np.abs(centered).sum(-1) if scale == 'l1' else (centered ** 2).sum(-1))
            scale = total / n if scale == 'l1' else (total / n) ** 0.5
            # dividing by a scale of 0 would turn flat rows into NaN
            self._check_values(scale, bool(self.n_vector))
        else:
            scale_norm = None  # the data is no longer normalized to the norm
            n_scale = self.n_rows // self.n_vector if self.n_vector else self.n_rows
            scale = np.broadcast_to(scale, (n_scale,)).astype(np.float64)
        return mean, scale, scale_norm

    def _apply_normalization(
            self,
            mean: np.ndarray,
            scale: np.ndarray,
            scale_norm: str | None,
    ):
        "Apply a normalization determined with :meth:`_normalization`"
        self._copy_data()
        center = mean.any()
        row_scale = self._row_scale(scale)
        for seg in self.segments:
            if center:
                seg -= mean[:, newaxis]
            seg /= row_scale[:, newaxis]
        self.scale_norm = scale_norm

        # Record normalization
        if self.mean is None:
            self.mean = mean
            self.scale = scale
        else:
            self.mean = self.mean + mean * self._row_scale(self.scale)
            self.scale = self.scale * scale

    @property
    def pads(self) -> np.ndarray:
        "Value of each row outside the data (the normalized value of 0)"
        if self.mean is None:
            return np.zeros(self.n_rows)
        return -self.mean / self._row_scale(self.scale)

    def check(self):
        "Raise an error for rows that are flat or contain NaN"
        if self.scale_norm is None:  # range of each row (NaN for rows containing NaN)
            values = np.max([seg.max(-1) for seg in self.segments], 0) - np.min([seg.min(-1) for seg in self.segments], 0)
            self._check_values(values, False)
        else:  # the scale was estimated from the data (one entry per vector for vector data)
            self._check_values(self.scale, bool(self.n_vector))

    def _check_values(self, values: np.ndarray, per_vector: bool):
        "Raise an error for rows (or vectors, with ``per_vector``) whose ``values`` are 0 (flat) or NaN"
        names = [name for block in self.blocks for name in block.names(per_vector)]
        is_flat = values == 0
        if np.any(is_flat):
            if names:
                raise ValueError(f"{self.name}: flat data in {', '.join(names[i] for i in np.flatnonzero(is_flat))}")
            raise ValueError(f"{self.name}: contains {is_flat.sum()} flat time series")
        has_nan = np.isnan(values)
        if np.any(has_nan):
            if names:
                raise ValueError(f"{self.name}: NaN in {', '.join(names[i] for i in np.flatnonzero(has_nan))}")
            raise ValueError(f"{self.name}: contains NaN")

    def package_values(
            self,
            value: np.ndarray,
            name: str | None = None,
            info: dict | None = None,
            dims: tuple[Dimension, ...] = (),
    ) -> list[NDVar | float]:
        "Package per-row values (e.g., :attr:`mean`) as one NDVar per block (see :func:`package_values`)"
        return package_values(self.blocks, value, name, info, dims, self.n_vector)


def _time_structure(v: NDVar | Datalist) -> tuple[bool, int, UTS | list[UTS]]:
    "``(is_ragged, n_cases, time_dim)`` of an NDVar or ragged list of NDVars (``n_cases=0`` without case dimension)"
    if isinstance(v, NDVar):
        n_cases = len(v) if v.has_case else 0
        return False, n_cases, v.get_dim('time')
    return True, len(v), [vi.get_dim('time') for vi in v]


def _ndvar_segments(
        v: NDVar | Datalist,
        dimnames: Sequence[str],
        n_rows: int,
        case_to_segments: bool,
) -> tuple[list[np.ndarray], np.ndarray | None, bool]:
    """Extract per-segment ``(n_rows, n_times_i)`` arrays from an NDVar (or ragged list of NDVars)

    Parameters
    ----------
    v
        NDVar, or list of NDVars for ragged trials.
    dimnames
        Order of the dimensions, with ``'time'`` last (and ``'case'`` first
        when ``case_to_segments``).
    n_rows
        Product of the lengths of the dimensions other than case and time.
    case_to_segments
        Split the case dimension of ``v`` into segments.

    Returns
    -------
    segments, flat, owned
        See :class:`SegmentedArray`. ``flat`` is only available when splitting
        the case dimension of a single row (a reshape of the data; otherwise
        concatenating would require a transposed copy). The data is ``owned``
        when extracting it already required a copy.
    """
    if not isinstance(v, NDVar):
        segments = [vi.get_data(dimnames).reshape((n_rows, -1)) for vi in v]
        return segments, None, not any(np.may_share_memory(seg, vi.x) for seg, vi in zip(segments, v))
    elif not case_to_segments:
        data = v.get_data(dimnames).reshape((n_rows, -1))
        return [data], None, not np.may_share_memory(data, v.x)
    cases = v.get_data(dimnames).reshape((len(v), n_rows, -1))
    if n_rows != 1:
        return list(cases), None, not np.may_share_memory(cases, v.x)
    flat = cases.reshape((1, -1))
    return _segment_views(flat, [cases.shape[-1]] * len(v)), flat, not np.may_share_memory(flat, v.x)


class DeconvolutionData:
    """Restructure input NDVars into arrays for deconvolution

    The data is stored as one array per segment (trial), with segments as views
    into the input :class:`NDVar` data when possible (see
    :class:`SegmentedArray`). The segment structure (cases or ragged trials)
    is determined by ``x``, and ``y`` has to match it.

    Parameters
    ----------
    x
        Predictors.
    y
        Dependent variable (optional, e.g. for predicting from a fitted model).
    data
        Dataset in which to evaluate ``y`` and ``x`` if they are strings.
    in_place
        Modify the data of the input NDVars in place when normalizing (saves
        memory).

    Attributes
    ----------
    x : SegmentedArray
        Predictors, ``(n_predictors, n_times_i)`` per segment, with one
        :class:`RowBlock` per predictor variable.
    y : SegmentedArray
        Dependent variable, ``(n_signals, n_times_i)`` per segment (``None``
        without ``y``), with a single :class:`RowBlock`; ``y.scale`` has one
        entry per vector.
    segments : array
        ``(n_segments, 2)`` array of segment ``[start, stop]`` indices on the
        concatenated time axis. The segments delimit chunks of continuous data,
        such as trials.
    segment_n_times : list of int
        Number of time samples in each segment.
    time : UTS
        Time dimension (of the first segment for ragged data).
    time_dim : UTS | list of UTS
        Time dimension of each segment (a list for ragged data).
    n_times_flat : int
        Total number of time samples across all segments.
    x_name : str | list of str
        Name of the predictor (a list for multiple predictors).
    basis : float
        Length of the basis window applied to ``x`` (0 for none).
    splits : Splits
        Cross-validation scheme (``None`` before :meth:`initialize_cross_validation`).
    """
    y: SegmentedArray | None = None  # None without y
    # cross-validation
    splits: Splits | None = None

    def __init__(
            self,
            x: NDVarArg | Sequence[NDVarArg],
            y: NDVarArg | None = None,
            data: Dataset = None,
            in_place: bool = False,
    ):
        if isinstance(x, (NDVar, Datalist, str)):
            multiple_x = False
            xs = [asndvar(x, data=data, ragged=True)]
        else:
            multiple_x = True
            xs = [asndvar(x_, data=data, ragged=True) for x_ in x]
            if len(xs) == 0:
                raise ValueError(f"{x=} of length 0")
        # segment structure
        structure = is_ragged, n_cases, time_dim = _time_structure(xs[0])
        for xi in xs[1:]:
            if _time_structure(xi) != structure:
                raise ValueError(f"x={xs}: not all predictors have the same time dimension and cases")
        has_case = is_ragged or n_cases > 0
        case_to_segments = has_case and not is_ragged
        n_segments = n_cases if has_case else 1
        if is_ragged:
            n_times = [len(uts) for uts in time_dim]
            segment_n_times = n_times
        else:
            n_times = len(time_dim)
            segment_n_times = [n_times] * n_segments
        stops = np.cumsum(segment_n_times, dtype=np.int64)
        segments = np.hstack(((stops - segment_n_times)[:, newaxis], stops[:, newaxis]))

        self.x = SegmentedArray.from_ndvars(xs, case_to_segments, False, 'x', in_place)
        self.is_ragged = is_ragged
        self.has_case = has_case
        self.n_cases = n_cases
        self.case_to_segments = case_to_segments
        self.time_dim = time_dim
        self.time = time_dim[0] if is_ragged else time_dim
        self.n_times = n_times
        self.n_times_flat = int(stops[-1])
        self.multiple_x = multiple_x
        self.x_name = [block.name for block in self.x.blocks] if multiple_x else self.x.blocks[0].name
        self.segments = segments
        self.segment_n_times = segment_n_times
        # basis
        self.basis = 0
        self.basis_window = None

        # y
        if y is None:
            return
        if isinstance(y, (list, tuple)) and isinstance(y[0], str):
            raise TypeError(f"{y=}: need a single NDVar (or list with ragged trials) as dependent variable")
        y = asndvar(y, data=data, ragged=is_ragged)
        y_structure = _time_structure(y)
        if y_structure != structure:
            raise ValueError(f"y={dataobj_repr(y)}: does not have the same time dimension and cases as x: y={y_structure[1:]}, x={structure[1:]}")
        self.y = SegmentedArray.from_ndvars([y], case_to_segments, True, f"y={dataobj_repr(y)}", in_place)

    def flatten(self):
        "Build the concatenated arrays (see :meth:`SegmentedArray.flatten`)"
        self.x.flatten()
        if self.y is not None:
            self.y.flatten()

    def apply_basis(self, basis: float, basis_window: str):
        """Apply basis to x

        Parameters
        ----------
        basis
            Length of the basis window (in seconds); 0 to skip.
        basis_window
            Window shape (see :func:`scipy.signal.get_window`).

        Notes
        -----
        Normalize after applying basis (basis can smooth out variance).
        The basis is applied to each segment separately.
        """
        if self.basis != 0:
            raise NotImplementedError("Applying basis more than once")
        elif not basis:
            return
        self.x._copy_data()
        n = int(round(basis / self.time.tstep))
        w = scipy.signal.get_window(basis_window, n, False)
        if len(w) <= 1:
            raise ValueError(f"{basis=}: Window is {len(w)} samples long")
        w /= w.sum()
        for seg in self.x.segments:
            seg[:] = scipy.signal.convolve(seg, w[newaxis], 'same')
        self.basis = basis
        self.basis_window = basis_window

    def normalize(self, error: str):
        """Center and scale ``x`` and ``y`` in place (see :meth:`SegmentedArray.normalize`)

        Parameters
        ----------
        error
            Scale by the mean absolute value (``'l1'``) or the standard
            deviation (``'l2'``).

        Notes
        -----
        To normalize only the predictors, use ``data.x.normalize(True, error)``.
        """
        arrays = [self.x] if self.y is None else [self.x, self.y]
        # check all data before modifying any of it, so that a failure leaves the data intact
        normalizations = [array._normalization(True, error) for array in arrays]
        for array, normalization in zip(arrays, normalizations):
            array._apply_normalization(*normalization)

    def check_data(self):
        "Raise an error for data that is flat or contains NaN (see :meth:`SegmentedArray.check`)"
        self.x.check()
        if self.y is not None:
            self.y.check()

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
            ``y.segments[i_segment][:, start:stop]`` is the corresponding data.
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

    def initialize_cross_validation(
            self,
            partitions: int | None = None,
            model: CategorialArg | None = None,
            data: Dataset = None,
            validate: int = 1,
            test: int = 1,
    ):
        """Initialize the cross-validation scheme (:attr:`splits`)

        Parameters
        ----------
        partitions
            Number of partitions to divide the data into (by default, determined
            from the number of segments).
        model
            Sample partitions evenly from the cells of this model (one entry per
            segment).
        data
            Dataset in which to evaluate ``model`` if it is a string.
        validate
            Number of partitions in each validation set (0 or 1).
        test
            Number of partitions in each test set (0 or 1).

        Notes
        -----
        See :func:`split_data` for the algorithm.
        """
        self.splits = split_data(self.segments, partitions, model, data, validate, test)

    def data_scale_ndvars(self) -> tuple:
        """Normalization parameters packaged like the input data

        Returns
        -------
        y_mean, y_scale
            Normalization of ``y`` (:class:`NDVar`, or scalar for a single time
            series; ``None`` if ``y`` was not normalized). ``y_scale`` does not
            include the vector dimension.
        x_mean, x_scale
            Normalization of ``x`` (:class:`NDVar` or scalar; a tuple with one
            item per predictor for multiple predictors).

        All four are ``None`` if the data was not normalized.
        """
        if self.x.mean is None:
            return None, None, None, None
        if self.y is None or self.y.mean is None:
            y_mean = y_scale = None
        else:
            y_mean, = self.y.package_values(self.y.mean)
            y_scale, = self.y.package_values(self.y.scale)
        x_mean = self.x.package_values(self.x.mean)
        x_scale = self.x.package_values(self.x.scale)
        if self.multiple_x:
            return y_mean, y_scale, tuple(x_mean), tuple(x_scale)
        return y_mean, y_scale, x_mean[0], x_scale[0]

    def package_kernel(
            self,
            h: np.ndarray,
            tstart: float,
    ) -> NDVar | tuple[NDVar, ...]:
        """Package kernel as NDVar

        Parameters
        ----------
        h
            Kernel data, ``(n_y, n_x, n_times)``.
        tstart
            Time of the first kernel sample.

        Returns
        -------
        kernel
            One :class:`NDVar` per predictor (a tuple for multiple predictors).
        """
        h_time = UTS(tstart, self.time.tstep, h.shape[-1], self.time.unit)
        y_info = self.y.blocks[0].info
        if self.x.mean is not None:
            info = _info.for_normalized_data(y_info, 'Response')
        else:
            info = y_info
        hs = [self.y.package_values(h[:, block.index], block.name, info, (*block.dims, h_time))[0] for block in self.x.blocks]
        if self.multiple_x:
            return tuple(hs)
        return hs[0]

    def package_y_like(
            self,
            data: np.ndarray,
            name: str,
    ) -> NDVar:
        """Package data with the shape of ``y`` (e.g., predicted ``y``) as NDVar

        Parameters
        ----------
        data
            Data with the same number of elements as the input ``y``.
        name
            NDVar name.
        """
        if self.is_ragged:
            raise NotImplementedError("package_y_like() for ragged data")
        dims = self.y.blocks[0].dims
        shape = tuple(len(dim) for dim in dims)
        if self.case_to_segments:
            data = np.rollaxis(data.reshape((*shape, self.n_cases, self.n_times)), -2)
            return NDVar(data, (Case, *dims, self.time), name)
        return NDVar(data.reshape((*shape, self.n_times)), (*dims, self.time), name)
