from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import reduce
from itertools import product
from operator import mul
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy import newaxis
import scipy.signal
from scipy.linalg import norm

from .. import _info
from .._data_obj import CategorialArg, NDVarArg, Datalist, Dataset, Factor, Interaction, NDVar, Case, UTS, dataobj_repr, ascategorial, asndvar
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

    A split divides the data available to it into training segments and
    (optionally) one held-out partition. The held-out partition is either a
    *test* set, which is never consulted while estimating the kernel, or a
    *validation* set, which is consulted repeatedly during fitting (for early
    stopping). A test split can in turn contain validation splits, which divide
    its training data again; each split without inner splits (leaf)
    corresponds to one boosting run.

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
            # flat format (before nested splits): one split per boosting run
            train, validate, test, i_test = (state[key] for key in ['train', 'validate', 'test', 'i_test'])
            i_validate = state.get('i_validate')  # not stored before BoostingResult version 16 (restored by Splits.__setstate__)
            if validate is None:
                state = {'train': train, 'reserved': test, 'i_reserved': None if test is None else i_test, 'is_test': test is not None}
            elif test is None:
                state = {'train': train, 'reserved': validate, 'i_reserved': i_validate}
            else:  # nested: test split with a single validation split
                validation_split = Split(train, validate, i_validate)
                state = {'train': validation_split.train_and_reserved, 'reserved': test, 'i_reserved': i_test, 'is_test': True, 'splits': [validation_split]}
        PickleableDataClass.__setstate__(self, state)


def merge_segments(
        segments: np.ndarray,
        soft_splits: bool | np.ndarray = None,
) -> np.ndarray:
    """Merge adjacent segments at soft boundaries

    Parameters
    ----------
    segments
        ``(n, 2)`` array of ``[start, stop]`` indices, sorted by time.
    soft_splits
        Boundaries at which adjacent segments are merged: ``True`` to merge at
        all boundaries, an array of boundary indices to merge only at those, or
        ``None`` to return ``segments`` unchanged.
    """
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


def _restore_validation_indices(
        splits: list[Split],
        n_partitions: int,
        i_test: int | None = None,
):
    "Fill in validation partition indices missing from splits pickled before BoostingResult version 16 (created in partition order, skipping the test partition)"
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
        splits = state['splits']
        if splits:
            _restore_validation_indices(splits, state['n_partitions'])
        test_splits = [split for split in splits if split.is_test] if splits else []
        if len({split.i_reserved for split in test_splits}) < len(test_splits):
            # flat format (before nested splits): one top-level split per boosting run; merge test splits with the same test partition
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
    if validate < 0:
        raise ValueError(f"{validate=}")
    elif validate > 1:
        raise NotImplementedError(f"{validate=}: currently has to be 0 or 1")
    if test < 0:
        raise ValueError(f"{test=}")
    elif test > 1:
        raise NotImplementedError(f"{test=}: currently has to be 0 or 1")
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
    n_vector
        For vector data, the number of components in each vector; the scale is
        then one entry per vector rather than per row.
    name
        Description of the data for error messages.
    names
        Name of each row for error messages.

    Attributes
    ----------
    segments : list of array
        Data for each segment, ``(n_rows, n_times_i)``; views into :attr:`flat`
        after :meth:`flatten`.
    mean, scale : array
        Normalization that was applied to the data (``None`` before :meth:`normalize`).
    scale_norm : str
        The ``error`` argument of :meth:`normalize` (``None`` before :meth:`normalize`).
    """
    mean: np.ndarray = None
    scale: np.ndarray = None
    scale_norm: str = None

    def __init__(
            self,
            segments: list[np.ndarray],
            owned: bool,
            flat: np.ndarray = None,
            n_vector: int = 0,
            name: str = 'data',
            names: Sequence[str] = None,
    ):
        self.segments = segments
        self._owned = owned
        self._flat = flat
        self.n_vector = n_vector
        self.name = name
        self.names = names

    def __getstate__(self) -> dict:
        # the segments hold all the data; the concatenated array is rebuilt by flatten()
        return {**self.__dict__, '_flat': None, '_owned': True}

    @property
    def n_rows(self) -> int:
        "Number of time series (rows in each segment)"
        return len(self.segments[0])

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
        place through :attr:`flat`; use :meth:`normalize` and
        :meth:`apply_normalization`, which copy the data when it is not owned.
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

    def _record_normalization(
            self,
            mean: np.ndarray,
            scale: np.ndarray,
    ):
        "Record a normalization step applied to the current data, composing it with previous ones"
        if self.mean is None:
            self.mean = mean
            self.scale = scale
        else:
            self.mean = self.mean + mean * self._row_scale(self.scale)
            self.scale = self.scale * scale

    def normalize(self, error: str):
        """Center and scale each row in place

        Parameters
        ----------
        error
            Scale by the mean absolute value (``'l1'``) or the standard
            deviation (``'l2'``).
        """
        self._copy_data()
        n = sum(self._n_times)
        mean = sum(seg.sum(-1) for seg in self.segments) / n
        for seg in self.segments:
            seg -= mean[:, newaxis]
        if self.n_vector:
            magnitudes = [norm(seg.reshape((-1, self.n_vector, seg.shape[-1])), axis=1) for seg in self.segments]
        else:
            magnitudes = self.segments
        if error == 'l1':
            scale = sum(np.abs(x).sum(-1) for x in magnitudes) / n
        elif error == 'l2':
            scale = (sum((x ** 2).sum(-1) for x in magnitudes) / n) ** 0.5
        else:
            raise ValueError(f"{error=}")
        row_scale = self._row_scale(scale)
        for seg in self.segments:
            seg /= row_scale[:, newaxis]
        self._record_normalization(mean, scale)
        self.scale_norm = error

    def apply_normalization(
            self,
            mean: np.ndarray | float,
            scale: np.ndarray | float,
    ):
        """Normalize in place with given values (e.g., from a fitted model)

        Parameters
        ----------
        mean
            Value to subtract from each row (``(n_rows,)`` array or scalar).
        scale
            Value by which to divide each row (``(n_rows,)`` array or scalar;
            one entry per vector for vector data).
        """
        n_scale = self.n_rows // self.n_vector if self.n_vector else self.n_rows
        mean = np.broadcast_to(mean, (self.n_rows,)).astype(np.float64)
        scale = np.broadcast_to(scale, (n_scale,)).astype(np.float64)
        self._copy_data()
        row_scale = self._row_scale(scale)
        for seg in self.segments:
            seg -= mean[:, newaxis]
            seg /= row_scale[:, newaxis]
        self._record_normalization(mean, scale)

    @property
    def pads(self) -> np.ndarray:
        "Value of each row outside the data (the normalized value of 0)"
        if self.mean is None:
            return np.zeros(self.n_rows)
        return -self.mean / self._row_scale(self.scale)

    def check(self):
        "Raise an error for rows that are flat or contain NaN"
        if self.scale is None:  # range of each row (NaN for rows containing NaN)
            check = np.max([seg.max(-1) for seg in self.segments], 0) - np.min([seg.min(-1) for seg in self.segments], 0)
        else:
            check = self.scale
        is_flat = check == 0
        if np.any(is_flat):
            if self.names:
                raise ValueError(f"{self.name}: flat data in {', '.join(self.names[i] for i in np.flatnonzero(is_flat))}")
            raise ValueError(f"{self.name}: contains {is_flat.sum()} flat time series")
        has_nan = np.isnan(check)
        if np.any(has_nan):
            if self.names:
                raise ValueError(f"{self.name}: NaN in {', '.join(self.names[i] for i in np.flatnonzero(has_nan))}")
            raise ValueError(f"{self.name}: contains NaN")


def _time_structure(v: NDVar | Datalist) -> tuple[bool, int, UTS | list[UTS]]:
    "``(is_ragged, n_cases, time_dim)`` of an NDVar or ragged list of NDVars (``n_cases=0`` without case dimension)"
    if isinstance(v, NDVar):
        return False, len(v) if v.has_case else 0, v.get_dim('time')
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
        Predictors, ``(n_predictors, n_times_i)`` per segment.
    y : SegmentedArray
        Dependent variable, ``(n_signals, n_times_i)`` per segment (``None``
        without ``y``); ``y.scale`` has one entry per vector.
    segments : array
        ``(n_segments, 2)`` array of segment ``[start, stop]`` indices on the
        concatenated time axis. The segments delimit chunks of continuous data,
        such as trials.
    time : UTS
        Time dimension (of the first segment for ragged data).
    time_dim : UTS | list of UTS
        Time dimension of each segment (a list for ragged data).
    n_times_flat : int
        Total number of time samples across all segments.
    x_name : str | list of str
        Name of the predictor (a list for multiple predictors).
    x_meta : list of tuple
        ``(name, dims, index)`` for each predictor variable: its name, its
        dimensions other than case and time, and the index of its rows in ``x``.
    basis : float
        Length of the basis window applied to ``x`` (0 for none).
    y_name : str
        Name of the dependent variable.
    ydims : tuple of Dimension
        Dimensions of ``y`` other than case and time (vector dimension last).
    yshape : tuple of int
        Shape corresponding to ``ydims``.
    vector_dim : Dimension
        Vector dimension of ``y`` (``None`` for scalar data).
    y_info : dict
        Info dictionary of ``y``.
    splits : Splits
        Cross-validation scheme (``None`` before :meth:`initialize_cross_validation`).
    """
    # y (None without y)
    y: SegmentedArray = None
    y_name: str = None
    y_info: dict = None
    ydims: tuple = None
    yshape: tuple = None
    vector_dim = None
    full_y_dims: tuple = None
    # cross-validation
    splits: Splits = None

    def __init__(
            self,
            x: NDVarArg | Sequence[NDVarArg],
            y: NDVarArg = None,
            data: Dataset = None,
            in_place: bool = False,
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

        # x: rows for each predictor
        x_meta = []  # [(x.name, xdims, index), ...]; index is int or slice
        x_names = []
        x_arrays = []  # (segments, flat, owned) for each predictor
        for xi in xs:
            x0i = xi[0] if is_ragged else xi
            dimnames = x0i.get_dimnames(first='case' if case_to_segments else None, last='time')
            dims = x0i.get_dims(dimnames[1:-1] if case_to_segments else dimnames[:-1])
            n_rows = reduce(mul, [len(dim) for dim in dims], 1)
            index = len(x_names) if n_rows == 1 else slice(len(x_names), len(x_names) + n_rows)
            x_repr = dataobj_repr(x0i)
            if dims:
                x_names.extend("-".join((x_repr, *map(str, v))) for v in product(*dims))
            else:
                x_names.append(x_repr)
            x_meta.append((x0i.name, dims, index))
            x_arrays.append(_ndvar_segments(xi, dimnames, n_rows, case_to_segments))
        if multiple_x:
            x_flat = np.empty((len(x_names), int(stops[-1])))
            for (x_segments, _, _), (_, _, index) in zip(x_arrays, x_meta):
                for seg, (start, stop) in zip(x_segments, segments):
                    x_flat[index, start:stop] = seg
            self.x = SegmentedArray(_segment_views(x_flat, segment_n_times), True, x_flat, name='x', names=x_names)
        else:
            x_segments, x_flat, x_owned = x_arrays[0]
            self.x = SegmentedArray(x_segments, x_owned or in_place, x_flat, name='x', names=x_names)

        self.is_ragged = is_ragged
        self.has_case = has_case
        self.n_cases = n_cases
        self.case_to_segments = case_to_segments
        self.time_dim = time_dim
        self.time = time_dim[0] if is_ragged else time_dim
        self.n_times = n_times
        self.n_times_flat = int(stops[-1])
        self.multiple_x = multiple_x
        self.x_name = x_name
        self.x_meta = x_meta
        self.segments = segments
        self.shortest_segment_n_times = np.min(np.diff(segments, axis=1))
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
        y0 = y[0] if is_ragged else y
        # vector dimension
        vector_dims = [dim.name for dim in y0.dims if dim._adjacency_type == 'vector']
        if not vector_dims:
            vector_dim = None
        elif len(vector_dims) == 1:
            vector_dim = y0.get_dim(vector_dims.pop())
        else:
            raise NotImplementedError(f"{y=}: more than one vector dimension ({', '.join(vector_dims)})")
        # y rows: ydims x time
        last = (vector_dim.name, 'time') if vector_dim else ('time',)
        y_dimnames = y0.get_dimnames(first='case' if case_to_segments else None, last=last)
        ydims = y0.get_dims(y_dimnames[1:-1] if case_to_segments else y_dimnames[:-1])
        n_rows = reduce(mul, map(len, ydims), 1)
        y_segments, y_flat, y_owned = _ndvar_segments(y, y_dimnames, n_rows, case_to_segments)
        n_vector = len(vector_dim) if vector_dim else 0
        self.y = SegmentedArray(y_segments, y_owned or in_place, y_flat, n_vector, f"y={dataobj_repr(y)}")
        self.y_name = y.name
        self.y_info = _info.copy(y0.info)
        self.ydims = ydims
        self.yshape = tuple(map(len, ydims))
        self.vector_dim = vector_dim
        if is_ragged:
            self.full_y_dims = None
        elif case_to_segments:
            self.full_y_dims = y.get_dims((*y_dimnames[1:-1], 'case', 'time'))
        else:
            self.full_y_dims = y.get_dims(y_dimnames)

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
        To normalize only the predictors, use ``data.x.normalize(error)``.
        """
        self.x.normalize(error)
        if self.y is not None:
            self.y.normalize(error)

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
            partitions: int = None,
            model: CategorialArg = None,
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
        if self.x.mean is not None:
            # y
            if self.y is None or self.y.mean is None:
                y_mean = y_scale = None
            else:
                if self.yshape:
                    y_mean = NDVar(self.y.mean.reshape(self.yshape), self.ydims, self.y_name, self.y_info)
                else:
                    y_mean = self.y.mean[0]
                # scale does not include vector dim
                if self.vector_dim:
                    dims = self.ydims[:-1]
                    shape = self.yshape[:-1]
                else:
                    dims = self.ydims
                    shape = self.yshape
                if shape:
                    y_scale = NDVar(self.y.scale.reshape(shape), dims, self.y_name, self.y_info)
                else:
                    y_scale = self.y.scale[0]
            # x
            x_mean = []
            x_scale = []
            for name, xdims, index in self.x_meta:
                if xdims:
                    shape = [len(dim) for dim in xdims]
                    x_mean.append(NDVar(self.x.mean[index].reshape(shape), xdims, name))
                    x_scale.append(NDVar(self.x.scale[index].reshape(shape), xdims, name))
                else:
                    x_mean.append(self.x.mean[index])
                    x_scale.append(self.x.scale[index])
            if self.multiple_x:
                x_mean = tuple(x_mean)
                x_scale = tuple(x_scale)
            else:
                x_mean = x_mean[0]
                x_scale = x_scale[0]
        else:
            y_mean = y_scale = x_mean = x_scale = None
        return y_mean, y_scale, x_mean, x_scale

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
        hs = []
        if self.x.mean is not None:
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
            value: np.ndarray,
            name: str,
            info: dict = None,
            meas: str = None,
    ) -> NDVar | float:
        """Package a per-signal value (e.g., a fit metric) like ``y``

        Parameters
        ----------
        value
            One value per signal, ``(n_y,)``, or one value per vector,
            ``(n_y / n_vector,)``, for values pooled over the vector dimension.
        name
            NDVar name.
        info
            NDVar info (default: the info of ``y``).
        meas
            Measurement type, used to generate the NDVar info.

        Returns
        -------
        value
            :class:`NDVar` with the dimensions of ``y`` (without case and time),
            or a scalar for a single time series.
        """
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
