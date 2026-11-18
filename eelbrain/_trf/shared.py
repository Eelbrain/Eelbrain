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
        the default).
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
    split
        How the partitions were formed, ``'cases'``, ``'time'`` or
        ``'within-cases'`` (see :func:`split_data`; ``None`` for splits
        created before this attribute was added).
    split_arg
        The ``split`` argument used to create the splits (``None`` for the
        default).
    validate_arg
        The ``validate`` argument used to create the splits (``None`` for the
        default).
    min_section_arg
        The ``min_section`` argument used to create the splits (``None`` for
        the default).
    tstep
        Time step between samples in seconds (``None`` for data without time
        axis).
    scheme
        Which partitions are used for training when a partition is held out,
        ``'k-fold'`` or ``'walk-forward'`` (see :func:`split_data`).
    """
    splits: list[Split]
    partitions_arg: int | None
    n_partitions: int
    n_validate: int
    n_test: int
    model: CategorialArg = None
    segments: np.ndarray = None
    split_segments: np.ndarray = None
    split: Literal['cases', 'time', 'within-cases'] = None
    split_arg: Literal['cases', 'time', 'within-cases'] = None
    validate_arg: float = None
    min_section_arg: float = None
    tstep: float = None
    scheme: Literal['k-fold', 'walk-forward'] = 'k-fold'

    def __setstate__(self, state: dict):
        # Backwards compatibility for BoostingResult version < 18
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
        if self.split:
            items.append(f'split={self.split!r}')
        if self.scheme != 'k-fold':
            items.append(f'scheme={self.scheme!r}')
        if self.validate_arg is None:
            if self.n_validate:
                items.append(f'n_validate={self.n_validate}')
        else:
            items.append(f'validate={self.validate_arg:g}')
        if self.n_test:
            items.append(f'n_test={self.n_test}')
        if self.model is not None:
            items.append(f'model={dataobj_repr(self.model)}')
        if self.min_section_arg is not None:
            items.append(f'min_section={self.min_section_arg:g}')
        items = ', '.join(items)
        return f"<Splits: {desc} split into {len(self.split_segments)} sections{items}>"

    @property
    def leaves(self) -> list[tuple[Split, Split]]:
        "``(top-level split, leaf split)`` for each boosting run, grouped by top-level split (see :attr:`Split.leaves`)"
        return [(top_split, leaf) for top_split in self.splits for leaf in top_split.leaves]

    def plot(self, **kwargs) -> DataSplit:
        """Plot data splits (see :class:`plot.DataSplit` for parameters)

        The x-axis shows time in seconds if :attr:`tstep` is known.
        """
        from ..plot import DataSplit

        return DataSplit(self, **kwargs)


def split_data(
        segments: np.ndarray,
        *,
        tstep: float = None,
        partitions: int | None = 5,
        split: Literal['cases', 'time', 'within-cases'] | None = 'time',
        scheme: Literal['k-fold', 'walk-forward'] = 'k-fold',
        validate: float = None,
        test: int = 1,
        model: CategorialArg = None,
        min_section: float = None,
        data: Dataset = None,
) -> Splits:
    """Split data into training, validation and test sets for cross-validation

    This function defines the cross-validation parameters that are shared by
    :func:`boosting`, :func:`plot.preview_partitions` and other estimators.
    The data consists of one or several *segments* of contiguous samples: for
    :class:`NDVar` data, each case (trial) is one segment, and continuous
    data is a single segment.

    Parameters
    ----------
    segments
        ``(n_segments, 2)`` array of ``[start, stop]`` indices of the data
        segments (on the concatenated time axis).
    tstep
        Time step between samples in seconds (by default, the data has no time
        axis, and parameters in seconds are not available).
    partitions
        Number of partitions (*k*) to divide the data into. Each partition is
        held out in turn (see ``validate`` and ``test``). The default is 5.
        With ``None``, use the number of segments for ``split='cases'`` (only
        for 3 to 10 segments), 4 for ``scheme='walk-forward'`` and 5 otherwise.
    split
        How to form partitions (see notes for details):

        - ``'time'`` (default): cut the concatenated time axis into
          ``partitions`` contiguous spans of equal length. Segments are only
          cut into pieces that are at least ``min_section`` long.
        - ``'cases'``: each partition consists of whole segments, so that
          segment boundaries are always partition boundaries. Requires at least
          as many segments as partitions.
        - ``'within-cases'``: cut each segment into ``partitions`` pieces of
          equal length, and form partition *i* from the *i*-th piece of every
          segment, so that each partition samples every segment.

        With ``None``, use ``'cases'`` for data with more than one segment and
        ``'time'`` for continuous data (``'within-cases'`` with
        ``scheme='walk-forward'``).
    scheme
        Which partitions are used for training when a partition is held out
        (see notes for details):

        - ``'k-fold'`` (default): all the other partitions.
        - ``'walk-forward'``: only the partitions preceding it, so that the
          training data always precedes the held-out data. The partitions
          after the first are short windows (see ``validate``), which are
          held out one at a time. Requires ``test=0``, and
          ``split='time'`` or ``split='within-cases'``.
    validate
        Size of each validation set (the data that is held out to evaluate a
        model while fitting it, e.g., for early stopping in :func:`boosting`).
        With ``scheme='k-fold'``, the number of partitions (0 or 1; default
        1). With ``scheme='walk-forward'``, the length of each validation set
        as proportion of the data that is cut (between 0 and 1; default 0.05).
    test
        Number of partitions in each test set (0 or 1). With ``test=1``, each
        partition is held out in turn as test set, i.e., data that is never
        consulted while fitting the model, and the validation sets are formed
        from the remaining partitions. With ``validate=1``, this results in
        ``k * (k - 1)`` model fits instead of ``k``.
    model
        With ``split='cases'``, sample partitions evenly from the cells of this
        model (one entry per segment).
    min_section
        With ``split='time'``, minimum duration (in seconds) of the pieces left
        when a segment is cut: a span boundary that would leave a shorter piece
        is moved to the closest segment boundary instead, so that segments
        shorter than ``2 * min_section`` are never cut. The default is 15 s.
        Requires ``tstep``; for data without time axis, segments are always
        cut at the span boundaries. Only available with ``split='time'`` and
        ``scheme='k-fold'``.
    data
        Dataset in which to evaluate ``model`` if it is a string.

    See Also
    --------
    plot.preview_partitions : Plot the splits resulting from different parameters

    Notes
    -----
    Splitting always follows the same four steps:

    1. Divide the data into *sections*. A section is either a whole segment or
       a piece of a segment. Boundaries between the original segments are
       *hard*; boundaries introduced by cutting a segment are *soft*.
    2. Assign each section to one of ``partitions`` partitions.
    3. Create the :class:`Split` tree (this describes ``scheme='k-fold'``;
       see below for walk-forward). With ``test=1``, create one test split
       for each partition *i*: partition *i* is held out as test set and all
       remaining partitions form the training set. With ``validate=1``, each
       split created so far (or, with ``test=0``, the whole data) is further
       divided into validation splits, one for each partition *j* in its
       training set: partition *j* is held out as validation set and the
       remaining training partitions form the training set. With ``validate=1``
       and ``test=1`` this yields *k* test splits with ``k - 1`` validation
       splits each, i.e., ``k * (k - 1)`` model fits (for *k* =
       ``partitions``); with one of them set to 0, it yields *k* top-level
       splits without inner splits; with both set to 0, a single split that
       uses all data for training.
    4. Within each set of a split, merge adjacent sections at soft boundaries,
       so that pieces of a segment that end up in the same set form one
       continuous segment again. Hard boundaries are never merged.

    How the sections are formed and assigned to partitions (steps 1 and 2)
    depends on ``split``. Except for ``split=None``, this never depends on the
    number of segments, so that the same arguments lead to the same kind of
    cross-validation for datasets that differ in the number of segments (e.g.,
    a participant with a missing trial).

    **Whole cases** (``split='cases'``): whole segments are used as sections, so that segment boundaries
    are always partition boundaries. The segments are assigned to partitions in
    an interleaved fashion, i.e., partition *p* receives segments
    ``[p::partitions]``. With ``model``, this is done separately within each
    cell, so that every partition samples each cell evenly (``model`` is only
    allowed with this scheme). With ``partitions=None``, the default is
    ``partitions = n_segments`` (number of segments in the smallest cell with
    ``model``), which amounts to leave-one-out and is only available for 3 to
    10 segments (per cell); with ``model`` it further requires all cells to
    have the same size. More partitions than segments (per cell) raise an
    error.

    **Time** (``split='time'``, the default): the
    concatenated time axis is divided into ``partitions`` spans of equal
    length, and the data is cut at these span boundaries (soft) as
    well as at all segment boundaries (hard). Each section is assigned to the
    partition whose span contains its start, so partitions contain
    (approximately) equal numbers of samples and correspond to different
    periods of time, while a span that straddles a segment boundary yields two
    sections in the same partition. For continuous data (a single segment),
    this simply cuts the data into ``partitions`` contiguous sections of equal
    length. A span boundary that falls within ``min_section`` of a segment
    boundary is moved to that segment boundary (the closer one, if both are
    within reach), so that cutting a segment never leaves a piece shorter than
    ``min_section``. Segments shorter than ``2 * min_section`` are thus never
    cut, and partitions are formed from whole segments wherever that is
    possible within this tolerance. This always works when the spans are at
    least ``2 * min_section`` long; with shorter spans, an error is raised if
    two span boundaries would be moved to the same segment boundary, or if a
    span boundary that can not be moved leaves a shorter piece (use fewer
    partitions or a smaller ``min_section``).

    **Within cases** (``split='within-cases'``): each segment is cut into
    ``partitions`` pieces of equal length, and partition *i*
    consists of the *i*-th piece of every segment. Every partition thus
    samples every segment in proportion to its length, at the cost of more
    segment boundaries. Note that partition *i* always covers the same
    relative position within the segments (e.g., the first partition contains
    the beginning of every segment).

    **Walk-forward** (``scheme='walk-forward'``): the training data always
    precedes the held-out data. For steps 1 and 2, the data is cut into
    ``partitions`` spans, of which the last ``partitions - 1`` are
    windows of equal length, and the first is the remaining data before them.
    With ``split='within-cases'``, every
    segment is cut this way, and partition *i* consists of the *i*-th piece of
    every segment. With ``split='time'``, the concatenated time axis is cut
    this way, so that all windows are at the end of the data (and cut
    additionally at segment boundaries); for continuous data, both are the
    same. Each window is ``ceil(validate * n)`` samples long, with ``n`` the
    number of samples in the segment (``'within-cases'``) or in the whole data
    (``'time'``), and ``validate`` the proportion (default 0.05). In step 3,
    each partition of windows is held out in turn as validation set, with all
    the partitions preceding it as training set; the partitions following it
    are not used in that split. This results in ``partitions - 1`` splits,
    because the first partition is never held out. The scheme requires a
    validation set and ``test=0``; ``min_section`` and ``split='cases'`` (in
    which partitions are not ordered in time) are not available. Note that
    the training sets of the splits differ by one window only. This is the
    scheme the NCRF package uses to select its regularization parameter (with
    ``split='within-cases'``): NCRF's ``n_splits`` is ``partitions - 1``, and
    NCRF's ``r``, which specifies the window relative to the rest of the
    segment, corresponds to ``validate = r / (1 + r)``.

    See :ref:`exa-data_split` for illustrations of these schemes, and use
    :meth:`Splits.plot` (or :func:`plot.preview_partitions`) to inspect a
    specific configuration.
    """
    partitions_arg, split_arg, validate_arg, min_section_arg = partitions, split, validate, min_section
    if partitions is not None and int(partitions) != partitions:
        raise TypeError(f"{partitions=}")
    if int(test) != test:
        raise TypeError(f"{test=}")
    elif test < 0:
        raise ValueError(f"{test=}")
    elif test > 1:
        raise NotImplementedError(f"{test=}: currently has to be 0 or 1")
    if scheme not in ('k-fold', 'walk-forward'):
        raise ValueError(f"{scheme=}: must be 'k-fold' or 'walk-forward'")
    if split is None:
        if scheme == 'walk-forward':
            split = 'within-cases'
        elif len(segments) > 1:
            split = 'cases'
        else:
            split = 'time'
    elif split not in ('cases', 'time', 'within-cases'):
        raise ValueError(f"{split=}: must be 'cases', 'time' or 'within-cases'")
    if scheme == 'walk-forward':
        if split == 'cases':
            raise NotImplementedError(f"{scheme=} with {split=}: partitions of whole cases are not ordered in time; use split='within-cases' or split='time'")
        elif test:
            raise NotImplementedError(f"{scheme=} with {test=}: the walk-forward scheme only implements validation splits (test=0)")
        if validate is None:
            validate = 0.05
        elif not 0 < validate < 1:
            raise ValueError(f"{validate=} with {scheme=}: needs to be the length of each validation window as proportion of the data (between 0 and 1)")
        n_validate = 1
    else:
        if validate is None:
            validate = 1
        elif validate not in (0, 1):
            raise ValueError(f"{validate=} with {scheme=}: needs to be 0 or 1 (a proportion is only available with scheme='walk-forward')")
        n_validate = int(validate)
    if partitions is not None and partitions <= n_validate + test:
        raise ValueError(f"{partitions=}: need at least {n_validate + test + 1} partitions with {validate=} and {test=}")
    if model is not None and split != 'cases':
        raise TypeError(f"model={dataobj_repr(model)!r} with {split=}: model can only be used with split='cases'")
    if min_section is None:
        min_section = 15
    elif split != 'time' or scheme != 'k-fold':
        raise TypeError(f"{min_section=} with {split=}, {scheme=}: min_section can only be used with split='time' and scheme='k-fold'")
    elif tstep is None:
        raise ValueError(f"{min_section=}: a duration in seconds requires tstep")
    elif min_section < 0:
        raise ValueError(f"{min_section=}: needs to be a duration in seconds (0 or more)")
    if split == 'cases':
        n_segments = len(segments)
        # determine model cells
        if model is None:
            categories = [np.arange(n_segments)]
            cell_size = n_segments
        else:
            model = ascategorial(model, data=data, n=n_segments)
            categories = [np.flatnonzero(model == cell) for cell in model.cells]
            cell_sizes = [len(i) for i in categories]
            cell_size = min(cell_sizes)
            if partitions is None and len(set(cell_sizes)) > 1:
                raise NotImplementedError(f'Automatic partition for variable cell size {dict(zip(model.cells, cell_sizes))}')
        # automatic selection of partitions
        if partitions is None:
            if 3 <= cell_size <= 10:
                partitions = cell_size
            else:
                raise NotImplementedError(f"Automatic partition for {cell_size} cases")
        elif partitions > cell_size:
            raise ValueError(f"{partitions=} with {split=}: only {cell_size} cases{'' if model is None else ' per cell'}; use fewer partitions, or split='time' or split='within-cases' to cut cases into pieces")
        # whole segments, interleaved within each cell
        soft_splits = False
        split_segments = segments
        piece_partition = np.empty(n_segments, np.int64)
        for cell_index in categories:
            piece_partition[cell_index] = np.arange(len(cell_index)) % partitions
    else:
        if partitions is None:
            partitions = 4 if scheme == 'walk-forward' else 5

        def cut(start: int, stop: int) -> np.ndarray:
            "Boundaries that cut the samples from ``start`` to ``stop`` into ``partitions`` spans"
            if scheme == 'walk-forward':
                # the first span is the data preceding the windows, the remaining spans are the windows
                n_times = stop - start
                n_window = _ceil_samples(validate * n_times)
                n_train = n_times - (partitions - 1) * n_window
                if n_train < 1:
                    raise ValueError(f"{partitions=} with {scheme=}: {n_times} samples are too short for {partitions - 1} windows of {n_window} samples and training data before them; use fewer partitions")
                return start + np.concatenate([[0], n_train + n_window * np.arange(partitions)])
            return np.round(np.linspace(start, stop, partitions + 1)).astype(np.int64)

        if split == 'within-cases':
            # cut every segment; the i-th piece of every segment forms partition i
            shortest_n_times = np.diff(segments, axis=1).min()
            if scheme == 'k-fold' and shortest_n_times < partitions:
                raise ValueError(f"{partitions=} with {split=}: not enough samples in the shortest segment ({shortest_n_times}) to cut it into {partitions} pieces")
            segment_cuts = np.vstack([cut(start, stop) for start, stop in segments])
            split_segments = np.stack([segment_cuts[:, :-1].ravel(), segment_cuts[:, 1:].ravel()], 1)
            soft_splits = segment_cuts[:, 1:-1].ravel()
            piece_partition = np.tile(np.arange(partitions), len(segments))
        else:
            # cut the concatenated time axis, and additionally at segment boundaries
            if tstep is None or scheme == 'walk-forward':
                n_min_section = 0
            else:
                n_min_section = int(round(min_section / tstep))
            split_points = cut(segments[0, 0], segments[-1, 1])
            boundaries = segments[1:, 0]
            if n_min_section and len(segments) > 1:
                # move span boundaries that would leave a piece shorter than min_section to the segment boundary
                for i in range(1, partitions):
                    i_segment = np.searchsorted(segments[:, 0], split_points[i], side='right') - 1
                    start, stop = segments[i_segment]
                    to_start = split_points[i] - start
                    to_stop = stop - split_points[i]
                    if to_start < n_min_section and to_start <= to_stop and i_segment > 0:
                        split_points[i] = start
                    elif to_stop < n_min_section and i_segment < len(segments) - 1:
                        split_points[i] = stop
            cuts = np.union1d(split_points, boundaries)
            split_segments = np.vstack([cuts[:-1], cuts[1:]]).T
            soft_splits = np.setdiff1d(split_points[1:-1], boundaries)
            if n_min_section and len(segments) > 1:
                # pieces of a cut segment: sections that start or end at a soft split
                is_piece = np.isin(cuts[:-1], soft_splits) | np.isin(cuts[1:], soft_splits)
                if len(np.unique(split_points)) <= partitions or np.any(np.diff(cuts)[is_piece] < n_min_section):
                    span = (segments[-1, 1] - segments[0, 0]) / partitions * tstep
                    raise ValueError(f"{partitions=} with {split=}: can not form {partitions} partitions of {span:g} s without cutting a segment into pieces shorter than min_section ({n_min_section * tstep:g} s); use fewer partitions or a smaller min_section")
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
    if scheme == 'walk-forward':
        splits = [make_split(all_partitions[:i], i, False) for i in all_partitions[1:]]
    elif test:
        splits = [make_split([p for p in all_partitions if p != i], i, True, bool(n_validate)) for i in all_partitions]
    elif n_validate:
        splits = [make_split([p for p in all_partitions if p != i], i, False) for i in all_partitions]
    else:
        splits = [make_split(all_partitions, None, False)]
    return Splits(splits, partitions_arg, partitions, n_validate, test, model, segments, split_segments, split, split_arg, validate_arg, min_section_arg, tstep, scheme)


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


def per_predictor(
        t: float | Sequence[float],
        n_predictors: int,
        name: str,
) -> list[float]:
    """One value per predictor

    Parameters
    ----------
    t
        A scalar applies to all predictors; a sequence specifies one value
        per predictor.
    n_predictors
        Number of predictors.
    name
        Name of the parameter for error messages.
    """
    if isinstance(t, (Sequence, np.ndarray)):
        values = list(t)
        if len(values) != n_predictors:
            raise ValueError(f"{name}={t!r}: need one value per predictor ({n_predictors})")
        return values
    return [t] * n_predictors


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

    i_start = [int(round(t / tstep)) for t in per_predictor(tstart, n_predictors, 'tstart')]
    if tstop is None:
        i_stop = [int(round(t / tstep)) + 1 for t in per_predictor(tmax, n_predictors, 'tmax')]
    else:
        i_stop = [_ceil_samples(t / tstep) for t in per_predictor(tstop, n_predictors, 'tstop')]
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
            v_name = v0.name if v.name is None else v.name  # a list of trials without name is named after the first trial
            blocks.append(RowBlock(v_name, dims, slice(start, start + n_rows), _info.copy(v0.info)))
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
                values = seg - mean[:, newaxis] if center else seg
                if self.n_vector:
                    values = norm(values.reshape((-1, self.n_vector, seg.shape[-1])), axis=1)
                out = None if values is seg else values  # reuse the temporary array, but do not modify the data
                if scale == 'l1':
                    total = total + np.abs(values, out=out).sum(-1)
                else:
                    total = total + np.square(values, out=out).sum(-1)
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
        concatenating would require a transposed copy). The data is float64
        (as required by the Cython routines), and ``owned`` when extracting it
        already required a copy.
    """
    if not isinstance(v, NDVar):
        segments = [np.asarray(vi.get_data(dimnames), np.float64).reshape((n_rows, -1)) for vi in v]
        return segments, None, not any(np.may_share_memory(seg, vi.x) for seg, vi in zip(segments, v))
    elif not case_to_segments:
        data = np.asarray(v.get_data(dimnames), np.float64).reshape((n_rows, -1))
        return [data], None, not np.may_share_memory(data, v.x)
    cases = np.asarray(v.get_data(dimnames), np.float64).reshape((len(v), n_rows, -1))
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
            partitions: int | None = 5,
            *,
            split: Literal['cases', 'time', 'within-cases'] | None = 'time',
            scheme: Literal['k-fold', 'walk-forward'] = 'k-fold',
            validate: float = None,
            test: int = 1,
            model: CategorialArg = None,
            min_section: float = None,
            data: Dataset = None,
    ):
        """Initialize the cross-validation scheme (:attr:`splits`)

        Parameters
        ----------
        partitions
            Number of partitions to divide the data into.
        split
            How to form partitions.
        scheme
            Which partitions are used for training when a partition is held
            out.
        validate
            Size of each validation set.
        test
            Number of partitions in each test set (0 or 1).
        model
            Sample partitions evenly from the cells of this model.
        min_section
            Minimum duration (in seconds) of the pieces left when a segment
            is cut.
        data
            Dataset in which to evaluate ``model`` if it is a string.

        Notes
        -----
        See :func:`split_data` for a full description of the parameters.
        """
        self.splits = split_data(self.segments, tstep=self.time.tstep, partitions=partitions, split=split, scheme=scheme, validate=validate, test=test, model=model, min_section=min_section, data=data)

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
