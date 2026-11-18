# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot data split for cross-validation"""

from collections.abc import Sequence
from numbers import Integral
from typing import Literal

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np

from .._trf.shared import Splits, split_data
from .._data_obj import Dataset, CategorialArg, NDVar, NDVarArg, asndvar, cellname
from ._base import EelFigure, Layout, LegendMixin, LegendArg
from ._styles import colors_for_oneway


class DataSplit(EelFigure, LegendMixin):
    """Plot data splits

    Parameters
    ----------
    splits
        The data splits.
    legend
        Matplotlib figure legend location argument
        (see ``loc`` argument to :class:`matplotlib.legend.Legend`)
        or 'fig' to plot the legend in a separate figure.
    colors
        Customize colors (uses the following keys: ``'train', 'validate',
        'test', 'unused'``, as well as the cells of :attr:`Splits.model`; a
        cell with one of those names uses the same entry).
    xlabel
        X-axis label (default ``'Time [s]'`` with ``tstep``; otherwise
        ``'Case'`` for data with several segments and ``'Sample'`` for
        continuous data).
    tstep
        Time step between samples (default :attr:`Splits.tstep`); with a
        time step, the x-axis shows time in seconds. Without one, it labels
        each data segment (case) at its center, or shows sample indices for
        continuous data.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    Each row corresponds to one boosting run (a leaf :class:`Split`). For
    cross-validation with a test set, rows are grouped by test partition
    (separated by black lines), and the y-axis labels the test partition.
    Data that a run uses neither for training nor as held-out set is
    grayed out. If the partitions were sampled from the cells of a
    ``model``, a line above the plot indicates the cell of each data
    segment.

    This plot can also be generated from a :class:`BoostingResult` object
    through :meth:`BoostingResult.splits.plot` method.
    """

    def __init__(
            self,
            splits: Splits,
            legend: LegendArg = 'upper right',
            colors: dict = None,
            xlabel: str = None,
            tstep: float = None,
            **kwargs,
    ):
        roles = ['train', 'validate', 'test']
        labels = {'train': 'Training', 'validate': 'Validation', 'test': 'Testing', 'unused': 'Unused'}
        cells = [] if splits.model is None else splits.model.cells
        if len(cells) <= 4:
            cell_colors = colors_for_oneway(cells, unambiguous=[7, 4, 8, 2][:len(cells)])
        else:
            cell_colors = colors_for_oneway(cells)
        # cells are kept apart from the roles, with which they can share names
        cell_colors.update(colors or {})
        colors = {**colors_for_oneway(roles, unambiguous=[6, 3, 5]), 'unused': '0.85', **(colors or {})}
        if tstep is None:
            tstep = splits.tstep
        label_cases = tstep is None and len(splits.segments) > 1
        if tstep is None:
            scale = 1
            if xlabel is None:
                xlabel = 'Case' if label_cases else 'Sample'
        else:
            scale = tstep
            if xlabel is None:
                xlabel = 'Time [s]'

        self.splits = splits
        rows = splits.leaves  # one row per boosting run: (top-level split, leaf split)
        is_nested = any(top_split is not split for top_split, split in rows)

        h_default = max(2, 0.5 + 0.15 * len(rows))
        layout = Layout(1, 16 / 9, h_default, **kwargs)
        EelFigure.__init__(self, None, layout)
        ax = self.figure.axes[0]

        x_start, x_stop = splits.segments[0, 0], splits.segments[-1, 1]
        n_samples = np.diff(splits.segments, axis=1).sum()
        for x in splits.segments[:-1, 1]:
            ax.axvline(x * scale, color='k')
        handles = {}
        for y, (top_split, split) in enumerate(rows):
            sets = [(split.train, 'train')]
            for split_i in ([split] if split is top_split else [split, top_split]):
                if split_i.reserved is not None:
                    sets.append((split_i.reserved, 'test' if split_i.is_test else 'validate'))
            # gray background for data the run does not use
            if sum(np.diff(segments, axis=1).sum() for segments, _ in sets) < n_samples:
                handles['unused'] = ax.add_artist(Rectangle((x_start * scale, y - 0.5), (x_stop - x_start) * scale, 1, facecolor=colors['unused'], linewidth=0))
            for segments, role in sets:
                for x0, x1 in segments:
                    rect = Rectangle((x0 * scale, y - 0.5), (x1 - x0) * scale, 1, facecolor=colors[role], linewidth=0)
                    ax.add_artist(rect)
                    handles[role] = rect
            # separate splits (black line between groups of nested splits)
            if y:
                ax.axhline(y - 0.5, color='k' if is_nested and top_split is not rows[y - 1][0] else 'white')
        labels = {key: labels[key] for key in handles}
        # line above the plot indicating the model cell of each data segment
        cell_transform = ax.get_xaxis_transform() + ScaledTranslation(0, 4 / 72, self.figure.dpi_scale_trans)
        for cell in cells:
            lines = [[(x0 * scale, 1), (x1 * scale, 1)] for x0, x1 in splits.segments[splits.model == cell]]
            handles['cell', cell] = ax.add_collection(LineCollection(lines, colors=[cell_colors[cell]], linewidths=4, capstyle='butt', transform=cell_transform, clip_on=False), autolim=False)
            labels['cell', cell] = cellname(cell)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        if is_nested:
            # label groups by top-level split
            ticks = [np.mean([y for y, (top_split, _) in enumerate(rows) if top_split is top_split_i]) for top_split_i in splits.splits]
            ax.set_yticks(ticks, [str(top_split.i_reserved) for top_split in splits.splits])
            ax.set_ylabel('Test partition')
        else:
            ax.set_ylabel('Split')
            if len(rows) == 1:
                ax.set_yticks([])
            else:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_xlim(x_start * scale, x_stop * scale)
        if label_cases:
            centers = splits.segments.mean(1)
            ax.xaxis.set_major_locator(FixedLocator(centers, nbins=10))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(np.searchsorted(centers, x))))
        LegendMixin.__init__(self, legend, handles, labels)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


def preview_partitions(
    cases: int | Sequence[float] | NDVarArg = 0,
    *,
    partitions: int | None = 5,
    split: Literal['cases', 'time', 'within-cases'] | None = 'time',
    scheme: Literal['k-fold', 'walk-forward'] = 'k-fold',
    validate: float = None,
    test: int = 1,
    model: CategorialArg = None,
    min_section: float = None,
    data: Dataset = None,
    **kwargs,
) -> DataSplit:
    """Preview how data will be partitioned for cross-validation

    Parameters
    ----------
    cases
        Description of the data that will be used. Can be specified as
        :class:`NDVar` data, as integer describing the number of trials
        (the default assumes continuous data), or as a sequence of numbers
        describing the duration (in seconds) of each trial, which allows
        trials of different lengths. Trials specified by their number have no
        duration, so ``min_section`` can not be used with them.
    partitions
        Number of partitions to divide the data into (default 5).
    split
        How to form partitions (default ``'time'``).
    scheme
        Which partitions are used for training when a partition is held out.
    validate
        Size of each validation set.
    test
        Number of partitions in each test set (0 or 1).
    model
        Sample partitions evenly from the cells of this model (one entry per
        trial; requires ``split='cases'``). A line above the plot indicates the cell of each trial.
    min_section
        Minimum duration (in seconds) of the pieces left when a trial is cut.
    data
        Dataset in which to evaluate ``cases`` and ``model`` if they are
        strings.
    ...
        For plotting parameters see :class:`DataSplit`.

    See Also
    --------
    :ref:`exa-data_split` example

    Notes
    -----
    See :func:`~eelbrain.split_data` for a full description of the
    cross-validation parameters, which are the same as in
    :func:`~eelbrain.boosting`.
    """
    tstep = None  # trials specified by number: no time information
    if isinstance(cases, Integral):
        # a nominal number of samples per trial (a single trial is continuous data)
        ns = [1000] * max(int(cases), 1)
    elif isinstance(cases, (str, NDVar)):
        y = asndvar(cases, data=data)
        n_cases = len(y) if y.has_case else 1
        ns = [len(y.get_dim('time'))] * n_cases
        tstep = y.time.tstep
    else:
        tstep = 0.01  # resolution for trials specified by duration
        ns = [int(round(duration / tstep)) for duration in cases]
    index = np.cumsum([0] + ns)
    segments = np.hstack([index[:-1, None], index[1:, None]])
    splits = split_data(segments, tstep=tstep, partitions=partitions, split=split, scheme=scheme, validate=validate, test=test, model=model, min_section=min_section, data=data)
    return DataSplit(splits, **kwargs)
