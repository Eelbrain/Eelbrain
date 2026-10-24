# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot data split for cross-validation"""

from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy

from .._trf.shared import Splits, split_data
from .._data_obj import Dataset, CategorialArg, NDVarArg, asndvar
from .._utils import deprecate_ds_arg
from ._base import EelFigure, Layout, LegendMixin, LegendArg
from ._styles import colors_for_oneway


class DataSplit(EelFigure, LegendMixin):

    def __init__(
            self,
            splits: Splits,
            legend: LegendArg = 'upper right',
            colors: dict = None,
            xlabel: str = 'Segment',
            **kwargs,
    ):
        """Plot data splits

        Parameters
        ----------
        splits
            The data splits.
        legend
            Matplotlib figure legend location argument or 'fig' to plot the
            legend in a separate figure.
        colors
            Customize colors (uses the following keys: ``'train', 'validate',
            'test'``).
        ...
            Also accepts :ref:`general-layout-parameters`.

        Notes
        -----
        Each row corresponds to one boosting run (a leaf :class:`Split`). For
        cross-validation with a test set, rows are grouped by test partition
        (separated by black lines), and the y-axis labels the test partition.

        This plot can also be generated from a :class:`BoostingResult` object
        through :meth:`BoostingResult.splits.plot` method.
        """
        roles = ['train', 'validate', 'test']
        labels = {'train': 'Training', 'validate': 'Validation', 'test': 'Testing'}
        if colors is None:
            colors = colors_for_oneway(roles, unambiguous=[6, 3, 5])

        rows = splits.leaves  # one row per boosting run: (top-level split, leaf split)
        is_nested = any(top_split is not split for top_split, split in rows)

        h_default = max(2, 0.5 + 0.15 * len(rows))
        layout = Layout(1, 16 / 9, h_default, **kwargs)
        EelFigure.__init__(self, None, layout)
        ax = self.figure.axes[0]

        for x in splits.segments[:-1, 1]:
            ax.axvline(x, color='k')
        handles = {}
        for y, (top_split, split) in enumerate(rows):
            sets = [(split.train, 'train')]
            for split_i in ([split] if split is top_split else [split, top_split]):
                if split_i.reserved is not None:
                    sets.append((split_i.reserved, 'test' if split_i.is_test else 'validate'))
            for segments, role in sets:
                for x0, x1 in segments:
                    rect = Rectangle((x0, y - 0.5), x1 - x0, 1, color=colors[role])
                    ax.add_artist(rect)
                    handles[role] = rect
            # separate splits (black line between groups of nested splits)
            if y:
                ax.axhline(y - 0.5, color='k' if is_nested and top_split is not rows[y - 1][0] else 'white')
        labels = {key: labels[key] for key in handles}
        ax.set_ylim(-0.5, len(rows) - 0.5)
        if is_nested:
            # label groups by top-level split
            ticks = [numpy.mean([y for y, (top_split, _) in enumerate(rows) if top_split is top_split_i]) for top_split_i in splits.splits]
            ax.set_yticks(ticks, [str(top_split.i_reserved) for top_split in splits.splits])
            ax.set_ylabel('Test partition')
        else:
            ax.set_ylabel('Split')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_xlim(splits.segments[0, 0], splits.segments[-1, 1])
        LegendMixin.__init__(self, legend, handles, labels)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


@deprecate_ds_arg
def preview_partitions(
    cases: int | NDVarArg = 0,
    partitions: int = None,
    model: CategorialArg = None,
    validate: int = 1,
    test: int = 1,
    data: Dataset = None,
    **kwargs,
) -> DataSplit:
    """Preview how data will be partitioned for the boosting function

    Parameters
    ----------
    cases
        Description of the data that will be used. Can be specified as
        :class:`NDVar` data, or as integer describing the number of trials
        (the default assumes continuous data).
    ...
        For a description of the splitting parameters see :func:`~eelbrain.boosting`.
        For plotting parameters see :class:`DataSplit`.

    See Also
    --------
    :ref:`exa-data_split` example
    """
    if isinstance(cases, int):
        if cases == 0:
            has_case = False
            if partitions is None:
                partitions = 5
            ns = [partitions]
        else:
            has_case = True
            if partitions is None:
                raise NotImplementedError('Automatic partitions with trials')
            ns = [1] * cases
    else:
        y = asndvar(cases, data=data)
        has_case = y.has_case
        if y.has_case:
            n_cases = len(y)
        else:
            n_cases = 1
        ns = [len(y.get_dim('time'))] * n_cases
    index = numpy.cumsum([0] + ns)
    segments = numpy.hstack([index[:-1, None], index[1:, None]])
    splits = split_data(segments, partitions, model, data, validate, test)
    kwargs.setdefault('xlabel', 'Case' if has_case else 'Segment')
    return DataSplit(splits, **kwargs)
