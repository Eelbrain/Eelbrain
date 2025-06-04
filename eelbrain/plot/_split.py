# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot data split for cross-validation"""
from typing import Union

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
        This plot can also be generated from a :class:`BoostingResult` object
        through :meth:`BoostingResult.splits.plot` method.
        """
        attrs = ['train', 'validate', 'test']
        labels = {'train': 'Training', 'validate': 'Validation', 'test': 'Testing'}
        if colors is None:
            colors = colors_for_oneway(attrs, unambiguous=[6, 3, 5])

        h_default = max(2, 0.5 + 0.15 * len(splits.splits))
        layout = Layout(1, 16 / 9, h_default, **kwargs)
        EelFigure.__init__(self, None, layout)
        ax = self.figure.axes[0]

        for x in splits.segments[:-1, 1]:
            ax.axvline(x, color='k')
        handles = {}
        for y, split in enumerate(splits.splits):
            for attr in attrs:
                segments = getattr(split, attr)
                if segments is None:
                    continue
                rect = None
                for x0, x1 in segments:
                    rect = Rectangle((x0, y - 0.5), x1 - x0, 1, color=colors[attr])
                    ax.add_artist(rect)
                handles[attr] = rect
            # separate splits
            if y:
                ax.axhline(y - 0.5, color='white')
        labels = {key: labels[key] for key in handles}
        ax.set_ylabel('Split')
        ax.set_ylim(-0.5, len(splits.splits) - 0.5)
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
    cases: Union[int, NDVarArg] = 0,
    partitions: int = None,
    model: CategorialArg = None,
    validate: int = 1,
    test: int = 0,
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
                partitions = 2 + test + validate if test else 10
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
