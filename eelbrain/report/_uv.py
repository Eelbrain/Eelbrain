# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from typing import Sequence, Tuple

from .._data_obj import Dataset, IndexArg, VarArg
from .. import fmtxt, plot


def scatter_table(
        xs: Sequence[VarArg],
        color: VarArg = None,
        sub: IndexArg = None,
        data: Dataset = None,
        diagonal: Tuple[float, float] = None,
        rasterize: bool = None,
        markers: str = '.',
        alpha: float = 0.5,
        **kwargs,
) -> fmtxt.Table:
    """Table with pairwise scatter-plots (see :class:`plot.Scatter`)

    Parameters
    ----------
    xs
        List of variables for pairwise comparisons.
    color
        Plot the correlation separately for different categories.
    sub
        Plot a subset of cases.
    data
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    diagonal
        Lower and upper value for plotting a diagonal (e.g. ``(0, 10)`` to plot
        diagonal from ``(0, 0)`` to ``(10, 10)``).
    rasterize
        Rasterize images (improves display performance for plots with many
        points; default: rasterize for datasets with > 500 cases).
    **
        Other :class:`plot.Scatter` arguments.
    """
    # row: start at 0; column: start at 1
    n_xs = len(xs)
    if n_xs < 2:
        raise ValueError(f"xs={xs!r}: need at least 2 items")
    # color-bar placement
    cbar_row = n_xs - 2
    cbar_column = 1 if n_xs > 2 else 2
    xs_columns = xs[1:] if n_xs > 2 else [xs[1], None]
    # generate table
    table = fmtxt.Table('l' * max(2, (n_xs - 1)))
    for i_row, y in enumerate(xs):
        for i_column, x in enumerate(xs_columns, 1):
            if i_column == cbar_column and i_row == cbar_row and color is not None:
                p_cbar = p.plot_colorbar(orientation='vertical', h=p._layout.h, width=0.2)  # noqa: F821
                table.cell(fmtxt.asfmtext(p_cbar, rasterize=rasterize))
                continue
            elif x is None:
                return table
            elif i_column <= i_row:
                table.cell()
                continue
            p = plot.Scatter(y, x, color, sub=sub, markers=markers, data=data, alpha=alpha, **kwargs)
            if diagonal:
                p.axes[0].plot(diagonal, diagonal, color='k')
            if rasterize is None:
                rasterize = p._n_cases > 500
            table.cell(fmtxt.asfmtext(p, rasterize=rasterize))
        table.endline()
    return table
