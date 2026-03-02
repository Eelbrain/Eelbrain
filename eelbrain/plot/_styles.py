# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Style specification"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import reduce
from itertools import chain, product
from math import ceil
import operator
from typing import Any
from collections.abc import Sequence

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba

from .._colorspaces import LocatedListedColormap, lch_to_rgb, rgb_to_lch, oneway_colors, twoway_colors, SYMMETRIC_CMAPS
from .._data_obj import Factor, Interaction, CellArg
from .._exceptions import KeysMissing
from functools import cached_property


# masked keys that act as modifier rather than replacement
modifer_keys = {'alpha', 'saturation'}


@dataclass
class Style:
    """Plotting style for :class:`~matplotlib.lines.Line2D`, :class:`~matplotlib.patches.Patch` and :func:`matplotlib.pyplot.scatter` plot types.

    Parameters
    ----------
    color
        Line color or face-color for patches and markers.
    marker
        Maker shape (see :mod:`matplotlib.markers`). Use ``line_marker`` to
        control whether markers are used for lines.
    hatch
        For patches (see :meth:`~matplotlib.patches.Patch.set_hatch`).
    linestyle
        For lines, see :meth:`matplotlib.lines.Line2D.set_linestyle`).
    linewidth
        For lines.
    zorder
        Z-order for plot elements, relative to object-specific default (use
        positive values to bring this category to the front).
    masked
        Alternate style for masked elements (e.g., when plotting permutation
        test results, lines that are not significant).
    edgecolor
        For patches and markers.
    edgewidth
        Line-width for the edges of patches and markers.
    linemarker
        Plot markers in line plots.

    Examples
    --------
    - :ref:`exa-utsstat`
    """
    color: Any = (0, 0, 0)
    marker: str = None
    hatch: str = ''
    linestyle: str = None
    linewidth: float = None
    zorder: float = 0  # z-order shift (relative to plot element default)
    masked: Any | dict[str, Any] = None  # Any should be Style, but autodoc does not support foward reference under decorator yet https://github.com/agronholm/sphinx-autodoc-typehints/issues/76
    edgecolor: Any = None
    edgewidth: float = 0,
    linemarker: bool | str = False
    alias: CellArg = None  # This style should not appear in legends

    @cached_property
    def line_args(self):
        args = {'color': self.color, 'linestyle': self.linestyle, 'markerfacecolor': self.color, 'zorder': 2 + self.zorder}
        if self.linewidth is not None:
            args['linewidth'] = self.linewidth
        if self.linemarker is True:
            args['marker'] = self.marker
        elif self.linemarker:
            args['marker'] = self.linemarker
        return args

    def fill_args(self, alpha: float):
        r, g, b, a = to_rgba(self.color)
        a *= alpha
        return {'color': (r, g, b, a), 'zorder': 1.5 + self.zorder}

    @cached_property
    def scatter_args(self):
        args = {'color': self.color, 'edgecolor': self.edgecolor, 'linewidth': self.edgewidth, 'facecolor': self.color, 'zorder': 2 + self.zorder}
        if self.marker:
            args['marker'] = self.marker
        return args

    @cached_property
    def patch_args(self):
        return {'facecolor': self.color, 'hatch': self.hatch, 'zorder': 1 + self.zorder}

    @cached_property
    def masked_style(self):  # -> 'Style'
        if self.masked is None:
            return replace(self, color=(0.7, 0.7, 0.7, 0.4))
        elif isinstance(self.masked, Style):
            return self.masked
        elif isinstance(self.masked, dict):
            kwargs = self.masked
            if modifer_keys.intersection(self.masked):
                kwargs = dict(kwargs)
                r, g, b, a = to_rgba(self.color)
                a *= kwargs.pop('alpha', 1)
                l, c, h = rgb_to_lch(r, g, b)
                c *= kwargs.pop('saturation', 1)
                kwargs['color'] = (*lch_to_rgb(l, c, h), a)
            return replace(self, **kwargs)
        else:
            raise TypeError(f"Style is invalid masked parameter: {self.masked!r}")

    @classmethod
    def _coerce(cls, arg):
        if isinstance(arg, cls):
            return arg
        elif arg is None:
            return cls()
        else:
            return cls(arg)


def to_styles_dict(colors: dict[CellArg, Any]) -> StylesDict:
    return {cell: Style._coerce(spec) for cell, spec in colors.items()}


def find_cell_styles(
        cells: Sequence[CellArg] = None,
        colors: ColorsArg = None,
        fallback: bool = True,
        default_cells: Sequence[CellArg] = None,
) -> StylesDict:
    """Process the colors arg from plotting functions

    Parameters
    ----------
    cells
        Cells for which colors are needed.
    colors
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; cells are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as cells.
        **dict**: A dictionary mapping each cell to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    fallback
        If a cell is missing, fall back on partial cells (on by default).
    default_cells
        Alternative cells to use if ``colors`` is ``None`` or a :class:`dict`.
        For example, when plots use the ``xax`` parameter, cells will contain a
        ``xax`` component, but colors should be consistent across axes.
    raise_for_missing
        Raise an error for missig cells.
    """
    if cells is None:
        cells = (None,)
    if cells == (None,):
        if isinstance(colors, dict):
            if None in colors:
                out = colors
            else:
                raise KeysMissing((None,), 'colors', colors)
        else:
            if colors is None:
                colors = 'k'
            out = {None: colors}
    elif isinstance(colors, (list, tuple)):
        if len(colors) < len(cells):
            raise ValueError(f"{colors=}: only {len(colors)} colors for {len(cells)} cells.")
        out = dict(zip(cells, colors))
    elif isinstance(colors, dict):
        out = to_styles_dict(colors)
        missing = [cell for cell in cells if cell not in out]
        if missing:
            if fallback:
                for cell in missing[:]:
                    if isinstance(cell, str):
                        continue
                    i_max = len(cell) - 1
                    super_cells = chain((cell[:-i] for i in range(1, i_max)), (cell[i:] for i in range(1, i_max)), (cell[0], cell[-1]))
                    for super_cell in super_cells:
                        if super_cell in out:
                            out[cell] = replace(out[super_cell], alias=super_cell)
                            missing.remove(cell)
                            break
            if missing:
                raise KeysMissing(missing, 'colors', colors)
        return out
    elif colors is None or isinstance(colors, str):
        if default_cells is not None:
            default_colors = find_cell_styles(default_cells, colors)
            return find_cell_styles(cells, default_colors)
        elif all(isinstance(cell, str) for cell in cells):
            out = colors_for_oneway(cells, cmap=colors)
        elif all(isinstance(cell, tuple) for cell in cells):
            ns = {len(cell) for cell in cells}
            if len(ns) == 1:
                out = colors_for_nway(list(zip(*cells)))
            else:
                raise NotImplementedError(f"{cells=}: unequal cell size")
        else:
            raise NotImplementedError(f"{cells=}: unequal cell size")
    else:
        raise TypeError(f"{colors=}")
    return to_styles_dict(out)


def colors_for_categorial(x, hue_start=0.2, cmap=None):
    """Automatically select colors for a categorial model

    Parameters
    ----------
    x : categorial
        Model defining the cells for which to define colors.
    hue_start : 0 <= scalar < 1
        First hue value (only for two-way or higher level models).
    cmap : str (optional)
        Name of a matplotlib colormap to use instead of default hue-based
        colors (only used for one-way models).

    Returns
    -------
    colors : dict {cell -> color}
        Dictionary providing colors for the cells in x.
    """
    if isinstance(x, Factor):
        return colors_for_oneway(x.cells, hue_start, cmap=cmap)
    elif isinstance(x, Interaction):
        return colors_for_nway([f.cells for f in x.base], hue_start)
    else:
        raise TypeError(f"{x=}: needs to be Factor or Interaction")


def colors_for_oneway(
        cells,
        hue_start: float | Sequence[float] = 0.2,
        light_range: float | tuple[float, float] = 0.5,
        cmap: str = None,
        light_cycle: int = None,
        always_cycle_hue: bool = False,
        locations: Sequence[float] = None,
        unambiguous: bool | Sequence[int] = None,
) -> dict:
    """Define colors for a single factor design

    Parameters
    ----------
    cells : sequence of str
        Cells for which to assign colors.
    hue_start
        First hue value (``0 <= hue < 1``) or list of hue values.
    light_range : scalar | tuple of 2 scalar
        Scalar that specifies the amount of lightness variation (default 0.5).
        If positive, the first color is lightest; if negative, the first color
        is darkest. A tuple can be used to specify exact end-points (e.g.,
        ``(1.0, 0.4)``). ``0.2`` is equivalent to ``(0.4, 0.6)``.
        The ``light_cycle`` parameter can be used to cycle between light and
        dark more than once.
    cmap
        Use a matplotlib colormap instead of the default color generation
        algorithm. Name of a matplotlib colormap to use (e.g., 'jet'). If
        specified, ``hue_start`` and ``light_range`` are ignored.
    light_cycle
        Cycle from light to dark in ``light_cycle`` cells to make nearby colors
        more distinct (default cycles once).
    always_cycle_hue
        Cycle hue even when cycling lightness. With ``False`` (default), hue
        is constant within a lightness cycle.
    locations
        Locations of the cells on the color-map (all in range [0, 1]; default is
        evenly spaced; example: ``numpy.linspace(0, 1, len(cells)) ** 0.5``).
    unambiguous
        Use `unambiguos colors <https://jfly.uni-koeln.de/html/color_blind/
        #pallet>`_. If ``True``, choose the ``n`` first colors; use a list of
        ``int`` to pick specific colors. Other parameters are ignored.
        For resources for constructing custom color palettes, see also:
        https://davidmathlogic.com/colorblind/.

    Returns
    -------
    Mapping from cells to colors.
    """
    if isinstance(cells, Iterator):
        cells = tuple(cells)
    n = len(cells)
    if cmap is None:
        colors = oneway_colors(n, hue_start, light_range, light_cycle, always_cycle_hue, locations, unambiguous)
    else:
        cm = mpl.colormaps.get_cmap(cmap)
        if locations is None:
            if n == 1:
                locations = [0.5]
            else:
                imax = n - 1
                locations = (i / imax for i in range(n))
        colors = (cm(x) for x in locations)
    return dict(zip(cells, colors))


def colors_for_twoway(
        x1_cells: Sequence[str],
        x2_cells: Sequence[str],
        hue_start: float = 0.2,
        hue_shift: float = 0.,
        hues: Sequence[float] = None,
        lightness: float | Sequence[float] = None,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Define cell colors for a two-way design

    Parameters
    ----------
    x1_cells
        Cells of the major factor.
    x2_cells
        Cells of the minor factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    hues
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    lightness
        If specified as scalar, colors will occupy the range
        ``[lightness, 100-lightness]``. Can also be given as list with one
        value corresponding to each element in the second factor.

    Returns
    -------
    Mapping from cells to colors.
    """
    x1_cells = list(x1_cells)
    x2_cells = list(x2_cells)
    n1 = len(x1_cells)
    n2 = len(x2_cells)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 cells on each factor")

    clist = twoway_colors(n1, n2, hue_start, hue_shift, hues, lightness)
    return dict(zip(product(x1_cells, x2_cells), clist))


def colors_for_nway(
        cell_lists: Sequence[CellArg],
        hue_start: float = 0.2,
) -> dict[CellArg, tuple]:
    """Define cell colors for a two-way design

    Parameters
    ----------
    cell_lists : sequence of of tuple of str
        List of the cells for each factor. E.g. for ``A % B``:
        ``[('a1', 'a2'), ('b1', 'b2', 'b3')]``.
    hue_start : 0 <= scalar < 1
        First hue value.

    Returns
    -------
    Mapping from cells to colors.
    """
    if len(cell_lists) == 1:
        return colors_for_oneway(cell_lists[0])
    elif len(cell_lists) == 2:
        return colors_for_twoway(cell_lists[0], cell_lists[1], hue_start)
    elif len(cell_lists) > 2:
        ns = tuple(map(len, cell_lists))
        n_outer = reduce(operator.mul, ns[:-1])
        n_inner = ns[-1]

        # outer circle
        hues = np.linspace(hue_start, 1 + hue_start, ns[0], False)
        # subdivide for each  level
        distance = 1. / ns[0]
        for n_current in ns[1:-1]:
            new = []
            d = distance / 3
            for hue in hues:
                new.extend(np.linspace(hue - d, hue + d, n_current))
            hues = new
            distance = 2 * d / (n_current - 1)
        hues = np.asarray(hues)
        hues %= 1
        colors = twoway_colors(n_outer, n_inner, hues=hues)
        return dict(zip(product(*cell_lists), colors))
    else:
        return {}


def styles_for_twoway(
        x1_cells: Sequence[str],
        x2_cells: Sequence[str],
        colors: dict[CellArg, Any] = None,
        linestyles: Sequence[Any] = ('-', '--', ':', '-.'),
) -> dict[tuple[str, str], Style]:
    """Cross color with linestyle

    Parameters
    ----------
    x1_cells
        Cells to alternate colors.
    x2_cells
        Cells to alternate linestyle.
    colors
        Colors for ``x1_cells`` (as generated by :func:`colors_for_oneway`).
    linestyles
        Linestyles for ``x2_cells``.

    Returns
    -------
    Mapping from cells to plot styles.
    """
    x1_cells = list(x1_cells)
    x2_cells = list(x2_cells)
    if colors is None:
        colors = colors_for_oneway(x1_cells)
    out = {}
    for x1_cell in x1_cells:
        for linestyle, x2_cell in zip(linestyles, x2_cells):
            out[x1_cell, x2_cell] = Style(colors[x1_cell], linestyle=linestyle)
    return out


def single_hue_colormap(hue: ColorArg) -> LinearSegmentedColormap:
    """Colormap ranging from transparent to ``hue``

    Parameters
    ----------
    hue : matplotlib color
        Base RGB color.
    """
    name = str(hue)
    color = to_rgb(hue)
    start = color + (0.,)
    stop = color + (1.,)
    return LinearSegmentedColormap.from_list(name, (start, stop))


def soft_threshold_colormap(
        cmap: str,
        threshold: float,
        vmax: float,
        subthreshold: ColorArg = None,
        symmetric: bool = None,
        alpha: float = 1,
) -> mpl.colors.ListedColormap:
    """Soft-threshold a colormap to make small values transparent

    Parameters
    ----------
    cmap
        Base colormap.
    threshold
        Value at which to threshold the colormap (i.e., the value at which to
        start the colormap).
    vmax
        Intended largest value of the colormap (used to infer the location of
        the ``threshold``).
    subthreshold : matplotlib color
        Color of sub-threshold values (the default is the end or middle of
        the colormap, depending on whether it is symmetric).
    symmetric
        Whether the ``cmap`` is symmetric (ranging from ``-vmax`` to ``vmax``)
        or not (ranging from ``0`` to ``vmax``). The default is ``True`` for
        known symmetric colormaps and ``False`` otherwise.
    alpha
        Control the global alpha level (opacity) of the colormap (original
        colormap alpha is multiplied by ``alpha``).
    """
    assert vmax > threshold >= 0
    cmap = mpl.colormaps.get_cmap(cmap)
    if symmetric is None:
        symmetric = cmap.name in SYMMETRIC_CMAPS

    colors = cmap(np.linspace(0., 1., cmap.N))
    if subthreshold is None:
        subthreshold_color = cmap(0.5) if symmetric else cmap(0)
    else:
        subthreshold_color = to_rgba(subthreshold)

    n = int(round(vmax / ((vmax - threshold) / cmap.N)))
    out_colors = np.empty((n, 4))
    if symmetric:
        i_threshold = int(ceil(cmap.N / 2))
        out_colors[:i_threshold] = colors[:i_threshold]
        out_colors[i_threshold:-i_threshold] = subthreshold_color
        out_colors[-i_threshold:] = colors[-i_threshold:]
    else:
        out_colors[:-cmap.N] = subthreshold_color
        out_colors[-cmap.N:] = colors
    if alpha != 1:
        out_colors[:, 3] *= alpha
    out = LocatedListedColormap(out_colors, cmap.name)
    out.vmax = vmax
    out.vmin = -vmax if symmetric else 0
    out.symmetric = symmetric
    return out


ColorArg = str | Sequence[float]
ColorsArg = ColorArg | dict[CellArg, ColorArg] | Sequence[ColorArg]
StylesDict = dict[CellArg | None, Style]
