# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plots for color/style legends"""
from collections.abc import Iterator
import colorsys
from itertools import chain
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.cm
from matplotlib.colors import Colormap, Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches
from matplotlib.ticker import FixedFormatter, MaxNLocator

from .._data_obj import CellArg, cellname
from .._utils import IS_WINDOWS
from ._base import EelFigure, Layout, AxisScale, CMapArg, ColorArg, fix_vlim_for_cmap, inch_to_figure
from ._styles import find_cell_styles


POINT_SIZE = 0.0138889  # 1 point in inches
LEGEND_SIZE = 1.2  # times font.size


class ColorGrid(EelFigure):
    """Plot colors for a two-way design in a grid

    Parameters
    ----------
    row_cells
        Cells contained in the rows.
    column_cells
        Cells contained in the columns.
    colors
        Colors for cells.
    size
        Size (width and height) of the color squares (the default is to
        scale them to fit the font size).
    column_label_position : 'top' | 'bottom' | 'none'
        Where to place the column labels (default is 'top').
    row_first
        Whether the row cell precedes the column cell in color keys. By
        default this is inferred from the existing keys.
    labels
        Condition labels that are used instead of the keys in ``row_cells`` and
        ``column_cells``.
    shape : 'box' | 'line'
        Shape for color samples (default 'box').
    ...
        Also accepts :ref:`general-layout-parameters`.

    Attributes
    ----------
    column_labels : list of :class:`matplotlib.text.Text`
        Column labels.
    row_labels : list of :class:`matplotlib.text.Text`
        Row labels.
    """
    def __init__(
            self,
            row_cells: Sequence[str],
            column_cells: Sequence[str],
            colors: Dict[CellArg, Any],
            size: float = None,
            column_label_position: str = 'top',
            row_first: bool = None,
            labels: dict = None,
            shape: str = 'box',
            **kwargs):
        row_cells = list(row_cells)
        column_cells = list(column_cells)
        if row_first is None:
            row_cell_0 = row_cells[0]
            col_cell_0 = column_cells[0]
            if (row_cell_0, col_cell_0) in colors:
                row_first = True
            elif (col_cell_0, row_cell_0) in colors:
                row_first = False
            else:
                raise KeyError(f"Neither {(row_cell_0, col_cell_0)} nor {(col_cell_0, row_cell_0)} exist as a key in colors")

        if row_first:
            cells = list(zip(row_cells, column_cells))
        else:
            cells = list(zip(column_cells, row_cells))
        styles = find_cell_styles(cells, colors)

        # reverse rows so we can plot upwards
        row_cells = tuple(reversed(row_cells))
        n_rows = len(row_cells)
        n_cols = len(column_cells)

        # prepare labels
        if labels:
            column_labels = [labels.get(c, c) for c in column_cells]
            row_labels = [labels.get(c, c) for c in row_cells]
        else:
            column_labels = column_cells
            row_labels = row_cells

        # default size
        chr_size = matplotlib.rcParams['font.size'] * POINT_SIZE
        if size is None:
            size = chr_size * LEGEND_SIZE
        w_default = size * (n_cols + 1) + chr_size * max(len(l) for l in row_labels)
        h_default = size * n_rows
        if column_label_position != 'none':
            h_default += chr_size * max(len(l) for l in column_labels)
        aspect = w_default / h_default
        layout = Layout(0, aspect, h_default, tight=False, **kwargs)
        EelFigure.__init__(self, None, layout)
        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()
        self._ax = ax

        # color patches
        for col in range(n_cols):
            for row in range(n_rows):
                if row_first:
                    cell = (row_cells[row], column_cells[col])
                else:
                    cell = (column_cells[col], row_cells[row])

                if shape == 'box':
                    patch = matplotlib.patches.Rectangle((col, row), 1, 1, ec='none', **styles[cell].patch_args)
                    ax.add_patch(patch)
                elif shape == 'line':
                    y = row + 0.5
                    ax.plot([col, col + 1], [y, y], **styles[cell].line_args)
                else:
                    raise ValueError(f"shape={shape!r}")

        # column labels
        self.column_labels = []
        if column_label_position == 'none':
            ymin = 0
            ymax = self._layout.h / size
        else:
            tilt_labels = any(len(label) > 1 for label in column_labels)
            if column_label_position == 'top':
                y = n_rows + 0.1
                va = 'bottom'
                rotation = 40 if tilt_labels else 0
                ymin = 0
                ymax = self._layout.h / size
            elif column_label_position == 'bottom':
                y = -0.1
                va = 'top'
                rotation = -40 if tilt_labels else 0
                ymax = n_rows
                ymin = n_rows - self._layout.h / size
            else:
                raise ValueError(f"column_label_position={column_label_position!r}")

            ha = 'left' if tilt_labels else 'center'
            x_offset = 0 if tilt_labels else 0.5
            y_offset = tilt_labels * size * 1.5
            for col, label in enumerate(column_labels):
                y_col = y + y_offset * (len(column_labels) - col - 1)
                h = ax.text(col + x_offset, y_col, label, va=va, ha=ha, rotation=rotation)
                self.column_labels.append(h)

        # row labels
        x = n_cols + 0.1
        self.row_labels = []
        for row, label in enumerate(row_labels):
            h = ax.text(x, row + 0.5, label, va='center', ha='left')
            self.row_labels.append(h)

        if size is not None:
            self._ax.set_xlim(0, self._layout.w / size)
            self._ax.set_ylim(ymin, ymax)

        self._show()

    def _tight(self):
        # arbitrary default with equal aspect
        self._ax.set_ylim(0, 1)
        self._ax.set_xlim(0, 1 * self._layout.w / self._layout.h)

        # draw to compute text coordinates
        self.draw()

        # find label bounding box
        xmax = 0
        ymax = 0
        for h in chain(self.column_labels, self.row_labels):
            bbox = h.get_window_extent()
            if bbox.xmax > xmax:
                xmax = bbox.xmax
                xpos = h.get_position()[0]
            if bbox.ymax > ymax:
                ymax = bbox.ymax
                ypos = h.get_position()[1]
        xmax += 2
        ymax += 2

        # transform from display coordinates -> data coordinates
        trans = self._ax.transData.inverted()
        xmax, ymax = trans.transform((xmax, ymax))

        # calculate required movement
        _, ax_xmax = self._ax.get_xlim()
        _, ax_ymax = self._ax.get_ylim()
        xtrans = ax_xmax - xmax
        ytrans = ax_ymax - ymax

        # calculate the scale factor:
        # new_coord = x * coord
        # new_coord = coord + trans
        # x = (coord + trans) / coord
        scale = (xpos + xtrans) / xpos
        scale_y = (ypos + ytrans) / ypos
        if scale_y <= scale:
            scale = scale_y

        self._ax.set_xlim(0, ax_xmax / scale)
        self._ax.set_ylim(0, ax_ymax / scale)


class ColorList(EelFigure):
    """Plot colors with labels

    Parameters
    ----------
    colors
        Colors for cells.
    cells
        Cells for which to plot colors (default is ``colors.keys()``).
    labels
        Condition labels that are used instead of the keys in ``colors``. This
        is useful if ``colors`` uses abbreviated labels, but the color legend
        should contain more intelligible labels.
    size
        Size (width and height) of the color squares (the default is to
        scale them to fit the font size).
    h
        Height of the figure in inches (default is chosen to fit all labels).
    shape
        Shape for color samples (default 'box').
    markersize
        Size of markers in points.
    linewidth
        Linewidth when plotting colors as lines.
    label_position
        Whether to place labels to the left or right of the colors.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Attributes
    ----------
    labels : list of :class:`matplotlib.text.Text`
        Color labels.
    """
    def __init__(
            self,
            colors: Dict[CellArg, Any],
            cells: Sequence[CellArg] = None,
            labels: Dict[CellArg, str] = None,
            size: float = None,
            h: float = None,
            shape: Literal['box', 'line', 'marker'] = 'box',
            markersize: float = None,
            linewidth: float = None,
            label_position: Literal['left', 'right'] = 'right',
            **kwargs):
        if label_position not in ('left', 'right'):
            raise ValueError(f"{label_position=}")
        if cells is None:
            cells = colors.keys()
        elif isinstance(cells, Iterator):
            cells = tuple(cells)
        styles = find_cell_styles(cells, colors)

        if h is None:
            if size is None:
                size = matplotlib.rcParams['font.size'] * LEGEND_SIZE * POINT_SIZE
            h = len(cells) * size
        elif size is None:  # size = h / len(cells)
            pass
        else:
            raise NotImplementedError("specifying size and h parameters together")

        if labels is None:
            labels = {cell: cellname(cell) for cell in cells}
        elif not isinstance(labels, dict):
            raise TypeError(f"{labels=}")

        layout = Layout(0, 1.5, 2, False, h=h, **kwargs)
        EelFigure.__init__(self, None, layout)

        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()

        if label_position == 'right':
            x = 1.1
            ha = 'left'
        else:
            x = -0.1
            ha = 'right'

        n = len(cells)
        self.labels = []
        for i, cell in enumerate(cells):
            bottom = n - i - 1
            y = bottom + 0.5
            if shape == 'box':
                patch = matplotlib.patches.Rectangle((0, bottom), 1, 1, ec='none', **styles[cell].patch_args)
                ax.add_patch(patch)
            elif shape == 'line':
                ax.plot([0, 1], [y, y], **styles[cell].line_args, markersize=markersize, linewidth=linewidth)
            elif shape == 'marker':
                ax.scatter(0.5, y, markersize, **styles[cell].scatter_args)
            else:
                raise ValueError(f"{shape=}")
            h = ax.text(x, y, labels.get(cell, cell), va='center', ha=ha, zorder=2)
            self.labels.append(h)

        ax.set_ylim(0, n)
        width = n * self._layout.w / self._layout.h
        if label_position == 'right':
            ax.set_xlim(0, width)
        else:
            ax.set_xlim(-width + 1, 1)

        self._draw_hooks.append(self.__update_frame)
        self._ax = ax
        self._show()
        if IS_WINDOWS and self._has_frame:
            self._frame.Fit()

    def __update_frame(self):
        if self._layout.w_fixed or not self._has_frame:
            return

        # resize figure to match legend
        # (all calculation in pixels)
        fig_bb = self.figure.get_window_extent()
        x_max = max(h.get_window_extent().x1 for h in self.labels)
        w0, h0 = self._frame.GetSize()
        new_w = w0 + (x_max - fig_bb.x1) + 4
        self._frame.SetSize(int(new_w), h0)
        # adjust x-limits
        n = len(self.labels)
        ax_bb = self._ax.get_window_extent()
        self._ax.set_xlim(0, n * ax_bb.width / ax_bb.height)


class ColorBar(EelFigure):
    """A color-bar for a matplotlib color-map

    Parameters
    ----------
    cmap
        Name of the color-map, or a matplotlib Colormap, or LUT.
    vmin
        Lower end of the scale mapped onto cmap.
    vmax
        Upper end of the scale mapped onto cmap.
    label
        Label for the x-axis (default is the unit, or if no unit is provided
        the name of the colormap).
    label_position
        Position of the axis label. Valid values depend on orientation.
    label_rotation
        Angle of the label in degrees (For horizontal colorbars, the default is
        0; for vertical colorbars, the default is 0 for labels of 3 characters
        and shorter, and 90 for longer labels).
    clipmin
        Clip the color-bar below this value.
    clipmax
        Clip the color-bar above this value.
    orientation
        Orientation of the bar (default is horizontal).
    unit
        Unit for the axis to determine tick labels (for example, ``'µV'`` to
        label 0.000001 as '1') or multiplier (e.g., ``1e-6``).
    contours
        Plot contour lines at these values.
    width
        Width of the color-bar in inches.
    ticks
        Control tick-labels on the colormap. Can be:

         - An integer number of evenly spaced ticks to draw
         - A sequence of tick locations (``()`` to draw no ticks at all)
         - A ``{float: str}`` dictionary with tick-locations and labels

    threshold
        Set the alpha of values below ``threshold`` to 0 (as well as for
        negative values above ``abs(threshold)``).
    ticklocation
        Where to place ticks and label.
    background : matplotlib color
        Background color (for colormaps including transparency).
    ...
        Also accepts :ref:`general-layout-parameters`.
    right_of
        Plot the colorbar to the right of, and matching in height of the axes
        specified in ``right_of``.
    left_of
        Plot the colorbar to the keft of, and matching in height of these axes.
    below
        Plot the colorbar below, and matching in width of these axes.
    offset
        Additional offset when using ``right_of``/``left_of``/``below``.
    """
    def __init__(
            self,
            cmap: CMapArg,
            vmin: float = None,
            vmax: float = None,
            label: Union[bool, str] = True,
            label_position: Literal['left', 'right', 'top', 'bottom'] = None,
            label_rotation: float = None,
            clipmin: float = None,
            clipmax: float = None,
            orientation: Literal['horizontal', 'vertical'] = 'horizontal',
            unit: Union[str, float] = None,
            contours: Sequence[float] = (),
            width: float = None,
            ticks: Union[int, Dict[float, str], Sequence[float]] = None,
            threshold: float = None,
            ticklocation: Literal['auto', 'top', 'bottom', 'left', 'right'] = 'auto',
            background: ColorArg = None,
            tight: bool = True,
            h: float = None,
            w: float = None,
            axes: matplotlib.axes.Axes = None,
            right_of: matplotlib.axes.Axes = None,
            left_of: matplotlib.axes.Axes = None,
            below: matplotlib.axes.Axes = None,
            offset: float = 0,
            **kwargs,
    ):
        # get Colormap
        if isinstance(cmap, np.ndarray):
            if threshold is not None:
                raise NotImplementedError("threshold parameter with cmap=<array>")
            if cmap.max() > 1:
                cmap = cmap / 255.
            cm = matplotlib.colors.ListedColormap(cmap, 'LUT')
        elif isinstance(cmap, Colormap):
            cm = cmap
        else:
            cm = matplotlib.colormaps.get_cmap(cmap)

        if any([right_of, left_of, below]):
            if sum(map(bool, [axes, right_of, left_of, below])) != 1:
                raise TypeError(f"{axes=}, {right_of=}, {left_of=}, {below=}: can only specify one at a time")
            source_axes = right_of or left_of or below
            figure = source_axes.get_figure()
            source_bbox = source_axes.get_position()
            if right_of or left_of:
                thickness = inch_to_figure(figure, x=width or 0.05)
                y0 = source_bbox.y0
                if h:
                    height = inch_to_figure(figure, y=h)
                    y0 += (source_bbox.bounds[3] - height) / 2
                else:
                    height = source_bbox.bounds[3]
                if right_of:
                    x0 = source_bbox.x1 + 2 * thickness
                else:
                    x0 = source_bbox.x0 - 3 * thickness
                    if ticklocation == 'auto':
                        ticklocation = 'left'
                if offset:
                    x0 += inch_to_figure(figure, x=offset)
                rect = (x0, y0, thickness, height)
                orientation = 'vertical'
            elif below:
                thickness = inch_to_figure(figure, y=width or 0.05)
                x0 = source_bbox.x0
                if w:
                    ax_width = inch_to_figure(figure, x=w)
                    x0 += (source_bbox.bounds[2] - ax_width) / 2
                else:
                    ax_width = source_bbox.bounds[2]
                y0 = source_bbox.y0 - 3 * thickness
                if offset:
                    y0 += inch_to_figure(figure, y=offset)
                rect = (x0, y0, ax_width, thickness)
                orientation = 'horizontal'
            else:
                raise RuntimeError
            axes = figure.add_axes(rect)

        # prepare layout
        if axes:
            ax_aspect = 1
        elif orientation == 'horizontal':
            if h is None and w is None:
                h = 1
            ax_aspect = 4
        elif orientation == 'vertical':
            if h is None and w is None:
                h = 4
            ax_aspect = 0.3
        else:
            raise ValueError(f"{orientation=}")

        layout = Layout(1, ax_aspect, 2, tight, h=h, w=w, axes=axes, **kwargs)
        EelFigure.__init__(self, cm.name, layout)
        ax = self.axes[0]

        # translate between axes and data coordinates
        if isinstance(vmin, Normalize):
            norm = vmin
        else:
            vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cm)
            norm = Normalize(vmin, vmax)

        if isinstance(unit, AxisScale):
            scale = unit
        else:
            scale = AxisScale(unit or 1, label)

        # value ticks
        if isinstance(ticks, int):
            if ticks == 0:
                tick_locs = ()
            elif ticks == 1:
                raise ValueError(f'{ticks=}')
            else:
                tick_locs = np.linspace(vmin, vmax, ticks)
            formatter = scale.formatter
        elif isinstance(ticks, dict):
            tick_locs = sorted(ticks)
            formatter = FixedFormatter([ticks[t] for t in tick_locs])
        else:
            if ticks is None:
                tick_locs = MaxNLocator(4)
            else:
                tick_locs = ticks
            formatter = scale.formatter

        if orientation == 'horizontal':
            axis = ax.xaxis
            contour_func = ax.axhline
        else:
            axis = ax.yaxis
            contour_func = ax.axvline

        if label is True:
            label = scale.label or cm.name
        if not label:
            label = ''

        # show only part of the colorbar
        if clipmin is not None or clipmax is not None:
            boundaries = norm.inverse(np.linspace(0, 1, cm.N + 1))
            if clipmin is None:
                start = None
            else:
                start = np.digitize(clipmin, boundaries)
                boundaries[start] = clipmin
            if clipmax is None:
                stop = None
            else:
                stop = np.digitize(clipmax, boundaries) + 1
                boundaries[stop - 1] = clipmax
            boundaries = boundaries[start:stop]
        else:
            boundaries = None

        colorbar = ColorbarBase(ax, cmap=cm, norm=norm, boundaries=boundaries, orientation=orientation, ticklocation=ticklocation, ticks=tick_locs, label=label, format=formatter)

        # label position/rotation
        if label_position is not None:
            axis.set_label_position(label_position)
        if label_rotation is not None:
            axis.label.set_rotation(label_rotation)
            if orientation == 'vertical':
                if (label_rotation + 10) % 360 < 20:
                    axis.label.set_va('center')
        elif orientation == 'vertical' and len(label) <= 3:
            axis.label.set_rotation(0)
            axis.label.set_va('center')

        if background is not None:
            ax.set_facecolor(background)

        self._contours = [contour_func(c, c='k') for c in contours]
        self._draw_hooks.append(self.__update_bar_tickness)

        self._colorbar = colorbar
        self._orientation = orientation
        self._width = width
        self._show()

    def _tight(self):
        # make sure ticklabels have space
        ax = self.axes[0]
        if self._orientation == 'vertical' and not self._layout.w_fixed:
            self.draw()
            labels = ax.get_yticklabels()
            # wmax = max(l.get_window_extent().width for l in labels)
            x0 = min(l.get_window_extent().x0 for l in labels)
            if x0 < 0:
                w, h = self.figure.get_size_inches()
                w -= (x0 / self._layout.dpi)
                self.figure.set_size_inches(w, h, forward=True)
        super(ColorBar, self)._tight()

    def __update_bar_tickness(self):
        # Override to keep bar thickness
        if not self._width:
            return
        ax = self.axes[0]
        x = (self._width, self._width)
        x = self.figure.dpi_scale_trans.transform(x)
        x = self.figure.transFigure.inverted().transform(x)
        pos = ax.get_position()
        xmin, ymin, width, height = pos.xmin, pos.ymin, pos.width, pos.height
        if self._orientation == 'vertical':
            if self._layout._margins_arg and 'right' in self._layout._margins_arg:
                xmin += width - x[0]
            width = x[0]
        else:
            if self._layout._margins_arg and 'top' in self._layout._margins_arg:
                ymin += height - x[1]
            height = x[1]
        ax.set_position((xmin, ymin, width, height))
        return True


def adjust_hls(
        color: Any,
        hue: float = 0,
        lightness: float = 0,
        saturation: float = 0,
) -> Tuple[float, float, float, float]:
    *rgb, alpha = matplotlib.colors.to_rgba(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    if hue:
        h = (h + hue) % 1
    if lightness:
        l = max(0, min(1, l + lightness))
    if saturation:
        s = max(0, min(1, s + saturation))
    return *colorsys.hls_to_rgb(h, l, s), alpha
