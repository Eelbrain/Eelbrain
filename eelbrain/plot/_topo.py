# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot topographic maps of sensor space data."""
from __future__ import annotations

import dataclasses
from itertools import repeat
from math import floor, sqrt
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import matplotlib
import matplotlib.axes
import matplotlib.markers
import matplotlib.patches
import numpy as np
from scipy import interpolate, linalg
from scipy.spatial import ConvexHull

from .._colorspaces import UNAMBIGUOUS_COLORS
from .._data_obj import NDVarArg, CategorialArg, IndexArg, Dataset
from .._text import ms
from .._utils import deprecate_ds_arg
from ._base import (
    CMapArg, ColorArg,
    PlotType, EelFigure, PlotData, AxisData, DataLayer,
    Layout, ImLayout, VariableAspectLayout,
    ColorMapMixin, TimeSlicerEF, TopoMapKey, XAxisMixin, YLimMixin,
)
from ._utsnd import AxButterfly, AxImArray, PltIm
from ._sensors import SENSORMAP_FRAME, SensorMapMixin, PltMap2d


InterpolationArg = Literal[None, 'nearest', 'linear', 'spline']
SensorLabelsArg = Literal['', 'none', 'index', 'name', 'fullname']


class Topomap(SensorMapMixin, ColorMapMixin, TopoMapKey, EelFigure):
    """Plot individual topogeraphies

    Parameters
    ----------
    y
        Data to plot.
    xax
        Create a separate plot for each cell in this model.
    data
        If a Dataset is provided, data can be specified as strings.
    sub
        Specify a subset of the data.
    vmax
        Upper limits for the colormap (default is determined from data).
    vmin
        Lower limit for the colormap (default ``-vmax``).
    cmap
        Colormap (default depends on the data).
    contours
        Draw contours. Can be an int (number of contours, including
        ``vmin`` and ``vmax``), a sequence (values at which to draw
        contours), or a dictionary with ``**kwargs`` for
        :meth:`~matplotlib.axes.Axes.contour` (must include a ``"levels"`` key).
        Default is no contours.
    proj
        The sensor projection to use for topomaps (or one projection per plot).
    res
        Resolution of the topomaps (width = height = ``res``).
    interpolation
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    clip : bool | 'even' | 'circle'
        Outline for clipping topomaps: 'even' to clip at a constant distance
        (default), 'circle' to clip using a circle.
    clip_distance
        How far from sensor locations to clip (1 is the axes height/width).
    head_radius
        Radius of the head outline drawn over sensors (on sensor plots with
        normalized positions, 0.45 is the outline of the topomap); 0 to plot no
        outline; tuple for separate (right, anterior) radius.
        The default is determined automatically.
    head_pos
        Head outline position along the anterior axis (0 is the center, 0.5 is
        the top end of the plot).
    im_interpolation
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    sensors
        How to mark sensor locations in the topomap (empty string ``''`` to
        omit marks).
    sensorlabels
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    mark
        Sensors which to mark.
    mcolor
        Color for marked sensors (see :func:`matplotlib.pyplot.scatter`).
    msize
        Size of the markers (see :func:`matplotlib.pyplot.scatter`).
    marker
        Marker shape (see :func:`matplotlib.pyplot.scatter`).
    axtitle
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlabel
        Label below the topomaps (default is no label; ``True`` to use ``y``
        names).
    margins
        Layout parameter.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    Keys:
     - ``t``: open a ``Topomap`` plot for the region under the mouse pointer.
     - ``T``: open a larger ``Topomap`` plot with visible sensor names for the
       map under the mouse pointer.
    """
    @deprecate_ds_arg
    def __init__(
            self,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            xax: CategorialArg = None,
            data: Dataset = None,
            sub: IndexArg = None,
            vmax: float = None,
            vmin: float = None,
            cmap: CMapArg = None,
            contours: Union[int, Sequence, Dict] = None,
            # topomap args
            proj: str = 'default',
            res: int = None,
            interpolation: InterpolationArg = None,
            clip: Union[bool, str] = 'even',
            clip_distance: float = 0.05,
            head_radius: Union[float, Tuple[float, float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            im_interpolation: str = None,
            # sensor-map args
            sensors: Union[str, matplotlib.markers.MarkerStyle] = '.',
            sensorlabels: SensorLabelsArg = None,
            mark: IndexArg = None,
            mcolor: Union[ColorArg, Sequence[ColorArg]] = None,
            msize: Union[float, Sequence[float]] = 20,
            marker: Union[str, matplotlib.markers.MarkerStyle] = 'o',
            # layout
            axtitle: Union[bool, Sequence[str]] = True,
            xlabel: Union[bool, str] = None,
            margins: Dict[str, float] = None,
            **kwargs,
    ):
        plot_data = PlotData.from_args(y, ('sensor',), xax, data, sub)
        self.plots = []
        ColorMapMixin.__init__(self, plot_data.data, cmap, vmax, vmin, contours, self.plots)
        if isinstance(proj, str):
            proj = repeat(proj, plot_data.n_plots)
        elif not isinstance(proj, Sequence):
            raise TypeError(f"proj={proj!r}")
        elif len(proj) != plot_data.n_plots:
            raise ValueError(f"proj={proj!r}: need as many proj as axes ({plot_data.n_plots})")

        layout = ImLayout(plot_data.plot_used, 1.1, 2, margins, axtitle=axtitle, **kwargs)
        EelFigure.__init__(self, plot_data.frame_title, layout)
        self._set_axtitle(axtitle, plot_data, verticalalignment='top', pad=-1)

        # plots
        axes_data = plot_data.for_plot(PlotType.IMAGE)
        for ax, layers, proj_ in zip(self.axes, axes_data, proj):
            h = AxTopomap(ax, layers, clip, clip_distance, sensors, sensorlabels, mark, mcolor, msize, marker, proj_, res, im_interpolation, xlabel, self._vlims, self._cmaps, self._contours, interpolation, head_radius, head_pos)
            self.plots.append(h)

        TopoMapKey.__init__(self, self._topo_data)
        SensorMapMixin.__init__(self, [h.sensors for h in self.plots])
        self._show()

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)
        SensorMapMixin._fill_toolbar(self, tb)

    def _topo_data(self, event):
        if event.inaxes:
            ax_i = self.axes.index(event.inaxes)
            p = self.plots[ax_i]
            return p.data, p.title, p.proj


class TopomapBins(SensorMapMixin, ColorMapMixin, TopoMapKey, EelFigure):
    """Topomaps in time-bins

    Parameters
    ----------
    y
        Data to plot.
    xax
        Create a separate plot for each cell in this model.
    data
        If a Dataset is provided, data can be specified as strings.
    sub
        Specify a subset of the data.
    bin_length
        Length ofthe time bins for topo-plots.
    tstart
        Beginning of the first time bin (default is the beginning of ``y``).
    tstop
        End of the last time bin (default is the end of ``y``).
    vmax
        Upper limits for the colormap (default is determined from data).
    vmin
        Lower limit for the colormap (default ``-vmax``).
    cmap
        Colormap (default depends on the data).
    contours
        Draw contours. Can be an int (number of contours, including
        ``vmin`` and ``vmax``), a sequence (values at which to draw
        contours), or a dictionary with ``**kwargs`` for
        :meth:`~matplotlib.axes.Axes.contour` (must include a ``"levels"`` key).
        Default is no contours.
    proj
        The sensor projection to use for topomaps.
    res
        Resolution of the topomaps (width = height = ``res``).
    interpolation
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    clip : bool | 'even' | 'circle'
        Outline for clipping topomaps: 'even' to clip at a constant distance
        (default), 'circle' to clip using a circle.
    clip_distance
        How far from sensor locations to clip (1 is the axes height/width).
    head_radius
        Radius of the head outline drawn over sensors (on sensor plots with
        normalized positions, 0.45 is the outline of the topomap); 0 to plot no
        outline; tuple for separate (right, anterior) radius.
        The default is determined automatically.
    head_pos
        Head outline position along the anterior axis (0 is the center, 0.5 is
        the top end of the plot).
    im_interpolation
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    sensors
        How to mark sensor locations in the topomap (empty string ``''`` to
        omit marks).
    sensorlabels
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    mark : Sensor index
        Sensors which to mark.
    mcolor
        Color for marked sensors (see :func:`matplotlib.pyplot.scatter`).
    msize
        Size of the markers (see :func:`matplotlib.pyplot.scatter`).
    marker
        Marker shape (see :func:`matplotlib.pyplot.scatter`).
    axtitle
        Title for the individual axes. The default is the time bin center.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    Keys:
     - ``t``: open a ``Topomap`` plot for the map under the mouse pointer.
     - ``T``: open a larger ``Topomap`` plot with visible sensor names for the
       map under the mouse pointer.
    """
    @deprecate_ds_arg
    def __init__(
            self,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            xax: CategorialArg = None,
            data: Dataset = None,
            sub: IndexArg = None,
            bin_length: float = 0.050,
            tstart: float = None,
            tstop: float = None,
            vmax: float = None,
            vmin: float = None,
            cmap: CMapArg = None,
            contours: Union[int, Sequence, Dict] = None,
            # topomap args
            proj: str = 'default',
            res: int = None,
            interpolation: InterpolationArg = None,
            clip: Union[bool, str] = 'even',
            clip_distance: float = 0.05,
            head_radius: Union[float, Tuple[float, float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            im_interpolation: str = None,
            # sensor-map args
            sensors: Union[str, matplotlib.markers.MarkerStyle] = '.',
            sensorlabels: SensorLabelsArg = None,
            mark: IndexArg = None,
            mcolor: Union[ColorArg, Sequence[ColorArg]] = None,
            msize: Union[float, Sequence[float]] = 20,
            marker: Union[str, matplotlib.markers.MarkerStyle] = 'o',
            # layout
            axtitle: Union[bool, Sequence[str]] = True,
            **kwargs,
    ):
        plot_data = PlotData.from_args(y, ('sensor', 'time'), xax, data, sub)
        self._plots = []
        plot_data._cannot_skip_axes(self)
        bin_data = plot_data.for_plot(PlotType.IMAGE).bin(bin_length, tstart, tstop)
        ColorMapMixin.__init__(self, plot_data.data, cmap, vmax, vmin, contours, self._plots)

        # create figure
        time = bin_data.y0.get_dim('time')
        n_bins = len(time)
        n_rows = bin_data.n_plots
        kwargs.setdefault('tight', False)
        layout = Layout(n_bins * n_rows, 1, 1.5, rows=n_rows, columns=n_bins, **kwargs)
        EelFigure.__init__(self, plot_data.frame_title, layout)
        self._plots.extend(repeat(None, n_bins * n_rows))

        for column, t in enumerate(time):
            t_data = bin_data.sub_time(t)
            for row, layers in enumerate(t_data):
                i = row * n_bins + column
                ax = self.axes[i]
                self._plots[i] = AxTopomap(ax, layers, clip, clip_distance, sensors, sensorlabels, mark, mcolor, msize, marker, proj, res, im_interpolation, None, self._vlims, self._cmaps, self._contours, interpolation, head_radius, head_pos)

        # Time labels
        if axtitle is True:
            fmt, _, _ = time._axis_format(True, True)
            self._set_axtitle((fmt(t) for t in time), axes=self.axes[:len(time)])
        elif axtitle:
            self._set_axtitle(axtitle)
        TopoMapKey.__init__(self, self._topo_data)
        SensorMapMixin.__init__(self, [h.sensors for h in self._plots])
        self._show()

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)
        SensorMapMixin._fill_toolbar(self, tb)

    def _topo_data(self, event):
        if event.inaxes:
            ax_i = self.axes.index(event.inaxes)
            p = self._plots[ax_i]
            return p.data, p.title, p.proj


class TopoButterfly(ColorMapMixin, TimeSlicerEF, TopoMapKey, YLimMixin, XAxisMixin, EelFigure):
    """Butterfly plot with corresponding topomaps

    Parameters
    ----------
    y : (list of) NDVar
        Data to plot.
    xax
        Create a separate plot for each cell in this model.
    data
        If a Dataset is provided, data can be specified as strings.
    sub
        Specify a subset of the data.
    vmax
        Upper limits for the colormap (default is determined from data).
    vmin
        Lower limit for the colormap (default ``-vmax``).
    cmap
        Colormap (default depends on the data).
    contours
        Draw contours. Can be an int (number of contours, including
        ``vmin`` and ``vmax``), a sequence (values at which to draw
        contours), or a dictionary with ``**kwargs`` for
        :meth:`~matplotlib.axes.Axes.contour` (must include a ``"levels"`` key).
        Default is no contours.
    color : matplotlib color
        Color of the butterfly plots.
    linewidth : scalar
        Linewidth for plots (defult is to use ``matplotlib.rcParams``).
    t
        Time to display in the topomap.
    proj
        The sensor projection to use for topomaps.
    res
        Resolution of the topomaps (width = height = ``res``).
    interpolation
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    clip : bool | 'even' | 'circle'
        Outline for clipping topomaps: 'even' to clip at a constant distance
        (default), 'circle' to clip using a circle.
    clip_distance : scalar
        How far from sensor locations to clip (1 is the axes height/width).
    head_radius : scalar | tuple
        Radius of the head outline drawn over sensors (on sensor plots with
        normalized positions, 0.45 is the outline of the topomap); 0 to plot no
        outline; tuple for separate (right, anterior) radius.
        The default is determined automatically.
    head_pos : scalar
        Head outline position along the anterior axis (0 is the center, 0.5 is
        the top end of the plot).
    im_interpolation : str
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    sensors
        How to mark sensor locations in the topomap (empty string ``''`` to
        omit marks).
    sensorlabels
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    mark : Sensor index
        Sensors to mark in the topo-map. To highlight sensors in the butterfly
        plot, consider using :meth:`NDVar.mask` on ``y``.
    mcolor : matplotlib color
        Color for marked sensors.
    xlabel
        X-axis label. By default the label is inferred from the data.
    ylabel
        Y-axis label. By default the label is inferred from the data.
    xticklabels
        Specify which axes should be annotated with x-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    yticklabels
        Specify which axes should be annotated with y-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlim : scalar | (scalar, scalar)
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    Topomap control:
     - LMB click in a butterfly plot fixates the topomap time
     - RMB click in a butterfly plot removes the time point, the topomaps
       follow the mouse pointer
     - ``.``: Increment the current topomap time (got right)
     - ``,``: Decrement the current topomap time (go left)
     - ``t``: open a ``Topomap`` plot for the time point under the mouse
       pointer
     - ``T``: open a larger ``Topomap`` plot with visible sensor names for the
       time point under the mouse pointer

    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: x-axis zoom in (reduce x axis range)
     - ``d``: x-axis zoom out (increase x axis range)
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    _default_xlabel_ax = -2

    @deprecate_ds_arg
    def __init__(
            self,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            xax: CategorialArg = None,
            data: Dataset = None,
            sub: IndexArg = None,
            vmax: float = None,
            vmin: float = None,
            cmap: CMapArg = None,
            contours: Union[int, Sequence, Dict] = None,
            color: Any = None,
            linewidth: float = None,
            # topomap args
            t: float = None,
            proj: str = 'default',
            res: int = None,
            interpolation: InterpolationArg = None,
            clip: Union[bool, str] = 'even',
            clip_distance: float = 0.05,
            head_radius: Union[float, Tuple[float, float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            im_interpolation: str = None,
            # sensor-map args
            sensors: Union[str, matplotlib.markers.MarkerStyle] = '.',
            sensorlabels: SensorLabelsArg = None,
            mark: IndexArg = None,
            mcolor: ColorArg = None,
            # layout
            xlabel: Union[bool, str] = True,
            ylabel: Union[bool, str] = True,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            axtitle: Union[bool, Sequence[str]] = True,
            frame: bool = True,
            xlim: Union[float, Tuple[float, float]] = None,
            **kwargs,
    ):
        plot_data = PlotData.from_args(y, ('sensor', None), xax, data, sub)
        plot_data._cannot_skip_axes(self)
        xdim = plot_data.dims[1]
        self._topomap_data = plot_data.for_plot(PlotType.IMAGE)

        # create figure
        row_titles = self._set_axtitle(axtitle, plot_data, plot_data.n_plots)
        layout = VariableAspectLayout(plot_data.n_plots, 3, 10, aspect=(None, 1), ax_frames=(frame, False), row_titles=row_titles, **kwargs)
        EelFigure.__init__(self, plot_data.frame_title, layout)

        self.bfly_axes = self.axes[0::2]
        self.topo_axes = self.axes[1::2]
        self.bfly_plots = []
        self.topo_plots = []
        self.t_markers = []  # vertical lines on butterfly plots

        ColorMapMixin.__init__(self, plot_data.data, cmap, vmax, vmin, contours, self.topo_plots)

        self._topo_kwargs = {
            'clip': clip,
            'clip_distance': clip_distance,
            'head_radius': head_radius,
            'head_pos': head_pos,
            'proj': proj,
            'contours': self._contours,
            'res': res,
            'interpolation': interpolation,
            'im_interpolation': im_interpolation,
            'sensors': sensors,
            'sensorlabels': sensorlabels,
            'mark': mark,
            'mcolor': mcolor,
        }

        # plot epochs (x/y are in figure coordinates)
        for ax, layers in zip(self.bfly_axes, plot_data.for_plot(PlotType.LINE)):
            h = AxButterfly(ax, layers, 'time', 'sensor', mark, color, linewidth, self._vlims, clip)
            self.bfly_plots.append(h)

        # decorate axes
        self._configure_axis_dim('x', plot_data.time_dim, xlabel, xticklabels, self.bfly_axes)
        self._configure_axis_data('y', plot_data, ylabel, yticklabels, self.bfly_axes)

        # setup callback
        XAxisMixin._init_with_data(self, plot_data.data, xdim, xlim, self.bfly_axes)
        YLimMixin.__init__(self, self.bfly_plots + self.topo_plots)
        TimeSlicerEF.__init__(self, xdim, plot_data.time_dim, self.bfly_axes, False, initial_time=t)
        TopoMapKey.__init__(self, self._topo_data)
        self._t_label = None  # time label under lowest topo-map
        self._frame .store_canvas()

        self._show(crosshair_axes=self.bfly_axes)
        self._init_controller()

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)

    def _update_topo(self, t):
        if not self.topo_plots:
            data = self._topomap_data.sub_time(t)
            for ax, layers in zip(self.topo_axes, data):
                p = AxTopomap(ax, layers, cmaps=self._cmaps, vlims=self._vlims, **self._topo_kwargs)
                self.topo_plots.append(p)
        else:
            data = self._topomap_data.sub_time(t, data_only=True)
            for p, layers in zip(self.topo_plots, data):
                p.set_data(layers)

    def _topo_data(self, event):
        ax = event.inaxes
        if ax is None:
            return
        p = self.bfly_plots[ax.id // 2]
        if ax in self.bfly_axes:
            t = event.xdata
        elif ax in self.topo_axes:
            t = self._current_time
        else:
            return
        seg = [l.sub(time=t) for l in p.data]
        return seg, f"{ms(t)} ms", self._topo_kwargs['proj']

    def _on_leave_axes_status_text(self, event):
        return "Topomap: t = %.3f" % self._current_time

    def _update_time(self, t, fixate):
        TimeSlicerEF._update_time(self, t, fixate)
        self._update_topo(t)
        if fixate:
            # add time label
            text = "t = %i ms" % round(t * 1e3)
            if self._t_label:
                self._t_label.set_text(text)
            else:
                ax = self.topo_axes[-1]
                self._t_label = ax.text(.5, -0.1, text, ha='center', va='top')
            self.canvas.draw()  # otherwise time label does not get redrawn
        elif self._time_fixed:
            self._t_label.remove()
            self._t_label = None
            self.canvas.draw()  # otherwise time label does not get redrawn
        elif hasattr(self.canvas, 'redraw'):
            self.canvas.redraw(self.topo_axes)


class PltTopomap(PltIm):
    _aspect = 'equal'

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            layer: DataLayer,
            proj: str,
            res: int,
            im_interpolation: str,
            vlims,
            cmaps,
            contours,
            interpolation: InterpolationArg,  # Method for interpolating topo-map between sensors
            clip: str,
            clip_distance: float,
    ):
        # store attributes
        self._proj = proj
        self._visible_data = layer.y.sensor._visible_sensors(proj)
        self._grid = np.linspace(0, 1, res)
        self._mgrid = tuple(np.meshgrid(self._grid, self._grid))
        if interpolation is None and layer.y.x.dtype.kind in 'biu':
            interpolation = 'nearest'
        self._method = interpolation

        # clip mask
        if clip:
            locs = layer.y.sensor.get_locs_2d(self._proj, frame=SENSORMAP_FRAME)
            if clip == 'even':
                hull = ConvexHull(locs)
                points = locs[hull.vertices]
                default_head_radius = sqrt(np.min(np.sum((points - [0.5, 0.5]) ** 2, 1)))
                # find offset due to clip_distance
                tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
                verticals = np.dot(tangents, [[0, -1], [1, 0]])
                verticals /= np.sqrt(np.sum(verticals ** 2, 1)[:, None])
                verticals *= clip_distance
                # apply offset
                points += verticals
                mask = matplotlib.patches.Polygon(points, transform=ax.transData)
            elif clip == 'circle':
                clip_radius = sqrt(np.max(np.sum((locs - [0.5, 0.5]) ** 2, 1)))
                mask = matplotlib.patches.Circle((0.5, 0.5), clip_radius, transform=ax.transData)
                default_head_radius = clip_radius
            else:
                raise ValueError('clip=%r' % (clip,))
        else:
            mask = None
            default_head_radius = None

        self._default_head_radius = default_head_radius
        PltIm.__init__(self, ax, layer, cmaps, vlims, contours, (0, 1, 0, 1), im_interpolation, mask)

    def _data_from_ndvar(self, ndvar):
        v = ndvar.get_data(('sensor',))
        locs = ndvar.sensor.get_locs_2d(self._proj, frame=SENSORMAP_FRAME)
        if self._visible_data is not None:
            v = v[self._visible_data]
            locs = locs[self._visible_data]

        if self._method is None:
            # interpolate data
            xi, yi = self._mgrid

            # code adapted from mne-python topmap _griddata()
            xy = locs[:, 0] + locs[:, 1] * -1j
            d = np.abs(xy - xy[:, None])
            diagonal_step = len(locs) + 1
            d.flat[::diagonal_step] = 1.

            g = (d * d) * (np.log(d) - 1.)
            g.flat[::diagonal_step] = 0.
            try:
                weights = linalg.solve(g, v.ravel())
            except ValueError:
                if np.isnan(v).any():
                    raise NotImplementedError("Can't interpolate sensor data with NaN")
                unique_locs = np.unique(locs, axis=0)
                if len(unique_locs) < len(locs):
                    raise NotImplementedError("Error determining sensor map projection due to more than one sensor in a single location; try using a different projection.")
                raise

            m, n = xi.shape
            out = np.empty_like(xi)

            g = np.empty(xy.shape)
            for i in range(m):
                for j in range(n):
                    d = np.abs(xi[i, j] + -1j * yi[i, j] - xy)
                    mask = np.where(d == 0)[0]
                    if len(mask):
                        d[mask] = 1.
                    np.log(d, out=g)
                    g -= 1.
                    g *= d * d
                    if len(mask):
                        g[mask] = 0.
                    out[i, j] = g.dot(weights)
            return out
        elif self._method == 'spline':
            k = int(floor(sqrt(len(locs)))) - 1
            tck = interpolate.bisplrep(locs[:, 1], locs[:, 0], v, kx=k, ky=k)
            return interpolate.bisplev(self._grid, self._grid, tck)
        else:
            isnan = np.isnan(v)
            if np.any(isnan):
                nanmap = interpolate.griddata(locs, isnan, self._mgrid, self._method)
                mask = nanmap > 0.5
                v = np.where(isnan, 0, v)
                vmap = interpolate.griddata(locs, v, self._mgrid, self._method)
                np.place(vmap, mask, np.nan)
                return vmap
            return interpolate.griddata(locs, v, self._mgrid, self._method)


class AxTopomap(AxImArray):
    """Axes with a topomap

    Parameters
    ----------
    mark : list of IDs
        highlight a subset of the sensors
    """
    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            layers: AxisData,
            clip: str = 'even',  # even or circle (only applies if interpolation is None)
            clip_distance: float = 0.05,  # distance from outermost sensor for clip=='even'
            sensors: Union[str, matplotlib.markers.MarkerStyle] = '.',
            sensorlabels: SensorLabelsArg = None,
            mark: IndexArg = None,
            mcolor: Union[ColorArg, Sequence[ColorArg]] = None,
            msize: Union[float, Sequence[float]] = 20,
            mmarker: Union[str, matplotlib.markers.MarkerStyle] = 'o',
            proj: str = 'default',  # topomap projection method
            res: int = None,  # topomap image resolution
            im_interpolation: str = None,  # matplotlib imshow interpolation method
            xlabel: Union[bool, str] = None,
            vlims: dict = {},
            cmaps: dict = {},
            contours: dict = {},
            interpolation: InterpolationArg = None,  # topomap interpolation method
            head_radius: Union[float, Sequence[float]] = None,
            head_pos: Union[float, Tuple[float, float]] = 0.,
            head_linewidth: float = None,
    ):
        self.ax = ax
        self.data = layers  # will not update from .set_data()
        self.proj = proj
        sensor_dim = layers.y0.sensor

        if xlabel is True:
            xlabel = layers.y0.name
        if im_interpolation is None:
            im_interpolation = 'bilinear'
        if res is None:
            res = 64 if interpolation is None else 100

        ax.set_axis_off()
        self.plots = [PltTopomap(ax, layer, proj, res, im_interpolation, vlims, cmaps, contours, interpolation, clip, clip_distance) for layer in layers]

        # head outline
        if head_radius is None and clip == 'circle' and interpolation is None and sensor_dim._topomap_outlines(proj) == 'top':
            head_radius = self.plots[0]._default_head_radius

        # plot sensors
        self.sensors = PltMap2d(ax, sensor_dim, proj, 1, sensors, 1, 'k', mark, mcolor, msize, mmarker, sensorlabels, False, head_radius, head_pos, head_linewidth)

        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if isinstance(xlabel, str):
            x, y = ax.transData.inverted().transform(ax.transAxes.transform((0.5, 0)))
            ax.text(x, y, xlabel, ha='center', va='top')

    def set_ylim(self, bottom, top):  # Alias for YLimMixin
        self.set_vlim(bottom, top)


@dataclasses.dataclass
class TopoWindow:
    """Helper class for TopoArray.

    Maintains a topomap corresponding to one segment with flexible time point.
    """
    ax: matplotlib.axes.Axes  # topomap-axes
    parent: AxImArray  # array-plot
    topomap_args: dict
    connectionstyle: str = "angle3,angleA=90,angleB=0"
    label_position: Literal['above', 'below', 'none'] = 'above'
    color: ColorArg = UNAMBIGUOUS_COLORS['bluish green']
    annotation_xy: Tuple[float, float] = (0.5, 1.05)
    # internal plot handles
    t_line = None
    pointer = None
    text_pointer = None
    plot = None
    t = None

    def update(self, t):
        if t is not None:
            if self.t_line:
                self.t_line.remove()
            self.t_line = self.parent.ax.axvline(t, c=self.color)

            t_str = f"{ms(t)} ms"
            if self.pointer:
                self.pointer.axes = self.parent.ax
                self.pointer.xy = (t, 1)
                if self.text_pointer:
                    self.text_pointer.set_text(t_str)
                self.pointer.set_visible(True)
            else:
                text = t_str if self.label_position == 'above' else ''
                arrowprops = {'arrowstyle': '-', 'shrinkB': 0, 'color': self.color}
                if self.connectionstyle:
                    arrowprops['connectionstyle'] = self.connectionstyle
                self.pointer = self.parent.ax.annotate(text, (t, 0), xycoords='data', xytext=self.annotation_xy, textcoords=self.ax.transData, horizontalalignment='center', verticalalignment='center', arrowprops=arrowprops, zorder=4)
                if self.label_position == 'above':
                    self.text_pointer = self.pointer
                elif self.label_position == 'below':
                    self.text_pointer = self.ax.text(0.5, 0, t_str, va='top', ha='center', transform=self.ax.transAxes)

            if self.plot is None:
                layers = self.parent.data.sub_time(t)
                self.plot = AxTopomap(self.ax, layers, **self.topomap_args)
            else:
                layers = self.parent.data.sub_time(t, data_only=True)
                self.plot.set_data(layers)
            self.t = t

    def clear(self):
        self.ax.cla()
        self.ax.set_axis_off()
        self.plot = None
        self.t = None
        if self.t_line:
            self.t_line.remove()
            self.t_line = None
        if self.pointer:
            self.pointer.remove()
            self.pointer = None

    def add_contour(self, meas, level, color):
        if self.plot:
            self.plot.add_contour(meas, level, color)

    def set_cmap(self, cmap, meas):
        if self.plot:
            self.plot.set_cmap(cmap, meas)

    def set_vlim(self, v, vmax=None, meas=None):
        if self.plot:
            self.plot.set_vlim(v, vmax, meas)


class TopoArray(ColorMapMixin, TopoMapKey, XAxisMixin, EelFigure):
    """Channel by sample plots with topomaps for individual time points

    Parameters
    ----------
    y
        Data to plot.
    xax
        Create a separate plot for each cell in this model.
    data
        If a Dataset is provided, data can be specified as strings.
    sub
        Specify a subset of the data.
    vmax
        Upper limits for the colormap (default is determined from data).
    vmin
        Lower limit for the colormap (default ``-vmax``).
    cmap
        Colormap (default depends on the data).
    contours
        Draw contours. Can be an int (number of contours, including
        ``vmin`` and ``vmax``), a sequence (values at which to draw
        contours), or a dictionary with ``**kwargs`` for
        :meth:`~matplotlib.axes.Axes.contour` (must include a ``"levels"`` key).
        Default is no contours.
    ntopo
        number of topomaps per array-plot.
    t
        Time points for topomaps.
    xlim : scalar | (scalar, scalar)
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    proj
        The sensor projection to use for topomaps.
    res
        Resolution of the topomaps (width = height = ``res``).
    interpolation
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    clip : bool | 'even' | 'circle'
        Outline for clipping topomaps: 'even' to clip at a constant distance
        (default), 'circle' to clip using a circle.
    clip_distance
        How far from sensor locations to clip (1 is the axes height/width).
    head_radius
        Radius of the head outline drawn over sensors (on sensor plots with
        normalized positions, 0.45 is the outline of the topomap); 0 to plot no
        outline; tuple for separate (right, anterior) radius.
        The default is determined automatically.
    head_pos
        Head outline position along the anterior axis (0 is the center, 0.5 is
        the top end of the plot).
    im_interpolation
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    sensors
        How to mark sensor locations in the topomap (empty string ``''`` to
        omit marks).
    sensorlabels
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown. Set to ``''`` to
        hide sensor position markers completely.
    mark : Sensor index
        Sensors which to mark.
    mcolor : matplotlib color
        Color for marked sensors.
    axtitle
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlabel
        X-axis label. By default the label is inferred from the data.
    ylabel
        Y-axis label. By default the label is inferred from the data.
    xticklabels
        Specify which axes should be annotated with x-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    yticklabels
        Specify which axes should be annotated with y-axis tick labels.
        Use ``int`` for a single axis, a sequence of ``int`` for multiple
        specific axes, or one of ``'left' | 'bottom' | 'all' | 'none'``.
    connectionstyle
        Style for the connections between the image array-plot and the
        topo-maps. Set to ``''`` for straight connections. See
        `Matplotlib demo <https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html>`_.
    connection_color
        Color for connection line.
    topo_labels
        Where to label time on topo-maps.
    gridspec_kw
        The figure attempts to produce a useful layout, but sometimes this will
        still result in unwanted artifacts like overlapping text; use gridspec
        parameters to customize the spacing (see
        :class:~matplotlib.gridspec.GridSpec).
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
     - LMB click on a topomap selects it for tracking the mouse pointer
         - LMB on the array plot fixates the topomap time point
     - RMB on a topomap removes the topomap

    """
    _make_axes = False

    @deprecate_ds_arg
    def __init__(
            self,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            xax: CategorialArg = None,
            data: Dataset = None,
            sub: IndexArg = None,
            vmax: float = None,
            vmin: float = None,
            cmap: CMapArg = None,
            contours: Union[int, Sequence, Dict] = None,
            ntopo: int = None,
            t: Sequence[float] = (),
            xlim: Union[float, Tuple[float, float]] = None,
            # topomap args
            proj: str = 'default',
            res: int = None,
            interpolation: InterpolationArg = None,
            clip: Union[bool, str] = 'even',
            clip_distance: float = 0.05,
            head_radius: Union[float, Tuple[float, float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            im_interpolation: str = None,
            # sensor-map args
            sensors: Union[str, matplotlib.markers.MarkerStyle] = '.',
            sensorlabels: SensorLabelsArg = None,
            mark: IndexArg = None,
            mcolor: ColorArg = None,
            # layout
            axtitle: Union[bool, Sequence[str]] = True,
            xlabel: Union[bool, str] = True,
            ylabel: Union[bool, str] = True,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            connectionstyle: str = "angle3,angleA=90,angleB=0",
            connection_color: ColorArg = UNAMBIGUOUS_COLORS['bluish green'],
            topo_labels: Literal['above', 'below', 'none'] = 'above',
            gridspec_kw: dict = None,
            **kwargs,
    ):
        if ntopo is None:
            ntopo = len(t) if t else 3

        plot_data = PlotData.from_args(y, ('time', 'sensor'), xax, data, sub).for_plot(PlotType.IMAGE)

        # create figure
        if 'columns' not in kwargs and 'rows' not in kwargs:
            kwargs['rows'] = 1
        layout = Layout(plot_data.plot_used, 1.5, 3, tight=False, **kwargs)
        EelFigure.__init__(self, plot_data.frame_title, layout)
        all_plots = []
        ColorMapMixin.__init__(self, plot_data.data, cmap, vmax, vmin, contours, all_plots)
        TopoMapKey.__init__(self, self._topo_data)

        # save important properties
        self._data = plot_data
        self._ntopo = ntopo
        self._default_xlabel_ax = -1 - ntopo
        self._proj = proj

        # prepare axes
        if layout.user_axes:
            self.axes = layout.user_axes
        else:
            x_frame_l = .6 / layout.axw / plot_data.n_plots
            x_frame_r = .025 / plot_data.n_plots
            kw = dict(left=x_frame_l, right=1 - x_frame_r, bottom=.05, top=.9, wspace=.1, hspace=.3)
            if gridspec_kw:
                kw.update(gridspec_kw)
            gs = self.figure.add_gridspec(layout.rows * 2, layout.columns * ntopo, **kw)
            if layout.rows == 1:
                for col, used in enumerate(plot_data.plot_used):
                    if not used:
                        continue
                    self.figure.add_subplot(gs[0, col * ntopo:(col + 1) * ntopo], picker=True)
                    for j in range(ntopo):
                        self.figure.add_subplot(gs[1, col * ntopo + j], picker=True, xticks=[], yticks=[])
            elif layout.columns == 1:
                for row, used in enumerate(plot_data.plot_used):
                    if not used:
                        continue
                    self.figure.add_subplot(gs[row * 2, 0:ntopo], picker=True)
                    for j in range(ntopo):
                        self.figure.add_subplot(gs[row * 2 + 1, j], picker=True, xticks=[], yticks=[])
            else:
                raise ValueError("Layout with multiple columns and rows; set either columns=1 or rows=1")
            self.axes = self.figure.axes

        # im_array plots
        self._array_axes = []
        self._array_plots = []
        self._topo_windows = []
        topomap_args = dict(clip=clip, clip_distance=clip_distance, sensors=sensors, sensorlabels=sensorlabels, mark=mark, mcolor=mcolor, proj=proj, res=res, im_interpolation=im_interpolation, vlims=self._vlims, cmaps=self._cmaps, contours=self._contours, interpolation=interpolation, head_radius=head_radius, head_pos=head_pos)
        for i, layers in enumerate(plot_data):
            ax_i = i * (ntopo + 1)
            ax = self.axes[ax_i]
            ax.ID = i
            ax.type = 'main'
            im_plot = AxImArray(ax, layers, 'time', im_interpolation, self._vlims, self._cmaps, self._contours)
            self._array_axes.append(ax)
            self._array_plots.append(im_plot)
            if i > 0:
                ax.yaxis.set_visible(False)

            # topo plots
            for j in range(ntopo):
                ax = self.axes[ax_i + 1 + j]
                ax.ID = i * ntopo + j
                ax.type = 'window'
                win = TopoWindow(ax, im_plot, topomap_args, connectionstyle, topo_labels, connection_color)
                self.axes.append(ax)
                self._topo_windows.append(win)
        all_plots.extend(self._array_plots)
        all_plots.extend(self._topo_windows)

        # if t argument is provided, set topo-map time points
        if t:
            if np.isscalar(t):
                t = [t]
            self.set_topo_ts(*t)

        self._set_axtitle(axtitle, plot_data, self._array_axes)
        self._configure_axis_dim('x', plot_data.y0.time, xlabel, xticklabels, self._array_axes)
        self._configure_axis_dim('y', 'sensor', ylabel, yticklabels, self._array_axes, False, plot_data.data)

        # setup callback
        XAxisMixin._init_with_data(self, plot_data.data, 'time', xlim, self._array_axes)
        self._selected_window = None
        self.canvas.mpl_connect('pick_event', self._pick_handler)
        self._frame .store_canvas()
        self._show(crosshair_axes=self._array_axes)

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)

    def _topo_data(self, event):
        ax = event.inaxes
        if ax in self._array_axes:
            t = event.xdata
            data = self._array_plots[ax.ID].data.sub_time(t)
        else:
            topo_window = self._topo_windows[ax.ID]
            t = topo_window.t
            if t is None:
                return
            data = topo_window.plot.data
        return data, f"{ms} ms", self._proj

    def _iter_plots(self):
        "Iterate through non-empty plots"
        yield from self._array_plots
        for w in self._topo_windows:
            if w.plot is not None:
                yield w.plot

    def set_cmap(self, cmap, meas=None):
        """Change the colormap

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        self._cmaps[meas] = cmap
        for p in self._iter_plots():
            p.set_cmap(cmap, meas)
        self.draw()

    def set_topo_t_single(self, topo_id, t):
        """
        Set the time for a single topomap.

        Parameters
        ----------
        topo_id : int
            Index of the topomap (numbered throughout the figure).
        t : scalar or ``None``
            time point; ``None`` clears the topomap
        """
        # get window ax
        w = self._topo_windows[topo_id]
        w.clear()

        if t is not None:
            w.update(t)

        self.canvas.draw()

    def set_topo_t(self, topo_id, t):
        """
        Set the time point for a topo-map (same for all array plots)

        Parameters
        ----------
        topo_id : int
            Index of the topomap (numberd for each array-plot).
        t : scalar or ``None``
            time point; ``None`` clears the topomap

        See Also
        --------
        .set_topo_ts : set several topomap time points at once
        .set_topo_t_single : set the time point of a single topomap
        """
        for i in range(len(self._array_plots)):
            _topo = self._ntopo * i + topo_id
            self.set_topo_t_single(_topo, t)

    def set_topo_ts(self, *t_list):
        """Set the time points displayed in topo-maps across all array-plots"""
        for i, t in enumerate(t_list):
            self.set_topo_t(i, t)

    def _pick_handler(self, pickevent):
        mouseevent = pickevent.mouseevent
        ax = pickevent.artist
        if ax.type == 'window':
            button = mouseevent.button  # 1: Left
            window = self._topo_windows[ax.ID]
            if button == 1:
                self._selected_window = window
            elif button in (2, 3):
                ax_id = window.ax.ID % self._ntopo
                self.set_topo_t(ax_id, None)
            else:
                pass
        elif (ax.type == 'main') and (self._selected_window is not None):
            self._selected_window.clear()  # to side track pdf export transparency issue
            # update corresponding topo_windows
            t = mouseevent.xdata
            ax_id = self._selected_window.ax.ID % self._ntopo
            self.set_topo_t(ax_id, t)

            self._selected_window = None
            self.canvas.draw()

    def _on_motion_sub(self, event):
        if (self._selected_window is not None
                and event.inaxes
                and event.inaxes.type == 'main'
                and event.xdata in self._data.plot_data[event.inaxes.ID].y0.time):
            self._selected_window.update(event.xdata)
            return {self._selected_window.ax}
        return set()
