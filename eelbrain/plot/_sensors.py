# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot sensor maps."""
from math import sin, cos, asin
import os
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import matplotlib as mpl
import matplotlib.axes
import matplotlib.markers
from matplotlib.lines import Line2D

from .._data_obj import Datalist, Sensor, as_sensor
from ._base import EelFigure, ImLayout, ColorArg

SENSORMAP_FRAME = 0.1

# some useful kwarg dictionaries for different plot layouts
kwargs_mono = dict(mc='k',
                   lc='.5',
                   hllc='k',
                   hlmc='k',
                   hlms=7,
                   strlc='k')


def _head_outlines(
        head_radius: Union[float, Sequence[float]] = None,
        head_pos: Union[float, Sequence[float]] = 0,
):
    # generate outlines for center 0, radius 1
    nose_alpha = 0.2
    l = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(l)
    head_y = np.sin(l)
    w = sin(nose_alpha)
    nose_x = np.array((-w, -w * 0.5, -w * 0.2, 0, w * 0.2, w * 0.5, w))
    ymin = cos(nose_alpha)
    nose_y = np.array((ymin, 1.02, 1.09, 1.1, 1.09, 1.02, ymin))
    ear_y = np.array((0.15, 0.145, 0.135, 0.125, 0.111, -0.011, -0.1864,
                      -0.2626, -0.2768, -0.2398))
    ear_x_right = np.array((cos(asin(ear_y[0])), 1., 1.02, 1.025, 1.03, 1.04,
                            1.07, 1.06, 1.02, cos(asin(ear_y[-1]))))
    ear_x_left = -ear_x_right

    # apply radius and center
    if isinstance(head_radius, Sequence):
        rx, ry = head_radius
    else:
        rx = ry = head_radius

    if isinstance(head_pos, Sequence):
        cx, cy = head_pos
        cx += 0.5
        cy += 0.5
    else:
        cx = 0.5
        cy = head_pos + 0.5

    for item in (head_x, nose_x, ear_x_right, ear_x_left):
        item *= rx
        item += cx

    for item in (head_y, nose_y, ear_y):
        item *= ry
        item += cy

    return ((head_x, head_y), (nose_x, nose_y), (ear_x_left, ear_y),
            (ear_x_right, ear_y))


class PltAdjacency:
    def __init__(self, ax, locs, adjacency, **linestyle):
        self.ax = ax
        self.locs = locs
        self._h = []
        self.show(adjacency, **linestyle)

    def show(self, adjacency, **linestyle):
        while self._h:
            self._h.pop().remove()

        if adjacency is None:
            return

        for c, r in adjacency:
            x = self.locs[[c, r], 0]
            y = self.locs[[c, r], 1]
            line = Line2D(x, y, **linestyle)
            self.ax.add_line(line)
            self._h.append(line)


class AxMap2d:

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            sensors: Sensor,
            proj: str,
            extent: float,
            marker: str,
            size: float,
            color: ColorArg,
            mark: Union[Sequence[str], str, Sequence[int], int] = None,
            head_radius: Union[float, Sequence[float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            head_linewidth: float = None,
            labels: Literal['none', 'index', 'name', 'fullname'] = 'none',
    ):
        self.ax = ax

        self.sensors = PltMap2d(ax, sensors, proj, extent, marker, size, color, mark, labels=labels, invisible=True, head_radius=head_radius, head_pos=head_pos, head_linewidth=head_linewidth)

        locs = sensors.get_locs_2d(proj, extent, SENSORMAP_FRAME)
        self.adjacency = PltAdjacency(ax, locs, None)

        ax.set_aspect('equal', 'datalim', 'C')
        if extent:
            ax.set_xlim(0, extent)
            ax.set_ylim(0, extent)

    def mark_sensors(self, *args, **kwargs):
        self.sensors.mark_sensors(*args, **kwargs)

    def remove(self):
        "Remove from axes"
        self.sensors.remove()


def sensor_labels(
        sensors: Sensor,
        labels: Literal['none', 'index', 'name', 'fullname'] = 'none',
) -> Optional[Sequence[str]]:
    if labels == 'none':
        return
    elif labels == 'index':
        return list(map(str, range(len(sensors))))
    elif labels == 'name':
        out = sensors.names
        prefix = os.path.commonprefix(out)
        pf_len = len(prefix)
        return [label[pf_len:] for label in out]
    elif labels == 'fullname':
        return sensors.names
    else:
        raise ValueError(f"{labels=}")


class PltMap2d:
    "Sensor-map plot"
    _labels_arg = 'none'  # currently shown label text

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            sensors: Sensor,
            proj: str,
            extent: float,
            marker: Union[str, matplotlib.markers.MarkerStyle],
            size: float,
            color: ColorArg,
            mark: Union[Sequence[str], str, Sequence[int], int],
            mcolor: Union[ColorArg, Sequence[ColorArg]] = None,
            msize: Union[float, Sequence[float]] = 20,
            mmarker: Union[str, matplotlib.markers.MarkerStyle] = 'o',
            labels: Literal['', 'none', 'index', 'name', 'fullname'] = 'none',
            invisible: bool = False,
            head_radius: Union[float, Sequence[float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            head_linewidth: float = None,
    ):
        self.ax = ax
        self.sensors = sensors
        self.locs = sensors.get_locs_2d(proj, extent, SENSORMAP_FRAME)
        self._index = None if invisible else sensors._visible_sensors(proj)
        self._extent = extent
        self._label_h = []  # handles for label text objects
        self._mark_handles = []

        # head outline
        if head_radius:
            if head_radius is True:
                head_radius = 0.5 * (1 - (SENSORMAP_FRAME * 0.9))

            if head_linewidth is None:
                head_linewidth = mpl.rcParams['lines.linewidth']

            for x, y in _head_outlines(head_radius, head_pos):
                ax.plot(x, y, color='k', linewidth=head_linewidth, solid_capstyle='butt', clip_on=False)

        # sensors
        if marker:
            index = slice(None) if self._index is None else self._index
            self._sensor_h = ax.scatter(self.locs[index, 0], self.locs[index, 1], size, color, marker=marker)
        else:
            self._sensor_h = None

        if labels:
            self.show_labels(labels)

        if mark is not None:
            self.mark_sensors(mark, size=msize, color=mcolor, marker=mmarker)

    def mark_sensors(self, sensors, **kwargs):
        """Mark specific sensors

        Parameters
        ----------
        sensors : None | Sensor dimension index
            Sensors which should be marked (None to clear all markings).
        size : scalar | sequence of scalars
            Marker size(s) in points^2 (default 20).
        color : color | sequence of colors
            Marker color(s) (default ``'yellow'``).
        marker : str
            Marker style (default: ``'o'``).
        ... :
            Matplotlib :func:`~matplotlib.axes.Axes.scatter` parameters.
        """
        if sensors is None:
            while self._mark_handles:
                self._mark_handles.pop().remove()
            return
        kwargs.setdefault('s', kwargs.pop('size', 20))
        kwargs.setdefault('c', kwargs.pop('color', 'yellow'))
        kwargs.setdefault('marker', kwargs.pop('m', 'o'))
        idx = self.sensors._array_index(sensors)
        h = self.ax.scatter(self.locs[idx, 0], self.locs[idx, 1], **kwargs)
        self._mark_handles.append(h)

    def remove(self):
        "Remove from axes"
        if self._sensor_h:
            self._sensor_h.remove()
        while self._mark_handles:
            self._mark_handles.pop().remove()

    def show_labels(self, labels='name', xpos=0, ypos=0, ha='center', va='bottom', **text_kwargs):
        """Plot labels for the sensors

        Parameters
        ----------
        labels : 'none' | 'index' | 'name' | 'fullname'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        xpos, ypos : scalar
            The position offset of the labels from the sensor markers.
        text_kwargs : **
            Matplotlib text parameters.
        """
        if labels == self._labels_arg:
            return
        label_text = sensor_labels(self.sensors, labels)

        # remove existing labels
        while self._label_h:
            self._label_h.pop().remove()
        self._labels_arg = labels

        if label_text is None:
            return

        locs = self.locs
        if self._index is not None:
            label_text = Datalist(label_text)[self._index]
            locs = locs[self._index]

        locs = locs + [[xpos, ypos]]
        for (x, y), txt in zip(locs, label_text):
            h = self.ax.text(x, y, txt, ha=ha, va=va, **text_kwargs)
            self._label_h.append(h)

    def separate_labels(self, pad):
        # separate overlapping labels
        locs = self.locs
        data_to_screen = self.ax.transData.transform_point
        screen_to_data = self.ax.transData.inverted().transform_point

        if self._extent:
            center = self._extent / 2.
            label_order = np.argsort((locs[:, 0] - center) ** 2)
        else:
            center = 0
            label_order = np.argsort(locs[:, 0] ** 2)

        n_labels = len(label_order)
        for i in range(1, n_labels):
            i_i = label_order[i]
            i_loc = locs[i_i, 0]
            if i_loc == center:
                continue
            h = self._label_h[i_i]
            i_bbox = h.get_window_extent()
            padded_bbox = i_bbox.expanded(pad, 0)
            side = -1 if i_loc < center else 1

            dx = 0
            for j in range(i):
                j_i = label_order[j]
                j_bbox = self._label_h[j_i].get_window_extent()
                if padded_bbox.overlaps(j_bbox):
                    if side == -1:
                        dx = min(dx, j_bbox.xmin - i_bbox.xmax - pad)
                    else:
                        dx = max(dx, j_bbox.xmax - i_bbox.xmin + pad)

            if dx:
                x, y = data_to_screen(h.get_position())
                h.set_position(screen_to_data((x + dx, y)))

    def set_label_color(self, color='k'):
        """Change the color of all sensor labels

        Parameters
        ----------
        color : matplotlib color
            New color for the sensor labels.
        """
        for h in self._label_h:
            h.set_color(color)


class SensorMapMixin:
    __label_options = ['None', 'Index', 'Name', 'Full Name']
    __label_option_args = ['none', 'index', 'name', 'fullname']

    def __init__(
            self,
            sensor_plots: Sequence[PltMap2d],
    ):
        "Call after EelFigure init (toolbar fill)"
        self.__label_color = 'k'
        self.__sensor_plots = sensor_plots
        if not self._has_frame:
            return
        # find current label text
        sel = self.__label_option_args.index(sensor_plots[0]._labels_arg)
        self.__LabelChoice.SetSelection(sel)

    def _fill_toolbar(self, tb):
        from .._wxgui import wx

        # sensor labels
        lbl = wx.StaticText(tb, -1, "Labels:")
        tb.AddControl(lbl)
        choice = wx.Choice(tb, -1, choices=self.__label_options)
        tb.AddControl(choice)
        self.__LabelChoice = choice
        choice.Bind(wx.EVT_CHOICE, self.__OnSensorLabelChoice)

        # sensor label color
        choices = ['black', 'white', 'blue', 'green', 'red', 'cyan', 'magenta',
                   'yellow']
        choice = wx.Choice(tb, -1, choices=choices)
        tb.AddControl(choice)
        self.__LabelColorChoice = choice
        choice.Bind(wx.EVT_CHOICE, self.__OnSensorLabelColorChoice)

        btn = wx.Button(tb, label="Mark")  # , style=wx.BU_EXACTFIT)
        btn.Bind(wx.EVT_BUTTON, self.__OnMarkSensor)
        tb.AddControl(btn)

    def __OnMarkSensor(self, event):  # noqa
        from .._wxgui import wx

        msg = "Channels to mark, separated by comma"
        dlg = wx.TextEntryDialog(self._frame, msg, "Mark Sensor")
        if dlg.ShowModal() != wx.ID_OK:
            return

        chs = tuple(filter(None, map(str.strip, dlg.GetValue().split(','))))
        try:
            self.mark_sensors(chs)
        except Exception as exc:
            msg = '%s: %s' % (type(exc).__name__, exc)
            sty = wx.OK | wx.ICON_ERROR
            wx.MessageBox(msg, "Mark Sensors Failed for %r" % chs, style=sty)

    def __OnSensorLabelChoice(self, event):  # noqa
        sel = event.GetSelection()
        sel_arg = self.__label_option_args[sel]
        self.set_label_text(sel_arg)

    def __OnSensorLabelColorChoice(self, event):  # noqa
        sel = event.GetSelection()
        color = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y'][sel]
        self.set_label_color(color)

    def mark_sensors(self, sensors, axis=None, **kwargs):
        """Mark given sensors on the plots

        Parameters
        ----------
        sensors : None | Sensor dimension index
            Sensors which should be marked (None to clear all markings).
        axis : int | list of int
            Which axes to mark (default is all).
        size : scalar | sequence of scalars
            Marker size(s) in points^2 (default 20).
        color : color | sequence of colors
            Marker color(s) (default ``'yellow'``).
        marker : str
            Marker style (default: ``'o'``).
        ... :
            Matplotlib :func:`~matplotlib.axes.Axes.scatter` parameters.
        """
        if axis is None:
            plots = self.__sensor_plots
        elif isinstance(axis, int):
            plots = [self.__sensor_plots[axis]]
        else:
            plots = [self.__sensor_plots[i] for i in axis]

        for p in plots:
            p.mark_sensors(sensors, **kwargs)
        self.draw()

    def separate_labels(self, pad=10):
        """Move overlapping labels apart along the x axis

        Parameters
        ----------
        pad : scalar
            Minimum amount of padding between labels (in pixels; default 5).
        """
        for p in self.__sensor_plots:
            p.separate_labels(pad)
        self.draw()

    def set_label_color(self, color='w'):
        if hasattr(self, '_SensorLabelChoice'):
            sels = ['k', 'w', 'b', 'g', 'r', 'c', 'm', 'y']
            if color in sels:
                sel = sels.index(color)
                self.__LabelColorChoice.SetSelection(sel)

        self.__label_color = color
        for p in self.__sensor_plots:
            p.set_label_color(color)
        self.draw()

    def set_label_text(self, text='name'):
        """Add/remove sensor labels

        Parameters
        ----------
        labels : 'none' | 'index' | 'name' | 'fullname'
            Content of the labels. For 'name', any prefix common to all names
            is removed; with 'fullname', the full name is shown.
        """
        # update plots
        for p in self.__sensor_plots:
            p.show_labels(text, color=self.__label_color)
        # update GUI
        self.__LabelChoice.SetSelection(self.__label_option_args.index(text))
        self.draw()


class SensorMaps(EelFigure):
    """Multiple views on a sensor layout.

    Allows selecting sensor groups and retrieving corresponding indices.

    Parameters
    ----------
    sensors : Sensor | NDVar
        The :class:`Sensor` dimension, or an :class:`NDVar` with a sensor
        dimension.
    select : list of int
        Initial selection.
    proj : str
        Sensor projection for the fourth plot.
    size : scalar
        Size for the sensor markers.
    color : matplotlib color
        Color for the sensor markers.
    marker : str
        Marker for the sensor positions.
    frame : scalar
        Size of the empty space around sensors in axes.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
    **Selecting Sensor Groups:**

     - Dragging with the left mouse button adds sensors to the selection.
     - Dragging with the right mouse button removes sensors from the current
       selection.
     - The 'Clear' button (or :meth:`clear`) clears the selection.

    """
    def __init__(self, sensors, select=[], proj='default', size=1,
                 color='k', marker='.', frame=0.05, **kwargs):
        sensors = as_sensor(sensors)

        # layout figure
        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None
        layout = ImLayout(4, 1, 3, None, {}, frame='off', columns=2, rows=2, **kwargs)
        EelFigure.__init__(self, sensors.sysname, layout)
        self.figure.subplots_adjust(left=0, bottom=0, right=1, top=1,
                                    wspace=.1, hspace=.1)

        # store args
        self._sensors = sensors

        min_coords = sensors.locations.min(0) - frame
        max_coords = sensors.locations.max(0) + frame
        xlim, ylim, zlim = list(zip(min_coords, max_coords))

        # back
        ax = self.ax0 = self.axes[0]
        ax.proj = 'back'  # y-
        ax.extent = False
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        self._h0 = AxMap2d(ax, sensors, ax.proj, ax.extent, marker, size, color)

        # left
        ax = self.ax1 = self.axes[1]
        ax.proj = 'left'  # x-
        ax.extent = False
        ax.set_xlim(ylim)
        ax.set_ylim(zlim)
        self._h1 = AxMap2d(ax, sensors, ax.proj, ax.extent, marker, size, color)

        # top
        ax = self.ax2 = self.axes[2]
        ax.proj = 'top'  # z+
        ax.extent = False
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self._h2 = AxMap2d(ax, sensors, ax.proj, ax.extent, marker, size, color)

        # proj
        ax = self.ax3 = self.axes[3]
        ax.proj = proj
        ax.extent = 1
        self._h3 = AxMap2d(ax, sensors, ax.proj, ax.extent, marker, size, color)
        self.ax3.set_xlim(-frame, 1 + frame)
        self.ax3.set_ylim(-frame, 1 + frame)

        self._sensor_maps = (self._h0, self._h1, self._h2, self._h3)

        # selection
        self.sel_kwargs = dict(marker='o', s=5, c='r', linewidths=.9)
        self._sel_h = []
        if select is not None:
            self.set_selection(select)
        else:
            self._selection = None

        # setup mpl event handling
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)

        self._show()

    def _fill_toolbar(self, tb):
        from .._wxgui import wx

        tb.AddSeparator()

        # plot labels
        btn = wx.Button(tb, wx.ID_CLEAR, "Clear")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnClear)

    def clear(self):
        "Clear the current sensor selection."
        self._selection = None
        self._update_mark_plot()

    def get_selection(self):
        """Retrieve the current selection

        Returns
        -------
        selection : list
            The current selection as a list of indices.
        """
        if self._selection is None:
            return []
        else:
            return np.where(self._selection)[0]

    def _on_button_press(self, event):
        ax = event.inaxes
        if ax:
            self._is_dragging = True
            self._drag_ax = event.inaxes
            self._drag_x = event.xdata
            self._drag_y = event.ydata

            self._frame.store_canvas()
            x = np.ones(5) * event.xdata
            y = np.ones(5) * event.ydata
            self._drag_rect = ax.plot(x, y, '-k')[0]

    def _on_button_release(self, event):
        if not hasattr(self, '_drag_rect'):
            return

        x = self._drag_rect.get_xdata()
        y = self._drag_rect.get_ydata()
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)

        ax = self._drag_ax
        locs = self._sensors.get_locs_2d(ax.proj, ax.extent, SENSORMAP_FRAME)
        x = locs[:, 0]
        y = locs[:, 1]
        sel = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)

        if self._selection is None:
            self._selection = sel
        elif event.button == 1:
            self._selection[sel] = True
        else:
            self._selection[sel] = False

        # clear dragging-related attributes
        self._drag_rect.remove()
        del self._drag_rect
        self._drag_ax = None
        self._drag_x = None
        self._drag_y = None

        self._update_mark_plot()

    @staticmethod
    def _on_motion_status_text(event):
        ax = event.inaxes
        if ax:
            out = "Projection: " + ax.proj
            if not ax.extent:
                out += ';  x = %i, y = %i' % (round(event.xdata * 1e3),
                                              round(event.ydata * 1e3))
            return out
        return ''

    def _on_motion_sub(self, event):
        ax = event.inaxes
        if ax and ax is self._drag_ax:
            x0 = self._drag_x
            x1 = event.xdata
            y0 = self._drag_y
            y1 = event.ydata
            x = [x0, x1, x1, x0, x0]
            y = [y0, y0, y1, y1, y0]
            self._drag_rect.set_data(x, y)
            return {ax}
        return set()

    def _OnClear(self, event):  # noqa
        self.clear()

    def set_selection(self, select):
        """
        Set the current selection with a list of indices.

        Parameters
        ----------
        select : sensor index
            Index for sensor dimension, for example array_like of int, or list
            of sensor names.
        """
        idx = self._sensors._array_index(select)
        self._selection = np.zeros(len(self._sensors), dtype=bool)
        self._selection[idx] = True
        self._update_mark_plot()

    def _update_mark_plot(self):
        for h in self._sensor_maps:
            h.sensors.mark_sensors(None)  # clear old selection
            h.sensors.mark_sensors(self._selection, **self.sel_kwargs)
        self.canvas.draw()


class SensorMap(SensorMapMixin, EelFigure):
    """Plot sensor positions in 2 dimensions

    Parameters
    ----------
    sensors : Sensor | NDVar
        The :class:`Sensor` dimension, or an :class:`NDVar` with a sensor
        dimension.
    labels
        Content of the labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    proj:
        Transform to apply to 3 dimensional sensor coordinates for plotting
        locations in a plane
    size : scalar
        Size for the sensor markers.
    color : matplotlib color
        Color for the sensor markers.
    marker : str
        Marker for the sensor positions.
    mark : None | list of int
        List of sensor indices to mark.
    head_radius : scalar | tuple | True
        Radius of the head outline drawn over sensors (on sensor plots with
        normalized positions, 0.45 is the outline of the topomap); 0 to plot no
        outline; tuple for separate (right, anterior) radius. True to be equal
        to :class:`plot.Topomap` with ``method="mne"``.
        The default is determined automatically.
    head_pos : scalar
        Head outline position along the anterior axis (0 is the center, 0.5 is
        the top end of the plot).
    adjacency
        Show sensor adjacency (default False).
    ...
        Also accepts :ref:`general-layout-parameters`.
    """
    def __init__(
            self,
            sensors: Any,
            labels: Literal['none', 'index', 'name', 'fullname'] = 'name',
            proj: str = 'default',
            size: float = 1,
            color: ColorArg = 'k',
            marker: str = '.',
            mark: Sequence = None,
            head_radius: Union[float, Sequence[float]] = None,
            head_pos: Union[float, Sequence[float]] = 0,
            adjacency: bool = False,
            **kwargs):
        sensors = as_sensor(sensors)
        kwargs.setdefault('frame', 'none')
        layout = ImLayout(1, 1, 5, default_margins={}, **kwargs)
        EelFigure.__init__(self, sensors.sysname, layout)
        ax = self.axes[0]

        # store args
        self._sensors = sensors
        self._proj = proj
        self._marker_handles = []
        self._adjacency = None

        self._markers = AxMap2d(ax, sensors, proj, 1, marker, size, color, mark, head_radius, head_pos, labels=labels)
        SensorMapMixin.__init__(self, [self._markers.sensors])

        if adjacency:
            self.show_adjacency()

        self._show()

    def remove_markers(self):
        "Remove all sensor markers."
        while len(self._marker_handles) > 0:
            h = self._marker_handles.pop(0)
            h.remove()
        self.canvas.draw()

    def show_adjacency(self, show=True):
        """Show the sensor adjacency as lines connecting sensors.

        Parameters
        ----------
        show : bool
            Show or hide the sensor adjacency.
        """
        if show:
            conn = self._sensors.adjacency()
            self._markers.adjacency.show(conn)
        else:
            self._markers.adjacency.show(None)
        self.draw()


class SensorMap3d(EelFigure):
    """Plot sensor positions in 3 dimensions

    Parameters
    ----------
    sensors : Sensor | NDVar
        The :class:`Sensor` dimension, or an :class:`NDVar` with a sensor
        dimension.
    labels
        Content of the labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    adjacency
        Show sensor adjacency (default False).
    ...
        Also accepts :ref:`general-layout-parameters`.
    """
    _make_axes = False

    def __init__(
            self,
            sensors: Any,
            labels: Literal['none', 'index', 'name', 'fullname'] = 'none',
            adjacency: bool = False,
            **kwargs):
        sensors = as_sensor(sensors)
        layout = ImLayout(1, 1, 5, default_margins={}, **kwargs)
        EelFigure.__init__(self, sensors.sysname, layout)

        ax = self.figure.add_subplot(111, projection='3d')
        locs = sensors.locations
        ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2])
        ax.set_xlabel('Left -> Right')
        ax.set_ylabel('Posterior -> Anterior')
        ax.set_zlabel('Inferior -> Superior')

        if adjacency:
            from mpl_toolkits.mplot3d.art3d import Line3D

            for c, r in sensors.adjacency():
                xs, ys, zs = locs[[c, r]].T
                line = Line3D(xs, ys, zs)
                ax.add_line(line)

        label_text = sensor_labels(sensors, labels)
        if label_text:
            for loc, txt in zip(locs, label_text):
                ax.text(*loc, txt)

        self._show()
