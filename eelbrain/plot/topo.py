"""
plot.topo
=========

Plots with topographic maps.

"""

from __future__ import division

import numpy as np

import _base
import utsnd
import sensors as _plt_sensors

# try:
#    from _topo3d import *
# except:
#    logging.info("eelbrain.plot.topo: _topo3d import failed, 3d-plots not available")


__hide__ = ['cs', 'test', 'utsnd']



class topomap(_base.eelfigure):
    def __init__(self, epochs, sensors=True, proj='default',
                 size=5, dpi=100, title="plot.topomap",
                 res=100, interpolation='nearest'):
        """
        Plots individual topogeraphies.

        Parameters
        ----------

        sensors : bool | 'id' | 'name'

        """
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1)

        # create figure
        n_plots = len(epochs)
        x_size = size * n_plots
        y_size = size
        figsize = (x_size, y_size)

        super(topomap, self).__init__(title=title, figsize=figsize, dpi=dpi)

        # plot epochs (x/y are in figure coordinates)
        frame = .05

        topo_kwargs = dict(res=res,
                           interpolation=interpolation,
                           proj=proj,
                           sensors=sensors,
                           )

        self.axes = []
        for i, layers in enumerate(epochs):
            # axes coordinates
            left = (i + frame) / n_plots
            bottom = frame
            width = (1 - 2 * frame) / n_plots
            height = 1 - 3 * frame

            ax_rect = [left, bottom, width, height]
            ax = self.figure.add_axes(ax_rect)
            ax.ID = i
            self.axes.append(ax)

            _ax_topomap(ax, layers, title=True, **topo_kwargs)

        self._show()



class butterfly(_base.eelfigure):
    """
    Butterfly plot with corresponding topomaps.

     - LMB click in butterfly plots fixates the topomap time.
     - RMB click in butterfly plots removes the time point, the topomaps follow
       the mouse pointer.

    """
    def __init__(self, epochs, size=2, bflywidth=3, dpi=90,
                 proj='default', res=100, interpolation='nearest',
                 title=True, xlabel=True, ylabel=True,
                 color=True, sensors=True, ROI=None, ylim=None):
        """
        **Plot atributes:**

        ROI : list of indices
            plot a subset of sensors
        sensors : bool
            determines whether all sensors are marked in the topo-maps

        **Figure Layout:**

        size : scalar
            in inches: height of the butterfly axes as well as side length of
            the topomap axes
        bflywidth : scalar
            multiplier for the width of butterfly plots based on their height
        res : int
            resolution of the topomap plots (res x res pixels)
        interpolation : 'nearest' | ...
            matplotlib imshow kwargs

        """
        frame_title = "plot.topo.butterfly: %r"
        if isinstance(title, basestring):
            frame_title = frame_title % title
        else:
            frame_title = frame_title % getattr(epochs, 'name', '')

        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 2)
        n_plots = len(epochs)

        # create figure
        x_size = size * (1 + bflywidth)
        y_size = size * n_plots
        figsize = (x_size, y_size)

        super(butterfly, self).__init__(title=frame_title, figsize=figsize, dpi=dpi)

        # axes sizes
        frame = .05  # in inches; .4

        xframe = frame / x_size
        x_left_ylabel = 0.5 / x_size if ylabel else 0
        x_left_title = 0.5 / x_size
        x_text = x_left_title / 3
        ax1_left = xframe + x_left_title + x_left_ylabel
        ax1_width = bflywidth / (bflywidth + 1) - ax1_left - xframe / 2
        ax2_left = bflywidth / (bflywidth + 1) + xframe / 2
        ax2_width = 1 / (bflywidth + 1) - 1.5 * xframe

        yframe = frame / y_size
        y_bottomframe = 0.5 / y_size
#        y_bottom = yframe
        y_sep = (1 - y_bottomframe) / n_plots
        height = y_sep - yframe

        self.topo_kwargs = {'proj': proj,
                            'res': res,
                            'interpolation': interpolation,
                            'sensors': sensors,
                            'ROI': ROI,
                            'ROIcolor': color,
                            'title': False}

        t = 0
        self.topo_axes = []
        self.bfly_axes = []
        self.topos = []
        self.t_markers = []

        # plot epochs (x/y are in figure coordinates)
        for i, layers in enumerate(epochs):
            bottom = 1 - y_sep * (1 + i)

            ax1_rect = [ax1_left, bottom, ax1_width, height]
            ax2_rect = [ax2_left, bottom, ax2_width, height]
            ax1 = self.figure.add_axes(ax1_rect)
            ax1.ID = i

            ax2 = self.figure.add_axes(ax2_rect, frameon=False)
            ax2.set_axis_off()

            # t - label
            if len(self.topo_axes) == n_plots - 1:
#                ax2.set_title('t = %.3f' % t)
#                self._t_title = ax2.title
                self._t_title = ax2.text(.0, 0, 't = %.3f' % t, ha='center')

            self.bfly_axes.append(ax1)
            self.topo_axes.append(ax2)
            self.topos.append((ax2, layers))

            show_x_axis = (i == n_plots - 1)

            utsnd._ax_butterfly(ax1, layers, sensors=ROI, ylim=ylim,
                                title=False, xlabel=show_x_axis, ylabel=ylabel,
                                color=color)

            if not show_x_axis:
#                ax1.xaxis.set_visible(False)
                ax1.xaxis.set_ticklabels([])

            # find and print epoch title
            title = True
            for l in layers:
                if title is True:
                    title = getattr(l, 'name', True)
            if isinstance(title, str):
                y_text = bottom + y_sep / 2
                ax1.text(x_text, y_text, title, transform=self.figure.transFigure,
                         ha='center', va='center', rotation='vertical')

        # setup callback
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self._realtime_topo = True
        self._frame.store_canvas()
        self._draw_topo(0, draw=False)
        self._show()

    def _draw_topo(self, t, draw=True):
        self._current_t = t
        for topo_ax, layers in self.topos:
            topo_ax.cla()
            layers = [l.subdata(time=t) for l in layers]
            _ax_topomap(topo_ax, layers, **self.topo_kwargs)

        if draw:
            self._frame.redraw(axes=self.topo_axes)  # , artists=self.t_markers)

    def _rm_markers(self):
        if self.t_markers:
            for m in self.t_markers:
                m.remove()
            self.t_markers = []

    def set_topo_t(self, t):
        "set the time point of the topo-maps"
        self._realtime_topo = False
        self._draw_topo(t, draw=False)

        # update t-markers
        self._rm_markers()
        for ax in self.bfly_axes:
            t_marker = ax.axvline(t, color='k')
            self.t_markers.append(t_marker)

        # add time label
        ax = self.topo_axes[-1]
        t_str = "t = %.3f" % t
        self._t_label = ax.text(.5, -0.1, t_str, ha='center', va='top')

        self.canvas.draw()  # otherwise time label does not get redrawn

    def _on_click(self, event):
        ax = event.inaxes
        if ax and hasattr(ax, 'ID'):
            button = {1:'l', 2:'r', 3:'r'}[event.button]
            if button == 'l':
                t = event.xdata
                self.set_topo_t(t)
            elif (button == 'r') and (self._realtime_topo == False):
                self._rm_markers()
                self._t_label.remove()
                self._realtime_topo = True
                self.canvas.draw()
#                self._frame.redraw(axes=self.bfly_axes) # this leaves the time label

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        txt = "Topomap: t = %.3f" % self._current_t
        self._frame.SetStatusText(txt)

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        ax = event.inaxes
        if ax and hasattr(ax, 'ID'):
            super(self.__class__, self)._on_motion(event)
        if self._realtime_topo and ax and hasattr(ax, 'ID'):
            self._draw_topo(event.xdata)

    def set_ylim(self, *ylim):
        """
        Change the range of values displayed in butterfly-plots.

        """
        if len(ylim) == 1:
            ymin, ymax = -ylim[0], ylim[0]
        elif len(ylim) == 2:
            ymin, ymax = ylim
        else:
            raise ValueError("Wrong number of values for ylim (need 1 or 2)")

        for i, ax in enumerate(self.figure.axes):
            if i % 2 == 0:
                ax.set_ylim(ymin, ymax)
        self.canvas.draw()


def _plt_topomap(ax, epoch, proj='default', res=100,
                 im_frame=0.02,  # empty space around sensors in the im
                 colorspace=None,
                 **im_kwargs):
    colorspace = _base.read_cs_arg(epoch, colorspace)
    handles = {}

    Y = epoch.get_data(('sensor',))
    Ymap = epoch.sensor.get_im_for_topo(Y, proj=proj, res=res, frame=im_frame)

    emin = -im_frame
    emax = 1 + im_frame
    map_kwargs = {'origin': "lower",
                  'extent': (emin, emax, emin, emax)}

    if colorspace.cmap:
        im_kwargs.update(map_kwargs)
        im_kwargs.update(colorspace.get_imkwargs())
        handles['im'] = ax.imshow(Ymap, **im_kwargs)

    # contours
    if colorspace.contours:
        # print "contours: {0}".format(colorspace.contours)
        map_kwargs.update(colorspace.get_contour_kwargs())
        h = ax.contour(Ymap, **map_kwargs)
        handles['contour'] = h

    return handles



def _ax_topomap(ax, layers, title=True,
                sensors=None, ROI=None, ROIcolor=True,
                proj='default', xlabel=None,
                im_frame=0.02, **im_kwargs):
    """
    sensors : bool | str
        plot sensor markers (str to add label:
    ROI : list of IDs
        highlight a subset of the sensors

    """
    ax.set_axis_off()
    handles = {}
    for layer in layers:
        handles[layer.name] = _plt_topomap(ax, layer, im_frame=im_frame,
                                           proj=proj, **im_kwargs)
        if title is True:
            title = getattr(layer, 'name', True)

    # plot sensors
    if sensors:
        sensor_net = layers[0].sensor
        _plt_sensors._plt_map2d(ax, sensor_net, proj=proj)

        if isinstance(sensors, str):
            _plt_sensors._plt_map2d_labels(ax, sensor_net, proj=proj, text=sensors)

    if ROI is not None:
        sensor_net = layers[0].sensor
        kw = dict(marker='.',  # symbol
                ms=3,  # marker size
                markeredgewidth=1,
                ls='')

        if ROIcolor is not True:
            kw['color'] = ROIcolor

        _plt_sensors._plt_map2d(ax, sensor_net, proj=proj, ROI=ROI, kwargs=kw)


    ax.set_xlim(-im_frame, 1 + im_frame)
    ax.set_ylim(-im_frame, 1 + im_frame)

    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)

    if isinstance(title, str):
        handles['title'] = ax.set_title(title)

    return handles





# MARK: XBYX plots (ANOVA results plots)


class _Window_Topo:
    """Helper class for array"""
    def __init__(self, ax, layers):
        self.ax = ax
        # initial plot state
        self.t_line = None
        self.pointer = None
        self.layers = layers

    def update(self, parent_ax=None, t=None, cs=None, sensors=None):
        if t != None:
            if self.t_line:
                self.t_line.remove()
            self.t_line = parent_ax.axvline(t, c='r')
            # self.pointer.xy=(t,1)
            # self.pointer.set_text("t = %s"%t)
            if self.pointer:
                # print 't =', t
                self.pointer.set_axes(parent_ax)
                self.pointer.xy = (t, 1)
                self.pointer.set_text("t=%.3g" % t)
                self.pointer.set_visible(True)
            else:
                xytext = self.ax.transAxes.transform((.5, 1))
                # These coordinates are in 'figure pixels'. They do not scale
                # when the figure is rescaled, so we need to transform them
                # into 'figure fraction' coordinates
                inv = self.ax.figure.transFigure.inverted()
                xytext = inv.transform(xytext)
                self.pointer = parent_ax.annotate("t=%.3g" % t, (t, 0),
                                    xycoords='data',
                                    xytext=xytext,
                                    textcoords='figure fraction',
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    arrowprops={'arrowstyle': '-',
                                                'shrinkB': 0,
                                                'connectionstyle': "angle3,angleA=90,angleB=0",
                                                'color': 'r'},
#                                    arrowprops={'width':1, 'frac':0,
#                                                'headwidth':0, 'color':'r',
#                                                'shrink':.05},
                                    zorder=99)

            self.ax.cla()
            layers = [l.subdata(time=t) for l in self.layers]
            _ax_topomap(self.ax, layers, title=False)

    def clear(self):
        self.ax.cla()
        self.ax.set_axis_off()
        if self.t_line:
            self.t_line.remove()
            self.t_line = None
        if self.pointer:
            self.pointer.remove()
            self.pointer = None




class array(_base.eelfigure):
    """
    Channel by sample plots with corresponding topomaps

     - LMB click on a topomap selects it for tracking the mouse pointer
         - LMB on the array plot fixates the topomap time point
     - RMB on a topomap removes the topomap

    """
    def __init__(self, epochs, title=None, height=3, width=2.5, ntopo=3, dpi=100,
                 ylim=None, t=[]):
        """
        Channel by sample array-plots with topomaps corresponding to
        individual time points.

        kwargs
        ------
        title : str | None
            title
        ntopo | int
            number of topomaps per array-plot

        """
        frame_title = "plot.topo.array: %r"
        if isinstance(title, basestring):
            frame_title = frame_title % title
        else:
            frame_title = frame_title % getattr(epochs, 'name', '')

        # convenience for single segment
        epochs = _base.unpack_epochs_arg(epochs, 2)

        # figure properties
        n_epochs = len(epochs)
        n_topo_total = ntopo * n_epochs
        left_rim = width / 4
        fig_width, fig_height = n_epochs * width + left_rim, height
        figsize = (fig_width, fig_height)

        # fig coordinates
        x_frame_l = .6 / width / n_epochs
        x_frame_r = .025 / n_epochs
        x_sep = .01 / n_epochs

        x_per_ax = (1 - x_frame_l - x_frame_r) / n_epochs

        # create figure
        super(array, self).__init__(title=frame_title, dpi=dpi, figsize=figsize)
        fig = self.figure

        fig.subplots_adjust(left=x_frame_l,
                            bottom=.05,
                            right=1 - x_frame_r,
                            top=.9,
                            wspace=.1, hspace=.3)
        if title:
            fig.suptitle(title)
        self.title = title

        # im_array plots
        self.main_axes = []
        ax_height = .4 + .07 * (not title)
        ax_bottom = .45  # + .05*(not title)
        for i, layers in enumerate(epochs):
            ax_left = x_frame_l + i * (x_per_ax + x_sep)
            ax_right = 1 - x_frame_r - (n_epochs - i - 1) * (x_per_ax + x_sep)
            ax_width = ax_right - ax_left
            ax = fig.add_axes((ax_left, ax_bottom, ax_width, ax_height),
                              picker=True)
            self.main_axes.append(ax)
            ax.ID = i
            ax.type = 'main'
            utsnd._ax_im_array(ax, layers)
            if i > 0:
                ax.yaxis.set_visible(False)

        # topo plots
        self.windows = []
        for i, layers in enumerate(epochs):
            for j in range(ntopo):
                ID = i * ntopo + j
                ax = fig.add_subplot(3, n_topo_total, 2 * n_topo_total + 1 + ID,
                                     picker=True, xticks=[], yticks=[])
                ax.ID = ID
                ax.type = 'window'
                self.windows.append(_Window_Topo(ax, layers))

        # save important properties
        self.epochs = epochs
        self._ntopo = ntopo

        # if t argument is provided, set topo-pol time points
        if t:
            if np.isscalar(t):
                t = [t]
            self.setwins(*t)

        # setup callback
        self._selected_window = None
        self.canvas.mpl_connect('pick_event', self._pick_handler)
        self.canvas.mpl_connect('motion_notify_event', self._motion_handler)
        self._frame.store_canvas()
        self._show()

    def __repr__(self):
        e_repr = []
        for e in self.epochs:
            if hasattr(e, 'name'):
                e_repr.append(e.name)
            else:
                e_repr.append([ie.name for ie in e])
        kwargs = {'s': repr(e_repr),
                  't': ' %r' % self.title if self.title else ''}
        txt = "<plot.topo.array{t} ({s})>".format(**kwargs)
        return txt

    def set_topo_single(self, topo, t, parent_im_id='auto'):
        """
        Set the time for a single topomap

        topo : int
            Id of the topomap (numbered throughout the figure).
        t : scalar or ``None``
            time point; ``None`` clears the topomap

        """
        # get parent ax
        if parent_im_id == 'auto':
            parent_im_id = int(topo / self._ntopo)
        parent_ax = self.main_axes[parent_im_id]
        # get window ax
        w = self.windows[topo]
        w.clear()

        if t is not None:
            w.update(parent_ax=parent_ax, t=t)

        self.canvas.draw()

    def set_topowin(self, topo_id, t):
        """
        Set the time point for a topo-map (for all xbyx plots; In order to
        modify a single topoplot, use setone method).

        """
        for i in xrange(len(self.main_axes)):
            _topo = self._ntopo * i + topo_id
            self.set_topo_single(_topo, t, parent_im_id=i)

    def set_topowins(self, *t_list):
        """
        Set time points for several topomaps (calls self.set() for each value
        in t_list)

        """
        for i, t in enumerate(t_list):
            self.set_topowin(i, t)

    def _window_update(self, mouseevent, parent_ax):
        "update a window (used for mouse-over and for pick)"
        t = mouseevent.xdata
        self._selected_window.update(parent_ax=parent_ax, t=t)
        self._frame.redraw(axes=[self._selected_window.ax])

    def _pick_handler(self, pickevent):
        mouseevent = pickevent.mouseevent
        ax = pickevent.artist
        button = {1:'l', 2:'r', 3:'r'}[mouseevent.button]
        if ax.type == 'window':
            window = self.windows[ax.ID]
            if button == 'l':
                self._selected_window = window
            elif button == 'r':
                Id = window.ax.ID % self._ntopo
                self.set_topowin(Id, None)
            else:
                pass
        elif (ax.type == 'main') and (self._selected_window != None):
            self._selected_window.clear()  # to side track pdf export transparency issue
#            self._window_update(mouseevent, ax)

            # update corresponding windows
            t = mouseevent.xdata
            Id = self._selected_window.ax.ID % self._ntopo
            self.set_topowin(Id, t)

            self._selected_window = None
            self.canvas.draw()

    def _motion_handler(self, mouseevent):
        ax = mouseevent.inaxes
        if ax in self.main_axes:
            if self._selected_window != None:
                self._window_update(mouseevent, ax)


