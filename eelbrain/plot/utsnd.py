"""
plot.utsnd
==========

plots for multivariate uniform time series

"""

from __future__ import division

import logging
import math

import numpy as np
import scipy as sp
# import matplotlib as mpl
import matplotlib.pyplot as plt

import eelbrain.fmtxt as fmtxt
import eelbrain.vessels.data as _dta

from eelbrain.utils import _basic_ops_
from eelbrain.vessels import colorspaces as _cs

import _base


__hide__ = ['plt', 'math']


# MARK: im arrays
def _plt_im_array(ax, epoch, dims=('time', 'sensor'), colorspace=None,
                  **kwargs):
    handles = []
    colorspace = _base.read_cs_arg(epoch, colorspace)
    data = epoch.get_data(dims)
    if data.ndim > 2:
#        print data.shape
        assert data.shape[0] == 1
        data = data[0]

    if colorspace.cmap:
        im_kwargs = kwargs.copy()
        im_kwargs.update(colorspace.get_imkwargs())
        h = ax.imshow(data, origin='lower', **im_kwargs)
        handles.append(h)

    if colorspace.contours:
        c_kwargs = kwargs.copy()
        c_kwargs.update(colorspace.get_contour_kwargs())
        h = ax.contour(data, **c_kwargs)
        handles.append(h)

    return handles



def _ax_im_array(ax, layers, x='time',  # vmax=None,
                 xlabel=True, ylabel=True, title=None, tick_spacing=.3):
    """
    plots segment data as im

    define a colorspace by supplying one of those kwargs: ``colorspace`` OR
    ``p`` OR ``vmax``

    """
    handles = []
    epoch = layers[0]

    xdim = epoch.get_dim(x)
    if epoch.ndim == 2:
        xdim_i = epoch.dimnames.index(x)
        ydim_i = {1:0, 0:1}[xdim_i]
        y = epoch.dimnames[ydim_i]
    else:
        err = ("Need 2 dimensions, got %i" % epoch.ndim)
        raise ValueError(err)

    ydim = epoch.get_dim(y)
    if y == 'sensor':
        ydim = _dta.var(np.arange(len(ydim)), y)

    map_kwargs = {'extent': [xdim[0], xdim[-1], ydim[0], ydim[-1]],
                  'aspect': 'auto'}

    # plot
    for l in layers:
        h = _plt_im_array(ax, l, dims=(y, x), **map_kwargs)
        handles.append(h)

    if xlabel:
        if xlabel is True:
            xlabel = xdim.name
        ax.set_xlabel(xlabel)

    if ylabel:
        if ylabel is True:
            ylabel = ydim.name
        ax.set_ylabel(ylabel)

    # x-ticks
    tickstart = math.ceil(xdim[0] / tick_spacing) * tick_spacing
    tickend = xdim[-1] + tick_spacing / 1e4
    ticklabels = np.arange(tickstart, tickend, tick_spacing)
    ax.xaxis.set_ticks(ticklabels)
    ax.x_fmt = "t = %.3f s"

    # y-ticks
    if y == 'sensor':  # make sure y-ticklabels are all integers
        locs = ax.yaxis.get_ticklocs()
        if any(locs != locs.round()):
            idx = np.where(locs == locs.round())[0]
            locs = locs[idx]
            labels = map(lambda x: str(int(x)), locs)
            ax.yaxis.set_ticks(locs)
            ax.yaxis.set_ticklabels(labels)

    # title
    if title is None:
        if plt.rcParams['text.usetex']:
            title = fmtxt.texify(epoch.name)
        else:
            title = epoch.name
    ax.set_title(title)

    return handles


def array(epochs, xlabel=True, ylabel=True,
          w=4, h=3, dpi=50):
    """
    plots uts data to a rectangular grid. I data has only 1 dimension, the
    x-axis defines epochs.

    **Arguments:**

    h : scalar
        plot height in inches
    w : scalar
        plot width in inches

    """
    epochs = _base.unpack_epochs_arg(epochs, 2)

    n_plots = len(epochs)
#    n = round(np.sqrt(n_plots))
    figsize = (n_plots * w, h)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(.1, .1, .95, .95, .1, .4)

    for i, layers in enumerate(epochs):
        ax = fig.add_subplot(1, n_plots, i + 1)

        _ylabel = ylabel if i == 1 else None
        _xlabel = xlabel if i == n_plots - 1 else None

        _ax_im_array(ax, layers, xlabel=_xlabel, ylabel=ylabel)

    fig.tight_layout()
    fig.show()
    return fig



def _t_axes(rect, xmin= -.5, xmax=1., vmax=1.5, vbar=1., markevery=.5, ticks=False):
    """
    creates and returns a new t-axes using rect

    vbar: extent of the vertical bar at the origin (+vbar to -vbar)

    """
    ax = plt.axes(rect, frameon=False)
    ax.set_axis_off()

    # vertical bar
    plt.plot([0, 0], [-vbar, vbar], 'k',
           marker='_', mec='k', mew=1, mfc='k')

    # horizontal bar
    xdata = np.arange(-markevery, xmax + 1e-5, markevery)
    xdata[0] = xmin
    plt.plot(xdata, xdata * 0, 'k', marker='|', mec='k', mew=1, mfc='k')
    logging.debug("xdata: %s" % str(xdata))

    # labels
    if ticks:
        ax.text(xmax, vbar * .1, r"$%s s$" % xmax, verticalalignment='bottom', horizontalalignment='center')
        ax.text(-vbar * .05, vbar, r"$%s \mu V$" % vbar, verticalalignment='center', horizontalalignment='right')
        ax.text(-vbar * .05, -vbar, r"$-%s \mu V$" % vbar, verticalalignment='center', horizontalalignment='right')
    ax.set_ylim(-vmax, vmax)
    ax.set_xlim(xmin, xmax)
    return ax



def _axgrid_sensors(sensorLocs2d, figsize=_base.defaults['figsize'],
                    spacing=.2, frame=.01, figvstretch=1,
                    header=0, footer=0,  # in inches
                    axes_legend_loc='lower left', **axkwargs):
    """
    creates topographocally distributed t-axes

     returns
        - list of t-axes
        - axes-legend (indicating t and V) (or None if axes_legend_loc == False)
    """
    # determine figure size and create figure
    sensorLocs2d[:, 1] *= figvstretch
    x, y = sensorLocs2d.max(axis=0) - sensorLocs2d.min(axis=0)
    ratio = (x + spacing) / (y + spacing)
    x_size, y_size = figsize
    if x_size == -1:
        x_size = y_size * ratio
    elif y_size == -1:
        y_size = x_size / ratio
    # take into account footer & header
    y_size += footer + header
    relative_footer = footer / float(y_size)
    relative_header = header / float(y_size)
    logging.debug(" _axgrid_sensors determined figsize %s x %s" % (x_size, y_size))
    fig = plt.figure(figsize=(x_size, y_size))
    # determine axes locations
    locs = sensorLocs2d
    # normalize range to 0--1
    locs -= locs.min(0)
    locs /= locs.max(0)
    # locs--> lower left points of axes
    locs[:, 0] *= (1 - spacing - 2 * frame)
    locs[:, 0] += frame
    locs[:, 1] *= (1 - spacing - 2 * frame - relative_header - relative_footer)
    locs[:, 1] += frame + relative_footer
    # print locs
    # add axes
    axes = [ _t_axes([x, y, spacing, spacing], **axkwargs) for x, y in locs ]
    if axes_legend_loc:
        x, y = _base._loc(axes_legend_loc, size=(1.1 * spacing, spacing), frame=frame)
        axes_legend = _t_axes([x, y, spacing, spacing], ticks=True, **axkwargs)
    else:
        axes_legend = None
    return axes, axes_legend


def _plt_uts(ax, epoch,
             sensors=None,  # sensors (ID) to plot
             lp=None,  # line-properties
             test_epoch=False, p=.05, softStats=False,  # testWindowFreq='max',
             sem=None,  # 'sem' (float multiplier)
             plotLabel=False,
             **plot_kwargs):
    """
    plots a uts plot for a single epoch


    Arguments
    ---------

    lp: dictionary (line-properties)
        any keyword-arguments for matplotlib plot
    sem: = None or float
        plot standard error of the mean (e.g., ``sem=2`` plots the mean +/- 2
        sem)
    test_epoch:
        submit a test_epoch to add to plot (efficient because usually
        _ax_utsStats is called more than once for several epochs
    plottype:
        ``'mean'``: plots the mean for each stats epoch
        ``'all'``:  plots all data traces contained in the stats, colors mark the
        epoch NOT MAINTAINED

    NOT IMPLEMENTED epochs: submit mean epochs

    """
    Y = epoch.get_data(('time', 'sensor'))
    if sensors is not None:
        Y = Y[:, sensors]
    T = epoch.time  # .x[...,None]

    handles = ax.plot(T, Y, label=epoch.name, **plot_kwargs)

    if plotLabel:
        Ymax = np.max(Y)
        ax.text(T[0] / 2, Ymax / 2, plotLabel, horizontalalignment='center')

    return handles



def _plt_extrema(ax, epoch, **plot_kwargs):
    data = epoch.get_data(('time', 'sensor'))
    Ymin = data.min(1)
    Ymax = data.max(1)
    T = epoch.time

    handle = ax.fill_between(T, Ymin, Ymax, **plot_kwargs)
    ax.set_xlim(T[0], T[-1])

    return handle


def _ax_butterfly(ax, layers, sensors=None, ylim=None, extrema=False,
                  title='{name}', xlabel=True, ylabel=True, color=None,
                  **plot_kwargs):
    """
    Arguments
    ---------

    ylim:
        y axis limits (scalar or (min, max) tuple)

    """
    handles = []

    xmin = []
    xmax = []
    name = ''
    for l in layers:
        colorspace = _base.read_cs_arg(l)
        if not colorspace.cmap:
            continue

        if color is None:
            plot_kwargs['color'] = l.properties.get('color', 'k')
        elif color is True:
            pass  # no color kwarg to use mpl's color_cycle
        else:
            plot_kwargs['color'] = color

        # plot
        if extrema:
            h = _plt_extrema(ax, l, **plot_kwargs)
        else:
            h = _plt_uts(ax, l, sensors=sensors, **plot_kwargs)

        handles.append(h)
        xmin.append(l.time[0])
        xmax.append(l.time[-1])

        if not name:
            name = getattr(l, 'name', '')

    # axes decoration
    l = layers[0]
    if xlabel is True:
        xlabel = 'Time [s]'
    if ylabel is True:
        ylabel = l.properties.get('unit', None)
    if ylim is None:
        ylim = l.properties.get('ylim', None)

    ax.set_xlim(min(xmin), max(xmax))

    if ylim:
        if np.isscalar(ylim):
            ax.set_ylim(-ylim, ylim)
        else:
            y_min, y_max = ylim
            ax.set_ylim(y_min, y_max)

    if xlabel not in [False, None]:
        ax.set_xlabel(xlabel)
    if ylabel not in [False, None]:
        ax.set_ylabel(ylabel)

#    ax.yaxis.set_offset_position('right')
    ax.yaxis.offsetText.set_va('top')

    ax.x_fmt = "t = %.3f s"
    if isinstance(title, str):
        ax.set_title(title.format(name=name))

    return handles


class butterfly(_base.eelfigure):
    "Plot data in a butterfly plot."
    def __init__(self, epochs, sensors=None, ylim=None, w=4, h=2, dpi=90,
                 ncol=3, title=None, axtitle='{name}',
                 xlabel=True, ylabel=True, color=None):
        """
        Parameters
        ----------

        sensors: None or list of sensor IDs
            sensors to plot (``None`` = all)
        w, h : scalar
            width and height of the individual aces in inches
        color: (mpl color)
            default (``None``): use segment color if available, otherwise
            black; ``True``: alternate colors (mpl default)
        title: bool or string
            Title for the axes. If ``True``, the segment's name is used.
        ylim:
            scalar or (min, max) tuple specifying the y-axis limits (the
            default ``None`` leaves mpl's default limits unaffected)

        """
        epochs = _base.unpack_epochs_arg(epochs, 2)

        n_plots = len(epochs)
        nrow = math.ceil(n_plots / ncol)
        ncol = min(n_plots, ncol)

        figsize = (w * ncol, h * nrow)
        win_title = title if isinstance(title, str) else 'plot.butterfly'
        super(butterfly, self).__init__(title=win_title, figsize=figsize, dpi=dpi)
        fig = self.figure

        for i, layers in enumerate(epochs):
            ax = fig.add_subplot(nrow, ncol, i + 1)

            if i == n_plots - 1:
                _xlabel = xlabel
            else:
                _xlabel = None

            _ax_butterfly(ax, layers, sensors=sensors, ylim=ylim, title=axtitle,
                          xlabel=_xlabel, ylabel=ylabel, color=color)

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        self._show()





def _ax_sensor(segments, sensor, labelVar=None, **kwargs):
    """
    NOT MAINTAINED
    plots one sensor (index) for a list of segments
    return plot
    """
    # make sure segment and sensors are iterable
    if type(segments) != list:
        segments = [segments]
    if labelVar == None:
        labelVar = segments[0].experiment.subjectVariable
    print "ndim:", segments[0].ndim
    if segments[0].hasStatistics:
        return plotSensorStats(segments, sensor, **kwargs)
    elif segments[0].ndim == 1:
        plots = []
        for s in segments:
            try:
                label = s[labelVar]
            except:
                label = "no name"
            print "plotting segment", s
            plots.append(plt.plot(s[:, sensor], label=label, **kwargs))
        return plots
    elif segments[0].ndim == 2:
        raise NotImplementedError("ndim 2 not implemented")







