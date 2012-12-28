"""
Base Class
==========

The module defines a baseclass for eelbrain plots (:py:class:`eelfigure`).
The figure and canvas are created either by a
:py:class:`eelbrain.wxutils.mpl_canvas.CanvasFrame` (:py:class:`wx.Frame`
subclass), or by a :py:class:`mpl_figure` which is based on standard
:py:mod:`pylab` functionality.

If the mpl figure is used, pyplot.show() is called after the plot is done.
The module attribute ``show_block_arg`` is submitted to
``plt.show(block=show_block_arg)``.


Implementation
==============

Types of plotting functions:

...
    creates new figure
_ax_...
    creates the layout for one axes object (provided as the ``ax`` kwarg)
    and plots a collection of data of a certain type.
_plt_...
    plots a collection of data of a certain type to an axes object, but does
    not take care of the general axes layout


OLD:

_row_...
    (row=?, nrows=?)     fill a whole row
?col_...


TODO
----

- generalized management of spacing


"""

import os
import shutil
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL

from eelbrain import vessels as _vsl
from eelbrain.vessels import data as _dt

try:
    from eelbrain.wxutils.mpl_canvas import CanvasFrame
    backend = 'wx'
except:
    backend = 'mpl'


defaults = {
          'figsize':(6, -1),  # was 7
        }
title_kwargs = {'size': 18,
                'family': 'serif'}
figs = []  # store callback figures (they need to be preserved)
show_block_arg = True  # if the mpl figure is used, this is submitted to plt.show(block=show_block_arg)


def unpack_epochs_arg(ndvars, ndim, dataset=None, levels=1):
    """
    Returns a nested list of epochs (through the summary method)
    """
    ndvars = getattr(ndvars, '_default_plot_obj', getattr(ndvars, 'all', ndvars))
    if not isinstance(ndvars, (tuple, list)):
        ndvars = [ndvars]

    if levels > 0:
        return [unpack_epochs_arg(v, ndim, dataset, levels - 1) for v in ndvars]
    else:
        out = []
        for ndvar in ndvars:
            if isinstance(ndvar, basestring):
                ndvar = dataset[ndvar]

            if ndvar.ndim == ndim + 1:
                if ndvar.has_case:
                    if len(ndvar) == 1:
                        ndvar = ndvar.summary(name='{name}')
                    else:
                        ndvar = ndvar.summary()

            if ndvar.ndim == ndim:
                pass
            else:
                err = ("Plot requires ndim=%i; %r ndim==%i" %
                       (ndim, ndvar, ndvar.ndim))
                raise _dt.DimensionMismatchError(err)
            out.append(ndvar)
        return out



def read_cs_arg(epoch, colorspace=None):
    if (colorspace is None):
        if 'colorspace' in epoch.properties:
            colorspace = epoch.properties['colorspace']
        else:
            colorspace = _vsl.colorspaces.get_default()

    return colorspace




class mpl_figure:
    "cf. wxutils.mpl_canvas"
    def __init__(self, **fig_kwargs):
        "creates self.figure and self.canvas attributes and returns the figure"
        self.figure = plt.figure(**fig_kwargs)
        self.canvas = self.figure.canvas
        figs.append(self)

    def Close(self):
        plt.close(self.figure)

    def SetStatusText(self, text):
        pass

    def Show(self):
        if mpl.get_backend() == 'WXAgg':
            plt.show(block=show_block_arg)

    def redraw(self, axes=[], artists=[]):
        "Adapted duplicate of mpl_canvas.FigureCanvasPanel"
        self.canvas.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.canvas.blit(extent)
        for artist in artists:
            ax = artist.get_axes()
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.canvas.blit(extent)

    def store_canvas(self):
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)


# MARK: figure composition

def _loc(name, size=(0, 0), title_space=0, frame=.01):
    """
    takes a loc argument and returns x,y of bottom left edge

    """
    if isinstance(name, basestring):
        y, x = name.split()
    # interpret x
    elif len(name) == 2:
        x, y = name
    else:
        raise NotImplementedError("loc needs to be string or len=2 tuple/list")
    if isinstance(x, basestring):
        if x == 'left':
            x = frame
        elif x in ['middle', 'center', 'centre']:
            x = .5 - size[0] / 2
        elif x == 'right':
            x = 1 - frame - size[0]
        else:
            raise ValueError(x)
    # interpret y
    if isinstance(y, basestring):
        if y in ['top', 'upper']:
            y = 1 - frame - title_space - size[1]
        elif y in ['middle', 'center', 'centre']:
            y = .5 - title_space / 2. - size[1] / 2.
        elif y in ['lower', 'bottom']:
            y = frame
        else:
            raise ValueError(y)
    return x, y


class eelfigure(object):
    """
    Parent class for eelbrain figures.

    In order to subclass:

     - find desired figure properties and then use them to initialize
       the eelfigure superclass; then use the
       :py:attr:`eelfigure.figure` and :py:attr:`eelfigure.canvas` attributes.
     - end the initialization by calling `eelfigure._show()`
     - add the :py:meth:`_fill_toolbar` method


    """
    def __init__(self, title="Eelbrain Figure", **fig_kwargs):
        # find the right frame
        frame = None
        self._is_wx = False
        if backend == 'wx':
            try:
                frame = CanvasFrame(title=title, eelfigure=self, **fig_kwargs)
                self._is_wx = True
            except:
                pass
        if frame is None:
            frame = mpl_figure(**fig_kwargs)

        # store attributes
        self._frame = frame
        self.figure = frame.figure
        self.canvas = frame.canvas

        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)

    def _get_statusbar_text(self, event):
        "subclass to add figure-specific content"
        return '%s'

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        self._frame.SetStatusText(':-)')

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        ax = event.inaxes
        if ax:
            y_fmt = getattr(ax, 'y_fmt', 'y = %.3g')
            x_fmt = getattr(ax, 'x_fmt', 'x = %.3g')
            # update status bar
            y_txt = y_fmt % event.ydata
            x_txt = x_fmt % event.xdata
            pos_txt = ',  '.join((x_txt, y_txt))

            txt = self._get_statusbar_text(event)
            self._frame.SetStatusText(txt % pos_txt)

    def _show(self):
        self.draw()
        self._frame.Show()

    def _fill_toolbar(self, tb):
        """
        Subclasses should add their toolbar items in this function which
        is called by CanvasFrame.FillToolBar()

        """
        pass

    def close(self):
        self._frame.Close()

    def draw(self):
        self._frame.canvas.draw()



class subplot_figure(eelfigure):
    def _show(self, figtitle=None):
        self.figure.tight_layout()
        if figtitle:
            t = self.figure.suptitle(figtitle)
            trans = self.figure.transFigure.inverted()
            bbox = trans.transform(t.get_window_extent(self.figure.canvas.renderer))
            print bbox
            t_bottom = bbox[0, 1]
            self.figure.subplots_adjust(top=1 - 2 * (1 - t_bottom))

        super(subplot_figure, self)._show()



class legend(eelfigure):
    def __init__(self, handles, labels, dpi=90, figsize=(2, 2)):
        super(legend, self).__init__(title="Legend", dpi=dpi, figsize=figsize)

        self.legend = self.figure.legend(handles, labels, loc=2)

        self._show()


def unpack(Y, X):
    "Returns a list of Y[cell] corresponding to the cells in X"
    epochs = []
    for cell in X.cells:
        y = Y[X == cell]
        y.name = cell
        epochs.append(y)
    return epochs


class ImageTiler(object):
    """
    http://stackoverflow.com/q/4567409/166700

    """
    def __init__(self, dest, ncol, nrow=1):
        self.dest = dest
        self.ncol = ncol
        self.nrow = nrow
        _, self.ext = os.path.splitext(dest)
        self.tempdir = None
        self.files = []
        self.n_files = ncol * nrow
        self._i_temp = 0

    def __del__(self):
        shutil.rmtree(self.tempdir)

    def _get_tempdir(self):
        if not self.tempdir:
            self.tempdir = tempfile.mkdtemp()
        return self.tempdir

    def add_fname(self, fname):
        self.files.append(fname)
        if len(self.files) < self.n_files:
            return

        # finalize
        images = []
        colw = [0] * self.ncol
        rowh = [0] * self.nrow
        for r in xrange(self.nrow):
            row = []
            for c in xrange(self.ncol):
                fname = self.files.pop(0)
                im = PIL.Image.open(fname)
                row.append(im)
                colw[c] = max(colw[c], im.size[0])
                rowh[r] = max(rowh[r], im.size[1])
            images.append(row)

        cpos = np.cumsum([0] + colw)
        rpos = np.cumsum([0] + rowh)
        out = PIL.Image.new('RGB', (cpos[-1], rpos[-1]))
        for r, row in enumerate(images):
            for c, im in enumerate(row):
                out.paste(im, (cpos[c], rpos[r]))
        out.save(self.dest)

    def get_temp_fname(self):
        fname = os.path.join(self._get_tempdir(), str(self._i_temp) + self.ext)
        self._i_temp += 1
        return fname

