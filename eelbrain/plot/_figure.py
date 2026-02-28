"""Figures for custom plots"""
from functools import wraps
from numbers import Real

import matplotlib.figure
import matplotlib.patches
import numpy as np

from ._base import EelFigure, Layout, XAxisMixin, format_axes


class Figure(EelFigure):
    """Empty figure

    Parameters
    ----------
    nax : int (optional)
        Create this many axes (default is to not create any axes).
    ...
        Also accepts :ref:`general-layout-parameters`.
    autoscale : bool
        Autoscale data axes (default False).
    """

    def __init__(self, nax=0, **kwargs):
        layout = Layout(nax, 1, 2, **kwargs)
        EelFigure.__init__(self, None, layout)

    def show(self):
        self._show()


class XFigure(XAxisMixin, Figure):

    def __init__(self, nax, xmin, xmax, xlim, **kwargs):
        Figure.__init__(self, nax, **kwargs)
        self._args = (xmin, xmax, xlim)

    def show(self):
        XAxisMixin.__init__(self, *self._args)
        Figure.show(self)


class AbsoluteLayoutFigure(XAxisMixin, Figure):
    """Layout for plots sharing y-axis scaling but with different limits

    Parameters
    ----------
    y_per_inch
        Y-axis scaling.
    xlim
        Set default x-axis limits for axes.
    ystep
        Space between ticks on the y-axis. Ticks will be aligned to include 0.
        Leave this parameter unspecified to use the :mod:`matplotlib` default.
    kwargs
        Other :class:`Figure` parameters.

    Attributes
    ----------
    h
        Figure height in inches (determine matplotlib ``y`` coordinate:
        ``y_relative = y_inches / figure.h``).
    w
        Figure width in inches (determine matplotlib ``x`` coordinate:
        ``x_relative = x_inches / figure.w``).

    Notes
    -----
    Usage:

     - Create ``TRFFigure``
     - Add axes using :meth:`.add_axes`
     - Finish by calling :meth:`.finalize`
    """

    def __init__(
            self,
            y_per_inch: Real,
            xlim: tuple[Real, Real] = None,
            ystep: float = None,  # spacing of y ticks
            **kwargs):
        self.y_per_inch = y_per_inch
        kwargs.setdefault('tight', False)
        Figure.__init__(self, **kwargs)
        self._xlim = xlim
        self._ylims = []
        self._ystep = ystep
        self.w = self._layout.w
        self.h = self._layout.h

    def add_axes(
            self,
            y_origin: Real,
            ylim: Real | tuple[Real, Real],
            left: Real,
            width: Real,
            frame: bool | str = 't',
            yaxis: bool = True,
            **kwargs,  # for matplotlib Axes
    ):
        """Add axes based on y-axis (origin and extent in figure scaling)

        Parameters
        ----------
        y_origin
            Axes origin (y = 0) location from the bottom of the figure.
        ylim
            Data limits.
        left
            Left axes border from the left figure border.
        width
            Axes width.
        **
            General axes parameters and
            :class:`matplotlib.figure.Figure.add_axes` parameters.
        """
        if isinstance(ylim, tuple):
            ymin, ymax = ylim
        elif isinstance(ylim, Real):
            ymin, ymax = -ylim, ylim
            ylim = (ymin, ymax)
        else:
            raise TypeError(f"{ylim=}")

        # [left, bottom, width, height]
        rect = [
            left / self._layout.w,
            (y_origin + (ymin / self.y_per_inch)) / self._layout.h,
            width / self._layout.w,
            (ymax - ymin) / self.y_per_inch / self._layout.h,
        ]
        kwargs.setdefault('autoscale_on', False)
        ax = self.figure.add_axes(rect, ylim=ylim, **kwargs)
        format_axes(ax, frame, yaxis)
        ax.patch.set_visible(False)
        if self._ystep is not None:
            ytick_start = (ymin // self._ystep) * self._ystep
            ytick_stop = (ymax // self._ystep + 1) * self._ystep
            ticks = np.arange(ytick_start, ytick_stop, self._ystep)
            if isinstance(self._ystep, int):
                ticks = ticks.astype(int)
            ax.set_yticks(ticks)
        self.axes.append(ax)
        self._ylims.append(ylim)
        return ax

    def add_axes_rect(
            self,
            left: Real,
            bottom: Real,
            width: Real,
            height: Real,
            **kwargs,
    ):
        """Add axes based on rectangle (all values in inches)

        Parameters
        ----------
        left
            Left axes border from the left figure border.
        bottom
            Axes bottom from the figure bottom.
        width
            Axes width.
        height
            Axes height.
        **
            :class:`matplotlib.figure.Figure.add_axes` parameters.
        """
        rect = [
            left / self._layout.w,
            bottom / self._layout.h,
            width / self._layout.w,
            height / self._layout.h,
        ]
        ax = self.figure.add_axes(rect, **kwargs)
        ax.patch.set_visible(False)
        self.axes.append(ax)
        self._ylims.append(None)
        return ax

    def finalize(
            self,
            outline: bool = False,
    ):
        """Finalize figure creation

        Parameters
        ----------
        outline
            Draw the outline of the figure (Area that will be exported when
            saving the figure). This is mainly useful for fine-tuning the figure
            size in Jupyter, which crops the display area based on figure
            elements rather than actual figure size.
        """
        for ax, ylim in zip(self.axes, self._ylims):
            if ylim is not None:
                ax.set_ylim(ylim)
        XAxisMixin.__init__(self, None, None, self._xlim, self.axes)
        if outline:
            self.draw_outline()
        self._show()

    @wraps(matplotlib.figure.Figure.text)
    def text(self, x, y, *args, **kwargs):
        "Add text at ``x, y`` in absolute coordinates (otherwise like :meth:`matplotlib.figure.Figure.text`)"
        return self.figure.text(x / self.w, y / self.h, *args, **kwargs)
