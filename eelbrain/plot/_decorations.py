# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from typing import Any, Literal

import matplotlib.axes
import matplotlib.patches

from .._stats import test


def figure_outline(color='k', figure=None, **kwargs):
    """Draw the outline of the current figure

    Mainly for fine-tuning the figure layout in Jupyter, which crops
    the display area based on figure elements rather than actual figure size.
    """
    if figure is None:
        from matplotlib import pyplot
        figure = pyplot.gcf()
    kwargs.setdefault('fc', 'none')
    artist = matplotlib.patches.Rectangle((0, 0), 1, 1, ec=color, **kwargs)
    figure.add_artist(artist)


def mark_difference(
        x1: float,
        x2: float,
        y: float,
        mark: float | str = None,
        dy: float = 0,
        color: Any = None,
        nudge: bool | float = None,
        location: Literal['top', 'bottom', 'left', 'right'] = 'top',
        ax: matplotlib.axes.Axes = None,
        line_args: dict = None,
        **text_args,
):
    """Mark a pair of categories with a line and a label

    Parameters
    ----------
    x1
        Location on category axis.
    x2
        Second location on category axis.
    y
        Level above which to plot the bar.
    mark
        Text label, or p-value to automatically determine the label and
        ``color``.
    dy
        Length of vertical ticks on each side of the bar (offsets the
        location of the bar itself to ``y + dy``).
    color
        Color for bar and ``label``.
    nudge
        Nudge the edges of the bar inwards to allow multiple bars
        side-by-side on the same level of ``y``.
    location
        Location relative to plots.
    ax
        Axes to which to plot (default is ``pyplot.gca()``).
    line_args
        Additional parameters for :func:`matplotlib.pyplot.plot`.
    ...
        All other parameters are used to plot the text label with
        :meth:`matplotlib.axes.Axes.text`.
    """
    if ax is None:
        from matplotlib import pyplot
        ax = pyplot.gca()

    if isinstance(mark, str):
        label = mark
    elif mark is None:
        label = ''
    else:
        p = mark
        n_stars = test._n_stars(p)
        if not n_stars:
            return
        if color is None:
            color = ('#FFCC00', '#FF6600', '#FF3300')[n_stars - 1]
        label = '*' * n_stars
        text_args.setdefault('size', matplotlib.rcParams['font.size'] * 1.5)

    if color is None:
        color = 'k'

    if x1 > x2:
        x1, x2 = x2, x1

    if nudge:
        if nudge is True:
            nudge = 0.025
        x1 += nudge
        x2 -= nudge

    if location in ('bottom', 'left'):
        dy = -dy

    x_text = (x1 + x2) / 2
    if dy:
        xs, ys = [x1, x1, x2, x2], [y, y + dy, y + dy, y]
        y_text = y + dy
    else:
        xs, ys = [x1, x2], [y, y]
        y_text = y

    if location in ('right', 'left'):
        xs, ys = ys, xs
        x_text, y_text = y_text, x_text

    if x1 != x2:
        line_args_ = {'color': color, 'clip_on': False}
        if line_args:
            line_args_.update(line_args)
        ax.plot(xs, ys, **line_args_)
    if label:
        rotation = {'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[location]
        args = {'ha': 'center', 'va': 'center', 'rotation': rotation, 'clip_on': False, **text_args}
        ax.text(x_text, y_text, label, color=color, **args)
