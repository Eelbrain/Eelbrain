.. currentmodule:: eelbrain

*******
Recipes
*******

.. contents:: Contents
   :local:


^^^^^^^^^^^^^^^^^^^^^
Plots for Publication
^^^^^^^^^^^^^^^^^^^^^

Style
-----

.. seealso::

   * Example on :ref:`exa-customizing-plots`
   * Matplotlib page on `styles <https://matplotlib.org/tutorials/introductory/
     customizing.html>`_

In order to produce multiple plots with consistent style it is useful to set
some :mod:`matplotlib` options globally. One way to do this is by updating
:data:`matplotlib.rcParams` at the beginning of a script/notebook, e.g.::

    from matplotlib import pyplot

    FONT = 'Arial'
    FONT_SIZE = 8
    LINEWIDTH = 0.5
    pyplot.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.transparent': True,
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': FONT,
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        # line width
        'axes.linewidth': LINEWIDTH,
        'grid.linewidth': LINEWIDTH,
        'lines.linewidth': LINEWIDTH,
        'patch.linewidth': LINEWIDTH,
        'xtick.major.width': LINEWIDTH,
        'xtick.minor.width': LINEWIDTH,
        'ytick.major.width': LINEWIDTH,
        'ytick.minor.width': LINEWIDTH,
    })


Plot sizes
----------

Matplotlib's :func:`~matplotlib.pyplot.tight_layout` functionality provides an easy way for plots to fill the available space, and most Eelbrain plots use it by default.
However, when trying to generate multiple figures with identical scaling it can lead to unwanted differences between figures.

For a consistent layout, set the font size through ``rcParams``, and then control the size of plots and figures using the corresponding parameters (``w, h, axw, axh``).
Subplot placement can also be specified in absolute scale through the ``margins`` parameter (see :ref:`general-layout-parameters`).


Various
-------

If a script produces several plots and there is no need to show them on the
screen, they can be kept hidden through the ``show=False`` argument::

    p = plot.UTSStat('uts', 'A', data=data, w=5, tight=False, show=False)

When working on more complex figures, it is often desirable to save the legend
separately and combine it in a layout application::

    p = plot.UTSStat('uts', 'A', data=data, w=5, tight=False, show=False, legend=False)
    p.save('plot.pdf', facecolor="none")
    p.save_legend('legend.pdf', facecolor="none")
