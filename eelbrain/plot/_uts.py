# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot uniform time-series of one variable."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.colors
import matplotlib.patches
import numpy as np

from .._colorspaces import oneway_colors
from .._data_obj import NDVarArg, Case, CategorialArg, CellArg, IndexArg, Dataset, NDVar, asndvar, cellname, longname
from .._stats.testnd import NDTest
from .._utils import deprecate_ds_arg
from . import _base
from ._base import AxisData, EelFigure, Layout, LegendArg, LegendMixin, PlotType, PlotData, StatLayer, DataLayer, TimeSlicerEF, XAxisMixin, YLimMixin
from ._styles import Style, to_styles_dict


class UTSStat(LegendMixin, XAxisMixin, YLimMixin, EelFigure):
    """
    Plot statistics for a one-dimensional NDVar

    Parameters
    ----------
    y
        One or several dependent variable(s) (one-dimensional NDVar).
    x
        Model: specification of conditions which should be plotted separately.
    xax
        Make separate axes for each category in this categorial model.
    match
        Identifier for repeated measures data.
    sub
        Only use a subset of the data provided.
    data
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    main : func | None
        Measure for the central tendency (function that takes an ``axis``
        argument). The default is numpy.mean.
    error
        Measure of variability to plot. Examples:
        ``sem``: Standard error of the mean;
        ``2sem``: 2 standard error of the mean;
        ``ci``: 95% confidence interval;
        ``99%ci``: 99% confidence interval;
        ``sd``: standard deviation;
        ``all``: Show all traces;
        ``none``: No variability indication.
    within_subject_error
        Within-subject error bars (see Loftus & Masson, 1994; default is
        ``True`` for complete related measures designs, ``False`` otherwise).
    legend
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    labels
        Alternative labels for legend as ``{cell: label}`` dictionary (preserves
        order).
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
    invy
        Invert the y axis (if ``bottom`` and/or ``top`` are specified explicitly
        they take precedence; an inverted y-axis can also be produced by
        specifying ``bottom > top``).
    bottom
        The lower end of the plot's y axis.
    top
        The upper end of the plot's y axis.
    case
        Dimension to treat as case (default is ``'case'``).
    xdim
        Dimension to plot along the x-axis (default is ``'time'``)
    xlim
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    clip
        Clip lines outside of axes (the default depends on whether ``frame`` is
        closed or open).
    color : matplotlib color
        Color if just a single category of data is plotted.
    colors
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; Cells of ``x`` are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as ``x.cells``.
        **dict**: A dictionary mapping each cell in ``x`` to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    error_alpha
        Alpha of the error plot (default 0.3).
    mask : NDVar | {cell: NDVar}
        Mask certain time points. To control appearance of masked regions, set
        ``colors`` using :class:`plot.Style`.
    clusters
        Clusters to add to the plots. The clusters should be provided as
        Dataset, as stored in test results' :py:attr:`.clusters`.
    pmax
        Maximum p-value of clusters to plot as solid.
    ptrend
        Maximum p-value of clusters to plot as trend.
    tight
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title
        Figure title.
    ...
        Also accepts :ref:`general-layout-parameters`.

    Examples
    --------
    - :ref:`exa-utsstat`

    Notes
    -----
    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)

    Examples
    --------
    Single :class:`NDVar` from a Dataset::

        plot.UTSStat('uts', 'A', 'B', match='rm', ds=ds)

    Multiple :class:`NDVar` in different axes

        plot.UTSStat(['uts1', 'uts2'], 'A', match='rm', ds=ds)

    Multiple :class:`NDVar` in a single axes::

        plot.UTSStat([['uts1', 'uts2']], match=True, ds=ds)

    """
    @deprecate_ds_arg
    def __init__(
            self,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            x: CategorialArg = None,
            xax: CategorialArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            main: Callable = np.mean,
            error: Union[str, Callable] = 'sem',
            within_subject_error: bool = None,
            legend: LegendArg = None,
            labels: Dict[CellArg, str] = None,
            axtitle: Union[bool, Sequence[str]] = True,
            xlabel: Union[bool, str] = True,
            ylabel: Union[bool, str] = True,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            invy: bool = False,
            bottom: float = None,
            top: float = None,
            case: str = None,
            xdim: str = None,
            xlim: Union[float, Tuple[float, float]] = None,
            clip: bool = None,
            colors: Dict[CellArg, Any] = None,
            error_alpha: float = 0.3,
            mask: Union[NDVar, Dict[CellArg, NDVar]] = None,
            clusters: Dataset = None,
            pmax: float = 0.05,
            ptrend: float = 0.1,
            **kwargs):
        if case and case != 'case':
            if x or xax or match:
                raise ValueError('x, xax and match cannot be specified with case')
            y = asndvar(y, sub, data)
            dimnames = y.get_dimnames(first=case)
            new_x = y.get_data(dimnames)
            dims = y.get_dims(dimnames[1:])
            y = NDVar(new_x, (Case, *dims))

        plot_data = PlotData.from_stats(y, x, xax, match, sub, data, (xdim,), colors, mask).for_plot(PlotType.LINE)
        xdim, = plot_data.dims

        layout = Layout(plot_data.plot_used, 2, 4, **kwargs)
        EelFigure.__init__(self, plot_data.frame_title, layout)
        if clip is None:
            clip = layout.frame is True

        # create plots
        self._plots = []
        legend_handles = {}
        ymax = ymin = None
        for ax, ax_data in zip(self.axes, plot_data):
            p = AxUTSStat(ax, ax_data, xdim, main, error, within_subject_error, clusters, pmax, ptrend, clip, error_alpha)
            self._plots.append(p)
            legend_handles.update(p.legend_handles)
            ymin = p.vmin if ymin is None else min(ymin, p.vmin)
            ymax = p.vmax if ymax is None else max(ymax, p.vmax)
        self._set_axtitle(axtitle, plot_data)

        # The legend should only display cells with distinct styles: remap legend handles to source cells
        alias_cells = {cell: style.alias for cell, style in plot_data.styles.items() if style.alias}
        if alias_cells:
            legend_handles = {alias_cells[cell]: handle for cell, handle in legend_handles.items()}

        # axes limits
        if top is not None:
            ymax = top
        if bottom is not None:
            ymin = bottom
        if invy:
            ymin, ymax = ymax, ymin
        for p in self._plots:
            p.set_ylim(ymin, ymax)

        self._configure_axis_data('y', plot_data.ct.y, ylabel, yticklabels)
        self._configure_axis_dim('x', plot_data.ct.y.get_dim(xdim), xlabel, xticklabels)
        XAxisMixin._init_with_data(self, ((plot_data.ct.y,),), xdim, xlim)
        YLimMixin.__init__(self, self._plots)
        LegendMixin.__init__(self, legend, legend_handles, labels, alt_sort=colors)
        self._update_ui_cluster_button()
        self._show()

    def _fill_toolbar(self, tb):
        from .._wxgui import wx

        btn = self._cluster_btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        btn.Enable(False)
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._on_show_cluster_info)

        LegendMixin._fill_toolbar(self, tb)

    def _on_show_cluster_info(self, event):
        from .._wxgui import show_text_dialog

        if len(self._plots) == 1:
            clusters = self._plots[0].cluster_plt.clusters
            all_clusters = None
            all_plots_same = True
        else:
            all_clusters = [p.cluster_plt.clusters is None for p in self._plots]
            clusters = all_clusters[0]
            if all(c is clusters for c in all_clusters[1:]):
                all_plots_same = True
            else:
                all_plots_same = False

        if all_plots_same:
            info = str(clusters)
        else:
            info = []
            for i, clusters in enumerate(all_clusters):
                if clusters is None:
                    continue
                title = "Axes %i" % i
                info.append(title)
                info.append('\n')
                info.append('-' * len(title))
                info.append(str(clusters))
            info = '\n'.join(info)

        show_text_dialog(self._frame, info, "Clusters")

    def _update_ui_cluster_button(self):
        if hasattr(self, '_cluster_btn'):
            enable = not all(p.cluster_plt.clusters is None for p in self._plots)
            self._cluster_btn.Enable(enable)

    def set_clusters(
            self,
            clusters: Union[Dataset, None],
            pmax: float = 0.05,
            ptrend: float = None,
            color: Any = '.7',
            ax: int = None,
            y: Union[float, Dict] = None,
            dy: float = None,
            **kwargs,
    ):
        """Add clusters from a cluster test to the plot.

        Parameters
        ----------
        clusters
            A dataset with the clusters to visualize.
            Uses the format of clusters stored in test results'
            :py:attr:`.clusters`.
            This should include at least ``tstart`` and ``tstop`` columns.
            It can also optionally contain an ``effect`` column to use different
            ``color`` and ``y`` levels for different types of clusters.
            Use ``None`` to remove the clusters plotted on a given axis.
        pmax
            Only plot clusters with ``p <= pmax``.
        ptrend
            Maximum p-value of clusters to plot as trend.
        color : matplotlib color | dict
            Color for the clusters, or a ``{effect: color}`` dictionary.
        ax : None | int
            Index of the axes to which the clusters are to be added. If None,
            add the clusters to all axes.
        y
            Visualize clusters as horizontal bars at specific ``y`` levels in
            the plot (default is shaded boxes spanning the whole y-axis).
            As with ``color``, this can be an ``{effect: y}`` dictionary.
        dy
            Height of the bars when using ``y``.
        ...
            Additional arguments for the matplotlib artists (e.g., ``zorder``).

        Notes
        -----
        Previously added clusters will be removed.
        """
        axes = range(len(self.axes)) if ax is None else [ax]

        # update plots
        for ax in axes:
            p = self._plots[ax].cluster_plt
            p.kwargs = kwargs
            p.set_clusters(clusters, False)
            p.set_color(color, False)
            p.set_y(y, dy, False)
            p.set_pmax(pmax, ptrend)
        self.draw()

        self._update_ui_cluster_button()


class UTS(TimeSlicerEF, LegendMixin, YLimMixin, XAxisMixin, EelFigure):
    """Value by time plot for uniform time series (UTS) data

    Parameters
    ----------
    y : (list of) NDVar
        UTS data to plot. For a list (``[a, b]``) each item will be plotted
        separately. To combine multiple UTSs in a common axes, use a nested
        list (``[[a, b]]`` would plot ``a`` and ``b`` together in a common axes).
    xax : categorial
        Make separate axes for each category in this categorial model.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    sub
        Specify a subset of the data.
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
    bottom
        The lower end of the plot's y axis.
    top
        The upper end of the plot's y axis.
    legend
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    labels
        Alternative labels for legend as ``{cell: label}`` dictionary (preserves
        order).
    xlim
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    clip
        Clip lines outside of axes (the default depends on whether ``frame`` is
        closed or open).
    colors
        Dictionary mapping ``y`` names to color, or a single color to use for
        all lines.
    stem
        Plot as stem-plot (default is a line-plot).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    ...
        Also accepts :ref:`general-layout-parameters`.

    Notes
    -----
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
    @deprecate_ds_arg
    def __init__(
            self,
            y: Union[NDVarArg, Sequence, NDTest],
            xax: CategorialArg = None,
            axtitle: Union[bool, Sequence[str]] = True,
            data: Dataset = None,
            sub: IndexArg = None,
            xlabel: Union[bool, str] = True,
            ylabel: Union[bool, str] = True,
            xticklabels: Union[str, int, Sequence[int]] = 'bottom',
            yticklabels: Union[str, int, Sequence[int]] = 'left',
            bottom: float = None,
            top: float = None,
            legend: LegendArg = None,
            labels: Dict[CellArg, str] = None,
            xlim: Union[float, Tuple[float, float]] = None,
            clip: bool = None,
            colors: Union[Any, dict] = None,
            color: Any = None,
            stem: bool = False,
            **kwargs):
        plot_data = PlotData.from_args(y, (None,), xax, data, sub).for_plot(PlotType.LINE)
        xdim = plot_data.dims[0]
        layout = Layout(plot_data.plot_used, 2, 4, **kwargs)
        EelFigure.__init__(self, plot_data.frame_title, layout)
        self._set_axtitle(axtitle, plot_data)
        self._configure_axis_dim('x', xdim, xlabel, xticklabels, data=plot_data.data)
        self._configure_axis_data('y', plot_data, ylabel, yticklabels)
        if clip is None:
            clip = layout.frame is True

        self.plots = []
        legend_handles = {}
        vlims = _base.find_fig_vlims(plot_data.data, top, bottom)

        n_colors = max(map(len, plot_data.data))
        if colors is None:
            if color is None:
                styles = [Style._coerce(color) for color in oneway_colors(n_colors)]
            else:
                styles = [Style._coerce(color)] * n_colors
        elif isinstance(colors, dict):
            styles = to_styles_dict(colors)
        else:
            styles = (Style._coerce(colors),) * n_colors

        for ax, layers in zip(self.axes, plot_data):
            h = AxUTS(ax, layers, xdim, vlims, styles, stem, clip)
            self.plots.append(h)
            legend_handles.update(h.legend_handles)

        self.epochs = plot_data.data
        XAxisMixin._init_with_data(self, plot_data.data, xdim, xlim)
        YLimMixin.__init__(self, self.plots)
        LegendMixin.__init__(self, legend, legend_handles, labels)
        TimeSlicerEF.__init__(self, xdim, plot_data.time_dim)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


class AxUTSStat:

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            data: AxisData,
            xdim: str,
            main: Callable,
            error: Union[str, Callable],
            within_subject_error: bool,
            clusters,
            pmax,
            ptrend,
            clip: bool,
            error_alpha,
    ):
        # stat plots
        self.ax = ax
        self.stat_plots = []
        self.legend_handles = {}

        for layer in data:
            plt = PltUTSStat(ax, layer, xdim, main, error, within_subject_error, clip, error_alpha)
            self.stat_plots.append(plt)
            if plt.main is not None:
                self.legend_handles[layer.style_key] = plt.main[0]

        # cluster plot
        self.cluster_plt = PltUTSClusters(ax, clusters, pmax, ptrend)

        # format y axis
        ax.autoscale(True, 'y')
        ax.autoscale(False, 'x')
        self.vmin, self.vmax = self.ax.get_ylim()

    @property
    def title(self):
        return self.ax.get_title()

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class AxUTS:

    def __init__(
            self,
            ax,
            axis_data: AxisData,
            xdim: str,
            vlims,
            styles: Union[Sequence, Dict],
            stem: bool,
            clip: bool = True,
    ):
        vmin, vmax = _base.find_uts_ax_vlim(axis_data.ndvars, vlims)
        if isinstance(styles, dict):
            styles = [styles[layer.y.name] for layer in axis_data]

        self.legend_handles = {}
        for layer, style in zip(axis_data, styles):
            p = PltUTS(ax, layer, xdim, style, stem, clip)
            self.legend_handles[longname(layer.y)] = p.plot_handle
            contours = layer.y.info.get('contours', None)
            if contours:
                for v, c in contours.items():
                    if v in contours:
                        continue
                    contours[v] = ax.axhline(v, color=c)

        self.ax = ax
        self.set_ylim(vmin, vmax)

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class PltUTS:

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            layer: DataLayer,
            xdim: str,
            style: Style,
            stem: bool = False,
            clip: bool = True,
    ):
        y = layer.y.get_data((xdim,))
        x = layer.y.get_dim(xdim)._axis_data()
        label = longname(layer.y)
        if stem:
            nonzero = y != 0
            nonzero[0] = True
            nonzero[-1] = True
            color = matplotlib.colors.to_hex(style.color)
            self.plot_handle = ax.stem(x[nonzero], y[nonzero], bottom=0, linefmt=color, markerfmt=' ', basefmt='#808080', label=label)
        else:
            kwargs = layer.plot_args(style.line_args)
            self.plot_handle = ax.plot(x, y, label=label, clip_on=clip, **kwargs)[0]

        for y, kwa in _base.find_uts_hlines(layer.y):
            ax.axhline(y, **kwa)


class PltUTSClusters:
    """UTS cluster plot"""
    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            clusters: Dataset,  # 'tstart', 'tstop', 'p', 'effect'
            pmax: float,
            ptrend: float,
            style: Style = None,
    ):
        self.pmax = pmax
        self.ptrend = ptrend
        self.h = []
        self.ax = ax
        self.clusters = clusters
        self.style = style
        self.y = None
        self.dy = None
        self.kwargs = {}
        self.update()

    def set_clusters(self, clusters, update=True):
        self.clusters = clusters
        if update:
            self.update()

    def set_color(self, color, update=True):
        if isinstance(color, dict):
            self.style = to_styles_dict(color)
        else:
            self.style = Style._coerce(color)
        if update:
            self.update()

    def set_pmax(self, pmax, ptrend, update=True):
        self.pmax = pmax
        self.ptrend = ptrend
        if update:
            self.update()

    def set_y(self, y, dy, update=True):
        self.y = y
        self.dy = dy
        if update:
            self.update()

    def update(self):
        h = self.h
        while len(h):
            h.pop().remove()

        clusters = self.clusters
        if clusters is None:
            return

        if self.dy is None:
            bottom, top = self.ax.get_ylim()
            dy = (top - bottom) / 40.
        else:
            dy = self.dy

        p_include = self.ptrend or self.pmax
        for cluster in clusters.itercases():
            if 'p' in cluster:
                p = cluster['p']
                if p > p_include:
                    continue
                alpha = 0.5 if p < self.pmax else 0.2
            else:
                alpha = 0.5

            tstart = cluster['tstart']
            tstop = cluster['tstop']
            effect = cluster.get('effect')
            y = self.y[effect] if isinstance(self.y, dict) else self.y
            style = self.style[effect] if isinstance(self.style, dict) else self.style
            if y is None:
                kwargs = {'zorder': -10, **self.kwargs}
                if style:
                    kwargs = {**style.patch_args, **kwargs}
                h = self.ax.axvspan(tstart, tstop, fill=True, alpha=alpha, **kwargs)
            else:
                if style:
                    kwargs = {**style.patch_args, **self.kwargs}
                else:
                    kwargs = self.kwargs
                h = matplotlib.patches.Rectangle((tstart, y - dy / 2.), tstop - tstart, dy, linewidth=0, **kwargs)
                self.ax.add_patch(h)
            self.h.append(h)


class PltUTSStat:

    def __init__(
            self,
            ax: matplotlib.axes.Axes,
            layer: StatLayer,
            xdim: str,
            main: Callable,
            error: Union[str, Callable],
            within_subject_error: bool,
            clip: bool,
            error_alpha: float,
    ):
        # zorder defaults: 1 for patches, 2 for lines
        label = cellname(layer.cell)
        x = layer.y.get_dim(xdim)._axis_data()
        # plot main
        if callable(main):
            y_main = layer.get_statistic(main)
            kwargs = layer.style.line_args
            if error == 'all':
                lw = kwargs.get('linewidth') or matplotlib.rcParams['lines.linewidth']
                kwargs = {**kwargs, 'linewidth': lw * 2}
            self.main = ax.plot(x, y_main, label=label, clip_on=clip, **kwargs)
        elif error == 'all':
            self.main = y_main = None
        else:
            raise TypeError(f"{main=}")

        # plot error
        if error == 'all':
            y_all = layer.y.get_data((xdim, 'case'))
            self.error = ax.plot(x, y_all, alpha=error_alpha, clip_on=clip, **layer.style.line_args)
        elif error and error != 'none' and len(layer.y.get_dim('case')) > 1:
            if callable(error):
                dev_data = layer.get_statistic(error)
            else:
                dev_data = layer.get_dispersion(error, within_subject_error)
            lower = y_main - dev_data
            upper = y_main + dev_data
            self.error = ax.fill_between(x, lower, upper, linewidth=0, clip_on=clip, **layer.style.fill_args(error_alpha))
        else:
            self.error = None
