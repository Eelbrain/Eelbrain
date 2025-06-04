# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""PySurfer Brain subclass to embed in Eelbrain"""
from functools import cached_property, partial
import os
import sys
import packaging.version
from tempfile import mkdtemp
from typing import Any, Callable, Literal, Sequence, Union
from time import time, sleep
from warnings import warn

from matplotlib.colors import ListedColormap, Colormap, to_rgb, to_rgba
from matplotlib.image import imsave
from mne.io.constants import FIFF
import numpy as np
import scipy.ndimage

from .._data_obj import NDVar, SourceSpace, UTS, asndvar
from .._exceptions import KeysMissing
from .._text import ms
from .._types import PathArg
from .._utils import IS_OSX
from ..fmtxt import Image
from ..mne_fixes import reset_logger
from ._base import CONFIG, TimeSlicer, AxisScale, do_autorun, find_fig_cmaps, find_fig_vlims, fix_vlim_for_cmap, use_inline_backend
from ._color_luts import p_lut
from ._colors import ColorBar, ColorList
from ._styles import colors_for_oneway

# Traits-GUI related imports
# --------------------------
# - Set ETS toolkit before importing traits-GUI
# - Readthedocs does not support mayavi import, so we can't use surfer
# - if this is the first surfer import, lower screen logging level
first_import = 'surfer' not in sys.modules
INLINE_DISPLAY = False
try:
    from traits.trait_base import ETSConfig
    if use_inline_backend() or not CONFIG['eelbrain']:
        ETSConfig.toolkit = 'qt'
        INLINE_DISPLAY = True
        from mayavi import mlab
        mlab.options.offscreen = True
    else:
        ETSConfig.toolkit = 'wx'
    import surfer
except ImportError as exception:
    from . import _mock_surfer as surfer
    warn(f"Error importing PySurfer: {exception}")
    SURFER_IMPORTED = False
else:
    SURFER_IMPORTED = True
    if first_import:
        reset_logger(surfer.utils.logger)
del first_import


HEMI_ID_TO_STR = {FIFF.FIFFV_MNE_SURF_LEFT_HEMI: 'lh',
                  FIFF.FIFFV_MNE_SURF_RIGHT_HEMI: 'rh'}
OTHER_HEMI = {'lh': 'rh', 'rh': 'lh'}
# default size
BRAIN_H = 250
BRAIN_W = 300


def get_source_dim(ndvar):
    if isinstance(ndvar, SourceSpace):
        return ndvar
    elif isinstance(ndvar, NDVar):
        for source in ndvar.dims:
            if isinstance(source, SourceSpace):
                return source
        raise ValueError("ndvar has no surface-based SourceSpace dimension")
    else:
        raise TypeError(f"{ndvar=}; needs to be NDVar or surface-based SourceSpace")


class Brain(TimeSlicer, surfer.Brain):
    """PySurfer :class:`Brain` subclass returned by :mod:`plot.brain` functions

    PySurfer :class:`surfer.Brain` subclass adding Eelbrain GUI integration and
    methods to visualize data in :class:`NDVar` format.

    Parameters
    ----------
    subject : str
        Subject name.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        'both': both hemispheres are shown in the same window;
        'split': hemispheres are displayed side-by-side in different viewing
        panes.
    surf : str
        Freesurfer surface mesh name (ie 'white', 'inflated', etc.).
    title : str
        Title for the window.
    cortex : str, tuple, dict, or None
        Specifies how the cortical surface is rendered. Options:

            1. The name of one of the preset cortex styles:
               ``'classic'`` (default), ``'high_contrast'``,
               ``'low_contrast'``, or ``'bone'``.
            2. A color-like argument to render the cortex as a single
               color, e.g. ``'red'`` or ``(0.1, 0.4, 1.)``. Setting
               this to ``None`` is equivalent to ``(0.5, 0.5, 0.5)``.
            3. The name of a colormap used to render binarized
               curvature values, e.g., ``Grays``.
            4. A list of colors used to render binarized curvature
               values. Only the first and last colors are used. E.g.,
               ['red', 'blue'] or [(1, 0, 0), (0, 0, 1)].
            5. A container with four entries for colormap (string
               specifiying the name of a colormap), vmin (float
               specifying the minimum value for the colormap), vmax
               (float specifying the maximum value for the colormap),
               and reverse (bool specifying whether the colormap
               should be reversed. E.g., ``('Greys', -1, 2, False)``.
            6. A dict of keyword arguments that is passed on to the
               call to surface.
    alpha : float in [0, 1]
        Alpha level to control opacity of the cortical surface.
    background, foreground : matplotlib colors
        color of the background and foreground of the display window
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment
        variable.
    views : list | str
        views to use
    offset : bool
        If True, aligs origin with medial wall. Useful for viewing inflated
        surface where hemispheres typically overlap (Default: True)
    show_toolbar : bool
        If True, toolbars will be shown for each view.
    offscreen : bool
        If True, rendering will be done offscreen (not shown). Useful
        mostly for generating images or screenshots, but can be buggy.
        Use at your own risk.
    interaction : str
        Can be "trackball" (default) or "terrain", i.e. a turntable-style
        camera.
    w, h : int
        Figure width and height.
    axw, axh : int
        Width and height of the individual viewing panes.
    name : str
        Window title (alternative to ``title`` for consistency with other
        Eelbrain figures).
    pos : tuple of int
        Position of the new window on the screen.
    show : bool
        Currently meaningless due to limitation in VTK that does not allow
        hidden plots.
    run : bool
        Run the Eelbrain GUI app (default is True for interactive plotting and
        False in scripts).

    Notes
    -----
    The documentation lists only the methods that Eelbrain adds to or overrides
    from the PySurfer :class:`~surfer.Brain` super-class. For complete PySurfer
    functionality see te PySurfer documentation.

    Initializing a :class:`Brain` object is expensive. When plotting multiple
    overlays on the same brain, it is more economical to initialize only one
    :class:`Brain` object and adding/removing the data layers in a loop.
    Pseudo-code for an example adding images to a table::

        table = fmtxt.Table('ll')
        b = brain(source_space, hemi=hemi, …)
        for data_layer in data_layers:
            table.cell(data_layer.name)
            b.add_ndvar(data_layer, ...)
            im = b.image()
            table.cell(im)
            b.remove_data()
            b.remove_labels()

    For another example see the implementation of :class:`SequencePlotter`.
    """
    _display_time_in_frame_title = True

    def __init__(self, subject, hemi, surf='inflated', title=None,
                 cortex="classic", alpha=1.0, background="white",
                 foreground="black", subjects_dir=None, views='lat',
                 offset=True, show_toolbar=False, offscreen=False,
                 interaction='trackball', w=None, h=None, axw=None, axh=None,
                 name=None, pos=None, source_space=None, show=True, run=None):
        if not SURFER_IMPORTED:
            raise RuntimeError("PySurfer import failed. You should have seen a warning 'Error importing PySurfer' earlier.")

        from ._wx_brain import BrainFrame

        self.__data = []
        self.__annot = None
        self.__labels = {}  # {name: color}
        self.__time_index = 0
        self.__source_space = source_space
        self.hemi = hemi

        if isinstance(views, str):
            views = [views]
        elif not isinstance(views, list):
            views = list(views)
        n_rows = len(views)

        if hemi == 'split':
            n_columns = 2
        elif hemi in ('lh', 'rh', 'both'):
            n_columns = 1
        else:
            raise ValueError(f"{hemi=}")

        # layout
        if w is None and axw is None:
            if h is None and axh is None:
                axw = BRAIN_W
                axh = BRAIN_H
            else:
                if axh is None:
                    axh = int(round(h / n_rows))
                axw = int(round(axh * (BRAIN_W / BRAIN_H)))
        elif h is None and axh is None:
            if axw is None:
                axw = int(round(w / n_columns))
            axh = int(round(axw * (BRAIN_H / BRAIN_W)))

        if w is None:
            w = axw * n_columns

        if h is None:
            h = axh * n_rows

        if title is None:
            if name is None:
                title = subject
            elif isinstance(name, str):
                title = name
            else:
                raise TypeError(f"{name=} (str required)")
        elif not isinstance(title, str):
            raise TypeError(f"{title=} (str required)")

        if foreground is None:
            foreground = 'black'
        if background is None:
            background = 'white'

        if INLINE_DISPLAY:
            self._frame = figure = None
            self._png_repr_meta = {'height': h, 'width': w}
        else:
            self._frame = BrainFrame(None, self, title, w, h, n_rows, n_columns, surf, pos)
            figure = self._frame.figure
            self._png_repr_meta = None

        if subjects_dir is not None:
            subjects_dir = os.path.expanduser(subjects_dir)
        surfer.Brain.__init__(self, subject, hemi, surf, '', cortex, alpha, (w, h), background, foreground, figure, subjects_dir, views, offset, show_toolbar, offscreen, interaction)
        TimeSlicer.__init__(self)

        if self._frame and CONFIG['show'] and show:
            self._frame.Show()
            if CONFIG['eelbrain'] and do_autorun(run):
                from .._wxgui import run as run_gui  # lazy import for docs
                run_gui()

    def __repr__(self):
        args = [self.subject_id, self._hemi, self.surf]
        if self.n_times:
            args.append(f"{self.n_times} time points {ms(self._time_dim.tmin)}-{ms(self._time_dim.tstop)} ms")
        return f"<plot.brain.Brain: {', '.join(args)}>"

    def _asfmtext(self, **_):
        return self.image()

    # pysurfer 0.10
    def _ipython_display_(self):
        """Called by Jupyter notebook to display a brain."""
        if use_inline_backend():
            import IPython.display
            IPython.display.display(self.image())
        else:
            print(repr(self))

    def _check_source_space(self, source):
        "Make sure SourceSpace is compatible"
        source = get_source_dim(source)
        if source.subject != self.subject_id:
            raise ValueError(f"Trying to plot NDVar with {source.subject=} on Brain with subject={self.subject_id!r}")
        elif self._hemi == 'lh' and source.lh_n == 0:
            raise ValueError("Trying to add NDVar without lh data to plot of lh")
        elif self._hemi == 'rh' and source.rh_n == 0:
            raise ValueError("Trying to add NDVar without rh data to plot of rh")
        return source

    def add_label(self, label, color=None, alpha=1, scalar_thresh=None, borders=False, hemi=None, subdir=None, lighting=False, **kwargs):
        surfer.Brain.add_label(self, label, color, alpha, scalar_thresh,
                               borders, hemi, subdir)
        self.labels_dict[label.name][0].actor.property.lighting = lighting
        if color is None:
            color = getattr(label, 'color', None) or "crimson"
        name = label if isinstance(label, str) else label.name
        self.__labels[name] = color

    def add_mask(self, source, color=(0, 0, 0, 0.5), smoothing_steps=None,
                 alpha=None, subjects_dir=None):
        """Add a mask shading areas that are not included in an NDVar

        Parameters
        ----------
        source : SourceSpace
            SourceSpace.
        color : matplotlib color
            Mask color, can include alpha (defauls is black with alpha=0.5:
            ``(0, 0, 0, 0.5)``).
        smoothing_steps : scalar (optional)
            Smooth transition at the mask's border. If smoothing, the mask is
            added as data layer, otherwise it is added as label.
        alpha : scalar
            Alpha for the mask (supercedes alpha in ``color``).
        subjects_dir : str
            Use this directory as the subjects directory.
        """
        source = self._check_source_space(source)
        if color is True:
            color = (0, 0, 0, 0.5)
        color = to_rgba(color, alpha)
        if smoothing_steps is not None:
            # generate LUT
            lut = np.repeat(np.reshape(color, (1, 4)), 256, 0)
            lut[:, 3] = np.linspace(color[-1], 0, 256)
            np.clip(lut, 0, 1, lut)
            lut *= 255
            lut = np.round(lut).astype(np.uint8)
            # generate mask Label
            mask_ndvar = source._mask_ndvar(subjects_dir)
            self.add_ndvar(mask_ndvar, lut, 0., 1., smoothing_steps, False, None, False)
        else:
            lh, rh = source._mask_label(subjects_dir)
            if self._hemi == 'lh':
                rh = None
            elif self._hemi == 'rh':
                lh = None

            if source.lh_n and lh:
                self.add_label(lh, color[:3], color[3])
                self.labels_dict['mask-lh'][0].actor.property.lighting = False
            if source.rh_n and rh:
                self.add_label(rh, color[:3], color[3])
                self.labels_dict['mask-rh'][0].actor.property.lighting = False

    def add_ndvar(
            self,
            ndvar: NDVar,
            cmap: Any = None,
            vmin: float = None,
            vmax: float = None,
            smoothing_steps: int = None,
            colorbar: bool = False,
            time_label: Union[str, Callable] = 'ms',
            lighting: bool = False,
            contours: Union[bool, Sequence[float]] = None,
            alpha: float = 1,
            remove_existing: bool = False,
    ):
        """Add data layer form an NDVar

        Parameters
        ----------
        ndvar : NDVar  ([case,] source[, time])
            NDVar with SourceSpace dimension and optional time dimension. If it
            contains a :class:`Case` dimension, the average over cases is
            displayed. Values outside of the source-space, as well as masked
            values are set to 0, assuming a colormap in which 0 is transparent.
        cmap : str | list of matplotlib colors | array
            Colormap. Can be the name of a matplotlib colormap, a list of
            colors, or a custom lookup table (an n x 4 array with RBGA values
            between 0 and 255).
        vmin
            Lower endpoint for the colormap. Needs to be set explicitly if
            ``cmap`` is a LUT array.
        vmax
            Upper endpoint for the colormap. Needs to be set explicitly if
            ``cmap`` is a LUT array.
        smoothing_steps
            Number of smoothing steps if data is spatially undersampled
            (PySurfer ``Brain.add_data()`` argument).
        colorbar
            Add a colorbar to the figure (use ``.plot_colorbar()`` to plot a
            colorbar separately).
        time_label
            Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
            milliseconds or in seconds, or supply a custom formatter for time
            values in seconds (default is ``'ms'``).
        lighting
            The data overlay is affected by light sources (default ``False``,
            i.e. data overlays appear luminescent).
        contours
            Draw contour lines instead of a solid overlay. Set to a list of
            contour levels or ``True`` for automatic contours.
        alpha
            Alpha value for the data layer (0 = tranparent, 1 = opaque).
        remove_existing
            Remove data layers that have been added previously (default False).
        """
        ndvar = asndvar(ndvar)
        # check input data and dimensions
        source = self._check_source_space(ndvar)
        # find ndvar time axis
        if ndvar.ndim == 1 + ndvar.has_case:
            if ndvar.has_case:
                ndvar = ndvar.mean('case')
            time_dim = times = None
            data_dims = (source.name,)
        elif ndvar.ndim != 2:
            if ndvar.has_case:
                raise ValueError(f"{ndvar=}: must be one- or two dimensional. If you meant to plot the average of cases, use ndvar.mean('case')")
            raise ValueError(f"{ndvar=}: must be one- or two dimensional")
        elif ndvar.has_dim('time'):
            time_dim = ndvar.time
            times = ndvar.time.times
            data_dims = (source.name, 'time')
            if time_label == 'ms':
                def time_label(x): return '%s ms' % int(round(x * 1000))
            elif time_label == 's':
                time_label = '%.3f s'
            elif time_label is False:
                time_label = None
        else:
            data_dims = ndvar.get_dimnames((source.name, None))
            time_dim = ndvar.get_dim(data_dims[1])
            times = np.arange(len(time_dim))
            time_label = None

        # remove existing data before modifying attributes
        if remove_existing:
            self.remove_data()

        # make sure time axis is compatible with existing data
        if isinstance(time_dim, UTS):
            self._init_time_dim(time_dim)

        # find colormap parameters
        meas = ndvar.info.get('meas')
        if cmap is None or isinstance(cmap, (str, Colormap)):
            epochs = ((ndvar,),)
            cmaps = find_fig_cmaps(epochs, cmap, alpha=True)
            vlims = find_fig_vlims(epochs, vmax, vmin, cmaps, unmask=False)
            vmin, vmax = vlims[meas]
        # colormap
        if contours is not None:
            if cmap is None:
                cmap = ('w', 'w')
            elif isinstance(cmap, str) and len(cmap) > 1:
                cmap = cmaps[meas]
            else:
                contour_color = to_rgb(cmap)
                cmap = (contour_color, contour_color)
        elif cmap is None or isinstance(cmap, str):
            cmap = cmaps[meas]

        # general PySurfer data args
        if smoothing_steps is None and source.kind == 'ico':
            smoothing_steps = source.grade + 1

        # determine which hemi we're adding data to
        if self._hemi in ('lh', 'rh'):
            data_hemi = self._hemi
        elif not source.lh_n:
            data_hemi = 'rh'
        elif not source.rh_n:
            data_hemi = 'lh'
        else:
            data_hemi = 'both'
        # remember where to find data_dict
        dict_hemi = 'rh' if data_hemi == 'rh' else 'lh'
        data_index = len(self._data_dicts[dict_hemi])

        # add data
        new_surfaces = []
        if data_hemi != 'rh':
            if self._hemi == 'lh':
                colorbar_ = colorbar
                colorbar = False
                time_label_ = time_label
                time_label = None
            else:
                colorbar_ = False
                time_label_ = None

            src_hemi = ndvar.sub(**{source.name: 'lh'})
            data = src_hemi.get_data(data_dims, 0)
            vertices = source.lh_vertices
            self.add_data(data, vmin, vmax, None, cmap, alpha, vertices, smoothing_steps, times, time_label_, colorbar_, 'lh')
            new_surfaces.extend(self.data_dict['lh']['surfaces'])

        if data_hemi != 'lh':
            src_hemi = ndvar.sub(**{source.name: 'rh'})
            data = src_hemi.get_data(data_dims, 0)
            vertices = source.rh_vertices
            self.add_data(data, vmin, vmax, None, cmap, alpha, vertices, smoothing_steps, times, time_label, colorbar, 'rh')
            new_surfaces.extend(self.data_dict['rh']['surfaces'])

        # update surfaces
        for surface in new_surfaces:
            if contours is not None:
                surface.enable_contours = True
                # http://code.enthought.com/projects/files/ets_api/enthought.mayavi.components.contour.Contour.html
                surface.contour.auto_update_range = False
                # surface.contour.maximum_contour = ndvar.max()
                # surface.contour.minimum_contour = ndvar.min()
                if contours is not True:
                    surface.contour.contours = contours
                    surface.contour.auto_contours = False

            if not lighting:
                surface.actor.property.lighting = False

        self.__data.append({
            'hemi': data_hemi,
            'data': ndvar,
            'dict_hemi': dict_hemi,
            'dict_index': data_index,
            'cmap': cmap,
            'vmin': vmin,
            'vmax': vmax,
        })

    def add_ndvar_annotation(self, ndvar, colors=None, borders=True, alpha=1, lighting=True):
        """Add annotation from labels in an NDVar

        Parameters
        ----------
        ndvar : NDVar of int
            NDVar in which each unique integer indicates a label. By default,
            ``0`` is interpreted as unlabeled, but this can be overridden by
            providing a ``colors`` dictionary that contains an entry for ``0``.
        colors : dict
            Dictionary mapping label IDs to colors.
        borders : bool | int
            Show label borders (instead of solid labels). If int, specify the
            border width.
        alpha : scalar [0, 1]
            Opacity of the labels (default 1).
        lighting : bool
            Labels are affected by lights (default True).
        """
        source = self._check_source_space(ndvar)
        x = ndvar.get_data(source.name)
        if x.dtype.kind not in 'iu':
            raise TypeError("Need NDVar of integer type, not %r" % (x.dtype,))
        # determine colors
        label_values = np.unique(x)
        if colors is None or isinstance(colors, str):
            cells = np.setdiff1d(label_values, [0], assume_unique=True)
            colors = plot_colors = colors_for_oneway(cells, cmap=colors)
        elif isinstance(colors, dict):
            missing = np.setdiff1d(label_values, colors)
            # if 0 should be plotted, we need to shift values
            if 0 in colors and 0 in label_values:
                plot_colors = {k + 1: v for k, v in colors.items()}
                x = x + 1
                label_values += 1
            else:
                plot_colors = colors
                missing = np.setdiff1d(missing, [0], assume_unique=True)
            if missing:
                raise KeysMissing(missing, 'colors', colors)
        else:
            raise TypeError(f"colors={colors}")
        # generate color table
        ctab = np.zeros((len(label_values), 5), int)
        ctab[:, 4] = label_values
        for i, v in enumerate(label_values):
            if v == 0:
                continue
            ctab[i, :4] = [int(round(c * 255.)) for c in to_rgba(plot_colors[v])]
        # generate annotation
        sss = source.get_source_space()
        indexes = (slice(None, source.lh_n), slice(source.lh_n, None))
        annot = []
        has_annot = []
        for ss, vertices, index in zip(sss, source.vertices, indexes):
            hemi = HEMI_ID_TO_STR[ss['id']]
            if self._hemi == OTHER_HEMI[hemi]:
                continue
            # expand to full source space
            ss_map = np.zeros(ss['nuse'], int)
            ss_map[np.isin(ss['vertno'], vertices)] = x[index]
            # select only the used colors; Mayavi resets the range of the data-
            # to-LUT mapping to the extrema of the data at various points, so it
            # is safer to restrict the LUT to used colors
            ctab_index = np.isin(ctab[:, 4], ss_map)
            hemi_ctab = ctab[ctab_index]
            if np.any(hemi_ctab):
                # map nearest from vertex to index
                nearest = np.searchsorted(ss['vertno'], ss['nearest'])
                # expand to full brain
                full_map = ss_map[nearest]
                annot.append((full_map, hemi_ctab))
                has_annot.append(hemi)

        if len(annot) == 0:
            return
        elif len(annot) == 1:
            annot = annot[0]
            hemi = has_annot[0]
        else:
            hemi = None

        self.add_annotation(annot, borders, alpha, hemi)
        self.__annot = colors
        if not lighting:
            for annot in self.annot_list:
                annot['surface'].actor.property.lighting = False

    def add_ndvar_label(self, ndvar, color=(1, 0, 0), borders=False, name=None,
                        alpha=None, lighting=False):
        """Draw a boolean NDVar as label.

        Parameters
        ----------
        ndvar : NDVar
            Boolean NDVar.
        color : matplotlib-style color | None
            anything matplotlib accepts: string, RGB, hex, etc. (default
            "crimson")
        borders : bool | int
            Show only label borders. If int, specify the number of steps
            (away from the true border) along the cortical mesh to include
            as part of the border definition.
        name : str
            Name for the label (for display in legend).
        alpha : float in [0, 1]
            alpha level to control opacity
        lighting : bool
            Whether label should be affected by lighting (default False).

        Notes
        -----
        To remove previously added labels, run Brain.remove_labels(). This
        method can only plot static labels; to plot a contour that varies over
        time, use :meth:`Brain.add_ndvar` with the ``contours`` parameter.
        """
        if color is None:
            color = (1, 0, 0)
        source = self._check_source_space(ndvar)
        x = ndvar.get_data(source.name)
        if x.dtype.kind != 'b':
            raise ValueError("Require NDVar of type bool, got %r" % (x.dtype,))
        if name is None:
            name = str(ndvar.name)
        color = to_rgba(color, alpha)
        lh_vertices = source.lh_vertices[x[:source.lh_n]]
        rh_vertices = source.rh_vertices[x[source.lh_n:]]
        lh, rh = source._label((lh_vertices, rh_vertices), name, color[:3])
        if lh and self._hemi != 'rh':
            while lh.name in self.labels_dict:
                lh.name += '_'
            self.add_label(lh, color[:3], color[3], borders=borders)
            self.labels_dict[lh.name][0].actor.property.lighting = lighting
        if rh and self._hemi != 'lh':
            while rh.name in self.labels_dict:
                rh.name += '_'
            self.add_label(rh, color[:3], color[3], borders=borders)
            self.labels_dict[rh.name][0].actor.property.lighting = lighting
        self.__labels[name] = color

    def add_ndvar_p_map(self, p_map, param_map=None, p0=0.05, p1=0.01, p0alpha=0.5, *args, **kwargs):
        """Add a map of p-values as data-layer

        Parameters
        ----------
        p_map : NDVar | NDTest
            Map of p values, or test result.
        param_map : NDVar
            Statistical parameter covering the same data points as p_map. Only the
            sign is used, for incorporating the directionality of the effect into
            the plot.
        p0 : scalar
            Highest p-value that is visible.
        p1 : scalar
            P-value where the colormap changes from ramping alpha to ramping color.
        p0alpha : 1 >= float >= 0
            Alpha at ``p0``. Set to 0 for a smooth transition, or a larger value to
            clearly delineate significant regions (default 0.5).
        ...
            Other parameters for :meth:`.add_ndvar`.
        """
        from .._stats.testnd import NDTest, MultiEffectNDTest

        if isinstance(p_map, NDTest):
            if isinstance(p_map, MultiEffectNDTest):
                raise NotImplementedError(f"plot.brain.p_map for {p_map.__class__.__name__}")
            elif param_map is not None:
                raise TypeError(f"param_map={param_map!r} when p_map is NDTest result")
            res = p_map
            p_map = res.p
            param_map = res._statistic_map
        p_map, cmap, vmax = p_lut(p_map, param_map, p0, p1, p0alpha)
        self.add_ndvar(p_map, cmap, -vmax, vmax, *args, **kwargs)

    def close(self):
        "Close the figure window"
        if self._frame:
            self._frame.Close()
        else:
            self._surfer_close()

    @property
    def closed(self):
        return self._figures[0][0] is None

    def _surfer_close(self):
        surfer.Brain.close(self)

    def copy_screenshot(self):
        "Copy the currently shown image to the clipboard"
        from .._wxgui import wx

        tempdir = mkdtemp()
        tempfile = os.path.join(tempdir, "brain.png")
        self.save_image(tempfile, 'rgba', True)

        bitmap = wx.Bitmap(tempfile, wx.BITMAP_TYPE_PNG)
        bitmap_obj = wx.BitmapDataObject(bitmap)

        if not wx.TheClipboard.IsOpened():
            open_success = wx.TheClipboard.Open()
            if open_success:
                wx.TheClipboard.SetData(bitmap_obj)
                wx.TheClipboard.Close()
                wx.TheClipboard.Flush()

    def _get_cmap_params(self, layer=0, label=True):
        """Return parameters required to plot a colorbar"""
        data = self.__data[layer]
        data_dict = self._data_dicts[data['dict_hemi']][data['dict_index']]
        colormap = ListedColormap(data_dict['orig_ctable'] / 255., label)
        return colormap, data_dict['fmin'], data_dict['fmax']

    def _has_annot(self):
        return bool(self.__annot)

    def _has_data(self):
        return bool(self.__data)

    def _has_labels(self):
        return bool(self.__labels)

    def enable_vertex_selection(self, color='red'):
        """Find source space vertice by right-clicking on the brain

        After enabling this functionality, each right-click on the brain will
        mark the closest vertex and the vertex number will be printed in the
        terminal.

        Parameters
        ----------
        color : mayavi color
            Color for the vertex marker.

        Examples
        --------
        Load a source space and plot it to be able to select vertices::

            ss = SourceSpace.from_file('directory/mri_subjects', 'fsaverage', 'ico-4')
            brain = plot.brain.brain(ss)
            brain.enable_vertex_selection()

        """
        if self.__source_space is None:
            raise RuntimeError("Can't enable vertex selection for brian without source space")
        for brain in self.brains:
            func = partial(self._select_nearest_source, hemi=brain.hemi, color=color)
            brain._f.on_mouse_pick(func, button="Right")

    @cached_property
    def _tris_lh(self):
        return self.__source_space._read_surf('lh')[1]

    @cached_property
    def _tris_rh(self):
        return self.__source_space._read_surf('rh')[1]

    def _select_nearest_source(self, vertex, hemi, color='red'):
        if not isinstance(vertex, int):
            vertex = vertex.point_id

        ss_vertices = self.__source_space.vertices[hemi == 'rh']
        if vertex not in ss_vertices:
            tris = self._tris_lh if hemi == 'lh' else self._tris_rh
            selected = np.any(tris == vertex, 1)
            for i in range(7):
                selected_v = np.unique(tris[selected])
                index = np.isin(ss_vertices, selected_v)
                if index.any():
                    vertex = ss_vertices[index][0]
                    break
                for v in selected_v:
                    selected |= np.any(tris == v, 1)
            else:
                print("No vertex found in 7 iterations")

        if 'selection' in self.foci_dict:
            self.foci_dict.pop('selection')[0].remove()
        self.add_foci([vertex], True, hemi=hemi, scale_factor=0.5, name='selection', color=color)
        tag = 'L' if hemi == 'lh' else 'R'
        print(f'{tag}{vertex}')

    def image(self, name=None, format='png', alt=None, mode='rgb'):
        """Create an FMText Image from a screenshot

        Parameters
        ----------
        name : str
            Name for the file (without extension; default is ``data.name`` or
            'brain').
        format : str
            File format (default 'png').
        alt : None | str
            Alternate text, placeholder in case the image can not be found
            (HTML `alt` tag).
        mode : ``'rgb'`` | ``'rgba'``
            ``'rgb'`` to render solid background, or ``'rgba'`` to
            include alpha channel for a transparent background (default).
        """
        if name is None:
            for data in self.__data:
                name = data['data'].name
                if name:
                    break
            else:
                name = 'brain'
        im = self.screenshot(mode, True)
        if self._png_repr_meta is None:
            w, h = self._frame.GetClientSize()
            meta = {'height': h, 'width': w}
        else:
            meta = self._png_repr_meta
        return Image.from_array(im, name, format, alt, **meta)

    def _im_array(self):
        im = self.screenshot('rgba', True)
        im *= 255
        return im.astype(np.int8)

    def plot_colorbar(self, label=True, label_position=None, label_rotation=None,
                      clipmin=None, clipmax=None, orientation='horizontal',
                      width=None, ticks=None, layer=None, *args, **kwargs):
        """Plot a colorbar corresponding to the displayed data

        Parameters
        ----------
        label : str | bool
            Label for the x-axis (default is based on the data).
        label_position : 'left' | 'right' | 'top' | 'bottom'
            Position of the axis label. Valid values depend on orientation.
        label_rotation : scalar
            Angle of the label in degrees (For horizontal colorbars, the default is
            0; for vertical colorbars, the default is 0 for labels of 3 characters
            and shorter, and 90 for longer labels).
        clipmin : scalar
            Clip the color-bar below this value.
        clipmax : scalar
            Clip the color-bar above this value.
        orientation : 'horizontal' | 'vertical'
            Orientation of the bar (default is horizontal).
        width : scalar
            Width of the color-bar in inches.
        ticks : {float: str} dict | sequence of float
            Customize tick-labels on the colormap; either a dictionary with
            tick-locations and labels, or a sequence of tick locations.
        layer : int
            If the brain contains multiple data layers, plot a colorbar for
            only one (int in the order ndvars were added; default is to plot
            colorbars for all layers).

        Returns
        -------
        colorbar : :class:`~eelbrain.plot.ColorBar` | list
            ColorBar plot object (list of colorbars if more than one data layer
            are present).
        """
        if not self.__data:
            raise RuntimeError("Brain has no data to plot colorbar for")

        if layer is None:
            layers = range(len(self.__data))
        else:
            layers = (layer,)

        out = []
        for layer in layers:
            data = self.__data[layer]
            ndvar = data['data']
            if ticks is None:
                ticks = ndvar.info.get('cmap ticks')
            scale = AxisScale(ndvar, label)
            colormap, vmin, vmax = self._get_cmap_params(layer, label)
            out.append(ColorBar(colormap, vmin, vmax, label, label_position, label_rotation, clipmin, clipmax, orientation, scale, (), width, ticks, *args, **kwargs))
            # reset parames
            label = True

        if len(out) == 1:
            return out[0]
        else:
            return out

    def _set_annot(self, annot, borders, alpha):
        "Store annot name to enable plot_legend()"
        self.add_annotation(annot, borders, alpha)
        self.__annot = annot

    def play_movie(self, tstart=None, tstop=None, fps=10):
        """Play an animation by setting time"""
        max_wait = 1 / fps
        start = 0 if tstart is None else self.index_for_time(tstart)
        stop = self.n_times if tstop is None else self.index_for_time(tstop)
        for i in range(start, stop):
            t0 = time()
            self.set_data_time_index(i)
            dt = time() - t0
            if dt < max_wait:
                sleep(max_wait - dt)

    def plot_legend(self, *args, **kwargs):
        """Plot legend for parcellation

        Parameters
        ----------
        labels : dict (optional)
            Alternative (text) label for (brain) labels.
        h : 'auto' | scalar
            Height of the figure in inches. If 'auto' (default), the height is
            automatically increased to fit all labels.

        Returns
        -------
        legend : :class:`~eelbrain.plot.ColorList`
            Figure with legend for the parcellation.

        See Also
        --------
        plot.brain.annot_legend : plot a legend without plotting the brain
        """
        from ._brain import annot_legend
        if self.__labels:
            return ColorList(self.__labels, self.__labels.keys(), *args, **kwargs)
        elif self.__annot is None:
            raise RuntimeError("Can only plot legend for brain displaying "
                               "parcellation")
        elif isinstance(self.__annot, str):
            lh = os.path.join(self.subjects_dir, self.subject_id, 'label',
                              'lh.%s.annot' % self.__annot)
            rh = os.path.join(self.subjects_dir, self.subject_id, 'label',
                              'rh.%s.annot' % self.__annot)
            return annot_legend(lh, rh, *args, **kwargs)
        else:
            return ColorList(self.__annot, sorted(self.__annot), *args, **kwargs)

    def remove_data(self, hemi=None):
        """Remove data shown with ``Brain.add_ndvar``"""
        surfer.Brain.remove_data(self, None)
        del self.__data[:]
        self._time_dim = None

    def remove_labels(
            self,
            labels: Sequence[str] = None,
            mask: bool = False,
    ):
        """Remove labels shown with ``Brain.add_ndvar_label``

        Parameters
        ----------
        labels
            Labels to remove.
        mask
            Also remove the mask (labels with names starting with ``mask-``).
        """
        if labels is None and not mask:
            labels = [l for l in self.__labels if not l.startswith('mask-')]
        elif isinstance(labels, str):
            labels = [labels]
        else:
            labels = list(labels)
            if not all(isinstance(l, str) for l in labels):
                raise TypeError("labels=%r" % (labels,))
        surfer_labels = set(labels).intersection(self._label_dicts)
        surfer.Brain.remove_labels(self, surfer_labels)
        for label in labels:
            del self.__labels[label]

    def save_image(
            self,
            filename: PathArg,
            mode: Literal['rgb', 'rgba'] = 'rgb',
            antialiased: bool = False,
            fake_transparency: Any = None,
    ):
        """Save view from all panels to disk

        Parameters
        ----------
        filename: string
            Path to new image file.
        mode : ``'rgb'`` | ``'rgba'``
            ``'rgb'`` to render solid background (default), or ``'rgba'`` to
            include alpha channel for a transparent background.
        antialiased : bool
            Antialias the image (see :func:`mayavi.mlab.screenshot`
            for details; default False).

            .. warning::
               Antialiasing can interfere with ``rgba`` mode, leading to opaque
               background.

        fake_transparency
            Use this color as background color and make it transparent.
            Workaround if ``mode='rgba'`` is broken.

        See also
        --------
        .screenshot : grab current image as array

        Notes
        -----
        Due to limitations in TraitsUI, if multiple views or hemi='split'
        is used, there is no guarantee painting of the windows will
        complete before control is returned to the command line. Thus
        we strongly recommend using only one figure window (which uses
        a Mayavi figure to plot instead of TraitsUI) if you intend to
        script plotting commands.
        """
        if fake_transparency is not None:
            bg_color = to_rgb(fake_transparency)
            for figures in self._figures:
                for figure in figures:
                    figure.scene.background = bg_color
        else:
            bg_color = None
        im: np.ndarray = self.screenshot(mode, antialiased)
        if bg_color is not None:
            seed = np.zeros(im.shape[:2], bool)
            seed[0, 0] = True
            seed[0, -1] = True
            seed[-1, 0] = True
            seed[-1, -1] = True
            mask = np.all(im[:, :, :3] == [[bg_color]], 2)
            transparent = scipy.ndimage.binary_propagation(seed, mask=mask)
            im[:, :, 3] = ~transparent
        imsave(filename, im)

    def set_parallel_view(self, forward=None, up=None, scale=None):
        """Set view to parallel projection

        Parameters
        ----------
        forward : scalar
            Move the view forward (mm).
        up : scalar
            Move the view upward (mm).
        scale : scalar
            Mayavi parallel_scale parameter. Default is 95 for the inflated
            surface, 75 otherwise. Smaller numbers correspond to zooming in.
        """
        import mayavi
        from mayavi import mlab
        from traits.trait_base import ETSConfig

        if scale is True:
            surf = self.geo['rh' if self._hemi == 'rh' else 'lh'].surf
            if surf == 'inflated':
                scale = 95 if ETSConfig.toolkit == 'wx' else 115
            else:
                scale = 75

        # Which mayavi scaling issues are we dealing with
        if IS_OSX and self._frame and scale is not None:
            scale_bug = packaging.version.parse(mayavi.__version__) < packaging.version.parse('4.8.2')
            position_bug = not scale_bug
        else:
            scale_bug = position_bug = 0

        i = 0
        for figs in self._figures:
            for fig in figs:
                if forward is not None or up is not None:
                    mlab.view(focalpoint=(0, forward or 0, up or 0), figure=fig)
                if scale is not None:
                    if scale_bug and i == 0:
                        fig.scene.camera.parallel_scale = 2 * scale
                    else:
                        fig.scene.camera.parallel_scale = scale
                fig.scene.camera.parallel_projection = True
                fig.render()
                i += 1

        if position_bug:
            fig = self._figures[0][0]
            is_rh = self.hemi == 'rh'
            if is_rh:
                is_lateral = mlab.view(figure=fig)[0] < 90
            else:
                is_lateral = mlab.view(figure=fig)[0] > 90
            if is_rh == is_lateral:
                mlab.view(focalpoint=(0, -57, -47), figure=fig)
            else:
                mlab.view(focalpoint=(0, 57, -47), figure=fig)

    def set_size(self, width, height):
        """Set image size in pixels"""
        self._frame.SetImageSize(width, height)

    def set_surf(self, surf):
        from ._wx_brain import SURFACES

        self._set_surf(surf)
        if self._frame and surf in SURFACES:
            self._frame._surf_selector.SetSelection(SURFACES.index(surf))

    def _set_surf(self, surf):
        surfer.Brain.set_surf(self, surf)
        self.set_parallel_view(scale=True)

    def set_title(self, title):
        "Set the window title"
        self._frame.SetTitle(str(title))

    def set_vlim(self, v=None, vmax=None):
        """Change the colormap limits

        If the limit is symmetric, use ``set_vlim(vlim)``; if it is not, use
        ``set_vlim(vmin, vmax)``.

        Parameters
        ----------
        v : scalar
            If this is the only value specified it is interpreted as the upper
            end of the scale, and the lower end is determined based on
            the colormap to be ``-v`` or ``0``. If ``vmax`` is also specified,
            ``v`` specifies the lower end of the scale.
        vmax : scalar (optional)
            Upper end of the color scale.

        Notes
        -----
        Only affects the most recently added data layer.
        """
        if self.__data:
            data = self.__data[-1]
            cmap = data['cmap']
            if vmax is None:
                vmin, vmax = fix_vlim_for_cmap(None, abs(v), cmap)
            else:
                vmin = v
        else:
            data = None
            if vmax is None:
                surfer_data = self.data_dict.values()[0]
                vmax = abs(v)
                if surfer_data['fmin'] >= 0:
                    vmin = surfer_data['fmin']
                else:
                    vmin = -vmax
            else:
                vmin = v
        self.scale_data_colormap(vmin, (vmin + vmax) / 2, vmax, False)
        if data is not None:
            data['vmin'] = vmin
            data['vmax'] = vmax

    def get_vlim(self):
        "``(vmin, vmax)`` for the most recently added data layer"
        if not self.__data:
            raise RuntimeError('No data added with Brain.add_ndvar()')
        data = self.__data[-1]
        return data['vmin'], data['vmax']

    def _update_time(self, t, fixate):
        index = self._time_dim._array_index(t)
        if index == self.__time_index:
            return
        elif self.closed:
            return
        self.set_data_time_index(index)
        self.__time_index = index


if hasattr(surfer.Brain, 'add_label'):
    Brain.add_label.__doc__ = surfer.Brain.add_label.__doc__  # py3
