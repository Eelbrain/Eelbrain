# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Framework for figures embedded in a GUI

Implementation
==============

Plotting is implemented hierarchically in 3 different types of
functions/classes:

top-level (public names)
    Top-level functions or classes have public names create an entire figure.
    Some classes also retain the figure and provide methods for manipulating
    it.

_ax_
    Functions beginning with _ax_ organize an axes object. They do not
    create their own axes object (this is provided by the top-level function),
    but change axes formatting such as labels and extent.

_plt_
    Functions beginning with _plt_ only plot data to a given axes object
    without explicitly changing aspects of the axes themselves.


Top-level plotters can be called with nested lists of data-objects (NDVar
instances). They create a separate axes for each list element. Axes
themselves can have multiple layers (e.g., a difference map visualized through
a colormap, and significance levels indicated by contours).


Example: t-test
---------------

For example, the default plot for testnd.ttest() results is the
following list (assuming the test compares A and B):

``[A, B, [diff(A,B), p(A, B)]]``

where ``diff(...)`` is a difference map and ``p(...)`` is a map of p-values.
The main plot function creates a separate axes object for each list element:

- ``A``
- ``B``
- ``[diff(A,B), p(A, B)]``

Each of these element is then plotted with the corresponding _ax_ function.
The _ax_ function calls _plt_ for each of its input elements. Thus, the
functions executed are:

#. plot([A, B, [diff(A,B), p(A, B)]])
#. ---> _ax_(A)
#. ------> _plt_(A)
#. ---> _ax_(B)
#. ------> _plt_(B)
#. ---> _ax_([diff(A,B), p(A, B)])
#. ------> _plt_(diff(A,B))
#. ------> _plt_(p(A, B))


Vision
------

Modularize organization

Data organizers:
    Take data from input arguments and organize it into axes and layers
Style mappers:
    Take data and input arguments and determine matplotlib parameters
Figure components
    Manage styling and interaction for figure properties (XAxisMixin,
    TimeController, ...)

"""
from __future__ import annotations
import __main__

from collections.abc import Iterable
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import cached_property, reduce
from itertools import chain, cycle, repeat
from logging import getLogger
import math
from numbers import Real
import os
import re
import time
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union
import weakref

import matplotlib as mpl
import matplotlib.axes
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.text
from matplotlib.colors import Colormap
from matplotlib.figure import SubplotParams
from matplotlib.ticker import FuncFormatter
import numpy as np

from .._celltable import Celltable
from .._colorspaces import LocatedColormap, SYMMETRIC_CMAPS, ZEROBASED_CMAPS, ALPHA_CMAPS
from .._config import CONFIG
from .._data_obj import Dimension, Dataset, Factor, Interaction, NDVar, Var, Case, UTS, NDVarArg, CategorialArg, IndexArg, CellArg, NDVarTypes, ascategorial, asndvar, assub, isnumeric, isdataobject, combine_cells, cellname
from .._utils.notebooks import use_inline_backend
from .._stats import testnd
from .._utils import IS_WINDOWS, intervals, ui
from .._ndvar import erode, resample
from .._text import enumeration, ms
from ..fmtxt import FMTextArg, Image, asfmtext, asfmtext_or_none
from ..table import melt_ndvar
from ._decorations import mark_difference
from ._styles import Style, find_cell_styles
from ._utils import adjust_hsv


# constants
POINT = 0.013888888888898  # inches

# defaults
defaults = {'maxw': 16, 'maxh': 10}

# Types
CMapArg = Any
ColorArg = Any
LegendArg = Optional[Union[str, int, Tuple[float, float], bool]]


class PlotType(Enum):
    GENERAL = auto()
    LEGACY = auto()
    LINE = auto()
    IMAGE = auto()
    CONTOUR = auto()


def do_autorun(run=None):
    # http://stackoverflow.com/a/2356420/166700
    if run is not None:
        return run
    elif CONFIG['autorun'] is None:
        return not hasattr(__main__, '__file__')
    else:
        return CONFIG['autorun']


def mpl_font_size(key: str) -> float:
    "Font size in inches"
    p = matplotlib.font_manager.FontProperties(size=mpl.rcParams[key])
    return p.get_size() * POINT


def inch_to_figure(figure: matplotlib.figure.Figure, x: float = None, y: float = None):
    "Transform (x, y) vector in inches to figure coordinates"
    coords = figure.dpi_scale_trans.transform((x or 0, y or 0))
    coords_t = figure.transFigure.inverted().transform(coords)
    if x is None:
        return coords_t[1]
    elif y is None:
        return coords_t[0]
    else:
        return coords_t


DISPLAY_UNIT = {
    's': 'ms',
    'V': 'µV',
    'T': 'fT',
    'sensor': int,
}
UNIT_FORMAT = {
    'A': 1,
    'Am': 1,
    'V': 1,
    'ms': 1e3,
    'mV': 1e3,
    'µV': 1e6,
    'pT': 1e12,
    'fT': 1e15,
    'dSPM': 1,
    'p': 1,
    'T': 1,
    'n': int,  # %i format
    'normalized': 1,
    int: int,
}
DEFAULT_CMAPS = {
    'B': 'xpolar',
    'V': 'xpolar',
    'p': 'sig',
    'f': 'viridis',
    'r': 'xpolar',
    't': 'xpolar',
}


class AxisScale:
    """Find matching number formatter and label for display unit (!= data unit)

    Parameters
    ----------
    v
        Display unit or scale of the axis, or daat to infer these. See
        ``unit_format`` dict above for options.
    label
        If ``label is True``, try to infer a label from ``v``.
    """
    def __init__(
            self,
            v: Union[NDVar, Var, Real, str, 'PlotData'],
            label: Union[bool, str, Sequence[str]] = True,
    ):
        if isinstance(v, str):
            data_unit = None
            meas = None
            unit = v
            scale = UNIT_FORMAT.get(v, 1)
        elif isinstance(v, Real):
            data_unit = None
            meas = None
            unit = None
            scale = 1 / v
        else:
            if isnumeric(v):
                meas = v.info.get('meas')
                data_unit = v.info.get('unit')
            elif isinstance(v, PlotData):
                meas = v.meas
                data_unit = v.unit
            else:
                raise TypeError(f"unit={v!r}")

            if data_unit in DISPLAY_UNIT:
                unit = DISPLAY_UNIT[data_unit]
                scale = UNIT_FORMAT[unit]
                if data_unit in UNIT_FORMAT:
                    scale /= UNIT_FORMAT[data_unit]
            else:
                scale = 1
                unit = data_unit
        self.data_unit = data_unit  # None | str
        self.display_unit = unit
        if scale == 1:
            self.formatter = None
        elif scale is int:
            self.formatter = FuncFormatter(lambda x, pos: f'{x:.0f}')
        else:
            self.formatter = FuncFormatter(lambda x, pos: f'{scale * x:g}')

        if label is True:
            if meas and unit and meas not in unit:
                label = f'{meas} [{unit}]'
            elif unit:
                label = unit
            elif meas:
                label = meas
            elif isinstance(v, PlotData):
                label = v.default_y_label
            elif isnumeric(v):
                label = v.name
            else:
                label = None
        self.label = label


def find_uts_hlines(ndvar):
    """Find horizontal lines for uts plots (based on contours)

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.

    Returns
    -------
    h_lines : iterator
        Iterator over (y, kwa) tuples.
    """
    contours = ndvar.info.get('contours', None)
    if contours:
        for level in sorted(contours):
            args = contours[level]
            if isinstance(args, dict):
                yield level, args.copy()
            else:
                yield level, {'color': args}


def find_uts_ax_vlim(
        layers: Sequence[NDVar],
        vlims: Dict = (),
) -> (Optional[float], Optional[float]):
    """Find y axis limits for uts axes

    Parameters
    ----------
    layers
        Data to be plotted.
    vlims
        Vmax and vmin values by (meas, cmap).

    Returns
    -------
    bottom
        Lowest value on y axis.
    top
        Highest value on y axis.
    """
    bottom = None
    top = None
    for ndvar in layers:
        meas = ndvar.info.get('meas')
        if meas in vlims:
            bottom_, top_ = vlims[meas]
            if bottom is None:
                bottom = bottom_
            elif bottom_ != bottom:
                bottom = min(bottom, bottom_)
            if top is None:
                top = top_
            elif top_ != top:
                top = max(top, top_)

    return bottom, top


def find_fig_cmaps(
        epochs: Sequence[Sequence[NDVar]],
        cmap: Union[dict, CMapArg] = None,
        alpha: bool = False,
) -> Dict[str, CMapArg]:
    """Find cmap for every meas

    Parameters
    ----------
    epochs
        All NDVars in the plot.
    cmap
        Use this instead of the default for the first ``meas`` (for user
        argument).
    alpha
        If possible, use cmaps with alpha.

    Returns
    -------
    cmaps
        {meas: cmap} dict for all meas.
    """
    if isinstance(cmap, dict):
        out = cmap.copy()
        cmap = None
    else:
        out = {}

    for ndvar in chain(*epochs):
        meas = ndvar.info.get('meas')

        if meas in out and out[meas]:
            pass
        elif cmap is not None:
            out[meas] = cmap
            cmap = None
        elif 'cmap' in ndvar.info:
            out[meas] = ndvar.info['cmap']
        else:
            out[meas] = None

    for k in out.keys():
        if out[k] is None:
            out[k] = DEFAULT_CMAPS.get(meas, 'xpolar')
        # prefer cmaps with alpha
        if alpha and isinstance(out[k], str) and out[k] in ALPHA_CMAPS:
            out[k] = ALPHA_CMAPS[out[k]]

    return out


def find_fig_contours(epochs, vlims, contours_arg):
    """Find contour arguments for every meas type

    Parameters
    ----------
    epochs : list of list of NDVar
        Data to be plotted.
    vlims : dist
        Vlims dict (used to interpret numerical arguments)
    contours_arg : int | sequence | dict
        User argument. Can be an int (number of contours), a sequence (values
        at which to draw contours), a kwargs dict (must contain the "levels"
        key), or a {meas: kwargs} dictionary.

    Returns
    -------
    contours : dict
        {meas: kwargs} mapping for contour plots.

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if isinstance(contours_arg, dict) and 'levels' not in contours_arg:
        out = contours_arg.copy()
        contours_arg = None
    else:
        out = {}

    for ndvars in epochs:
        for layer, ndvar in enumerate(ndvars):
            meas = ndvar.info.get('meas')
            if meas in out:
                continue

            if contours_arg is not None:
                param = contours_arg
                contours_arg = None
            else:
                if layer:  # overlay
                    kind = ndvar.info.get('overlay', ('contours',))
                else:
                    kind = ndvar.info.get('base', ())

                if 'contours' in kind:
                    param = ndvar.info.get('contours', None)
                    if layer:
                        param = ndvar.info.get('overlay_contours', param)
                    else:
                        param = ndvar.info.get('base_contours', param)

                    if isinstance(param, dict) and 'levels' not in param:
                        levels = sorted(param)
                        colors = [param[v] for v in levels]
                        param = {'levels': levels, 'colors': colors}
                else:
                    param = None

            if param is None:
                out[meas] = None
            elif isinstance(param, dict):
                out[meas] = param
            elif isinstance(param, int):
                vmin, vmax = vlims[meas]
                out[meas] = {'levels': np.linspace(vmin, vmax, param),
                             'colors': 'k'}
            else:
                out[meas] = {'levels': tuple(param), 'colors': 'k'}

    return out


def find_fig_vlims(
        plots: Sequence[Sequence[NDVar]],
        vmax: Union[dict, float] = None,
        vmin: Union[dict, float] = None,
        cmaps: dict = None,
        unmask: bool = True,
):
    """Find vmin and vmax parameters for every (meas, cmap) combination

    Parameters
    ----------
    plots : nested list of NDVar
        Unpacked plot data.
    vmax : None | dict | scalar
        Dict: predetermined vlims (take precedence). Scalar: user-specified
        vmax parameter (used for for the first meas kind).
    vmin : None | scalar
        User-specified vmin parameter. If vmax is user-specified but vmin is
        None, -vmax is used.
    cmaps : dict
        If provided, vlims will be fixed to match symmetric or 0-based cmaps.
    unmask
        Also consider masked values for limits.

    Returns
    -------
    vlims : dict
        Dictionary of im limits: {meas: (vmin, vmax)}.
    """
    vlims = {}
    if isinstance(vmax, dict):
        vlims.update(vmax)
        ndvars = [v for v in chain.from_iterable(plots) if v.info.get('meas') not in vlims]
    else:
        ndvars = [*chain.from_iterable(plots)]

        if vmin is None and vmax is not None:
            if cmaps is None and any(v.min() < 0 for v in ndvars):
                vmin = -vmax
            else:
                vmin = 0

        # apply user specified vlim
        if vmin is not None or vmax is not None:
            meas = ndvars[0].info.get('meas')
            if vmax is None:
                meas_ndvars = [v for v in ndvars if v.info.get('meas') == meas]
                for ndvar in meas_ndvars:
                    _, vmax_ = find_vlim_args(ndvar, unmask=unmask)
                    vmax = vmax_ if vmax is None else max(vmax, vmax_)

            vlims[meas] = (vmin, vmax)
            ndvars = [v for v in ndvars if v.info.get('meas') != meas]

    # for other meas, fill in data limits
    for ndvar in ndvars:
        meas = ndvar.info.get('meas')
        vmin, vmax = find_vlim_args(ndvar, unmask=unmask)
        if meas in vlims:
            vmin_, vmax_ = vlims[meas]
            vmin = vmin if vmin_ is None else min(vmin, vmin_)
            vmax = vmax if vmax_ is None else max(vmax, vmax_)

        vlims[meas] = (vmin, vmax)

    # fix vlims
    for meas in vlims.keys():
        vmin, vmax = vlims[meas]
        if cmaps is not None:
            vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cmaps[meas])
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        vlims[meas] = vmin, vmax

    return vlims


def find_vlim_args(
        ndvar: NDVar,
        vmin: float = None,
        vmax: float = None,
        unmask: bool = True,
):
    if vmax is None:
        vmax = ndvar.info.get('vmax', None)
        if vmin is None:
            vmin = ndvar.info.get('vmin', None)

    if vmax is None or vmin is None:
        if unmask and np.ma.isMaskedArray(ndvar.x):
            data = ndvar.x.data
        else:
            data = ndvar.x
        xmax = np.nanmax(data)
        if np.ma.isMaskedArray(xmax):
            xmax = xmax.data
        xmin = np.nanmin(ndvar.x)
        if np.ma.isMaskedArray(xmin):
            xmin = xmin.data
        abs_max = max(abs(xmax), abs(xmin)) or 1e-14
        if np.isnan(abs_max):
            raise ValueError("Can't plot all NaN input")
        scale = math.floor(np.log10(abs_max))
        if vmax is None:
            vmax = math.ceil(xmax * 10 ** -scale) * 10 ** scale
        if vmin is None:
            vmin = math.floor(xmin * 10 ** -scale) * 10 ** scale

    return vmin, vmax


def fix_vlim_for_cmap(vmin, vmax, cmap):
    "Fix the vmin value to yield an appropriate range for the cmap"
    if isinstance(cmap, LocatedColormap):
        vmin, vmax = cmap.vmin, cmap.vmax
        is_symmetric = cmap.symmetric
        starts_at_zero = False
    else:
        if isinstance(cmap, Colormap):
            cmap = cmap.name
        is_symmetric = cmap in SYMMETRIC_CMAPS
        starts_at_zero = cmap in ZEROBASED_CMAPS

    if is_symmetric:
        if vmax is None and vmin is None:
            pass
        elif vmin is None:
            vmax = abs(vmax)
            vmin = -vmax
        elif vmax is None:
            vmax = abs(vmin)
            vmin = -vmax
        else:
            vmax = max(abs(vmax), abs(vmin))
            vmin = -vmax
    elif starts_at_zero:
        vmin = 0
    return vmin, vmax


def find_data_dims(
        ndvar: NDVar,
        dims: Union[int, Tuple[str, ...]],
        extra_dim: str = None,
) -> Tuple[Union[str, None], List[str]]:
    """Find dimensions in data.

    Raise a ValueError if the dimensions don't match, except when the ``case``
    dimension is omitted in ``dims``.

    Parameters
    ----------
    ndvar
        NDVar instance to query.
    dims
        The requested dimensions.
    extra_dim
        Dimension that will be removed by other operation (e.g. ``xax``).

    Returns
    -------
    agg : None | str
        Dimension to aggregate over.
    dims : list of str
        Dimension names with all instances of ``None`` replaced by a string.
    """
    if isinstance(dims, int):
        if extra_dim:
            dims += 1

        dimnames = list(ndvar.dimnames)
        if ndvar.ndim == dims:
            agg = None
        elif ndvar.ndim == dims + 1:
            for agg in dimnames:
                if agg != extra_dim:
                    break
            dimnames.remove(agg)
        else:
            raise ValueError(f"y={ndvar} has wrong number of dimensions; {dims} or {dims + 1} required")
    else:
        required_dims = (extra_dim, *dims) if extra_dim else dims
        if ndvar.ndim == len(required_dims):
            agg = None
            dimnames = list(ndvar.get_dimnames(required_dims))
        elif ndvar.ndim == len(required_dims) + 1:
            if any(d is None for d in required_dims):
                if ndvar.has_case and 'case' not in required_dims:
                    agg = 'case'
                else:
                    raise ValueError(f"y={ndvar} is ambiguous for required dimensions {required_dims}")
            else:
                agg = None
            dimnames = list(ndvar.get_dimnames((agg, *required_dims)))
            agg = dimnames.pop(0)
        else:
            n_required = len(required_dims)
            specific = ', '.join([dim for dim in required_dims if dim is not None])
            raise ValueError(f"y={ndvar} has wrong dimensions; need {n_required} or {n_required + 1} dimensions, including {specific}")

        if extra_dim:
            dimnames.remove(extra_dim)
    return agg, dimnames


def find_labels(
        cells: Sequence[CellArg],
        labels_arg: Dict[CellArg: str] = None,
        delim: str = ' ',
) -> Dict[CellArg, str]:
    if not labels_arg:
        return {cell: cellname(cell, delim) for cell in cells}
    labels = {}
    for cell in cells:
        if cell in labels_arg:
            label = labels_arg[cell]
        elif isinstance(cell, str):
            label = cell
        elif isinstance(cell, tuple):
            label = cellname([labels_arg.get(item, item) for item in cell], delim)
        else:
            raise TypeError(f"{cell=}")
        labels[cell] = label
    return labels


def brain_data(
        data: Union[NDVar, testnd.NDTest],
        dataset: Dataset = None,
):
    # for GlassBrain and surfer brain
    if isinstance(data, testnd.NDDifferenceTest):
        return data.masked_difference()
    else:
        return asndvar(data, data=dataset)


def butterfly_data(
        data: Union[NDVar, testnd.NDTest],
        hemi: str,
        resample_: int = None,
        colors: bool = False,
        return_vector_data: bool = False,
):
    """Data for plotting butterfly plot with brain

    Returns
    -------
    hemis : list of str
        Hemispheres in the data.
    butterfly_daya :
        Data for Butterfly plot.
    brain_data :
        Data for brain plot.
    """
    # find input type
    if isinstance(data, NDVar):
        y = data
        kind = 'ndvar'
    elif isinstance(data, testnd.NDDifferenceTest):
        y = data.masked_difference()
        kind = 'ndvar'
    else:
        raise TypeError(f"ndvar={data!r}")
    source = y.get_dim('source')

    # find samplingrate
    if resample_ is not None:
        raise NotImplementedError(f"resample_={resample_}")

    # find hemispheres to include
    if hemi is None:
        hemis = []
        if source.lh_n:
            hemis.append('lh')
        if source.rh_n:
            hemis.append('rh')
    elif hemi in ('lh', 'rh'):
        hemis = [hemi]
    else:
        raise ValueError("hemi=%r" % (hemi,))

    if kind == 'ndvar':
        if y.has_case:
            y = y.mean('case')
        if resample_:
            y = resample(y, resample_, window='hamming')
        if y.has_dim('space'):
            if return_vector_data:
                brain_data = y
                y = y.norm('space')
            else:
                y = y.norm('space')
                brain_data = y
        else:
            brain_data = y
        bfly_data = [y.sub(source=hemi, name=hemi.capitalize()) for hemi in hemis]
    elif kind == 'test':
        sig = data.p <= 0.05
        y_magnitude = y.rms('time')
        # resample
        if resample_:
            y = resample(y, resample_, window='hamming')
            sig = resample(sig, resample_) > 0.5
        brain_data = y.mask(~sig)
        # mask
        non_sig = erode(~sig, 'time')
        y_sig = y.mask(non_sig)
        y_ns = y.mask(sig)
        # line-styles
        if colors:
            lh_color = '#046AAD'
            rh_color = '#A60628'
            line_color_sig = {'lh': lh_color, 'rh': rh_color}
            line_color_ns = {
                'lh': adjust_hsv(lh_color, 0, -0.5, -0.),
                'rh': adjust_hsv(rh_color, 0, -0.7, -0.),
            }
        else:
            color_sig = (0,) * 3
            color_ns = (.7,) * 3
            line_color_sig = {'lh': color_sig, 'rh': color_sig}
            line_color_ns = {'lh': color_ns, 'rh': color_ns}
        linestyle_ns = {'linewidth': 0.2, 'color': line_color_ns, 'alpha': 0.2}
        linestyle_sig = {'linewidth': 0.2, 'color': line_color_sig, 'alpha': 1.0}
        # layer-data
        axes = []
        for hemi in hemis:
            # z-order
            y_mag = y_magnitude.sub(source=hemi)
            z_order = dict(zip(y_mag.source, -y_mag.x.argsort()))
            # data
            layers = []
            for y, linestyle in ((y_ns, linestyle_ns), (y_sig, linestyle_sig)):
                kwargs = {'zorder': z_order, **linestyle}
                layers.append(DataLayer(y.sub(source=hemi), PlotType.LINE, kwargs))
            axes.append(AxisData(layers))
        bfly_data = PlotData(axes, ('time', 'source'), plot_names=hemis)
    else:
        raise RuntimeError(f"{kind=}")
    return hemis, bfly_data, brain_data


def pop_if_dict(kwargs, key):
    "Helper for artist-sepcific matplotlib kwargs"
    if key in kwargs and isinstance(kwargs[key], dict):
        return kwargs.pop(key)


def set_dict_arg(key, arg, line_dim_obj, artists, legend_handles=None):
    "Helper for artist-sepcific matplotlib kwargs"
    set_attr_name = 'set_' + key
    for dim_index, value in arg.items():
        index = line_dim_obj._array_index(dim_index)
        if isinstance(index, int):
            key_artists = [artists[index]]
        else:
            key_artists = artists[index]

        if not key_artists:
            continue

        for artist in key_artists:
            getattr(artist, set_attr_name)(value)

        if legend_handles is not None:
            for artist in key_artists:
                artist.set_label(dim_index)
            legend_handles[dim_index] = artist


_remap_args = {'c': 'color'}


def _dict_arg(arg: dict = None) -> dict:
    if arg is None:
        return {}
    elif any(k in arg for k in _remap_args):
        return {_remap_args.get(k, k): v for k, v in arg.items()}
    else:
        return arg


@dataclass(eq=False)
class Layer:
    y: NDVar
    plot_type: PlotType = PlotType.GENERAL

    def bin(self, bin_length, tstart, tstop):
        raise NotImplementedError

    def sub_time(self, time: float, data_only: bool = False):
        raise NotImplementedError

    def for_plot(self, plot_type: PlotType) -> Iterator['DataLayer']:
        raise NotImplementedError


@dataclass(eq=False)
class DataLayer(Layer):
    """Data for one subplot layer"""
    _plot_args: dict = None
    _plot_args_2: dict = None  # alternate (contour plot of IMAGE layer)
    _bin_func: callable = np.mean

    def __post_init__(self):
        self._plot_args = _dict_arg(self._plot_args)
        if self.plot_type == PlotType.IMAGE:
            self._plot_args_2 = _dict_arg(self._plot_args_2)
        elif self._plot_args_2:
            raise TypeError(f"plot_args_2={self._plot_args_2!r} for {self.plot_type}")

    def plot_args(self, kwargs: dict) -> dict:
        # needs to be a copy?
        return {**_dict_arg(kwargs), **self._plot_args}

    def contour_plot_args(self, contours):
        out = {}
        # contours arg
        meas = self.y.info.get('meas')
        if meas in contours:
            if contours[meas] is not None:
                out.update(contours[meas])
        # layer
        if self.plot_type == PlotType.IMAGE:
            out.update(self._plot_args_2)
        elif self.plot_type == PlotType.CONTOUR:
            out.update(self._plot_args)
        else:
            raise RuntimeError(f"layer of type {self.plot_type}")
        return out

    def im_plot_args(self, vlims: dict, cmaps: dict) -> dict:
        assert self.plot_type == PlotType.IMAGE
        meas = self.y.info.get('meas')
        if meas in cmaps:
            cmap = cmaps[meas]
        elif 'cmap' in self.y.info:
            cmap = self.y.info['cmap']
        else:
            cmap = DEFAULT_CMAPS.get(meas, 'xpolar')

        if meas in vlims:
            vmin, vmax = vlims[meas]
        else:
            vmin, vmax = find_vlim_args(self.y)
            vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cmap)
        return {'cmap': cmap, 'vmin': vmin, 'vmax': vmax, **self._plot_args}

    def for_plot(self, plot_type: PlotType) -> Iterator['DataLayer']:
        if self.plot_type == plot_type:
            yield self
        elif not np.ma.isMaskedArray(self.y.x):
            yield DataLayer(self.y, plot_type, self._plot_args)
        elif plot_type == PlotType.LEGACY:
            yield DataLayer(self.y.unmask(), plot_type, self._plot_args)
        elif self.plot_type != PlotType.GENERAL:
            raise RuntimeError(f"Invalid PlotData conversion: {self.plot_type} -> {plot_type}")
        elif plot_type == PlotType.LINE:
            un_mask = NDVar(~self.y.x.mask, self.y.dims)
            # kwargs = {}
            if self.y.has_dim('time'):
                un_mask = erode(un_mask, 'time')
                # if self.y.ndim == 2:
                #     mag = self.y.rms('time')
                #     z_dim = mag.dimnames[0]
                #     kwargs['zorder'] = dict(zip(mag.get_dim(z_dim), -mag.x.argsort()))
            y_masked = self.y.unmask().mask(un_mask)
            args_main = {'alpha': 1., 'zorder': 1}
            args_masked = {'alpha': 0.4, 'color': (.7, .7, .7), 'zorder': 0}
            for y, args in ((self.y, args_main), (y_masked, args_masked)):
                yield DataLayer(y, plot_type, args)
        elif plot_type == PlotType.IMAGE:
            x = NDVar(self.y.x.data, self.y.dims, self.y.name, self.y.info)
            yield DataLayer(x, PlotType.IMAGE)
            x = NDVar(1. - self.y.x.mask, self.y.dims, self.y.name, {'meas': 'mask'})
            yield DataLayer(x, PlotType.CONTOUR, {'levels': [0.5], 'colors': ['black']}, _bin_func=np.max)
        else:
            raise RuntimeError(f"{plot_type=}")

    def sub_time(self, time: float, data_only: bool = False):
        y = self.y.sub(time=time)
        if data_only:
            return y
        else:
            return DataLayer(y, self.plot_type, self._plot_args, self._plot_args_2, self._bin_func)

    def bin(self, bin_length, tstart, tstop):
        y = self.y.bin(bin_length, tstart, tstop, self._bin_func)
        return replace(self, y=y)


@dataclass(eq=False)
class StatLayer(Layer):
    style: Style = None
    ct: Celltable = None
    cell: CellArg = None  # key for Celltable
    mask: NDVar = None  # mask needs to be applied to stats
    mask_missing: bool = True
    style_key: CellArg = None  # Key for legend - usually equal to cell

    def __post_init__(self):
        if self.style_key is None:
            self.style_key = self.cell

    def _apply_mask(self, y: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return y
        if np.ma.isMaskedArray(y):
            y = y.data
        return NDVar(y, self.y.dims[1:]).mask(self.mask, missing=self.mask_missing).x

    def get_statistic(self, func: Callable = np.mean) -> np.ndarray:
        return self._apply_mask(self.ct._get_func(self.cell, func))

    def get_dispersion(self, spec, within_subject) -> np.ndarray:
        return self._apply_mask(self.ct.variability(spec, within_subject, self.cell, True))

    def for_plot(self, plot_type: PlotType) -> Iterator['DataLayer']:
        if self.plot_type == plot_type:
            yield self
        elif self.mask is None:
            yield replace(self, plot_type=plot_type)
        elif plot_type == PlotType.LINE:
            inverse_mask = ~self.mask
            if self.y.has_dim('time'):
                inverse_mask = erode(inverse_mask, 'time')
            yield replace(self, plot_type=plot_type, style=self.style.masked_style, mask=inverse_mask, mask_missing=False)
            yield replace(self, plot_type=plot_type)
        else:
            raise NotImplementedError


@dataclass(eq=False)
class AxisData:
    """Represent one axis (multiple layers)"""
    layers: List[Layer]
    title: str = None

    def __iter__(self):
        return iter(self.layers)

    @property
    def y0(self):
        for layer in self.layers:
            return layer.y
        raise IndexError("No data")

    @cached_property
    def ndvars(self):
        return [layer.y for layer in self.layers]

    def for_plot(self, plot_type: PlotType) -> 'AxisData':
        return replace(self, layers=[l for layer in self.layers for l in layer.for_plot(plot_type)])

    def bin(self, bin_length, tstart, tstop):
        return replace(self, layers=[l.bin(bin_length, tstart, tstop) for l in self.layers])

    def sub_time(self, time: float, data_only: bool = False):
        axis = []
        for layer in self.layers:
            if time in layer.y.time:
                axis.append(layer.sub_time(time, data_only))
        if data_only:
            return axis
        else:
            return replace(self, layers=axis)


def x_arg(x: CategorialArg):
    if isinstance(x, str) and x.startswith('.'):
        return None, x[1:]
    else:
        return x, None


def combine_x_args(x, xax):
    if x is None:
        return xax
    elif xax is None:
        return x
    elif not isinstance(xax, x.__class__):
        raise TypeError(f"x={x}, xax={xax}: x and xax must be of same type or None")
    elif isinstance(xax, str):
        return f"({x}) % ({xax})"
    else:
        return x % xax


@dataclass(eq=False)
class PlotData:
    """Organize nd-data for plotting

    Notes
    -----
    Two-stage initialization:

     - Initially independent of plot type
       - Layers contain masked NDVars

     - Converted into specific plot-types with :meth:`.for_plot` methods.
       - Masks converted into layers


    Additional consideration for statistics-plots (UTSStat)
     - Need central tendency and dispersion
     - Dispersion is not derivable from layer data (within-subject SEM)
     - Ideally potential to be dynamic (switch between viewing all data and statistic)

     Implementation
     - DataLayer subclass that keeps reference to a CellTable?

    """
    plot_data: List[AxisData]  # Data for each axis
    dims: Sequence[str]  # Dimensions assigned to the axes
    frame_title: str = "unnamed data"  # Default window title
    plot_names: List[Union[str, None]] = None  # Titles for the plots (all non-None axes)
    plot_used: List[bool] = None  # List indicating which plot slots are used
    plot_type: PlotType = PlotType.GENERAL
    ct: Celltable = None
    x: Union[Factor, Interaction] = None
    xax: Union[Factor, Interaction] = None
    styles: Dict[CellArg, Style] = None

    def __post_init__(self):
        self.n_plots = len(self.plot_data)
        if self.plot_used is None:
            self.plot_used = [True] * self.n_plots
        else:
            assert sum(self.plot_used) == self.n_plots
        if self.plot_names is None:
            self.plot_names = []
            for axis_data in self.plot_data:
                if axis_data.title:
                    axis_title = axis_data.title
                else:
                    for layer in axis_data:
                        if layer.y.name:
                            axis_title = layer.y.name
                            break
                    else:
                        axis_title = None
                self.plot_names.append(axis_title)

    def __repr__(self):
        desc = [f'{self.n_plots} plots']
        if not all(self.plot_used):
            desc.append(f'{len(self.plot_used) - self.n_plots} empty')
        desc.append(' x '.join(self.dims))
        return f"<PlotData {self.frame_title!r}: {', '.join(desc)}>"

    def _cannot_skip_axes(self, parent):
        if not all(self.plot_used):
            raise NotImplementedError(f"y can not contain None for {parent.__class__.__name__} plot")

    def __iter__(self):
        return iter(self.plot_data)

    @classmethod
    def from_args(
            cls,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            dims: Union[int, Tuple[Optional[str], ...]],
            xax: CategorialArg = None,
            data: Dataset = None,
            sub: IndexArg = None,
    ):
        """Unpack the first argument to top-level NDVar plotting functions

        Parameters
        ----------
        y
            the first argument.
        dims
            The dimensions needed for the plotting function. ``None`` to indicate
            arbitrary dimensions.
        xax
            A model to divide ``y`` into different axes. ``xax`` is currently
            applied on the first level, i.e., it assumes that ``y``'s first
            dimension is cases.
        data
            Dataset containing data objects which are provided as :class:`str`.
        sub
            Index selecting a subset of cases.

        Notes
        -----
        Ndvar plotting functions above 1-d UTS level should support the following
        API:

         - simple NDVar: summary ``plot(meg)``
         - by dim: each case ``plot(meg, '.case')``
         - NDVar and xax argument: summary for each  ``plot(meg, subject)
         - nested list of layers (e.g., ttest results: [c1, c0, [c1-c0, p]])
        """
        if isinstance(y, cls):
            return y
        elif isinstance(y, AxisData):
            for layer in y.layers:
                dims = find_data_dims(layer.y, dims)
            return PlotData([y], dims)
        sub = assub(sub, data)
        if hasattr(y, '_default_plot_obj'):
            ys = getattr(y, '_default_plot_obj')()
        elif isinstance(y, (tuple, list)):
            ys = [getattr(yi, '_default_plot_obj')() if hasattr(yi, '_default_plot_obj') else yi for yi in y]
        else:
            ys = y

        if isinstance(ys, NDVarTypes):
            ys = (ys,)

        ax_names = None
        if xax is None:
            # y=[[y1], y2], xax=None
            axes = []
            for ax in ys:
                if ax is None:
                    axes.append(None)
                elif isinstance(ax, NDVarTypes):
                    ax = asndvar(ax, sub, data)
                    agg, dims = find_data_dims(ax, dims)
                    layer = aggregate(ax, agg)
                    axes.append([layer])
                else:
                    layers = []
                    for layer in ax:
                        layer = asndvar(layer, sub, data)
                        agg, dims = find_data_dims(layer, dims)
                        layers.append(aggregate(layer, agg))
                    axes.append(layers)
            x_name = None
            # determine y names
            y_names = []
            for layers in axes:
                if layers is None:
                    continue
                for layer in layers:
                    if layer.name and layer.name not in y_names:
                        y_names.append(layer.name)
        elif all(ax is None or isinstance(ax, NDVarTypes) for ax in ys):
            ys = [asndvar(layer, sub, data) for layer in ys]
            y_names = [layer.name for layer in ys]
            layers = []
            if isinstance(xax, str) and xax.startswith('.'):
                # y=[y1, y2], xax='.dim'
                dimname, attr = re.match(r'\.(\w+)(?:\.(\w+))?$', xax).groups()
                xax_dim = indexes = dissolve_dim = None
                for layer in ys:
                    dim = layer.get_dim(dimname)
                    if xax_dim is None:
                        xax_dim = dim
                        if attr is None:
                            indexes = dim
                            dissolve_dim = dimname  # dissolved by indexing
                            # axis labels
                            unit = f' {xax_dim._axis_unit}' if xax_dim._axis_unit else ''
                            ax_names = [f'{v}{unit}' for v in xax_dim]
                        else:
                            f = getattr(dim, attr)
                            if not isinstance(f, Factor):
                                raise ValueError(f'{xax=}')
                            indexes = [f == cell for cell in f.cells]
                            ax_names = f.cells
                    elif dim != xax_dim:
                        raise ValueError(f"{y=}, {xax=}: dimension not equal on different y")
                    agg, dims = find_data_dims(layer, dims, dissolve_dim)
                    layers.append([aggregate(layer.sub(**{dimname: i}), agg) for i in indexes])
                x_name = xax
            else:
                # y=[y1, y2], xax=categorial
                xax = ascategorial(xax, sub, data)
                xax_indexes = [xax == cell for cell in xax.cells]
                for layer in ys:
                    agg, dims = find_data_dims(layer, dims)
                    layers.append([aggregate(layer.sub(index), agg) for index in xax_indexes])
                x_name = xax.name
                ax_names = [cellname(cell) for cell in xax.cells]
            axes = list(zip(*layers))
        else:
            raise TypeError(f"{y=}, {xax=}: y can't be nested list if xax is specified, use single list")

        if len(y_names) == 0:
            y_name = None
        elif len(y_names) == 1:
            y_name = y_names[0]
        else:
            y_name = ', '.join(y_names)

        use_axes = [ax is not None for ax in axes]
        axes = [AxisData([DataLayer(l) for l in ax]) for ax in axes if ax]
        title = frame_title(y_name, x_name)
        return cls(axes, dims, title, ax_names, use_axes)

    @classmethod
    def from_stats(
            cls,
            y: Union[NDVarArg, Sequence[NDVarArg]],
            x: CategorialArg = None,
            xax: CategorialArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            ds: Dataset = None,
            dims: Tuple[Union[str, None]] = None,
            colors: dict = None,
            mask: Union[NDVar, Dict[CellArg, NDVar]] = None,
            level: Literal[1, 2, 3] = 1,
    ):
        if isinstance(y, (tuple, list)):
            if level == 2:
                if x is not None:
                    raise TypeError(f"Nested y, {x=}: x cannot be specified with nested y")
                data = [cls.from_stats(yi, x, xax, match, sub, ds, dims, colors, mask, level=3) for yi in y]
                layers = [data_i.plot_data[0].layers[0] for data_i in data]
                plot_data = [replace(data[0].plot_data[0], layers=layers)]
                return replace(data[0], plot_data=plot_data)
            elif level == 3:
                raise TypeError("Doubly nested y")
            elif xax is not None:
                raise TypeError(f"{y=}, {xax=}: xax cannot be specified with multiple y")
            plot_used = [yi is not None for yi in y]
            axes_data = [cls.from_stats(yi, x, xax, match, sub, ds, dims, mask=mask, level=2) for yi in y if yi is not None]
            axes = list(chain.from_iterable(ax.plot_data for ax in axes_data))
            # cells for styles are based on y and x
            for ax in axes:
                for layer in ax.layers:
                    layer.style_key = combine_cells(layer.ct.y.name, layer.cell)
            cells = {layer.style_key for ax in axes for layer in ax.layers}
            styles = find_cell_styles(cells, colors)
            for ax in axes:
                for layer in ax.layers:
                    layer.style = styles[layer.style_key]
            return replace(axes_data[0], plot_data=axes, plot_used=plot_used, plot_names=None, styles=styles)
        x, x_dim = x_arg(x)
        if x_dim is not None:
            ds = melt_ndvar(y, x_dim, ds=ds)
            y = ds.info['varname']
            x = combine_x_args(x, x_dim)
        xax, xax_dim = x_arg(xax)
        if xax_dim is not None:
            ds = melt_ndvar(y, xax_dim, ds=ds)
            y = ds.info['varname']
            xax = combine_x_args(xax, xax_dim)
        x_full = combine_x_args(x, xax)
        ct = Celltable(y, x_full, match, sub, data=ds)
        # data dimensions
        agg, dims = find_data_dims(ct.y, dims, 'case')
        if agg:
            raise NotImplementedError
        # reconstruct x/xax
        if x is not None:
            x = ct._align(x, data=ds, coerce=ascategorial)
        if xax is None:
            default_color_cells = None
            ax_cells = [None]
        else:
            xax = ct._align(xax, data=ds, coerce=ascategorial)
            ax_cells = xax.cells
            if x is None:
                default_color_cells = None
            else:
                default_color_cells = x.cells
        title = frame_title(y, x, xax)
        # find styles
        styles = find_cell_styles(ct.cells, colors, default_cells=default_color_cells)
        # find masks
        if mask is None:
            masks = defaultdict(lambda: None)
        elif isinstance(mask, NDVar):
            masks = defaultdict(lambda: mask)
        elif isinstance(mask, dict):
            masks = defaultdict(lambda: None, **mask)
        else:
            raise TypeError(f"{mask=}")
        # assemble layers
        axes = []
        for ax_cell in ax_cells:
            if x is None:
                cells = [ax_cell]
            elif ax_cell is None:
                cells = x.cells
            else:
                x_cells = x.cells
                cells = [combine_cells(x_cell, ax_cell) for x_cell in x_cells]
                cells = [cell for cell in cells if cell in ct.data]
            layers = [StatLayer(ct.data[cell], style=styles[cell], ct=ct, cell=cell, mask=masks[cell]) for cell in cells]
            axes.append(AxisData(layers, cellname(ax_cell)))
        return cls(axes, dims, title, ct=ct, x=x, xax=xax, styles=styles)

    @classmethod
    def empty(cls, plots: Union[int, List[bool]], dims: Sequence[str], title: str):
        """Empty PlotData object that can be filled by appending to layers

        Parameters
        ----------
        plots : int | list of bool
            Number of plots, or list of booleans indicating for each plot
            whether its slot is used.
        dims : sequence of str
            Names of the dimensions.
        title : str
            Data description for the plot frame.
        """
        if isinstance(plots, int):
            plots = [AxisData([]) for _ in range(plots)]
        else:
            plots = [AxisData([]) if p else None for p in plots]
        return cls(plots, dims, title)

    @property
    def y0(self):
        for ax in self.plot_data:
            for layer in ax:
                return layer.y
        raise IndexError("No data")

    @property
    def default_y_label(self):
        "Y-label in case meas and unit are uninformative"
        names = {l.y.name for ax in self.plot_data for l in ax}
        names.discard(None)
        if len(names) == 1:
            return names.pop()
        return None

    @property
    def meas(self):
        meass = {l.y.info.get('meas') for ax in self.plot_data for l in ax}
        meass.discard(None)
        if len(meass) == 1:
            return meass.pop()
        return None

    @property
    def unit(self):
        units = {l.y.info.get('unit') for ax in self.plot_data for l in ax}
        units.discard(None)
        if len(units) == 1:
            return units.pop()
        return None

    @cached_property
    def data(self):
        "For backwards compatibility with nested list of NDVar"
        data = self.for_plot(PlotType.LEGACY)
        return [[l.y for l in layers] for layers in data.plot_data]

    @cached_property
    def time_dim(self):
        "UTS dimension to expose for time slicer"
        time_dims = [l.y.get_dim('time') for ax in self.plot_data for l in ax.layers if l.y.has_dim('time')]
        if time_dims:
            return reduce(UTS._union, time_dims)

    def for_plot(self, plot_type: PlotType) -> 'PlotData':
        if self.plot_type == plot_type:
            return self
        plot_data = [ax.for_plot(plot_type) for ax in self.plot_data]
        return replace(self, plot_data=plot_data, plot_type=plot_type)

    def bin(self, bin_length, tstart, tstop):
        axes = [ax.bin(bin_length, tstart, tstop) for ax in self.plot_data]
        return PlotData(axes, self.dims, self.frame_title, self.plot_names, self.plot_used, self.plot_type)

    def sub_time(self, time: float, data_only: bool = False):
        axes = [ax.sub_time(time, data_only) for ax in self.plot_data]
        if data_only:
            return axes
        else:
            dims = [dim for dim in self.dims if dim != 'time']
            return PlotData(axes, dims, self.frame_title, self.plot_names, self.plot_used, self.plot_type)


def aggregate(y, agg):
    return y if agg is None else y.mean(agg)


class FigureFrame:
    """Stand in for a wx.Frame with other backends"""

    def __init__(self, figure):
        self.figure = figure
        self.canvas = self.figure.canvas
        self._background = None

    def Close(self):  # noqa
        pass

    def SetStatusText(self, text):  # noqa
        pass

    def Show(self):  # noqa
        pass

    def redraw(self, axes=()):
        "Adapted duplicate of mpl_canvas.FigureCanvasPanel"
        self.canvas.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
        self.canvas.blit(self.figure)

    def store_canvas(self):
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)


class MatplotlibFrame(FigureFrame):
    "Cf. _wxgui.mpl_canvas"
    def __init__(self, **fig_kwargs):
        "Create self.figure and self.canvas attributes and return the figure"
        from matplotlib import pyplot

        figure = pyplot.figure(**fig_kwargs)
        FigureFrame.__init__(self, figure)
        self._plt = pyplot

    def Close(self):  # noqa
        self._plt.close(self.figure)

    def Show(self):  # noqa
        if mpl.get_backend() == 'WXAgg' and do_autorun():
            self._plt.show()


def frame_title(y, x=None, xax=None):
    """Generate frame title from common data structure

    Parameters
    ----------
    y : data-obj | str
        Dependent variable.
    x : data-obj | str
        Predictor.
    xax : data-obj | str
        Grouping variable for axes.
    """
    if isdataobject(y):
        y = y.name
    if isdataobject(x):
        x = x.name
    if isdataobject(xax):
        xax = xax.name

    if xax is None:
        if x is None:
            return "%s" % (y,)
        else:
            return "%s ~ %s" % (y, x)
    elif x is None:
        return "%s | %s" % (y, xax)
    else:
        return "%s ~ %s | %s" % (y, x, xax)


class MatplotlibFigure:
    """Wrap a matplotlib figure for FMText"""
    def __init__(self, figure):
        self.figure = figure

    def _asfmtext(
            self,
            rasterize: bool = None,
            close_figures: bool = None,
    ):
        if rasterize is None:
            format = None
        elif rasterize:
            format = 'png'
        else:
            format = 'svg'
        return self.image(format=format, close=close_figures)

    def close(self):
        from matplotlib import pyplot
        pyplot.close(self.figure)

    def image(self, name: str = None, format: str = None, close: bool = None):
        """Create FMTXT Image from the figure

        Parameters
        ----------
        name
            Name for the file (without extension; default is 'image').
        format
            File format. For HTML, use ``svg`` for vector graphics and ``png``
            for pixel graphics. The default is ``svg`` and can be changed with
            :func:`configure`).
        close
            Close the figure after writing to the ``image``. By default, this is
            ``True`` when in an inline context (Jupyter notebook), ``False``
            otherwise).

        Returns
        -------
        image : fmtxt.Image
            Image FMTXT object.
        """
        if format is None:
            format = CONFIG['format']
        image = Image(name, format)
        self.figure.savefig(image, format=format)
        if close or (close is None and use_inline_backend()):
            self.close()
        return image


class EelFigure(MatplotlibFigure):
    """Parent class for Eelbrain figures.

    In order to subclass:

     - find desired figure properties and then use them to initialize
       the _EelFigure superclass; then use the
       :py:attr:`_EelFigure.figure` and :py:attr:`_EelFigure.canvas` attributes.
     - end the initialization by calling `_EelFigure._show()`
     - add the :py:meth:`_fill_toolbar` method
    """
    _default_xlabel_ax = -1
    _default_ylabel_ax = 0
    _make_axes = True
    _can_set_time = False
    _can_set_vlim = False
    _can_set_ylim = False
    _can_set_xlim = False
    _has_frame = False

    def __init__(self, data_desc: Optional[str], layout: BaseLayout):
        """Parent class for Eelbrain figures.

        Parameters
        ----------
        data_desc
            Data description for frame title.
        layout
            Layout that determines figure dimensions.
        """
        name = self.__class__.__name__
        desc = layout.name or data_desc
        self._title = f'{name}: {desc}' if desc else name

        # Use Eelbrain frame or pyplot
        if layout.user_axes:
            ax = layout.user_axes[0]
            frame = FigureFrame(ax.get_figure())
        elif CONFIG['eelbrain'] and not use_inline_backend():
            from .._wxgui import wx, get_app
            from .._wxgui.mpl_canvas import CanvasFrame
            get_app()
            pos = wx.DefaultPosition if layout.pos is None else layout.pos
            frame = CanvasFrame(title=self._title, eelfigure=self, pos=pos, **layout.fig_kwa())
            self._has_frame = True
        else:
            frame = MatplotlibFrame(**layout.fig_kwa())

        figure = frame.figure
        if layout.title:
            self._figtitle = figure.suptitle(layout.title)
        else:
            self._figtitle = None

        # make axes
        if self._make_axes:
            axes = layout.make_axes(figure)
        else:
            axes = []

        # store attributes
        MatplotlibFigure.__init__(self, figure)
        self._frame = frame
        self.axes = axes
        self.canvas = frame.canvas
        self._layout = layout
        self._last_draw_time = 1.
        self.__callback_key_press = {}
        self.__callback_key_release = {}

        # containers for hooks
        self._draw_hooks = []
        self._untight_draw_hooks = []

        # options
        self._draw_crosshairs = False
        self._crosshair_lines = None
        self._crosshair_axes = None

        # add callbacks
        if use_inline_backend():
            return
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)
        self.canvas.mpl_connect('resize_event', self._on_resize)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.canvas.mpl_connect('key_release_event', self._on_key_release)

    def __repr__(self):
        title = self._frame.GetTitle() if self._has_frame else self._title
        return f'<{title}>'

    def _ipython_display_(self):
        from IPython.display import display
        display(self.figure)

    def _set_axtitle(
            self,
            axtitle: Union[bool, str, Iterator[str]] = None,
            data: PlotData = None,
            axes: Union[List[matplotlib.axes.Axes], int] = None,
            names: Sequence[str] = None,
            **kwargs,
    ):
        """Set axes titles automatically

        Parameters
        ----------
        axtitle
            Plot parameter.
        data
            Plotted data (if available).
        axes
            Axes for which to set title (default is self.axes). If an int,
            (n axes) the method does not set axes title but returns ``None``
            or a tuple of titles.
        names
            Instead of using ``epochs`` name attributes, use these names.
        ...
            Matplotlib ``Axes.set_title()`` parameters.
        """
        if axtitle is False or axtitle is None:
            return

        if axes is None:
            axes = self.axes

        naxes = axes if isinstance(axes, int) else len(axes)

        if axtitle is True and naxes == 1:
            return
        elif axtitle is True or isinstance(axtitle, str):
            if names is None:
                if data is None:
                    raise RuntimeError(f"data=None and names=None with {axtitle=}")
                names = data.plot_names

            if axtitle is True:
                axtitle = names
            else:
                axtitle = [axtitle.format(name=n) if n else None for n in names]

        if isinstance(axes, int):
            return axtitle

        for title, ax in zip(axtitle, axes):
            if title is not None:
                ax.set_title(asfmtext(title), **kwargs)

    def _show(self, crosshair_axes=None):
        if self._layout.user_axes:
            return
        if self._layout.tight:
            self._tight()

        if crosshair_axes is None:
            self._crosshair_axes = self.axes
        else:
            self._crosshair_axes = crosshair_axes

        self.draw()

        # Allow hooks to modify figure after first draw
        need_redraw = any([func() for func in self._draw_hooks])
        if not self._layout.tight:
            need_redraw = any([func() for func in self._untight_draw_hooks]) or need_redraw
        if need_redraw:
            self.draw()

        if CONFIG['show'] and self._layout.show:
            self._frame.Show()
            if self._has_frame and do_autorun(self._layout.run):
                from .._wxgui import run
                run()

        if self._has_frame and not self.canvas._background:
            self._frame.store_canvas()

    def _tight(self):
        "Default implementation based on matplotlib"
        try:
            self.figure.tight_layout()
        except ValueError as exception:
            getLogger('eelbrain').debug('tight-layout: %s', exception)

        if self._figtitle:
            trans = self.figure.transFigure.inverted()
            extent = self._figtitle.get_window_extent(self.figure.canvas.renderer)
            bbox = trans.transform(extent)
            t_bottom = bbox[0, 1]
            self.figure.subplots_adjust(top=1 - 2 * (1 - t_bottom))

    def _on_key_press(self, event):
        if event.key in self.__callback_key_press:
            self.__callback_key_press[event.key](event)
            event.guiEvent.Skip(False)  # Matplotlib Skip()s all events

    def _on_key_release(self, event):
        if event.key in self.__callback_key_release:
            self.__callback_key_release[event.key](event)
            event.guiEvent.Skip(False)

    def _on_leave_axes(self, event):
        "Update the status bar when the cursor leaves axes"
        if self._frame is None:
            return
        self._frame.SetStatusText(self._on_leave_axes_status_text(event))
        if self._draw_crosshairs:
            self._remove_crosshairs(True)

    def _on_leave_axes_status_text(self, event):
        return '☺︎'

    def _on_motion(self, event):
        "Update the status bar for mouse movement"
        if self._frame is None:
            return
        redraw_axes = self._on_motion_sub(event)
        ax = event.inaxes
        # draw crosshairs
        if self._draw_crosshairs and ax in self._crosshair_axes:
            if self._crosshair_lines is None:
                self._crosshair_lines = tuple(
                    (ax.axhline(event.ydata, color='k'),
                     ax.axvline(event.xdata, color='k'))
                    for ax in self._crosshair_axes)
            else:
                for hline, vline in self._crosshair_lines:
                    hline.set_ydata([event.ydata, event.ydata])
                    vline.set_xdata([event.xdata, event.xdata])
            redraw_axes.update(self._crosshair_axes)
        # update status bar
        self._frame.SetStatusText(self._on_motion_status_text(event))
        # redraw
        self.canvas.redraw(redraw_axes)

    @staticmethod
    def _on_motion_status_text(event):
        ax = event.inaxes
        if ax:
            return ('x = %s, y = %s' % (
                ax.xaxis.get_major_formatter().format_data_short(event.xdata),
                ax.yaxis.get_major_formatter().format_data_short(event.ydata)))
        return ''

    def _on_motion_sub(self, event):
        "Subclass action on mouse motion, return set of axes to redraw"
        return set()

    def _on_resize(self, event):
        if self._layout.tight:
            self._tight()

    def _register_key(self, key, press=None, release=None):
        if press:
            if key in self.__callback_key_press:
                raise RuntimeError("Attempting to assign key press %r twice" %
                                   key)
            self.__callback_key_press[key] = press
        if release:
            if key in self.__callback_key_release:
                raise RuntimeError("Attempting to assign key release %r twice" %
                                   key)
            self.__callback_key_release[key] = release

    def _remove_crosshairs(self, draw=False):
        if self._crosshair_lines is not None:
            for hline, vline in self._crosshair_lines:
                hline.remove()
                vline.remove()
            self._crosshair_lines = None
            if draw:
                self.canvas.redraw(self._crosshair_axes)

    def _fill_toolbar(self, tb):
        """
        Add toolbar tools

        Subclasses should add their toolbar items in this function which
        is called by ``CanvasFrame.FillToolBar()``.
        """
        pass

    def close(self):
        "Close the figure."
        self._frame.Close()

    def _get_axes(self, axes):
        "Iterate over axes corresponding to ``axes`` parameter"
        if axes is None:
            return self.axes
        elif isinstance(axes, int):
            return self.axes[axes],
        else:
            return (self.axes[i] for i in axes)

    def _configure_axis(
            self,
            axis: str,  # 'x' | 'y'
            ticklabels: Union[str, int, Sequence[int]],  # where to show tick-labels
            params: Iterable,  # (formatter, locator, label) for each Axes
            axes: List[matplotlib.axes.Axes] = None,  # axes which to format
    ):
        if axes is None:
            axes = self.axes

        # find axes with tick-labels
        nax = len(axes)
        if isinstance(ticklabels, bool):
            show_ticklabels = [ticklabels] * nax
        elif isinstance(ticklabels, str):
            if ticklabels == 'bottom':
                if all(isinstance(ax, matplotlib.axes.SubplotBase) for ax in axes):
                    subplotspecs = [ax.get_subplotspec() for ax in axes]
                    bottom = max([spec.rowspan.stop for spec in subplotspecs])
                    show_ticklabels = [spec.rowspan.stop == bottom for spec in subplotspecs]
                else:
                    first = len(axes) - min(self._layout.columns, nax)
                    show_ticklabels = [i >= first for i in range(len(axes))]
            elif ticklabels == 'left':
                if all(isinstance(ax, matplotlib.axes.SubplotBase) for ax in axes):
                    subplotspecs = [ax.get_subplotspec() for ax in axes]
                    left = min([spec.colspan.start for spec in subplotspecs])
                    show_ticklabels = [spec.colspan.start == left for spec in subplotspecs]
                else:
                    ncol = self._layout.columns or nax
                    show_ticklabels = [i % ncol == 0 for i in range(len(axes))]
            elif ticklabels == 'all':
                show_ticklabels = [True] * nax
            elif ticklabels == 'none':
                show_ticklabels = [False] * nax
            else:
                raise ValueError(f"{ticklabels=}")
        else:
            show_ticklabels = [False] * nax
            if isinstance(ticklabels, int):
                show_ticklabels[ticklabels] = True
            else:
                for i in ticklabels:
                    show_ticklabels[i] = True

        # parameter for hiding tick-labels
        if axis == 'y':
            tick_params = {'labelleft': False}
        else:
            tick_params = {'labelbottom': False}

        # format ticks
        labels = []
        for ax, (formatter, locator, label_), show_ticklabels_ in zip(axes, params, show_ticklabels):
            axis_ = ax.yaxis if axis == 'y' else ax.xaxis
            if locator:
                axis_.set_major_locator(locator)
            if formatter:
                axis_.set_major_formatter(formatter)
            if not show_ticklabels_:
                ax.tick_params(**tick_params)
            labels.append(label_)

        # set labels
        if any(labels):
            if len(set(labels)) == 1:
                # default positioning
                if axis == 'y':
                    self.set_ylabel(labels[0])
                else:
                    self.set_xlabel(labels[0])
            else:
                for ax, label in zip(axes, labels):
                    if label:
                        if axis == 'y':
                            ax.set_ylabel(label)
                        else:
                            ax.set_xlabel(label)

    def _configure_axis_data(
            self,
            axis: str,  # 'x' | 'y'
            data: Union[NDVar, Var],  # data for default label
            label: Union[bool, str],  # override label
            ticklabels: Union[str, int, Sequence[int]] = True,  # where to show tick-labels
            axes: List[matplotlib.axes.Axes] = None,  # axes which to format
            labels: Dict[str, str] = None,  # user-provided labels
    ):
        "Configure an axis based on data"
        if label is True and labels is not None:
            label = labels.get(data.name, label)
        scale = AxisScale(data, label)
        formatters = repeat(scale.formatter)
        # One label for each subplot
        if not isinstance(scale.label, str) and isinstance(scale.label, Iterable):
            labels = chain(scale.label, repeat(None))
        else:
            labels = repeat(scale.label)
        params = zip(formatters, repeat(None), labels)
        self._configure_axis(axis, ticklabels, params, axes)

    def _configure_axis_dim(
            self,
            axis: str,  # 'x' | 'y'
            dim: Union[str, Dimension],  # The dimension assigned to the axis
            label: Union[bool, str],  # axis labale
            ticklabels: Union[str, int, Sequence[int]],  # where to show tick-labels
            axes: List[matplotlib.axes.Axes] = None,  # axes which to format
            scalar: bool = True,
            data: List = None,
    ):
        "Configure an axis based on a dimension"
        # Dimension objects
        if isinstance(dim, str):
            dims = [layers[0].get_dim(dim) for layers in data]
        else:
            dims = repeat(dim)
        params = (dim._axis_format(scalar, label) for dim in dims)
        self._configure_axis(axis, ticklabels, params, axes)

    def draw(self):
        "(Re-)draw the figure (after making manual changes)."
        if self._frame is None:
            return
        t0 = time.time()
        self._frame.canvas.draw()
        self._last_draw_time = time.time() - t0

    def draw_crosshairs(self, enable=True):
        """Draw crosshairs under the cursor

        Parameters
        ----------
        enable : bool
            Enable drawing crosshairs (default True, set to False to disable).
        """
        self._draw_crosshairs = enable
        if not enable:
            self._remove_crosshairs(True)

    def draw_outline(self, color='k', **kwargs):
        """Draw the outline of the figure

        Mainly for fine-tuning the figure layout in Jupyter, which crops
        the display area based on figure elements rather than actual figure size.
        """
        kwargs.setdefault('fc', 'none')
        artist = matplotlib.patches.Rectangle((0, 0), 1, 1, ec=color, **kwargs)
        self.figure.add_artist(artist)

    def save(self, *args, **kwargs):
        """Short-cut for Matplotlib's :meth:`~matplotlib.figure.Figure.savefig()`

        Notes
        -----
        Important parameters (for details see :meth:`~matplotlib.figure.Figure.savefig()`):

         - ``facecolor="none"`` to save the figure with transparent background.
         - ``bbox_inches='tight'`` to ignore the figure boundaries and instead save
           an area including all plotted elements.
        """
        self.figure.savefig(*args, **kwargs)

    def add_hline(self, y, axes=None, *args, **kwargs):
        """Draw a horizontal line on one or more axes

        Parameters
        ----------
        y : scalar
            Level at which to draw the line.
        axes : int | list of int
            Which axes to mark (default is all axes).
        ...
            :meth:`matplotlib.axes.Axes.axhline` parameters.


        Notes
        -----
        See Matplotlib's :meth:`matplotlib.axes.Axes.axhline` for more
        arguments.
        """
        for ax in self._get_axes(axes):
            ax.axhline(y, *args, **kwargs)
        self.draw()

    def add_hspan(self, bottom, top, axes=None, *args, **kwargs):
        """Draw a horizontal bar on one or more axes

        Parameters
        ----------
        bottom : scalar
            Bottom end of the horizontal bar.
        top : scalar
            Top end of the horizontal bar.
        axes : int | list of int
            Which axes to mark (default is all axes).
        ...
            :meth:`matplotlib.axes.Axes.axvspan` parameters.


        Notes
        -----
        See Matplotlib's :meth:`matplotlib.axes.Axes.axhspan` for more
        arguments.
        """
        for ax in self._get_axes(axes):
            ax.axhspan(bottom, top, *args, **kwargs)
        self.draw()

    def add_vline(self, x, axes=None, *args, **kwargs):
        """Draw a vertical line on one or more axes

        Parameters
        ----------
        x : scalar
            Value at which to place the vertical line.
        axes : int | list of int
            Which axes to mark (default is all axes).
        ...
            :meth:`matplotlib.axes.Axes.axvspan` parameters.


        Notes
        -----
        See Matplotlib's :meth:`matplotlib.axes.Axes.axvline` for more
        arguments.
        """
        for ax in self._get_axes(axes):
            ax.axvline(x, *args, **kwargs)
        self.draw()

    def add_vspan(self, xmin, xmax, axes=None, *args, **kwargs):
        """Draw a vertical bar on one or more axes

        Parameters
        ----------
        xmin : scalar
            Start value on the x-axis.
        xmax : scalar
            Last value on the x-axis.
        axes : int | list of int
            Which axes to mark (default is all axes).
        ...
            :meth:`matplotlib.axes.Axes.axvspan` parameters.


        Notes
        -----
        See Matplotlib's :meth:`matplotlib.axes.Axes.axvspan` for more
        arguments.
        """
        for ax in self._get_axes(axes):
            ax.axvspan(xmin, xmax, *args, **kwargs)
        self.draw()

    def set_name(self, name):
        """Set the figure window title"""
        plot_name = self.__class__.__name__
        self._frame.SetTitle(f'{plot_name}: {name}' if name else plot_name)

    def set_xtick_rotation(self, rotation):
        """Rotate every x-axis tick-label by an angle (counterclockwise, in degrees)

        Parameters
        ----------
        rotation : scalar
            Counterclockwise rotation angle, in degrees.
        """
        for ax in self.axes:
            for t in ax.get_xticklabels():
                t.set_rotation(rotation)
        self.draw()

    def set_xlabel(self, label: str, ax: int = None):
        """Set the label for the x-axis

        Parameters
        ----------
        label
            X-axis label.
        ax
            Axis on which to set the label (default is usually the last axis).
        """
        if ax is None:
            ax = self._default_xlabel_ax
        self.axes[ax].set_xlabel(label)

    def set_ylabel(self, label: str, ax: int = None):
        """Set the label for the y-axis

        Parameters
        ----------
        label
            Y-axis label.
        ax
            Axis on which to set the label (default is usually the first axis).
        """
        if ax is None:
            ax = self._default_ylabel_ax
        self.axes[ax].set_ylabel(label)


def format_axes(
        ax: mpl.axes.Axes,
        frame: Union[bool, str],
        yaxis: bool,
):
    if frame == 't':
        ax.tick_params(direction='inout', bottom=False, top=True, left=False, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.spines['right'].set_position('zero')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_position('zero')
        ax.spines['bottom'].set_visible(False)
    elif frame == 'none':
        for spine in ax.spines.values():
            spine.set_visible(False)
    elif frame == 'off':
        ax.axis('off')
    elif not frame:
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['top'].set_visible(False)

    if not yaxis:
        ax.yaxis.set_ticks(())
        ax.spines['left'].set_visible(False)


class BaseLayout:
    def __init__(
            self,
            h: float,
            w: float,
            dpi: float,
            tight: bool,
            show: bool,
            run: bool,
            title: FMTextArg = None,
            autoscale: bool = False,
            name: str = None,
            right_of: Union[EelFigure, int] = None,
            below: Union[EelFigure, int] = None,
            axes: Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]] = None,
    ):
        self.h = h
        self.w = w
        self.dpi = dpi or mpl.rcParams['figure.dpi']
        self.tight = tight
        self.show = show
        self.run = run
        self.autoscale = autoscale
        self.title = asfmtext_or_none(title)
        self.name = asfmtext_or_none(name) or title
        if isinstance(axes, matplotlib.axes.Axes):
            axes = [axes]
        elif axes is not None:
            axes = list(axes)
        self.user_axes = axes

        x = y = None
        if hasattr(right_of, '_frame'):
            rect = right_of._frame.GetRect()
            x = rect.GetRight() + 1
            if below is None:
                y = rect.GetTop()
        elif isinstance(right_of, int):
            x = right_of
        elif right_of is not None:
            raise TypeError(f"{right_of=}")

        if hasattr(below, '_frame'):
            rect = below._frame.GetRect()
            y = rect.GetBottom() + 1
            if x is None:
                x = rect.GetLeft()
        elif isinstance(below, int):
            y = below
        elif below is not None:
            raise TypeError(f"{below=}")

        if x is None and y is None:
            self.pos = None
        else:
            if x is None:
                x = -1
            if y is None:
                y = -1
            self.pos = (x, y)

    def fig_kwa(self):
        out = {'figsize': (self.w, self.h), 'dpi': self.dpi}
        if CONFIG['figure_background'] is not False:
            out['facecolor'] = CONFIG['figure_background']
        return out

    def make_axes(self, figure):
        if self.user_axes:
            axes = self.user_axes
        else:
            axes = self._make_axes(figure)
        self._configure_axes(axes)
        return axes

    def _make_axes(self, figure):
        raise NotImplementedError

    def _configure_axes(self, axes):
        raise NotImplementedError


def resolve_plot_rect(w, h, dpi):
    # infer figure dimensions from screen size
    w_applies = w is not None and w <= 0
    h_applies = h is not None and h <= 0
    if w_applies or h_applies:
        from .._wxgui import wx, get_app

        get_app()
        effective_dpi = dpi or mpl.rcParams['figure.dpi']
        display_w, display_h = wx.GetDisplaySize()
        if h_applies:
            effective_display_h = display_h - 50
            h = effective_display_h / effective_dpi + h
        if w_applies:
            w = display_w / effective_dpi + w
    return w, h


class LayoutDim:
    "Helper function to determine figure spacing"
    _properties = ('total', 'ax', 'first', 'last', 'space')
    _equations = dict(
        total='first + n_ax * ax + (n_ax - 1) * space + last',
        ax='(total - first - last - (n_ax - 1) * space) / n_ax',
        first='total - n_ax * ax  - (n_ax - 1) * space - last',
        last='total - first - n_ax * ax - (n_ax - 1) * space',
        space='(total - first - n_ax * ax - last) / (n_ax - 1)',
    )

    def __init__(self, n_ax, total, ax, first, space, last, ax_default, first_default, space_default, last_default):
        if space is None and n_ax == 1:
            space = 0.
        values = {'total': total, 'first': first, 'space': space, 'last': last, 'ax': ax, 'n_ax': n_ax}
        defaults = {'first': first_default, 'space': space_default, 'last': last_default, 'ax': ax_default}
        for i, p in enumerate(self._properties):
            if values[p] is None:
                for p2 in self._properties[i + 1:]:
                    if values[p2] is None:
                        values[p2] = defaults[p2]
                values[p] = eval(self._equations[p], values)
                break

        self.total = values['total']
        self.ax = values['ax']
        self.first = values['first']
        self.space = values['space']
        self.last = values['last']


class Layout(BaseLayout):
    """Layout for figures with several axes of the same size"""
    _default_margins = {'left': 0.4, 'bottom': 0.5, 'right': 0.05, 'top': 0.05, 'wspace': 0.1, 'hspace': 0.1}

    def __init__(
            self,
            nax: Union[int, List[bool]],
            ax_aspect: float,  # width / height
            axh_default: float,
            tight: bool = True,
            title: str = None,
            h: float = None,
            w: float = None,
            axh: float = None,
            axw: float = None,
            rows: int = None,
            columns: int = None,
            dpi: float = None,
            margins: Dict[str, float] = None,
            show: bool = True,
            run: bool = None,
            frame: Union[bool, str] = True,
            yaxis=True,
            share_axes: bool = False,
            **kwargs):
        """Create a grid of axes based on variable parameters.

        Parameters
        ----------
        nax
            Number of axes required. If provided as a list, axes are only added
            for items where ``item`` is True.
        ax_aspect
            Width / height aspect of the axes.
        axh_default
            The default axes height if it can not be determined from the other
            parameters.
        tight
            Rescale axes so that the space in the figure is used optimally
            (default True).
        title : str
            Figure title.
        h
            Height of the figure.
        w
            Width of the figure.
        axh
            Height of the axes.
        axw
            Width of the axes.
        rows
            Set a limit to the number of rows (default is no limit).
        columns
            Set a limit to the number of columns (defaut is no limit). If
            neither nrow or ncol is specified, a square layout is preferred.
        dpi
            DPI for the figure (default is to use matplotlib rc parameters).
        margins
            Absolute subplot parameters (in inches). Implies ``tight=False``.
            If ``margins`` is specified, ``axw`` and ``axh`` are interpreted
            exclusive of the margins, i.e., ``axh=2, margins={'top': .5}`` for
            a plot with one axes will result in a total height of 2.5.
        show
            Show the figure in the GUI (default True). Use False for creating
            figures and saving them without displaying them on the screen.
        run
            Run the Eelbrain GUI app (default is True for interactive plotting and
            False in scripts).
        frame : bool | 't' | 'none'
            Draw frame around axes:
            - True: all four spines
            - False: only spines with ticks
            - 't': spines at x=0 and y=0
            - 'none': no spines at all
        """
        if h and axh:
            if h < axh:
                raise ValueError("h < axh")
        if w and axw:
            if w < axw:
                raise ValueError("w < axw")
        w, h = resolve_plot_rect(w, h, dpi)

        self.h_fixed = h
        self.w_fixed = w if w is not None else axw
        self._margins_arg = margins

        if margins is True:
            use_margins = True
            tight = False
            margins = self._default_margins.copy()
        elif margins is not None:
            use_margins = True
            tight = False
            margins = dict(margins)
            invalid = set(margins).difference(self._default_margins)
            if invalid:
                raise ValueError(f"{margins=}: Unknown keys {invalid}")
        else:
            margins = {k: 0 for k in self._default_margins}
            use_margins = False

        h_is_implicit = h is None
        w_is_implicit = w is None

        if nax is None:
            axes = None
        elif isinstance(nax, int):
            axes = list(range(nax))
        elif isinstance(nax, (list, tuple)):
            axes = [i for i, ax in enumerate(nax) if ax]
            nax = len(nax)
        else:
            raise TypeError("nax=%r" % (nax,))

        trim = None
        if not nax:
            if w is None:
                if h is None:
                    h = axh_default
                w = ax_aspect * h
            elif h is None:
                h = w / ax_aspect
        elif nax == 1:
            columns = columns or 1
            rows = rows or 1
        elif rows is None and columns is None:
            if w and axw:
                trim = 'row'
                columns = math.floor(w / axw)
            elif h and axh:
                trim = 'col'
                rows = math.floor(h / axh)
            elif w:
                trim = 'row'
                if axh:
                    columns = round(w / (axh * ax_aspect))
                else:
                    columns = round(w / (axh_default * ax_aspect))
                columns = max(1, min(nax, columns))
            elif h:
                trim = 'col'
                if axw:
                    rows = round(h / (axw / ax_aspect))
                else:
                    rows = round(h / axh_default)
                rows = max(1, min(nax, rows))
            elif axh or axw:
                trim = 'row'
                if not axh:
                    axh = axw / ax_aspect
                rows = min(nax, math.floor(defaults['maxh'] / axh))
            else:
                trim = 'row'
                # default: minimum number of columns (max number of rows)
                hspace = margins.get('hspace', 0)
                maxh = defaults['maxh'] - margins.get('top', 0) - margins.get('bottom', 0) + hspace
                axh_with_space = axh_default + hspace
                rows = min(nax, math.floor(maxh / axh_with_space))
                columns = math.ceil(nax / rows)
                # test width
                wspace = margins.get('wspace', 0)
                maxw = defaults['maxw'] - margins.get('left', 0) - margins.get('right', 0) + wspace
                axw_with_space = axh_default * ax_aspect + wspace
                if columns * axw_with_space > maxw:
                    # nrow/ncol proportional to (maxh / axh) / (maxw / axw)
                    ratio = (maxh / axh_with_space) / (maxw / axw_with_space)
                    # nax = ncol * (ncol * ratio)
                    # ncol = sqrt(nax / ratio)
                    columns = math.floor(math.sqrt(nax / ratio))
                    rows = math.ceil(nax / columns)
                    axh = (maxh - rows * hspace) / rows
                    axw = axh * ax_aspect

        if nax:
            if rows is None:
                rows = math.ceil(nax / columns)
            elif columns is None:
                columns = math.ceil(nax / rows)

            if trim == 'row':
                if (rows * columns) - nax >= columns:
                    rows -= 1
            elif trim == 'col':
                if (rows * columns) - nax >= rows:
                    rows -= 1

            if axw:
                axh_default = axw / ax_aspect
            elif w:
                axh_default = w / columns / ax_aspect
            h_dim = LayoutDim(rows, h, axh, margins.get('top'), margins.get('hspace'), margins.get('bottom'), axh_default, self._default_margins['top'], self._default_margins['hspace'], self._default_margins['bottom'])
            w_dim = LayoutDim(columns, w, axw, margins.get('left'), margins.get('wspace'), margins.get('right'), h_dim.ax * ax_aspect, self._default_margins['left'], self._default_margins['wspace'], self._default_margins['right'])
            h = h_dim.total
            w = w_dim.total
            axh = h_dim.ax
            axw = w_dim.ax
            margins = {
                'top': h_dim.first, 'bottom': h_dim.last, 'hspace': h_dim.space,
                'left': w_dim.first, 'right': w_dim.last, 'wspace': w_dim.space}
            h_is_implicit = w_is_implicit = False

        if h_is_implicit:
            hspace = 0 if rows is None else margins['hspace'] * (rows - 1)
            h += margins['bottom'] + hspace + margins['top']
        if w_is_implicit:
            wspace = 0 if columns is None else margins['wspace'] * (columns - 1)
            w += margins['left'] + wspace + margins['right']

        BaseLayout.__init__(self, h, w, dpi, tight, show, run, title, **kwargs)
        self.nax = nax
        self.axes = axes
        self.axh = axh
        self.axw = axw
        self.rows = rows
        self.columns = columns
        self.frame = frame
        self.yaxis = yaxis
        self.share_axes = share_axes
        self.margins = margins if use_margins else None

    def __repr__(self):
        kwargs = self.fig_kwa()
        if 'subplotpars' in kwargs:
            pars = kwargs['subplotpars']
            attrs = ((k, getattr(pars, k)) for k in ('left', 'right', 'bottom', 'top', 'wspace', 'hspace'))
            desc = ', '.join(f'{k}={v}' for k, v in attrs if v is not None)
            kwargs['subplotpars'] = f"<{desc}>"
        args = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        return f'<Layout: {args}>'

    def fig_kwa(self):
        out = BaseLayout.fig_kwa(self)

        if self.margins:  # absolute subplot parameters
            out['subplotpars'] = SubplotParams(
                self.margins['left'] / self.w,
                self.margins['bottom'] / self.h,
                1 - self.margins['right'] / self.w,
                1 - self.margins['top'] / self.h,
                # space expressed as a fraction of the average axis height/width
                self.margins['wspace'] / self.axw,
                self.margins['hspace'] / self.axh)

        return out

    def _make_axes(self, figure):
        if not self.nax:
            return []
        axes = []
        kwargs = {}
        for i in self.axes:
            ax = figure.add_subplot(self.rows, self.columns, i + 1, autoscale_on=self.autoscale, **kwargs)
            axes.append(ax)
            if self.share_axes:
                kwargs.update(sharex=ax, sharey=ax)
        return axes

    def _configure_axes(self, axes):
        for ax in axes:
            format_axes(ax, self.frame, self.yaxis)
        return axes


class ImLayout(Layout):
    """Layout subclass for axes without space

    Make sure to specify the ``margins`` parameter for absolute spacing
    """

    def __init__(
            self,
            nax: Union[int, List[bool]],
            ax_aspect: float,  # width / height
            axh_default: float,
            margins: dict = None,
            default_margins: dict = None,
            title: str = None,
            axtitle: Union[bool, Sequence[str]] = False,  # for default spacing
            **kwargs,
    ):
        if axtitle is True:
            has_axtitle = (len(nax) if isinstance(nax, list) else nax) > 1
        else:
            has_axtitle = True if isinstance(axtitle, np.ndarray) else bool(axtitle)
        title_space = 1.5 * mpl_font_size('figure.titlesize') if title else 0
        axtitle_space = 1.5 * mpl_font_size('axes.titlesize') if has_axtitle else 0
        margins_ = {
            'left': 0, 'wspace': 0, 'right': 0,
            'top': axtitle_space + title_space,
            'hspace': axtitle_space,
            'bottom': 0,
        }
        if default_margins:
            margins_.update(default_margins)
        if margins:
            margins_.update(margins)
        Layout.__init__(self, nax, ax_aspect, axh_default, title=title, margins=margins_, **kwargs)

    def _make_axes(self, figure):
        axes = []
        for i in self.axes:
            ax = figure.add_subplot(self.rows, self.columns, i + 1, autoscale_on=self.autoscale)
            axes.append(ax)
        return axes

    def _configure_axes(self, axes):
        for ax in axes:
            ax.axis('off')


class VariableAspectLayout(BaseLayout):
    """Layout with a fixed number of columns that differ in spacing

    Axes are originally created to fill the whole rectangle allotted to them.
    Developed for TopoButterfly plot: one variable aspect butterfly plot, and
    one square topomap plot.

    Parameters
    ----------
    rows
        Number of rows.
    axh_default
        Default row height.
    w_default
        Default figure width.
    aspect
        Axes aspect ratio (w/h) for each column; None for axes with flexible
        width.
    ax_kwargs
        Parameters for :meth:`figure.add_axes` for each column.
    ax_frames
        ``frame`` parameter for :func:`format_axes` for each column.
    row_titles
        One title per row.
    """
    def __init__(
            self,
            rows: int,
            axh_default: float,
            w_default: float,
            aspect: Sequence[Optional[float]] = (None, 1),
            ax_kwargs: Sequence[dict] = None,
            ax_frames: Sequence[bool] = None,
            row_titles: Sequence[Optional[str]] = None,
            title: FMTextArg = None,
            h: float = None,
            w: float = None,
            axh: float = None,
            dpi: float = None,
            show: bool = True,
            run: bool = None,
            **kwargs,
    ):
        w, h = resolve_plot_rect(w, h, dpi)
        self.w_fixed = w

        if axh and h:
            raise ValueError("h and axh can not be specified both at the same time")
        elif h:
            axh = h / rows
        elif axh:
            h = rows * axh
        else:
            axh = axh_default
            h = rows * axh

        if w is None:
            w = w_default

        if ax_kwargs is None:
            ax_kwargs = [{}] * len(aspect)
        if ax_frames is None:
            ax_frames = [True] * len(aspect)

        BaseLayout.__init__(self, h, w, dpi, False, show, run, title, **kwargs)
        self.nax = rows * len(aspect)
        self.axh = axh
        self.rows = rows
        self.columns = len(aspect)
        self.share_axes = False
        self.row_titles = row_titles
        self.aspect = aspect
        self.n_flexible = self.aspect.count(None)
        self.ax_kwargs = ax_kwargs
        self.ax_frames = ax_frames

        # Compute axes outlines for given height and width
        h = self.h
        w = self.w
        text_buffer = 20 * POINT

        # buffers for legends
        left_buffer = text_buffer * (3 + (self.row_titles is not None))
        bottom_buffer = text_buffer * 2
        top_buffer = text_buffer * (1 + 2 * bool(self.title))

        # rectangle base in inches
        axh = (h - bottom_buffer - top_buffer) / self.rows
        axws = [None if a is None else a * axh for a in self.aspect]
        fixed = sum(axw for axw in axws if axw is not None)
        w_free = (w - fixed - left_buffer) / self.n_flexible
        widths = [w_free if axw is None else axw for axw in axws]
        lefts = (sum(widths[:i]) + left_buffer for i in range(len(widths)))
        bottoms = (i * axh + bottom_buffer for i in range(self.rows - 1, -1, -1))

        # convert to figure coords
        height = axh / h
        lefts_ = [l / w for l in lefts]
        widths_ = [w_ / w for w_ in widths]
        bottoms_ = [b / h for b in bottoms]

        # rectangles:  (left, bottom, width, height)
        self._ax_rects = [[(l, bottom, w, height) for l, w in zip(lefts_, widths_)] for bottom in bottoms_]

    def _make_axes(self, figure):
        axes = []
        for row, row_rects in enumerate(self._ax_rects):
            for rect, kwa, frame in zip(row_rects, self.ax_kwargs, self.ax_frames):
                ax = figure.add_axes(rect, autoscale_on=self.autoscale, **kwa)
                axes.append(ax)

            if self.row_titles and self.row_titles[row]:
                bottom, height = rect[1], rect[3]
                figure.text(0, bottom + height / 2, self.row_titles[row], ha='left', va='center', rotation='vertical')
        return axes

    def _configure_axes(self, axes):
        for ax, frame in zip(axes, cycle(self.ax_frames)):
            format_axes(ax, frame, True)
        # id axes for callbacks
        for i, ax in enumerate(axes):
            ax.id = i


def subplots(
        rows: int = 1,
        columns: int = 1,
        axh: float = None,
        axw: float = None,
        h: float = None,
        w: float = None,
        left: float = None,
        right: float = None,
        wspace: float = None,
        width_ratios: Sequence[float] = None,
        bottom: float = None,
        top: float = None,
        hspace: float = None,
        height_ratios: Sequence[float] = None,
        **kwargs,
) -> (matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]):
    """Specify :func:`matplotlib.pyplot.subplots` parameters in inches

    Parameters
    ----------
    rows
        Number of subplot rows.
    columns
        Number of subplot columns.
    axh
        Height of each axes.
    axw
        Width of each axes.
    h
        Figure height.
    w
        Figure width.
    left
        Margin to the left of the axes.
    right
        Margin to the right of the axes.
    wspace
        Width of the margin between axes.
    width_ratios
        The relative widths of the columns (see :class:`matplotlib.gridspec.GridSpec`).
    bottom
        Margin below the axes.
    top
        Margin above the axes.
    hspace
        Height of the margin between axes.
    height_ratios
        The relative heights of the rows (see :class:`matplotlib.gridspec.GridSpec`).
    **
        Other parameters for :func:`matplotlib.pyplot.subplots`.
    """
    from matplotlib import pyplot

    margins = {'left': left, 'bottom': bottom, 'right': right, 'top': top, 'wspace': wspace, 'hspace': hspace}
    layout = Layout(rows * columns, 1, 2, False, None, h, w, axh, axw, rows, columns, None, margins)
    gridspec_kw = {
        'left': layout.margins['left'] / layout.w,
        'right': 1 - layout.margins['right'] / layout.w,
        'wspace': layout.margins['wspace'] / layout.axw,
        'width_ratios': width_ratios,
        'bottom': layout.margins['bottom'] / layout.h,
        'top': 1 - layout.margins['top'] / layout.h,
        'hspace': layout.margins['hspace'] / layout.axh,
        'height_ratios': height_ratios,
    }
    return pyplot.subplots(layout.rows, layout.columns, figsize=(layout.w, layout.h), gridspec_kw=gridspec_kw, **kwargs)


class ColorBarMixin:
    """Colorbar toolbar button mixin

    Parameters
    ----------
    param_func : func
        Function that returns color-bar parameters.
    """
    def __init__(
            self,
            param_func: Callable = None,  # function to get cmap, vmin, vmax
            data: Union[NDVar, Var, Real, str, 'PlotData'] = None,  # to infer unit
            mappable: Any = None,  # matplotlib mappable object
            default_kwargs: dict = None,  # default parameters for plotting colorbar
    ):
        self.__get_params = param_func
        self.__mappable = mappable
        self.__default_kwargs = default_kwargs or {}
        if data is None:
            self.__scale = AxisScale(1)
        else:
            self.__scale = AxisScale(data)

    def _fill_toolbar(self, tb):
        from .._wxgui import wx, ID, Icon

        tb.AddTool(ID.PLOT_COLORBAR, "Plot Colorbar", Icon("plot/colorbar"))
        tb.Bind(wx.EVT_TOOL, self.__OnPlotColorBar, id=ID.PLOT_COLORBAR)

    def __OnPlotColorBar(self, event):  # noqa
        return self.plot_colorbar()

    def plot_colorbar(
            self,
            label: Union[bool, str] = True,
            label_position: Literal['left', 'right', 'top', 'bottom'] = None,
            label_rotation: float = None,
            **kwargs,
    ):
        """Plot a colorbar corresponding to the displayed data

        Parameters
        ----------
        label
            Label for the x-axis (default is based on the data).
        label_position
            Position of the axis label. Valid values depend on orientation.
        label_rotation
            Angle of the label in degrees (For horizontal colorbars, the default is
            0; for vertical colorbars, the default is 0 for labels of 3 characters
            and shorter, and 90 for longer labels).
        ...
            More parameters for :class:`plot.ColorBar`.

        Returns
        -------
        colorbar : plot.ColorBar
            ColorBar plot object.
        """
        # cf. matplorlib.colorbar.Colorbar transforming mappable to color-map
        from . import ColorBar
        if self.__mappable is not None:
            cmap = self.__mappable.cmap
            vmin = self.__mappable.norm
            vmax = None
        elif self.__get_params is not None:
            cmap, vmin, vmax = self.__get_params()
        else:
            raise RuntimeError(f"No colormap on {self}")
        kwargs = {'unit': self.__scale, **self.__default_kwargs, **kwargs}
        return ColorBar(cmap, vmin, vmax, label, label_position, label_rotation, **kwargs)


class ColorMapMixin(ColorBarMixin):
    """takes care of color-map and includes color-bar"""
    _can_set_vlim = True

    def __init__(self, epochs, cmap, vmax, vmin, contours, plots):
        ColorBarMixin.__init__(self, self.__get_cmap_params, epochs[0][0])
        self.__plots = plots  # can be empty list at __init__
        self._cmaps = find_fig_cmaps(epochs, cmap)
        self._vlims = find_fig_vlims(epochs, vmax, vmin, self._cmaps)
        self._contours = find_fig_contours(epochs, self._vlims, contours)
        self._first_meas = epochs[0][0].info.get('meas')

    def __get_cmap_params(self):
        return (self._cmaps[self._first_meas],) + self._vlims[self._first_meas]

    def add_contour(self, level: float, color: Any = 'k', meas: str = None):
        """Add a contour line

        Parameters
        ----------
        level : scalar
            The value at which to draw the contour.
        color : matplotlib color
            The color of the contour line.
        meas : str
            The measurement for which to add a contour line (default is the
            measurement plotted first).
        """
        if meas is None:
            meas = self._first_meas

        for p in self.__plots:
            p.add_contour(meas, level, color)
        self.draw()

    def set_cmap(self, cmap, meas=None):
        """Change the colormap in the array plots

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        if meas is None:
            meas = self._first_meas

        for p in self.__plots:
            p.set_cmap(cmap, meas)
        self._cmaps[meas] = cmap
        if isinstance(cmap, LocatedColormap):
            self.set_vlim(cmap.vmin, cmap.vmax, meas)
        else:
            self.draw()

    def set_vlim(self, v=None, vmax=None, meas=None):
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
        meas : str (optional)
            Measurement type to apply (default is the first one found).
        """
        if meas is None:
            meas = self._first_meas
        elif meas not in self._cmaps:
            raise ValueError("meas=%r" % (meas,))

        if vmax is None:
            vmin, vmax = fix_vlim_for_cmap(None, abs(v), self._cmaps[meas])
        else:
            vmin = v

        for p in self.__plots:
            p.set_vlim(vmin, vmax, meas)
        self._vlims[meas] = vmin, vmax

        if self._can_set_ylim:
            self.set_ylim(vmin, vmax)
        else:
            self.draw()

    def get_vlim(self, meas=None):
        "Retrieve colormap value limits as ``(vmin, vmax)`` tuple"
        if meas is None:
            meas = self._first_meas
        return self._vlims[meas]


class LegendMixin:
    __choices = ('invisible', 'separate window', 'draggable', 'upper right',
                 'upper left', 'lower left', 'lower right', 'right',
                 'center left', 'center right', 'lower center', 'upper center',
                 'center')
    __args = (False, 'fig', 'draggable', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    _has_frame = None

    def __init__(
            self,
            loc: LegendArg,
            handles: Dict[CellArg, Any],
            labels: Dict[CellArg, str] = None,
            alt_sort: Iterable[CellArg] = None,
    ):
        """Legend toolbar menu mixin

        Parameters
        ----------
        loc
            Matplotlib figure legend location argument or 'fig' to plot the
            legend in a separate figure.
        handles : dict
            {cell: handle} dictionary.
        labels : dict
            Dictionary with labels for cells.
        alt_sort
            Alternative legend order (if labels is not specified).
        """
        # sort legend handles based on colors dict
        if alt_sort and all(key in alt_sort for key in handles):
            keys = [key for key in alt_sort if key in handles]
            handles = {key: handles[key] for key in keys}
        # whether to plot default legend
        if loc is not None:
            initial_loc = loc
        elif len(handles) > 1:
            initial_loc = 'upper right'
        else:
            initial_loc = False
        self.__handles = handles
        self.legend = None
        self.__labels = None
        self.__set_labels(labels)
        self.plot_legend(initial_loc)

    def __set_labels(self, labels: Dict[CellArg, str] = None):
        if labels is not None:
            self.__labels = {key: asfmtext(label) for key, label in labels.items()}

    def _fill_toolbar(self, tb):
        from .._wxgui import wx

        choices = [name.title() for name in self.__choices]
        self.__ctrl = wx.Choice(tb, choices=choices, name='Legend')
        tb.AddControl(self.__ctrl, "Legend")
        self.__ctrl.Bind(wx.EVT_CHOICE, self.__OnChoice, source=self.__ctrl)

    def __OnChoice(self, event):  # noqa
        self.__plot(self.__args[event.GetSelection()])

    def plot_legend(
            self,
            loc: LegendArg = 'fig',
            labels: Dict[CellArg, str] = None,
            **kwargs,
    ):
        """Plot the legend (or remove it from the figure).

        Parameters
        ----------
        loc
            Where to plot the legend (see Notes; default 'fig').
        labels
            Dictionary with alternate labels for all cells.
        ... :
            Additional legend parameters (for :class:`eelbrain.plot.Legend` if
            ``loc=='fig'``, otherwise for :math:`matplotlib.figure.Figure.legend`).

        Returns
        -------
        legend_figure : None | legend
            If loc=='fig' the Figure, otherwise None.

        Notes
        -----
        legend content can be modified through the figure's
        ``legend_handles`` and ``legend_labels`` attributes.

        Possible values for the ``loc`` argument:

        ``False``:
            Make the current legend invisible
        ``'fig'``:
            Plot the legend in a new figure
        ``'draggable'``:
            The legend can be dragged to the desired position with the mouse
            pointer.
        str | int | (float, float):
            Matplotlib :meth:`~matplotlib.figure.Figure.legend` position argument.
        """
        if loc in self.__choices:
            choice = self.__choices.index(loc)
            arg = self.__args[choice]
        elif loc is None:
            choice = 0
            arg = False
        elif loc is True:
            choice = 3
            arg = 'best'
        elif isinstance(loc, Sequence) and not isinstance(loc, str):
            choice = 0
            arg = loc
        elif loc not in self.__args:
            raise ValueError(f"Invalid legend location: {loc!r}; use one of: {enumeration(map(repr, self.__choices), 'or')}")
        else:
            choice = self.__args.index(loc)
            arg = loc

        if self._has_frame:
            self.__ctrl.SetSelection(choice)

        if arg is not False:
            return self.__plot(loc, labels, **kwargs)

    def save_legend(self, *args, **kwargs):
        """Save the legend as image file

        Parameters
        ----------
        ... :
            Parameters for Matplotlib's figure.savefig()
        """
        p = self.plot_legend(show=False)
        p.save(*args, **kwargs)
        p.close()

    def __plot(self, loc: LegendArg, labels: Dict[CellArg, str] = None, **kwargs):
        self.__set_labels(labels)
        if loc and self.__handles:
            if self.__labels is None:
                cells = list(self.__handles)
                labels = [cellname(cell) for cell in cells]
            elif isinstance(self.__labels, dict):
                cells = list(self.__labels.keys())
                labels = list(self.__labels.values())
            else:
                raise TypeError(f"{labels=}; needs to be dict")
            handles = [self.__handles[cell] for cell in cells]
            if loc == 'fig':
                return Legend(handles, labels, **kwargs)
            else:
                # take care of old legend
                if self.legend is not None and loc == 'draggable':
                    self.legend.set_draggable(True)
                elif self.legend is not None:
                    self.legend.remove()
                elif loc == 'draggable':
                    self.legend = self.figure.legend(handles, labels, loc=1, **kwargs)
                    self.legend.set_draggable(True)

                if loc != 'draggable':
                    self.legend = self.figure.legend(handles, labels, loc=loc, **kwargs)
                self.draw()
        elif self.legend is not None:
            self.legend.remove()
            self.legend = None
            self.draw()
        elif not self.__handles:
            raise RuntimeError("No handles to produce legend.")


class Legend(EelFigure):

    def __init__(self, handles, labels, **kwargs):
        layout = Layout(0, 1, 2, tight=False, **kwargs)
        EelFigure.__init__(self, None, layout)

        self.legend = self.figure.legend(handles, labels, loc=2)

        # resize figure to match legend
        if not self._layout.w_fixed and self._has_frame:
            self.draw()
            bb = self.legend.get_window_extent()
            w0, h0 = self._frame.GetSize()
            h = int(h0 + bb.x0 - bb.y0)
            w = int(bb.x0 + bb.x1)
            self._frame.SetSize((w, h))

        self._show()


class TimeController:
    # Link plots that have the TimeSlicer mixin
    def __init__(
            self,
            t: float = 0,
            fixate: bool = False,
    ):
        self._plots = []  # list of weakref to plots
        self.current_time = t
        self.fixate = fixate

    def add_plot(self, plot: 'TimeSlicer'):
        if plot._time_controller is None:
            t = plot._validate_time(self.current_time)
            plot._set_time(t, self.fixate)
            self._plots.append(weakref.ref(plot))
            plot._time_controller = self
        elif plot._time_controller is not self:
            self.merge(plot._time_controller)

    def iter_plots(self):
        needs_cleaning = False
        for ref in self._plots:
            plot = ref()
            if plot is None:
                needs_cleaning = True
            else:
                yield plot
        if needs_cleaning:
            self._plots = [ref for ref in self._plots if ref() is not None]

    def merge(self, time_controller):
        "Merge another TimeController into self"
        for plot in time_controller.iter_plots():
            plot._time_controller = None
            self.add_plot(plot)

    def set_time(self, t, fixate):
        if t == self.current_time and fixate == self.fixate:
            return
        for p in self.iter_plots():
            t = p._validate_time(t)
        for p in self.iter_plots():
            p._update_time_wrapper(t, fixate)
        self.current_time = t
        self.fixate = fixate

    def set_xlim(self, xmin, xmax):
        for p in self.iter_plots():
            if isinstance(p, XAxisMixin):
                p._set_xlim(xmin, xmax, draw=True)


class TimeSlicer:
    # Interface to link time axes of multiple plots.
    # update data in a child plot of time-slices
    _time_dim = None
    _current_time = None  # needs to reflect what is currently displayed
    _initial_time = None  # used by delayed initialization of time-controller
    _display_time_in_frame_title = False

    def __init__(
            self,
            time_dim: Union[UTS, Case] = None,
            time_fixed: bool = None,
            display_text: matplotlib.text.Text = None,
            initial_time: float = None,
    ):
        self._time_controller = None
        self._time_fixed = time_fixed if time_fixed is not None else (initial_time is not None)
        self.__display_text = display_text
        self._initial_time = initial_time
        if time_dim is not None:
            self._init_time_dim(time_dim)

    def _init_time_dim(self, time_dim: Union[UTS, Case]):
        if self._time_dim is not None:
            if time_dim == self._time_dim:
                return
            raise ValueError(f"An incompatible time dimension is already set on {self}\nold: {self._time_dim}\nnew: {time_dim}")
        self._time_dim = time_dim
        if isinstance(time_dim, UTS):
            if self._initial_time is None:
                self._initial_time = time_dim.tmin
        elif isinstance(time_dim, Case):
            if self._initial_time is None:
                self._initial_time = 0
        else:
            raise TypeError(f'{time_dim=}')

    def _init_controller(self):
        # Only instantiated if more than one plots need to be linked
        tc = TimeController(self._initial_time, self._time_fixed)
        tc.add_plot(self)

    def link_time_axis(self, other):
        """Link the time axis of this figure with another figure"""
        if self._time_dim is None:
            raise NotImplementedError("Slice plot for dimension other than time")
        elif not isinstance(other, TimeSlicer):
            raise TypeError(f"{other.__class__.__name__} plot does not support linked time axes")
        elif other._time_dim is None:
            raise NotImplementedError("Slice plot for dimension other than time")
        elif other._time_controller:
            other._time_controller.add_plot(self)
        else:
            if not self._time_controller:
                self._init_controller()
            self._time_controller.add_plot(other)

    def _nudge_time(self, offset):
        if self._time_dim is None:
            return
        current_i = self._time_dim._array_index(self.get_time())
        if offset > 0:
            new_i = min(self._time_dim.nsamples - 1, current_i + offset)
        else:
            new_i = max(0, current_i + offset)
        self._set_time(self._time_dim[new_i], True)

    def get_time(self):
        "Retrieve the current time"
        if self._current_time is None:
            return self._initial_time
        return self._current_time

    def play_movie(self, time_dilation=4.):
        """Cycle through the time axis

        See Also
        --------
        .save_movie : Save a movie to disk for smoother playback
        """
        t = self._time_dim[0]
        self.set_time(t)
        tmax = self._time_dim[-1]
        last_frame = time.time()
        time.sleep(0.05)
        while True:
            now = time.time()
            t += (now - last_frame) / time_dilation
            last_frame = now
            if t > tmax:
                break
            self.set_time(t)
        self.set_time(tmax)

    def set_time(self, time):
        """Set the time point to display

        Parameters
        ----------
        time : scalar
            Time to display.
        """
        self._set_time(time, True)

    def _set_time(self, t, fixate=False):
        "Called by the plot"
        if self._time_controller is None:
            self._update_time_wrapper(t, fixate)
        else:
            self._time_controller.set_time(t, fixate)

    def _update_time_wrapper(self, t, fixate):
        "Called by the TimeController"
        if t == self._current_time and fixate == self._time_fixed:
            return
        self._update_time(t, fixate)
        self._current_time = t
        self._time_fixed = fixate
        if self._display_time_in_frame_title and self._frame:
            self._frame.SetTitleSuffix(f' [{ms(t)} ms]')
        if self.__display_text is not None:
            self.__display_text.set_text(f'{ms(t)} ms')

    def _update_time(self, t, fixate):
        raise NotImplementedError

    def _validate_time(self, t):
        if self._time_dim is not None:
            if t < self._time_dim.tmin:
                return self._time_dim.tmin
            elif t > self._time_dim.tmax:
                return self._time_dim.tmax
        return t

    def _im_array(self):
        # for movies
        raise NotImplementedError


class TimeSlicerEF(TimeSlicer):
    # TimeSlicer for Eelfigure
    _can_set_time = True

    def __init__(
            self,
            x_dimname: str,
            x_dim: Dimension,
            axes: Sequence[matplotlib.axes.Axes] = None,
            redraw: bool = True,
            display_text: matplotlib.text.Text = None,
            initial_time: float = None,
    ):
        if x_dimname != 'time':
            TimeSlicer.__init__(self, time_fixed=True, display_text=display_text)
            return
        TimeSlicer.__init__(self, x_dim, display_text=display_text, initial_time=initial_time)
        self.__axes = self.axes if axes is None else axes
        self.__time_lines = []
        self.__redraw = redraw
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self._register_key('.', self._on_nudge_time)
        self._register_key(',', self._on_nudge_time)

    def _on_click(self, event):
        if self._time_controller and event.inaxes in self.__axes:
            self._set_time(event.xdata, fixate=event.button == 1)

    def _on_motion_sub(self, event):
        if not self._time_fixed and event.inaxes in self.__axes:
            self._set_time(event.xdata)
        return set()

    def _on_nudge_time(self, event):
        self._nudge_time(1 if event.key == '.' else -1)

    def _update_time(self, t, fixate):
        # Implementation for a plot with time axes
        if fixate:
            redraw = True
            if self.__time_lines:
                xdata = (t, t)
                for line in self.__time_lines:
                    line.set_xdata(xdata)
            else:
                for ax in self.__axes:
                    self.__time_lines.append(ax.axvline(t, color='k'))
        else:
            redraw = bool(self.__time_lines)
            while self.__time_lines:
                self.__time_lines.pop().remove()

        if self.__redraw and redraw and self._frame is not None:
            self.canvas.redraw(self.__axes)

    def save_movie(self, filename=None, time_dilation=4., **kwargs):
        """Save the figure with moving time axis as movie

        Parameters
        ----------
        filename : path-like
            Filename for the movie (omit to use a GUI).
        time_dilation : float
            Factor by which to stretch time (default 4). Time dilation is
            controlled through the frame-rate; if the ``fps`` keyword argument
            is specified, ``time_dilation`` is ignored.
        ...
            :func:`imageio.mimwrite` parmeters.

        Notes
        -----
        For a given size (figure width and height in inches), control export
        resolution through ``pyplot.rcParams['figure.dpi']``.
        """
        import imageio

        if filename is None:
            filename = ui.ask_saveas("Save movie...", None, [('Movie (*.mov)', '*.mov')])
            if not filename:
                return
        else:
            filename = os.path.expanduser(filename)

        if 'fps' not in kwargs:
            kwargs['fps'] = 1. / self._time_dim.tstep / time_dilation

        ims = []
        for t in self._time_dim:
            self._set_time(t, True)
            im = self._im_array()
            ims.append(im)
        imageio.mimwrite(filename, ims, **kwargs)

    def _im_array(self):
        # private attr usage is official: https://matplotlib.org/gallery/misc/agg_buffer_to_array.html
        return np.array(self.figure.canvas.renderer._renderer)


class TopoMapKey:

    def __init__(self, data_func):
        self.__topo_data = data_func
        self._register_key('t', self.__on_topo)
        self._register_key('T', self.__on_topo)

    def __on_topo(self, event):
        topo_data = self.__topo_data(event)
        if topo_data is None:
            return

        from ._topo import Topomap

        data, title, proj = topo_data
        if event.key == 't':
            Topomap(data, proj=proj, cmap=self._cmaps, vmax=self._vlims,
                    contours=self._contours, title=title)
        else:
            Topomap(data, proj=proj, cmap=self._cmaps, vmax=self._vlims,
                    contours=self._contours, title=title, axw=9,
                    sensorlabels='name')


class CategorialAxisMixin:

    def __init__(self, ax, axis, layout, label, model, ticks, labels, tick_delim, tick_pos, cells, rotation, origin=None):
        self.__ax = ax
        self.__axis = axis
        self.__cells = cells
        if axis == 'x':
            self.__axis_obj = ax.xaxis
            if origin is not None:
                ax.axhline(origin, color='k', linewidth=mpl.rcParams['axes.linewidth'], clip_on=False)
        elif axis == 'y':
            self.__axis_obj = ax.yaxis
            if origin is not None:
                ax.axvline(origin, color='k', linewidth=mpl.rcParams['axes.linewidth'], clip_on=False)
        else:
            raise ValueError(f"axis={axis!r}")

        # axis label
        if label is True:
            if model is not None and model.name:
                label = model.name.replace('_', ' ')
            else:
                label = False
        if label:
            self.__axis_obj.set_label_text(label)

        # ticks
        self.__axis_obj.set_ticks_position('none')
        if ticks:
            offset = None  # Transform to properly align labels with bars
            kwargs = {}
            if isinstance(ticks, dict) or ticks is True:
                labels_ = find_labels(cells, labels, tick_delim)
                if isinstance(ticks, dict):
                    labels_.update(ticks)
                tick_labels = [labels_[cell] for cell in cells]
            else:
                tick_labels = ticks
            if isinstance(rotation, Real):
                if rotation:
                    ax.tick_params(axis, pad=0)

                if rotation == -90:
                    offset = -mpl_font_size('xtick.labelsize') / 4
                elif 0 < rotation < 90:
                    kwargs['ha'] = 'right'
                    offset = mpl_font_size('xtick.labelsize')
                elif -90 < rotation < 0:
                    kwargs['ha'] = 'left'
                    kwargs['rotation_mode'] = 'anchor'
                    if rotation < -60:
                        offset = mpl_font_size('xtick.labelsize') * ((-rotation - 60) / 90)
            self.__axis_obj.set_ticks(tick_pos, tick_labels, rotation=rotation, **kwargs)
            if offset:
                offset_transform = matplotlib.transforms.ScaledTranslation(offset, 0, ax.figure.dpi_scale_trans)
                for label in ax.xaxis.get_majorticklabels():
                    label.set_transform(label.get_transform() + offset_transform)
        elif ticks is False:
            self.__axis_obj.set_ticks(())

        if axis == 'x' and self._has_frame and not self._layout.w_fixed:
            self._draw_hooks.append(self.__separate_categorial_labels)

    def __separate_categorial_labels(self):
        # make sure x axis labels don't overlap
        labels = self.__axis_obj.get_ticklabels()
        n = len(labels)
        if n > 1:
            bbs = [l.get_window_extent(self.figure.canvas.renderer) for l in labels]
            overlap = max(bbs[i].x1 - bbs[i + 1].x0 for i in range(n - 1))
            if overlap <= 0:
                return False
            extend = n * (overlap + 10)
            w, h = self._frame.GetSize()
            w += int(extend)
            self._frame.SetSize((w, h))
            return True

    def mark_pair(
            self,
            cell_1: Union[float, CellArg],
            cell_2: Union[float, CellArg],
            y: float,
            dy: float = None,
            mark: Union[float, str] = None,
            color: Any = None,
            nudge: Union[bool, float] = None,
            **text_args,
    ):
        """Mark a pair of categories with a line and a label

        Parameters
        ----------
        cell_1
            Data-cell to be compared (can be specified as cell or as
            x-coordinate)
        cell_2
            Second cell to be compared.
        y
            Level above which to plot the bar.
        dy
            Length of vertical ticks on each side of the bar (offsets the
            location of the bar itself to ``y + dy``; use negative values to
            flip orientation).
        mark
            Text label, or p-value to automatically determine the label and
            ``color``.
        color
            Color for bar and ``label``.
        nudge
            Nudge the edges of the bar inwards to allow multiple bars
            side-by-side on the same level of ``y``.
        ...
            All other parameters are used to plot the text label with
            :meth:`matplotlib.axes.Axes.text`.
        """
        if isinstance(cell_1, (str, tuple)):
            x1 = self.__cells.index(cell_1)
        else:
            x1 = cell_1
        if isinstance(cell_2, (str, tuple)):
            x2 = self.__cells.index(cell_2)
        else:
            x2 = cell_2
        location = {'x': 'top', 'y': 'right'}[self.__axis]
        mark_difference(x1, x2, y, mark, dy, color, nudge, location, self.__ax, **text_args)


class XAxisMixin:
    """Manage x-axis

    Parameters
    ----------
    xmin : scalar
        Lower bound of the x axis.
    xmin : scalar
        Upper bound of the x axis.
    axes : list of Axes
        Axes that should be managed by the mixin.
    xlim : scalar | (scalar, scalar)
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).

    Notes
    -----
    Navigation:
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: x-axis zoom in (reduce x axis range)
     - ``d``: x-axis zoom out (increase x axis range)
    """
    _can_set_xlim = True

    def __init__(self, xmin, xmax, xlim=None, axes=None):
        self.__xmin = xmin
        self.__xmax = xmax
        self.__axes = axes or self.axes
        self.__vspans = []
        self._register_key('f', self.__on_zoom_plus)
        self._register_key('d', self.__on_zoom_minus)
        self._register_key('j' if IS_WINDOWS else 'left', self.__on_left)
        self._register_key('l' if IS_WINDOWS else 'right', self.__on_right)
        self._register_key('home', self.__on_beginning)
        self._register_key('end', self.__on_end)
        if xlim is None:
            xlim = (self.__xmin, self.__xmax)
        elif np.isscalar(xlim):
            xlim = (self.__xmin, self.__xmin + xlim)
        self._set_xlim(*xlim)

    def _init_with_data(
            self,
            epochs: Sequence[Sequence[NDVar]],
            xdim: str,
            xlim: Union[float, Tuple[float, float]] = None,
            axes: List[matplotlib.axes.Axes] = None,
            im: bool = False,
    ):
        """Compute axis bounds from data

        Parameters
        ----------
        epochs
            The data that is plotted (to determine axis range).
        xdim
            Dimension that is plotted on the x-axis.
        axes
            Axes that should be managed by the mixin.
        xlim
            Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
            scalar (default is the full x-axis in the data).
        im
            Plot displays an im, i.e. the axes limits need to extend beyond the
            dimension endpoints by half a step (default False).
        """
        dims = (e.get_dim(xdim) for e in chain(*epochs))
        if im:
            dim_extent = [dim._axis_im_extent() for dim in dims]
        else:
            dim_extent = [dim._axis_extent() for dim in dims]
        xmin = min(e[0] for e in dim_extent)
        xmax = max(e[1] for e in dim_extent)
        XAxisMixin.__init__(self, xmin, xmax, xlim, axes)

    def get_xlim(self):
        return self.__axes[0].get_xlim()

    def __animate(self, vmin, vmin_dst, vmax, vmax_dst):
        n_steps = int(0.1 // self._last_draw_time)
        if n_steps > 1:
            vmin_d = vmin_dst - vmin
            vmax_d = vmax_dst - vmax
            for i in range(1, n_steps):
                x = i / n_steps
                self.set_xlim(vmin + x * vmin_d, vmax + x * vmax_d)
        self.set_xlim(vmin_dst, vmax_dst)

    def __on_beginning(self, event):
        left, right = self.get_xlim()
        d = right - left
        self.set_xlim(self.__xmin, min(self.__xmax, self.__xmin + d))

    def __on_end(self, event):
        left, right = self.get_xlim()
        d = right - left
        self.set_xlim(max(self.__xmin, self.__xmax - d), self.__xmax)

    def __on_zoom_plus(self, event):
        left, right = self.get_xlim()
        d = (right - left) / 4.
        self.__animate(left, left + d, right, right - d)

    def __on_zoom_minus(self, event):
        left, right = self.get_xlim()
        d = right - left
        new_left = max(self.__xmin, left - (d / 2.))
        new_right = min(self.__xmax, new_left + 2 * d)
        self.__animate(left, new_left, right, new_right)

    def __on_left(self, event):
        left, right = self.get_xlim()
        d = right - left
        new_left = max(self.__xmin, left - d)
        self.__animate(left, new_left, right, new_left + d)

    def __on_right(self, event):
        left, right = self.get_xlim()
        d = right - left
        new_right = min(self.__xmax, right + d)
        self.__animate(left, new_right - d, right, new_right)

    def _set_xlim(self, left, right, draw=False):
        for ax in self.__axes:
            ax.set_xlim(left, right)
        if draw:
            self.draw()

    def add_vspans(self, intervals, axes=None, *args, **kwargs):
        """Draw vertical bars over axes

        Parameters
        ----------
        intervals : sequence of (start, stop) tuples
            Start and stop positions on the x-axis.
        axes : int | list of int
            Which axes to mark (default is all axes).
        additonal arguments :
            Additional arguments for :func:`matplotlib.axvspan`.
        """
        if axes is None:
            axes = self.__axes
        elif isinstance(axes, int):
            axes = (self.__axes[axes],)
        else:
            axes = [self.__axes[i] for i in axes]

        for ax in axes:
            for xmin, xmax in intervals:
                self.__vspans.append(ax.axvspan(xmin, xmax, *args, **kwargs))
        self.draw()

    def set_xlim(self, left=None, right=None):
        """Set the x-axis limits for all axes"""
        if isinstance(self, TimeSlicer) and self._time_controller is not None:
            if left is None or right is None:
                ax_left, ax_right = self.__axes[0].get_xlim()
                if left is None:
                    left = ax_left
                if right is None:
                    right = ax_right
            self._time_controller.set_xlim(left, right)
        else:
            self._set_xlim(left, right, draw=True)


class YLimMixin:
    """Manage y-axis

    Parameters
    ----------
    plots : Sequence
        Plots to manage. Plots must have ``.ax`` attribute.

    Notes
    -----
    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    # Keep Y-lim and V-lim separate. For EEG, one might want to invert the
    # y-axis without inverting the colormap

    # What should be the organizing principle for different vlims within
    # one figure? Use cases:
    # - 2 axes with different data
    # - (not implemented) one axis with two y-axes
    _can_set_ylim = True

    def __init__(self, plots):
        self.__plots = plots
        self._register_key('r', self.__on_zoom_in)
        self._register_key('c', self.__on_zoom_out)
        self._register_key('i' if IS_WINDOWS else 'up', self.__on_move_up)
        self._register_key('k' if IS_WINDOWS else 'down', self.__on_move_down)
        self._draw_hooks.append(self.__draw_hook)
        # disable because it changes y-limits
        # self._untight_draw_hooks.append(self.__untight_draw_hook)

    def __draw_hook(self):
        need_draw = False
        for p in self.__plots:
            # decimate overlapping ticklabels
            locs = p.ax.yaxis.get_ticklocs()
            we = tuple(l.get_window_extent(self.canvas.renderer) for l in p.ax.yaxis.get_ticklabels())
            start = 0
            step = 1
            locs_list = list(locs) if 0 in locs else None
            while any(e1.ymin < e0.ymax for e0, e1 in intervals(we[start::step])):
                step += 1
                if locs_list:
                    start = locs_list.index(0) % step
            if step > 1:
                p.ax.yaxis.set_ticks(locs[start::step])
                need_draw = True
        return need_draw

    def __untight_draw_hook(self):
        for p in self.__plots:
            # remove the top-most y tick-label if it is outside the figure
            extent = p.ax.yaxis.get_ticklabels()[-1].get_window_extent()
            if extent.height and extent.y1 > self.figure.get_window_extent().y1:
                p.ax.set_yticks(p.ax.get_yticks()[:-1])

    def get_ylim(self):
        vmin = min(p.vmin for p in self.__plots)
        vmax = max(p.vmax for p in self.__plots)
        return vmin, vmax

    def set_ylim(self, bottom=None, top=None):
        """Set the y-axis limits

        Parameters
        ----------
        bottom : scalar
            Lower y-axis limit.
        top : scalar
            Upper y-axis limit.
        """
        if bottom is None and top is None:
            return

        for p in self.__plots:
            p.set_ylim(bottom, top)
        self.draw()

    def __animate(self, vmin, vmin_d, vmax, vmax_d):
        n_steps = int(0.1 // self._last_draw_time)
        if n_steps <= 1:
            self.set_ylim(vmin + vmin_d, vmax + vmax_d)
        else:
            for i in range(1, n_steps + 1):
                x = i / n_steps
                self.set_ylim(vmin + x * vmin_d, vmax + x * vmax_d)

    def __on_move_down(self, event):
        vmin, vmax = self.get_ylim()
        d = (vmax - vmin) * 0.1
        self.__animate(vmin, -d, vmax, -d)

    def __on_move_up(self, event):
        vmin, vmax = self.get_ylim()
        d = (vmax - vmin) * 0.1
        self.__animate(vmin, d, vmax, d)

    def __on_zoom_in(self, event):
        vmin, vmax = self.get_ylim()
        d = (vmax - vmin) * 0.05
        self.__animate(vmin, d, vmax, -d)

    def __on_zoom_out(self, event):
        vmin, vmax = self.get_ylim()
        d = (vmax - vmin) * (1 / 22)
        self.__animate(vmin, -d, vmax, d)
