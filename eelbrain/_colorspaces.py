"""Data-specific colormaps"""
from itertools import cycle
from math import ceil
from numbers import Real
from typing import Sequence, List, Tuple, Union

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
import numpy as np


# https://jfly.uni-koeln.de/html/color_blind/ (Fig. 16)
UNAMBIGUOUS_COLORS = {
    'black': (0.00, 0.00, 0.00),
    'orange': (0.90, 0.60, 0.00),
    'sky blue': (0.35, 0.70, 0.90),
    'bluish green': (0.00, 0.60, 0.50),
    'yellow': (0.95, 0.90, 0.25),
    'blue': (0.00, 0.45, 0.70),
    'vermilion': (0.80, 0.40, 0.00),
    'reddish purple': (0.80, 0.60, 0.70),
}

# How to handle different colormaps
SYMMETRIC_CMAPS = [
    'cold_hot', 'cold_white_hot',
]
ZEROBASED_CMAPS = []
ALPHA_CMAPS = {}  # corresponding cmaps with transparency (alpha channel)


def register_cmap(
        cmap: matplotlib.colors.Colormap,
        symmetric: bool = False,
        zero_based: bool = False,
        alpha_version: Union[bool, str] = None,
):
    matplotlib.colormaps.register(cmap)
    if symmetric:
        SYMMETRIC_CMAPS.append(cmap.name)
    if zero_based:
        ZEROBASED_CMAPS.append(cmap.name)
    if alpha_version is True:
        ALPHA_CMAPS[cmap.name] = f'{cmap.name}-a'
    elif alpha_version:
        ALPHA_CMAPS[cmap.name] = alpha_version


class LocatedColormap:
    vmin = None
    vmax = None
    symmetric = False


class LocatedListedColormap(ListedColormap, LocatedColormap):
    pass


class LocatedLinearSegmentedColormap(LinearSegmentedColormap, LocatedColormap):
    pass


def lch_to_rgb(lightness, chroma, hue):
    "Convert Lightness/Chroma/Hue color representation to RGB"
    psych = LCHabColor(lightness, chroma, hue * 360)
    rgb = convert_color(psych, sRGBColor)
    return rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b


def rgb_to_lch(r, g, b):
    rgb = sRGBColor(r, g, b)
    lch = convert_color(rgb, LCHabColor)
    return lch.lch_l, lch.lch_c, lch.lch_h / 360


def _to_rgb(arg, alpha=False):
    if isinstance(arg, (float, int)):
        arg = lch_to_rgb(50, 100, arg)

    if alpha:
        return to_rgba(arg)
    else:
        return to_rgb(arg)


def make_seq_cmap(seq, val, name):
    """Colormap from sequence of RGB values

    Parameters
    ----------
    seq : iterator
        Each entry is either an RGB tuple (if pre- and post color are
        identical) or a tuple with two RGB tuples (separate pre- and post-
        colors).
    val : iterator
        For each entry in ``seq``, the coordinate on the colormap.
    name : str
        Colormap name.
    """
    red = []
    green = []
    blue = []
    for v, col in zip(val, seq):
        if len(col) == 3:
            r0, g0, b0 = col
            r1, g1, b1 = col
        elif len(col) == 2:
            (r0, g0, b0), (r1, g1, b1) = col
        else:
            raise ValueError('col in seq: %s' % str(col))
        red.append((v, r0, r1))
        green.append((v, g0, g1))
        blue.append((v, b0, b1))
    cdict = {'red': red, 'green': green, 'blue': blue}
    return LinearSegmentedColormap(name, cdict)


def unambiguous_color(color: str, lightness: float = None, chroma: float = None):
    """Generate an `unambiguous color <https://jfly.uni-koeln.de/html/color_blind/#pallet>`_

    Parameters
    ----------
    color
        Color name, one of: black, orange, sky blue, bluish green, yellow,
        blue, vermilion, reddish purple.
    lightness
        Adjust the lightness (in percent).
    chroma
        Adjust the saturation (in percent).
    """
    r, g, b = UNAMBIGUOUS_COLORS[color]
    l, c, h = rgb_to_lch(r, g, b)
    if lightness is not None:
        l = lightness
    if chroma is not None:
        c = chroma
    return lch_to_rgb(l, c, h)


def twoway_cmap(n1, hue_start=0.1, hue_shift=0.5, name=None, hues=None):
    """Create colormap for two-way interaction

    Parameters
    ----------
    n1 : int
        Number of levels on the first factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    name : str
        Name of the colormap.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    """
    # within each hue, create values for [-1, -0.5, 0.5, 1]
    # return list of [i, (r, g, b), (r, g, b)]
    if hues is None:
        hues = np.linspace(hue_start, hue_start + 1, n1, False) % 1.
    hue_shift *= (0.5 / n1)

    seqs = []
    for h in hues:
        h_pre = (h - hue_shift) % 1
        h_post = (h + hue_shift) % 1
        seqs.append((lch_to_rgb(0, 100, h_pre),
                     lch_to_rgb(50, 100, h),
                     lch_to_rgb(100, 100, h_post)))

    seq = []
    for i in range(n1):
        seq.append((seqs[i - 1][-1], seqs[i][0]))
        seq.append(seqs[i][1])
        if i == n1 - 1:
            seq.append((seqs[i][2], seqs[0][0]))

    loc = np.linspace(0, 1, n1 * 2 + 1)

    if name is None:
        name = "%i_by_n" % n1

    return make_seq_cmap(seq, loc, name)


def oneway_colors(
        n: int,
        hue_start: Union[float, Sequence[float]] = 0.2,
        light_range: Union[float, Tuple[float, float]] = 0.5,
        light_cycle: int = None,
        always_cycle_hue: bool = False,
        locations: Sequence[float] = None,
        unambiguous: Union[bool, Sequence[int]] = None,
) -> list:
    "Create colors for categories (see docs at colors_for_oneway)"
    if unambiguous:
        if unambiguous is True:
            if n > 8:
                raise ValueError("unambiguous=True for n > 8")
            indices = list(range(n))
        else:
            if min(unambiguous) < 1 or max(unambiguous) > 8:
                raise ValueError(f"unambiguous={unambiguous}: values outside range (1, 8)")
            indices = [i - 1 for i in unambiguous]
        colors = list(UNAMBIGUOUS_COLORS.values())
        return [colors[i] for i in indices]

    if locations is not None:
        raise NotImplementedError('locations for non-cmap based colors')

    if light_cycle is None or always_cycle_hue:
        n_hues = n
    else:
        n_hues = int(ceil(n / light_cycle))

    if isinstance(hue_start, Real):
        hue = np.linspace(hue_start, hue_start + 1, n_hues, False) % 1.
        if light_cycle and not always_cycle_hue:
            hue = np.repeat(hue, light_cycle)
    elif len(hue_start) >= n_hues:
        hue = hue_start
    else:
        raise ValueError("If list of hues is provided it needs ot contain at least as many hues as there are cells")

    if isinstance(light_range, (list, tuple)):
        start, stop = light_range
    else:
        start = 0.5 + 0.5 * light_range
        stop = 0.5 - 0.5 * light_range

    if light_cycle is None:
        lightness = np.linspace(100 * start, 100 * stop, n)
    else:
        tile = np.linspace(100 * start, 100 * stop, light_cycle)
        lightness = cycle(tile)
        if n % light_cycle:
            hue = hue[:n]

    return [lch_to_rgb(l, 100, h) for l, h in zip(lightness, hue)]


def twoway_colors(
        n1: int,
        n2: int,
        hue_start: float = 0.2,
        hue_shift: float = 0.,
        hues: Sequence[float] = None,
        lightness: Union[float, Sequence[float]] = None,
) -> List[Tuple[float, float, float]]:
    """Create colors for two-way interaction

    Parameters
    ----------
    n1
        Number of levels on the first factor.
    n2
        Number of levels on the second factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    hues
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    lightness
        If specified as scalar, colors will occupy the range
        ``[lightness, 100-lightness]``. Can also be given as list with one
        value corresponding to each element in the second factor.
    """
    if hues is None:
        hues = np.linspace(hue_start, hue_start + 1, n1, False) % 1.
    else:
        hues = np.asarray(hues)
        if np.any(hues > 1) or np.any(hues < 0):
            raise ValueError(f"hues={hues}: values out of range, need to be in [0, 1]")
        elif len(hues) < n1:
            raise ValueError(f"hues={hues}: need as many hues as levels in the first factor (got {len(hues)}, need {n1})")
    hue_shift *= (1. / 3. / n1)

    if lightness is None:
        lstart = 60. / n2
        ls = np.linspace(lstart, 100 - lstart, n2)
    elif isinstance(lightness, Real):
        ls = np.linspace(lightness, 100 - lightness, n2)
    else:
        if len(lightness) != n2:
            raise ValueError(f"lightness={lightness!r}: need {n2} values")
        ls = lightness

    colors = []
    for hue in hues:
        hs = np.linspace(hue - hue_shift, hue + hue_shift, n2) % 1
        colors.extend(lch_to_rgb(l, 100, h) for l, h in zip(ls, hs))

    return colors


def two_step_colormap(left_max, left, center='transparent', right=None, right_max=None, name='two-step'):
    """Colormap with 2 or 4 gradients (e.g., ``white - red - transparent - blue - white``)

    Parameters
    ----------
    left_max : matplotlib color
        Left end of the colormap.
    left : matplotlib color
        Left middle of the colormap.
    center : matplotlib color | 'transparent'
        Color for the middle value; 'transparent to make the middle transparent
        (default).
    right : matplotlib color
        Right middle of the colormap (if not specified, the colormap ends at
        the location specified by ``center``).
    right_max : matplotlib color
        Right end of the colormap.
    name : str
        Name for the colormap.

    Examples
    --------
    Standard red/blue::

        >>> cmap = plot.two_step_colormap('black', 'red', 'transparent', 'blue', 'black', name='red-blue')
        >>> plot.ColorBar(cmap, 1)

    Or somewhat more adventurous::

        >>> cmap = plot.two_step_colormap('black', (1, 0, 0.3), 'transparent', (0.3, 0, 1), 'black', name='red-blue-2')
    """
    if center == 'transparent':
        center_ = None
        transparent_middle = True
    else:
        center_ = _to_rgb(center, False)
        transparent_middle = False
    left_max_ = _to_rgb(left_max, transparent_middle)
    left_ = _to_rgb(left, transparent_middle)
    is_symmetric = right is not None
    if is_symmetric:
        right_ = _to_rgb(right, transparent_middle)
        right_max_ = _to_rgb(right_max, transparent_middle)
    else:
        right_ = right_max_ = None

    kind = (is_symmetric, transparent_middle)
    if kind == (False, False):
        clist = (
            (0.0, center_),
            (0.5, left_),
            (1.0, left_max_),
        )
    elif kind == (False, True):
        clist = (
            (0.0, (*left_[:3], 0)),
            (0.5, left_),
            (1.0, left_max_),
        )
    elif kind == (True, False):
        clist = (
            (0.0, left_max_),
            (0.25, left_),
            (0.5, center_),
            (0.75, right_),
            (1.0, right_max_),
        )
    elif kind == (True, True):
        clist = (
            (0.0, left_max_),
            (0.25, left_),
            (0.5, (*left_[:3], 0)),
            (0.5, (*right_[:3], 0)),
            (0.75, right_),
            (1.0, right_max_),
        )
    else:
        raise RuntimeError
    cmap = LocatedLinearSegmentedColormap.from_list(name, clist)
    cmap.set_bad('w', alpha=0.)
    cmap.symmetric = is_symmetric
    return cmap


def make_cmaps():
    """Create custom colormaps and register them with matplotlib"""
    # Polar
    # -----
    # bi-polar, blue-white-red based
    cmap = LinearSegmentedColormap.from_list(
        "polar", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (1.0, (1.0, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    # polar-alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "polar-a", (
            (0.0, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (1.0, (1.0, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    # xpolar ("extra-polar"): fade ends into black
    cmap = LinearSegmentedColormap.from_list(
        "xpolar", (
            (0.0, (0.0, 0.0, 0.0)),
            (0.3, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (0.7, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    # extra-polar alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "xpolar-a", (
            (0.0, (0.0, 0.0, 0.0, 1.0)),
            (0.3, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.7, (1.0, 0.0, 0.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    # extra-polar light: ends are light instead of dark
    cmap = LinearSegmentedColormap.from_list(
        "lpolar", (
            (0.0, (0.5, 1.0, 1.0, 1.0)),
            (0.1, (0.0, 1.0, 1.0, 1.0)),
            (0.3, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.7, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 0.5, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    # -alpha
    cmap = LinearSegmentedColormap.from_list(
        "lpolar-a", (
            (0.0, (0.5, 1.0, 1.0, 1.0)),
            (0.1, (0.0, 1.0, 1.0, 1.0)),
            (0.3, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.7, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 0.5, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)
    # version with alpha higher up
    cmap = LinearSegmentedColormap.from_list(
        "lpolar-aa", (
            (0.0, (0.5, 1.0, 1.0, 1.0)),
            (0.1, (0.0, 1.0, 1.0, 1.0)),
            (0.3, (0.0, 0.0, 1.0, 0.5)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.7, (1.0, 0.0, 0.0, 0.5)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 0.5, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    # Lux
    # ---
    # lux:  cyan vs orange
    cmap = LinearSegmentedColormap.from_list(
        "lux", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.15, (0.0, 1.0, 1.0, 1.0)),
            (0.4, (0.0, 0.3, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.6, (1.0, 0.3, 0.0, 1.0)),
            (0.85, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    # lux-gray:  gray center
    cmap = LinearSegmentedColormap.from_list(
        "lux-gray", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.15, (0.0, 1.0, 1.0, 1.0)),
            (0.4, (0.0, 0.3, 1.0, 1.0)),
            (0.5, (0.9, 0.9, 0.9, 1.0)),
            (0.6, (1.0, 0.3, 0.0, 1.0)),
            (0.85, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version='lux-a')
    # -alpha
    cmap = LinearSegmentedColormap.from_list(
        "lux-a", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.15, (0.0, 1.0, 1.0, 1.0)),
            (0.4, (0.0, 0.3, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.6, (1.0, 0.3, 0.0, 1.0)),
            (0.85, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    # polar-lux:  polar on the inside, blend to lux
    cmap = LinearSegmentedColormap.from_list(
        "polar-lux", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.1, (0.0, 1.0, 1.0, 1.0)),
            (0.4, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.6, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    # -alpha
    cmap = LinearSegmentedColormap.from_list(
        "polar-lux-a", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.1, (0.0, 1.0, 1.0, 1.0)),
            (0.4, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.6, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    # lux-purple:  on the negative side, blend into purple instead of cyan
    cmap = LinearSegmentedColormap.from_list(
        "lux-purple", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.15, (1.0, 0.0, 1.0, 1.0)),
            (0.4, (0.3, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.6, (1.0, 0.3, 0.0, 1.0)),
            (0.85, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    cmap = LinearSegmentedColormap.from_list(
        "lux-purple-a", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.15, (1.0, 0.0, 1.0, 1.0)),
            (0.4, (0.3, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.6, (1.0, 0.3, 0.0, 1.0)),
            (0.85, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    cmap = LinearSegmentedColormap.from_list(
        "polar-lux-purple", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.1, (1.0, 0.0, 1.0, 1.0)),
            (0.4, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (0.6, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True, alpha_version=True)
    cmap = LinearSegmentedColormap.from_list(
        "polar-lux-purple-a", (
            (0.0, (1.0, 1.0, 1.0, 1.0)),
            (0.1, (1.0, 0.0, 1.0, 1.0)),
            (0.4, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.6, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap, symmetric=True)

    # phase
    # -----
    cmap = LinearSegmentedColormap.from_list(
        "phase", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap)

    # Significance
    # ------------
    # sig: significance map for specific vmax=0.05
    cmap = LinearSegmentedColormap.from_list(
        "sig", (
            (0.0,  (1.0, 1.0, 1.0)),
            (0.02, (1.0, 1.0, 0.0)),
            (0.2,  (1.0, 0.5, 0.0)),
            (1.0,  (1.0, 0.0, 0.0)),
        ))
    cmap.set_over('k', alpha=0.)
    cmap.set_bad('b', alpha=0.)
    register_cmap(cmap, zero_based=True)


make_cmaps()

# https://matplotlib.org/tutorials/colors/colormaps.html
mpl_diverging = ('PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic')
SYMMETRIC_CMAPS.extend(mpl_diverging)
SYMMETRIC_CMAPS.extend(f'{name}_r' for name in mpl_diverging)
