import numpy as np


def dspm_lut(fmin, fmid, fmax, n=256):
    """Create a color look up table (lut) for a dSPM plot

    Parameters
    ----------
    fmin, fmid, fmax : scalar
        Start-, mid- and endpoint for the color gradient.
    n : int
        Number of distinct color values in the table.

    Notes
    -----
    Transitions:

    0-fmin
        Transparent.
    fmin-fmid
        Transparent - opaque.
    fmid-fmax
        Hue shift.
    """
    if not (fmin < fmid < fmax):
        raise ValueError("Invalid colormap, we need fmin < fmid < fmax")
    elif fmin < 0:
        raise ValueError("The dSPM color gradient is symmetric around 0, fmin "
                         "needs to be > 0 (got %s)." % fmin)

    lut = np.zeros((n, 4), dtype=np.uint8)
    i0 = int(round(n / 2))  # v=0 (middle of the LUT)
    imin = int(round((fmin / fmax) * i0))  # i0 is the range of one side of the LUT
    min_n = i0 - imin
    min_p = i0 + imin
    imid = int(round((fmid / fmax) * i0))
    mid_n = i0 - imid
    mid_p = i0 + imid

    # red end
    lut[i0:, 0] = 255
    lut[mid_p:, 1] = np.linspace(0, 255, n - mid_p)

    # blue end
    lut[:i0, 2] = 255
    lut[:mid_n, 0] = np.linspace(127, 0, mid_n)
    lut[:mid_n, 1] = np.linspace(127, 0, mid_n)

    # alpha
    lut[:mid_n, 3] = 255
    lut[mid_n:min_n, 3] = np.linspace(255, 0, min_n - mid_n)
    lut[min_n:min_p, 3] = 0
    lut[min_p:mid_p, 3] = np.linspace(0, 255, mid_p - min_p)
    lut[mid_p:, 3] = 255

    return lut


def p_lut(pmap, tmap=None, p0=0.05, p1=0.01, p0alpha=0.5, n=256):
    """Creat a color look up table (lut) for p-values

    Parameters
    ----------
    pmap : NDVar
        Map of p-values.
    tmap : NDVar
        Map of signed statistic (only used to code the sign of each p-value).
    p0 : scalar
        Highest p-value that is visible.
    p1 : scalar
        P-value where the colormap changes from ramping alpha to ramping color.
    p0alpha : 1 >= float >= 0
        Alpha at ``p0``. Set to 0 for a smooth transition, or a larger value to
        clearly delineate significant regions (default 0.5).
    n : int
        Number of color categories in the lut.
    """
    if p1 >= p0:
        raise ValueError("p1 needs to be smaller than p0.")

    pstep = 2 * p0 / (n - 3)  # there are n - 1 steps, 2 leave the visible range

    # max p-value that needs to be represented (1 step out of visible)
    vmax = p0 + pstep

    # bring interesting p-values to the range [pstep vmax]
    pmap = vmax - pmap

    # set uninteresting values to zero
    pmap.x.clip(0, vmax, pmap.x)

    # add sign to p-values
    if tmap is not None:
        pmap.x *= np.sign(tmap.x)

    # http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
    lut = np.zeros((n, 4), dtype=np.uint8)

    middle = n // 2
    p0p = middle + 1
    p1n = int(round(p1 / pstep))
    p1p = n - p1n

    # negative colors
    lut[:middle, 2] = 255
    lut[:p1n, 0] = np.linspace(255, 0, p1n)

    # positive colors
    lut[middle:, 0] = 255
    lut[p1p:, 1] = np.linspace(0, 255, n - p1p)

    # alpha
    if p0alpha == 1:
        lut[:middle, 3] = 255
        lut[p0p:, 3] = 255
    elif not 0 <= p0alpha <= 1:
        raise ValueError(f"{p0alpha=}")
    else:
        p0_alpha = int(round(p0alpha * 255))
        lut[:p1n, 3] = 255
        lut[p1n:middle, 3] = np.linspace(255, p0_alpha, middle - p1n)
        lut[p0p:p1p, 3] = np.linspace(p0_alpha, 255, p1p - p0p)
        lut[p1p:, 3] = 255

    pmap.info['cmap ticks'] = {
        -vmax: '<' + str(p1 / 10)[1:],
        -vmax + p1: str(p1)[1:],
        0: str(p0)[1:],
        vmax - p1: str(p1)[1:],
        vmax: '<' + str(p1 / 10)[1:],
    }

    return pmap, lut, vmax
