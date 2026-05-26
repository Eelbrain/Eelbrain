import numpy as np

SHUFFLE_METHODS = ('shift',)


def shuffle(ndvar, method, band=slice(None), angle=180):
    """Shuffle NDVar on the time axis."""
    if band is None:
        band = slice(None)

    if method not in SHUFFLE_METHODS:
        raise ValueError(f"method={method!r}; need one of {' | '.join(SHUFFLE_METHODS)}")
    if method != 'shift':
        raise RuntimeError(f"method={method!r}")

    assert 0 <= angle < 360
    time_ax = ndvar.get_axis('time')
    assert 1 <= ndvar.ndim <= 2, f"ndvar must be 1d or 2d, got {ndvar!r}"
    i_mid = int(round(ndvar.x.shape[time_ax] * (angle / 360)))
    out = ndvar.copy()
    if ndvar.ndim == 1:
        out.x[:i_mid] = ndvar.x[-i_mid:]
        out.x[i_mid:] = ndvar.x[:-i_mid]
    elif time_ax == 1:
        out.x[band, :i_mid] = ndvar.x[band, -i_mid:]
        out.x[band, i_mid:] = ndvar.x[band, :-i_mid]
    elif time_ax == 0:
        out.x[:i_mid, band] = ndvar.x[-i_mid:, band]
        out.x[i_mid:, band] = ndvar.x[:-i_mid, band]
    else:
        raise ValueError(f"{ndvar}: More than 2d")
    return out
