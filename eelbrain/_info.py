# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Info dictionary

Main entries
------------

meas
    Measurement:

    B
        Magnetic field strength (MEG).
    V
        Voltage (EEG).
    p
        Probability (statistics).
    r, t, f
        Statistic (correlation, t- and f- values).

contours
    Default plotting argument.
cmap
    Default plotting argument.
vmin, vmax
    Default plotting argument.


Used by GUIs
------------

bad_channels
    Bad channels.


Internally used
---------------

cmap ticks
    Non-standard cmap-ticks; used by plot.brain.p_map to transmit proper tick
    labels for remapped p-values
"""
from __future__ import annotations

import mne
import numpy as np
from typing import TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from ._data_obj import Dataset


# Key constants for info dictionaries
BAD_CHANNELS = 'bad_channels'
INTERPOLATE_CHANNELS = "interpolate_channels"
INTERPOLATE_WINDOWS = "interpolate_windows"
INTERPOLATE_WINDOWS_MAX = "interpolate_windows_max"
# Parameters that should be reset if measurement type changes
MAIN_ARGS = ('meas', 'unit', 'cmap', 'vmin', 'vmax', 'contours')


###########
# Constants
###########

_unit_fmt = {
    1: "%s",
    1e-3: "m%s",
    1e-6: "µ%s",
    1e-9: "n%s",
    1e-12: "p%s",
    1e-15: "f%s",
}


#####################
# Generate info dicts
#####################

def copy(old: dict):
    "Carry over all meaningful parameters"
    return {k: v for k, v in old.items() if k in MAIN_ARGS}


def default_info(meas, old=None):
    "Default colorspace info"
    info = {'meas': meas}
    return _update(info, old)


def for_cluster_pmap(old=None):
    info = {
        'meas': 'p',
        'cmap': 'sig',
        'vmax': 0.05,
        'contours': {0.05: (0., 0., 0.)},
    }
    return _update(info, old)


def for_p_map(old=None):
    "Info dict for significance map"
    info = {
        'meas': 'p',
        'cmap': 'sig',
        'vmax': .05,
        'contours': {.01: '.5', .001: '0'},
    }
    return _update(info, old)


def for_normalized_data(old, default_meas):
    info = {'meas': old.get('meas', default_meas), 'unit': 'normalized'}
    return _update(info, old)


def for_stat_map(meas, c0=None, c1=None, c2=None, tail=0, contours=None, old=None):
    if meas == 'r':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 't':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 'f' or meas == 't2':
        info = {'meas': meas, 'cmap': 'BuPu_r', 'vmin': 0}
    else:
        info = default_info(meas)

    if contours is None:
        contours = {}
        if c0 is not None:
            if tail >= 0:
                contours[c0] = (1.0, 0.5, 0.1)
            if tail <= 0:
                contours[-c0] = (0.5, 0.1, 1.0)
        if c1 is not None:
            if tail >= 0:
                contours[c1] = (1.0, 0.9, 0.2)
            if tail <= 0:
                contours[-c1] = (0.9, 0.2, 1.0)
        if c2 is not None:
            if tail >= 0:
                contours[c2] = (1.0, 1.0, 0.8)
            if tail <= 0:
                contours[-c2] = (1.0, 0.8, 1.0)
    if contours:
        info['contours'] = contours
    return _update(info, old)


def for_eeg(vmax=None, mult=1, old=None):
    info = {
        'meas': 'V',
        'unit': _unit_fmt[1 / mult] % 'V',
        'cmap': 'xpolar',
    }
    if vmax is not None:
        info['vmax'] = vmax
    return _update(info, old)


def for_meg(vmax=None, mult=1, meas="B", unit='T', old=None):
    info = {
        'meas': meas,
        'unit': _unit_fmt[1 / mult] % unit,
        'cmap': 'xpolar',
    }
    if vmax is not None:
        info['vmax'] = vmax
    return _update(info, old)


def for_boolean(old=None):
    return _update({}, old)


def for_data(x, old=None):
    "For data, depending on type"
    if x.dtype.kind == 'b':
        return for_boolean(old)
    elif old:
        return old.copy()
    else:
        return {}


###################
# Update info dicts
###################

def _update(new, old):
    """Update the plotting arguments in ``old`` to reflect a new colorspace

    Parameters
    ----------
    new : dict
        The new info that should be updated.
    old : dict
        The previous info dictionary.

    Returns
    -------
    info : dict
        The updated dictionary.
    """
    if old:
        new.update({k: v for k, v in old.items() if k not in MAIN_ARGS})
    return new


#######################
# operate on info dicts
#######################

def _values_equal(a, b):
    "Test equality, taking into account array values"
    if a is b:
        return True
    elif type(a) is not type(b):
        return False
    elif not np.iterable(a):
        return a == b
    elif len(a) != len(b):
        return False
    elif isinstance(a, (tuple, list)):
        return all(_values_equal(a_, b_) for a_, b_ in zip(a, b))
    elif isinstance(a, dict):
        if a.keys() == b.keys():
            return all(_values_equal(a[k], b[k]) for k in a)
        else:
            return False
    elif isinstance(a, mne.io.BaseRaw):
        return isinstance(b, a.__class__) and _values_equal(a.info, b.info)
    from ._data_obj import NDVar
    if isinstance(a, (np.ndarray, NDVar)):
        if a.shape == b.shape:
            return (a == b).all()
        else:
            return False
    else:
        return a == b


def merge_info(items: Sequence[Dataset]) -> dict:
    "Merge info dicts from several objects"
    info0 = items[0].info
    other_infos = [i.info for i in items[1:]]
    # find shared keys
    info_keys = set(info0.keys())
    for info in other_infos:
        info_keys.intersection_update(info.keys())
    # find shared values
    out = {}
    for key in info_keys:
        v0 = info0[key]
        if all(_values_equal(info[key], v0) for info in other_infos):
            out[key] = v0
    return out
