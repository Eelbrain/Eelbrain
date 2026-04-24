# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import defaultdict

import numpy as np

from ._data_obj import Datalist, Dataset
from ._ndvar import neighbor_correlation
from ._info import BAD_CHANNELS


def _out(out, epochs):
    if out is None:
        return Datalist([[] for _ in range(len(epochs))])
    elif len(out) != len(epochs):
        raise ValueError("out needs same length as epochs, got %i/%i" %
                         (len(out), len(epochs)))
    return out


def new_rejection_ds(ds):
    """Create a rejection Dataset from a Dataset with epochs"""
    out = Dataset(info={BAD_CHANNELS: [], 'epochs.selection': ds.info.get('epochs.selection')})
    out['value'] = ds['value']
    out[:, 'accept'] = True
    out[:, 'rej_tag'] = ''
    return out


def find_flat_epochs(epochs, flat=1e-13, out=None):
    out = _out(out, epochs)
    d = epochs.max('time') - epochs.min('time')
    for i, chi in zip(*np.nonzero(d.get_data(('case', 'sensor')) < flat)):
        ch = epochs.sensor.names[chi]
        if ch not in out[i]:
            out[i].append(ch)

    return out


def find_flat_evoked(epochs, flat=1e-14):
    average = epochs.mean('case')
    d = average.max('time') - average.min('time')
    return epochs.sensor.names[d < flat]


def find_noisy_channels(epochs, mincorr=0.35):
    names = epochs.sensor.names
    out_e = Datalist([list(names[neighbor_correlation(ep) < mincorr]) for ep in epochs])
    return out_e


def channel_listlist_to_dict(listlist):
    out = defaultdict(list)
    for i, chs in enumerate(listlist):
        for ch in chs:
            out[ch].append(i)
    return dict(out)
