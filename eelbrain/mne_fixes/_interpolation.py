# Mostly retaining MNE-Python functions to compensate for API changes
import logging

import numpy as np

import mne
from mne.bem import _fit_sphere
from mne.channels.interpolation import _make_interpolation_matrix
try:
    from mne.forward import _map_meg_or_eeg_channels
except ImportError:  # mne < 0.21
    from mne.forward import _map_meg_channels as _map_meg_or_eeg_channels


# mne 0.10 function
def map_meg_channels(inst, picks_good, picks_bad, mode):
    info_from = mne.pick_info(inst.info, picks_good, copy=True)
    info_to = mne.pick_info(inst.info, picks_bad, copy=True)
    return _map_meg_or_eeg_channels(info_from, info_to, mode=mode, origin='auto')


# private in 0.9.0 (Epochs method)
def get_channel_positions(self, picks=None):
    """Gets channel locations from info

    Parameters
    ----------
    picks : array-like of int | None
        Indices of channels to include. If None (default), all meg and eeg
        channels that are available are returned (bad channels excluded).
    """
    if picks is None:
        picks = mne.pick_types(self.info, meg=True, eeg=True)
    chs = self.info['chs']
    pos = np.array([chs[k]['loc'][:3] for k in picks])
    n_zero = np.sum(np.sum(np.abs(pos), axis=1) == 0)
    if n_zero > 1:  # XXX some systems have origin (0, 0, 0)
        raise ValueError('Could not extract channel positions for '
                         '{} channels'.format(n_zero))
    return pos


def _make_interpolator(inst, bad_channels):
    """Find indexes and interpolation matrix to interpolate bad channels

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    """
    logger = logging.getLogger(__name__)

    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)

    picks = mne.pick_types(inst.info, meg=False, eeg=True, exclude=[])
    bads_idx[picks] = [inst.ch_names[ch] in bad_channels for ch in picks]
    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = get_channel_positions(inst, picks)

    # Make sure only EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]

    pos_good = pos[goods_idx_pos]
    pos_bad = pos[bads_idx_pos]

    # test spherical fit
    radius, center = _fit_sphere(pos_good, False)
    distance = np.sqrt(np.sum((pos_good - center) ** 2, 1))
    distance = np.mean(distance / radius)
    if np.abs(1. - distance) > 0.1:
        logger.warning('Your spherical fit is poor, interpolation results are '
                       'likely to be inaccurate.')

    logger.info('Computing interpolation matrix from {0} sensor '
                'positions'.format(len(pos_good)))

    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    return goods_idx, bads_idx, interpolation


def _interpolate_bads_eeg(epochs, bad_channels_by_epoch):
    """Interpolate bad channels per epoch

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    bad_channels_by_epoch : list of list of str
        Bad channel names specified for each epoch. For example, for an Epochs
        instance containing 3 epochs: ``[['F1'], [], ['F3', 'FZ']]``
    """
    logger = logging.getLogger(__name__)

    if len(bad_channels_by_epoch) != len(epochs):
        raise ValueError("Unequal length of epochs (%i) and "
                         "bad_channels_by_epoch (%i)"
                         % (len(epochs), len(bad_channels_by_epoch)))

    interp_cache = {}
    for i, bad_channels in enumerate(bad_channels_by_epoch):
        if not bad_channels:
            continue

        # find interpolation matrix
        key = tuple(sorted(bad_channels))
        if key in interp_cache:
            goods_idx, bads_idx, interpolation = interp_cache[key]
        else:
            goods_idx, bads_idx, interpolation = interp_cache[key] = _make_interpolator(epochs, key)

        # apply interpolation
        logger.info('Interpolating %i sensors on epoch %i', bads_idx.sum(), i)
        epochs._data[i, bads_idx, :] = np.dot(interpolation, epochs._data[i, goods_idx, :])


def _interpolate_bads_meg(epochs, bad_channels_by_epoch, interp_cache):
    """Interpolate bad MEG channels per epoch

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    bad_channels_by_epoch : list of list of str
        Bad channel names specified for each epoch. For example, for an Epochs
        instance containing 3 epochs: ``[['F1'], [], ['F3', 'FZ']]``
    interp_cache : dict
        Will be updated.

    Notes
    -----
    Based on mne 0.9.0 MEG channel interpolation.
    """
    logger = logging.getLogger(__name__)
    if len(bad_channels_by_epoch) != len(epochs):
        raise ValueError("Unequal length of epochs (%i) and "
                         "bad_channels_by_epoch (%i)"
                         % (len(epochs), len(bad_channels_by_epoch)))

    import time
    logger.debug("starting interpolation")
    t0 = time.time()

    # make sure bad_chs includes only existing channels
    all_chs = set(epochs.ch_names)
    bad_channels_by_epoch = [all_chs.intersection(chs) for chs in
                             bad_channels_by_epoch]

    # find needed interpolators
    sorted_bad_chs_by_epoch = [tuple(sorted(bad_channels)) for bad_channels in
                               bad_channels_by_epoch]
    needed = set(sorted_bad_chs_by_epoch)
    needed.discard(())
    n_keys = len(needed)
    if not n_keys:
        return
    bads = tuple(sorted(epochs.info['bads']))

    # make sure the cache is based on the correct channels
    if 'ch_names' not in interp_cache or interp_cache['ch_names'] != epochs.ch_names:
        interp_cache.clear()
        interp_cache['ch_names'] = epochs.ch_names

    # create interpolators
    make_interpolators(interp_cache, needed, bads, epochs)
    t1 = time.time()

    logger.debug("interpolate epochs")
    for i, key in enumerate(sorted_bad_chs_by_epoch):
        if not key:
            continue
        # apply interpolation
        picks_good, picks_bad, interpolation = interp_cache[bads, key]
        logger.info('Interpolating sensors %s on epoch %s', picks_bad, i)
        epochs._data[i, picks_bad, :] = interpolation.dot(epochs._data[i, picks_good, :])
    t2 = time.time()

    logger.debug("Interpolation took %s/%s seconds" % (t1 - t0, t2 - t1))


def make_interpolators(interp_cache, keys, bads, epochs):
    make = [k for k in keys if (bads, k) not in interp_cache]
    logger = logging.getLogger(__name__)
    logger.debug("Making %i of %i interpolators" % (len(make), len(keys)))
    for key in make:
        picks_good = mne.pick_types(epochs.info, meg=True, ref_meg=False, exclude=key)
        picks_bad = mne.pick_channels(epochs.ch_names, key)
        interpolation = map_meg_channels(epochs, picks_good, picks_bad, 'accurate')
        interp_cache[bads, key] = picks_good, picks_bad, interpolation
