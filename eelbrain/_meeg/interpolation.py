# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Per-epoch and time-windowed bad-channel interpolation.

Eelbrain interpolates bad channels separately for each epoch (and, for long
epochs, only over the time window in which a channel is bad). This builds on the
vendored MNE-Python spherical-spline / field-mapping primitives in
:mod:`eelbrain.mne_fixes._interpolation`.
"""
from __future__ import annotations

import logging

import numpy as np

import mne

from ..mne_fixes._interpolation import get_channel_positions, map_meg_channels, _make_interpolation_matrix
from .base import BadChannelWindow


def _make_interpolator(
        inst: mne.Epochs,
        bad_channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find indexes and interpolation matrix to interpolate bad channels

    Parameters
    ----------
    inst
        The data to interpolate. Must be preloaded.
    bad_channels
        Names of the channels to interpolate.

    Returns
    -------
    goods_idx
        Boolean mask of the channels used as predictors.
    bads_idx
        Boolean mask of the interpolated channels.
    interpolation
        Interpolation matrix mapping good to bad channels.
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

    # test spherical fit (linear least-squares sphere fit)
    a = np.c_[2 * pos_good, np.ones(len(pos_good))]
    b = (pos_good ** 2).sum(1)
    x = np.linalg.lstsq(a, b, rcond=1e-6)[0]
    center = x[:3]
    radius = np.sqrt((x[:3] ** 2).sum() + x[3])
    distance = np.sqrt(np.sum((pos_good - center) ** 2, 1))
    distance = np.mean(distance / radius)
    if np.abs(1. - distance) > 0.1:
        logger.warning('Your spherical fit is poor, interpolation results are likely to be inaccurate.')

    logger.info(f'Computing interpolation matrix from {len(pos_good)} sensor positions')

    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    return goods_idx, bads_idx, interpolation


def _interpolate_bads_eeg(
        epochs: mne.Epochs,
        bad_channels_by_epoch: list[list[str]],
) -> None:
    """Interpolate bad channels per epoch

    Parameters
    ----------
    epochs
        The data to interpolate. Must be preloaded.
    bad_channels_by_epoch
        Bad channel names specified for each epoch. For example, for an Epochs
        instance containing 3 epochs: ``[['F1'], [], ['F3', 'FZ']]``
    """
    logger = logging.getLogger(__name__)

    if len(bad_channels_by_epoch) != len(epochs):
        raise ValueError(f"Unequal length of epochs ({len(epochs)}) and bad_channels_by_epoch ({len(bad_channels_by_epoch)})")

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
        logger.info(f'Interpolating {bads_idx.sum()} sensors on epoch {i}')
        epochs._data[i, bads_idx, :] = np.dot(interpolation, epochs._data[i, goods_idx, :])


def _interpolate_bads_meg(
        epochs: mne.Epochs,
        bad_channels_by_epoch: list[list[str]],
        interp_cache: dict,
) -> None:
    """Interpolate bad MEG channels per epoch

    Parameters
    ----------
    epochs
        The data to interpolate. Must be preloaded.
    bad_channels_by_epoch
        Bad channel names specified for each epoch. For example, for an Epochs
        instance containing 3 epochs: ``[['F1'], [], ['F3', 'FZ']]``
    interp_cache
        Will be updated.

    Notes
    -----
    Based on mne 0.9.0 MEG channel interpolation.
    """
    logger = logging.getLogger(__name__)
    if len(bad_channels_by_epoch) != len(epochs):
        raise ValueError(f"Unequal length of epochs ({len(epochs)}) and bad_channels_by_epoch ({len(bad_channels_by_epoch)})")

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
        logger.info(f'Interpolating sensors {picks_bad} on epoch {i}')
        epochs._data[i, picks_bad, :] = interpolation.dot(epochs._data[i, picks_good, :])
    t2 = time.time()

    logger.debug(f"Interpolation took {t1 - t0}/{t2 - t1} seconds")


def make_interpolators(
        interp_cache: dict,
        keys: set[tuple[str, ...]],
        bads: tuple[str, ...],
        epochs: mne.Epochs,
) -> None:
    """Add MEG interpolators for ``keys`` to ``interp_cache`` (keyed by ``(bads, key)``).

    Parameters
    ----------
    interp_cache
        Cache of interpolators; will be updated in place.
    keys
        Bad-channel sets (sorted tuples) to create interpolators for.
    bads
        The recording's globally bad channels (part of the cache key).
    epochs
        The data providing the sensor geometry.
    """
    make = [k for k in keys if (bads, k) not in interp_cache]
    logger = logging.getLogger(__name__)
    logger.debug(f"Making {len(make)} of {len(keys)} interpolators")
    for key in make:
        picks_good = mne.pick_types(epochs.info, meg=True, ref_meg=False, exclude=key)
        picks_bad = mne.pick_channels(epochs.ch_names, key)
        interpolation = map_meg_channels(epochs, picks_good, picks_bad, 'accurate')
        interp_cache[bads, key] = picks_good, picks_bad, interpolation


def _window_intervals(
        windows: list[BadChannelWindow],
        epochs: mne.Epochs,
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Decompose one epoch's bad-channel windows into constant-bad-set intervals.

    Parameters
    ----------
    windows
        Bad-channel time windows for a single epoch (``channel``/``tmin``/``tmax``
        attributes; ``tmax`` exclusive, epoch-relative seconds).
    epochs
        The epochs the windows apply to (provides the time base).

    Returns
    -------
    intervals
        Maximal half-open sample ranges ``[a, b)`` paired with the sorted tuple
        of channels bad throughout that range (the interpolation cache key).
    """
    t0 = epochs.times[0]
    sfreq = epochs.info['sfreq']
    n_times = len(epochs.times)
    spans = []
    for w in windows:
        a = max(0, int(round((w.tmin - t0) * sfreq)))
        b = min(n_times, int(round((w.tmax - t0) * sfreq)))
        if b > a:
            spans.append((w.channel, a, b))
    boundaries = sorted({a for _, a, _ in spans} | {b for _, _, b in spans})
    intervals = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        active = tuple(sorted(ch for ch, sa, sb in spans if sa <= a and b <= sb))
        if active:
            intervals.append((a, b, active))
    return intervals


def _interpolate_bad_windows_eeg(
        epochs: mne.Epochs,
        windows_by_epoch: list[list[BadChannelWindow]],
        max_interpolate: int,
) -> None:
    """Interpolate bad EEG channels over the time window in which they are bad

    Like :func:`_interpolate_bads_eeg`, but each channel is only interpolated
    over the time window(s) in which it is bad rather than across the whole
    epoch.

    Parameters
    ----------
    epochs
        The data to interpolate. Must be preloaded.
    windows_by_epoch
        Bad-channel time windows specified for each epoch.
    max_interpolate
        Maximum number of channels to interpolate simultaneously. In a time
        interval where more channels than this are bad, all channels are set to
        0 instead of interpolated (too few good channels remain for a reliable
        interpolation).
    """
    if len(windows_by_epoch) != len(epochs):
        raise ValueError(f"Unequal length of epochs ({len(epochs)}) and windows_by_epoch ({len(windows_by_epoch)})")

    eeg_chs = {epochs.ch_names[p] for p in mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])}
    interp_cache = {}
    for i, windows in enumerate(windows_by_epoch):
        windows = [w for w in windows if w.channel in eeg_chs]
        for a, b, key in _window_intervals(windows, epochs):
            if len(key) > max_interpolate:
                # Too many bad channels to interpolate reliably: mask the interval.
                epochs._data[i, :, a:b] = 0
                continue
            if key in interp_cache:
                goods_idx, bads_idx, interpolation = interp_cache[key]
            else:
                goods_idx, bads_idx, interpolation = interp_cache[key] = _make_interpolator(epochs, key)
            epochs._data[i, bads_idx, a:b] = np.dot(interpolation, epochs._data[i, goods_idx, a:b])


def _interpolate_bad_windows_meg(
        epochs: mne.Epochs,
        windows_by_epoch: list[list[BadChannelWindow]],
        interp_cache: dict,
) -> None:
    """Interpolate bad MEG channels over the time window in which they are bad

    Like :func:`_interpolate_bads_meg`, but each channel is only interpolated
    over the time window(s) in which it is bad rather than across the whole
    epoch.

    Parameters
    ----------
    epochs
        The data to interpolate. Must be preloaded.
    windows_by_epoch
        Bad-channel time windows specified for each epoch.
    interp_cache
        Will be updated.
    """
    if len(windows_by_epoch) != len(epochs):
        raise ValueError(f"Unequal length of epochs ({len(epochs)}) and windows_by_epoch ({len(windows_by_epoch)})")

    # make sure the cache is based on the correct channels
    if 'ch_names' not in interp_cache or interp_cache['ch_names'] != epochs.ch_names:
        interp_cache.clear()
        interp_cache['ch_names'] = epochs.ch_names
    bads = tuple(sorted(epochs.info['bads']))
    meg_chs = {epochs.ch_names[p] for p in mne.pick_types(epochs.info, meg=True, ref_meg=False)}

    for i, windows in enumerate(windows_by_epoch):
        windows = [w for w in windows if w.channel in meg_chs]
        intervals = _window_intervals(windows, epochs)
        make_interpolators(interp_cache, {key for _, _, key in intervals}, bads, epochs)
        for a, b, key in intervals:
            picks_good, picks_bad, interpolation = interp_cache[bads, key]
            epochs._data[i, picks_bad, a:b] = interpolation.dot(epochs._data[i, picks_good, a:b])
