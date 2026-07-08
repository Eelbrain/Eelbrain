# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for per-epoch and time-windowed bad-channel interpolation."""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import mne

from eelbrain import datasets
from eelbrain._meeg import BadChannelWindow
from eelbrain._meeg.interpolation import _interpolate_bads_eeg, _interpolate_bads_meg, _interpolate_bad_windows_eeg
from eelbrain.testing import requires_mne_sample_data


@requires_mne_sample_data
def test_interpolation_meg():
    "Per-epoch MEG channel interpolation"
    ds = datasets.get_mne_sample(sub=[0, 1, 2, 3])
    bads1 = ['MEG 0531', 'MEG 1321']
    bads3 = ['MEG 0531', 'MEG 2231']
    bads_list = [[], bads1, [], bads3]
    test_epochs = ds['epochs']
    index_0531 = test_epochs.ch_names.index('MEG 0531')
    test_epochs._data[1, index_0531] = 0
    epochs1 = test_epochs.copy()
    epochs3 = test_epochs.copy()

    _interpolate_bads_meg(test_epochs, bads_list, {})
    assert_array_equal(test_epochs._data[0], epochs1._data[0])
    assert_array_equal(test_epochs._data[2], epochs1._data[2])
    epochs1.info['bads'] = bads1
    epochs1.interpolate_bads(mode='accurate', origin='auto')
    assert_array_almost_equal(test_epochs._data[1], epochs1._data[1], 25)
    epochs3.info['bads'] = bads3
    epochs3.interpolate_bads(mode='accurate', origin='auto')
    assert_array_almost_equal(test_epochs._data[3], epochs3._data[3], 25)


def test_interpolation_eeg():
    "Per-epoch EEG spherical-spline interpolation only touches each epoch's bads"
    rng = np.random.default_rng(0)
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    info = mne.create_info(ch_names, 100., 'eeg')
    info.set_montage(montage)
    epochs = mne.EpochsArray(rng.standard_normal((3, len(ch_names), 50)), info, verbose='error')
    original = epochs._data.copy()
    i_c3 = ch_names.index('C3')
    i_f3 = ch_names.index('F3')
    i_f4 = ch_names.index('F4')

    _interpolate_bads_eeg(epochs, [['C3'], [], ['F3', 'F4']])
    # epoch 0: only C3 changed
    assert not np.allclose(original[0, i_c3], epochs._data[0, i_c3])
    assert_array_equal(np.delete(original[0], i_c3, 0), np.delete(epochs._data[0], i_c3, 0))
    # epoch 1: no bad channels, unchanged
    assert_array_equal(original[1], epochs._data[1])
    # epoch 2: only F3 and F4 changed
    assert not np.allclose(original[2, i_f3], epochs._data[2, i_f3])
    assert not np.allclose(original[2, i_f4], epochs._data[2, i_f4])
    assert_array_equal(np.delete(original[2], [i_f3, i_f4], 0), np.delete(epochs._data[2], [i_f3, i_f4], 0))


def test_interpolate_bad_windows_eeg():
    "Windowed EEG interpolation touches only the window and matches whole-epoch"
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz']
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names, 100., 'eeg')
    info.set_montage(montage)
    rng = np.random.RandomState(0)
    n_ep, n_t = 3, 50
    data = rng.randn(n_ep, len(ch_names), n_t) * 1e-6
    epochs = mne.EpochsArray(data.copy(), info, verbose='error')

    # Cz bad over samples 10:30 in epoch 1
    windows = [[], [BadChannelWindow('Cz', 0.10, 0.30)], []]
    windowed = epochs.copy()
    _interpolate_bad_windows_eeg(windowed, windows, max_interpolate=5)

    # reference: whole-epoch interpolation of Cz on epoch 1
    reference = epochs.copy()
    _interpolate_bads_eeg(reference, [[], ['Cz'], []])

    ci = ch_names.index('Cz')
    d0 = epochs.get_data(copy=True)
    dw = windowed.get_data(copy=True)
    dr = reference.get_data(copy=True)
    # inside the window matches whole-epoch interpolation
    assert np.allclose(dw[1, ci, 10:30], dr[1, ci, 10:30])
    # outside the window unchanged
    assert np.array_equal(dw[1, ci, :10], d0[1, ci, :10])
    assert np.array_equal(dw[1, ci, 30:], d0[1, ci, 30:])
    # other channels and epochs untouched
    assert np.array_equal(np.delete(dw[1], ci, 0), np.delete(d0[1], ci, 0))
    assert np.array_equal(dw[0], d0[0])
    assert np.array_equal(dw[2], d0[2])


def test_interpolate_bad_windows_eeg_zeroes_when_too_many_bad():
    "Windowed EEG interpolation zeroes intervals when more than max_interpolate channels are bad"
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz']
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names, 100., 'eeg')
    info.set_montage(montage)
    rng = np.random.RandomState(0)
    data = rng.randn(1, len(ch_names), 50) * 1e-6
    epochs = mne.EpochsArray(data.copy(), info, verbose='error')

    # 3 channels bad over samples 10:30, but max_interpolate=2 -> zero the interval instead
    bad = ['F3', 'C3', 'P3']
    windows = [[BadChannelWindow(ch, 0.10, 0.30) for ch in bad]]
    _interpolate_bad_windows_eeg(epochs, windows, max_interpolate=2)

    d = epochs.get_data(copy=True)
    # the whole interval is set to 0
    assert np.array_equal(d[0, :, 10:30], np.zeros((len(ch_names), 20)))
    # outside the interval, data are unchanged
    assert np.array_equal(d[0, :, :10], data[0, :, :10])
    assert np.array_equal(d[0, :, 30:], data[0, :, 30:])
