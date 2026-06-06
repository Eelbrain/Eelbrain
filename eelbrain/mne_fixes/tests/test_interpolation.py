# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import mne
from mne.channels.interpolation import _make_interpolation_matrix as mne_make_interpolation_matrix

from eelbrain import datasets
from eelbrain.mne_fixes import _interpolate_bads_eeg, _interpolate_bads_meg
from eelbrain.mne_fixes._interpolation import _make_interpolation_matrix
from eelbrain.testing import requires_mne_sample_data


@requires_mne_sample_data
def test_interpolation():
    "Test MNE channel interpolation by epoch"
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
    "Test EEG spherical-spline interpolation by epoch"
    # vendored kernel matches MNE-Python's implementation
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((20, 3)) + [0, 0, 0.05]
    assert_array_equal(
        _make_interpolation_matrix(pos[:15], pos[15:]),
        mne_make_interpolation_matrix(pos[:15].copy(), pos[15:].copy()))

    # per-epoch interpolation only touches each epoch's bad channels
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
