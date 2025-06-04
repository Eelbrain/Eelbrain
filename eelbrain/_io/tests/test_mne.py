# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
from pathlib import Path

import mne
from mne import pick_types
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from eelbrain import datasets, load
from eelbrain.testing import assert_dataobj_equal, requires_mne_sample_data, requires_mne_testing_data, file_path


def test_load_events():
    # simulate raw
    data = np.random.random((2, 5000))
    samplingrate = 500
    info = mne.create_info(['Fp', 'Cz'], samplingrate, ['eeg', 'eeg'])
    raw = mne.io.RawArray(data, info)
    onset = [0.1, 1.2, 2.3, 4.63]
    labels = ['test1', 'test2', 'test1', 'test2']
    annotations = mne.Annotations(onset, 0.100, labels)
    raw.set_annotations(annotations)

    # test load events
    events = load.mne.events(raw)
    assert_array_equal(events['i_start'], [time * samplingrate for time in onset])
    assert_array_equal(events['event'], labels)


@requires_mne_testing_data
def test_load_fiff_ctf():
    path = Path(mne.datasets.testing.data_path())
    raw_path = path / 'CTF' / 'testdata_ctf.ds'
    raw = mne.io.read_raw_ctf(raw_path)
    y = load.mne.raw_ndvar(raw)
    assert_array_equal(y.sensor.adjacency()[:3], [[0, 1], [0, 16], [0, 44]])


@requires_mne_sample_data
def test_load_fiff_mne():
    data_path = mne.datasets.sample.data_path()
    fwd_path = os.path.join(data_path, 'MEG', 'sample', 'sample-ico-4-fwd.fif')
    evoked_path = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis-no-filter-ave.fif')
    cov_path = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
    mri_sdir = os.path.join(data_path, 'subjects')

    # make sure forward solution exists
    if not os.path.exists(fwd_path):
        datasets.get_mne_sample(src='ico')

    mne_evoked = mne.read_evokeds(evoked_path, 'Left Auditory')
    mne_fwd = mne.read_forward_solution(fwd_path)
    mne_fwd = mne.convert_forward_solution(mne_fwd, force_fixed=True, use_cps=True)
    cov = mne.read_cov(cov_path)

    picks = mne.pick_types(mne_evoked.info, 'mag')
    channels = [mne_evoked.ch_names[i] for i in picks]

    mne_evoked = mne_evoked.pick_channels(channels)
    mne_fwd = mne.pick_channels_forward(mne_fwd, channels)
    cov = mne.pick_channels_cov(cov, channels)

    mne_inv = mne.minimum_norm.make_inverse_operator(mne_evoked.info, mne_fwd, cov, 0, None, True)

    mne_stc = mne.minimum_norm.apply_inverse(mne_evoked, mne_inv, 1., 'MNE')

    meg = load.mne.evoked_ndvar(mne_evoked)
    inv = load.mne.inverse_operator(mne_inv, 'ico-4', mri_sdir)
    stc = inv.dot(meg)
    assert_array_almost_equal(stc.get_data(('source', 'time')), mne_stc.data)

    fwd = load.mne.forward_operator(mne_fwd, 'ico-4', mri_sdir)
    reconstruct = fwd.dot(stc)
    mne_reconstruct = mne.apply_forward(mne_fwd, mne_stc, mne_evoked.info)
    assert_array_almost_equal(reconstruct.get_data(('sensor', 'time')), mne_reconstruct.data)


@requires_mne_testing_data
def test_load_fiff_ndvar():
    data_path = Path(mne.datasets.testing.data_path())

    # raw_ndvar for
    raw = mne.io.read_raw_fif(data_path / 'MEG/sample/sample_audvis_trunc_raw.fif')
    # stim
    ndvar = load.fiff.raw_ndvar(raw, data='stim')
    data = raw.copy().pick('stim').get_data()
    assert_array_equal(ndvar.get_data(('sensor', 'time')), data)
    # EOG
    ndvar = load.fiff.raw_ndvar(raw, data='eog')
    data = raw.copy().pick('eog').get_data()
    assert_array_equal(ndvar.get_data(('sensor', 'time')), data)


def test_load_fiff_sensor():
    umd_sqd_path = file_path('test_umd-raw.sqd')
    raw = mne.io.read_raw_kit(umd_sqd_path)

    sensor = load.mne.sensor_dim(raw)
    assert sensor.sysname == 'KIT-UMD-3'


@requires_mne_sample_data
@pytest.mark.filterwarnings("ignore:The measurement information")
def test_load_fiff_from_raw():
    "Test loading data from a fiff raw file"
    data_path = mne.datasets.sample.data_path()
    meg_path = os.path.join(data_path, 'MEG', 'sample')
    raw_path = os.path.join(meg_path, 'sample_audvis_filt-0-40_raw.fif')
    evt_path = os.path.join(meg_path, 'sample_audvis_filt-0-40_raw-eve.fif')

    # load events
    ds = load.mne.events(raw_path, merge=-1, stim_channel='STI 014')
    assert ds['i_start'].x.dtype.kind == 'i'
    # compare with mne
    ds_evt = load.mne.events(events=evt_path)
    ds = ds[np.arange(ds.n_cases) != 289]  # mne is missing an event
    assert_dataobj_equal(ds, ds_evt, name=False)

    # add epochs as ndvar
    ds = ds.sub('trigger == 32')
    ds_ndvar = load.mne.add_epochs(ds, -0.1, 0.3, decim=10, data='mag', proj=False, reject=2e-12)
    meg = ds_ndvar['meg']
    assert meg.ndim == 3
    data = meg.get_data(('case', 'sensor', 'time'))

    # compare with mne epochs
    ds_mne = load.mne.add_mne_epochs(ds, -0.1, 0.3, decim=10, proj=False, reject={'mag': 2e-12})
    epochs = ds_mne['epochs']
    # events
    assert_array_equal(epochs.events[:, 1], 0)
    assert_array_equal(epochs.events[:, 2], 32)
    # data
    picks = pick_types(epochs.info, meg='mag')
    mne_data = epochs.get_data()[:, picks]
    assert_array_equal(meg.sensor.names, [epochs.info['ch_names'][i] for i in picks])
    assert_array_equal(data, mne_data)
    assert_array_almost_equal(meg.time, epochs.times)

    # with proj
    meg = load.mne.epochs(ds, -0.1, 0.3, decim=10, data='mag', proj=True, reject=2e-12)
    epochs = load.mne.mne_epochs(ds, -0.1, 0.3, decim=10, proj=True, reject={'mag': 2e-12})
    picks = pick_types(epochs.info, meg='mag')
    mne_data = epochs.get_data()[:, picks]
    assert_array_almost_equal(meg.x, mne_data, 10)
