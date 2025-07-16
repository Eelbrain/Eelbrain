# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join
from warnings import catch_warnings, filterwarnings

import mne
from eelbrain import gui, load
from eelbrain.testing import gui_test, TempDir
from eelbrain._wxgui import ID


@gui_test
def test_select_components():
    "Test Select-Epochs GUI Document"
    tempdir = TempDir()
    path = join(tempdir, 'test-ica.fif')

    data_path = mne.datasets.testing.data_path()
    raw_path = join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
    raw = mne.io.Raw(raw_path, preload=True)
    ds = load.mne.events(raw, stim_channel='STI 014')
    ds['epochs'] = load.mne.mne_epochs(ds, tmax=0.1)
    ica = mne.preprocessing.ICA(0.95, max_iter=1)
    with catch_warnings():
        filterwarnings('ignore', 'FastICA did not converge')
        ica.fit(raw)
    ica.save(path)

    frame = gui.select_components(path, ds)
    frame.model.toggle(1)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(path)
    assert ica.exclude == [1]

    frame.OnUndo(None)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(path)
    assert ica.exclude == []

    # plotting
    for i in [ID.BASELINE_NONE, ID.BASELINE_GLOABL_MEAN, ID.BASELINE_CUSTOM]:
        frame.butterfly_baseline = i
        frame.OnPlotGrandAverage(None)

    frame.Close()
