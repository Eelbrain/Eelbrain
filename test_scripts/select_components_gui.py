# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from eelbrain import *
import mne


data_dir = mne.datasets.sample.data_path()
PATH = 'test-ica.fif'

data = mne.io.read_raw_fif(data_dir / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif', preload=True)
data.set_eeg_reference()
data.pick_types(eeg=True)
# data.pick_types('mag')
data.filter(1, 40)
if os.path.exists(PATH):
    ica = mne.preprocessing.read_ica(PATH)
else:
    ica = mne.preprocessing.ICA()
    # ica = mne.preprocessing.ICA(0.95)
    # ica = mne.preprocessing.ICA(0.99)
    ica.fit(data)
    ica.save(PATH, overwrite=True)
g = gui.select_components(PATH, data)
