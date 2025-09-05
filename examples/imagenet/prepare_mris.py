# skip test: data unavailable
from os import makedirs, mkdir
from pathlib import Path
from posixpath import dirname
import mne

# Download the raw dataset at https://openneuro.org/datasets/ds005810/versions/1.0.6, then run this script.

MRI_SDIR = '/mnt/d/Data/ds005810/derivatives/freesurfer'
RAW_FILE = '/mnt/d/Data/ds005810/sub-{subject}/ses-ImageNet01/meg/sub-{subject}_ses-ImageNet01_task-ImageNet_run-01_meg.fif'
TRANS_FILE = '/mnt/d/Data/ds005810/derivatives/trans/{subject}_trans.fif'
subjects = ['01']

makedirs(dirname(TRANS_FILE), exist_ok=True)
for subject in subjects:
    raw = mne.io.read_raw_fif(RAW_FILE.format(subject=subject), preload=False)
    coreg = mne.coreg.Coregistration(raw.info, subject=subject, subjects_dir=MRI_SDIR)
    coreg.fit_fiducials()
    coreg.fit_icp()
    mne.write_trans(TRANS_FILE.format(subject=subject), coreg.trans, overwrite=True)
