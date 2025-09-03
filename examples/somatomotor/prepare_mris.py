# skip test: data unavailable
from os import makedirs, mkdir
from pathlib import Path
from posixpath import dirname
import mne

# Download the raw dataset at https://openneuro.org/datasets/ds006035/versions/1.0.0, then run this script.

MRI_SDIR = '/mnt/d/Data/Somatomotor/derivatives/freesurfer'
RAW_FILE = '/mnt/d/Data/Somatomotor/sub-{subject}/ses-meeg/meg/sub-{subject}_ses-meeg_task-somatomotor_run-1_meg.fif'
TRANS_FILE = '/mnt/d/Data/Somatomotor/derivatives/trans/{subject}_trans.fif'
subjects = ['sm04', 'sm06', 'sm07', 'sm09', 'sm12']

mne.datasets.fetch_fsaverage(subjects_dir=Path(MRI_SDIR))
makedirs(dirname(TRANS_FILE), exist_ok=True)
for subject in subjects:
    mne.scale_mri('fsaverage', subject, 1., subjects_dir=MRI_SDIR, labels=False)
    raw = mne.io.read_raw_fif(RAW_FILE.format(subject=subject), preload=False)
    coreg = mne.coreg.Coregistration(raw.info, subject=subject, subjects_dir=MRI_SDIR)
    coreg.fit_fiducials()
    coreg.fit_icp()
    mne.write_trans(TRANS_FILE.format(subject=subject), coreg.trans, overwrite=True)
