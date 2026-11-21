# skip test: data unavailable
from os import makedirs
from pathlib import Path
from posixpath import dirname
import mne

# Download the raw dataset at https://openneuro.org/datasets/ds006035/versions/1.0.0, then run this script.

MRI_SDIR = '/mnt/d/Data/Somatomotor/derivatives/freesurfer'
RAW_FILE = '/mnt/d/Data/Somatomotor/sub-{subject}/ses-meeg/meg/sub-{subject}_ses-meeg_task-somatomotor_run-1_meg.fif'
TRANS_FILE = '/mnt/d/Data/Somatomotor/derivatives/mne/sub-{subject}/ses-meeg/meg/sub-{subject}_ses-meeg_trans.fif'  # named after the MRI subject (sub-{subject} by default)
subjects = ['sm04', 'sm06', 'sm07', 'sm09', 'sm12']

mne.datasets.fetch_fsaverage(subjects_dir=Path(MRI_SDIR))
for subject in subjects:
    mrisubject = f'sub-{subject}'
    mne.scale_mri('fsaverage', mrisubject, 1., subjects_dir=MRI_SDIR, labels=False)
    raw = mne.io.read_raw_fif(RAW_FILE.format(subject=subject), preload=False)
    coreg = mne.coreg.Coregistration(raw.info, subject=mrisubject, subjects_dir=MRI_SDIR)
    coreg.fit_fiducials()
    coreg.fit_icp()
    trans_file = TRANS_FILE.format(subject=subject)
    makedirs(dirname(trans_file), exist_ok=True)
    mne.write_trans(trans_file, coreg.trans, overwrite=True)
