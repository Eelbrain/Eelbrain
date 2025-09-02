import os
import mne

MRI_SDIR = '/mnt/d/Data/Somatomotor/derivatives/freesurfer'
RAW_FILE = '/mnt/d/Data/Somatomotor/sub-{subject}/ses-meeg/meg/sub-{subject}_ses-meeg_task-somatomotor_run-1_meg.fif'
TRANS_FILE = '/mnt/d/Data/Somatomotor/derivatives/trans/{subject}_trans.fif'
subjects = ['sm04', 'sm06', 'sm07', 'sm09', 'sm12']

mne.datasets.fetch_fsaverage(subjects_dir=MRI_SDIR)
for subject in subjects:
    mne.scale_mri('fsaverage', subject, 1., subjects_dir=MRI_SDIR, labels=False)
    raw = mne.io.read_raw_fif(RAW_FILE.format(subject=subject), preload=False)
    coreg = mne.coreg.Coregistration(raw.info, subject=subject, subjects_dir=MRI_SDIR)
    coreg.fit_fiducials()
    coreg.fit_icp()
    mne.write_trans(TRANS_FILE.format(subject=subject), coreg.trans, overwrite=True)
