# skip test: data unavailable
from os import makedirs
from posixpath import dirname
import mne


MRI_SDIR = '/mnt/d/Data/ds005810/derivatives/freesurfer'
RAW_FILE = '/mnt/d/Data/ds005810/sub-{subject}/ses-ImageNet01/meg/sub-{subject}_ses-{session}_task-ImageNet_run-01_meg.fif'
TRANS_FILE = '/mnt/d/Data/ds005810/derivatives/mne/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_trans.fif'  # named after the MRI subject (sub-{subject} by default)
subjects = ['01', '03', '04']
sessions = ['ImageNet01']


for subject in subjects:
    for session in sessions:
        raw = mne.io.read_raw_fif(RAW_FILE.format(subject=subject, session=session), preload=False)
        coreg = mne.coreg.Coregistration(raw.info, subject=subject, subjects_dir=MRI_SDIR)
        coreg.fit_fiducials()
        coreg.fit_icp()
        trans_file = TRANS_FILE.format(subject=subject, session=session)
        makedirs(dirname(trans_file), exist_ok=True)
        mne.write_trans(trans_file, coreg.trans, overwrite=True)
