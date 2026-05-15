# skip test: data unavailable
from os import makedirs
from posixpath import dirname
import mne


MRI_SDIR = '/mnt/d/Data/ds005810/derivatives/freesurfer'
RAW_FILE = '/mnt/d/Data/ds005810/sub-{subject}/ses-ImageNet01/meg/sub-{subject}_ses-{session}_task-ImageNet_run-01_meg.fif'
TRANS_FILE = '/mnt/d/Data/ds005810/derivatives/trans/sub-{subject}_ses-{session}_trans.fif'
subjects = ['01', '03', '04']
sessions = ['ImageNet01']


makedirs(dirname(TRANS_FILE), exist_ok=True)
for subject in subjects:
    for session in sessions:
        raw = mne.io.read_raw_fif(RAW_FILE.format(subject=subject, session=session), preload=False)
        coreg = mne.coreg.Coregistration(raw.info, subject=subject, subjects_dir=MRI_SDIR)
        coreg.fit_fiducials()
        coreg.fit_icp()
        mne.write_trans(TRANS_FILE.format(subject=subject, session=session), coreg.trans, overwrite=True)
