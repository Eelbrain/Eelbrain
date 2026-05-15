"Utilities for MNE data processing"
import os
from pathlib import Path

import mne
from mne.label import _get_annot_fname
from mne.utils import get_subjects_dir
from nibabel.freesurfer import read_annot, write_annot


def fix_annot_names(subject, parc, clean_subject=None, clean_parc=None,
                    hemi='both', subjects_dir=None):
    """Fix for Freesurfer's mri_surf2surf corrupting label names in annot files

    Notes
    -----
    Requires nibabel > 1.3.0 for annot file I/O
    """
    # process args
    subjects_dir = get_subjects_dir(subjects_dir)
    if clean_subject is None:
        clean_subject = subject
    if clean_parc is None:
        clean_parc = parc

    fpaths, hemis = _get_annot_fname(None, subject, hemi, parc, subjects_dir)
    clean_fpaths, _ = _get_annot_fname(None, clean_subject, hemi, clean_parc,
                                       subjects_dir)

    for fpath, clean_fpath, hemi in zip(fpaths, clean_fpaths, hemis):
        labels, ctab, names = read_annot(fpath)
        _, _, clean_names = read_annot(clean_fpath)
        if all(n == nc for n, nc in zip(names, clean_names)):
            continue

        if len(clean_names) != len(names):
            raise ValueError(f"Different names in {hemi} annot files: {names} vs. {clean_names}")

        for clean_name, name in zip(clean_names, names):
            if not name.startswith(clean_name):
                raise ValueError(f"{name} does not start with {clean_name}")

        write_annot(fpath, labels, ctab, clean_names)


def is_fake_mri(mri_dir):
    """Check whether a directory is a fake MRI subject directory

    Parameters
    ----------
    mri_dir : str(path)
        Path to a directory.

    Returns
    -------
    True is `mri_dir` is a fake MRI directory.

    Notes
    -----
    Based entirely on the presence of the `MRI scaling parameters.cfg` file.
    """
    fname = os.path.join(mri_dir, 'MRI scaling parameters.cfg')
    return os.path.exists(fname)


def get_volume_source_space_labels():
    # see mne.source_space._get_lut
    path = Path(mne.__file__).parent / 'data' / 'FreeSurferColorLUT.txt'
    text = path.read_text()
    lines = (line for line in text.splitlines() if line and not line.startswith('#'))
    data = (line.split(maxsplit=2) for line in lines)
    return {int(id_): label for id_, label, _ in data}
