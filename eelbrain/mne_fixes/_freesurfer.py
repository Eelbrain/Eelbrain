# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os


def rename_mri(subject_from, subject_to, subjects_dir, preview=False):
    """Rename an MRI subject

    Parameters
    ----------
    subject_from : str
        Name of the subject that should be renamed.
    subject_to : str
        Name of the new subject name.
    subjects_dir : str
        Subjects-directory.
    preview : bool
        Show all the files that would be renamed instead of actually renaming
        them.

    Notes
    -----
    Does not update the ``MRI scaling parameters.cfg`` file for scaled MRIs.
    """
    sdir = os.path.join(subjects_dir, subject_from)
    if not os.path.exists(sdir):
        raise OSError(f"Directory for {subject_from} does not exist at {subjects_dir}")

    n_sdir = len(subjects_dir) + 1
    n_from = len(subject_from)

    pairs = []
    for root, dirs, files in os.walk(sdir):
        for filename in files:
            if filename.startswith(subject_from):
                old = os.path.join(root, filename)
                new = os.path.join(root, subject_to + filename[n_from:])
                pairs.append((old, new))
    pairs.append((sdir, os.path.join(subjects_dir, subject_to)))

    if preview:
        for old, new in pairs:
            print(f"  {old[n_sdir:]}\n->{new[n_sdir:]}")
    else:
        for old, new in pairs:
            os.rename(old, new)
