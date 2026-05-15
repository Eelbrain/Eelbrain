# Fixes for defective mne functions
from collections import defaultdict
import os.path as op
import packaging.version

import numpy as np

from mne.surface import read_surface
from mne.utils import get_subjects_dir, logger, verbose
from mne.label import _get_annot_fname, _n_colors, _write_annot
import nibabel

if packaging.version.parse(nibabel.__version__) == packaging.version.parse('2.3.0'):
    raise ImportError(f"The installed version of nibabel ({nibabel.__version__}) is defective (see https://github.com/nipy/nibabel/issues/649). Please install a newer version.")


@verbose
def write_labels_to_annot(labels, subject=None, parc=None, overwrite=False,
                          subjects_dir=None, annot_fname=None,
                          colormap='hsv', hemi='both', verbose=None):
    """Create a FreeSurfer annotation from a list of labels

    FIX: always write both hemispheres

    Parameters
    ----------
    labels : list with instances of mne.Label
        The labels to create a parcellation from.
    subject : str | None
        The subject for which to write the parcellation for.
    parc : str | None
        The parcellation name to use.
    overwrite : bool
        Overwrite files if they already exist.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    annot_fname : str | None
        Filename of the .annot file. If not None, only this file is written
        and 'parc' and 'subject' are ignored.
    colormap : str
        Colormap to use to generate label colors for labels that do not
        have a color specified.
    hemi : 'both' | 'lh' | 'rh'
        The hemisphere(s) for which to write *.annot files (only applies if
        annot_fname is not specified; default is 'both').
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Notes
    -----
    Vertices that are not covered by any of the labels are assigned to a label
    named "unknown".
    """
    subjects_dir = get_subjects_dir(subjects_dir)

    # get the .annot filenames and hemispheres
    annot_fname, hemis = _get_annot_fname(annot_fname, subject, hemi, parc,
                                          subjects_dir)

    if not overwrite:
        for fname in annot_fname:
            if op.exists(fname):
                raise ValueError('File %s exists. Use "overwrite=True" to '
                                 'overwrite it' % fname)

    # prepare container for data to save:
    to_save = []
    # keep track of issues found in the labels
    duplicate_colors = []
    invalid_colors = []
    overlap = []
    no_color = (-1, -1, -1, -1)
    no_color_rgb = (-1, -1, -1)
    for hemi, fname in zip(hemis, annot_fname):
        hemi_labels = [label for label in labels if label.hemi == hemi]
        n_hemi_labels = len(hemi_labels)

        if n_hemi_labels == 0:
            ctab = np.empty((0, 4), dtype=np.int32)
            ctab_rgb = ctab[:, :3]
        else:
            hemi_labels.sort(key=lambda label: label.name)

            # convert colors to 0-255 RGBA tuples
            hemi_colors = [no_color if label.color is None else
                           tuple(int(round(255 * i)) for i in label.color)
                           for label in hemi_labels]
            ctab = np.array(hemi_colors, dtype=np.int32)
            ctab_rgb = ctab[:, :3]

            # make color dict (for annot ID, only R, G and B count)
            labels_by_color = defaultdict(list)
            for label, color in zip(hemi_labels, ctab_rgb):
                labels_by_color[tuple(color)].append(label.name)

            # check label colors
            for color, names in labels_by_color.items():
                if color == no_color_rgb:
                    continue

                if color == (0, 0, 0):
                    # we cannot have an all-zero color, otherw. e.g. tksurfer
                    # refuses to read the parcellation
                    msg = ('At least one label contains a color with, "r=0, '
                           'g=0, b=0" value. Some FreeSurfer tools may fail '
                           'to read the parcellation')
                    logger.warning(msg)

                if any(i > 255 for i in color):
                    msg = f"{color}: {', '.join(names)} ({hemi})"
                    invalid_colors.append(msg)

                if len(names) > 1:
                    msg = f"{color}: {', '.join(names)} ({hemi})"
                    duplicate_colors.append(msg)

            # replace None values (labels with unspecified color)
            if labels_by_color[no_color_rgb]:
                default_colors = _n_colors(n_hemi_labels, bytes_=True,
                                           cmap=colormap)
                safe_color_i = 0  # keep track of colors known to be in hemi_colors
                for i in range(n_hemi_labels):
                    if ctab[i, 0] == -1:
                        color = default_colors[i]
                        # make sure to add no duplicate color
                        while np.any(np.all(color[:3] == ctab_rgb, 1)):
                            color = default_colors[safe_color_i]
                            safe_color_i += 1
                        # assign the color
                        ctab[i] = color

        # find number of vertices in surface
        if subject is not None and subjects_dir is not None:
            fpath = op.join(subjects_dir, subject, 'surf', f'{hemi}.white')
            points, _ = read_surface(fpath)
            n_vertices = len(points)
        else:
            if len(hemi_labels) > 0:
                max_vert = max(np.max(label.vertices) for label in hemi_labels)
                n_vertices = max_vert + 1
            else:
                n_vertices = 1
            msg = ('    Number of vertices in the surface could not be '
                   'verified because the surface file could not be found; '
                   'specify subject and subjects_dir parameters.')
            logger.warning(msg)

        # Create annot and color table array to write
        annot = np.empty(n_vertices, dtype=np.intp)
        annot[:] = -1
        # create the annotation ids from the colors
        annot_id_coding = np.array((1, 2 ** 8, 2 ** 16))
        annot_ids = list(np.sum(ctab_rgb * annot_id_coding, axis=1))
        for label, annot_id in zip(hemi_labels, annot_ids):
            # make sure the label is not overwriting another label
            if np.any(annot[label.vertices] != -1):
                other_ids = set(annot[label.vertices])
                other_ids.discard(-1)
                other_indices = (annot_ids.index(i) for i in other_ids)
                other_names = (hemi_labels[i].name for i in other_indices)
                other_repr = ', '.join(other_names)
                msg = f"{hemi}: {label.name} overlaps {other_repr}"
                overlap.append(msg)

            annot[label.vertices] = annot_id

        hemi_names = [label.name for label in hemi_labels]

        # Assign unlabeled vertices to an "unknown" label
        unlabeled = (annot == -1)
        if np.any(unlabeled):
            msg = ("Assigning %i unlabeled vertices to "
                   "'unknown-%s'" % (unlabeled.sum(), hemi))
            logger.info(msg)

            # find an unused color (try shades of gray first)
            for i in range(1, 257):
                if not np.any(np.all((i, i, i) == ctab_rgb, 1)):
                    break
            if i < 256:
                color = (i, i, i, 0)
            else:
                err = ("Need one free shade of gray for 'unknown' label. "
                       "Please modify your label colors, or assign the "
                       "unlabeled vertices to another label.")
                raise ValueError(err)

            # find the id
            annot_id = np.sum(annot_id_coding * color[:3])

            # update data to write
            annot[unlabeled] = annot_id
            ctab = np.vstack((ctab, color))
            hemi_names.append("unknown")

        # convert to FreeSurfer alpha values
        ctab[:, 3] = 255 - ctab[:, 3]

        # remove hemi ending in names
        hemi_names = [name[:-3] if name.endswith(hemi) else name
                      for name in hemi_names]

        to_save.append((fname, annot, ctab, hemi_names))

    issues = []
    if duplicate_colors:
        msg = ("Some labels have the same color values (all labels in one "
               "hemisphere must have a unique color):")
        duplicate_colors.insert(0, msg)
        issues.append('\n'.join(duplicate_colors))
    if invalid_colors:
        msg = ("Some labels have invalid color values (all colors should be "
               "RGBA tuples with values between 0 and 1)")
        invalid_colors.insert(0, msg)
        issues.append('\n'.join(invalid_colors))
    if overlap:
        msg = ("Some labels occupy vertices that are also occupied by one or "
               "more other labels. Each vertex can only be occupied by a "
               "single label in *.annot files.")
        overlap.insert(0, msg)
        issues.append('\n'.join(overlap))

    if issues:
        raise ValueError('\n\n'.join(issues))

    # write it
    for fname, annot, ctab, hemi_names in to_save:
        logger.info(f'   writing {len(hemi_names)} labels to {fname}')
        _write_annot(fname, annot, ctab, hemi_names)

    logger.info('[done]')
