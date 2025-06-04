# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Functions from MNE-Python that have been removed from public API
"""
import numpy as np


##############################################################################
# formerly in mne.source_space

_src_kind_dict = {
    "vol": "volume",
    "surf": "surface",
    "discrete": "discrete",
}


def label_src_vertno_sel(label, src):
    """Find vertex numbers and indices from label.

    Parameters
    ----------
    label : Label
        Source space label.
    src : dict
        Source space.

    Returns
    -------
    vertices : list of length 2
        Vertex numbers for lh and rh.
    src_sel : array of int (len(idx) = len(vertices[0]) + len(vertices[1]))
        Indices of the selected vertices in sourse space.
    """
    if src[0]["type"] != "surf":
        raise ValueError(
            "Labels are only supported with surface source spaces, "
            f"got {_src_kind_dict[src[0]['type']]} source space"
        )

    vertno = [src[0]["vertno"], src[1]["vertno"]]

    if label.hemi == "lh":
        vertno_sel = np.intersect1d(vertno[0], label.vertices)
        src_sel = np.searchsorted(vertno[0], vertno_sel)
        vertno[0] = vertno_sel
        vertno[1] = np.array([], int)
    elif label.hemi == "rh":
        vertno_sel = np.intersect1d(vertno[1], label.vertices)
        src_sel = np.searchsorted(vertno[1], vertno_sel) + len(vertno[0])
        vertno[0] = np.array([], int)
        vertno[1] = vertno_sel
    elif label.hemi == "both":
        vertno_sel_lh = np.intersect1d(vertno[0], label.lh.vertices)
        src_sel_lh = np.searchsorted(vertno[0], vertno_sel_lh)
        vertno_sel_rh = np.intersect1d(vertno[1], label.rh.vertices)
        src_sel_rh = np.searchsorted(vertno[1], vertno_sel_rh) + len(vertno[0])
        src_sel = np.hstack((src_sel_lh, src_sel_rh))
        vertno = [vertno_sel_lh, vertno_sel_rh]
    else:
        raise Exception("Unknown hemisphere type")

    return vertno, src_sel
