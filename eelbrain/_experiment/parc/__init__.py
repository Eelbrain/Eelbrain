# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Parcellation configuration and the annotation derivative.

Parcellations define how the cortical surface is divided into regions of
interest. Each parcellation type is a :class:`Parcellation` (a
:class:`~configuration.Configuration`) subclass that the user attaches to
:class:`~pipeline.Pipeline` by name.

The supported parcellation types are:

:class:`FreeSurferParc`
    A named parcellation that already exists in the FreeSurfer subject's
    ``label/`` directory (e.g. ``'aparc'``).
:class:`FSAverageParc`
    A parcellation defined on the ``fsaverage`` surface and morphed to each
    individual subject.
:class:`EelbrainParc`
    A parcellation provided by Eelbrain (e.g. a functional atlas).
:class:`SeededParc` / :class:`IndividualSeededParc`
    A parcellation grown from a set of seed coordinates (MNI or subject-space).
:class:`CombinationParc`
    A parcellation derived from another by merging or renaming labels using a
    declarative expression language.
:class:`LabelParc`
    A parcellation defined directly from a list of :class:`mne.Label` objects.
:class:`VolumeParc`
    A volumetric parcellation for volume source spaces.

The configuration classes live in :mod:`._experiment.parc.config`; the shared
:class:`AnnotDerivative` graph node that produces or loads the ``*.annot`` files
lives in :mod:`._experiment.parc.nodes`.
"""

from .config import (
    SEEDED_PARC_RE,
    CombinationParc,
    EelbrainParc,
    FSAverageParc,
    FreeSurferParc,
    IndividualSeededParc,
    LabelParc,
    Parcellation,
    SeededParc,
    SubParc,
    VolumeParc,
    _resolve_parc,
)
from .nodes import AnnotDerivative
