# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Parcellation configurations.

Each parcellation type is a :class:`Parcellation` (a
:class:`~configuration.Configuration`) subclass that the user attaches to
:class:`~pipeline.Pipeline` by name. The graph node that builds or loads the
corresponding ``*.annot`` files lives in :mod:`._experiment.parc.nodes`.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING
from collections.abc import Sequence

import mne

from ..._mne import combination_label, labels_from_mni_coords, rename_label, dissolve_label
from ..pathing import MRI_SDIR, mri_dir
from ..configuration import Configuration, ConfigurationError, sequence_arg

if TYPE_CHECKING:
    from ..derivative_cache import Request
    from .nodes import AnnotDerivative


SEEDED_PARC_RE = re.compile(r'^(.+)-(\d+)$')


def _resolve_parc(parcs: dict[str, Parcellation], parc: str) -> tuple[str, Parcellation | None]:
    if parc == '':
        return '', None
    if parc in parcs:
        return parc, parcs[parc]
    match = SEEDED_PARC_RE.match(parc)
    if match is None:
        raise ValueError(f"{parc=}: unknown parcellation")
    name = match.group(1)
    resolved = parcs.get(name)
    if not isinstance(resolved, SeededParc):
        raise ValueError(f"{parc=}: unknown parcellation")
    return parc, resolved


class Parcellation(Configuration):
    DICT_ATTRS = ('kind',)
    kind = None  # used when comparing dict representations
    morph_from_fsaverage = False

    def __init__(
            self,
            views: str | Sequence[str] = None,
    ):
        self.views = views

    def _make(
            self,
            ctx: Request,
            annot: AnnotDerivative,
            parc: str,  # the name (contains radius for seeded parcellations)
    ) -> list:
        raise RuntimeError(f"Trying to make {self.__class__.__name__}")


class SubParc(Parcellation):
    """A subset of labels in another parcellation

    Parameters
    ----------
    base
        The name of the parcellation that provides the input labels. A common
        ``base`` is the ``'aparc'`` parcellation [1]_.
    labels
        Labels to copy from ``base``. In order to include a label in both
        hemispheres, omit the ``*-hemi`` tag. For example, with
        ``base='aparc'``, ``labels=('transversetemporal',)`` would include the
        transverse temporal gyrus in both hemisphere, whereas
        ``labels=('transversetemporal-lh',)`` would include the transverse
        temporal gyrus of only the left hemisphere.
    views
        Views shown in anatomical plots, e.g. ``("medial", "lateral")``.

    See Also
    --------
    Pipeline.parcs

    Examples
    --------
    Masks for temporal and frontal lobes::

        parcs = {
            'STG': SubParc('aparc', ('transversetemporal', 'superiortemporal')),
            'IFG': SubParc('aparc', ('parsopercularis', 'parsorbitalis', 'parstriangularis')),
            'lateraltemporal': SubParc('aparc', (
                'transversetemporal', 'superiortemporal', 'bankssts',
                'middletemporal', 'inferiortemporal')),
        }

    References
    ----------
    .. [1] Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B.
           C., Blacker, D., … Killiany, R. J. (2006). An automated labeling system
           for subdividing the human cerebral cortex on MRI scans into gyral based
           regions of interest. NeuroImage, 31(3), 968–980.
           `10.1016/j.neuroimage.2006.01.021
           <https://surfer.nmr.mgh.harvard.edu/ftp/articles/desikan06-parcellation.pdf>`_
    """
    DICT_ATTRS = ('kind', 'base', 'labels')
    kind = 'combination'

    def __init__(
            self,
            base: str,
            labels: Sequence[str],
            views: str | Sequence[str] = None,
    ):
        Parcellation.__init__(self, views)
        self.base = base
        self.labels = sequence_arg('labels', labels)

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        base = {l.name: l for l in ctx.load('base')}
        hemis = ('-lh', '-rh')
        labels = []
        for label in self.labels:
            if label.endswith(hemis):
                labels.append(base[label])
            else:
                for hemi in hemis:
                    labels.append(base[label + hemi])
        return labels

    def _base_labels(self) -> set:
        return set(self.labels)


class CombinationParc(Parcellation):
    """Recombine labels from an existing parcellation

    Parameters
    ----------
    base
        The name of the parcellation that provides the input labels. A common
        ``base`` is the ``'aparc'`` parcellation [1]_.
    labels : dict  {str: str}
        New labels to create in ``{name: expression}`` format. All label names
        should be composed of alphanumeric characters (plus underline) and should
        not contain the -hemi tags. In order to create a given label only on one
        hemisphere, add the -hemi tag in the name (not in the expression, e.g.,
        ``{'occipitotemporal-lh': "occipital + temporal"}``).
    views
        Views shown in anatomical plots, e.g. ``("medial", "lateral")``.

    See Also
    --------
    Pipeline.parcs

    Examples
    --------
    These are pre-defined parcellations::

        parcs = {
            'lobes-op': CombinationParc('lobes', {'occipitoparietal': "occipital + parietal"}),
            'lobes-ot': CombinationParc('lobes', {'occipitotemporal': "occipital + temporal"}),
        }

    An example using a split label. In ``split(superiorfrontal, 3)[2]``, ``3``
    indicates a split into three parts, and the index ``[2]`` picks the last
    one. Label are split along their longest axis, and ordered posterior to
    anterior, so ``[2]`` picks the most anterior part of ``superiorfrontal``::

        parcs = {
            'medial': CombinationParc('aparc', {
                'medialparietal': 'precuneus + posteriorcingulate',
                'medialfrontal': 'medialorbitofrontal + rostralanteriorcingulate'
                                 ' + split(superiorfrontal, 3)[2]',
                }, views='medial'),
        }

    Posterior 2/3 of the combined superior temporal gyrus and Heschl's gyrus::

        parcs = {
            'STG301': CombinationParc('aparc', {'STG301': "split(transversetemporal + superiortemporal, 3)[:2]"}),
        }


    References
    ----------
    .. [1] Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B.
           C., Blacker, D., … Killiany, R. J. (2006). An automated labeling system
           for subdividing the human cerebral cortex on MRI scans into gyral based
           regions of interest. NeuroImage, 31(3), 968–980.
           `10.1016/j.neuroimage.2006.01.021
           <https://surfer.nmr.mgh.harvard.edu/ftp/articles/desikan06-parcellation.pdf>`_
    """
    DICT_ATTRS = ('kind', 'base', 'labels')
    kind = 'combination'

    def __init__(
            self,
            base: str,
            labels: dict,
            views: str | Sequence[str] = None,
    ):
        Parcellation.__init__(self, views)
        self.base = base
        self.labels = labels

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        base = {l.name: l for l in ctx.load('base')}
        subjects_dir = ctx.root / MRI_SDIR
        labels = []
        for name, exp in self.labels.items():
            labels += combination_label(name, exp, base, subjects_dir)
        return labels

    def _base_labels(self) -> set:
        base_labels = set()
        for name, exp in self.labels.items():
            exp_labels = re.findall(r'[^\W0-9]\w*', exp)
            base_labels.update(exp_labels)
        base_labels.remove('split')
        return base_labels


class EelbrainParc(Parcellation):
    "Parcellation that has special make rule"
    kind = 'eelbrain_parc'
    base = 'PALS_B12_Lobes'

    def __init__(
            self,
            morph_from_fsaverage: bool,
            views: str | Sequence[str] = None,
    ):
        Parcellation.__init__(self, views)
        self.morph_from_fsaverage = morph_from_fsaverage

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        assert parc == 'lobes'
        subject = ctx.state['mrisubject']
        subjects_dir = ctx.root / MRI_SDIR
        if subject != 'fsaverage':
            raise RuntimeError(f"lobes parcellation can only be created for fsaverage, not for {subject}")

        # load source annot
        labels = ctx.load('base')

        # sort labels
        labels = [l for l in labels if l.name[:-3] != 'MEDIAL.WALL']

        # rename good labels
        rename_label(labels, 'LOBE.FRONTAL', 'frontal')
        rename_label(labels, 'LOBE.OCCIPITAL', 'occipital')
        rename_label(labels, 'LOBE.PARIETAL', 'parietal')
        rename_label(labels, 'LOBE.TEMPORAL', 'temporal')

        # reassign unwanted labels
        targets = ('frontal', 'occipital', 'parietal', 'temporal')
        dissolve_label(labels, 'LOBE.LIMBIC', targets, subjects_dir)
        dissolve_label(labels, 'GYRUS', targets, subjects_dir, 'rh')
        dissolve_label(labels, '???', targets, subjects_dir)
        dissolve_label(labels, '????', targets, subjects_dir, 'rh')
        dissolve_label(labels, '???????', targets, subjects_dir, 'rh')

        return labels


class FreeSurferParc(Parcellation):
    """Parcellation that is created outside Eelbrain for each subject

    Parcs that can not be generated automatically (e.g.,
    parcellation that comes with FreeSurfer). These parcellations are
    automatically scaled for brains based on scaled versions of fsaverage, but
    for individual MRIs the user is responsible for creating the respective
    annot-files.

    See Also
    --------
    Pipeline.parcs

    Examples
    --------
    Predefined parcellations::

        parcs = {
            'aparc': FreeSurferParc(),
            }
    """
    kind = 'subject_parc'


class FSAverageParc(Parcellation):
    """Fsaverage parcellation that is morphed to individual subjects

    Parcs that are defined for the fsaverage brain and should be morphed
    to every other subject's brain. These parcellations are automatically
    morphed to individual subjects' MRIs.

    See Also
    --------
    Pipeline.parcs

    Examples
    --------
    Predefined parcellations::

        parcs = {
            'PALS_B12_Brodmann': FSAverageParc(),
            }
    """
    kind = 'fsaverage_parc'
    morph_from_fsaverage = True


class LabelParc(Parcellation):
    """Assemble parcellation from FreeSurfer labels

    Combine one or several ``*.label`` files into a parcellation.

    """
    DICT_ATTRS = ('kind', 'labels')
    kind = 'label_parc'
    make = True

    def __init__(
            self,
            labels: Sequence[str],
            views: str | Sequence[str] = None,
    ):
        Parcellation.__init__(self, views)
        self.labels = sequence_arg('labels', labels)

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        labels = []
        hemis = ('lh.', 'rh.')
        path = os.path.join(ctx.root / mri_dir(ctx.state), 'label', '%s.label')
        for label in self.labels:
            if label.startswith(hemis):
                labels.append(mne.read_label(path % label))
            else:
                labels.extend(mne.read_label(path % (hemi + label)) for hemi in hemis)
        return labels


class SeededParc(Parcellation):
    """Parcellation that is grown from seed coordinates

    Seeds are defined on fsaverage which is in MNI305 space (`FreeSurfer wiki
    <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_).
    For each seed entry, the source space vertex closest to the given coordinate
    will be used as actual seed, and a label will be created including all
    points with a surface distance smaller than a given extent from the seed
    vertex/vertices. The spatial extent is determined when setting the parc as
    analysis parameter as in ``e.set(parc="myparc-25")``, which specifies a
    radius of 25 mm.

    See Also
    --------
    Pipeline.parcs

    Parameters
    ----------
    seeds : dict
        ``{name: seed(s)}`` dictionary, where names are strings, including
        hemisphere tags (e.g., ``"mylabel-lh"``) and seed(s) are array-like,
        specifying one or more seed coordinate (shape ``(3,)`` or
        ``(n_seeds, 3)``).
    mask : str
        Name of a parcellation to use as mask (i.e., anything that is "unknown"
        in that parcellation is excluded from the new parcellation. For example,
        use ``{'mask': 'lobes'}`` to exclude the subcortical areas around the
        diencephalon.

    Examples
    --------
    Example with multiple seeds::

         parcs = {
             'stg': SeededParc({
                 'anteriorstg-lh': ((-54, 10, -8), (-47, 14, -28)),
                 'middlestg-lh': (-66, -24, 8),
                 'posteriorstg-lh': (-54, -57, 16),
             },
             mask='lobes'),
         }
    """
    DICT_ATTRS = ('kind', 'seeds', 'surface', 'mask')
    kind = 'seeded'
    make = True

    def __init__(self, seeds, mask=None, surface='white', views=None):
        Parcellation.__init__(self, views)
        self.seeds = seeds
        self.mask = mask
        self.surface = surface

    def _seeds_for_subject(self, subject):
        return self.seeds

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        if self.mask:
            ctx.load('mask')
        subject = ctx.state['mrisubject']
        subjects_dir = ctx.root / MRI_SDIR
        seeds = self._seeds_for_subject(subject)
        name, extent = SEEDED_PARC_RE.match(parc).groups()
        return labels_from_mni_coords(seeds, float(extent), subject, self.surface, self.mask, subjects_dir, parc)


class IndividualSeededParc(SeededParc):
    """Seed parcellation with individual seeds for each subject

    Analogous to :class:`SeededParc`, except that seeds are
    provided for each subject.

    See Also
    --------
    Pipeline.parcs

    Examples
    --------
    Parcellation with subject-specific seeds::

        parcs = {
            'stg': IndividualSeededParc({
                'anteriorstg-lh': {
                    'R0001': (-54, 10, -8),
                    'R0002': (-47, 14, -28),
                },
                'middlestg-lh': {
                    'R0001': (-66, -24, 8),
                    'R0002': (-60, -26, 9),
                }
                mask='lobes'),
        }
    """
    kind = 'individual seeded'
    morph_from_fsaverage = False

    def __init__(self, seeds, mask=None, surface='white', views=None):
        SeededParc.__init__(self, seeds, mask, surface, views)
        labels = tuple(self.seeds)
        label_subjects = {label: sorted(self.seeds[label].keys()) for label in labels}
        subjects = label_subjects[labels[0]]
        if not all(label_subjects[label] == subjects for label in labels[1:]):
            raise ConfigurationError("Some labels are missing subjects")
        self.subjects = subjects

    def _seeds_for_subject(self, subject):
        if subject not in self.subjects:
            raise ConfigurationError(f"Parcellation {self.name} not defined for subject {subject}")
        seeds = {name: self.seeds[name][subject] for name in self.seeds}
        # filter out missing
        return {name: seed for name, seed in seeds.items() if seed}


class VolumeParc(Parcellation):
    "Assume it exists"
    kind = 'volume'
