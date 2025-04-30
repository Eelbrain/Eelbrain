from copy import deepcopy
import os
import re
from typing import Sequence, Union

import mne

from .._mne import combination_label, labels_from_mni_coords, rename_label, dissolve_label
from .definitions import DefinitionError, Definition, tuple_arg


SEEDED_PARC_RE = re.compile(r'^(.+)-(\d+)$')


class Parcellation(Definition):
    DICT_ATTRS = ('kind',)
    kind = None  # used when comparing dict representations
    morph_from_fsaverage = False

    def __init__(
            self,
            views: Union[str, Sequence[str]] = None,
    ):
        self.views = views

    def _link(self, name):
        out = deepcopy(self)
        out.name = name
        return out

    def _make(
            self,
            e,  # the MneExperiment instance
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
    MneExperiment.parcs

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
            views: Union[str, Sequence[str]] = None,
    ):
        Parcellation.__init__(self, views)
        self.base = base
        self.labels = tuple_arg('labels', labels)

    def _make(self, e, parc):
        with e._temporary_state:
            base = {l.name: l for l in e.load_annot(parc=self.base)}
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
    MneExperiment.parcs

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
            views: Union[str, Sequence[str]] = None,
    ):
        Parcellation.__init__(self, views)
        self.base = base
        self.labels = labels

    def _make(self, e, parc):
        with e._temporary_state:
            base = {l.name: l for l in e.load_annot(parc=self.base)}
        subjects_dir = e.get('mri-sdir')
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

    def __init__(
            self,
            morph_from_fsaverage: bool,
            views: Union[str, Sequence[str]] = None,
    ):
        Parcellation.__init__(self, views)
        self.morph_from_fsaverage = morph_from_fsaverage

    def _make(self, e, parc):
        assert parc == 'lobes'
        subject = e.get('mrisubject')
        subjects_dir = e.get('mri-sdir')
        if subject != 'fsaverage':
            raise RuntimeError(f"lobes parcellation can only be created for fsaverage, not for {subject}")

        # load source annot
        with e._temporary_state:
            labels = e.load_annot(parc='PALS_B12_Lobes')

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
    MneExperiment.parcs

    Examples
    --------
    Predefined parcellations::

        parcs = {
            'aparc': FreeSurferParc(),
            }
    """
    kind = 'subject_parc'

    def _make(self, e, parc):
        subject = e.get('mrisubject')
        raise FileNotFoundError(f"At least one annot file for the parcellation {parc} is missing for {subject}")


class FSAverageParc(Parcellation):
    """Fsaverage parcellation that is morphed to individual subjects

    Parcs that are defined for the fsaverage brain and should be morphed
    to every other subject's brain. These parcellations are automatically
    morphed to individual subjects' MRIs.

    See Also
    --------
    MneExperiment.parcs

    Examples
    --------
    Predefined parcellations::

        parcs = {
            'PALS_B12_Brodmann': FSAverageParc(),
            }
    """
    kind = 'fsaverage_parc'
    morph_from_fsaverage = True

    def _make(self, e, parc):
        common_brain = e.get('common_brain')
        assert e.get('mrisubject') == common_brain
        raise FileNotFoundError(f"At least one annot file for the parcellation {parc} is missing for {common_brain}")


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
            views: Union[str, Sequence[str]] = None,
    ):
        Parcellation.__init__(self, views)
        self.labels = tuple_arg('labels', labels)

    def _make(self, e, parc):
        labels = []
        hemis = ('lh.', 'rh.')
        path = os.path.join(e.get('mri-dir'), 'label', '%s.label')
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
    MneExperiment.parcs

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

    def seeds_for_subject(self, subject):
        return self.seeds

    def _make(self, e, parc):
        if self.mask:
            with e._temporary_state:
                e.make_annot(parc=self.mask)
        subject = e.get('mrisubject')
        subjects_dir = e.get('mri-sdir')
        seeds = self.seeds_for_subject(subject)
        name, extent = SEEDED_PARC_RE.match(parc).groups()
        return labels_from_mni_coords(seeds, float(extent), subject, self.surface, self.mask, subjects_dir, parc)


class IndividualSeededParc(SeededParc):
    """Seed parcellation with individual seeds for each subject

    Analogous to :class:`SeededParc`, except that seeds are
    provided for each subject.

    See Also
    --------
    MneExperiment.parcs

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
            raise DefinitionError("Some labels are missing subjects")
        self.subjects = subjects

    def seeds_for_subject(self, subject):
        if subject not in self.subjects:
            raise DefinitionError(f"Parcellation {self.name} not defined for subject {subject}")
        seeds = {name: self.seeds[name][subject] for name in self.seeds}
        # filter out missing
        return {name: seed for name, seed in seeds.items() if seed}


class VolumeParc(Parcellation):
    "Assume it exists"
    kind = 'volume'


def parc_from_dict(name, params):
    p = params.copy()
    kind = p.pop('kind', None)
    if kind is None:
        raise KeyError(f"Parcellation {name} does not contain the required 'kind' entry")
    elif kind not in PARC_CLASSES:
        raise ValueError(f"Parcellation {name} contains an invalid 'kind' entry: {kind!r}")
    cls = PARC_CLASSES[kind]
    return cls(**p)


PARC_CLASSES = {p.kind: p for p in (CombinationParc, FreeSurferParc, FSAverageParc, SeededParc, IndividualSeededParc)}


def assemble_parcs(items):
    parcs = {}
    for name, obj in items:
        if isinstance(obj, Parcellation):
            parc = obj
        elif isinstance(obj, dict):
            parc = parc_from_dict(name, obj)
        elif obj == FreeSurferParc.kind:
            parc = FreeSurferParc(('lateral', 'medial'))
        elif obj == FSAverageParc.kind:
            parc = FSAverageParc(('lateral', 'medial'))
        else:
            raise DefinitionError(f"parcellation {name!r}: {obj!r}")
        parcs[name] = parc._link(name)
    return parcs
