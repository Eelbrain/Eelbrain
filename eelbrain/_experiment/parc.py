from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any
from collections.abc import Sequence

import mne

from .._mne import combination_label, labels_from_mni_coords, rename_label, dissolve_label
from ..mne_fixes import write_labels_to_annot
from .._utils import subp
from .._utils.mne_utils import fix_annot_names, is_fake_mri
from .pathing import annot_file_path, annot_stamp_path, label_dir, mri_dir, mri_sdir
from .derivative_cache import Dependency, Derivative, Request, file_fingerprint
from .configuration import Configuration, ConfigurationError, sequence_arg


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
        base = {l.name: l for l in annot.load_annot(ctx, parc=self.base)}
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
        base = {l.name: l for l in annot.load_annot(ctx, parc=self.base)}
        subjects_dir = mri_sdir(ctx.state)
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
            views: str | Sequence[str] = None,
    ):
        Parcellation.__init__(self, views)
        self.morph_from_fsaverage = morph_from_fsaverage

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        assert parc == 'lobes'
        subject = ctx.state['mrisubject']
        subjects_dir = mri_sdir(ctx.state)
        if subject != 'fsaverage':
            raise RuntimeError(f"lobes parcellation can only be created for fsaverage, not for {subject}")

        # load source annot
        labels = annot.load_annot(ctx, parc='PALS_B12_Lobes')

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

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        subject = ctx.state['mrisubject']
        raise FileNotFoundError(f"At least one annot file for the parcellation {parc} is missing for {subject}")


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

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        common_brain = ctx.state['common_brain']
        assert ctx.state['mrisubject'] == common_brain
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
            views: str | Sequence[str] = None,
    ):
        Parcellation.__init__(self, views)
        self.labels = sequence_arg('labels', labels)

    def _make(self, ctx: Request, annot: AnnotDerivative, parc: str):
        labels = []
        hemis = ('lh.', 'rh.')
        path = os.path.join(mri_dir(ctx.state), 'label', '%s.label')
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
            annot.ensure_annot(ctx, parc=self.mask)
        subject = ctx.state['mrisubject']
        subjects_dir = mri_sdir(ctx.state)
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


class AnnotDerivative(Derivative[list[mne.Label]]):
    name = 'annot'
    key_fields = ('mrisubject', 'parc')

    def __init__(self, parcs: dict[str, Parcellation], hemis: tuple[str, ...]):
        self.parcs = parcs
        self.hemis = hemis

    def annot_file_paths(self, state: dict[str, Any]) -> list[Path]:
        return [annot_file_path(state, hemi) for hemi in self.hemis]

    def annot_file_fingerprints(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            file_fingerprint(
                state['root'],
                annot_file_path(state, hemi),
                'annot-file',
                metadata={'mrisubject': state['mrisubject'], 'parc': state['parc'], 'hemi': hemi},
            )
            for hemi in self.hemis
        ]

    def label_file_fingerprints(self, state: dict[str, Any], parc_def: LabelParc) -> list[dict[str, Any]]:
        hemis = ('lh.', 'rh.')
        pattern = os.path.join(label_dir(state), '%s.label')
        labels = []
        for label in parc_def.labels:
            if label.startswith(hemis):
                labels.append(label)
            else:
                labels.extend(f'{hemi}{label}' for hemi in hemis)
        return [
            file_fingerprint(
                state['root'],
                pattern % label,
                'label-file',
                metadata={'label': label, 'parc': state['parc']},
            )
            for label in labels
        ]

    def annot_labels(self, state: dict[str, Any]) -> list[mne.Label]:
        return mne.read_labels_from_annot(state['mrisubject'], state['parc'], 'both', subjects_dir=mri_sdir(state))

    def managed_annot(self, state: dict[str, Any], parc_def: Parcellation) -> bool:
        if isinstance(parc_def, FreeSurferParc):
            return False
        if isinstance(parc_def, FSAverageParc):
            return state['mrisubject'] != state['common_brain']
        return True

    def load_annot(
            self,
            ctx: Request,
            *,
            parc: str | None = None,
            mrisubject: str | None = None,
    ) -> list[mne.Label]:
        state = {}
        if parc is not None:
            state['parc'] = parc
        if mrisubject is not None:
            state['mrisubject'] = mrisubject
        return ctx.load('annot', state=state)

    def ensure_annot(
            self,
            ctx: Request,
            *,
            parc: str | None = None,
            mrisubject: str | None = None,
    ) -> None:
        self.load_annot(ctx, parc=parc, mrisubject=mrisubject)

    def make_parcellation(
            self,
            ctx: Request,
            parc: str,
            parc_def: Parcellation,
    ) -> list[mne.Label]:
        labels = parc_def._make(ctx, self, parc)
        write_labels_to_annot(labels, ctx.state['mrisubject'], parc, True, mri_sdir(ctx.state))
        return labels

    def path(
            self,
            ctx: Request,
    ) -> Path:
        return annot_stamp_path(ctx.state)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        parc, parc_def = _resolve_parc(self.parcs, ctx.state['parc'])
        if parc_def is None or isinstance(parc_def, VolumeParc):
            return ()

        deps = []
        base = getattr(parc_def, 'base', None)
        if base:
            deps.append(Dependency('annot', label='base', state={'parc': base}))
        mask = getattr(parc_def, 'mask', None)
        if mask:
            deps.append(Dependency('annot', label='mask', state={'parc': mask}))

        mrisubject = ctx.state['mrisubject']
        common_brain = ctx.state['common_brain']
        fake_mri = is_fake_mri(mri_dir(ctx.state))
        if mrisubject != common_brain and (parc_def.morph_from_fsaverage or fake_mri):
            deps.append(Dependency('annot', label='common-brain', state={'mrisubject': common_brain}))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        parc, parc_def = _resolve_parc(self.parcs, ctx.state['parc'])
        if parc_def is None:
            return {'parc': parc, 'kind': 'none'}

        fingerprint = {
            'parc': parc,
            'definition': ctx.registry.canonicalize(parc_def._as_dict()),
        }
        if not self.managed_annot(ctx.state, parc_def):
            fingerprint['files'] = self.annot_file_fingerprints(ctx.state)
        elif isinstance(parc_def, LabelParc):
            fingerprint['labels'] = self.label_file_fingerprints(ctx.state, parc_def)
        return fingerprint

    def build(self, ctx: Request) -> list[mne.Label]:
        parc, parc_def = _resolve_parc(self.parcs, ctx.state['parc'])
        if parc_def is None or isinstance(parc_def, VolumeParc):
            return []
        if not self.managed_annot(ctx.state, parc_def):
            return self.annot_labels(ctx.state)

        mrisubject = ctx.state['mrisubject']
        common_brain = ctx.state['common_brain']
        fake_mri = is_fake_mri(mri_dir(ctx.state))
        if mrisubject != common_brain and (parc_def.morph_from_fsaverage or fake_mri):
            if fake_mri:
                for hemi in self.hemis:
                    src = annot_file_path({**ctx.state, 'mrisubject': common_brain}, hemi)
                    dst = annot_file_path(ctx.state, hemi)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
            else:
                Path(label_dir(ctx.state)).mkdir(parents=True, exist_ok=True)
                subjects_dir = mri_sdir(ctx.state)
                for hemi in ('lh', 'rh'):
                    cmd = [
                        "mri_surf2surf",
                        "--srcsubject", common_brain,
                        "--trgsubject", mrisubject,
                        "--sval-annot", parc,
                        "--tval", parc,
                        "--hemi", hemi,
                    ]
                    subp.run_freesurfer_command(cmd, subjects_dir)
                fix_annot_names(mrisubject, parc, common_brain, subjects_dir=subjects_dir)
            return self.annot_labels(ctx.state)

        return self.make_parcellation(ctx, parc, parc_def)

    def load(
            self,
            ctx: Request,
            path: Path) -> list[mne.Label]:
        return self.annot_labels(ctx.state)

    def save(
            self,
            ctx: Request,
            path: Path,
            value: list[mne.Label],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("annot\n")

    def validate(
            self,
            ctx: Request,
            path: Path,
            manifest,
    ) -> bool:
        return all(path.exists() for path in self.annot_file_paths(ctx.state))


class VolumeParc(Parcellation):
    "Assume it exists"
    kind = 'volume'
