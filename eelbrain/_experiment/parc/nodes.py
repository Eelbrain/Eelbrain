# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""The annotation derivative graph node.

All parcellation types produce or load ``*.annot`` files via a single shared
:class:`AnnotDerivative` graph node keyed on ``(mrisubject, parc)``. The
derivative dispatches to the appropriate build logic based on the
:class:`~._experiment.parc.config.Parcellation` subtype stored in the registry
at construction time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mne

from ...mne_fixes import write_labels_to_annot
from ..._utils import subp
from ..._mne import find_source_subject
from ..._utils.mne_utils import fix_annot_names, is_fake_mri
from ..derivative_cache import Dependency, ExternalArtifactDerivative, Request, file_fingerprint
from ..pathing import MRI_SDIR, annot_file_path, annot_stamp_path, label_dir
from .config import Parcellation, FreeSurferParc, FSAverageParc, LabelParc, VolumeParc, _resolve_parc


class AnnotDerivative(ExternalArtifactDerivative[list[mne.Label]]):
    name = 'annot'
    key_fields = ('mrisubject', 'parc')

    def __init__(self, parcs: dict[str, Parcellation]):
        self.parcs = parcs

    def annot_file_paths(self, state: dict[str, Any]) -> list[Path]:
        return [annot_file_path(state, hemi) for hemi in ('lh', 'rh')]

    def annot_file_fingerprints(self, ctx: Request) -> list[dict[str, Any]]:
        return [
            file_fingerprint(
                ctx.root,
                ctx.root / annot_file_path(ctx.state, hemi),
                metadata={'mrisubject': ctx.state['mrisubject'], 'parc': ctx.state['parc'], 'hemi': hemi},
            )
            for hemi in ('lh', 'rh')
        ]

    def label_file_fingerprints(self, ctx: Request, parc_def: LabelParc) -> list[dict[str, Any]]:
        hemis = ('lh.', 'rh.')
        pattern = os.path.join(str(ctx.root / label_dir(ctx.state)), '%s.label')
        labels = []
        for label in parc_def.labels:
            if label.startswith(hemis):
                labels.append(label)
            else:
                labels.extend(f'{hemi}{label}' for hemi in hemis)
        return [
            file_fingerprint(ctx.root, pattern % label, metadata={'parc': ctx.state['parc']})
            for label in labels
        ]

    def annot_labels(self, ctx: Request) -> list[mne.Label]:
        return mne.read_labels_from_annot(ctx.state['mrisubject'], ctx.state['parc'], 'both', subjects_dir=ctx.root / MRI_SDIR)

    def managed_annot(self, state: dict[str, Any], parc_def: Parcellation) -> bool:
        if isinstance(parc_def, FreeSurferParc):
            return False
        if isinstance(parc_def, FSAverageParc):
            return state['mrisubject'] != 'fsaverage'
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
        write_labels_to_annot(labels, ctx.state['mrisubject'], parc, True, ctx.root / MRI_SDIR)
        return labels

    def path(
            self,
            ctx: Request,
    ) -> Path:
        return ctx.root / annot_stamp_path(ctx.state)

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
        if parc_def.morph_from_fsaverage:
            source_subject = 'fsaverage'
        else:
            source_subject = find_source_subject(mrisubject, ctx.root / MRI_SDIR)
        if source_subject and source_subject != mrisubject:
            deps.append(Dependency('annot', label='source-subject', state={'mrisubject': source_subject}))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        parc, parc_def = _resolve_parc(self.parcs, ctx.state['parc'])
        if parc_def is None:
            return {'parc': parc, 'kind': 'none'}

        fingerprint = {
            'parc': parc,
            'definition': parc_def,
        }
        if not self.managed_annot(ctx.state, parc_def):
            fingerprint['files'] = self.annot_file_fingerprints(ctx)
        elif isinstance(parc_def, LabelParc):
            fingerprint['labels'] = self.label_file_fingerprints(ctx, parc_def)
        return fingerprint

    def build(self, ctx: Request) -> None:
        parc, parc_def = _resolve_parc(self.parcs, ctx.state['parc'])
        if parc_def is None or isinstance(parc_def, VolumeParc):
            return
        if not self.managed_annot(ctx.state, parc_def):
            return  # annot files are externally managed; load() reads them

        mrisubject = ctx.state['mrisubject']
        if 'source-subject' in ctx.declared_dependencies:
            source_subject = ctx.declared_dependencies['source-subject'].state['mrisubject']
            # materialize the source-subject annotation through the dependency tree
            # (its build writes the source .annot) rather than reading it off-disk
            common_brain_labels = ctx.load('source-subject')
            (ctx.root / label_dir(ctx.state)).mkdir(parents=True, exist_ok=True)
            subjects_dir = ctx.root / MRI_SDIR
            if is_fake_mri(subjects_dir / mrisubject):
                # a scaled MRI shares the common brain's surface topology, so its
                # labels can be written for this subject directly
                write_labels_to_annot(common_brain_labels, mrisubject, parc, True, ctx.root / MRI_SDIR)
            else:
                for hemi in ('lh', 'rh'):
                    cmd = [
                        "mri_surf2surf",
                        "--srcsubject", source_subject,
                        "--trgsubject", mrisubject,
                        "--sval-annot", parc,
                        "--tval", parc,
                        "--hemi", hemi,
                    ]
                    subp.run_freesurfer_command(cmd, subjects_dir)
                fix_annot_names(mrisubject, parc, source_subject, subjects_dir=subjects_dir)
        else:
            self.make_parcellation(ctx, parc, parc_def)

    def load(
            self,
            ctx: Request,
            path: Path) -> list[mne.Label]:
        return self.annot_labels(ctx)
