# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model cache nodes."""

from __future__ import annotations

from itertools import product
from os.path import basename, join
from typing import Any

import mne
from mne.minimum_norm import make_inverse_operator

from .derivative_cache import Artifact, CachePolicy, Dependency, Derivative, DerivativeContext
from .._utils.mne_utils import is_fake_mri
from ..mne_fixes._source_space import merge_volume_source_space, prune_volume_source_space, restrict_volume_source_space


class SrcDerivative(Derivative[mne.SourceSpaces]):
    name = 'src'
    path_template = 'src-file'
    key_fields = ('mrisubject', 'src')

    def _is_scaled(self, ctx: DerivativeContext) -> bool:
        return ctx.get('mrisubject') != ctx.get('common_brain') and is_fake_mri(ctx.get('mri-dir'))

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        deps = []
        if self._is_scaled(ctx):
            deps.append(Dependency(
                'src',
                label='common-brain-src',
                state=lambda c: {'mrisubject': c.get('common_brain')},
            ))
        elif ctx.get('src').startswith('vol'):
            deps.append(Dependency('bem-input'))
        return tuple(deps)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'mrisubject': ctx.get('mrisubject'),
            'src': ctx.get('src'),
            'common_brain': ctx.get('common_brain'),
            'fake_mri': is_fake_mri(ctx.get('mri-dir')),
        }

    def build(self, ctx: DerivativeContext) -> mne.SourceSpaces:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            dst = ctx.path('src-file', mkdir=True)
            subject = p.get('mrisubject')
            common_brain = p.get('common_brain')
            src = p.get('src')

            if self._is_scaled(ctx):
                ctx.load('src', mrisubject=common_brain)
                p._log.info(f"Scaling {src} source space for {subject}...")
                mne.scale_source_space(subject, f'{{subject}}-{src}-src.fif', subjects_dir=p.get('mri-sdir'), n_jobs=1)
                return mne.read_source_spaces(dst)

            mri_sdir = p.get('mri-sdir')
            kind, param, special = p._eval_src(src)
            grade = int(param)
            p._log.info(f"Generating {src} source space for {subject}...")
            if kind == 'vol':
                if subject == 'fsaverage':
                    bem = p.get('bem-file')
                else:
                    raise NotImplementedError("Volume source space for subject other than fsaverage")
                if special == 'brainstem':
                    name = 'brainstem'
                    voi = ['Brain-Stem', '3rd-Ventricle']
                    voi_lat = ('Thalamus-Proper', 'VentralDC')
                    remove_midline = False
                elif special == 'cortex':
                    name = 'cortex'
                    voi = []
                    voi_lat = ('Cerebral-Cortex',)
                    remove_midline = True
                elif not special:
                    name = 'cortex'
                    voi = []
                    voi_lat = ('Cerebral-Cortex', 'Cerebral-White-Matter')
                    remove_midline = True
                else:
                    raise RuntimeError(f'{src=}')
                voi.extend('%s-%s' % fmt for fmt in product(('Left', 'Right'), voi_lat))
                mri_dir = p.get('mri-dir', make=True)
                sss = mne.setup_volume_source_space(
                    subject,
                    pos=float(param),
                    bem=bem,
                    mri=join(mri_dir, 'mri', 'aseg.mgz'),
                    volume_label=voi,
                    subjects_dir=mri_sdir,
                )
                sss = merge_volume_source_space(sss, name)
                if special is None:
                    sss = restrict_volume_source_space(sss, grade, mri_sdir, subject, grow=1)
                return prune_volume_source_space(sss, grade, 3, remove_midline=remove_midline, fill_holes=4)

            spacing = kind + param
            return mne.setup_source_space(subject, spacing=spacing, add_dist=True, subjects_dir=mri_sdir, n_jobs=1)

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.SourceSpaces:
        return mne.read_source_spaces(artifact.path)

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.SourceSpaces,
    ) -> None:
        mne.write_source_spaces(artifact.path, value, overwrite=True)


class SourceMorphDerivative(Derivative[mne.SourceMorph]):
    name = 'source-morph'
    path_template = 'source-morph-file'
    key_fields = ('mrisubject', 'common_brain', 'src')

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency('src', label='src-from'),
            Dependency(
                'src',
                label='src-to',
                state=lambda c: {'mrisubject': c.get('common_brain')},
            ),
        )

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'mrisubject': ctx.get('mrisubject'),
            'common_brain': ctx.get('common_brain'),
            'src': ctx.get('src'),
        }

    def build(self, ctx: DerivativeContext) -> mne.SourceMorph:
        subject_from = ctx.get('mrisubject')
        subject_to = ctx.get('common_brain')
        subjects_dir = ctx.get('mri-sdir')
        src_from = ctx.load('src')
        src_to = ctx.load('src', mrisubject=subject_to)
        return mne.compute_source_morph(
            src_from,
            subject_from,
            subject_to,
            subjects_dir,
            src_to=src_to,
            precompute=True,
        )

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.SourceMorph:
        return mne.read_source_morph(artifact.path)

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.SourceMorph,
    ) -> None:
        value.save(artifact.path, overwrite=True)


class FwdDerivative(Derivative[mne.Forward]):
    name = 'fwd'
    path_template = 'fwd-file'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'mrisubject', 'src',
    )

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency(
                'raw-input-meeg',
                state=lambda c: {'raw': 'raw'},
                options=lambda c: {'add_bads': False},
            ),
            Dependency('trans-input'),
            Dependency('src'),
        )

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'raw': ctx.get('raw'),
            'epoch': ctx.get('epoch'),
            'mrisubject': ctx.get('mrisubject'),
            'src': ctx.get('src'),
        }

    def build(self, ctx: DerivativeContext) -> mne.Forward:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            raw = p.load_raw(add_bads=False)
            src = ctx.load('src')
            dst = p.get('fwd-file')
            p._log.debug(f"make_fwd {basename(dst)}...")
            if p.get('mrisubject') == 'fsaverage':
                bemsol = join(p.get('mri-dir'), 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
            else:
                bem = p._load_bem()
                bemsol = mne.make_bem_solution(bem)
            if 'kit_system_id' in raw.info:
                is_kit = raw.info['kit_system_id'] is not None
            else:
                raise RuntimeError("Unclear how to set ignor_ref for legacy file without kit_system_id")
            fwd = mne.make_forward_solution(
                raw.info,
                p.get('trans-file'),
                src,
                bemsol,
                ignore_ref=is_kit,
            )
            for s, s0 in zip(fwd['src'], src):
                if s['nuse'] != s0['nuse']:
                    raise RuntimeError(
                        f"The forward solution {basename(dst)} contains fewer sources than the source space. "
                        "This could be due to a corrupted bem file with sources outside of the inner skull surface."
                    )
            return fwd

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.Forward:
        return mne.read_forward_solution(artifact.path)

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.Forward,
    ) -> None:
        mne.write_forward_solution(artifact.path, value, overwrite=True)


class InvDerivative(Derivative[mne.minimum_norm.InverseOperator]):
    name = 'inv'
    path_template = 'inv-file'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv',
    )
    cache_policy = CachePolicy.OPTIONAL

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('fwd'), Dependency('cov'))

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'raw': ctx.get('raw'),
            'epoch': ctx.get('epoch'),
            'rej': ctx.get('rej'),
            'cov': ctx.get('cov'),
            'src': ctx.get('src'),
            'inv': ctx.get('inv'),
        }

    def build(self, ctx: DerivativeContext) -> mne.minimum_norm.InverseOperator:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            src = p.get('src')
            inv = p.get('inv')
            if src[:3] == 'vol' and not (inv.startswith('vec') or inv.startswith('free')):
                raise ValueError(f'{inv=} with {src=}: volume source space requires free or vector inverse')
            fiff = ctx.option('fiff')
            if fiff is None:
                fiff = p.load_raw()
            _, make_kw, _ = p._inv_params()
            return make_inverse_operator(
                fiff.info,
                ctx.load('fwd'),
                ctx.load('cov'),
                use_cps=True,
                **make_kw,
            )

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.minimum_norm.InverseOperator:
        return mne.minimum_norm.read_inverse_operator(artifact.path)

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.minimum_norm.InverseOperator,
    ) -> None:
        mne.minimum_norm.write_inverse_operator(artifact.path, value, overwrite=True)
