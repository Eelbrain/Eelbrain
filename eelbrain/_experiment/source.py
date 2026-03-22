# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model cache nodes."""

from __future__ import annotations

from os.path import basename, join
from typing import Any

import mne
from mne.minimum_norm import make_inverse_operator

from .derivative_cache import Artifact, CachePolicy, Dependency, Derivative, DerivativeContext


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
            Dependency('src-input'),
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
            src = mne.read_source_spaces(p.get('src-file', make=True))
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
