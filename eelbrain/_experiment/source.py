# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model cache nodes."""

from __future__ import annotations

from itertools import product
from os.path import basename, join
import re
from typing import Any

import mne
from mne.minimum_norm import make_inverse_operator

from .covariance import cov_node_name
from .derivative_cache import Artifact, CachePolicy, Dependency, Derivative, DerivativeContext, Input, file_fingerprint
from .preprocessing import load_raw_dependency, raw_meeg_input_name
from .._utils.mne_utils import is_fake_mri
from ..mne_fixes._source_space import merge_volume_source_space, prune_volume_source_space, restrict_volume_source_space


INV_METHODS = ('MNE', 'dSPM', 'sLORETA', 'eLORETA', 'champ')
SRC_RE = re.compile(r'^(ico|vol)-(\d+)(?:-(cortex|brainstem))?$')
INV_RE = re.compile(r"^(free|fixed|loose\.\d+|vec)"
                    r"(?:-(\d*\.?\d+))?"
                    rf"-({'|'.join(INV_METHODS)})"
                    r"(?:-((?:0\.)?\d+))?"
                    r"(?:-(pick_normal))?"
                    r"$")


def parse_src(src: str) -> tuple[str, str, str | None]:
    m = SRC_RE.match(src)
    if not m:
        raise ValueError(f'src={src}')
    kind, param, special = m.groups()
    if special and kind != 'vol':
        raise ValueError(f'src={src}')
    return kind, param, special


def eval_src(src: str) -> str:
    parse_src(src)
    return src


def inv_str(
        ori: str = 'free',
        snr: float = 3,
        method: str = 'dSPM',
        depth: float = 0.8,
        pick_normal: bool = False,
) -> str:
    if isinstance(ori, str):
        if ori not in ('free', 'fixed', 'vec'):
            raise ValueError(f"{ori=}; needs to be 'free', 'fixed', 'vec', or float")
    elif not 0 < ori < 1:
        raise ValueError(f"{ori=}; must be in range (0, 1)")
    else:
        ori = f'loose{str(ori)[1:]}'
    items = [ori]

    if snr > 0:
        items.append(f'{snr:g}')
    elif snr < 0:
        raise ValueError(f"{snr=}")

    if method in INV_METHODS:
        items.append(method)
    else:
        raise ValueError(f"{method=}")

    if not 0 <= depth <= 1:
        raise ValueError(f"{depth=}; must be in range [0, 1]")
    elif depth != 0.8:
        items.append(f'{depth:g}')

    if pick_normal:
        if ori in ('vec', 'fixed'):
            raise ValueError(f"{ori=} and pick_normal=True are incompatible")
        items.append('pick_normal')

    return '-'.join(items)


def parse_inv(inv: str) -> tuple[str | float, float, str, float, bool]:
    m = INV_RE.match(inv)
    if m is None:
        raise ValueError(f"{inv=}: invalid inverse specification")

    ori, snr, method, depth, pick_normal = m.groups()
    if ori.startswith('loose'):
        ori = float(ori[5:])
        if not 0 < ori < 1:
            raise ValueError(f"{inv=}: loose parameter needs to be in range (0, 1)")
    elif pick_normal and ori in ('vec', 'fixed'):
        raise ValueError(f"{inv=}: {ori} incompatible with pick_normal")

    if snr is None:
        snr = 0
    else:
        snr = float(snr)
        if snr < 0:
            raise ValueError(f"{inv=}: {snr=}")

    if method not in INV_METHODS:
        raise ValueError(f"{inv=}: {method=}")

    if depth is None:
        depth = 0.8
    else:
        depth = float(depth)
        if not 0 <= depth <= 1:
            raise ValueError(f"{inv=}: {depth=}, needs to be in range [0, 1]")

    return ori, snr, method, depth, bool(pick_normal)


def eval_inv(inv: str) -> str:
    return inv_str(*parse_inv(inv))


def update_inv_cache(fields: dict[str, Any]) -> str:
    if '*' in fields['inv']:
        return fields['inv']
    ori, _, _, depth, _ = INV_RE.match(fields['inv']).groups()
    if depth:
        return f'{ori}-{depth}'
    return ori


def inverse_operator_params(inv: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if '*' in inv:
        raise ValueError(f'{inv=} with wildcard')

    ori, snr, method, depth, pick_normal = parse_inv(inv)
    if ori == 'fixed':
        make_kw = {'fixed': True}
    elif ori == 'free' or ori == 'vec':
        make_kw = {'loose': 1}
    elif isinstance(ori, float):
        make_kw = {'loose': ori}
    else:
        raise RuntimeError(f"{inv=} (orientation={ori!r})")

    if depth is None:
        make_kw['depth'] = 0.8
    elif depth == 0:
        make_kw['depth'] = None
    else:
        make_kw['depth'] = depth

    apply_kw = {'method': method, 'lambda2': 1. / snr ** 2 if snr else 0}
    if ori == 'vec':
        apply_kw['pick_ori'] = 'vector'
    elif pick_normal:
        apply_kw['pick_ori'] = 'normal'

    return method, make_kw, apply_kw


class _NamedFileInput(Input):
    def __init__(
            self,
            name: str,
            *,
            template: str,
            kind: str,
            digest: bool = False,
            metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.template = template
        self.kind = kind
        self.digest = digest
        self.metadata = metadata

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            return file_fingerprint(p.root, p.get(self.template), self.kind, self.digest, self.metadata)


class TransInput(_NamedFileInput):
    def __init__(self):
        super().__init__('trans-input', template='trans-file', kind='trans-file')


class BemInput(_NamedFileInput):
    def __init__(self):
        super().__init__('bem-input', template='bem-file', kind='bem-file')


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
            subject = ctx.get('mrisubject')
            common_brain = ctx.get('common_brain')
            src = ctx.get('src')

            if self._is_scaled(ctx):
                ctx.load('src', mrisubject=common_brain)
                p._log.info(f"Scaling {src} source space for {subject}...")
                mne.scale_source_space(subject, f'{{subject}}-{src}-src.fif', subjects_dir=ctx.get('mri-sdir'), n_jobs=1)
                return mne.read_source_spaces(dst)

            mri_sdir = ctx.get('mri-sdir')
            kind, param, special = parse_src(src)
            grade = int(param)
            p._log.info(f"Generating {src} source space for {subject}...")
            if kind == 'vol':
                if subject == 'fsaverage':
                    bem = ctx.path('bem-file')
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
                mri_dir = ctx.path('mri-dir', make=True)
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
                raw_meeg_input_name('raw'),
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
            raw = load_raw_dependency(ctx, ctx.get('raw'), add_bads=False)
            src = ctx.load('src')
            dst = ctx.path('fwd-file')
            p._log.debug(f"make_fwd {basename(dst)}...")
            if ctx.get('mrisubject') == 'fsaverage':
                bemsol = join(ctx.get('mri-dir'), 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
            else:
                bem = p._load_bem()
                bemsol = mne.make_bem_solution(bem)
            if 'kit_system_id' in raw.info:
                is_kit = raw.info['kit_system_id'] is not None
            else:
                raise RuntimeError("Unclear how to set ignor_ref for legacy file without kit_system_id")
            fwd = mne.make_forward_solution(
                raw.info,
                ctx.path('trans-file'),
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
        return (Dependency('fwd'), Dependency(cov_node_name(ctx.get('cov'))))

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
        src = ctx.get('src')
        inv = ctx.get('inv')
        if src[:3] == 'vol' and not (inv.startswith('vec') or inv.startswith('free')):
            raise ValueError(f'{inv=} with {src=}: volume source space requires free or vector inverse')
        fiff = ctx.option('fiff')
        if fiff is None:
            fiff = load_raw_dependency(ctx, ctx.get('raw'))
        _, make_kw, _ = inverse_operator_params(inv)
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
