# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model cache nodes."""

from __future__ import annotations

import logging
from itertools import product
import os
from os.path import basename, join
from pathlib import Path
import re
from typing import Any

import mne
from mne.minimum_norm import make_inverse_operator

from .covariance import cov_node_name
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext, Input, file_fingerprint
from .pathing import (
    bem_dir, bem_file_path, fwd_file_path, inv_file_path, mri_dir,
    mri_sdir, source_morph_file_path, src_file_path, trans_file_path,
)
from .preprocessing import load_raw_dependency
from .._text import enumeration, plural
from .._utils import subp
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


def _load_bem(state: dict[str, Any], log: logging.Logger) -> mne.ConductorModel:
    subject = state['mrisubject']
    if subject == 'fsaverage' or is_fake_mri(mri_dir(state)):
        return mne.read_bem_surfaces(bem_file_path(state))

    bem_dir_ = bem_dir(state)
    surfs = ('brain', 'inner_skull', 'outer_skull', 'outer_skin')
    paths = {surf: join(bem_dir_, surf + '.surf') for surf in surfs}
    missing = [surf for surf in surfs if not Path(paths[surf]).exists()]
    if missing:
        for surf in missing[:]:
            path = Path(paths[surf])
            if path.is_symlink():
                new_target = Path('watershed') / f'{subject}_{surf}_surface'
                if (Path(bem_dir_) / new_target).exists():
                    log.info("Fixing broken symlink for %s %s surface file", subject, surf)
                    path.unlink()
                    path.symlink_to(new_target)
                    missing.remove(surf)
                else:
                    log.error("%s missing for %s", new_target, subject)
        if missing:
            log.info("%s %s missing for %s. Running mne.make_watershed_bem()...", enumeration(missing).capitalize(), plural('surface', len(missing)), subject)
            os.environ['FREESURFER_HOME'] = subp.get_fs_home()
            mne.bem.make_watershed_bem(subject, mri_sdir(state), overwrite=True)
    return mne.make_bem_model(subject, conductivity=(0.3,), subjects_dir=mri_sdir(state))


class TransInput(Input):
    name = 'trans-input'

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return file_fingerprint(ctx.get('root'), trans_file_path(ctx.state), 'trans-file')


class BemInput(Input):
    name = 'bem-input'

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return file_fingerprint(ctx.get('root'), bem_file_path(ctx.state), 'bem-file')


class SrcDerivative(Derivative[mne.SourceSpaces]):
    name = 'src'
    key_fields = ('mrisubject', 'src')

    def __init__(self, log: logging.Logger):
        self.log = log

    def _is_scaled(self, ctx: DerivativeContext) -> bool:
        return ctx.get('mrisubject') != ctx.get('common_brain') and is_fake_mri(mri_dir(ctx.state))

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = src_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

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
            'fake_mri': is_fake_mri(mri_dir(ctx.state)),
        }

    def build(self, ctx: DerivativeContext) -> mne.SourceSpaces:
        dst = self.path(ctx, mkdir=True)
        subject = ctx.get('mrisubject')
        common_brain = ctx.get('common_brain')
        src = ctx.get('src')

        if self._is_scaled(ctx):
            ctx.load('src', mrisubject=common_brain)
            self.log.info("Scaling %s source space for %s...", src, subject)
            mne.scale_source_space(subject, f'{{subject}}-{src}-src.fif', subjects_dir=mri_sdir(ctx.state), n_jobs=1)
            return mne.read_source_spaces(dst)

        subjects_dir = mri_sdir(ctx.state)
        kind, param, special = parse_src(src)
        grade = int(param)
        self.log.info("Generating %s source space for %s...", src, subject)
        if kind == 'vol':
            if subject == 'fsaverage':
                bem = bem_file_path(ctx.state)
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
            mri_dir_ = Path(mri_dir(ctx.state))
            mri_dir_.mkdir(parents=True, exist_ok=True)
            sss = mne.setup_volume_source_space(
                subject,
                pos=float(param),
                bem=bem,
                mri=join(mri_dir_, 'mri', 'aseg.mgz'),
                volume_label=voi,
                subjects_dir=subjects_dir,
            )
            sss = merge_volume_source_space(sss, name)
            if special is None:
                sss = restrict_volume_source_space(sss, grade, subjects_dir, subject, grow=1)
            return prune_volume_source_space(sss, grade, 3, remove_midline=remove_midline, fill_holes=4)

        spacing = kind + param
        return mne.setup_source_space(subject, spacing=spacing, add_dist=True, subjects_dir=subjects_dir, n_jobs=1)

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.SourceSpaces:
        return mne.read_source_spaces(path)

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.SourceSpaces,
    ) -> None:
        mne.write_source_spaces(path, value, overwrite=True)


class SourceMorphDerivative(Derivative[mne.SourceMorph]):
    name = 'source-morph'
    key_fields = ('mrisubject', 'common_brain', 'src')

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = source_morph_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

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
        subjects_dir = mri_sdir(ctx.state)
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
            path: Path) -> mne.SourceMorph:
        return mne.read_source_morph(path)

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.SourceMorph,
    ) -> None:
        value.save(path, overwrite=True)


class FwdDerivative(Derivative[mne.Forward]):
    name = 'fwd'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'mrisubject', 'src',
    )

    def __init__(self, log: logging.Logger):
        self.log = log

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = fwd_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency(
                'raw',
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
        raw = load_raw_dependency(ctx, ctx.get('raw'), add_bads=False)
        src = ctx.load('src')
        dst = self.path(ctx)
        self.log.debug("make_fwd %s...", basename(dst))
        if ctx.get('mrisubject') == 'fsaverage':
            bemsol = join(mri_dir(ctx.state), 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        else:
            bem = _load_bem(ctx.state, self.log)
            bemsol = mne.make_bem_solution(bem)
        if 'kit_system_id' in raw.info:
            is_kit = raw.info['kit_system_id'] is not None
        else:
            raise RuntimeError("Unclear how to set ignor_ref for legacy file without kit_system_id")
        fwd = mne.make_forward_solution(
            raw.info,
            trans_file_path(ctx.state),
            src,
            bemsol,
            ignore_ref=is_kit,
        )
        for src_part, src_ref in zip(fwd['src'], src):
            if src_part['nuse'] != src_ref['nuse']:
                raise RuntimeError(
                    f"The forward solution {basename(dst)} contains fewer sources than the source space. "
                    "This could be due to a corrupted bem file with sources outside of the inner skull surface."
                )
        return fwd

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.Forward:
        return mne.read_forward_solution(path)

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.Forward,
    ) -> None:
        mne.write_forward_solution(path, value, overwrite=True)


class InvDerivative(Derivative[mne.minimum_norm.InverseOperator]):
    name = 'inv'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv',
    )
    cache_policy = CachePolicy.OPTIONAL

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = inv_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

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
            path: Path) -> mne.minimum_norm.InverseOperator:
        return mne.minimum_norm.read_inverse_operator(path)

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.minimum_norm.InverseOperator,
    ) -> None:
        mne.minimum_norm.write_inverse_operator(path, value, overwrite=True)
