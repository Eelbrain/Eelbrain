# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model and source-data derivatives.

These nodes own the reusable source-space products behind
``Pipeline.load_inv``, ``Pipeline.load_evoked_stc``, and
``Pipeline.load_epochs_stc``. Higher-level derivatives should load them
through :meth:`DerivativeContext.load` instead of relying on injected facade
methods.
"""

from __future__ import annotations

import logging
from itertools import product
import os
from os.path import basename, join
from pathlib import Path
import re
from typing import Any
from collections.abc import Sequence

import mne
import numpy as np
from mne.minimum_norm import apply_inverse, apply_inverse_epochs, make_inverse_operator
from mne.morph import SourceMorph
from scipy import sparse

from .. import load, save
from .._data_obj import Dataset, Datalist, NDVar, combine
from .covariance import cov_node_name
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext, Input, file_fingerprint
from .events import EPOCHS_DATA
from .pathing import (
    bem_dir, bem_file_path, mri_dir, mri_sdir, src_file_path, trans_file_path,
)
from .preprocessing import load_raw_dependency, raw_node_name
from .._text import enumeration, plural
from .._utils import subp
from .._utils.mne_utils import is_fake_mri
from ..mne_fixes._source_space import merge_volume_source_space, prune_volume_source_space, restrict_volume_source_space
from .._mne import find_source_subject, label_from_annot, morph_source_space


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


def _selected_parc(
        ctx: DerivativeContext,
        mask: bool | str = False,
) -> str | None:
    parc = ctx.option('parc') or ctx.get('parc') or None
    if isinstance(mask, str):
        return mask
    return parc


def _identity_source_morph(
        subject_from: str,
        subject_to: str,
        src_from: mne.SourceSpaces,
        src_to: mne.SourceSpaces,
) -> SourceMorph:
    """Create a trivial surface :class:`mne.SourceMorph` for scaled template brains.

    This is only for the public ``load_source_morph()`` API in the special case
    where a subject source space is a scaled copy of ``subject_to``. It is not
    a general fallback for missing Freesurfer morph data.

    The source spaces must therefore match exactly in their per-hemisphere
    vertex definitions. If they do not, the scaled-source-space invariant of
    the pipeline is broken and a real morph would be required.
    """
    vertices_from = [np.array(src['vertno'], int) for src in src_from[:2]]
    vertices_to = [np.array(src['vertno'], int) for src in src_to[:2]]
    if not all(np.array_equal(v_from, v_to) for v_from, v_to in zip(vertices_from, vertices_to)):
        raise RuntimeError(
            "Scaled source-space morph requires identical per-hemisphere vertices in source and target source spaces"
        )
    n_from = sum(len(vertices) for vertices in vertices_from)
    return SourceMorph(
        subject_from,
        subject_to,
        'surface',
        None,
        None,
        None,
        None,
        None,
        False,
        sparse.eye(n_from, format='csr'),
        vertices_to,
        None,
        None,
        None,
        None,
        {'vertices_from': vertices_from},
        None,
    )


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
                state={'mrisubject': ctx.get('common_brain')},
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
            ctx.registry.log.info("Scaling %s source space for %s...", src, subject)
            mne.scale_source_space(subject, f'{{subject}}-{src}-src.fif', subjects_dir=mri_sdir(ctx.state), n_jobs=1)
            return mne.read_source_spaces(dst)

        subjects_dir = mri_sdir(ctx.state)
        kind, param, special = parse_src(src)
        grade = int(param)
        ctx.registry.log.info("Generating %s source space for %s...", src, subject)
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
    cache_suffix = '-morph.h5'

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency('src', label='src-from'),
            Dependency(
                'src',
                label='src-to',
                state={'mrisubject': ctx.get('common_brain')},
            ),
        )

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'mrisubject': ctx.get('mrisubject'),
            'common_brain': ctx.get('common_brain'),
            'src': ctx.get('src'),
            'fake_mri': is_fake_mri(mri_dir(ctx.state)),
        }

    def build(self, ctx: DerivativeContext) -> mne.SourceMorph:
        subject_from = ctx.get('mrisubject')
        subject_to = ctx.get('common_brain')
        subjects_dir = mri_sdir(ctx.state)
        src_to = ctx.load('src', mrisubject=subject_to)
        if is_fake_mri(mri_dir(ctx.state)) and subject_from != subject_to:
            src_from = ctx.load('src')
            return _identity_source_morph(subject_from, subject_to, src_from, src_to)
        src_from = ctx.load('src')
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
    cache_suffix = '-fwd.fif'

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency(
                raw_node_name('raw'),
                state={'raw': 'raw'},
                options={'add_bads': False},
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
        if ctx.get('mrisubject') == 'fsaverage':
            bemsol = join(mri_dir(ctx.state), 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        else:
            bem = _load_bem(ctx.state, ctx.registry.log)
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
    cache_suffix = '-inv.fif'

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
            ctx.load(cov_node_name(ctx.get('cov'))),
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


def _mask_ndvar(y: NDVar):
    if y.source.parc is None:
        raise RuntimeError(f'{y} has no parcellation')
    mask = y.source.parc.startswith('unknown')
    if mask.any():
        return y.sub(source=np.invert(mask))
    return y


def load_epochs_stc_group(
        registry,
        subjects: Sequence[str],
        state: dict[str, Any],
        options: dict[str, Any],
        mri_subjects: dict[str, dict[str, str]],
        common_brain: str,
) -> Dataset:
    if options['data_raw']:
        raise ValueError(f"data_raw={options['data_raw']!r} with group: Can not combine raw data from multiple subjects.")
    if options['keep_epochs']:
        raise ValueError(f"keep_epochs={options['keep_epochs']!r} with group: Can not combine Epochs objects for different subjects. Set keep_epochs=False (default).")
    morph = options.get('morph')
    if morph is None:
        options = {**options, 'morph': True}
    elif not morph:
        raise ValueError(f"morph={morph!r} with group: Source estimates can only be combined after morphing data to common brain model. Set morph=True.")
    dss = [
        registry.load('epochs-stc', state=_subject_state(state, subject, mri_subjects, common_brain), options=options)
        for subject in subjects
    ]
    return combine(dss)


def load_epochs_stc_request(
        registry,
        groups: dict[str, Sequence[str]],
        current_group: str,
        state: dict[str, Any],
        options: dict[str, Any],
        mri_subjects: dict[str, dict[str, str]],
        common_brain: str,
        subjects,
) -> Dataset:
    if subjects is True:
        subjects = current_group
    elif subjects in (None, 1):
        return registry.load('epochs-stc', state=state, options=options)
    if isinstance(subjects, Sequence) and not isinstance(subjects, str):
        return load_epochs_stc_group(registry, subjects, state, options, mri_subjects, common_brain)
    if isinstance(subjects, str) and subjects in groups:
        return load_epochs_stc_group(registry, groups[subjects], state, options, mri_subjects, common_brain)
    return registry.load('epochs-stc', state=_subject_state(state, subjects, mri_subjects, common_brain), options=options)


def load_evoked_stc_group(
        registry,
        subjects: Sequence[str],
        state: dict[str, Any],
        options: dict[str, Any],
        mri_subjects: dict[str, dict[str, str]],
        common_brain: str,
) -> Dataset:
    morph = options.get('morph')
    if options['ndvar']:
        if morph is None:
            options = {**options, 'morph': True}
        elif not morph:
            raise ValueError("ndvar=True, morph=False with multiple subjects: Can't create ndvars with data from different brains")
    dss = [
        registry.load('evoked-stc', state=_subject_state(state, subject, mri_subjects, common_brain), options=options)
        for subject in subjects
    ]
    return combine(dss)


def load_evoked_stc_request(
        registry,
        groups: dict[str, Sequence[str]],
        current_group: str,
        state: dict[str, Any],
        options: dict[str, Any],
        mri_subjects: dict[str, dict[str, str]],
        common_brain: str,
        subjects,
) -> Dataset:
    if subjects is True:
        subjects = current_group
    elif subjects in (None, 1):
        return registry.load('evoked-stc', state=state, options=options)
    if isinstance(subjects, Sequence) and not isinstance(subjects, str):
        return load_evoked_stc_group(registry, subjects, state, options, mri_subjects, common_brain)
    if isinstance(subjects, str) and subjects in groups:
        return load_evoked_stc_group(registry, groups[subjects], state, options, mri_subjects, common_brain)
    return registry.load('evoked-stc', state=_subject_state(state, subjects, mri_subjects, common_brain), options=options)


def _subject_state(
        state: dict[str, Any],
        subject: str,
        mri_subjects: dict[str, dict[str, str]],
        common_brain: str,
) -> dict[str, Any]:
    out = {**state, 'subject': subject}
    mri = out.get('mri')
    if mri not in (None, '', '*'):
        mrisubject = mri_subjects[mri][subject]
        if mrisubject != common_brain and not mrisubject.startswith('sub-'):
            mrisubject = 'sub-' + mrisubject
        out['mrisubject'] = mrisubject
    return out


def _prepare_inv(
        ctx: DerivativeContext,
        fiff: Any,
        mask: bool | str,
        morph: bool | None,
):
    parc = _selected_parc(ctx, mask)
    if parc:
        ctx.load('annot', state={'parc': parc})

    inv = ctx.load('inv', options={'fiff': fiff})
    subjects_dir = mri_sdir(ctx.state)
    mrisubject = ctx.get('mrisubject')
    is_scaled = find_source_subject(mrisubject, subjects_dir)
    if mask and (is_scaled or not morph):
        label = label_from_annot(inv['src'], mrisubject, subjects_dir, parc)
    else:
        label = None
    return inv, label, subjects_dir, mrisubject, is_scaled, parc


class EpochsStcDerivative(Derivative[Dataset]):
    name = 'epochs-stc'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.pickle'

    def __init__(self, epochs: dict[str, Any]):
        self.epochs = epochs

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        deps = [
            Dependency(EPOCHS_DATA, options={
                'baseline': ctx.option('baseline'),
                'ndvar': False if ctx.option('keep_epochs', False) in (True, False) else 'both',
                'reject': ctx.option('reject', True),
                'cat': ctx.option('cat'),
                'samplingrate': ctx.option('samplingrate'),
                'decim': ctx.option('decim'),
                'pad': ctx.option('pad', 0),
                'data_raw': ctx.option('data_raw', False),
                'vardef': ctx.option('vardef'),
                'data': 'sensor',
                'add_bads': True,
            }),
            Dependency('inv'),
        ]
        mask = ctx.option('mask', False)
        if mask:
            parc = _selected_parc(ctx, mask)
            deps.append(Dependency('annot', state={'parc': parc}))
        if ctx.option('morph', False):
            deps.append(Dependency('source-morph'))
        return tuple(deps)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.canonicalize({
            'baseline': ctx.option('baseline'),
            'src_baseline': ctx.option('src_baseline'),
            'cat': ctx.option('cat'),
            'keep_epochs': ctx.option('keep_epochs', False),
            'morph': ctx.option('morph'),
            'mask': ctx.option('mask'),
            'data_raw': ctx.option('data_raw', False),
            'vardef': repr(ctx.option('vardef')),
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'pad': ctx.option('pad', 0),
            'ndvar': ctx.option('ndvar', True),
            'reject': ctx.option('reject', True),
            'adjacency': ctx.get('adjacency'),
        })

    def build(self, ctx: DerivativeContext) -> Dataset:
        epoch = self.epochs[ctx.get('epoch')]
        keep_epochs = ctx.option('keep_epochs', False)
        if keep_epochs is True:
            sns_ndvar = False
            del_epochs = False
        elif keep_epochs is False:
            sns_ndvar = False
            del_epochs = True
        elif keep_epochs == 'ndvar':
            sns_ndvar = 'both'
            del_epochs = True
        elif keep_epochs == 'both':
            sns_ndvar = 'both'
            del_epochs = False
        else:
            raise ValueError(f"{keep_epochs=}")

        ds = ctx.load(EPOCHS_DATA, options={
            'baseline': ctx.option('baseline'),
            'ndvar': sns_ndvar,
            'reject': ctx.option('reject', True),
            'cat': ctx.option('cat'),
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'pad': ctx.option('pad', 0),
            'data_raw': ctx.option('data_raw', False),
            'vardef': ctx.option('vardef'),
            'data': 'sensor',
            'add_bads': True,
        })

        src_baseline = ctx.option('src_baseline', False)
        if not ctx.option('baseline') and src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError("src_baseline with post_baseline_trigger_shift")
        if src_baseline is True:
            src_baseline = epoch.baseline

        epoch_list = ds['epochs'] if isinstance(ds['epochs'], Datalist) else [ds['epochs']]
        inv, label, subjects_dir, mrisubject, is_scaled, parc = _prepare_inv(ctx, epoch_list[0], ctx.option('mask', False), ctx.option('morph', False))
        method, make_kw, apply_kw = inverse_operator_params(ctx.get('inv'))
        stc_list = [apply_inverse_epochs(epoch_obj, inv, label=label, **apply_kw) for epoch_obj in epoch_list]
        is_variable_time = isinstance(ds['epochs'], Datalist)
        if is_variable_time:
            stc_list = [stc for stc, in stc_list]

        if ctx.option('ndvar', True):
            src = ctx.get('src')
            ndvar_list = [
                load.mne.stc_ndvar(stc, mrisubject, src, subjects_dir, method, make_kw.get('fixed', False), parc=parc, adjacency=ctx.get('adjacency'))
                for stc in stc_list
            ]
            if src_baseline:
                for value in ndvar_list:
                    value -= value.summary(time=src_baseline)
            if ctx.option('morph', False):
                common_brain = ctx.get('common_brain')
                ctx.load('annot', state={'mrisubject': common_brain})
                ndvar_list = [morph_source_space(value, common_brain) for value in ndvar_list]
                if ctx.option('mask', False) and not is_scaled:
                    ndvar_list = [_mask_ndvar(value) for value in ndvar_list]
                key = 'srcm'
            else:
                key = 'src'
            src_var = ndvar_list
        else:
            if src_baseline:
                raise NotImplementedError("Baseline for SourceEstimate")
            if ctx.option('morph', False):
                raise NotImplementedError("Morphing for SourceEstimate")
            key = 'stc'
            src_var = stc_list

        ds[key] = src_var if is_variable_time else src_var[0]
        if del_epochs:
            del ds['epochs']
        return ds

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class EvokedStcDerivative(Derivative[Dataset]):
    name = 'evoked-stc'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count', 'cov', 'mrisubject',
        'src', 'inv',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.pickle'

    def __init__(self, epochs: dict[str, Any]):
        self.epochs = epochs

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        deps = [
            Dependency('evoked', options={
                'baseline': ctx.option('baseline'),
                'ndvar': 2 if ctx.option('keep_evoked', False) and ctx.option('ndvar', True) else False,
                'cat': ctx.option('cat'),
                'samplingrate': ctx.option('samplingrate'),
                'decim': ctx.option('decim'),
                'data_raw': ctx.option('data_raw', False),
                'vardef': ctx.option('vardef'),
                'data': 'sensor',
            }),
            Dependency('inv'),
        ]
        mask = ctx.option('mask', False)
        if mask:
            parc = _selected_parc(ctx, mask)
            deps.append(Dependency('annot', state={'parc': parc}))
        if ctx.option('morph', False):
            deps.append(Dependency('source-morph'))
        return tuple(deps)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.canonicalize({
            'baseline': ctx.option('baseline'),
            'src_baseline': ctx.option('src_baseline'),
            'cat': ctx.option('cat'),
            'keep_evoked': ctx.option('keep_evoked', False),
            'morph': ctx.option('morph'),
            'mask': ctx.option('mask'),
            'data_raw': ctx.option('data_raw', False),
            'vardef': repr(ctx.option('vardef')),
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'ndvar': ctx.option('ndvar', True),
            'adjacency': ctx.get('adjacency'),
        })

    def build(self, ctx: DerivativeContext) -> Dataset:
        keep_evoked = ctx.option('keep_evoked', False)
        ndvar = ctx.option('ndvar', True)
        sensor_ndvar = 2 if keep_evoked and ndvar else False
        ds = ctx.load('evoked', options={
            'baseline': ctx.option('baseline'),
            'ndvar': sensor_ndvar,
            'cat': ctx.option('cat'),
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'data_raw': ctx.option('data_raw', False),
            'vardef': ctx.option('vardef'),
            'data': 'sensor',
        })

        src_baseline = ctx.option('src_baseline', False)
        epoch = self.epochs[ctx.get('epoch')]
        if src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError(f"{src_baseline=}: post_baseline_trigger_shift is not implemented for baseline correction in source space")
        if src_baseline is True:
            src_baseline = epoch.baseline
        invs = {}
        stcs = []
        subject = ctx.get('subject')
        mrisubject = ctx.get('mrisubject')
        common_brain = ctx.get('common_brain')
        if is_fake_mri(mri_dir(ctx.state)):
            subject_from = common_brain
        else:
            subject_from = mrisubject
        parc = _selected_parc(ctx, ctx.option('mask', False))
        if ndvar:
            target_subject = common_brain if ctx.option('morph', False) else mrisubject
            if parc:
                ctx.load('annot', state={'mrisubject': target_subject, 'parc': parc})
        source_morph = None
        if ctx.option('morph', False) and subject_from != common_brain:
            source_morph = ctx.load('source-morph', state={'mrisubject': subject_from})

        method, make_kw, apply_kw = inverse_operator_params(ctx.get('inv'))
        for evoked in ds['evoked']:
            inv = invs.setdefault(subject, ctx.load('inv', options={'fiff': evoked}))
            stc = apply_inverse(evoked, inv, **apply_kw)
            if src_baseline:
                mne.baseline.rescale(stc._data, stc.times, src_baseline, 'mean', copy=False)
            if ctx.option('morph', False):
                if subject_from == common_brain:
                    stc.subject = common_brain
                else:
                    stc = source_morph.apply(stc)
            stcs.append(stc)

        if ndvar:
            key = 'srcm' if ctx.option('morph', False) else 'src'
            target_subject = common_brain if ctx.option('morph', False) else mrisubject
            stc_value = load.mne.stc_ndvar(stcs, target_subject, ctx.get('src'), mri_sdir(ctx.state), method, make_kw.get('fixed', False), parc=parc, adjacency=ctx.get('adjacency'))
            if ctx.option('mask', False):
                stc_value = _mask_ndvar(stc_value)
        else:
            key = 'stcm' if ctx.option('morph', False) else 'stc'
            stc_value = stcs
        ds[key] = stc_value
        if ndvar == 1 or not keep_evoked:
            del ds['evoked']
        return ds

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        save.pickle(value, path)
