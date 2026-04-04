# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model and source-data derivatives.

These nodes own the reusable source-space products behind
``Pipeline.load_inv``, ``Pipeline.load_evoked_stc``, and
``Pipeline.load_epochs_stc``. Higher-level derivatives should load them
through :meth:`Request.load` instead of relying on injected facade
methods.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from itertools import product
import os
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
from .derivative_cache import CachePolicy, Dependency, Derivative, Request, Input, UncachedDerivative, file_fingerprint
from .pathing import (
    bem_dir, bem_file_path, mri_dir, mri_sdir, src_file_path, trans_file_path,
)
from .preprocessing import load_raw_dependency, raw_node_name
from .test_def import TestDims
from .._text import enumeration, plural
from .._utils import subp
from .._utils.mne_utils import is_fake_mri
from ..mne_fixes._source_space import merge_volume_source_space, prune_volume_source_space, restrict_volume_source_space
from .._mne import find_source_subject, label_from_annot


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
        ctx: Request,
        mask: bool | str = False,
) -> str | None:
    parc = ctx.options['parc'] or ctx.state['parc'] or None
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
    paths = {surf: bem_dir_ / f'{surf}.surf' for surf in surfs}
    missing = [surf for surf in surfs if not paths[surf].exists()]
    if missing:
        for surf in missing[:]:
            path = paths[surf]
            if path.is_symlink():
                new_target = Path('watershed') / f'{subject}_{surf}_surface'
                if (bem_dir_ / new_target).exists():
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

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return file_fingerprint(ctx.state['root'], trans_file_path(ctx.state), 'trans-file')


class BemInput(Input):
    name = 'bem-input'

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return file_fingerprint(ctx.state['root'], bem_file_path(ctx.state), 'bem-file')


class SrcDerivative(Derivative[mne.SourceSpaces]):
    name = 'src'
    key_fields = ('mrisubject', 'src')

    def _is_scaled(self, ctx: Request) -> bool:
        return ctx.state['mrisubject'] != ctx.state['common_brain'] and is_fake_mri(mri_dir(ctx.state))

    def path(self, ctx: Request) -> Path:
        return src_file_path(ctx.state)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = []
        if self._is_scaled(ctx):
            deps.append(Dependency(
                'src',
                label='common-brain-src',
                state={'mrisubject': ctx.state['common_brain']},
            ))
        elif ctx.state['src'].startswith('vol'):
            deps.append(Dependency('bem-input'))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'mrisubject': ctx.state['mrisubject'],
            'src': ctx.state['src'],
            'common_brain': ctx.state['common_brain'],
            'fake_mri': is_fake_mri(mri_dir(ctx.state)),
        }

    def build(self, ctx: Request) -> mne.SourceSpaces:
        dst = self.path(ctx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        subject = ctx.state['mrisubject']
        common_brain = ctx.state['common_brain']
        src = ctx.state['src']

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
                mri=mri_dir_ / 'mri' / 'aseg.mgz',
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
            ctx: Request,
            path: Path) -> mne.SourceSpaces:
        return mne.read_source_spaces(path)

    def save(
            self,
            ctx: Request,
            path: Path,
            value: mne.SourceSpaces,
    ) -> None:
        mne.write_source_spaces(path, value, overwrite=True)


class SourceMorphDerivative(Derivative[mne.SourceMorph]):
    name = 'source-morph'
    key_fields = ('mrisubject', 'common_brain', 'src')
    cache_suffix = '-morph.h5'

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (
            Dependency('src', label='src-from'),
            Dependency(
                'src',
                label='src-to',
                state={'mrisubject': ctx.state['common_brain']},
            ),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'mrisubject': ctx.state['mrisubject'],
            'common_brain': ctx.state['common_brain'],
            'src': ctx.state['src'],
            'fake_mri': is_fake_mri(mri_dir(ctx.state)),
        }

    def build(self, ctx: Request) -> mne.SourceMorph:
        subject_from = ctx.state['mrisubject']
        subject_to = ctx.state['common_brain']
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
            ctx: Request,
            path: Path) -> mne.SourceMorph:
        return mne.read_source_morph(path)

    def save(
            self,
            ctx: Request,
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

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (
            Dependency(
                raw_node_name('raw'),
                state={'raw': 'raw'},
                options={'add_bads': False},
            ),
            Dependency('trans-input'),
            Dependency('src'),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'raw': ctx.state['raw'],
            'epoch': ctx.state['epoch'],
            'mrisubject': ctx.state['mrisubject'],
            'src': ctx.state['src'],
        }

    def build(self, ctx: Request) -> mne.Forward:
        raw = load_raw_dependency(ctx, ctx.state['raw'], add_bads=False)
        src = ctx.load('src')
        dst = self.path(ctx)
        if ctx.state['mrisubject'] == 'fsaverage':
            bemsol = mri_dir(ctx.state) / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
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
                    f"The forward solution {dst.name} contains fewer sources than the source space. "
                    "This could be due to a corrupted bem file with sources outside of the inner skull surface."
                )
        return fwd

    def load(
            self,
            ctx: Request,
            path: Path) -> mne.Forward:
        return mne.read_forward_solution(path)

    def save(
            self,
            ctx: Request,
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
    VIEW_OPTION_DEFAULTS = {'fiff': None}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('fwd'), Dependency(cov_node_name(ctx.state['cov'])))

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'raw': ctx.state['raw'],
            'epoch': ctx.state['epoch'],
            'rej': ctx.state['rej'],
            'cov': ctx.state['cov'],
            'src': ctx.state['src'],
            'inv': ctx.state['inv'],
        }

    def build(self, ctx: Request) -> mne.minimum_norm.InverseOperator:
        src = ctx.state['src']
        inv = ctx.state['inv']
        if src[:3] == 'vol' and not (inv.startswith('vec') or inv.startswith('free')):
            raise ValueError(f'{inv=} with {src=}: volume source space requires free or vector inverse')
        fiff = ctx.view_options['fiff']
        if fiff is None:
            fiff = load_raw_dependency(ctx, ctx.state['raw'])
        _, make_kw, _ = inverse_operator_params(inv)
        return make_inverse_operator(
            fiff.info,
            ctx.load('fwd'),
            ctx.load(cov_node_name(ctx.state['cov'])),
            use_cps=True,
            **make_kw,
        )

    def load(
            self,
            ctx: Request,
            path: Path) -> mne.minimum_norm.InverseOperator:
        return mne.minimum_norm.read_inverse_operator(path)

    def save(
            self,
            ctx: Request,
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


def _subject_state(
        state: dict[str, Any],
        subject: str,
        mri_subjects: dict[str, dict[str, str]],
        common_brain: str,
) -> dict[str, Any]:
    out = {**state, 'subject': subject, 'group': None}
    mri = out.get('mri')
    if mri not in (None, '', '*'):
        mrisubject = mri_subjects[mri][subject]
        if mrisubject != common_brain and not mrisubject.startswith('sub-'):
            mrisubject = 'sub-' + mrisubject
        out['mrisubject'] = mrisubject
    return out


def _prepare_inv(
        ctx: Request,
        fiff: Any,
        mask: bool | str,
        morph: bool | None,
):
    parc = _selected_parc(ctx, mask)
    if parc:
        ctx.load('annot', state={'parc': parc})

    inv = ctx.load('inv', options={'fiff': fiff})
    subjects_dir = mri_sdir(ctx.state)
    mrisubject = ctx.state['mrisubject']
    is_scaled = find_source_subject(mrisubject, subjects_dir)
    if mask and (is_scaled or not morph):
        label = label_from_annot(inv['src'], mrisubject, subjects_dir, parc)
    else:
        label = None
    return inv, label, subjects_dir, mrisubject, is_scaled, parc


class EpochsStcDerivative(Derivative[Dataset]):
    """Source-space single-trial dataset derived from cached epochs.

    Options
    -------
    baseline
        Sensor-space baseline correction before inverse application.
    src_baseline
        Source-space baseline correction after inverse application.
    cat
        Optional subset of model cells to keep.
    keep_epochs
        Whether to keep the sensor epochs alongside source output.
    morph
        Whether to morph source data to the common brain.
    mask
        Optional source-space mask/parcellation to apply.
    data_raw
        Whether to keep raw objects in the dataset info.
    samplingrate
        Sampling rate override for the underlying epochs artifact.
    decim
        Decimation override for the underlying epochs artifact.
    pad
        Extra time padding to add before epoch extraction.
    ndvar
        Whether to return source output as NDVars.
    reject
        Whether to apply epoch rejection/interpolation state.
    """
    name = 'epochs-stc'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.pickle'
    OPTION_DEFAULTS = {
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'parc': None,
        'morph': None,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'reject': True,
    }
    VIEW_OPTION_DEFAULTS = {'ndvar': True, 'data_raw': False, 'keep_epochs': False}

    def __init__(self, raw, epochs: dict[str, Any]):
        self.raw = raw
        self.epochs = epochs

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = [
            Dependency('epochs-dataset', options=ctx.options_for(
                'epochs-dataset',
                baseline=ctx.options['baseline'],
                ndvar=False,
                reject=ctx.options['reject'],
                cat=ctx.options['cat'],
                samplingrate=ctx.options['samplingrate'],
                decim=ctx.options['decim'],
                pad=ctx.options['pad'],
                data_raw=False,
                data='sensor',
            )),
            Dependency('inv'),
        ]
        mask = ctx.options['mask']
        if mask:
            parc = _selected_parc(ctx, mask)
            deps.append(Dependency('annot', state={'parc': parc}))
        if ctx.options['morph']:
            deps.append(Dependency('source-morph'))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize(ctx.options)

    def build(self, ctx: Request) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        ds = ctx.load('epochs-dataset', options=ctx.options_for(
            'epochs-dataset',
            baseline=ctx.options['baseline'],
            ndvar=False,
            reject=ctx.options['reject'],
            cat=ctx.options['cat'],
            samplingrate=ctx.options['samplingrate'],
            decim=ctx.options['decim'],
            pad=ctx.options['pad'],
            data_raw=False,
            data='sensor',
        ))

        src_baseline = ctx.options['src_baseline']
        if not ctx.options['baseline'] and src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError("src_baseline with post_baseline_trigger_shift")
        if src_baseline is True:
            src_baseline = epoch.baseline

        epoch_list = ds['epochs'] if isinstance(ds['epochs'], Datalist) else [ds['epochs']]
        inv, label, subjects_dir, mrisubject, is_scaled, parc = _prepare_inv(ctx, epoch_list[0], ctx.options['mask'], ctx.options['morph'])
        method, make_kw, apply_kw = inverse_operator_params(ctx.state['inv'])
        stc_list = [apply_inverse_epochs(epoch_obj, inv, label=label, **apply_kw) for epoch_obj in epoch_list]
        is_variable_time = isinstance(ds['epochs'], Datalist)
        if is_variable_time:
            stc_list = [stc for stc, in stc_list]

        if src_baseline:
            for value in stc_list:
                values = value if isinstance(value, list) else [value]
                for stc in values:
                    mne.baseline.rescale(stc._data, stc.times, src_baseline, 'mean', copy=False)

        if ctx.options['morph']:
            common_brain = ctx.state['common_brain']
            target_subject = common_brain
            ctx.load('annot', state={'mrisubject': common_brain})
            subject_from = common_brain if is_fake_mri(mri_dir(ctx.state)) else mrisubject
            if subject_from == common_brain:
                for value in stc_list:
                    values = value if isinstance(value, list) else [value]
                    for stc in values:
                        stc.subject = common_brain
            else:
                source_morph = ctx.load('source-morph')
                stc_list = [
                    [source_morph.apply(stc) for stc in value] if isinstance(value, list) else source_morph.apply(value)
                    for value in stc_list
                ]
            stc_key = 'stcm'
            src_key = 'srcm'
        else:
            target_subject = mrisubject
            stc_key = 'stc'
            src_key = 'src'

        src = ctx.state['src']
        ndvar_list = [
            load.mne.stc_ndvar(value, target_subject, src, subjects_dir, method, make_kw.get('fixed', False), parc=parc, adjacency=ctx.state['adjacency'])
            for value in stc_list
        ]
        if ctx.options['mask'] and ctx.options['morph'] and not is_scaled:
            ndvar_list = [_mask_ndvar(value) for value in ndvar_list]

        ds[stc_key] = stc_list if is_variable_time else stc_list[0]
        ds[src_key] = ndvar_list if is_variable_time else ndvar_list[0]
        return ds

    def apply_view_options(self, ctx: Request, ds: Dataset) -> Dataset:
        ds = ds.copy()
        ndvar = ctx.view_options['ndvar']
        keep_epochs = ctx.view_options['keep_epochs']
        if keep_epochs not in (True, False, 'ndvar', 'both'):
            raise ValueError(f"{keep_epochs=}")

        stc_key = 'stcm' if 'stcm' in ds else 'stc'
        src_key = 'srcm' if 'srcm' in ds else 'src'
        if ndvar:
            del ds[stc_key]
        else:
            del ds[src_key]

        if keep_epochs in ('ndvar', 'both'):
            epochs_value = ds['epochs']
            epochs_list = epochs_value if isinstance(epochs_value, Datalist) else [epochs_value]
            info = epochs_list[0].info
            sensor_types = TestDims.coerce('sensor').data_to_ndvar(info)
            ds.info['sensor_types'] = sensor_types
            raw_pipe = self.raw[ctx.state['raw']]
            for data_kind in sensor_types:
                sysname = raw_pipe.get_sysname(info, ds.info['subject'], data_kind, self.raw)
                adjacency = raw_pipe.get_adjacency(data_kind, self.raw)
                name = 'meg' if data_kind == 'mag' and 'grad' not in sensor_types else data_kind
                if isinstance(epochs_value, Datalist):
                    ys = [load.mne.epochs_ndvar(value, data=data_kind, sysname=sysname, adjacency=adjacency, name=data_kind)[0] for value in epochs_value]
                else:
                    ys = load.mne.epochs_ndvar(epochs_value, data=data_kind, sysname=sysname, adjacency=adjacency)
                ds[name] = ys
            if keep_epochs == 'ndvar':
                del ds['epochs']
        elif not keep_epochs:
            del ds['epochs']

        if ctx.view_options['data_raw']:
            ds.info['raw'] = load_raw_dependency(ctx, add_bads=True, preload=False, noise=False)
        else:
            ds.info.pop('raw', None)
        return ds

    def load(self, ctx: Request, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class EvokedStcDerivative(Derivative[Dataset]):
    """Source-space evoked dataset derived from cached evokeds.

    Options
    -------
    baseline
        Sensor-space baseline correction before inverse application.
    src_baseline
        Source-space baseline correction after inverse application.
    cat
        Optional subset of model cells to keep.
    keep_evoked
        Whether to keep the sensor evoked data alongside source output.
    morph
        Whether to morph source data to the common brain.
    mask
        Optional source-space mask/parcellation to apply.
    data_raw
        Whether to keep raw objects in the dataset info.
    samplingrate
        Sampling rate override for the underlying evoked artifact.
    decim
        Decimation override for the underlying evoked artifact.
    ndvar
        Whether to return source output as NDVars.
    """
    name = 'evoked-stc'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count', 'cov', 'mrisubject',
        'src', 'inv',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.pickle'
    OPTION_DEFAULTS = {
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'parc': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
    }
    VIEW_OPTION_DEFAULTS = {'ndvar': True, 'data_raw': False, 'keep_evoked': False}

    def __init__(self, raw, epochs: dict[str, Any]):
        self.raw = raw
        self.epochs = epochs

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = [
            Dependency('evoked-dataset', options=ctx.options_for(
                'evoked-dataset',
                baseline=ctx.options['baseline'],
                ndvar=False,
                cat=ctx.options['cat'],
                samplingrate=ctx.options['samplingrate'],
                decim=ctx.options['decim'],
                data_raw=False,
                data='sensor',
            )),
            Dependency('inv'),
        ]
        mask = ctx.options['mask']
        if mask:
            parc = _selected_parc(ctx, mask)
            deps.append(Dependency('annot', state={'parc': parc}))
        if ctx.options['morph']:
            deps.append(Dependency('source-morph'))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize(ctx.options)

    def build(self, ctx: Request) -> Dataset:
        ds = ctx.load('evoked-dataset', options=ctx.options_for(
            'evoked-dataset',
            baseline=ctx.options['baseline'],
            ndvar=False,
            cat=ctx.options['cat'],
            samplingrate=ctx.options['samplingrate'],
            decim=ctx.options['decim'],
            data_raw=False,
            data='sensor',
        ))

        src_baseline = ctx.options['src_baseline']
        epoch = self.epochs[ctx.state['epoch']]
        if src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError(f"{src_baseline=}: post_baseline_trigger_shift is not implemented for baseline correction in source space")
        if src_baseline is True:
            src_baseline = epoch.baseline
        invs = {}
        stcs = []
        subject = ctx.state['subject']
        mrisubject = ctx.state['mrisubject']
        common_brain = ctx.state['common_brain']
        if is_fake_mri(mri_dir(ctx.state)):
            subject_from = common_brain
        else:
            subject_from = mrisubject
        parc = _selected_parc(ctx, ctx.options['mask'])
        target_subject = common_brain if ctx.options['morph'] else mrisubject
        if parc:
            ctx.load('annot', state={'mrisubject': target_subject, 'parc': parc})
        source_morph = None
        if ctx.options['morph'] and subject_from != common_brain:
            source_morph = ctx.load('source-morph', state={'mrisubject': subject_from})

        method, make_kw, apply_kw = inverse_operator_params(ctx.state['inv'])
        for evoked in ds['evoked']:
            inv = invs.setdefault(subject, ctx.load('inv', options={'fiff': evoked}))
            stc = apply_inverse(evoked, inv, **apply_kw)
            if src_baseline:
                mne.baseline.rescale(stc._data, stc.times, src_baseline, 'mean', copy=False)
            if ctx.options['morph']:
                if subject_from == common_brain:
                    stc.subject = common_brain
                else:
                    stc = source_morph.apply(stc)
            stcs.append(stc)

        src_key = 'srcm' if ctx.options['morph'] else 'src'
        stc_key = 'stcm' if ctx.options['morph'] else 'stc'
        ds[src_key] = load.mne.stc_ndvar(stcs, target_subject, ctx.state['src'], mri_sdir(ctx.state), method, make_kw.get('fixed', False), parc=parc, adjacency=ctx.state['adjacency'])
        if ctx.options['mask']:
            ds[src_key] = _mask_ndvar(ds[src_key])
        ds[stc_key] = stcs
        return ds

    def apply_view_options(self, ctx: Request, ds: Dataset) -> Dataset:
        ds = ds.copy()
        ndvar = ctx.view_options['ndvar']
        keep_evoked = ctx.view_options['keep_evoked']
        stc_key = 'stcm' if 'stcm' in ds else 'stc'
        src_key = 'srcm' if 'srcm' in ds else 'src'
        if ndvar:
            del ds[stc_key]
        else:
            del ds[src_key]

        if keep_evoked and ndvar:
            evoked = ds['evoked']
            pipe = self.raw[ctx.state['raw']]
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = TestDims.coerce('sensor').data_to_ndvar(info)
            for sensor_type in sensor_types:
                sysname = pipe.get_sysname(info, ctx.state['subject'], sensor_type, self.raw)
                adjacency = pipe.get_adjacency(sensor_type, self.raw)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
            del ds['evoked']
        elif not keep_evoked:
            del ds['evoked']

        if ctx.view_options['data_raw']:
            ds.info['raw'] = load_raw_dependency(ctx, add_bads=True, preload=False, noise=False)
        else:
            ds.info.pop('raw', None)
        return ds

    def load(self, ctx: Request, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class EpochsStcGroupDatasetDerivative(UncachedDerivative[Dataset]):
    """Group-level dataset assembled from subject ``epochs-stc`` datasets.

    Options
    -------
    Same options as :class:`EpochsStcDerivative`.

    Notes
    -----
    ``data_raw`` and ``keep_epochs`` must be falsey, and
    ``morph`` defaults to ``True`` when omitted.
    """
    name = 'epochs-stc-group-dataset'
    OPTION_DEFAULTS = {**EpochsStcDerivative.OPTION_DEFAULTS, **EpochsStcDerivative.VIEW_OPTION_DEFAULTS}

    def __init__(self, groups: dict[str, Sequence[str]], mri_subjects: dict[str, dict[str, str]], common_brain: str):
        self.groups = groups
        self.mri_subjects = mri_subjects
        self.common_brain = common_brain

    def key(self, ctx: Request) -> dict[str, Any]:
        group = ctx.state['group']
        if group in (None, '', '*'):
            raise RuntimeError(f"{self.name!r} requires an explicit group")
        return ctx.registry.canonicalize({'group': group, 'options': ctx.registry.canonicalize(ctx.options)})

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(ctx, state_fields=('group',))

    def _group_options(self, ctx: Request) -> dict[str, Any]:
        data_raw = ctx.options['data_raw']
        if data_raw:
            raise ValueError(f"data_raw={data_raw!r} with group: Can not combine raw data from multiple subjects.")
        keep_epochs = ctx.options['keep_epochs']
        if keep_epochs:
            raise ValueError(f"keep_epochs={keep_epochs!r} with group: Can not combine Epochs objects for different subjects. Set keep_epochs=False (default).")
        morph = ctx.options['morph']
        if morph is None:
            return ctx.options_for('epochs-stc', *EpochsStcDerivative.OPTION_DEFAULTS, *EpochsStcDerivative.VIEW_OPTION_DEFAULTS, morph=True)
        if not morph:
            raise ValueError(f"morph={morph!r} with group: Source estimates can only be combined after morphing data to common brain model. Set morph=True.")
        return ctx.options_for('epochs-stc', *EpochsStcDerivative.OPTION_DEFAULTS, *EpochsStcDerivative.VIEW_OPTION_DEFAULTS)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = self._group_options(ctx)
        return tuple(
            Dependency('epochs-stc', label=subject, state=_subject_state(ctx.state, subject, self.mri_subjects, self.common_brain), options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        options = self._group_options(ctx)
        dss = [
            ctx.load('epochs-stc', state=_subject_state(ctx.state, subject, self.mri_subjects, self.common_brain), options=options)
            for subject in self.groups[ctx.state['group']]
        ]
        return combine(dss)


class EvokedStcGroupDatasetDerivative(UncachedDerivative[Dataset]):
    """Group-level dataset assembled from subject ``evoked-stc`` datasets.

    Options
    -------
    Same options as :class:`EvokedStcDerivative`.

    Notes
    -----
    ``ndvar=True`` requires morphing to a common brain,
    and ``morph`` defaults to ``True`` when omitted in that case.
    """
    name = 'evoked-stc-group-dataset'
    OPTION_DEFAULTS = {**EvokedStcDerivative.OPTION_DEFAULTS, **EvokedStcDerivative.VIEW_OPTION_DEFAULTS}

    def __init__(self, groups: dict[str, Sequence[str]], mri_subjects: dict[str, dict[str, str]], common_brain: str):
        self.groups = groups
        self.mri_subjects = mri_subjects
        self.common_brain = common_brain

    def key(self, ctx: Request) -> dict[str, Any]:
        group = ctx.state['group']
        if group in (None, '', '*'):
            raise RuntimeError(f"{self.name!r} requires an explicit group")
        return ctx.registry.canonicalize({'group': group, 'options': ctx.registry.canonicalize(ctx.options)})

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(ctx, state_fields=('group',))

    def _group_options(self, ctx: Request) -> dict[str, Any]:
        morph = ctx.options['morph']
        if ctx.options['ndvar']:
            if morph is None:
                return ctx.options_for('evoked-stc', *EvokedStcDerivative.OPTION_DEFAULTS, *EvokedStcDerivative.VIEW_OPTION_DEFAULTS, morph=True)
            if not morph:
                raise ValueError("ndvar=True, morph=False with multiple subjects: Can't create ndvars with data from different brains")
        return ctx.options_for('evoked-stc', *EvokedStcDerivative.OPTION_DEFAULTS, *EvokedStcDerivative.VIEW_OPTION_DEFAULTS)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = self._group_options(ctx)
        return tuple(
            Dependency('evoked-stc', label=subject, state=_subject_state(ctx.state, subject, self.mri_subjects, self.common_brain), options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        options = self._group_options(ctx)
        dss = [
            ctx.load('evoked-stc', state=_subject_state(ctx.state, subject, self.mri_subjects, self.common_brain), options=options)
            for subject in self.groups[ctx.state['group']]
        ]
        return combine(dss)


def roi_data_from_subject_datasets(dss: Sequence[Dataset], reducer: str) -> ROIData:
    n_trials_dss = []
    label_dss = {}
    for ds in dss:
        n_trials_dss.append(ds)
        ds_n = ds.copy()
        src = ds_n.pop(next(name for name in ('srcm', 'src', 'stcm', 'stc') if name in ds_n))
        for label in src.source.parc.cells:
            if label.startswith('unknown-'):
                continue
            label_ds = ds_n.copy()
            label_ds['label_tc'] = getattr(src, reducer)(source=label)
            label_dss.setdefault(label, []).append(label_ds)
    return ROIData({label: combine(label_ds, incomplete='drop') for label, label_ds in label_dss.items()}, combine(n_trials_dss, incomplete='drop'))


@dataclass
class ROIData:
    label_data: dict[str, Dataset]
    n_trials_ds: Dataset
