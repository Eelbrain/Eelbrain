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
from .configuration import Configuration
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
INV_RE = re.compile(
    r"^"
    r"(free|fixed|loose\.\d+|vec)"
    r"(?:-(\d*\.?\d+))?"
    rf"-({'|'.join(INV_METHODS)})"
    r"(?:-((?:0\.)?\d+))?"
    r"(?:-(pick_normal))?"
    r"$"
)


class InverseSolution(Configuration):
    """Internal normalized inverse-operator configuration."""

    @classmethod
    def _coerce(cls, inv: str | InverseSolution) -> InverseSolution:
        if isinstance(inv, InverseSolution):
            return inv
        if isinstance(inv, str):
            return MinimumNormInverseSolution._from_string(inv)
        raise TypeError(f"{inv=}: invalid inverse solution specification")

    def _string(self) -> str:
        raise NotImplementedError(f"{self.__class__.__name__}._string()")

    def _validate_for_source_space(self, src: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}._validate_for_source_space()")

    def _build_operator(
            self,
            info: mne.Info,
            fwd: mne.Forward,
            cov: mne.Covariance,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._build_operator()")

    def _load_operator(self, path: Path):
        raise NotImplementedError(f"{self.__class__.__name__}._load_operator()")

    def _save_operator(self, path: Path, value) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}._save_operator()")

    def _apply_epochs(self, epochs_obj, operator, label=None):
        raise NotImplementedError(f"{self.__class__.__name__}._apply_epochs()")

    def _apply_evoked(self, evoked, operator):
        raise NotImplementedError(f"{self.__class__.__name__}._apply_evoked()")

    def _to_ndvar(
            self,
            stc,
            subject: str,
            src: str,
            subjects_dir: Path,
            *,
            parc: str | None,
            adjacency: str,
    ) -> NDVar:
        return load.mne.stc_ndvar(stc, subject, src, subjects_dir, self.method, self._fixed, parc=parc, adjacency=adjacency)


class MinimumNormInverseSolution(InverseSolution):
    """Normalized minimum-norm inverse configuration."""

    DICT_ATTRS = ('kind', 'ori', 'snr', 'method', 'depth', 'pick_normal')

    def __init__(
            self,
            ori: str | float = 'free',
            snr: float = 3,
            method: str = 'dSPM',
            depth: float = 0.8,
            pick_normal: bool = False,
    ):
        if isinstance(ori, str):
            if ori not in ('free', 'fixed', 'vec'):
                raise ValueError(f"{ori=}; needs to be 'free', 'fixed', 'vec', or float")
        elif not 0 < ori < 1:
            raise ValueError(f"{ori=}; must be in range (0, 1)")
        if snr < 0:
            raise ValueError(f"{snr=}")
        if method not in INV_METHODS:
            raise ValueError(f"{method=}")
        if not 0 <= depth <= 1:
            raise ValueError(f"{depth=}; must be in range [0, 1]")
        if pick_normal and ori in ('vec', 'fixed'):
            raise ValueError(f"{ori=} and pick_normal=True are incompatible")

        self.kind = 'minimum_norm'
        self.ori = ori
        self.snr = snr
        self.method = method
        self.depth = depth
        self.pick_normal = pick_normal

    @classmethod
    def _from_string(cls, inv: str) -> MinimumNormInverseSolution:
        m = INV_RE.match(inv)
        if m is None:
            raise ValueError(f"{inv=}: invalid inverse specification")

        ori, snr, method, depth, pick_normal = m.groups()
        if ori.startswith('loose'):
            ori = float(ori[5:])
            if not 0 < ori < 1:
                raise ValueError(f"{inv=}: loose parameter needs to be in range (0, 1)")

        if snr is None:
            snr = 0
        else:
            snr = float(snr)

        if depth is None:
            depth = 0.8
        else:
            depth = float(depth)

        return cls(ori, snr, method, depth, bool(pick_normal))

    def _string(self) -> str:
        if isinstance(self.ori, str):
            ori = self.ori
        else:
            ori = f'loose{str(self.ori)[1:]}'
        items = [ori]
        if self.snr > 0:
            items.append(f'{self.snr:g}')
        items.append(self.method)
        if self.depth != 0.8:
            items.append(f'{self.depth:g}')
        if self.pick_normal:
            items.append('pick_normal')
        return '-'.join(items)

    def _validate_for_source_space(self, src: str) -> None:
        if src[:3] == 'vol' and self.ori not in ('free', 'vec'):
            raise ValueError(f"{self._string()=!r} with {src=}: volume source space requires free or vector inverse")

    @property
    def _make_kw(self) -> dict[str, Any]:
        if self.ori == 'fixed':
            out = {'fixed': True}
        elif self.ori in ('free', 'vec'):
            out = {'loose': 1}
        else:
            out = {'loose': self.ori}

        if self.depth == 0:
            out['depth'] = None
        else:
            out['depth'] = self.depth
        return out

    @property
    def _apply_kw(self) -> dict[str, Any]:
        out = {'method': self.method, 'lambda2': 1. / self.snr ** 2 if self.snr else 0}
        if self.ori == 'vec':
            out['pick_ori'] = 'vector'
        elif self.pick_normal:
            out['pick_ori'] = 'normal'
        return out

    @property
    def _fixed(self) -> bool:
        return self._make_kw.get('fixed', False)

    def _build_operator(
            self,
            info: mne.Info,
            fwd: mne.Forward,
            cov: mne.Covariance,
    ):
        return make_inverse_operator(info, fwd, cov, use_cps=True, **self._make_kw)

    def _load_operator(self, path: Path):
        return mne.minimum_norm.read_inverse_operator(path)

    def _save_operator(self, path: Path, value) -> None:
        mne.minimum_norm.write_inverse_operator(path, value, overwrite=True)

    def _apply_epochs(self, epochs_obj, operator, label=None):
        return apply_inverse_epochs(epochs_obj, operator, label=label, **self._apply_kw)

    def _apply_evoked(self, evoked, operator):
        return apply_inverse(evoked, operator, **self._apply_kw)


def parse_src(src: str) -> tuple[str, str, str | None]:
    m = SRC_RE.match(src)
    if not m:
        raise ValueError(f'{src=}')
    kind, param, special = m.groups()
    if special and kind != 'vol':
        raise ValueError(f'{src=}')
    return kind, param, special


def eval_src(src: str) -> str:
    parse_src(src)
    return src


def _source_parc(state: dict[str, Any]) -> str | None:
    if state['src'].startswith('vol'):
        return None
    parc = state['parc']
    if not parc:
        raise ValueError("Surface source-space workflows require state parc to be set")
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
        return file_fingerprint(ctx.registry.root, trans_file_path(ctx.state), 'trans-file')


class BemInput(Input):
    name = 'bem-input'

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return file_fingerprint(ctx.registry.root, bem_file_path(ctx.state), 'bem-file')


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
            ctx.load('src', state={'mrisubject': common_brain})
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
        src_to = ctx.load('src', state={'mrisubject': subject_to})
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
        'subject', 'session', 'task', 'acquisition', 'run', 'raw',
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
        'subject', 'session', 'task', 'acquisition', 'run', 'raw',
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
        solution = InverseSolution._coerce(ctx.state['inv'])
        solution._validate_for_source_space(ctx.state['src'])
        fiff = ctx.view_options['fiff']
        if fiff is None:
            fiff = load_raw_dependency(ctx, ctx.state['raw'])
        return solution._build_operator(fiff.info, ctx.load('fwd'), ctx.load(cov_node_name(ctx.state['cov'])))

    def load(
            self,
            ctx: Request,
            path: Path) -> mne.minimum_norm.InverseOperator:
        return InverseSolution._coerce(ctx.state['inv'])._load_operator(path)

    def save(
            self,
            ctx: Request,
            path: Path,
            value: mne.minimum_norm.InverseOperator,
    ) -> None:
        InverseSolution._coerce(ctx.state['inv'])._save_operator(path, value)


def _drop_unknown_labels(y: NDVar):
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
    out = {**state, 'subject': subject}
    mri = out.get('mri')
    if mri not in (None, '', '*'):
        mrisubject = mri_subjects[mri][subject]
        if mrisubject != common_brain and not mrisubject.startswith('sub-'):
            mrisubject = 'sub-' + mrisubject
        out['mrisubject'] = mrisubject
    return out


@dataclass
class SourceProjection:
    solution: InverseSolution
    operator: Any
    label: Any
    subjects_dir: Path
    target_subject: str
    source_morph: SourceMorph | None
    set_subject: str | None
    parc: str | None
    remove_unknown_after_ndvar: bool
    stc_key: str
    src_key: str

    def morph_stcs(self, stc_value):
        values = stc_value if isinstance(stc_value, list) else [stc_value]
        if self.source_morph is not None:
            values = [self.source_morph.apply(stc) for stc in values]
        elif self.set_subject is not None:
            for stc in values:
                stc.subject = self.set_subject
        return values if isinstance(stc_value, list) else values[0]

    def add_to_dataset(self, ctx: Request, ds: Dataset, stc_value, *, variable_time: bool = False) -> None:
        if variable_time:
            src_value = [
                self.solution._to_ndvar(stc, self.target_subject, ctx.state['src'], self.subjects_dir, parc=self.parc, adjacency=ctx.state['adjacency'])
                for stc in stc_value
            ]
        else:
            src_value = self.solution._to_ndvar(stc_value, self.target_subject, ctx.state['src'], self.subjects_dir, parc=self.parc, adjacency=ctx.state['adjacency'])
        if self.remove_unknown_after_ndvar:
            if variable_time:
                src_value = [_drop_unknown_labels(value) for value in src_value]
            else:
                src_value = _drop_unknown_labels(src_value)
        ds[self.stc_key] = stc_value
        ds[self.src_key] = src_value


def _prepare_source_projection(
        ctx: Request,
        fiff: Any,
        morph: bool | None,
        solution: InverseSolution,
) -> SourceProjection:
    subjects_dir = mri_sdir(ctx.state)
    mrisubject = ctx.state['mrisubject']
    source_subject = find_source_subject(mrisubject, subjects_dir) or mrisubject
    is_scaled = source_subject != mrisubject
    target_subject = ctx.state['common_brain'] if morph else mrisubject
    parc = _source_parc(ctx.state)
    if parc:
        ctx.load('annot', state={'mrisubject': target_subject, 'parc': parc})
        if (is_scaled or not morph) and source_subject != target_subject:
            ctx.load('annot', state={'mrisubject': source_subject, 'parc': parc})

    operator = ctx.load('inv', options={'fiff': fiff})
    if parc and (is_scaled or not morph):
        label = label_from_annot(operator['src'], source_subject, subjects_dir, parc)
    else:
        label = None

    source_morph = None
    set_subject = None
    stc_key = 'stc'
    src_key = 'src'
    remove_unknown_after_ndvar = False
    if morph:
        target_subject = ctx.state['common_brain']
        stc_key = 'stcm'
        src_key = 'srcm'
        subject_from = ctx.state['common_brain'] if is_fake_mri(mri_dir(ctx.state)) else mrisubject
        if subject_from == ctx.state['common_brain']:
            set_subject = ctx.state['common_brain']
        else:
            source_morph = ctx.load('source-morph', state={'mrisubject': subject_from})
        remove_unknown_after_ndvar = bool(parc and not is_scaled)

    return SourceProjection(
        solution,
        operator,
        label,
        subjects_dir,
        target_subject,
        source_morph,
        set_subject,
        parc,
        remove_unknown_after_ndvar,
        stc_key,
        src_key,
    )


def _apply_source_baseline(stc_value, baseline) -> None:
    if not baseline:
        return
    values = stc_value if isinstance(stc_value, list) else [stc_value]
    for stc in values:
        mne.baseline.rescale(stc._data, stc.times, baseline, 'mean', copy=False)


def _source_dependencies(ctx: Request, sensor_dependency: Dependency) -> tuple[Dependency, ...]:
    deps = [sensor_dependency, Dependency('inv')]
    parc = _source_parc(ctx.state)
    if parc:
        subjects_dir = mri_sdir(ctx.state)
        mrisubject = ctx.state['mrisubject']
        source_subject = find_source_subject(mrisubject, subjects_dir) or mrisubject
        target_subject = ctx.state['common_brain'] if ctx.options['morph'] else mrisubject
        deps.append(Dependency('annot', state={'mrisubject': target_subject, 'parc': parc}))
        if (source_subject != target_subject) and (source_subject != mrisubject or not ctx.options['morph']):
            deps.append(Dependency('annot', label='source', state={'mrisubject': source_subject, 'parc': parc}))
    if ctx.options['morph'] and (ctx.state['common_brain'] if is_fake_mri(mri_dir(ctx.state)) else ctx.state['mrisubject']) != ctx.state['common_brain']:
        deps.append(Dependency('source-morph'))
    return tuple(deps)


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
        'subject', 'session', 'task', 'acquisition', 'run', 'raw',
        'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv', 'parc',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.pickle'
    OPTION_DEFAULTS = {
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'morph': None,
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
        return _source_dependencies(ctx, Dependency('epochs', options=ctx.options_for('epochs', baseline=ctx.options['baseline'], ndvar=False, reject=ctx.options['reject'], cat=ctx.options['cat'], samplingrate=ctx.options['samplingrate'], decim=ctx.options['decim'], pad=ctx.options['pad'], data_raw=False, data='sensor')))

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize(ctx.options)

    def build(self, ctx: Request) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        solution = InverseSolution._coerce(ctx.state['inv'])
        ds = ctx.load('epochs', options=ctx.options_for('epochs', baseline=ctx.options['baseline'], ndvar=False, reject=ctx.options['reject'], cat=ctx.options['cat'], samplingrate=ctx.options['samplingrate'], decim=ctx.options['decim'], pad=ctx.options['pad'], data_raw=False, data='sensor'))

        src_baseline = ctx.options['src_baseline']
        if not ctx.options['baseline'] and src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError("src_baseline with post_baseline_trigger_shift")
        if src_baseline is True:
            src_baseline = epoch.baseline

        epochs_value = ds['epochs']
        epoch_list = epochs_value if isinstance(epochs_value, Datalist) else [epochs_value]
        variable_time = isinstance(epochs_value, Datalist)
        projection = _prepare_source_projection(ctx, epoch_list[0], ctx.options['morph'], solution)
        stc_value = [solution._apply_epochs(epoch_obj, projection.operator, label=projection.label) for epoch_obj in epoch_list]
        if variable_time:
            stc_value = [value[0] for value in stc_value]
        else:
            stc_value = stc_value[0]
        _apply_source_baseline(stc_value, src_baseline)
        stc_value = projection.morph_stcs(stc_value)
        projection.add_to_dataset(ctx, ds, stc_value, variable_time=variable_time)
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
                sysname = raw_pipe._get_sysname(info, ds.info['subject'], data_kind, self.raw)
                adjacency = raw_pipe._get_adjacency(data_kind, self.raw)
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
        'subject', 'session', 'task', 'acquisition', 'run', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count', 'cov', 'mrisubject',
        'src', 'inv', 'parc',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.pickle'
    OPTION_DEFAULTS = {
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'samplingrate': None,
        'decim': None,
    }
    VIEW_OPTION_DEFAULTS = {'ndvar': True, 'data_raw': False, 'keep_evoked': False}

    def __init__(self, raw, epochs: dict[str, Any]):
        self.raw = raw
        self.epochs = epochs

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return _source_dependencies(ctx, Dependency('evoked', options=ctx.options_for('evoked', baseline=ctx.options['baseline'], ndvar=False, cat=ctx.options['cat'], samplingrate=ctx.options['samplingrate'], decim=ctx.options['decim'], data_raw=False, data='sensor')))

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize(ctx.options)

    def build(self, ctx: Request) -> Dataset:
        solution = InverseSolution._coerce(ctx.state['inv'])
        ds = ctx.load('evoked', options=ctx.options_for('evoked', baseline=ctx.options['baseline'], ndvar=False, cat=ctx.options['cat'], samplingrate=ctx.options['samplingrate'], decim=ctx.options['decim'], data_raw=False, data='sensor'))

        src_baseline = ctx.options['src_baseline']
        epoch = self.epochs[ctx.state['epoch']]
        if src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError(f"{src_baseline=}: post_baseline_trigger_shift is not implemented for baseline correction in source space")
        if src_baseline is True:
            src_baseline = epoch.baseline
        projection = _prepare_source_projection(ctx, ds['evoked'][0], ctx.options['morph'], solution)
        stc_value = [solution._apply_evoked(evoked, projection.operator) for evoked in ds['evoked']]
        _apply_source_baseline(stc_value, src_baseline)
        stc_value = projection.morph_stcs(stc_value)
        projection.add_to_dataset(ctx, ds, stc_value)
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
                sysname = pipe._get_sysname(info, ctx.state['subject'], sensor_type, self.raw)
                adjacency = pipe._get_adjacency(sensor_type, self.raw)
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

    def __init__(self, mri_subjects: dict[str, dict[str, str]], common_brain: str, groups: dict[str, tuple[str, ...]]):
        self.mri_subjects = mri_subjects
        self.common_brain = common_brain
        self.groups = groups

    def key(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize({'parc': ctx.state['parc'], 'subjects': self.groups[ctx.state['group']], 'options': ctx.registry.canonicalize(ctx.options)})

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(ctx, state_fields=('parc',))

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

    def __init__(self, mri_subjects: dict[str, dict[str, str]], common_brain: str, groups: dict[str, tuple[str, ...]]):
        self.mri_subjects = mri_subjects
        self.common_brain = common_brain
        self.groups = groups

    def key(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize({'parc': ctx.state['parc'], 'subjects': self.groups[ctx.state['group']], 'options': ctx.registry.canonicalize(ctx.options)})

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(ctx, state_fields=('parc',))

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
