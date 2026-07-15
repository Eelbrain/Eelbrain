# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model and source-data derivatives.

These nodes own the reusable source-space products behind
``Pipeline.load_inv`` and the source-space branch of
``Pipeline.load_evoked``/``Pipeline.load_epochs`` (selected via a non-empty
``inv``). Higher-level derivatives should load them through
:meth:`Request.load` instead of relying on injected facade methods.

The inverse-solution and source-space configurations they build on live in
:mod:`._experiment.source.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.morph import SourceMorph
from scipy import sparse

from ... import load
from ..._data_obj import Dataset, Datalist, NDVar, combine
from ..derivative_cache import CachePolicy, Dependency, Derivative, ExternalArtifactDerivative, OptionSpec, Request, Input, UncachedDerivative, file_fingerprint
from ..pathing import (
    MRI_SDIR, bem_dir, bem_file_path, mri_dir, src_file_path, trans_file_path,
)
from ..preprocessing import Reference, canonical_recording, raw_node_name
from ..data import DataSpec
from ..._text import enumeration, plural
from ..._utils import subp
from ..._utils.mne_utils import is_fake_mri
from ...mne_fixes._source_space import merge_volume_source_space, prune_volume_source_space, restrict_volume_source_space
from ..._mne import find_source_subject, label_from_annot
from .config import InverseSolution, parse_src


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


class TransInput(Input):
    name = 'trans-input'
    key_fields = ('subject', 'session')

    def path(self, ctx: Request) -> Path:
        return ctx.root / trans_file_path(ctx.state, datatype=ctx.datatype)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return file_fingerprint(ctx.root, self.path(ctx))

    def load(self, ctx: Request) -> mne.transforms.Transform:
        return mne.read_trans(self.path(ctx))


class BemInput(Input):
    name = 'bem-input'
    key_fields = ('mrisubject',)

    def path(self, ctx: Request) -> Path:
        return ctx.root / bem_file_path(ctx.state)

    def _surface_paths(self, ctx: Request) -> dict[str, Path]:
        bem_dir_ = ctx.root / bem_dir(ctx.state)
        return {surf: bem_dir_ / f'{surf}.surf' for surf in ('brain', 'inner_skull', 'outer_skull', 'outer_skin')}

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        subject = ctx.state['mrisubject']
        if subject == 'fsaverage' or is_fake_mri(ctx.root / mri_dir(ctx.state)):
            return file_fingerprint(ctx.root, self.path(ctx))
        return {surf: file_fingerprint(ctx.root, path) for surf, path in self._surface_paths(ctx).items()}

    def load(self, ctx: Request) -> mne.ConductorModel:
        subject = ctx.state['mrisubject']
        if subject == 'fsaverage' or is_fake_mri(ctx.root / mri_dir(ctx.state)):
            return mne.read_bem_surfaces(self.path(ctx))
        bem_dir_ = ctx.root / bem_dir(ctx.state)
        paths = self._surface_paths(ctx)
        missing = [surf for surf, path in paths.items() if not paths.exists()]
        if missing:
            # Test for broken FreeSurfer symlinks
            for surf in missing[:]:
                path = paths[surf]
                if path.is_symlink():
                    new_target = Path('watershed') / f'{subject}_{surf}_surface'
                    if (bem_dir_ / new_target).exists():
                        ctx.registry.log.info("Fixing broken symlink for %s %s surface file", subject, surf)
                        path.unlink()
                        path.symlink_to(new_target)
                        missing.remove(surf)
                    else:
                        ctx.registry.log.error("%s missing for %s", new_target, subject)
            if missing:
                ctx.registry.log.info("%s %s missing for %s. Running mne.make_watershed_bem()...", enumeration(missing).capitalize(), plural('surface', len(missing)), subject)
                os.environ['FREESURFER_HOME'] = subp.get_fs_home()
                mne.bem.make_watershed_bem(subject, ctx.root / MRI_SDIR, overwrite=True)
        return mne.make_bem_model(subject, conductivity=(0.3,), subjects_dir=ctx.root / MRI_SDIR)


class SrcDerivative(ExternalArtifactDerivative[mne.SourceSpaces]):
    name = 'src'
    key_fields = ('mrisubject', 'src')

    def _source_subject(self, ctx: Request) -> str | None:
        """The subject a scaled MRI was scaled from, or ``None`` for a real MRI"""
        return find_source_subject(ctx.state['mrisubject'], ctx.root / MRI_SDIR)

    def path(self, ctx: Request) -> Path:
        return ctx.root / src_file_path(ctx.state)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        source_subject = self._source_subject(ctx)
        if source_subject is not None:
            return Dependency(
                'src',
                label='source-src',
                state={'mrisubject': source_subject},
            ),
        elif ctx.state['src'].startswith('vol'):
            return Dependency('bem-input'),
        return ()

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        out = {'fake_mri': is_fake_mri(ctx.root / mri_dir(ctx.state))}
        if ctx.state['src'].startswith('vol'):
            # volume source spaces are built from the aseg segmentation (see build())
            out['aseg'] = file_fingerprint(ctx.root, ctx.root / mri_dir(ctx.state) / 'mri' / 'aseg.mgz')
        return out

    def build(self, ctx: Request) -> None:
        dst = self.path(ctx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        subject = ctx.state['mrisubject']
        src = ctx.state['src']

        if self._source_subject(ctx) is not None:
            ctx.load('source-src')
            ctx.registry.log.info("Scaling %s source space for %s...", src, subject)
            mne.scale_source_space(subject, f'{{subject}}-{src}-src.fif', subjects_dir=ctx.root / MRI_SDIR, n_jobs=1)
            return

        subjects_dir = ctx.root / MRI_SDIR
        kind, param, special = parse_src(src)
        grade = int(param)
        ctx.registry.log.info("Generating %s source space for %s...", src, subject)
        if kind == 'vol':
            if subject == 'fsaverage':
                bem = ctx.root / bem_file_path(ctx.state)
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
            mri_dir_ = ctx.root / mri_dir(ctx.state)
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
            sss = prune_volume_source_space(sss, grade, 3, remove_midline=remove_midline, fill_holes=4)
        else:
            spacing = kind + param
            sss = mne.setup_source_space(subject, spacing=spacing, add_dist=True, subjects_dir=subjects_dir, n_jobs=1)

        mne.write_source_spaces(dst, sss, overwrite=True)

    def load(
            self,
            ctx: Request,
            path: Path) -> mne.SourceSpaces:
        return mne.read_source_spaces(path)


class SourceMorphDerivative(Derivative[mne.SourceMorph]):
    name = 'source-morph'
    key_fields = ('mrisubject', 'common_brain', 'src')
    cache_suffix = '-morph.h5'

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (
            Dependency('src', label='src-from'),
            Dependency('src', label='src-to', state={'mrisubject': ctx.state['common_brain']}),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'fake_mri': is_fake_mri(ctx.root / mri_dir(ctx.state))}

    def build(self, ctx: Request) -> mne.SourceMorph:
        subject_from = ctx.state['mrisubject']
        subject_to = ctx.state['common_brain']
        subjects_dir = ctx.root / MRI_SDIR
        src_to = ctx.load('src-to')
        src_from = ctx.load('src-from')
        if is_fake_mri(ctx.root / mri_dir(ctx.state)) and subject_from != subject_to:
            return _identity_source_morph(subject_from, subject_to, src_from, src_to)
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


def _eeg_channel_names(info: mne.Info) -> set[str]:
    names = info['ch_names']
    return {names[i] for i in mne.pick_types(info, meg=False, eeg=True, exclude=[])}


class FwdDerivative(Derivative[mne.Forward]):
    name = 'fwd'
    key_fields = ('subject', 'session', 'acquisition', 'mrisubject', 'src')
    cache_suffix = '-fwd.fif'

    def __init__(self, raw, references: dict[str, Reference | None], recordings: frozenset[tuple[str, str, str, str, str]]):
        self.raw = raw
        self._references = references
        self._recordings = recordings

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        # The forward solution only needs the raw sensor info (shared across a
        # subject's recordings), so pin a canonical recording rather than key
        # on the ambient task/run.
        recording = canonical_recording(self._recordings, ctx.state['subject'], ctx.state.get('session'), ctx.state.get('acquisition'))
        raw_state = {'task': recording[0], 'run': recording[1]} if recording else None
        deps = [
            Dependency(raw_node_name('raw'), state=raw_state),
            Dependency('trans-input'),
            Dependency('src'),
            Dependency('canonical-head-position'),
        ]
        # fsaverage uses a precomputed BEM solution (see build/fingerprint); other subjects build it from bem-input
        if ctx.state['mrisubject'] != 'fsaverage':
            deps.append(Dependency('bem-input'))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        out = {'source_reference_add': self._references['average'].add}
        if ctx.state['mrisubject'] == 'fsaverage':
            bemsol = ctx.root / mri_dir(ctx.state) / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
            out['bem_solution'] = file_fingerprint(ctx.root, bemsol)
        return out

    def build(self, ctx: Request) -> mne.Forward:
        raw = ctx.load(raw_node_name('raw'))
        reference = self._references['average']
        if reference.add:
            raw.crop(tmax=0).load_data()
            reference._prepare_source_data(raw, self.raw.root_source_pipe('raw').montage)
        median_head_pos = ctx.load('canonical-head-position')
        info = raw.info
        if median_head_pos is not None:
            info = info.copy()
            with info._unlock():
                info['dev_head_t'] = median_head_pos
        src = ctx.load('src')
        dst = self.path(ctx)
        if ctx.state['mrisubject'] == 'fsaverage':
            bemsol = ctx.root / mri_dir(ctx.state) / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
        else:
            bemsol = mne.make_bem_solution(ctx.load('bem-input'))
        if 'kit_system_id' in info:
            is_kit = (info['kit_system_id'] is not None) or (info['chs'][0]['coil_type'] == FIFF.FIFFV_COIL_KIT_GRAD)
        else:
            raise RuntimeError("Unclear how to set ignor_ref for legacy file without kit_system_id")
        fwd = mne.make_forward_solution(info, ctx.load('trans-input'), src, bemsol, ignore_ref=is_kit)
        for src_part, src_ref in zip(fwd['src'], src):
            if src_part['nuse'] != src_ref['nuse']:
                raise RuntimeError(f"The forward solution {dst.name} contains fewer sources than the source space. This could be due to a corrupted bem file with sources outside of the inner skull surface.")
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
    key_fields = ('subject', 'session', 'acquisition', 'raw', 'epoch', 'epoch_rejection', 'cov', 'mrisubject', 'src', 'inv')
    cache_suffix = '-inv.fif'

    def __init__(self, raw, references: dict[str, Reference | None], recordings: frozenset[tuple[str, str, str, str, str]], cache: bool = True):
        self.raw = raw
        self._references = references
        self._recordings = recordings
        if not cache:
            self.cache_policy = CachePolicy.NEVER

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        # Only the raw sensor info is used (see build), so pin a canonical
        # recording rather than key on the ambient task/run.
        recording = canonical_recording(self._recordings, ctx.state['subject'], ctx.state.get('session'), ctx.state.get('acquisition'))
        raw_state = {'task': recording[0], 'run': recording[1]} if recording else None
        return (
            Dependency(raw_node_name(ctx.state['raw']), label='raw', state=raw_state),
            Dependency('fwd'),
            Dependency('cov'),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'solution': InverseSolution._coerce(ctx.state['inv']),
            'source_reference_add': self._references['average'].add,
        }

    def build(self, ctx: Request) -> mne.minimum_norm.InverseOperator:
        solution = InverseSolution._coerce(ctx.state['inv'])
        solution._validate_for_source_space(ctx.state['src'])
        raw = ctx.load('raw')
        fwd = ctx.load('fwd')
        reference = self._references['average']
        if reference.add:
            # only raw.info is needed for the inverse operator, so crop to a single sample
            raw.crop(tmax=0).load_data()
        reference._prepare_source_data(raw, self.raw.root_source_pipe(ctx.state['raw']).montage)
        if reference.add and _eeg_channel_names(fwd['info']) != _eeg_channel_names(raw.info):
            raise NotImplementedError(f"EEG channels differ between the forward solution and the {ctx.state['raw']!r} raw used for the inverse operator; source localization with a reconstructed reference channel ({reference.add}) requires the inverse raw to keep the same EEG channels as the root 'raw' source used for the forward solution.")
        return solution._build_operator(raw.info, fwd, ctx.load('cov'))

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
        common_brain: str = 'fsaverage',
) -> dict[str, Any]:
    """The state fields to override to switch to ``subject`` (a dependency delta).

    Only the changed fields are returned; the parent state propagates to the
    dependency automatically.
    """
    mri = state['mri']
    mrisubject = mri_subjects[mri][subject]
    if mrisubject != common_brain and not mrisubject.startswith('sub-'):
        mrisubject = 'sub-' + mrisubject
    return {'subject': subject, 'mrisubject': mrisubject}


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
        morph: bool | None,
        solution: InverseSolution,
) -> SourceProjection:
    subjects_dir = ctx.root / MRI_SDIR
    mrisubject = ctx.state['mrisubject']
    source_subject = find_source_subject(mrisubject, subjects_dir) or mrisubject
    is_scaled = source_subject != mrisubject
    target_subject = ctx.state['common_brain'] if morph else mrisubject
    parc = _source_parc(ctx.state)
    if parc:
        ctx.ensure('annot')

    operator = ctx.load('inv')
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
        subject_from = ctx.state['common_brain'] if is_fake_mri(ctx.root / mri_dir(ctx.state)) else mrisubject
        if subject_from == ctx.state['common_brain']:
            set_subject = ctx.state['common_brain']
        else:
            source_morph = ctx.load('source-morph')
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


def _check_head_position_alignment(ctx: Request, info: mne.Info) -> None:
    """Raise if the data's head position doesn't match the canonical session position."""
    median_head_pos = ctx.load('canonical-head-position')
    if median_head_pos is not None:
        if not np.allclose(info['dev_head_t']['trans'], median_head_pos['trans']):
            raise RuntimeError("The data head position does not match the canonical session head position. Apply Maxwell filtering before computing source estimates.")


def _source_dependencies(ctx: Request, sensor_dependency: Dependency) -> tuple[Dependency, ...]:
    deps = [sensor_dependency, Dependency('inv'), Dependency('canonical-head-position')]
    parc = _source_parc(ctx.state)
    if parc:
        if ctx.options['morph']:
            target_subject = ctx.state['common_brain']
        else:
            target_subject = ctx.state['mrisubject']
        deps.append(Dependency('annot', state={'mrisubject': target_subject, 'parc': parc}))
    if ctx.options['morph']:
        source_subject = find_source_subject(ctx.state['mrisubject'], ctx.root / MRI_SDIR)
        if source_subject == ctx.state['common_brain']:
            pass  # no morph required
        else:
            deps.append(Dependency('source-morph'))
    return tuple(deps)


class EpochsStcDerivative(UncachedDerivative[Dataset]):
    """Source-space single-trial dataset derived from epochs.

    Options
    -------
    baseline
        Sensor-space baseline correction before inverse application.
    src_baseline
        Source-space baseline correction after inverse application.
    keep_epochs
        Whether to keep the sensor epochs alongside source output.
    morph
        Whether to morph source data to the common brain.
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
    # source localization handles EEG referencing internally
    fixed_state = {'reference': ''}
    key_options = {
        'baseline': False,
        'src_baseline': False,
        'morph': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'reject': True,
    }
    view_options = {
        'ndvar': True,
        'keep_epochs': False,
    }

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # ``common_brain`` is only used when morphing the estimate to it
        fields = ('subject', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'inv', 'cov', 'raw', 'src', 'parc', 'mrisubject', 'adjacency')
        if ctx.options['morph']:
            fields += ('common_brain',)
        return fields

    def __init__(self, raw, epochs: dict[str, Any], references: dict[str, Reference | None]):
        self.raw = raw
        self.epochs = epochs
        self._references = references

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = ctx.options_for('epochs', 'baseline', 'reject', 'samplingrate', 'decim', 'pad', ndvar=False, data='sensor')
        return _source_dependencies(ctx, Dependency('epochs', options=options))

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'source_reference_add': self._references['average'].add}

    def build(self, ctx: Request) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        solution = InverseSolution._coerce(ctx.state['inv'])
        ds = ctx.load('epochs')
        epochs_value = ds['epochs']
        epoch_list = epochs_value if isinstance(epochs_value, Datalist) else [epochs_value]
        variable_time = isinstance(epochs_value, Datalist)
        reference = self._references['average']
        montage = self.raw.root_source_pipe(ctx.state['raw']).montage
        for epoch_obj in epoch_list:
            reference._prepare_source_data(epoch_obj, montage)
        _check_head_position_alignment(ctx, epoch_list[0].info)

        src_baseline = ctx.options['src_baseline']
        if src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError("src_baseline with post_baseline_trigger_shift")
        if src_baseline is True:
            src_baseline = epoch.baseline
        projection = _prepare_source_projection(ctx, ctx.options['morph'], solution)
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

        if ndvar:
            del ds['stc']
        else:
            del ds['src']

        if keep_epochs in ('ndvar', 'both'):
            epochs_value = ds['epochs']
            epochs_list = epochs_value if isinstance(epochs_value, Datalist) else [epochs_value]
            info = epochs_list[0].info
            sensor_types = DataSpec('sensor').find_ndvar_channel_types(info)
            ds.info['sensor_types'] = sensor_types
            raw_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            for data_kind in sensor_types:
                sysname = raw_pipe._get_sysname(info, ds.info['subject'], data_kind)
                adjacency = raw_pipe._get_adjacency(data_kind)
                if isinstance(epochs_value, Datalist):
                    ys = [load.mne.epochs_ndvar(value, data=data_kind, sysname=sysname, adjacency=adjacency, name=data_kind)[0] for value in epochs_value]
                else:
                    ys = load.mne.epochs_ndvar(epochs_value, data=data_kind, sysname=sysname, adjacency=adjacency)
                ds[data_kind] = ys
            if keep_epochs == 'ndvar':
                del ds['epochs']
        elif not keep_epochs:
            del ds['epochs']

        ds.info.pop('raw', None)
        return ds


class EvokedStcDerivative(UncachedDerivative[Dataset]):
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
    samplingrate
        Sampling rate override for the underlying evoked artifact.
    decim
        Decimation override for the underlying evoked artifact.
    ndvar
        Whether to return source output as NDVars.
    """
    name = 'evoked-stc'
    # source localization handles EEG referencing internally
    fixed_state = {'reference': ''}
    key_options = {
        'model': '',
        'baseline': False,
        'src_baseline': False,
        'morph': False,
        'samplingrate': None,
        'decim': None,
    }
    view_options = {
        'ndvar': True,
        'keep_evoked': False,
        'cat': None,
    }

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # ``common_brain`` is only used when morphing the estimate to it
        fields = ('subject', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'inv', 'cov', 'raw', 'src', 'parc', 'mrisubject', 'adjacency', 'equalize_evoked_count')
        if ctx.options['morph']:
            fields += ('common_brain',)
        return fields

    def __init__(self, raw, epochs: dict[str, Any], references: dict[str, Reference | None]):
        self.raw = raw
        self.epochs = epochs
        self._references = references

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = ctx.options_for('evoked', 'model', 'baseline', 'samplingrate', 'decim')
        return _source_dependencies(ctx, Dependency('evoked', options=options))

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'source_reference_add': self._references['average'].add}

    def build(self, ctx: Request) -> Dataset:
        solution = InverseSolution._coerce(ctx.state['inv'])
        ds = ctx.load('evoked')
        reference = self._references['average']
        montage = self.raw.root_source_pipe(ctx.state['raw']).montage
        for evoked in ds['evoked']:
            reference._prepare_source_data(evoked, montage)
        _check_head_position_alignment(ctx, ds['evoked'][0].info)

        src_baseline = ctx.options['src_baseline']
        epoch = self.epochs[ctx.state['epoch']]
        if src_baseline and epoch.post_baseline_trigger_shift:
            raise NotImplementedError(f"{src_baseline=}: post_baseline_trigger_shift is not implemented for baseline correction in source space")
        if src_baseline is True:
            src_baseline = epoch.baseline
        projection = _prepare_source_projection(ctx, ctx.options['morph'], solution)
        stc_value = [solution._apply_evoked(evoked, projection.operator) for evoked in ds['evoked']]
        _apply_source_baseline(stc_value, src_baseline)
        stc_value = projection.morph_stcs(stc_value)
        projection.add_to_dataset(ctx, ds, stc_value)
        return ds

    def apply_view_options(self, ctx: Request, ds: Dataset) -> Dataset:
        ds = ds.copy()
        cat = ctx.view_options['cat']
        if cat:
            ds = ds.sub(ds.eval(ctx.options['model']).isin(cat))
        ndvar = ctx.view_options['ndvar']
        keep_evoked = ctx.view_options['keep_evoked']
        if ndvar:
            del ds['stc']
        else:
            del ds['src']

        if keep_evoked and ndvar:
            evoked = ds['evoked']
            pipe = self.raw.root_source_pipe(ctx.state['raw'])
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = DataSpec('sensor').find_ndvar_channel_types(info)
            for sensor_type in sensor_types:
                sysname = pipe._get_sysname(info, ctx.state['subject'], sensor_type)
                adjacency = pipe._get_adjacency(sensor_type)
                ds[sensor_type] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
            del ds['evoked']
        elif not keep_evoked:
            del ds['evoked']

        ds.info.pop('raw', None)
        return ds


@dataclass
class ROIData:
    label_data: dict[str, Dataset]
    n_trials_ds: Dataset


class EvokedStcGroupDatasetDerivative(UncachedDerivative[Dataset | ROIData]):
    """Group-level dataset assembled from subject ``evoked-stc`` datasets.

    Options
    -------
    Same options as :class:`EvokedStcDerivative`.

    Notes
    -----
    ``morph`` defaults to ``True``. With ``ndvar=True, morph=False``, source
    NDVars from different brains are retained as a list in the ``src`` column.
    """
    name = 'evoked-stc-group-dataset'
    key_options = {
        **EvokedStcDerivative.key_options,
        **EvokedStcDerivative.view_options,
        'data': OptionSpec(DataSpec('source'), DataSpec),
        'morph': True,
    }

    def __init__(self, mri_subjects: dict[str, dict[str, str]], groups: dict[str, tuple[str, ...]]):
        self.mri_subjects = mri_subjects
        self.groups = groups

    def override_key_fields(self, ctx: Request) -> tuple[str, ...] | None:
        fields = ('group', 'mri', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'equalize_evoked_count', 'inv', 'cov', 'raw', 'src', 'parc', 'mrisubject', 'adjacency')
        if ctx.options['morph']:
            fields += ('common_brain',)
        return fields

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'subjects': tuple(self.groups[ctx.state['group']])}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = ctx.options_for('evoked-stc', *EvokedStcDerivative.key_options, *EvokedStcDerivative.view_options)
        data = ctx.options['data']
        if data.aggregate:
            assert not ctx.options['morph']
        return tuple(
            Dependency('evoked-stc', label=subject, state=_subject_state(ctx.state, subject, self.mri_subjects), options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset | ROIData:
        data = ctx.options['data']
        subjects = self.groups[ctx.state['group']]
        if data is not None and data.aggregate:
            label_dss = {}
            n_trials_dss = []
            for subject in subjects:
                ds = ctx.load(subject)
                roi_data = roi_data_from_dataset(ds, data.aggregate)
                for label, label_ds in roi_data.label_data.items():
                    label_dss.setdefault(label, []).append(label_ds)
                n_trials_dss.append(roi_data.n_trials_ds)
            label_data = {label: combine(label_ds, incomplete='drop') for label, label_ds in label_dss.items()}
            n_trials_ds = combine(n_trials_dss, incomplete='drop')
            return ROIData(label_data, n_trials_ds)
        else:
            dss = [ctx.load(subject) for subject in subjects]
            return combine(dss, to_list=True)


def roi_data_from_dataset(
        ds: Dataset,
        reducer: str,
) -> ROIData:
    """Extract ROI time courses from a group or subject dataset.

    Parameters
    ----------
    ds
        Dataset containing source estimates in ``src``.
        This function removes ``src`` from ``ds``.
    reducer
        NDVar method used to reduce each parcellation label (``'mean'`` or
        ``'rms'``).
    """
    src = ds.pop('src')
    label_data = {}
    for label in src.source.parc.cells:
        if label.startswith('unknown-'):
            continue
        label_ds = ds.copy()
        label_ds['label_tc'] = getattr(src, reducer)(source=label)
        label_data[label] = label_ds
    return ROIData(label_data, ds)
