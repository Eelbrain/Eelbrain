# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Raw preprocessing graph nodes.

Each configured source :class:`~._experiment.preprocessing.config.RawPipe`
produces one raw input node, and each configured processed ``RawPipe`` produces
one raw derivative node; :class:`~._experiment.preprocessing.config.RawICA`
additionally produces an ICA input node. These graph nodes use the bound
``RawPipe`` objects to build and load concrete artifacts, managing artifact
identity, dependency edges, and cache integration.

The configuration classes these nodes build on live in
:mod:`._experiment.preprocessing.config`.
"""
from __future__ import annotations
from datetime import datetime
import itertools
import json
import logging
from pathlib import Path
from typing import Any
import warnings
from collections.abc import Sequence

import mne
import mne_bids
from mne_bids import BIDSPath
import numpy
import pandas as pd
from scipy.spatial.transform import Rotation

from ..._exceptions import DataError
from ..derivative_cache import (
    ArtifactManifest, CachePolicy, Dependency, Derivative, UncachedDerivative,
    Request, Input, MANIFEST_SCHEMA_VERSION, ProtectedArtifactError,
    canonical_state_subset, compare_manifests, file_fingerprint,
)
from ..logging import find_difference, format_difference_path
from ..exceptions import FileMissingError
from ..pathing import bids_path, DERIV_DIR
from .config import (
    MNE_VERBOSITY, RawPipeGraph, RawSource, CachedRawPipe, RawICA, RawApplyICA, RawMaxwell,
    raw_node_name, raw_bad_channels_input_name, raw_input_name, ica_input_name,
)

LOG = logging.getLogger(__name__)
REINDEX_ICA = 'reindex_ica'
# Scaling factors from BIDS coordinate units to metres
COORD_SCALE = {'mm': 1e-3, 'cm': 1e-2, 'm': 1.0}
BIDS_MEG_CHANNEL_TYPES = (
    'MEGGRADAXIAL', 'MEGMAG', 'MEGGRAD', 'MEGREFGRADAXIAL',
    'MEGGRADPLANAR', 'MEGREFMAG', 'MEGOTHER', 'HLU',
)
# Adapted from mne-bids _get_ch_type_mapping (BSD-3-Clause; MNE-BIDS developers).
BIDS_TO_MNE_CHANNEL_TYPES = {
    'EEG': 'eeg',
    'MISC': 'misc',
    'TRIG': 'stim',
    'EMG': 'emg',
    'ECOG': 'ecog',
    'SEEG': 'seeg',
    'EOG': 'eog',
    'ECG': 'ecg',
    'RESP': 'resp',
    'GSR': 'gsr',
    'TEMP': 'temperature',
    'NIRSCWAMPLITUDE': 'fnirs_cw_amplitude',
    'NIRS': 'fnirs_cw_amplitude',
    'VEOG': 'eog',
    'HEOG': 'eog',
    'DBS': 'dbs',
    'EYEGAZE': 'eyegaze',
    'PUPIL': 'pupil',
}


def canonical_recording(recordings: frozenset[tuple[str, str, str, str, str]], subject: str, session: str | None, acquisition: str | None) -> tuple[str, str] | None:
    """Return a deterministic ``(task, run)`` recording for one subject/session/acquisition.

    Used to pin an info-only raw load (forward/inverse/covariance) to a single
    representative recording, so the derivative's identity does not depend on the
    ambient ``task``/``run``. The sensor ``info`` (channel geometry) is shared
    across a subject's recordings, so any existing recording is equivalent.

    Parameters
    ----------
    recordings
        Existing ``(subject, session, task, acquisition, run)`` recordings.
    subject
        Subject to select a recording for.
    session
        Session to select a recording for (``None`` is treated as ``''``).
    acquisition
        Acquisition to select a recording for (``None`` is treated as ``''``).

    Returns
    -------
    The first ``(task, run)`` in sorted order for the subject/session/acquisition, or
    ``None`` when no recording exists.
    """
    matches = sorted((task, run) for subject_, session_, task, acquisition_, run in recordings if subject_ == subject and session_ == (session or '') and acquisition_ == (acquisition or ''))
    return matches[0] if matches else None


class RawBadChannelsInput(Input[list[str]]):
    """Access to Pipeline-specific bad channel definitions.

    User-specified bad channels are stored in an Eelbrain-specific  ``channels.tsv`` file under the ``derivatives/mne/`` hierarchy  rather than in the BIDS source dataset, so that re-downloading the dataset does not overwrite them.
    The BIDS source ``channels.tsv`` is used as seed when the derivatives file is first written.
    """
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run')
    key_options = {'noise': False}

    def __init__(
            self,
            raw_name: str,
            pipe: RawSource,
            extension: str,
    ):
        self.name = raw_bad_channels_input_name(raw_name)
        self.raw_name = raw_name
        self.fixed_state = {'raw': raw_name}
        self.pipe = pipe
        self.extension = extension

    def path(self, ctx: Request) -> Path:
        """Path to the Pipeline-specific bad-channels ``channels.tsv`` file."""
        # Same sidecar as the BIDS source, relocated under derivatives/mne so the source dataset is never modified.
        bpath = self._bids_path(ctx)
        return bpath.update(root=ctx.root / DERIV_DIR / 'mne', check=False).fpath

    def _bids_path(self, ctx: Request) -> BIDSPath:
        """Noise-resolved ``channels.tsv`` :class:`BIDSPath` in the source dataset."""
        bpath = bids_path(ctx.root, ctx.state, self.extension, datatype=ctx.datatype, noise=ctx.options['noise'])
        return bpath.update(suffix='channels', extension='.tsv')

    def _active_path(self, ctx: Request) -> Path:
        """The file ``load`` reads from: derivatives file if present, else BIDS source."""
        path = self.path(ctx)
        if path.exists():
            return path
        return self._bids_path(ctx).fpath

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'bads': self.load(ctx)}

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict[str, Any] | None:
        return file_fingerprint(ctx.root, self._active_path(ctx))

    def load(self, ctx: Request) -> list[str]:
        path = self._active_path(ctx)
        if not path.exists():
            return []
        channels_df = pd.read_csv(path, sep='\t')
        if 'status' not in channels_df.columns:
            return []
        if 'name' not in channels_df.columns:
            raise RuntimeError(f"channels.tsv file at {path} is missing required column 'name'.")
        return channels_df.query('status == "bad"')['name'].tolist()

    def write(
            self,
            ctx: Request,
            raw: mne.io.BaseRaw,
            new_bads: list[str],
            redo: bool,
            *,
            create: bool = False,
    ) -> None:
        """Write bad-channel status to the Pipeline-specific ``channels.tsv`` file.

        Bad channels are written to the ``derivatives/mne/`` hierarchy so
        that the BIDS source dataset is never modified. With ``create=True``, a
        missing file is initialized from the BIDS source ``channels.tsv`` (to
        preserve any bad channels shipped with the dataset), or from ``raw`` if
        no source sidecar exists, so the resulting file contains one row for
        every channel in the recording.
        Channel names in ``new_bads`` are normalized against the raw file using
        the associated :class:`RawSource`. By default, new bad channels are
        added to any channels that are already marked bad. With ``redo=True``,
        all channels are first reset to good and only ``new_bads`` are marked
        bad.

        Parameters
        ----------
        ctx
            Request describing the recording and ``noise`` option.
        raw
            Raw file used to validate channel names and initialize a missing
            ``channels.tsv`` file.
        new_bads
            Channels to mark bad.
        redo
            Replace existing bad-channel markings instead of adding to them.
        create
            Create a missing ``channels.tsv`` file before writing.
        """
        path = self.path(ctx)
        if path.exists():
            channels_df = pd.read_csv(path, sep='\t')
            if 'name' not in channels_df.columns:
                raise RuntimeError(f"channels.tsv file at {path} is missing required column 'name'.")
            if 'status' not in channels_df.columns:
                channels_df['status'] = 'good'
            created = False
        elif create:
            source_path = self._bids_path(ctx).fpath
            if source_path.exists():
                LOG.info("No bad-channels file found at %s, seeding from BIDS source %s.", path, source_path)
                channels_df = pd.read_csv(source_path, sep='\t')
                if 'name' not in channels_df.columns:
                    raise RuntimeError(f"channels.tsv file at {source_path} is missing required column 'name'.")
                if 'status' not in channels_df.columns:
                    channels_df['status'] = 'good'
            else:
                LOG.info("No bad-channels file found at %s, creating one from raw.", path)
                ch_status = ['bad' if ch in raw.info['bads'] else 'good' for ch in raw.ch_names]
                channels_df = pd.DataFrame({'name': raw.ch_names, 'status': ch_status})
            path.parent.mkdir(parents=True, exist_ok=True)
            created = True
        else:
            raise FileMissingError(f"Bad channels file does not exist at {path}")

        old_bads = channels_df.query('status == "bad"')['name'].tolist()
        new_bads = self.pipe._normalize_channel_names(raw, new_bads)
        if not redo:
            new_bads = sorted(set(old_bads).union(new_bads))
        LOG.info("Bad channels: %s -> %s for %s", old_bads, new_bads, path)
        if new_bads == old_bads and not created:
            return

        missing = [ch for ch in new_bads if ch not in set(channels_df['name'])]
        if missing:
            raise RuntimeError(f"channels.tsv file at {path} is missing bad channel names: {missing!r}.")
        if redo:
            channels_df['status'] = 'good'
        channels_df.loc[channels_df['name'].isin(new_bads), 'status'] = 'bad'
        channels_df.to_csv(path, sep='\t', index=False)


class RawSourceInput(Input[mne.io.BaseRaw]):
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run')
    key_options = {'noise': False}
    view_options = {'preload': False}

    def __init__(
            self,
            raw_name: str,
            pipe: RawSource,
            extension: str,
    ):
        self.name = raw_input_name(raw_name)
        self.raw_name = raw_name
        self.fixed_state = {'raw': raw_name}
        self.pipe = pipe
        self.extension = extension

    def _resolve_bids_path(self, ctx: Request, require: bool = False) -> BIDSPath:
        """Return the noise-resolved BIDSPath and the actual file path on disk."""
        bids_path_ = bids_path(ctx.root, ctx.state, self.extension, datatype=ctx.datatype)
        if ctx.options['noise']:
            bids_path_ = bids_path_.find_empty_room()
        if bids_path_.fpath.exists():
            return bids_path_
        # Alternative path: split files
        split_path = bids_path_.copy().update(split='01')
        if split_path.fpath.exists():
            return split_path
        if require:
            raise FileMissingError(f"Raw input file does not exist at expected location {bids_path_.fpath}")
        return bids_path_

    def path(self, ctx: Request) -> Path:
        return self._resolve_bids_path(ctx).fpath

    @staticmethod
    def _read_raw(path: BIDSPath, preload: bool) -> mne.io.BaseRaw:
        """Read a raw file using the MNE reader appropriate for its BIDS extension."""
        kwargs = {'preload': preload, 'verbose': MNE_VERBOSITY}
        match path.extension:
            case '.fif':
                reader = mne.io.read_raw_fif
                kwargs['allow_maxshield'] = True
            case '.edf':
                reader = mne.io.read_raw_edf
            case '.vhdr':
                reader = mne.io.read_raw_brainvision
            case '.set':
                reader = mne.io.read_raw_eeglab
            case '.bdf':
                reader = mne.io.read_raw_bdf
            case _:
                raise RuntimeError(f"Unrecognized file format: {path.extension}")
        return reader(path.fpath, **kwargs)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        path = self._resolve_bids_path(ctx)
        fp = {
            'raw': self.raw_name,
            'pipe': self.pipe,
            'source': file_fingerprint(ctx.root, path.fpath),
        }
        channels_path = self._find_bids_channels(path)
        if channels_path is not None:
            fp['channels'] = file_fingerprint(ctx.root, channels_path)
        if path.datatype == 'eeg':
            elec_pair = self._find_bids_electrodes(path)
            if elec_pair is not None:
                elec_path, coord_path = elec_pair
                fp['electrodes'] = file_fingerprint(ctx.root, elec_path)
                fp['coordsystem'] = file_fingerprint(ctx.root, coord_path)
        return fp

    def load(self, ctx: Request) -> mne.io.BaseRaw:
        return self._load_raw(ctx, ctx.view_options['preload'])

    def load_view(self, ctx: Request, view: str):
        if view != 'info':
            return super().load_view(ctx, view)
        raw = self._load_raw(ctx, preload=False)
        return raw.info

    def _load_raw(self, ctx: Request, preload: bool):
        path = self._resolve_bids_path(ctx, require=True)
        raw = self._read_raw(path, preload=preload)
        self._apply_bids_channels(path, raw)
        if self.pipe.rename_channels:
            if rename := {k: v for k, v in self.pipe.rename_channels.items() if k in raw.ch_names}:
                raw.rename_channels(rename)
        if self.pipe.montage:
            raw.set_montage(self.pipe.montage)
        elif path.datatype == 'eeg':
            self._apply_bids_electrodes(path, raw)
        return raw

    @staticmethod
    def _find_bids_channels(path: BIDSPath) -> Path | None:
        """Find the BIDS channels.tsv sidecar for a recording."""
        channels_path = path.find_matching_sidecar(suffix='channels', extension='.tsv', on_error='ignore')
        return Path(channels_path) if channels_path is not None else None

    @staticmethod
    def _apply_bids_channels(path: BIDSPath, raw: mne.io.BaseRaw) -> None:
        """Apply channel metadata from BIDS channels.tsv sidecar if present."""
        channels_path = RawSourceInput._find_bids_channels(path)
        if channels_path is None:
            return
        channels_df = pd.read_csv(channels_path, sep='\t')
        if 'name' not in channels_df.columns:
            warnings.warn(f"{channels_path} has no 'name' column; skipping channel metadata.")
            return

        if 'type' in channels_df.columns:
            channel_types = {}
            for ch_name, ch_type in zip(channels_df['name'], channels_df['type']):
                if ch_name not in raw.ch_names:
                    continue
                ch_type_bids = str(ch_type).upper()
                if ch_type_bids in BIDS_MEG_CHANNEL_TYPES:
                    continue
                updated_ch_type = BIDS_TO_MNE_CHANNEL_TYPES.get(ch_type_bids)
                if updated_ch_type is None:
                    updated_ch_type = 'misc'
                    warnings.warn(f"No BIDS -> MNE mapping found for channel type {ch_type_bids!r}. Type of channel {ch_name!r} will be set to 'misc'.")
                channel_types[ch_name] = updated_ch_type
            if channel_types:
                raw.set_channel_types(channel_types, on_unit_change='ignore')

    @staticmethod
    def _find_bids_electrodes(path: BIDSPath) -> tuple[Path, Path] | None:
        """Find the BIDS electrode sidecar pair for an EEG recording.

        Looks first for space-entity files (``sub-X_space-*_electrodes.tsv``),
        which is the pattern written by mne-bids. Falls back to a task-matched
        file if no space files are found. Returns
        ``(electrodes_path, coordsystem_path)`` or ``None``. Returned paths
        are not guaranteed to exist.
        """
        data_dir = path.fpath.parent
        sub_prefix = f"sub-{path.subject}"
        if path.session:
            sub_prefix += f"_ses-{path.session}"
        space_candidates = sorted(data_dir.glob(f"{sub_prefix}_space-*_electrodes.tsv"))
        if space_candidates:
            if len(space_candidates) > 1:
                warnings.warn(f"Multiple electrodes.tsv files found in {data_dir}; using {space_candidates[0].name}")
            elec_path = space_candidates[0]
        else:
            elec_path = path.copy().update(suffix='electrodes', extension='.tsv').fpath
            if not elec_path.exists():
                return None
        coord_path = elec_path.with_name(elec_path.name.replace('_electrodes.tsv', '_coordsystem.json'))
        return elec_path, coord_path

    @staticmethod
    def _apply_bids_electrodes(path: BIDSPath, raw: mne.io.BaseRaw) -> None:
        """Apply electrode positions from BIDS electrodes.tsv sidecar if present."""
        elec_pair = RawSourceInput._find_bids_electrodes(path)
        if elec_pair is None:
            return
        elec_path, coord_path = elec_pair
        if not coord_path.exists():
            warnings.warn(f"No matching coordsystem.json found for {elec_path.name}; electrode positions not applied.")
            return
        with open(coord_path, encoding='utf-8-sig') as f:
            coordsystem = json.load(f)
        coord_frame_bids = coordsystem.get('EEGCoordinateSystem', '')
        coord_unit = coordsystem.get('EEGCoordinateUnits', 'm')
        coord_frame = mne_bids.config.BIDS_TO_MNE_FRAMES.get(coord_frame_bids)
        if coord_frame is None:
            warnings.warn(f"Unrecognized EEG coordinate system {coord_frame_bids!r} in {coord_path.name}; electrode positions not applied.")
            return
        scale = COORD_SCALE.get(coord_unit)
        if scale is None:
            warnings.warn(f"Unrecognized EEG coordinate unit {coord_unit!r} in {coord_path.name}; electrode positions not applied.")
            return
        elec_df = pd.read_csv(elec_path, sep='\t')
        numeric = elec_df[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
        valid = numeric.notna().all(axis=1)
        ch_pos = {
            name: numpy.array([x, y, z]) * scale
            for name, x, y, z in zip(elec_df.loc[valid, 'name'], numeric.loc[valid, 'x'], numeric.loc[valid, 'y'], numeric.loc[valid, 'z'])
        }
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame=coord_frame)
        raw.set_montage(montage, on_missing='warn')


class RawSourceDerivative(UncachedDerivative[mne.io.BaseRaw]):
    """Orchestrating node combining the raw source file and bad-channel sidecar.

    Downstream pipeline steps depend on this node via :func:`raw_node_name`.
    Write operations route here so that they can load the raw file (owned by
    :class:`RawSourceInput`) before delegating the actual sidecar write to
    :class:`RawBadChannelsInput`.
    """
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run')
    key_options = {'noise': False}
    view_options = {'preload': False}

    def __init__(
            self,
            raw_name: str,
            pipe: RawSource,
            extension: str,
    ):
        self.name = raw_node_name(raw_name)
        self.raw_name = raw_name
        self.fixed_state = {'raw': raw_name}
        self.pipe = pipe
        self.extension = extension

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        source_name = raw_input_name(self.raw_name)
        bads_name = raw_bad_channels_input_name(self.raw_name)
        return (
            Dependency(source_name, options=ctx.options_for(source_name, 'noise', preload=False)),
            Dependency(bads_name, options=ctx.options_for(bads_name, 'noise')),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'pipe': self.pipe}

    def build(self, ctx: Request) -> mne.io.BaseRaw:
        source_name = raw_input_name(self.raw_name)
        raw = ctx.load(source_name)
        raw.info['bads'] = self._load_bad_channels(ctx)
        return raw

    def apply_view_options(self, ctx: Request, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if ctx.view_options['preload'] and not raw.preload:
            raw.load_data()
        return raw

    def load_view(self, ctx: Request, view: str):
        source_name = raw_input_name(self.raw_name)
        if view == 'bads':
            return self._load_bad_channels(ctx)
        if view == 'info':
            info = ctx.load(source_name, options=ctx.options_for(source_name, 'noise'), view='info')
            with info._unlock():
                info['bads'] = self._load_bad_channels(ctx)
            return info
        return super().load_view(ctx, view)

    def _load_bad_channels(self, ctx: Request) -> list[str]:
        tsv_bads = ctx.load(raw_bad_channels_input_name(self.raw_name))
        raw = ctx.load(raw_input_name(self.raw_name))
        raw_bads = raw.info['bads']
        all_bads = set(tsv_bads) | set(raw_bads)

        # Detect EEG channels whose positions contain NaN
        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=())
        nan_bads = {raw.info['chs'][i]['ch_name'] for i in eeg_picks if numpy.any(numpy.isnan(raw.info['chs'][i]['loc'][:3]))}
        nan_bads.difference_update(all_bads)
        if nan_bads:
            eeg_names = {raw.info['chs'][i]['ch_name'] for i in eeg_picks}
            if eeg_names and eeg_names.issubset(nan_bads):
                raise DataError("All EEG channel positions are NaN. This usually means that the raw file does not contain electrode positions and a montage needs to be applied. Set the montage parameter in RawSource to supply channel positions.")
            warnings.warn(f"Channels with NaN position marked as bad: {', '.join(sorted(nan_bads))}", RuntimeWarning)
            all_bads |= nan_bads

        return sorted(all_bads)


class ICAInput(Input[mne.preprocessing.ICA]):
    key_fields = ('subject', 'session', 'acquisition', 'run')
    version = 1

    def __init__(
            self,
            raw_name: str,
            pipe: RawICA,
            recordings: frozenset[tuple[str, str, str, str, str]],
            runs: Sequence[str],
    ):
        self.name = ica_input_name(raw_name)
        self.raw_name = raw_name
        self.fixed_state = {'raw': raw_name}
        self.pipe = pipe
        self._recordings = recordings
        self._runs = runs or ['']
        # When runs are concatenated, the ICA spans every run, so it is cached
        # per subject/session/acquisition rather than per run.
        if pipe._concatenate_runs:
            self.key_fields = ('subject', 'session', 'acquisition')

    def path(self, ctx: Request) -> Path:
        return self.pipe.path(ctx)

    def _key(self, ctx: Request) -> dict[str, Any]:
        return canonical_state_subset({**ctx.state, 'raw': self.raw_name}, self.key_fields)

    def _manifest(self, ctx: Request) -> ArtifactManifest | None:
        return ctx.registry.read_manifest(ctx.registry.manifest_path(self.path(ctx), self.name))

    def _load_value(self, ctx: Request) -> mne.preprocessing.ICA:
        return self.pipe._load_ica(ctx)

    def _source_states(self, ctx: Request, tasks: Sequence[str]) -> list[dict[str, str]]:
        """Existing source ``{'task', 'run'}`` states for the current subject/session/acquisition.

        Runs are included only when the ICA step concatenates runs (after
        :class:`RawMaxwell`); otherwise the current run is used. Combinations
        without a recording for the current subject/session/acquisition are skipped.
        """
        subject = ctx.state['subject']
        session = ctx.state.get('session') or ''
        acquisition = ctx.state.get('acquisition') or ''
        if self.pipe._concatenate_runs:
            # Spans every run, so identity is keyed on subject/session/acquisition only; the
            # ambient run must not be read (it is not in key_fields here).
            runs = self._runs
        else:
            runs = [ctx.state.get('run') or '']
        states = []
        for task in tasks:
            for run in runs:
                if (subject, session, task, acquisition, run) in self._recordings:
                    states.append({'task': task, 'run': run})
        return states

    def _load_bad_channels(self, ctx: Request) -> list[str]:
        bads = set()
        source_raw = raw_node_name(self.pipe.source)
        for state in self._source_states(ctx, self.pipe.task):
            bads.update(ctx.load(source_raw, state=state, options={'noise': False}, view='bads'))
        return sorted(bads)

    def load_concatenated_source_raw(
            self,
            ctx: Request,
            tasks: tuple[str, ...],
    ) -> mne.io.BaseRaw:
        bad_channels = self._load_bad_channels(ctx)
        states = self._source_states(ctx, tasks)
        if not states:
            raise FileMissingError(f"No source recordings found to estimate ICA {self.raw_name!r} ({ctx.state['subject']=}, session={ctx.state.get('session')!r}).")
        raw = load_raw_dependency(ctx, self.pipe.source, preload=True, state=states[0])
        raw.info['bads'] = bad_channels
        for state in states[1:]:
            raw_ = load_raw_dependency(ctx, self.pipe.source, preload=True, state=state)
            raw_.info['bads'] = bad_channels
            raw.append(raw_)
        return raw

    def _reindex_existing(self, ctx: Request) -> mne.preprocessing.ICA:
        value = self._load_value(ctx)
        ctx.registry.write_manifest(ctx.registry.manifest_path(self.path(ctx), self.name), self._build_manifest(ctx, value))
        return value

    @staticmethod
    def _manifest_matches(
            previous: ArtifactManifest | None,
            current: ArtifactManifest,
    ) -> bool:
        return compare_manifests(previous, current) is None

    def _stale_reason(
            self,
            previous: ArtifactManifest | None,
            current: ArtifactManifest,
    ) -> str:
        if previous is None:
            return "Eelbrain has no saved record for how this ICA file was created."

        diff = find_difference(previous.fingerprint.get('pipe'), current.fingerprint.get('pipe'), coarsen=False)
        if diff is not None:
            path, old, new = diff
            field = self._format_pipe_setting(path)
            return f"The ICA step {self.raw_name!r} changed ({field}: {old!r} -> {new!r})."

        diff = find_difference(previous.dependencies, current.dependencies, strip_quick=True)
        if diff is not None:
            path, old, new = diff
            dep = path[0]
            if dep.endswith(':raw'):
                raw_name = self._dependency_raw_name(previous, current, dep)
                if path[-1] == 'bads':
                    return f"This ICA was estimated using different bad channels: {old!r} -> {new!r}."
                if any(a == 'fingerprint' and b == 'source' for a, b in zip(path, path[1:])):
                    def _fmt_mtime(v: Any) -> str:
                        t = v if isinstance(v, (int, float)) else (v.get('mtime') if isinstance(v, dict) else None)
                        return datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') if t is not None else '?'
                    return f"The source data for raw step {raw_name!r} was modified ({_fmt_mtime(old)} -> {_fmt_mtime(new)})."
                field = self._format_pipe_setting(path[1:], ('fingerprint', 'pipe'))
                return f"This ICA was estimated using different settings for raw step {raw_name!r} ({field}: {old!r} -> {new!r})."
            field = format_difference_path(path)
            return f"One of the recorded ICA inputs changed ({field}: {old!r} -> {new!r})."

        diff = find_difference(previous.fingerprint, current.fingerprint)
        if diff is not None:
            path, old, new = diff
            if path == ('bads',):
                old_set = set(old or [])
                new_set = set(new or [])
                removed = sorted(old_set - new_set)
                added = sorted(new_set - old_set)
                shared = sorted(old_set & new_set)
                lines = ["The set of bad channels used for ICA estimation changed."]
                if shared:
                    lines.append(f"  shared: {', '.join(shared)}")
                if added:
                    lines.append(f"  added: {', '.join(added)}")
                if removed:
                    lines.append(f"  removed: {', '.join(removed)}")
                return '\n'.join(lines)
            field = format_difference_path(path)
            return f"The recorded ICA settings changed ({field}: {old!r} -> {new!r})."

        return "This ICA file no longer matches the current data and settings."

    @staticmethod
    def _dependency_raw_name(
            previous: ArtifactManifest | None,
            current: ArtifactManifest,
            dependency: str,
    ) -> str:
        current_dep = current.dependencies.get(dependency, {})
        previous_dep = {} if previous is None else previous.dependencies.get(dependency, {})
        current_fingerprint = current_dep.get('fingerprint', {})
        previous_fingerprint = previous_dep.get('fingerprint', {})
        return (
            current_fingerprint.get('raw')
            or current_fingerprint.get('definitions', {}).get('raw')
            or previous_fingerprint.get('raw')
            or previous_fingerprint.get('definitions', {}).get('raw')
            or '?'
        )

    @staticmethod
    def _format_pipe_setting(
            path: tuple[str, ...],
            strip_prefix: tuple[str, ...] = (),
    ) -> str:
        parts = list(path)
        if strip_prefix and tuple(parts[:len(strip_prefix)]) == strip_prefix:
            parts = parts[len(strip_prefix):]
        if not parts:
            return 'settings'
        if parts[0] in ('kwargs', 'fit_kwargs'):
            parts = parts[1:] or [parts[0]]
        return format_difference_path(tuple(parts))

    def _current_value_manifest(
            self,
            ctx: Request,
    ) -> tuple[mne.preprocessing.ICA, ArtifactManifest]:
        value = self._load_value(ctx)
        return value, self._build_manifest(ctx, value)

    def _build_manifest(
            self,
            ctx: Request,
            value: mne.preprocessing.ICA,
    ) -> ArtifactManifest:
        resolve_state, resolve_options = ctx._resolve_context()
        return ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=self.name,
            derivative_version=self.version,
            key=self._key(ctx),
            fingerprint=ctx.registry.canonicalize(self.fingerprint(ctx)),
            dependencies=ctx.registry.dependency_fingerprints(ctx),
            cache_policy='external',
            software={'eelbrain_cache_schema': str(MANIFEST_SCHEMA_VERSION), 'mne': mne.__version__},
            resolve_state=resolve_state,
            resolve_options=resolve_options,
        )

    def is_valid(self, ctx: Request) -> bool:
        path = self.path(ctx)
        if not path.exists():
            return False
        return self._manifest_matches(self._manifest(ctx), self._current_value_manifest(ctx)[1])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = []
        for i, state in enumerate(self._source_states(ctx, self.pipe.task)):
            deps.append(Dependency(
                raw_node_name(self.pipe.source),
                label=f'source-{i}:raw',
                state=state,
            ))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        path = self.path(ctx)
        return {
            'raw': self.raw_name,
            'pipe': self.pipe,
            'bads': self._load_bad_channels(ctx),
            'ica_path': path.relative_to(ctx.root),
            'exists': path.exists(),
        }

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        fingerprint = self.fingerprint(ctx)
        path = self.path(ctx)
        fingerprint['ica_file'] = file_fingerprint(ctx.root, path)
        if path.exists():
            fingerprint['exclude'] = self.pipe._load_ica(ctx).exclude
        else:
            fingerprint['exclude'] = []
        return fingerprint

    def load(self, ctx: Request) -> mne.preprocessing.ICA:
        path = self.path(ctx)
        if not path.exists():
            raise FileMissingError(f"ICA file {path.name} does not exist. Run e.make_ica() to create it.")
        value, current = self._current_value_manifest(ctx)
        previous = self._manifest(ctx)
        if not self._manifest_matches(previous, current):
            if ctx.has_control(REINDEX_ICA):
                ctx.registry.write_manifest(ctx.registry.manifest_path(path, self.name), current)
                return value
            reason = self._stale_reason(previous, current)
            raise ProtectedArtifactError(self.name, path, message=f"Existing ICA file {path.name!r} no longer matches the current data and ICA settings.", reason=reason, instructions=f"{reason}\nTo make this ICA match the current pipeline again, revert the raw pipeline change or recompute the ICA. To keep using this existing ICA anyway, call e.load_ica(raw={self.raw_name!r}, accept_stale=True) once or run e.make_ica(raw={self.raw_name!r}) and choose 'incorporate'. To recompute it from the current data, run e.make_ica(raw={self.raw_name!r}) and choose 'overwrite'.")
        return value

    def load_view(
            self,
            ctx: Request,
            view: str,
    ):
        if view == 'bads':
            # The (existence-filtered) source bad channels that the fit uses. The ICA object
            # itself cannot report them: channels bad at fit time are excluded from the
            # decomposition, so they are absent from ica.info['bads'].
            return self._load_bad_channels(ctx)
        if view == 'status':
            if self.path(ctx).exists():
                return 'ok'
            if self._source_states(ctx, self.pipe.task):
                return 'missing-ica'
            return 'missing-raw'
        return super().load_view(ctx, view)

    def materialize(
            self,
            ctx: Request,
            allow_protected_overwrite: bool = False,
            allow_protected_reindex: bool = False,
    ) -> mne.preprocessing.ICA:
        """Build and save the ICA, or load it if already up-to-date.

        Unlike a standard :class:`Derivative`, ICA files may contain manual
        component-rejection decisions and must not be silently overwritten when
        they are stale. This method therefore raises
        :exc:`ProtectedArtifactError` instead of rebuilding automatically.

        The caller (``make_ica``) catches that error, prompts the user for a
        choice, and calls this method again with the appropriate flag set:

        - ``allow_protected_overwrite=True`` — recompute ICA and overwrite the
          existing file.
        - ``allow_protected_reindex=True`` — keep the existing file and rewrite
          its manifest to match the current pipeline state (``incorporate``).

        Parameters
        ----------
        ctx
            Bound request for the current ICA input.
        allow_protected_overwrite
            If ``True``, recompute ICA even when an existing file is stale.
        allow_protected_reindex
            If ``True``, keep the existing ICA file but update its manifest so
            it is no longer considered stale.
        """
        path = self.path(ctx)
        previous = self._manifest(ctx)
        if path.exists():
            value, current = self._current_value_manifest(ctx)
            if self._manifest_matches(previous, current):
                return value
            elif allow_protected_reindex:
                assert current is not None
                ctx.registry.write_manifest(ctx.registry.manifest_path(path, self.name), current)
                return value
            elif not allow_protected_overwrite:
                reason = self._stale_reason(previous, current)
                raise ProtectedArtifactError(self.name, path, message=f"Existing ICA file {path.name!r} no longer matches the current data and ICA settings.", instructions=f"{reason}\nUse allow_protected_reindex=True to keep this ICA file and rewrite its manifest, or allow_protected_overwrite=True to recompute it.")
        raw = self.load_concatenated_source_raw(ctx, self.pipe.task)
        value = self.pipe._fit_ica(raw, ctx.state['subject'], self.raw_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path, overwrite=True)
        ctx.registry.write_manifest(ctx.registry.manifest_path(path, self.name), self._build_manifest(ctx, value))
        return self.load(ctx)


class RawDerivative(Derivative[mne.io.BaseRaw]):
    """Cached raw pipeline artifact.

    Options
    -------
    preload
        Whether to preload the returned raw object.
    noise
        Whether to resolve the corresponding empty-room recording instead of
        the subject recording.
    """
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run')
    cache_suffix = '-raw.fif'
    key_options = {'noise': False}
    view_options = {'preload': False}

    def __init__(
            self,
            raw_name: str,
            pipe: CachedRawPipe,
            pipes: RawPipeGraph,
            extension: str,
    ):
        self.name = raw_node_name(raw_name)
        self.raw_name = raw_name
        self.fixed_state = {'raw': raw_name}
        self.pipe = pipe
        self.pipes = pipes
        self.extension = extension
        if not pipe.cache:
            self.cache_policy = CachePolicy.NEVER

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        source_node = raw_node_name(self.pipe.source)
        deps = [
            Dependency(
                source_node,
                options=ctx.options_for(source_node, 'noise', preload=True),
            ),
        ]
        if isinstance(self.pipe, RawICA):
            ica_name = self.pipes.ica_name(self.raw_name)
            ica_node = ica_input_name(ica_name)
            deps.append(Dependency(ica_node))
            deps.append(Dependency(ica_node, view='bads', label=f'{ica_node}:bads'))
            if ctx.options['noise']:
                deps.append(Dependency(
                    source_node, view='bads',
                    options={'noise': True},
                    label=f'{source_node}:noise_bads',
                ))
            elif ctx.state['task'] not in self.pipe.task:
                deps.append(Dependency(source_node, view='bads', label=f'{source_node}:task_bads'))
        elif isinstance(self.pipe, RawApplyICA):
            ica_name = self.pipes.ica_name(self.raw_name)
            deps.append(Dependency(ica_input_name(ica_name)))
            deps.append(Dependency(
                source_node, view='bads',
                options=ctx.options_for(source_node, 'noise'),
                label=f'{source_node}:bads',
            ))
            deps.append(Dependency(raw_node_name(self.pipe.ica_source), view='bads'))
        elif isinstance(self.pipe, RawMaxwell):
            deps.append(Dependency('maxwell-calibration'))
            deps.append(Dependency('maxwell-crosstalk'))
            deps.append(Dependency('canonical-head-position'))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'pipe': self.pipe, 'raw': self.raw_name}

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view == 'bads':
            return {
                'raw': self.raw_name,
                'pipe': self.pipe,
                'bads': self.pipe._collect_bads(ctx, noise=ctx.options['noise']),
            }
        return super().dependency_fingerprint(ctx, view)

    def build(self, ctx: Request) -> mne.io.BaseRaw:
        source_node = raw_node_name(self.pipe.source)
        path = bids_path(ctx.root, ctx.state, self.extension, datatype=ctx.datatype)
        source_pipe = self.pipes.root_source_pipe(self.raw_name)
        raw = ctx.load(source_node)
        if not raw.preload:
            raw.load_data()
        if isinstance(self.pipe, (RawICA, RawApplyICA)):
            ica_name = self.pipes.ica_name(self.raw_name)
            ica_pipe = self.pipes.ica_pipe(self.raw_name)
            ica = ctx.load(ica_input_name(ica_name))
            if isinstance(self.pipe, RawICA):
                ica_node = ica_input_name(ica_name)
                bads = set(ctx.load(f'{ica_node}:bads'))
                if ctx.options['noise']:
                    bads.update(ctx.load(f'{source_node}:noise_bads'))
                elif ctx.state['task'] not in self.pipe.task:
                    bads.update(ctx.load(f'{source_node}:task_bads'))
                bad_channels = sorted(bads)
            else:
                bad_channels = sorted(
                    set(ctx.load(f'{source_node}:bads')) | set(ctx.load(raw_node_name(self.pipe.ica_source)))
                )
            return ica_pipe._apply_ica(raw, ica, bad_channels, self.raw_name, log=ctx.registry.log)
        if isinstance(self.pipe, RawMaxwell):
            calibration = ctx.load('maxwell-calibration')
            cross_talk = ctx.load('maxwell-crosstalk')
            destination = ctx.load('canonical-head-position')
            return self.pipe._make(raw, path=path, noise=ctx.options['noise'], raw_name=self.raw_name, log=ctx.registry.log, source_pipe=source_pipe, calibration=calibration, cross_talk=cross_talk, destination=destination)
        return self.pipe._make(raw, path=path, noise=ctx.options['noise'], raw_name=self.raw_name, log=ctx.registry.log, source_pipe=source_pipe)

    def load(self, ctx: Request, path: Path) -> mne.io.BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'This filename', module='mne')
            raw = mne.io.read_raw_fif(path, preload=False, verbose=MNE_VERBOSITY)
        return raw

    def load_view(self, ctx: Request, view: str):
        if view == 'bads':
            return self.pipe._collect_bads(ctx, noise=ctx.options['noise'])
        if view != 'info':
            return super().load_view(ctx, view)

        state = {**ctx.state, 'raw': self.raw_name}
        path = bids_path(ctx.root, state, self.extension, datatype=ctx.datatype)
        upstream_info = load_raw_info_dependency(ctx, self.pipe.source, noise=ctx.options['noise']).copy()
        info = self.pipe._make_info(upstream_info, path=path, noise=ctx.options['noise'], raw_name=self.raw_name, log=ctx.registry.log)
        if info is None:
            info = ctx.load_artifact().info

        with info._unlock():
            info['bads'] = self.pipe._collect_bads(ctx, noise=ctx.options['noise'])
        return info

    def apply_view_options(self, ctx: Request, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if ctx.view_options['preload'] and not raw.preload:
            raw.load_data()
        return raw

    def save(
            self,
            ctx: Request,
            path: Path,
            value: mne.io.BaseRaw,
    ) -> None:
        value.save(path, overwrite=True, verbose='ERROR')


class MaxwellCalibrationInput(Input[Path]):
    """Input node for the fine-calibration file (acq-calibration_meg.dat/.fif)."""
    name = 'maxwell-calibration'
    key_fields = ('subject', 'session')

    def path(self, ctx: Request) -> Path:
        for ext in ('.dat', '.fif'):
            p = BIDSPath(
                root=ctx.root,
                subject=ctx.state.get('subject') or None,
                session=ctx.state.get('session') or None,
                acquisition='calibration',
                suffix='meg',
                extension=ext,
                datatype='meg',
            ).fpath
            if p.exists():
                return p
        return BIDSPath(
            root=ctx.root,
            subject=ctx.state.get('subject') or None,
            session=ctx.state.get('session') or None,
            acquisition='calibration',
            suffix='meg',
            extension='.dat',
            datatype='meg',
        ).fpath

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        path = self.path(ctx)
        if path.exists():
            return file_fingerprint(ctx.root, path)
        return {'maxwell-calibration': None}

    def load(self, ctx: Request) -> Path | None:
        path = self.path(ctx)
        return path if path.exists() else None


class MaxwellCrosstalkInput(Input[Path]):
    """Input node for the cross-talk compensation file (acq-crosstalk_meg.fif)."""
    name = 'maxwell-crosstalk'
    key_fields = ('subject', 'session')

    def path(self, ctx: Request) -> Path:
        return BIDSPath(
            root=ctx.root,
            subject=ctx.state.get('subject') or None,
            session=ctx.state.get('session') or None,
            acquisition='crosstalk',
            suffix='meg',
            extension='.fif',
            datatype='meg',
        ).fpath

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        path = self.path(ctx)
        if path.exists():
            return file_fingerprint(ctx.root, path)
        return {'maxwell-crosstalk': None}

    def load(self, ctx: Request) -> Path | None:
        path = self.path(ctx)
        return path if path.exists() else None


class RawHeadPositionDerivative(UncachedDerivative[numpy.ndarray]):
    """Head position samples extracted from one raw recording.

    For recordings with cHPI active the full tracked position time-series is
    returned; otherwise the static ``dev_head_t`` transform is returned as a
    single sample.

    Returns an ``(n, 6)`` float array with columns
    ``[q1, q2, q3, tx, ty, tz]`` using MNE's compact quaternion convention.
    An empty ``(0, 6)`` array is returned when no head position information is
    available in the file.
    """

    name = 'raw-head-position'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run')

    def __init__(self, raw_input_name: str):
        self._raw_input_name = raw_input_name

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency(self._raw_input_name),)

    def build(self, ctx: Request) -> numpy.ndarray | None:
        raw = ctx.load(self._raw_input_name)
        info = raw.info
        hpi_freqs, _, _ = mne.chpi.get_chpi_info(info, on_missing='ignore')
        if len(hpi_freqs):
            chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
            chpi_locs = mne.chpi.compute_chpi_locs(info, chpi_amplitudes)
            head_pos = mne.chpi.compute_head_pos(info, chpi_locs)
            # head_pos columns: [t, q1, q2, q3, tx, ty, tz, gof, err, v]
            return head_pos[:, 1:7]
        dev_head_t = info.get('dev_head_t')
        if dev_head_t is None:
            return None
        trans = dev_head_t['trans']
        quat = mne.transforms.rot_to_quat(trans[:3, :3])
        return numpy.array([[*quat, *trans[:3, 3]]])


class CanonicalHeadPositionDerivative(Derivative):
    """Canonical head position for Maxwell filtering across tasks and runs.

    Computes a single representative head-to-device transform for a given
    subject, session, and acquisition, suitable as the ``destination`` parameter of
    :func:`mne.preprocessing.maxwell_filter`.

    The rotation is the Fréchet mean on SO(3) computed via
    :meth:`scipy.spatial.transform.Rotation.mean` (eigenvector method).
    The translation is the arithmetic mean.  For recordings with cHPI all
    tracked position samples contribute, not just the starting position.

    Returns ``None`` when only one position sample exists across all
    tasks/runs, in which case each file's own ``dev_head_t`` is used directly
    by Maxwell filtering to avoid round-trip conversion noise.

    Parameters
    ----------
    recordings
        Existing ``(subject, session, task, acquisition, run)`` recordings, used for
        existence checks in :meth:`dependencies`.
    tasks
        All task names defined in the experiment.
    runs
        All run values defined in the experiment, or an empty sequence when the
        experiment has no run entity (in which case run ``''`` is used).
    """

    name = 'canonical-head-position'
    key_fields = ('subject', 'session', 'acquisition')
    cache_suffix = '.fif'

    def __init__(
            self,
            recordings: frozenset[tuple[str, str, str, str, str]],
            tasks: Sequence[str],
            runs: Sequence[str],
    ):
        self._recordings = recordings
        self._tasks = tasks
        self._runs = runs or ['']

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        subject = ctx.state['subject']
        session = ctx.state.get('session') or ''
        acquisition = ctx.state.get('acquisition') or ''
        deps = []
        for task, run in itertools.product(self._tasks, self._runs):
            if (subject, session, task, acquisition, run) in self._recordings:
                deps.append(Dependency(
                    name='raw-head-position',
                    label=f'task-{task}_run-{run}' if run else f'task-{task}',
                    state={'task': task, 'run': run},
                ))
        return tuple(deps)

    def build(self, ctx: Request) -> mne.transforms.Transform | None:
        all_positions = []
        for label in ctx.declared_dependencies:
            positions = ctx.load(label)  # (n, 6): [q1, q2, q3, tx, ty, tz], or None
            if positions is not None:
                all_positions.append(positions)
        if len(all_positions) <= 1:
            return None
        all_pos = numpy.vstack(all_positions)  # (N, 6): [q1, q2, q3, tx, ty, tz]
        if numpy.allclose(all_pos[1:], all_pos[0]):
            return None
        # MNE compact quaternions [q1, q2, q3] → scipy [x, y, z, w] (scalar last)
        q = all_pos[:, :3]
        q0 = numpy.sqrt(numpy.maximum(1.0 - numpy.sum(q ** 2, axis=1), 0.0))
        trans = numpy.eye(4)
        trans[:3, :3] = Rotation.from_quat(numpy.column_stack([q, q0])).mean().as_matrix()
        trans[:3, 3] = numpy.mean(all_pos[:, 3:], axis=0)
        return mne.transforms.Transform(fro='meg', to='head', trans=trans)

    def save(self, ctx: Request, path: Path, value: mne.transforms.Transform | None) -> None:
        if value is None:
            path.touch()
        else:
            mne.write_trans(path, value, overwrite=True)

    def load(self, ctx: Request, path: Path) -> mne.transforms.Transform | None:
        if path.stat().st_size == 0:
            return None
        return mne.read_trans(path)


def load_raw_dependency(
        ctx: Request,
        raw: str | None = None,
        *,
        preload: bool = False,
        noise: bool = False,
        state: dict[str, Any] | None = None,
) -> mne.io.BaseRaw:
    merged_state = dict(state or ())
    if raw is None:
        raw = ctx.state['raw']
    merged_state['raw'] = raw
    return ctx.load(raw_node_name(raw), state=merged_state, options={'preload': preload, 'noise': noise})


def load_raw_info_dependency(
        ctx: Request,
        raw: str | None = None,
        *,
        noise: bool = False,
        state: dict[str, Any] | None = None,
) -> mne.Info:
    merged_state = dict(state or ())
    if raw is None:
        raw = ctx.state['raw']
    merged_state['raw'] = raw
    return ctx.load(raw_node_name(raw), state=merged_state, options={'noise': noise}, view='info')
