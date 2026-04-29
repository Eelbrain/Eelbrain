# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Raw preprocessing configuration and graph-node support.

The raw preprocessing subsystem is organized around user-supplied
configuration objects plus graph nodes built from them.

:class:`RawPipe` is the configuration base class for the raw pipeline.
:class:`RawPipe` subclasses implement a specific preprocessing step by
overriding the :meth:`RawPipe._make` method,
and expose user-configurable parameters during initialization.
Users add these :class:`RawPipe` subclass objects to :class:`Pipeline`.

During :class:`Pipeline` initialization, the configured :class:`RawPipe`
objects are normalized and registered as explicit raw-side graph nodes.
Each configured source :class:`RawPipe` produces one raw input node, and each
configured processed :class:`RawPipe` produces one raw derivative node.
:class:`RawICA` additionally produces an ICA input node. Those graph nodes use
the bound :class:`RawPipe` objects to build and load concrete artifacts. The
nodes manage artifact identity, dependency edges, and cache integration, while
the :class:`RawPipe` objects supply preprocessing behavior and configuration.

The public ``Pipeline.load_raw`` method is a facade over these graph nodes.
Raw orchestration, chaining, cached artifact loading, and internal cache-path
generation belong in the raw derivative family, not in bound
:class:`Pipeline` methods.

Caching, manifests, dependency traversal, protected-artifact handling, and
cache policy belong to the lower cache and graph layers, not to
:class:`RawPipe`. Extending the raw pipeline should work by adding
:class:`RawPipe` subclasses and supplying them through :class:`Pipeline`,
without editing the cache kernel or injecting facade behavior into nodes.
"""
from __future__ import annotations
import warnings
import fnmatch
import logging
from os.path import exists, relpath
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

import mne
from scipy import signal
from mne_bids import BIDSPath
import pandas as pd

from .. import load
from .._data_obj import NDVar, Sensor
from .._exceptions import ConfigurationError
from .._io.fiff import KIT_NEIGHBORS
from .._io.txt import read_adjacency
from .._ndvar import filter_data
from .._text import enumeration
from .._utils import user_activity
from .derivative_cache import (
    ArtifactManifest, CachePolicy, Dependency, Derivative, UncachedDerivative,
    Request, Input, MANIFEST_SCHEMA_VERSION, ProtectedArtifactError,
    canonical_state_subset, file_fingerprint,
)
from .configuration import Configuration, sequence_arg, typed_arg
from .exceptions import FileMissingError
from .pathing import LOG_DIR, bids_path, ica_file_path

MNE_VERBOSITY = 'WARNING'
LOG = logging.getLogger(__name__)
_RAW_READER_WARNING_STATE: dict[str, dict[str, Any]] = {}
REINDEX_ICA = 'reindex_ica'


def _raw_reader_warnings_path(root: Path) -> Path:
    return root / LOG_DIR / 'raw-reader-warnings.log'


def _record_raw_reader_warnings(
        root: str | Path,
        raw_path: str | Path,
        warning_list: list[warnings.WarningMessage],
        log: logging.Logger | None = None,
) -> None:
    if not warning_list:
        return
    root_path = Path(root)
    state = _RAW_READER_WARNING_STATE.get(str(root_path))
    if state is None:
        details_path = _raw_reader_warnings_path(root_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text("Warnings emitted while reading raw data files.\n")
        state = {'count': 0, 'details_path': details_path, 'seen': set(), 'warned': False}
        _RAW_READER_WARNING_STATE[str(root_path)] = state
    raw_path = Path(raw_path)
    new_entries = []
    for message in warning_list:
        category = message.category.__name__
        text = str(message.message)
        key = (raw_path, category, text)
        if key in state['seen']:
            continue
        state['seen'].add(key)
        state['count'] += 1
        new_entries.append((category, text))
    if not new_entries:
        return
    with state['details_path'].open('a') as fid:
        fid.write(f"\n{raw_path}\n")
        for category, text in new_entries:
            fid.write(f"  {category}: {text}\n")
    if not state['warned']:
        count = state['count']
        noun = 'warning was' if count == 1 else 'warnings were'
        (log or LOG).warning("%s %s issued while reading raw data files. Full details were written to %s. Additional raw-reader warnings will be suppressed in the terminal for this experiment.", count, noun, state['details_path'])
        state['warned'] = True


class RawPipe(Configuration):
    """Base class for raw-pipeline configurations."""
    DICT_ATTRS = ()

    def _can_resolve(self, pipes: Mapping[str, RawPipe]) -> bool:
        """Determine whether this pipe's dependencies are available in ``pipes``."""
        raise NotImplementedError

    def _get_adjacency(self, data: str) -> str | list[tuple[str, str]] | None:
        raise NotImplementedError

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
    ) -> str | None:
        raise NotImplementedError

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        """Assemble bad channels list from sources"""
        raise NotImplementedError


def raw_node_name(raw: str) -> str:
    return f'raw:{raw}'


def raw_bad_channels_input_name(raw: str) -> str:
    return f'raw-input-bads:{raw}'


def raw_input_name(raw: str) -> str:
    return f'raw-input:{raw}'


def ica_input_name(raw: str) -> str:
    return f'ica-input:{raw}'


class RawBadChannelsInput(Input[list[str]]):
    """Access to source bad channel definitions from ``channels.tsv`` files."""
    OPTION_DEFAULTS = {'noise': False}

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
        """Path to the BIDS channels sidecar (.tsv) for this request."""
        bpath = bids_path(ctx.root, ctx.state, self.extension)
        if ctx.options['noise']:
            bpath = bpath.find_empty_room()
        return Path(bpath.copy().update(suffix='channels', extension='.tsv').fpath)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'bads': self.load(ctx)}

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict[str, Any] | None:
        return file_fingerprint(ctx.root, self.path(ctx), 'bads-file')

    def load(self, ctx: Request) -> list[str]:
        path = self.path(ctx)
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
        """Write bad-channel status to the BIDS ``channels.tsv`` sidecar.

        With ``create=True``, missing sidecar files are initialized from
        ``raw`` before writing, so the resulting file contains one row for
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
            Create a missing ``channels.tsv`` sidecar from ``raw`` before
            writing.
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
            LOG.info("No channels.tsv found at %s, creating an empty one.", path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ch_status = ['bad' if ch in raw.info['bads'] else 'good' for ch in raw.ch_names]
            channels_df = pd.DataFrame({'name': raw.ch_names, 'status': ch_status})
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
    OPTION_DEFAULTS = {'noise': False}
    VIEW_OPTION_DEFAULTS = {'preload': False}

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
        bids_path_ = bids_path(ctx.root, ctx.state, self.extension)
        if ctx.options['noise']:
            bids_path_ = bids_path_.find_empty_room()
        if bids_path_.fpath.exists():
            return bids_path_
        split_path = bids_path_.copy()
        split_path.update(split='01')
        if split_path.exists():
            return split_path
        if require:
            raise FileMissingError(f"Raw input file does not exist at expected location {bids_path_.fpath}")
        return bids_path_

    def path(self, ctx: Request) -> Path:
        return self._resolve_bids_path(ctx).fpath

    @staticmethod
    def _read_raw(
            path: BIDSPath,
            preload: bool,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        """Read a raw file using the MNE reader appropriate for its BIDS extension."""
        match path.extension:
            case '.fif':
                reader = mne.io.read_raw_fif
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
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            raw = reader(path.fpath, preload=preload, verbose=MNE_VERBOSITY)
        _record_raw_reader_warnings(path.root, path.fpath, caught_warnings, log)
        return raw

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'raw': self.raw_name,
            'pipe': self.pipe._as_dict(),
            'source': file_fingerprint(ctx.root, self.path(ctx), 'raw-source'),
        }

    def load(self, ctx: Request) -> mne.io.BaseRaw:
        path = self._resolve_bids_path(ctx, require=True)
        raw = self._read_raw(path, preload=ctx.view_options['preload'], log=ctx.registry.log)
        if self.pipe.rename_channels:
            if rename := {k: v for k, v in self.pipe.rename_channels.items() if k in raw.ch_names}:
                raw.rename_channels(rename)
        if self.pipe.montage:
            raw.set_montage(self.pipe.montage)
        return raw

    def load_view(self, ctx: Request, view: str):
        if view != 'info':
            return super().load_view(ctx, view)
        path = self._resolve_bids_path(ctx, require=True)
        raw = self._read_raw(path, preload=False, log=ctx.registry.log)
        return raw.info


class RawSourceDerivative(UncachedDerivative[mne.io.BaseRaw]):
    """Orchestrating node combining the raw source file and bad-channel sidecar.

    Downstream pipeline steps depend on this node via :func:`raw_node_name`.
    Write operations route here so that they can load the raw file (owned by
    :class:`RawSourceInput`) before delegating the actual sidecar write to
    :class:`RawBadChannelsInput`.
    """
    OPTION_DEFAULTS = {'noise': False}
    VIEW_OPTION_DEFAULTS = {'preload': False}

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
        return {'pipe': self.pipe._as_dict()}

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
        raw_bads = ctx.load(raw_input_name(self.raw_name)).info['bads']
        if tsv_bads and raw_bads:
            return sorted(set(tsv_bads) | set(raw_bads))
        if tsv_bads:
            return tsv_bads
        return raw_bads


class ICAInput(Input[mne.preprocessing.ICA]):
    key_fields = ('subject', 'session', 'acquisition', 'run')
    version = 1

    def __init__(
            self,
            raw_name: str,
            pipe: RawICA,
            pipes: RawPipeGraph,
            extension: str,
    ):
        self.name = ica_input_name(raw_name)
        self.raw_name = raw_name
        self.fixed_state = {'raw': raw_name}
        self.pipe = pipe
        self.pipes = pipes
        self.extension = extension

    def _path(self, ctx: Request) -> Path:
        return ctx.root / ica_file_path(ctx.state, self.raw_name)

    def _key(self, ctx: Request) -> dict[str, Any]:
        return canonical_state_subset({**ctx.state, 'raw': self.raw_name}, self.key_fields)

    def _manifest(self, ctx: Request) -> ArtifactManifest | None:
        return ctx.registry.read_manifest(ctx.registry.manifest_path(self._path(ctx)))

    def _load_value(self, ctx: Request) -> mne.preprocessing.ICA:
        return self.pipe._load_ica(ctx)

    def _load_bad_channels(self, ctx: Request) -> list[str]:
        bads = set()
        source_raw = raw_node_name(self.pipe.source)
        for task in self.pipe.task:
            bads.update(ctx.load(source_raw, state={'task': task}, options={'noise': False}, view='bads'))
        return sorted(bads)

    def load_concatenated_source_raw(
            self,
            ctx: Request,
            tasks: tuple[str, ...],
    ) -> mne.io.BaseRaw:
        bad_channels = self._load_bad_channels(ctx)
        raw = load_raw_dependency(ctx, self.pipe.source, preload=True, state={'task': tasks[0]})
        raw.info['bads'] = bad_channels
        for task in tasks[1:]:
            raw_ = load_raw_dependency(ctx, self.pipe.source, preload=True, state={'task': task})
            raw_.info['bads'] = bad_channels
            raw.append(raw_)
        return raw

    def _reindex_existing(self, ctx: Request) -> mne.preprocessing.ICA:
        value = self._load_value(ctx)
        ctx.registry.write_manifest(ctx.registry.manifest_path(self._path(ctx)), self._build_manifest(ctx, value))
        return value

    @staticmethod
    def _manifest_matches(
            previous: ArtifactManifest | None,
            current: ArtifactManifest,
    ) -> bool:
        return (
            previous is not None
            and previous.schema_version == current.schema_version
            and previous.derivative == current.derivative
            and previous.derivative_version == current.derivative_version
            and previous.key == current.key
            and previous.fingerprint == current.fingerprint
            and previous.dependencies == current.dependencies
        )

    @classmethod
    def _first_difference(
            cls,
            old: Any,
            new: Any,
            path: tuple[str, ...] = (),
    ) -> tuple[tuple[str, ...], Any, Any] | None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old).union(new), key=str):
                if key not in old:
                    return (*path, str(key)), None, new[key]
                if key not in new:
                    return (*path, str(key)), old[key], None
                diff = cls._first_difference(old[key], new[key], (*path, str(key)))
                if diff is not None:
                    return diff
            return None
        if isinstance(old, list) and isinstance(new, list):
            for i in range(max(len(old), len(new))):
                if i >= len(old):
                    return (*path, f'[{i}]'), None, new[i]
                if i >= len(new):
                    return (*path, f'[{i}]'), old[i], None
                diff = cls._first_difference(old[i], new[i], (*path, f'[{i}]'))
                if diff is not None:
                    return diff
            return None
        if old != new:
            return path, old, new
        return None

    @staticmethod
    def _format_difference_path(path: tuple[str, ...], strip_prefix: tuple[str, ...] = ()) -> str:
        parts = list(path)
        if strip_prefix and tuple(parts[:len(strip_prefix)]) == strip_prefix:
            parts = parts[len(strip_prefix):]
        out = []
        for part in parts:
            if part.startswith('['):
                if out:
                    out[-1] += part
                else:
                    out.append(part)
            else:
                out.append(part)
        return '.'.join(out) or 'value'

    def _stale_reason(
            self,
            previous: ArtifactManifest | None,
            current: ArtifactManifest,
    ) -> str:
        if previous is None:
            return "Eelbrain has no saved record for how this ICA file was created."

        diff = self._first_difference(previous.fingerprint.get('pipe'), current.fingerprint.get('pipe'))
        if diff is not None:
            path, old, new = diff
            field = self._format_pipe_setting(path)
            return f"The ICA step {self.raw_name!r} changed ({field}: {old!r} -> {new!r})."

        diff = self._first_difference(previous.dependencies, current.dependencies)
        if diff is not None:
            path, old, new = diff
            dep = path[0]
            if dep.endswith(':raw'):
                raw_name = self._dependency_raw_name(previous, current, dep)
                field = self._format_pipe_setting(path[1:], ('fingerprint', 'definitions', 'pipe'))
                return f"This ICA was estimated using different settings for raw step {raw_name!r} ({field}: {old!r} -> {new!r})."
            field = self._format_difference_path(path)
            return f"One of the recorded ICA inputs changed ({field}: {old!r} -> {new!r})."

        diff = self._first_difference(previous.fingerprint, current.fingerprint)
        if diff is not None:
            path, old, new = diff
            if path == ('bads',):
                return f"The set of bad channels used for ICA estimation changed ({old!r} -> {new!r})."
            field = self._format_difference_path(path)
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
        return ICAInput._format_difference_path(tuple(parts))

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
        return ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=self.name,
            derivative_version=self.version,
            key=self._key(ctx),
            fingerprint=ctx.registry.canonicalize(self.fingerprint(ctx)),
            dependencies=ctx.registry.dependency_fingerprints(self, ctx, None),
            cache_policy='external',
            software={'eelbrain_cache_schema': str(MANIFEST_SCHEMA_VERSION), 'mne': mne.__version__},
        )

    def is_valid(self, ctx: Request) -> bool:
        path = self._path(ctx)
        if not path.exists():
            return False
        return self._manifest_matches(self._manifest(ctx), self._current_value_manifest(ctx)[1])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = []
        for i, task in enumerate(self.pipe.task):
            deps.append(Dependency(
                raw_node_name(self.pipe.source),
                label=f'source-{i}:raw',
                state={'task': task},
            ))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        path = self._path(ctx)
        return {
            'raw': self.raw_name,
            'pipe': self.pipe._as_dict(),
            'bads': self._load_bad_channels(ctx),
            'ica_path': relpath(path, ctx.root),
            'exists': exists(path),
        }

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        fingerprint = self.fingerprint(ctx)
        path = self._path(ctx)
        fingerprint['ica_file'] = file_fingerprint(ctx.root, path, 'ica-file')
        if exists(path):
            fingerprint['exclude'] = self.pipe._load_ica(ctx).exclude
        else:
            fingerprint['exclude'] = []
        return fingerprint

    def load(self, ctx: Request) -> mne.preprocessing.ICA:
        path = self._path(ctx)
        if not exists(path):
            raise FileMissingError(f"ICA file {path.name} does not exist. Run e.make_ica() to create it.")
        value, current = self._current_value_manifest(ctx)
        previous = self._manifest(ctx)
        if not self._manifest_matches(previous, current):
            if ctx.has_control(REINDEX_ICA):
                ctx.registry.write_manifest(ctx.registry.manifest_path(path), current)
                return value
            reason = self._stale_reason(previous, current)
            raise ProtectedArtifactError(self.name, path, message=f"Existing ICA file {path.name!r} no longer matches the current data and ICA settings.", instructions=f"{reason} To make this ICA match the current pipeline again, revert the raw pipeline change or recompute the ICA. To keep using this existing ICA anyway, call e.load_ica(raw={self.raw_name!r}, accept_stale=True) once or run e.make_ica(raw={self.raw_name!r}) and choose 'incorporate'. To recompute it from the current data, run e.make_ica(raw={self.raw_name!r}) and choose 'overwrite'.")
        return value

    def load_view(
            self,
            ctx: Request,
            view: str,
    ):
        if view == 'bads':
            return sorted(self.load(ctx).info['bads'])
        if view == 'status':
            if exists(self._path(ctx)):
                return 'ok'
            source_node = raw_input_name(self.pipes.root_source_name(self.pipe.source))
            if all(ctx.registry.resolve(source_node, state={**ctx.state, 'task': task}, options={'noise': False}).exists() for task in self.pipe.task):
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
        path = self._path(ctx)
        previous = self._manifest(ctx)
        current = None
        if exists(path):
            value, current = self._current_value_manifest(ctx)
            if self._manifest_matches(previous, current):
                return value
        if exists(path) and not allow_protected_overwrite:
            if allow_protected_reindex:
                assert current is not None
                ctx.registry.write_manifest(ctx.registry.manifest_path(path), current)
                return value
            reason = self._stale_reason(previous, current)
            raise ProtectedArtifactError(self.name, path, message=f"Existing ICA file {path.name!r} no longer matches the current data and ICA settings.", instructions=f"{reason} Use allow_protected_reindex=True to keep this ICA file and rewrite its manifest, or allow_protected_overwrite=True to recompute it.")
        raw = self.load_concatenated_source_raw(ctx, self.pipe.task)
        value = self.pipe._fit_ica(raw, ctx.state['subject'], self.raw_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path, overwrite=True)
        ctx.registry.write_manifest(ctx.registry.manifest_path(path), self._build_manifest(ctx, value))
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
    cache_policy = CachePolicy.OPTIONAL
    cache_suffix = '-raw.fif'
    OPTION_DEFAULTS = {'noise': False}
    VIEW_OPTION_DEFAULTS = {'preload': False}

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
        if not pipe._cache:
            self.cache_policy = CachePolicy.DISABLED_BY_DEFAULT

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
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(ctx, definitions={'pipe': self.pipe._as_dict()}, extra={'raw': self.raw_name})

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view == 'bads':
            return {
                'raw': self.raw_name,
                'pipe': self.pipe._as_dict(),
                'bads': self.pipe._collect_bads(ctx, noise=ctx.options['noise']),
            }
        return super().dependency_fingerprint(ctx, view)

    def build(self, ctx: Request) -> mne.io.BaseRaw:
        source_node = raw_node_name(self.pipe.source)
        path = bids_path(ctx.root, ctx.state, self.extension)
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
        path = bids_path(ctx.root, state, self.extension)
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


class RawSource(RawPipe):
    """Raw data source

    Parameters
    ----------
    sysname
        Used to determine sensor positions (not needed for KIT files, or when a
        montage is specified).
    rename_channels
        Rename channels based on a ``{from: to}`` dictionary. This happens
        *after* calling the ``reader``, and *before* applying the ``montage``.
        Useful to convert system-specific channel names to those of a standard montages.
    montage
        Name of a montage that is applied to raw data to set sensor positions
        (see :meth:`mne.io.Raw.set_montage`).
    adjacency
        Ajacency between sensors. Can be specified as:

        - ``'auto'`` to use :func:`mne.channels.find_ch_adjacency`
        - Pre-defined adjacency (one of :func:`mne.channels.get_builtin_ch_adjacencies`)
        - Path to load adjacency from a file
        - ``"none"`` for no connections
        - ``"grid"`` for grid connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.

        If unspecified, it is inferred from ``sysname`` if possible.
    ...
    """
    DICT_ATTRS = ('sysname', 'rename_channels', 'montage', 'adjacency', 'kwargs')

    def __init__(
            self,
            sysname: str = None,
            rename_channels: dict = None,
            montage: str = None,
            adjacency: str | list[tuple[str, str]] | Path = None,
            **kwargs,
    ):
        RawPipe.__init__(self)
        if isinstance(adjacency, str):
            if adjacency not in ('auto', 'grid', 'none') and adjacency not in mne.channels.get_builtin_ch_adjacencies():
                adjacency = Path(adjacency)
        if isinstance(adjacency, Path):
            adjacency = read_adjacency(adjacency)
        self.sysname = sysname
        self.rename_channels = typed_arg(rename_channels, dict)
        self.montage = montage
        self.adjacency = adjacency
        self.kwargs = kwargs

    def _can_resolve(self, pipes: dict[str, RawPipe]) -> bool:
        return True

    def _normalize_channel_names(self, raw: mne.io.BaseRaw, bad_chs: list[str]) -> list[str]:
        """Validate and normalize channel names against the raw file's sensor layout."""
        sensor = load.mne.sensor_dim(raw.info, adjacency=self.adjacency)
        return sensor._normalize_sensor_names(bad_chs)

    def _detect_flat_channels(self, path: BIDSPath, raw: mne.io.BaseRaw, flat: float = None) -> list[str] | None:
        """Detect flat channels; returns None if the operation should be skipped."""
        if flat is None:
            if path.datatype == 'meg':
                flat = 1e-14
            elif path.datatype == 'eeg':
                return None
            else:
                raise NotImplementedError(f"{path.datatype=}")
        elif flat == 0:
            return None
        bad_chs = list(raw.info['bads'])
        sysname = self._get_sysname(raw.info, path.subject, path.datatype)
        raw_ndvar = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=self.adjacency)
        bad_chs.extend(raw_ndvar.sensor.names[raw_ndvar.std('time') < flat])
        return bad_chs

    def _as_dict(self) -> dict:
        out = RawPipe._as_dict(self)
        if isinstance(self.montage, mne.channels.DigMontage):
            out['montage'] = Sensor.from_montage(self.montage)
        return out

    def _get_adjacency(self, data: str) -> str | list[tuple[str, str]] | None:
        if data == 'eog':
            return None
        else:
            return self.adjacency

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
    ) -> str | None:
        if data == 'eog':
            return None
        elif isinstance(self.sysname, str):
            return self.sysname
        elif isinstance(self.sysname, dict):
            for k, v in self.sysname.items():
                if fnmatch.fnmatch(subject, k):
                    return v
        kit_system_id = info.get('kit_system_id')
        return KIT_NEIGHBORS.get(kit_system_id)


class CachedRawPipe(RawPipe):
    _bad_chs_affect_cache: bool = False
    DICT_ATTRS = ('source',)

    def __init__(self, source: str, cache: bool = True):
        RawPipe.__init__(self)
        self.source = source
        self._cache = cache

    def _can_resolve(self, pipes: Mapping[str, RawPipe]) -> bool:
        return self.source in pipes

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return info

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        return ctx.load(raw_node_name(self.source), options={'noise': noise}, view='bads')


class RawFilter(CachedRawPipe):
    """Filter raw pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    l_freq
        Low cut-off frequency in Hz.
    h_freq
        High cut-off frequency in Hz.
    cache
        Cache the resulting raw files (default ``True``).
    n_jobs
        Parameter for :meth:`mne.io.Raw.filter`; Values other than 1 are slower
        in most cases due to added overhead except for very large files.
    ...
        :meth:`mne.io.Raw.filter` parameters.

    See Also
    --------
    Pipeline.raw
    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('l_freq', 'h_freq', 'n_jobs', 'kwargs')

    def __init__(
            self,
            source: str,
            l_freq: float = None,
            h_freq: float = None,
            cache: bool = True,
            n_jobs: str | int | None = 1,
            **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.kwargs = kwargs
        self.n_jobs = n_jobs

    def _filter_ndvar(self, ndvar, **kwargs):
        return filter_data(ndvar, self.l_freq, self.h_freq, **self.kwargs, **kwargs)

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: filtering for %s...", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        raw.filter(self.l_freq, self.h_freq, **self.kwargs, n_jobs=self.n_jobs, verbose=MNE_VERBOSITY)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        if self.l_freq and self.l_freq > (info['highpass'] or 0):
            with info._unlock():
                info['highpass'] = float(self.l_freq)
        if self.h_freq and self.h_freq < (info['lowpass'] or info['sfreq']):
            with info._unlock():
                info['lowpass'] = float(self.h_freq)
        return info


class RawFilterElliptic(CachedRawPipe):
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('low_stop', 'low_pass', 'high_pass', 'high_stop', 'gpass', 'gstop')

    def __init__(self, source, low_stop, low_pass, high_pass, high_stop, gpass, gstop):
        CachedRawPipe.__init__(self, source)
        self.low_stop = low_stop
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.high_stop = high_stop
        self.gpass = gpass
        self.gstop = gstop

    def _sos(self, sfreq):
        nyq = sfreq / 2.
        low_stop = self.low_stop
        low_pass = self.low_pass
        high_pass = self.high_pass
        high_stop = self.high_stop
        gpass = self.gpass
        gstop = self.gstop
        if high_stop is None:
            assert low_stop is not None
            assert high_pass is None
        else:
            high_stop /= nyq
            high_pass /= nyq

        if low_stop is None:
            assert low_pass is None
        else:
            low_pass /= nyq
            low_stop /= nyq

        if low_stop is None:
            btype = 'lowpass'
            wp, ws = high_pass, high_stop
        elif high_stop is None:
            btype = 'highpass'
            wp, ws = low_pass, low_stop
        else:
            btype = 'bandpass'
            wp, ws = (low_pass, high_pass), (low_stop, high_stop)
        order, wn = signal.ellipord(wp, ws, gpass, gstop)
        return signal.ellip(order, gpass, gstop, wn, btype, output='sos')

    def _filter_ndvar(self, ndvar):
        axis = ndvar.get_axis('time')
        sos = self._sos(1. / ndvar.time.tstep)
        x = signal.sosfilt(sos, ndvar.x, axis)
        return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: filtering for %s...", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        sos = self._sos(raw.info['sfreq'])
        for i in picks:
            raw._data[i] = signal.sosfilt(sos, raw._data[i])
        low, high = self.low_pass, self.high_pass
        with raw.info._unlock():
            if high and raw.info['lowpass'] > high:
                raw.info['lowpass'] = float(high)
            if low and raw.info['highpass'] < low:
                raw.info['highpass'] = float(low)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        low, high = self.low_pass, self.high_pass
        if high and high < (info['lowpass'] or info['sfreq']):
            with info._unlock():
                info['lowpass'] = float(high)
        if low and low > (info['highpass'] or 0):
            with info._unlock():
                info['highpass'] = float(low)
        return info


class RawICA(CachedRawPipe):
    """ICA raw pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    task
        Task(s) to use for estimating ICA components.
    method
        Method for ICA decomposition (default: ``'extended-infomax'``; see
        :class:`mne.preprocessing.ICA`).
    random_state
        Set the random state for ICA decomposition to make results reproducible
        (default 0, see :class:`mne.preprocessing.ICA`).
    fit_kwargs
        A dictionary with keyword arguments that should be passed to
        :meth:`mne.preprocessing.ICA.fit`. This includes
        ``reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}`` unless
        a different value for ``reject`` is specified here.
    cache : bool
        Cache the resulting raw files (default ``False``).
    ...
        Additional parameters for :class:`mne.preprocessing.ICA`.

    See Also
    --------
    Pipeline.raw
    RawApplyICA

    Notes
    -----
    This preprocessing step estimates one set of ICA components per subject,
    using the data specified in the ``task`` parameter. The selected
    components are then removed from all data tasks during this preprocessing
    step, regardless of whether they were used to estimate the components or
    not.

    Use :meth:`Pipeline.make_ica_selection` for each subject to
    select ICA components that should be removed. The arguments to that function
    determine what data is used to visualize the component time courses.

    This step merges bad channels from all tasks.

    Examples
    --------
    Some ICA examples::

        class Experiment(Pipeline):

            raw = {
                '1-40': RawFilter('raw', 1, 40),
                # Extended infomax with PCA preprocessing
                'ica': RawICA('1-40', 'extended-infomax', n_components=0.99),
                # Fast ICA
                'fastica': RawICA('1-40', 'task', 'fastica', n_components=0.9),
                # Change thresholds for data rejection using fit_kwargs
                'ica-rej': RawICA('1-40', 'task', 'fastica', fit_kwargs=dict(
                    reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 500e-6},
                )),
            }

    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('task', 'kwargs', 'fit_kwargs')

    run: str | Sequence[str] = None

    def __init__(
            self,
            source: str,
            task: str | Sequence[str],
            method: str = 'extended-infomax',
            random_state: int = 0,
            fit_kwargs: dict[str, Any] = None,
            cache: bool = False,
            **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.task = sequence_arg('task', task, allow_none=False)
        self.kwargs = {'method': method, 'random_state': random_state, **kwargs}
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}

    def _load_ica(
            self,
            ctx: Request,
    ) -> mne.preprocessing.ICA:
        ica_path = ctx.root / ica_file_path(ctx.state, self.name)
        if not exists(ica_path):
            raise FileMissingError(f"ICA file {ica_path.name} does not exist for raw={self.name!r}. Run e.make_ica() to create it.")
        return mne.preprocessing.read_ica(ica_path)

    @staticmethod
    def _check_ica_channels(
            ica: mne.preprocessing.ICA,
            info: mne.Info,
            return_missing: bool = False,  # if ICA is missing channels, return those (they can be dropped in data)
    ) -> bool | tuple:
        "Check whether `ica` and `info` contain the same channels"
        picks = mne.pick_types(info, meg=True, eeg=True, ref_meg=False)
        raw_ch_names = [info.ch_names[i] for i in picks]
        if return_missing:
            raw_set = set(raw_ch_names)
            ica_set = set(ica.ch_names)
            if ica_set - raw_set:
                raise RuntimeError(f"ICA contains channels not present in data: {enumeration(sorted(ica_set - raw_set))}")
            else:
                return tuple(raw_set - ica_set)
        else:
            return raw_ch_names == ica.ch_names

    def _fit_ica(
            self,
            raw: mne.io.BaseRaw,
            subject: str,
            raw_name: str,
    ) -> mne.preprocessing.ICA:
        LOG.info("Raw %s: computing ICA decomposition for %s", raw_name, subject)
        kwargs = self.kwargs.copy()
        kwargs.setdefault('max_iter', 256)
        if kwargs['method'] == 'extended-infomax':
            kwargs['method'] = 'infomax'
            kwargs['fit_params'] = {'extended': True}

        ica = mne.preprocessing.ICA(**kwargs)
        fit_kwargs = {'reject': {'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}, **self.fit_kwargs}
        with user_activity:
            ica.fit(raw, **fit_kwargs)
        return ica

    def _apply_ica(
            self,
            raw: mne.io.BaseRaw,
            ica: mne.preprocessing.ICA,
            bad_channels: list[str],
            raw_name: str,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.debug("Raw %s: applying ICA...", raw_name)
        raw.info['bads'] = [ch for ch in bad_channels if ch in raw.ch_names]
        missing = self._check_ica_channels(ica, raw.info, return_missing=True)
        if missing:
            raw.drop_channels(missing)
        ica.apply(raw)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        bads = set()
        # Try to read bad channels on ICA
        try:
            bads.update(ctx.load(ica_input_name(self.name), view='bads'))
        except FileMissingError:
            # Merged task file bad channels
            for task in self.task:
                bads.update(ctx.load(raw_node_name(self.source), state={'task': task}, view='bads'))
        # Task that has not been used for ICA fit
        if noise:
            bads.update(ctx.load(raw_node_name(self.source), options={'noise': True}, view='bads'))
        elif ctx.state['task'] not in self.task:
            bads.update(ctx.load(raw_node_name(self.source), view='bads'))
        return sorted(bads)


class RawApplyICA(CachedRawPipe):
    """Apply ICA estimated in a :class:`RawICA` pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    ica
        Name of the :class:`RawICA` pipe from which to load the ICA components.
    cache
        Cache the resulting raw files (default ``False``).

    See Also
    --------
    Pipeline.raw

    Notes
    -----
    This pipe inherits bad channels from the ICA.

    Examples
    --------
    Estimate ICA components with 1-40 Hz band-pass filter and apply the ICA
    to data that is high pass filtered at 0.1 Hz::

        class Experiment(Pipeline):

            raw = {
                '1-40': RawFilter('raw', 1, 40),
                'ica': RawICA('1-40', 'task', 'extended-infomax', n_components=0.99),
                '0.1-40': RawFilter('raw', 0.1, 40),
                '0.1-40-ica': RawApplyICA('0.1-40', 'ica'),
            }

    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('ica_source',)

    def __init__(
            self,
            source: str,
            ica: str,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.ica_source = ica

    def _can_resolve(self, pipes: Mapping[str, RawPipe]) -> bool:
        return CachedRawPipe._can_resolve(self, pipes) and self.ica_source in pipes

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        bads = set()
        bads.update(ctx.load(raw_node_name(self.source), options={'noise': noise}, view='bads'))
        bads.update(ctx.load(raw_node_name(self.ica_source), view='bads'))
        return sorted(bads)


class RawMaxwell(CachedRawPipe):
    """Maxwell filter raw pipe.

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    bad_condition
        How to deal with ill-conditioned SSS matrices; by default, an error is
        raised, which might prevent the process to complete for some subjects.
        Set to ``'warning'`` to proceed anyways.
    cache
        Cache the resulting raw files (default ``True``).
    flat
        Threshold for marking flat channels as bad (default 1e-14).
    ...
        :func:`mne.preprocessing.maxwell_filter` parameters.

    See Also
    --------
    Pipeline.raw

    Notes
    -----
    For empty room recordings, there is no ``dev_head_t`` information, ``coord_frame = 'meg'`` will be used automatically.
    Flat channels are automatically marked as bad with a threshold of parameter ``flat``.
    """

    _bad_chs_affect_cache = True
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('bad_condition', 'flat', 'kwargs')

    def __init__(
        self,
        source: str,
        bad_condition: str = 'error',
        cache: bool = True,
        flat: float = 1e-14,
        **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.kwargs = kwargs
        self.bad_condition = bad_condition
        self.flat = flat

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        assert source_pipe is not None
        sysname = source_pipe._get_sysname(raw.info, path.subject, path.datatype)
        adjacency = source_pipe._get_adjacency(path.datatype)
        raw_ndvar = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=adjacency)
        raw.info['bads'].extend(raw_ndvar.sensor.names[raw_ndvar.std('time') < self.flat])
        logger = log or LOG
        logger.info("Raw %s: computing Maxwell filter for %s", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            coord_frame = 'meg' if noise else 'head'
            return mne.preprocessing.maxwell_filter(raw, bad_condition=self.bad_condition, coord_frame=coord_frame, verbose=MNE_VERBOSITY, **self.kwargs)

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None


class RawOversampledTemporalProjection(CachedRawPipe):
    """Oversampled temporal projection: see :func:`mne.preprocessing.oversampled_temporal_projection`"""
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('duration',)

    def __init__(
            self,
            source: str,
            duration: float = 10.0,
            cache: bool = True,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.duration = duration

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: computing oversampled temporal projection for %s", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            return mne.preprocessing.oversampled_temporal_projection(raw, self.duration)


class RawReReference(CachedRawPipe):
    """Re-reference EEG data

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    reference
        New reference: ``'average'`` (default) or one or several electrode
        names.
    add
        Reconstruct reference channels with given names and set them to 0.
    drop
        Drop these channels after applying the reference.
    cache
        Cache the resulting raw files (default ``False``).

    See Also
    --------
    Pipeline.raw
    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('reference', 'add', 'drop')

    def __init__(
            self,
            source: str,
            reference: str | Sequence[str] = 'average',
            add: str | Sequence[str] = None,
            drop: str | Sequence[str] = None,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        if isinstance(reference, str):
            self.reference = reference
        else:
            self.reference = sequence_arg('reference', reference, allow_none=False, sequence_type=list)
        self.add = sequence_arg('add', add, sequence_type=list)
        self.drop = sequence_arg('drop', drop, sequence_type=list)

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        montage = raw.get_montage()
        if self.add:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The locations of multiple reference channels are ignored', module='mne')
                raw = mne.add_reference_channels(raw, self.add, copy=False)
            if montage:
                raw.set_montage(montage)
        raw.set_eeg_reference(self.reference)
        if self.drop:
            raw = raw.drop_channels(self.drop)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None


class RawPipeGraph(Mapping[str, RawPipe]):
    """Resolved raw-pipeline graph with convenience lineage lookups."""

    def __init__(
            self,
            pipes: dict[str, RawPipe],
            source_names: dict[str, str | None],
            root_source_names: dict[str, str],
            ica_names: dict[str, str | None],
            lineages: dict[str, tuple[str, ...]],
    ):
        self._pipes = pipes
        self._source_names = source_names
        self._root_source_names = root_source_names
        self._ica_names = ica_names
        self._lineages = lineages

    def __getitem__(self, item: str) -> RawPipe:
        return self._pipes[item]

    def __iter__(self):
        return iter(self._pipes)

    def __len__(self) -> int:
        return len(self._pipes)

    def source_name(self, raw_name: str) -> str | None:
        """Return the immediate upstream raw name for ``raw_name``."""
        return self._source_names[raw_name]

    def source_pipe(self, raw_name: str) -> RawPipe | None:
        """Return the immediate upstream raw pipe for ``raw_name``."""
        source_name = self.source_name(raw_name)
        if source_name is None:
            return None
        return self[source_name]

    def root_source_name(self, raw_name: str) -> str:
        """Return the source raw name at the root of ``raw_name``."""
        return self._root_source_names[raw_name]

    def root_source_pipe(self, raw_name: str) -> RawSource:
        """Return the source raw pipe at the root of ``raw_name``."""
        pipe = self[self.root_source_name(raw_name)]
        assert isinstance(pipe, RawSource)
        return pipe

    def ica_name(self, raw_name: str) -> str:
        """Return the ICA raw name associated with ``raw_name``."""
        if raw_name in self._ica_names:
            return self._ica_names[raw_name]
        raise ValueError(f"{raw_name=} does not involve ICA")

    def ica_pipe(self, raw_name: str) -> RawICA:
        """Return the ICA raw pipe associated with ``raw_name``."""
        ica_name = self.ica_name(raw_name)
        pipe = self[ica_name]
        assert isinstance(pipe, RawICA)
        return pipe

    def lineage_names(self, raw_name: str) -> tuple[str, ...]:
        """Return the raw-step names from source to ``raw_name``."""
        return self._lineages[raw_name]

    def lineage_pipes(self, raw_name: str) -> tuple[RawPipe, ...]:
        """Return the raw-step pipes from source to ``raw_name``."""
        return tuple(self[name] for name in self.lineage_names(raw_name))


def assemble_raw_pipes(
        raw: dict[str, RawPipe],
        tasks: tuple[str],
) -> RawPipeGraph:
    """Resolve raw-pipe dependencies and bind pipe names."""
    pending = dict(raw)
    resolved = {}
    source_names = {}
    root_source_names = {}
    ica_names = {}
    lineages = {}
    for name, pipe in pending.items():
        pipe._store_name(name)
    while pending:
        n_pending = len(pending)
        for key in list(pending):
            if pending[key]._can_resolve(resolved):
                pipe = pending.pop(key)
                if isinstance(pipe, RawICA):
                    missing = set(pipe.task).difference(tasks)
                    if missing:
                        raise ConfigurationError(f"RawICA {key!r} lists one or more non-exising tasks: {', '.join(missing)}. Available tasks: {', '.join(tasks)}.")
                if isinstance(pipe, RawSource):
                    source_names[key] = None
                    root_source_names[key] = key
                    ica_names[key] = None
                    lineages[key] = (key,)
                else:
                    source_names[key] = pipe.source
                    root_source_names[key] = root_source_names[pipe.source]
                    if isinstance(pipe, RawICA):
                        ica_names[key] = key
                    elif isinstance(pipe, RawApplyICA):
                        ica_names[key] = pipe.ica_source
                    else:
                        ica_names[key] = ica_names[pipe.source]
                    lineages[key] = (*lineages[pipe.source], key)
                resolved[key] = pipe
        if len(pending) == n_pending:
            raise ConfigurationError(f"Unable to resolve source for raw {enumeration(pending)}, circular dependency?")
    return RawPipeGraph(raw, source_names, root_source_names, ica_names, lineages)
