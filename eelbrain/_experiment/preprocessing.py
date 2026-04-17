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
from collections.abc import Sequence

import mne
from scipy import signal
from mne_bids import BIDSPath, mark_channels
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
    ArtifactManifest, CachePolicy, Dependency, Derivative,
    Request, Input, MANIFEST_SCHEMA_VERSION, ProtectedArtifactError,
    canonical_state_subset, file_fingerprint,
)
from .configuration import Configuration, sequence_arg, typed_arg
from .exceptions import FileMissingError
from .pathing import LOG_DIR, bids_path, ica_file_path

MNE_VERBOSITY = 'WARNING'
AddBadsArg = bool | Sequence[str]
LOG = logging.getLogger(__name__)
_RAW_READER_WARNING_STATE: dict[str, dict[str, Any]] = {}
REINDEX_ICA = 'reindex_ica'


def _raw_reader_warnings_path(root: str | Path) -> Path:
    return Path(root) / LOG_DIR / 'raw-reader-warnings.log'


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

    def _can_resolve(self, pipes: dict[str, RawPipe]) -> bool:
        "Determine whether this pipe's dependencies are available in ``pipes``."
        raise NotImplementedError

    def _as_dict(self) -> dict:
        return {'type': self.__class__.__name__, **Configuration._as_dict(self)}

    def _find_input_pipe(self, raw_dict: dict[str, RawPipe]) -> RawSource:
        raise NotImplementedError

    def _get_adjacency(self, data: str, pipes: dict[str, RawPipe] = None) -> str | list[tuple[str, str]] | Path:
        raise NotImplementedError

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
            pipes: dict[str, RawPipe] = None,
    ) -> str | None:
        raise NotImplementedError

    def _load_with_bads(
            self,
            path: BIDSPath,
            add_bads: AddBadsArg = True,
            preload: bool = False,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        "Call _load() and add bad channels"
        raw = self._load(path, preload, noise=noise, pipes=pipes, log=log)
        # bad channels
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is True:
            raw.info['bads'] = self._load_bad_channels(path, noise=noise, pipes=pipes, log=log)
        elif add_bads is False:
            raw.info['bads'] = []
        else:
            raise TypeError(f"{add_bads=}")
        return raw

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        "Process the info without processing the raw data, return the processed info"
        raise NotImplementedError

    def _load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None, log: logging.Logger | None = None) -> list[str]:
        raise NotImplementedError

    def _make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        raise NotImplementedError

    def _make_bad_channels_auto(
            self,
            path: BIDSPath,
            flat: float,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        raise NotImplementedError


def raw_node_name(raw: str) -> str:
    return f'raw:{raw}'


def raw_bad_channels_input_name(raw: str) -> str:
    return f'raw-input-bads:{raw}'


def ica_input_name(raw: str) -> str:
    return f'ica-input:{raw}'


def get_ica_pipe_name(pipes: dict[str, RawPipe], raw: str | RawPipe) -> str:
    if isinstance(raw, str):
        pipe_name = raw
        pipe = pipes[raw]
    else:
        for pipe_name, pipe in pipes.items():
            if pipe is raw:
                break
        else:
            raise ValueError(f"{raw=} raw pipe is not registered")
    while not isinstance(pipe, RawICA):
        if isinstance(pipe, RawSource):
            raise ValueError(f"raw={pipe_name!r} does not involve ICA")
        if isinstance(pipe, RawApplyICA):
            pipe_name = pipe.ica_source
        else:
            pipe_name = pipe.source
        pipe = pipes[pipe_name]
    return pipe_name


def get_ica_pipe(pipes: dict[str, RawPipe], raw: str | RawPipe) -> RawICA:
    return pipes[get_ica_pipe_name(pipes, raw)]


def raw_state(
        state: dict[str, Any],
        raw: str,
) -> dict[str, Any]:
    out = dict(state)
    out['raw'] = raw
    return out


class RawBadChannelsInput(Input[list[str]]):
    """Access to bad channel definitions from channels sidecar files

    There is one instance for every :class:`RawPipe` in the graph.
    The instance's load method delegates to the RawPipe's load method,
    which in turn traverses the :class:`RawPipe` tree.

    Effectively, bad channels come from the :class:`RawSource`,
    but some :class:`RawPipe` like ICA may modify them along the way.

    In additiom to the channels file, bad channels can also be embedded in the
    raw FIFF file. :class:`RawBadChannelsInput` is not aware of those.
    """
    OPTION_DEFAULTS = {'noise': False}

    def __init__(
            self,
            raw_name: str,
            pipe: RawPipe,
            pipes: dict[str, RawPipe],
    ):
        self.name = raw_bad_channels_input_name(raw_name)
        self.raw_name = raw_name
        self.pipe = pipe
        self.pipes = pipes

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        noise = ctx.options['noise']
        state = {**ctx.state, 'raw': self.raw_name}
        path = bids_path(ctx.registry.root, state)
        return {
            'raw': self.raw_name,
            'noise': noise,
            'pipeline': self.pipe._as_dict(),
            'bad_channels': self.pipe._load_bad_channels(path, noise=noise, pipes=self.pipes, log=ctx.registry.log),
        }

    def load(self, ctx: Request) -> list[str]:
        state = {**ctx.state, 'raw': self.raw_name}
        return self.pipe._load_bad_channels(bids_path(ctx.registry.root, state), noise=ctx.options['noise'], pipes=self.pipes, log=ctx.registry.log)


class RawSourceInput(Input[mne.io.BaseRaw]):
    OPTION_DEFAULTS = {'noise': False}
    VIEW_OPTION_DEFAULTS = {'add_bads': True, 'preload': False}

    def __init__(
            self,
            raw_name: str,
            pipe: RawSource,
            pipes: dict[str, RawPipe],
    ):
        self.name = raw_node_name(raw_name)
        self.raw_name = raw_name
        self.pipe = pipe
        self.pipes = pipes

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if ctx.view_options['add_bads'] is True:
            return (
                Dependency(
                    raw_bad_channels_input_name(self.raw_name),
                    state={'raw': self.raw_name},
                    options=ctx.options_for(raw_bad_channels_input_name(self.raw_name), noise=ctx.options['noise']),
                ),
            )
        return ()

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        noise = ctx.options['noise']
        state = {**ctx.state, 'raw': self.raw_name}
        path = bids_path(ctx.registry.root, state, noise=noise)
        actual_path = self.pipe._find_raw_path(path) or path.fpath
        return {
            'raw': self.raw_name,
            'noise': noise,
            'pipe': self.pipe._as_dict(),
            'source': file_fingerprint(
                ctx.registry.root,
                actual_path,
                'raw-source',
                metadata={'raw': self.raw_name, 'noise': noise},
            ),
        }

    def load(self, ctx: Request) -> mne.io.BaseRaw:
        return self.pipe._load_with_bads(
            bids_path(ctx.registry.root, {**ctx.state, 'raw': self.raw_name}),
            add_bads=ctx.view_options['add_bads'],
            preload=ctx.view_options['preload'],
            noise=ctx.options['noise'],
            pipes=self.pipes,
            log=ctx.registry.log,
        )


class ICAInput(Input[mne.preprocessing.ICA]):
    key_fields = ('subject', 'session', 'acquisition', 'run')
    version = 1

    def __init__(
            self,
            raw_name: str,
            pipe: RawICA,
            pipes: dict[str, RawPipe],
            runs: tuple[str, ...] | None,
    ):
        self.name = ica_input_name(raw_name)
        self.raw_name = raw_name
        self.pipe = pipe
        self.pipes = pipes
        self.runs = runs

    def _path(self, ctx: Request) -> Path:
        return ctx.registry.root / ica_file_path({**ctx.state, 'raw': self.raw_name}, raw=self.raw_name)

    def _key(self, ctx: Request) -> dict[str, Any]:
        return canonical_state_subset({**ctx.state, 'raw': self.raw_name}, self.key_fields)

    def _manifest(self, ctx: Request) -> ArtifactManifest | None:
        return ctx.registry.read_manifest(ctx.registry.manifest_path(self._path(ctx)))

    def _load_value(self, ctx: Request) -> mne.preprocessing.ICA:
        state = {**ctx.state, 'raw': self.raw_name}
        return self.pipe._load_ica(bids_path(ctx.registry.root, state), raw_name=self.raw_name)

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
                field = self._format_pipe_setting(path[1:], ('fingerprint', 'pipeline'))
                return f"This ICA was estimated using different settings for raw step {raw_name!r} ({field}: {old!r} -> {new!r})."
            if dep.endswith(':bads'):
                raw_name = self._dependency_raw_name(previous, current, dep)
                if len(path) > 2 and path[1] == 'fingerprint' and path[2] == 'pipeline':
                    field = self._format_pipe_setting(path[1:], ('fingerprint', 'pipeline'))
                    return f"This ICA was estimated using different settings for raw step {raw_name!r} ({field}: {old!r} -> {new!r})."
                field = self._format_difference_path(path[1:], ('fingerprint',))
                return f"The bad-channel information used to estimate this ICA changed for raw step {raw_name!r} ({field}: {old!r} -> {new!r})."
            field = self._format_difference_path(path)
            return f"One of the recorded ICA inputs changed ({field}: {old!r} -> {new!r})."

        diff = self._first_difference(previous.fingerprint, current.fingerprint)
        if diff is not None:
            path, old, new = diff
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
        return current_dep.get('fingerprint', {}).get('raw') or previous_dep.get('fingerprint', {}).get('raw') or '?'

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
        return self.pipe._ica_dependencies(ctx, self.raw_name, self.runs)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        path = self._path(ctx)
        return {
            'raw': self.raw_name,
            'pipe': self.pipe._as_dict(),
            'runs': self.runs,
            'ica_path': relpath(path, ctx.registry.root),
            'exists': exists(path),
        }

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        fingerprint = dict(self.fingerprint(ctx))
        path = self._path(ctx)
        fingerprint['ica_file'] = file_fingerprint(ctx.registry.root, path, 'ica-file')
        if exists(path):
            state = {**ctx.state, 'raw': self.raw_name}
            fingerprint['exclude'] = self.pipe._load_ica(bids_path(ctx.registry.root, state), raw_name=self.raw_name).exclude
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
        value = self.pipe._build_ica(ctx, self.pipes, self.raw_name, self.runs)
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path, overwrite=True)
        ctx.registry.write_manifest(ctx.registry.manifest_path(path), self._build_manifest(ctx, value))
        return self.load(ctx)


class RawDerivative(Derivative[mne.io.BaseRaw]):
    """Cached raw pipeline artifact.

    Options
    -------
    add_bads
        Whether to apply bad channels to the loaded raw object.
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
    VIEW_OPTION_DEFAULTS = {'add_bads': True, 'preload': False}

    def __init__(
            self,
            raw_name: str,
            pipe: CachedRawPipe,
            pipes: dict[str, RawPipe],
    ):
        self.name = raw_node_name(raw_name)
        self.raw_name = raw_name
        self.pipe = pipe
        self.pipes = pipes
        if not pipe._cache:
            self.cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = [
            Dependency(
                raw_node_name(self.pipe.source),
                state={'raw': self.pipe.source},
                options=ctx.options_for(raw_node_name(self.pipe.source), add_bads=True, preload=False, noise=ctx.options['noise']),
            ),
        ]
        if isinstance(self.pipe, RawICA):
            deps.append(Dependency(ica_input_name(self.raw_name), state={'raw': self.raw_name}))
        elif isinstance(self.pipe, RawApplyICA):
            deps.append(Dependency(ica_input_name(self.pipe.ica_source), state={'raw': self.pipe.ica_source}))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(ctx, state_fields=self.key_fields, definitions={'pipe': self.pipe._as_dict()})

    def build(self, ctx: Request) -> mne.io.BaseRaw:
        noise = ctx.options['noise']
        state = {**ctx.state, 'raw': self.raw_name}
        path = bids_path(ctx.registry.root, state)

        raw = load_raw_dependency(ctx, self.pipe.source, add_bads=True, preload=True, noise=noise)
        if not raw.preload:
            raw.load_data()
        if isinstance(self.pipe, RawICA):
            ica = ctx.load(ica_input_name(self.raw_name), state={'raw': self.raw_name})
            return self.pipe._apply_ica(path, raw, ica, self.raw_name, pipes=self.pipes, log=ctx.registry.log)
        if isinstance(self.pipe, RawApplyICA):
            ica = ctx.load(ica_input_name(self.pipe.ica_source), state={'raw': self.pipe.ica_source})
            ica_pipe = get_ica_pipe(self.pipes, self.pipe.ica_source)
            return ica_pipe._apply_ica(path, raw, ica, self.raw_name, pipes=self.pipes, log=ctx.registry.log)
        return self.pipe._make(path, True, noise=noise, raw=raw, pipes=self.pipes, raw_name=self.raw_name, log=ctx.registry.log)

    def load(self, ctx: Request, path: Path) -> mne.io.BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'This filename', module='mne')
            raw = mne.io.read_raw_fif(path, preload=False, verbose=MNE_VERBOSITY)
        return raw

    def apply_view_options(self, ctx: Request, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if ctx.view_options['preload'] and not raw.preload:
            raw.load_data()
        add_bads = ctx.view_options['add_bads']
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is True:
            state = {**ctx.state, 'raw': self.raw_name}
            raw.info['bads'] = self.pipe._load_bad_channels(bids_path(ctx.registry.root, state), noise=ctx.options['noise'], pipes=self.pipes, log=ctx.registry.log)
        elif add_bads is False:
            raw.info['bads'] = []
        else:
            raise TypeError(f"{add_bads=}")
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
        add_bads: AddBadsArg = True,
        preload: bool = False,
        noise: bool = False,
        state: dict[str, Any] | None = None,
) -> mne.io.BaseRaw:
    merged_state = dict(state or ())
    if raw is None:
        raw = ctx.state['raw']
    merged_state['raw'] = raw
    return ctx.load(raw_node_name(raw), state=merged_state, options={'add_bads': add_bads, 'preload': preload, 'noise': noise})


def raw_data_dependency(
        ctx: Request,
        *,
        raw: str | None = None,
        label: str | None = None,
        noise: bool = False,
        add_bads: AddBadsArg = True,
) -> Dependency:
    if raw is None:
        raw = ctx.state['raw']
    return Dependency(
        raw_node_name(raw),
        label=label,
        state={'raw': raw},
        options={'add_bads': add_bads, 'preload': False, 'noise': noise},
    )


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

    def _find_input_pipe(self, raw_dict: dict[str, RawPipe]) -> RawSource:
        return self

    def _find_raw_path(self, path: BIDSPath) -> Path | None:
        """Return the raw file path, or None if it does not exist.

        Checks the path as given first.  If the file is absent, falls back to
        the BIDS first-split variant (``split-01``) so that split recordings
        are located transparently — MNE then reads all subsequent parts.
        """
        raw_path = Path(path.fpath)
        if raw_path.exists():
            return raw_path
        split_path = Path(path.copy().update(split='01').fpath)
        if split_path.exists():
            return split_path
        return None

    def _raw_path(self, path: BIDSPath) -> Path:
        """Get path to the raw file. Enforce existence."""
        raw_path = self._find_raw_path(path)
        if raw_path is None:
            raise FileMissingError(f"Raw input file does not exist at expected location {path.fpath}")
        return raw_path

    def _bads_path(self, path: BIDSPath) -> Path:
        return Path(path.copy().update(
            suffix='channels',
            extension='.tsv',
        ).fpath)

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        "Load raw data from different formats"
        path = path if not noise else path.find_empty_room()
        raw_path = self._raw_path(path)
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
                raise RuntimeError(f"Unrecognized file format: {path.suffix}")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            raw = reader(raw_path, preload=preload, verbose=MNE_VERBOSITY)
        _record_raw_reader_warnings(path.root, raw_path, caught_warnings, log)
        if self.rename_channels:
            if rename := {k: v for k, v in self.rename_channels.items() if k in raw.ch_names}:
                raw.rename_channels(rename)
        if self.montage:
            raw.set_montage(self.montage)
        # Empty room may have missing raw.info['dig'], find alternative ways to do this when encountered
        # if not raw.info['dig'] and self._dig_sessions is not None and self._dig_sessions[subject]:
        #     dig_recording = self._dig_sessions[subject][recording]
        #     if dig_recording != recording:
        #         dig_raw = self._load(subject, dig_recording, False)
        #         raw.set_montage(mne.channels.DigMontage(dig=dig_raw.info['dig']))
        return raw

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        return self._load_with_bads(path, pipes=pipes).info

    def _load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None, log: logging.Logger | None = None) -> list[str]:
        "Get channels.tsv content, create if it does not exist"
        path = path if not noise else path.find_empty_room()
        bads_path = self._bads_path(path)
        if exists(bads_path):
            channels_df = pd.read_csv(bads_path, sep='\t')
            if 'status' not in channels_df.columns:
                return []
            elif 'name' not in channels_df.columns:
                raise RuntimeError(f"channels.tsv file at {bads_path} is missing required column 'name'. Please regenerate the file.")
            return channels_df.query('status == "bad"')['name'].tolist()
        # Create a channels file if none exists
        LOG.info("No channels.tsv found for %s, creating an empty one.", path.fpath)
        bads_path.parent.mkdir(parents=True, exist_ok=True)
        raw = self._load(path, preload=False, log=log)
        ch_names = raw.ch_names
        ch_status = ['bad' if ch in raw.info['bads'] else 'good' for ch in ch_names]
        channels_df = pd.DataFrame({
            'name': ch_names,
            'status': ch_status,
        })
        channels_df.to_csv(bads_path, sep='\t', index=False)
        return channels_df.query('status == "bad"')['name'].tolist()

    def _make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        path = path if not noise else path.find_empty_room()
        # check input list
        if isinstance(bad_chs, (str, int)):
            bad_chs = (bad_chs,)
        raw = self._load(path, False)
        sensor = load.mne.sensor_dim(raw.info, adjacency=self.adjacency)
        new_bads = sensor._normalize_sensor_names(bad_chs)
        # merge with old bads
        old_bads = self._load_bad_channels(path)
        if old_bads is not None and not redo:
            new_bads = sorted(set(old_bads).union(new_bads))
        # print change
        LOG.info("Bad channels: %s -> %s for %s", old_bads, new_bads, self._bads_path(path))
        if new_bads == old_bads:
            return
        # write new bad channels
        if redo:
            mark_channels(path, ch_names='all', status='good', verbose=MNE_VERBOSITY)
        if len(new_bads):
            mark_channels(path, ch_names=new_bads, status='bad', verbose=MNE_VERBOSITY)

    def _make_bad_channels_auto(
            self,
            path: BIDSPath,
            flat: float = None,
            redo: bool = False,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        if noise:
            path = path.find_empty_room()
        if flat is None:
            if path.datatype == 'meg':
                flat = 1e-14
            elif path.datatype == 'eeg':
                return
            else:
                raise NotImplementedError(f"{path.datatype=}")
        elif flat == 0:
            return
        raw = self._load(path, False)
        bad_chs: list[str] = raw.info['bads']
        sysname = self._get_sysname(raw.info, path.subject, path.datatype)
        raw = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=self.adjacency)
        bad_chs.extend(raw.sensor.names[raw.std('time') < flat])
        self._make_bad_channels(path, bad_chs, redo)

    def _as_dict(self) -> dict:
        out = RawPipe._as_dict(self)
        if isinstance(self.montage, mne.channels.DigMontage):
            out['montage'] = Sensor.from_montage(self.montage)
        return out

    def _get_adjacency(self, data: str, pipes: dict[str, RawPipe] = None) -> str | list[tuple[str, str]] | Path | None:
        if data == 'eog':
            return None
        else:
            return self.adjacency

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
            pipes: dict[str, RawPipe] = None,
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

    def _can_resolve(self, pipes: dict[str, RawPipe]) -> bool:
        return self.source in pipes

    def _find_input_pipe(self, raw_dict: dict[str, RawPipe]) -> RawSource:
        return raw_dict[self.source]._find_input_pipe()

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        return self._make(path, preload, noise=noise, pipes=pipes, log=log)

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        return pipes[self.source]._load_info(path, pipes)

    def _load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None, log: logging.Logger | None = None) -> list[str]:
        return pipes[self.source]._load_bad_channels(path, noise=noise, pipes=pipes, log=log)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        pipes[self.source]._make_bad_channels(path, bad_chs, redo, noise=noise, pipes=pipes)

    def _make_bad_channels_auto(self, *args, **kwargs) -> None:
        pipes = kwargs.pop('pipes', None)
        pipes[self.source]._make_bad_channels_auto(*args, pipes=pipes, **kwargs)

    def _get_adjacency(self, data: str, pipes: dict[str, RawPipe] = None) -> str | list[tuple[str, str]] | Path:
        return pipes[self.source]._get_adjacency(data, pipes)

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
            pipes: dict[str, RawPipe] = None,
    ) -> str | None:
        return pipes[self.source]._get_sysname(info, subject, data, pipes)


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
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, preload=True, noise=noise, pipes=pipes, log=log)
        logger = log or LOG
        logger.info("Raw %s: filtering for %s...", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        raw.filter(self.l_freq, self.h_freq, **self.kwargs, n_jobs=self.n_jobs, verbose=MNE_VERBOSITY)
        return raw

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        info = super()._load_info(path, pipes)
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
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, preload=True, noise=noise, pipes=pipes, log=log)
        logger = log or LOG
        logger.info("Raw %s: filtering for %s...", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        sos = self._sos(raw.info['sfreq'])
        for i in picks:
            raw._data[i] = signal.sosfilt(sos, raw._data[i])
        low, high = self.low_pass, self.high_pass
        if high and raw.info['lowpass'] > high:
            raw.info['lowpass'] = float(high)
        if low and raw.info['highpass'] < low:
            raw.info['highpass'] = float(low)
        return raw


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

    def _load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None, log: logging.Logger | None = None) -> list[str]:
        source_pipe = pipes[self.source]
        bad_chs = set()
        for task in self.task:
            path_ = path.copy().update(task=task)
            bad_chs.update(source_pipe._load_bad_channels(path_, pipes=pipes, log=log))
        if noise:
            bad_chs.update(source_pipe._load_bad_channels(path, noise=noise, pipes=pipes, log=log))
        return sorted(bad_chs)

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        info = super()._load_info(path, pipes)
        info['bads'] = self._load_bad_channels(path, pipes=pipes)
        return info

    def _load_ica(
            self,
            path: BIDSPath,
            raw_name: str,
    ) -> mne.preprocessing.ICA:
        ica_path = path.root / ica_file_path(raw_state({
            'subject': path.subject,
            'session': path.session,
            'task': path.task,
            'acquisition': path.acquisition,
            'run': path.run,
            'datatype': path.datatype,
            'suffix': path.suffix,
            'extension': path.extension,
        }, raw_name), raw=raw_name)
        if not exists(ica_path):
            raise FileMissingError(f"ICA file {ica_path.name} does not exist for raw={raw_name!r}. Run e.make_ica() to create it.")
        return mne.preprocessing.read_ica(ica_path)

    def _source_states(
            self,
            ctx: Request,
            runs: tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        out = []
        for task in self.task:
            for run in (runs or (None,)):
                state = {'raw': self.source, 'task': task}
                if run is not None:
                    state['run'] = run
                out.append(state)
        return out

    def _ica_dependencies(
            self,
            ctx: Request,
            raw_name: str,
            runs: tuple[str, ...] | None,
    ) -> tuple[Dependency, ...]:
        deps = []
        for i, state in enumerate(self._source_states(ctx, runs)):
            source_state = dict(state)
            stem = f"source-{i}"
            source_raw = source_state.pop('raw')
            deps.append(
                raw_data_dependency(
                    ctx,
                    raw=source_raw,
                    label=f'{stem}:raw',
                    add_bads=False,
                )
            )
            deps[-1] = Dependency(
                deps[-1].name,
                label=deps[-1].label,
                state=dict(state),
                options=deps[-1].options,
            )
            deps.append(
                Dependency(
                    raw_bad_channels_input_name(source_raw),
                    label=f'{stem}:bads',
                    state=dict(state),
                )
            )
        return tuple(deps)

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

    def _load_concatenated_source_raw(
            self,
            path: BIDSPath,
            tasks: tuple[str],
            runs: tuple[str],
            pipes: dict[str, RawPipe],
    ) -> mne.io.BaseRaw:
        "Concatenate raws from different tasks and runs."
        # NOTE: this use bad channels in RawICA while loading tasks from user input.
        source_pipe = pipes[self.source]
        bad_channels = self._load_bad_channels(path, pipes=pipes)
        path_list = []
        for task in tasks:
            path_ = path.copy().update(task=task)
            if not runs:
                path_list.append(path_)
                continue
            for run in runs:
                path_list.append(path_.copy().update(run=run))
        raw = source_pipe._load_with_bads(path_list[0], bad_channels, pipes=pipes)
        for path_ in path_list[1:]:
            raw_ = source_pipe._load_with_bads(path_, bad_channels, pipes=pipes)
            raw.append(raw_)
        return raw

    def _build_ica(
            self,
            ctx: Request,
            pipes: dict[str, RawPipe],
            raw_name: str,
            runs: tuple[str, ...] | None,
    ) -> mne.preprocessing.ICA:
        state = raw_state(ctx.state, raw_name)
        path = bids_path(ctx.registry.root, state)
        bad_channels = self._load_bad_channels(path, pipes=pipes)
        path_list = []
        runs = runs or ()
        for task in self.task:
            if not runs:
                path_list.append({'task': task})
                continue
            for run in runs:
                path_list.append({'task': task, 'run': run})

        raw = load_raw_dependency(ctx, self.source, add_bads=bad_channels, preload=True, state=path_list[0])
        for state in path_list[1:]:
            raw_ = load_raw_dependency(ctx, self.source, add_bads=bad_channels, preload=True, state=state)
            raw.append(raw_)
        return self._fit_ica(raw, ctx.state['subject'], raw_name)

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

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, preload=True, noise=noise, pipes=pipes, log=log)
        return self._apply(path, raw, raw_name, pipes=pipes, log=log)

    def _apply_ica(
            self,
            path: BIDSPath,
            raw: mne.io.BaseRaw,
            ica: mne.preprocessing.ICA,
            raw_name: str,
            pipes: dict[str, RawPipe] = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.debug("Raw %s: applying ICA for %s...", raw_name, path.fpath)
        raw.info['bads'] = [ch for ch in self._load_bad_channels(path, pipes=pipes) if ch in raw.ch_names]
        missing = self._check_ica_channels(ica, raw.info, return_missing=True)
        if missing:
            raw.drop_channels(missing)
        ica.apply(raw)
        return raw

    def _apply(
            self,
            path: BIDSPath,
            raw: mne.io.BaseRaw,
            raw_name: str,
            pipes: dict[str, RawPipe] = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        return self._apply_ica(path, raw, self._load_ica(path, raw_name), raw_name, pipes=pipes, log=log)


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

    def _can_resolve(self, pipes: dict[str, RawPipe]) -> bool:
        return CachedRawPipe._can_resolve(self, pipes) and self.ica_source in pipes

    def _load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None, log: logging.Logger | None = None) -> list[str]:
        return pipes[self.ica_source]._load_bad_channels(path, noise=noise, pipes=pipes, log=log)

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        info = super()._load_info(path, pipes)
        info['bads'] = self._load_bad_channels(path, pipes=pipes)
        return info

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, preload=True, noise=noise, pipes=pipes, log=log)
        return pipes[self.ica_source]._apply(path, raw, raw_name, pipes=pipes, log=log)


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
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, noise=noise, pipes=pipes, log=log)
        sysname = self._get_sysname(raw.info, path.subject, path.datatype, pipes)
        adjacency = self._get_adjacency(path.datatype, pipes)
        raw_ndvar = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=adjacency)
        raw.info['bads'].extend(raw_ndvar.sensor.names[raw_ndvar.std('time') < self.flat])
        logger = log or LOG
        logger.info("Raw %s: computing Maxwell filter for %s", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            coord_frame = 'meg' if noise else 'head'
            return mne.preprocessing.maxwell_filter(raw, bad_condition=self.bad_condition, coord_frame=coord_frame, verbose=MNE_VERBOSITY, **self.kwargs)


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
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, noise=noise, pipes=pipes, log=log)
        logger = log or LOG
        logger.info("Raw %s: computing oversampled temporal projection for %s", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            return mne.preprocessing.oversampled_temporal_projection(raw, self.duration)


class RawUpdateBadChannels(CachedRawPipe):
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('bad_channels',)

    def __init__(
            self,
            source: str,
            bad_channels: dict[str, Sequence[str]],
    ):
        CachedRawPipe.__init__(self, source, False)
        self.bad_channels = bad_channels
        self._pattern_keys = [key for key in bad_channels if ('*' in key or '?' in key)]

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, preload=preload, noise=noise, pipes=pipes, log=log)
        return raw

    def _load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None, log: logging.Logger | None = None) -> list[str]:
        bad_channels = pipes[self.source]._load_bad_channels(path, noise=noise, pipes=pipes, log=log)
        subject = path.subject
        if subject in self.bad_channels:
            key = subject
        else:
            for key in self._pattern_keys:
                if fnmatch.fnmatch(subject, key):
                    break
            else:
                raise KeyError(subject)
        return sorted(set(bad_channels).union(self.bad_channels[key]))


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
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = pipes[self.source]._load_with_bads(path, preload=True, noise=noise, pipes=pipes, log=log)
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

    def _load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        if self.add or self.drop:
            return self._load_with_bads(path, pipes=pipes).info
        else:
            return super()._load_info(path, pipes)


def assemble_raw_pipes(
        raw: dict[str, RawPipe],
        tasks: tuple[str],
) -> dict[str, RawPipe]:
    """Resolve raw-pipe dependencies and bind pipe names."""
    pending = dict(raw)
    resolved = {}
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
                resolved[key] = pipe
        if len(pending) == n_pending:
            raise ConfigurationError(f"Unable to resolve source for raw {enumeration(pending)}, circular dependency?")
    return raw
