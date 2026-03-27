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
objects are normalized and bound into graph nodes. Those graph nodes
use the bound :class:`RawPipe` objects to build and load concrete artifacts.
The nodes manage artifact identity, dependency edges, and cache integration,
while the :class:`RawPipe` objects supply preprocessing behavior and
configuration.

The public ``Pipeline.load_raw`` method is a facade over these graph nodes.
Raw orchestration, chaining, and cached artifact loading belong in the raw
derivative family, not in bound :class:`Pipeline` methods.

Caching, manifests, dependency traversal, protected-artifact handling, and
cache policy belong to the lower cache and graph layers, not to
:class:`RawPipe`. Extending the raw pipeline should work by adding
:class:`RawPipe` subclasses and supplying them through :class:`Pipeline`,
without editing the cache kernel or injecting facade behavior into nodes.
"""
from __future__ import annotations
import warnings
from copy import deepcopy
import fnmatch
from itertools import chain
import logging
from os import makedirs
from os.path import basename, exists, relpath
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import mne
from scipy import signal
from mne_bids import BIDSPath, mark_channels
import pandas as pd

from .. import load
from .._data_obj import NDVar, Sensor
from .._exceptions import DefinitionError
from .._io.fiff import KIT_NEIGHBORS
from .._io.txt import read_adjacency
from .._ndvar import filter_data
from .._text import enumeration
from .._utils import deprecate_kwarg, user_activity
from .derivative_cache import (
    ArtifactManifest, CachePolicy, Dependency, DependencyNode,
    Derivative, DerivativeContext, Input, MANIFEST_SCHEMA_VERSION,
    ProtectedArtifactError, canonical_state_subset, file_fingerprint,
)
from .definitions import sequence_arg, typed_arg
from .exceptions import FileMissingError
from .pathing import bids_path, cached_raw_file_path

MNE_VERBOSITY = 'WARNING'
AddBadsArg = bool | Sequence[str]


class RawPipe:
    name: str = None  # set on linking
    log: logging.Logger = None

    def _can_link(self, pipes: dict[str, RawPipe]) -> bool:
        raise NotImplementedError

    def _link(
            self,
            name: str,
            pipes: dict[str, RawPipe],
            cache_path: str,
            log: logging.Logger
    ) -> RawPipe:
        raise NotImplementedError

    def _link_base(
            self,
            name: str,
            log: logging.Logger,
    ) -> RawPipe:
        out = deepcopy(self)
        out.name = name
        out.log = log
        return out

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = {arg: getattr(self, arg) for arg in chain(args, ('name',))}
        out['type'] = self.__class__.__name__
        return out

    @staticmethod
    def _normalize_dict(state: dict) -> None:
        pass

    def get_adjacency(self, data: str, pipes: dict[str, RawPipe] = None) -> str | list[tuple[str, str]] | Path:
        raise NotImplementedError

    def get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
            pipes: dict[str, RawPipe] = None,
    ) -> str | None:
        raise NotImplementedError

    def load(
            self,
            path: BIDSPath,
            add_bads: AddBadsArg = True,
            preload: bool = False,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        "Call _load() and add bad channels"
        raw = self._load(path, preload, noise=noise, pipes=pipes)
        # bad channels
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is True:
            raw.info['bads'] = self.load_bad_channels(path, noise=noise, pipes=pipes)
        elif add_bads is False:
            raw.info['bads'] = []
        else:
            raise TypeError(f"{add_bads=}")
        return raw

    def load_derivative_name(self) -> str | None:
        return None

    def raw_meeg_input_name(self) -> str:
        return raw_meeg_input_name(self.name)

    def raw_bad_channels_input_name(self) -> str:
        return raw_bad_channels_input_name(self.name)

    def load_derivative_cache(self) -> bool | None:
        return None

    def load_derivative_options(
            self,
            *,
            add_bads: AddBadsArg = True,
            preload: bool = False,
            noise: bool = False,
    ) -> dict[str, Any]:
        return {
            'add_bads': add_bads,
            'preload': preload,
            'noise': noise,
        }

    def raw_dependency_name(self) -> str:
        return self.load_derivative_name() or self.raw_meeg_input_name()

    def raw_dependency_options(
            self,
            *,
            add_bads: AddBadsArg = True,
            noise: bool = False,
    ) -> dict[str, Any]:
        return self.load_derivative_options(add_bads=add_bads, preload=False, noise=noise)

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        "Process the info without processing the raw data, return the processed info"
        raise NotImplementedError

    def load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None) -> list[str]:
        raise NotImplementedError

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        raise NotImplementedError

    def make_bad_channels_auto(
            self,
            path: BIDSPath,
            flat: float,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        raise NotImplementedError

    def cache_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return ()

    def input_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return (RawBadChannelsInput(self),)


def raw_cache_node_name(raw: str) -> str:
    return f'raw-cache:{raw}'


def ica_cache_node_name(raw: str) -> str:
    return f'ica-cache:{raw}'


def raw_bad_channels_input_name(raw: str) -> str:
    return f'raw-input-bads:{raw}'


def raw_meeg_input_name(raw: str) -> str:
    return f'raw-input-meeg:{raw}'


RAW_NODE_NAME = 'raw'
RAW_BAD_CHANNELS_INPUT = 'raw-input-bads'
RAW_MEEG_INPUT = 'raw-input-meeg'
ICA_INPUT = 'ica-input'


def get_source_pipe(pipes: dict[str, RawPipe], pipe: CachedRawPipe) -> RawPipe:
    return pipes[pipe._source_name]


def get_root_pipe(pipes: dict[str, RawPipe], pipe: RawPipe) -> RawSource:
    while not isinstance(pipe, RawSource):
        pipe = pipes[pipe._source_name]
    return pipe


def state_bids_path(
        state: dict[str, Any],
        *,
        noise: bool = False,
) -> BIDSPath:
    path = bids_path(state)
    return path if not noise else path.find_empty_room()


class RawBadChannelsInput(Input[list[str]]):
    name = RAW_BAD_CHANNELS_INPUT

    def __init__(self, raw: RawDerivative):
        self.raw = raw

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        pipe = self.raw.pipe(ctx.get('raw'))
        noise = ctx.option('noise', False)
        path = state_bids_path(ctx.state)
        return {
            'raw': pipe.name,
            'noise': noise,
            'pipeline': pipe._as_dict(),
            'bad_channels': pipe.load_bad_channels(path, noise=noise, pipes=self.raw.pipes),
        }

    def load(self, ctx: DerivativeContext) -> list[str]:
        pipe = self.raw.pipe(ctx.get('raw'))
        return pipe.load_bad_channels(state_bids_path(ctx.state), noise=ctx.option('noise', False), pipes=self.raw.pipes)


class RawMEEGInput(Input[mne.io.BaseRaw]):
    name = RAW_MEEG_INPUT

    def __init__(self, raw: RawDerivative):
        self.raw = raw

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        if ctx.option('add_bads', True) is True:
            return (
                Dependency(
                    RAW_BAD_CHANNELS_INPUT,
                    state=lambda c: {'raw': c.get('raw')},
                    options=lambda c: {'noise': c.option('noise', False)},
                ),
            )
        return ()

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        pipe = self.raw.pipe(ctx.get('raw'))
        noise = ctx.option('noise', False)
        path = state_bids_path(ctx.state, noise=noise)
        return {
            'source': file_fingerprint(
                ctx.get('root'),
                path.fpath,
                'raw-source',
                metadata={
                    'raw': pipe.name,
                    'noise': noise,
                    'pipeline': pipe._as_dict(),
                },
            ),
        }

    def load(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        pipe = self.raw.pipe(ctx.get('raw'))
        return pipe.load(
            state_bids_path(ctx.state),
            add_bads=ctx.option('add_bads', True),
            preload=ctx.option('preload', False),
            noise=ctx.option('noise', False),
            pipes=self.raw.pipes,
        )


class ICAInput(Input[mne.preprocessing.ICA]):
    name = ICA_INPUT
    key_fields = ('subject', 'session', 'acquisition', 'run', 'split', 'raw')
    version = 1

    def __init__(self, raw: RawDerivative):
        self.raw = raw

    def _pipe(self, ctx: DerivativeContext) -> RawICA:
        pipe = self.raw.pipe(ctx.get('raw'))
        if not isinstance(pipe, RawICA):
            raise RuntimeError(f"raw={pipe.name!r} does not define an ICA input")
        return pipe

    def _path(self, ctx: DerivativeContext) -> Path:
        return self._pipe(ctx).ica_artifact_path(ctx)

    def _key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return canonical_state_subset(ctx.state, self.key_fields)

    def _manifest(self, ctx: DerivativeContext) -> ArtifactManifest | None:
        return ctx.registry.read_manifest(ctx.registry.manifest_path(self._path(ctx)))

    def _load_value(self, ctx: DerivativeContext) -> mne.preprocessing.ICA:
        return self._pipe(ctx).load_ica(state_bids_path(ctx.state))

    def _build_manifest(
            self,
            ctx: DerivativeContext,
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
            provenance=ctx.registry.canonicalize({'n_components': value.n_components_, 'exclude': list(value.exclude)}),
        )

    def is_valid(self, ctx: DerivativeContext) -> bool:
        path = self._path(ctx)
        manifest = self._manifest(ctx)
        if manifest is None or not path.exists():
            return False
        current = self._build_manifest(ctx, self._load_value(ctx))
        return (
            manifest.schema_version == current.schema_version
            and manifest.derivative == current.derivative
            and manifest.derivative_version == current.derivative_version
            and manifest.key == current.key
            and manifest.fingerprint == current.fingerprint
            and manifest.dependencies == current.dependencies
        )

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return self._pipe(ctx).ica_dependencies(ctx)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        pipe = self._pipe(ctx)
        path = pipe._ica_path(state_bids_path(ctx.state))
        return {
            'raw': pipe.name,
            'pipe': pipe._as_dict(),
            'runs': self.raw.runs,
            'ica_path': relpath(path, ctx.get('root')),
            'exists': exists(path),
        }

    def dependency_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        fingerprint = dict(self.fingerprint(ctx))
        path = self._pipe(ctx)._ica_path(state_bids_path(ctx.state))
        fingerprint['ica_file'] = file_fingerprint(ctx.get('root'), path, 'ica-file')
        fingerprint['exclude'] = self._pipe(ctx).load_ica(state_bids_path(ctx.state)).exclude if exists(path) else []
        return fingerprint

    def load(self, ctx: DerivativeContext) -> mne.preprocessing.ICA:
        path = self._path(ctx)
        if not exists(path):
            raise FileMissingError(f"ICA file {basename(path)} does not exist. Run e.make_ica() to create it.")
        if not self.is_valid(ctx):
            raise ProtectedArtifactError(self.name, path)
        return self._load_value(ctx)

    def materialize(
            self,
            ctx: DerivativeContext,
            allow_protected_overwrite: bool = False,
    ) -> mne.preprocessing.ICA:
        if self.is_valid(ctx):
            return self.load(ctx)
        path = self._path(ctx)
        if exists(path) and not allow_protected_overwrite:
            raise ProtectedArtifactError(self.name, path)
        value = self._pipe(ctx).build_ica(ctx, self.raw.pipes)
        path.parent.mkdir(parents=True, exist_ok=True)
        value.save(path, overwrite=True)
        ctx.registry.write_manifest(ctx.registry.manifest_path(path), self._build_manifest(ctx, value))
        return self.load(ctx)


class RawDerivative(Derivative[mne.io.BaseRaw]):
    name = RAW_NODE_NAME
    path_template = 'cached-raw-file'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw')
    cache_policy = CachePolicy.OPTIONAL

    def __init__(self, pipes: dict[str, RawPipe], runs: tuple[str, ...] | None = None):
        self.pipes = pipes
        self.runs = runs
        for pipe in pipes.values():
            if isinstance(pipe, RawICA):
                pipe.runs = runs
        self.bad_channels_input = RawBadChannelsInput(self)
        self.meeg_input = RawMEEGInput(self)
        self.ica_input = ICAInput(self) if any(isinstance(pipe, RawICA) for pipe in pipes.values()) else None

    def __getitem__(self, key: str) -> RawPipe:
        return self.pipes[key]

    def __iter__(self):
        return iter(self.pipes)

    def values(self):
        return self.pipes.values()

    def pipe(self, raw: str) -> RawPipe:
        return self.pipes[raw]

    def root_pipe(self, raw: str | RawPipe) -> RawSource:
        pipe = self.pipe(raw) if isinstance(raw, str) else raw
        return get_root_pipe(self.pipes, pipe)

    def ica_pipe(self, raw: str | RawPipe) -> RawICA:
        pipe = self.pipe(raw) if isinstance(raw, str) else raw
        while not isinstance(pipe, RawICA):
            if isinstance(pipe, RawSource):
                raise ValueError(f"raw={pipe.name!r} does not involve ICA")
            if isinstance(pipe, RawApplyICA):
                pipe = self.pipes[pipe._ica_source]
            else:
                pipe = self.pipes[pipe._source_name]
        return pipe

    def input_nodes(self) -> tuple[DependencyNode[Any], ...]:
        nodes = [self.meeg_input, self.bad_channels_input]
        if self.ica_input is not None:
            nodes.append(self.ica_input)
        return tuple(nodes)

    def should_cache(
            self,
            ctx: DerivativeContext,
            cache: bool | None,
    ) -> bool:
        if cache is not None:
            return cache
        pipe = self.pipe(ctx.get('raw'))
        return isinstance(pipe, CachedRawPipe) and pipe._cache

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        raw = ctx.get('raw')
        pipe = self.pipe(raw)
        if isinstance(pipe, RawSource):
            return (
                Dependency(
                    RAW_MEEG_INPUT,
                    state=lambda c, raw_name=raw: {'raw': raw_name},
                    options=lambda c: {
                        'add_bads': c.option('add_bads', True),
                        'preload': c.option('preload', False),
                        'noise': c.option('noise', False),
                    },
                ),
            )

        deps = [
            Dependency(
                self.name,
                state=lambda c, raw_name=pipe._source_name: {'raw': raw_name},
                options=lambda c: {'add_bads': True, 'preload': False, 'noise': c.option('noise', False)},
            ),
        ]
        if isinstance(pipe, RawICA):
            deps.append(Dependency(ICA_INPUT, state=lambda c, raw_name=pipe.name: {'raw': raw_name}))
        elif isinstance(pipe, RawApplyICA):
            deps.append(Dependency(ICA_INPUT, state=lambda c, raw_name=pipe._ica_source: {'raw': raw_name}))
        return tuple(deps)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        pipe = self.pipe(ctx.get('raw'))
        return {
            'raw': pipe.name,
            'noise': bool(ctx.option('noise', False)),
            'pipe': pipe._as_dict(),
        }

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = cached_raw_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def build(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        pipe = self.pipe(ctx.get('raw'))
        if isinstance(pipe, RawSource):
            return ctx.load(RAW_MEEG_INPUT)

        noise = ctx.option('noise', False)
        bids_path = state_bids_path(ctx.state)

        raw = load_raw_dependency(ctx, pipe._source_name, add_bads=True, preload=True, noise=noise)
        if isinstance(pipe, RawICA):
            ica = ctx.load(ICA_INPUT, state={'raw': pipe.name})
            return pipe.apply_ica(bids_path, raw, ica, pipe.name, pipes=self.pipes)
        if isinstance(pipe, RawApplyICA):
            ica = ctx.load(ICA_INPUT, state={'raw': pipe._ica_source})
            ica_pipe = self.pipe(pipe._ica_source)
            return ica_pipe.apply_ica(bids_path, raw, ica, pipe.name, pipes=self.pipes)
        return pipe._make(bids_path, True, noise=noise, raw=raw, pipes=self.pipes)

    def load(self, ctx: DerivativeContext, path: Path) -> mne.io.BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'This filename', module='mne')
            raw = mne.io.read_raw_fif(
                path,
                preload=ctx.option('preload', False),
                verbose=MNE_VERBOSITY,
            )
        add_bads = ctx.option('add_bads', True)
        pipe = self.pipe(ctx.get('raw'))
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is True:
            raw.info['bads'] = pipe.load_bad_channels(state_bids_path(ctx.state), noise=ctx.option('noise', False), pipes=self.pipes)
        elif add_bads is False:
            raw.info['bads'] = []
        else:
            raise TypeError(f"{add_bads=}")
        return raw

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.io.BaseRaw,
    ) -> None:
        value.save(path, overwrite=True, verbose='ERROR')


def load_raw_dependency(
        ctx: DerivativeContext,
        raw: str | None = None,
        *,
        pipe: RawPipe | None = None,
        add_bads: AddBadsArg = True,
        preload: bool = False,
        noise: bool = False,
        state: dict[str, Any] | None = None,
) -> mne.io.BaseRaw:
    merged_state = dict(state or ())
    if raw is None:
        if pipe is None:
            raise TypeError("load_raw_dependency() needs raw or pipe")
        raw = pipe.name
    merged_state['raw'] = raw
    return ctx.load(RAW_NODE_NAME, state=merged_state, options={'add_bads': add_bads, 'preload': preload, 'noise': noise})


def raw_data_dependency(
        ctx: DerivativeContext,
        *,
        raw: str | None = None,
        pipe: RawPipe | None = None,
        label: str | None = None,
        noise: bool = False,
        add_bads: AddBadsArg = True,
) -> Dependency:
    if raw is None:
        raw = pipe.name if pipe is not None else ctx.get('raw')
    return Dependency(
        RAW_NODE_NAME,
        label=label,
        state=lambda c, raw_name=raw: {'raw': raw_name},
        options=lambda c, noise_=noise, add_bads_=add_bads: {'add_bads': add_bads_, 'preload': False, 'noise': noise_},
    )


class ProcessedRawDerivative(Derivative[mne.io.BaseRaw]):
    path_template = 'cached-raw-file'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw')
    cache_policy = CachePolicy.OPTIONAL

    def __init__(self, pipe: CachedRawPipe):
        self.pipe = pipe
        self.name = raw_cache_node_name(pipe.name)

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return self.pipe.cache_dependencies(ctx)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.pipe.cache_fingerprint(ctx)

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> Path:
        return self.pipe.cache_artifact_path(ctx, mkdir)

    def build(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        return self.pipe.build_cache(ctx)

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.io.BaseRaw:
        return self.pipe.load_cache(ctx, path)

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.io.BaseRaw,
    ) -> None:
        self.pipe.save_cache(ctx, path, value)


class ICADerivative(Derivative[mne.preprocessing.ICA]):
    path_template = 'ica-file'
    key_fields = ('subject', 'session', 'acquisition', 'run', 'split', 'raw')

    def __init__(self, pipe: RawICA):
        self.pipe = pipe
        self.name = ica_cache_node_name(pipe.name)

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return self.pipe.ica_dependencies(ctx)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        fingerprint = self.pipe.ica_fingerprint(ctx)
        fingerprint.pop('exclude', None)
        return fingerprint

    def dependency_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.pipe.ica_fingerprint(ctx)

    def can_reindex_protected_artifact(
            self,
            ctx: DerivativeContext,
            path: Path,
            manifest,
            cache: bool | None = None,
    ) -> bool:
        if manifest.key != canonical_state_subset(ctx.state, self.key_fields):
            return False
        if manifest.dependencies != ctx.registry.dependency_fingerprints(self, ctx, cache):
            return False
        current = dict(self.fingerprint(ctx))
        previous = dict(manifest.fingerprint)
        current.pop('exclude', None)
        previous.pop('exclude', None)
        return current == previous

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> Path:
        return self.pipe.ica_artifact_path(ctx, mkdir)

    def build(self, ctx: DerivativeContext) -> mne.preprocessing.ICA:
        return self.pipe.build_ica(ctx)

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.preprocessing.ICA:
        return self.pipe.load_ica_cache(ctx, path)

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.preprocessing.ICA,
    ) -> None:
        self.pipe.save_ica_cache(ctx, path, value)


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

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
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
        self._kwargs = kwargs

    def _can_link(self, pipes: dict[str, RawPipe]) -> bool:
        return True

    def _link(
            self,
            name: str,
            pipes: dict[str, RawPipe],
            cache_path: str,
            log: logging.Logger,
    ) -> RawPipe:
        return RawPipe._link_base(self, name, log)

    def _raw_path(self, path: BIDSPath) -> str:
        "Get path to the raw file. Enforce existence."
        raw_path = str(path.fpath)
        if not exists(raw_path):
            raise FileMissingError(f"Raw input file does not exist at expected location {path.fpath}")
        return raw_path

    def _bads_path(self, path: BIDSPath) -> str:
        return str(path.copy().update(
            suffix='channels',
            extension='.tsv',
            split=None,
        ).fpath)

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
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
        raw = reader(
            raw_path,
            preload=preload,
            verbose=MNE_VERBOSITY,
        )
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

    def load_derivative_name(self) -> str | None:
        return self.raw_meeg_input_name()

    def input_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return (
            RawMEEGInput(self),
            *RawPipe.input_nodes(self),
        )

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        return self.load(path).info

    def load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None) -> list[str]:
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
        # create channels file
        self.log.info("No channels.tsv found for %s, creating an empty one.", path.fpath)
        makedirs(Path(bads_path).parent, exist_ok=True)
        if exists(path.fpath):
            raw = self._load(path, preload=False)
            ch_names = raw.ch_names
        else:
            raise FileMissingError(f"Raw input file does not exist at expected location {path.fpath}")
        ch_status = ['bad' if ch in raw.info['bads'] else 'good' for ch in ch_names]
        channels_df = pd.DataFrame({
            'name': ch_names,
            'status': ch_status,
        })
        channels_df.to_csv(bads_path, sep='\t', index=False)
        return channels_df.query('status == "bad"')['name'].tolist()

    def make_bad_channels(
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
        old_bads = self.load_bad_channels(path)
        if old_bads is not None and not redo:
            new_bads = sorted(set(old_bads).union(new_bads))
        # print change
        self.log.info("Bad channels: %s -> %s for %s", old_bads, new_bads, self._bads_path(path))
        if new_bads == old_bads:
            return
        # write new bad channels
        if redo:
            mark_channels(path, ch_names='all', status='good', verbose=MNE_VERBOSITY)
        if len(new_bads):
            mark_channels(path, ch_names=new_bads, status='bad', verbose=MNE_VERBOSITY)

    def make_bad_channels_auto(
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
        sysname = self.get_sysname(raw.info, path.subject, path.datatype)
        raw = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=self.adjacency)
        bad_chs.extend(raw.sensor.names[raw.std('time') < flat])
        self.make_bad_channels(path, bad_chs, redo)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = RawPipe._as_dict(self, args)
        out.update(self._kwargs)
        if self.rename_channels:
            out['rename_channels'] = self.rename_channels
        if self.montage:
            if isinstance(self.montage, mne.channels.DigMontage):
                out['montage'] = Sensor.from_montage(self.montage)
            else:
                out['montage'] = self.montage
        if self.adjacency is not None:
            out['connectivity'] = self.adjacency
        return out

    def get_adjacency(self, data: str, pipes: dict[str, RawPipe] = None) -> str | list[tuple[str, str]] | Path | None:
        if data == 'eog':
            return None
        else:
            return self.adjacency

    def get_sysname(
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
    # set on linking
    cache_path: str = None

    def __init__(self, source, cache=True):
        RawPipe.__init__(self)
        self._source_name = source
        self._cache = cache

    def _can_link(self, pipes: dict[str, RawPipe]) -> bool:
        return self._source_name in pipes

    def _link(
            self,
            name: str,
            pipes: dict[str, RawPipe],
            cache_path: str,
            log: logging.Logger,
    ) -> RawPipe:
        if self._source_name not in pipes:
            raise DefinitionError(f"{self.__class__.__name__} {name!r} source {self._source_name!r} does not exist")
        out = RawPipe._link_base(self, name, log)
        out.cache_path = cache_path
        return out

    def source_name(self) -> str:
        return self._source_name

    def _cache_path(self, path: BIDSPath) -> str:
        "Get path to the cached raw file"
        return self.cache_path.format(raw=self.name, suffix=path.datatype, **path.entities)

    def raw_cache_node_name(self) -> str:
        return raw_cache_node_name(self.name)

    def cache_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return (ProcessedRawDerivative(self),)

    def cache_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            raw_data_dependency(
                ctx,
                raw=self._source_name,
                noise=ctx.option('noise', False),
            ),
        )

    def cache_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'raw': ctx.get('raw'),
            'noise': bool(ctx.option('noise', False)),
            'pipe': self._as_dict(),
        }

    def cache_artifact_path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> Path:
        path = Path(self._cache_path(state_bids_path(ctx.state, noise=ctx.option('noise', False))))
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_source_raw(
            self,
            ctx: DerivativeContext,
            *,
            preload: bool = False,
            add_bads: AddBadsArg = True,
            noise: bool = False,
            state: dict[str, Any] | None = None,
    ) -> mne.io.BaseRaw:
        return load_raw_dependency(
            ctx,
            self._source_name,
            add_bads=add_bads,
            preload=preload,
            noise=noise,
            state=state,
        )

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        path = state_bids_path(ctx.state)
        raw = self.load_source_raw(ctx, preload=True, noise=noise)
        return self._make(path, True, noise=noise, raw=raw, pipes=None)

    def load_cache(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.io.BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'This filename', module='mne')
            raw = mne.io.read_raw_fif(
                path,
                preload=ctx.option('preload', False),
                verbose=MNE_VERBOSITY,
            )
        add_bads = ctx.option('add_bads', True)
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is True:
            raw.info['bads'] = self.load_bad_channels(state_bids_path(ctx.state), noise=ctx.option('noise', False), pipes=None)
        elif add_bads is False:
            raw.info['bads'] = []
        else:
            raise TypeError(f"{add_bads=}")
        return raw

    def save_cache(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.io.BaseRaw,
    ) -> None:
        value.save(path, overwrite=True, verbose='ERROR')

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        return self._make(path, preload, noise=noise, pipes=pipes)

    def load_derivative_name(self) -> str | None:
        return self.raw_cache_node_name()

    def load_derivative_cache(self) -> bool | None:
        return self._cache

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        return get_source_pipe(pipes, self).load_info(path, pipes)

    def load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None) -> list[str]:
        return get_source_pipe(pipes, self).load_bad_channels(path, noise=noise, pipes=pipes)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
            pipes: dict[str, RawPipe] = None,
    ) -> None:
        get_source_pipe(pipes, self).make_bad_channels(path, bad_chs, redo, noise=noise, pipes=pipes)

    def make_bad_channels_auto(self, *args, **kwargs) -> None:
        pipes = kwargs.pop('pipes', None)
        get_source_pipe(pipes, self).make_bad_channels_auto(*args, pipes=pipes, **kwargs)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = RawPipe._as_dict(self, args)
        out['source'] = self._source_name
        return out

    def get_adjacency(self, data: str, pipes: dict[str, RawPipe] = None) -> str | list[tuple[str, str]] | Path:
        return get_source_pipe(pipes, self).get_adjacency(data, pipes)

    def get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
            pipes: dict[str, RawPipe] = None,
    ) -> str | None:
        return get_source_pipe(pipes, self).get_sysname(info, subject, data, pipes)


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
        self.args = (l_freq, h_freq)
        self.kwargs = kwargs
        self.n_jobs = n_jobs
        # mne backwards compatibility (fir_design default change 0.15 -> 0.16)
        if 'use_kwargs' in kwargs:
            self._use_kwargs = kwargs.pop('use_kwargs')
        else:
            self._use_kwargs = kwargs

    def filter_ndvar(self, ndvar, **kwargs):
        return filter_data(ndvar, *self.args, **self._use_kwargs, **kwargs)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, preload=True, noise=noise, pipes=pipes)
        self.log.info("Raw %s: filtering for %s...", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        raw.filter(*self.args, **self._use_kwargs, n_jobs=self.n_jobs, verbose=MNE_VERBOSITY)
        return raw

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        info = super().load_info(path, pipes)
        l_freq, h_freq = self.args
        if l_freq and l_freq > (info['highpass'] or 0):
            with info._unlock():
                info['highpass'] = float(l_freq)
        if h_freq and h_freq < (info['lowpass'] or info['sfreq']):
            with info._unlock():
                info['lowpass'] = float(h_freq)
        return info

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'args', 'kwargs'])


class RawFilterElliptic(CachedRawPipe):

    def __init__(self, source, low_stop, low_pass, high_pass, high_stop, gpass, gstop):
        CachedRawPipe.__init__(self, source)
        self.args = (low_stop, low_pass, high_pass, high_stop, gpass, gstop)

    def _sos(self, sfreq):
        nyq = sfreq / 2.
        low_stop, low_pass, high_pass, high_stop, gpass, gstop = self.args
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

    def filter_ndvar(self, ndvar):
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
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, preload=True, noise=noise, pipes=pipes)
        self.log.info("Raw %s: filtering for %s...", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        sos = self._sos(raw.info['sfreq'])
        for i in picks:
            raw._data[i] = signal.sosfilt(sos, raw._data[i])
        low, high = self.args[1], self.args[2]
        if high and raw.info['lowpass'] > high:
            raw.info['lowpass'] = float(high)
        if low and raw.info['highpass'] < low:
            raw.info['highpass'] = float(low)
        return raw

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'args'])


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

    # set on linking
    ica_path: str = None
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

    def load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None) -> list[str]:
        source_pipe = get_source_pipe(pipes, self)
        bad_chs = set()
        for task in self.task:
            path_ = path.copy().update(task=task)
            bad_chs.update(source_pipe.load_bad_channels(path_, pipes=pipes))
        if noise:
            bad_chs.update(source_pipe.load_bad_channels(path, noise=noise, pipes=pipes))
        return sorted(bad_chs)

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        info = super().load_info(path, pipes)
        info['bads'] = self.load_bad_channels(path, pipes=pipes)
        return info

    def _ica_path(self, path: BIDSPath) -> str:
        return self.ica_path.format(raw='ica', suffix=path.datatype, **path.entities)

    def ica_cache_node_name(self) -> str:
        return ica_cache_node_name(self.name)

    def cache_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return (*CachedRawPipe.cache_nodes(self), ICADerivative(self))

    def cache_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            *CachedRawPipe.cache_dependencies(self, ctx),
            Dependency(
                self.ica_cache_node_name(),
                state=lambda c, raw_name=self.name: {'raw': raw_name},
            ),
        )

    def load_ica(self, path: BIDSPath) -> mne.preprocessing.ICA:
        ica_path = self._ica_path(path)
        if not exists(ica_path):
            raise FileMissingError(f"ICA file {basename(ica_path)} does not exist for raw={self.name!r}. Run e.make_ica() to create it.")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Version 0.23 introduced max_iter', DeprecationWarning)
            return mne.preprocessing.read_ica(ica_path)

    def _source_states(
            self,
            ctx: DerivativeContext,
    ) -> list[dict[str, Any]]:
        out = []
        runs = self.runs or (None,)
        for task in self.task:
            for run in runs:
                state = {'raw': self._source_name, 'task': task}
                if run is not None:
                    state['run'] = run
                out.append(state)
        return out

    def ica_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        deps = []
        for i, state in enumerate(self._source_states(ctx)):
            source_state = dict(state)
            stem = f"source-{i}"
            deps.append(
                raw_data_dependency(
                    ctx,
                    raw=source_state.pop('raw'),
                    label=f'{stem}:raw',
                    add_bads=False,
                )
            )
            deps[-1] = Dependency(
                deps[-1].name,
                label=deps[-1].label,
                state=lambda c, state_=state: dict(state_),
                options=deps[-1].options,
            )
            deps.append(
                Dependency(
                    RAW_BAD_CHANNELS_INPUT,
                    label=f'{stem}:bads',
                    state=lambda c, state_=state: dict(state_),
                )
            )
        return tuple(deps)

    def ica_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        bids_path = state_bids_path(ctx.state)
        ica_path = self._ica_path(bids_path)
        exclude = []
        if exists(ica_path):
            exclude = self.load_ica(bids_path).exclude
        return {
            'raw': ctx.get('raw'),
            'pipe': self._as_dict(),
            'runs': self.runs,
            'exclude': exclude,
        }

    def ica_artifact_path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> Path:
        path = Path(self._ica_path(state_bids_path(ctx.state)))
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

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

    def load_concatenated_source_raw(
            self,
            path: BIDSPath,
            tasks: tuple[str],
            runs: tuple[str],
            pipes: dict[str, RawPipe],
    ) -> mne.io.BaseRaw:
        "Concatenate raws from different tasks and runs."
        # NOTE: this use bad channels in RawICA while loading tasks from user input.
        source_pipe = get_source_pipe(pipes, self)
        bad_channels = self.load_bad_channels(path, pipes=pipes)
        path_list = []
        for task in tasks:
            path_ = path.copy().update(task=task)
            if not runs:
                path_list.append(path_)
                continue
            for run in runs:
                path_list.append(path_.copy().update(run=run))
        raw = source_pipe.load(path_list[0], bad_channels, pipes=pipes)
        for path_ in path_list[1:]:
            raw_ = source_pipe.load(path_, bad_channels, pipes=pipes)
            raw.append(raw_)
        return raw

    def build_ica(
            self,
            ctx: DerivativeContext,
            pipes: dict[str, RawPipe],
    ) -> mne.preprocessing.ICA:
        bids_path = state_bids_path(ctx.state)
        bad_channels = self.load_bad_channels(bids_path, pipes=pipes)
        path_list = []
        runs = self.runs or ()
        for task in self.task:
            if not runs:
                path_list.append({'task': task})
                continue
            for run in runs:
                path_list.append({'task': task, 'run': run})

        raw = self.load_source_raw(
            ctx,
            add_bads=bad_channels,
            preload=True,
            state=path_list[0],
        )
        for state in path_list[1:]:
            raw_ = self.load_source_raw(
                ctx,
                add_bads=bad_channels,
                preload=True,
                state=state,
            )
            raw.append(raw_)
        return self.fit_ica(raw, ctx.get('subject'))

    def load_ica_cache(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.preprocessing.ICA:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Version 0.23 introduced max_iter', DeprecationWarning)
            return mne.preprocessing.read_ica(path)

    def save_ica_cache(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.preprocessing.ICA,
    ) -> None:
        value.save(path, overwrite=True)

    def fit_ica(
            self,
            raw: mne.io.BaseRaw,
            subject: str,
    ) -> mne.preprocessing.ICA:
        self.log.info("Raw %s: computing ICA decomposition for %s", self.name, subject)
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
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, preload=True, noise=noise, pipes=pipes)
        return self._apply(path, raw, self.name, pipes=pipes)

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        raise RuntimeError("RawICA.build_cache() is obsolete; use RawDerivative to materialize cached raws")

    def apply_ica(
            self,
            path: BIDSPath,
            raw: mne.io.BaseRaw,
            ica: mne.preprocessing.ICA,
            raw_name: str,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        self.log.debug("Raw %s: applying ICA for %s...", raw_name, path.fpath)
        raw.info['bads'] = [ch for ch in self.load_bad_channels(path, pipes=pipes) if ch in raw.ch_names]
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
    ) -> mne.io.BaseRaw:
        return self.apply_ica(path, raw, self.load_ica(path), raw_name, pipes=pipes)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, [*args, 'task', 'kwargs'])
        if self.fit_kwargs:
            out['fit_kwargs'] = self.fit_kwargs
        return out


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

    def __init__(
            self,
            source: str,
            ica: str,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self._ica_source = ica

    def _can_link(self, pipes: dict[str, RawPipe]) -> bool:
        return CachedRawPipe._can_link(self, pipes) and self._ica_source in pipes

    def _link(
            self,
            name: str,
            pipes: dict[str, RawPipe],
            cache_path: str,
            log: logging.Logger,
    ) -> RawPipe:
        return CachedRawPipe._link(self, name, pipes, cache_path, log)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, args)
        out['ica_source'] = self._ica_source
        return out

    def load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None) -> list[str]:
        return pipes[self._ica_source].load_bad_channels(path, noise=noise, pipes=pipes)

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        info = super().load_info(path, pipes)
        info['bads'] = self.load_bad_channels(path, pipes=pipes)
        return info

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, preload=True, noise=noise, pipes=pipes)
        return pipes[self._ica_source]._apply(path, raw, self.name, pipes=pipes)

    def cache_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            *CachedRawPipe.cache_dependencies(self, ctx),
            Dependency(
                ica_cache_node_name(self._ica_source),
                state=lambda c, raw_name=self._ica_source: {'raw': raw_name},
            ),
        )

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        raise RuntimeError("RawApplyICA.build_cache() is obsolete; use RawDerivative to materialize cached raws")


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
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, noise=noise, pipes=pipes)
        sysname = self.get_sysname(raw.info, path.subject, path.datatype, pipes)
        adjacency = self.get_adjacency(path.datatype, pipes)
        raw_ndvar = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=adjacency)
        raw.info['bads'].extend(raw_ndvar.sensor.names[raw_ndvar.std('time') < self.flat])
        self.log.info("Raw %s: computing Maxwell filter for %s", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            coord_frame = 'meg' if noise else 'head'
            return mne.preprocessing.maxwell_filter(raw, bad_condition=self.bad_condition, coord_frame=coord_frame, verbose=MNE_VERBOSITY, **self.kwargs)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'kwargs'])


class RawOversampledTemporalProjection(CachedRawPipe):
    """Oversampled temporal projection: see :func:`mne.preprocessing.oversampled_temporal_projection`"""

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
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, noise=noise, pipes=pipes)
        self.log.info("Raw %s: computing oversampled temporal projection for %s", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            return mne.preprocessing.oversampled_temporal_projection(raw, self.duration)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'duration'])


class RawUpdateBadChannels(CachedRawPipe):

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
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, preload=preload, noise=noise, pipes=pipes)
        return raw

    def load_bad_channels(self, path: BIDSPath, noise: bool = False, pipes: dict[str, RawPipe] = None) -> list[str]:
        bad_channels = get_source_pipe(pipes, self).load_bad_channels(path, noise=noise, pipes=pipes)
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

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'bad_channels'])


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

    def __init__(
            self,
            source: str,
            reference: str | Sequence[str] = 'average',
            add: str | Sequence[str] = None,
            drop: str | Sequence[str] = None,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        if not isinstance(reference, str):
            reference = sequence_arg('reference', reference, allow_none=False, sequence_type=list)
        self.reference = reference
        self.add = sequence_arg('add', add, sequence_type=list)
        self.drop = sequence_arg('drop', drop, sequence_type=list)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
            raw: mne.io.BaseRaw = None,
            pipes: dict[str, RawPipe] = None,
    ) -> mne.io.BaseRaw:
        if raw is None:
            raw = get_source_pipe(pipes, self).load(path, preload=True, noise=noise, pipes=pipes)
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

    def load_info(self, path: BIDSPath, pipes: dict[str, RawPipe] = None) -> mne.Info:
        if self.add or self.drop:
            return self.load(path, pipes=pipes).info
        else:
            return super().load_info(path, pipes)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, [*args, 'reference'])
        if self.add:
            out['add'] = self.add
        if self.drop:
            out['drop'] = self.drop
        return out

    @staticmethod
    def _normalize_dict(state: dict) -> None:
        if not isinstance(state['reference'], str):
            state['reference'] = sequence_arg('reference', state['reference'])
        for key in ['add', 'drop']:
            if key in state:
                state[key] = sequence_arg(key, state[key])


def assemble_pipeline(
        raw: dict[str, RawPipe],
        tasks: tuple[str],
        cache_path: str,
        ica_path: str,
        log: logging.Logger,
) -> dict[str, RawPipe]:
    "Assemble preprocessing pipeline form a definition in a dict"
    linked_raw = {}
    while raw:
        n = len(raw)
        for key in list(raw):
            if raw[key]._can_link(linked_raw):
                pipe = raw.pop(key)._link(key, linked_raw, cache_path, log)
                if isinstance(pipe, RawICA):
                    missing = set(pipe.task).difference(tasks)
                    if missing:
                        raise DefinitionError(f"RawICA {key!r} lists one or more non-exising tasks: {', '.join(missing)}. Available tasks: {', '.join(tasks)}.")
                    pipe.ica_path = ica_path
                linked_raw[key] = pipe
        if len(raw) == n:
            raise DefinitionError(f"Unable to resolve source for raw {enumeration(raw)}, circular dependency?")
    return linked_raw


def normalize_dict(raw: dict) -> None:
    "Normalize pipeline state with latest pipeline classes"
    for key, params in raw.items():
        pipe_class = globals()[params['type']]
        pipe_class._normalize_dict(params)
