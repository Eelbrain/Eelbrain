# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pre-processing operations based on NDVars

Path templating: passed from `Pipeline`
Entity information: BIDSPath object as a parameter

"""
from __future__ import annotations
import warnings
from copy import deepcopy
import fnmatch
from itertools import chain
import logging
from os import makedirs
from os.path import basename, exists
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
from .derivative_cache import Artifact, CachePolicy, Dependency, DependencyNode, Derivative, DerivativeContext
from .definitions import sequence_arg, typed_arg
from .exceptions import FileMissingError

MNE_VERBOSITY = 'WARNING'
AddBadsArg = bool | Sequence[str]


class RawPipe:
    name: str = None  # set on linking
    log: logging.Logger = None
    pipeline: Any = None

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

    def get_adjacency(self, data: str) -> str | list[tuple[str, str]] | Path:
        raise NotImplementedError

    def get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str
    ) -> str | None:
        raise NotImplementedError

    def load(
            self,
            path: BIDSPath,
            add_bads: AddBadsArg = True,
            preload: bool = False,
            noise: bool = False,
    ) -> mne.io.BaseRaw:
        "Call _load() and add bad channels"
        raw = self._load(path, preload, noise=noise)
        # bad channels
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is True:
            raw.info['bads'] = self.load_bad_channels(path, noise=noise)
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
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def load_info(self, path: BIDSPath) -> mne.Info:
        "Process the info without processing the raw data, return the processed info"
        raise NotImplementedError

    def load_bad_channels(self, path: BIDSPath, noise: bool = False) -> list[str]:
        raise NotImplementedError

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
    ) -> None:
        raise NotImplementedError

    def make_bad_channels_auto(
            self,
            path: BIDSPath,
            flat: float,
            redo: bool,
            noise: bool = False,
    ) -> None:
        raise NotImplementedError

    def cache_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return ()


def raw_cache_node_name(raw: str) -> str:
    return f'raw-cache:{raw}'


def ica_cache_node_name(raw: str) -> str:
    return f'ica-cache:{raw}'


def load_raw_dependency(
        ctx: DerivativeContext,
        raw: str,
        *,
        add_bads: AddBadsArg = True,
        preload: bool = False,
        noise: bool = False,
        state: dict[str, Any] | None = None,
) -> mne.io.BaseRaw:
    state_ = dict(state or ())
    state_['raw'] = raw
    pipe = ctx.pipeline._raw[raw]
    options = {
        'add_bads': add_bads,
        'preload': preload,
        'noise': noise,
    }
    if isinstance(pipe, CachedRawPipe):
        return ctx.load(pipe.raw_cache_node_name(), state=state_, options=options)
    return ctx.load('raw-input-meeg', state=state_, options=options)


def raw_data_dependency(
        ctx: DerivativeContext,
        *,
        raw: str | None = None,
        noise: bool = False,
        add_bads: AddBadsArg = True,
) -> Dependency:
    raw = ctx.get('raw') if raw is None else raw
    pipe = ctx.pipeline._raw[raw]
    if isinstance(pipe, CachedRawPipe):
        return Dependency(
            pipe.raw_cache_node_name(),
            state=lambda c, raw_name=raw: {'raw': raw_name},
            options=lambda c, noise_=noise: {'noise': noise_},
        )
    return Dependency(
        'raw-input-meeg',
        state=lambda c, raw_name=raw: {'raw': raw_name},
        options=lambda c, noise_=noise, add_bads_=add_bads: {
            'noise': noise_,
            'add_bads': add_bads_,
        },
    )


class ProcessedRawDerivative(Derivative[mne.io.BaseRaw]):
    path_template = 'cached-raw-file'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw')
    cache_policy = CachePolicy.OPTIONAL

    def __init__(self, raw_name: str):
        self.raw_name = raw_name
        self.name = raw_cache_node_name(raw_name)

    def _pipe(self, ctx: DerivativeContext) -> CachedRawPipe:
        pipe = ctx.pipeline._raw[self.raw_name]
        if not isinstance(pipe, CachedRawPipe):
            raise RuntimeError(f"ProcessedRawDerivative requires a cached raw pipe, got {pipe.__class__.__name__}")
        return pipe

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return self._pipe(ctx).cache_dependencies(ctx)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self._pipe(ctx).cache_fingerprint(ctx)

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> str:
        return self._pipe(ctx).cache_artifact_path(ctx, mkdir)

    def build(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        return self._pipe(ctx).build_cache(ctx)

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.io.BaseRaw:
        return self._pipe(ctx).load_cache(ctx, artifact)

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.io.BaseRaw,
    ) -> None:
        self._pipe(ctx).save_cache(ctx, artifact, value)


class ICADerivative(Derivative[mne.preprocessing.ICA]):
    path_template = 'ica-file'
    key_fields = ('subject', 'session', 'acquisition', 'run', 'split', 'raw')

    def __init__(self, raw_name: str):
        self.raw_name = raw_name
        self.name = ica_cache_node_name(raw_name)

    def _pipe(self, ctx: DerivativeContext) -> RawICA:
        pipe = ctx.pipeline._raw[self.raw_name]
        if not isinstance(pipe, RawICA):
            raise RuntimeError(f"ICADerivative requires a RawICA pipe, got {pipe.__class__.__name__}")
        return pipe

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return self._pipe(ctx).ica_dependencies(ctx)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        fingerprint = self._pipe(ctx).ica_fingerprint(ctx)
        fingerprint.pop('exclude', None)
        return fingerprint

    def dependency_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self._pipe(ctx).ica_fingerprint(ctx)

    def can_reindex_protected_artifact(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            manifest,
            cache: bool | None = None,
    ) -> bool:
        if manifest.key != ctx.registry.normalize_state(self.key_fields, ctx.state):
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
    ) -> str:
        return self._pipe(ctx).ica_artifact_path(ctx, mkdir)

    def build(self, ctx: DerivativeContext) -> mne.preprocessing.ICA:
        return self._pipe(ctx).build_ica(ctx)

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.preprocessing.ICA:
        return self._pipe(ctx).load_ica_cache(ctx, artifact)

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.preprocessing.ICA,
    ) -> None:
        self._pipe(ctx).save_ica_cache(ctx, artifact, value)


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

    def load_info(self, path: BIDSPath) -> mne.Info:
        return self.load(path).info

    def load_bad_channels(self, path: BIDSPath, noise: bool = False) -> list[str]:
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

    def get_adjacency(self, data: str) -> str | list[tuple[str, str]] | Path | None:
        if data == 'eog':
            return None
        else:
            return self.adjacency

    def get_sysname(
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
    # set on linking
    source: RawPipe = None
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
        out.source = pipes[self._source_name]
        out.cache_path = cache_path
        return out

    def _cache_path(self, path: BIDSPath) -> str:
        "Get path to the cached raw file"
        return self.cache_path.format(raw=self.name, suffix=path.datatype, **path.entities)

    def raw_cache_node_name(self) -> str:
        return raw_cache_node_name(self.name)

    def cache_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return (ProcessedRawDerivative(self.name),)

    def cache_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            raw_data_dependency(
                ctx,
                raw=self.source.name,
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
    ) -> str:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            bids_path = p._bids_path if not ctx.option('noise', False) else p._bids_path.find_empty_room()
            path = self._cache_path(bids_path)
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
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
            self.source.name,
            add_bads=add_bads,
            preload=preload,
            noise=noise,
            state=state,
        )

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            return self._make(p._bids_path, True, noise=ctx.option('noise', False))

    def load_cache(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.io.BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'This filename', module='mne')
            raw = mne.io.read_raw_fif(
                artifact.path,
                preload=ctx.option('preload', False),
                verbose=MNE_VERBOSITY,
            )
        add_bads = ctx.option('add_bads', True)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            if isinstance(add_bads, Sequence):
                raw.info['bads'] = list(add_bads)
            elif add_bads is True:
                raw.info['bads'] = self.load_bad_channels(p._bids_path, noise=ctx.option('noise', False))
            elif add_bads is False:
                raw.info['bads'] = []
            else:
                raise TypeError(f"{add_bads=}")
        return raw

    def save_cache(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.io.BaseRaw,
    ) -> None:
        value.save(artifact.path, overwrite=True, verbose='ERROR')

    def _load(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
    ) -> mne.io.BaseRaw:
        if self.pipeline is None:
            return self._make(path, preload, noise=noise)

        state = {
            key: value
            for key, value in path.entities.items()
            if value is not None and key in ('subject', 'session', 'task', 'acquisition', 'run', 'split')
        }
        state['raw'] = self.name
        return self.pipeline._load_derivative(
            self.raw_cache_node_name(),
            cache=self._cache,
            state=state,
            options={
                'add_bads': False,
                'preload': preload,
                'noise': noise,
            },
        )

    def load_info(self, path: BIDSPath) -> mne.Info:
        return self.source.load_info(path)

    def load_bad_channels(self, path: BIDSPath, noise: bool = False) -> list[str]:
        return self.source.load_bad_channels(path, noise=noise)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs: tuple[str] | str | int,
            redo: bool,
            noise: bool = False,
    ) -> None:
        self.source.make_bad_channels(path, bad_chs, redo, noise=noise)

    def make_bad_channels_auto(self, *args, **kwargs) -> None:
        self.source.make_bad_channels_auto(*args, **kwargs)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = RawPipe._as_dict(self, args)
        out['source'] = self._source_name
        return out

    def get_adjacency(self, data: str) -> str | list[tuple[str, str]] | Path:
        return self.source.get_adjacency(data)

    def get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
    ) -> str | None:
        return self.source.get_sysname(info, subject, data)


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
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True, noise=noise)
        self.log.info("Raw %s: filtering for %s...", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        raw.filter(*self.args, **self._use_kwargs, n_jobs=self.n_jobs, verbose=MNE_VERBOSITY)
        return raw

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            path = p._bids_path
            raw = self.load_source_raw(ctx, preload=True, noise=noise)
            self.log.info("Raw %s: filtering for %s...", self.name, path.fpath if not noise else path.find_empty_room().fpath)
            raw.filter(*self.args, **self._use_kwargs, n_jobs=self.n_jobs, verbose=MNE_VERBOSITY)
            return raw

    def load_info(self, path: BIDSPath) -> mne.Info:
        info = super().load_info(path)
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
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True, noise=noise)
        self.log.info("Raw %s: filtering for %s...", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        # filter data
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        sos = self._sos(raw.info['sfreq'])
        for i in picks:
            raw._data[i] = signal.sosfilt(sos, raw._data[i])
        # update info
        low, high = self.args[1], self.args[2]
        if high and raw.info['lowpass'] > high:
            raw.info['lowpass'] = float(high)
        if low and raw.info['highpass'] < low:
            raw.info['highpass'] = float(low)
        return raw

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            path = p._bids_path
            raw = self.load_source_raw(ctx, preload=True, noise=noise)
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

    def load_bad_channels(self, path: BIDSPath, noise: bool = False) -> list[str]:
        bad_chs = set()
        for task in self.task:
            path_ = path.copy().update(task=task)
            bad_chs.update(self.source.load_bad_channels(path_))
        if noise:
            bad_chs.update(self.source.load_bad_channels(path, noise=noise))
        return sorted(bad_chs)

    def load_info(self, path: BIDSPath) -> mne.Info:
        info = super().load_info(path)
        info['bads'] = self.load_bad_channels(path)
        return info

    def _ica_path(self, path: BIDSPath) -> str:
        return self.ica_path.format(raw='ica', suffix=path.datatype, **path.entities)

    def ica_cache_node_name(self) -> str:
        return ica_cache_node_name(self.name)

    def cache_nodes(self) -> tuple[DependencyNode[Any], ...]:
        return (*CachedRawPipe.cache_nodes(self), ICADerivative(self.name))

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
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            out = []
            runs = p._runs or (None,)
            for task in self.task:
                for run in runs:
                    state = {'raw': self.source.name, 'task': task}
                    if run is not None:
                        state['run'] = run
                    out.append(state)
        return out

    def ica_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        deps = []
        for state in self._source_states(ctx):
            source_state = dict(state)
            deps.append(
                raw_data_dependency(
                    ctx,
                    raw=source_state.pop('raw'),
                    add_bads=False,
                )
            )
            deps[-1] = Dependency(
                deps[-1].name,
                state=lambda c, state_=state: dict(state_),
                options=deps[-1].options,
            )
            deps.append(
                Dependency(
                    'raw-input-bads',
                    state=lambda c, state_=state: dict(state_),
                )
            )
        return tuple(deps)

    def ica_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            ica_path = self._ica_path(p._bids_path)
            exclude = []
            if exists(ica_path):
                exclude = self.load_ica(p._bids_path).exclude
            return {
                'raw': ctx.get('raw'),
                'pipe': self._as_dict(),
                'runs': p._runs,
                'exclude': exclude,
            }

    def ica_artifact_path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> str:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            path = self._ica_path(p._bids_path)
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    ) -> mne.io.BaseRaw:
        "Concatenate raws from different tasks and runs."
        # NOTE: this use bad channels in RawICA while loading tasks from user input.
        bad_channels = self.load_bad_channels(path)
        path_list = []
        for task in tasks:
            path_ = path.copy().update(task=task)
            if not runs:
                path_list.append(path_)
                continue
            for run in runs:
                path_list.append(path_.copy().update(run=run))
        raw = self.source.load(path_list[0], bad_channels)
        for path_ in path_list[1:]:
            raw_ = self.source.load(path_, bad_channels)
            raw.append(raw_)
        return raw

    def build_ica(self, ctx: DerivativeContext) -> mne.preprocessing.ICA:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)

            bad_channels = self.load_bad_channels(p._bids_path)
            path_list = []
            runs = p._runs or ()
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
            return self.fit_ica(raw, p._bids_path.subject)

    def load_ica_cache(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> mne.preprocessing.ICA:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Version 0.23 introduced max_iter', DeprecationWarning)
            return mne.preprocessing.read_ica(artifact.path)

    def save_ica_cache(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: mne.preprocessing.ICA,
    ) -> None:
        value.save(artifact.path, overwrite=True)

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
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True, noise=noise)
        return self._apply(path, raw, self.name)

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            raw = self.load_source_raw(ctx, preload=True, noise=noise)
            ica = ctx.load(self.ica_cache_node_name(), state={'raw': self.name})
            return self.apply_ica(p._bids_path, raw, ica, self.name)

    def apply_ica(
            self,
            path: BIDSPath,
            raw: mne.io.BaseRaw,
            ica: mne.preprocessing.ICA,
            raw_name: str,
    ) -> mne.io.BaseRaw:
        self.log.debug("Raw %s: applying ICA for %s...", raw_name, path.fpath)
        raw.info['bads'] = [ch for ch in self.load_bad_channels(path) if ch in raw.ch_names]
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
    ) -> mne.io.BaseRaw:
        return self.apply_ica(path, raw, self.load_ica(path), raw_name)

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
    ica_source = None  # set on linking

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
        out = CachedRawPipe._link(self, name, pipes, cache_path, log)
        out.ica_source = pipes[self._ica_source]
        return out

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, args)
        out['ica_source'] = self._ica_source
        return out

    def load_bad_channels(self, path: BIDSPath, noise: bool = False) -> list[str]:
        return self.ica_source.load_bad_channels(path, noise=noise)

    def load_info(self, path: BIDSPath) -> mne.Info:
        info = super().load_info(path)
        info['bads'] = self.load_bad_channels(path)
        return info

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
            noise: bool = False,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True, noise=noise)
        return self.ica_source._apply(path, raw, self.name)

    def cache_dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            *CachedRawPipe.cache_dependencies(self, ctx),
            Dependency(
                self.ica_source.ica_cache_node_name(),
                state=lambda c, raw_name=self.ica_source.name: {'raw': raw_name},
            ),
        )

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            raw = self.load_source_raw(ctx, preload=True, noise=noise)
            ica = ctx.load(self.ica_source.ica_cache_node_name(), state={'raw': self.ica_source.name})
            return self.ica_source.apply_ica(p._bids_path, raw, ica, self.name)


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
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, noise=noise)
        sysname = self.get_sysname(raw.info, path.subject, path.datatype)
        adjacency = self.get_adjacency(path.datatype)
        raw_ndvar = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=adjacency)
        raw.info['bads'].extend(raw_ndvar.sensor.names[raw_ndvar.std('time') < self.flat])
        self.log.info("Raw %s: computing Maxwell filter for %s", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            coord_frame = 'meg' if noise else 'head'
            return mne.preprocessing.maxwell_filter(raw, bad_condition=self.bad_condition, coord_frame=coord_frame, verbose=MNE_VERBOSITY, **self.kwargs)

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            path = p._bids_path
            raw = self.load_source_raw(ctx, noise=noise)
            sysname = self.get_sysname(raw.info, path.subject, path.datatype)
            adjacency = self.get_adjacency(path.datatype)
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
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, noise=noise)
        self.log.info("Raw %s: computing oversampled temporal projection for %s", self.name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            return mne.preprocessing.oversampled_temporal_projection(raw, self.duration)

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            path = p._bids_path
            raw = self.load_source_raw(ctx, noise=noise)
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
    ) -> mne.io.BaseRaw:
        return self.source.load(path, preload=preload, noise=noise)

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        return self.load_source_raw(
            ctx,
            preload=True,
            noise=ctx.option('noise', False),
        )

    def load_bad_channels(self, path: BIDSPath, noise: bool = False) -> list[str]:
        bad_channels = self.source.load_bad_channels(path, noise=noise)
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
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True, noise=noise)
        if self.add:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The locations of multiple reference channels are ignored', module='mne')
                raw = mne.add_reference_channels(raw, self.add, copy=False)
            # apply new channel position
            pipe = self.source
            while not isinstance(pipe, RawSource):
                pipe = pipe.source
            if pipe.montage:
                raw.set_montage(pipe.montage)
        raw.set_eeg_reference(self.reference)
        if self.drop:
            raw = raw.drop_channels(self.drop)
        return raw

    def build_cache(self, ctx: DerivativeContext) -> mne.io.BaseRaw:
        noise = ctx.option('noise', False)
        raw = self.load_source_raw(ctx, preload=True, noise=noise)
        if self.add:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The locations of multiple reference channels are ignored', module='mne')
                raw = mne.add_reference_channels(raw, self.add, copy=False)
            pipe = self.source
            while not isinstance(pipe, RawSource):
                pipe = pipe.source
            if pipe.montage:
                raw.set_montage(pipe.montage)
        raw.set_eeg_reference(self.reference)
        if self.drop:
            raw = raw.drop_channels(self.drop)
        return raw

    def load_info(self, path: BIDSPath) -> mne.Info:
        if self.add or self.drop:
            return self.load(path).info
        else:
            return super().load_info(path)

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
