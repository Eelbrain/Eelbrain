# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Event derivatives - prepare events for epoch extraction (epoch-events).

Dependency structure:

    epoch-events
    ├── PrimaryEpoch (single run) / ContinuousEpoch
    │     └── selected-events
    │           ├── labeled-events
    │           │     ├── events-input   (BIDS sidecar, preferred when present)
    │           │     └── events         (trigger-based fallback)
    │           └── rejection            (epoch-rejection-input | epoch-rejection-channel-model;
    │                                      only when epoch_rejection is set and reject != False)
    │
    ├── PrimaryEpoch (combine runs: run=None and multiple runs exist)
    │     └── selected-events  ×N  (one per run)
    │           ├── labeled-events
    │           │     ├── events-input
    │           │     └── events
    │           └── rejection
    │
    ├── SecondaryEpoch
    │     └── epoch-events  (base epoch, recursively)
    │
    └── SuperEpoch
          └── epoch-events  ×N  (one per sub-epoch)

:class:`EventsInput` (``'events-input'``)
    External input that reads events directly from a BIDS sidecar
    ``*_events.tsv`` file when one is present alongside the raw recording.

:class:`EventsDerivative` (``'events'``)
    Reads raw trigger data for one recording file and applies the experiment's
    :meth:`~Pipeline.fix_events` hook to produce a corrected
    :class:`~eelbrain.Dataset` of trial events.  Used as a fallback when no
    BIDS events sidecar is found.

:class:`LabeledEventsDerivative` (``'labeled-events'``)
    Applies built-in variable definitions and the experiment's
    :meth:`~Pipeline.label_events` hook on top of the cached events.
    Prefers :class:`EventsInput` over :class:`EventsDerivative` when a BIDS
    sidecar file is present.

:class:`SelectedEventsDerivative` (``'selected-events'``)
    Applies epoch-specific trial selection (``sel`` predicate), artifact
    rejection, and bad-channel annotations for a single raw recording file.
    Always restricted to one task/run combination.  Adds the rejection node
    (``epoch-rejection-input`` for a manual rejection, or
    ``epoch-rejection-channel-model`` for an automatic one) as a dependency when
    epoch rejection is active (``epoch_rejection`` is set and ``reject`` is not
    ``False``).

:class:`EpochEventsDerivative` (``'epoch-events'``)
    Epoch-level event aggregation.  For :class:`~epochs.PrimaryEpoch` and
    :class:`~epochs.ContinuousEpoch` this delegates to
    :class:`SelectedEventsDerivative`, combining across runs for combine-all
    epochs.  For :class:`~epochs.SecondaryEpoch` and
    :class:`~epochs.SuperEpoch` it delegates to base/sub-epoch
    ``epoch-events`` nodes.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from .. import load, save
from .._data_obj import Datalist, Dataset, Factor, Var, combine
from .._exceptions import ConfigurationError
from .._info import BAD_CHANNELS, INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS, INTERPOLATE_WINDOWS_MAX
from .derivative_cache import CachePolicy, Dependency, Derivative, Input, Request, UncachedDerivative, file_fingerprint
from .epoch_rejection import EpochRejection, ManualRejection
from .epochs import EPOCH_EXTRACT_OPTIONS, EpochCollection, SecondaryEpoch, SuperEpoch, PrimaryEpoch, ContinuousEpoch, single_recording_run
from .pathing import BIDS_ENTITY_KEYS, bids_path
from .preprocessing import raw_node_name
from .variable_def import Variables


def function_fingerprint(function) -> str:
    """SHA-256 digest of a function's source code, truncated to 16 hex chars.

    Falls back to ``__qualname__`` when source is not accessible (compiled
    extensions, interactive sessions).
    """
    try:
        src = inspect.getsource(function)
    except (OSError, TypeError):
        return getattr(function, '__qualname__', repr(function))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


class EventsInput(Input[Dataset]):
    """Read events from a BIDS sidecar ``*_events.tsv`` file.

    The file is expected to follow the BIDS specification with at least the
    columns ``onset`` (seconds), ``sample`` (integer sample index), and
    ``value`` (integer trigger code).  Additional columns are passed through
    to the returned :class:`~eelbrain.Dataset` so that they are available in
    :meth:`~Pipeline.label_events`.

    """
    name = 'events-input'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run')

    def __init__(
            self,
            raw_extension: str,
    ):
        self.raw_extension = raw_extension

    def _resolve_bids_events_path(self, ctx: Request) -> BIDSPath:
        return bids_path(ctx.root, ctx.state, extension='.tsv', datatype=ctx.datatype, suffix='events')

    def path(self, ctx: Request) -> Path:
        return self._resolve_bids_events_path(ctx).fpath

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return file_fingerprint(ctx.root, self.path(ctx))

    def load(self, ctx: Request) -> Dataset | None:
        path = self.path(ctx)
        if not path.exists():
            return None
        df = pd.read_csv(path, sep='\t')
        entities = {k: ctx.state[k] for k in BIDS_ENTITY_KEYS}
        return Dataset.from_dataframe(df, info=entities)


def _check_ds(ds: Dataset, source: str, info: dict[str, Any]) -> Dataset:
    if not isinstance(ds, Dataset):
        raise ConfigurationError(f"{source} needs to return the events Dataset. Got {ds!r}.")
    if 'sample' not in ds:
        raise ConfigurationError(f"The Dataset returned by {source} does not contain a variable called `sample`. This variable is required to ascribe events to data samples.")
    if 'value' not in ds:
        raise ConfigurationError(f"The Dataset returned by {source} does not contain a variable called `value`. This variable is required to check rejection files.")
    if ds.info is not info:
        # Make sure to keep some required information
        ds.info.update({k: v for k, v in info.items() if k not in ds.info})
    return ds


class EventsDerivative(Derivative[Dataset]):
    """Extract events form M/EEG data files"""
    name = 'events'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'raw')
    cache_suffix = '.pickle'

    def __init__(
            self,
            trigger_shift: float | dict[str | tuple[str, str], float],
            stim_channel: str | list[str],
            merge_triggers: Any,
            preload: bool,
            fix_events,
            owner_name: str,
    ):
        self.trigger_shift = trigger_shift
        self.stim_channel = stim_channel
        self.merge_triggers = merge_triggers
        self.preload = preload
        self.fix_events_impl = fix_events
        self.owner_name = owner_name

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        raw_name = ctx.state['raw']
        return (Dependency(raw_node_name(raw_name), state={'raw': raw_name}, options={'preload': False, 'noise': False}),)

    def _get_trigger_shift(self, subject: str, session: str):
        if isinstance(self.trigger_shift, dict):
            for key in ((subject, session), subject):
                if key in self.trigger_shift:
                    return self.trigger_shift[key]
            return 0
        return self.trigger_shift

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        subject = ctx.state['subject']
        session = ctx.state['session']
        trigger_shift = self._get_trigger_shift(subject, session)
        return {
            'stim_channel': self.stim_channel,
            'merge_triggers': self.merge_triggers,
            'trigger_shift': trigger_shift,
            'fix_events': function_fingerprint(self.fix_events_impl),
        }

    def build(self, ctx: Request) -> Dataset:
        entities = {k: ctx.state[k] for k in BIDS_ENTITY_KEYS}
        subject = entities['subject']
        session = entities['session']
        raw = ctx.load(raw_node_name(ctx.state['raw']))
        if self.preload and not raw.preload:
            raw.load_data()
        try:
            ds = load.mne.events(raw, self.merge_triggers, stim_channel=self.stim_channel)
        except ValueError:
            # No trigger channel present (e.g. sidecar-only dataset); return empty events
            ds = Dataset({'i_start': Var(np.zeros(0, int)), 'trigger': Var(np.zeros(0, int))}, info={'raw': raw})
        del ds.info['raw']
        ds.rename('i_start', 'sample')
        ds.rename('trigger', 'value')
        ds.info['raw.samplingrate'] = raw.info['sfreq']
        ds.info['raw.first_samp'] = raw.first_samp
        ds.info['raw.last_samp'] = raw.last_samp
        ds.info.update(entities)

        trigger_shift = self._get_trigger_shift(subject, session)
        if trigger_shift:
            ds['sample'] += int(round(trigger_shift * ds.info['raw.samplingrate']))

        # Apply e.fix_events()
        info = ds.info
        n_args = len(inspect.signature(self.fix_events_impl).parameters)
        if n_args == 1:
            ds = self.fix_events_impl(ds)
        else:
            raise ValueError(f"{self.owner_name}.label_events {self.label_events_impl!r}: number of arguments: {n_args}; should take one argument, {self.owner_name}.label_events(self, ds) or label_events(ds) ")
        return _check_ds(ds, f'{self.owner_name}.fix_events()', info)

    def load(self, ctx: Request, path: Path) -> Dataset:
        ds = load.unpickle(path)
        ds.info.update({k: ctx.state[k] for k in BIDS_ENTITY_KEYS})
        return ds

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class LabeledEventsDerivative(Derivative[Dataset]):
    """Labeled event dataset produced by applying :meth:`~Pipeline.label_events`.

    Caching is controlled by :attr:`Pipeline.cache_event_labels`.  When
    ``True`` (the default) the labeled events are cached, and the fingerprint
    detects changes to ``label_events`` via source-code hashing.  When
    ``False`` this node is always rebuilt from the cached unlabeled events —
    the correct choice when ``label_events`` reads external files whose changes
    cannot be detected without executing the hook.
    """
    name = 'labeled-events'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'raw')
    cache_suffix = '.pickle'

    def __init__(
            self,
            label_events: Callable[[Dataset], Dataset],
            owner_name: str,
            multi_task: bool,
            multi_session: bool,
            variables: Variables,
            groups: dict[str, Any],
            cache: bool,
    ):
        self.label_events_impl = label_events
        self.owner_name = owner_name
        self.multi_task = multi_task
        self.multi_session = multi_session
        self._variables = variables
        self._groups = groups
        if not cache:
            self.cache_policy = CachePolicy.NEVER

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (
            Dependency('events-input'),
            Dependency('events'),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'variables': self._variables,
            'label_events': function_fingerprint(self.label_events_impl),
        }

    def build(self, ctx: Request) -> Dataset:
        sidecar = ctx.load('events-input')
        trigger_events = ctx.load('events')
        if sidecar is not None:
            ds = sidecar
            # Override sfreq with the authoritative value from the raw file, and
            # adjust samples: BIDS TSV is 0-indexed from file start (MNE-BIDS
            # subtracts raw.first_samp on write), so add it back.
            ds.info['raw.samplingrate'] = trigger_events.info['raw.samplingrate']
            ds.info['raw.first_samp'] = trigger_events.info['raw.first_samp']
            ds.info['raw.last_samp'] = trigger_events.info['raw.last_samp']
            if trigger_events.info['raw.first_samp']:
                ds['sample'] = ds['sample'] + trigger_events.info['raw.first_samp']
        else:
            ds = trigger_events
        ds['subject'] = Factor([ctx.state['subject']], repeat=ds.n_cases, random=True)
        if self.multi_task:
            ds[:, 'task'] = ctx.state['task']
        if self.multi_session:
            ds[:, 'session'] = ctx.state['session']
        self._variables._apply(ds, self._groups)

        # Apply e.label_events()
        info = ds.info
        n_args = len(inspect.signature(self.label_events_impl).parameters)
        if n_args == 1:
            ds = self.label_events_impl(ds)
        else:
            raise ValueError(f"{self.owner_name}.label_events {self.label_events_impl!r}: number of arguments: {n_args}; should take one argument, {self.owner_name}.label_events(self, ds) or label_events(ds) ")
        return _check_ds(ds, f'{self.owner_name}.label_events()', info)

    def load(self, ctx: Request, path: Path) -> Dataset:
        ds = load.unpickle(path)
        ds.info.update({k: ctx.state[k] for k in BIDS_ENTITY_KEYS})
        return ds

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class SelectedEventsDerivative(UncachedDerivative[Dataset]):
    """Selected events for a single raw recording (one task/run).

    Applies epoch-specific trial selection (``sel`` predicate), artifact
    rejection, and bad-channel annotations for a single recording file.
    Always restricted to one task/run combination; multi-run aggregation is
    handled by :class:`EpochEventsDerivative`.
    """
    name = 'selected-events'
    key_fields = ('subject', 'session', 'acquisition', 'run', 'raw', 'epoch', 'epoch_rejection')
    key_options = {
        'reject': True,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'tmin': None,
        'tmax': None,
        'tstop': None,
    }

    def __init__(
            self,
            epochs: dict[str, Any],
            epoch_rejection: dict[str, EpochRejection | None],
    ):
        self.epochs = epochs
        self.epoch_rejection = epoch_rejection

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        reject = ctx.options['reject']
        if reject not in (True, False, 'keep'):
            raise ValueError(f"{reject=}")
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r}; can't load events for epoch collection")
        elif isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            rejection_params = self.epoch_rejection[ctx.state['epoch_rejection']]
            state = {'task': epoch.task}
            if epoch.run:
                state['run'] = epoch.run
            deps = [Dependency('labeled-events', state=state)]
            if rejection_params is not None and reject:
                node = 'epoch-rejection-input' if isinstance(rejection_params, ManualRejection) else 'epoch-rejection-channel-model'
                deps.append(Dependency(node, label='rejection', state=state))
            return tuple(deps)
        elif isinstance(epoch, SecondaryEpoch):
            options = ctx.options_for('selected-events', 'reject', *EPOCH_EXTRACT_OPTIONS)
            state = {'epoch': epoch.sel_epoch}
            return (Dependency('selected-events', options=options, state=state),)
        else:
            raise RuntimeError(f"{epoch=}")

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'epoch': self.epochs[ctx.state['epoch']]}

    def build(self, ctx: Request) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        subject = ctx.state['subject']
        if isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            ds = ctx.load('labeled-events')
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if epoch.n_cases is not None and ds.n_cases != epoch.n_cases:
                raise RuntimeError(f"Number of epochs {ds.n_cases}, expected {epoch.n_cases}")
            ds.index()

            # Trial rejection
            reject = ctx.options['reject']
            rejection_params = self.epoch_rejection[ctx.state['epoch_rejection']]
            if rejection_params is not None and reject:
                rejection_ds = ctx.load('rejection')

                # Handle event mismatches
                if rejection_ds.info.get('epochs.selection') is not None:
                    ds = ds[rejection_ds.info['epochs.selection']]
                if rejection_ds.n_cases != ds.n_cases or np.any(ds['value'] != rejection_ds['value']):
                    raise RuntimeError(f"The epoch selection file contains different events from the data loaded from the raw file. If the events included in the epoch were changed intentionally, redo epoch selection for {subject}/{epoch.name}")

                # Channel interpolation
                if rejection_params.interpolation:
                    ds.info[INTERPOLATE_CHANNELS] = True
                    if INTERPOLATE_CHANNELS in rejection_ds:
                        ds[INTERPOLATE_CHANNELS] = rejection_ds[INTERPOLATE_CHANNELS]
                    else:
                        ds[INTERPOLATE_CHANNELS] = Datalist([[]] * ds.n_cases, INTERPOLATE_CHANNELS, 'strlist')
                    # Time-resolved interpolation windows (long epochs)
                    if INTERPOLATE_WINDOWS in rejection_ds:
                        ds.info[INTERPOLATE_WINDOWS] = True
                        ds.info[INTERPOLATE_WINDOWS_MAX] = rejection_ds.info[INTERPOLATE_WINDOWS_MAX]
                        ds[INTERPOLATE_WINDOWS] = rejection_ds[INTERPOLATE_WINDOWS]
                    else:
                        ds.info[INTERPOLATE_WINDOWS] = False
                else:
                    ds.info[INTERPOLATE_CHANNELS] = False
                    ds.info[INTERPOLATE_WINDOWS] = False

                if reject == 'keep':
                    ds['accept'] = rejection_ds['accept']
                elif reject is True:
                    ds = ds.sub(rejection_ds['accept'])
                elif reject is not False:
                    raise RuntimeError(f"{reject=}")

                ds.info[BAD_CHANNELS] = rejection_ds.info.get(BAD_CHANNELS, [])
            else:
                ds.info[INTERPOLATE_CHANNELS] = False
                ds.info[INTERPOLATE_WINDOWS] = False
                ds.info[BAD_CHANNELS] = []
        elif isinstance(epoch, SecondaryEpoch):
            ds = ctx.load('selected-events')
            if epoch.sel:
                ds = ds.sub(epoch.sel)
                ds.index()
        else:
            raise RuntimeError(f"{epoch=}")

        return epoch._prepare_selected_events(ds, subject, ctx.options)


class EpochEventsDerivative(UncachedDerivative[Dataset]):
    """Epoch-level event aggregation.

    For :class:`~epochs.PrimaryEpoch` and :class:`~epochs.ContinuousEpoch`,
    delegates to :class:`SelectedEventsDerivative`.  For combine-all
    PrimaryEpochs aggregates across runs and adds a ``'run'`` column.  For
    :class:`~epochs.SecondaryEpoch` and :class:`~epochs.SuperEpoch` delegates
    to the appropriate base/sub-epoch ``epoch-events`` nodes.

    Options
    -------
    reject
        Whether to apply artifact rejection (``True``, ``False``, or ``'keep'``).
    """
    name = 'epoch-events'
    key_fields = ('subject', 'session', 'acquisition', 'epoch', 'raw', 'epoch_rejection')
    key_options = {
        'reject': True,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'tmin': None,
        'tmax': None,
        'tstop': None,
    }

    def __init__(
            self,
            epochs: dict[str, Any],
            runs_for: dict[tuple[str, str, str, str], tuple[str, ...]],
    ):
        self.epochs = epochs
        self._runs_for = runs_for

    def _find_runs(self, ctx: Request, epoch) -> tuple[str, ...]:
        """Runs to aggregate over"""
        if isinstance(epoch, PrimaryEpoch):
            if epoch.run is None:
                key = (ctx.state['subject'], ctx.state['session'], epoch.task, ctx.state['acquisition'])
                if key in self._runs_for:
                    return self._runs_for[key]
            return ()
        if isinstance(epoch, SecondaryEpoch):
            return self._find_runs(ctx, self.epochs[epoch.sel_epoch])
        return ()

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        runs = self._find_runs(ctx, epoch)
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r}; can't load events for epoch collection")
        elif isinstance(epoch, (PrimaryEpoch, SecondaryEpoch)) and runs:
            # Combine-all: per-run selected-events; index applied after combining
            rec_options = ctx.options_for('selected-events', 'reject', *EPOCH_EXTRACT_OPTIONS)
            return tuple(
                Dependency('selected-events', label=f'selected-events-{run}',
                           state={'task': epoch.task, 'run': run}, options=rec_options)
                for run in runs
            )
        elif isinstance(epoch, (PrimaryEpoch, SecondaryEpoch, ContinuousEpoch)):
            return (Dependency('selected-events', state={'task': epoch.task, 'run': single_recording_run(self.epochs, epoch)},
                               options=ctx.options_for('selected-events', 'reject', *EPOCH_EXTRACT_OPTIONS)),)
        else:
            options = ctx.options_for('epoch-events', 'reject', *EPOCH_EXTRACT_OPTIONS)
            if isinstance(epoch, SuperEpoch):
                return tuple(
                    Dependency('epoch-events', label=f'{sub_epoch}:events', options=options,
                               state={'epoch': sub_epoch, 'task': self.epochs[sub_epoch].task})
                    for sub_epoch in epoch.sub_epochs
                )
            else:
                raise RuntimeError(f"{epoch=}")

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'epoch': self.epochs[ctx.state['epoch']]}

    def build(self, ctx: Request) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, (PrimaryEpoch, SecondaryEpoch, ContinuousEpoch)):
            runs = self._find_runs(ctx, epoch)
            if runs:
                dss = []
                for run in runs:
                    ds = ctx.load(f'selected-events-{run}')
                    ds[:, 'run'] = run
                    dss.append(ds)
                ds = combine(dss)
                ds.info[BAD_CHANNELS] = sorted({ch for d in dss for ch in d.info.get(BAD_CHANNELS, [])})
                ds.info[INTERPOLATE_CHANNELS] = any(d.info.get(INTERPOLATE_CHANNELS, False) for d in dss)
                ds.info[INTERPOLATE_WINDOWS] = any(d.info.get(INTERPOLATE_WINDOWS, False) for d in dss)
                windows_max = {d.info.get(INTERPOLATE_WINDOWS_MAX) for d in dss} - {None}
                if windows_max:
                    assert len(windows_max) == 1
                    ds.info[INTERPOLATE_WINDOWS_MAX] = windows_max.pop()
                if epoch.n_cases is not None and ds.n_cases != epoch.n_cases:
                    raise RuntimeError(f"Number of epochs {ds.n_cases}, expected {epoch.n_cases}")
                return ds
            return ctx.load('selected-events')
        elif isinstance(epoch, SuperEpoch):
            dss = []
            bad_channels = set()
            for sub_epoch in epoch.sub_epochs:
                ds = ctx.load(f'{sub_epoch}:events')
                ds[:, 'epoch'] = sub_epoch
                dss.append(ds)
                bad_channels.update(ds.info[BAD_CHANNELS])
            ds = combine(dss)
            ds.info[BAD_CHANNELS] = sorted(bad_channels)
        else:
            raise RuntimeError(f"{epoch=}")
        return epoch._prepare_selected_events(ds, ctx.state['subject'], ctx.options)
