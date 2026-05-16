# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Event and selected-event derivatives.

The event pipeline is split into three nodes:

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
    Applies epoch-specific trial selection (``sel`` predicate, rejection,
    bad-channel interpolation) on top of the labeled events.
    Cache policy is :attr:`~CachePolicy.DISABLED_BY_DEFAULT`
    because the result is a small dataset that is cheap to recompute and is
    usually only needed as an intermediate for :class:`~epochs.EpochsDerivative`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import load, save
from .._data_obj import Datalist, Dataset, Factor, combine
from .._exceptions import ConfigurationError
from .._info import BAD_CHANNELS, INTERPOLATE_CHANNELS
from .derivative_cache import CachePolicy, Dependency, Derivative, Input, Request, UncachedDerivative, file_fingerprint
from .epochs import EPOCH_EXTRACT_OPTIONS, EpochCollection, SecondaryEpoch, SuperEpoch, PrimaryEpoch, ContinuousEpoch
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

    The recording sampling frequency is read from the accompanying
    ``*_<datatype>.json`` metadata sidecar (``SamplingFrequency`` field).
    """
    name = 'events-input'

    def __init__(
            self,
            raw_extension: str,
    ):
        self.raw_extension = raw_extension

    def path(self, ctx: Request) -> Path:
        return bids_path(ctx.root, ctx.state, extension='.tsv', suffix='events').fpath

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return file_fingerprint(ctx.root, self.path(ctx), 'events-tsv')

    def load(self, ctx: Request) -> Dataset:
        path = self.path(ctx)
        df = pd.read_csv(path, sep='\t')
        # Read SamplingFrequency from the accompanying JSON metadata sidecar
        json_path = Path(bids_path(ctx.root, ctx.state, '.json').fpath)
        if json_path.exists():
            with open(json_path) as f:
                sfreq = float(json.load(f).get('SamplingFrequency', 0.))
        else:
            sfreq = 0.
        entities = {k: ctx.state[k] for k in BIDS_ENTITY_KEYS}
        ds = Dataset.from_dataframe(df)
        ds.info['raw.samplingrate'] = sfreq
        ds.info.update(entities)
        return ds


def _check_ds(ds: Dataset, source: str, info: dict[str, Any]) -> Dataset:
    if not isinstance(ds, Dataset):
        raise ConfigurationError(f"{source} needs to return the events Dataset. Got {ds!r}.")
    if 'sample' not in ds:
        raise ConfigurationError(f"The Dataset returned by {source} does not contain a variable called `sample`. This variable is required to ascribe events to data samples.")
    if 'value' not in ds:
        raise ConfigurationError(f"The Dataset returned by {source} does not contain a variable called `value`. This variable is required to check rejection files.")
    if ds.info is not info:
        ds.info.update(info)
    return ds


class EventsDerivative(Derivative[Dataset]):
    """Extract events form M/EEG data files"""
    name = 'events'
    key_fields = ('subject', 'session', 'task', 'run', 'raw')
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
            'raw': ctx.state['raw'],
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
        ds = load.mne.events(raw, self.merge_triggers, stim_channel=self.stim_channel)
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

        ds = _check_ds(self.fix_events_impl(self, ds), f'{self.owner_name}.fix_events()', ds.info)
        ds['onset'] = ds['sample'] / ds.info['raw.samplingrate']
        return ds

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
    key_fields = ('subject', 'session', 'task', 'run', 'raw')
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
            self.cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def _has_sidecar(self, ctx: Request) -> bool:
        """Return whether a BIDS events sidecar exists for the current state."""
        return ctx.registry.resolve('events-input', state=ctx.state).exists()

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if self._has_sidecar(ctx):
            return (
                Dependency('events-input', label='events'),
                Dependency(raw_node_name(ctx.state['raw']), label='raw'),
            )
        return (Dependency('events'),)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(
            ctx,
            definitions={
                'variables': self._variables,
                'label_events': function_fingerprint(self.label_events_impl),
            },
        )

    def build(self, ctx: Request) -> Dataset:
        ds = ctx.load('events')
        if self._has_sidecar(ctx):
            raw = ctx.load('raw')
            ds.info['raw.samplingrate'] = raw.info['sfreq']
            ds.info['raw.first_samp'] = raw.first_samp
            ds.info['raw.last_samp'] = raw.last_samp
            # BIDS TSV 'sample' is 0-indexed from file start (MNE-BIDS subtracts
            # raw.first_samp on write); adjust to raw's absolute sample frame.
            if raw.first_samp:
                ds['sample'] = ds['sample'] + raw.first_samp
        ds['subject'] = Factor([ctx.state['subject']], repeat=ds.n_cases, random=True)
        if self.multi_task:
            ds[:, 'task'] = ctx.state['task']
        if self.multi_session:
            ds[:, 'session'] = ctx.state['session']
        self._variables._apply(ds, self._groups)
        info = ds.info
        return _check_ds(self.label_events_impl(self, ds), f'{self.owner_name}.label_events()', info)

    def load(self, ctx: Request, path: Path) -> Dataset:
        ds = load.unpickle(path)
        ds.info.update({k: ctx.state[k] for k in BIDS_ENTITY_KEYS})
        return ds

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class SelectedEventsDerivative(UncachedDerivative[Dataset]):
    """Selected event dataset for one epoch/rejection state.

    Options
    -------
    reject
        Whether to apply artifact rejection (`True`, `False`, or `'keep'`).
    index
        Add an index column the returned dataset, and which column name to use.
        Indexing occurs after trial selection and before artifact rejection.
    cat
        Optional subset of model cells to keep.
    """
    name = 'selected-events'
    # key_fields = ('subject', 'session', 'task', 'run', 'raw', 'epoch', 'rej')
    # cache_suffix = '.pickle'
    # cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    OPTION_DEFAULTS = {
        'index': True,
        'reject': True,
        'baseline': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'tmin': None,
        'tmax': None,
        'tstop': None,
    }
    VIEW_OPTION_DEFAULTS = {'cat': None}

    def __init__(
            self,
            raw,
            epochs: dict[str, Any],
            artifact_rejection: dict[str, dict[str, Any]],
    ):
        self.raw = raw
        self.epochs = epochs
        self.artifact_rejection = artifact_rejection

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r}; can't load events for collection epoch")
        elif isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            reject = ctx.options['reject']
            if reject not in (True, False, 'keep'):
                raise ValueError(f"{reject=}")
            state = {'task': epoch.task}
            deps = [Dependency('labeled-events', state=state)]
            rejection_params = self.artifact_rejection[ctx.state['rej']]
            if rejection_params['kind'] and reject:
                deps.append(Dependency('rej-input', state=state))
            return tuple(deps)
        else:
            options = ctx.options_for('selected-events', 'reject', *EPOCH_EXTRACT_OPTIONS)
            if isinstance(epoch, SecondaryEpoch):
                state = {'epoch': epoch.sel_epoch, 'task': self.epochs[epoch.sel_epoch].task}
                return (Dependency('selected-events', options=options, state=state),)
            elif isinstance(epoch, SuperEpoch):
                return tuple(Dependency('selected-events', label=f'{sub_epoch}:events', options=options, state={'epoch': sub_epoch, 'task': self.epochs[sub_epoch].task}) for sub_epoch in epoch.sub_epochs)
            else:
                raise RuntimeError(f"{epoch=}")

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        epoch = self.epochs[ctx.state['epoch']]
        return self.standard_fingerprint(ctx, definitions={'epoch': epoch})

    def build(self, ctx: Request) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        subject = ctx.state['subject']
        if isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            ds = ctx.load('labeled-events')
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if epoch.n_cases is not None and ds.n_cases != epoch.n_cases:
                raise RuntimeError(f"Number of epochs {ds.n_cases}, expected {epoch.n_cases}")
            if ctx.options['index']:
                ds.index(ctx.options['index'])

            # Trial rejection
            reject = ctx.options['reject']
            rejection_params = self.artifact_rejection[ctx.state['rej']]
            if rejection_params['kind'] and reject:
                rejection_ds = ctx.load('rej-input')

                # Handle event mismatches
                if rejection_ds.info.get('epochs.selection') is not None:
                    ds = ds[rejection_ds.info['epochs.selection']]
                if rejection_ds.n_cases != ds.n_cases or np.any(ds['value'] != rejection_ds['value']):
                    raise RuntimeError(f"The epoch selection file contains different events from the data loaded from the raw file. If the events included in the epoch were changed intentionally, redo epoch selection for {subject}/{epoch.name}")

                # Channel interpolation
                if rejection_params['interpolation']:
                    ds.info[INTERPOLATE_CHANNELS] = True
                    if INTERPOLATE_CHANNELS in rejection_ds:
                        ds[INTERPOLATE_CHANNELS] = rejection_ds[INTERPOLATE_CHANNELS]
                    else:
                        ds[INTERPOLATE_CHANNELS] = Datalist([[]] * ds.n_cases, INTERPOLATE_CHANNELS, 'strlist')
                else:
                    ds.info[INTERPOLATE_CHANNELS] = False

                if reject == 'keep':
                    ds['accept'] = rejection_ds['accept']
                elif reject is True:
                    ds = ds.sub(rejection_ds['accept'])
                elif reject is not False:
                    raise RuntimeError(f"{reject=}")

                ds.info[BAD_CHANNELS] = rejection_ds.info.get(BAD_CHANNELS, [])
            else:
                ds.info[INTERPOLATE_CHANNELS] = False
                ds.info[BAD_CHANNELS] = []
        elif isinstance(epoch, SecondaryEpoch):
            ds = ctx.load('selected-events')
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if ctx.options['index']:
                ds.index(ctx.options['index'])
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

    def apply_view_options(self, ctx: Request, ds: Dataset) -> Dataset:
        if ctx.view_options['cat']:
            model = ds.eval(ctx.state['model'])
            ds = ds.sub(model.isin(ctx.view_options['cat']))
        return ds
