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
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .. import load, save
from .._data_obj import Datalist, Dataset, Factor, Var, combine
from .._exceptions import ConfigurationError
from .._info import BAD_CHANNELS
from .._names import INTERPOLATE_CHANNELS
from .derivative_cache import CachePolicy, Dependency, Derivative, Input, Request, file_fingerprint
from .epochs import EpochCollection, SecondaryEpoch, SuperEpoch
from .exceptions import FileMissingError
from .pathing import BIDS_ENTITY_KEYS, bids_path, rej_file_path
from .preprocessing import load_raw_dependency, raw_data_dependency, raw_node_name
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
        ds.info['sfreq'] = sfreq
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
        return (raw_data_dependency(ctx, add_bads=False),)

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
        ds.info['sfreq'] = raw.info['sfreq']
        ds.info.update(entities)

        trigger_shift = self._get_trigger_shift(subject, session)
        if trigger_shift:
            ds['sample'] += int(round(trigger_shift * ds.info['sfreq']))

        ds = _check_ds(self.fix_events_impl(self, ds), f'{self.owner_name}.fix_events()', ds.info)
        ds['onset'] = ds['sample'] / ds.info['sfreq']
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
            self.cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def _has_sidecar(self, ctx: Request) -> bool:
        """Return whether a BIDS events sidecar exists for the current state."""
        return ctx.registry.resolve('events-input', state=ctx.state).exists()

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if self._has_sidecar(ctx):
            return (Dependency('events-input', label='events'),)
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


class SelectedEventsDerivative(Derivative[Dataset]):
    """Selected event dataset for one epoch/rejection state.

    Options
    -------
    reject
        Whether to apply artifact rejection (`True`, `False`, or `'keep'`).
    add_bads
        Whether to load current bad channels into attached raw objects.
    index
        Whether to index the returned dataset, and which index name to use.
    data_raw
        Whether to keep the raw object in ``ds.info['raw']``.
    cat
        Optional subset of model cells to keep.
    """
    name = 'selected-events'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'raw', 'epoch', 'rej')
    cache_suffix = '.pickle'
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    OPTION_DEFAULTS = {'reject': True}
    VIEW_OPTION_DEFAULTS = {'add_bads': True, 'index': True, 'data_raw': False, 'cat': None}

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
        tasks = epoch.tasks if hasattr(epoch, 'tasks') else (epoch.task,)
        deps = [Dependency('rej-input', label='rej')]
        for task in tasks:
            task_state = {'task': task}
            deps.append(Dependency('labeled-events', label=f'{task}:labeled-events', state=task_state))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        epoch = self.epochs[ctx.state['epoch']]
        return self.standard_fingerprint(
            ctx,
            definitions={'epoch': epoch._as_dict()},
        )

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view is None:
            return self.fingerprint(ctx)
        if view != 'epochs':
            raise ValueError(f"{self.name!r} does not define dependency view {view!r}")
        ds = ctx.load()
        return {'sample': ctx.registry.canonicalize(ds['sample'])}

    def _build_selected_events(
            self,
            ctx: Request,
            epoch,
            reject: bool | str,
            add_bads: bool | list[str],
            index: str | bool,
            data_raw: bool,
            cat,
    ) -> Dataset:
        subject = ctx.state['subject']
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r}; can't load events for collection epoch")

        if isinstance(epoch, SuperEpoch):
            dss = []
            raw = None
            if isinstance(add_bads, Sequence):
                bad_channels = list(add_bads)
            else:
                bad_channels = []
            for task in epoch.tasks:
                task_dss = []
                for sub_epoch in epoch.sub_epochs:
                    if self.epochs[sub_epoch].task != task:
                        continue
                    ds = self._build_selected_events(ctx, self.epochs[sub_epoch], reject, add_bads, index, data_raw, None)
                    ds[:, 'epoch'] = sub_epoch
                    task_dss.append(ds)
                if add_bads is True:
                    for task_ds in task_dss:
                        task_bads = task_ds.info[BAD_CHANNELS]
                        if not task_bads and data_raw:
                            task_bads = task_ds.info['raw'].info['bads']
                        bad_channels.extend(task_bads)
                ds = combine(task_dss)
                dss.append(ds)
                if data_raw:
                    raw_ = task_dss[0].info['raw']
                    if raw is None:
                        raw = raw_
                    else:
                        ds['sample'] += raw.last_samp + 1 - raw_.first_samp
                        raw.append(raw_)
            if add_bads is True:
                bad_channels = sorted(set(bad_channels))
            ds = combine(dss)
            if data_raw:
                raw.info['bads'] = bad_channels
                ds.info['raw'] = raw
            ds.info[BAD_CHANNELS] = bad_channels
        elif isinstance(epoch, SecondaryEpoch):
            ds = self._build_selected_events(ctx, self.epochs[epoch.sel_epoch], 'keep' if reject else False, add_bads, index, data_raw, None)
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if index:
                ds.index(index)
            if reject is True and self.artifact_rejection[ctx.state['rej']]['kind'] is not None:
                ds = ds.sub('accept')
        else:
            rej_params = self.artifact_rejection[ctx.state['rej']]
            selection_state = {**ctx.state, 'epoch': epoch.name, 'task': epoch.task}
            if reject and rej_params['kind'] is not None:
                rej_file = ctx.root / rej_file_path(selection_state)
                if rej_file.exists():
                    ds_sel = load.unpickle(rej_file)
                else:
                    raise FileMissingError(f"The rejection file at {rej_file.relative_to(ctx.root)} does not exist. Run .make_epoch_selection() first.")
            else:
                ds_sel = None
            ds = ctx.load(f'{epoch.task}:labeled-events')
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if index:
                ds.index(index)
            if epoch.n_cases is not None and ds.n_cases != epoch.n_cases:
                raise RuntimeError(f"Number of epochs {ds.n_cases}, expected {epoch.n_cases}")
            if ds_sel is not None:
                test_passed = False
                if ds_sel.info.get('epochs.selection') is not None:
                    ds = ds[ds_sel.info['epochs.selection']]
                # Support both new ('value') and old ('trigger') rejection files
                sel_col = 'value' if 'value' in ds_sel else 'trigger'
                if ds_sel.n_cases != ds.n_cases:
                    if np.all(ds[:ds_sel.n_cases, 'value'] == ds_sel[sel_col]):
                        ds = ds[:ds_sel.n_cases]
                        test_passed = True
                    elif np.all(ds[-ds_sel.n_cases:, 'value'] == ds_sel[sel_col]):
                        ds = ds[-ds_sel.n_cases:]
                        test_passed = True
                elif np.all(ds['value'] == ds_sel[sel_col]):
                    test_passed = True
                if not test_passed:
                    raise RuntimeError(f"The epoch selection file contains different events (trigger IDs) from the data loaded from the raw file. If the events included in the epoch were changed intentionally, redo epoch selection for {subject}/{epoch.name}")

                if rej_params['interpolation']:
                    ds.info[INTERPOLATE_CHANNELS] = True
                    if INTERPOLATE_CHANNELS in ds_sel:
                        ds[INTERPOLATE_CHANNELS] = ds_sel[INTERPOLATE_CHANNELS]
                    else:
                        ds[INTERPOLATE_CHANNELS] = Datalist([[]] * ds.n_cases, INTERPOLATE_CHANNELS, 'strlist')
                else:
                    ds.info[INTERPOLATE_CHANNELS] = False

                if reject == 'keep':
                    ds['accept'] = ds_sel['accept']
                elif reject is True:
                    ds = ds.sub(ds_sel['accept'])
                elif reject is not False:
                    raise RuntimeError(f"{reject=}")

                if add_bads:
                    ds.info[BAD_CHANNELS] = ds_sel.info.get(BAD_CHANNELS, [])
                else:
                    ds.info[BAD_CHANNELS] = []
            else:
                ds.info[INTERPOLATE_CHANNELS] = False
                ds.info[BAD_CHANNELS] = []

        if epoch.trigger_shift:
            shift = epoch.trigger_shift
            if isinstance(shift, str):
                shift = ds.eval(shift)
            if isinstance(shift, Var):
                shift = shift.x
                if np.isnan(shift).any():
                    raise RuntimeError(f"The epoch shift contains NaNs for {subject}/{epoch.name}\n{shift=}")
            if np.isscalar(shift):
                ds['sample'] += int(round(shift * ds.info['sfreq']))
            else:
                ds['sample'] += np.round(shift * ds.info['sfreq']).astype(int)

        if cat:
            model = ds.eval(ctx.state['model'])
            ds = ds.sub(model.isin(cat))
        if not data_raw and 'raw' in ds.info:
            del ds.info['raw']
        return ds

    def _view_raw(
            self,
            ctx: Request,
            ds: Dataset,
            epoch,
            add_bads: bool | list[str],
            data_raw: bool,
    ) -> Dataset:
        tasks = list(ds['task'].cells) if 'task' in ds else []
        if not tasks:
            if hasattr(epoch, 'task'):
                tasks = [epoch.task]
            elif isinstance(epoch, SecondaryEpoch):
                sel_epoch = self.epochs[epoch.sel_epoch]
                tasks = list(getattr(sel_epoch, 'tasks', (sel_epoch.task,)))
            else:
                tasks = list(getattr(epoch, 'tasks', ()))
        if not tasks:
            return ds

        ds = ds.copy()
        raw_all = None
        bad_channels = list(add_bads) if isinstance(add_bads, Sequence) else None
        for i, task in enumerate(tasks):
            raw = load_raw_dependency(ctx, add_bads=add_bads, preload=data_raw, noise=False, state={'task': task})
            if bad_channels is None:
                task_bads = raw.info['bads'] if add_bads else []
            else:
                task_bads = bad_channels
            if data_raw:
                if raw_all is None:
                    raw_all = raw
                else:
                    offset = raw_all.last_samp + 1 - raw.first_samp
                    if 'task' in ds:
                        ds['sample'].x[ds['task'] == task] += offset
                    else:
                        ds['sample'] += offset
                    raw_all.append(raw)
            if bad_channels is None:
                bad_channels = task_bads if i == 0 else sorted(set(bad_channels).union(task_bads))
        ds.info[BAD_CHANNELS] = bad_channels or []
        if data_raw and raw_all is not None:
            raw_all.info['bads'] = ds.info[BAD_CHANNELS]
            ds.info['raw'] = raw_all
        else:
            ds.info.pop('raw', None)
        return ds

    def build(self, ctx: Request) -> Dataset:
        reject = ctx.options['reject']
        if reject not in (True, False, 'keep'):
            raise ValueError(f"{reject=}")
        return self._build_selected_events(
            ctx,
            self.epochs[ctx.state['epoch']],
            reject,
            False,
            False,
            False,
            None,
        )

    def apply_view_options(self, ctx: Request, ds: Dataset) -> Dataset:
        epoch = self.epochs[ctx.state['epoch']]
        add_bads = ctx.view_options['add_bads']
        data_raw = ctx.view_options['data_raw']
        if add_bads or data_raw:
            ds = self._view_raw(ctx, ds, epoch, add_bads, data_raw)
        else:
            ds = ds.copy()
            ds.info[BAD_CHANNELS] = []
            ds.info.pop('raw', None)
        cat = ctx.view_options['cat']
        if cat:
            model = ds.eval(ctx.state['model'])
            ds = ds.sub(model.isin(cat))
        index = ctx.view_options['index']
        if index is True:
            index = 'index'
        elif index and not isinstance(index, str):
            raise TypeError(f"{index=}")
        if index:
            ds.index(index)
        return ds

    def load(self, ctx: Request, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)
