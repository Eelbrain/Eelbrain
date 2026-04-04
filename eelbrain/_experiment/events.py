# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Event and selected-event derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np

from .. import load, save
from .._data_obj import Datalist, Dataset, Factor, Var, combine
from .._exceptions import ConfigurationError
from .._info import BAD_CHANNELS
from .._names import INTERPOLATE_CHANNELS
from .derivative_cache import CachePolicy, Dependency, Derivative, Request
from .epochs import EpochCollection, SecondaryEpoch, SuperEpoch
from .exceptions import FileMissingError
from .pathing import rej_file_path
from .preprocessing import load_raw_dependency, raw_data_dependency
from .variable_def import Variables


BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')
SELECTED_EVENTS = 'selected-events'


def _check_ds(ds: Dataset, source: str, info: dict[str, Any]) -> Dataset:
    if not isinstance(ds, Dataset):
        raise ConfigurationError(f"{source} needs to return the events Dataset. Got {ds!r}.")
    if 'i_start' not in ds:
        raise ConfigurationError(f"The Dataset returned by {source} does not contain a variable called `i_start`. This variable is required to ascribe events to data samples.")
    if 'trigger' not in ds:
        raise ConfigurationError(f"The Dataset returned by {source} does not contain a variable called `trigger`. This variable is required to check rejection files.")
    if ds.info is not info:
        ds.info.update(info)
    return ds


class EventsDerivative(Derivative[Dataset]):
    name = 'events'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw')
    cache_suffix = '.pickle'

    def __init__(
            self,
            trigger_shift: int | dict[str, int] | dict[tuple[str, str], int],
            stim_channel: str | list[str],
            merge_triggers: Any,
            variables: Variables,
            groups: dict[str, Any],
            preload: bool,
            fix_events,
            label_events,
            owner_name: str,
            multi_task: bool,
            multi_session: bool,
    ):
        self.trigger_shift = trigger_shift
        self.stim_channel = stim_channel
        self.merge_triggers = merge_triggers
        self._variables = variables
        self._groups = groups
        self.preload = preload
        self.fix_events_impl = fix_events
        self.label_events_impl = label_events
        self.owner_name = owner_name
        self.multi_task = multi_task
        self.multi_session = multi_session
        self.variables_repr = repr(variables)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (raw_data_dependency(ctx, add_bads=False),)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        subject = ctx.state['subject']
        session = ctx.state['session']
        trigger_shift = self.trigger_shift
        if isinstance(trigger_shift, dict):
            trigger_shift = trigger_shift.get((subject, session), trigger_shift.get(subject, 0))
        return {
            'raw': ctx.state['raw'],
            'stim_channel': self.stim_channel,
            'merge_triggers': self.merge_triggers,
            'trigger_shift': trigger_shift,
            'variables': self.variables_repr,
            'fix_events': getattr(self.fix_events_impl, '__qualname__', repr(self.fix_events_impl)),
            'label_events': getattr(self.label_events_impl, '__qualname__', repr(self.label_events_impl)),
        }

    def build(self, ctx: Request) -> Dataset:
        entities = {k: ctx.state[k] for k in BIDS_ENTITY_KEYS}
        subject = entities['subject']
        session = entities['session']
        raw = load_raw_dependency(ctx, add_bads=False, preload=self.preload, noise=False)
        ds = load.mne.events(raw, self.merge_triggers, stim_channel=self.stim_channel)
        del ds.info['raw']
        ds.info['sfreq'] = raw.info['sfreq']
        ds.info.update(entities)

        info = ds.info
        ds = _check_ds(self.fix_events_impl(self, ds), f'{self.owner_name}.fix_events()', info)
        ds['time'] = ds['i_start'] / ds.info['sfreq']
        ds['SOA'] = ds['time'].diff(0)
        ds['subject'] = Factor([subject], repeat=ds.n_cases, random=True)
        if self.multi_task:
            ds[:, 'task'] = entities['task']
        if self.multi_session:
            ds[:, 'session'] = entities['session']
        self._variables.apply(ds, self._groups)
        info = ds.info
        ds = _check_ds(self.label_events_impl(self, ds), f'{self.owner_name}.label_events()', info)

        trigger_shift = self.trigger_shift
        if isinstance(trigger_shift, dict):
            trigger_shift = trigger_shift.get((subject, session), trigger_shift.get(subject, 0))
        if trigger_shift:
            ds['i_start'] += int(round(trigger_shift * ds.info['sfreq']))
        return ds

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
    name = SELECTED_EVENTS
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw', 'epoch', 'rej')
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
            deps.append(Dependency('events', label=f'{task}:events', state=task_state))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        epoch = self.epochs[ctx.state['epoch']]
        return self.standard_fingerprint(
            ctx,
            definitions={'epoch': epoch._as_dict()},
            extra={'rej': ctx.state['rej']},
        )

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view is None:
            return self.fingerprint(ctx)
        if view != 'epochs':
            raise ValueError(f"{self.name!r} does not define dependency view {view!r}")
        ds = self.build(ctx)
        return {'i_start': ds['i_start'].x.tolist()}

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
                        ds['i_start'] += raw.last_samp + 1 - raw_.first_samp
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
                rej_file = rej_file_path(selection_state)
                if rej_file.exists():
                    ds_sel = load.unpickle(rej_file)
                else:
                    raise FileMissingError(f"The rejection file at {rej_file.relative_to(Path(ctx.state['root']))} does not exist. Run .make_epoch_selection() first.")
            else:
                ds_sel = None
            state = {**ctx.state, 'task': epoch.task}
            ds = ctx.load('events', state=state)
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
                if ds_sel.n_cases != ds.n_cases:
                    if np.all(ds[:ds_sel.n_cases, 'trigger'] == ds_sel['trigger']):
                        ds = ds[:ds_sel.n_cases]
                        test_passed = True
                    elif np.all(ds[-ds_sel.n_cases:, 'trigger'] == ds_sel['trigger']):
                        ds = ds[-ds_sel.n_cases:]
                        test_passed = True
                elif np.all(ds['trigger'] == ds_sel['trigger']):
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
                ds['i_start'] += int(round(shift * ds.info['sfreq']))
            else:
                ds['i_start'] += np.round(shift * ds.info['sfreq']).astype(int)

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
                        ds['i_start'].x[ds['task'] == task] += offset
                    else:
                        ds['i_start'] += offset
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
