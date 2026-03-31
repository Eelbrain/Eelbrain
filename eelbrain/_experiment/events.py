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
from .configuration import sequence_arg
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext, file_fingerprint
from .epochs import EpochCollection, SecondaryEpoch, SuperEpoch
from .exceptions import FileMissingError
from .pathing import rej_file_path
from .preprocessing import load_raw_dependency, raw_bad_channels_input_name, raw_data_dependency, raw_node_name
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


def _coerce_vardef(vardef: None | str | Variables, tests: dict[str, Any]) -> Variables | None:
    if vardef is None:
        return None
    if isinstance(vardef, str):
        try:
            vardef = tests[vardef].vars
        except KeyError:
            raise ValueError(f"{vardef=}") from None
    elif not isinstance(vardef, Variables):
        vardef = Variables(vardef)
    return vardef


def _apply_vardef(ds: Dataset, vardef: None | str | Variables, tests: dict[str, Any], groups: dict[str, Any]) -> None:
    vardef = _coerce_vardef(vardef, tests)
    if vardef is not None:
        vardef.apply(ds, groups)


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

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (raw_data_dependency(ctx, add_bads=False),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        subject = ctx.get('subject')
        session = ctx.get('session')
        trigger_shift = self.trigger_shift
        if isinstance(trigger_shift, dict):
            trigger_shift = trigger_shift.get((subject, session), trigger_shift.get(subject, 0))
        return {
            'raw': ctx.get('raw'),
            'stim_channel': self.stim_channel,
            'merge_triggers': self.merge_triggers,
            'trigger_shift': trigger_shift,
            'variables': self.variables_repr,
            'fix_events': getattr(self.fix_events_impl, '__qualname__', repr(self.fix_events_impl)),
            'label_events': getattr(self.label_events_impl, '__qualname__', repr(self.label_events_impl)),
        }

    def build(self, ctx: DerivativeContext) -> Dataset:
        entities = {k: ctx.get(k) for k in BIDS_ENTITY_KEYS}
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

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        ds = load.unpickle(path)
        ds.info.update({k: ctx.get(k) for k in BIDS_ENTITY_KEYS})
        return ds

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class SelectedEventsDerivative(Derivative[Dataset]):
    name = SELECTED_EVENTS
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw', 'epoch', 'rej')
    cache_suffix = '.pickle'
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def __init__(
            self,
            raw,
            epochs: dict[str, Any],
            tests: dict[str, Any],
            artifact_rejection: dict[str, dict[str, Any]],
            groups: dict[str, Any],
    ):
        self.raw = raw
        self.epochs = epochs
        self.tests = tests
        self.artifact_rejection = artifact_rejection
        self.groups = groups

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        epoch = self.epochs[ctx.get('epoch')]
        state = dict(ctx.state)
        task_dependencies = []
        tasks = epoch.tasks if hasattr(epoch, 'tasks') else (epoch.task,)
        for task in tasks:
            task_state = {**state, 'task': task}
            task_dependencies.append({
                'task': task,
                'events': ctx.registry.resolve('events', state=task_state).describe_dependency(),
                'raw': ctx.registry.resolve(
                    raw_node_name(task_state['raw']),
                    state={**task_state, 'raw': task_state['raw']},
                    options={'add_bads': ctx.option('add_bads', True), 'preload': True, 'noise': False},
                ).describe_dependency(),
            })
        rej_files = None
        if self.artifact_rejection[ctx.get('rej')]['kind'] is not None:
            rej_files = [
                file_fingerprint(
                    ctx.get('root'),
                    rej_file_path({**state, 'epoch': epoch_name, 'task': self.epochs[epoch_name].task}),
                    'rej-file',
                    metadata={'epoch': epoch_name},
                )
                for epoch_name in epoch.rej_file_epochs
            ]
        return {
            'epoch': epoch._as_dict(),
            'rej': ctx.get('rej'),
            'vardef': repr(_coerce_vardef(ctx.option('vardef'), self.tests)),
            'options': ctx.registry.canonicalize({
                'reject': ctx.option('reject', True),
                'add_bads': ctx.option('add_bads', True),
                'index': ctx.option('index', True),
                'data_raw': ctx.option('data_raw', False),
                'cat': sequence_arg('cat', ctx.option('cat')) if ctx.option('cat') else None,
            }),
            'dependencies': task_dependencies,
            'rej-files': rej_files,
        }

    def dependency_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        epoch = self.epochs[ctx.get('epoch')]
        state = dict(ctx.state)
        tasks = epoch.tasks if hasattr(epoch, 'tasks') else (epoch.task,)
        raw_dependencies = []
        for task in tasks:
            task_state = {**state, 'task': task}
            raw_dependencies.append({
                'task': task,
                'raw': ctx.registry.resolve(
                    raw_node_name(task_state['raw']),
                    state=task_state,
                    options={'add_bads': ctx.option('add_bads', True), 'preload': True, 'noise': False},
                ).describe_dependency(),
            })

        ds = self.build(ctx)
        out = {
            'n_cases': ds.n_cases,
            'i_start': ds['i_start'].x.tolist(),
            'trigger': ds['trigger'].x.tolist(),
            'sfreq': ds.info['sfreq'],
            'raw': raw_dependencies,
        }
        if ctx.option('interpolate_bads', False):
            out['interpolate_channels'] = bool(ds.info.get(INTERPOLATE_CHANNELS, False))
            if out['interpolate_channels'] and INTERPOLATE_CHANNELS in ds:
                out[INTERPOLATE_CHANNELS] = [list(channels) for channels in ds[INTERPOLATE_CHANNELS]]

        if isinstance(epoch.trigger_shift, str):
            shift = ds.eval(epoch.trigger_shift)
            out['trigger_shift'] = shift.x.tolist() if isinstance(shift, Var) else shift

        tmax = ctx.option('tmax')
        if tmax is None and ctx.option('tstop') is None:
            tmax = epoch.tmax
        if isinstance(tmax, str):
            tmax_value = ds.eval(tmax)
            out['tmax'] = tmax_value.x.tolist() if isinstance(tmax_value, Var) else tmax_value

        if ctx.option('trigger_shift', True) and epoch.post_baseline_trigger_shift:
            shift_values = ds[epoch.post_baseline_trigger_shift]
            out['post_baseline_trigger_shift'] = shift_values.x.tolist() if isinstance(shift_values, Var) else list(shift_values)

        return out

    def _load_events(self, ctx: DerivativeContext, task: str, add_bads: bool | list[str], data_raw: bool) -> Dataset:
        state = {**ctx.state, 'task': task}
        ds = ctx.load('events', state=state)
        raw = load_raw_dependency(ctx, add_bads=add_bads, preload=True, noise=False, state=state)
        ds.info['raw'] = raw
        if not data_raw and 'raw' in ds.info:
            del ds.info['raw']
        return ds

    def _build_selected_events(
            self,
            ctx: DerivativeContext,
            epoch,
            reject: bool | str,
            add_bads: bool | list[str],
            index: str | bool,
            data_raw: bool,
            vardef: str | Variables | None,
            cat,
    ) -> Dataset:
        subject = ctx.get('subject')
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r}; can't load events for collection epoch")

        if isinstance(epoch, SuperEpoch):
            dss = []
            raw = None
            if isinstance(add_bads, Sequence):
                bad_channels = list(add_bads)
            elif add_bads:
                bad_channels = sorted(set().union(*(
                    set(ctx.load(raw_bad_channels_input_name(ctx.get('raw')), state={**ctx.state, 'task': task}, options={'noise': False}))
                    for task in epoch.tasks
                )))
            else:
                bad_channels = []
            for task in epoch.tasks:
                task_dss = []
                for sub_epoch in epoch.sub_epochs:
                    if self.epochs[sub_epoch].task != task:
                        continue
                    ds = self._build_selected_events(ctx, self.epochs[sub_epoch], reject, add_bads, index, data_raw, None, None)
                    ds[:, 'epoch'] = sub_epoch
                    task_dss.append(ds)
                ds = combine(task_dss)
                dss.append(ds)
                if data_raw:
                    raw_ = task_dss[0].info['raw']
                    raw_.info['bads'] = bad_channels
                    if raw is None:
                        raw = raw_
                    else:
                        ds['i_start'] += raw.last_samp + 1 - raw_.first_samp
                        raw.append(raw_)
            ds = combine(dss)
            if data_raw:
                ds.info['raw'] = raw
            ds.info[BAD_CHANNELS] = bad_channels
        elif isinstance(epoch, SecondaryEpoch):
            ds = self._build_selected_events(ctx, self.epochs[epoch.sel_epoch], 'keep' if reject else False, add_bads, index, data_raw, None, None)
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if index:
                ds.index(index)
            if reject is True and self.artifact_rejection[ctx.get('rej')]['kind'] is not None:
                ds = ds.sub('accept')
        else:
            rej_params = self.artifact_rejection[ctx.get('rej')]
            selection_state = {**ctx.state, 'epoch': epoch.name, 'task': epoch.task}
            if reject and rej_params['kind'] is not None:
                rej_file = rej_file_path(selection_state)
                if rej_file.exists():
                    ds_sel = load.unpickle(rej_file)
                else:
                    raise FileMissingError(f"The rejection file at {rej_file.relative_to(Path(ctx.get('root')))} does not exist. Run .make_epoch_selection() first.")
            else:
                ds_sel = None
            ds = self._load_events(ctx, epoch.task, add_bads, data_raw)
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

        _apply_vardef(ds, epoch.vars, self.tests, self.groups)
        _apply_vardef(ds, vardef, self.tests, self.groups)

        if cat:
            model = ds.eval(ctx.get('model'))
            ds = ds.sub(model.isin(cat))
        if not data_raw and 'raw' in ds.info:
            del ds.info['raw']
        return ds

    def build(self, ctx: DerivativeContext) -> Dataset:
        reject = ctx.option('reject', True)
        if reject not in (True, False, 'keep'):
            raise ValueError(f"{reject=}")
        index = ctx.option('index', True)
        if index is True:
            index = 'index'
        elif index and not isinstance(index, str):
            raise TypeError(f"{index=}")
        return self._build_selected_events(
            ctx,
            self.epochs[ctx.get('epoch')],
            reject,
            ctx.option('add_bads', True),
            index,
            ctx.option('data_raw', False),
            ctx.option('vardef'),
            ctx.option('cat'),
        )

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        save.pickle(value, path)
