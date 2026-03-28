# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Event, selected-event, epoch, and evoked derivatives.

These graph nodes own the work behind the event/epoch/evoked ``load_x`` paths.
They may be initialized with configuration objects, immutable values, and
explicit dataset-transform hooks, but they must not receive bound
:class:`Pipeline` methods as execution backends.

When higher-level derivatives need events, epochs, or evoked sensor data, they
should load these nodes through :meth:`DerivativeContext.load` rather than call
back into :class:`Pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Sequence

import mne
import numpy as np

from .. import load, save
from .._data_obj import Datalist, Dataset, Factor, Var, combine
from .._exceptions import DefinitionError, DimensionMismatchError
from .._info import BAD_CHANNELS
from .._names import INTERPOLATE_CHANNELS
from .._text import n_of
from .._mne import shift_mne_epoch_trigger
from ..mne_fixes import _interpolate_bads_eeg, _interpolate_bads_meg
from .definitions import sequence_arg
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext, file_fingerprint
from .epochs import ContinuousEpoch, EpochCollection, SecondaryEpoch, SuperEpoch, decim_param
from .exceptions import FileMissingError
from .pathing import (
    epochs_file_path, event_file_path, evoked_dataset_file_path, evoked_file_path,
    rej_file_path, selected_events_file_path,
)
from .preprocessing import load_raw_dependency, raw_bad_channels_input_name, raw_data_dependency, raw_node_name
from .test_def import TestDims
from .variable_def import Variables


BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')
SELECTED_EVENTS = 'selected-events'
EPOCHS_DATA = 'epochs-ds'
EVOKED_DATA = 'evoked-ds'


def _evoked_comments(evoked: list[mne.Evoked]) -> list[str]:
    return [e.comment or 'No comment' for e in evoked]


def load_evoked_group(
        registry,
        subjects: Sequence[str],
        state: dict[str, Any],
        options: dict[str, Any],
) -> Dataset:
    data = TestDims.coerce(options['data'])
    individual_ndvar = isinstance(data.sensor, str)
    dss = [registry.load(EVOKED_DATA, state={**state, 'subject': subject}, options={**options, 'ndvar': individual_ndvar}) for subject in subjects]
    ndvar = options['ndvar']
    if individual_ndvar:
        ndvar = False
    elif ndvar:
        for ds in dss:
            for evoked in ds['evoked']:
                evoked.info['bads'] = []
    ds = combine(dss, incomplete='drop')
    if not ndvar and not individual_ndvar:
        lens = [len(evoked.times) for evoked in ds['evoked']]
        ulens = set(lens)
        if len(ulens) > 1:
            err = ["Unequal time axis sampling (len):"]
            alens = np.array(lens)
            for length in ulens:
                subjects_ = ', '.join(ds[alens == length, 'subject'].cells)
                err.append(f"{length}: {subjects_}")
            raise DimensionMismatchError('\n'.join(err))
        return ds
    if ndvar and not individual_ndvar:
        evoked = ds['evoked']
        del ds['evoked']
        raw_node = registry._get_node(raw_node_name(state['raw']))
        pipe = raw_node.pipe
        info = evoked[0].info
        sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
        subject = ds[0, 'subject']
        for sensor_type in sensor_types:
            sysname = pipe.get_sysname(info, subject, sensor_type, raw_node.pipes)
            adjacency = pipe.get_adjacency(sensor_type, raw_node.pipes)
            name = 'meg' if sensor_type == 'mag' else sensor_type
            ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
            if sensor_type != 'eog' and isinstance(data.sensor, str):
                ds[name] = getattr(ds[name], data.sensor)('sensor')
    return ds


def load_evoked_request(
        registry,
        groups: dict[str, Sequence[str]],
        current_group: str,
        state: dict[str, Any],
        options: dict[str, Any],
        subjects,
) -> Dataset:
    if subjects is True:
        subjects = current_group
    elif subjects in (None, 1):
        return registry.load(EVOKED_DATA, state=state, options=options)
    if isinstance(subjects, Sequence) and not isinstance(subjects, str):
        return load_evoked_group(registry, subjects, state, options)
    if isinstance(subjects, str) and subjects in groups:
        return load_evoked_group(registry, groups[subjects], state, options)
    return registry.load(EVOKED_DATA, state={**state, 'subject': subjects}, options=options)


def _check_ds(ds: Dataset, source: str, info: dict[str, Any]) -> Dataset:
    if not isinstance(ds, Dataset):
        raise DefinitionError(f"{source} needs to return the events Dataset. Got {ds!r}.")
    if 'i_start' not in ds:
        raise DefinitionError(f"The Dataset returned by {source} does not contain a variable called `i_start`. This variable is required to ascribe events to data samples.")
    if 'trigger' not in ds:
        raise DefinitionError(f"The Dataset returned by {source} does not contain a variable called `trigger`. This variable is required to check rejection files.")
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

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = event_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

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

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = selected_events_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

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


class EpochsDerivative(Derivative[Dataset]):
    name = EPOCHS_DATA
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw', 'epoch', 'rej')
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def __init__(
            self,
            raw,
            epochs: dict[str, Any],
            log,
    ):
        self.raw = raw
        self.epochs = epochs
        self.log = log

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = epochs_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _selected_events_options(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'reject': ctx.option('reject', True),
            'add_bads': ctx.option('add_bads', True),
            'index': False,
            'data_raw': True,
            'vardef': ctx.option('vardef'),
            'cat': ctx.option('cat'),
        }

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.get('epoch')]
        if isinstance(epoch, EpochCollection):
            return tuple(
                Dependency(
                    EPOCHS_DATA,
                    state={'epoch': sub_epoch},
                    options={**ctx.options, 'data_raw': ctx.option('data_raw', False)},
                )
                for sub_epoch in epoch.collect
            )
        return (Dependency(SELECTED_EVENTS, options=self._selected_events_options(ctx)),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        epoch = self.epochs[ctx.get('epoch')]
        return {
            'epoch': epoch._as_dict(),
            'options': ctx.registry.canonicalize({
                key: ctx.option(key)
                for key in (
                    'baseline', 'samplingrate', 'decim', 'pad', 'data_raw',
                    'data', 'trigger_shift', 'tmin', 'tmax', 'tstop',
                    'interpolate_bads',
                )
            }),
        }

    def build(self, ctx: DerivativeContext) -> Dataset:
        epoch_name = ctx.get('epoch')
        epoch = self.epochs[epoch_name]
        if isinstance(epoch, EpochCollection):
            dss = []
            for sub_epoch in epoch.collect:
                ds = ctx.load(EPOCHS_DATA, state={'epoch': sub_epoch}, options={**ctx.options, 'data_raw': ctx.option('data_raw', False)})
                ds[:, 'epoch'] = sub_epoch
                dss.append(ds)
            return combine(dss)

        data = TestDims.coerce(ctx.option('data', 'sensor'))
        if not data.sensor:
            raise ValueError(f"data={data.string!r}; load_evoked is for loading sensor data")
        if data.sensor is not True and not ctx.option('ndvar', True):
            raise ValueError(f"data={data.string!r} with ndvar=False")

        ds = ctx.load(SELECTED_EVENTS, options=self._selected_events_options(ctx))
        if ds.n_cases == 0:
            raise RuntimeError(f"No events left for epoch={epoch.name!r}, subject={ctx.get('subject')!r}")

        tmin = epoch.tmin if ctx.option('tmin') is None else ctx.option('tmin')
        tmax = ctx.option('tmax')
        tstop = ctx.option('tstop')
        if tmax is None and tstop is None:
            tmax = epoch.tmax
        baseline = ctx.option('baseline', False)
        if baseline is True:
            baseline = epoch.baseline
        pad = ctx.option('pad', 0)
        if isinstance(tmax, str):
            tmax = ds.eval(tmax)
            assert isinstance(tmax, Var)
            assert not epoch.post_baseline_trigger_shift, 'not implemented with variable tmax'
            variable_tmax = True
        else:
            variable_tmax = False
        if pad:
            if baseline:
                b0, b1 = baseline
                baseline = (tmin if b0 is None else b0, tmax if b1 is None else b1)
            tmin -= pad
            if tmax is not None:
                tmax = tmax + pad
            elif tstop is not None:
                tstop = tstop + pad
        decim = decim_param(ctx.option('samplingrate'), ctx.option('decim'), epoch, ds.info)

        if isinstance(epoch, ContinuousEpoch):
            split_threshold = epoch.split + (epoch.pad_end + epoch.pad_start)
            diff = ds['time'].diff(to_begin=split_threshold + 1)
            onsets = np.flatnonzero(diff >= split_threshold)
            illegal = {'T_relative', 'events', 'tmax'}.intersection(ds)
            if illegal:
                raise RuntimeError(f"Events contain variables with reserved names: {', '.join(illegal)}")
            event_stops = [*onsets[1:], None]
            events = [ds[i1:i2] for i1, i2 in zip(onsets, event_stops)]
            raw_samplingrate = ds.info['raw'].info['sfreq']
            for events_i in events:
                sample_i = events_i['i_start'] - events_i[0, 'i_start']
                events_i['T_relative'] = sample_i / raw_samplingrate
            ds = ds[onsets]
            ds.info['nested_events'] = 'events'
            ds['events'] = events
            tmin = -epoch.pad_start
            ds['tmax'] = Var([e[-1, 'time'] - e[0, 'time'] + epoch.pad_end for e in events])
            tmax = ds.eval('tmax')
            variable_tmax = True

        if variable_tmax:
            ds['epochs'] = load.mne.variable_length_mne_epochs(ds, tmin, tmax, baseline, allow_truncation=True, decim=decim, reject_by_annotation=False)
            epochs_list = ds['epochs']
        else:
            n = ds.n_cases
            ds = load.mne.add_mne_epochs(ds, tmin, tmax, baseline, decim=decim, drop_bad_chs=False, tstop=tstop, reject_by_annotation=False)
            if ds.n_cases != n:
                self.log.warning("%s missing for %s/%s", n_of(n - ds.n_cases, 'epoch'), ctx.get('subject'), epoch_name)
            if ctx.option('trigger_shift', True) and epoch.post_baseline_trigger_shift:
                ds['epochs'] = shift_mne_epoch_trigger(ds['epochs'], ds[epoch.post_baseline_trigger_shift], epoch.post_baseline_trigger_shift_min, epoch.post_baseline_trigger_shift_max)
            epochs_list = [ds['epochs']]

        info = epochs_list[0].info
        sensor_types = data.data_to_ndvar(info)
        bads_all = None
        bads_individual = None
        interpolate_bads = ctx.option('interpolate_bads', False)
        if interpolate_bads:
            bads_all = info['bads']
            if ds.info[INTERPOLATE_CHANNELS] and any(ds[INTERPOLATE_CHANNELS]):
                bads_individual = ds[INTERPOLATE_CHANNELS]
                if bads_all:
                    base = set(bads_all)
                    bads_individual = [sorted(base.union(bads)) if set(bads).difference(base) else [] for bads in bads_individual]

        if bads_all:
            reset_bads = interpolate_bads != 'keep'
            for epochs in epochs_list:
                epochs.interpolate_bads(reset_bads=reset_bads)
        if ctx.option('reject', True) and bads_individual:
            assert not variable_tmax
            if 'mag' in sensor_types:
                interp_cache = {}
                _interpolate_bads_meg(ds['epochs'], bads_individual, interp_cache)
            if 'eeg' in sensor_types:
                _interpolate_bads_eeg(ds['epochs'], bads_individual)

        ndvar = ctx.option('ndvar', True)
        if ndvar:
            ds.info['sensor_types'] = sensor_types
            pipe = self.raw[ctx.get('raw')]
            for data_kind in sensor_types:
                sysname = pipe.get_sysname(info, ds.info['subject'], data_kind, self.raw)
                adjacency = pipe.get_adjacency(data_kind, self.raw)
                name = 'meg' if data_kind == 'mag' and 'grad' not in sensor_types else data_kind
                if variable_tmax:
                    ys = [load.mne.epochs_ndvar(e, data=data_kind, sysname=sysname, adjacency=adjacency, name=data_kind)[0] for e in ds['epochs']]
                    if isinstance(data.sensor, str):
                        ys = [getattr(y, data.sensor)('sensor') for y in ys]
                else:
                    ys = load.mne.epochs_ndvar(ds['epochs'], data=data_kind, sysname=sysname, adjacency=adjacency)
                    if isinstance(data.sensor, str):
                        ys = getattr(ys, data.sensor)('sensor')
                ds[name] = ys
            if ndvar != 'both':
                del ds['epochs']

        if not ctx.option('data_raw', False):
            del ds.info['raw']
        return ds

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        save.pickle(value, path)


class EvokedDerivative(Derivative[Dataset]):
    name = 'evoked'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count',
    )
    cache_policy = CachePolicy.OPTIONAL

    def __init__(self, epochs: dict[str, Any]):
        self.epochs = epochs

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = evoked_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency(EPOCHS_DATA, options={
            'baseline': True if self.epochs[ctx.get('epoch')].post_baseline_trigger_shift else False,
            'ndvar': False,
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'data_raw': False,
            'vardef': ctx.option('vardef'),
            'interpolate_bads': 'keep',
            'add_bads': True,
            'reject': True,
            'cat': None,
            'trigger_shift': True,
            'data': 'sensor',
        }),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        epoch = self.epochs[ctx.get('epoch')]
        return {
            'epoch': epoch._as_dict(),
            'model': ctx.get('model'),
            'equalize_evoked_count': ctx.get('equalize_evoked_count'),
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'vardef': repr(ctx.option('vardef')),
        }

    def _build_evoked_dataset(self, ctx: DerivativeContext) -> Dataset:
        epoch = self.epochs[ctx.get('epoch')]
        ds = ctx.load(EPOCHS_DATA, options={
            'baseline': True if epoch.post_baseline_trigger_shift else False,
            'ndvar': False,
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'data_raw': False,
            'vardef': ctx.option('vardef'),
            'interpolate_bads': 'keep',
            'add_bads': True,
            'reject': True,
            'cat': None,
            'trigger_shift': True,
            'data': 'sensor',
        })
        model = ctx.get('model')
        equal_count = ctx.get('equalize_evoked_count') == 'eq'
        ds_agg = ds.aggregate(model, drop_bad=True, equal_count=equal_count, drop=('i_start', 't_edf', 'time', 'index', 'trigger'), never_drop=('epochs',))
        ds_agg.rename('epochs', 'evoked')
        model_vars = model.split('%') if model else ()
        for evoked, *cell in ds_agg.zip('evoked', *model_vars):
            evoked.info['description'] = "Eelbrain"
            evoked.comment = ' % '.join(cell)
        return ds_agg

    def build(self, ctx: DerivativeContext) -> Dataset:
        return self._build_evoked_dataset(ctx)

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        evoked = mne.read_evokeds(path, proj=False)
        ds = self._build_evoked_dataset(ctx)
        model = ctx.get('model')
        model_vars = model.split('%') if model else ()
        if model_vars:
            cells = [' % '.join(cell) or 'No comment' for cell in ds.zip(*model_vars)]
        else:
            cells = ['No comment']
        comments = [e.comment for e in evoked]
        if comments != cells:
            if set(comments) == set(cells):
                index = [comments.index(cell) for cell in cells]
                evoked = [evoked[i] for i in index]
            else:
                raise RuntimeError(f"Error reading cached evoked: {comments=}, {cells=}")
        ds['evoked'] = evoked
        return ds

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        mne.write_evokeds(path, value['evoked'], overwrite=True)

    def validate(self, ctx: DerivativeContext, path: Path, manifest) -> bool:
        evoked = mne.read_evokeds(path, proj=False)
        return _evoked_comments(evoked) == manifest.provenance.get('comments', [])

    def provenance(self, ctx: DerivativeContext, value: Dataset) -> dict[str, Any]:
        return {'comments': _evoked_comments(value['evoked'])}


class EvokedDataDerivative(Derivative[Dataset]):
    name = EVOKED_DATA
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count',
    )
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def __init__(self, raw, epochs: dict[str, Any]):
        self.raw = raw
        self.epochs = epochs

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = evoked_dataset_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('evoked', options={
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'vardef': ctx.option('vardef'),
        }),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        data = TestDims.coerce(ctx.option('data', 'sensor'))
        return {
            'baseline': ctx.option('baseline'),
            'cat': sequence_arg('cat', ctx.option('cat')) if ctx.option('cat') else None,
            'ndvar': ctx.option('ndvar', True),
            'data': data.string,
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'vardef': repr(ctx.option('vardef')),
            'data_raw': ctx.option('data_raw', False),
        }

    def build(self, ctx: DerivativeContext) -> Dataset:
        epoch = self.epochs[ctx.get('epoch')]
        ds = ctx.load('evoked', options={
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
            'vardef': ctx.option('vardef'),
        })
        cat = ctx.option('cat')
        model = ctx.get('model')
        if cat:
            if not model:
                raise TypeError(f"{cat=} with {model=}: the cat parameter only applies when a model is specified")
            idx = ds.eval(model).isin(cat)
            ds = ds.sub(idx)
            if ds.n_cases == 0:
                raise RuntimeError(f"Selection with {cat=} resulted in empty Dataset")

        baseline = ctx.option('baseline', False)
        if baseline is True:
            baseline = epoch.baseline
        if baseline and not epoch.post_baseline_trigger_shift:
            for evoked in ds['evoked']:
                mne.baseline.rescale(evoked.data, evoked.times, baseline, 'mean', copy=False)

        ndvar = ctx.option('ndvar', True)
        data = TestDims.coerce(ctx.option('data', 'sensor'))
        if ndvar:
            evoked = ds['evoked']
            if ndvar == 1:
                del ds['evoked']
            pipe = self.raw[ctx.get('raw')]
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
            subject = ctx.get('subject')
            for sensor_type in sensor_types:
                sysname = pipe.get_sysname(info, subject, sensor_type, self.raw)
                adjacency = pipe.get_adjacency(sensor_type, self.raw)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
                if sensor_type != 'eog' and isinstance(data.sensor, str):
                    ds[name] = getattr(ds[name], data.sensor)('sensor')
        if ctx.option('data_raw', False):
            ds.info['raw'] = load_raw_dependency(ctx, add_bads=True, preload=False, noise=False)
        return ds

    def load(self, ctx: DerivativeContext, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        save.pickle(value, path)
