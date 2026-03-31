# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Epoch definitions and epoch/evoked sensor derivatives."""

from pathlib import Path
from typing import Any
from copy import deepcopy
import inspect
from collections.abc import Sequence
import math

import mne
import numpy as np

from .. import load, save
from .._data_obj import Dataset, Var, combine
from .._exceptions import ConfigurationError, DimensionMismatchError
from .._mne import shift_mne_epoch_trigger
from .._names import INTERPOLATE_CHANNELS
from .._text import enumeration
from .._text import n_of
from ..mne_fixes import _interpolate_bads_eeg, _interpolate_bads_meg
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext, Input, file_fingerprint
from .configuration import Configuration, typed_arg
from .pathing import rej_file_path
from .preprocessing import load_raw_dependency, raw_node_name
from .test_def import TestDims


def assemble_epochs(epoch_def, epoch_default):
    epochs = {}
    secondary_epochs = {}
    super_epochs = {}
    collections = {}
    for name, parameters in epoch_def.items():
        # into Epochs object
        if isinstance(parameters, EpochBase):
            epoch = parameters
        elif isinstance(parameters, dict):
            if 'base' in parameters:
                epoch = SecondaryEpoch(**parameters)
            elif 'sub_epochs' in parameters:
                epoch = SuperEpoch(**parameters)
            elif 'collect' in parameters:
                epoch = EpochCollection(**parameters)
            else:
                kwargs = {**epoch_default, **parameters}
                epoch = PrimaryEpoch(**kwargs)
        else:
            raise TypeError(f"Epoch {name}: {parameters!r}")

        if isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            epochs[name] = epoch._link(name, epochs)
        elif isinstance(epoch, SecondaryEpoch):
            secondary_epochs[name] = epoch
        elif isinstance(epoch, SuperEpoch):
            super_epochs[name] = epoch
        elif isinstance(epoch, EpochCollection):
            collections[name] = epoch
        else:
            raise RuntimeError(f"epoch_type={epoch.__class__.__name__}")

    secondary_epochs.update(super_epochs)
    secondary_epochs.update(collections)
    # integrate secondary epochs (epochs with base parameter)
    while secondary_epochs:
        n = len(secondary_epochs)
        for key in list(secondary_epochs):
            if secondary_epochs[key]._can_link(epochs):
                epochs[key] = secondary_epochs.pop(key)._link(key, epochs)
        if len(secondary_epochs) == n:
            raise ConfigurationError(f"Can't resolve epoch dependencies for {enumeration(secondary_epochs)}")
    return epochs


class RejectionInput(Input):
    name = 'rej-input'

    def __init__(
            self,
            root: str | Path,
            artifact_rejection: dict[str, dict[str, Any]],
            epochs: dict[str, Any],
    ):
        self.root = Path(root)
        self.artifact_rejection = artifact_rejection
        self.epochs = epochs

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        rej = self.artifact_rejection[ctx.get('rej')]
        if rej['kind'] is None:
            return {'kind': 'none'}
        epoch = self.epochs[ctx.get('epoch')]
        return {
            'rej': ctx.get('rej'),
            'files': [
                file_fingerprint(self.root, rej_file_path(ctx.state, epoch=e), 'rej-file')
                for e in epoch.rej_file_epochs
            ],
        }


EPOCHS_DATA = 'epochs-ds'


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
    dss = [registry.load('evoked', state={**state, 'subject': subject}, options={**options, 'ndvar': individual_ndvar}) for subject in subjects]
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
        state: dict[str, Any],
        options: dict[str, Any],
        subjects,
) -> Dataset:
    if isinstance(subjects, Sequence) and not isinstance(subjects, str):
        return load_evoked_group(registry, subjects, state, options)
    if isinstance(subjects, str) and subjects in groups:
        return load_evoked_group(registry, groups[subjects], state, options)
    return registry.load('evoked', state={**state, 'subject': subjects}, options=options)


class EpochBase(Configuration):
    baseline = None
    n_cases = None
    trigger_shift = None
    post_baseline_trigger_shift = None
    decim = None

    def _repr_args(self):
        args = []
        for name, param in inspect.signature(self.__class__).parameters.items():
            value = getattr(self, name)
            if param.default is param.empty:
                args.append(repr(value))
            elif value != param.default:
                args.append(f'{name}={value!r}')
        return args

    def __repr__(self):
        args = ', '.join(self._repr_args())
        return f"{self.__class__.__name__}({args})"

    def _link(self, name, epochs):
        out = deepcopy(self)
        out.name = name
        return out


class Epoch(EpochBase):
    """Epoch definition base (non-functional baseclass)"""
    DICT_ATTRS = ('name', 'tmin', 'tmax', 'decim', 'samplingrate', 'baseline', 'vars', 'trigger_shift', 'post_baseline_trigger_shift', 'post_baseline_trigger_shift_min', 'post_baseline_trigger_shift_max')

    # to be set by subclass
    rej_file_epochs = None
    tasks = None

    def __init__(
            self,
            tmin: float | str = -0.1,
            tmax: float | str = 0.6,
            samplingrate: float = None,
            decim: int = None,
            baseline: tuple[float | None, float | None] = None,
            vars: dict = None,
            trigger_shift: float | str = 0.,
            post_baseline_trigger_shift: str = None,
            post_baseline_trigger_shift_min: float = None,
            post_baseline_trigger_shift_max: float = None,
    ):
        if post_baseline_trigger_shift is not None:
            if post_baseline_trigger_shift_min is None or post_baseline_trigger_shift_max is None:
                raise ConfigurationError(f"{post_baseline_trigger_shift=} but missing post_baseline_trigger_shift_min and/or post_baseline_trigger_shift_max")
            cut_time = post_baseline_trigger_shift_max - post_baseline_trigger_shift_min
            if not isinstance(tmax, str) and cut_time >= tmax - tmin:
                raise ConfigurationError("No data remaining after trigger shift")

        if decim is not None:
            if decim < 1:
                raise ValueError(f"{decim=}")
            elif samplingrate is not None:
                raise TypeError(f"{decim=} with {samplingrate=}: only one of these parameters can be specified at a time")
        elif samplingrate is not None:
            if samplingrate <= 0:
                raise ValueError(f"{samplingrate=}")
        else:
            samplingrate = 200

        if baseline is None:
            if tmin >= 0:
                baseline = False
            elif not isinstance(tmax, str) and tmax < 0:
                baseline = (None, None)
            else:
                baseline = (None, 0)
        elif baseline is False:
            pass
        elif len(baseline) != 2:
            raise ValueError(f"{baseline=}: needs to be length 2 tuple")
        else:
            baseline = (typed_arg(baseline[0], float), typed_arg(baseline[1], float))

        if not isinstance(trigger_shift, (float, str)):
            raise TypeError(f"{trigger_shift=}: needs to be float or str")

        self.tmin = typed_arg(tmin, float)
        self.tmax = typed_arg(tmax, float, str)
        self.samplingrate = typed_arg(samplingrate, float, int)
        self.decim = typed_arg(decim, int)
        self.baseline = baseline
        self.vars = vars
        self.trigger_shift = trigger_shift
        self.post_baseline_trigger_shift = post_baseline_trigger_shift
        self.post_baseline_trigger_shift_min = post_baseline_trigger_shift_min
        self.post_baseline_trigger_shift_max = post_baseline_trigger_shift_max


class PrimaryEpoch(Epoch):
    """Epoch based on selecting events from a raw file

    Parameters
    ----------
    task
        Task (raw file) from which to load data.
    sel
        Expression which evaluates in the events Dataset to the index of the
        events included in this Epoch specification.
    tmin
        Start of the epoch, or an expression that evaluates to a
        trial-specific ``tmin`` value in the events dataset (default -0.1).
    tmax
        End of the epoch, or an expression that evaluates to a
        trial-specific ``tmax`` value in the events dataset (default 0.6).
    samplingrate
        Target samplingrate. Needs to divide data samplingrate evenly (e.g.
        ``200`` for data sampled at 1000 Hz; default ``200``).
    decim
        Alternative to ``samplingrate``. Decimate the data by this factor
        (i.e., only keep every ``decim``'th sample).
    baseline : tuple
        The baseline of the epoch (default ``(None, 0)``; if ``tmin > 0``: no
        baseline; if ``tmax < 0``: the whole interval).
    vars
        Add new variables only for this epoch.
        Each entry specifies a variable with the following schema:
        ``{name: definition}``. ``definition`` can be either a string that is
        evaluated in the events-Dataset`, or a
        ``(source_name, {value: code})``-tuple.
        ``source_name`` can also be an interaction, in which case cells are joined
        with spaces (``"f1_cell f2_cell"``).
    trigger_shift
        Shift event triggers before extracting the data [in seconds]. Can be a
        float to shift all triggers by the same value, or a str indicating an event
        variable that specifies the trigger shift for each trigger separately.
        The ``trigger_shift`` applied after loading selected events.
        For secondary epochs the ``trigger_shift`` is applied additively with the
        ``trigger_shift`` of their base epoch.
    post_baseline_trigger_shift
        Shift the trigger (i.e., where epoch time = 0) after baseline correction.
        The value of this entry has to be the name of an event variable providing
        for each epoch the actual amount of time shift (in seconds). If the
        ``post_baseline_trigger_shift`` parameter is specified, the parameters
        ``post_baseline_trigger_shift_min`` and ``post_baseline_trigger_shift_max``
        are also needed, specifying the smallest and largest possible shift. These
        are used to crop the resulting epochs appropriately, to the region from
        ``new_tmin = epoch['tmin'] - post_baseline_trigger_shift_min`` to
        ``new_tmax = epoch['tmax'] - post_baseline_trigger_shift_max``.
    n_cases
        Expected number of epochs. If n_cases is defined, a ``RuntimeError``
        will be raised whenever the actual number of matching events is different.

    See Also
    --------
    Pipeline.epochs

    Examples
    --------
    Selecting events based on a categorial label::

        PrimaryEpoch('task', "variable == 'label'")

    Based on multiple categorial labels::

        PrimaryEpoch('task', "variable.isin(['label1', 'label2'])")

    Based on multiple categorial variables::

        PrimaryEpoch('task', "(variable == 'label') & (other_variable == 'other_label)")

    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sel',)

    def __init__(
            self,
            task: str,
            sel: str = None,
            tmin: float | str = -0.1,
            tmax: float | str = 0.6,
            samplingrate: float = None,
            decim: int = None,
            baseline: tuple[float | None, float | None] = None,
            vars: dict = None,
            trigger_shift: float | str = 0.,
            post_baseline_trigger_shift: str = None,
            post_baseline_trigger_shift_min: float = None,
            post_baseline_trigger_shift_max: float = None,
            n_cases: int = None,
    ):
        Epoch.__init__(self, tmin, tmax, samplingrate, decim, baseline, vars, trigger_shift, post_baseline_trigger_shift, post_baseline_trigger_shift_min, post_baseline_trigger_shift_max)
        self.task = task
        self.sel = typed_arg(sel, str)
        self.n_cases = typed_arg(n_cases, int)
        self.tasks = (task,)

    def _repr_args(self):
        args = [repr(self.task)]
        if self.sel is not None:
            args.append(repr(self.sel))
        for name, param in inspect.signature(Epoch).parameters.items():
            value = getattr(self, name)
            if value != param.default:
                args.append(f'{name}={value!r}')
        return args

    def _link(self, name, epochs):
        out = Epoch._link(self, name, epochs)
        out.rej_file_epochs = (name,)
        return out


class SecondaryEpoch(Epoch):
    """Epoch inheriting events from another epoch

    Secondary epochs inherits events and corresponding trial rejection from
    another epoch (the ``base``). They also inherit all other parameters unless
    they are explicitly overridden. For example ``sel`` can be used to select
    a subset of the events in the ``base`` epoch.

    Parameters
    ----------
    base
        Name of the epoch whose parameters provide defaults for all parameters.
        Additional parameters override parameters of the ``base`` epoch, with the
        except for ``trigger_shift``, which is applied additively to the
        ``trigger_shift`` of the ``base`` epoch.
    sel
        Apply additional event selection `after` applying ``sel`` of the
        ``base`` epoch.
    ...
        Override base-epoch parameters (see :class:`PrimaryEpoch`).

    See Also
    --------
    Pipeline.epochs
    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sel_epoch', 'sel')
    INHERITED_PARAMS = ('tmin', 'tmax', 'decim', 'samplingrate', 'baseline', 'post_baseline_trigger_shift', 'post_baseline_trigger_shift_min', 'post_baseline_trigger_shift_max')

    def __init__(
            self,
            base: str,
            sel: str = None,
            **kwargs,
    ):
        self.sel_epoch = base
        self.sel = typed_arg(sel, str)
        self._kwargs = kwargs

    def _repr_args(self):
        args = [repr(self.sel_epoch)]
        if self.sel is not None:
            args.append(repr(self.sel))
        args.extend([f'{key}={value!r}' for key, value in self._kwargs.items()])
        return args

    def _can_link(self, epochs):
        return self.sel_epoch in epochs

    def _link(self, name, epochs):
        base = epochs[self.sel_epoch]
        if not isinstance(base, (PrimaryEpoch, SecondaryEpoch)):
            raise ConfigurationError(f"Epoch {name}, base={self.sel_epoch!r}: is {base.__class__.__name__}, needs to be PrimaryEpoch or SecondaryEpoch")
        kwargs = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            if param not in kwargs:
                kwargs[param] = getattr(base, param)
        out = Epoch._link(self, name, epochs)
        Epoch.__init__(out, **kwargs)
        out.rej_file_epochs = base.rej_file_epochs
        out.task = base.task
        out.tasks = base.tasks
        return out


class SuperEpoch(Epoch):
    """Combine several other epochs

    Parameters
    ----------
    sub_epochs : sequence of str
        Tuple of epoch names. These epochs are combined to form the super-epoch.
        Epochs are merged at the level of events, so the base epochs can not
        contain post-baseline trigger shifts which are applied after loading
        data (however, the super-epoch can have a post-baseline trigger shift).
    ...
        Override sub-epoch parameters (see :class:`PrimaryEpoch`).

    See Also
    --------
    Pipeline.epochs
    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sub_epochs',)
    INHERITED_PARAMS = ('tmin', 'tmax', 'decim', 'samplingrate', 'baseline')

    def __init__(self, sub_epochs, **kwargs):
        self.sub_epochs = tuple(sub_epochs)
        self._kwargs = kwargs

    def _repr_args(self):
        return [repr(self.sub_epochs), *[f'{k}={v!r}' for k, v in self._kwargs.items()]]

    def _can_link(self, epochs):
        return all(name in epochs for name in self.sub_epochs)

    def _link(self, name, epochs):
        sub_epochs = [epochs[e] for e in self.sub_epochs]
        # check sub-epochs
        for e in sub_epochs:
            if isinstance(e, SuperEpoch):
                raise ConfigurationError(f"Epoch {name}: SuperEpochs can not be defined recursively")
            elif not isinstance(e, Epoch):
                raise ConfigurationError(f"Epoch {name}: sub-epochs must all by PrimaryEpochs")
            elif e.post_baseline_trigger_shift is not None:
                raise ConfigurationError(f"Epoch {name}: Super-epochs are merged on the level of events and can't contain epochs with post_baseline_trigger_shift")
        # find inherited epoch parameters
        kwargs = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            if param in kwargs:
                continue
            values = {getattr(e, param) for e in sub_epochs}
            if len(values) > 1:
                param_repr = ', '.join(repr(v) for v in values)
                raise ConfigurationError(f"Epoch {name}: All sub_epochs must have the same setting for {param}, got {param_repr}")
            kwargs[param] = values.pop()
        out = Epoch._link(self, name, epochs)
        Epoch.__init__(out, **kwargs)
        # tasks, with preserved order
        out.tasks = []
        out.rej_file_epochs = []
        for e in sub_epochs:
            if e.task not in out.tasks:
                out.tasks.append(e.task)
            out.rej_file_epochs.extend(e.rej_file_epochs)
        return out


class EpochCollection(EpochBase):
    """A collection of epochs that are loaded separately.

    For TRFs, a separate TRF will be estimated for each collected epoch (as
    opposed to a :class:`SuperEpoch`, for which sub-epochs will be merged
    before estimating a single TRF).

    Parameters
    ----------
    collect
        Epochs to collect.

    See Also
    --------
    Pipeline.epochs
    """
    # IMPLEMENTATION ALTERNATIVE?
    # ---------------------------
    # In analogy to standard epochs, the "model" parameter could be used to fit
    # a separate TRF per cell.
    #
    #  - Logistic complication: I would want to be able to fit only cell 1
    #    first, and later fit cell 2, without redundant refitting.
    DICT_ATTRS = ('collect',)

    def __init__(self, collect: Sequence[str]):
        self.collect = collect
        EpochBase.__init__(self)

    def _repr_args(self):
        return [repr(self.collect)]

    def _can_link(self, epochs):
        return all(name in epochs for name in self.collect)

    def _link(self, name, epochs):
        sub_epochs = [epochs[e] for e in self.collect]
        out = EpochBase._link(self, name, epochs)
        # make sure basic attributes match
        for param in SuperEpoch.INHERITED_PARAMS:
            values = {getattr(e, param) for e in sub_epochs}
            if len(values) > 1:
                param_repr = ', '.join(repr(v) for v in values)
                raise ConfigurationError(f"Epoch {name}: All sub-epochs must have the same setting for {param}, got {param_repr}")
            setattr(out, param, values.pop())
        # dependencies
        tasks = set()
        rej_file_epochs = set()
        for e in sub_epochs:
            tasks.update(e.tasks)
            rej_file_epochs.update(e.rej_file_epochs)
        out.tasks = sorted(tasks)
        out.rej_file_epochs = sorted(rej_file_epochs)
        return out


class ContinuousEpoch(EpochBase):
    """Epoch spanning multiple events for continuous analysis

    A :class:`ContinuousEpoch` will extract a continuous segment of data from
    the first event to the last event. ``pad_start`` and ``pad_stop`` determine
    how much extra time to include before the first event and after the last
    event (to allow using the data surrounding these events for estimating TRFs
    with negative and positive lags). ``split`` controls whether to break up the
    data into multuple segments when there are long pauses between successive
    events.

    When using :meth:`Pipeline.load_epochs`, each row of the returned
    :class:`Dataset` will contain the events in the epoch alongside the data.

    Parameters
    ----------
    task
        Task (raw file) from which to load data.
    sel
        Expression which evaluates in the events Dataset to the index of the
        events included in this Epoch specification (default is all events).
    pad_start
        Time to add before the first event (in seconds, default 0.100).
    pad_end
        Time to add after the last event (in seconds, default 1).
    split
        Split into several continuous epochs whenever time between used data
        (event times ± ``pad``) is larger than ``split`` (default 10). For
        example, in an experiment with many 2 s long trials which are grouped
        into 2 blocks with a break of 50 s, this would result in two epochs, one
        for each block.
    samplingrate
        Target samplingrate. Needs to divide data samplingrate evenly (e.g.
        ``200`` for data sampled at 1000 Hz; default ``200``).
    vars
        Add new variables only for this epoch.
        Each entry specifies a variable with the following schema:
        ``{name: definition}``. ``definition`` can be either a string that is
        evaluated in the events-Dataset`, or a
        ``(source_name, {value: code})``-tuple.
        ``source_name`` can also be an interaction, in which case cells are joined
        with spaces (``"f1_cell f2_cell"``).
    """
    DICT_ATTRS = ('name', 'task', 'sel', 'pad_start', 'pad_end', 'split', 'samplingrate', 'vars')

    def __init__(
            self,
            task: str,
            sel: str = None,
            pad_start: float = 0.100,
            pad_end: float = 1.000,
            split: float = 10,
            samplingrate: float = 200,
            vars: dict = None,
    ):
        EpochBase.__init__(self)
        self.task = typed_arg(task, str)
        self.sel = typed_arg(sel, str)
        self.pad_start = typed_arg(pad_start, float)
        self.pad_end = typed_arg(pad_end, float)
        self.split = typed_arg(split, float)
        self.samplingrate = typed_arg(samplingrate, float, int)
        self.vars = vars
        self.tasks = (task,)


class EpochsDerivative(Derivative[Dataset]):
    name = EPOCHS_DATA
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw', 'epoch', 'rej')
    cache_suffix = '.pickle'
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def __init__(
            self,
            raw,
            epochs: dict[str, Any],
    ):
        self.raw = raw
        self.epochs = epochs

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
                Dependency(EPOCHS_DATA, state={'epoch': sub_epoch}, options={**ctx.options, 'data_raw': ctx.option('data_raw', False)})
                for sub_epoch in epoch.collect
            )
        return (Dependency('selected-events', options=self._selected_events_options(ctx)),)

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

        ds = ctx.load('selected-events', options=self._selected_events_options(ctx))
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
                ctx.registry.log.warning("%s missing for %s/%s", n_of(n - ds.n_cases, 'epoch'), ctx.get('subject'), epoch_name)
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
    cache_suffix = '-ave.fif'

    def __init__(self, epochs: dict[str, Any]):
        self.epochs = epochs

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

    def _apply_load_options(self, ctx: DerivativeContext, ds: Dataset) -> Dataset:
        epoch = self.epochs[ctx.get('epoch')]
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
            raw_node = ctx.registry._get_node(raw_node_name(ctx.get('raw')))
            pipe = raw_node.pipe
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
            subject = ctx.get('subject')
            for sensor_type in sensor_types:
                sysname = pipe.get_sysname(info, subject, sensor_type, raw_node.pipes)
                adjacency = pipe.get_adjacency(sensor_type, raw_node.pipes)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
                if sensor_type != 'eog' and isinstance(data.sensor, str):
                    ds[name] = getattr(ds[name], data.sensor)('sensor')
        if ctx.option('data_raw', False):
            ds.info['raw'] = load_raw_dependency(ctx, add_bads=True, preload=False, noise=False)
        return ds

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
        return self._apply_load_options(ctx, ds)

    def save(self, ctx: DerivativeContext, path: Path, value: Dataset) -> None:
        mne.write_evokeds(path, value['evoked'], overwrite=True)

    def validate(self, ctx: DerivativeContext, path: Path, manifest) -> bool:
        evoked = mne.read_evokeds(path, proj=False)
        return _evoked_comments(evoked) == manifest.provenance.get('comments', [])

    def provenance(self, ctx: DerivativeContext, value: Dataset) -> dict[str, Any]:
        return {'comments': _evoked_comments(value['evoked'])}


def decim_param(
        samplingrate: int,
        decim: int,
        epoch: Epoch | None,
        info: dict,
        minimal: bool = False,  # try to infer minimally necessary samplingrate
) -> int:
    if samplingrate is not None:
        if decim is not None:
            raise TypeError(f"{samplingrate=}, {decim=}: can only specify one at a time")
    elif decim is not None:
        return decim
    elif epoch is not None and not minimal:
        if epoch.decim is not None:
            return epoch.decim
        elif epoch.samplingrate is not None:
            samplingrate = epoch.samplingrate

    if samplingrate is not None:
        decim_ratio = info['sfreq'] / samplingrate
        rounded_decim_ratio = round(decim_ratio)
        if not math.isclose(decim_ratio, rounded_decim_ratio, rel_tol=1e-3):
            raise ValueError(f"{samplingrate=} with data at {info['sfreq']:g} Hz: needs to be integer ratio")
        return rounded_decim_ratio

    if minimal:
        if h_freq := info.get('lowpass'):
            return int(info['sfreq'] / (h_freq * 2.5))
        else:
            return int(info['sfreq'] / 100)

    return 1
