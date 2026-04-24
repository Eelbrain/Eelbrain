# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Epoch definitions and epoch/evoked sensor derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import inspect
from collections.abc import Mapping, Sequence
import math
import shutil

import mne
import numpy as np

from .. import load
from .._data_obj import Datalist, Dataset, Var, combine
from .._exceptions import ConfigurationError, DimensionMismatchError
from .._info import BAD_CHANNELS
from .._mne import shift_mne_epoch_trigger
from .._names import INTERPOLATE_CHANNELS
from .._text import enumeration
from .._text import n_of
from ..mne_fixes import _interpolate_bads_eeg, _interpolate_bads_meg
from .derivative_cache import CachePolicy, Dependency, Derivative, Request, Input, UncachedDerivative, file_fingerprint
from .configuration import Configuration, typed_arg
from .pathing import rej_file_path
from .preprocessing import load_raw_dependency, raw_node_name
from .test_def import TestDims


EpochBaselineArg = Literal[False] | tuple[float | None, float | None] | None


def _shared_sub_epoch_parameters(name: str, sub_epochs: Sequence[EpochBase], parameters: Sequence[str]) -> dict[str, Any]:
    out = {}
    for param in parameters:
        values = {getattr(sub_epoch, param) for sub_epoch in sub_epochs}
        if len(values) > 1:
            param_repr = ', '.join(repr(v) for v in values)
            raise ConfigurationError(f"Epoch {name}: All sub-epochs must have the same setting for {param}, got {param_repr}")
        out[param] = values.pop()
    return out


def assemble_epochs(epoch_def: Mapping[str, EpochBase]) -> dict[str, EpochBase]:
    """Resolve epoch definitions and cache epoch-family dependent parameters.

    This binds each epoch object's ``name`` and lets the epoch classes cache
    deterministic graph-dependent parameters such as inherited epoch
    parameters, ``task``/``tasks``, and ``rej_file_epochs``.
    """
    epochs = {}
    unresolved_epochs = {}
    seen_epoch_ids = {}
    for name, epoch in epoch_def.items():
        if not isinstance(epoch, EpochBase):
            raise TypeError(f"Epoch {name}: {epoch!r}; need an epoch definition")
        previous_name = seen_epoch_ids.setdefault(id(epoch), name)
        if previous_name != name:
            raise TypeError(f"Epoch {name}: reuses the same epoch object as {previous_name!r}; define a separate EpochBase instance for each name")
        epoch._store_name(name)

        if isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            epoch._store_dependent_parameters()
            epochs[name] = epoch
        elif isinstance(epoch, (SecondaryEpoch, SuperEpoch, EpochCollection)):
            unresolved_epochs[name] = epoch
        else:
            raise RuntimeError(f"epoch_type={epoch.__class__.__name__}")

    while unresolved_epochs:
        n = len(unresolved_epochs)
        for key in list(unresolved_epochs):
            epoch = unresolved_epochs[key]
            if isinstance(epoch, SecondaryEpoch):
                ready = epoch.sel_epoch in epochs
            elif isinstance(epoch, SuperEpoch):
                ready = all(name in epochs for name in epoch.sub_epochs)
            elif isinstance(epoch, EpochCollection):
                ready = all(name in epochs for name in epoch.collect)
            else:
                raise RuntimeError(f"epoch_type={epoch.__class__.__name__}")
            if not ready:
                continue
            epoch = unresolved_epochs.pop(key)
            epoch._store_dependent_parameters(epochs)
            epochs[key] = epoch
        if len(unresolved_epochs) == n:
            raise ConfigurationError(f"Can't resolve epoch dependencies for {enumeration(unresolved_epochs)}")
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

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        rej = self.artifact_rejection[ctx.state['rej']]
        if rej['kind'] is None:
            return {'kind': 'none'}
        epoch = self.epochs[ctx.state['epoch']]
        return {
            'rej': ctx.state['rej'],
            'files': [
                file_fingerprint(self.root, self.root / rej_file_path(ctx.state, epoch=e), 'rej-file')
                for e in epoch.rej_file_epochs
            ],
        }


def _evoked_comments(evoked: list[mne.Evoked]) -> list[str]:
    return [e.comment or 'No comment' for e in evoked]


def _epochs_selection_metadata(value) -> list[int] | list[list[int] | None] | None:
    if isinstance(value, Datalist):
        selections = []
        for epochs in value:
            selection = getattr(epochs, 'selection', None)
            selections.append(None if selection is None else selection.tolist())
        return selections
    selection = getattr(value, 'selection', None)
    return None if selection is None else selection.tolist()


def _apply_epochs_selection(ds: Dataset, selection: np.ndarray | None) -> Dataset:
    if selection is None or (len(selection) == ds.n_cases and np.array_equal(selection, np.arange(ds.n_cases))):
        return ds
    ds = ds[selection]
    ds.info = ds.info.copy()
    ds.info['epochs.selection'] = selection
    return ds


class EpochBase(Configuration):
    """Base class for epoch definitions."""
    baseline = None
    n_cases = None
    trigger_shift = None
    post_baseline_trigger_shift = None
    decim = None
    _rej_file_epochs_from_name = False

    def _prepare_selected_events(
            self,
            ds: Dataset,
            subject: str,
    ) -> Dataset:
        """Prepare the selected-events shell for this epoch.

        Parameters
        ----------
        ds
            Selected-events dataset for one subject after graph-level event
            selection and load options have been applied.
        subject
            Subject identifier used for error messages.

        Returns
        -------
        ds
            Dataset to use as the event shell for epoch extraction.
            Implementations may return a rewritten dataset, for example for
            continuous epochs.
        """
        raise NotImplementedError(f"{self.__class__.__name__}._prepare_selected_events()")

    def _extraction_parameters(
            self,
            ds: Dataset,
            options: dict[str, Any],
    ) -> tuple[float, Any, float | None, Any, int, bool]:
        """Compute epoch extraction parameters for a prepared shell.

        Parameters
        ----------
        ds
            Prepared event shell returned by :meth:`_prepare_selected_events`.
        options
            `epochs` node options that affect extraction, such as time-window,
            baseline, padding, and decimation overrides.

        Returns
        -------
        tmin
            Start of the extraction window in seconds.
        tmax
            End of the extraction window, or a per-epoch :class:`Var`.
        tstop
            Optional explicit stop time for fixed-length extraction.
        baseline
            Baseline interval to apply during epoch extraction.
        decim
            Decimation factor for MNE epoch extraction.
        variable_tmax
            Whether ``tmax`` varies per epoch.
        """
        raise NotImplementedError(f"{self.__class__.__name__}._extraction_parameters()")

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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase] = None) -> None:
        if self._rej_file_epochs_from_name:
            self.rej_file_epochs = (self.name,)


class Epoch(EpochBase):
    """Epoch definition base (non-functional baseclass)"""
    DICT_ATTRS = ('tmin', 'tmax', 'decim', 'samplingrate', 'baseline', 'trigger_shift', 'post_baseline_trigger_shift', 'post_baseline_trigger_shift_min', 'post_baseline_trigger_shift_max')

    # to be set by subclass
    rej_file_epochs = None
    tasks = None

    def _set_epoch_parameters(
            self,
            tmin: float | str = -0.1,
            tmax: float | str = 0.6,
            samplingrate: float = None,
            decim: int = None,
            baseline: EpochBaselineArg = None,
            trigger_shift: float | str = 0.,
            post_baseline_trigger_shift: str = None,
            post_baseline_trigger_shift_min: float = None,
            post_baseline_trigger_shift_max: float = None,
    ) -> None:
        if post_baseline_trigger_shift is not None:
            if post_baseline_trigger_shift_min is None or post_baseline_trigger_shift_max is None:
                raise ConfigurationError(f"{post_baseline_trigger_shift=} but missing post_baseline_trigger_shift_min and/or post_baseline_trigger_shift_max")
            cut_time = post_baseline_trigger_shift_max - post_baseline_trigger_shift_min
            if not isinstance(tmax, str) and cut_time >= tmax - tmin:
                raise ConfigurationError("No data remaining after trigger shift")

        if decim is not None:
            if decim < 1:
                raise ValueError(f"{decim=}")
            if samplingrate is not None:
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
        elif baseline is not False:
            if len(baseline) != 2:
                raise ValueError(f"{baseline=}: needs to be length 2 tuple")
            baseline = (typed_arg(baseline[0], float), typed_arg(baseline[1], float))

        if not isinstance(trigger_shift, (float, str)):
            raise TypeError(f"{trigger_shift=}: needs to be float or str")

        self.tmin = typed_arg(tmin, float)
        self.tmax = typed_arg(tmax, float, str)
        self.samplingrate = typed_arg(samplingrate, float, int)
        self.decim = typed_arg(decim, int)
        self.baseline = baseline
        self.trigger_shift = trigger_shift
        self.post_baseline_trigger_shift = post_baseline_trigger_shift
        self.post_baseline_trigger_shift_min = post_baseline_trigger_shift_min
        self.post_baseline_trigger_shift_max = post_baseline_trigger_shift_max

    def __init__(
            self,
            tmin: float | str = -0.1,
            tmax: float | str = 0.6,
            samplingrate: float = None,
            decim: int = None,
            baseline: EpochBaselineArg = None,
            trigger_shift: float | str = 0.,
            post_baseline_trigger_shift: str = None,
            post_baseline_trigger_shift_min: float = None,
            post_baseline_trigger_shift_max: float = None,
    ):
        self._set_epoch_parameters(
            tmin,
            tmax,
            samplingrate,
            decim,
            baseline,
            trigger_shift,
            post_baseline_trigger_shift,
            post_baseline_trigger_shift_min,
            post_baseline_trigger_shift_max,
        )

    def _prepare_selected_events(
            self,
            ds: Dataset,
            subject: str,
    ) -> Dataset:
        if ds.n_cases == 0:
            raise RuntimeError(f"No events left for epoch={self.name!r}, subject={subject!r}")
        return ds

    def _extraction_parameters(
            self,
            ds: Dataset,
            options: dict[str, Any],
    ) -> tuple[float, Any, float | None, Any, int, bool]:
        tmin = self.tmin if options['tmin'] is None else options['tmin']
        tmax = options['tmax']
        tstop = options['tstop']
        if tmax is None and tstop is None:
            tmax = self.tmax
        baseline = self.baseline if options['baseline'] is True else options['baseline']
        if isinstance(tmax, str):
            tmax = ds.eval(tmax)
            assert isinstance(tmax, Var)
            assert not self.post_baseline_trigger_shift, 'not implemented with variable tmax'
            variable_tmax = True
        else:
            variable_tmax = False
        if pad := options['pad']:
            if baseline:
                b0, b1 = baseline
                baseline = (tmin if b0 is None else b0, tmax if b1 is None else b1)
            tmin -= pad
            if tmax is not None:
                tmax = tmax + pad
            elif tstop is not None:
                tstop = tstop + pad
        decim = decim_param(options['samplingrate'], options['decim'], self, ds.info)
        return tmin, tmax, tstop, baseline, decim, variable_tmax


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
    trigger_shift
        Shift event triggers before extracting the data [in seconds]. Can be a
        float to shift all triggers by the same value, or a str indicating an event
        variable that specifies the trigger shift for each trigger separately.
        The ``trigger_shift`` applied after loading selected events.
        For secondary epochs the ``trigger_shift`` is applied additively with the
        ``trigger_shift`` of their base epoch.
    post_baseline_trigger_shift
        Shift the trigger (i.e., where epoch time = 0) after baseline correction.
        The value of this entry is an expression that is evaluated in the
        selected-events Dataset and needs to yield the actual amount of time
        shift (in seconds) for each epoch. If the
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
    _rej_file_epochs_from_name = True

    def __init__(
            self,
            task: str,
            sel: str = None,
            tmin: float | str = -0.1,
            tmax: float | str = 0.6,
            samplingrate: float = None,
            decim: int = None,
            baseline: EpochBaselineArg = None,
            trigger_shift: float | str = 0.,
            post_baseline_trigger_shift: str = None,
            post_baseline_trigger_shift_min: float = None,
            post_baseline_trigger_shift_max: float = None,
            n_cases: int = None,
    ):
        Epoch.__init__(self, tmin, tmax, samplingrate, decim, baseline, trigger_shift, post_baseline_trigger_shift, post_baseline_trigger_shift_min, post_baseline_trigger_shift_max)
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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase]) -> None:
        base = epochs[self.sel_epoch]
        if not isinstance(base, (PrimaryEpoch, SecondaryEpoch)):
            raise ConfigurationError(f"Epoch {self.name}, base={self.sel_epoch!r}: is {base.__class__.__name__}, needs to be PrimaryEpoch or SecondaryEpoch")
        params = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            params.setdefault(param, getattr(base, param))
        self._set_epoch_parameters(**params)
        self.rej_file_epochs = base.rej_file_epochs
        self.task = base.task
        self.tasks = base.tasks


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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase]) -> None:
        sub_epochs = [epochs[sub_epoch] for sub_epoch in self.sub_epochs]
        for sub_epoch in sub_epochs:
            if isinstance(sub_epoch, SuperEpoch):
                raise ConfigurationError(f"Epoch {self.name}: SuperEpochs can not be defined recursively")
            if not isinstance(sub_epoch, Epoch):
                raise ConfigurationError(f"Epoch {self.name}: sub-epochs must all by PrimaryEpochs")
            if sub_epoch.post_baseline_trigger_shift is not None:
                raise ConfigurationError(f"Epoch {self.name}: Super-epochs are merged on the level of events and can't contain epochs with post_baseline_trigger_shift")
        params = self._kwargs.copy()
        for param, value in _shared_sub_epoch_parameters(self.name, sub_epochs, self.INHERITED_PARAMS).items():
            params.setdefault(param, value)
        self._set_epoch_parameters(**params)
        self.tasks = list(dict.fromkeys(sub_epoch.task for sub_epoch in sub_epochs))
        self.rej_file_epochs = [epoch_name for sub_epoch in sub_epochs for epoch_name in sub_epoch.rej_file_epochs]


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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase]) -> None:
        sub_epochs = [epochs[sub_epoch] for sub_epoch in self.collect]
        for param, value in _shared_sub_epoch_parameters(self.name, sub_epochs, SuperEpoch.INHERITED_PARAMS).items():
            setattr(self, param, value)
        self.tasks = sorted({task for sub_epoch in sub_epochs for task in sub_epoch.tasks})
        self.rej_file_epochs = sorted({epoch_name for sub_epoch in sub_epochs for epoch_name in sub_epoch.rej_file_epochs})


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
    """
    DICT_ATTRS = ('task', 'sel', 'pad_start', 'pad_end', 'split', 'samplingrate')
    _rej_file_epochs_from_name = True

    def __init__(
            self,
            task: str,
            sel: str = None,
            pad_start: float = 0.100,
            pad_end: float = 1.000,
            split: float = 10,
            samplingrate: float = 200,
    ):
        EpochBase.__init__(self)
        self.task = typed_arg(task, str)
        self.sel = typed_arg(sel, str)
        self.pad_start = typed_arg(pad_start, float)
        self.pad_end = typed_arg(pad_end, float)
        self.split = typed_arg(split, float)
        self.samplingrate = typed_arg(samplingrate, float, int)
        self.tasks = (task,)

    def _prepare_selected_events(
            self,
            ds: Dataset,
            subject: str,
    ) -> Dataset:
        if ds.n_cases == 0:
            raise RuntimeError(f"No events left for epoch={self.name!r}, subject={subject!r}")

        split_threshold = self.split + self.pad_start + self.pad_end
        onsets = np.flatnonzero(ds['onset'].diff(to_begin=split_threshold + 1) >= split_threshold)
        illegal = {'T_relative', 'events', 'tmax'}.intersection(ds)
        if illegal:
            raise RuntimeError(f"Events contain variables with reserved names: {', '.join(illegal)}")
        events = [ds[i1:i2] for i1, i2 in zip(onsets, [*onsets[1:], None])]
        raw_samplingrate = ds.info['raw'].info['sfreq']
        for events_i in events:
            sample_i = events_i['sample'] - events_i[0, 'sample']
            events_i['T_relative'] = sample_i / raw_samplingrate
        ds = ds[onsets]
        ds.info['nested_events'] = 'events'
        ds['events'] = events
        ds['tmax'] = Var([events_i[-1, 'onset'] - events_i[0, 'onset'] + self.pad_end for events_i in events])
        return ds

    def _extraction_parameters(
            self,
            ds: Dataset,
            options: dict[str, Any],
    ) -> tuple[float, Any, float | None, Any, int, bool]:
        baseline = self.baseline if options['baseline'] is True else options['baseline']
        decim = decim_param(options['samplingrate'], options['decim'], self, ds.info)
        return -self.pad_start, ds.eval('tmax'), None, baseline, decim, True


class EpochsDerivative(Derivative[Any]):
    """Epoch dataset with cached MNE epochs as internal artifact.

    Options
    -------
    ndvar
        Whether to convert epoch data to NDVars (`True`, `False`, or `'both'`).
    data
        Sensor representation to return.
    data_raw
        Whether to keep the raw object in ``ds.info['raw']``.
    add_bads
        Whether to include current bad-channel information in the returned
        selected-events dataset.
    baseline
        Baseline correction to apply while creating cached epochs.
    samplingrate
        Sampling rate override for epoch extraction.
    decim
        Decimation override for epoch extraction.
    pad
        Extra time padding to add before epoch extraction.
    trigger_shift
        Whether to apply trigger shifting from the epoch definition.
    tmin, tmax, tstop
        Time window overrides for epoch extraction.
    interpolate_bads
        Whether and how to interpolate bad channels while building epochs.
    reject
        Whether to apply per-epoch interpolation/rejection state.
    cat
        Optional subset of model cells to keep before epoch creation.
    """
    name = 'epochs'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'raw', 'epoch', 'rej')
    cache_suffix = '.epochs'
    cache_policy = CachePolicy.OPTIONAL
    OPTION_DEFAULTS = {
        'baseline': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'trigger_shift': True,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'reject': True,
        'cat': None,
    }
    VIEW_OPTION_DEFAULTS = {'ndvar': True, 'data': 'sensor', 'data_raw': False, 'add_bads': True}

    def __init__(
            self,
            raw,
            epochs: dict[str, Any],
    ):
        self.raw = raw
        self.epochs = epochs

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            return tuple(
                Dependency(
                    'epochs',
                    label=sub_epoch,
                    state={'epoch': sub_epoch},
                    options=ctx.options_for('epochs', *self.OPTION_DEFAULTS, ndvar=False, data='sensor', data_raw=False),
                )
                for sub_epoch in epoch.collect
            )
        return (
            Dependency(
                raw_node_name(ctx.state['raw']),
                options=ctx.options_for(raw_node_name(ctx.state['raw']), add_bads=bool(ctx.options['interpolate_bads']), preload=False, noise=False),
            ),
            Dependency(
                'selected-events',
                options=ctx.options_for('selected-events', reject=ctx.options['reject'], add_bads=True, index=False, data_raw=True, cat=ctx.options['cat']),
                view='epochs',
            ),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        epoch = self.epochs[ctx.state['epoch']]
        return self.standard_fingerprint(ctx, definitions={'epoch': epoch._as_dict()})

    def build(self, ctx: Request):
        epoch_name = ctx.state['epoch']
        epoch = self.epochs[epoch_name]
        if isinstance(epoch, EpochCollection):
            epochs_list = []
            for sub_epoch in epoch.collect:
                ds = ctx.load(sub_epoch)
                epoch_value = ds['epochs']
                if isinstance(epoch_value, Datalist):
                    epochs_list.extend(epoch_value)
                else:
                    epochs_list.append(epoch_value)
            return Datalist(epochs_list, 'epochs')

        ds = ctx.load(view='shell')
        tmin, tmax, tstop, baseline, decim, variable_tmax = epoch._extraction_parameters(ds, ctx.options)
        if variable_tmax:
            epoch_value = load.mne.variable_length_mne_epochs(ds, tmin, tmax, baseline, allow_truncation=True, decim=decim, reject_by_annotation=False, i_start='sample', trigger='value')
            epochs_list = epoch_value
        else:
            n = ds.n_cases
            ds = load.mne.add_mne_epochs(ds, tmin, tmax, baseline, decim=decim, drop_bad_chs=False, tstop=tstop, reject_by_annotation=False, i_start='sample', trigger='value')
            if ds.n_cases != n:
                ctx.registry.log.warning("%s missing for %s/%s", n_of(n - ds.n_cases, 'epoch'), ctx.state['subject'], epoch_name)
            if ctx.options['trigger_shift'] and epoch.post_baseline_trigger_shift:
                shift = ds.eval(epoch.post_baseline_trigger_shift)
                ds['epochs'] = shift_mne_epoch_trigger(ds['epochs'], shift, epoch.post_baseline_trigger_shift_min, epoch.post_baseline_trigger_shift_max)
            epoch_value = ds['epochs']
            epochs_list = [epoch_value]

        info = epochs_list[0].info
        bads_all = None
        bads_individual = None
        interpolate_bads = ctx.options['interpolate_bads']
        if interpolate_bads:
            bads_all = info['bads']
            if ds.info[INTERPOLATE_CHANNELS] and any(ds[INTERPOLATE_CHANNELS]):
                bads_individual = ds[INTERPOLATE_CHANNELS]
                base = set(bads_all or ())
                bads_individual = [sorted(base.union(bads)) if set(bads).difference(base) else [] for bads in bads_individual]

        if bads_all:
            reset_bads = interpolate_bads != 'keep'
            for epochs in epochs_list:
                epochs.interpolate_bads(reset_bads=reset_bads)
        if ctx.options['reject'] and bads_individual:
            assert not variable_tmax
            if 'mag' in TestDims.coerce('sensor').data_to_ndvar(info):
                interp_cache = {}
                _interpolate_bads_meg(epoch_value, bads_individual, interp_cache)
            if 'eeg' in TestDims.coerce('sensor').data_to_ndvar(info):
                _interpolate_bads_eeg(epoch_value, bads_individual)
        return epoch_value

    def load(self, ctx: Request, path: Path):
        metadata = ctx.artifact_metadata
        if metadata['kind'] == 'datalist':
            return Datalist(
                [mne.read_epochs(path / relpath, proj=False) for relpath in metadata['files']],
                metadata['name'],
                metadata['fmt'],
            )
        return mne.read_epochs(path / metadata['file'], proj=False)

    def save(self, ctx: Request, path: Path, value) -> None:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        path.mkdir()
        if isinstance(value, Datalist):
            for i, epochs in enumerate(value):
                epochs.save(path / f'epochs-{i:04d}-epo.fif', overwrite=True)
        else:
            value.save(path / 'epochs-0000-epo.fif', overwrite=True)

    def artifact_metadata(self, ctx: Request, value) -> dict[str, Any]:
        if isinstance(value, Datalist):
            return {
                'kind': 'datalist',
                'files': [f'epochs-{i:04d}-epo.fif' for i in range(len(value))],
                'name': value.name,
                'fmt': value._fmt,
                'selections': _epochs_selection_metadata(value),
            }
        return {'kind': 'single', 'file': 'epochs-0000-epo.fif', 'selection': _epochs_selection_metadata(value)}

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view is None:
            return self.fingerprint(ctx)
        if view != 'shell':
            raise ValueError(f"{self.name!r} does not define dependency view {view!r}")
        return self.fingerprint(ctx)

    def load_view(self, ctx: Request, view: str):
        if view != 'shell':
            return super().load_view(ctx, view)

        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"{self.name!r} does not define view {view!r} for {epoch.__class__.__name__}")

        ds = ctx.load(
            'selected-events',
            options=ctx.options_for(
                'selected-events',
                reject=ctx.options['reject'],
                add_bads=True,
                index='index' if ctx.options['cat'] else False,
                data_raw=True,
                cat=ctx.options['cat'],
            ),
        )
        ds = epoch._prepare_selected_events(ds, ctx.state['subject'])
        selection = ctx.artifact_metadata.get('selection')
        if selection is None and ctx.artifact_path.exists():
            selection = getattr(ctx.load_artifact(), 'selection', None)
        if selection is not None:
            selection = np.asarray(selection, dtype=int)
            if 'index' in ds:
                selection = np.flatnonzero(np.isin(ds['index'].x, selection))
            ds = _apply_epochs_selection(ds, selection)
        return ds

    def apply_view_options(self, ctx: Request, epoch_value):
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            dss = []
            for sub_epoch in epoch.collect:
                ds = ctx.load(
                    'epochs',
                    state={'epoch': sub_epoch},
                    options=ctx.options_for('epochs', *self.OPTION_DEFAULTS, *self.VIEW_OPTION_DEFAULTS),
                )
                ds[:, 'epoch'] = sub_epoch
                dss.append(ds)
            return combine(dss)

        data = TestDims.coerce(ctx.view_options['data'])
        if not data.sensor:
            raise ValueError(f"data={data.string!r}; load_evoked is for loading sensor data")
        if data.sensor is not True and not ctx.view_options['ndvar']:
            raise ValueError(f"data={data.string!r} with ndvar=False")

        ds = ctx.load(view='shell')
        _, _, _, _, _, variable_tmax = epoch._extraction_parameters(ds, ctx.options)
        selected_events = ctx.load(
            'selected-events',
            options=ctx.options_for(
                'selected-events',
                reject=ctx.options['reject'],
                add_bads=ctx.view_options['add_bads'],
                index=False,
                data_raw=ctx.view_options['data_raw'],
                cat=ctx.options['cat'],
            ),
        )
        ds.info[BAD_CHANNELS] = selected_events.info.get(BAD_CHANNELS, [])
        if 'raw' in selected_events.info:
            ds.info['raw'] = selected_events.info['raw']
        else:
            ds.info.pop('raw', None)
        raw = ds.info.get('raw')
        bads = raw.info['bads'] if raw is not None else ds.info.get(BAD_CHANNELS, [])
        if isinstance(epoch_value, Datalist):
            if any(epochs.info['bads'] != bads for epochs in epoch_value):
                epoch_value = Datalist([epochs.copy() for epochs in epoch_value], epoch_value.name, epoch_value._fmt)
                for epochs in epoch_value:
                    epochs.info['bads'] = bads
        elif epoch_value.info['bads'] != bads:
            epoch_value = epoch_value.copy()
            epoch_value.info['bads'] = bads
        ds['epochs'] = epoch_value

        ndvar = ctx.view_options['ndvar']
        if ndvar:
            epochs_list = epoch_value if isinstance(epoch_value, Datalist) else [epoch_value]
            info = epochs_list[0].info
            sensor_types = data.data_to_ndvar(info)
            ds.info['sensor_types'] = sensor_types
            source_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            for data_kind in sensor_types:
                sysname = source_pipe._get_sysname(info, ds.info['subject'], data_kind)
                adjacency = source_pipe._get_adjacency(data_kind)
                name = 'meg' if data_kind == 'mag' and 'grad' not in sensor_types else data_kind
                if variable_tmax:
                    ys = [load.mne.epochs_ndvar(e, data=data_kind, sysname=sysname, adjacency=adjacency, name=data_kind)[0] for e in epoch_value]
                    if isinstance(data.sensor, str):
                        ys = [getattr(y, data.sensor)('sensor') for y in ys]
                else:
                    ys = load.mne.epochs_ndvar(epoch_value, data=data_kind, sysname=sysname, adjacency=adjacency)
                    if isinstance(data.sensor, str):
                        ys = getattr(ys, data.sensor)('sensor')
                ds[name] = ys
            if ndvar != 'both':
                del ds['epochs']

        if not ctx.view_options['data_raw']:
            ds.info.pop('raw', None)
        return ds


class EvokedDerivative(Derivative[list[mne.Evoked]]):
    """Evoked dataset with cached MNE evoked objects as internal artifact.

    Options
    -------
    baseline
        Baseline correction to apply at load time.
    ndvar
        Whether to convert the returned data to NDVars.
    cat
        Optional subset of model cells to keep.
    data_raw
        Whether to keep the raw object in ``ds.info['raw']``.
    data
        Sensor representation to return.
    samplingrate
        Sampling rate override for the underlying epochs artifact.
    decim
        Decimation override for the underlying epochs artifact.
    """
    name = 'evoked'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count',
    )
    cache_policy = CachePolicy.OPTIONAL
    cache_suffix = '-ave.fif'
    OPTION_DEFAULTS = {'samplingrate': None, 'decim': None}
    VIEW_OPTION_DEFAULTS = {'baseline': False, 'ndvar': True, 'cat': None, 'data_raw': False, 'data': 'sensor'}

    def __init__(self, raw, epochs: dict[str, Any]):
        self.raw = raw
        self.epochs = epochs

    def _epochs_options(self, ctx: Request, *, cat=None, add_bads: bool = True) -> dict[str, Any]:
        return {
            'baseline': True if self.epochs[ctx.state['epoch']].post_baseline_trigger_shift else False,
            'samplingrate': ctx.options['samplingrate'],
            'decim': ctx.options['decim'],
            'interpolate_bads': 'keep',
            'reject': True,
            'cat': cat,
            'trigger_shift': True,
            'ndvar': False,
            'data_raw': False,
            'data': 'sensor',
            'add_bads': add_bads,
        }

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('epochs', options=self._epochs_options(ctx)),)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        fingerprint = self.standard_fingerprint(ctx)
        ds = ctx.load(view='shell')
        model = ctx.state['model']
        if model:
            model_value = ds.eval(model)
            fingerprint['model_signature'] = ctx.registry.canonicalize(model_value)
        else:
            fingerprint['model_signature'] = ds.n_cases
        return fingerprint

    def build(self, ctx: Request) -> list[mne.Evoked]:
        model = ctx.state['model']
        data = ctx.load('epochs')
        data = data.aggregate(
            model,
            never_drop=('epochs',),
            drop_bad=True,
            equal_count=ctx.state['equalize_evoked_count'] == 'eq',
            drop=('sample', 't_edf', 'onset', 'index', 'value'),
        )
        data.rename('epochs', 'evoked')
        model_vars = model.split('%') if model else ()
        for evoked, *cell in data.zip('evoked', *model_vars):
            evoked.info['description'] = "Eelbrain"
            evoked.comment = ' | '.join(cell)
        return data['evoked']

    def load(self, ctx: Request, path: Path) -> list[mne.Evoked]:
        return mne.read_evokeds(path, proj=False)

    def save(self, ctx: Request, path: Path, value: list[mne.Evoked]) -> None:
        mne.write_evokeds(path, value, overwrite=True)

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view is None:
            return self.fingerprint(ctx)
        if view != 'shell':
            raise ValueError(f"{self.name!r} does not define dependency view {view!r}")
        return self.fingerprint(ctx)

    def load_view(self, ctx: Request, view: str):
        if view != 'shell':
            return super().load_view(ctx, view)

        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            dss = []
            for sub_epoch in epoch.collect:
                ds = ctx.load(
                    'evoked',
                    state={'epoch': sub_epoch},
                    options=ctx.options_for('evoked', *self.OPTION_DEFAULTS),
                    view='shell',
                )
                ds[:, 'epoch'] = sub_epoch
                dss.append(ds)
            return combine(dss)

        data = ctx.load('epochs', options=self._epochs_options(ctx), view='shell')
        return data.aggregate(
            ctx.state['model'],
            drop_bad=True,
            equal_count=ctx.state['equalize_evoked_count'] == 'eq',
            drop=('sample', 't_edf', 'onset', 'index', 'value'),
        )

    def apply_view_options(self, ctx: Request, evoked: list[mne.Evoked]) -> Dataset:
        ds = ctx.load(view='shell')
        cat = ctx.view_options['cat']
        if cat:
            ds = ds.sub(ds.eval(ctx.state['model']).isin(cat))
        raw = ds.info.get('raw')
        bads = raw.info['bads'] if raw is not None else ds.info.get(BAD_CHANNELS, [])
        model = ctx.state['model']
        model_vars = model.split('%') if model else ()
        cells = [' | '.join(cell) or 'No comment' for cell in ds.zip(*model_vars)] if model_vars else ['No comment']
        evoked_by_cell = dict(zip(_evoked_comments(evoked), evoked))
        if len(evoked_by_cell) != len(evoked):
            raise RuntimeError(f"Cached evoked data contains duplicate comments: {_evoked_comments(evoked)!r}")
        try:
            evoked = [evoked_by_cell[cell] for cell in cells]
        except KeyError:
            raise RuntimeError(f"Error reading cached evoked: available={tuple(evoked_by_cell)}, requested={tuple(cells)}") from None
        if any(evoked_i.info['bads'] != bads for evoked_i in evoked):
            evoked = [evoked_i.copy() for evoked_i in evoked]
            for evoked_i in evoked:
                evoked_i.info['bads'] = bads
        ds['evoked'] = evoked

        epoch = self.epochs[ctx.state['epoch']]
        baseline = ctx.view_options['baseline']
        if baseline is True:
            baseline = epoch.baseline
        if baseline and not epoch.post_baseline_trigger_shift:
            for evoked_i in ds['evoked']:
                mne.baseline.rescale(evoked_i.data, evoked_i.times, baseline, 'mean', copy=False)

        data = TestDims.coerce(ctx.view_options['data'])
        ndvar = ctx.view_options['ndvar']
        if ndvar:
            evoked = ds['evoked']
            if ndvar == 1:
                del ds['evoked']
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
            source_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            for sensor_type in sensor_types:
                sysname = source_pipe._get_sysname(info, ctx.state['subject'], sensor_type)
                adjacency = source_pipe._get_adjacency(sensor_type)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
                if sensor_type != 'eog' and isinstance(data.sensor, str):
                    ds[name] = getattr(ds[name], data.sensor)('sensor')
        if ctx.view_options['data_raw']:
            ds.info['raw'] = load_raw_dependency(ctx, add_bads=True, preload=False, noise=False)
        elif 'raw' in ds.info:
            del ds.info['raw']
        return ds


class EvokedGroupDatasetDerivative(UncachedDerivative[Dataset]):
    """Group-level sensor evoked dataset assembled from subject datasets.

    Options
    -------
    baseline
        Baseline correction to apply at load time.
    ndvar
        Whether to convert the returned data to NDVars.
    cat
        Optional subset of model cells to keep.
    samplingrate
        Sampling rate override for the underlying evoked artifact.
    decim
        Decimation override for the underlying evoked artifact.
    data_raw
        Whether to keep raw objects in ``ds.info['raw']`` for the subject
        datasets before combining.
    data
        Sensor representation to return.
    """
    name = 'evoked-group-dataset'
    OPTION_DEFAULTS = {'baseline': False, 'ndvar': True, 'cat': None, 'samplingrate': None, 'decim': None, 'data_raw': False, 'data': 'sensor'}

    def __init__(self, raw, groups):
        self.raw = raw
        self.groups = groups

    def key(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize({'subjects': tuple(self.groups[ctx.state['group']]), 'options': ctx.registry.canonicalize(ctx.options)})

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.key(ctx)

    def _subject_options(self, ctx: Request) -> dict[str, Any]:
        return ctx.options_for('evoked', 'baseline', 'cat', 'samplingrate', 'decim', 'data_raw', 'data', ndvar=isinstance(TestDims.coerce(ctx.options['data']).sensor, str))

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = self._subject_options(ctx)
        return tuple(
            Dependency('evoked', label=subject, state={'subject': subject}, options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        dss = [ctx.load(subject) for subject in self.groups[ctx.state['group']]]
        data = TestDims.coerce(ctx.options['data'])
        ndvar = ctx.options['ndvar']
        individual_ndvar = isinstance(data.sensor, str)
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
                    subjects = ', '.join(ds[alens == length, 'subject'].cells)
                    err.append(f"{length}: {subjects}")
                raise DimensionMismatchError('\n'.join(err))
            return ds
        if ndvar and not individual_ndvar:
            evoked = ds['evoked']
            del ds['evoked']
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
            source_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            subject = ds[0, 'subject']
            for sensor_type in sensor_types:
                sysname = source_pipe._get_sysname(info, subject, sensor_type)
                adjacency = source_pipe._get_adjacency(sensor_type)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
                if sensor_type != 'eog' and isinstance(data.sensor, str):
                    ds[name] = getattr(ds[name], data.sensor)('sensor')
        return ds


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
