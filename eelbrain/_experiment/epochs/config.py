# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Epoch definitions.

Each epoch type is a :class:`EpochBase` (a
:class:`~configuration.Configuration`) subclass that the user attaches to
:class:`~pipeline.Pipeline` by name. The graph nodes that extract the
corresponding sensor data live in :mod:`._experiment.epochs.nodes`.
"""

from __future__ import annotations

from typing import Any, Literal
from collections.abc import Iterator, Mapping, Sequence
import math

import numpy as np

from ..._data_obj import Dataset, Var
from ..._exceptions import ConfigurationError
from ..._text import enumeration
from ..configuration import Configuration, typed_arg


EpochBaselineArg = Literal[False] | tuple[float | None, float | None] | None
EPOCH_EXTRACT_OPTIONS = ('samplingrate', 'decim', 'pad', 'tmin', 'tmax', 'tstop')


def _shared_sub_epoch_parameters(name: str, sub_epochs: Sequence[EpochBase], parameters: Sequence[str]) -> dict[str, Any]:
    out = {}
    for param in parameters:
        values = {getattr(sub_epoch, param) for sub_epoch in sub_epochs}
        if len(values) > 1:
            param_repr = ', '.join(repr(v) for v in values)
            raise ConfigurationError(f"Epoch {name}: All sub-epochs must have the same setting for {param}, got {param_repr}")
        out[param] = values.pop()
    return out


def assemble_epochs(epoch_def: Mapping[str, EpochBase], tasks: Sequence[str]) -> dict[str, EpochBase]:
    """Resolve epoch definitions and cache epoch-family dependent parameters.

    This binds each epoch object's ``name`` and lets the epoch classes cache
    deterministic graph-dependent parameters such as inherited epoch
    parameters, ``task``/``tasks``, and ``rej_file_epochs``.
    """
    epochs = {}
    unresolved_epochs = {}
    for name, epoch in epoch_def.items():
        if not isinstance(epoch, EpochBase):
            raise TypeError(f"Epoch {name}: {epoch!r}; need an epoch definition")
        epoch._store_name(name)
        if isinstance(epoch, (PrimaryEpoch, ContinuousEpoch)):
            epoch._store_dependent_parameters(epochs, tasks)
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
            epoch._store_dependent_parameters(epochs, tasks)
            epochs[key] = epoch
        if len(unresolved_epochs) == n:
            raise ConfigurationError(f"Can't resolve epoch dependencies for {enumeration(unresolved_epochs)}")

    return epochs


class EpochBase(Configuration):
    """Base class for epoch definitions."""
    baseline = None
    n_cases = None
    trigger_shift = None
    post_baseline_trigger_shift = None
    decim = None
    _rej_file_epochs_from_name = False
    _needs_task: bool = False
    _tasks = None  # set if tasks is not (task,)
    # Attributes that can contain a str to be evaluated in the events Dataset
    # Used to construct the fingerprint for the events this epoch depends on
    _allowed_eval_attrs: tuple[str, ...] = ('trigger_shift', 'post_baseline_trigger_shift')

    def _prepare_selected_events(
            self,
            ds: Dataset,
            subject: str,
            options: dict[str, Any],
    ) -> Dataset:
        """Prepare the selected-events shell for this epoch.

        Parameters
        ----------
        ds
            Selected-events dataset for one subject after graph-level event
            selection and load options have been applied.
        subject
            Subject identifier used for error messages.
        options
            Selected-events load options that affect epoch extraction.

        Returns
        -------
        Dataset to use as the event shell for epoch extraction.
        Implementations may return a rewritten dataset, for example for
        continuous epochs.
        """
        if ds.n_cases == 0:
            raise RuntimeError(f"No events left for epoch {subject}/{self.name}")

        if self.trigger_shift:
            shift = self.trigger_shift
            if isinstance(shift, str):
                shift = ds.eval(shift)
            if isinstance(shift, Var):
                shift = shift.x
            # Apply
            if np.isscalar(shift):
                ds['sample'] += int(round(shift * ds.info['raw.samplingrate']))
            elif np.isnan(shift).any():
                raise RuntimeError(f"The epoch trigger_shift contains NaNs for {subject}/{self.name}\n{shift=}")
            else:
                ds['sample'] += np.round(shift * ds.info['raw.samplingrate']).astype(int)
        return ds

    def _eval_attrs(self) -> Iterator[str]:
        """Yield settings that must be evaluated in the events dataset."""
        for attr in self._allowed_eval_attrs:
            if isinstance(getattr(self, attr), str):
                yield attr

    def _extraction_parameters(
            self,
            ds: Dataset,
            options: dict[str, Any],
    ) -> tuple[float, Any, float | None, int, bool]:
        """Compute epoch extraction parameters for a prepared shell.

        Parameters
        ----------
        ds
            Prepared event shell returned by :meth:`_prepare_selected_events`.
        options
            `epochs` node options that affect extraction, such as time-window,
            padding, and decimation overrides.

        Returns
        -------
        tmin
            Start of the extraction window in seconds.
        tmax
            End of the extraction window, or a per-epoch :class:`Var`.
        tstop
            Optional explicit stop time for fixed-length extraction.
        decim
            Decimation factor for MNE epoch extraction.
        variable_tmax
            Whether ``tmax`` varies per epoch.
        """
        raise NotImplementedError(f"{self.__class__.__name__}._extraction_parameters()")

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase], tasks: Sequence[str]) -> None:
        """Bind epoch-graph parameters after all epoch names are known."""
        if self._rej_file_epochs_from_name:
            self.rej_file_epochs = (self.name,)
        if self._needs_task:
            if self.task:
                if self.task not in tasks:
                    raise ConfigurationError(f"Unknown task for epoch {self.name!r}: {self.task!r}; should be one of {enumeration(tasks)}")
            elif len(tasks) == 1:
                self.task = tasks[0]
            else:
                raise ConfigurationError(f"Epoch {self.name!r} has no task specified, but multiple tasks exist: {enumeration(tasks)}")

    @property
    def tasks(self) -> tuple[str, ...]:
        """Tasks contributing data to this epoch definition."""
        if self._tasks:
            return self._tasks
        return (self.task,)


class Epoch(EpochBase):
    """Fixed-length, time-locked epoch base (non-functional baseclass)"""
    DICT_ATTRS = ('tmin', 'tmax', 'decim', 'samplingrate', 'baseline', 'trigger_shift', 'post_baseline_trigger_shift', 'post_baseline_trigger_shift_min', 'post_baseline_trigger_shift_max')
    _allowed_eval_attrs = ('tmin', 'tmax', *EpochBase._allowed_eval_attrs)

    # to be set by subclass
    rej_file_epochs = None

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
        """Store and validate common fixed-length epoch parameters."""
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

    def _prepare_selected_events(
            self,
            ds: Dataset,
            subject: str,
            options: dict[str, Any],
    ) -> Dataset:
        """Apply common event preparation and discard out-of-bounds epochs."""
        ds = super()._prepare_selected_events(ds, subject, options)
        return self._trim_to_raw_boundaries(ds, options)

    def _trim_to_raw_boundaries(
            self,
            ds: Dataset,
            options: dict[str, Any],
    ) -> Dataset:
        """Remove events whose requested epoch window exceeds raw bounds."""
        tmin, tmax, tstop, decim, variable_tmax = self._extraction_parameters(ds, options)
        if variable_tmax:
            return ds
        raw_sfreq = ds.info['raw.samplingrate']
        if tmax is None:
            if tstop is None:
                tmax = 0.6
            else:
                sfreq = raw_sfreq / decim
                start_index = int(round(tmin * sfreq))
                stop_index = int(round(tstop * sfreq))
                tmax = tmin + (stop_index - start_index - 1) / sfreq
        elif tstop is not None:
            raise TypeError(f"tmax and tstop can not both be specified at the same time, got tmax={tmax}, tstop={tstop}")
        sample = ds['sample'].x
        i_min = sample + math.floor(tmin * raw_sfreq)
        i_max = sample + math.floor(tmax * raw_sfreq)
        selection = np.flatnonzero((i_min >= ds.info['raw.first_samp']) & (i_max <= ds.info['raw.last_samp']))
        if len(selection) == ds.n_cases and np.array_equal(selection, np.arange(ds.n_cases)):
            return ds
        ds = ds[selection]
        ds.info = ds.info.copy()
        ds.info['epochs.selection'] = selection
        return ds

    def _extraction_parameters(
            self,
            ds: Dataset,
            options: dict[str, Any],
    ) -> tuple[float, Any, float | None, int, bool]:
        """Resolve fixed-length extraction settings with load-time overrides."""
        tmin = self.tmin if options['tmin'] is None else options['tmin']
        tmax = options['tmax']
        tstop = options['tstop']
        if tmax is None and tstop is None:
            tmax = self.tmax
        if isinstance(tmax, str):
            tmax = ds.eval(tmax)
            assert isinstance(tmax, Var)
            assert not self.post_baseline_trigger_shift, 'not implemented with variable tmax'
            variable_tmax = True
        else:
            variable_tmax = False
        if pad := options['pad']:
            tmin -= pad
            if tmax is not None:
                tmax = tmax + pad
            elif tstop is not None:
                tstop = tstop + pad
        decim = decim_param(options['samplingrate'], options['decim'], self, ds.info)
        return tmin, tmax, tstop, decim, variable_tmax


class PrimaryEpoch(Epoch):
    """Epoch based on selecting events from a raw file

    Parameters
    ----------
    task
        Task from which to load data.
        Can be omitted if the experiment has only a single task.
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
        ``200`` for data sampled at 1000 Hz; by default, use the raw data
        samplingrate).
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
        shift (in seconds) for each epoch.
        Typically, this parameter is defined on a :class:`SecondaryEpoch`, such that
        trial rejection can be performed on a larger :class:`PrimaryEpoch` that
        encompasses the baseline as well as the target time window.
        If the ``post_baseline_trigger_shift`` parameter is specified, the parameters
        ``post_baseline_trigger_shift_min`` and ``post_baseline_trigger_shift_max``
        are also needed, specifying the smallest and largest possible shift. These
        are used to crop the resulting epochs appropriately, to the region from
        ``new_tmin = epoch['tmin'] - post_baseline_trigger_shift_min`` to
        ``new_tmax = epoch['tmax'] - post_baseline_trigger_shift_max``.
    n_cases
        Expected number of epochs. If n_cases is defined, a ``RuntimeError``
        will be raised whenever the actual number of matching events is different.
    run
        Restrict the epoch to a specific run. By default (``None``), events are
        combined across all available runs for the given task.

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
    DICT_ATTRS = Epoch.DICT_ATTRS + ('task', 'run', 'sel',)
    _rej_file_epochs_from_name = True
    _needs_task = True

    def __init__(
            self,
            task: str = None,
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
            run: str | None = None,
    ):
        super().__init__(tmin, tmax, samplingrate, decim, baseline, trigger_shift, post_baseline_trigger_shift, post_baseline_trigger_shift_min, post_baseline_trigger_shift_max)
        self.task = task
        self.run = typed_arg(run, str)
        self.sel = typed_arg(sel, str)
        self.n_cases = typed_arg(n_cases, int)


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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase], tasks: Sequence[str]) -> None:
        base = epochs[self.sel_epoch]
        if not isinstance(base, (PrimaryEpoch, SecondaryEpoch)):
            raise ConfigurationError(f"Epoch {self.name}, base={self.sel_epoch!r}: is {base.__class__.__name__}, needs to be PrimaryEpoch or SecondaryEpoch")
        params = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            params.setdefault(param, getattr(base, param))
        Epoch.__init__(self, **params)
        self.rej_file_epochs = base.rej_file_epochs
        self.task = base.task


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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase], tasks: Sequence[str]) -> None:
        sub_epochs = [epochs[sub_epoch] for sub_epoch in self.sub_epochs]
        for sub_epoch in sub_epochs:
            if isinstance(sub_epoch, SuperEpoch):
                raise ConfigurationError(f"Epoch {self.name}: SuperEpochs can not be defined recursively")
            if not isinstance(sub_epoch, Epoch):
                raise ConfigurationError(f"Epoch {self.name}: sub-epochs must all by PrimaryEpochs")
            if sub_epoch.post_baseline_trigger_shift is not None:
                raise ConfigurationError(f"Epoch {self.name}: Super-epochs are merged on the level of events and can't contain epochs with post_baseline_trigger_shift")
        params = self._kwargs.copy()
        # Only check sub-epoch agreement for params that are not explicitly overridden
        non_overridden = [p for p in self.INHERITED_PARAMS if p not in params]
        params.update(_shared_sub_epoch_parameters(self.name, sub_epochs, non_overridden))
        Epoch.__init__(self, **params)
        # Record which params were explicitly overridden; _kwargs is kept for idempotent re-resolution
        self._explicit_params = tuple(self._kwargs)
        self._tasks = tuple(sorted({sub_epoch.task for sub_epoch in sub_epochs}))
        self.rej_file_epochs = [epoch_name for sub_epoch in sub_epochs for epoch_name in sub_epoch.rej_file_epochs]

    def _trim_to_raw_boundaries(self, ds: Dataset, options: dict[str, Any]) -> Dataset:
        return ds


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

    def _store_dependent_parameters(self, epochs: Mapping[str, EpochBase], tasks: Sequence[str]) -> None:
        sub_epochs = [epochs[sub_epoch] for sub_epoch in self.collect]
        for param, value in _shared_sub_epoch_parameters(self.name, sub_epochs, SuperEpoch.INHERITED_PARAMS).items():
            setattr(self, param, value)
        self._tasks = tuple(sorted({task for sub_epoch in sub_epochs for task in sub_epoch.tasks}))
        self.rej_file_epochs = sorted({epoch_name for sub_epoch in sub_epochs for epoch_name in sub_epoch.rej_file_epochs})


class ContinuousEpoch(EpochBase):
    """Epoch spanning multiple events for continuous analysis

    A :class:`ContinuousEpoch` will extract a continuous segment of data from
    the first event to the last event. ``pad_start`` and ``pad_stop`` determine
    how much extra time to include before the first event and after the last
    event (to allow using the data surrounding these events for estimating TRFs
    with negative and positive lags). ``split`` controls whether to break up the
    data into multiple segments when there are long pauses between successive
    events.

    When using :meth:`Pipeline.load_epochs`, each row of the returned
    :class:`Dataset` will contain the events in the epoch alongside the data.

    Parameters
    ----------
    task
        Task from which to load data.
        Can be omitted if the experiment has only a single task.
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
        ``200`` for data sampled at 1000 Hz; by default, use the raw data
        samplingrate).
    """
    DICT_ATTRS = ('task', 'sel', 'pad_start', 'pad_end', 'split', 'samplingrate')
    _rej_file_epochs_from_name = True
    _needs_task = True

    def __init__(
            self,
            task: str = None,
            sel: str = None,
            pad_start: float = 0.100,
            pad_end: float = 1.000,
            split: float = 10,
            samplingrate: float = None,
    ):
        self.task = typed_arg(task, str)
        self.sel = typed_arg(sel, str)
        self.pad_start = typed_arg(pad_start, float)
        self.pad_end = typed_arg(pad_end, float)
        self.split = typed_arg(split, float)
        self.samplingrate = typed_arg(samplingrate, float, int)

    def _prepare_selected_events(
            self,
            ds: Dataset,
            subject: str,
            options: dict[str, Any],
    ) -> Dataset:
        ds = super()._prepare_selected_events(ds, subject, options)

        split_threshold = self.split + self.pad_start + self.pad_end
        onsets = np.flatnonzero(ds['onset'].diff(to_begin=split_threshold + 1) >= split_threshold)
        illegal = {'T_relative', 'events', 'tmax'}.intersection(ds)
        if illegal:
            raise RuntimeError(f"Events contain variables with reserved names: {', '.join(illegal)}")
        events = [ds[i1:i2] for i1, i2 in zip(onsets, [*onsets[1:], None])]
        raw_samplingrate = ds.info['raw.samplingrate']
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
    ) -> tuple[float, Any, float | None, int, bool]:
        decim = decim_param(options['samplingrate'], options['decim'], self, ds.info)
        return -self.pad_start, ds.eval('tmax'), None, decim, True


def decim_param(
        samplingrate: int,
        decim: int,
        epoch: Epoch | None,
        info: dict,
        minimal: bool = False,  # try to infer minimally necessary samplingrate
) -> int:
    raw_samplingrate = info['raw.samplingrate'] if 'raw.samplingrate' in info else info['sfreq']
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
        decim_ratio = raw_samplingrate / samplingrate
        rounded_decim_ratio = round(decim_ratio)
        if not math.isclose(decim_ratio, rounded_decim_ratio, rel_tol=1e-3):
            raise ValueError(f"{samplingrate=} with data at {raw_samplingrate:g} Hz: needs to be integer ratio")
        return rounded_decim_ratio

    if minimal:
        if h_freq := info.get('lowpass'):
            return int(raw_samplingrate / (h_freq * 2.5))
        else:
            return int(raw_samplingrate / 100)

    return 1


def single_recording_run(epochs: Mapping[str, EpochBase], epoch: EpochBase) -> str:
    """Run value for an epoch that wraps a single recording.

    Used to pin the ``run`` of a per-recording dependency when an aggregating
    node (``epochs``/``epoch-events``) is not combining across multiple runs, so
    the pinned value comes from the epoch definition rather than ambient state.
    Returns the :class:`PrimaryEpoch` ``run`` (following ``SecondaryEpoch``
    bases), or ``''`` when the experiment has no run entity.

    Parameters
    ----------
    epochs
        All epoch definitions, to resolve ``SecondaryEpoch`` bases.
    epoch
        The epoch being resolved.
    """
    while isinstance(epoch, SecondaryEpoch):
        epoch = epochs[epoch.sel_epoch]
    return getattr(epoch, 'run', None) or ''
