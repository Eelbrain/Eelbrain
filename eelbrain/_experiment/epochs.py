# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from copy import deepcopy
import inspect
from typing import Optional

from .._exceptions import DefinitionError
from .._text import enumeration
from .definitions import Definition, typed_arg


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
            raise DefinitionError(f"Can't resolve epoch dependencies for {enumeration(secondary_epochs)}")
    return epochs


class EpochBase(Definition):
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
    sessions = None

    def __init__(self, tmin=-0.1, tmax=0.6, samplingrate=None, decim=None, baseline=None,
                 vars=None, trigger_shift=0., post_baseline_trigger_shift=None,
                 post_baseline_trigger_shift_min=None,
                 post_baseline_trigger_shift_max=None):
        if post_baseline_trigger_shift is not None:
            if post_baseline_trigger_shift_min is None or post_baseline_trigger_shift_max is None:
                raise DefinitionError(f"post_baseline_trigger_shift={post_baseline_trigger_shift} but missing post_baseline_trigger_shift_min and/or post_baseline_trigger_shift_max")
            cut_time = post_baseline_trigger_shift_max - post_baseline_trigger_shift_min
            if not isinstance(tmax, str) and cut_time >= tmax - tmin:
                raise DefinitionError("No data remaining after trigger shift")

        if decim is not None:
            if decim < 1:
                raise ValueError(f"decim={decim!r}")
            elif samplingrate is not None:
                raise TypeError(f"deimc={decim} with samplingrate={samplingrate}: only one of these parameters can be specified at a time")
        elif samplingrate is not None:
            if samplingrate <= 0:
                raise ValueError(f"samplingrate={samplingrate!r}")
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
            raise ValueError(f"baseline={baseline!r}: needs to be length 2 tuple")
        else:
            baseline = (typed_arg(baseline[0], float), typed_arg(baseline[1], float))

        if not isinstance(trigger_shift, (float, str)):
            raise TypeError(f"trigger_shift={trigger_shift!r}: needs to be float or str")

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

    def _as_dict_24(self):
        "Dict to be compared with Eelbrain 0.24 cache"
        out = {k: v for k, v in self._as_dict().items() if v is not None}
        if isinstance(self, (SecondaryEpoch, SuperEpoch)):
            out['_rej_file_epochs'] = self.rej_file_epochs
        if isinstance(self, PrimaryEpoch) and self.n_cases:
            out['n_cases'] = self.n_cases
        if out['trigger_shift'] == 0:
            del out['trigger_shift']
        return out


class PrimaryEpoch(Epoch):
    """Epoch based on selecting events from a raw file

    Parameters
    ----------
    session : str
        Session (raw file) from which to load data.
    sel : str
        Expression which evaluates in the events Dataset to the index of the
        events included in this Epoch specification.
    tmin : float
        Start of the epoch (default -0.1).
    tmax : float
        End of the epoch (default 0.6).
    samplingrate : scalar
        Target samplingrate. Needs to divide data samplingrate evenly (e.g.
        ``200`` for data sampled at 1000 Hz; default ``200``).
    decim : int
        Alternative to ``samplingrate``. Decimate the data by this factor
        (i.e., only keep every ``decim``'th sample).
    baseline : tuple
        The baseline of the epoch (default ``(None, 0)``; if ``tmin > 0``: no
        baseline; if ``tmax < 0``: the whole interval).
    n_cases : int
        Expected number of epochs. If n_cases is defined, a RuntimeError error
        will be raised whenever the actual number of matching events is different.
    trigger_shift : float | str
        Shift event triggers before extracting the data [in seconds]. Can be a
        float to shift all triggers by the same value, or a str indicating an event
        variable that specifies the trigger shift for each trigger separately.
        The ``trigger_shift`` applied after loading selected events.
        For secondary epochs the ``trigger_shift`` is applied additively with the
        ``trigger_shift`` of their base epoch.
    post_baseline_trigger_shift : str
        Shift the trigger (i.e., where epoch time = 0) after baseline correction.
        The value of this entry has to be the name of an event variable providing
        for each epoch the actual amount of time shift (in seconds). If the
        ``post_baseline_trigger_shift`` parameter is specified, the parameters
        ``post_baseline_trigger_shift_min`` and ``post_baseline_trigger_shift_max``
        are also needed, specifying the smallest and largest possible shift. These
        are used to crop the resulting epochs appropriately, to the region from
        ``new_tmin = epoch['tmin'] - post_baseline_trigger_shift_min`` to
        ``new_tmax = epoch['tmax'] - post_baseline_trigger_shift_max``.
    vars : dict
        Add new variables only for this epoch.
        Each entry specifies a variable with the following schema:
        ``{name: definition}``. ``definition`` can be either a string that is
        evaluated in the events-Dataset`, or a
        ``(source_name, {value: code})``-tuple.
        ``source_name`` can also be an interaction, in which case cells are joined
        with spaces (``"f1_cell f2_cell"``).

    See Also
    --------
    MneExperiment.epochs

    Examples
    --------
    Selecting events based on a categorial label::

        PrimaryEpoch('session', "variable == 'label'")

    Based on multiple categorial labels::

        PrimaryEpoch('session', "variable.isin(['label1', 'label2'])")

    Based on multiple categorial variables::

        PrimaryEpoch('session', "(variable == 'label') & (other_variable == 'other_label)")

    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sel',)

    def __init__(self, session, sel=None, **kwargs):
        n_cases = kwargs.pop('n_cases', None)
        Epoch.__init__(self, **kwargs)
        self.session = session
        self.sel = typed_arg(sel, str)
        self.n_cases = typed_arg(n_cases, int)
        self.sessions = (session,)

    def _repr_args(self):
        args = [repr(self.session)]
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
    a subset of the events in the base epoch.

    Parameters
    ----------
    base : str
        Name of the epoch whose parameters provide defaults for all parameters.
        Additional parameters override parameters of the ``base`` epoch, with the
        exception of ``trigger_shift``, which is applied additively to the
        ``trigger_shift`` of the ``base`` epoch.
    ...
        Override base-epoch parameters (see :class:`PrimaryEpoch`).

    See Also
    --------
    MneExperiment.epochs
    """
    DICT_ATTRS = Epoch.DICT_ATTRS + ('sel_epoch', 'sel')
    INHERITED_PARAMS = ('tmin', 'tmax', 'decim', 'samplingrate', 'baseline', 'post_baseline_trigger_shift', 'post_baseline_trigger_shift_min', 'post_baseline_trigger_shift_max')

    def __init__(self, base, sel=None, **kwargs):
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
            raise DefinitionError(f"Epoch {name}, base={self.sel_epoch!r}: is {base.__class__.__name__}, needs to be PrimaryEpoch or SecondaryEpoch")
        kwargs = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            if param not in kwargs:
                kwargs[param] = getattr(base, param)
        out = Epoch._link(self, name, epochs)
        Epoch.__init__(out, **kwargs)
        out.rej_file_epochs = base.rej_file_epochs
        out.session = base.session
        out.sessions = base.sessions
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
    MneExperiment.epochs
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
                raise DefinitionError(f"Epoch {name}: SuperEpochs can not be defined recursively")
            elif not isinstance(e, Epoch):
                raise DefinitionError(f"Epoch {name}: sub-epochs must all by PrimaryEpochs")
            elif e.post_baseline_trigger_shift is not None:
                raise DefinitionError(f"Epoch {name}: Super-epochs are merged on the level of events and can't contain epochs with post_baseline_trigger_shift")
        # find inherited epoch parameters
        kwargs = self._kwargs.copy()
        for param in self.INHERITED_PARAMS:
            if param in kwargs:
                continue
            values = {getattr(e, param) for e in sub_epochs}
            if len(values) > 1:
                param_repr = ', '.join(repr(v) for v in values)
                raise DefinitionError(f"Epoch {name}: All sub_epochs must have the same setting for {param}, got {param_repr}")
            kwargs[param] = values.pop()
        out = Epoch._link(self, name, epochs)
        Epoch.__init__(out, **kwargs)
        # sessions, with preserved order
        out.sessions = []
        out.rej_file_epochs = []
        for e in sub_epochs:
            if e.session not in out.sessions:
                out.sessions.append(e.session)
            out.rej_file_epochs.extend(e.rej_file_epochs)
        return out


class EpochCollection(EpochBase):
    """A collection of epochs that are loaded separately.

    For TRFs, a separate TRF will be estimated for each collected epoch (as
    opposed to a SuperEpoch, for which sub-epochs will be merged before estimating
    a single TRF).

    Parameters
    ----------
    collect : Sequence of str
        Epochs to collect.

    See Also
    --------
    MneExperiment.epochs
    """
    # IMPLEMENTATION ALTERNATIVE?
    # ---------------------------
    # In analogy to standard epochs, the "model" parameter could be used to fit
    # a separate TRF per cell.
    #
    #  - Logistic complication: I would want to be able to fit only cell 1
    #    first, and later fit cell 2, without redundant refitting.
    DICT_ATTRS = ('collect',)

    def __init__(self, collect):
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
                raise DefinitionError(f"Epoch {name}: All sub-epochs must have the same setting for {param}, got {param_repr}")
            setattr(out, param, values.pop())
        # dependencies
        sessions = set()
        rej_file_epochs = set()
        for e in sub_epochs:
            sessions.update(e.sessions)
            rej_file_epochs.update(e.rej_file_epochs)
        out.sessions = sorted(sessions)
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

    When using :meth:`MneExperiment.load_epochs`, each row of the returned
    :class:`Dataset` will contain the events in the epoch alongside the data.

    Parameters
    ----------
    session
        Session (raw file) from which to load data.
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
    DICT_ATTRS = ('name', 'session', 'sel', 'pad_start', 'pad_end', 'split', 'samplingrate', 'vars')

    def __init__(
            self,
            session: str,
            sel: str = None,
            pad_start: float = 0.100,
            pad_end: float = 1.000,
            split: float = 10,
            samplingrate: float = 200,
            vars: dict = None,
    ):
        EpochBase.__init__(self)
        self.session = typed_arg(session, str)
        self.sel = typed_arg(sel, str)
        self.pad_start = typed_arg(pad_start, float)
        self.pad_end = typed_arg(pad_end, float)
        self.split = typed_arg(split, float)
        self.samplingrate = typed_arg(samplingrate, float, int)
        self.vars = vars
        self.sessions = (session,)


def decim_param(
        samplingrate: int,
        decim: int,
        epoch: Optional[Epoch],
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
        if decim_ratio % 1:
            raise ValueError(f"{samplingrate=} with data at {info['sfreq']:g} Hz: needs to be integer ratio")
        return int(decim_ratio)

    if minimal:
        if h_freq := info.get('lowpass'):
            return int(info['sfreq'] / (h_freq * 2.5))
        else:
            return int(info['sfreq'] / 100)

    return 1
