# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections.abc import Mapping
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import numpy

from ..._data_obj import Dataset, Factor, NDVar, UTS, Var, combine
from ..._ndvar import resample
from ..._ndvar.uts import pad
from ..._trf._predictors import epoch_impulse_predictor, event_impulse_predictor
from ..configuration import Configuration, typed_arg
from ..pathing import PREDICTOR_DIR, subject_predictor_path
from .model import Term, TRFModelError


def t_stop_ds(ds: Dataset, t: float):
    "Dummy-event for the end of the last step"
    t_stop = ds.info['tstop'] + t
    out = {}
    for k, v in ds.items():
        if k == 'time':
            out['time'] = Var([t_stop])
        elif isinstance(v, Var):
            out[k] = Var(numpy.asarray([0], v.x.dtype))
        elif isinstance(v, Factor):
            out[k] = Factor([''])
        else:
            raise ValueError(f"{k!r} in predictor: {v!r}")
    return Dataset(out)


class EventPredictor(Configuration):
    """Generate an impulse for each event

    For epoched data, one impulse per epoch;
    for continuous data, one impulse per event in the event list.

    Parameters
    ----------
    value
        Name of a :class:`Var` or :class:`Factor` in the events :class:`Dataset`
        (or expression resulting in one).
    latency
        Latency of the impulse relative to the event in seconds (or expression
        retrieving it from the events dataset).
    sel
        Subset of events.
    """
    DICT_ATTRS = ('value', 'latency', 'sel')

    def __init__(
            self,
            value: float | str = 1.,
            latency: float | str = 0.,
            sel: str = None,
    ):
        self.value = typed_arg(value, float, str)
        self.latency = typed_arg(latency, float, str)
        self.sel = typed_arg(sel, str)

    def _generate(self, uts: UTS, ds: Dataset, term: Term):
        assert term.stimulus is None
        if self.sel:
            raise NotImplementedError
        return epoch_impulse_predictor((ds.n_cases, uts), self.value, self.latency, term.string, ds)

    def _generate_continuous(self, uts: UTS, events: Dataset, term: Term) -> NDVar:
        "Impulse for each event in one ContinuousEpoch segment, placed at ``epoch_time``"
        assert term.stimulus is None
        if self.sel:
            events = events.sub(self.sel)
        return event_impulse_predictor(uts, 'epoch_time', self.value, self.latency, term.code, events)


class FilePredictorBase(Configuration):
    """Base class for predictors stored in files corresponding to specific stimuli

    Use :class:`UTSPredictor` for predictors stored as uniform time series
    (:class:`NDVar`) and :class:`NUTSPredictor` for predictors stored as
    non-uniform time series (:class:`Dataset`).
    """
    DICT_ATTRS = ('resample', 'sampling')
    # State fields that select the predictor file (stimulus-based by default)
    _key_fields: tuple[str, ...] = ()

    def _path(self, term: Term, state: Mapping[str, Any], root: Path) -> Path:
        "Absolute path of the predictor file backing ``term``"
        return root / PREDICTOR_DIR / f"{self._file_stem(term)}.pickle"

    def _resample(self, x: NDVar, tstep: float = None):
        if tstep is None or x.time.tstep == tstep:
            pass
        elif x.time.tstep > tstep:
            raise ValueError(f"Requested samplingrate rate is higher than in file ({1 / tstep:g} > {1 / x.time.tstep:g})")
        elif self.resample == 'bin':
            x = x.bin(tstep, label='start')
        elif self.resample == 'resample':
            srate = 1 / tstep
            int_srate = int(round(srate))
            srate = int_srate if abs(int_srate - srate) < .001 else srate
            x = resample(x, srate)
        elif self.resample is None:
            raise RuntimeError(f"{x.name} has tstep={x.time.tstep}, not {tstep}. Set the {self.__class__.__name__} resample parameter to enable automatic resampling.")
        else:
            raise RuntimeError(f"{self.resample=}")
        return x


def _arrays_equal(a: numpy.ndarray, b: numpy.ndarray) -> bool:
    "Exact array equality; NaN counts as equal to itself"
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in 'fc':
        return bool(numpy.array_equal(a, b, equal_nan=True))
    return bool(numpy.array_equal(a, b))


def _columns_equal(a: Var | Factor, b: Var | Factor) -> bool:
    "Exact equality for one Dataset column"
    if type(a) is not type(b):
        return False
    if isinstance(a, Factor):
        return len(a) == len(b) and all(ai == bi for ai, bi in zip(a, b))
    return _arrays_equal(a.x, b.x)


class UTSPredictor(FilePredictorBase):
    """Uniform time series predictor, stored as :class:`NDVar` files

    Parameters
    ----------
    resample
        How to resample the predictor when an analysis is done at a lower
        sampling rate than the stored :class:`NDVar`:

         - ``bin``: averaging the values in time bins
         - ``resample``: use appropriate filter followed by decimation

        For predictors with non-continuous information, such as impulses,
        binning is more appropriate.
    sampling
        Whether the predictor is continuous or discrete. Used to decide
        whether to filter this predictor with ``filter_x='continuous'``
        (default ``'continuous'``).

    Notes
    -----
    UTS predictors are stored as :class:`NDVar` objects with time axis
    matching the data.

    Predictor files are expected for each stimulus at::

        {root}/derivatives/predictors/{stimulus}~{key}[-...].pickle

    Where ``stimulus`` refers to the name provided by ``stim_var`` and ``key``
    refers to the predictor's name (the key used in
    :attr:`TRFExperiment.predictors`). Tags starting with a dash (``-``)
    following the ``key`` can be used to distinguish different versions of
    a given preditor (``{stimulus}~{key}-{variant}``).
    """
    DICT_ATTRS = ('resample', 'sampling')

    def __init__(
            self,
            resample: Literal['bin', 'resample'] = None,
            sampling: Literal['continuous', 'discrete'] = 'continuous',
    ):
        assert resample in (None, 'bin', 'resample')
        assert sampling in ('continuous', 'discrete')
        self.resample = resample
        self.sampling = sampling

    def _file_stem(self, term: Term) -> str:
        "File name (without extension) of the predictor file backing ``term``"
        return term.uts_file_name

    def _reference_stem(self, term: Term, state: Mapping[str, Any]) -> str:
        "Identifier for the cache-internal reference copy of ``term``'s relevant data"
        return self._file_stem(term)

    def _prepare(self, x: NDVar, tstep: float) -> NDVar:
        "Resample the raw file contents to ``tstep``"
        if not isinstance(x, NDVar):
            raise TypeError(f"UTSPredictor file must contain an NDVar, contains {x!r}")
        return self._resample(x, tstep)

    def _relevant_data(self, contents: NDVar, term: Term) -> NDVar:
        "The subset of the file contents that actually feeds the predictor (the whole NDVar)"
        if not isinstance(contents, NDVar):
            raise TypeError(f"UTSPredictor file must contain an NDVar, contains {contents!r}")
        return contents

    def _data_equal(self, stored: NDVar, current: NDVar) -> bool:
        "Exact comparison of two versions of the relevant data"
        return isinstance(stored, NDVar) and stored.dims == current.dims and _arrays_equal(stored.x, current.x)

    def _generate(self, x: NDVar, tmin: float, tstep: float, n_samples: int, term: Term) -> NDVar:
        # build the predictor for one input file from its raw (unpickled) contents
        x = self._prepare(x, tstep)
        if term.nuts_method:
            raise TRFModelError(f"{term.string}: suffix {term.nuts_method} reserved for non-uniform time series predictors")
        x = pad(x, tmin, nsamples=n_samples, set_tmin=True)
        x.info['sampling'] = self.sampling
        return x

    def _prepare_stimulus(self, contents: NDVar, tstep: float) -> NDVar:
        "One stimulus' relevant data, resampled to ``tstep`` for continuous placement"
        return self._prepare(contents, tstep)

    def _generate_continuous(self, uts: UTS, events: Dataset, stim_var: str, term: Term, cache: dict) -> NDVar:
        "Place per-stimulus predictors into a continuous segment at their ``epoch_time``"
        if term.nuts_method:
            raise TRFModelError(f"{term.string}: suffix {term.nuts_method} reserved for non-uniform time series predictors")
        v = cache[events[0, stim_var]]
        dimnames = v.get_dimnames(first='time')
        dims = (uts, *v.get_dims(dimnames[1:]))
        x = NDVar.zeros(dims, term.key)
        for t, stim in events.zip('epoch_time', stim_var):
            x_stim = cache[stim]
            i_start = uts._array_index(t + x_stim.time.tmin)
            i_stop = i_start + len(x_stim.time)
            if i_stop > len(uts):
                raise ValueError(f"{term.string} for {stim} is longer than the data")
            x.x[i_start:i_stop] = x_stim.get_data(dimnames)
        x.info['sampling'] = self.sampling
        return x


class NUTSPredictor(FilePredictorBase):
    """Non-uniform time series predictor, stored as :class:`Dataset` files

    NUTS predictors are specified as :class:`Dataset` objects with a ``time``
    column (time stamp of each event in seconds) and further columns with
    event values. When loading a predictor, they are converted to uniform time
    series by placing impulses at the time stamps. The columns to use are
    specified in the model term, as ``{key}-{value-column}`` or
    ``{key}-{value-column}-{mask-column}`` (the boolean mask column sets
    ``value`` to zero wherever it is ``False``). The term ``{key}`` alone
    invokes an intercept, i.e. a value of 1 at each time point.

    Notes
    -----
    See :class:`FilePredictorBase` for the predictor file location.

    Examples
    --------
    Assume a :class:`Dataset` stored at ``predictors/story~word.pickle``, etc.,
    with the following columns:

    - ``time``, indicating the word's onset time
    - ``frequency``, the word frequency
    - ``surprisal``, how surprising the word is in its context
    - ``noun``, ``True`` if the word is a noun, ``False`` otherwise

    This could be added to the experiment as follows::

        predictors = {
            'word': NUTSPredictor(),
        }

    With this predictor, the following terms could be used for TRF models:

    - ``word``: Unit size impulse at every word onset
    - ``word-frequency``: An impulse at each word onset reflecting the word's frequency
    - ``word-frequency-noun``: An impulse at each noun's onset reflecting the noun's frequency

    These terms in turn could be used to construct the following model::

        experiment.load_trfs(x="word + word-frequency + word-surprisal")

    """
    DICT_ATTRS = ()

    def _file_stem(self, term: Term) -> str:
        "File name (without extension) of the predictor file backing ``term``"
        return term.nuts_file_name

    def _reference_stem(self, term: Term, state: Mapping[str, Any]) -> str:
        "Identifier for the cache-internal reference copy of ``term``'s relevant data"
        # stimulus~file-column[-mask]
        return term.string_without_nuts_method

    def _sampling(self, nuts_method: str = None) -> Literal['continuous', 'discrete'] | None:
        if nuts_method == 'step':
            return 'continuous'
        elif nuts_method is None:
            return 'discrete'
        else:
            raise RuntimeError(f'{nuts_method=}')

    def _relevant_data(self, contents: Dataset, term: Term) -> Dataset:
        "The subset of the file contents that actually feeds the predictor"
        if not isinstance(contents, Dataset):
            raise TypeError(f"NUTSPredictor file must contain a Dataset, contains {contents!r}")
        column_key, mask_key = term.nuts_columns
        keys = ['time']
        for key in (column_key, mask_key):
            if key is not None and key in contents:
                keys.append(key)
        out = contents[keys]
        if 'tstop' in contents.info:
            out.info['tstop'] = contents.info['tstop']
        return out

    def _data_equal(self, stored: Dataset, current: Dataset) -> bool:
        "Exact comparison of two versions of the relevant data"
        if not isinstance(stored, Dataset) or set(stored.keys()) != set(current.keys()):
            return False
        if stored.info.get('tstop') != current.info.get('tstop'):
            return False
        return all(_columns_equal(stored[key], current[key]) for key in current)

    def _generate(self, x: Dataset, tmin: float, tstep: float, n_samples: int, term: Term) -> NDVar:
        # build the predictor for one input file from its raw (unpickled) contents
        if tmin is None:
            tmin = 0
        if tstep is None:
            tstep = 0.001
        if n_samples is None:
            if 'tstop' in x.info:
                tstop = x.info['tstop']
            else:
                tstop = x[-1, 'time'] + 0.5
            n_samples = int((tstop - tmin) // tstep)
        uts = UTS(tmin, tstep, n_samples)
        x = self._ds_to_ndvar(x, uts, term)
        x.info['sampling'] = self._sampling(term.nuts_method)
        return x

    def _prepare_stimulus(self, contents: Dataset, tstep: float) -> Dataset:
        "One stimulus' relevant data (resampling happens later at the segment's ``uts``)"
        return contents

    def _generate_continuous(self, uts: UTS, events: Dataset, stim_var: str, term: Term, cache: dict) -> NDVar:
        "Place per-stimulus event tables into a continuous segment at their ``epoch_time``"
        dss = []
        for t, stim in events.zip('epoch_time', stim_var):
            x = cache[stim].copy()
            x['time'] += t
            dss.append(x)
            if term.nuts_method:
                dss.append(t_stop_ds(x, t))
        x = self._ds_to_ndvar(combine(dss), uts, term)
        x.info['sampling'] = self._sampling(term.nuts_method)
        return x

    def _ds_to_ndvar(self, ds: Dataset, uts: UTS, term: Term):
        column_key, mask_key = term.nuts_columns
        if column_key is None:
            column_key = 'value'
            ds[:, column_key] = 1

        if mask_key:
            mask = ds[mask_key].x
            assert mask.dtype.kind == 'b', "'mask' must be boolean"
        else:
            mask = None

        if mask is not None:
            ds[column_key] *= mask

        # prepare output NDVar
        x = NDVar.zeros(uts, name=term.key)

        # fill in values
        dt = uts.tstep / 2
        ds = ds[(ds['time'] > uts.tmin - dt) & (ds['time'] < uts.tmax + dt)]
        if term.nuts_method is None:
            for t, v in ds.zip('time', column_key):
                x[t] += v
        elif term.nuts_method == 'step':
            t_stops = ds[1:, 'time']
            if ds[-1, column_key] != 0:
                if 'tstop' not in ds.info:
                    raise TRFModelError(f"{term.string}: for step representation, the predictor datasets needs to contain ds.info['tstop'] to determine the end of the last step")
                t_stops = chain(t_stops, [ds.info['tstop']])
            for t0, t1, v in zip(ds['time'], t_stops, ds[column_key]):
                x[t0:t1] = v
        else:
            raise TRFModelError(f"{term.string}: NUTS-method={term.nuts_method!r}")
        return x


class SubjectUTSPredictor(UTSPredictor):
    """Subject-specific uniform time series predictor

    Parameters
    ----------
    resample
        See :class:`UTSPredictor`.
    sampling
        See :class:`UTSPredictor`.
    per_event
        How to model a :class:`ContinuousEpoch` that contains multiple events:

         - ``False`` (default): the predictor is a single time series spanning
           the whole recording (one file per recording), modeling the
           subject-specific response to the entire *sequence*. For a
           :class:`ContinuousEpoch`, the predictor time axis matches
           ``epoch_time`` (zero at the first selected event), and each segment
           is cut out directly at its position on that axis.
         - ``True``: the predictor is placed per event, exactly like a
           :class:`UTSPredictor`, but from subject-specific files. These files
           are keyed by subject, session, acquisition, and stimulus and are
           shared across task/run recordings.

    Notes
    -----
    In contrast to a :class:`UTSPredictor`, which represents a specific stimulus
    and is shared across subjects, a :class:`SubjectUTSPredictor` provides a
    separate predictor file for each recording in sequence mode. These files
    are identified by the ``subject``, ``session``, ``task``, ``acquisition``,
    and ``run`` BIDS entities.

    With ``per_event=False`` the file for a term is expected at::

        {root}/derivatives/subject-predictors/sub-{subject}[/ses-{session}]/sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_run-{run}]_desc-{code}.pickle

    and the term cannot be combined with a stimulus. With ``per_event=True``,
    task and run do not select the predictor file, and the file for each
    stimulus is expected at::

        {root}/derivatives/subject-predictors/sub-{subject}[/ses-{session}]/sub-{subject}[_ses-{session}][_acq-{acquisition}]_desc-{stimulus}~{code}.pickle
    """
    DICT_ATTRS = ('resample', 'sampling', 'per_event')

    def __init__(
            self,
            resample: Literal['bin', 'resample'] = None,
            sampling: Literal['continuous', 'discrete'] = 'continuous',
            per_event: bool = False,
    ):
        super().__init__(resample, sampling)
        self.per_event = per_event
        if per_event:
            self._key_fields = ('subject', 'session', 'acquisition')
        else:
            self._key_fields = ('subject', 'session', 'task', 'acquisition', 'run')

    def _prepare_sequence(self, x: NDVar, tstep: float, term: Term) -> NDVar:
        "Prepare a recording-long sequence predictor"
        if term.nuts_method:
            raise TRFModelError(f"{term.string}: suffix {term.nuts_method} reserved for non-uniform time series predictors")
        x = self._prepare(x, tstep)
        x.info['sampling'] = self.sampling
        return x

    def _path(self, term: Term, state: Mapping[str, Any], root: Path) -> Path:
        # term.string is the bare code (per_event=False) or {stimulus}~{code} (per_event=True)
        return root / subject_predictor_path(state, term.string, self._key_fields)

    def _reference_stem(self, term: Term, state: Mapping[str, Any]) -> str:
        # BIDS entities in _key_fields followed by desc-{code|stimulus~code}
        return self._path(term, state, Path()).stem
