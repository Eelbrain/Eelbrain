from collections import Counter
from dataclasses import dataclass, asdict
from itertools import repeat
from pathlib import Path
from typing import Any
import warnings

import mne
import numpy as np

from ... import load, save
from ..._data_obj import Dataset, Datalist, Factor, NDVar, Var, combine, isuv
from ..._io.pickle import update_subjects_dir
from ..._mne import morph_source_space
from ..._ndvar import set_tmin
from ..._ndvar.uts import pad
from ..._utils.mne_utils import is_fake_mri
from ..configuration import Configuration
from ..data import DataSpec
from ..derivative_cache import Dependency, Derivative, OptionSpec, Request, UncachedDerivative, VersionedInput, file_fingerprint
from ..epochs.config import EpochBase
from ..pathing import BIDS_ENTITY_KEYS, MRI_SDIR, mri_dir
from ..preprocessing import RawFilter, RawPipe, RawSource
from ..source.nodes import _subject_state
from ..statistics.config import ResolvedTestNDSpec, TTestOneSample, TTestRelated, Test, TwoStageTest
from ..variable_def import Variables
from .estimator import Estimator
from .job import TRFJob
from .model import Comparison, Model, Term, TRFModelError
from .predictor import EventPredictor, NUTSPredictor, SubjectUTSPredictor, UTSPredictor


@dataclass(frozen=True)
class Recording:
    """BIDS entities identifying one recording"""
    subject: str
    session: str
    task: str
    acquisition: str
    run: str

    def dependency_label(self, term: Term):
        """Dependency label for a predictor file tied to this recording"""
        suffix = f'task-{self.task}'
        if self.run:
            suffix += f'_run-{self.run}'
        return f'{term.without_lags().string}@{suffix}'


def find_bids_recordings(ds: Dataset) -> list[Recording]:
    """Move BIDS entities into dataset columns"""
    values = []
    for key in BIDS_ENTITY_KEYS:
        if key in ds:
            values.append(ds[key])
        else:
            values.append(repeat(ds.info[key], ds.n_cases))
    return [Recording(*case) for case in zip(*values)]


def filter_pipes(raw: dict[str, RawPipe], raw_name: str) -> list[RawFilter]:
    "The RawFilter pipes for ``raw_name``, ordered from source to output"
    pipe = raw[raw_name]
    pipes = []
    while not isinstance(pipe, RawSource):
        if isinstance(pipe, RawFilter):
            pipes.append(pipe)
        pipe = raw[pipe.source]
    pipes.reverse()
    return pipes


def filter_predictor(x: NDVar, raw: dict[str, RawPipe], raw_name: str, filter_x: bool | str) -> NDVar:
    "Filter a predictor with the current ``raw`` pipeline's :class:`RawFilter` pipes when requested"
    if isinstance(filter_x, str):
        if filter_x == 'continuous':
            filter_x = x.info['sampling'] == 'continuous'
        else:
            raise ValueError(f"{filter_x=}")
    if filter_x:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'filter_length ', RuntimeWarning)
            for pipe in filter_pipes(raw, raw_name):
                x = pipe._filter_ndvar(x, pad='edge')
    return x


def _post_process_trfs(
        ds: Dataset,
        smooth: float | None,
        common_brain: str | None = None,
        source_morph: mne.SourceMorph | None = None,
) -> None:
    """Prepare TRFs for statistical analysis (morphing and smoothing)"""
    # should_morph = common_brain is not None or source_morph is not None
    if not smooth and not common_brain:
        return
    keys = [key for key in (*ds.info['xs'], *ds.info['metrics']) if isinstance(ds[key], NDVar) and ds[key].has_dim('source')]
    for key in keys:
        # if should_morph:
        if common_brain:
            ds[key] = morph_source_space(ds[key], common_brain, morph=source_morph)
        if smooth:
            # OPT: pre-compute smoothing matrix
            ds[key] = ds[key].smooth('source', smooth, 'gaussian')


class PredictorInput(VersionedInput[NDVar]):
    """Read the relevant data of a single predictor file

    Reads one predictor file and returns the subset of its contents that
    actually feeds the predictor (for a :class:`NUTSPredictor`, only the
    ``time`` and value/mask columns; a :class:`UTSPredictor` NDVar is returned
    unchanged). Shaping that data into a predictor on the M/EEG time axis
    (resampling, NUTS conversion, padding) is done by :class:`TRFDerivative`,
    which knows the response sampling rate.

    The predictor definition owns its file identity: a stimulus-based predictor
    (:class:`UTSPredictor`, :class:`NUTSPredictor`) resolves to one file per
    stimulus, keyed entirely by the ``term``. A
    :class:`SubjectUTSPredictor` additionally uses the BIDS entities declared
    by its ``_key_fields``; these depend on whether it represents a recording
    sequence or per-event stimuli.

    Because the relevant data can be large, dependent manifests do not embed
    it; they store a small version identity backed by one canonical reference
    copy per (file, relevant columns) in the cache (see
    :class:`~..derivative_cache.VersionedInput`).

    Parameters
    ----------
    root
        Experiment root directory.
    predictors
        Mapping of predictor key to predictor definition (the
        :attr:`Pipeline.predictors` attribute), used to resolve the file name
        and the relevant columns.
    """
    name = 'predictor'
    key_options = {
        'term': OptionSpec(None, Term, normalize=lambda x: Term._coerce(x).without_lags()),
    }

    def __init__(
            self,
            root: str | Path,
            predictors: dict[str, Configuration],
    ):
        self.root = Path(root)
        self.predictors = predictors

    def _resolve(self, ctx: Request) -> tuple[Term, UTSPredictor | NUTSPredictor]:
        term = ctx.options['term']
        predictor = self.predictors[term.predictor_key]
        if isinstance(predictor, SubjectUTSPredictor):
            if term.stimulus and not predictor.per_event:
                raise TRFModelError(f"{term.string}: {type(predictor).__name__}(per_event=False) cannot be combined with a stimulus")
        elif not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")
        return term, predictor

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        term, predictor = self._resolve(ctx)
        return predictor._key_fields

    def path(self, ctx: Request) -> Path:
        term, predictor = self._resolve(ctx)
        return predictor._path(term, ctx.state, self.root)

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict:
        """Quickest comparison, avoiding reference .json read"""
        term, predictor = self._resolve(ctx)
        return {
            'config': predictor,
            'file': file_fingerprint(self.root, self.path(ctx)),
        }

    def fingerprint(self, ctx: Request) -> dict:
        term, predictor = self._resolve(ctx)
        return {'config': predictor, 'version': self.reference_version(ctx)}

    def _reference_stem(self, ctx: Request) -> str:
        """Identifies the data relevant for this term"""
        term, predictor = self._resolve(ctx)
        return predictor._reference_stem(term, ctx.state)

    def _data_equal(self, ctx: Request, stored, current) -> bool:
        term, predictor = self._resolve(ctx)
        return predictor._data_equal(stored, current)

    def load(self, ctx: Request):
        term, predictor = self._resolve(ctx)
        contents = load.unpickle(self.path(ctx))
        return predictor._relevant_data(contents, term)


def _normalize_trf_options(options: dict) -> None:
    """Canonicalize the ``(x, tstart, tstop)`` options in place (see :meth:`Model.normalize_lags`), so that equivalent spellings share one cache key"""
    if options['x'] is None:
        return
    x, tstart, tstop = options['x'].normalize_lags(options['tstart'], options['tstop'])
    options.update(x=x.sorted(), tstart=tstart, tstop=tstop)


class TRFDerivative(Derivative[object]):
    """Fit and cache a TRF for one subject

    Parameters
    ----------
    root
        Experiment root directory.
    estimators
        Mapping of estimator name to :class:`Estimator` definition (the
        :attr:`Pipeline.estimators` attribute).
    predictors
        Mapping of predictor key to predictor definition.
    stim_var
        Mapping of stimulus key to the events :class:`Dataset` column that
        identifies the stimulus (the assembled ``Pipeline._stim_var``).
    raw
        Assembled raw pipeline definitions (for predictor filtering).
    """
    name = 'trf'
    cache_suffix = '.pickle'
    fixed_state = {'adjacency': ''}
    key_options = {
        'x': OptionSpec(None, Model, normalize=Model.coerce),
        'tstart': 0.0,
        'tstop': 0.5,
        'estimator': 'boosting',
        'data': OptionSpec(DataSpec('sensor'), DataSpec),
        'samplingrate': None,
        'decim': None,
        'filter_x': False,
    }

    def __init__(
            self,
            root: str | Path,
            estimators: dict[str, Estimator],
            predictors: dict[str, Configuration],
            stim_var: str,
            raw: dict[str, RawPipe],
    ):
        self.root = Path(root)
        self.estimators = estimators
        self.predictors = predictors
        self.stim_var = stim_var
        self.raw = raw

    def _term_predictor(self, term: Term) -> tuple[Configuration, str]:
        """The ``(predictor_definition, stimulus_column)`` for a model term"""
        predictor = self.predictors[term.predictor_key]
        stim_var = term.stimulus or self.stim_var
        return predictor, stim_var

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # source vs sensor space changes which fields identify the artifact.
        # This is also the read-enforcement set, so it must cover every state
        # field the build may read: 'inv' is always read (to pick the space).
        fields = ('subject', 'session', 'acquisition', 'raw', 'epoch', 'epoch_rejection', 'inv')
        est = self.estimators[ctx.options['estimator']]
        if ctx.state['inv']:  # non-empty inverse → source space
            fields += ('cov', 'mrisubject', 'src', 'parc')
        if est.extra_input_fields:
            fields += est.extra_input_fields
        else:
            fields += ('reference',)
        return tuple(fields)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'estimator': self.estimators[ctx.options['estimator']]}

    def normalize_stored_fingerprint(self, fingerprint: dict[str, Any]) -> None:
        est = fingerprint.get('estimator')
        if isinstance(est, dict) and est.get('scale_data') == 'inplace':  # < 0.43.0a3
            est['scale_data'] = True

    def validate_options(self, ctx: Request) -> None:
        _normalize_trf_options(ctx.options)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        est = self.estimators[ctx.options['estimator']]

        # M/EEG response: sensor (inv='') vs source space
        if ctx.state['inv']:  # source space
            node = 'epochs-stc'
            option_kwargs = {}
        else:
            node = 'epochs'
            option_kwargs = {
                'data': ctx.options['data'],  # resolved sensor kind
                'interpolate_bads': est.interpolate_bads,
            }
        options = ctx.options_for(node, 'samplingrate', 'decim', **option_kwargs)
        deps = [Dependency(node, label='response', options=options)]

        for extra in est.extra_inputs:
            deps.append(Dependency(extra))

        # One predictor-file edge per input. Stimulus predictors, including
        # per-event SubjectUTSPredictors, are shared across recordings;
        # sequence-mode SubjectUTSPredictors have one edge per recording.
        edges: dict[str, Dependency] = {}
        events = nested = recordings = None
        for term in ctx.options['x'].terms:
            predictor, stim_var = self._term_predictor(term)
            if isinstance(predictor, EventPredictor):
                continue  # EventPredictor generates from the events, no file edge
            elif not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
                raise RuntimeError(f"{predictor=}")
            # Lazy load events
            if events is None:
                options = ctx.options_for('epoch-events', 'samplingrate', 'decim')
                events = ctx.load('epoch-events', options=options)
                nested = events.info.get('nested_events')
            if isinstance(predictor, SubjectUTSPredictor) and not predictor.per_event:
                if recordings is None:
                    recordings = find_bids_recordings(events)
                for recording in dict.fromkeys(recordings):  # ordered set
                    label = recording.dependency_label(term)
                    edges[label] = Dependency('predictor', label=label, state=asdict(recording), options={'term': term})
                continue
            elif nested:
                stims = {stim for i in range(events.n_cases) for stim in events[i, nested][stim_var].cells}
            elif stim_var in events:
                stims = set(events[stim_var].cells)
            else:
                raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the events")
            for stim in stims:
                label = term.file_label(stim)
                edges[label] = Dependency('predictor', label=label, options={'term': term.with_stimulus(stim)})
        deps.extend(edges.values())
        return tuple(deps)

    def build(self, ctx: Request) -> object:
        return self.make_job(ctx)()

    def make_job(self, ctx: Request) -> TRFJob:
        """Load the data and assemble a picklable :class:`pipeline.TRFJob` (the fit deferred).

        Parameters
        ----------
        ctx
            Resolved request for this TRF (carries state and options).
        """
        # ctx.load('response'/<predictor code>) resolves dependency labels, which
        # requires the build-deps context; it is re-entrant, so this is safe both
        # from build() (already inside it) and from JobSpec.make_job() (fresh).
        with ctx._build_deps_context():
            est = self.estimators[ctx.options['estimator']]
            model = ctx.options['x']
            if not model.terms:
                raise TRFModelError(f"{ctx.options['x']!r}: empty model")
            tstart = ctx.options['tstart']
            tstop = ctx.options['tstop']
            if tstart is None:
                # canonical per-term lag windows (see Model.normalize_lags): every term carries explicit bounds; the estimators accept one (tstart, tstop) per predictor
                tstart = [term.tstart for term in model.terms]
                tstop = [term.tstop for term in model.terms]
            ds = ctx.load('response')
            y = ds[ctx.options['data'].response_key(ds)]
            # name predictors by the bare term; the lag window disambiguates a predictor that occurs with several windows
            base_counts = Counter((term.stimulus, term.code) for term in model.terms)
            xs = []
            for term in model.terms:
                x = self._load_predictor(ctx, ds, term, y)
                name = term.string if base_counts[term.stimulus, term.code] > 1 else term.without_lags().string
                if isinstance(x, Datalist):
                    for xi in x:
                        xi.name = name
                x.name = name
                xs.append(x)
            fwd = cov = None
            if 'fwd' in est.extra_inputs:
                fwd = ctx.load('fwd')  # ensure built and tracked as a dependency
                fwd = load.mne.forward_operator(fwd, ctx.state['src'], self.root / MRI_SDIR, None)
            if 'cov' in est.extra_inputs:
                cov = ctx.load('cov')
        return TRFJob(est, y, xs, tstart, tstop, fwd, cov)

    def _load_predictor(self, ctx: Request, ds, term: Term, y) -> NDVar | Datalist:
        "Assemble one model term's predictor, shaped to the response time axis"
        predictor, stim_var = self._term_predictor(term)
        is_variable_time = isinstance(y, Datalist)
        is_nested = ds.info.get('nested_events')  # 'events' for a ContinuousEpoch
        filter_x = ctx.options['filter_x']

        if isinstance(predictor, EventPredictor):
            if filter_x:
                raise ValueError(f"filter_x: not available for {type(predictor).__name__}")
            if is_nested:
                return Datalist([predictor._generate_continuous(yi.time, ds[i, is_nested], term) for i, yi in enumerate(y)])
            if is_variable_time:
                raise NotImplementedError(f"{type(predictor).__name__} for variable-length epochs")
            return predictor._generate(y.time, ds, term)
        elif not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")

        # per-subject sequence predictor: one recording-long file, cut per case
        if isinstance(predictor, SubjectUTSPredictor) and not predictor.per_event:
            return self._load_subject_predictor(ctx, predictor, term, ds, y, filter_x, is_nested)

        if stim_var not in ds:
            raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the data")

        if is_nested:
            return self._load_predictor_nested(ctx, predictor, term, stim_var, ds, y, is_nested, filter_x)

        # single-event epoch: one stimulus per case, aligned to the response
        stim_factor = ds[stim_var]
        if is_variable_time:
            xs = [self._aligned_predictor(ctx, predictor, term, s, yi.time, filter_x) for s, yi in zip(stim_factor, y)]
            return Datalist(xs)
        time = y.time
        cache = {s: self._aligned_predictor(ctx, predictor, term, s, time, filter_x) for s in stim_factor.cells}
        return combine([cache[s] for s in stim_factor])

    def _load_predictor_nested(self, ctx: Request, predictor: UTSPredictor | NUTSPredictor, term: Term, stim_var: str, ds, y: Datalist, nested: str, filter_x: bool | str) -> Datalist:
        "Assemble a per-event (ContinuousEpoch) predictor on the shared ``epoch_time`` axis"
        tstep = y[0].time.tstep
        stims = {stim for i in range(ds.n_cases) for stim in ds[i, nested][stim_var].cells}
        cache = {stim: predictor._prepare_stimulus(ctx.load(term.file_label(stim)), tstep) for stim in stims}
        xs = []
        for i, yi in enumerate(y):
            x = predictor._generate_continuous(yi.time, ds[i, nested], stim_var, term, cache)
            xs.append(filter_predictor(x, self.raw, ctx.state['raw'], filter_x))
        return Datalist(xs)

    def _load_subject_predictor(self, ctx: Request, predictor: SubjectUTSPredictor, term: Term, ds, y, filter_x: bool | str, is_nested: bool) -> NDVar | Datalist:
        "Cut recording-long predictors into response cases"
        times = [yi.time for yi in y] if isinstance(y, Datalist) else [y.time] * ds.n_cases
        recordings = find_bids_recordings(ds)
        x_fulls = {}
        for recording in set(recordings):
            label = recording.dependency_label(term)
            x_full = predictor._prepare_sequence(ctx.load(label), times[0].tstep, term)
            x_fulls[recording] = filter_predictor(x_full, self.raw, ctx.state['raw'], filter_x)

        if is_nested:
            offsets = [0.] * ds.n_cases
        else:
            sfreq = ds.info['raw.samplingrate']
            sample_0 = {}
            for recording, sample_i in zip(recordings, ds['sample']):
                if recording not in sample_0:
                    sample_0[recording] = sample_i
            offsets = [(sample_i - sample_0[recording]) / sfreq for recording, sample_i in zip(recordings, ds['sample'])]

        def chunk(time, recording, offset):
            x_full = x_fulls[recording]
            offset = x_full.time.tstep * round(offset / x_full.time.tstep)  # snap to the predictor's sample grid
            x = set_tmin(x_full, x_full.time.tmin - offset) if offset else x_full  # global time offset -> local 0
            return pad(x, time.tmin, nsamples=time.nsamples, set_tmin=True)  # crop to the segment

        xs = [chunk(time, recording, offset) for time, recording, offset in zip(times, recordings, offsets)]
        if isinstance(y, Datalist):
            return Datalist(xs)
        return combine(xs)

    def _aligned_predictor(self, ctx: Request, predictor: UTSPredictor | NUTSPredictor, term: Term, stim: str | None, time, filter_x: bool | str) -> NDVar:
        "Build one stimulus' predictor from its file data and align it to ``time``"
        subset = ctx.load(term.file_label(stim))
        x = predictor._generate(subset, None, time.tstep, None, term)
        x = filter_predictor(x, self.raw, ctx.state['raw'], filter_x)
        return pad(x, time.tmin, nsamples=time.nsamples, set_tmin=True)

    def save(self, ctx: Request, path: Path, value: object) -> None:
        save.pickle(value, path)

    def load(self, ctx: Request, path: Path) -> object:
        return load.unpickle(path)


# Options shared by the TRF-dataset nodes: the :class:`TRFDerivative` options that
# select the fit, plus the dataset-shaping ``scale``, ``smooth``, and ``trfs``.
_TRF_DATASET_OPTIONS = {
    'x': OptionSpec(None, Model, normalize=Model.coerce),
    'tstart': 0.0,
    'tstop': 0.5,
    'estimator': 'boosting',
    'data': OptionSpec(None, DataSpec),
    'samplingrate': None,
    'decim': None,
    'filter_x': False,
    'scale': None,
    'smooth': None,
    'trfs': True,
}


class TRFDatasetDerivative(UncachedDerivative[Dataset]):
    """Assemble one subject's TRF result(s) into a :class:`Dataset`

    Wraps the cached :class:`TRFDerivative` result into a single-case dataset of
    fit metrics and TRF kernels (one case per member epoch for an
    :class:`EpochCollection`). Source-space data is morphed to the common brain
    so that subjects can be combined.

    Parameters
    ----------
    root
        Experiment root directory.
    estimators
        Mapping of estimator name to :class:`Estimator` definition.
    epochs
        Assembled epoch definitions.
    """
    name = 'trf-dataset'
    key_options = _TRF_DATASET_OPTIONS

    def __init__(
            self,
            root: str | Path,
            estimators: dict[str, Estimator],
            epochs: dict[str, EpochBase],
    ):
        self.root = Path(root)
        self.estimators = estimators
        self.epochs = epochs

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = ['subject', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'reference', 'raw', 'inv']
        if ctx.state['inv']:
            fields += ['cov', 'src', 'parc', 'adjacency', 'mrisubject', 'common_brain']
        return tuple(fields)

    def validate_options(self, ctx: Request) -> None:
        _normalize_trf_options(ctx.options)
        if not ctx.state['inv'] and (smooth := ctx.options['smooth']):
            raise ValueError(f"{smooth=}: smoothing is only available for source-space data")

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        trf_options = ctx.options_for('trf', 'x', 'tstart', 'tstop', 'estimator', 'data', 'samplingrate', 'decim', 'filter_x')
        epoch_def = self.epochs[ctx.state['epoch']]
        deps = [Dependency('trf', label=epoch, state={'epoch': epoch}, options=trf_options) for epoch in epoch_def.collected_epochs]
        if ctx.state['inv'] and not is_fake_mri(self.root / mri_dir(ctx.state)):
            deps.append(Dependency('source-morph'))
        return tuple(deps)

    def build(self, ctx: Request) -> Dataset:
        est = self.estimators[ctx.options['estimator']]
        scale = ctx.options['scale']
        trfs = ctx.options['trfs']
        epoch_def = self.epochs[ctx.state['epoch']]
        dss = [est._result_dataset(ctx.load(epoch), scale=scale, trfs=trfs) for epoch in epoch_def.collected_epochs]
        ds = combine(dss, name=ctx.options['x'].name)
        # Morphing/smoothing
        if ctx.state['inv']:
            common_brain = ctx.state['common_brain']
            if is_fake_mri(self.root / mri_dir(ctx.state)):
                source_morph = None
            else:
                source_morph = ctx.load('source-morph')
            _post_process_trfs(ds, ctx.options['smooth'], common_brain, source_morph)
        return ds

    def apply_view_options(self, ctx: Request, value: Dataset) -> Dataset:
        """Add the columns identifying the cases, which :meth:`build` does not

        One case per collected epoch, so these follow from the epoch definition and
        the subject alone; :meth:`load_view` produces them without loading data.
        ``task`` is added for all cases or for none, since the cases are combined
        across subjects.
        """
        epoch_def = self.epochs[ctx.state['epoch']]
        value['epoch'] = Factor(epoch_def.collected_epochs)
        if tasks := epoch_def.collected_tasks:
            value['task'] = Factor(tasks)
        value['subject'] = Factor([ctx.state['subject']], repeat=value.n_cases, random=True)
        return value

    def load_view(self, ctx: Request, view: str):
        """The ``shell`` view: the non-data columns of this dataset, without loading data

        A TRF dataset carries no event columns, so the shell is what
        :meth:`apply_view_options` adds, and nothing else.
        """
        if view != 'shell':
            return super().load_view(ctx, view)
        return self.apply_view_options(ctx, Dataset())


class TRFGroupDatasetDerivative(UncachedDerivative[Dataset]):
    """Combine per-subject TRF datasets for a group into one :class:`Dataset`

    Parameters
    ----------
    mri_subjects
        Mapping of ``mri`` value to subject→MRI-subject (for per-subject state).
    variables
        Global pipeline variable definitions; the across-subject ones are added
        here.
    groups
        Mapping of group name to the sequence of member subjects.

    Notes
    -----
    Across-subject variables are added here, because this is where subjects are
    combined. A TRF dataset carries no event columns, so only a variable keyed on
    ``subject`` applies (a ``task`` restriction is honored through the ``task``
    column added by :class:`TRFDatasetDerivative`). They are added in
    :meth:`apply_view_options`, i.e. they are part of the returned data but not of
    this node's fingerprint; a cached consumer that reads them is responsible for
    recording their values (see
    :meth:`~eelbrain._experiment.variable_def.Variables.resolve`).
    """
    name = 'trf-group-dataset'
    key_options = _TRF_DATASET_OPTIONS

    def __init__(
            self,
            mri_subjects: dict[str, dict[str, str]],
            variables: Variables,
            groups: dict[str, tuple[str, ...]],
    ):
        self.mri_subjects = mri_subjects
        self.variables = variables
        self.groups = groups

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = ['group', 'mri', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'reference', 'raw', 'inv']
        if ctx.state['inv']:
            fields += ['cov', 'src', 'parc', 'adjacency', 'mrisubject', 'common_brain']
        return tuple(fields)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subjects': self.groups[ctx.state['group']]}

    def _subject_options(self, ctx: Request) -> dict[str, Any]:
        "Options for the per-subject datasets: smoothing is applied to the combined dataset instead, so that the source smoothing matrix is calculated only once"
        subject_options = [key for key in self.key_options if key != 'smooth']
        return ctx.options_for('trf-dataset', *subject_options, smooth=None)

    def validate_options(self, ctx: Request) -> None:
        _normalize_trf_options(ctx.options)
        if not ctx.state['inv'] and (smooth := ctx.options['smooth']):
            raise ValueError(f"{smooth=}: smoothing is only available for source-space data")

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = self._subject_options(ctx)
        return tuple(
            Dependency('trf-dataset', label=subject, state=_subject_state(ctx.state, subject, self.mri_subjects), options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        dss = [ctx.load(subject) for subject in self.groups[ctx.state['group']]]
        ds = combine(dss, to_list=True)
        _post_process_trfs(ds, ctx.options['smooth'])
        return ds

    def apply_view_options(self, ctx: Request, value: Dataset) -> Dataset:
        self.variables.resolve(value, self.groups, across_subject_only=True)
        return value

    def load_view(self, ctx: Request, view: str):
        """The ``shell`` view: the non-data columns of this dataset, without loading data

        Assembled like the data in :meth:`build`, from the subjects' shells, so that
        a consumer can fingerprint the variables it reads (see
        :meth:`~eelbrain._experiment.variable_def.Variables.resolve`) against exactly
        the columns it will get. A TRF dataset carries no event columns, so those
        shells need no data of their own, which makes this cheap enough for a
        cache-validity check.
        """
        if view != 'shell':
            return super().load_view(ctx, view)
        options = self._subject_options(ctx)
        dss = [ctx.load('trf-dataset', state=_subject_state(ctx.state, subject, self.mri_subjects), options=options, view='shell') for subject in self.groups[ctx.state['group']]]
        return self.apply_view_options(ctx, combine(dss))


class TRFModelTestDerivative(Derivative[Any]):
    """Cache a statistical comparison of TRF model-fit metrics.

    The derivative depends on group-level metric datasets for the test and,
    unless the comparison is against zero, baseline model. The cached artifact
    contains only the statistical result; ``return_data`` is a view option that
    reconstructs the uncached input dataset on demand.

    Parameters
    ----------
    tests
        Configured :attr:`Pipeline.tests` definitions.
    groups
        Mapping of group names to their member subjects.
    """
    name = 'trf-model-test'
    cache_suffix = '.pickle'
    key_options = {
        'x': OptionSpec(None, Comparison, normalize=Comparison._coerce),
        'tstart': 0.0,
        'tstop': 0.5,
        'estimator': 'boosting',
        'data': OptionSpec(None, DataSpec),
        'samplingrate': None,
        'filter_x': False,
        'metric': 'ev',
        'smooth': None,
        'test': None,
        'pmin': 'tfce',
        'samples': 10000,
    }
    view_options = {
        'return_data': OptionSpec(False, bool),
    }

    def __init__(
            self,
            tests: dict[str, Test],
            groups: dict[str, tuple[str, ...]],
    ):
        self.tests = tests
        self.groups = groups

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = ['group', 'mri', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'reference', 'raw', 'inv']
        if ctx.state['inv']:
            fields += ['cov', 'src', 'parc', 'adjacency', 'mrisubject', 'common_brain']
        return tuple(fields)

    def validate_options(self, ctx: Request) -> None:
        _normalize_trf_options(ctx.options)
        metric, pmin, samples = ctx.options['metric'], ctx.options['pmin'], ctx.options['samples']
        _, reducer = self._metric_parts(metric)
        # A reducer always leaves one value per case; that an unreduced metric is already univariate only the data shows (checked in _test_data)
        if reducer and (pmin is not None or samples):
            raise ValueError(f"{metric=} leaves one value per case, which is tested parametrically; {pmin=} and {samples=} do not apply (use pmin=None, samples=0)")

    def _test_obj(self, ctx: Request) -> Test | None:
        "The named test, or ``None`` for the default incremental model test"
        test_name = ctx.options['test']
        if test_name is None:
            return None
        test_obj = self.tests[test_name]
        if isinstance(test_obj, TwoStageTest):
            raise NotImplementedError(f"test={test_name!r}: TwoStageTest not implemented for TRF model tests")
        return test_obj

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        if test_obj := self._test_obj(ctx):
            return {'test': test_obj._as_dict_without_vars()}
        return {}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        comparison = ctx.options['x']
        option_names = ('tstart', 'tstop', 'estimator', 'data', 'samplingrate', 'filter_x', 'smooth')
        x1_options = ctx.options_for('trf-group-dataset', *option_names, x=comparison.x1, scale=None, trfs=False)
        deps = [Dependency('trf-group-dataset', label='x1', options=x1_options)]
        if comparison.x0:
            options = ctx.options_for('trf-group-dataset', *option_names, x=comparison.x0, scale=None, trfs=False)
            deps.append(Dependency('trf-group-dataset', label='x0', options=options))
        test_obj = self._test_obj(ctx)
        if test_obj and test_obj._test_vars:
            # The same dataset's shell to fingerprint the columns used by the test's variables
            deps.append(Dependency('trf-group-dataset', label='events', view='shell', options=x1_options))
        return tuple(deps)

    def dependency_fingerprint_override(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> dict[str, Any] | None:
        """Depend on the values the test reads, not the definitions behind them

        The shell already carries the across-subject :attr:`Pipeline.variables`; only
        the test's own are applied on top, exactly as in :meth:`build` (see
        :meth:`Test._resolve_vars`).
        """
        if dep.label != 'events':
            return None
        return self._test_obj(ctx)._resolve_vars(ctx.load(dep.label), self.groups)

    @staticmethod
    def _metric_parts(metric: str) -> tuple[str, str | None]:
        if '.' not in metric:
            return metric, None
        metric_, reducer = metric.rsplit('.', 1)
        if reducer not in ('sum', 'mean', 'max'):
            raise ValueError(f"{metric=} with reducer {reducer!r}: expected 'sum', 'mean', or 'max'")
        return metric_, reducer

    @staticmethod
    def _assert_aligned(ds1: Dataset, ds0: Dataset) -> None:
        if ds1.n_cases != ds0.n_cases:
            desc = "case number"
        elif not np.all(ds1['subject'] == ds0['subject']):
            desc = "subject"
        elif 'epoch' in ds1 and not np.all(ds1['epoch'] == ds0['epoch']):
            desc = "epoch"
        else:
            return
        raise RuntimeError(f"TRF model datasets are not aligned by {desc}")

    def _test_data(
            self,
            ctx: Request,
    ) -> tuple[Dataset, Var | NDVar, Test]:
        comparison = ctx.options['x']
        metric, reducer = self._metric_parts(ctx.options['metric'])
        test_obj = self._test_obj(ctx)
        ds1 = ctx.load('x1')
        if metric not in ds1:
            available = ', '.join(ds1.info.get('metrics', ()))
            raise ValueError(f"{metric=} is not available from estimator {ctx.options['estimator']!r}; available metrics: {available}")

        if comparison.x0:
            ds0 = ctx.load('x0')
            self._assert_aligned(ds1, ds0)
            # Only variables that are identical between the datasets should matter for tests
            keep = tuple([key for key in ds1 if key in ds0 and isuv(ds1[key]) and isuv(ds0[key]) and np.all(ds1[key] == ds0[key])])
            if test_obj is None:
                keep += (metric,)
                ds = combine((ds1[keep], ds0[keep]))
                ds['model'] = Factor(('test', 'baseline'), repeat=ds1.n_cases)
                test_obj = TTestRelated('model', 'test', 'baseline', comparison.tail)
            else:
                ds = ds1[keep]
                if isinstance(ds1[metric], Datalist):
                    ds[metric] = Datalist([value1 - value0 for value1, value0 in zip(ds1[metric], ds0[metric])])
                else:
                    ds[metric] = ds1[metric] - ds0[metric]
        else:
            ds = ds1
            if test_obj is None:
                test_obj = TTestOneSample(comparison.tail)

        test_obj._resolve_vars(ds, self.groups)
        y = ds[metric]
        if reducer is None:
            if isinstance(y, Datalist):
                raise ValueError(f"{metric=} has inconsistent spatial dimensions across cases; specify a .sum, .mean, or .max reduction")
        elif isinstance(y, Var):
            raise ValueError(f"{metric=} is already univariate and can not be reduced with .{reducer}")
        elif isinstance(y, Datalist):
            dim = 'sensor' if y[0].has_dim('sensor') else 'source'
            y = combine([getattr(yi, reducer)(dim) for yi in y])
        else:
            dim = 'sensor' if y.has_dim('sensor') else 'source'
            y = getattr(y, reducer)(dim)
        return ds, y, test_obj

    def build(self, ctx: Request) -> Any:
        ds, y, test_obj = self._test_data(ctx)
        test_spec = ResolvedTestNDSpec.from_request(ctx, time=False)
        return test_spec.make_result(self, y, ds, test_obj)

    def apply_view_options(self, ctx: Request, value: Any) -> Any:
        if not ctx.view_options['return_data']:
            return value
        ds, _, _ = self._test_data(ctx)
        return ds, value

    def load(self, ctx: Request, path: Path) -> Any:
        value = load.unpickle(path)
        if ctx.options['data'].source:
            update_subjects_dir(value, ctx.root / MRI_SDIR, 2)
        return value

    def save(self, ctx: Request, path: Path, value: Any) -> None:
        save.pickle(value, path)
