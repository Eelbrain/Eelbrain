from pathlib import Path
import warnings

from ... import load, save
from ..._data_obj import Dataset, Datalist, Factor, NDVar, combine
from ..._mne import morph_source_space
from ..._ndvar.uts import pad
from ..._utils.mne_utils import is_fake_mri
from ..configuration import Configuration
from ..derivative_cache import Dependency, Derivative, Input, Request, UncachedDerivative, canonical_state_subset, file_fingerprint
from ..epochs.config import EpochCollection
from ..pathing import MRI_SDIR, mri_dir
from ..preprocessing import RawFilter, RawPipe, RawSource
from ..source.nodes import _subject_state
from .estimator import Estimator
from .job import TRFJob
from .model import Model, Term, TRFModelError, parse_term
from .predictor import EventPredictor, FilePredictor, SessionPredictor


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


class PredictorInput(Input[NDVar]):
    """Read the relevant data of a single predictor file

    Reads one ``{stimulus}~{code}.pickle`` file of a :class:`FilePredictor` and
    returns the subset of its contents that actually feeds the predictor (for a
    NUTS :class:`Dataset`, only the ``time`` and value/mask columns; an
    :class:`NDVar`/list is returned unchanged). Shaping that data into a
    predictor on the M/EEG time axis (resampling, NUTS conversion, padding) is
    done by :class:`TRFDerivative`, which knows the response sampling rate.

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
    OPTION_DEFAULTS = {
        'code': None,
    }

    def __init__(
            self,
            root: str | Path,
            predictors: dict[str, FilePredictor],
    ):
        self.root = Path(root)
        self.predictors = predictors
        self.directory = self.root / 'derivatives' / 'predictors'

    def _resolve(self, ctx: Request) -> tuple[Term, FilePredictor]:
        term = parse_term(ctx.options['code'])
        predictor = self.predictors[term.predictor_key]
        if not isinstance(predictor, FilePredictor):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")
        return term, predictor

    def path(self, ctx: Request) -> Path:
        term, predictor = self._resolve(ctx)
        return self.directory / f"{term.nuts_file_name(predictor.columns)}.pickle"

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict:
        term, predictor = self._resolve(ctx)
        return {'file': file_fingerprint(self.root, self.path(ctx), 'predictor-file'), 'config': predictor}

    def fingerprint(self, ctx: Request) -> dict:
        return {'data': self.load(ctx)}

    def load(self, ctx: Request):
        term, predictor = self._resolve(ctx)
        contents = load.unpickle(self.path(ctx))
        return predictor._relevant_data(contents, term)


# Response NDVar keys in the loaded Dataset, ordered by preference
_Y_NAMES = ('srcm', 'src', 'meg', 'eeg')


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
    named_models
        Named models for expanding model abbreviations.
    stim_var
        Mapping of stimulus key to the events :class:`Dataset` column that
        identifies the stimulus (the assembled ``Pipeline._stim_var``).
    raw
        Assembled raw pipeline definitions (for predictor filtering).
    """
    name = 'trf'
    cache_suffix = '.pickle'
    key_fields = ('subject', 'session', 'raw', 'epoch', 'epoch_rejection', 'reference', 'cov', 'mrisubject', 'src', 'inv', 'parc')
    OPTION_DEFAULTS = {
        'x': None,
        'tstart': 0.0,
        'tstop': 0.5,
        'estimator': 'boosting',
        'data': None,
        'mask': None,
        'samplingrate': None,
        'decim': None,
        'filter_x': False,
    }

    def __init__(
            self,
            root: str | Path,
            estimators: dict[str, Estimator],
            predictors: dict[str, FilePredictor],
            named_models: dict[str, Model],
            stim_var: str,
            raw: dict[str, RawPipe],
    ):
        self.root = Path(root)
        self.estimators = estimators
        self.predictors = predictors
        self.named_models = named_models
        self.stim_var = stim_var
        self.raw = raw

    def _estimator(self, ctx: Request) -> Estimator:
        return self.estimators[ctx.options['estimator']]

    def _model(self, ctx: Request) -> Model:
        return Model.coerce(ctx.options['x']).initialize(self.named_models)

    def _term_predictor(self, term: Term) -> tuple[Configuration, str]:
        """The ``(predictor_definition, stimulus_column)`` for a model term"""
        predictor = self.predictors[term.predictor_key]
        stim_var = term.stimulus or self.stim_var
        return predictor, stim_var

    def key(self, ctx: Request) -> dict[str, object]:
        est = self._estimator(ctx)
        data = est.resolve_data(ctx.options['data'])
        fields = ['subject', 'session', 'raw', 'epoch', 'epoch_rejection', 'reference']
        if data not in ('sensor', 'meg', 'eeg'):
            fields += ['cov', 'mrisubject', 'src', 'parc']
            if 'fwd' not in est.extra_inputs:  # boosting source uses the inverse
                fields.append('inv')
        elif est.extra_inputs:  # NCRF: sensor data + forward solution
            fields += ['cov', 'mrisubject', 'src']
        key = canonical_state_subset(ctx.state, tuple(fields))
        key.update(ctx.options)
        key['x'] = self._model(ctx).name
        return key

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'estimator': self._estimator(ctx)}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        est = self._estimator(ctx)
        data = est.resolve_data(ctx.options['data'])

        # M/EEG data
        option_kwargs = {}
        if data in ('meg', 'eeg'):
            option_kwargs['data'] = data
        if data in (None, 'sensor', 'meg', 'eeg'):
            node = 'epochs'
            option_kwargs['interpolate_bads'] = est.interpolate_bads
        else:
            node = 'epochs-stc'
        options = ctx.options_for(node, 'samplingrate', 'decim', **option_kwargs)
        deps = [Dependency(node, label='response', options=options)]

        for extra in est.extra_inputs:
            deps.append(Dependency(extra))

        # one predictor-file edge per (FilePredictor term, stimulus); the stimuli
        # are data-derived, so enumerate them from the (lightweight) epoch events
        edges: dict[str, Dependency] = {}
        events = None
        for term in self._model(ctx).terms:
            predictor, stim_var = self._term_predictor(term)
            if not isinstance(predictor, FilePredictor):
                continue
            if events is None:
                events = ctx.registry.resolve('epoch-events', state=dict(ctx.state)).load()
            if stim_var not in events:
                raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the events")
            for stim in events[stim_var].cells:
                code = term.with_stimulus(stim).string
                edges[code] = Dependency('predictor', label=code, options={'code': code})
        deps.extend(edges.values())
        return tuple(deps)

    def build(self, ctx: Request) -> object:
        est = self._estimator(ctx)
        model = self._model(ctx)
        if not model.terms:
            raise TRFModelError(f"{ctx.options['x']!r}: empty model")
        tstart = ctx.options['tstart']
        tstop = ctx.options['tstop']
        ds = ctx.load('response')
        for y_name in _Y_NAMES:
            if y_name in ds:
                break
        else:
            raise RuntimeError(f"No response NDVar in loaded data (keys: {', '.join(ds.keys())})")
        y = ds[y_name]
        xs = [self._load_predictor(ctx, ds, term, y) for term in model.terms]
        fwd = cov = None
        if 'fwd' in est.extra_inputs:
            fwd = ctx.load('fwd')  # ensure built and tracked as a dependency
            fwd = load.mne.forward_operator(fwd, ctx.state['src'], self.root / MRI_SDIR, None)
        if 'cov' in est.extra_inputs:
            cov = ctx.load('cov')
        return est._fit(y, xs, tstart, tstop, fwd=fwd, cov=cov)

    def _load_predictor(self, ctx: Request, ds, term: Term, y) -> NDVar:
        "Assemble one model term's predictor, shaped to the response time axis"
        predictor, stim_var = self._term_predictor(term)
        is_variable_time = isinstance(y, Datalist)
        filter_x = ctx.options['filter_x']

        if isinstance(predictor, EventPredictor):
            if filter_x:
                raise ValueError(f"filter_x: not available for {type(predictor).__name__}")
            if is_variable_time:
                raise NotImplementedError(f"{type(predictor).__name__} for variable-length epochs")
            x = predictor._generate(y.time, ds, term)
            x.name = term.string
            return x
        if isinstance(predictor, SessionPredictor):
            raise NotImplementedError(f"{term.string}: {type(predictor).__name__} is not supported yet")
        if not isinstance(predictor, FilePredictor):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")

        # FilePredictor: build each stimulus' predictor from its file data at the
        # response sampling rate, then align per case to the response
        if stim_var not in ds:
            raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the data")
        stim_factor = ds[stim_var]
        if is_variable_time:
            xs = [self._aligned_predictor(ctx, predictor, term, s, yi.time, filter_x) for s, yi in zip(stim_factor, y)]
            return Datalist(xs)
        time = y.time
        cache = {s: self._aligned_predictor(ctx, predictor, term, s, time, filter_x) for s in stim_factor.cells}
        x = combine([cache[s] for s in stim_factor])
        x.name = term.string
        return x

    def _aligned_predictor(self, ctx: Request, predictor: FilePredictor, term: Term, stim: str, time, filter_x: bool | str) -> NDVar:
        "Build one stimulus' predictor from its file data and align it to ``time``"
        subset = ctx.load(term.with_stimulus(stim).string)
        x = predictor._generate(subset, None, time.tstep, None, term)
        x = filter_predictor(x, self.raw, ctx.state['raw'], filter_x)
        x = pad(x, time.tmin, nsamples=time.nsamples, set_tmin=True)
        x.name = term.string
        return x

    def save(self, ctx: Request, path: Path, value: object) -> None:
        save.pickle(value, path)

    def load(self, ctx: Request, path: Path) -> object:
        return load.unpickle(path)

    def make_job(self, ctx: Request, experiment_class: type) -> TRFJob:
        """Create a picklable :class:`TRFJob` for computing this TRF elsewhere.

        Parameters
        ----------
        ctx
            Resolved request for this TRF (carries state and options).
        experiment_class
            The :class:`Pipeline` subclass to reconstruct on the worker.
        """
        return TRFJob(experiment_class, str(self.root), dict(ctx.state), dict(ctx.options), ctx.artifact_path)


# Options shared by the TRF-dataset nodes: the :class:`TRFDerivative` options that
# select the fit, plus the dataset-shaping ``scale`` and ``trfs``.
_TRF_DATASET_OPTIONS = {
    'x': None,
    'tstart': 0.0,
    'tstop': 0.5,
    'estimator': 'boosting',
    'data': None,
    'mask': None,
    'samplingrate': None,
    'decim': None,
    'filter_x': False,
    'scale': None,
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
    named_models
        Named models for expanding model abbreviations.
    epochs
        Assembled epoch definitions (for :class:`EpochCollection` expansion).
    """
    name = 'trf-dataset'
    OPTION_DEFAULTS = _TRF_DATASET_OPTIONS

    def __init__(
            self,
            root: str | Path,
            estimators: dict[str, Estimator],
            named_models: dict[str, Model],
            epochs: dict[str, object],
    ):
        self.root = Path(root)
        self.estimators = estimators
        self.named_models = named_models
        self.epochs = epochs

    def _estimator(self, ctx: Request) -> Estimator:
        return self.estimators[ctx.options['estimator']]

    def _model(self, ctx: Request) -> Model:
        return Model.coerce(ctx.options['x']).initialize(self.named_models)

    def _is_source(self, ctx: Request) -> bool:
        return self._estimator(ctx).resolve_data(ctx.options['data']) not in ('sensor', 'meg', 'eeg')

    def _epoch_names(self, ctx: Request) -> list[str]:
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            return list(epoch.collect)
        return [ctx.state['epoch']]

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        trf_options = ctx.options_for('trf', 'x', 'tstart', 'tstop', 'estimator', 'data', 'mask', 'samplingrate', 'decim', 'filter_x')
        deps = [Dependency('trf', label=epoch, state={'epoch': epoch}, options=trf_options) for epoch in self._epoch_names(ctx)]
        if self._is_source(ctx) and not is_fake_mri(self.root / mri_dir(ctx.state)):
            deps.append(Dependency('source-morph'))
        return tuple(deps)

    def build(self, ctx: Request) -> Dataset:
        est = self._estimator(ctx)
        scale = ctx.options['scale']
        trfs = ctx.options['trfs']
        subject = ctx.state['subject']
        is_source = self._is_source(ctx)
        source_morph = None
        if is_source and not is_fake_mri(self.root / mri_dir(ctx.state)):
            source_morph = ctx.load('source-morph')
        common_brain = ctx.state['common_brain']
        dss = []
        for epoch in self._epoch_names(ctx):
            res = ctx.load(epoch)
            ds = est._result_dataset(res, scale=scale, trfs=trfs)
            ds['subject'] = Factor([subject], random=True)
            ds[:, 'epoch'] = epoch
            if is_source:
                for key in (*ds.info['xs'], *ds.info['metrics']):
                    if key in ds and isinstance(ds[key], NDVar) and ds[key].has_dim('source'):
                        ds[key] = morph_source_space(ds[key], common_brain, morph=source_morph)
            dss.append(ds)
        ds = combine(dss)
        ds.name = self._model(ctx).name
        return ds


class TRFGroupDatasetDerivative(UncachedDerivative[Dataset]):
    """Combine per-subject TRF datasets for a group into one :class:`Dataset`

    Parameters
    ----------
    mri_subjects
        Mapping of ``mri`` value to subject→MRI-subject (for per-subject state).
    common_brain
        Common-brain MRI subject (morph target for source data).
    groups
        Mapping of group name to the sequence of member subjects.
    """
    name = 'trf-group-dataset'
    OPTION_DEFAULTS = _TRF_DATASET_OPTIONS

    def __init__(
            self,
            mri_subjects: dict[str, dict[str, str]],
            common_brain: str,
            groups: dict[str, tuple[str, ...]],
    ):
        self.mri_subjects = mri_subjects
        self.common_brain = common_brain
        self.groups = groups

    def key(self, ctx: Request) -> dict[str, object]:
        return {'subjects': tuple(self.groups[ctx.state['group']]), 'options': ctx.options}

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subjects': tuple(self.groups[ctx.state['group']])}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = ctx.options_for('trf-dataset', *self.OPTION_DEFAULTS)
        return tuple(
            Dependency('trf-dataset', label=subject, state=_subject_state(ctx.state, subject, self.mri_subjects, self.common_brain), options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        dss = [ctx.load(subject) for subject in self.groups[ctx.state['group']]]
        return combine(dss, to_list=True)
