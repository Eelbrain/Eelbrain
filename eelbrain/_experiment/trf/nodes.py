from pathlib import Path
import warnings

from ... import load, save
from ..._data_obj import Datalist, NDVar, combine
from ..._ndvar.uts import pad
from ..configuration import Configuration
from ..derivative_cache import Dependency, Derivative, Input, Request, canonical_state_subset, file_fingerprint
from ..pathing import MRI_SDIR
from ..preprocessing import RawFilter, RawPipe, RawSource
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
