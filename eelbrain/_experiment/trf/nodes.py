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


class PredictorInput(Input[NDVar]):
    """Materialize a predictor :class:`NDVar` from its source file

    Parameters
    ----------
    root
        Experiment root directory.
    predictors
        Mapping of predictor key to predictor definition (the
        :attr:`Pipeline.predictors` attribute).
    raw
        Mapping of raw pipeline definitions (the assembled ``Pipeline._raw``),
        used to filter predictors when ``filter_x`` is requested.
    """
    name = 'predictor'
    OPTION_DEFAULTS = {
        'code': None,
        'tstep': None,
        'tmin': None,
        'n_samples': None,
        'filter_x': False,
    }

    def __init__(
            self,
            root: str | Path,
            predictors: dict[str, FilePredictor],
            raw: dict[str, RawPipe],
    ):
        self.root = Path(root)
        self.predictors = predictors
        self.raw = raw
        self.directory = self.root / 'derivatives' / 'predictors'

    def _filter_pipes(self, raw_name: str) -> list[RawFilter]:
        "The RawFilter pipes for ``raw_name``, ordered from source to output"
        pipe = self.raw[raw_name]
        pipes = []
        while not isinstance(pipe, RawSource):
            if isinstance(pipe, RawFilter):
                pipes.append(pipe)
            pipe = self.raw[pipe.source]
        pipes.reverse()
        return pipes

    def _resolve(self, ctx: Request) -> tuple[Term, FilePredictor]:
        term = parse_term(ctx.options['code'])
        key = term.code.split('-')[0]
        try:
            predictor = self.predictors[key]
        except KeyError:
            raise TRFModelError(f"{term.string}: predictor {key!r} not defined")
        if not isinstance(predictor, FilePredictor):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")
        return term, predictor

    def path(self, ctx: Request) -> Path:
        term, predictor = self._resolve(ctx)
        return self.directory / f"{term.nuts_file_name(predictor.columns)}.pickle"

    def fingerprint(self, ctx: Request) -> dict:
        term, predictor = self._resolve(ctx)
        fp = {
            'file': file_fingerprint(self.root, self.path(ctx), 'predictor-file'),
            'config': predictor,
        }
        if ctx.options['filter_x']:
            raw_name = ctx.state['raw']
            fp['raw'] = self._filter_pipes(raw_name)
        return fp

    def load(self, ctx: Request) -> NDVar:
        term, predictor = self._resolve(ctx)
        options = ctx.options
        x = predictor._generate(options['tmin'], options['tstep'], options['n_samples'], term, self.directory)
        filter_x = options['filter_x']
        if isinstance(filter_x, str):
            if filter_x == 'continuous':
                filter_x = x.info['sampling'] == 'continuous'
            else:
                raise ValueError(f"{filter_x=}")
        if filter_x:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'filter_length ', RuntimeWarning)
                for pipe in self._filter_pipes(ctx.state['raw']):
                    x = pipe._filter_ndvar(x, pad='edge')
        x.name = term.string
        return x


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
        'filter_x': False,
    }

    def __init__(
            self,
            root: str | Path,
            estimators: dict[str, Estimator],
            predictors: dict[str, FilePredictor],
            named_models: dict[str, Model],
            stim_var: dict[str, str],
            raw: dict[str, RawPipe],
    ):
        self.root = Path(root)
        self.estimators = estimators
        self.predictors = predictors
        self.named_models = named_models
        self.stim_var = stim_var
        self.raw = raw
        self.directory = self.root / 'derivatives' / 'predictors'

    def _estimator(self, ctx: Request) -> Estimator:
        name = ctx.options['estimator']
        try:
            return self.estimators[name]
        except KeyError:
            raise TRFModelError(f"estimator {name!r} not defined in Pipeline.estimators")

    def _model(self, ctx: Request) -> Model:
        return Model.coerce(ctx.options['x']).initialize(self.named_models)

    def _term_predictor(self, term: Term) -> tuple[Configuration, str]:
        """The ``(predictor_definition, stimulus_column)`` for a model term"""
        key = term.code.split('-')[0]
        try:
            predictor = self.predictors[key]
        except KeyError:
            raise TRFModelError(f"{term.string}: predictor {key!r} not defined")
        stim = term.stimulus
        if stim is None:
            stim_var = self.stim_var['']
        elif stim in self.stim_var:
            stim_var = self.stim_var[stim]
        else:
            stim_var = stim
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
        model = self._model(ctx)
        # predictor definitions: covers EventPredictor (which has no dependency
        # edge) and is harmless redundancy for FilePredictor (tracked via edges)
        predictors = {term.string: self._term_predictor(term)[0] for term in model.terms}
        return {'model': model.name, 'estimator': self._estimator(ctx), 'predictors': predictors}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        est = self._estimator(ctx)
        data = est.resolve_data(ctx.options['data'])
        samplingrate = ctx.options['samplingrate']
        filter_x = ctx.options['filter_x']
        deps = []
        options = {'samplingrate': samplingrate}
        if data in ('sensor', 'meg', 'eeg'):
            if data in ('meg', 'eeg'):
                options['data'] = data
            deps.append(Dependency('epochs', label='response', options=options))
        else:
            deps.append(Dependency('epochs-stc', label='response', options=options))
        for extra in est.extra_inputs:
            deps.append(Dependency(extra))
        # one predictor edge per (FilePredictor term, stimulus); the stimuli are
        # data-derived, so enumerate them from the (lightweight) epoch events
        pred_state = {'raw': ctx.state['raw']} if filter_x else None
        edges: dict[str, Dependency] = {}
        events = None
        for term in self._model(ctx).terms:
            predictor, stim_var = self._term_predictor(term)
            if not isinstance(predictor, FilePredictor):
                continue
            if samplingrate is None:
                raise TRFModelError(f"{term.string}: samplingrate must be specified for FilePredictor TRFs")
            if events is None:
                events = ctx.registry.resolve('epoch-events', state=dict(ctx.state)).load()
            if stim_var not in events:
                raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the events")
            for stim in events[stim_var].cells:
                code = term.with_stimulus(stim).string
                edges[code] = Dependency('predictor', label=code, state=pred_state, options={'code': code, 'tstep': 1 / samplingrate, 'tmin': None, 'n_samples': None, 'filter_x': filter_x})
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
        xs = [self._load_predictor(ctx, ds, term, y, y_name) for term in model.terms]
        fwd = cov = None
        if 'fwd' in est.extra_inputs:
            ctx.load('fwd')  # ensure built and tracked as a dependency
            fwd_path = ctx.registry.resolve('fwd', state=dict(ctx.state)).artifact_path
            fwd = load.mne.forward_operator(fwd_path, ctx.state['src'], self.root / MRI_SDIR, None, adjacency=False)
        if 'cov' in est.extra_inputs:
            cov = ctx.load('cov')
        return est._fit(y, xs, tstart, tstop, fwd=fwd, cov=cov)

    def _load_predictor(self, ctx: Request, ds, term: Term, y, y_name: str) -> NDVar:
        "Assemble one model term's predictor, aligned per case to the response"
        predictor, stim_var = self._term_predictor(term)
        is_variable_time = isinstance(y, Datalist)

        if isinstance(predictor, EventPredictor):
            if ctx.options['filter_x']:
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

        # FilePredictor: one declared predictor edge per stimulus (the full
        # predictor at the analysis tstep), aligned per case to the response
        if stim_var not in ds:
            raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the data")
        stim_factor = ds[stim_var]

        if is_variable_time:
            xs = [self._aligned_predictor(ctx, term, s, yi.time) for s, yi in zip(stim_factor, y)]
            return Datalist(xs)
        time = y.time
        cache = {s: self._aligned_predictor(ctx, term, s, time) for s in stim_factor.cells}
        x = combine([cache[s] for s in stim_factor])
        x.name = term.string
        return x

    def _aligned_predictor(self, ctx: Request, term: Term, stim: str, time) -> NDVar:
        "Load the declared predictor edge for one stimulus and align it to ``time``"
        code = term.with_stimulus(stim).string
        x = ctx.load(code)
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
