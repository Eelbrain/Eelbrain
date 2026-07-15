from pathlib import Path
import warnings

import mne

from ... import load, save
from ..._data_obj import Dataset, Datalist, Factor, NDVar, combine
from ..._mne import morph_source_space
from ..._ndvar.uts import pad
from ..._utils.mne_utils import is_fake_mri
from ..configuration import Configuration
from ..data import DataSpec
from ..derivative_cache import Dependency, Derivative, OptionSpec, Request, UncachedDerivative, VersionedInput, file_fingerprint
from ..epochs.config import EpochCollection
from ..pathing import MRI_SDIR, mri_dir
from ..preprocessing import RawFilter, RawPipe, RawSource
from ..source.nodes import _subject_state
from .estimator import Estimator
from .job import TRFJob
from .model import Model, Term, TRFModelError, parse_term
from .predictor import EventPredictor, NUTSPredictor, UTSPredictor


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

    Reads one ``{stimulus}~{code}.pickle`` predictor file and returns the
    subset of its contents that actually feeds the predictor (for a
    :class:`NUTSPredictor`, only the ``time`` and value/mask columns; a
    :class:`UTSPredictor` NDVar is returned unchanged). Shaping that data into
    a predictor on the M/EEG time axis (resampling, NUTS conversion, padding)
    is done by :class:`TRFDerivative`, which knows the response sampling rate.

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
    key_fields = ()  # identity is fully option-based (the predictor ``code``)
    key_options = {
        'code': None,
    }

    def __init__(
            self,
            root: str | Path,
            predictors: dict[str, Configuration],
    ):
        self.root = Path(root)
        self.predictors = predictors
        self.directory = self.root / 'derivatives' / 'predictors'

    def _resolve(self, ctx: Request) -> tuple[Term, UTSPredictor | NUTSPredictor]:
        term = parse_term(ctx.options['code'])
        predictor = self.predictors[term.predictor_key]
        if not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")
        return term, predictor

    def path(self, ctx: Request) -> Path:
        term, predictor = self._resolve(ctx)
        return self.directory / f"{predictor._file_stem(term)}.pickle"

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict:
        term, predictor = self._resolve(ctx)
        return {
            'config': predictor,
            'file': file_fingerprint(self.root, self.path(ctx)),
        }

    def fingerprint(self, ctx: Request) -> dict:
        term, predictor = self._resolve(ctx)
        return {'config': predictor, 'version': self.reference_version(ctx)}

    def _reference_stem(self, ctx: Request) -> str:
        term, predictor = self._resolve(ctx)
        return predictor._reference_stem(term)

    def _source_fingerprint(self, ctx: Request) -> dict:
        return file_fingerprint(self.root, self.path(ctx))

    def _current_data(self, ctx: Request):
        return self.load(ctx)

    def _data_equal(self, ctx: Request, stored, current) -> bool:
        term, predictor = self._resolve(ctx)
        return predictor._data_equal(stored, current)

    def load(self, ctx: Request):
        term, predictor = self._resolve(ctx)
        contents = load.unpickle(self.path(ctx))
        return predictor._relevant_data(contents, term)


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
        'mask': None,
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

    def _estimator(self, ctx: Request) -> Estimator:
        return self.estimators[ctx.options['estimator']]

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
        if ctx.state['inv']:  # non-empty inverse → source space
            fields += ('cov', 'mrisubject', 'src', 'parc')
        elif self._estimator(ctx).extra_inputs:  # NCRF: sensor data + forward solution
            fields += ('cov', 'mrisubject', 'src')
        else:
            fields += ('reference',)
        return tuple(fields)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'estimator': self._estimator(ctx)}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        est = self._estimator(ctx)

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

        # one predictor-file edge per
        edges: dict[str, Dependency] = {}
        events = None
        for term in ctx.options['x'].terms:
            predictor, stim_var = self._term_predictor(term)
            if not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
                continue
            if events is None:
                options = ctx.options_for('epoch-events', 'samplingrate', 'decim')
                events = ctx.load('epoch-events', options=options)
            if stim_var not in events:
                raise TRFModelError(f"{term.string}: stimulus variable {stim_var!r} not in the events")
            for stim in events[stim_var].cells:
                code = term.with_stimulus(stim).string
                edges[code] = Dependency('predictor', label=code, options={'code': code})
        deps.extend(edges.values())
        return tuple(deps)

    def build(self, ctx: Request) -> object:
        return self.make_job(ctx).fit()

    def make_job(self, ctx: Request) -> TRFJob:
        """Load the data and assemble a picklable :class:`TRFJob` (the fit deferred).

        Parameters
        ----------
        ctx
            Resolved request for this TRF (carries state and options).
        """
        # ctx.load('response'/<predictor code>) resolves dependency labels, which
        # requires the build-deps context; it is re-entrant, so this is safe both
        # from build() (already inside it) and from TRFJobSpec.make_job() (fresh).
        with ctx._build_deps_context():
            est = self._estimator(ctx)
            model = ctx.options['x']
            if not model.terms:
                raise TRFModelError(f"{ctx.options['x']!r}: empty model")
            tstart = ctx.options['tstart']
            tstop = ctx.options['tstop']
            ds = ctx.load('response')
            y = ds[ctx.options['data'].response_key(ds)]
            xs = [self._load_predictor(ctx, ds, term, y) for term in model.terms]
            fwd = cov = None
            if 'fwd' in est.extra_inputs:
                fwd = ctx.load('fwd')  # ensure built and tracked as a dependency
                fwd = load.mne.forward_operator(fwd, ctx.state['src'], self.root / MRI_SDIR, None)
            if 'cov' in est.extra_inputs:
                cov = ctx.load('cov')
        return TRFJob(est, y, xs, tstart, tstop, fwd, cov, key=ctx.key())

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
        elif not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
            raise NotImplementedError(f"{term.string}: loading {type(predictor).__name__} is not supported")

        # file predictor: build each stimulus' predictor from its file data at the
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

    def _aligned_predictor(self, ctx: Request, predictor: UTSPredictor | NUTSPredictor, term: Term, stim: str, time, filter_x: bool | str) -> NDVar:
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


# Options shared by the TRF-dataset nodes: the :class:`TRFDerivative` options that
# select the fit, plus the dataset-shaping ``scale``, ``smooth``, and ``trfs``.
_TRF_DATASET_OPTIONS = {
    'x': OptionSpec(None, Model, normalize=Model.coerce),
    'tstart': 0.0,
    'tstop': 0.5,
    'estimator': 'boosting',
    'data': OptionSpec(None, DataSpec),
    'mask': None,
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
        Assembled epoch definitions (for :class:`EpochCollection` expansion).
    """
    name = 'trf-dataset'
    key_options = _TRF_DATASET_OPTIONS

    def __init__(
            self,
            root: str | Path,
            estimators: dict[str, Estimator],
            epochs: dict[str, object],
    ):
        self.root = Path(root)
        self.estimators = estimators
        self.epochs = epochs

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = ['subject', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'reference', 'raw', 'inv']
        if ctx.state['inv']:
            fields += ['cov', 'src', 'parc', 'adjacency', 'mrisubject', 'common_brain']
        return tuple(fields)

    def _estimator(self, ctx: Request) -> Estimator:
        return self.estimators[ctx.options['estimator']]

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
        if ctx.state['inv']:
            if not is_fake_mri(self.root / mri_dir(ctx.state)):
                deps.append(Dependency('source-morph'))
        elif smooth := ctx.options['smooth']:
            raise ValueError(f"{smooth=}: smoothing is only available for source-space data")
        return tuple(deps)

    def build(self, ctx: Request) -> Dataset:
        est = self._estimator(ctx)
        scale = ctx.options['scale']
        trfs = ctx.options['trfs']
        subject = ctx.state['subject']
        dss = []
        for epoch in self._epoch_names(ctx):
            res = ctx.load(epoch)
            ds = est._result_dataset(res, scale=scale, trfs=trfs)
            ds[:, 'epoch'] = epoch
            dss.append(ds)
        ds = combine(dss, name=ctx.options['x'].name)
        ds['subject'] = Factor([subject], repeat=ds.n_cases, random=True)
        # Morphing/smoothing
        if ctx.state['inv']:
            common_brain = ctx.state['common_brain']
            if is_fake_mri(self.root / mri_dir(ctx.state)):
                source_morph = None
            else:
                source_morph = ctx.load('source-morph')
            _post_process_trfs(ds, ctx.options['smooth'], common_brain, source_morph)
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
    key_options = _TRF_DATASET_OPTIONS

    def __init__(
            self,
            mri_subjects: dict[str, dict[str, str]],
            groups: dict[str, tuple[str, ...]],
    ):
        self.mri_subjects = mri_subjects
        self.groups = groups

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = ['group', 'mri', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'reference', 'raw', 'inv']
        if ctx.state['inv']:
            fields += ['cov', 'src', 'parc', 'adjacency', 'mrisubject', 'common_brain']
        return tuple(fields)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subjects': tuple(self.groups[ctx.state['group']])}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if not ctx.state['inv']:
            if smooth := ctx.options['smooth']:
                raise ValueError(f"{smooth=}: smoothing is only available for source-space data")
        # Smooth the combined dataset so that the source smoothing matrix is calculated only once
        subject_options = tuple(key for key in self.key_options if key != 'smooth')
        options = ctx.options_for('trf-dataset', *subject_options, smooth=None)
        return tuple(
            Dependency('trf-dataset', label=subject, state=_subject_state(ctx.state, subject, self.mri_subjects), options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        dss = [ctx.load(subject) for subject in self.groups[ctx.state['group']]]
        ds = combine(dss, to_list=True)
        _post_process_trfs(ds, ctx.options['smooth'])
        return ds
