from pathlib import Path
import warnings

from ..._data_obj import NDVar
from ..derivative_cache import Input, Request, file_fingerprint
from ..preprocessing import RawFilter, RawPipe, RawSource
from .model import Term, TRFModelError, parse_term
from .predictor import FilePredictor


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
    OPTION_DEFAULTS = {'code': None, 'tstep': None, 'tmin': None, 'n_samples': None, 'filter_x': False}

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
