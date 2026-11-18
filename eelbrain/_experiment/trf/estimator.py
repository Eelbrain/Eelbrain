# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""TRF estimators: configurations selecting and parametrizing a fitting algorithm

An :class:`Estimator` is a :class:`Configuration` that both selects a fitting
algorithm (boosting or NCRF) and carries the algorithm-specific parameters.
TRF-general parameters (model, ``tstart``, ``tstop``, ``data``,
``samplingrate``) stay on :meth:`Pipeline.load_trf`; estimator-specific
parameters (``basis``, ``delta``, ``mu``, …) live on the estimator.
"""
from numbers import Real
from typing import Any, Literal
from collections.abc import Sequence

import numpy as np

from ..._data_obj import Dataset, Datalist, NDVar, Var
from ..._ndvar import concatenate
from ..._trf._boosting import boosting
from ..configuration import Configuration, typed_arg


def arctanh(r: NDVar | float) -> NDVar | float:
    "Fisher z-transform of a correlation (preserving :class:`NDVar` dims)"
    if isinstance(r, NDVar):
        return NDVar(np.arctanh(r.x), r.dims, r.name, {'unit': 'z(r)'})
    return np.arctanh(r)


class Estimator(Configuration):
    """Base class for TRF estimators"""
    # Subclasses select a fitting algorithm and declare its parameters through ``DICT_ATTRS``.
    # The cross-validation parameters are those of :func:`eelbrain.split_data`; each estimator owns their defaults and validation, because the constraints differ between algorithms.

    # Dependencies (registered derivative names) that :meth:`_fit` needs beyond
    # the response, loaded by the TRF node and passed as keyword arguments.
    extra_inputs: tuple[str, ...] = ()
    extra_input_fields: tuple[str, ...] = ()
    # Whether the estimator requires sensor-space data (inv=''; e.g., NCRF)
    requires_sensor_space: bool = False
    # Fit-quality metric columns this estimator contributes to the TRFs Dataset.
    metric_keys: tuple[str, ...] = ()
    # Whether the cases are concatenated to continuous data before fitting
    _concatenate_cases: bool = False

    @property
    def interpolate_bads(self) -> bool:
        "Whether bad channels should be interpolated when loading sensor-space epochs"
        return False

    @property
    def _split_args(self) -> dict[str, Any]:
        "Cross-validation parameters, as keyword arguments for :func:`eelbrain.split_data`"
        raise NotImplementedError

    def _result_metrics(self, result) -> dict[str, NDVar | float]:
        """Fit-quality metrics for one result, keyed by output-Dataset column.

        Subclasses return the columns named in :attr:`metric_keys` that apply to
        ``result`` (e.g. boosting omits the vector-only ``r1``/``z1`` for scalar
        data).
        """
        raise NotImplementedError

    def _result_tstep(self, result) -> float:
        "Time-step of the estimated TRF (for the output ``samplingrate``)"
        raise NotImplementedError

    def _result_kernels(self, result, *, scale: str) -> list[NDVar]:
        "TRF kernels for one result as a list of :class:`NDVar`"
        if scale == 'original':
            h = result.h_scaled
        elif scale is None:
            h = result.h
        else:
            raise ValueError(f"{scale=}")
        return [h] if isinstance(h, NDVar) else list(h)

    def _result_dataset(self, result, *, scale: str, trfs: bool) -> Dataset:
        """Single-case :class:`Dataset` of fit metrics and (optionally) TRF kernels.

        ``ds.info['metrics']`` lists the metric columns and ``ds.info['xs']`` the
        kernel columns (both consumed by group morphing and smoothing).

        Parameters
        ----------
        result
            A single fitted result (e.g. :class:`~eelbrain.BoostingResult`).
        scale
            Kernel scaling (``None`` or ``'original'``, see
            :meth:`Pipeline.load_trfs`).
        trfs
            Include the TRF kernels (set ``False`` to load metrics only).
        """
        ds = Dataset()
        metrics = self._result_metrics(result)
        for key, value in metrics.items():
            ds[key] = value[np.newaxis] if isinstance(value, NDVar) else Var([value])
        xs = []
        if trfs:
            for h in self._result_kernels(result, scale=scale):
                key = Dataset.as_key(h.name)
                ds[key] = h[np.newaxis]
                xs.append(key)
        ds.info['metrics'] = list(metrics)
        ds.info['xs'] = xs
        ds.info['samplingrate'] = 1. / self._result_tstep(result)
        return ds

    def _fit(
            self,
            y: NDVar | Datalist,
            xs: list[NDVar | Datalist],
            tstart: float | list[float],
            tstop: float | list[float],
            *,
            fwd: NDVar = None,
            cov=None,
    ):
        """Fit the TRF and return the result object.

        Parameters
        ----------
        y
            Response (a single :class:`NDVar`, or a :class:`Datalist` of
            :class:`NDVar` for variable-length epochs).
        xs
            Predictors, one entry per model term (each a :class:`NDVar` or
            :class:`Datalist` matching ``y``).
        tstart
            Start of the TRF in seconds (or one value per predictor).
        tstop
            Stop of the TRF in seconds (or one value per predictor).
        fwd
            Forward solution (NCRF only).
        cov
            Noise covariance (NCRF only).
        """
        raise NotImplementedError


class Boosting(Estimator):
    """Boosting estimator

    Parameters
    ----------
    basis
        Width of the basis window for the response function in seconds.
    basis_window
        Window shape for the basis (see :func:`eelbrain.boosting`).
    error
        Error function: ``'l1'`` or ``'l2'``.
    delta
        Boosting step size.
    mindelta
        If the error for the training data can't be reduced, divide ``delta``
        in half until it is smaller than ``mindelta``.
    selective_stopping
        Stop boosting each predictor separately (see :func:`eelbrain.boosting`).
    scale_data
        Normalize ``y`` and ``x`` before fitting.
    partitions
        Number of partitions for cross-validation. The cross-validation
        parameters (``partitions``, ``split``, ``validate``, ``test`` and
        ``min_section``) are described in :func:`eelbrain.split_data`; use
        :meth:`Pipeline.show_partitions` to inspect how they partition the
        data.
    split
        How to form partitions: ``'time'`` (default) forms partitions from
        contiguous spans of time, ``'cases'`` from whole cases,
        ``'within-cases'`` from pieces of every case, and ``'walk-forward'``
        holds out windows at the end of every case (requires ``test=0``).
        With ``None``, use ``'cases'`` for data with more than one case and
        ``'time'`` for continuous data.
    validate
        Size of each validation set, which is used for early stopping. Only
        used with ``split='walk-forward'``: the length of each validation set
        as proportion of the case (default 0.05). With the other schemes, the
        validation set is always one partition.
    test
        Number of partitions in each test set (0 or 1). With ``test=1``
        (default), each partition is held out as test set in turn, and fit
        metrics reflect how well the TRFs predict data that were not used to
        estimate them. With ``test=0``, TRFs are estimated from all data, and
        fit metrics (e.g., ``r`` and ``ev``) are not available, because they
        would be computed from the same data that were used to estimate the
        TRF, and overestimate the model's predictive power. Model comparisons
        (e.g., :meth:`Pipeline.load_model_test`) thus require ``test=1``. See
        the ``test`` parameter of :func:`eelbrain.boosting` for details.
    min_section
        With ``split='time'``, minimum duration (in seconds) of the pieces
        left when a case is cut (default 15).
    partition_results
        Keep the result for each test partition (or, with ``test=0``, for
        each validation partition).
    backward
        Fit a backward model (predict the stimulus from the response). Only
        valid with a single-term model.
    """
    DICT_ATTRS = ('basis', 'basis_window', 'error', 'delta', 'mindelta', 'selective_stopping', 'scale_data', 'partitions', 'split', 'validate', 'test', 'min_section', 'partition_results', 'backward')
    DICT_DEFAULTS = {'split': None, 'validate': None, 'min_section': None}  # added after 0.43.0b4; the values that reproduce the cross-validation of results cached before

    def __init__(
            self,
            basis: float = 0.050,
            basis_window: str = 'hamming',
            error: Literal['l1', 'l2'] = 'l1',
            delta: float = 0.005,
            mindelta: float = None,
            selective_stopping: int = 0,
            scale_data: bool = True,
            partitions: int | None = 5,
            split: Literal['cases', 'time', 'within-cases', 'walk-forward'] | None = 'time',
            validate: float = None,
            test: int = 1,
            min_section: float = None,
            partition_results: bool = False,
            backward: bool = False,
    ):
        self.basis = typed_arg(basis, float)
        self.basis_window = basis_window
        self.error = error
        self.delta = typed_arg(delta, float)
        self.mindelta = typed_arg(mindelta, float, allow_none=True)
        self.selective_stopping = typed_arg(selective_stopping, int)
        self.scale_data = typed_arg(scale_data, bool)
        if test not in (0, 1):
            raise ValueError(f"{test=}: needs to be 0 or 1")
        elif validate == 0:
            raise ValueError(f"{validate=}: boosting requires a validation set for early stopping")
        elif split == 'walk-forward' and test:
            raise ValueError(f"{split=} with {test=}: the walk-forward scheme has no test set; use test=0")
        if partitions is not None and partitions < 0:
            # legacy (results cached up to 0.43.0b4): concatenate the cases to continuous data and form abs(partitions) partitions of equal length
            if split not in (None, 'time'):
                raise ValueError(f"{partitions=} with {split=}")
            split = None  # as stored for those results
        if partitions is not None and abs(partitions) < 2 + test:
            raise ValueError(f"{partitions=}: need at least {2 + test} partitions with {test=}")
        self.partitions = partitions
        self.split = split
        self.validate = typed_arg(validate, float, allow_none=True)
        self.test = int(test)
        self.min_section = typed_arg(min_section, float, allow_none=True)
        self.partition_results = partition_results
        self.backward = backward

    @property
    def metric_keys(self) -> tuple[str, ...]:
        # boosting computes fit metrics only with a test set
        return ('r', 'z', 'residual', 'ev', 'r1', 'z1') if self.test else ()

    @property
    def interpolate_bads(self) -> bool:
        # A forward model predicts the sensor response, so bad channels must be interpolated to
        # a consistent set; a backward model predicts the stimulus and uses the channels as-is.
        return not self.backward

    @property
    def _concatenate_cases(self) -> bool:
        return self.partitions is not None and self.partitions < 0

    @property
    def _split_args(self) -> dict[str, Any]:
        if self.partitions is None:
            partitions = None
        else:
            partitions = abs(self.partitions)
        return {'partitions': partitions, 'split': self.split, 'validate': self.validate, 'test': self.test, 'min_section': self.min_section}

    def _fit(self, y, xs, tstart, tstop, *, fwd=None, cov=None):
        if self._concatenate_cases:
            y = concatenate(y)
            xs = [concatenate(x) for x in xs]
        if len(xs) == 1:
            x = xs[0]
            if self.backward:
                y, x = x, y
        elif self.backward:
            raise ValueError("backward model with more than one predictor")
        else:
            names = [xi.name for xi in xs]
            if len(set(names)) < len(names):
                raise ValueError(f"Multiple predictors with the same name: {', '.join(names)}")
            x = xs
        scale_data = 'inplace' if self.scale_data else False
        return boosting(y, x, tstart, tstop, scale_data, self.delta, self.mindelta, self.error, self.basis, self.basis_window, selective_stopping=self.selective_stopping, partition_results=self.partition_results, **self._split_args)

    def _result_metrics(self, result) -> dict[str, NDVar | float]:
        if not self.test:
            return {}
        r = result.r
        metrics = {'r': r, 'z': arctanh(r), 'residual': result.residual, 'ev': result.proportion_explained}
        if result.r_l1 is not None:  # vector data
            r1 = result.r_l1
            metrics['r1'] = r1
            metrics['z1'] = arctanh(r1)
        return metrics

    def _result_tstep(self, result) -> float:
        h = result.h_source
        if not isinstance(h, NDVar):
            h = h[0]
        return h.time.tstep


class NCRF(Estimator):
    """Neuro-Current Response Function estimator

    Fits the TRF directly in source space from sensor data using the ``ncrf``
    package. The ``data`` argument of :meth:`Pipeline.load_trf` must be left
    unset for NCRF (sensor data is used and localized internally), and the
    forward solution and noise covariance are loaded automatically.

    Parameters
    ----------
    mu
        Regularization parameter (``'auto'`` to determine through
        cross-validation, or a numeric value / sequence of values).
    basis_stride
        Spacing between neighboring Gabor basis atoms, in samples.
    n_iter
        Number of outer iterations.
    n_iterc
        Number of Champagne iterations within each outer iteration.
    n_iterf
        Number of FASTA iterations within each outer iteration.
    partitions
        Number of partitions for the cross-validation that selects ``mu``
        (unless ``mu`` is a single value). The cross-validation parameters
        (``partitions``, ``split``, ``validate`` and ``min_section``) are
        described in :func:`eelbrain.split_data`; use
        :meth:`Pipeline.show_partitions` to inspect how they partition the
        data. With the default ``split``, the last ``partitions - 1``
        partitions of every case are held out one at a time (default 4, i.e.,
        3 splits).
    split
        How to form partitions. The default (``'walk-forward'``) holds out
        windows at the end of every case, each with only the data preceding it
        as training set.
    validate
        Size of each validation set. For ``split='walk-forward'``, the length
        of each validation set as proportion of the case (default 0.05).
    min_section
        With ``split='time'``, minimum duration (in seconds) of the pieces
        left when a case is cut (default 15).
    tol
        Convergence tolerance.
    use_ES
        Refine the cross-validated ``mu`` with the estimation stability criterion.
    basis_std
        Standard deviation of the Gaussian basis atoms (in seconds).

    Notes
    -----
    The fit-quality metrics (see :meth:`Pipeline.load_trfs`) are the selected
    ``mu`` and the training-set scores of the :class:`ncrf.NCRFFit`.
    """
    extra_inputs = ('fwd', 'cov')
    extra_input_fields = ('cov', 'mrisubject', 'src')
    requires_sensor_space = True
    DICT_ATTRS = ('mu', 'basis_stride', 'n_iter', 'n_iterc', 'n_iterf', 'partitions', 'split', 'validate', 'min_section', 'tol', 'use_ES', 'basis_std')
    metric_keys = ('mu', 'explained_variance', 'l2_error', 'cross_fit', 'weighted_l2_error')

    def __init__(
            self,
            mu: str | float | Sequence[float] = 'auto',
            basis_stride: int = 1,
            n_iter: int = 10,
            n_iterc: int = 10,
            n_iterf: int = 100,
            partitions: int | None = None,
            split: Literal['cases', 'time', 'within-cases', 'walk-forward'] | None = 'walk-forward',
            validate: float = None,
            min_section: float = None,
            tol: float = 0.001,
            use_ES: bool = False,
            basis_std: float = 0.0085,
    ):
        self.mu = mu
        self.basis_stride = typed_arg(basis_stride, int)
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf
        self.partitions = typed_arg(partitions, int, allow_none=True)
        self.split = split
        self.validate = typed_arg(validate, float, allow_none=True)
        self.min_section = typed_arg(min_section, float, allow_none=True)
        self.tol = typed_arg(tol, float)
        self.use_ES = use_ES
        self.basis_std = typed_arg(basis_std, float)

    @property
    def _split_args(self) -> dict[str, Any]:
        # no test set: NCRF cross-validates only to select mu
        return {'partitions': self.partitions, 'split': self.split, 'validate': self.validate, 'test': 0, 'min_section': self.min_section}

    def _fit(self, y, xs, tstart, tstop, *, fwd=None, cov=None):
        if fwd is None or cov is None:
            raise RuntimeError("NCRF requires a forward solution and noise covariance")
        # NCRF assumes the data contains a subset of the sensors in the covariance
        y0 = y[0] if isinstance(y, Datalist) else y
        if set(y0.sensor.names).difference(cov.ch_names):
            if isinstance(y, Datalist):
                y = Datalist([yi.sub(sensor=cov.ch_names) for yi in y])
            else:
                y = y.sub(sensor=cov.ch_names)
        x = xs[0] if len(xs) == 1 else xs
        # Use component API because xs is organized by predictor [pred1, pred2, ...] whereas fit_ncrf expects xs organized by trial
        from ncrf import ChampLasso, NCRFEstimator, RegressionData
        data = RegressionData.from_data(y, x, tstart, tstop, self.basis_stride, basis_std=self.basis_std)
        if isinstance(self.mu, Real):
            splits = None  # a single mu is not cross-validated
        else:
            data.data.initialize_cross_validation(**self._split_args)
            splits = data.data.splits
        estimator = NCRFEstimator.from_lead_field(fwd, cov)
        solver = ChampLasso(mu=self.mu, n_iter=self.n_iter, n_iterc=self.n_iterc, n_iterf=self.n_iterf, tol=self.tol, use_es=self.use_ES)
        return estimator.fit(data, solver, cv=splits, compute_explained_variance=True)

    # ``result`` is an :class:`ncrf.NCRFFit` report; the kernels and TRF timing live on its ``model``
    def _result_metrics(self, result) -> dict[str, NDVar | float]:
        return {'mu': result.solver.mu, **result.scores}

    def _result_tstep(self, result) -> float:
        return result.model.design.tstep

    def _result_kernels(self, result, *, scale: str) -> list[NDVar]:
        return super()._result_kernels(result.model, scale=scale)
