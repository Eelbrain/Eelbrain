# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""TRF estimators: configurations selecting and parametrizing a fitting algorithm

An :class:`Estimator` is a :class:`Configuration` that both selects a fitting
algorithm (boosting or NCRF) and carries the algorithm-specific parameters.
TRF-general parameters (model, ``tstart``, ``tstop``, ``data``, ``mask``,
``samplingrate``) stay on :meth:`Pipeline.load_trf`; estimator-specific
parameters (``basis``, ``delta``, ``mu``, …) live on the estimator.
"""
from typing import Literal
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
    """Base class for TRF estimators

    Subclasses select a fitting algorithm and declare its parameters through
    :attr:`DICT_ATTRS`. Cross-validation / partitioning is owned by each
    estimator because the schemes differ between algorithms.
    """
    # Dependencies (registered derivative names) that :meth:`_fit` needs beyond
    # the response, loaded by the TRF node and passed as keyword arguments.
    extra_inputs: tuple[str, ...] = ()
    # Whether the estimator requires sensor-space data (inv=''; e.g., NCRF)
    requires_sensor_space: bool = False
    # Fit-quality metric columns this estimator contributes to the TRFs Dataset.
    metric_keys: tuple[str, ...] = ()

    @property
    def interpolate_bads(self) -> bool:
        "Whether bad channels should be interpolated when loading sensor-space epochs"
        return False

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
            tstart: float,
            tstop: float,
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
            Start of the TRF in seconds.
        tstop
            Stop of the TRF in seconds.
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
    tstart
        Not set here; see :meth:`Pipeline.load_trf`.
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
        Scale ``y`` and ``x`` before fitting; ``'inplace'`` to save memory.
    partitions
        Number of partitions for cross-validation. ``None`` to infer from the
        number of cases; a negative value concatenates the cases and uses
        ``-partitions`` partitions (``-1`` to let boosting infer them).
    cv
        Use cross-validation (hold out a test partition).
    partition_results
        Keep the result for each test partition.
    backward
        Fit a backward model (predict the stimulus from the response). Only
        valid with a single-term model.
    """
    DICT_ATTRS = ('basis', 'basis_window', 'error', 'delta', 'mindelta', 'selective_stopping', 'scale_data', 'partitions', 'cv', 'partition_results', 'backward')
    metric_keys = ('r', 'z', 'residual', 'det', 'r1', 'z1')

    def __init__(
            self,
            basis: float = 0.050,
            basis_window: str = 'hamming',
            error: Literal['l1', 'l2'] = 'l1',
            delta: float = 0.005,
            mindelta: float = None,
            selective_stopping: int = 0,
            scale_data: bool | Literal['inplace'] = 'inplace',
            partitions: int = None,
            cv: bool = True,
            partition_results: bool = False,
            backward: bool = False,
    ):
        self.basis = typed_arg(basis, float)
        self.basis_window = basis_window
        self.error = error
        self.delta = typed_arg(delta, float)
        self.mindelta = typed_arg(mindelta, float)
        self.selective_stopping = selective_stopping
        self.scale_data = scale_data
        self.partitions = partitions
        self.cv = cv
        self.partition_results = partition_results
        self.backward = backward

    @property
    def interpolate_bads(self) -> bool:
        # A forward model predicts the sensor response, so bad channels must be interpolated to
        # a consistent set; a backward model predicts the stimulus and uses the channels as-is.
        return not self.backward

    def _fit(self, y, xs, tstart, tstop, *, fwd=None, cov=None):
        partitions = self.partitions
        if partitions is not None and partitions < 0:
            partitions = None if partitions == -1 else -partitions
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
        return boosting(y, x, tstart, tstop, self.scale_data, self.delta, self.mindelta, self.error, self.basis, self.basis_window, partitions=partitions, test=int(self.cv), selective_stopping=self.selective_stopping, partition_results=self.partition_results)

    def _result_metrics(self, result) -> dict[str, NDVar | float]:
        r = result.r
        metrics = {'r': r, 'z': arctanh(r), 'residual': result.residual, 'det': result.proportion_explained}
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
    nlevels
        Number of levels for the lead-field decomposition.
    n_iter
        Number of iterations.
    n_iterc
        Number of coordinate-descent iterations.
    n_iterf
        Number of FASTA iterations.
    n_splits
        Number of cross-validation splits for ``mu='auto'``.
    tol
        Convergence tolerance.
    use_ES
        Use the early-stopping strategy.
    basis_std
        Standard deviation of the temporal basis (in seconds).
    """
    extra_inputs = ('fwd', 'cov')
    requires_sensor_space = True
    DICT_ATTRS = ('mu', 'nlevels', 'n_iter', 'n_iterc', 'n_iterf', 'n_splits', 'tol', 'use_ES', 'basis_std')
    metric_keys = ('mu',)

    def __init__(
            self,
            mu: str | float | Sequence[float] = 'auto',
            nlevels: int = 1,
            n_iter: int = 10,
            n_iterc: int = 10,
            n_iterf: int = 100,
            n_splits: int = 3,
            tol: float = 0.001,
            use_ES: bool = False,
            basis_std: float = 0.0085,
    ):
        self.mu = mu
        self.nlevels = nlevels
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf
        self.n_splits = n_splits
        self.tol = typed_arg(tol, float)
        self.use_ES = use_ES
        self.basis_std = typed_arg(basis_std, float)

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
        if len(xs) == 1:
            x = xs[0]
        else:
            x = xs
        from ncrf import fit_ncrf
        return fit_ncrf(y, x, fwd, cov, tstart, tstop, nlevels=self.nlevels, n_iter=self.n_iter, n_iterc=self.n_iterc, n_iterf=self.n_iterf, normalize=True, in_place=True, mu=self.mu, tol=self.tol, n_splits=self.n_splits, use_ES=self.use_ES, basis_std=self.basis_std)

    def _result_metrics(self, result) -> dict[str, NDVar | float]:
        return {'mu': result.mu}

    def _result_tstep(self, result) -> float:
        return result.tstep
