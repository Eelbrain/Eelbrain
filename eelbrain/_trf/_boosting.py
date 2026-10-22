"""Boosting as described by David et al. (2007).

Versions
--------
Stored in ``algorithm_version`` attribute (see docstring)

Profiling
---------
ds = datasets._get_continuous()
y = ds['y']
x1 = ds['x1']
x2 = ds['x2']

%prun -s cumulative res = boosting(y, x1, 0, 1)

"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cached_property, reduce
import inspect
from itertools import chain, repeat
from math import ceil
from operator import mul
import platform
import sys
import time
from typing import Literal
from collections.abc import Callable, Sequence

import mne
import numpy as np
from numpy import newaxis

from .._config import CONFIG
from .._data_obj import Case, Dataset, Dimension, SourceSpaceBase, NDVar, CategorialArg, NDVarArg, UTS, dataobj_repr
from .._exceptions import OldVersionError
from .._ndvar.ndvar import _concatenate_values, set_adjacency, set_parc
from .._ndvar._convolve import convolve_1d, convolve_2d
from .._utils import PickleableDataClass, deprecate_ds_arg, user_activity
from .shared import PredictorData, DeconvolutionData, Split, Splits, merge_segments
from ._fit_metrics import get_evaluators
from . import _boosting_opt as opt


def to_array(ndvar: NDVar | tuple[NDVar, ...] | float) -> np.ndarray:
    if isinstance(ndvar, NDVar):
        return ndvar.x.ravel()
    elif isinstance(ndvar, tuple):
        return np.concatenate([to_array(x) for x in ndvar])
    else:
        return np.array([ndvar])


def to_index(ndvar: Sequence[NDVar | float]) -> list[int | slice]:
    out = []
    i0 = 0
    for x in ndvar:
        if isinstance(x, NDVar):
            i1 = i0 + reduce(mul, x.shape, 1)
            out.append(slice(i0, i1))
            i0 = i1
        else:
            out.append(i0)
            i0 += 1
    return out


@dataclass(eq=False)
class BoostingResult(PickleableDataClass):
    """Result from boosting

    Fit metrics are computed from all time points in ``y``.

    - For models estimated with cross-validation (``test=1``, the default),
      ``y_pred`` in each test segment is predicted from the averaged TRF from
      the corresponding training runs (each non-test segment used as
      validation set once), and the different test segments are then
      concatenated to compute the fit-metrics in comparison with ``y``.
    - For models estimated with ``test=0``, fit metrics are not computed
      (``None``), because they could only be computed from the same data that
      were used to estimate the TRF, and would overestimate the model's
      predictive power (see the ``test`` parameter of :func:`boosting`).
      With ``partition_results=True``, the fit of each run to its validation
      and training data is available in :attr:`BoostingResult.partition_results`.


    Attributes
    ----------
    h : NDVar | tuple of NDVar
        The temporal response function (the average of all TRFs from the
        different runs/partitions).
        Whether ``h`` is an :class:`NDVar` or a :class:`tuple` of :class:`NDVar`
        depends on whether the ``x`` parameter to :func:`boosting` was an
        :class:`NDVar` or a sequence of :class:`NDVar`.
    h_scaled : NDVar | tuple of NDVar
        ``h`` scaled such that it applies to the original input ``y`` and ``x``.
        If boosting was done with ``scale_data=False``, ``h_scaled`` is the same
        as ``h``.
    h_source : NDVar | tuple of NDVar
        If ``h`` was constructed using a basis, ``h_source`` represents the
        source of ``h`` before being convolved with the basis.
    h_time : UTS
        Time dimension of the kernel.
    r
        Correlation between the measured response ``y`` and the predicted
        response ``h * x``. With cross-validation (the default,
        :func:`boosting` with ``test=1``), each partition of ``y`` is
        predicted using the ``h`` estimated from the corresponding training
        partitions. With ``test=0``, ``r`` is ``None`` (see above).
        For vector data, measured and predicted responses are normalized, and ``r``
        is computed as the average dot product over time.
        The type of ``r`` depends on the ``y`` parameter to :func:`boosting`:
        If ``y`` is one-dimensional, ``r`` is scalar, otherwise it is a :class:`NDVar`.
        Note that ``r`` does not take into account the model's ability to predict
        the magnitude of the response, only its shape; for a measure that reflects both,
        consider using ``proportion_explained``.
    r_rank : float | NDVar
        As ``r``, the Spearman rank correlation.
    t_run
        Time it took to run the boosting algorithm (in seconds).
    error
        The error evaluation method used.
    residual : float | NDVar
        The residual of the final result

        - ``error='l1'``: the sum of the absolute differences between ``y`` and
          ``h * x``.
        - ``error='l2'``: the sum of the squared differences between ``y`` and
          ``h * x``.

        For vector ``y``, the error is defined based on the distance in space
        for each data point.
    delta
        Kernel modification step used.
    mindelta
        Mindelta parameter used.
    n_samples
        Number of samples in the data described by the fit metrics (all data
        unless only a single test partition was evaluated).
    proportion_explained : float | NDVar
        The proportion of the variation in ``y`` that is explained by the model.
        Calculated as ``1 - (variation(residual) / variation(y))``.
        Variation is caculated as the ``l1`` or ``l2`` norm,
        depending on the ``error`` that was used for model fitting.
        Note that this does not correspond to ``r**2`` even for ``error='l2'``,
        because residuals are not guaranteed to be orthogonal to predictions.
        As for ``r``, this is ``None`` for models estimated with ``test=0``.
    scale_data
        Scale_data parameter used.
    y_mean
        Mean that was subtracted from ``y``.
    y_scale
        Scale by which ``y`` was divided.
    x_mean
        Mean that was subtracted from ``x``.
    x_scale
        Scale by which ``x`` was divided.
    splits
        Data splits used for cross-validation.
        Use :meth:`Splits.plot` to visualize the cross-validation scheme.
    partition_results
        If :func:`boosting` is called with ``partition_results=True``, this
        attribute contains the results for the individual test partitions
        (``test=1``), or for the individual validation partitions (``test=0``).
        For validation partitions, fit metrics (e.g., ``r``) reflect the fit to
        the validation data, and ``train_*`` attributes the fit to the training
        data. Note that the fit to the validation data is also biased upward,
        because the validation data determined when boosting stopped.
        :meth:`BoostingResult.partition_result_data` combines them in a
        :class:`Dataset`.
    i_test
        Index of the test partition whose data the fit metrics describe (see
        :attr:`Split.i_test`): set for results in
        :attr:`BoostingResult.partition_results` of models estimated with
        ``test=1``, and for results from :meth:`Boosting.evaluate_fit` with
        ``i_test``. ``None`` when fit metrics pool all test partitions.
    i_validate
        For results in :attr:`BoostingResult.partition_results` of models
        estimated with ``test=0``: index of the validation partition (see
        :attr:`Split.i_validate`).
    train_r, train_r_rank, train_r_l1, train_residual, train_proportion_explained
        Fit to the training data (only for results in
        :attr:`BoostingResult.partition_results` of models estimated with
        ``test=0``).
    algorithm_version
        Version of the algorithm with which the model was estimated

          - -1: results from before this attribute was added
          - 0: Normalize ``x`` after applying basis
          - 1: Numba implementation
          - 2: Cython multiprocessing implementation (Eelbrain 0.38)
          - 3: Cython based convolution (Eelbrain 0.40)

    execution_context
        Information on the context in which the model was estimated.


    Examples
    --------
    To compare the fit of models estimated with different loss metrics
    (``error`` parameter) calculate the proportion of explained variance,
    for example::

        data = datasets._get_continuous()
        trf_l1 = boosting('y', 'x1', 0, 1, data=data, error='l1', partitions=3)
        trf_l2 = boosting('y', 'x1', 0, 1, data=data, error='l2', partitions=3)
        l1_explained_variance = 1 - (trf_l1.l2_residual / trf_l1.l2_total)
        l2_explained_variance = 1 - (trf_l2.l2_residual / trf_l2.l2_total)

    """
    # basic parameters
    y: str | None
    x: str | None | tuple[str | None]
    tstart: float | tuple[float, ...]
    tstop: float | tuple[float, ...]
    scale_data: bool
    delta: float
    mindelta: float
    error: str
    selective_stopping: int
    # data properties
    y_mean: NDVar | float
    y_scale: NDVar | float
    x_mean: NDVar | float | tuple[NDVar | float, ...]
    x_scale: NDVar | float | tuple[NDVar | float, ...]
    # results
    _h: NDVar | tuple[NDVar, ...]
    _isnan: np.ndarray
    t_run: float
    # advanced parameters
    basis: float
    basis_window: str

    splits: Splits = None
    # advanced data properties
    n_samples: int = None
    _y_info: dict = field(default_factory=dict)
    _y_dims: tuple[Dimension, ...] = None
    # fit metrics
    i_test: int = None  # test partition for fit metrics
    i_validate: int = None  # validation partition (partition results without test set)
    l1_residual: float | NDVar = None
    l2_residual: float | NDVar = None
    l1_total: float | NDVar = None
    l2_total: float | NDVar = None
    r: float | NDVar = None
    r_rank: float | NDVar = None
    r_l1: NDVar = None
    # training fit metrics (partition results of models without test set)
    train_l1_residual: float | NDVar = None
    train_l2_residual: float | NDVar = None
    train_l1_total: float | NDVar = None
    train_l2_total: float | NDVar = None
    train_r: float | NDVar = None
    train_r_rank: float | NDVar = None
    train_r_l1: NDVar = None
    partition_results: list[BoostingResult] = None
    # store the version of the boosting algorithm with which model was fit
    version: int = 16  # file format (updates when re-saving)
    algorithm_version: int = -1  # do not change when re-saving
    execution_context: dict[str, str | tuple[str, ...]] = None  # do not change when re-saving
    # debug parameters
    y_pred: NDVar = None
    fit: Boosting = None

    def __post_init__(self):
        if self.splits is None:
            self.partitions = None
        else:
            self.partitions = self.splits.n_partitions

    def __setstate__(self, state: dict):
        # backwards compatibility
        version = state.pop('version')
        if version < self.version:
            if version == 7:
                state['partitions'] = state.pop('n_partitions')
                state['partitions_arg'] = state.pop('n_partitions_arg')
            if version < 9:
                state['residual'] = state.pop('fit_error')
                if state.pop('prefit', None):
                    raise OSError('Boosting result used the prefit functionality that has been removed. Use an older version of eelbrain to open this result.')
            if version < 11:
                for key in ['partitions_arg', 'h', 'isnan', 'y_info']:
                    state[f'_{key}'] = state.pop(key, None)
                state['r_rank'] = state.pop('spearmanr')
            if version < 12:
                # Vector residuals are averaged
                if state['r_rank'] is None:
                    state['residual'] *= state['n_samples']
            if version < 13:
                state['splits'] = Splits(None, state.pop('_partitions_arg'), state.pop('partitions'), state.pop('validate', 1), state.pop('test', 0), state.pop('model'))
            if version < 14:
                # state[f"y_{state['error']}_scale"] = state.pop('y_scale')
                state[f"{state['error']}_residual"] = state.pop('residual')
            if version == 14 and 'eelbrain_version' in state:
                state['execution_context'] = {'eelbrain_version': state.pop('eelbrain_version')}
            if version < 15:
                if state.pop('prefit', None):
                    raise OSError('Boosting result used the prefit functionality that has been removed. Use an older version of eelbrain to open this result.')
            if version < 16 and state.get('i_test') is not None:
                # n_samples reflects all data, which is wrong for a single test partition
                state['n_samples'] = None
        PickleableDataClass.__setstate__(self, state)

    def __repr__(self):
        items = []
        if isinstance(self.tstart, tuple):
            x = ' + '.join(f'{x} ({t1:g} - {t2:g})' for x, t1, t2 in zip(self.x, self.tstart, self.tstop))
        else:
            if self.x is None or isinstance(self.x, str):
                x = self.x
            else:
                x = ' + '.join(map(str, self.x))
            items.append(f'{self.tstart:g} - {self.tstop:g}')
        items.insert(0, f'{self.__class__.__name__} {self.y} ~ {x}')
        for name, param in inspect.signature(boosting).parameters.items():
            if param.default is inspect.Signature.empty or name == 'data':
                continue
            elif name == 'debug':
                continue
            elif name == 'partition_results':
                value = bool(self.partition_results)
            elif name == 'partitions':
                value = None if self.splits is None else self.splits.partitions_arg
            elif name == 'model':
                if self.splits is None or self.splits.model is None:
                    continue
                value = dataobj_repr(self.splits.model)
            elif name == 'validate':
                value = None if self.splits is None else self.splits.n_validate
            elif name == 'test':
                value = None if self.splits is None else self.splits.n_test
            else:
                value = getattr(self, name)
            if value is not None and value != param.default:
                items.append(f'{name}={value}')
        return f"<{', '.join(items)}>"

    @cached_property
    def _x_mean_array(self):
        return to_array(self.x_mean)

    @cached_property
    def _x_scale_array(self):
        return to_array(self.x_scale)

    @cached_property
    def h(self):
        if not self.basis:
            return self._h
        elif isinstance(self._h, tuple):
            return tuple(h.smooth('time', self.basis, self.basis_window, 'full') for h in self._h)
        else:
            return self._h.smooth('time', self.basis, self.basis_window, 'full')

    @cached_property
    def h_scaled(self) -> NDVar | tuple[NDVar, ...]:
        if self.y_scale is None:
            return self.h
        elif isinstance(self.h, NDVar):
            out = self.h * (self.y_scale / self.x_scale)
            out.info = self._y_info.copy()
            return out
        else:
            out = []
            for h, sx in zip(self.h, self.x_scale):
                h = h * (self.y_scale / sx)
                h.info = self._y_info.copy()
                out.append(h)
            return tuple(out)

    @cached_property
    def h_source(self) -> NDVar | tuple[NDVar, ...]:
        return self._h

    @cached_property
    def h_time(self) -> UTS:
        if isinstance(self.h, NDVar):
            return self.h.time
        else:
            return self.h[0].time

    @cached_property
    def residual(self) -> float | NDVar | None:
        return getattr(self, f'{self.error}_residual')

    @cached_property
    def train_residual(self) -> float | NDVar | None:
        return getattr(self, f'train_{self.error}_residual')

    @cached_property
    def train_proportion_explained(self) -> float | NDVar | None:
        if self.train_residual is None:
            return None
        return 1 - (self.train_residual / getattr(self, f'train_{self.error}_total'))

    @cached_property
    def _variability(self):
        # variability in the data
        total = getattr(self, f'{self.error}_total')
        if total is not None:
            # variability in the evaluated segments (e.g., one test partition)
            return total
        elif self.y_scale is None:
            raise NotImplementedError("Not implemented for scale_data=False")
        elif self.n_samples is None:
            raise OldVersionError("This is an older result object which did not store some necessary information; refit the model to use this attribute")
        else:
            # Due to the normalization:
            return self.n_samples

    @deprecate_ds_arg
    def cross_predict(
            self,
            x: NDVarArg | Sequence[NDVarArg] = None,
            data: Dataset = None,
            scale: Literal['original', 'normalized'] = 'original',
            name: str = None,
    ) -> NDVar:
        """Predict responses to ``x`` using complementary training data

        Parameters
        ----------
        x
            Predictors used in the original model fit, or a subset thereof.
            In order for cross-prediction to be accurate, ``x`` needs to match the ``x``
            used in the original fit exactly in cases and time.
        data
            Dataset with predictors. If ``ds`` is specified, ``x`` can be omitted.
        scale
            Return predictions at the scale of the original data (the ``y``
            supplied to the :func:`boosting` function) or at the normalized
            scale that is used for model fitting (``y`` and ``x`` normalized).
        name
            Name for the output :class:`NDVar`.

        See Also
        --------
        convolve : Simple prediction of linear model

        Notes
        -----
        This function does not adjust the mean across time of predicted
        responses; subtract the mean in order to compute explained variance.

        Examples
        --------
        Fit a TRF and reproduce the error using the cross-predict function::

            trf = boosting(y, x, 0, 0.5, partitions=5, partition_results=True)
            y_pred = trf.cross_predict(x, scale='normalized')

            y_normalized = (y - trf.y_mean) / trf.y_scale
            y_residual = y_normalized - y_pred
            proportion_explained_l1 = 1 - (y_residual.abs().sum('time') / y_normalized.abs().sum('time'))
            proportion_explained_l2 = 1 - ((y_residual ** 2).sum('time') / (y_normalized ** 2).sum('time'))

        """
        if scale not in ('original', 'normalized'):
            raise ValueError(f"{scale=}")
        if not self.partition_results:
            raise ValueError("BoostingResult does not contain partition-specific models; fit with partition_results=True")
        elif not self.splits.n_test:
            raise ValueError("Cross-prediction requires a model fit with cross-validation (test=1)")
        # predictors
        x_ = self.x if x is None else x
        x_data = PredictorData(x_, data, copy=True)
        # check predictors match h
        if x_data.x_name == self.x:
            x_use = None
            x_use_index = slice(None)
        else:
            if isinstance(self.x, str):
                raise ValueError(f'{x=} for {self}')
            elif x_data.multiple_x:
                x_use = x_data.x_name
            else:
                x_use = [x_data.x_name]
            missing = set(x_use).difference(self.x)
            if missing:
                raise ValueError(f"{x=}: has variables not in {self}:\nx: {', '.join(missing)}")
            x_use_orig_order = [xi for xi in self.x if xi in x_use]
            if x_use_orig_order != x_use:
                raise NotImplementedError("x in different order than original fit")
            # index for subset of self.x
            x_use_index = np.ones(self._x_mean_array.shape[0], bool)
            for index, name in zip(to_index(self.x_mean), self.x):
                x_use_index[index] = name in x_use
        # prepare output array
        if self._y_dims is None:  # only possible in results from dev version
            y_dims = self.y_mean.dims
        else:
            y_dims = self._y_dims
        y_dimnames = [dim.name for dim in y_dims]
        n_y = sum(len(dim) for dim in y_dims) or 1
        y_pred = np.empty((1, n_y, x_data.n_times_flat))
        # prepare x:  (n_x_only, n_shared, n_x_times)
        x_array = x_data.data
        if self.scale_data:
            x_mean = self._x_mean_array[x_use_index]
            x_scale = self._x_scale_array[x_use_index]
            x_array -= x_mean[:, newaxis]
            x_pads = -(x_mean / x_scale)[newaxis]
            x_array /= x_scale[:, newaxis]
        else:
            x_pads = np.zeros((1, len(x_array)))
        x_array = x_array[newaxis, :, :]
        # prepare h
        h_i_start = int(round(self.h_time.tmin / self.h_time.tstep))
        # iterate through partitions
        for result in self.partition_results:
            # find segments
            for split in self.splits.splits:
                if split.i_test == result.i_test:
                    segments = split.test
                    break
            else:
                raise RuntimeError(f"Split missing for test segment {result.i_test}")
            # h to flat array: (n_h_only == in y, n_shared == in x, n_h_times)
            hs = [result.h] if isinstance(result.h, NDVar) else result.h
            if x_use:
                hs = {h.name: h for h in hs}
                hs = [hs[x] for x in x_use]
            h_array = []
            for h, (_, xdims, _) in zip(hs, x_data.x_meta):
                dimnames = [*y_dimnames, *[dim.name for dim in xdims], 'time']
                h_data = h.get_data(dimnames)
                h_data = h_data.reshape((n_y, -1, h_data.shape[-1]))
                h_array.append(h_data)
            h_array = np.concatenate(h_array, 1)
            # convolution
            convolve_2d(h_array, x_array, x_pads, h_i_start, segments, y_pred)
        # package output
        if name is None:
            name = self.y

        if x_data.is_ragged:
            split_points = [0, *np.cumsum(x_data.n_times)]
            shape = [*[len(dim) for dim in y_dims], split_points[-1]]
            y_pred = y_pred.reshape(shape)
            y_pred = [y_pred[..., i0: i1] for i0, i1 in zip(split_points[:-1], split_points[1:])]
            y_pred = [NDVar(y_, [*y_dims, uts], name, self._y_info) for y_, uts in zip(y_pred, x_data.time_dim)]
            if scale == 'original' and self.scale_data:
                for y_ in y_pred:
                    y_ *= self.y_scale
                    y_ += self.y_mean
        else:
            dims = [*y_dims, x_data.time_dim]
            shape = [len(dim) for dim in dims]
            if x_data.case_to_segments:
                dims = (Case, *dims)
                shape.insert(-1, x_data.n_cases)
            y_pred = y_pred.reshape(shape)
            if x_data.case_to_segments:
                y_pred = np.rollaxis(y_pred, -2)
            y_pred = NDVar(y_pred, dims, name, self._y_info)
            if scale == 'original' and self.scale_data:
                y_pred *= self.y_scale
                y_pred += self.y_mean
        return y_pred

    def partition_result_data(self) -> Dataset:
        """Results from the different partitions in a :class:`Dataset`

        For models estimated with cross-validation (``test=1``), the
        :class:`Dataset` contains one case per test partition (``i_test``),
        with fit metrics for that test partition. For models estimated with
        ``test=0``, it contains one case per validation partition
        (``i_validate``), with fit metrics for the validation partition
        (``r``, ``ev``) and for the training data (``train_r``, ``train_ev``).
        """
        if self.splits.n_test:
            keys = ['i_test', 'r', 'ev']
            rows = [[res.i_test, res.r, res.proportion_explained] for res in self.partition_results]
        else:
            keys = ['i_validate', 'r', 'ev', 'train_r', 'train_ev']
            rows = [[res.i_validate, res.r, res.proportion_explained, res.train_r, res.train_proportion_explained] for res in self.partition_results]
        h_is_list = isinstance(self._h, tuple)
        for row, res in zip(rows, self.partition_results):
            row.extend(res.h if h_is_list else [res.h])
        if self.x in (None, (None,)):
            xs = ['x']
        elif isinstance(self.x, str):
            xs = [self.x]
        else:
            xs = [f'x_{i}' if x is None else x for i, x in enumerate(self.x)]
        return Dataset.from_caselist([*keys, *xs], rows)

    @cached_property
    def proportion_explained(self) -> float | NDVar | None:
        if self.residual is None:
            return None
        return 1 - (self.residual / self._variability)

    def _apply_ndvar_transform(self, func: Callable):
        "Apply func to all NDVars in-place (only for source space transformation)"
        def sub_func(obj):
            if obj is None:
                return None
            elif isinstance(obj, tuple):
                return tuple(sub_func(obj_) for obj_ in obj)
            return func(obj)

        # NDVars
        for attr in ('_h', 'r', 'r_rank', 'r_l1', 'l1_residual', 'l2_residual', 'l1_total', 'l2_total', 'train_r', 'train_r_rank', 'train_r_l1', 'train_l1_residual', 'train_l2_residual', 'train_l1_total', 'train_l2_total', 'y_mean', 'y_scale'):
            setattr(self, attr, sub_func(getattr(self, attr)))
        # Invalidate cached properties derived from the transformed NDVars
        for attr in [name for name, value in vars(type(self)).items() if isinstance(value, cached_property)]:
            self.__dict__.pop(attr, None)

        # List of Dimension
        if self._y_dims is not None:
            self._y_dims = tuple([sub_func(dim) if isinstance(dim, SourceSpaceBase) else dim for dim in self._y_dims])

        if self.partition_results:
            for res in self.partition_results:
                res._apply_ndvar_transform(func)

    def _morph(
            self,
            to_subject: str,
            source_morph: mne.SourceMorph = None,
    ):
        "Morph source space"
        from .._mne import morph_source_space

        def func(obj: NDVar):
            return morph_source_space(obj, to_subject, morph=source_morph)
        self._apply_ndvar_transform(func)

    def _set_adjacency(self, dim, adjacency):
        def func(obj: NDVar):
            return set_adjacency(obj, dim, adjacency)
        self._apply_ndvar_transform(func)

    def _set_parc(self, parc: str):
        """Change the parcellation of source-space result

        Notes
        -----
        No warning for missing sources!
        """
        def func(obj: NDVar):
            return set_parc(obj, parc, mask=True)
        self._apply_ndvar_transform(func)

    @classmethod
    def _eelbrain_concatenate(
            cls,
            results: Sequence[BoostingResult],
            dim: str = 'source',
    ):
        "Combine multiple complementary source-space BoostingResult objects"
        out = {}
        for field_i in fields(cls):
            if field_i.name == 'version':
                continue
            elif field_i.name == '_isnan':
                out['_isnan'] = None
                continue
            values = [getattr(result, field_i.name) for result in results]
            if field_i.name == 't_run':
                out['t_run'] = None if any(v is None for v in values) else sum(values)
                continue
            if field_i.name == 'partition_results' and any(v is not None for v in values):
                if not all(v is not None for v in values):
                    raise ValueError('partition_results available for some but not all part-results')
                new_value = [cls._eelbrain_concatenate(p_results) for p_results in zip(*values)]
            elif field_i.name in ('algorithm_version',):
                values = set(values)
                if len(values) == 1:
                    new_value = values.pop()
                else:
                    new_value = tuple(sorted(values))
            elif field_i.name == 'execution_context':
                new_values = values[:1]
                for value in values[1:]:
                    if value not in new_values:
                        new_values.append(value)
                if len(new_values) == 1:
                    new_value = new_values[0]
                else:
                    new_value = values
            elif any(v is None for v in values):
                new_value = None
            else:
                new_value = _concatenate_values(values, dim, field_i.name)
            out[field_i.name] = new_value
        return cls(**out)


class SplitResult:
    __slots__ = ('split', 'h', 'h_failed')

    def __init__(self, split: Split, h: np.ndarray, h_failed: np.ndarray):
        self.split = split
        self.h = h  # (n_y, n_x, n_times_h)
        self.h_failed = h_failed  # (n_y,)


class Boosting:
    """Object-oriented API for boosting

    Examples
    --------
    Standard usage of the object-oriented API for a model with
    cross-validation, comparable to the :func:`boosting` function::

        data = DeconvolutionData(y, x, ds)
        data.apply_basis(0.05, 'hamming'')
        data.normalize('l1')
        data.initialize_cross_validation(5)
        model = Boosting(data)
        model.fit(0, 0.500, selective_stopping=1, error='l1')
        result = model.evaluate_fit()
    """
    # fit parameters
    tstart = None
    tstart_h = None
    tstop = None
    selective_stopping = None
    error = None
    delta = None
    mindelta = None
    # timing
    t_fit_start = None
    t_fit_done = None
    # fit result
    _i_start = None
    split_results = None
    # eval result
    y_pred = None

    def __init__(
            self,
            data: DeconvolutionData,
    ):
        self.data = data

    def fit(
            self,
            tstart: float | Sequence[float],
            tstop: float | Sequence[float],
            selective_stopping: int = 1,
            error: str = 'l1',
            delta: float = 0.005,  # coordinate search step
            mindelta: float = None,  # narrow search by reducing delta until reaching mindelta
    ):
        self.data._check_data()
        assert error in ('l1', 'l2')
        error_id = int(error[1])
        mindelta_ = delta if mindelta is None else mindelta
        self.selective_stopping = selective_stopping
        self.error = error
        self.delta = delta
        self.mindelta = mindelta
        n_x = len(self.data.x)
        # find TRF start/stop for each x
        if isinstance(tstart, (tuple, list, np.ndarray)):
            if len(tstart) != len(self.data.x_name):
                raise ValueError(f'{tstart=}: {len(tstart)=} is different from len(x)={len(self.data.x_name)}')
            elif len(tstart) != len(tstop):
                raise ValueError(f'{tstop=}: {len(tstop)=} does not match {len(tstart)=}')
            self.tstart = tuple(tstart)
            self.tstart_h = min(self.tstart)
            self.tstop = tuple(tstop)
            if any(start >= stop for start, stop in zip(self.tstart, self.tstop)):
                raise ValueError(f"Some tstart > tstop: {tstart=} and {tstop=}")
            n_xs = [reduce(mul, map(len, xdims), 1) for _, xdims, _ in self.data._x_meta]
            tstart_by_x = [t for t, n in zip(tstart, n_xs) for _ in range(n)]
            tstop_by_x = [t for t, n in zip(tstop, n_xs) for _ in range(n)]
        else:
            if tstart >= tstop:
                raise ValueError(f"{tstart=} > {tstop=}")
            self.tstart = self.tstart_h = tstart
            self.tstop = tstop
            tstart_by_x = [tstart] * n_x
            tstop_by_x = [tstop] * n_x

        # TRF extent in indices
        tstep = self.data.time.tstep
        i_start_by_x = np.asarray([int(round(t / tstep)) for t in tstart_by_x], np.int64)
        i_stop_by_x = np.asarray([int(ceil(t / tstep)) for t in tstop_by_x], np.int64)
        self._i_start = i_start = np.min(i_start_by_x)
        i_stop = np.max(i_stop_by_x)
        h_n_times = i_stop - i_start
        # Data segments need to be at least as long as the TRF
        h_n_times_max = np.max(h_n_times)
        if h_n_times_max > self.data.shortest_segment_n_times:
            raise ValueError(f"{tstart=}, {tstop=}: the kernel is longer than shortest data segment")
        shortest_cv_segment_n_times = np.min(np.diff(self.data.splits.split_segments, axis=1))
        if h_n_times_max > shortest_cv_segment_n_times:
            raise ValueError(f"{tstart=}, {tstop=}: the kernel is longer than shortest cross-validation data segment")

        self.t_fit_start = time.time()

        # boosting
        num_threads = CONFIG['n_workers']
        split_train = package_splits([split.train for split in self.data.splits.splits])
        split_validate = package_splits([split.validate for split in self.data.splits.splits])
        split_train_and_validate = package_splits([split.train_and_validate for split in self.data.splits.splits])
        hs, hs_failed = opt.boosting_runs(self.data.y, self.data.x, self.data.x_pads, split_train, split_validate, split_train_and_validate, i_start_by_x, i_stop_by_x, delta, mindelta_, error_id, selective_stopping, num_threads)
        self.split_results = [SplitResult(split, h, h_failed) for split, h, h_failed in zip(self.data.splits.splits, hs, hs_failed)]

        self.t_fit_done = time.time()

    def _get_i_tests(self):
        assert self.data.splits.n_test
        return sorted({split.split.i_test for split in self.split_results})

    @staticmethod
    def _get_h(split_results: list[SplitResult]) -> np.ndarray:
        "Average kernel, skipping failed kernels (channels that failed in all splits are 0)"
        hs = np.array([split.h for split in split_results])  # (n_splits, n_y, n_x, n_times_h)
        ok = ~np.array([split.h_failed for split in split_results])  # (n_splits, n_y)
        h = np.einsum('sy,syxt->yxt', ok.astype(hs.dtype), hs)
        h /= np.maximum(ok.sum(0), 1)[:, newaxis, newaxis]
        return h

    def _get_h_failed(self, split_results: list[SplitResult]) -> np.ndarray:
        out = np.all([split.h_failed for split in split_results], 0)
        if self.data.vector_dim:
            out = np.all(out.reshape((len(self.data.vector_dim), -1)), 0)
        return out

    def _evaluate(
            self,
            metrics: Sequence[str],
            hs: list[tuple[np.ndarray, np.ndarray]],  # [(h, segments), ...]: segments predicted by h
            eval_segments: list[np.ndarray],  # segments for each set of fit metrics
            debug: bool = False,
    ) -> list:
        "Predict y and compute fit metrics (see :meth:`Evaluator.get`)"
        # y dimensions
        n_y = len(self.data.y)
        if self.data.vector_dim:
            n_vec = len(self.data.vector_dim)
            n_vecs = n_y // n_vec
        else:
            n_vec = n_vecs = 0

        # predicted y
        if debug:
            self.y_pred = y_pred_iter = y_pred = np.full(self.data.y.shape, np.nan)  # NaN for samples that are not predicted
        elif n_vecs:
            y_pred = np.empty((n_vec, *self.data.y.shape[1:]))
            y_pred_iter = chain.from_iterable(repeat(tuple(y_pred), n_vecs))
        else:
            y_pred = np.empty(self.data.y.shape[1:])
            y_pred_iter = repeat(y_pred, n_y)

        all_evaluators, evaluators_s, evaluators_v = get_evaluators(metrics, self.data, eval_segments)

        # fit and evaluate each y
        for i_y, y_pred_i in enumerate(y_pred_iter):
            # for cross-validation, different segments are predicted by different h:
            for h, segments in hs:
                convolve_1d(h[i_y], self.data.x, self.data.x_pads, self._i_start, segments, y_pred_i)

            if evaluators_s:
                for e in evaluators_s:
                    e.add_y(i_y, self.data.y[i_y], y_pred_i)

            if evaluators_v and i_y % n_vec == n_vec - 1:
                i_vec = i_y // n_vec
                i_y_vec = slice(i_y - n_vec + 1, i_y + 1)
                y_pred_i_vec = y_pred[i_y_vec] if debug else y_pred
                for e in evaluators_v:
                    e.add_y(i_vec, self.data.y[i_y_vec], y_pred_i_vec)
        return all_evaluators

    def evaluate_fit(
            self,
            i_test: int = None,
            metrics: Sequence[str] = None,
            cross_fit: bool = None,
            partition_results: bool = False,
            debug: bool = False,
    ):
        """Compute average TRF and fit metrics

        Parameters
        ----------
        i_test
            Test partition index (only applies for models fit with
            cross-validation). By default, ``y`` in each test segment is
            predicted from the averaged model from the corresponding training
            sets, and all test segments are pooled for computing fit metrics.
            Set ``i_test`` with an integer to compute fit metrics for a single
            test partition.
        metrics
            Which model fit metrics to compute (default depends on the data).
            If no metrics are needed, ``metrics=()`` is faster.
        cross_fit
            Compute fit metrics from cross-validation (only applies to model
            with cross-validation; default ``True``). Without cross-validation,
            fit metrics are only computed for ``partition_results``.
        partition_results
            Keep results (TRFs and model evaluation) for each test-partition
            (with ``cross_fit``) or for each validation partition (without
            test set).
        debug
            Add additional attributes to the returned result.
        """
        from .. import __version__ as eelbrain_version

        if cross_fit is None:
            cross_fit = bool(self.data.splits.n_test)
        elif cross_fit and not self.data.splits.n_test:
            raise ValueError(f"{cross_fit=} for model without cross-validation")
        if partition_results and not cross_fit and self.data.splits.n_test:
            raise ValueError(f"{partition_results=} with {cross_fit=} for model with test set")

        # fit evaluation
        if metrics is None:
            if self.data.vector_dim:
                metrics = [f'vec-{self.error}', f'vec-{self.error}-total', 'vec-corr']
                if self.error == 'l1':
                    metrics.append('vec-corr-l1')
            else:
                metrics = ['l1_residual', 'l2_residual', 'r', 'r_rank', 'l1_total', 'l2_total']

        # partitions: [[SplitResult, ...], ...]
        if cross_fit:
            i_tests = self._get_i_tests()
            if i_test is not None:
                if i_test not in i_tests:
                    raise ValueError(f"{i_test=}: test partitions are {i_tests}")
                i_tests = [i_test]
            partitions = [[split for split in self.split_results if split.split.i_test == i] for i in i_tests]
        elif i_test is not None:
            raise ValueError(f"{i_test=} without cross_fit")
        elif partition_results:
            # without test set, each split has a different validation partition
            partitions = [[split] for split in self.split_results]
        else:
            partitions = []
        hs = [self._get_h(partition) for partition in partitions]
        n_samples = self.data.y.shape[1]

        evaluations = {}
        partition_evaluations = [{} for _ in partitions]
        if cross_fit:
            # each test partition is predicted by the h estimated without it
            test_segments = [partition[0].split.test for partition in partitions]
            hs_segments = list(zip(hs, test_segments))
            all_segments = np.sort(np.vstack(test_segments), 0)
            n_samples = int(np.sum(all_segments[:, 1] - all_segments[:, 0]))  # all data unless i_test is specified
            if metrics:
                eval_segments = [merge_segments(all_segments, True)]
                if partition_results:
                    eval_segments.extend(merge_segments(segments, True) for segments in test_segments)
                evaluators = self._evaluate(metrics, hs_segments, eval_segments, debug)
                evaluations = {e.attr: e.get() for e in evaluators}
                if partition_results:
                    partition_evaluations = [{e.attr: e.get(i + 1) for e in evaluators} for i in range(len(partitions))]
            elif debug:
                self._evaluate((), hs_segments, [], True)
        elif metrics:
            # Without test set, fit metrics would be computed from the data used to estimate h
            # (overestimating predictive power); only report validation and training fit for each split
            for h_i, (split_result,), evaluations_i in zip(hs, partitions, partition_evaluations):
                split = split_result.split
                evaluators = self._evaluate(metrics, [(h_i, split.train_and_validate)], [split.validate, split.train])
                evaluations_i.update({e.attr: e.get() for e in evaluators})
                evaluations_i.update({f'train_{e.attr}': e.get(1) for e in evaluators})

        # package h
        if cross_fit:
            h_x = hs[0] if len(hs) == 1 else np.mean(hs, 0)
        else:
            h_x = self._get_h(self.split_results)
            if debug:
                # in-sample prediction from the averaged kernel
                self._evaluate((), [(h_x, self.data.segments)], [], True)
        h = self.data.package_kernel(h_x, self.tstart_h)
        # package model parameters
        y_mean, y_scale, x_mean, x_scale = self.data.data_scale_ndvars()
        if debug:
            evaluations['y_pred'] = self.data.package_y_like(self.y_pred, 'y-pred')
            evaluations['fit'] = self
        t_run = self.t_fit_done - self.t_fit_start
        execution_context = {
            'machine': platform.machine(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'platform': sys.platform,
            'platform_version': platform.version(),
            'eelbrain_version': eelbrain_version,
        }
        # partition-specific results
        if partition_results:
            partition_results_list = []
            for partition, h_i, evaluations_i in zip(partitions, hs, partition_evaluations):
                split = partition[0].split
                eval_segments = split.test if cross_fit else split.validate
                h_i = self.data.package_kernel(h_i, self.tstart_h)
                result = BoostingResult(
                    self.data.y_name, self.data.x_name, self.tstart, self.tstop, bool(self.data.scale_data), self.delta, self.mindelta, self.error, self.selective_stopping,
                    y_mean, y_scale, x_mean, x_scale,
                    h_i, self._get_h_failed(partition), 0,
                    self.data.basis, self.data.basis_window, None,
                    int(np.sum(eval_segments[:, 1] - eval_segments[:, 0])), self.data.y_info, self.data.ydims,
                    algorithm_version=3, execution_context=execution_context,
                    i_test=split.i_test, i_validate=None if cross_fit else split.i_validate, **evaluations_i)
                partition_results_list.append(result)
        else:
            partition_results_list = None
        return BoostingResult(
            # basic parameters
            self.data.y_name, self.data.x_name, self.tstart, self.tstop, bool(self.data.scale_data), self.delta, self.mindelta, self.error, self.selective_stopping,
            # data properties
            y_mean, y_scale, x_mean, x_scale,
            # results
            h, self._get_h_failed(self.split_results), t_run,
            # advanced parameters
            self.data.basis, self.data.basis_window, self.data.splits,
            # advanced data properties
            n_samples, self.data.y_info, self.data.ydims,
            partition_results=partition_results_list,
            algorithm_version=3, execution_context=execution_context,
            i_test=i_test, **evaluations)


@user_activity
@deprecate_ds_arg
def boosting(
        y: NDVarArg,
        x: NDVarArg | Sequence[NDVarArg],
        tstart: float | Sequence[float],
        tstop: float | Sequence[float],
        scale_data: bool | Literal['inplace'] = True,  # normalize y and x
        delta: float = 0.005,  # coordinate search step
        mindelta: float = None,  # narrow search by reducing delta until reaching mindelta
        error: Literal['l1', 'l2'] = 'l2',
        basis: float = 0,
        basis_window: str = 'hamming',
        partitions: int = None,  # Number of partitionings for cross-validation
        model: CategorialArg = None,
        validate: int = 1,  # Number of segments in validation set
        test: int = 1,  # Number of segments in test set
        data: Dataset = None,
        selective_stopping: int = 0,
        partition_results: bool = False,
        debug: bool = False,
) -> BoostingResult:
    """Estimate a linear filter with coordinate descent

    Parameters
    ----------
    y
        Signal to predict. When ``y`` contains more than one signal (e.g.,
        multiple EEG channels), results for each signal will be computed
        independently. Muiltiple cases along a :class:`Case` dimension are
        treated as different trials which share a filter. For correlation fit
        metrics, a :class:`Space` dimension is interpreted as defining a vector
        measure.
    x
        Signal to use to predict ``y``. Can be sequence of NDVars to include
        multiple predictors. Time dimension must correspond to ``y``.
    tstart
        Start of the TRF in seconds. A list can be used to specify different
        values for each item in ``x``.
    tstop
        Stop of the TRF in seconds. Format must match ``tstart``.
    scale_data
        Scale ``y`` and ``x`` before boosting: subtract the mean and divide by
        the standard deviation (when ``error='l2'``) or the mean absolute
        value (when ``error='l1'``). Use ``'inplace'`` to save memory by scaling
        the original objects specified as ``y`` and ``x`` instead of making a
        copy. The data scale is stored in the :class:`BoostingResult`:
        :attr:`BoostingResult.y_mean`, :attr:`BoostingResult.y_scale`, :attr:`BoostingResult.x_mean`, and :attr:`BoostingResult.x_scale`
        attributes.
    delta
        Step for changes in the kernel.
    mindelta
        If the error for the training data can't be reduced, divide ``delta``
        in half until ``delta < mindelta``. The default is ``mindelta = delta``,
        i.e. ``delta`` is constant.
    error
        Error function to use (default is ``l2``).

        - ``error='l1'``: the sum of the absolute differences between ``y`` and
          ``h * x``.
        - ``error='l2'``: the sum of the squared differences between ``y`` and
          ``h * x``.

        For vector ``y``, the error is defined based on the distance in space
        for each data point.
    basis
        Use a basis of windows with this length for the kernel (by default,
        impulses are used).
    basis_window
        Basis window (see :func:`scipy.signal.get_window` for options; default
        is ``'hamming'``).
    partitions
        Divide the data into this many ``partitions`` (*k*) for
        cross-validation. Each boosting run uses one partition for validation
        (early stopping), one partition as test set (with ``test=1``), and the
        remaining partitions for training. With ``test=1``, this requires at
        least 3 partitions and results in ``k * (k - 1)`` boosting runs
        (instead of ``k`` runs with ``test=0``).
        If data is continuous, data are divided into contiguous segments of
        equal length (default 5).
        If data has cases, cases are divided with ``[::partitions]`` slices
        (default ``n_cases``, which is only possible with 3 to 10 cases;
        if ``model`` is specified, all cells need to have the same number of
        cases, and ``n_cases`` is the number of cases per cell).
        See :ref:`exa-data_split` example.
    model
        If data has cases, divide cases into different categories (division
        for crossvalidation is done separately for each cell).
    data
        If provided, other parameters can be specified as string for items in
        ``ds``.
    validate
        Number of segments in validation dataset (currently has to be 1).
    test
        By default (``test=1``), the boosting algorithm performs *k*-fold
        cross-validation (with *k* = ``partitions``): Each partition is used as
        test dataset in turn, while the remaining ``k - 1`` partitions are used
        to estimate the kernel. The fit metrics (e.g.,
        :attr:`BoostingResult.r` and :attr:`BoostingResult.proportion_explained`)
        reflect the re-combination of all partitions, each one predicted from
        the corresponding, independent training set. They thus estimate the
        model's predictive power for data that were not used to fit it.

        With ``test=0``, the kernel is estimated from all available data. This
        requires fewer boosting runs for a given number of partitions, and the
        kernel (:attr:`BoostingResult.h`) is still regularized through early
        stopping. However, fit metrics could then only be computed from the
        same data that were used to estimate the kernel. Such in-sample fit
        metrics overestimate the model's predictive power: even predictors that
        are unrelated to ``y`` yield positive fit metrics, and the bias grows
        with the number of model parameters (predictors and kernel length).
        Fit metrics are therefore not computed (:attr:`BoostingResult.r` etc.
        are ``None``). Use ``test=0`` only when the kernel itself is of
        interest, or when the model will be evaluated on independent data
        (e.g., using :func:`convolve`). To inspect the fit of the individual
        boosting runs, use ``partition_results=True``.
    selective_stopping
        By default, the boosting algorithm stops when the validation error
        stops decreasing. With ``selective_stopping=True``, boosting continues but
        excludes the predictor (one time-series in ``x``) that caused the
        increase in validation error, and continues until all predictors are
        stopped. The integer value of ``selective_stopping`` determines after
        how many steps with error increases each predictor is excluded.
    partition_results
        Keep results (TRFs and model evaluation) for each test partition, or,
        with ``test=0``, for each validation partition (including the fit to
        the training data; see :attr:`BoostingResult.partition_results`).
        This is disabled by default to reduce file size when saving results.
    debug
        Add additional attributes to the returned result.

    See Also
    --------
    plot.preview_partitions : preview data partitions for cross-validation

    Notes
    -----
    The boosting algorithm is described in [1]_.

    In order to predict data, use the :func:`convolve` function::

    >>> data = datasets.get_uts()
    >>> data['a1'] = epoch_impulse_predictor('uts', 'A=="a1"', data=data)
    >>> data['a0'] = epoch_impulse_predictor('uts', 'A=="a0"', data=data)
    >>> res = boosting('uts', ['a0', 'a1'], 0, 0.5, partitions=10, model='A', data=data)
    >>> y_pred = convolve(res.h_scaled, ['a0', 'a1'], data=data)
    >>> y = data['uts']
    >>> plot.UTS([y-y.mean('time'), y_pred], '.case')

    References
    ----------
    .. [1] David, S. V., Mesgarani, N., & Shamma, S. A. (2007). Estimating
        sparse spectro-temporal receptive fields with natural stimuli. Network:
        Computation in Neural Systems, 18(3), 191-212.
        `10.1080/09548980701609235 <https://doi.org/10.1080/09548980701609235>`_.
    """
    # scale_data
    if isinstance(scale_data, bool):
        scale_in_place = False
    elif isinstance(scale_data, str):
        if scale_data == 'inplace':
            scale_in_place = True
        else:
            raise ValueError(f"{scale_data=}")
    else:
        raise TypeError(f"{scale_data=}, need bool or str")
    # selective_stopping
    selective_stopping = int(selective_stopping)
    if selective_stopping < 0:
        raise ValueError(f"{selective_stopping=}")

    dec_data = DeconvolutionData(y, x, data, scale_in_place)
    dec_data.apply_basis(basis, basis_window)
    if scale_data:
        dec_data.normalize(error)
    dec_data.initialize_cross_validation(partitions, model, data, validate, test)

    fit = Boosting(dec_data)
    fit.fit(tstart, tstop, selective_stopping, error, delta, mindelta)
    return fit.evaluate_fit(debug=debug, partition_results=partition_results)


def package_splits(splits: Sequence[np.ndarray]) -> np.ndarray:
    n = max(len(split) for split in splits)
    out = np.empty((len(splits), n, 2), np.int64)
    out.fill(-1)
    for i, split in enumerate(splits):
        out[i, :len(split)] = split
    return out
