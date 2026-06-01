# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Predict each sensor from the other sensors with per-channel regression."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._data_obj import NDVar, NDVarArg, asndvar

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class ChannelModel:
    """Regression model predicting each sensor from the other sensors.

    A separate regression model is fit for each sensor, predicting that
    sensor's signal from all the other sensors. This can be used to
    reconstruct (e.g. interpolate) channels with :meth:`predict`, or to
    identify bad channels with :meth:`score`.

    Parameters
    ----------
    model
        The regression model to use for each sensor. ``'huber'`` (default)
        uses :class:`sklearn.linear_model.HuberRegressor`, which is robust to
        high-amplitude artifacts in the training data while also regularizing
        collinear channels through ``alpha``. ``'ridge'`` uses
        :class:`sklearn.linear_model.Ridge` (fast, but artifacts in the
        training data bias the fit). ``'ols'`` uses ordinary least squares
        (:class:`sklearn.linear_model.LinearRegression`). Alternatively, any
        scikit-learn estimator instance can be passed and is cloned for each
        sensor (in which case the other parameters are ignored).
    alpha
        L2 regularization strength (``'huber'`` and ``'ridge'`` only). Features
        and target are robustly scaled before fitting (see Notes), so ``alpha``
        applies in a unit-scale space and is independent of the data amplitude.
    epsilon
        Huber threshold: residuals smaller than this are treated with squared
        loss (OLS-like), larger ones with linear loss (robust). The smaller the
        value, the more robust to outliers (``'huber'`` only).
    fit_intercept
        Estimate an intercept for each sensor (default ``True``).
    ...
        Additional keyword arguments are passed to the estimator.

    Notes
    -----
    Before fitting, the predictor channels and the target channel are each
    scaled with :class:`sklearn.preprocessing.RobustScaler` (centered on the
    median, scaled by the inter-quartile range). This makes the fit invariant
    to the overall data amplitude (EEG in volts is ~1e-6, which otherwise makes
    regularized/robust estimators like ``'huber'`` collapse to flat
    predictions) and prevents high-amplitude artifacts from inflating the
    scaling. The scaling is inverted automatically, so predictions are returned
    in the original units.

    Attributes
    ----------
    sensor : Sensor
        The sensor dimension the model was fit with.
    estimators_ : list
        The fitted estimator for each sensor (in the order of ``sensor``).
    """

    def __init__(
            self,
            model: str | BaseEstimator = 'huber',
            alpha: float = 1e-4,
            epsilon: float = 1.35,
            fit_intercept: bool = True,
            **kwargs,
    ):
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        self.sensor = None
        self.estimators_ = None

    def _make_estimator(self):
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        # robustly scale features and target so the fit is amplitude-invariant
        # and artifacts do not inflate the scaling
        pipeline = make_pipeline(RobustScaler(), self._make_regressor())
        return TransformedTargetRegressor(pipeline, transformer=RobustScaler())

    def _make_regressor(self):
        if not isinstance(self.model, str):
            from sklearn.base import clone
            return clone(self.model)
        elif self.model == 'huber':
            from sklearn.linear_model import HuberRegressor
            return HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, fit_intercept=self.fit_intercept, **{'max_iter': 300, **self.kwargs})
        elif self.model == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.kwargs)
        elif self.model == 'ols':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(fit_intercept=self.fit_intercept, **self.kwargs)
        else:
            raise ValueError(f"{self.model=}; needs to be 'huber', 'ridge', 'ols' or a scikit-learn estimator")

    def fit(self, data: NDVarArg, threshold: float = 50e-6):
        """Fit the model.

        Parameters
        ----------
        data : NDVar
            EEG data with ``sensor`` and ``time`` dimensions (``[case x] sensor
            x time``). All non-sensor dimensions are flattened into regression
            samples.
        threshold
            Exclude data in which any channel exceeds this absolute value
            (default 50 µV). In epoched data (with a ``case`` dimension) the
            whole epoch is excluded; in continuous data the ±250 ms around each
            exceeding time point is excluded. Set to ``None`` to disable.

        Returns
        -------
        self
        """
        data = asndvar(data)
        if not data.has_dim('sensor'):
            raise ValueError(f"{data=}: needs a sensor dimension")
        if not data.has_dim('time'):
            raise ValueError(f"{data=}: needs a time dimension")
        sensor = data.get_dim('sensor')
        n_sensors = len(sensor)
        if data.has_case:
            # epoched: sensor x case x time
            x = data.get_data(('sensor', 'case', 'time'))
            if threshold is not None:
                keep = ~(np.abs(x) > threshold).any((0, 2))  # per epoch
                x = x[:, keep]
        else:
            # continuous: sensor x time
            x = data.get_data(('sensor', 'time'))
            if threshold is not None:
                bad = (np.abs(x) > threshold).any(0)  # per time point
                w = round(0.250 / data.get_dim('time').tstep)  # ±250 ms
                bad = np.convolve(bad, np.ones(2 * w + 1), 'same') > 0
                x = x[:, ~bad]
        # sensor x sample
        x = x.reshape(n_sensors, -1)
        if x.shape[1] == 0:
            raise ValueError(f"{threshold=}: excluded all data")
        estimators = []
        for i in range(n_sensors):
            others = np.arange(n_sensors) != i
            estimator = self._make_estimator()
            estimator.fit(x[others].T, x[i])
            estimators.append(estimator)
        self.sensor = sensor
        self.estimators_ = estimators

    def predict(self, data: NDVarArg) -> NDVar:
        """Predict each sensor from the other sensors.

        Parameters
        ----------
        data : NDVar
            EEG data (``[case x] sensor x time``) with the same sensors used for
            fitting.

        Returns
        -------
        prediction : NDVar
            Data with the same dimensions as ``data``, where each channel is
            predicted from the other channels.
        """
        data = self._check_data(data)
        time = data.get_dim('time')
        if data.has_case:
            x = data.get_data(('case', 'sensor', 'time'))
            out = np.stack([self._predict_raw(xi) for xi in x])
            dims = (data.get_dim('case'), self.sensor, time)
        else:
            out = self._predict_raw(data.get_data(('sensor', 'time')))
            dims = (self.sensor, time)
        return NDVar(out, dims, data.name, data.info)

    def score(self, data: NDVarArg, threshold: float = 50e-6, max_exclude: float = 0.25) -> NDVar:
        """Score each sensor by how badly it is predicted from the others.

        A high score identifies a bad channel. Within each epoch, the channel
        with the largest prediction error is scored with that error and then
        excluded (its input is replaced with its prediction from the other
        channels, so it no longer contaminates the remaining channels); this
        repeats until no channel's error exceeds ``threshold``, at which point
        the remaining channels are scored with their current error.

        Parameters
        ----------
        data : NDVar
            EEG data (``[case x] sensor x time``) with the same sensors used for
            fitting.
        threshold
            Stop excluding channels once the largest error drops to this
            absolute value (default 50 µV).
        max_exclude
            Maximum number of channels to exclude per epoch. A value < 1 is
            interpreted as a fraction of the sensors (default 0.25); a value
            ≥ 1 as an absolute count.

        Returns
        -------
        score : NDVar
            The per-channel error score (``[case x] sensor``).
        """
        data = self._check_data(data)
        n_sensors = len(self.sensor)
        max_n = int(max_exclude) if max_exclude >= 1 else int(max_exclude * n_sensors)
        if data.has_case:
            x = data.get_data(('case', 'sensor', 'time'))
            out = np.stack([self._score_block(xi, threshold, max_n) for xi in x])
            dims = (data.get_dim('case'), self.sensor)
        else:
            out = self._score_block(data.get_data(('sensor', 'time')), threshold, max_n)
            dims = (self.sensor,)
        return NDVar(out, dims, data.name)

    def _check_data(self, data: NDVarArg) -> NDVar:
        if self.estimators_ is None:
            raise RuntimeError("This ChannelModel has not been fit yet; call .fit() first")
        data = asndvar(data)
        if data.get_dim('sensor') != self.sensor:
            raise ValueError(f"{data=}: sensors do not match the sensors used for fitting")
        return data

    def _predict_raw(self, x: np.ndarray) -> np.ndarray:
        # predict each channel from the others; x and output are sensor x time
        out = np.empty_like(x)
        index = np.arange(len(x))
        for i in range(len(x)):
            out[i] = self.estimators_[i].predict(x[index != i].T)
        return out

    def _score_block(self, x: np.ndarray, threshold: float, max_n: int) -> np.ndarray:
        # step-down error score per channel for one block (sensor x time)
        n_sensors = len(x)
        scores = np.empty(n_sensors)
        xi = x
        bad = []
        while True:
            # impute the current bad channels with their predictions (fixed point)
            for _ in range(20):
                pred = self._predict_raw(xi)
                new = xi.copy()
                new[bad] = pred[bad]
                if np.max(np.abs(new - xi)) <= 1e-3 * np.max(np.abs(x)):
                    xi = new
                    break
                xi = new
            error = np.abs(x - self._predict_raw(xi)).max(1)  # per channel
            remaining = [c for c in range(n_sensors) if c not in bad]
            worst = remaining[np.argmax(error[remaining])]
            if error[worst] <= threshold or len(bad) >= max_n:
                scores[remaining] = error[remaining]
                return scores
            scores[worst] = error[worst]
            bad.append(worst)
            bad.append(worst)
