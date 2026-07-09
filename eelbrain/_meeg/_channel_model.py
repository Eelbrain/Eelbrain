# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Predict each sensor from the other sensors with per-channel regression."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._data_obj import Datalist, NDVar, NDVarArg, UTS, asndvar
from .base import BadChannelWindow

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

    def fit(self, data: NDVarArg | list, threshold: float = 50e-6):
        """Fit the model.

        Parameters
        ----------
        data
            EEG data with ``sensor`` and ``time`` dimensions (``[case x] sensor
            x time``). All non-sensor dimensions are flattened into regression
            samples. A ``list`` (or :class:`Datalist`) of long, variable-length
            epochs (each ``sensor x time``, e.g. :class:`mne.Epochs` or NDVar)
            is also accepted; each is treated like continuous data and all are
            concatenated.
        threshold
            Exclude data in which any channel exceeds this absolute value
            (default 50 µV). In epoched data (with a ``case`` dimension) the
            whole epoch is excluded; in continuous and long-epoch data the
            ±250 ms around each exceeding time point is excluded. Set to
            ``None`` to disable.

        Returns
        -------
        self
        """
        if isinstance(data, list):
            # long epochs: concatenate the good samples of each epoch
            blocks = self._as_blocks(data)
            sensor = blocks[0].get_dim('sensor')
            n_sensors = len(sensor)
            x = np.concatenate([self._exclude_continuous(ndvar.get_data(('sensor', 'time')), ndvar.get_dim('time').tstep, threshold) for ndvar in blocks], axis=1)
        else:
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
                x = x.reshape(n_sensors, -1)  # sensor x sample
            else:
                # continuous: sensor x time
                x = self._exclude_continuous(data.get_data(('sensor', 'time')), data.get_dim('time').tstep, threshold)
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

    def predict(self, data: NDVarArg | list) -> NDVar | Datalist:
        """Predict each sensor from the other sensors.

        Parameters
        ----------
        data
            EEG data (``[case x] sensor x time``) with the same sensors used for
            fitting. A ``list`` of long, variable-length epochs is also accepted
            (see :meth:`fit`).

        Returns
        -------
        prediction
            Data with the same dimensions as ``data``, where each channel is
            predicted from the other channels. For a list of long epochs, a
            :class:`Datalist` with one prediction NDVar per epoch.
        """
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [NDVar(self._predict_raw(ndvar.get_data(('sensor', 'time'))), (self.sensor, ndvar.get_dim('time')), ndvar.name, ndvar.info) for ndvar in blocks]
            return Datalist(out, blocks.name)
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

    def score(self, data: NDVarArg | list, threshold: float = 50e-6, max_exclude: float = 0.25) -> NDVar | Datalist:
        """Score each sensor by how badly it is predicted from the others.

        A high score identifies a bad channel. Within each epoch, the channel
        with the largest prediction error is scored with that error and then
        excluded (its input is replaced with its prediction from the other
        channels, so it no longer contaminates the remaining channels); this
        repeats until no channel's error exceeds ``threshold``, at which point
        the remaining channels are scored with their current error.

        Parameters
        ----------
        data
            EEG data (``[case x] sensor x time``) with the same sensors used for
            fitting. A ``list`` of long, variable-length epochs is also accepted;
            use :meth:`find_bad_windows` instead to score those time-resolved.
        threshold
            Stop excluding channels once the largest error drops to this
            absolute value (default 50 µV).
        max_exclude
            Maximum number of channels to exclude per epoch. A value < 1 is
            interpreted as a fraction of the sensors (default 0.25); a value
            ≥ 1 as an absolute count.

        Returns
        -------
        score
            The per-channel error score (``[case x] sensor``). For a list of long
            epochs, a :class:`Datalist` with one score NDVar per epoch.
        """
        n_sensors = len(self.sensor) if self.sensor is not None else 0
        max_n = int(max_exclude) if max_exclude >= 1 else int(max_exclude * n_sensors)
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [NDVar(self._score_block(ndvar.get_data(('sensor', 'time')), threshold, max_n), (self.sensor,), ndvar.name) for ndvar in blocks]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        if data.has_case:
            x = data.get_data(('case', 'sensor', 'time'))
            out = np.stack([self._score_block(xi, threshold, max_n) for xi in x])
            dims = (data.get_dim('case'), self.sensor)
        else:
            out = self._score_block(data.get_data(('sensor', 'time')), threshold, max_n)
            dims = (self.sensor,)
        return NDVar(out, dims, data.name)

    def find_bad_windows(
            self,
            data: NDVarArg | list,
            threshold: float = 50e-6,
            max_exclude: float = 0.25,
            window: float = 1.0,
            hop: float = 0.5,
            min_duration: float = 0.1,
            merge_gap: float | None = None,
    ) -> Datalist | list:
        """Find the time windows in which each sensor is bad.

        Like :meth:`score`, but time-resolved: instead of flagging a channel for
        a whole epoch, the channel is scored within sliding time windows so that
        a bad channel is only flagged over the interval in which it is actually
        bad.

        Parameters
        ----------
        data
            EEG data with the same sensors used for fitting; typically a ``list``
            (or :class:`Datalist`) of long, variable-length epochs (each
            ``sensor x time``). A single continuous NDVar (``sensor x time``) or
            epoched NDVar (``case x sensor x time``) is also accepted.
        threshold
            A channel is bad in a window when its error exceeds this absolute
            value (default 50 µV; see :meth:`score`).
        max_exclude
            Maximum number of channels to exclude per window (see :meth:`score`).
        window
            Length of the sliding scoring window in seconds (default 1.0).
        hop
            Step between successive windows in seconds (default 0.5).
        min_duration
            Discard bad windows shorter than this many seconds (default 0.1).
        merge_gap
            Merge two bad windows of the same channel separated by less than this
            many seconds (default: ``window``).

        Returns
        -------
        windows
            One list of :class:`BadChannelWindow` per epoch (per case for an
            epoched NDVar; a single list for a continuous NDVar).

        Notes
        -----
        The data are scanned with a window of length ``window`` seconds, stepped
        by ``hop`` seconds. Within each window the step-down scoring of
        :meth:`score` is applied, so every channel gets an error and hence a
        good/bad classification (bad when the error exceeds ``threshold``) for
        that window, with at most ``max_exclude`` channels flagged per window. A
        time point is then considered bad for a channel if it is covered by *any*
        window in which that channel was classified bad; since each window's
        verdict applies to the window's full width and successive windows overlap
        (when ``hop`` < ``window``), the bad time points form contiguous runs.
        Each run is returned as a :class:`BadChannelWindow`, after discarding
        runs shorter than ``min_duration`` and merging runs of the same channel
        separated by less than ``merge_gap``.

        Because a window's verdict spans its whole width, a localized artifact is
        bracketed by up to roughly one ``window`` length of margin on each side.
        ``window`` therefore effectively sets the amount of padding around
        detected artifacts, while ``hop`` controls how precisely the window edges
        are placed.
        """
        n_sensors = len(self.sensor) if self.sensor is not None else 0
        max_n = int(max_exclude) if max_exclude >= 1 else int(max_exclude * n_sensors)
        if merge_gap is None:
            merge_gap = window
        args = (threshold, max_n, window, hop, min_duration, merge_gap)
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [self._windows_for_block(ndvar.get_data(('sensor', 'time')), ndvar.get_dim('time'), *args) for ndvar in blocks]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        time = data.get_dim('time')
        if data.has_case:
            x = data.get_data(('case', 'sensor', 'time'))
            out = [self._windows_for_block(xi, time, *args) for xi in x]
            return Datalist(out, data.name)
        return self._windows_for_block(data.get_data(('sensor', 'time')), time, *args)

    def _as_blocks(self, data: list) -> Datalist:
        # normalize a list/Datalist of long epochs to a Datalist of validated
        # sensor x time NDVars
        data = asndvar(data, ragged=True)
        for ndvar in data:
            if not ndvar.has_dim('sensor'):
                raise ValueError(f"{ndvar=}: needs a sensor dimension")
            if not ndvar.has_dim('time'):
                raise ValueError(f"{ndvar=}: needs a time dimension")
        return data

    def _check_fit(self):
        if self.estimators_ is None:
            raise RuntimeError("This ChannelModel has not been fit yet; call .fit() first")

    def _check_data(self, data: NDVarArg) -> NDVar:
        self._check_fit()
        data = asndvar(data)
        if data.get_dim('sensor') != self.sensor:
            raise ValueError(f"{data=}: sensors do not match the sensors used for fitting")
        return data

    def _check_blocks(self, data: list) -> Datalist:
        self._check_fit()
        blocks = self._as_blocks(data)
        for ndvar in blocks:
            if ndvar.get_dim('sensor') != self.sensor:
                raise ValueError(f"{ndvar=}: sensors do not match the sensors used for fitting")
        return blocks

    @staticmethod
    def _exclude_continuous(
            x: np.ndarray,
            tstep: float,
            threshold: float | None,
    ) -> np.ndarray:
        # drop the ±250 ms around any time point where a channel exceeds threshold
        if threshold is None:
            return x
        bad = (np.abs(x) > threshold).any(0)  # per time point
        w = round(0.250 / tstep)  # ±250 ms
        bad = np.convolve(bad, np.ones(2 * w + 1), 'same') > 0
        return x[:, ~bad]

    def _predict_raw(self, x: np.ndarray) -> np.ndarray:
        # predict each channel from the others; x and output are sensor x time
        out = np.empty_like(x)
        index = np.arange(len(x))
        for i in range(len(x)):
            out[i] = self.estimators_[i].predict(x[index != i].T)
        return out

    def _score_block(self, x: np.ndarray, threshold: float, max_n: int) -> np.ndarray:
        # step-down error score per channel (sensor,) for one block (sensor x time)
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

    def _windows_for_block(
            self,
            x: np.ndarray,
            time: UTS,
            threshold: float,
            max_n: int,
            window: float,
            hop: float,
            min_duration: float,
            merge_gap: float,
    ) -> list[BadChannelWindow]:
        # time-resolved bad-channel windows for one block (sensor x time)
        error = self._score_windows(x, time, threshold, max_n, window, hop)
        return self._windows_from_error(error, time, threshold, min_duration, merge_gap)

    def _score_windows(
            self,
            x: np.ndarray,
            time: UTS,
            threshold: float,
            max_n: int,
            window: float,
            hop: float,
    ) -> np.ndarray:
        # per-sample error from sliding-window step-down scoring (sensor x time)
        n_times = x.shape[1]
        w = max(1, round(window / time.tstep))
        h = max(1, round(hop / time.tstep))
        starts = list(range(0, max(1, n_times - w + 1), h))
        if starts[-1] != n_times - w and n_times > w:
            starts.append(n_times - w)  # make sure the last samples are covered
        error = np.zeros_like(x)
        for s in starts:
            t0, t1 = s, min(s + w, n_times)
            block = self._score_block(x[:, t0:t1], threshold, max_n)  # per channel
            error[:, t0:t1] = np.maximum(error[:, t0:t1], block[:, None])
        return error

    def _windows_from_error(
            self,
            error: np.ndarray,
            time: UTS,
            threshold: float,
            min_duration: float,
            merge_gap: float,
    ) -> list[BadChannelWindow]:
        # convert a per-sample error array (sensor x time) into bad-channel windows
        mask = error > threshold
        n_times = mask.shape[1]
        min_samples = max(1, round(min_duration / time.tstep))
        merge_samples = round(merge_gap / time.tstep)
        names = self.sensor.names
        out = []
        for ci in np.flatnonzero(mask.any(1)):
            # half-open [start, stop) runs of bad samples
            edges = np.flatnonzero(np.diff(np.concatenate(([0], mask[ci].view(np.int8), [0]))))
            runs = []
            for start, stop in zip(edges[::2], edges[1::2]):
                if runs and start - runs[-1][1] < merge_samples:
                    runs[-1] = (runs[-1][0], stop)
                else:
                    runs.append((start, stop))
            for start, stop in runs:
                if stop - start < min_samples:
                    continue
                tmin = time.tmin + start * time.tstep
                tmax = time.tstop if stop == n_times else time.tmin + stop * time.tstep
                out.append(BadChannelWindow(names[ci], tmin, tmax))
        return out
