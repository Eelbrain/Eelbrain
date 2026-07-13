# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Predict each sensor from the other sensors with per-channel regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from joblib import Parallel, delayed
import numpy as np

from .._data_obj import Datalist, NDVar, NDVarArg, UTS, asndvar
from .base import BadChannelWindow

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class _BadChannelModel:
    """Shared input validation for :class:`ChannelModel` and
    :class:`ChannelRANSACModel`.

    Both models expose the same ``sensor`` / ``estimators_`` attributes and
    ``fit`` / ``predict`` / ``score`` / ``find_bad_windows`` interface, so
    that they can be used interchangeably.
    """

    sensor = None
    estimators_ = None

    def _as_blocks(self, data: list) -> Datalist:
        # normalize a list/Datalist of long epochs to a Datalist of validated
        # sensor x time NDVars
        data = asndvar(data, ragged=True)
        for ndvar in data:
            if not ndvar.has_dim("sensor"):
                raise ValueError(f"{ndvar=}: needs a sensor dimension")
            if not ndvar.has_dim("time"):
                raise ValueError(f"{ndvar=}: needs a time dimension")
        return data

    def _check_fit(self):
        if self.estimators_ is None:
            raise RuntimeError(
                f"This {self.__class__.__name__} has not been fit yet; call .fit() first"
            )

    def _check_data(self, data: NDVarArg) -> NDVar:
        self._check_fit()
        data = asndvar(data)
        if data.get_dim("sensor") != self.sensor:
            raise ValueError(
                f"{data=}: sensors do not match the sensors used for fitting"
            )
        return data

    def _check_blocks(self, data: list) -> Datalist:
        self._check_fit()
        blocks = self._as_blocks(data)
        for ndvar in blocks:
            if ndvar.get_dim("sensor") != self.sensor:
                raise ValueError(
                    f"{ndvar=}: sensors do not match the sensors used for fitting"
                )
        return blocks


class ChannelModel(_BadChannelModel):
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
        model: str | BaseEstimator = "huber",
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
        elif self.model == "huber":
            from sklearn.linear_model import HuberRegressor

            return HuberRegressor(
                epsilon=self.epsilon,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                **{"max_iter": 300, **self.kwargs},
            )
        elif self.model == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                **self.kwargs,
            )
        elif self.model == "ols":
            from sklearn.linear_model import LinearRegression

            return LinearRegression(
                fit_intercept=self.fit_intercept,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"{self.model=}; needs to be 'huber', 'ridge', 'ols' or a scikit-learn estimator"
            )

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
            sensor = blocks[0].get_dim("sensor")
            n_sensors = len(sensor)
            x = np.concatenate(
                [
                    self._exclude_continuous(
                        ndvar.get_data(("sensor", "time")),
                        ndvar.get_dim("time").tstep,
                        threshold,
                    )
                    for ndvar in blocks
                ],
                axis=1,
            )
        else:
            data = asndvar(data)
            if not data.has_dim("sensor"):
                raise ValueError(f"{data=}: needs a sensor dimension")
            if not data.has_dim("time"):
                raise ValueError(f"{data=}: needs a time dimension")
            sensor = data.get_dim("sensor")
            n_sensors = len(sensor)
            if data.has_case:
                # epoched: sensor x case x time
                x = data.get_data(("sensor", "case", "time"))
                if threshold is not None:
                    keep = ~(np.abs(x) > threshold).any((0, 2))  # per epoch
                    x = x[:, keep]
                x = x.reshape(n_sensors, -1)  # sensor x sample
            else:
                # continuous: sensor x time
                x = self._exclude_continuous(
                    data.get_data(("sensor", "time")),
                    data.get_dim("time").tstep,
                    threshold,
                )
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
        return self

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
            out = [
                NDVar(
                    self._predict_raw(ndvar.get_data(("sensor", "time"))),
                    (self.sensor, ndvar.get_dim("time")),
                    ndvar.name,
                    ndvar.info,
                )
                for ndvar in blocks
            ]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        time = data.get_dim("time")
        if data.has_case:
            x = data.get_data(("case", "sensor", "time"))
            out = np.stack([self._predict_raw(xi) for xi in x])
            dims = (data.get_dim("case"), self.sensor, time)
        else:
            out = self._predict_raw(data.get_data(("sensor", "time")))
            dims = (self.sensor, time)
        return NDVar(out, dims, data.name, data.info)

    def score(
        self,
        data: NDVarArg | list,
        threshold: float = 50e-6,
        max_exclude: float = 0.25,
    ) -> NDVar | Datalist:
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
            EEG data (``[case x] sensor x time``) with the same sensors used
            for fitting. A ``list`` of long, variable-length epochs is also
            accepted; use :meth:`find_bad_windows` instead to score those
            time-resolved.
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
            The per-channel error score (``[case x] sensor``). For a list of
            long epochs, a :class:`Datalist` with one score NDVar per epoch.
        """
        n_sensors = len(self.sensor) if self.sensor is not None else 0
        max_n = (
            int(max_exclude)
            if max_exclude >= 1
            else int(max_exclude * n_sensors)
        )
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [
                NDVar(
                    self._score_block(
                        ndvar.get_data(("sensor", "time")),
                        threshold,
                        max_n,
                    ),
                    (self.sensor,),
                    ndvar.name,
                )
                for ndvar in blocks
            ]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        if data.has_case:
            x = data.get_data(("case", "sensor", "time"))
            out = np.stack(
                [self._score_block(xi, threshold, max_n) for xi in x],
            )
            dims = (data.get_dim("case"), self.sensor)
        else:
            out = self._score_block(
                data.get_data(("sensor", "time")),
                threshold,
                max_n,
            )
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

        Like :meth:`score`, but time-resolved: instead of flagging a channel
        for a whole epoch, the channel is scored within sliding time windows
        so that a bad channel is only flagged over the interval in which it
        is actually bad.

        Parameters
        ----------
        data
            EEG data with the same sensors used for fitting; typically a
            ``list`` (or :class:`Datalist`) of long, variable-length epochs
            (each ``sensor x time``). A single continuous NDVar (``sensor x
            time``) or epoched NDVar (``case x sensor x time``) is also
            accepted.
        threshold
            A channel is bad in a window when its error exceeds this
            absolute value (default 50 µV; see :meth:`score`).
        max_exclude
            Maximum number of channels to exclude per window (see
            :meth:`score`).
        window
            Length of the sliding scoring window in seconds (default 1.0).
        hop
            Step between successive windows in seconds (default 0.5).
        min_duration
            Discard bad windows shorter than this many seconds (default 0.1).
        merge_gap
            Merge two bad windows of the same channel separated by less
            than this many seconds (default: ``window``).

        Returns
        -------
        windows
            One list of :class:`BadChannelWindow` per epoch (per case for an
            epoched NDVar; a single list for a continuous NDVar).

        Notes
        -----
        The data are scanned with a window of length ``window`` seconds,
        stepped by ``hop`` seconds. Within each window the step-down scoring
        of :meth:`score` is applied, so every channel gets an error and
        hence a good/bad classification (bad when the error exceeds
        ``threshold``) for that window, with at most ``max_exclude``
        channels flagged per window. A time point is then considered bad
        for a channel if it is covered by *any* window in which that
        channel was classified bad; since each window's verdict applies to
        the window's full width and successive windows overlap (when
        ``hop`` < ``window``), the bad time points form contiguous runs.
        Each run is returned as a :class:`BadChannelWindow`, after
        discarding runs shorter than ``min_duration`` and merging runs of
        the same channel separated by less than ``merge_gap``.

        Because a window's verdict spans its whole width, a localized
        artifact is bracketed by up to roughly one ``window`` length of
        margin on each side. ``window`` therefore effectively sets the
        amount of padding around detected artifacts, while ``hop`` controls
        how precisely the window edges are placed.
        """
        n_sensors = len(self.sensor) if self.sensor is not None else 0
        max_n = (
            int(max_exclude)
            if max_exclude >= 1
            else int(max_exclude * n_sensors)
        )
        if merge_gap is None:
            merge_gap = window
        args = (threshold, max_n, window, hop, min_duration, merge_gap)
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [
                self._windows_for_block(
                    ndvar.get_data(("sensor", "time")),
                    ndvar.get_dim("time"),
                    *args,
                )
                for ndvar in blocks
            ]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        time = data.get_dim("time")
        if data.has_case:
            x = data.get_data(("case", "sensor", "time"))
            out = [self._windows_for_block(xi, time, *args) for xi in x]
            return Datalist(out, data.name)
        return self._windows_for_block(
            data.get_data(("sensor", "time")),
            time,
            *args,
        )

    @staticmethod
    def _exclude_continuous(
        x: np.ndarray,
        tstep: float,
        threshold: float | None,
    ) -> np.ndarray:
        # drop the ±250 ms around any time point where a channel exceeds
        # threshold
        if threshold is None:
            return x
        bad = (np.abs(x) > threshold).any(0)  # per time point
        w = round(0.250 / tstep)  # ±250 ms
        bad = np.convolve(bad, np.ones(2 * w + 1), "same") > 0
        return x[:, ~bad]

    def _predict_raw(self, x: np.ndarray) -> np.ndarray:
        # predict each channel from the others; x and output are sensor x time
        out = np.empty_like(x)
        index = np.arange(len(x))
        for i in range(len(x)):
            out[i] = self.estimators_[i].predict(x[index != i].T)
        return out

    def _score_block(
        self,
        x: np.ndarray,
        threshold: float,
        max_n: int,
    ) -> np.ndarray:
        # step-down error score per channel (sensor,) for one block
        # (sensor x time)
        n_sensors = len(x)
        scores = np.empty(n_sensors)
        xi = x
        bad = []
        while True:
            # impute the current bad channels with their predictions
            # (fixed point)
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
        return self._windows_from_error(
            error,
            time,
            threshold,
            min_duration,
            merge_gap,
        )

    def _score_windows(
        self,
        x: np.ndarray,
        time: UTS,
        threshold: float,
        max_n: int,
        window: float,
        hop: float,
    ) -> np.ndarray:
        # per-sample error from sliding-window step-down scoring
        # (sensor x time)
        n_times = x.shape[1]
        w = max(1, round(window / time.tstep))
        h = max(1, round(hop / time.tstep))
        starts = list(range(0, max(1, n_times - w + 1), h))
        if starts[-1] != n_times - w and n_times > w:
            starts.append(n_times - w)  # make sure the last samples are
            # covered
        error = np.zeros_like(x)
        for s in starts:
            t0, t1 = s, min(s + w, n_times)
            # per channel
            block = self._score_block(x[:, t0:t1], threshold, max_n)
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
        # convert a per-sample error array (sensor x time) into bad-channel
        # windows
        mask = error > threshold
        n_times = mask.shape[1]
        min_samples = max(1, round(min_duration / time.tstep))
        merge_samples = round(merge_gap / time.tstep)
        names = self.sensor.names
        out = []
        for ci in np.flatnonzero(mask.any(1)):
            # half-open [start, stop) runs of bad samples
            edges = np.flatnonzero(
                np.diff(
                    np.concatenate(
                        ([0], mask[ci].view(np.int8), [0]),
                    )
                )
            )
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
                tmax = (
                    time.tstop
                    if stop == n_times
                    else time.tmin + stop * time.tstep
                )
                out.append(BadChannelWindow(names[ci], tmin, tmax))
        return out


class ChannelRANSACModel(_BadChannelModel):
    """Like :class:`ChannelModel`, but uses RANSAC for each sensor's regression.

    This is slower than :class:`ChannelModel`, but more robust to artifacts in
    the training data. It is therefore recommended when the training data are
    contaminated by artifacts, e.g. when using long epochs or continuous data.

    Parameters
    ----------
    window_len :
        Window length in seconds. Default 5.
    max_broken_time :
        Max fraction (< 1) or seconds (≥ 1) of bad time allowed. Default 0.4.
    n_resamples :
        Number of RANSAC random subsets. Default 50.
    subset_size :
        Fraction of channels in each RANSAC subset. Default 0.25.
    random_seed :
        Seed for reproducible RANSAC sampling. Default 0.
    n_jobs :
        Number of parallel jobs for RANSAC projector computation. Default 1.

    Attributes
    ----------
    sensor : Sensor
        The sensor dimension the model was fit with.
    estimators_ : RANSACProjector
        The fitted RANSAC projector (a single, shared projector is used for
        all sensors, unlike :attr:`ChannelModel.estimators_` which holds one
        estimator per sensor).
    """

    def __init__(
        self,
        window_len: float = 5.0,
        max_broken_time: float = 0.4,
        n_resamples: int = 50,
        subset_size: float = 0.25,
        random_seed: int = 0,
        n_jobs: int = 1,
    ):
        self.window_len = window_len
        self.max_broken_time = max_broken_time
        self.n_resamples = n_resamples
        self.subset_size = subset_size
        self.sensor = None
        self.estimators_ = None
        self.random_seed = random_seed
        self.n_jobs = n_jobs

    def _make_estimator(self):
        self.estimators_ = RANSACProjector(
            subset_size=self.subset_size,
            n_resamples=self.n_resamples,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
        )

    def fit(self, data: NDVarArg | list):
        """Fit the model.

        Parameters
        ----------
        data
            EEG data with ``sensor`` and ``time`` dimensions (``[case x] sensor
            x time``), used to determine the sensor positions. A ``list`` (or
            :class:`Datalist`) of long, variable-length epochs (each ``sensor
            x time``) is also accepted; only the sensors are used (see
            :meth:`ChannelModel.fit`).

        Returns
        -------
        self
        """
        if isinstance(data, list):
            blocks = self._as_blocks(data)
            ref = blocks[0]
        else:
            ref = asndvar(data)
            if not ref.has_dim("sensor"):
                raise ValueError(f"{data=}: needs a sensor dimension")
        self._make_estimator()
        self.sensor = ref.get_dim("sensor")
        self.estimators_.fit(ref)
        return self

    def predict(self, data: NDVarArg | list) -> NDVar | Datalist:
        """Predict each sensor from the RANSAC reconstruction of the others.

        Parameters
        ----------
        data
            EEG data (``[case x] sensor x time``) with the same sensors used for
            fitting. A ``list`` of long, variable-length epochs is also accepted
            (see :meth:`ChannelModel.predict`).

        Returns
        -------
        prediction
            Data with the same dimensions as ``data``. For a list of long
            epochs, a :class:`Datalist` with one prediction NDVar per epoch.
        """
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [
                self.estimators_.transform(ndvar, self.window_len)
                for ndvar in blocks
            ]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        return self.estimators_.transform(data, self.window_len)

    def score(
        self,
        data: NDVarArg | list,
        corr_threshold: float = 0.75,
        window_len: float | None = None,
    ) -> NDVar | Datalist:
        """Score each sensor by how badly it is predicted from the others.

        A high score identifies a bad channel. Within each epoch / window
        with corrleation of the channel recording with its RANSAC estimator
        is computed. Any channel with corrleation below the `corr_thershold`
        is scored 1 as bad channel. For continuous data, such scores are
        averaged across the windows for each sesnsor, and if it exceeds
        a prediefied value < 1, that channel will be treated as bad channel.
        Bad chananels can be determined based on the fraction or seconds
        of bad time allowed.

        Parameters
        ----------
        data
            EEG data (``[case x] sensor x time``) with the same sensors used
            for fitting. A ``list`` of long, variable-length epochs is also
            accepted; use :meth:`find_bad_windows` instead to score those
            time-resolved.
        corr_threshold
            Corr threshold for marking bad channels (default 0.75).
        window_len
            Window length in seconds for the RANSAC reconstruction (default:
            the ``window_len`` passed to the constructor).

        Returns
        -------
        score
            The per-channel error score (``[case x] sensor``). For a list of
            long epochs, a :class:`Datalist` with one score NDVar per epoch.
        """
        if window_len is None:
            window_len = self.window_len
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [
                self._score_block(ndvar, corr_threshold, window_len)
                for ndvar in blocks
            ]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        return self._score_block(data, corr_threshold, window_len)

    def _score_block(self, data, corr_threshold, window_len):
        # score is % of window having atleat 1 sensor with corr <
        # corr_threshold.
        corrs = self.estimators_.compute_correlation(data, window_len)
        flagged = corrs < corr_threshold
        if data.has_case:
            return flagged
        else:
            return flagged.mean("time")

    def find_bad_windows(
        self,
        data: NDVarArg | list,
        window_len: float = 1.0,
        corr_threshold: float = 0.7,
        min_duration: float | int = 0.1,
        merge_gap: float | None = None,
    ):
        """Find the time windows in which each sensor is bad.

        Like :meth:`score`, but time-resolved: instead of flagging a
        channel for a whole epoch, the channel is scored within sliding
        time windows so that a bad channel is only flagged over the
        interval in which it is actually bad.

        Parameters
        ----------
        data
            EEG data with the same sensors used for fitting; typically a
            ``list`` (or :class:`Datalist`) of long, variable-length epochs
            (each ``sensor x time``). A single continuous NDVar (``sensor x
            time``) or epoched NDVar (``case x sensor x time``) is also
            accepted.
        window_len
            Length of the RANSAC reconstruction window in seconds
            (default 1.0).
        corr_threshold
            A channel is bad in a window when its correlation with its
            RANSAC reconstruction drops below this value (default 0.7; see
            :meth:`score`).
        min_duration
            Discard bad windows shorter than this many seconds (default 0.1).
        merge_gap
            Merge two bad windows of the same channel separated by less
            than this many seconds (default: ``window_len``).

        Returns
        -------
        windows
            One list of :class:`BadChannelWindow` per epoch (per case for an
            epoched NDVar; a single list for a continuous NDVar).
        """
        if merge_gap is None:
            merge_gap = window_len
        args = (window_len, corr_threshold, min_duration, merge_gap)
        if isinstance(data, list):
            blocks = self._check_blocks(data)
            out = [self._windows_for_block(ndvar, *args) for ndvar in blocks]
            return Datalist(out, blocks.name)
        data = self._check_data(data)
        if data.has_case:
            time = data.get_dim("time")
            x = data.get_data(("case", "sensor", "time"))
            out = [
                self._windows_for_block(
                    NDVar(xi, (self.sensor, time), data.name),
                    *args,
                )
                for xi in x
            ]
            return Datalist(out, data.name)
        return self._windows_for_block(data, *args)

    def _windows_for_block(
        self,
        data: NDVar,
        window_len: float,
        corr_threshold: float,
        min_duration: float,
        merge_gap: float,
    ) -> list[BadChannelWindow]:
        # time-resolved bad-channel windows for one block (sensor x time)
        corr = self._score_windows(data, window_len)
        return self._windows_from_corr(
            corr,
            corr_threshold,
            min_duration,
            merge_gap,
        )

    def _score_windows(
        self,
        data: NDVar,  # sensor x time
        window_len: float,
    ) -> np.ndarray:
        # per-sample error from sliding-window step-down scoring
        # (sensor x time)
        block = self.estimators_.compute_correlation(data, window_len)
        x = data.get_data(("sensor", "time"))
        corr = np.empty_like(x)
        time = data.get_dim("time")
        n_times = time.nsamples
        w = max(1, round(window_len / time.tstep))
        starts = list(range(0, max(1, n_times - w + 1), w))
        if starts[-1] != n_times - w and n_times > w:
            starts.append(n_times - w)  # make sure the last samples are
            # covered
        for s, b in zip(starts, block.get_data(("time", "sensor"))):
            t0, t1 = s, min(s + w, n_times)
            corr[:, t0:t1] = b[:, None]
        corr = NDVar(
            corr,
            data.get_dims(("sensor", "time")),
            name="RANSAC corr",
        )
        return corr

    def _windows_from_corr(
        self,
        corr: NDVar,
        corr_threshold: float,
        min_duration: float,
        merge_gap: float,
    ) -> list[BadChannelWindow]:
        # convert a per-sample error array (sensor x time) into bad-channel
        # windows
        mask = corr.get_data(("sensor", "time")) < corr_threshold
        time = corr.get_dim("time")
        n_times = mask.shape[1]
        min_samples = max(1, round(min_duration / time.tstep))
        merge_samples = round(merge_gap / time.tstep)
        names = self.sensor.names
        out = []
        for ci in np.flatnonzero(mask.any(1)):
            # half-open [start, stop) runs of bad samples
            edges = np.flatnonzero(
                np.diff(
                    np.concatenate(
                        ([0], mask[ci].view(np.int8), [0]),
                    )
                )
            )
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
                tmax = (
                    time.tstop
                    if stop == n_times
                    else time.tmin + stop * time.tstep
                )
                out.append(BadChannelWindow(names[ci], tmin, tmax))
        return out

    def _mad(
        self,
        x: np.ndarray,
        median: bool = False,
        axis: int = -1,
    ) -> np.ndarray:
        """Mean/median absolute deviation (matches MATLAB mad(x))."""
        method = np.median if median else np.mean
        return method(
            np.abs(x - method(x, axis=axis, keepdims=True)),
            axis=axis,
        )


class RANSACProjector:
    n_sensors = None
    """RANSAC projector for one window of data.

    Parameters
    ----------
    subset_size : float
        Fraction of channels in each RANSAC subset.
    n_resamples : int
        Number of RANSAC random subsets.
    alpha : float
        Regularization parameter for RANSAC projection.
    random_seed : int
        Seed for reproducible RANSAC sampling.
    n_jobs : int
        Number of parallel jobs for RANSAC projector computation.
    """

    def __init__(
        self,
        subset_size: float = 0.25,
        n_resamples: int = 50,
        alpha: float = 1e-5,
        random_seed: int = 42,
        n_jobs: int = 1,
    ):
        self.subset_size = subset_size
        self.n_resamples = n_resamples
        self.alpha = alpha
        self._rng = np.random.default_rng(random_seed)
        self._workers = Parallel(n_jobs=n_jobs)
        self._n_splits = min(self._workers.n_jobs, n_resamples)

    def fit(self, data: NDVar) -> RANSACProjector:
        """Fit the RANSAC projector to the channel positions.

        Parameters
        ----------
        pos : array, shape (n_sensors, 3)
            3-D channel positions in metres.
        """
        pos = data.sensor.locs
        self.n_sensors = pos.shape[0]
        n_subset = max(1, round(self.subset_size * self.n_sensors))
        ch_subsets = self._get_random_subsets(
            self.n_resamples,
            n_subset,
            self._rng,
            self.n_sensors,
        )

        ch_subsets_split = np.array_split(ch_subsets, self._n_splits)
        projectors = self._workers(
            delayed(self._build_ransac_projector)(pos, ch_subset, self.alpha)
            for ch_subset in ch_subsets_split
        )
        self._projectors = np.vstack(projectors)
        return self

    @staticmethod
    def _transform(
        X: np.ndarray,
        projectors: np.ndarray,
        n_resamples: int,
        n_sensors: int,
    ) -> np.ndarray:
        YY_all = projectors.dot(X).reshape(n_resamples, n_sensors, -1)
        # take median across RANSAC samples
        YY = np.median(YY_all, axis=0)  # (N, win)
        return YY

    @staticmethod
    def _transform_window(
        X,
        offsets,
        win_samples,
        projectors,
        n_resamples,
        n_sensors,
    ):
        YYs = []
        for offset in offsets:
            XX = X[:, offset : offset + win_samples]
            # reconstruct: (n_sensors * num_samples, win_samples) →
            # (num_samples, n_sensors, win_samples)
            YY = RANSACProjector._transform(
                XX,
                projectors,
                n_resamples,
                n_sensors,
            )
            YYs.append(YY)
        return np.vstack(YYs)  # (n_sensors, n_times)

    def transform(self, data: NDVar, window_len: float = 5.0) -> NDVar:
        """Transform the data with the RANSAC projector.

        Parameters
        ----------
        data : NDVar
            EEG data with ``sensor`` and ``time`` dimensions (``[case x] sensor
            x time``). All non-sensor dimensions are flattened into regression
            samples. A ``list`` (or :class:`Datalist`) of long, variable-length
            epochs (each ``sensor x time``, e.g. :class:`mne.Epochs` or NDVar)
            is also accepted; each is treated like continuous data and all are
            concatenated.
        window_len : float
            Window length in seconds. Default 5.

        Returns
        -------
        transformed : NDVar
            Transformed data with the same dimensions as ``data``.
        """
        # Decide win_samples if not provided
        time = data.get_dim("time")
        sfreq = 1 / time.tstep
        if data.has_case:
            # Ignore window_len for epoched data, as we will transform
            # each epoch separately.
            win_samples = time.nsamples
            n_times = win_samples * len(data.get_dim("case"))
            X = data.get_data(("case", "sensor", "time"))
            X = np.hstack(list(X))  # (n_sensors, n_times)
        else:
            win_samples = int(window_len * sfreq)
            n_times = time.nsamples
            X = data.get_data(("sensor", "time"))

        assert self.n_sensors == X.shape[0]
        if data.has_case:
            # exactly one window per case, no remainder
            offsets = np.arange(0, n_times, win_samples)
        else:
            # if the data is not longer than one window, there are no full
            # windows and everything is handled by the "last window" below
            offsets = np.arange(0, max(0, n_times - win_samples), win_samples)
        n_blocks = len(offsets)

        if n_blocks:
            n_splits = min(self._workers.n_jobs, n_blocks)
            offsets_splits = np.array_split(offsets, n_splits)
            new_X = self._workers(
                delayed(self._transform_window)(
                    X,
                    offsets_split,
                    win_samples,
                    self._projectors,
                    self.n_resamples,
                    self.n_sensors,
                )
                for offsets_split in offsets_splits
            )
            new_X = np.vstack(new_X)  # (n_sensors, n_times)

        # Handle the remainder that is not a full window (or, if the data is
        # not longer than one window, the entire data)
        if n_blocks == 0 or offsets[-1] + win_samples < n_times:
            last_offset = offsets[-1] + win_samples if n_blocks else 0
            last_window = self._transform(
                X[:, last_offset:],
                self._projectors,
                self.n_resamples,
                self.n_sensors,
            )
        else:
            last_window = None

        if data.has_case:
            new_X = new_X.reshape(n_blocks, self.n_sensors, win_samples)
            transformed_data = NDVar(
                new_X,
                data.get_dims(("case", "sensor", "time")),
                name="RANSAC transformed",
                info=data.info,
            )
        else:
            if n_blocks:
                new_X = np.hstack(
                    new_X.reshape(n_blocks, self.n_sensors, win_samples),
                )
            else:
                new_X = np.empty((self.n_sensors, 0))
            if last_window is not None:
                new_X = np.hstack([new_X, last_window])
            time = UTS(time.tmin, time.tstep, new_X.shape[-1])
            transformed_data = NDVar(
                new_X,
                data.get_dims(("sensor", "time")),
                name="RANSAC transformed",
                info=data.info,
            )
        return transformed_data

    def compute_correlation(
        self,
        data: NDVar,
        window_len: float = 5.0,
    ) -> np.ndarray:
        # --- per-window correlation ---
        # Decide win_samples if not provided
        time = data.get_dim("time")
        sfreq = 1 / time.tstep
        if data.has_case:
            # Ignore window_len for epoched data, as we will compute
            # correlation for each epoch separately.
            win_samples = time.nsamples
            n_times = win_samples * len(data.get_dim("case"))
            X = data.get_data(("case", "sensor", "time"))
            X = np.hstack(list(X))  # (n_sensors, n_times)
        else:
            win_samples = int(window_len * sfreq)
            n_times = time.nsamples
            X = data.get_data(("sensor", "time"))

        assert self.n_sensors == X.shape[0]
        if n_times < win_samples:
            raise RuntimeError(
                f"Window length exceeds data horizon. Use smaller windows."
            )
        if data.has_case:
            # exactly one window per case, no remainder
            offsets = np.arange(0, n_times, win_samples)
        else:
            # if the data is not longer than one window, there are no full
            # windows and everything is handled by the "last window" below
            offsets = np.arange(0, max(0, n_times - win_samples), win_samples)
        n_windows = len(offsets)

        if n_windows:
            n_splits = min(self._workers.n_jobs, n_windows)
            offsets_splits = np.array_split(offsets, n_splits)
            corrs = self._workers(
                delayed(self._compute_correlation_window)(
                    X,
                    offsets_split,
                    win_samples,
                    self._projectors,
                    self.n_resamples,
                    self.n_sensors,
                )
                for offsets_split in offsets_splits
            )
            corrs = np.vstack(corrs)  # (n_wins, n_usable, )
        else:
            corrs = np.empty((0, self.n_sensors))

        # Handle the remainder that is not a full window (or, if the data is
        # not longer than one window, the entire data)
        if n_windows == 0 or offsets[-1] + win_samples < n_times:
            last_offset = offsets[-1] + win_samples if n_windows else 0
            last_corr = self._compute_correlation(
                X[:, last_offset:],
                self._projectors,
                self.n_resamples,
                self.n_sensors,
            )
            n_windows += 1
            corrs = np.vstack([corrs, last_corr])

        if data.has_case:
            corrs = NDVar(
                corrs,
                (data.get_dim("case"), data.get_dim("sensor")),
                name="RANSAC correlation",
                info=data.info,
            )
        else:
            window = UTS(window_len / 2, window_len, n_windows)
            corrs = NDVar(
                corrs.T,
                (data.get_dim("sensor"), window),
                name="RANSAC correlation",
                info=data.info,
            )
        return corrs

    @staticmethod
    def _compute_correlation(XX, projectors, n_resamples, n_sensors):
        YY = RANSACProjector._transform(
            XX,
            projectors,
            n_resamples,
            n_sensors,
        )
        # Compute correlation for each channel
        num = np.sum(XX * YY, axis=-1)
        denom = np.sqrt(np.sum(XX**2, axis=-1)) * np.sqrt(
            np.sum(YY**2, axis=-1)
        )
        corr = np.where(denom > 0, num / denom, 0.0)
        return corr

    @staticmethod
    def _compute_correlation_window(
        X,
        offsets,
        win_samples,
        projectors,
        n_resamples,
        n_sensors,
    ):  # Asuumes raw data.
        """Compute correlation of each channel to its RANSAC reconstruction."""
        corrs = list()

        for offset in offsets:
            XX = X[:, offset : offset + win_samples]
            # reconstruct: (n_sensors * num_samples, win_samples) →
            # (num_samples, n_sensors, win_samples)
            corr = RANSACProjector._compute_correlation(
                XX,
                projectors,
                n_resamples,
                n_sensors,
            )
            corrs.append(corr)
        return np.vstack(corrs)

    @staticmethod
    def _build_ransac_projector(
        pos: np.ndarray,
        ch_subsets: list[np.ndarray],
        alpha: float,
    ) -> np.ndarray:
        """Build RANSAC projection matrix P of shape (C, C*num_samples).

        Each CxC block in P reconstructs all channels from one random subset.
        """
        from ..mne_fixes._interpolation import _make_interpolation_matrix

        n_sensors = pos.shape[0]
        mappings = list()
        pick_to = range(n_sensors)
        for pick_from in ch_subsets:
            mapping = np.zeros((n_sensors, n_sensors))
            # rows=all channels, cols=subset
            mapping[:, pick_from] = _make_interpolation_matrix(
                pos[pick_from],
                pos[pick_to],
                alpha,
            )
            mappings.append(mapping)
        return np.vstack(mappings)

    def _get_random_subsets(self, n_resamples, subset_size, rng, n_sensors):
        pool_base = np.arange(n_sensors)
        ch_subsets = list()
        for _ in range(n_resamples):
            picks = rng.choice(pool_base, size=subset_size, replace=False)
            ch_subsets.append(picks)
        return ch_subsets  # (n_sensors*num_samples, n_samples)ß
