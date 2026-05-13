# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from warnings import catch_warnings, filterwarnings

import numpy as np
from scipy.linalg import norm
from scipy.stats import spearmanr
try:
    from scipy.stats import ConstantInputWarning  # >= 1.9
except ImportError:
    from scipy.stats import SpearmanRConstantInputWarning as ConstantInputWarning  # < 1.9

from ._boosting_opt import error_for_indexes
from .shared import DeconvolutionData


class Evaluator:
    vector = False
    attr = NotImplemented
    name = NotImplemented
    meas = None

    def __init__(
            self,
            data: DeconvolutionData,
            segments: list[np.ndarray] = None,  # evaluations for different test-segments
    ):
        n = len(data.y)
        if self.vector:
            if not data.vector_dim:
                raise ValueError(f"{self.__class__.__name__}: Vector evaluator for non-vector data")
            n //= len(data.vector_dim)
        self.data = data
        self.segments = segments
        self.xs = [np.empty(n, np.float64) for _ in segments]

    @classmethod
    def _crop_y(cls, segments, *ys):
        "For error functions that require one contiguous segment"
        if len(segments) == 1:
            istart, istop = segments[0]
            out = [y[..., istart: istop] for y in ys]
        else:
            # OPT: avoid new array creation
            out = [np.concatenate([y[..., i_start:i_stop] for i_start, i_stop in segments], axis=-1) for y in ys]
        return out[0] if len(out) == 1 else out

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} evaluator>"

    def get(self, i_test: int = -1):
        return self.data.package_value(self.xs[i_test + 1], self.name, meas=self.meas)


class L1(Evaluator):
    attr = 'l1_residual'
    name = 'L1 residuals'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        err = y - y_pred
        for x, segments in zip(self.xs, self.segments):
            x[i] = error_for_indexes(err, segments, 1)


class L2(Evaluator):
    attr = 'l2_residual'
    name = 'L2 residuals'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        err = y - y_pred
        for x, segments in zip(self.xs, self.segments):
            x[i] = error_for_indexes(err, segments, 2)


class L1Total(Evaluator):
    attr = 'l1_total'
    name = 'L1 total'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        for x, segments in zip(self.xs, self.segments):
            x[i] = error_for_indexes(y, segments, 1)


class L2Total(Evaluator):
    attr = 'l2_total'
    name = 'L2 total'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        for x, segments in zip(self.xs, self.segments):
            x[i] = error_for_indexes(y, segments, 2)


class Correlation(Evaluator):
    attr = 'r'
    name = 'Correlation'
    meas = 'r'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        for x, segments in zip(self.xs, self.segments):
            y_i, y_pred_i = self._crop_y(segments, y, y_pred)
            with catch_warnings():
                filterwarnings('ignore', "invalid value encountered", RuntimeWarning)
                r = np.corrcoef(y_i, y_pred_i)[0, 1]
            x[i] = 0 if np.isnan(r) else r


class RankCorrelation(Evaluator):
    attr = 'r_rank'
    name = 'Rank correlation'
    meas = 'r'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        for x, segments in zip(self.xs, self.segments):
            y_i, y_pred_i = self._crop_y(segments, y, y_pred)
            with catch_warnings():
                filterwarnings('ignore', "invalid value encountered", RuntimeWarning)
                filterwarnings('ignore', category=ConstantInputWarning)
                r = spearmanr(y_i, y_pred_i)[0]
            x[i] = 0 if np.isnan(r) else r


class VectorL1(Evaluator):
    vector = True
    attr = 'l1_residual'
    name = 'Vector l1 residuals'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        err = norm(y - y_pred, axis=0)
        for x, segments in zip(self.xs, self.segments):
            err_i = self._crop_y(segments, err)
            x[i] = err_i.sum(-1)


class VectorL2(Evaluator):
    vector = True
    attr = 'l2_residual'
    name = 'Vector l2 residuals'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        err = y - y_pred
        err **= 2
        err = err.sum(0)
        for x, segments in zip(self.xs, self.segments):
            err_i = self._crop_y(segments, err)
            x[i] = err_i.sum()


class VectorCorrelation(Evaluator):
    vector = True
    attr = 'r'
    name = 'Vector correlation'
    meas = 'r'

    def add_y(
            self,
            i: int,  # y index (row in data.y)
            y: np.ndarray,  # actual data
            y_pred: np.ndarray,  # data predicted by model
    ):
        for x, segments in zip(self.xs, self.segments):
            y, y_pred = self._crop_y(segments, y, y_pred)
            x[i] = self._r(y, y_pred)

    def _r(self, y, y_pred):
        # shape (space, time)
        y_norm = norm(y, axis=0)
        y_pred_norm = norm(y_pred, axis=0)
        # rescale y_pred
        y_pred_scale = (y_pred_norm ** 2).mean() ** 0.5
        if y_pred_scale == 0:
            return 0
        y_pred_l2 = y_pred / y_pred_scale
        # rescale y
        y_scale = (y_norm ** 2).mean() ** 0.5
        if y_scale == 0:
            return 0
        elif self.data.scale_data == 'l2':
            y_l2 = y
        else:
            y_l2 = y / y_scale
        # l2 correlation
        return np.multiply(y_l2, y_pred_l2, out=y_pred_l2).sum(0).mean(0)


class VectorCorrelationL1(VectorCorrelation):
    vector = True
    attr = 'r_l1'
    name = 'Vector l1-correlation'
    meas = 'r'

    def _r(self, y, y_pred):
        # shape (space, time)
        y_norm = norm(y, axis=0)
        y_pred_norm = norm(y_pred, axis=0)
        # rescale y_pred
        y_pred_scale = y_pred_norm.mean()
        if y_pred_scale == 0:
            return 0
        y_pred_l1 = y_pred / y_pred_scale
        # rescale y
        y_scale = y_norm.mean()
        if y_scale == 0:
            return 0
        elif self.data.scale_data == 'l1':
            y_l1 = y
        else:
            y_l1 = y / y_scale
        # l1 correlation
        # --------------
        # E|X| = 1 --> E√XX = 1
        yy = np.multiply(y_l1, y_pred_l1, out=y_pred_l1).sum(0)
        sign = np.sign(yy)
        np.abs(yy, out=yy)
        yy **= 0.5
        yy *= sign
        return yy.mean()


EVALUATORS = {
    'l1_residual': L1,
    'l2_residual': L2,
    'l1_total': L1Total,
    'l2_total': L2Total,
    'r': Correlation,
    'r_rank': RankCorrelation,
    'vec-l1': VectorL1,
    'vec-l2': VectorL2,
    'vec-corr': VectorCorrelation,
    'vec-corr-l1': VectorCorrelationL1,
}


def get_evaluators(
        keys: list[str],
        data: DeconvolutionData,
        segments: list[np.ndarray] = None,  # evaluations for different test-segments
) -> (list[Evaluator], list[Evaluator], list[Evaluator]):
    evaluators = [EVALUATORS[key](data, segments) for key in keys]
    # split into scalar and vector evaluators
    evaluators_s = [e for e in evaluators if not e.vector]
    evaluators_v = [e for e in evaluators if e.vector]
    return evaluators, evaluators_s, evaluators_v
