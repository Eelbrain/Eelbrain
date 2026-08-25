# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Separable, picklable TRF computing job

A TRF fit is split into two stages so that a single fit can be computed on a
machine that does not have the raw data; see
:mod:`eelbrain._experiment.derivative_cache.job` for the generic machinery.
:class:`pipeline.TRFJob` is the data-carrying half; the host-side half is the generic
:class:`~eelbrain._experiment.derivative_cache.job.JobSpec`, created by
:meth:`Pipeline._trf_job_spec`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..derivative_cache import Job

if TYPE_CHECKING:
    from ..._data_obj import Datalist, NDVar
    from .estimator import Estimator


@dataclass(frozen=True)
class TRFJob(Job):
    """A picklable TRF fitting job carrying its data

    Like a deferred :func:`functools.partial` over ``Estimator._fit``: it
    holds the estimator and the already-loaded fitting arguments, so it can be
    pickled, executed on a machine without the raw data, and the result pickled
    back. Created by ``TRFDerivative.make_job`` / :meth:`Pipeline.load_trf_job`.

    Parameters
    ----------
    estimator
        The ``Estimator`` that fits the model.
    y
        Response (a single :class:`NDVar`, or a :class:`Datalist` of
        :class:`NDVar` for variable-length epochs).
    xs
        Predictors, one entry per model term.
    tstart
        Start of the TRF in seconds (or one value per predictor).
    tstop
        Stop of the TRF in seconds (or one value per predictor).
    fwd
        Forward solution (NCRF only).
    cov
        Noise covariance (NCRF only).
    """
    estimator: Estimator
    y: NDVar | Datalist
    xs: list[NDVar | Datalist]
    tstart: float | list[float]
    tstop: float | list[float]
    fwd: NDVar | None = None
    cov: Any | None = None

    def __call__(self):
        "Fit the TRF and return the result object."
        return self.estimator._fit(self.y, self.xs, self.tstart, self.tstop, fwd=self.fwd, cov=self.cov)
