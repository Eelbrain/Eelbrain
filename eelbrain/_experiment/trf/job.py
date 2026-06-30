# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Separable, picklable TRF computing job

A TRF fit is split into two stages so that a single fit can be computed on a
machine that does not have the raw data:

- :class:`TRFJobSpec` is host-side, internal machinery (created by
  :meth:`Pipeline._trf_job_spec`). It holds only a reference to the resolved
  request (state and options), generates the corresponding :class:`TRFJob`,
  checks whether a fit is already cached, and incorporates an externally
  computed result back into the cache.
- :class:`TRFJob` is picklable and *data-carrying*: it holds the estimator and
  all already-loaded fitting arguments, so it can be shipped to another machine,
  fit there, and the result pickled back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._data_obj import Datalist, NDVar
    from ..derivative_cache import Request
    from .estimator import Estimator


@dataclass(frozen=True)
class TRFJob:
    """A picklable TRF fitting job carrying its data

    Like a deferred :func:`functools.partial` over :meth:`Estimator._fit`: it
    holds the estimator and the already-loaded fitting arguments, so it can be
    pickled, executed on a machine without the raw data, and the result pickled
    back. Created by :meth:`TRFDerivative.make_job` / :meth:`Pipeline.load_trf_job`.

    Parameters
    ----------
    estimator
        The :class:`Estimator` that fits the model.
    y
        Response (a single :class:`NDVar`, or a :class:`Datalist` of
        :class:`NDVar` for variable-length epochs).
    xs
        Predictors, one entry per model term.
    tstart
        Start of the TRF in seconds.
    tstop
        Stop of the TRF in seconds.
    fwd
        Forward solution (NCRF only).
    cov
        Noise covariance (NCRF only).
    key
        Cache key identifying the corresponding artifact, for matching the
        result back to a :class:`TRFJobSpec`.
    """
    estimator: Estimator
    y: NDVar | Datalist
    xs: list[NDVar | Datalist]
    tstart: float
    tstop: float
    fwd: NDVar | None = None
    cov: Any | None = None
    key: dict[str, Any] | None = None

    def fit(self):
        "Fit the TRF and return the result object."
        return self.estimator._fit(self.y, self.xs, self.tstart, self.tstop, fwd=self.fwd, cov=self.cov)

    def __call__(self):
        return self.fit()


@dataclass(frozen=True)
class TRFJobSpec:
    """Host-side handle for one TRF fit (internal)

    References the resolved request needed to (re)generate a data-carrying
    :class:`TRFJob` and to incorporate an externally computed result into the
    cache. Not required to be picklable.

    Parameters
    ----------
    ctx
        Resolved request for the TRF (carries state and options).
    """
    ctx: Request

    @property
    def path(self) -> Path:
        "Target cache artifact path"
        return self.ctx.artifact_path

    @property
    def key(self) -> dict[str, Any]:
        "Cache key identifying the artifact"
        return self.ctx.key()

    @property
    def is_done(self) -> bool:
        "Whether a valid cached artifact already exists"
        return self.ctx.is_valid()

    def make_job(self) -> TRFJob:
        "Load the data on the host and build a picklable :class:`TRFJob`"
        return self.ctx.node.make_job(self.ctx)

    def save_result(self, result) -> object:
        "Incorporate an externally computed result into the cache (artifact + manifest)"
        return self.ctx.save_artifact(result)
