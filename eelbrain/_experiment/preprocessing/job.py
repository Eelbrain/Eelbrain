# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Separable, picklable ICA computing job

An ICA decomposition is split into two stages so that a single decomposition can
be computed on a machine that does not have the raw data; see
:mod:`eelbrain._experiment.derivative_cache.job` for the generic machinery.
:class:`ICAJob` is the data-carrying half; the host-side half is the generic
:class:`~eelbrain._experiment.derivative_cache.job.JobSpec`, created by
:meth:`Pipeline._job_spec`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne

from ..._utils import user_activity
from ..derivative_cache import Job


@dataclass(frozen=True)
class ICAJob(Job):
    """A picklable ICA fitting job carrying its data

    Holds the source data and the resolved fitting arguments, so it can be
    pickled, executed on a machine without the raw data, and the result pickled
    back. Created by :meth:`ICAInput.make_job`.

    Parameters
    ----------
    raw
        Preloaded source data to decompose, concatenated across tasks/runs where
        the ICA step spans them, with the fit-time bad channels already set on
        ``raw.info['bads']``.
    kwargs
        Resolved arguments for :class:`mne.preprocessing.ICA`.
    fit_kwargs
        Resolved arguments for :meth:`mne.preprocessing.ICA.fit`.

    Notes
    -----
    ``key`` alone does not distinguish two :class:`RawICA` pipes over the same recording;
    the inherited ``node`` field does (each :class:`RawICA` step registers its
    own :class:`ICAInput`).
    """
    raw: mne.io.BaseRaw
    kwargs: dict[str, Any]
    fit_kwargs: dict[str, Any]

    def __call__(self) -> mne.preprocessing.ICA:
        "Fit the ICA and return the :class:`mne.preprocessing.ICA` object."
        ica = mne.preprocessing.ICA(**self.kwargs)
        with user_activity:
            ica.fit(self.raw, **self.fit_kwargs)
        return ica
