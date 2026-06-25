# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Separable, picklable TRF computing job

A :class:`TRFJob` is private machinery, constructed by
:meth:`TRFDerivative.make_job`, that allows a single TRF fit to be pickled and
computed on a different machine. It is *deferred*: it carries the experiment
class, root, state and options (not the loaded data), and reconstructs the
experiment on the worker to reload data and run the same build path as
:meth:`Pipeline.load_trf`.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TRFJob:
    """A picklable TRF fitting job

    Parameters
    ----------
    experiment_class
        The :class:`Pipeline` subclass to reconstruct on the worker.
    root
        Experiment root directory.
    state
        Experiment state to restore before fitting.
    options
        ``'trf'`` derivative options (model, ``tstart``, ``tstop``, estimator, …).
    path
        Target cache artifact path (for reference; the build writes here).
    """
    experiment_class: type
    root: str
    state: dict[str, Any]
    options: dict[str, Any]
    path: Path

    def fit(self):
        """Reconstruct the experiment, fit the TRF, cache and return the result."""
        experiment = self.experiment_class(self.root)
        experiment.set(**self.state)
        return experiment._load_derivative('trf', options=self.options)

    def __call__(self):
        return self.fit()
