"""
Boosting estimator for TRF pipeline.

Use in experiment definition:

    estimators = {
        'boosting': BoostingEstimator(basis=0.050),
        'boosting-l2': BoostingEstimator(error='l2', partitions=5),
    }
    # Then: e.load_trf('acoustic_envelop', 0, 0.5, estimator='boosting')

tstart & tstop are parameters of load_trf/load_trfs, not of the estimator.
"""

from typing import Any, Dict, Optional
from .estimator import Estimator

# Concrete strategy implementation for BoostingEstimator
class BoostingEstimator(Estimator):
    """
    Parameters
    ----------
    delta : float, default=0.005
        Step size for boosting updates.
        Smaller values lead to slower but more precise updates.
        Larger values speed up training but may overshoot optimal solutions.

    mindelta : float, optional
        Minimum step size threshold.
        Training may stop when updates fall below this value.

    error : str, default="l1"
        Loss function used during boosting.
        - "l1" → robust to outliers
        - "l2" → emphasizes larger errors

    basis : float, default=0.050
        Temporal basis width (in seconds).
        Controls temporal smoothness of the TRF.
        Larger values → smoother temporal filters.

    partitions : int, optional
        Number of partitions for cross-validation during boosting.
        Required when the number of trials is outside the default supported range.

    test : bool, default=True
        Whether to evaluate model performance during training.
        Used for early stopping and model selection.

    selective_stopping : int, default=0
        Enables early stopping on selected components.
        Helps prevent overfitting by stopping updates selectively.

    partition_results : bool, default=False
        Whether to store results separately for each partition.
        Useful for detailed analysis but increases memory usage.

    name : str, optional
        Name of the estimator (defaults to "boosting").

    Notes
    -----
    - Boosting is an iterative method that builds the TRF incrementally.
    - Compared to NCRF, boosting operates directly as a fitting algorithm
    rather than solving a regularized inverse problem.
    """
    def __init__(
        self,
        *,
        delta: float = 0.005,
        mindelta: Optional[float] = None,
        error: str = "l1",
        basis: float = 0.050,
        partitions: Optional[int] = None,
        test: bool = True,
        selective_stopping: int = 0,
        partition_results: bool = False,
        name: str = "",
    ):
        self.delta = delta
        self.mindelta = mindelta
        self.error = error
        self.basis = basis
        self.partitions = partitions
        self.test = test
        self.selective_stopping = selective_stopping
        self.partition_results = partition_results
        self.name = name or "boosting"


    # partial(boosting, y, xs, tstart, tstop, 'inplace', **self.parameters_for_partial())
    def parameters_for_partial(self) -> Dict[str, Any]:
        """Override base: return boosting kwargs for partial(boosting, ..., **kwargs)."""
        return {
            "delta": self.delta,
            "mindelta": self.mindelta,
            "error": self.error,
            "basis": self.basis,
            "partitions": self.partitions,
            "test": self.test,
            "selective_stopping": self.selective_stopping,
            "partition_results": self.partition_results,
        }
