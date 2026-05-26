from .estimator import Estimator
from typing import Dict, Any
from eelbrain._experiment.mne_experiment import TestDims

# Concrete strategy implementation for NCRFEstimator


class NCRFEstimator(Estimator):
    name: str = "ncrf"

    """
    NCRFEstimator: configuration for NCRF model fitting.

    This class defines the optimization and preprocessing behavior for NCRF.
    It does NOT implement the fitting itself, but prepares parameters and
    enforces valid usage before dispatching to the backend (fit_ncrf).

    Parameters
    ----------
    mu : float or 'auto'
        Regularization strength. Controls the trade-off between model complexity
        and smoothness.
        - larger mu → smoother / more stable model
        - smaller mu → more flexible but risk of overfitting
        - 'auto' → automatically selected (recommended for demo)

    n_iter : int, optional
        Total number of optimization iterations.
        More iterations improve convergence but increase runtime.

    n_iterf : int, optional
        Number of iterations for the forward step in the optimization.
        Used internally for staged or block-wise updates.

    n_iterc : int, optional
        Number of iterations for another internal update step (e.g., coordinate updates).
        Typically not needed unless tuning optimization behavior.

    normalize : bool, default=True
        Whether to normalize input data before fitting.
        Improves numerical stability and convergence.

    in_place : bool, default=True
        Whether to modify data in-place (faster, less memory) or create a copy (safer).
    """
    def __init__(self, *, mu: float = 'auto', n_iter: int = None, n_iterf: int = None, n_iterc: int = None, normalize: bool = True, in_place: bool = True):
        self.mu = mu
        self.n_iter = n_iter
        self.n_iterf = n_iterf
        self.n_iterc = n_iterc
        self.normalize = normalize
        self.in_place = in_place
    
    def parameters_for_partial(self) -> Dict[str, Any]:
        params = {
            "mu": self.mu,
            "n_iter": self.n_iter,
            "n_iterf": self.n_iterf,
            "n_iterc": self.n_iterc,
            "normalize": self.normalize,
            "in_place": self.in_place,
        }
        return {key: value for key, value in params.items() if value is not None}

    def normalize_trf_args(self, experiment, data, mask, state):
        state = dict(state)
        # NCRF always operates on sensor data and should not inherit
        # source-space configuration from the public TRF API.
        data = TestDims('sensor')
        mask = None
        state.pop('inv', None)
        return data, mask, state
