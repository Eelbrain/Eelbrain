# Base class for TRF estimators
from typing import Any, Dict

# Abstract strategy interface
class Estimator:
    """Base for TRF estimator configs. Subclasses override parameters_for_partial()."""

    name: str = ""

    def parameters_for_partial(self) -> Dict[str, Any]:
        """
        Return kwargs for partial(fitter, ..., **kwargs).
        Subclasses override this to return their estimator-specific parameters.
        """
        return {}

    def normalize_trf_args(self, experiment, data, mask, state):
        """Normalize estimator-specific TRF args while preserving public API compatibility."""
        return data, mask, dict(state)
