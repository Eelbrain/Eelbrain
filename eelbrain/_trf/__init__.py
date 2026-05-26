from importlib import import_module

__all__ = [
    "BoostingEstimator",
    "Code",
    "Estimator",
    "EventPredictor",
    "FilePredictor",
    "MakePredictor",
    "NCRFEstimator",
    "ResultCollection",
    "SessionPredictor",
    "TRFExperiment",
    "Term",
]

_MODULE_ATTRS = {
    "Code": ("._code", "Code"),
    "TRFExperiment": ("._experiment", "TRFExperiment"),
    "Term": ("._model", "Term"),
    "EventPredictor": ("._predictor", "EventPredictor"),
    "FilePredictor": ("._predictor", "FilePredictor"),
    "MakePredictor": ("._predictor", "MakePredictor"),
    "SessionPredictor": ("._predictor", "SessionPredictor"),
    "ResultCollection": ("._results", "ResultCollection"),
    "Estimator": ("._estimator", "Estimator"),
    "BoostingEstimator": ("._estimator", "BoostingEstimator"),
    "NCRFEstimator": ("._estimator", "NCRFEstimator"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _MODULE_ATTRS[name]
    except KeyError as err:
        raise AttributeError(name) from err
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
