# autoflake: skip_file
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Components for implementing TRF estimators

Public to share with NCRF, but may change together with NCRF in the future.
"""
from ._config import CONFIG, init_worker
from ._trf.shared import DeconvolutionData, RowBlock, SegmentedArray, Split, Splits, lag_matrix, lag_window, package_values, split_data


def n_workers() -> int:
    "Number of worker processes configured for multiprocessing (see :func:`eelbrain.configure`; 0 when multiprocessing is disabled)"
    return CONFIG['n_workers']
