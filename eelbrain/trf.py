# autoflake: skip_file
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Components for implementing TRF estimators

Public to share with NCRF, but may change together with NCRF in the future.
"""
import os

from ._config import CONFIG
from ._trf.shared import DeconvolutionData, RowBlock, SegmentedArray, Split, Splits, lag_matrix, lag_window, package_values, split_data


def n_workers() -> int:
    "Number of worker processes configured for multiprocessing (see :func:`eelbrain.configure`; 0 when multiprocessing is disabled)"
    return CONFIG['n_workers']


def init_worker() -> None:
    "Apply Eelbrain's configuration to a worker process (the scheduling priority, see :func:`eelbrain.configure`); call from the worker"
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])
