# autoflake: skip_file
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Components for implementing TRF estimators

Public to share with NCRF, but may change together with NCRF in the future.
The components for cross-validation are in the main namespace
(:func:`eelbrain.split_data`, :class:`eelbrain.Splits` and :class:`eelbrain.Split`).
"""
from ._config import init_worker, n_workers
from ._trf.shared import DeconvolutionData, RowBlock, SegmentedArray, lag_matrix, lag_window, package_values
