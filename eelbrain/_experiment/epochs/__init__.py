# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Epoch definitions and epoch/evoked sensor derivatives.

The epoch definition classes live in :mod:`._experiment.epochs.config`; the
graph nodes that extract epoch and evoked sensor data live in
:mod:`._experiment.epochs.nodes`.
"""

from .config import (
    EPOCH_EXTRACT_OPTIONS,
    ContinuousEpoch,
    Epoch,
    EpochBase,
    EpochBaselineArg,
    EpochCollection,
    PrimaryEpoch,
    SecondaryEpoch,
    SuperEpoch,
    assemble_epochs,
    decim_param,
    single_recording_run,
)
from .nodes import (
    EpochsDerivative,
    EvokedDerivative,
    EvokedGroupDatasetDerivative,
    RecordingEpochsDerivative,
)
