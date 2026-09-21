# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Raw preprocessing configuration and graph-node support.

The configuration classes (:class:`RawPipe` and subclasses, :class:`Reference`,
the :class:`RawPipeGraph` collection, and naming helpers) live in
:mod:`._experiment.preprocessing.config`; the graph nodes that build and load
the concrete raw artifacts live in :mod:`._experiment.preprocessing.nodes`.
"""

from .config import (
    MNE_VERBOSITY,
    CachedRawPipe,
    RawApplyICA,
    RawNotchFilter,
    RawFilter,
    RawFilterElliptic,
    RawICA,
    RawMaxwell,
    RawOversampledTemporalProjection,
    RawPipe,
    RawPipeGraph,
    RawReReference,
    RawSource,
    Reference,
    assemble_raw_pipes,
    ica_input_name,
    raw_bad_channels_input_name,
    raw_input_name,
    raw_node_name,
)
from .nodes import (
    COORD_SCALE,
    REINDEX_ICA,
    canonical_recording,
    ICAInput,
    MaxwellCalibrationInput,
    MaxwellCrosstalkInput,
    CanonicalHeadPositionDerivative,
    RawBadChannelsInput,
    RawDerivative,
    RawHeadPositionDerivative,
    RawSourceDerivative,
    RawSourceInput,
    load_raw_dependency,
    load_raw_info_dependency,
)
