# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Source-model and source-data configurations and derivatives.

The inverse-solution and source-space configuration classes live in
:mod:`._experiment.source.config`; the graph nodes that build the reusable
source-space products behind ``Pipeline.load_inv`` and the source-space branch
of ``Pipeline.load_evoked``/``Pipeline.load_epochs`` live in
:mod:`._experiment.source.nodes`.
"""

from .config import (
    INV_RE,
    SRC_RE,
    InverseSolution,
    MinimumNormInverseSolution,
    eval_src,
    parse_src,
)
from .nodes import (
    BemInput,
    EpochsStcDerivative,
    EvokedStcDerivative,
    EvokedStcGroupDatasetDerivative,
    FwdDerivative,
    InvDerivative,
    ROIData,
    SourceMorphDerivative,
    SourceProjection,
    SrcDerivative,
    TransInput,
    roi_data_from_dataset,
    _drop_unknown_labels,
    _source_parc,
    _subject_state,
)
