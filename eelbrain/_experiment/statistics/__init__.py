# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistical test definitions and result derivatives.

The test definition classes (:class:`Configuration` subclasses) live in
:mod:`._experiment.statistics.config`; the graph nodes that compute test
results live in :mod:`._experiment.statistics.nodes` and, for two-stage tests,
:mod:`._experiment.statistics.two_stage_nodes`.
"""

from .config import (
    ANOVA,
    ResolvedTestNDSpec,
    TContrastRelated,
    TTestIndependent,
    TTestOneSample,
    TTestRelated,
    Test,
    TwoStageTest,
    tail_arg,
    validate_tests,
)
from .nodes import (
    EvokedTestDataDerivative,
    ROITestResult,
    ResultOutputDerivative,
    TestResultDerivative,
)
from .two_stage_nodes import (
    ROI2StageResult,
    TwoStageDataDerivative,
    TwoStageLevel1Derivative,
    TwoStageLevel2Derivative,
)
