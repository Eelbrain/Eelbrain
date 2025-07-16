"""Some basic statistical tests for univariate data."""
__test__ = False
from ._stats.glm import ANOVA
from ._stats.test import (
    Correlation, RankCorrelation,
    TTestOneSample, TTestIndependent, TTestRelated, MannWhitneyU, WilcoxonSignedRank,
    pairwise, ttest, correlations,
    lilliefors,
    pairwise_correlations,
)
