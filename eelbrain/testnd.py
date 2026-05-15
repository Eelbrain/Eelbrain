# autoflake: skip_file
"""Statistical tests for multidimensional data in :class:`NDVar` objects"""
__test__ = False
from ._stats.testnd import NDTest, MultiEffectNDTest, Correlation, TTestOneSample, TTestIndependent, TTestRelated, TContrastRelated, ANOVA, Vector, VectorDifferenceRelated
from ._stats.spm import LM, LMGroup
