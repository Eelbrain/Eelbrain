# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain._experiment.configuration import ConfigurationError
from eelbrain._experiment.statistics.config import ANOVA, TTestIndependent, TTestRelated, TwoStageTest
from eelbrain._experiment.variable_def import EvalVar, LabelVar


def test_find_test_vars():
    none = set()
    # t-test
    test = TTestRelated('A', 'a', 'b')
    assert test._find_test_vars() == ({'A'}, none)
    # groups
    test = TTestIndependent('group', 'a', 'b')
    assert test._find_test_vars() == (none, {'a', 'b'})
    # within-ANOVA
    test = ANOVA('a * b * subject')
    assert test.model == 'a%b'
    assert test._find_test_vars() == ({'a', 'b'}, none)
    # between ANOVA
    with pytest.raises(ConfigurationError):
        ANOVA('a*b*c')
    test = ANOVA('a*b*c', model='')
    assert test.model == ''
    assert test._find_test_vars() == ({'a', 'b', 'c'}, none)
    # mixed ANOVA
    test = ANOVA('A * GR * subject(GR)')
    assert test.model == 'A'
    assert test._find_test_vars() == ({'A', 'GR'}, none)
    # two-stage
    test = TwoStageTest("a + b + a*b", vars={'a': EvalVar('c * d'), 'b': EvalVar('c * e')})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
    test = TwoStageTest("a + b + a*b", vars={'a': EvalVar('c * d'), 'b': EvalVar('c * e'), 'x': EvalVar('something * nonexistent')})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
    test = TwoStageTest("a + b + a*b", vars={'a': LabelVar('c%d', {1: 'x'}), 'b': LabelVar('c%e', {1: 'x'})})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
