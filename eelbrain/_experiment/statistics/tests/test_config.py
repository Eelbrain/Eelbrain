# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain._experiment.configuration import ConfigurationError
from eelbrain._experiment.statistics.config import ANOVA, TTestIndependent, TTestRelated, TwoStageTest
from eelbrain._experiment.variable_def import EvalVar, LabelVar


def test_test_vars():
    "Variables a test reads; these are resolved against an event shell for fingerprints"
    # t-test
    test = TTestRelated('A', 'a', 'b')
    assert set(test._test_vars) == {'A'}
    # groups: TTestIndependent synthesizes a GroupVar of its own
    test = TTestIndependent('group', 'a', 'b')
    assert set(test._test_vars) == {'group'}
    assert set(test.vars.across_subject_vars) == {'group'}
    # within-ANOVA
    test = ANOVA('a * b * subject')
    assert test.model == 'a%b'
    assert set(test._test_vars) == {'a', 'b'}
    # between ANOVA
    with pytest.raises(ConfigurationError):
        ANOVA('a*b*c')
    test = ANOVA('a*b*c', model='')
    assert test.model == ''
    assert set(test._test_vars) == {'a', 'b', 'c'}
    # mixed ANOVA: the between-subject factor enters outside the model
    test = ANOVA('A * GR * subject(GR)')
    assert test.model == 'A'
    assert set(test._test_vars) == {'A', 'GR'}
    # two-stage: the stage-1 terms
    test = TwoStageTest("a + b + a*b", vars={'a': EvalVar('c * d'), 'b': EvalVar('c * e')})
    assert set(test._test_vars) == {'a', 'b'}
    test = TwoStageTest("a + b + a*b", vars={'a': LabelVar('c%d', {1: 'x'}), 'b': LabelVar('c%e', {1: 'x'})})
    assert set(test._test_vars) == {'a', 'b'}
