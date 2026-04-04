# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import logging

import pytest

from eelbrain._experiment import test_def
from eelbrain._experiment.configuration import Configuration, ConfigurationError, find_dependent_epochs, find_epoch_vars, find_epochs_vars, sequence_arg
from eelbrain._experiment.derivative_cache import DerivativeRegistry
from eelbrain._experiment.preprocessing import RawFilter, RawICA, RawReReference
from eelbrain._experiment.two_stage import TwoStageTest
from eelbrain._experiment.variable_def import EvalVar, GroupVar, LabelVar, Variables
from eelbrain.testing import TempDir


class ExampleConfiguration(Configuration):
    DICT_ATTRS = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class ExampleSequenceConfiguration(Configuration):
    DICT_ATTRS = ('items',)

    def __init__(self, items):
        self.items = sequence_arg('items', items, str, sequence_type=list)


def test_find_epoch_vars():
    assert find_epoch_vars({'sel': "myvar == 'x'"}) == {'myvar'}
    assert find_epoch_vars({'post_baseline_trigger_shift': "myvar"}) == {'myvar'}

    epochs = {'a': {'sel': "vara == 'a'"},
              'b': {'sel': "logical_and(varb == 'b', varc == 'c')"},
              'sec': {'sel_epoch': 'a', 'sel': "svar == 's'"},
              'super': {'sub_epochs': ('a', 'b')}}
    assert find_epochs_vars(epochs) == {'a': {'vara'},
                                        'b': {'logical_and', 'varb', 'varc'},
                                        'sec': {'vara', 'svar'},
                                        'super': {'vara', 'logical_and', 'varb', 'varc'}}
    assert set(find_dependent_epochs('a', epochs)) == {'sec', 'super'}
    assert find_dependent_epochs('b', epochs) == ['super']
    assert find_dependent_epochs('sec', epochs) == []
    assert find_dependent_epochs('super', epochs) == []


def test_find_test_vars():
    none = set()
    # t-test
    test = test_def.TTestRelated('A', 'a', 'b')
    assert test._find_test_vars() == ({'A'}, none)
    # groups
    test = test_def.TTestIndependent('group', 'a', 'b')
    assert test._find_test_vars() == (none, {'a', 'b'})
    # within-ANOVA
    test = test_def.ANOVA('a * b * subject')
    assert test.model == 'a%b'
    assert test._find_test_vars() == ({'a', 'b'}, none)
    # between ANOVA
    with pytest.raises(ConfigurationError):
        test_def.ANOVA('a*b*c')
    test = test_def.ANOVA('a*b*c', model='')
    assert test.model == ''
    assert test._find_test_vars() == ({'a', 'b', 'c'}, none)
    # mixed ANOVA
    test = test_def.ANOVA('A * GR * subject(GR)')
    assert test.model == 'A'
    assert test._find_test_vars() == ({'A', 'GR'}, none)
    # two-stage
    test = TwoStageTest("a + b + a*b", vars={'a': 'c * d', 'b': 'c * e'})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
    test = TwoStageTest("a + b + a*b", vars={'a': 'c * d', 'b': 'c * e', 'x': 'something * nonexistent'})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
    test = TwoStageTest("a + b + a*b", vars={'a': ('c%d', {}), 'b': ('c%e', {})})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)


def test_sequence_arg():
    # single value
    assert sequence_arg('sequence', 'a', str) == ('a',)
    assert sequence_arg('sequence', 1, int) == (1,)
    assert sequence_arg('sequence', 1, int, sequence_type=list) == [1]
    # list/tuple
    assert sequence_arg('sequence', ['a', 'b'], str) == ('a', 'b')
    assert sequence_arg('sequence', ('a', 'b'), str) == ('a', 'b')
    assert sequence_arg('sequence', [1, 2], int) == (1, 2)
    assert sequence_arg('sequence', (1, 2), int) == (1, 2)
    # wrong type
    with pytest.raises(TypeError):
        sequence_arg('sequence', 1.5, int)
    with pytest.raises(TypeError):
        sequence_arg('sequence', ['a', 2], str)
    with pytest.raises(TypeError):
        sequence_arg('sequence', (1, 'b'), int)


def test_config_base():
    config = ExampleConfiguration('x', 1)
    assert config._as_dict() == {'a': 'x', 'b': 1}
    assert config == ExampleConfiguration('x', 1)
    assert config != ExampleConfiguration('x', 2)
    assert config == {'a': 'x', 'b': 1}


def test_config_normalization():
    config = ExampleSequenceConfiguration('x')
    assert config.items == ['x']
    assert config._as_dict() == {'items': ['x']}
    assert config == ExampleSequenceConfiguration(['x'])


def test_config_canonicalization_and_variables():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.config'))

    variables = Variables({'x': EvalVar('a + b', task='task-a')})
    canonical = registry.canonicalize({'vars': variables})
    assert canonical == {'vars': {'x': {'task': 'task-a', 'code': 'a + b'}}}

    test = test_def.ANOVA('x*subject', vars={'x': EvalVar('a + b', task='task-a')})
    canonical_test = registry.canonicalize(test._as_dict())
    assert canonical_test['vars'] == {'x': {'task': 'task-a', 'code': 'a + b'}}


def test_vardef_semantic_identity():
    assert EvalVar('a + b', task='task-a') != EvalVar('a + b', task='task-b')
    assert GroupVar(('g1', 'g2'), task='task-a') != GroupVar(('g1', 'g2'), task='task-b')

    compact = LabelVar('trigger', {(1, 2): 'target'}, task='task-a')
    expanded = LabelVar('trigger', {1: 'target', 2: 'target'}, task='task-a')
    assert compact == expanded
    assert compact != LabelVar('trigger', {1: 'target', 2: 'target'}, task='task-b')


def test_raw_pipe_semantic_dict():
    pipe = RawFilter('raw', 1, 40, n_jobs=2, method='iir')
    assert pipe._as_dict() == {
        'type': 'RawFilter',
        'source': 'raw',
        'l_freq': 1,
        'h_freq': 40,
        'n_jobs': 2,
        'kwargs': {'method': 'iir'},
    }
    assert 'name' not in pipe._as_dict()

    ica = RawICA('raw', 'task-a')
    assert ica.task == ('task-a',)
    assert ica._as_dict()['task'] == ('task-a',)

    reref = RawReReference('raw', ['A1', 'A2'], add='EXG1', drop='EXG8')
    assert reref.reference == ['A1', 'A2']
    assert reref.add == ['EXG1']
    assert reref.drop == ['EXG8']
