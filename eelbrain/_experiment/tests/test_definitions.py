# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain._experiment import test_def
from eelbrain._experiment.definitions import DefinitionError, find_dependent_epochs, find_epoch_vars, find_epochs_vars, sequence_arg


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
    with pytest.raises(DefinitionError):
        test_def.ANOVA('a*b*c')
    test = test_def.ANOVA('a*b*c', model='')
    assert test.model == ''
    assert test._find_test_vars() == ({'a', 'b', 'c'}, none)
    # mixed ANOVA
    test = test_def.ANOVA('A * GR * subject(GR)')
    assert test.model == 'A'
    assert test._find_test_vars() == ({'A', 'GR'}, none)
    # two-stage
    test = test_def.TwoStageTest("a + b + a*b", vars={'a': 'c * d', 'b': 'c * e'})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
    test = test_def.TwoStageTest("a + b + a*b", vars={'a': 'c * d', 'b': 'c * e', 'x': 'something * nonexistent'})
    assert test._find_test_vars() == ({'c', 'd', 'e'}, none)
    test = test_def.TwoStageTest("a + b + a*b", vars={'a': ('c%d', {}), 'b': ('c%e', {})})
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
