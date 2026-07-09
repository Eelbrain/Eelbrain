# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import logging

import pytest

from eelbrain._data_obj import Factor, Interaction, Var
from eelbrain._experiment.configuration import Configuration, ConfigurationError, find_dependent_epochs, find_epoch_vars, find_epochs_vars, sequence_arg
from eelbrain._experiment.derivative_cache import DerivativeRegistry
from eelbrain._experiment.preprocessing import RawApplyICA, RawFilter, RawICA, RawMaxwell, RawPipeGraph, RawReReference, RawSource, assemble_raw_pipes
from eelbrain._experiment.statistics import config as test_def
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
    assert config._as_dict() == {'type': 'ExampleConfiguration', 'a': 'x', 'b': 1}
    assert config == ExampleConfiguration('x', 1)
    assert config != ExampleConfiguration('x', 2)
    assert config == {'type': 'ExampleConfiguration', 'a': 'x', 'b': 1}


def test_config_normalization():
    config = ExampleSequenceConfiguration('x')
    assert config.items == ['x']
    assert config._as_dict() == {'type': 'ExampleSequenceConfiguration', 'items': ['x']}
    assert config == ExampleSequenceConfiguration(['x'])


def test_config_canonicalization_and_variables():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.config'))

    variables = Variables({'x': EvalVar('a + b', task='task-a')})
    canonical = registry.canonicalize({'vars': variables})
    assert canonical == {'vars': {'x': {'type': 'EvalVar', 'task': 'task-a', 'code': 'a + b'}}}

    test = test_def.ANOVA('x*subject', vars={'x': EvalVar('a + b', task='task-a')})
    canonical_test = registry.canonicalize(test._as_dict())
    assert canonical_test['vars'] == {'x': {'type': 'EvalVar', 'task': 'task-a', 'code': 'a + b'}}


def test_canonicalize_data_objects():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.config'))

    assert registry.canonicalize(Var([1, 2])) == [1, 2]
    assert registry.canonicalize(Factor(['a', 'b'], random=True)) == ['a', 'b']
    assert registry.canonicalize(Interaction([Factor(['a', 'b']), Factor(['x', 'y'])])) == [['a', 'x'], ['b', 'y']]


def test_vardef_semantic_identity():
    assert EvalVar('a + b', task='task-a') != EvalVar('a + b', task='task-b')
    assert GroupVar(('g1', 'g2'), task='task-a') != GroupVar(('g1', 'g2'), task='task-b')

    compact = LabelVar('value', {(1, 2): 'target'}, task='task-a')
    expanded = LabelVar('value', {1: 'target', 2: 'target'}, task='task-a')
    assert compact == expanded
    assert compact != LabelVar('value', {1: 'target', 2: 'target'}, task='task-b')


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


def test_epoch_rejection_semantic_dict():
    from eelbrain._experiment.epoch_rejection import ChannelModelRejection, EpochRejection, ManualRejection
    rej = ManualRejection(interpolation=False)
    assert isinstance(rej, EpochRejection)
    assert rej.interpolation is False
    assert rej._as_dict() == {'type': 'ManualRejection', 'interpolation': False}
    assert ManualRejection().interpolation is True

    auto = ChannelModelRejection(max_interpolate=3, score_threshold=1e-4, raw='1-40')
    assert isinstance(auto, EpochRejection)
    assert auto._as_dict() == {
        'type': 'ChannelModelRejection', 'interpolation': True, 'fit_threshold': 50e-6,
        'score_threshold': 1e-4, 'max_interpolate': 3, 'raw': '1-40', 'continuous': 5.,
        'window': 1.0, 'hop': 0.5, 'min_duration': 0.1, 'merge_gap': None,
        'model': 'huber', 'alpha': 1e-4, 'epsilon': 1.35,
    }


def test_reference_prepare_source_data():
    "Reference.prepare_source_data prepares EEG data for source localization"
    import numpy as np
    import mne
    from mne.minimum_norm.inverse import _check_reference
    from eelbrain._experiment.preprocessing import Reference
    mne.set_log_level('ERROR')

    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(['Fz', 'Pz', 'C3', 'C4'], 200., 'eeg')  # Cz absent
    raw = mne.io.RawArray(np.zeros((4, 200)), info)
    raw.set_montage(montage)

    # no add: adds an average-reference projection, accepted by MNE inverse modeling
    x = raw.copy()
    Reference('average')._prepare_source_data(x, montage)
    assert x.info['custom_ref_applied'] == 0
    _check_reference(x)  # must not raise

    # add: reconstruct the implicit channel as zeros + projection
    x = raw.copy()
    Reference('average', add='Cz')._prepare_source_data(x, montage)
    assert 'Cz' in x.ch_names
    assert np.allclose(x.get_data(picks=['Cz']), 0)
    assert x.info['custom_ref_applied'] == 0
    _check_reference(x)  # must not raise

    # MEG-only data: no-op (no EEG channels)
    meg = mne.io.RawArray(np.zeros((2, 200)), mne.create_info(['MEG 001', 'MEG 002'], 200., 'mag'))
    Reference('average')._prepare_source_data(meg)
    assert len(meg.info['projs']) == 0

    # only an average reference (optionally with add) is supported for source localization
    with pytest.raises(NotImplementedError):
        Reference(['M1', 'M2'])._prepare_source_data(raw.copy(), montage)
    with pytest.raises(NotImplementedError):
        Reference('average', drop='Fz')._prepare_source_data(raw.copy(), montage)


def test_raw_pipe_graph_lineage():
    raw = assemble_raw_pipes({
        'raw': RawSource(),
        '1-40': RawFilter('raw', 1, 40),
        'ica': RawICA('1-40'),
        'ica1-40': RawFilter('ica', 1, 40),
        'apply-ica': RawApplyICA('1-40', 'ica'),
    }, ('sample',))

    assert isinstance(raw, RawPipeGraph)
    assert raw.source_name('raw') is None
    assert raw.source_pipe('raw') is None
    assert raw.root_source_name('ica1-40') == 'raw'
    assert raw.root_source_pipe('apply-ica') is raw['raw']
    assert raw.ica_name('ica1-40') == 'ica'
    assert raw.ica_pipe('apply-ica') is raw['ica']
    assert tuple(pipe.name for pipe in raw.lineage_pipes('ica1-40')) == ('raw', '1-40', 'ica', 'ica1-40')
    assert raw['ica'].task == ('sample',)


def test_raw_configurations():
    # task=None with multiple tasks is only allowed after RawMaxwell
    with pytest.raises(ConfigurationError, match='RawMaxwell'):
        assemble_raw_pipes({
            'raw': RawSource(),
            'ica': RawICA('raw'),
        }, ('sample1', 'sample2'))

    # task=None with a single task: use that task, no run concatenation
    raw = assemble_raw_pipes({
        'raw': RawSource(),
        'ica': RawICA('raw'),
    }, ('sample',))
    assert raw['ica'].task == ('sample',)
    assert raw['ica']._concatenate_runs is False

    # task=None after RawMaxwell: accept all tasks and concatenate runs
    raw = assemble_raw_pipes({
        'raw': RawSource(),
        'maxwell': RawMaxwell('raw'),
        '1-40': RawFilter('maxwell', 1, 40),
        'ica': RawICA('1-40'),
    }, ('sample1', 'sample2'))
    assert raw['ica'].task == ('sample1', 'sample2')
    assert raw['ica']._concatenate_runs is True

    # explicit task after RawMaxwell also concatenates runs
    raw = assemble_raw_pipes({
        'raw': RawSource(),
        'maxwell': RawMaxwell('raw'),
        'ica': RawICA('maxwell', 'sample1'),
    }, ('sample1', 'sample2'))
    assert raw['ica'].task == ('sample1',)
    assert raw['ica']._concatenate_runs is True
