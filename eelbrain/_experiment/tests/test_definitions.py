# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import logging

import pytest

from eelbrain._data_obj import Dataset, Factor, Interaction, Var
from eelbrain._experiment.configuration import Configuration, ConfigurationError, find_dependent_epochs, find_epoch_vars, find_epochs_vars, sequence_arg
from eelbrain._experiment.derivative_cache import DerivativeRegistry
from eelbrain._experiment.preprocessing import RawApplyICA, RawFilter, RawICA, RawMaxwell, RawPipeGraph, RawReReference, RawSource, assemble_raw_pipes
from eelbrain._experiment.statistics import config as test_def
from eelbrain._experiment.variable_def import EvalVar, GroupVar, LabelVar, Variables, _find_unresolvable_columns
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


def test_reserved_variable_names():
    "Variables can not shadow event columns that the pipeline writes itself"
    for name in ['subject', 'acquisition', 'sample', 'value', 'index', 'epoch', 'accept', 'interpolate_channels', 'epochs', 'evoked', 'src', 'model', 'tmax']:
        with pytest.raises(ConfigurationError):
            Variables({name: EvalVar('a + b')})
    # a name that only resembles a reserved one is fine
    Variables({'epoch_index': EvalVar('a + b'), 'value_shifted': EvalVar('a + b')})


def test_variable_input_columns():
    "Which names in a definition are input columns is decided against the data"
    # a function, builtin or module is resolved by Dataset.eval, so it is not required
    # of the data, and such a variable is applied like any other
    events = Dataset({'value': Var([-1., 2.])})
    Variables({
        'absval': EvalVar('abs(value)'),
        'logval': EvalVar('numpy.log(absval)'),
        'intval': EvalVar('Var(value.x.astype(int))'),
        'labeled': LabelVar('abs(value)', {1.: 'a'}),
    }).resolve(events, input_events=True)
    assert list(events['absval']) == [1., 2.]
    assert list(events['labeled']) == ['a', '']
    assert _find_unresolvable_columns({'abs', 'numpy', 'Var', 'value'}, events) == set()

    # ... but a column of the same name is the data's, since it shadows the context in
    # Dataset.eval, so it is an input like any other and is tracked as one
    events = Dataset({'type': Factor(['a', 'b'])})
    assert EvalVar("type == 'a'")._input_vars() == {'type'}
    assert _find_unresolvable_columns({"type"}, events) == set()
    Variables({'is_a': EvalVar("type == 'a'")}).resolve(events, input_events=True)
    assert list(events['is_a']) == [True, False]
    # and its values reach a consumer that records them for a cache fingerprint
    assert list(Variables().resolve(events, names={'type'})['type']) == ['a', 'b']

    # where the column is absent, the context supplies the name instead and the
    # definition fails; that is reported rather than raised from deeper down
    with pytest.raises(ConfigurationError, match='evaluation context'):
        Variables({'is_a': EvalVar("type == 'a'")}).resolve(Dataset({'value': Var([1, 2])}), input_events=True)
    # a name that neither can supply is still reported as a missing input
    assert _find_unresolvable_columns({'type', 'typo'}, events) == {'typo'}


def test_variable_stages():
    "Partition into event and across-subject variables"
    variables = Variables({
        'side': LabelVar('value', {(1, 3): 'left', (2, 4): 'right'}),
        'score': LabelVar('subject', {'R0000': 1., 'R0001': 2.}),  # definition spans subjects
        'group': GroupVar(['g0', 'g1']),
        'is_g0': EvalVar("group == 'g0'"),  # derived from an across-subject variable
        'group_by_side': EvalVar("group % side"),  # mixes an across-subject and a trial-level input
    })
    assert list(variables.event_vars) == ['side']
    assert list(variables.across_subject_vars) == ['score', 'group', 'is_g0', 'group_by_side']
    # a variable keyed on the subject in some other way is not across-subject
    variables = Variables({'first': EvalVar("subject == 'R0000'")})
    assert list(variables.event_vars) == ['first']

    # variables from an enclosing scope (Test.vars nested in Pipeline.variables)
    test_vars = Variables({'is_g0': EvalVar("group == 'g0'"), 'target': EvalVar("value == 1")})
    assert list(test_vars.event_vars) == ['is_g0', 'target']
    assert list(test_vars._find_across_subject_vars({'group'})) == ['is_g0']


def test_resolve():
    "Variables.resolve: add what the data supports, and report what it does not"
    groups = {'g0': ('R0000',), 'g1': ('R0001', 'R0002'), 'all': ('R0000', 'R0001', 'R0002')}
    variables = Variables({
        'side': LabelVar('value', {(1, 3): 'left', (2, 4): 'right'}),
        'age': GroupVar(['g0', 'g1']),
        'is_g0': EvalVar("age == 'g0'"),
    })

    # full events: everything resolves
    events = Dataset({'subject': Factor(['R0000', 'R0000']), 'value': Var([1, 2])})
    assert variables.resolve(events, groups) == {}
    assert list(events['side']) == ['left', 'right']
    assert list(events['age']) == ['g0', 'g0']
    assert list(events['is_g0']) == [True, True]

    # an event shell: only what is computable from `subject` is added, silently
    shell = Dataset({'subject': Factor(['R0000', 'R0001'])})
    variables.resolve(shell, groups)
    assert list(shell) == ['subject', 'age', 'is_g0']

    # ... unless the caller says what it needs
    shell = Dataset({'subject': Factor(['R0000', 'R0001'])})
    assert list(variables.resolve(shell, groups, names={'age'})['age']) == ['g0', 'g1']
    with pytest.raises(ValueError, match="'side'"):
        variables.resolve(Dataset({'subject': Factor(['R0000'])}), groups, names={'side'})

    # without groups the data is from a single subject, where across-subject variables are absent
    events = Dataset({'subject': Factor(['R0000', 'R0000']), 'value': Var([1, 2])})
    variables.resolve(events)
    assert list(events) == ['subject', 'value', 'side']

    # the nodes that combine subjects get only the deferred definitions, so an event
    # variable is not re-derived from aggregated columns (Pipeline._across_subject_variables)
    deferred = Variables(variables.across_subject_vars)
    assert list(deferred.vars) == ['age', 'is_g0']
    aggregated = Dataset({'subject': Factor(['R0000', 'R0001']), 'value': Var([1.5, 2.5])})
    deferred.resolve(aggregated, groups)
    assert list(aggregated) == ['subject', 'value', 'age', 'is_g0']  # 'side' is not recomputed from the cell means


def test_resolve_overwrite():
    "A variable never replaces a column the data already provides"
    groups = {'g0': ('R0000',), 'g1': ('R0001',)}
    variables = Variables({'r': GroupVar(['g0', 'g1'])})
    # a TRF dataset provides its fit metrics, which no blacklist can enumerate
    trfs = Dataset({'subject': Factor(['R0000', 'R0001']), 'r': Var([0.1, 0.2])})
    with pytest.raises(ConfigurationError, match="'r'"):
        variables.resolve(trfs, groups)
    assert list(trfs['r']) == [0.1, 0.2]
    # the same name is fine where it is not a column
    evoked = Dataset({'subject': Factor(['R0000', 'R0001'])})
    variables.resolve(evoked, groups)
    assert list(evoked['r']) == ['g0', 'g1']
    # a variable that does not apply to the data is skipped rather than reported
    variables = Variables({'r': GroupVar(['g0', 'g1'], task='b')})
    variables.resolve(Dataset({'subject': Factor(['R0000']), 'r': Var([0.1])}, info={'task': 'a'}), groups)


def test_resolve_input_events():
    "Where the events are labeled, a variable that can not be computed is an error, not a later stage"
    variables = Variables({'side': LabelVar('valu', {1: 'left', 2: 'right'})})  # typo in the source column
    events = Dataset({'subject': Factor(['R0000', 'R0000']), 'value': Var([1, 2])})
    with pytest.raises(ConfigurationError, match="'valu'"):
        variables.resolve(events, input_events=True)
    # a variable for a different task is skipped rather than reported, since its inputs may be absent
    variables = Variables({'side': LabelVar('other', {1: 'left'}, task='b')})
    variables.resolve(Dataset({'value': Var([1, 2])}, info={'task': 'a'}), input_events=True)
    # and so are across-subject variables, which belong to a later stage
    variables = Variables({'age': GroupVar(['g0', 'g1']), 'is_g0': EvalVar("age == 'g0'")})
    events = Dataset({'subject': Factor(['R0000', 'R0000'])})
    variables.resolve(events, input_events=True)
    assert list(events) == ['subject']
    # a variable that names one defined later is an error where the events do not provide that column
    variables = Variables({'is_g0': EvalVar("age == 'g0'"), 'age': GroupVar(['g0', 'g1'])})
    events = Dataset({'subject': Factor(['R0000', 'R0000'])})
    with pytest.raises(ConfigurationError, match="'age' is defined as a variable but not applied before 'is_g0'"):
        variables.resolve(events, input_events=True)
    # ... but reads the input column where the events provide it
    events = Dataset({'subject': Factor(['R0000', 'R0000']), 'age': Factor(['g0', 'g1'])})
    variables.resolve(events, input_events=True)
    assert list(events['is_g0']) == [True, False]
    assert list(events['age']) == ['g0', 'g1']  # the deferred definition is not applied per subject
    # a variable can be named after an input column to relabel it (housekeeping)
    variables = Variables({'stimulus': LabelVar('stimulus', {'Clip 0': '0', 'Clip 1': '1'})})
    events = Dataset({'subject': Factor(['R0000', 'R0000']), 'stimulus': Factor(['Clip 0', 'Clip 1'])})
    variables.resolve(events, input_events=True)
    assert list(events['stimulus']) == ['0', '1']
    # ... but not where the column is not raw input (test_resolve_overwrite)
    with pytest.raises(ConfigurationError, match='would overwrite'):
        variables.resolve(Dataset({'stimulus': Factor(['Clip 0', 'Clip 1'])}))


def test_resolve_names_scope():
    "Only the variables the caller asks for have to resolve"
    variables = Variables({
        'side': LabelVar('value', {1: 'left'}, task='a'),
        'target': EvalVar('value == 1'),
    })
    # 'side' is restricted to another task, so it is not added; asking for 'target' alone is fine
    ds = Dataset({'value': Var([1, 2])}, info={'task': 'b'})
    assert list(variables.resolve(ds, names={'target'})) == ['target']
    assert 'side' not in ds
    # ... and a caller that does need it still gets told
    ds = Dataset({'value': Var([1, 2])}, info={'task': 'b'})
    with pytest.raises(ValueError, match="'side'"):
        variables.resolve(ds, names={'side', 'target'})


def test_resolve_task():
    "A task-restricted variable follows ds.info, or the task column where subjects are combined"
    variables = Variables({'side': LabelVar('value', {1: 'left', 2: 'right'}, task='a')})

    ds = Dataset({'value': Var([1, 2])}, info={'task': 'a'})
    variables.resolve(ds)
    assert list(ds['side']) == ['left', 'right']

    ds = Dataset({'value': Var([1, 2])}, info={'task': 'b'})
    variables.resolve(ds)
    assert 'side' not in ds

    # combined recordings carry the task in a column instead
    ds = Dataset({'value': Var([1, 2]), 'task': Factor(['a', 'a'])})
    variables.resolve(ds)
    assert list(ds['side']) == ['left', 'right']

    ds = Dataset({'value': Var([1, 2]), 'task': Factor(['b', 'b'])})
    variables.resolve(ds)
    assert 'side' not in ds

    # a task-restricted variable has no single answer for data that combines tasks
    ds = Dataset({'value': Var([1, 2]), 'task': Factor(['a', 'b'])})
    with pytest.raises(NotImplementedError, match='combines several tasks'):
        variables.resolve(ds)


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
