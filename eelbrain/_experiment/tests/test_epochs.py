# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain._exceptions import ConfigurationError
from eelbrain._experiment.epochs import assemble_epochs
from eelbrain._data_obj import Dataset, Var
from eelbrain.pipeline import PrimaryEpoch, SecondaryEpoch, SuperEpoch, EpochCollection, ContinuousEpoch


def test_epoch_repr():
    primary_epoch = PrimaryEpoch('task')
    assert repr(primary_epoch) == "PrimaryEpoch('task', samplingrate=200, baseline=(None, 0))"
    secondary_epoch = SecondaryEpoch('primary_epoch', 'v == 1')
    assert repr(secondary_epoch) == "SecondaryEpoch('primary_epoch', 'v == 1')"
    super_epoch = SuperEpoch(('e1', 'e2'))
    assert repr(super_epoch) == "SuperEpoch(('e1', 'e2'))"
    epoch_collection = EpochCollection(('e1', 'e2'))
    assert repr(epoch_collection) == "EpochCollection(('e1', 'e2'))"
    continuous_epoch = ContinuousEpoch('task', 'stim == 1')
    assert repr(continuous_epoch) == "ContinuousEpoch(task='task', sel='stim == 1')"


def test_prepare_continuous_epoch_dataset():
    epoch = ContinuousEpoch('task', 'stim == 1', pad_start=0.1, pad_end=0.2, split=0.5, samplingrate=200)
    assert 'name' not in epoch._as_dict()
    ds = Dataset({
        'onset': Var([0.0, 0.1, 0.2, 1.0, 1.1]),
        'sample': Var([0, 100, 200, 1000, 1100]),
    })
    ds.info['raw.samplingrate'] = 1000
    options = {
        'baseline': False,
        'samplingrate': None,
        'decim': None,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'pad': 0,
    }
    ds = epoch._prepare_selected_events(ds, 'R0001', options)
    tmin, tmax, tstop, baseline, decim, variable_tmax = epoch._extraction_parameters(ds, options)

    assert ds.n_cases == 2
    assert ds.info['nested_events'] == 'events'
    assert tmin == -0.1
    assert list(tmax.x) == pytest.approx([0.4, 0.3])
    assert tstop is None
    assert baseline is False
    assert decim == 5
    assert variable_tmax is True
    assert 'T_relative' in ds[0, 'events']


def test_assemble_epochs_requires_epoch_objects():
    with pytest.raises(TypeError, match='need an epoch definition'):
        assemble_epochs({'target': {'task': 'sample'}}, ('sample',))


def test_assemble_epochs_stores_dependent_parameters():
    epochs = assemble_epochs({
        'a': PrimaryEpoch('task-a'),
        'b': PrimaryEpoch('task-b'),
        'a-sub': SecondaryEpoch('a'),
        'ab': SuperEpoch(('a', 'b')),
        'collection': EpochCollection(('a', 'b')),
        'cont': ContinuousEpoch('task-c'),
    }, ('task-a', 'task-b', 'task-c'))

    primary = epochs['a']
    assert primary.name == 'a'
    assert primary.task == 'task-a'
    assert primary.tasks == ('task-a',)
    assert primary.rej_file_epochs == ('a',)
    assert 'name' not in primary._as_dict()

    secondary = epochs['a-sub']
    assert secondary.name == 'a-sub'
    assert secondary.task == 'task-a'
    assert secondary.tasks == ('task-a',)
    assert secondary.rej_file_epochs == ('a',)
    assert 'name' not in secondary._as_dict()
    assert 'task' not in secondary._as_dict()
    assert 'tasks' not in secondary._as_dict()
    assert 'rej_file_epochs' not in secondary._as_dict()

    super_epoch = epochs['ab']
    assert super_epoch.name == 'ab'
    assert super_epoch.tasks == ('task-a', 'task-b')
    assert super_epoch.rej_file_epochs == ['a', 'b']
    assert 'name' not in super_epoch._as_dict()

    collection = epochs['collection']
    assert collection.name == 'collection'
    assert collection.tasks == ('task-a', 'task-b')
    assert collection.rej_file_epochs == ['a', 'b']
    assert 'name' not in collection._as_dict()

    continuous = epochs['cont']
    assert continuous.name == 'cont'
    assert continuous.rej_file_epochs == ('cont',)
    assert 'name' not in continuous._as_dict()


def test_assemble_epochs_detects_cycles():
    with pytest.raises(ConfigurationError, match="Can't resolve epoch dependencies"):
        assemble_epochs({'a': SecondaryEpoch('b'), 'b': SecondaryEpoch('a')}, ('a', 'b'))
