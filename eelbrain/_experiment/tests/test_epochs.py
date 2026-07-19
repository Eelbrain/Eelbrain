# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne
import numpy as np
import pytest

from eelbrain._exceptions import ConfigurationError
from eelbrain._experiment.epochs import assemble_epochs
from eelbrain._experiment.epochs.nodes import _epochs_artifact_metadata, _load_epochs, _save_epochs
from eelbrain._experiment.events import _combine_event_datasets
from eelbrain._data_obj import Datalist, Dataset, Factor, Var
from eelbrain.pipeline import PrimaryEpoch, SecondaryEpoch, SuperEpoch, EpochCollection, ContinuousEpoch


def test_prepare_continuous_epoch_dataset():
    epoch = ContinuousEpoch('task', 'stim == 1', pad_start=0.1, pad_end=0.2, split=0.5, samplingrate=200, run='2')
    assert 'name' not in epoch._as_dict()
    assert epoch.run == '2'
    assert epoch._as_dict()['run'] == '2'
    ds = Dataset({
        'onset': Var([0.0, 0.1, 0.2, 1.0, 1.1]),
        'sample': Var([0, 100, 200, 1000, 1100]),
    })
    ds.info['raw.samplingrate'] = 1000
    options = {
        'samplingrate': None,
        'decim': None,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'pad': 0,
    }
    ds = epoch._prepare_selected_events(ds, 'R0001', options)
    tmin, tmax, tstop, decim, variable_tmax = epoch._extraction_parameters(ds, options)

    assert ds.n_cases == 2
    assert ds.info['nested_events'] == 'events'
    assert tmin == -0.1
    assert list(tmax.x) == pytest.approx([0.4, 0.3])
    assert tstop is None
    assert decim == 5
    assert variable_tmax is True
    assert list(ds['epoch_time']) == pytest.approx([0.0, 1.0])
    assert list(ds[0, 'events']['epoch_time']) == pytest.approx([0.0, 0.1, 0.2])
    assert list(ds[1, 'events']['epoch_time']) == pytest.approx([1.0, 1.1])


def test_combine_event_dataset_bids_entities():
    "Varying BIDS entities become columns; invariant entities remain in info"
    info = {'subject': 'R0001', 'session': '', 'acquisition': ''}
    ds_1 = Dataset({'value': Var([1, 2])}, info={**info, 'task': 'story', 'run': '1'})
    ds_2 = Dataset({'value': Var([3]), 'run': Factor(['2'])}, info={**info, 'task': 'rest', 'run': '2'})
    ds = _combine_event_datasets([ds_1, ds_2])

    assert tuple(ds['task']) == ('story', 'story', 'rest')
    assert tuple(ds['run']) == ('1', '1', '2')
    assert 'task' not in ds.info
    assert 'run' not in ds.info
    assert all(ds.info[key] == value for key, value in info.items())


def test_shifted_epoch_time_serialization(tmp_path):
    "Shifted MNE epoch time axes survive the recording-epochs cache."
    info = mne.create_info(['EEG 001'], 100, 'eeg')
    data = np.zeros((1, 1, 50))
    epochs_0 = mne.EpochsArray(data, info, tmin=-0.1, verbose=False)
    epochs_1 = mne.EpochsArray(data, info, tmin=-0.1, verbose=False).shift_time(2.0)
    epochs = Datalist([epochs_0, epochs_1], 'epochs')
    path = tmp_path / 'epochs'

    _save_epochs(path, epochs)
    loaded = _load_epochs(path, _epochs_artifact_metadata(epochs))

    assert isinstance(loaded, Datalist)
    assert loaded[0].times[[0, -1]] == pytest.approx(epochs_0.times[[0, -1]])
    assert loaded[1].times[[0, -1]] == pytest.approx(epochs_1.times[[0, -1]])


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
    # _explicit_params records which kwargs were explicitly provided (empty here)
    assert super_epoch._explicit_params == ()
    assert repr(super_epoch) == "SuperEpoch(('a', 'b'))"

    collection = epochs['collection']
    assert collection.name == 'collection'
    assert collection.tasks == ('task-a', 'task-b')
    assert collection.rej_file_epochs == ['a', 'b']
    assert 'name' not in collection._as_dict()

    continuous = epochs['cont']
    assert continuous.name == 'cont'
    assert continuous.rej_file_epochs == ('cont',)
    assert 'name' not in continuous._as_dict()


def test_super_epoch_parameter_overrides():
    """SuperEpoch overrides are resolved at assembly time and relax sub-epoch agreement checks."""
    # Without override, sub-epochs must agree on INHERITED_PARAMS
    with pytest.raises(ConfigurationError, match="All sub-epochs must have the same setting for tmin"):
        assemble_epochs({
            'a': PrimaryEpoch('task', tmin=-0.1),
            'b': PrimaryEpoch('task', tmin=-0.2),
            'ab': SuperEpoch(('a', 'b')),
        }, ('task',))

    # With an override, sub-epochs may differ on the overridden param
    epochs = assemble_epochs({
        'a': PrimaryEpoch('task', tmin=-0.1),
        'b': PrimaryEpoch('task', tmin=-0.2),
        'ab': SuperEpoch(('a', 'b'), tmin=-0.3),
    }, ('task',))
    super_epoch = epochs['ab']
    assert super_epoch.tmin == -0.3
    assert super_epoch._explicit_params == ('tmin',)
    assert repr(super_epoch) == "SuperEpoch(('a', 'b'), tmin=-0.3)"


def test_assemble_epochs_detects_cycles():
    with pytest.raises(ConfigurationError, match="Can't resolve epoch dependencies"):
        assemble_epochs({'a': SecondaryEpoch('b'), 'b': SecondaryEpoch('a')}, ('a', 'b'))
