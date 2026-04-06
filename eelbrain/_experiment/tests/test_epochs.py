# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from types import SimpleNamespace

import pytest

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
    assert repr(continuous_epoch) == "ContinuousEpoch('task', sel='stim == 1')"


def test_prepare_continuous_epoch_dataset():
    epoch = ContinuousEpoch('task', 'stim == 1', pad_start=0.1, pad_end=0.2, split=0.5, samplingrate=200)
    ds = Dataset({
        'time': Var([0.0, 0.1, 0.2, 1.0, 1.1]),
        'i_start': Var([0, 100, 200, 1000, 1100]),
    })
    ds.info['sfreq'] = 1000
    ds.info['raw'] = SimpleNamespace(info={'sfreq': 1000})
    ds = epoch.prepare_selected_events(ds, 'R0001')
    tmin, tmax, tstop, baseline, decim, variable_tmax = epoch.extraction_parameters(
        ds,
        {
            'baseline': False,
            'samplingrate': None,
            'decim': None,
            'tmin': None,
            'tmax': None,
            'tstop': None,
            'pad': 0,
        },
    )

    assert ds.n_cases == 2
    assert ds.info['nested_events'] == 'events'
    assert tmin == -0.1
    assert list(tmax.x) == pytest.approx([0.4, 0.3])
    assert tstop is None
    assert baseline is False
    assert decim == 5
    assert variable_tmax is True
    assert 'T_relative' in ds[0, 'events']
