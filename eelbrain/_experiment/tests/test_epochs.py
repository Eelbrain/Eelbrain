# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
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
