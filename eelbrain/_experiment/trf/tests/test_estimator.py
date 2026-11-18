import pickle

import pytest

from eelbrain import NDVar, concatenate, datasets, boosting
from eelbrain._experiment.trf.estimator import Boosting, Estimator, NCRF
from eelbrain._experiment.trf.nodes import TRFDerivative


def test_boosting():
    est = Boosting()
    assert isinstance(est, Estimator)
    assert est.requires_sensor_space is False
    assert est.extra_inputs == ()
    # _as_dict covers every DICT_ATTRS entry
    d = est._as_dict()
    assert d['type'] == 'Boosting'
    assert set(d) == {'type', *Boosting.DICT_ATTRS}.difference(('scheme', 'validate', 'min_section'))
    # equality / repr
    assert est == Boosting()
    assert est != Boosting(basis=0.1)
    # cross-validation
    assert est._split_args == {'partitions': 5, 'split': 'time', 'scheme': 'k-fold', 'validate': None, 'test': 1, 'min_section': None}
    with pytest.raises(ValueError):
        Boosting(partitions=-1)
    with pytest.raises(ValueError):
        Boosting(partitions=2)
    assert Boosting(partitions=2, test=0).partitions == 2
    with pytest.raises(ValueError, match='validation set'):
        Boosting(validate=0)
    with pytest.raises(ValueError, match='test=0'):
        Boosting(scheme='walk-forward')
    walk_forward = Boosting(scheme='walk-forward', validate=0.1, test=0)
    assert walk_forward._split_args == {'partitions': 5, 'split': 'time', 'scheme': 'walk-forward', 'validate': 0.1, 'test': 0, 'min_section': None}
    assert walk_forward._as_dict()['scheme'] == 'walk-forward'
    # results cached up to 0.43.0b4: no split parameter, cv instead of test, and negative partitions to concatenate the cases
    stored = {key: value for key, value in d.items() if key not in ('split', 'test')}
    stored['cv'] = True
    TRFDerivative.normalize_stored_fingerprint(None, fingerprint := {'estimator': {**stored, 'partitions': -5}})
    legacy = Boosting(partitions=-5)
    assert legacy == fingerprint['estimator']
    assert legacy._concatenate_cases
    assert legacy._split_args['partitions'] == 5
    assert not est._concatenate_cases
    TRFDerivative.normalize_stored_fingerprint(None, fingerprint := {'estimator': {**stored, 'partitions': 4, 'cv': False}})
    assert Boosting(partitions=4, split=None, test=0) == fingerprint['estimator']
    assert Boosting(partitions=4, test=0) != fingerprint['estimator']
    with pytest.raises(ValueError):
        Boosting(partitions=-5, split='case')
    with pytest.raises(ValueError):
        Boosting(partitions=-5, scheme='walk-forward', test=0)
    # no fit metrics without test set
    assert Boosting(test=0).metric_keys == ()
    assert repr(Boosting(basis=0.1, backward=True)) == "Boosting(basis=0.1, backward=True)"
    # picklable
    assert pickle.loads(pickle.dumps(est)) == est


def test_ncrf():
    est = NCRF(mu=0.5)
    assert isinstance(est, Estimator)
    assert est.extra_inputs == ('fwd', 'cov')
    assert est.requires_sensor_space is True
    d = est._as_dict()
    assert d['type'] == 'NCRF'
    assert set(d) == {'type', *NCRF.DICT_ATTRS}
    assert NCRF() != NCRF(mu=0.5)
    # cross-validation without test set
    assert est._split_args == {'partitions': None, 'split': None, 'scheme': 'walk-forward', 'validate': None, 'test': 0, 'min_section': None}
    assert pickle.loads(pickle.dumps(est)) == est


def test_boosting_fit():
    "Boosting._fit applies the estimator's cross-validation parameters"
    ds = datasets.get_uts()
    y, x = ds['uts'][:6], ds['uts'][6:12]
    x.name = 'x'
    res = Boosting(partitions=3, split='case', test=0)._fit(y.copy(), [x.copy()], 0, 0.1)
    assert (res.splits.split, res.splits.n_partitions, res.splits.n_test) == ('case', 3, 0)
    assert len(res.splits.segments) == 6
    # legacy: negative partitions concatenate the cases
    res = Boosting(partitions=-3, test=0)._fit(y.copy(), [x.copy()], 0, 0.1)
    assert len(res.splits.segments) == 1
    target = boosting(concatenate(y), concatenate(x), 0, 0.1, basis=0.05, partitions=3, split=None, test=0)
    assert res.splits == target.splits
    assert (res.h.x == target.h.x).all()


def test_boosting_result_dataset():
    "Boosting._result_dataset exposes the right columns and info"
    ds = datasets.get_uts(utsnd=True)
    res = boosting(ds['utsnd'], ds['uts'], 0, 0.1, partitions=3, test=1, basis=0.05)
    est = Boosting()

    out = est._result_dataset(res, scale=None, trfs=True)
    assert out.n_cases == 1
    assert out.info['metrics'] == ['r', 'z', 'residual', 'ev']  # no vector r1/z1
    assert 'det' not in out
    assert out.info['xs'] == ['uts']
    assert isinstance(out['r'], NDVar)
    assert isinstance(out['uts'], NDVar)
    assert out.info['samplingrate'] == 1 / res.h_source.time.tstep

    # scale='original' uses the rescaled kernel; trfs=False drops the kernel
    scaled = est._result_dataset(res, scale='original', trfs=True)
    assert (scaled['uts'].x != out['uts'].x).any()
    metrics_only = est._result_dataset(res, scale=None, trfs=False)
    assert metrics_only.info['xs'] == []
    assert 'uts' not in metrics_only

    with pytest.raises(ValueError):
        est._result_dataset(res, scale='bad', trfs=True)
