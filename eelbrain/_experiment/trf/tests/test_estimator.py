import pickle

import pytest

from eelbrain import NDVar, datasets, boosting
from eelbrain._experiment.trf.estimator import Boosting, Estimator, NCRF


def test_boosting():
    est = Boosting()
    assert isinstance(est, Estimator)
    assert est.requires_sensor_space is False
    assert est.extra_inputs == ()
    # _as_dict covers every DICT_ATTRS entry
    d = est._as_dict()
    assert d['type'] == 'Boosting'
    assert set(d) == {'type', *Boosting.DICT_ATTRS}
    # equality / repr
    assert est == Boosting()
    assert est != Boosting(basis=0.1)
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
    assert pickle.loads(pickle.dumps(est)) == est


def test_boosting_result_dataset():
    "Boosting._result_dataset exposes the right columns and info"
    ds = datasets.get_uts(utsnd=True)
    res = boosting(ds['utsnd'], ds['uts'], 0, 0.1, partitions=3, test=1, basis=0.05)
    est = Boosting()

    out = est._result_dataset(res, scale=None, trfs=True)
    assert out.n_cases == 1
    assert out.info['metrics'] == ['r', 'z', 'residual', 'det']  # no vector r1/z1
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
