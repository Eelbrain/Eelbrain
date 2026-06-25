import pickle

import pytest

from eelbrain._experiment.trf.estimator import Boosting, Estimator, NCRF


def test_boosting():
    est = Boosting()
    assert isinstance(est, Estimator)
    assert est.resolve_data(None) == 'source'
    assert est.resolve_data('sensor') == 'sensor'
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
    assert est.resolve_data(None) == 'sensor'
    with pytest.raises(ValueError):
        est.resolve_data('sensor')
    d = est._as_dict()
    assert d['type'] == 'NCRF'
    assert set(d) == {'type', *NCRF.DICT_ATTRS}
    assert NCRF() != NCRF(mu=0.5)
    assert pickle.loads(pickle.dumps(est)) == est
