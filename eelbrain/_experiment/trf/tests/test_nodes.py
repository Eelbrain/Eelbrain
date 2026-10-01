from types import SimpleNamespace

import pytest

from eelbrain import Dataset, Factor
from eelbrain._experiment.trf.estimator import Boosting
from eelbrain._experiment.trf.nodes import TRFDerivative, TRFModelTestDerivative


def test_trf_fingerprint_default_partitions():
    "TRFs with partitions inferred by boosting() record the default partitions"
    estimators = {
        'default': Boosting(),
        'inferred': Boosting(partitions=None),
        'inferred-no-cv': Boosting(partitions=None, cv=False),
        'explicit': Boosting(partitions=5),
    }
    node = TRFDerivative('.', estimators, {}, 'stimulus', {})
    fingerprints = {key: node.fingerprint(SimpleNamespace(options={'estimator': key})) for key in estimators}
    assert fingerprints['inferred']['default_partitions'] == 5
    assert fingerprints['inferred-no-cv']['default_partitions'] == 5
    assert 'default_partitions' not in fingerprints['default']
    assert 'default_partitions' not in fingerprints['explicit']


def test_model_test_dataset_alignment():
    ds1 = Dataset({'subject': Factor(('s1', 's2')), 'epoch': Factor(('a', 'a'))})
    ds0 = Dataset({'subject': Factor(('s2', 's1')), 'epoch': Factor(('a', 'a'))})
    with pytest.raises(RuntimeError, match='not aligned'):
        TRFModelTestDerivative._assert_aligned(ds1, ds0)


def test_model_test_metric_parts():
    assert TRFModelTestDerivative._metric_parts('ev') == ('ev', None)
    assert TRFModelTestDerivative._metric_parts('ev.sum') == ('ev', 'sum')
    assert TRFModelTestDerivative._metric_parts('ev.mean') == ('ev', 'mean')
    assert TRFModelTestDerivative._metric_parts('ev.max') == ('ev', 'max')
    with pytest.raises(ValueError, match="metric='ev.median'"):
        TRFModelTestDerivative._metric_parts('ev.median')
