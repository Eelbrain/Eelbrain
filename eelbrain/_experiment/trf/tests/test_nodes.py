from types import SimpleNamespace

import pytest

from eelbrain import Dataset, Factor
from eelbrain._experiment.trf.estimator import Boosting
from eelbrain._experiment.trf.nodes import TRFDatasetDerivative, TRFModelTestDerivative


def test_trf_dataset_without_metrics():
    "trfs=False is an empty dataset for an estimator without fit metrics"
    node = TRFDatasetDerivative('.', {'cv': Boosting(), 'no-cv': Boosting(cv=False)}, {})
    options = {'x': None, 'smooth': None, 'trfs': False}
    node.validate_options(SimpleNamespace(state={'inv': ''}, options={**options, 'estimator': 'cv'}))
    node.validate_options(SimpleNamespace(state={'inv': ''}, options={**options, 'estimator': 'no-cv', 'trfs': True}))
    with pytest.raises(ValueError, match='trfs=False'):
        node.validate_options(SimpleNamespace(state={'inv': ''}, options={**options, 'estimator': 'no-cv'}))


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
