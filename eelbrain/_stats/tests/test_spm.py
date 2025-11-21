# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import datasets, configure
from eelbrain._stats.spm import LM, LMGroup


@pytest.mark.parametrize('n_workers', [False, True])
def test_lm(n_workers):
    if os.getenv("SKIP_N_WORKERS_TESTS") == "1" and n_workers:
        pytest.skip("Skipping n_workers tests")
    configure(n_workers=n_workers)

    ds = datasets.get_uts(utsnd=True)
    model = ds.eval("A*B*Y")
    coeffs = ds['uts'].ols(model)

    lm = LM('uts', 'A*B*Y', data=ds, coding='effect', samples=0)
    assert repr(lm) == "<LM 'uts', 'A*B*Y', samples=0>"
    for i, effect in enumerate(model.effects):
        assert_array_equal(lm.coefficient(effect.name).x, coeffs.x[i])

    # Permutation
    lm = LM('uts', 'A*B*Y', data=ds, coding='effect', samples=100, tstart=0.1)
    clusters = lm.find_clusters()
    assert clusters.n_cases == 14
    # Cluster-based
    lm = LM('uts', 'A*B*Y', data=ds, coding='effect', samples=100, tstart=0.1, pmin=0.05)
    assert lm.find_clusters(0.05).n_cases == 6

    # N-dim
    lm = LM('utsnd', 'A*B*Y', data=ds, coding='effect', samples=100, tstart=0.1)
    clusters = lm.find_clusters()
    assert clusters.n_cases == 57
    lm = LM('utsnd', 'A*B*Y', data=ds, coding='effect', samples=100, tstart=0.1, pmin=0.05)
    assert lm.find_clusters(0.05).n_cases == 7


def test_random_lm():
    np.random.seed(0)

    ds = datasets.get_uts()
    lms_dummy = []
    lms_effect = []
    for i in range(5):
        ds['uts'].x += np.random.normal(0, 2, ds['uts'].shape)
        lms_dummy.append(LM('uts', 'A*B*Y', data=ds, samples=0))
        lms_effect.append(LM('uts', 'A*B*Y', data=ds, samples=0, coding='effect'))

    # dummy coding
    rlm = LMGroup(lms_dummy)
    assert repr(rlm) == "<LMGroup 'uts', 'A*B*Y', n=5>"
    # coefficients
    ds = rlm.coefficients_dataset(('A', 'A x B'), long=True)
    assert ds['term'].cells == ('A', 'A x B')
    # tests
    res = rlm.column_ttest('A x B', samples=100, pmin=0.05, mintime=0.025)
    assert res.clusters.n_cases == 2

    # effect coding
    rlm = LMGroup(lms_effect)
    res = rlm.column_ttest('A x B', samples=100, pmin=0.05, mintime=0.025)
    assert res.clusters.n_cases == 5
    # persistence
    rlm_p = pickle.loads(pickle.dumps(rlm, pickle.HIGHEST_PROTOCOL))
    assert rlm_p.dims == rlm.dims
