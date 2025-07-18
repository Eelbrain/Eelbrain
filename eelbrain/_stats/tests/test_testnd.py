# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import product
import pickle
import logging
import pytest
import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import eelbrain
from eelbrain import Dataset, NDVar, Categorial, Scalar, UTS, Sensor, configure, datasets, test, testnd, set_log_level, cwt_morlet
from eelbrain._exceptions import WrongDimensionError, ZeroVarianceError
from eelbrain._stats.testnd import Adjacency, NDPermutationDistribution, label_clusters, _MergedTemporalClusterDist, find_peaks, VectorDifferenceIndependent
from eelbrain._utils.system import IS_WINDOWS
from eelbrain.fmtxt import asfmtext
from eelbrain.testing import assert_dataobj_equal, assert_dataset_equal, requires_mne_sample_data


def test_anova():
    "Test testnd.ANOVA()"
    ds = datasets.get_uts(True, nrm=True)

    testnd.ANOVA('utsnd', 'A*B', data=ds)
    for samples in (0, 2):
        logging.info("TEST:  samples=%r" % samples)
        testnd.ANOVA('utsnd', 'A*B', data=ds, samples=samples)
        testnd.ANOVA('utsnd', 'A*B', data=ds, samples=samples, pmin=0.05)
        res = testnd.ANOVA('utsnd', 'A*B', data=ds, samples=samples, tfce=True)
        assert res._plot_model() == 'A%B'
    asfmtext(res)

    res = testnd.ANOVA('utsnd', 'A*B*rm', match=False, data=ds, samples=0, pmin=0.05)
    assert repr(res) == "<ANOVA 'utsnd', 'A*B*rm', match=False, samples=0, pmin=0.05, 'A': 17 clusters, 'B': 20 clusters, 'A x B': 22 clusters>"
    assert res._plot_model() == 'A%B'
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, samples=2, pmin=0.05)
    assert res.match == 'rm'
    assert repr(res) == "<ANOVA 'utsnd', 'A*B*rm', match='rm', samples=2, pmin=0.05, 'A': 17 clusters, p < .001, 'B': 20 clusters, p < .001, 'A x B': 22 clusters, p < .001>"
    assert res._plot_model() == 'A%B'

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert repr(res_) == repr(res)
    assert res_._plot_model() == 'A%B'

    # threshold-free
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, samples=10)
    assert res.match == 'rm'
    assert repr(res) == "<ANOVA 'utsnd', 'A*B*rm', match='rm', samples=10, 'A': p < .001, 'B': p < .001, 'A x B': p < .001>"
    assert 'A clusters' in res.clusters.info
    assert 'B clusters' in res.clusters.info
    assert 'A x B clusters' in res.clusters.info

    # no clusters
    res = testnd.ANOVA('uts', 'B', sub="A=='a1'", data=ds, samples=5, pmin=0.05, mintime=0.02)
    repr(res)
    assert 'v' in res.clusters
    assert 'p' in res.clusters
    assert res._plot_model() == 'B'

    # all effects with clusters
    res = testnd.ANOVA('uts', 'A*B*rm', match=False, data=ds, samples=5, pmin=0.05, tstart=0.1, mintime=0.02)
    assert set(res.clusters['effect'].cells) == set(res.effects)

    # some effects with clusters, some without
    res = testnd.ANOVA('uts', 'A*B*rm', data=ds, samples=5, pmin=0.05, tstart=0.37, mintime=0.02)
    assert res.match == 'rm'
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert_dataobj_equal(res.clusters, res_.clusters)

    # test multi-effect results (with persistence)
    # UTS
    res = testnd.ANOVA('uts', 'A*B*rm', data=ds, samples=5)
    assert res.match == 'rm'
    repr(res)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    resr = pickle.loads(string)
    tf_clusters = resr.find_clusters(pmin=0.05)
    peaks = resr.find_peaks()
    assert_dataobj_equal(tf_clusters, res.find_clusters(pmin=0.05))
    assert_dataobj_equal(peaks, res.find_peaks())
    assert tf_clusters.eval("p.min()") == peaks.eval("p.min()")
    unmasked = resr.f[0]
    masked = resr.masked_parameter_map(effect=0, pmin=0.05)
    assert_array_equal(masked.x <= unmasked.x, True)

    # reproducibility
    decimal = 12 if IS_WINDOWS else None  # FIXME: why is Windows sometimes different???
    res0 = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, pmin=0.05, samples=5)
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, pmin=0.05, samples=5)
    assert_dataset_equal(res.clusters, res0.clusters, decimal=decimal)
    configure(n_workers=0)
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, pmin=0.05, samples=5)
    assert_dataset_equal(res.clusters, res0.clusters, decimal=decimal)
    configure(n_workers=True)

    # permutation
    eelbrain._stats.permutation._YIELD_ORIGINAL = 1
    samples = 4
    # raw
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, samples=samples)
    for dist in res._cdist:
        assert len(dist.dist) == samples
        assert_array_equal(dist.dist, dist.parameter_map.abs().max())
    # TFCE
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, tfce=True, samples=samples)
    for dist in res._cdist:
        assert len(dist.dist) == samples
        assert_array_equal(dist.dist, dist.tfce_map.abs().max())
    # thresholded
    res1 = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, pmin=0.05, samples=samples)
    clusters = res1.find_clusters()
    for dist, effect in zip(res1._cdist, res1.effects):
        effect_idx = clusters.eval("effect == %r" % effect)
        vmax = clusters[effect_idx, 'v'].abs().max()
        assert len(dist.dist) == samples
        assert_array_equal(dist.dist, vmax)
    eelbrain._stats.permutation._YIELD_ORIGINAL = 0

    # 1d TFCE
    configure(n_workers=0)
    res = testnd.ANOVA('utsnd.rms(time=(0.1, 0.3))', 'A*B*rm', data=ds, tfce=True, samples=samples)
    configure(n_workers=True)

    # zero variance
    res2 = testnd.ANOVA('utsnd', 'A', data=ds)
    ds['utsnd'].x[:, 1, 10] = 0.
    zero_var = ds['utsnd'].var('case') == 0
    zv_index = tuple(i[0] for i in zero_var.nonzero())
    res1_zv = testnd.ANOVA('utsnd', 'A*B*rm', data=ds)
    res2_zv = testnd.ANOVA('utsnd', 'A', data=ds)
    for res, res_zv in ((res1, res1_zv), (res2, res2_zv)):
        for f, f_zv in zip(res.f, res_zv.f):
            assert_array_equal((f_zv == 0).x, zero_var.x)
            assert f_zv[zv_index] == 0
            f_zv[zv_index] = f[zv_index]
            assert_dataobj_equal(f_zv, f, decimal=decimal)

    # nested random effect
    res = testnd.ANOVA('uts', 'A * B * nrm(A)', data=ds, samples=10, tstart=.4)
    assert res.match == 'nrm(A)'
    assert [p.min() for p in res.p] == [0.0, 0.6, 0.9]

    # unequal argument length
    with pytest.raises(ValueError):
        testnd.ANOVA('uts', 'A[:-1]', data=ds)
    with pytest.raises(ValueError):
        testnd.ANOVA('uts[:-1]', 'A * B * nrm(A)', data=ds)


def test_anova_incremental():
    "Test testnd.ANOVA() with incremental f-tests"
    ds = datasets.get_uts()
    testnd.ANOVA('uts', 'A*B', data=ds[3:], pmin=0.05, samples=10)


@requires_mne_sample_data
def test_anova_parc():
    "Test ANOVA with parc argument and source space data"
    set_log_level('warning', 'mne')
    ds = datasets.get_mne_sample(src='ico', sub="side.isin(('L', 'R'))")
    y = ds['src'].sub(source=('lateraloccipital-lh', 'cuneus-lh'))
    y1 = y.sub(source='lateraloccipital-lh')
    y2 = y.sub(source='cuneus-lh')
    kwa = dict(ds=ds, tstart=0.2, tstop=0.3, samples=100)

    resp = testnd.ANOVA(y, "side*modality", pmin=0.05, parc='source', **kwa)
    c1p = resp.find_clusters(source='lateraloccipital-lh')
    c2p = resp.find_clusters(source='cuneus-lh')
    del c1p['p_parc', 'id']
    del c2p['p_parc', 'id']
    res1 = testnd.ANOVA(y1, "side*modality", pmin=0.05, **kwa)
    c1 = res1.find_clusters()
    del c1['id']
    res2 = testnd.ANOVA(y2, "side*modality", pmin=0.05, **kwa)
    c2 = res2.find_clusters()
    del c2['id']
    assert_dataset_equal(c1p, c1)
    assert_dataset_equal(c2p, c2)
    assert_array_equal(c2['p'], [0.85, 0.88, 0.97, 0.75, 0.99, 0.99, 0.98, 0.0,
                                 0.12, 0.88, 0.25, 0.97, 0.34, 0.96])

    # without multiprocessing
    configure(n_workers=0)
    ress = testnd.ANOVA(y, "side*modality", pmin=0.05, parc='source', **kwa)
    c1s = ress.find_clusters(source='lateraloccipital-lh')
    c2s = ress.find_clusters(source='cuneus-lh')
    del c1s['p_parc', 'id']
    del c2s['p_parc', 'id']
    assert_dataset_equal(c1s, c1)
    assert_dataset_equal(c2s, c2)
    configure(n_workers=True)

    # parc but single label
    resp2 = testnd.ANOVA(y2, "side*modality", pmin=0.05, parc='source', **kwa)
    c2sp = resp2.find_clusters(source='cuneus-lh')
    del c2sp['p_parc', 'id']
    assert_dataset_equal(c2sp, c2)

    # not defined
    with pytest.raises(NotImplementedError):
        testnd.ANOVA(y, "side*modality", tfce=True, parc='source', **kwa)


def test_clusterdist():
    "Test NDPermutationDistribution class"
    shape = (10, 6, 6, 4)
    locs = [[0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]]
    x = np.random.normal(0, 1, shape)
    sensor = Sensor(locs, ['0', '1', '2', '3'])
    sensor.set_adjacency(connect_dist=1.1)
    dims = ('case', UTS(-0.1, 0.1, 6), Scalar('dim2', range(6), 'unit'),
            sensor)
    y = NDVar(x, dims)

    # test connecting sensors
    logging.info("TEST:  connecting sensors")
    bin_map = np.zeros(shape[1:], bool)
    bin_map[:3, :3, :2] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = NDPermutationDistribution(y, 0, 1.5)
    print(repr(cdist))
    cdist.add_original(pmap)
    print(repr(cdist))
    assert cdist.n_clusters == 1
    assert_array_equal(cdist._original_cluster_map == cdist._cids[0],
                       cdist._crop(bin_map).swapaxes(0, cdist._nad_ax))
    assert cdist.parameter_map.dims == y.dims[1:]

    # test connecting many sensors
    logging.info("TEST:  connecting sensors")
    bin_map = np.zeros(shape[1:], bool)
    bin_map[:3, :3] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = NDPermutationDistribution(y, 0, 1.5)
    cdist.add_original(pmap)
    assert cdist.n_clusters == 1
    assert_array_equal(cdist._original_cluster_map == cdist._cids[0],
                       cdist._crop(bin_map).swapaxes(0, cdist._nad_ax))

    # test keeping sensors separate
    logging.info("TEST:  keeping sensors separate")
    bin_map = np.zeros(shape[1:], bool)
    bin_map[:3, :3, 0] = True
    bin_map[:3, :3, 2] = True
    pmap = np.random.normal(0, 1, shape[1:])
    np.clip(pmap, -1, 1, pmap)
    pmap[bin_map] = 2
    cdist = NDPermutationDistribution(y, 1, 1.5)
    cdist.add_original(pmap)
    assert cdist.n_clusters == 2

    # criteria
    ds = datasets.get_uts(True)
    res = testnd.TTestRelated('utsnd', 'A', match='rm', data=ds, samples=0, pmin=0.05)
    assert res.clusters['duration'].min() < 0.01
    assert res.clusters['n_sensors'].min() == 1
    res = testnd.TTestRelated('utsnd', 'A', match='rm', data=ds, samples=0, pmin=0.05,
                              mintime=0.02, minsensor=2)
    assert res.clusters['duration'].min() >= 0.02
    assert res.clusters['n_sensors'].min() == 2

    # 1d
    res1d = testnd.TTestRelated('utsnd.sub(time=0.1)', 'A', match='rm', data=ds,
                                samples=0, pmin=0.05)
    assert_dataobj_equal(res1d.p_uncorrected, res.p_uncorrected.sub(time=0.1))

    # TFCE
    logging.info("TEST:  TFCE")
    sensor = Sensor(locs, ['0', '1', '2', '3'])
    sensor.set_adjacency(connect_dist=1.1)
    time = UTS(-0.1, 0.1, 4)
    scalar = Scalar('scalar', range(10), 'unit')
    dims = ('case', time, sensor, scalar)
    rng = np.random.RandomState(0)
    y = NDVar(rng.normal(0, 1, (10, 4, 4, 10)), dims)
    cdist = NDPermutationDistribution(y, 3, None)
    cdist.add_original(y.x[0])
    cdist.finalize()
    assert cdist.dist.shape == (3,)
    # I/O
    string = pickle.dumps(cdist, pickle.HIGHEST_PROTOCOL)
    cdist_ = pickle.loads(string)
    assert repr(cdist_) == repr(cdist)
    # find peaks
    x = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [7, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 7, 5, 5, 0, 0],
                   [0, 0, 0, 0, 5, 4, 4, 4, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                   [0, 0, 0, 0, 7, 0, 0, 3, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    tgt = np.equal(x, 7)
    peaks = find_peaks(x, cdist._adjacency)
    logging.debug(' detected: \n%s' % (peaks.astype(int)))
    logging.debug(' target: \n%s' % (tgt.astype(int)))
    assert_array_equal(peaks, tgt)
    # testnd permutation result
    res = testnd.TTestOneSample(y, tfce=True, samples=3)
    if sys.version_info[0] == 3:
        target = [96.84232967, 205.83207424, 425.65942084]
    else:
        target = [77.5852307, 119.1976153, 217.6270428]
    assert_allclose(np.sort(res._cdist.dist), target)

    # parc with TFCE on unconnected dimension
    configure(False)
    x = rng.normal(0, 1, (10, 5, 2, 4))
    time = UTS(-0.1, 0.1, 5)
    categorial = Categorial('categorial', ('a', 'b'))
    y = NDVar(x, ('case', time, categorial, sensor))
    y0 = NDVar(x[:, :, 0], ('case', time, sensor))
    y1 = NDVar(x[:, :, 1], ('case', time, sensor))
    res = testnd.TTestOneSample(y, tfce=True, samples=3)
    res_parc = testnd.TTestOneSample(y, tfce=True, samples=3, parc='categorial')
    res0 = testnd.TTestOneSample(y0, tfce=True, samples=3)
    res1 = testnd.TTestOneSample(y1, tfce=True, samples=3)
    # cdist
    assert res._cdist.shape == (4, 2, 5)
    # T-maps don't depend on adjacency
    assert_array_equal(res.t.x[:, 0], res0.t.x)
    assert_array_equal(res.t.x[:, 1], res1.t.x)
    assert_array_equal(res_parc.t.x[:, 0], res0.t.x)
    assert_array_equal(res_parc.t.x[:, 1], res1.t.x)
    # TFCE-maps should always be the same because they're unconnected
    assert_array_equal(res.tfce_map.x[:, 0], res0.tfce_map.x)
    assert_array_equal(res.tfce_map.x[:, 1], res1.tfce_map.x)
    assert_array_equal(res_parc.tfce_map.x[:, 0], res0.tfce_map.x)
    assert_array_equal(res_parc.tfce_map.x[:, 1], res1.tfce_map.x)
    # Probability-maps should depend on what is taken into account
    p_a = res0.compute_probability_map().x
    p_b = res1.compute_probability_map().x
    assert_array_equal(res_parc.compute_probability_map(categorial='a').x, p_a)
    assert_array_equal(res_parc.compute_probability_map(categorial='b').x, p_b)
    p_parc = res_parc.compute_probability_map()
    assert_array_equal(p_parc.x, res.compute_probability_map().x)
    assert np.all(p_parc.sub(categorial='a').x >= p_a)
    assert np.all(p_parc.sub(categorial='b').x >= p_b)
    configure(True)


def test_corr():
    "Test testnd.Correlation()"
    ds = datasets.get_uts(True)

    # add correlation
    ds['utsnd'].x[:, 3:5, 50:65] += ds['Y'].x[:, None, None]

    res = testnd.Correlation('utsnd', 'Y', data=ds, samples=0)
    assert repr(res) == "<Correlation 'utsnd', 'Y', samples=0>"
    for s, t in product('01234', (0.1, 0.2, 0.35)):
        target = test.Correlation(ds['utsnd'].sub(sensor=s, time=t), ds['Y']).r
        assert res.r.sub(sensor=s, time=t) == pytest.approx(target)
    res = testnd.Correlation('utsnd', 'Y', 'rm', data=ds, samples=0)
    repr(res)
    res = testnd.Correlation('utsnd', 'Y', data=ds, samples=10, pmin=0.05)
    repr(res)
    res = testnd.Correlation('utsnd', 'Y', data=ds, samples=10, tfce=True)
    repr(res)

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert repr(res_) == repr(res)
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)
    assert_dataobj_equal(res.p, res_.p)


def test_t_contrast():
    ds = datasets.get_uts()

    # simple contrast
    res = testnd.TContrastRelated('uts', 'A', 'a1>a0', 'rm', data=ds, samples=10, pmin=0.05)
    assert repr(res) == "<TContrastRelated 'uts', 'A', 'a1>a0', match='rm', samples=10, pmin=0.05, 7 clusters, p < .001>"
    res_ = testnd.TTestRelated('uts', 'A', 'a1', 'a0', 'rm', data=ds)
    assert_array_equal(res.t.x, res_.t.x)

    # complex contrast
    res = testnd.TContrastRelated('uts', 'A%B', 'min(a0|b0>a1|b0, a0|b1>a1|b1)', 'rm', data=ds, samples=10, pmin=0.05)
    res_b0 = testnd.TTestRelated('uts', 'A%B', ('a0', 'b0'), ('a1', 'b0'), 'rm', data=ds)
    res_b1 = testnd.TTestRelated('uts', 'A%B', ('a0', 'b1'), ('a1', 'b1'), 'rm', data=ds)
    assert_array_equal(res.t.x, np.min([res_b0.t.x, res_b1.t.x], axis=0))

    # persistence
    string = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert repr(res_) == repr(res)
    assert_dataobj_equal(res.p, res_.p)

    # contrast with "*"
    res = testnd.TContrastRelated('uts', 'A%B', 'min(a1|b0>a0|b0, a1|b1>a0|b1)', 'rm', data=ds, tail=1, samples=0)

    # zero variance
    ds['uts'].x[:, 10] = 0.
    with pytest.raises(ZeroVarianceError):
        testnd.TContrastRelated('uts', 'A%B', 'min(a1|b0>a0|b0, a1|b1>a0|b1)', 'rm', tail=1, data=ds, samples=0)


def test_labeling():
    "Test cluster labeling"
    shape = (4, 20)
    pmap = np.empty(shape, float)
    edges = np.array([(0, 1), (0, 3), (1, 2), (2, 3)], np.uint32)
    conn = Adjacency((
        Scalar('graph', range(4), adjacency=edges),
        UTS(0, 0.01, 20)))
    criteria = None

    # some clusters
    pmap[:] = [[3, 3, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 8, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0],
               [0, 3, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0]]
    cmap, cids = label_clusters(pmap, 2, 0, conn, criteria)
    assert len(cids) == 6
    assert_array_equal(cmap > 0, np.abs(pmap) > 2)

    # some other clusters
    pmap[:] = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0],
               [0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 4, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 4, 1, 0, 4, 4, 0, 0],
               [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0]]
    cmap, cids = label_clusters(pmap, 2, 0, conn, criteria)
    assert len(cids) == 6
    assert_array_equal(cmap > 0, np.abs(pmap) > 2)


def test_ttest_1samp():
    "Test testnd.TTestOneSample()"
    ds = datasets.get_uts(True)

    # no clusters
    res0 = testnd.TTestOneSample('uts', sub="A == 'a0'", data=ds, samples=0)
    assert res0.p_uncorrected.min() < 0.05
    assert repr(res0) == "<TTestOneSample 'uts', sub=\"A == 'a0'\", samples=0>"

    # sub as array
    res1 = testnd.TTestOneSample('uts', sub=ds.eval("A == 'a0'"), data=ds, samples=0)
    assert repr(res1) == "<TTestOneSample 'uts', sub=<array>, samples=0>"

    # clusters without resampling
    res1 = testnd.TTestOneSample('uts', sub="A == 'a0'", data=ds, samples=0, pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert res1.clusters.n_cases == 1
    assert 'p' not in res1.clusters
    assert repr(res1) == "<TTestOneSample 'uts', sub=\"A == 'a0'\", samples=0, pmin=0.05, tstart=0, tstop=0.6, mintime=0.05, 1 clusters>"

    # persistence
    string = pickle.dumps(res1, pickle.HIGHEST_PROTOCOL)
    res1_ = pickle.loads(string)
    assert repr(res1_) == repr(res1)
    assert_dataobj_equal(res1.p_uncorrected, res1_.p_uncorrected)

    # clusters with resampling
    res2 = testnd.TTestOneSample('uts', sub="A == 'a0'", data=ds, samples=10, pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert res2.clusters.n_cases == 1
    assert res2.samples == 10
    assert 'p' in res2.clusters
    assert repr(res2) == "<TTestOneSample 'uts', sub=\"A == 'a0'\", samples=10, pmin=0.05, tstart=0, tstop=0.6, mintime=0.05, 1 clusters, p < .001>"

    # clusters with permutations
    dss = ds.sub("numpy.logical_and(A=='a0', B=='b0')")[:8]
    res3 = testnd.TTestOneSample('uts', sub="A == 'a0'", data=dss, samples=10000, pmin=0.05, tstart=0, tstop=0.6, mintime=0.05)
    assert repr(res3) == "<TTestOneSample 'uts', sub=\"A == 'a0'\", samples=255, pmin=0.05, tstart=0, tstop=0.6, mintime=0.05, 2 clusters, p = .020>"
    assert res3.clusters.n_cases == 2
    assert res3.samples == -1
    assert str(res3.clusters) == (
        '#   id   tstart   tstop   duration   v        p          sig\n'
        '------------------------------------------------------------\n'
        '0   3    0.08     0.34    0.26       95.692   0.019608   *  \n'
        '1   4    0.35     0.56    0.21       81.819   0.019608   *  \n'
        '------------------------------------------------------------\n'
        'NDVars: cluster')

    # nd
    dss = ds.sub("A == 'a0'")
    res = testnd.TTestOneSample('utsnd', data=dss, samples=1)
    res = testnd.TTestOneSample('utsnd', data=dss, pmin=0.05, samples=1)
    res = testnd.TTestOneSample('utsnd', data=dss, tfce=True, samples=1)

    # TFCE properties
    res = testnd.TTestOneSample('utsnd', sub="A == 'a0'", data=ds, samples=1)
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res = pickle.loads(string)
    tfce_clusters = res.find_clusters(pmin=0.05)
    peaks = res.find_peaks()
    assert tfce_clusters.eval("p.min()") == peaks.eval("p.min()")
    masked = res.masked_parameter_map(pmin=0.05)
    assert_array_equal(masked.abs().x <= res.t.abs().x, True)

    # zero variance
    ds['utsnd'].x[:, 1, 10] = 0.
    ds['utsnd'].x[:, 2, 10] = 0.1
    res = testnd.TTestOneSample('utsnd', data=ds, samples=0)
    assert res.t.x[1, 10] == 0.
    assert res.t.x[2, 10] > 1e10

    # argument length
    with pytest.raises(ValueError):
        testnd.TTestOneSample('utsnd', sub="A[:-1] == 'a0'", data=ds, samples=0)


def test_ttest_ind():
    "Test testnd.TTestIndependent()"
    ds = datasets.get_uts(True)

    # basic
    res = testnd.TTestIndependent('uts', 'A', 'a1', 'a0', data=ds, samples=0)
    assert repr(res) == "<TTestIndependent 'uts', 'A', 'a1' (n=30), 'a0' (n=30), samples=0>"
    assert res.p_uncorrected.min() < 0.05
    # persistence
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert repr(res_) == "<TTestIndependent 'uts', 'A', 'a1' (n=30), 'a0' (n=30), samples=0>"
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)
    # alternate argspec
    res_ = testnd.TTestIndependent("uts[A == 'a1']", "uts[A == 'a0']", data=ds, samples=0)
    assert repr(res_) == "<TTestIndependent 'uts' (n=30), 'uts' (n=30), samples=0>"
    assert_dataobj_equal(res_.t, res.t)

    # cluster
    res = testnd.TTestIndependent('uts', 'A', 'a1', 'a0', data=ds, tail=1, samples=1)
    # persistence
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert repr(res_) == repr(res)
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)

    # nd
    res = testnd.TTestIndependent('utsnd', 'A', 'a1', 'a0', data=ds, pmin=0.05, samples=2)
    assert res._cdist.n_clusters == 10

    # zero variance
    ds['utsnd'].x[:, 1, 10] = 0.
    res_zv = testnd.TTestIndependent('utsnd', 'A', 'a1', 'a0', data=ds, samples=0)
    assert_array_equal(res_zv.t.x[0], res.t.x[0])
    assert res_zv.t.x[1, 10] == 0.
    # argument mismatch
    with pytest.raises(ValueError):
        testnd.TTestIndependent(ds['utsnd'], ds[:-1, 'A'], samples=0)

    # large array (#52)
    ds = datasets.simulate_erp(60, time=UTS(-0.1, 0.001, 601), snr=0.7)
    res = testnd.TTestIndependent('eeg', 'predictability', 'low', 'high', data=ds, pmin=0.05, tstart=0.100, mintime=0.020, samples=20)
    clusters = res.find_clusters()
    assert_array_equal(clusters['p'], [0.4, 0.2, 0.4, 0.2, 0. ])


def test_ttest_rel():
    "Test testnd.TTestRelated()"
    ds = datasets.get_uts(True)

    # basic
    res = testnd.TTestRelated('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm', data=ds, samples=100)
    assert repr(res) == "<TTestRelated 'uts', 'A x B', ('a1', 'b1'), ('a0', 'b0'), 'rm' (n=15), samples=100, p < .001>"
    difference = res.masked_difference()
    assert difference.x.mask.sum() == 84
    c1 = res.masked_c1()
    assert c1.x.mask.sum() == 84
    assert_array_equal(c1.x.data, res.c1_mean.x)

    # alternate argspec
    res_ = testnd.TTestRelated("uts[A%B == ('a1', 'b1')]", "uts[A%B == ('a0', 'b0')]", data=ds, samples=100)
    assert repr(res_) == "<TTestRelated 'uts', 'uts' (n=15), samples=100, p < .001>"
    assert_dataobj_equal(res_.t, res.t)
    # alternate argspec 2
    ds1 = Dataset()
    ds1['a1b1'] = ds.eval("uts[A%B == ('a1', 'b1')]")
    ds1['a0b0'] = ds.eval("uts[A%B == ('a0', 'b0')]")
    res1 = testnd.TTestRelated('a1b1', 'a0b0', data=ds1, samples=100)
    assert_dataobj_equal(res1.t, res.t)
    assert repr(res1) == "<TTestRelated 'a1b1', 'a0b0' (n=15), samples=100, p < .001>"

    # persistence
    string = pickle.dumps(res, pickle.HIGHEST_PROTOCOL)
    res_ = pickle.loads(string)
    assert repr(res_) == repr(res)
    assert_dataobj_equal(res.p_uncorrected, res_.p_uncorrected)

    # collapsing cells
    res2 = testnd.TTestRelated('uts', 'A', 'a1', 'a0', 'rm', data=ds, samples=0)
    assert res2.p_uncorrected.min() < 0.05
    assert res2.n == res.n

    # reproducibility
    res3 = testnd.TTestRelated('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm', data=ds, samples=100)
    assert_dataset_equal(res3.find_clusters(maps=True), res.clusters)
    configure(n_workers=0)
    res4 = testnd.TTestRelated('uts', 'A%B', ('a1', 'b1'), ('a0', 'b0'), 'rm', data=ds, samples=100)
    assert_dataset_equal(res4.find_clusters(maps=True), res.clusters)
    configure(n_workers=True)
    sds = ds.sub("B=='b0'")
    # thresholded, UTS
    configure(n_workers=0)
    res0 = testnd.TTestRelated('uts', 'A', 'a1', 'a0', 'rm', data=sds, pmin=0.1, samples=100)
    tgt = res0.find_clusters()
    configure(n_workers=True)
    res1 = testnd.TTestRelated('uts', 'A', 'a1', 'a0', 'rm', data=sds, pmin=0.1, samples=100)
    assert_dataset_equal(res1.find_clusters(), tgt)
    # thresholded, UTSND
    configure(n_workers=0)
    res0 = testnd.TTestRelated('utsnd', 'A', 'a1', 'a0', 'rm', data=sds, pmin=0.1, samples=100)
    tgt = res0.find_clusters()
    configure(n_workers=True)
    res1 = testnd.TTestRelated('utsnd', 'A', 'a1', 'a0', 'rm', data=sds, pmin=0.1, samples=100)
    assert_dataset_equal(res1.find_clusters(), tgt)
    # TFCE, UTS
    configure(n_workers=0)
    res0 = testnd.TTestRelated('uts', 'A', 'a1', 'a0', 'rm', data=sds, tfce=True, samples=10)
    tgt = res0.compute_probability_map()
    configure(n_workers=True)
    res1 = testnd.TTestRelated('uts', 'A', 'a1', 'a0', 'rm', data=sds, tfce=True, samples=10)
    assert_dataobj_equal(res1.compute_probability_map(), tgt)

    # zero variance
    ds['utsnd'].x[:, 1, 10] = 0.
    res = testnd.TTestRelated('utsnd', 'A', match='rm', data=ds)
    assert res.t.x[1, 10] == 0

    # argument length
    with pytest.raises(ValueError):
        testnd.TTestRelated('utsnd', 'A[:-1]', match='rm', data=ds)
    with pytest.raises(ValueError):
        testnd.TTestRelated('utsnd', 'A', match='rm[:-1]', data=ds)


def test_vector():
    """Test vector tests"""
    # single vector
    ds = datasets.get_uv(vector=True)
    res = testnd.Vector('v[:40]', data=ds, samples=10)
    assert res.p == 0.0
    res = testnd.Vector('v[40:]', data=ds, samples=10)
    assert res.p == 1.0

    # single vector with norm stat
    res_t = testnd.Vector('v[:40]', data=ds, samples=10, norm=True)
    assert res_t.p == 0.0
    res_t = testnd.Vector('v[40:]', data=ds, samples=10, norm=True)
    assert res_t.p == 1.0

    # non-space tests should raise error
    with pytest.raises(WrongDimensionError):
        testnd.TTestOneSample('v', data=ds)
    with pytest.raises(WrongDimensionError):
        testnd.TTestRelated('v', 'A', match='rm', data=ds)
    with pytest.raises(WrongDimensionError):
        testnd.TTestIndependent('v', 'A', data=ds)
    with pytest.raises(WrongDimensionError):
        testnd.TContrastRelated('v', 'A', 'a0 > a1', 'rm', data=ds)
    with pytest.raises(WrongDimensionError):
        testnd.Correlation('v', 'fltvar', data=ds)
    with pytest.raises(WrongDimensionError):
        testnd.ANOVA('v', 'A * B', data=ds)

    # vector in time
    ds = datasets.get_uts(vector3d=True)
    v1 = ds[30:, 'v3d']
    v2 = ds[:30, 'v3d']
    vd = v1 - v2
    res = testnd.Vector(vd, samples=10)
    assert res.p.min() == 0
    difference = res.masked_difference(0.5)
    assert difference.x.mask.sum() == 288
    res_r = pickle.loads(pickle.dumps(res))
    assert repr(res_r) == repr(res)
    # diff related
    resd = testnd.VectorDifferenceRelated(v1, v2, samples=10)
    assert_dataobj_equal(resd.p, res.p, name=False)
    assert_dataobj_equal(resd.t2, res.t2, name=False)
    res_r = pickle.loads(pickle.dumps(resd))
    assert repr(res_r) == repr(resd)
    # diff independent
    res = VectorDifferenceIndependent(v1, v2, samples=10, norm=True)
    assert_dataobj_equal(res.difference, v1.mean('case') - v2.mean('case'), name=False)
    assert res.p.max() == 1
    assert res.p.min() == 0
    res_r = pickle.loads(pickle.dumps(res))
    assert repr(res_r) == repr(res)
    # with mp
    res = testnd.Vector(v1, samples=10)
    assert res.p.min() == 0.1
    # without mp
    configure(n_workers=0)
    res0 = testnd.Vector(v1, samples=10)
    assert_array_equal(np.sort(res0._cdist.dist), np.sort(res._cdist.dist))
    configure(n_workers=True)
    # time window
    res = testnd.Vector(v2, samples=10, tstart=0.1, tstop=0.4)
    assert res.p.min() == 0.2
    difference = res.masked_difference(0.5)
    assert difference.x.mask.sum() == 297

    # vector in time with norm stat
    res = testnd.Vector(vd, samples=10, norm=True)
    assert res.p.min() == 0
    difference = res.masked_difference()
    assert difference.x.mask.sum() == 285
    resd = testnd.VectorDifferenceRelated(v1, v2, samples=10, norm=True)
    assert_dataobj_equal(resd.p, res.p, name=False)
    assert_dataobj_equal(resd.difference, res.difference, name=False)

    v_small = v2 / 100
    res = testnd.Vector(v_small, tfce=True, samples=10, norm=True)
    assert 'WARNING' in repr(res)
    res = testnd.Vector(v_small, tfce=0.1, samples=10)
    assert res.p.min() == 0.0


def test_cwt():
    "Test tests with wavelet transform"
    ds = datasets.get_uts(True)
    ds['cwt'] = cwt_morlet(ds['utsnd'], np.arange(10, 20)).abs()
    res = testnd.TTestRelated('cwt', 'A', match='rm', data=ds, pmin=0.05, samples=10)
    cluster = res.clusters.sub("p == 0")
    assert_array_equal(cluster['frequency_min'], 10)
    assert_array_equal(cluster['frequency_max'], 19)


def test_merged_temporal_cluster_dist():
    "Test use of _MergedTemporalClusterDist with testnd test results"
    ds1 = datasets.get_uts()
    ds2 = datasets.get_uts(seed=42)

    anova_kw = dict(y='uts', x='A*B*rm', match='rm', pmin=0.05, samples=10)
    ttest_kw = dict(y='uts', x='A', c1='a1', c0='a0', pmin=0.05, samples=10)
    contrast_kw = dict(y='uts', x='A', contrast='a1>a0', pmin=0.05, samples=10)

    def test_merged(res1, res2):
        merged_dist = _MergedTemporalClusterDist([res1._cdist, res2._cdist])
        if isinstance(res1, testnd.ANOVA):
            assert len(merged_dist.dist) == len(res1.effects)
            for effect, dist in merged_dist.dist.items():
                assert effect in res1.effects
                assert len(dist) == res1.samples
        else:
            assert len(merged_dist.dist) == res1.samples
        res1_clusters = merged_dist.correct_cluster_p(res1)
        res2_clusters = merged_dist.correct_cluster_p(res2)
        for clusters in [res1_clusters, res2_clusters]:
            assert 'p_parc' in clusters
            for cl in clusters.itercases():
                assert cl['p_parc'] >= cl['p']

    # multi-effect
    res1 = testnd.ANOVA(data=ds1, **anova_kw)
    res2 = testnd.ANOVA(data=ds2, **anova_kw)
    test_merged(res1, res2)

    # TTestRelated
    res1 = testnd.TTestRelated(data=ds1, match='rm', **ttest_kw)
    res2 = testnd.TTestRelated(data=ds2, match='rm', **ttest_kw)
    test_merged(res1, res2)

    # TTestIndependent
    res1 = testnd.TTestIndependent(data=ds1, **ttest_kw)
    res2 = testnd.TTestIndependent(data=ds2, **ttest_kw)
    test_merged(res1, res2)

    # TTestOneSample
    res1 = testnd.TTestOneSample('uts', data=ds1, pmin=0.05, samples=10)
    res2 = testnd.TTestOneSample('uts', data=ds2, pmin=0.05, samples=10)
    test_merged(res1, res2)

    # TContrastRelated
    res1 = testnd.TContrastRelated(data=ds1, match='rm', **contrast_kw)
    res2 = testnd.TContrastRelated(data=ds2, match='rm', **contrast_kw)
    test_merged(res1, res2)
