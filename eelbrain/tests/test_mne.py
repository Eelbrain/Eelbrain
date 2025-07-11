"""Test mne interaction"""
import os
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import mne
from nibabel.freesurfer import read_annot
import pytest

from eelbrain import (
    datasets, load, table, testnd,
    Dataset, Factor,
    concatenate, labels_from_clusters, morph_source_space, set_parc, xhemi)
from eelbrain._data_obj import SourceSpace, asndvar, _matrix_graph
from eelbrain._mne import shift_mne_epoch_trigger, combination_label
from eelbrain.testing import requires_mne_sample_data, requires_mne_testing_data
from eelbrain.tests.test_data import assert_dataobj_equal


def assert_label_equal(l0, l1, comment=True, color=True):
    if isinstance(l0, mne.BiHemiLabel):
        assert isinstance(l1, mne.BiHemiLabel)
        assert_label_equal(l0.lh, l1.lh, comment, color)
        assert_label_equal(l0.rh, l1.rh, comment, color)
    else:
        if comment:
            assert l0.comment == l1.comment
        if color:
            assert l0.color == l1.color
        assert l0.hemi == l1.hemi
        assert l0.subject == l1.subject
        assert_array_equal(l0.vertices, l1.vertices)
        assert_allclose(l0.pos, l1.pos)
        assert_allclose(l0.values, l1.values)


@requires_mne_sample_data
def test_source_estimate():
    "Test SourceSpace dimension"
    mne.set_log_level('warning')
    # load data
    ds = datasets.get_mne_sample(src='ico')
    dsa = ds.aggregate('side')
    src = ds['src']
    source = src.source

    # test auto-conversion
    asndvar('epochs', data=ds)
    asndvar('epochs', data=dsa)
    asndvar(dsa['epochs'][0])

    # source space clustering
    res = testnd.TTestIndependent('src', 'side', data=ds, samples=0, pmin=0.05, tstart=0.05, mintime=0.02, minsource=10)
    assert res.clusters.n_cases == 52

    # test disconnecting parc
    orig_conn = set(map(tuple, source.adjacency()))
    disc_conn = set(map(tuple, source.adjacency(True)))
    assert len(disc_conn) < len(orig_conn)
    for pair in orig_conn:
        s, d = pair
        if pair in disc_conn:
            assert source.parc[s] == source.parc[d]
        else:
            assert source.parc[s] != source.parc[d]

    # threshold-based test with parc
    srcl = src.sub(source='lh')
    res = testnd.TTestIndependent(srcl, 'side', data=ds, samples=10, pmin=0.05, tstart=0.05, mintime=0.02, minsource=10, parc='source')
    assert res._cdist.dist.shape[1] == len(srcl.source.parc.cells)
    label = 'superiortemporal-lh'
    c_all = res.find_clusters(maps=True)
    c_label = res.find_clusters(maps=True, source=label)
    assert_array_equal(c_label['location'], label)
    for case in c_label.itercases():
        id_ = case['id']
        idx = c_all['id'].index(id_)[0]
        assert case['v'] == c_all[idx, 'v']
        assert case['tstart'] == c_all[idx, 'tstart']
        assert case['tstop'] == c_all[idx, 'tstop']
        assert case['p'] <= c_all[idx, 'p']
        assert_dataobj_equal(case['cluster'], c_all[idx, 'cluster'].sub(source=label))

    # threshold-free test with parc
    res = testnd.TTestIndependent(srcl, 'side', data=ds, samples=10, tstart=0.05, parc='source')
    cl = res.find_clusters(0.05)
    assert cl.eval("p.min()") == res.p.min()
    mp = res.masked_parameter_map()
    assert mp.min() == res.t.min()
    assert mp.max() == res.t.max(res.p <= 0.05)
    assert mp.max() == pytest.approx(-4.95817732)

    # indexing source space
    s_sub = src.sub(source='fusiform-lh')
    idx = source.index_for_label('fusiform-lh')
    s_idx = src[idx]
    assert_dataobj_equal(s_sub, s_idx)

    # concatenate
    src_reconc = concatenate((src.sub(source='lh'), src.sub(source='rh')), 'source')
    assert_dataobj_equal(src_reconc, src)
    # label
    src_sm = src.sub(source=['superiortemporal-lh', 'middletemporal-lh'])
    src_s = src.sub(source='superiortemporal-lh')
    src_m = src.sub(source='middletemporal-lh')
    src_conc = concatenate([src_s, src_m], 'source')
    assert_dataobj_equal(src_conc, src_sm)
    assert_array_equal(src_conc.source.parc, src_sm.source.parc)

    # melt NDVar
    dsa['src_sub'] = dsa['src'].sub(source=['lateraloccipital-lh', 'lateraloccipital-rh'])
    mdsa = table.melt_ndvar('src_sub', 'source', ds=dsa)
    assert_array_equal(mdsa['hemi'], Factor(['lh'] * 88 * 2 + ['rh'] * 87 * 2))


@requires_mne_sample_data
def test_dataobjects():
    "Test handing MNE-objects as data-objects"
    shift = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1])
    epochs = datasets.get_mne_epochs()
    ds = Dataset({'a': Factor('ab', repeat=8), 'epochs': epochs})
    ds['ets'] = shift_mne_epoch_trigger(epochs, shift, min(shift), max(shift))

    # ds operations
    ds.sub("a == 'a'")
    ds.aggregate('a')

    # asndvar
    asndvar(ds['epochs'])
    asndvar(ds['ets'])

    # adjacency
    ds = datasets.get_mne_sample(sub=[0], sns=True)
    sensor = ds['meg'].sensor
    c = sensor.adjacency()
    assert_array_equal(c[:, 0] < c[:, 1], True)
    assert c.max() == len(sensor) - 1


@requires_mne_sample_data
def test_epoch_trigger_shift():
    "Test the shift_mne_epoch_trigger() function"
    epochs = datasets.get_mne_sample(sns=True, sub="[1,2,3]")['epochs']
    n_lost_start = np.sum(epochs.times < epochs.tmin + 0.05)
    n_lost_end = np.sum(epochs.times > epochs.tmax - 0.05)
    data = epochs.get_data()

    # don't shift
    epochs_s = shift_mne_epoch_trigger(epochs, [0, 0, 0])
    assert_array_equal(epochs_s.get_data(), data)

    epochs_s = shift_mne_epoch_trigger(epochs, [-0.05, 0., 0.05])
    data_s = epochs_s.get_data()
    assert_array_equal(data_s[0], data[0, :, : -(n_lost_end + n_lost_start)])
    assert_array_equal(data_s[1], data[1, :, n_lost_start: -n_lost_end])
    assert_array_equal(data_s[2], data[2, :, n_lost_end + n_lost_start:])
    assert_allclose(epochs_s.times, epochs.times[n_lost_start: -n_lost_end],
                    rtol=1e-1, atol=1e-3)  # ms accuracy

    epochs_s = shift_mne_epoch_trigger(epochs, [0.05, 0., 0.05])
    data_s = epochs_s.get_data()
    assert_array_equal(data_s[0], data[0, :, n_lost_end:])
    assert_array_equal(data_s[1], data[1, :, :-n_lost_end])
    assert_array_equal(data_s[2], data[2, :, n_lost_end:])
    assert_allclose(epochs_s.times, epochs.times[:-n_lost_end],
                    rtol=1e-1, atol=1e-3)  # ms accuracy


@requires_mne_testing_data
def test_combination_label():
    "Test combination label creation"
    data_dir = mne.datasets.testing.data_path(download=False)
    subjects_dir = os.path.join(data_dir, 'subjects')

    labels = {l.name: l for l in mne.read_labels_from_annot('fsaverage', subjects_dir=subjects_dir)}

    # standard
    l = combination_label('temporal', "superiortemporal + middletemporal + inferiortemporal", labels, subjects_dir)
    lh = labels['superiortemporal-lh'] + labels['middletemporal-lh'] + labels['inferiortemporal-lh']
    lh.name = 'temporal-lh'
    rh = labels['superiortemporal-rh'] + labels['middletemporal-rh'] + labels['inferiortemporal-rh']
    rh.name = 'temporal-rh'
    assert len(l) == 2
    assert_label_equal(l[0], lh)
    assert_label_equal(l[1], rh)

    # only rh
    l = combination_label('temporal-rh', "superiortemporal + middletemporal + inferiortemporal",
                          labels, subjects_dir)
    assert len(l) == 1
    assert l[0].name == 'temporal-rh'
    assert_array_equal(l[0].vertices, rh.vertices)

    # with split_label
    l2 = combination_label('temporal-rh', "superiortemporal + middletemporal +"
                                          "split(inferiortemporal, 2)[0] +"
                                          "split(inferiortemporal, 2)[1]",
                           labels, subjects_dir)
    assert_label_equal(l2[0], l[0], comment=False, color=False)

    # names with .
    labels = {l.name: l for l in
              mne.read_labels_from_annot('fsaverage', 'PALS_B12_Lobes', subjects_dir=subjects_dir)}
    l = combination_label('frontal-lh', "LOBE.FRONTAL", labels, subjects_dir)[0]
    assert_array_equal(l.vertices, labels['LOBE.FRONTAL-lh'].vertices)


@requires_mne_testing_data
def test_morphing():
    # Test data
    data_dir = Path(mne.datasets.testing.data_path(download=False))
    subjects_dir = data_dir / 'subjects'
    stc = datasets.get_mne_stc()
    ndvar = load.mne.stc_ndvar(stc, 'sample', 'ico-5', subjects_dir, 'dSPM', name='src')
    labels_sample = {label.name: label for label in mne.read_labels_from_annot('sample', 'aparc', subjects_dir=subjects_dir)}
    labels_fsa = {label.name: label for label in mne.read_labels_from_annot('fsaverage', 'aparc', subjects_dir=subjects_dir)}
    morph = mne.compute_source_morph(stc, 'sample', 'fsaverage', subjects_dir)
    stc_fsa = morph.apply(stc)
    stc_fsa_ndvar = load.mne.stc_ndvar(stc_fsa, 'fsaverage', 'ico-5', subjects_dir, 'dSPM', False, 'src', parc=None)

    # sample to fsaverage
    # -------------------
    # Using only subject
    ndvar_fsa = morph_source_space(ndvar, 'fsaverage')
    assert_dataobj_equal(ndvar_fsa, stc_fsa_ndvar, decimal=6)
    # Using pre-computed morph-matrix
    ndvar_fsa = morph_source_space(ndvar, 'fsaverage', morph=morph)
    assert_dataobj_equal(ndvar_fsa, stc_fsa_ndvar, decimal=6)
    # Subset of STC using morph-matrix (equivalent to using stc.expand())
    for hemi in ['lh', 'rh', 'bi']:
        if hemi == 'bi':
            label_sample = labels_sample['transversetemporal-lh'] + labels_sample['transversetemporal-rh']
            label_fsa = labels_fsa['transversetemporal-lh'] + labels_fsa['transversetemporal-rh']
        else:
            label_sample = labels_sample[f'transversetemporal-{hemi}']
            label_fsa = labels_fsa[f'transversetemporal-{hemi}']
        stc_label = stc.in_label(label_sample)
        ndvar_label = load.mne.stc_ndvar(stc_label, 'sample', 'ico-5', subjects_dir, 'dSPM', name='src')
        stc_label_fsa = morph.apply(stc_label.copy().expand(stc.vertices)).in_label(label_fsa)
        stc_label_fsa_ndvar = load.mne.stc_ndvar(stc_label_fsa, 'fsaverage', 'ico-5', subjects_dir, 'dSPM', False, 'src', parc=None)
        ndvar_label_fsa = morph_source_space(ndvar_label, 'fsaverage', morph=morph)
        assert_dataobj_equal(ndvar_label_fsa, stc_label_fsa_ndvar, decimal=6)
        # Single time point
        ndvar_label_fsa = morph_source_space(ndvar_label.sub(time=0), 'fsaverage', morph=morph)
        assert_dataobj_equal(ndvar_label_fsa, stc_label_fsa_ndvar.sub(time=0), decimal=6)
        # Subset of STC (computing morph-matrix for subset)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'\d+/\d+ vertices not included in smoothing', module='mne')
            label_morph = mne.compute_source_morph(stc_label, 'sample', 'fsaverage', subjects_dir)
        stc_label_fsa = label_morph.apply(stc_label).in_label(label_fsa)
        stc_label_fsa_ndvar = load.mne.stc_ndvar(stc_label_fsa, 'fsaverage', 'ico-5', subjects_dir, 'dSPM', False, 'src', parc=None)
        ndvar_label_fsa = morph_source_space(ndvar_label, 'fsaverage')
        assert_dataobj_equal(ndvar_label_fsa, stc_label_fsa_ndvar, decimal=6)

    # Scaled brain to fsaverage
    # -------------------------
    y_scaled = datasets.get_mne_stc(True, subject='fsaverage_scaled')
    y_scaled_m = morph_source_space(y_scaled, 'fsaverage')
    assert y_scaled_m.source.subject == 'fsaverage'
    assert_array_equal(y_scaled_m.x, y_scaled.x)

    # masked data
    y_sub = y_scaled.sub(source='superiortemporal-lh')
    y_sub_m = morph_source_space(y_sub, 'fsaverage')
    assert y_sub_m.source.subject == 'fsaverage'
    assert_array_equal(y_sub_m.x, y_sub.x)


@requires_mne_sample_data
def test_xhemi():
    y = datasets.get_mne_stc(ndvar=True)
    data_dir = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(data_dir, 'subjects')
    load.update_subjects_dir(y, subjects_dir)

    lh, rh = xhemi(y, mask=False)
    assert lh.source.rh_n == 0
    assert rh.source.rh_n == 0
    assert lh.max() == pytest.approx(10.80, abs=1e-2)
    assert rh.max() == pytest.approx(7.91, abs=1e-2)

    # Surface source space
    ds = datasets.get_mne_sample(src='ico', hpf=1, fsaverage=True)
    y = ds['src']
    yl, yr = xhemi(y)
    assert yl.source == yr.source
    # FSAverage
    y = ds['srcm']
    yl, yr = xhemi(y)
    assert yl.source == yr.source
    # Subset
    y = ds['srcm'].sub(source=[f'{label}-{hemi}' for label in ('superiortemporal', 'transversetemporal') for hemi in ('lh', 'rh')])
    yl, yr = xhemi(y)
    assert yl.source == yr.source

    # Volume source space
    ds = datasets.get_mne_sample(src='vol', ori='vector', hpf=1)
    y = ds[0, 'src']
    with pytest.raises(NotImplementedError):
        xhemi(y)
    # make symmetric
    coords = list(map(tuple, y.source.coordinates))
    index = [r == 0 or (-r, a, s) in coords for r, a, s in coords]
    y = y.sub(source=np.array(index))
    # test xhemi
    yl, yr = xhemi(y)
    assert yl.source == yr.source
    # test vector mirroring
    r, a, s = coords[10]
    assert r  # make sure it's not on midline
    for i_orig, coords in enumerate(y.source.coordinates):
        if tuple(coords) == (r, a, s):
            break
    for i_flipped, coords in enumerate(yr.source.coordinates):
        if tuple(coords) == (-r, a, s):
            break
    assert_array_equal(yr.x[i_flipped], y.x[i_orig] * [[-1], [1], [1]])


@requires_mne_sample_data  # source space distance computation times out
def test_source_space():
    "Test SourceSpace dimension"
    data_dir = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(data_dir, 'subjects')
    annot_path = os.path.join(subjects_dir, '%s', 'label', '%s.%s.annot')

    for subject in ['fsaverage', 'sample']:
        mne_src = datasets._mne_source_space(subject, 'ico-4', subjects_dir)
        vertices = [mne_src[0]['vertno'], mne_src[1]['vertno']]
        ss = SourceSpace(vertices, subject, 'ico-4', subjects_dir)

        # labels
        for hemi_vertices, hemi in zip(ss.vertices, ('lh', 'rh')):
            labels, _, names = read_annot(annot_path % (subject, hemi, 'aparc'))
            start = 0 if hemi == 'lh' else len(ss.lh_vertices)
            hemi_tag = '-' + hemi
            for i, v in enumerate(hemi_vertices, start):
                label = labels[v]
                if label == -1:
                    assert ss.parc[i] == 'unknown' + hemi_tag
                else:
                    assert ss.parc[i] == names[label].decode() + hemi_tag

        # adjacency
        conn = ss.adjacency()
        mne_conn = mne.spatial_src_adjacency(mne_src)
        assert_array_equal(conn, _matrix_graph(mne_conn))

        # sub-space adjacency
        sssub = ss[ss._array_index('superiortemporal-rh')]
        ss2 = SourceSpace(vertices, subject, 'ico-4', subjects_dir, 'aparc')
        ss2sub = ss2[ss2._array_index('superiortemporal-rh')]
        assert_array_equal(sssub.adjacency(), ss2sub.adjacency())


@requires_mne_sample_data
def test_source_ndvar():
    "Test NDVar with source dimension"
    ds = datasets.get_mne_sample(-0.1, 0.1, src='ico', sub='index<=1')
    v = ds['src', 0]
    assert v.source.parc.name == 'aparc'
    v_2009 = set_parc(v, 'aparc.a2009s')
    assert v_2009.source.parc.name == 'aparc.a2009s'
    conn = v_2009.source.adjacency()
    assert np.sum(v.source.parc == v_2009.source.parc) < len(v.source)
    v_back = set_parc(v_2009, 'aparc')
    assert v_back.source.parc.name == 'aparc'
    assert_array_equal(v.source.parc, v_back.source.parc)
    assert v.x is v_back.x
    assert_array_equal(v_back.source.adjacency(), conn)

    # labels_from_cluster
    v1, v2 = ds['src']
    v1 = v1 * (v1 > 15)
    labels1 = labels_from_clusters(v1)
    assert len(labels1) == 1
    labels1s = labels_from_clusters(v1.sum('time'))
    assert len(labels1s) == 1
    assert_label_equal(labels1s[0], labels1[0])
    v2 = v2 * (v2 > 2)
    labels2 = labels_from_clusters(concatenate((v1, v2), 'case'))
    assert len(labels2) == 2
    assert_label_equal(labels1[0], labels2[0])


@requires_mne_sample_data
def test_vec_source():
    "Test vector source space"
    ds = datasets.get_mne_sample(0, 0.1, (0, 0), src='vol', sub="(modality=='A') & (side == 'L')", ori='vector', stc=True)
    # conversion: vector
    stc = ds[0, 'stc']
    stc2 = load.mne.stc_ndvar([stc, stc], ds.info['subject'], 'vol-10', ds.info['subjects_dir'])
    assert_dataobj_equal(stc2[1], ds[0, 'src'], name=False)
    # add parc
    stc2 = load.mne.stc_ndvar(stc, ds.info['subject'], 'vol-10', ds.info['subjects_dir'], parc='aparc+aseg')
    assert stc2.source.parc[0] == 'Right-Cerebellum-Cortex'
    # non-vector
    if hasattr(stc, 'magnitude'):  # added in mne 0.18
        stc = stc.magnitude()
        ndvar = load.mne.stc_ndvar(stc, ds.info['subject'], 'vol-10', ds.info['subjects_dir'])
        assert_dataobj_equal(ndvar, ds[0, 'src'].norm('space'), name=False)
    # test
    res = testnd.Vector('src', data=ds, samples=2)
    clusters = res.find_clusters()
    assert_array_equal(clusters['n_sources'], [808, 1, 8, 3, 1, 1])
    # NDVar
    v = ds['src']
    assert v.sub(source='lh', time=0).shape == (72, 712, 3)
    # parc
    v = ds[0, 'src']
    v = set_parc(v, Factor('abcdefg', repeat=227))
    v1 = v.sub(source='a')
    assert len(v1.source) == 227
    v2 = v.sub(source=('b', 'c'))
    assert len(v2.source) == 454
    assert 'b' in v2.source.parc
    assert 'd' not in v2.source.parc
    with pytest.raises(IndexError):
        v.sub(source='ab')
    with pytest.raises(IndexError):
        v.sub(source=['a', 'bc'])
