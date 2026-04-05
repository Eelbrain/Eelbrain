# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Test Pipeline using mne-python sample data"""
import json
import logging
from os.path import join, exists
from os import remove
from pathlib import Path
import pytest
import warnings
from warnings import catch_warnings, filterwarnings

import mne
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from eelbrain import *
from eelbrain.pipeline import *
from eelbrain._exceptions import ConfigurationError
from eelbrain._experiment.derivative_cache import ProtectedArtifactError
from eelbrain._experiment.pathing import ica_file_path, log_dir
from eelbrain._experiment.preprocessing import raw_node_name
from eelbrain._experiment.reports import _report_subject_info
from eelbrain._experiment.test_def import TestDims as _TestDims
from eelbrain._experiment.variable_def import LabelVar, Variables
from eelbrain.testing import TempDir, assert_dataobj_equal, requires_mne_sample_data


def _test_result_manifest_path(
        e,
        test: str,
        tstart: float,
        tstop: float,
        pmin,
        *,
        node: str = 'test-result',
        samples: int,
        data: str,
        baseline=True,
        src_baseline=None,
        parc=None,
        mask=None,
        smooth=None,
        samplingrate=None,
) -> Path:
    return e._derivatives.manifest_path(e._derivatives.resolve(node, state=e._derivative_state(), options={
        'data': _TestDims.coerce(data, morph=True),
        'samples': samples,
        'test': test,
        'tstart': tstart,
        'tstop': tstop,
        'pmin': pmin,
        'parc': parc,
        'mask': mask,
        'baseline': baseline,
        'src_baseline': src_baseline,
        'smooth': smooth,
        'samplingrate': samplingrate,
    }).artifact_path)


@requires_mne_sample_data
def test_sample():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=3, n_segments=2, mris=True)

    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)

    assert e.get('raw') == '1-40'
    assert e.get('subject') == 'R0000'
    assert e.get('subject', subject='R0002') == 'R0002'
    tree = e.show_dependencies('evoked', return_str=True)
    assert 'evoked [derivative]' in tree
    assert 'epochs [derivative]' in tree
    wrapped_tree = e.show_dependencies('evoked', max_line_length=60, return_str=True)
    assert all(len(line) <= 60 for line in wrapped_tree.splitlines())

    # wildcard formatting
    with e._temporary_state:
        state = e._derivative_state()
        state['subject'] = '*'
        assert str(ica_file_path(state, raw='*')) == join(root, 'derivatives', 'ica', 'sub-*_meg_raw-*_ica.fif')
        state['subject'] = 'R0002'
        assert str(ica_file_path(state, raw='*')) == join(root, 'derivatives', 'ica', 'sub-R0002_meg_raw-*_ica.fif')

    # events
    e.set('R0001', rej='')
    ds = e.load_selected_events(epoch='target')
    assert ds.n_cases == 39
    ds = e.load_selected_events(epoch='auditory')
    assert ds.n_cases == 20
    ds = e.load_selected_events(epoch='av')
    assert ds.n_cases == 39

    # mrisubject
    assert e.get('mrisubject') == 'sub-R0001'

    # covariance
    with e._temporary_state:
        raw = e.load_raw(raw='1-40')
        assert isinstance(raw, mne.io.BaseRaw)
        assert exists(e._derivatives.resolve(raw_node_name('1-40'), state=e._derivative_state({'raw': '1-40'})).manifest_path)
        e.set(cov='emptyroom', raw='tsss')
        cov = e.load_cov()
        assert isinstance(cov, mne.Covariance)
        assert exists(e._derivatives.resolve('cov:emptyroom', state=e._derivative_state()).manifest_path)
        assert e.load_bad_channels(noise=True) == []
        e.set(cov='emptyroom', raw='1-40')
        cov = e.load_cov()
        assert isinstance(cov, mne.Covariance)
        assert exists(e._derivatives.resolve('cov:emptyroom', state=e._derivative_state()).manifest_path)
        assert e.load_bad_channels(noise=True) == []
        e.load_cov()

    # evoked cache invalidated by change in bads
    e.set('R0001', rej='', epoch='target')
    e.load_events()
    assert exists(e._derivatives.resolve('events', state=e._derivative_state()).manifest_path)
    ds = e.load_evoked(ndvar=False)
    assert exists(e._derivatives.resolve('evoked', state=e._derivative_state()).manifest_path)
    assert ds[0, 'evoked'].info['bads'] == []
    e.make_bad_channels(['MEG 0331'])
    ds = e.load_evoked(ndvar=False)
    assert ds[0, 'evoked'].info['bads'] == ['MEG 0331']

    e.set(rej='man', model='modality')
    test_tree = e.show_dependencies(
        'test-result',
        options={
            'data': _TestDims.coerce('sensor.rms', morph=True),
            'samples': 100,
            'test': 'a>v',
            'tstart': 0.05,
            'tstop': 0.2,
            'pmin': 0.05,
            'parc': None,
            'mask': None,
            'baseline': False,
            'src_baseline': None,
            'smooth': None,
            'samplingrate': None,
        },
        return_str=True,
    )
    assert 'evoked-test-data [derivative]' in test_tree
    assert 'evoked-group-dataset [derivative]' in test_tree
    movie_tree = e.show_dependencies(
        'movie',
        options={
            'data': _TestDims.coerce('source', morph=True),
            'single_subject': False,
            'movie_kind': 'ttest',
            'subject': None,
            'baseline': False,
            'src_baseline': None,
            'cat': None,
            'p': 0.05,
            'pmin': 0.001,
            'pmid': 0.01,
            'surf': 'inflated',
            'time_dilation': 4.0,
            'cluster_state': {},
        },
        return_str=True,
    )
    assert 'evoked-stc-group-dataset [derivative]' in movie_tree
    sds = []
    for _ in e:
        e.make_epoch_selection(auto=2.5e-12)
        sds.append(e.load_evoked())
    ds_ind = combine(sds, dim_intersection=True)

    ds = e.load_evoked('all')
    ds['meg'] = ds['meg'].sub(sensor=ds['meg'].sensor.index(exclude='MEG 0331'))  # load_evoked interpolates bad channel
    assert_dataobj_equal(ds_ind, ds, decimal=19)  # make vs load evoked

    # sensor space tests
    megs = [e.load_evoked(cat='auditory')['meg'] for _ in e]
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms', baseline=False, make=True)
    test_manifest = _test_result_manifest_path(e, 'a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms', baseline=False)
    assert exists(test_manifest)
    with open(test_manifest) as fid:
        test_manifest_data = json.load(fid)
    assert test_manifest_data['fingerprint']['definitions']['test']['tail'] == 1
    assert test_manifest_data['fingerprint']['definitions']['epoch']['tmax'] == 0.3
    assert 'dependencies' not in test_manifest_data['fingerprint']
    assert 'evoked-test-data' in test_manifest_data['dependencies']
    assert 'evoked-group-dataset' in test_manifest_data['dependencies']['evoked-test-data']['dependencies']
    remove(test_manifest)
    with pytest.raises(IOError):
        e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms', baseline=False)
    _ = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms', baseline=False, make=True)
    assert exists(test_manifest)

    class ChangedTestExperiment(SampleExperiment):
        tests = {
            **SampleExperiment.tests,
            'a>v': TTestRelated('modality', 'auditory', 'visual'),
        }

    with pytest.raises(IOError):
        ChangedTestExperiment(root).load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms', baseline=False)

    class ChangedEpochExperiment(SampleExperiment):
        epochs = {
            **SampleExperiment.epochs,
            'target': PrimaryEpoch('sample', "event == 'target'", tmax=0.2, decim=5),
        }

    with pytest.raises(IOError):
        ChangedEpochExperiment(root).load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms', baseline=False)

    meg_rms = combine(meg.rms('sensor') for meg in megs).mean('case', name='auditory')
    assert_dataobj_equal(res.c1_mean, meg_rms, decimal=21)
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.mean', baseline=False, make=True)
    meg_mean = combine(meg.mean('sensor') for meg in megs).mean('case', name='auditory')
    assert_dataobj_equal(res.c1_mean, meg_mean, decimal=21)
    with pytest.raises(IOError):
        e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False)
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False, make=True)
    assert res.p.min() == pytest.approx(.143, abs=.001)
    assert res.difference.max() == pytest.approx(4.47e-13, 1e-15)
    # plot (skip to avoid using framework build)
    # e.plot_evoked(1, epoch='target', model='')

    # e._report_subject_info() broke with non-alphabetic subject order
    subjects = e.get_field_values('subject')
    ds = Dataset()
    ds['subject'] = Factor(reversed(subjects))
    ds['n'] = Var(range(3))
    _ = _report_subject_info(e._derivatives._get_node('source-report'), e._derivative_state(), ds, '')

    # post_baseline_trigger_shift
    # use multiple of tstep to shift by even number of samples
    tstep = 0.008324800548266162
    shift = -7 * tstep

    class Experiment(SampleExperiment):
        epochs = {
            **SampleExperiment.epochs,
            'visual-s': SecondaryEpoch('target', "modality == 'visual'", post_baseline_trigger_shift='shift', post_baseline_trigger_shift_max=0, post_baseline_trigger_shift_min=shift),
        }
        variables = {
            **SampleExperiment.variables,
            'shift': LabelVar('side', {'left': 0, 'right': shift}),
            'shift_t': LabelVar('trigger', {(1, 3): 0, (2, 4): shift})
        }
    e = Experiment(root)
    # test shift in events
    ds = e.load_events()
    assert_dataobj_equal(ds['shift_t'], ds['shift'], name=False)
    # compare against epochs (baseline correction on epoch level rather than evoked for smaller numerical error)
    ep = e.load_epochs(baseline=True, epoch='visual', rej='').aggregate('side')
    evs = e.load_evoked(baseline=True, epoch='visual-s', rej='', model='side')
    tstart = ep['meg'].time.tmin - shift
    assert_dataobj_equal(evs[0, 'meg'], ep[0, 'meg'].sub(time=(tstart, None)), decimal=19)
    tstop = ep['meg'].time.tstop + shift
    assert_almost_equal(evs[1, 'meg'].x, ep[1, 'meg'].sub(time=(None, tstop)).x, decimal=19)

    # post_baseline_trigger_shift
    class Experiment(SampleExperiment):
        epochs = {
            **SampleExperiment.epochs,
            'av_shift': SuperEpoch(
                ('visual', 'auditory'),
                post_baseline_trigger_shift="Var.from_dict(modality, {'visual': 0.0, 'auditory': 0.1})",
                post_baseline_trigger_shift_max=0.1,
                post_baseline_trigger_shift_min=0.0,
            ),
        }
        groups = {
            'group0': Group(['R0000']),
            'group1': SubGroup('all', ['R0000']),
        }
        variables = {
            'group': GroupVar(['group0', 'group1']),
            **SampleExperiment.variables,
        }
    e = Experiment(root)
    events = e.load_selected_events(epoch='av_shift')
    ds = e.load_epochs(baseline=True, epoch='av_shift')
    v = ds.sub("epoch=='visual'", 'meg')
    v_target = e.load_epochs(baseline=True, epoch='visual')['meg'].sub(time=(-0.1, v.time.tstop))
    assert_almost_equal(v.x, v_target.x)
    a = ds.sub("epoch=='auditory'", 'meg').sub(time=(-0.1, 0.099))
    a_target = e.load_epochs(baseline=True, epoch='auditory')['meg'].sub(time=(0, 0.199))
    assert_almost_equal(a.x, a_target.x, decimal=20)

    # duplicate subject
    class BadExperiment(SampleExperiment):
        groups = {'group': ('R0001', 'R0002', 'R0002')}
    with pytest.raises(ConfigurationError):
        BadExperiment(root)

    # non-existing subject
    class BadExperiment(SampleExperiment):
        groups = {'group': ('R0001', 'R0003', 'R0002')}
    with pytest.raises(ConfigurationError):
        BadExperiment(root)

    # unsorted subjects
    class Experiment(SampleExperiment):
        groups = {'group': ('R0002', 'R0000', 'R0001')}
    e = Experiment(root)
    assert [s for s in e] == ['R0000', 'R0001', 'R0002']

    # changes
    class Changed(SampleExperiment):
        variables = {
            'event': {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'},
            'side': {(1, 3): 'left', (2, 4): 'right_changed'},
            'modality': {(1, 2): 'auditory', (3, 4): 'visual'}
        }
        tests = {
            'twostage': TwoStageTest(
                'side_left + modality_a',
                {'side_left': "side == 'left'",
                 'modality_a': "modality == 'auditory'"}),
            'novars': TwoStageTest('side + modality'),
        }
    e = Changed(root)

    # changed variable, while a test with model=None is not changed
    class Changed(Changed):
        variables = {
            'side': {(1, 3): 'left', (2, 4): 'right_changed'},
            'modality': {(1, 2): 'auditory', (3, 4): 'visual_changed'}
        }
    e = Changed(root)

    # changed variable, unchanged test with vardef=None
    class Changed(Changed):
        variables = {
            'side': {(1, 3): 'left', (2, 4): 'right_changed'},
            'modality': {(1, 2): 'auditory', (3, 4): 'visual_changed'}
        }
    e = Changed(root)

    # ICA
    # ---
    class Experiment(SampleExperiment):
        raw = {
            **SampleExperiment.raw,
            'ica': RawICA('1-40', 'sample', method='fastica', n_components=0.95),
            'apply-ica': RawApplyICA('1-40', 'ica'),
        }
    e = Experiment(root)
    ica_path = e.make_ica(raw='ica')
    ica_manifest = e._derivatives.manifest_path(ica_path)
    assert exists(ica_manifest)

    class ChangedExperiment(Experiment):
        raw = {
            **Experiment.raw,
            '1-40': RawFilter('tsss', 1, 41),
            'ica': RawICA('1-40', 'sample', method='fastica', n_components=0.95),
        }
    e_changed = ChangedExperiment(root)
    with pytest.raises(ProtectedArtifactError, match='estimated using different settings for raw step') as error:
        e_changed.load_raw(raw='ica1-40')
    assert "'1-40'" in str(error.value)
    assert 'h_freq' in str(error.value)
    assert '40' in str(error.value)
    assert '41' in str(error.value)
    assert 'revert the raw pipeline change' in str(error.value)
    assert 'accept_stale=True' in str(error.value)
    assert 'cache directory' not in str(error.value)
    manifest_data = json.loads(Path(ica_manifest).read_text())
    manifest_data['derivative_version'] += 1
    Path(ica_manifest).write_text(json.dumps(manifest_data))
    with pytest.raises(ProtectedArtifactError, match='accept_stale=True'):
        e.load_raw(raw='ica1-40')
    with pytest.raises(ProtectedArtifactError, match='choose .*incorporate'):
        e.load_ica(raw='ica')
    ica = e.load_ica(raw='ica', accept_stale=True)
    assert isinstance(ica, mne.preprocessing.ICA)
    assert isinstance(e.load_ica(raw='ica'), mne.preprocessing.ICA)
    e.set(raw='ica1-40', model='')
    e.make_epoch_selection(auto=2e-12, overwrite=True)
    ds1 = e.load_evoked(raw='ica1-40')
    ica = e.load_ica(raw='ica')
    ica.exclude = [0, 1, 2]
    ica.save(ica_path, overwrite=True)
    ds2 = e.load_evoked(raw='ica1-40')
    assert not np.allclose(ds1['meg'].x, ds2['meg'].x, atol=1e-20), "ICA change ignored"
    # apply-ICA
    with catch_warnings():
        filterwarnings('ignore', "The measurement information indicates a low-pass frequency", RuntimeWarning)
        ds1 = e.load_evoked(raw='ica', rej='')
        ds2 = e.load_evoked(raw='apply-ica', rej='')
    assert_dataobj_equal(ds2, ds1)
    # Source-space forward/inverse coverage lives in test_sample_source(), so
    # this fast test stays comparable to main.

    # rename subject
    # --------------
    # e.set(subject='R0001')
    # src = Path(e._bids_path.directory)
    # dst = Path(str(src).replace('R0001', 'R0003'))
    # shutil.move(src, dst)
    # for path in dst.glob('*.fif'):
    #     shutil.move(path, dst / path.parent / path.name.replace('R0001', 'R0003'))
    # check subject list
    # e = SampleExperiment(root)
    # assert list(e) == ['R0000', 'R0002', 'R0003']
    # check that cached test got deleted
    # assert e.get('raw') == '1-40'
    # with pytest.raises(IOError):
    #     e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False)
    # res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False, make=True)
    # assert res.df == 2
    # assert res.p.min() == pytest.approx(.143, abs=.001)
    # assert res.difference.max() == pytest.approx(4.47e-13, 1e-15)

    # remove subject
    # --------------
    # shutil.rmtree(dst)
    # # check cache
    # e = SampleExperiment(root)
    # assert list(e) == ['R0000', 'R0002']
    # # check that cached test got deleted
    # assert e.get('raw') == '1-40'
    # with pytest.raises(IOError):
    #     e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False)

    # label_events
    # ------------
    class Experiment(SampleExperiment):
        def label_events(self, ds):
            ds = SampleExperiment.label_events(self, ds)
            ds = ds.sub("event == 'smiley'")
            ds['new_var'] = Var([i + 1 for i in ds['i_start']])
            return ds

    e = Experiment(root)
    events = e.load_events()
    assert_array_equal(events['new_var'], [67402, 75306])

    # Parc
    # ----
    labels = e.load_annot(parc='ac', mrisubject='fsaverage')
    assert len(labels) == 4
    annot_handle = e._derivatives.resolve('annot', state=e._derivative_state({'mrisubject': 'fsaverage', 'parc': 'ac'}))
    assert exists(annot_handle.manifest_path)
    # change parc definition

    class Experiment(SampleExperiment):
        parcs = {
            'ac': SubParc('aparc', ('transversetemporal', 'superiortemporal')),
        }
    e = Experiment(root)
    labels = e.load_annot(parc='ac', mrisubject='fsaverage')
    assert len(labels) == 6


@requires_mne_sample_data
@pytest.mark.slow
def test_sample_source():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=3, n_segments=2, mris=True)  # TODO: use sample MRI which already has forward solution
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)

    # source space tests
    e.set(src='ico-4', rej='', epoch='auditory')
    morph = e.load_source_morph(subject='R0000')
    assert isinstance(morph, mne.SourceMorph)
    assert exists(e._derivatives.resolve('source-morph', state=e._derivative_state({'subject': 'R0000'})).manifest_path)
    # These two tests are only identical if the evoked has been cached before the first test is loaded
    resp = e.load_test('left=right', 0.05, 0.2, 0.05, samples=100, parc='ac', make=True)
    resm = e.load_test('left=right', 0.05, 0.2, 0.05, samples=100, mask='ac', make=True)
    assert exists(e._derivatives.resolve('src', state=e._derivative_state()).manifest_path)
    assert exists(e._derivatives.resolve('fwd', state=e._derivative_state()).manifest_path)
    assert exists(e._derivatives.resolve('inv', state=e._derivative_state()).manifest_path)
    with open(_test_result_manifest_path(e, 'left=right', 0.05, 0.2, 0.05, samples=100, data='source', parc='ac')) as fid:
        source_manifest_data = json.load(fid)
    assert source_manifest_data['fingerprint']['definitions']['parc']['base'] == 'aparc'
    assert 'dependencies' not in source_manifest_data['fingerprint']
    assert 'evoked-test-data' in source_manifest_data['dependencies']
    assert 'evoked-stc-group-dataset' in source_manifest_data['dependencies']['evoked-test-data']['dependencies']
    assert set(source_manifest_data['dependencies']['evoked-test-data']['dependencies']['evoked-stc-group-dataset']['dependencies']) == {'R0000', 'R0001', 'R0002'}
    assert_dataobj_equal(resp.t, resm.t)
    # ROI tests
    e.set(epoch='target')
    ress = e.load_test('left=right', 0.05, 0.2, 0.05, samples=100, data='source.rms', parc='ac', make=True)
    with open(_test_result_manifest_path(e, 'left=right', 0.05, 0.2, 0.05, samples=100, data='source.rms', parc='ac')) as fid:
        roi_manifest_data = json.load(fid)
    assert 'evoked-test-data' in roi_manifest_data['dependencies']
    roi_deps = roi_manifest_data['dependencies']['evoked-test-data']['dependencies']
    assert set(roi_deps) == {'R0000', 'R0001', 'R0002'}
    assert all(roi_deps[subject]['name'] == 'evoked-stc' for subject in roi_deps)
    res = ress.res['transversetemporal-lh']
    assert res.p.min() == 1 / 7
    ress = e.load_test('twostage', 0.05, 0.2, 0.05, samples=100, data='source.rms', parc='ac', make=True)
    with open(_test_result_manifest_path(e, 'twostage', 0.05, 0.2, 0.05, node='two-stage-level-2', samples=100, data='source.rms', parc='ac')) as fid:
        two_stage_manifest_data = json.load(fid)
    assert 'two-stage-level-1' in {dep['name'] for dep in two_stage_manifest_data['dependencies'].values()}
    subject_dep = two_stage_manifest_data['dependencies']['R0000']
    with open(Path(subject_dep['manifest'])) as fid:
        level_1_manifest_data = json.load(fid)
    assert level_1_manifest_data['dependencies']['two-stage-data']['dependencies']['R0000']['name'] == 'epochs-stc'
    ds_return, _ = e.load_test('twostage', 0.05, 0.2, 0.05, samples=100, data='source', return_data=True, make=True)
    assert isinstance(ds_return, Dataset)
    assert 'subject' in ds_return
    res = ress.res['transversetemporal-lh']
    assert res.samples == -1
    assert res.tests['intercept'].p.min() == 1 / 7

    class ChangedParcExperiment(SampleExperiment):
        parcs = {
            **SampleExperiment.parcs,
            'ac': SubParc('aparc', ('superiortemporal',)),
        }

    with pytest.raises(IOError):
        ChangedParcExperiment(root).load_test('left=right', 0.05, 0.2, 0.05, samples=100, data='source.rms', parc='ac')


@requires_mne_sample_data
def test_sample_tasks():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 2, 1)

    class Experiment(SampleExperiment):

        raw = {
            'ica': RawICA('raw', ('sample1', 'sample2'), 'fastica', max_iter=1),
            'av-ref': RawReReference('raw'),
            **SampleExperiment.raw,
        }

    root = join(tempdir, 'SampleExperiment')
    e = Experiment(root)

    # get paths
    pipe = e._raw[e.get('raw', raw='raw')]
    bids_path = e._bids_path
    assert pipe._raw_path(bids_path) == Path(root) / 'sub-R0000' / 'meg' / 'sub-R0000_task-sample1_meg.fif'
    assert pipe._bads_path(bids_path) == Path(root) / 'sub-R0000' / 'meg' / 'sub-R0000_task-sample1_channels.tsv'
    pipe = e._raw[e.get('raw', raw='ica')]
    bids_path = e._bids_path
    state = e._derivative_state(raw='ica')
    handle = e._derivatives.resolve(raw_node_name('ica'), state=state)
    assert handle.artifact_path.is_relative_to(Path(root) / 'derivatives' / 'eelbrain' / 'cache' / 'raw-ica')
    assert handle.artifact_path.suffix == '.fif'
    assert '_key-' in handle.artifact_path.name
    assert str(ica_file_path(state, raw='ica')) == join(root, 'derivatives', 'ica', 'sub-R0000_meg_raw-ica_ica.fif')
    e.set(raw='raw')

    # automatically generate channels.tsv
    bad_path = join(root, 'sub-R0000', 'meg', 'sub-R0000_task-sample1_channels.tsv')
    remove(bad_path)
    assert not exists(bad_path)
    e.make_bad_channels('MEG 0111')
    assert exists(bad_path)
    assert e.load_bad_channels() == ['MEG 0111']
    # add another bad channel
    e.make_bad_channels('MEG 0121')
    assert e.load_bad_channels() == ['MEG 0111', 'MEG 0121']
    # redo bad channels
    e.make_bad_channels([], redo=True)
    assert e.load_bad_channels() == []
    e.make_bad_channels('MEG 0111', redo=True)
    assert e.load_bad_channels() == ['MEG 0111']

    # merge bad channels for ICA
    assert e.load_bad_channels(task='sample2') == []
    e.make_bad_channels('MEG 0121')
    assert e.load_bad_channels(raw='ica') == ['MEG 0111', 'MEG 0121']
    e.set(raw='raw')
    # merge_bad_channels
    e.merge_bad_channels()
    assert e.load_bad_channels(task='sample2') == ['MEG 0111', 'MEG 0121']
    e.show_bad_channels()

    # rejection
    for _ in e:
        for epoch in ('target1', 'target2'):
            e.set(epoch=epoch)
            e.make_epoch_selection(auto=2e-12)

    ds = e.load_evoked('R0000', epoch='target2')
    e.set(task='sample1')
    ds2 = e.load_evoked('R0000')
    assert_dataobj_equal(ds2, ds, decimal=19)

    # super-epoch
    ds1 = e.load_epochs(epoch='target1')
    ds2 = e.load_epochs(epoch='target2')
    ds_super = e.load_epochs(epoch='super')
    assert_dataobj_equal(ds_super['meg'], combine((ds1['meg'], ds2['meg'])))
    # evoked
    dse_super = e.load_evoked(epoch='super', model='modality%side')
    target = ds_super.aggregate('modality%side', drop=('i_start', 't_edf', 'time', 'index', 'trigger', 'task', 'interpolate_channels', 'epoch'))
    assert_dataobj_equal(dse_super, target, 19)

    # conflicting task and epoch settings
    rej_path = join(root, 'derivatives', 'eelbrain', 'epoch selection', 'sub-R0000_meg_raw-1-40_epoch-target2_rej-man_epoch.pickle')
    e.set(epoch='target2', raw='1-40')
    assert not exists(rej_path)
    e.set(task='sample1')
    e.make_epoch_selection(auto=2e-12)
    assert exists(rej_path)

    # ica
    e.set('R0000', raw='ica')
    with catch_warnings():
        filterwarnings('ignore', "FastICA did not converge", UserWarning)
        assert e.make_ica() == join(root, 'derivatives', 'ica', 'sub-R0000_meg_raw-ica_ica.fif')


@requires_mne_sample_data
def test_evoked_backed_test_vars_are_post_aggregation_only():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    class Experiment(SampleExperiment):
        tests = {
            **SampleExperiment.tests,
            'anova-ok': ANOVA('modality * modality_num * subject', model='modality', vars={'modality_num': ('modality', {'auditory': 0, 'visual': 1})}),
            'anova-bad': ANOVA('modality_num * subject', vars={'modality_num': ('modality', {'auditory': 0, 'visual': 1})}),
        }

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=3, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')
    e = Experiment(root, rej='')

    options = {
        'data': _TestDims.coerce('sensor.mean', morph=True),
        'test': 'anova-ok',
        'baseline': False,
        'src_baseline': None,
        'parc': None,
        'mask': None,
        'smooth': None,
        'samplingrate': None,
    }
    ds = e._derivatives.load('evoked-test-data', state=e._derivative_state({'test': 'anova-ok'}), options=options)
    assert 'modality_num' in ds

    with pytest.raises(ConfigurationError, match='post-aggregation dataset'):
        e._derivatives.load('evoked-test-data', state=e._derivative_state({'test': 'anova-bad'}), options={**options, 'test': 'anova-bad'})


@requires_mne_sample_data
def test_raw_reader_warnings_are_summarized(monkeypatch):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=1, mris=False)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)

    original = mne.io.read_raw_fif

    def read_raw_fif(*args, **kwargs):
        warnings.warn("Synthetic raw reader warning 1", RuntimeWarning)
        warnings.warn("Synthetic raw reader warning 2", RuntimeWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(mne.io, 'read_raw_fif', read_raw_fif)

    with catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        e.load_raw(raw='raw')
        e.load_raw(raw='raw')
    assert not any('issued while reading raw data files' in str(w.message) for w in record)

    details_path = Path(log_dir({'root': root})) / 'raw-reader-warnings.log'
    assert details_path.exists()
    text = details_path.read_text()
    assert 'Synthetic raw reader warning 1' in text
    assert 'Synthetic raw reader warning 2' in text

    log_path = Path(next(handler.baseFilename for handler in e._log.handlers if isinstance(handler, logging.FileHandler)))
    log_text = log_path.read_text()
    assert str(details_path) in log_text
    assert log_text.count('issued while reading raw data files') == 1


@requires_mne_sample_data
def test_evoked_cache_reuse():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 2, 1)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', rej='')

    _ = e.load_evoked(ndvar=False)
    handle = e._derivatives.resolve('evoked', state=e._derivative_state())
    evoked_path = handle.artifact_path
    manifest_path = handle.manifest_path
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest['dependencies']['epochs']['view'] == 'evoked'
    mtimes_1 = (evoked_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)

    _ = e.load_evoked(ndvar=False)
    mtimes_2 = (evoked_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)

    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_evoked_cached_load_bypasses_epochs(monkeypatch):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 2, 1)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', rej='')

    target = e.load_evoked(ndvar=False)
    epochs_node = e._derivatives._get_node('epochs')

    def fail(*args, **kwargs):
        raise AssertionError("Evoked dataset load should not rebuild epochs on an evoked cache hit")

    monkeypatch.setattr(epochs_node, 'load', fail)
    monkeypatch.setattr(epochs_node, 'build', fail)

    calls = 0
    original_read_evokeds = mne.read_evokeds

    def read_evokeds(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_read_evokeds(*args, **kwargs)

    monkeypatch.setattr(mne, 'read_evokeds', read_evokeds)

    ds = e.load_evoked(ndvar=False)
    assert_dataobj_equal(ds, target, decimal=19)
    assert calls == 1


@requires_mne_sample_data
def test_evoked_cached_load_applies_cat_without_rebuilding_epochs(monkeypatch):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 2, 1)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', rej='', model='modality')

    target = e.load_evoked(ndvar=False, cat='auditory')
    epochs_node = e._derivatives._get_node('epochs')

    def fail(*args, **kwargs):
        raise AssertionError("Evoked dataset load should not rebuild epochs on an evoked cache hit")

    monkeypatch.setattr(epochs_node, 'load', fail)
    monkeypatch.setattr(epochs_node, 'build', fail)

    calls = 0
    original_read_evokeds = mne.read_evokeds

    def read_evokeds(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_read_evokeds(*args, **kwargs)

    monkeypatch.setattr(mne, 'read_evokeds', read_evokeds)

    ds = e.load_evoked(ndvar=False, cat='auditory')
    assert ds.n_cases == 1
    assert_dataobj_equal(ds, target, decimal=19)
    assert calls == 1


@requires_mne_sample_data
def test_evoked_cache_ignores_irrelevant_selected_events_changes(monkeypatch):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 1, 2, 1)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', rej='', model='modality')

    _ = e.load_evoked(ndvar=False)
    handle = e._derivatives.resolve('evoked', state=e._derivative_state())
    mtimes_1 = (handle.artifact_path.stat().st_mtime_ns, handle.manifest_path.stat().st_mtime_ns)

    node = e._derivatives._get_node('selected-events')
    original_build = node.build
    original_fingerprint = node.fingerprint

    def build(ctx):
        ds = original_build(ctx)
        ds = ds.copy()
        ds['marker'] = Var(np.arange(ds.n_cases), 'marker')
        return ds

    def fingerprint(ctx):
        out = original_fingerprint(ctx)
        out['irrelevant-change'] = True
        return out

    monkeypatch.setattr(node, 'build', build)
    monkeypatch.setattr(node, 'fingerprint', fingerprint)

    assert handle.is_valid()
    _ = e.load_evoked(ndvar=False)
    mtimes_2 = (handle.artifact_path.stat().st_mtime_ns, handle.manifest_path.stat().st_mtime_ns)

    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_evoked_cache_stales_on_model_change():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', rej='', model='modality')
    _ = e.load_evoked(ndvar=False)

    class ChangedExperiment(SampleExperiment):
        variables = {
            **SampleExperiment.variables,
            'modality': {(1, 2): 'auditory_changed', (3, 4): 'visual'},
        }

    e_changed = ChangedExperiment(root)
    e_changed.set(subject='R0000', epoch='target', rej='', model='modality')
    handle = e_changed._derivatives.resolve('evoked', state=e_changed._derivative_state())

    assert not handle.is_valid()
    ds = e_changed.load_evoked(ndvar=False)
    assert set(ds['modality'].cells) == {'auditory_changed', 'visual'}


@requires_mne_sample_data
def test_epochs_dependency_views_distinguish_model_sensitivity():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', rej='', model='modality')
    evoked_handle = e._derivatives.resolve('evoked', state=e._derivative_state(), options={})
    epochs_handle = e._derivatives.resolve('epochs', state=evoked_handle.state, options=evoked_handle.node._epochs_options(evoked_handle))

    dataset_fingerprint = epochs_handle.describe_dependency()['fingerprint']
    evoked_fingerprint = epochs_handle.describe_dependency(view='evoked')['fingerprint']

    assert 'model_signature' not in dataset_fingerprint
    assert 'model_signature' in evoked_fingerprint

    class ChangedExperiment(SampleExperiment):
        variables = {
            **SampleExperiment.variables,
            'modality': {(1, 2): 'auditory_changed', (3, 4): 'visual'},
        }

    e_changed = ChangedExperiment(root)
    e_changed.set(subject='R0000', epoch='target', rej='', model='modality')
    evoked_handle_changed = e_changed._derivatives.resolve('evoked', state=e_changed._derivative_state(), options={})
    epochs_handle_changed = e_changed._derivatives.resolve('epochs', state=evoked_handle_changed.state, options=evoked_handle_changed.node._epochs_options(evoked_handle_changed))

    assert epochs_handle_changed.describe_dependency()['fingerprint'] == dataset_fingerprint
    assert epochs_handle_changed.describe_dependency(view='evoked')['fingerprint'] != evoked_fingerprint


@requires_mne_sample_data
def test_epochs_cache_uses_fif():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 1, 2, 1)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', rej='')

    options = {
        'baseline': False,
        'reject': True,
        'cat': None,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'trigger_shift': True,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'ndvar': False,
        'data': 'sensor',
        'data_raw': False,
        'add_bads': True,
    }
    handle = e._derivatives.resolve('epochs', state=e._derivative_state(), options=options)
    ds = handle.load(cache=True)
    epochs = handle.node.load(handle, handle.artifact_path)

    assert isinstance(ds['epochs'], mne.BaseEpochs)
    assert isinstance(epochs, mne.BaseEpochs)
    assert handle.artifact_path.is_dir()
    assert list(handle.artifact_path.glob('*-epo.fif'))
    manifest = json.loads(handle.manifest_path.read_text())
    assert manifest['artifact_metadata']['kind'] == 'single'
    assert manifest['artifact_metadata']['file'] == 'epochs-0000-epo.fif'
    assert manifest['dependencies']['selected-events']['view'] == 'epochs'
    assert tuple(manifest['dependencies']['selected-events']['fingerprint']) == ('i_start',)

    mtimes_1 = tuple(path.stat().st_mtime_ns for path in sorted(handle.artifact_path.iterdir()))
    ds_cached = handle.load(cache=True)
    mtimes_2 = tuple(path.stat().st_mtime_ns for path in sorted(handle.artifact_path.iterdir()))

    assert isinstance(ds_cached['epochs'], mne.BaseEpochs)
    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_epochs_cached_load_uses_current_selected_events(monkeypatch):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 1, 2, 1)
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', rej='')

    options = {
        'baseline': False,
        'reject': True,
        'cat': None,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'trigger_shift': True,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'ndvar': False,
        'data': 'sensor',
        'data_raw': False,
        'add_bads': True,
    }
    handle = e._derivatives.resolve('epochs', state=e._derivative_state(), options=options)
    ds = handle.load(cache=True)
    assert isinstance(ds['epochs'], mne.BaseEpochs)

    mtimes_1 = tuple(path.stat().st_mtime_ns for path in sorted(handle.artifact_path.iterdir()))
    node = e._derivatives._get_node('selected-events')
    original_build = node.build
    original_fingerprint = node.fingerprint

    def build(ctx):
        ds = original_build(ctx)
        ds = ds.copy()
        ds['marker'] = Var(np.arange(ds.n_cases), 'marker')
        return ds

    def fingerprint(ctx):
        out = original_fingerprint(ctx)
        out['irrelevant-change'] = True
        return out

    monkeypatch.setattr(node, 'build', build)
    monkeypatch.setattr(node, 'fingerprint', fingerprint)

    assert handle.is_valid()
    ds_cached = e.load_epochs(ndvar=False)
    mtimes_2 = tuple(path.stat().st_mtime_ns for path in sorted(handle.artifact_path.iterdir()))

    assert isinstance(ds_cached['epochs'], mne.BaseEpochs)
    assert 'marker' in ds_cached
    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_selected_events_manifest_uses_real_dependencies():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', rej='')
    handle = e._derivatives.resolve('selected-events', state=e._derivative_state(), options={
        'reject': True,
        'add_bads': True,
        'index': True,
        'data_raw': False,
        'cat': None,
    })
    _ = handle.load(cache=True)
    manifest = json.loads(handle.manifest_path.read_text())

    assert 'dependencies' not in manifest['fingerprint']
    assert 'rej' in manifest['dependencies']
    assert any(key.endswith(':events') for key in manifest['dependencies'])
    assert not any(key.endswith(':raw') for key in manifest['dependencies'])


@requires_mne_sample_data
def test_raw_cache_identity_ignores_view_options():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000')
    node_name = raw_node_name('1-40')
    base_state = e._derivative_state()

    handle_default = e._derivatives.resolve(node_name, state=base_state, options={'noise': False, 'preload': False, 'add_bads': True})
    handle_view = e._derivatives.resolve(node_name, state=base_state, options={'noise': False, 'preload': True, 'add_bads': False})
    handle_noise = e._derivatives.resolve(node_name, state=base_state, options={'noise': True, 'preload': False, 'add_bads': True})

    assert handle_default.current_fingerprint() == handle_view.current_fingerprint()
    assert handle_default.current_fingerprint() != handle_noise.current_fingerprint()


@requires_mne_sample_data
def test_selected_events_cache_identity_ignores_view_options():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', rej='', model='modality')
    base_state = e._derivative_state()

    handle_default = e._derivatives.resolve('selected-events', state=base_state, options={
        'reject': True,
        'add_bads': True,
        'index': True,
        'data_raw': False,
        'cat': None,
    })
    handle_view = e._derivatives.resolve('selected-events', state=base_state, options={
        'reject': True,
        'add_bads': False,
        'index': 'trial',
        'data_raw': True,
        'cat': ('auditory',),
    })
    handle_reject = e._derivatives.resolve('selected-events', state=base_state, options={
        'reject': False,
        'add_bads': True,
        'index': True,
        'data_raw': False,
        'cat': None,
    })

    assert handle_default.current_fingerprint() == handle_view.current_fingerprint()
    assert handle_default.current_fingerprint() != handle_reject.current_fingerprint()


@requires_mne_sample_data
def test_source_cache_identity_ignores_view_options():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=True)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', rej='', src='ico-4')
    base_state = e._derivative_state()

    epochs_stc_default = e._derivatives.resolve('epochs-stc', state=base_state, options={
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'reject': True,
        'ndvar': True,
        'data_raw': False,
        'keep_epochs': False,
    })
    epochs_stc_view = e._derivatives.resolve('epochs-stc', state=base_state, options={
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'reject': True,
        'ndvar': False,
        'data_raw': True,
        'keep_epochs': 'both',
    })
    epochs_stc_artifact = e._derivatives.resolve('epochs-stc', state=base_state, options={
        'baseline': (-0.1, 0),
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'reject': True,
        'ndvar': True,
        'data_raw': False,
        'keep_epochs': False,
    })

    assert epochs_stc_default.current_fingerprint() == epochs_stc_view.current_fingerprint()
    assert epochs_stc_default.current_fingerprint() != epochs_stc_artifact.current_fingerprint()

    evoked_stc_default = e._derivatives.resolve('evoked-stc', state=base_state, options={
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'ndvar': True,
        'data_raw': False,
        'keep_evoked': False,
    })
    evoked_stc_view = e._derivatives.resolve('evoked-stc', state=base_state, options={
        'baseline': False,
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'ndvar': False,
        'data_raw': True,
        'keep_evoked': True,
    })
    evoked_stc_artifact = e._derivatives.resolve('evoked-stc', state=base_state, options={
        'baseline': (-0.1, 0),
        'src_baseline': False,
        'cat': None,
        'morph': False,
        'mask': False,
        'samplingrate': None,
        'decim': None,
        'ndvar': True,
        'data_raw': False,
        'keep_evoked': False,
    })

    assert evoked_stc_default.current_fingerprint() == evoked_stc_view.current_fingerprint()
    assert evoked_stc_default.current_fingerprint() != evoked_stc_artifact.current_fingerprint()


@requires_mne_sample_data
def test_selected_events_vardef_is_local():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, n_segments=2, mris=False)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', rej='')
    options = {
        'reject': True,
        'add_bads': True,
        'index': True,
        'data_raw': False,
        'cat': None,
    }
    compact = Variables({'grouped': LabelVar('trigger', {(1, 2): 'target'}, task='sample')})
    changed = Variables({'grouped': LabelVar('trigger', {1: 'target', 2: 'nontarget'}, task='sample')})

    handle = e._derivatives.resolve('selected-events', state=e._derivative_state(), options=options)
    _ = handle.load(cache=True)

    assert 'vardef' not in handle.current_fingerprint()
    with pytest.raises(TypeError, match="undeclared option"):
        e._derivatives.resolve('selected-events', state=e._derivative_state(), options={**options, 'vardef': compact})

    ds_compact = e.load_selected_events(vardef=compact)
    ds_changed = e.load_selected_events(vardef=changed)
    assert set(ds_compact['grouped'].cells) == {'', 'target'}
    assert 'nontarget' in ds_changed['grouped'].cells


@requires_mne_sample_data
def test_coreg_report_dependencies_are_explicit():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=2, n_segments=2, mris=True)
    root = join(tempdir, 'SampleExperiment')

    e = SampleExperiment(root)
    handle = e._derivatives.resolve('coreg-report', state=e._derivative_state(), options={'dst': None})

    assert 'dependencies' not in handle.current_fingerprint()
    dependencies = handle.dependency_fingerprints()
    assert any(key.endswith(':raw') for key in dependencies)
    assert any(key.endswith(':trans') for key in dependencies)


@requires_mne_sample_data
def test_sample_neuromag():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, n_subjects=1, pick='')

    class Experiment(SampleExperiment):
        defaults = {'raw': '1-40'}
        # raw = {'1-40': RawFilter('raw', 1, 40)}

    root = join(tempdir, 'SampleExperiment')
    e = Experiment(root)
    e.set(raw='1-40', epoch='target', rej='')

    # Check original events
    ds = e.load_events()
    assert ds.n_cases == 80
    ds = e.load_selected_events()
    assert ds.n_cases == 73

    # Check auto-rejection
    e.set(rej='man')
    e.make_epoch_selection(auto={'mag': 2e-12, 'grad': 5e-11, 'eeg': 1.5e-4})
    ds = e.load_selected_events(reject='keep')
    assert ds['accept'].sum() == 69


@requires_mne_sample_data
def test_sample_eeg():
    set_log_level('warning', 'mne')

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 1, 1, pick='eeg')

    class Experiment(Pipeline):

        raw = {
            'av-ref': RawReReference('raw'),
        }

    root = join(tempdir, 'SampleExperiment')
    e = Experiment(root)

    # average reference
    raw = e.load_raw(raw='av-ref')
    assert raw.info['custom_ref_applied'] == True
