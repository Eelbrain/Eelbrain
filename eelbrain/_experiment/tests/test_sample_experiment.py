# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Test Pipeline using mne-python sample data"""
import itertools
import json
import logging
from os.path import join, exists
from os import remove
from pathlib import Path
import shutil
import tomllib
from unittest.mock import patch
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
from eelbrain._experiment.parc.nodes import AnnotDerivative
from eelbrain._experiment.pathing import BIDS_ENTITY_KEYS, LOG_DIR, ica_file_path
from eelbrain._experiment.preprocessing import RawFilterElliptic, ica_input_name, raw_node_name
from eelbrain._experiment.data import DataSpec
from eelbrain._experiment.variable_def import EvalVar, LabelVar, Variables
from eelbrain.testing import assert_dataobj_equal, requires_mne_sample_data


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
        disconnect_labels: bool = False,
        baseline=True,
        src_baseline=None,
        smooth=None,
        samplingrate=None,
) -> Path:
    options = {
        'data': DataSpec.coerce(data),
        'samples': samples,
        'test': test,
        'tstart': tstart,
        'tstop': tstop,
        'pmin': pmin,
        'disconnect_labels': disconnect_labels,
        'baseline': baseline,
        'src_baseline': src_baseline,
        'smooth': smooth,
        'samplingrate': samplingrate,
    }
    return e._derivatives.resolve(node, state=e.state, options=options).manifest_path


@pytest.fixture(scope='session')
def _samples_templates(tmp_path_factory):
    "Per-session cache of sample-experiment templates, keyed by setup configuration"
    return tmp_path_factory.mktemp('samples_templates'), {}


@pytest.fixture
def samples_experiment(_samples_templates, tmp_path):
    """Sample-experiment dataset roots backed by per-configuration templates.

    ``datasets.setup_samples_experiment`` is expensive, so each distinct
    configuration is built only once per test session and cached. Every call
    returns a fresh copy of the relevant template, so tests stay isolated while
    the dataset is generated only once per kind.
    """
    template_dir, cache = _samples_templates
    counter = itertools.count()

    def make(
            n_subjects: int = 3,
            n_tasks: int = 1,
            n_segments: int = 4,
            n_runs: int = 1,
            mris: bool = False,
            pick: str = 'mag',
    ) -> str:
        if not mne.datasets.has_dataset("sample"):
            pytest.skip("mne sample data unavailable")
        key = (n_subjects, n_tasks, n_segments, n_runs, mris, pick)
        if key not in cache:
            template = template_dir / f'template-{len(cache)}'
            template.mkdir()
            datasets.setup_samples_experiment(template, n_subjects, n_tasks, n_segments, n_runs, mris, pick=pick)
            cache[key] = template / 'SampleExperiment'
        root = tmp_path / f'experiment-{next(counter)}' / 'SampleExperiment'
        shutil.copytree(cache[key], root)
        return str(root)

    return make


@requires_mne_sample_data
def test_sample(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=3, n_segments=2, mris=True)
    e = SampleExperiment(root)

    assert e.get('raw') == '1-40'
    assert e.get('subject') == 'R0000'
    assert e.get('subject', subject='R0002') == 'R0002'
    assert e._raw['raw'].name == 'raw'
    assert e._parcs['ac'].name == 'ac'
    assert e._parcs['lobes'].name == 'lobes'
    tree = e._show_dependencies('evoked', return_str=True)
    assert 'evoked [derivative]' in tree
    # Dataset assembly is always uncached
    assert 'epochs [uncached]' in tree
    wrapped_tree = e._show_dependencies('evoked', max_line_length=60, return_str=True)
    assert all(len(line) <= 60 for line in wrapped_tree.splitlines())

    # wildcard formatting
    with e._temporary_state:
        state = e.state
        state['subject'] = '*'
        assert str(ica_file_path(state, '*', datatype='meg')) == join('derivatives', 'mne', 'sub-*', 'meg', 'sub-*_desc-*_ica.fif')
        state['subject'] = 'R0002'
        assert str(ica_file_path(state, '*', datatype='meg')) == join('derivatives', 'mne', 'sub-R0002', 'meg', 'sub-R0002_desc-*_ica.fif')

    # events
    e.set('R0001', epoch_rejection='')
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
        assert exists(e._derivatives.resolve(raw_node_name('1-40'), state=e.state).manifest_path)
        e.set(cov='emptyroom', raw='tsss')
        cov = e.load_cov()
        assert isinstance(cov, mne.Covariance)
        assert exists(e._derivatives.resolve('cov', state=e.state).manifest_path)
        assert e.load_bad_channels(noise=True) == []
        e.set(cov='emptyroom', raw='1-40')
        cov = e.load_cov()
        assert isinstance(cov, mne.Covariance)
        assert exists(e._derivatives.resolve('cov', state=e.state).manifest_path)
        assert e.load_bad_channels(noise=True) == []
        e.load_cov()

    # evoked cache invalidated by change in bads
    e.set('R0001', epoch_rejection='', epoch='target')
    e.load_events()
    assert exists(e._resolve_derivative('labeled-events').manifest_path)
    ds = e.load_evoked(ndvar=False)
    assert exists(e._resolve_derivative('evoked').manifest_path)
    assert ds[0, 'evoked'].info['bads'] == []
    e.make_bad_channels(['MEG 0331'])
    ds = e.load_evoked(ndvar=False)
    assert ds[0, 'evoked'].info['bads'] == ['MEG 0331']

    e.set(epoch_rejection='manual')
    test_tree = e._show_dependencies(
        'test-result',
        options={
            'data': DataSpec.coerce('meg.rms'),
            'samples': 100,
            'test': 'a>v',
            'tstart': 0.05,
            'tstop': 0.2,
            'pmin': 0.05,
            'baseline': False,
            'src_baseline': None,
            'smooth': None,
            'samplingrate': None,
        },
        return_str=True,
    )
    assert 'evoked-test-data [uncached]' in test_tree
    assert 'evoked-group-dataset [uncached]' in test_tree
    sds = []
    for _ in e:
        e.make_epoch_rejection(auto=2.5e-12)
        sds.append(e.load_evoked())
    ds_ind = combine(sds, dim_intersection=True)

    ds = e.load_evoked('all')
    ds['mag'] = ds['mag'].sub(sensor=ds['mag'].sensor.index(exclude='MEG 0331'))  # load_evoked interpolates bad channel
    assert_dataobj_equal(ds_ind, ds, decimal=19)  # make vs load evoked

    # sensor space tests
    megs = [e.load_evoked(cat='auditory', baseline=False, model='modality', interpolate_bads=True)['mag'] for _ in e]
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='meg.rms', inv='', baseline=False)
    test_manifest = _test_result_manifest_path(e, 'a>v', 0.05, 0.2, 0.05, samples=100, data='meg.rms', baseline=False)
    assert exists(test_manifest)
    with open(test_manifest) as fid:
        test_manifest_data = json.load(fid)
    assert test_manifest_data['fingerprint']['test']['tail'] == 1
    assert test_manifest_data['fingerprint']['epoch']['tmax'] == 0.3
    assert 'dependencies' not in test_manifest_data['fingerprint']
    assert 'evoked-test-data' in test_manifest_data['dependencies']
    remove(test_manifest)
    _ = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='meg.rms', inv='', baseline=False)
    assert exists(test_manifest)

    meg_rms = combine(meg.rms('sensor') for meg in megs).mean('case', name='auditory')
    assert_dataobj_equal(res.c1_mean, meg_rms, decimal=21)
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='meg.mean', inv='', baseline=False)
    meg_mean = combine(meg.mean('sensor') for meg in megs).mean('case', name='auditory')
    assert_dataobj_equal(res.c1_mean, meg_mean, decimal=21)
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, inv='', baseline=False)
    assert res.p.min() == pytest.approx(.143, abs=.001)
    assert res.difference.max() == pytest.approx(4.47e-13, 1e-15)
    # plot (skip to avoid using framework build)
    # e.plot_evoked(1, epoch='target', model='')

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
            'shift_t': LabelVar('value', {(1, 3): 0, (2, 4): shift})
        }
    e = Experiment(root)
    # test shift in events
    ds = e.load_events()
    assert_dataobj_equal(ds['shift_t'], ds['shift'], name=False)
    # compare against epochs (baseline correction on epoch level rather than evoked for smaller numerical error)
    ep = e.load_epochs(baseline=True, epoch='visual', epoch_rejection='').aggregate('side')
    evs = e.load_evoked(baseline=True, epoch='visual-s', epoch_rejection='', model='side')
    tstart = ep['mag'].time.tmin - shift
    assert_dataobj_equal(evs[0, 'mag'], ep[0, 'mag'].sub(time=(tstart, None)), decimal=19)
    tstop = ep['mag'].time.tstop + shift
    assert_almost_equal(evs[1, 'mag'].x, ep[1, 'mag'].sub(time=(None, tstop)).x, decimal=19)
    # baseline correction can not be deferred/disabled for post_baseline_trigger_shift epochs
    with pytest.raises(NotImplementedError):
        e.load_epochs(baseline=False, epoch='visual-s', epoch_rejection='')
    with pytest.raises(NotImplementedError):
        e.load_evoked(baseline=False, epoch='visual-s', epoch_rejection='', model='side')

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
    v = ds.sub("epoch=='visual'", 'mag')
    v_target = e.load_epochs(baseline=True, epoch='visual')['mag'].sub(time=(-0.1, v.time.tstop))
    assert_almost_equal(v.x, v_target.x)
    a = ds.sub("epoch=='auditory'", 'mag').sub(time=(-0.1, 0.099))
    a_target = e.load_epochs(baseline=True, epoch='auditory')['mag'].sub(time=(0, 0.199))
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

    class Experiment(SampleExperiment):
        groups = {
            'ab': ('R0002', 'R0000'),
            'alias': ('R0000', 'R0002'),
        }
    e = Experiment(root)
    assert e.get_field_values('subject', group='ab') == e.get_field_values('subject', group='alias') == ['R0000', 'R0002']
    # Group is part of the derivative's declared identity
    result_options = {
        'data': DataSpec.coerce('meg.rms'),
        'samples': 20,
        'test': 'a>v',
        'tstart': 0.05,
        'tstop': 0.2,
        'pmin': 0.05,
        'baseline': False,
        'src_baseline': None,
        'smooth': None,
        'samplingrate': None,
    }
    e.set(group='ab', epoch_rejection='manual')
    handle_ab = e._resolve_derivative('test-result', options=result_options)
    e.set(group='alias')
    handle_alias = e._resolve_derivative('test-result', options=result_options)
    assert handle_ab.artifact_path != handle_alias.artifact_path

    class BadExperiment(SampleExperiment):
        parcs = {'ac': 'aparc'}
    with pytest.raises(TypeError, match="need Parcellation"):
        BadExperiment(root)

    # changes
    class Changed(SampleExperiment):
        variables = {
            'event': LabelVar('value', {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'}),
            'side': LabelVar('value', {(1, 3): 'left', (2, 4): 'right_changed'}),
            'modality': LabelVar('value', {(1, 2): 'auditory', (3, 4): 'visual'}),
        }
        tests = {
            'twostage': TwoStageTest(
                'side_left + modality_a',
                {'side_left': EvalVar("side == 'left'"),
                 'modality_a': EvalVar("modality == 'auditory'")}),
            'novars': TwoStageTest('side + modality'),
        }
    e = Changed(root)

    # changed variable, while a test with model=None is not changed
    class Changed(Changed):
        variables = {
            'side': LabelVar('value', {(1, 3): 'left', (2, 4): 'right_changed'}),
            'modality': LabelVar('value', {(1, 2): 'auditory', (3, 4): 'visual_changed'}),
        }
    e = Changed(root)

    # changed variable, unchanged test with vardef=None
    class Changed(Changed):
        variables = {
            'side': LabelVar('value', {(1, 3): 'left', (2, 4): 'right_changed'}),
            'modality': LabelVar('value', {(1, 2): 'auditory', (3, 4): 'visual_changed'}),
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
    ica_manifest = e._derivatives.manifest_path(ica_path, ica_input_name('ica'))
    assert exists(ica_manifest)
    ica_manifest_data = json.loads(Path(ica_manifest).read_text())
    assert ica_manifest_data['resolve_state'] == {'subject': 'R0000', 'session': '', 'acquisition': ''}
    assert ica_manifest_data['resolve_options'] == {}
    assert not [entry for entry in e._derivatives.scan_cache().entries if entry.manifest_path == Path(ica_manifest)]

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
    e.set(raw='ica1-40', epoch_rejection='manual')
    e.make_epoch_rejection(auto=2e-12, overwrite=True)
    ds1 = e.load_evoked(raw='ica1-40')
    ica = e.load_ica(raw='ica')
    ica.exclude = [0, 1, 2]
    ica.save(ica_path, overwrite=True)
    ds2 = e.load_evoked(raw='ica1-40')
    assert not np.allclose(ds1['mag'].x, ds2['mag'].x, atol=1e-20), "ICA change ignored"
    # apply-ICA
    with catch_warnings():
        filterwarnings('ignore', "The measurement information indicates a low-pass frequency", RuntimeWarning)
        ds1 = e.load_evoked(raw='ica', epoch_rejection='')
        ds2 = e.load_evoked(raw='apply-ica', epoch_rejection='')
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
    # res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False)
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
            ds = ds.sub("event == 'smiley'")
            ds['new_var'] = Var([i + 1 for i in ds['sample']])
            return ds

    e = Experiment(root)
    events = e.load_events()
    assert_array_equal(events['new_var'], [67402, 75306])

    # Parc
    # ----
    labels = e.load_annot(parc='ac', mrisubject='fsaverage')
    assert len(labels) == 4
    annot_handle = e._resolve_derivative('annot')
    assert exists(annot_handle.manifest_path)
    # make_annot() only materializes the annot files; it does not read the labels back
    assert annot_handle.is_valid()
    with patch.object(AnnotDerivative, 'load', autospec=True) as annot_load:
        e.make_annot(parc='ac', mrisubject='fsaverage')
    annot_load.assert_not_called()
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
def test_sample_source(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=3, n_segments=1, mris=True)  # TODO: use sample MRI which already has forward solution
    e = SampleExperiment(root)

    # source space tests
    # ico-2 (320 vertices/hemi) keeps forward/inverse fast while still covering the transversetemporal ROI
    e.set(epoch='auditory', epoch_rejection='', src='ico-2', parc='ac', inv='free-3-dSPM')
    morph = e.load_source_morph(subject='R0000')
    assert isinstance(morph, mne.SourceMorph)
    assert exists(e._resolve_derivative('source-morph').manifest_path)
    res = e.load_test('left=right', 0.05, 0.2, 0.05, samples=8)
    res_labels = e.load_test('left=right', 0.05, 0.2, 0.05, samples=8, disconnect_labels=True)
    assert exists(e._resolve_derivative('src').manifest_path)
    assert exists(e._resolve_derivative('fwd').manifest_path)
    assert exists(e._resolve_derivative('inv').manifest_path)
    # cat is a view option on evoked-stc: subsetting model cells
    ds_all = e.load_evoked(model='side', ndvar=False, inv='free-3-dSPM')
    ds_left = e.load_evoked(model='side', cat=('left',), ndvar=False, inv='free-3-dSPM')
    assert set(ds_all['side'].cells) == {'left', 'right'}
    assert set(ds_left['side'].cells) == {'left'}
    assert ds_left.n_cases < ds_all.n_cases
    with open(_test_result_manifest_path(e, 'left=right', 0.05, 0.2, 0.05, samples=8, data='source')) as fid:
        source_manifest_data = json.load(fid)
    with open(_test_result_manifest_path(e, 'left=right', 0.05, 0.2, 0.05, samples=8, data='source', disconnect_labels=True)) as fid:
        disconnected_manifest_data = json.load(fid)
    assert source_manifest_data['fingerprint']['parc']['base'] == 'aparc'
    assert source_manifest_data['key']['parc'] == 'ac'
    assert 'dependencies' not in source_manifest_data['fingerprint']
    assert 'evoked-test-data' in source_manifest_data['dependencies']
    source_data_deps = source_manifest_data['dependencies']['evoked-test-data']['dependencies']
    assert source_data_deps['dataset']['name'] == 'evoked-stc-group-dataset'
    assert set(source_data_deps['dataset']['dependencies']) == {'R0000', 'R0001', 'R0002'}
    assert source_manifest_data['key']['options']['disconnect_labels'] is False
    assert disconnected_manifest_data['key']['options']['disconnect_labels'] is True
    assert_dataobj_equal(res.t, res_labels.t)
    # ROI tests
    e.set(epoch='target')
    ress = e.load_test('left=right', 0.05, 0.2, 0.05, samples=8, data='source.rms')
    with open(_test_result_manifest_path(e, 'left=right', 0.05, 0.2, 0.05, samples=8, data='source.rms')) as fid:
        roi_manifest_data = json.load(fid)
    assert 'evoked-test-data' in roi_manifest_data['dependencies']
    roi_deps = roi_manifest_data['dependencies']['evoked-test-data']['dependencies']
    assert set(roi_deps) == {'dataset'}
    assert roi_deps['dataset']['name'] == 'evoked-stc-group-dataset'
    group_deps = roi_deps['dataset']['dependencies']
    assert set(group_deps) == {'R0000', 'R0001', 'R0002'}
    assert all(group_deps[subject]['name'] == 'evoked-stc' for subject in group_deps)
    assert all('source-morph' not in group_deps[subject]['dependencies'] for subject in group_deps)
    res = ress.res['transversetemporal-lh']
    assert res.p.min() == 1 / 7
    with pytest.raises(TypeError, match='disconnect_labels'):
        e.load_test('left=right', 0.05, 0.2, 0.05, samples=8, data='source.rms', disconnect_labels=True)
    ress = e.load_test('twostage', 0.05, 0.2, 0.05, samples=8, data='source.rms')
    with open(_test_result_manifest_path(e, 'twostage', 0.05, 0.2, 0.05, node='two-stage-level-2', samples=8, data='source.rms')) as fid:
        two_stage_manifest_data = json.load(fid)
    assert 'two-stage-level-1' in {dep['name'] for dep in two_stage_manifest_data['dependencies'].values()}
    subject_dep = two_stage_manifest_data['dependencies']['R0000']
    with open(e._derivatives.cache_dir / subject_dep['manifest']) as fid:
        level_1_manifest_data = json.load(fid)
    assert level_1_manifest_data['dependencies']['two-stage-data']['dependencies']['data']['name'] == 'evoked-stc'
    # ds_return, _ = e.load_test('twostage', 0.05, 0.2, 0.05, samples=8, return_data=True)
    # assert isinstance(ds_return, Dataset)
    # assert 'subject' in ds_return
    res = ress.res['transversetemporal-lh']
    assert res.samples == -1
    assert res.tests['intercept'].p.min() == 1 / 7

    # Parc needs to be set
    with pytest.raises(ValueError, match='state parc'):
        e.load_test('left=right', 0.05, 0.2, 0.05, samples=8, parc='')

    # Outdated test requires make=True
    class ChangedParcExperiment(SampleExperiment):
        parcs = {
            **SampleExperiment.parcs,
            'ac': SubParc('aparc', ('superiortemporal',)),
        }


@requires_mne_sample_data
def test_sample_tasks(monkeypatch, samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(2, 2, 1)

    class Experiment(SampleExperiment):
        defaults = {**SampleExperiment.defaults, 'epoch_rejection': 'manual'}

        raw = {
            'ica': RawICA('raw', ('sample1', 'sample2'), 'fastica', max_iter=1, cache=True),
            'av-ref': RawReReference('raw'),
            **SampleExperiment.raw,
        }

    e = Experiment(root)

    # get paths
    handle = e._resolve_derivative(raw_node_name('ica'))
    assert 'root' not in e.state
    assert handle.root == Path(root)
    assert handle.artifact_path.is_relative_to(Path(root) / 'derivatives' / 'eelbrain' / 'cache' / 'raw@ica')
    assert handle.artifact_path.suffix == '.fif'
    assert '_key-' in handle.artifact_path.name
    assert str(ica_file_path(e.state, 'ica', datatype='meg')) == join('derivatives', 'mne', 'sub-R0000', 'meg', 'sub-R0000_desc-ica_ica.fif')
    ica_handle = e._resolve_derivative(ica_input_name('ica'))
    assert ica_handle.load(view='status') == 'missing-ica'
    e.set(raw='raw')

    # bad channels are stored in derivatives, not in the BIDS source dataset
    bad_path = join(root, 'derivatives', 'mne', 'sub-R0000', 'meg', 'sub-R0000_task-sample1_channels.tsv')
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
    e.show_bad_channels()

    # rejection
    for _ in e:
        for epoch in ('target1', 'target2'):
            e.set(epoch=epoch)
            e.make_epoch_rejection(auto=2e-12)

    ds = e.load_evoked('R0000', epoch='target2')
    e.set(task='sample1')
    ds2 = e.load_evoked('R0000')
    assert_dataobj_equal(ds2, ds, decimal=19)

    # super-epoch
    ds1 = e.load_epochs(epoch='target1', interpolate_bads=True)
    ds2 = e.load_epochs(epoch='target2', interpolate_bads=True)
    recording_epochs_node = e._derivatives._get_node('recording-epochs')
    recording_epochs_build = recording_epochs_node.build
    recording_epochs_builds = []

    def count_recording_epochs_builds(ctx):
        recording_epochs_builds.append(ctx.state['epoch'])
        return recording_epochs_build(ctx)

    monkeypatch.setattr(recording_epochs_node, 'build', count_recording_epochs_builds)
    ds_super = e.load_epochs(epoch='super', interpolate_bads=True)
    assert recording_epochs_builds == ['target1', 'target2']
    assert ds_super.info['epoch'] == 'super'
    assert_dataobj_equal(ds_super['mag'], combine((ds1['mag'], ds2['mag'])))
    # SuperEpoch should depend on the same sub-epoch request as direct loading.
    super_handle = e._resolve_derivative('epochs')
    super_dependencies = super_handle.dependency_fingerprints()
    target2_dependency = next(dep for dep in super_handle.node.dependencies(super_handle) if dep.label == 'target2')
    with e._temporary_state:
        e.set(epoch='target2')
        target2_entry = e._resolve_derivative('epochs', options=target2_dependency.options).describe_dependency()
    assert super_dependencies['target2'] == target2_entry
    # evoked
    dse_super = e.load_evoked(epoch='super', model='modality%side')
    ds_super_keep = e.load_epochs(epoch='super', interpolate_bads='keep')
    target = ds_super_keep.aggregate('modality%side', drop=('sample', 't_edf', 'onset', 'index', 'value', 'task', 'interpolate_channels', 'epoch'))
    assert_dataobj_equal(dse_super, target, 19)

    # conflicting task and epoch settings
    rej_path = join(root, 'derivatives', 'mne', 'sub-R0000', 'meg', 'sub-R0000_raw-1-40_epoch-target2_rej-manual_epoch.pickle')
    e.set(epoch='target2', raw='1-40')
    assert not exists(rej_path)
    e.set(task='sample1')
    e.make_epoch_rejection(auto=2e-12)
    assert exists(rej_path)

    # ica
    e.set('R0000', raw='ica')
    with catch_warnings():
        filterwarnings('ignore', "FastICA did not converge", UserWarning)
        ica_path = e.make_ica()
    assert ica_path == Path(root) / 'derivatives' / 'mne' / 'sub-R0000' / 'meg' / 'sub-R0000_desc-ica_ica.fif'


@requires_mne_sample_data
def test_ica_all_tasks_after_maxwell(samples_experiment):
    "task=None ICA after RawMaxwell uses all tasks and runs per subject/session/acquisition"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(n_subjects=2, n_tasks=2, n_segments=1, n_runs=2)

    # task=None with multiple tasks is rejected without a preceding RawMaxwell step
    class BadExperiment(SampleExperiment):
        raw = {**SampleExperiment.raw, 'ica': RawICA('1-40')}
    with pytest.raises(ConfigurationError, match='RawMaxwell'):
        BadExperiment(root)

    class Experiment(SampleExperiment):
        raw = {
            'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True, st_overlap=False),
            'ica': RawICA('tsss', method='fastica', max_iter=1, n_components=0.95),
            **SampleExperiment.raw,
        }
    e = Experiment(root)
    # task=None resolves to all tasks; after RawMaxwell runs are concatenated
    assert e._raw['ica'].task == ('sample1', 'sample2')
    assert e._raw['ica']._concatenate_runs is True

    e.set('R0000', raw='ica')
    # the ICA spans all tasks/runs, so the file is per subject/session/acquisition (no task/run entity)
    assert str(ica_file_path(e.state, 'ica', concatenate_runs=True, datatype='meg')) == join('derivatives', 'mne', 'sub-R0000', 'meg', 'sub-R0000_desc-ica_ica.fif')
    with catch_warnings():
        filterwarnings('ignore', "FastICA did not converge", UserWarning)
        ica_path = e.make_ica()
    assert ica_path == Path(root) / 'derivatives' / 'mne' / 'sub-R0000' / 'meg' / 'sub-R0000_desc-ica_ica.fif'
    assert exists(ica_path)
    assert isinstance(e.load_ica(), mne.preprocessing.ICA)
    # the ICA can be applied to an individual recording
    assert isinstance(e.load_raw(), mne.io.BaseRaw)


@requires_mne_sample_data
def test_epoch_reference(samples_experiment):
    "EEG re-referencing after channel interpolation (the 'reference' state)"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(1, 1, pick='')  # keep EEG channels

    class Experiment(SampleExperiment):
        references = {'avg': Reference('average')}

    e = Experiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='raw')

    # default reference='' leaves EEG unreferenced
    ds0 = e.load_epochs(reference='', interpolate_bads=False)
    assert float(ds0['eeg'].mean('sensor').abs().max()) > 1e-6

    # reference='avg' drives the EEG sensor-mean to ~0, with and without
    # interpolation (the reference is applied after interpolation either way)
    for interpolate_bads in (False, True):
        ds = e.load_epochs(reference='avg', interpolate_bads=interpolate_bads)
        assert float(ds['eeg'].mean('sensor').abs().max()) < 1e-15
        # MEG is untouched by EEG re-referencing
        ds_ref0 = e.load_epochs(reference='', interpolate_bads=interpolate_bads)
        assert_dataobj_equal(ds['mag'], ds_ref0['mag'], decimal=20)

    # changing the Reference config invalidates the cache (same name)
    e.set(reference='avg')
    e.load_evoked(ndvar=False, model='modality')

    class ChangedExperiment(Experiment):
        references = {'avg': Reference(['EEG 001'])}

    e_changed = ChangedExperiment(root)
    e_changed.set(subject='R0000', epoch='target', epoch_rejection='', raw='raw', reference='avg')
    assert not e_changed._resolve_derivative('evoked', options={'model': 'modality'}).is_valid()

    # MEG-only data: a reference with no EEG to apply raises (rather than
    # silently producing a duplicate cache entry); reference='' works
    meg_root = samples_experiment(1, 1, pick='mag')
    e_meg = Experiment(meg_root)
    e_meg.set(subject='R0000', epoch='target', epoch_rejection='', raw='raw')
    with pytest.raises(ConfigurationError):
        e_meg.load_epochs(reference='avg')
    e_meg.load_epochs(reference='')


@requires_mne_sample_data
def test_interpolate_bads(samples_experiment):
    "load_epochs interpolate_bads False / 'keep' / True semantics"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(1, 1, pick='mag')
    bad = 'MEG 0111'
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='raw')
    e.make_bad_channels(bad)

    epo_false = e.load_epochs(ndvar=False, interpolate_bads=False)['epochs']
    epo_keep = e.load_epochs(ndvar=False, interpolate_bads='keep')['epochs']
    epo_true = e.load_epochs(ndvar=False, interpolate_bads=True)['epochs']

    # the bad channel stays marked for False/'keep' and is reset for True
    assert epo_false.info['bads'] == [bad]
    assert epo_keep.info['bads'] == [bad]
    assert epo_true.info['bads'] == []

    # 'keep' interpolates the data (changed vs False); True yields the same data as 'keep',
    # differing only by the bad-channel marker (so it can share the cached artifact)
    i = epo_false.ch_names.index(bad)
    data_false = epo_false.get_data()[:, i]
    data_keep = epo_keep.get_data()[:, i]
    data_true = epo_true.get_data()[:, i]
    assert not np.array_equal(data_false, data_keep)
    assert_array_equal(data_true, data_keep)

    # the interpolated channel is included in NDVar output only for True
    assert bad not in e.load_epochs(interpolate_bads=False)['mag'].sensor.names
    assert bad not in e.load_epochs(interpolate_bads='keep')['mag'].sensor.names
    assert bad in e.load_epochs(interpolate_bads=True)['mag'].sensor.names


@requires_mne_sample_data
def test_interpolate_bads_after_ica(samples_experiment):
    "A channel bad at ICA-fit time is preserved (not dropped) for downstream interpolation"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(1, 1, pick='')  # keep EEG channels

    class Experiment(SampleExperiment):
        raw = {
            **SampleExperiment.raw,
            'ica': RawICA('tsss', 'sample', method='fastica', n_components=0.95, fit_kwargs={'reject': None}),
        }

    e = Experiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='ica')
    bad = 'EEG 003'
    e.make_bad_channels(bad)  # bad before fit -> excluded from the ICA decomposition
    with catch_warnings():
        filterwarnings('ignore', "FastICA did not converge", UserWarning)
        e.make_ica()

    # the channel is excluded from the ICA, so it is absent from ica.info['bads']
    assert bad not in e.load_ica().ch_names

    # the post-ICA raw still contains the bad channel (kept marked, not dropped)
    raw = e.load_raw()
    assert bad in raw.ch_names
    assert bad in raw.info['bads']

    # interpolation works just as without ICA: included for True, excluded otherwise
    assert bad in e.load_epochs(interpolate_bads=True)['eeg'].sensor.names
    assert bad not in e.load_epochs(interpolate_bads='keep')['eeg'].sensor.names
    assert bad not in e.load_epochs(interpolate_bads=False)['eeg'].sensor.names


def test_variable_length_epochs(samples_experiment):
    "load_epochs for variable-length (variable-tmax) epochs returns per-epoch NDVars"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(1, n_segments=2, mris=False)

    class Experiment(SampleExperiment):
        epochs = {
            **SampleExperiment.epochs,
            # tmax varies per epoch (0.2 or 0.3 s) -> variable-length epochs
            'varlen': PrimaryEpoch('sample', "event == 'target'", tmin=-0.1, tmax='0.2 + 0.1*(index % 2)', decim=5),
            # all selected events form a single ContinuousEpoch segment
            'cont-one': ContinuousEpoch('sample', "event == 'target'", pad_start=0.1, pad_end=0.1, split=1000),
            # every selected event forms an equal-length segment
            'cont-equal': ContinuousEpoch('sample', "event == 'target'", pad_start=0.1, pad_end=0.1, split=0),
        }

    e = Experiment(root)
    e.set(subject='R0000', epoch='varlen', epoch_rejection='', raw='raw')

    ds = e.load_epochs()
    n = ds.n_cases
    assert n > 0
    # each epoch becomes its own NDVar because the epochs have different lengths
    assert isinstance(ds['mag'], Datalist)
    assert len(ds['mag']) == n
    n_times = {y.time.nsamples for y in ds['mag']}
    assert len(n_times) == 2  # two distinct epoch lengths
    assert 'epochs' not in ds

    # ndvar=False keeps the raw MNE epochs as a Datalist, one per trial
    ds_mne = e.load_epochs(ndvar=False)
    assert isinstance(ds_mne['epochs'], Datalist)
    assert len(ds_mne['epochs']) == n

    # keep_mne keeps both the MNE epochs and the NDVars
    ds_both = e.load_epochs(keep_mne=True)
    assert isinstance(ds_both['epochs'], Datalist)
    assert isinstance(ds_both['mag'], Datalist)

    # ContinuousEpoch keeps the segmented representation even when there is
    # only one segment.
    e.set(epoch='cont-one')
    ds_cont = e.load_epochs(keep_mne=True)
    assert ds_cont.n_cases == 1
    assert isinstance(ds_cont['epochs'], Datalist)
    assert isinstance(ds_cont['mag'], Datalist)
    assert ds_cont['epochs'][0].times[0] == pytest.approx(-0.1, abs=0.002)
    assert ds_cont['mag'][0].time.tmin == pytest.approx(ds_cont['epochs'][0].times[0])

    # Equal-length ContinuousEpoch segments likewise remain separate because
    # they occupy different positions on the epoch-wide clock.
    e.set(epoch='cont-equal')
    ds_cont = e.load_epochs(keep_mne=True)
    assert ds_cont.n_cases > 1
    assert isinstance(ds_cont['epochs'], Datalist)
    assert isinstance(ds_cont['mag'], Datalist)
    assert len({y.time.nsamples for y in ds_cont['mag']}) == 1
    assert len({y.time.tmin for y in ds_cont['mag']}) == ds_cont.n_cases


@requires_mne_sample_data
def test_channel_model_rejection(samples_experiment):
    "Automatic epoch rejection via ChannelModel (the 'epoch_rejection' state)"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment
    from eelbrain._info import INTERPOLATE_CHANNELS

    root = samples_experiment(1, 1, pick='')  # keep EEG channels

    class Experiment(SampleExperiment):
        epoch_rejection = {'auto': ChannelModelRejection(model='ridge', fit_threshold=None, score_threshold=2e-5, max_interpolate=2)}

    e = Experiment(root)
    e.set(subject='R0000', epoch='target', raw='raw')
    n_total = e.load_epochs(epoch_rejection='', interpolate_bads=False).n_cases

    # build + cache the automatically generated rejection file
    e.set(epoch_rejection='auto')
    ctx = e._resolve_derivative('epoch-rejection-channel-model')
    rej_ds = ctx.load()
    cache_path = ctx.node.path(ctx)
    assert exists(str(cache_path))
    assert 'cache' in cache_path.parts and 'epoch-rejection-channel-model' in cache_path.parts
    assert rej_ds.n_cases == n_total
    n_rejected = int((~rej_ds['accept']).sum())
    n_interp = sum(1 for x in rej_ds[INTERPOLATE_CHANNELS] if x)
    assert n_rejected > 0  # some epochs rejected (> max_interpolate bad channels)
    assert n_interp > 0    # some epochs have channels marked for interpolation
    assert max(len(x) for x in rej_ds[INTERPOLATE_CHANNELS]) <= 2  # never exceeds max_interpolate
    assert set(rej_ds['rej_tag'][~rej_ds['accept'].x]) == {'channel-model'}

    # end-to-end: reject=True drops the rejected epochs
    ds = e.load_epochs(reject=True, interpolate_bads=False)
    assert ds.n_cases == n_total - n_rejected
    # second resolve is a cache hit (no rebuild)
    assert e._resolve_derivative('epoch-rejection-channel-model').is_valid()

    # MEG-only data: ChannelModelRejection has no EEG to model -> raises
    meg_root = samples_experiment(1, 1, pick='mag')
    e_meg = Experiment(meg_root)
    e_meg.set(subject='R0000', epoch='target', raw='raw', epoch_rejection='auto')
    with pytest.raises(ConfigurationError):
        e_meg._resolve_derivative('epoch-rejection-channel-model').load()


@requires_mne_sample_data
def test_channel_model_rejection_continuous(samples_experiment):
    "ChannelModelRejection: equal-length epochs longer than ``continuous`` use windowed detection"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment
    from eelbrain._info import INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS

    root = samples_experiment(1, 1, pick='')  # keep EEG channels

    # ``continuous`` below the (equal) epoch duration -> time-resolved detection
    class Experiment(SampleExperiment):
        epoch_rejection = {'auto': ChannelModelRejection(model='ridge', fit_threshold=None, score_threshold=1e-5, max_interpolate=2, continuous=0.1)}

    e = Experiment(root)
    e.set(subject='R0000', epoch='target', raw='raw', epoch_rejection='auto')
    rej_ds = e._resolve_derivative('epoch-rejection-channel-model').load()
    assert INTERPOLATE_WINDOWS in rej_ds
    assert INTERPOLATE_CHANNELS not in rej_ds
    assert rej_ds['accept'].x.all()  # windowed detection never rejects wholesale

    # with the default ``continuous`` (5 s) the same short epoch uses whole-epoch detection
    class Experiment2(SampleExperiment):
        epoch_rejection = {'auto': ChannelModelRejection(model='ridge', fit_threshold=None, score_threshold=1e-5, max_interpolate=2)}

    e2 = Experiment2(root)
    e2.set(subject='R0000', epoch='target', raw='raw', epoch_rejection='auto')
    rej_ds2 = e2._resolve_derivative('epoch-rejection-channel-model').load()
    assert INTERPOLATE_CHANNELS in rej_ds2
    assert INTERPOLATE_WINDOWS not in rej_ds2


@requires_mne_sample_data
def test_channel_model_rejection_variable_length(samples_experiment):
    "ChannelModelRejection on long, variable-length epochs -> time-windowed interpolation"
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment
    from eelbrain._info import INTERPOLATE_WINDOWS, INTERPOLATE_WINDOWS_MAX
    from eelbrain._meeg import BadChannelWindow

    root = samples_experiment(1, 1, pick='')  # keep EEG channels

    class Experiment(SampleExperiment):
        epochs = {
            **SampleExperiment.epochs,
            'varlen': PrimaryEpoch('sample', "event == 'target'", tmin=-0.1, tmax='0.2 + 0.1*(index % 2)'),
        }
        epoch_rejection = {'auto': ChannelModelRejection(model='ridge', fit_threshold=None, score_threshold=1e-5, max_interpolate=2)}

    e = Experiment(root)
    e.set(subject='R0000', epoch='varlen', raw='raw', epoch_rejection='auto')

    # the rejection file stores per-epoch BadChannelWindow lists
    ctx = e._resolve_derivative('epoch-rejection-channel-model')
    rej_ds = ctx.load()
    assert INTERPOLATE_WINDOWS in rej_ds
    windows = rej_ds[INTERPOLATE_WINDOWS]
    assert all(isinstance(w, BadChannelWindow) for epoch_windows in windows for w in epoch_windows)
    # nothing is rejected wholesale for long epochs
    assert rej_ds['accept'].x.all()
    n_windows = sum(len(epoch_windows) for epoch_windows in windows)
    assert n_windows > 0  # score_threshold low enough to flag something

    # end-to-end: interpolation runs and only touches samples inside the windows
    max_interpolate = rej_ds.info[INTERPOLATE_WINDOWS_MAX]
    ds0 = e.load_epochs(interpolate_bads=True, baseline=False, epoch_rejection='')
    ds1 = e.load_epochs(interpolate_bads=True, baseline=False, epoch_rejection='auto')
    assert isinstance(ds1['eeg'], Datalist)
    assert len(ds1['eeg']) == len(windows)
    changed = zeroed_any = False
    for i, (y0, y1, epoch_windows) in enumerate(zip(ds0['eeg'], ds1['eeg'], windows)):
        bad_by_channel = {}
        for w in epoch_windows:
            bad_by_channel.setdefault(w.channel, []).append((w.tmin, w.tmax))
        # intervals where more than max_interpolate channels are bad are zeroed
        # across all channels (too few good channels for reliable interpolation)
        n_bad = np.zeros(y0.time.nsamples, int)
        for spans in bad_by_channel.values():
            in_channel = np.zeros(y0.time.nsamples, bool)
            for tmin, tmax in spans:
                in_channel |= (y0.time.times >= tmin) & (y0.time.times < tmax)
            n_bad += in_channel
        zeroed = n_bad > max_interpolate
        if zeroed.any():
            zeroed_any = True
            assert_array_equal(y1.x[:, zeroed], 0.)
        for ci, ch in enumerate(y0.sensor.names):
            spans = bad_by_channel.get(ch, [])
            inside = np.zeros(y0.time.nsamples, bool)
            for tmin, tmax in spans:
                inside |= (y0.time.times >= tmin) & (y0.time.times < tmax)
            # samples outside any bad window (and outside zeroed intervals) are unchanged
            unchanged = ~inside & ~zeroed
            assert_array_equal(y0.x[ci, unchanged], y1.x[ci, unchanged])
            # flagged samples are modified, whether interpolated or zeroed
            if inside.any() and not np.array_equal(y0.x[ci, inside], y1.x[ci, inside]):
                changed = True
    assert changed  # flagged windows were actually modified
    assert zeroed_any  # some interval had more than max_interpolate bad channels


@requires_mne_sample_data
def test_evoked_backed_test_vars_are_post_aggregation_only(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    class Experiment(SampleExperiment):
        tests = {
            **SampleExperiment.tests,
            'anova-ok': ANOVA('modality * modality_num * subject', model='modality', vars={'modality_num': LabelVar('modality', {'auditory': 0, 'visual': 1})}),
            'anova-bad': ANOVA('modality_num * subject', vars={'modality_num': LabelVar('modality', {'auditory': 0, 'visual': 1})}),
        }

    root = samples_experiment(n_subjects=3, n_segments=2, mris=False)
    e = Experiment(root, epoch_rejection='')

    options = {
        'data': DataSpec.coerce('meg.mean'),
        'test': 'anova-ok',
        'baseline': False,
        'src_baseline': None,
        'smooth': None,
        'samplingrate': None,
    }
    ds = e._resolve_derivative('evoked-test-data', options=options).load()
    assert 'modality_num' in ds

    with pytest.raises(ConfigurationError, match='For evoked tests'):
        e._resolve_derivative('evoked-test-data', options={**options, 'test': 'anova-bad'}).load()


@requires_mne_sample_data
def test_raw_bad_channel_derivatives_follow_pipe_graph(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(1, 2, 1)

    class Experiment(SampleExperiment):
        raw = {
            'ica': RawICA('raw', ('sample1', 'sample2'), 'fastica', max_iter=1),
            'apply-ica': RawApplyICA('raw', 'ica'),
            **SampleExperiment.raw,
        }

    e = Experiment(root)

    e.set(subject='R0000', raw='raw', task='sample1')
    e.make_bad_channels('MEG 0111', redo=True)
    e.set(task='sample2')
    e.make_bad_channels('MEG 0121', redo=True)

    assert e.load_bad_channels(raw='raw', task='sample1') == ['MEG 0111']
    assert e.load_bad_channels(raw='ica', task='sample1') == ['MEG 0111', 'MEG 0121']
    assert e.load_bad_channels(raw='ica', task='sample2') == ['MEG 0111', 'MEG 0121']
    assert e.load_bad_channels(raw='apply-ica', task='sample1') == ['MEG 0111', 'MEG 0121']


@requires_mne_sample_data
def test_raw_reader_warnings_are_summarized(monkeypatch, samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=1, mris=False)
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
    assert not any('issued during raw-input@raw' in str(w.message) for w in record)

    details_path = e.root / LOG_DIR / 'raw-input@raw-warnings.toml'
    assert details_path.exists()
    text = details_path.read_text()
    assert 'Synthetic raw reader warning 1' in text
    assert 'Synthetic raw reader warning 2' in text
    data = tomllib.loads(text)
    assert len(data['warning']) == 2

    log_path = Path(next(handler.baseFilename for handler in e._log.handlers if isinstance(handler, logging.FileHandler)))
    log_text = log_path.read_text()
    assert str(details_path) in log_text
    assert log_text.count('issued during raw-input@raw') == 1

    e.load_raw(raw='raw')
    assert details_path.read_text() == text
    assert log_path.read_text().count('issued during raw-input@raw') == 1


@requires_mne_sample_data
def test_evoked_cache_reuse(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(2, 2, 1)
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', epoch_rejection='')

    _ = e.load_evoked(ndvar=False)
    handle = e._resolve_derivative('evoked')
    evoked_path = handle.artifact_path
    manifest_path = handle.manifest_path
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert 'view' not in manifest['dependencies']['epochs']
    mtimes_1 = (evoked_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)

    _ = e.load_evoked(ndvar=False)
    mtimes_2 = (evoked_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns)

    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_evoked_cached_load_bypasses_epochs(monkeypatch, samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(2, 2, 1)
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', epoch_rejection='')

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
def test_evoked_cached_load_applies_cat_without_rebuilding_epochs(monkeypatch, samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(2, 2, 1)
    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target1', epoch_rejection='')

    target = e.load_evoked(ndvar=False, cat='auditory', model='modality')
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

    ds = e.load_evoked(ndvar=False, cat='auditory', model='modality')
    assert ds.n_cases == 1
    assert_dataobj_equal(ds, target, decimal=19)
    assert calls == 1


@requires_mne_sample_data
def test_evoked_cache_ignores_irrelevant_selected_events_changes(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    root = samples_experiment(1, 2, 1)
    e = SampleExperiment(root)

    e.set(subject='R0000', epoch='target1', epoch_rejection='')
    assert not e._resolve_derivative('evoked', options={'model': 'modality'}).is_valid()
    e.load_evoked(ndvar=False, model='modality')
    assert e._resolve_derivative('evoked', options={'model': 'modality'}).is_valid()

    assert not e._resolve_derivative('evoked', options={'model': 'side'}).is_valid()
    e.load_evoked(ndvar=False, model='side')
    assert e._resolve_derivative('evoked', options={'model': 'side'}).is_valid()

    class SampleExperimentModified(SampleExperiment):

        variables = {
            **SampleExperiment.variables,
            'side': LabelVar('value', {(1, 3): 'left_', (2, 4): 'right_'}),
        }

    e = SampleExperimentModified(root)
    e.set(subject='R0000', epoch='target1', epoch_rejection='')
    assert e._resolve_derivative('evoked', options={'model': 'modality'}).is_valid()

    assert not e._resolve_derivative('evoked', options={'model': 'side'}).is_valid()


@requires_mne_sample_data
def test_evoked_cache_stales_on_model_change(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='')
    _ = e.load_evoked(ndvar=False, model='modality')

    class ChangedExperiment(SampleExperiment):
        variables = {
            **SampleExperiment.variables,
            'modality': LabelVar('value', {(1, 2): 'auditory_changed', (3, 4): 'visual'}),
        }

    e_changed = ChangedExperiment(root)
    e_changed.set(subject='R0000', epoch='target', epoch_rejection='')
    handle = e_changed._resolve_derivative('evoked', options={'model': 'modality'})

    assert not handle.is_valid()
    ds = e_changed.load_evoked(ndvar=False, model='modality')
    assert set(ds['modality'].cells) == {'auditory_changed', 'visual'}


@requires_mne_sample_data
def test_epochs_dependency_distinguishes_model_sensitivity(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    class CachedEpochsExperiment(SampleExperiment):
        cache_epochs = True

    e = CachedEpochsExperiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='')
    evoked_handle = e._resolve_derivative('evoked', options={'model': 'modality'})
    epochs_dep = next(dep for dep in evoked_handle.node.dependencies(evoked_handle) if dep.name == 'epochs')
    epochs_handle = e._resolve_derivative('epochs', options=epochs_dep.options)

    # Current model labels should not affect the epochs dependency because
    # epoch extraction only needs event timing and rejection-related metadata.
    epochs_dependency = epochs_handle.describe_dependency()

    # Build evoked once. Unlike epochs, evoked depends on the labels of the
    # current model because it stores one averaged response per model cell.
    assert not evoked_handle.is_valid()
    ds = e.load_evoked(ndvar=False, model='modality')
    assert set(ds['modality'].cells) == {'auditory', 'visual'}
    assert evoked_handle.is_valid()

    class ChangedExperiment(CachedEpochsExperiment):
        variables = {
            **SampleExperiment.variables,
            'modality': LabelVar('value', {(1, 2): 'auditory_changed', (3, 4): 'visual'}),
        }

    e_changed = ChangedExperiment(root)
    e_changed.set(subject='R0000', epoch='target', epoch_rejection='')
    evoked_handle_changed = e_changed._resolve_derivative('evoked', options={'model': 'modality'})
    epochs_handle_changed = e_changed._resolve_derivative('epochs', options=epochs_dep.options)

    assert epochs_handle_changed.describe_dependency() == epochs_dependency
    # The evoked artifact aggregates by model cells, so the same change should
    # invalidate evoked and rebuild it with the current labels.
    assert not evoked_handle_changed.is_valid()
    ds_changed = e_changed.load_evoked(ndvar=False, model='modality')
    assert set(ds_changed['modality'].cells) == {'auditory_changed', 'visual'}
    assert evoked_handle_changed.is_valid()


@requires_mne_sample_data
def test_recording_epochs_cache_uses_fif(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    class CachedEpochsExperiment(SampleExperiment):
        cache_epochs = True

    root = samples_experiment(1, 2, 1)
    e = CachedEpochsExperiment(root)
    e.set(subject='R0000', epoch='target1', epoch_rejection='')

    options = {
        'baseline': False,
        'reject': True,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'ndvar': False,
        'data': 'sensor',
    }
    epochs_handle = e._resolve_derivative('epochs', options=options)
    assert not epochs_handle.is_valid()
    dep = next(dep for dep in epochs_handle.node.dependencies(epochs_handle) if dep.name == 'recording-epochs')
    handle = e._derivatives.resolve(dep.name, state={**e.state, **dep.state}, options=dep.options)
    epochs = handle.load()

    assert isinstance(epochs, mne.BaseEpochs)
    assert handle.artifact_path.is_dir()
    assert list(handle.artifact_path.glob('*-epo.fif'))
    manifest = json.loads(handle.manifest_path.read_text())
    assert manifest['artifact_metadata']['kind'] == 'single'
    assert manifest['artifact_metadata']['file'] == 'epochs-0000-epo.fif'
    assert set(manifest['dependencies']) == {'raw', 'selected-events'}

    mtimes_1 = tuple(path.stat().st_mtime_ns for path in sorted(handle.artifact_path.iterdir()))
    epochs_cached = handle.load()
    mtimes_2 = tuple(path.stat().st_mtime_ns for path in sorted(handle.artifact_path.iterdir()))

    assert isinstance(epochs_cached, mne.BaseEpochs)
    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_epochs_with_cached_recording_use_current_selected_events(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    class CachedEpochsExperiment(SampleExperiment):
        cache_epochs = True

    root = samples_experiment(1, 2, 1)
    e = CachedEpochsExperiment(root)
    e.set(subject='R0000', epoch='target1', epoch_rejection='')

    options = {
        'baseline': False,
        'reject': True,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'ndvar': False,
        'data': 'sensor',
    }
    handle = e._resolve_derivative('epochs', options=options)
    dep = next(dep for dep in handle.node.dependencies(handle) if dep.name == 'recording-epochs')
    recording_handle = e._derivatives.resolve(dep.name, state={**e.state, **dep.state}, options=dep.options)

    # Compute epochs once to create the recording-level FIF artifact.
    ds = handle.load()
    assert isinstance(ds['epochs'], mne.BaseEpochs)
    assert 'marker' not in ds
    mtimes_1 = tuple(path.stat().st_mtime_ns for path in sorted(recording_handle.artifact_path.iterdir()))

    # Change selected-events in a way that affects the returned event shell but
    # not the epochs artifact stored on disk.
    class ChangedExperiment(CachedEpochsExperiment):
        variables = {
            **SampleExperiment.variables,
            'marker': LabelVar('value', {(1, 2): 'early', (3, 4): 'late'}),
        }

    # Loading epochs should reuse the cached FIF artifact while applying the
    # current selected-events shell to the returned Dataset.
    e_changed = ChangedExperiment(root)
    e_changed.set(subject='R0000', epoch='target1', epoch_rejection='')
    handle_changed = e_changed._resolve_derivative('epochs', options=options)
    dep_changed = next(dep for dep in handle_changed.node.dependencies(handle_changed) if dep.name == 'recording-epochs')
    recording_handle_changed = e_changed._derivatives.resolve(dep_changed.name, state={**e_changed.state, **dep_changed.state}, options=dep_changed.options)
    assert recording_handle_changed.artifact_path == recording_handle.artifact_path
    ds_cached = handle_changed.load()
    mtimes_2 = tuple(path.stat().st_mtime_ns for path in sorted(recording_handle.artifact_path.iterdir()))

    assert isinstance(ds_cached['epochs'], mne.BaseEpochs)
    assert 'marker' in ds_cached
    assert mtimes_1 == mtimes_2


@requires_mne_sample_data
def test_selected_events_manifest_uses_real_dependencies(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='')
    handle = e._resolve_derivative('epoch-events', options={
        'reject': True,
    })
    dependencies = handle.dependency_fingerprints()
    assert 'dependencies' not in handle.current_fingerprint()
    assert set(dependencies) == {'selected-events'}
    rec_events_deps = dependencies['selected-events']['dependencies']
    assert set(rec_events_deps) == {'labeled-events'}
    assert set(rec_events_deps['labeled-events']['dependencies']) == {'events-input', 'events'}

    # Secondary epochs with no additional selection should still inherit the
    # primary epoch's event selection when extracting epochs.
    target_events = e.load_selected_events(epoch='target')
    cov_events = e.load_selected_events(epoch='cov')
    assert cov_events.n_cases == target_events.n_cases
    cov_epochs = e.load_epochs(epoch='cov', ndvar=False)
    assert cov_epochs.n_cases == cov_events.n_cases


@requires_mne_sample_data
def test_labeled_events_sidecar_copies_raw_info_from_raw(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='')
    raw = e.load_raw()

    # labeled-events always depends on both events-input and events (trigger-based)
    labeled_handle = e._resolve_derivative('labeled-events')
    dependencies = labeled_handle.dependency_fingerprints()
    labeled_events = labeled_handle.load()

    assert set(dependencies) == {'events-input', 'events'}
    # Raw timing info is copied from trigger events (which read the raw file),
    # so that the epoch boundary check in _prepare_selected_events works correctly.
    assert labeled_events.info['raw.samplingrate'] == raw.info['sfreq']
    assert labeled_events.info['raw.first_samp'] == raw.first_samp
    assert labeled_events.info['raw.last_samp'] == raw.last_samp


@requires_mne_sample_data
def test_raw_cache_identity_ignores_view_options(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    e = SampleExperiment(root)
    e.set(subject='R0000')
    node_name = raw_node_name('1-40')

    handle_default = e._resolve_derivative(node_name, options={'noise': False, 'preload': False})
    handle_view = e._resolve_derivative(node_name, options={'noise': False, 'preload': True})
    handle_noise = e._resolve_derivative(node_name, options={'noise': True, 'preload': False})

    # View options (preload) must not affect cache identity; artifact options (noise) must.
    assert handle_default.key() == handle_view.key()
    assert handle_default.key() != handle_noise.key()


@requires_mne_sample_data
def test_raw_info_view_matches_source_and_processed_raws(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    e = SampleExperiment(root)
    e.set(subject='R0000')
    e.make_bad_channels('MEG 0111', redo=True)

    source_info = e._resolve_derivative(raw_node_name('raw'), options={'noise': False, 'preload': False}).load(view='info')
    source_raw = e.load_raw(raw='raw')
    assert source_info['sfreq'] == source_raw.info['sfreq']
    assert source_info['bads'] == source_raw.info['bads'] == ['MEG 0111']

    processed_info = e._resolve_derivative(raw_node_name('1-40'), options={'noise': False, 'preload': False}).load(view='info')
    processed_raw = e.load_raw(raw='1-40')
    assert processed_info['bads'] == processed_raw.info['bads'] == ['MEG 0111']
    assert processed_info['highpass'] == pytest.approx(processed_raw.info['highpass'])
    assert processed_info['lowpass'] == pytest.approx(processed_raw.info['lowpass'])


@requires_mne_sample_data
def test_raw_filter_elliptic_info_view_matches_artifact(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    class Experiment(SampleExperiment):
        raw = {
            **SampleExperiment.raw,
            'ellip': RawFilterElliptic('raw', None, None, 40, 45, 1, 20),
        }

    e = Experiment(root)
    e.set(subject='R0000')

    info = e._resolve_derivative(raw_node_name('ellip'), options={'noise': False, 'preload': False}).load(view='info')
    raw = e.load_raw(raw='ellip')

    assert info['sfreq'] == raw.info['sfreq']
    assert info['highpass'] == pytest.approx(raw.info['highpass'])
    assert info['lowpass'] == pytest.approx(raw.info['lowpass'])


@requires_mne_sample_data
def test_selected_events_vardef_is_local(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, n_segments=2, mris=False)

    e = SampleExperiment(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='')
    options = {
        'reject': True,
    }
    compact = Variables({'grouped': LabelVar('value', {(1, 2): 'target'}, task='sample')})
    changed = Variables({'grouped': LabelVar('value', {1: 'target', 2: 'nontarget'}, task='sample')})

    handle = e._resolve_derivative('epoch-events', options=options)
    _ = handle.load()

    assert 'vardef' not in handle.current_fingerprint()
    with pytest.raises(TypeError, match="undeclared option"):
        e._resolve_derivative('epoch-events', options={**options, 'vardef': compact})

    ds_compact = e.load_selected_events(vardef=compact)
    ds_changed = e.load_selected_events(vardef=changed)
    assert set(ds_compact['grouped'].cells) == {'', 'target'}
    assert 'nontarget' in ds_changed['grouped'].cells


@requires_mne_sample_data
def test_sample_neuromag(samples_experiment):
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    root = samples_experiment(n_subjects=1, pick='')

    class Experiment(SampleExperiment):
        defaults = {'raw': '1-40', 'epoch_rejection': 'manual'}

    e = Experiment(root)
    e.set(raw='1-40', epoch='target', epoch_rejection='')

    # Check original events
    ds = e.load_events()
    assert ds.n_cases == 80
    ds = e.load_selected_events()
    assert ds.n_cases == 73

    # Check auto-rejection
    e.set(epoch_rejection='manual')
    e.make_epoch_rejection(auto={'mag': 2e-12, 'grad': 5e-11, 'eeg': 1.5e-4})
    ds = e.load_selected_events(reject='keep')
    assert ds['accept'].sum() == 69


@requires_mne_sample_data
def test_epoch_run(samples_experiment):
    """Test run aggregation for PrimaryEpoch and ContinuousEpoch."""
    set_log_level('warning', 'mne')

    root = samples_experiment(n_subjects=2, n_segments=2, n_runs=2)

    class MultiRunExperiment(Pipeline):
        stim_channel = 'STI 014'
        merge_triggers = -1
        variables = {
            'event': LabelVar('value', {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'}),
        }
        # combine-all epoch: run=None → load events from all runs
        # explicit-run epochs: run='1'/'2' → load events only from that run
        epochs = {
            'target': PrimaryEpoch('sample', "event == 'target'", tmax=0.3, decim=5),
            'target-copy': SecondaryEpoch('target'),
            'target-r1': PrimaryEpoch('sample', "event == 'target'", tmax=0.3, decim=5, run='1'),
            'target-r2': PrimaryEpoch('sample', "event == 'target'", tmax=0.3, decim=5, run='2'),
            'cont': ContinuousEpoch('sample', "event == 'target'", pad_start=0.1, pad_end=0.1, split=0.5),
            'cont-r1': ContinuousEpoch('sample', "event == 'target'", pad_start=0.1, pad_end=0.1, split=0.5, run='1'),
            'cont-r2': ContinuousEpoch('sample', "event == 'target'", pad_start=0.1, pad_end=0.1, split=0.5, run='2'),
        }

    e = MultiRunExperiment(root)

    # With explicit-run epoch, run state should be forced to the epoch's run
    e.set(epoch='target-r1')
    assert e.get('run') == '1', "explicit-run epoch should set run='1' in state"

    # Combine-all: events from both runs combined; explicit-run loads only that run
    e.set(epoch='target', epoch_rejection='')
    ds_all = e.load_selected_events()
    assert ds_all.n_cases > 0

    e.set(epoch='target-r1', epoch_rejection='')
    ds_r1 = e.load_selected_events()
    assert ds_r1.n_cases > 0

    e.set(epoch='target-r2', epoch_rejection='')
    ds_r2 = e.load_selected_events()
    assert ds_r2.n_cases > 0

    # Combined events = run-1 + run-2
    assert ds_all.n_cases == ds_r1.n_cases + ds_r2.n_cases, f"combine-all ({ds_all.n_cases}) != run-1 ({ds_r1.n_cases}) + run-2 ({ds_r2.n_cases})"

    # Epochs can be loaded from combine-all epoch
    e.set(epoch='target')
    ds_epochs = e.load_epochs()
    assert 'mag' in ds_epochs
    assert ds_epochs.n_cases == ds_all.n_cases

    # A secondary epoch based on a combine-all primary should load epochs from
    # every run, not just the current run.
    e.set(epoch='target-copy', epoch_rejection='')
    ds_secondary = e.load_selected_events()
    assert ds_secondary.n_cases == ds_all.n_cases
    ds_secondary_epochs = e.load_epochs()
    assert ds_secondary_epochs.n_cases == ds_secondary.n_cases

    # Continuous epochs are prepared independently in each recording, then
    # combined as segments carrying their run identity.
    e.set(epoch='cont-r1', epoch_rejection='')
    assert e.get('run') == '1'
    cont_r1 = e.load_selected_events()
    e.set(epoch='cont-r2')
    assert e.get('run') == '2'
    cont_r2 = e.load_selected_events()
    e.set(epoch='cont')
    cont_all = e.load_selected_events()
    assert cont_all.n_cases == cont_r1.n_cases + cont_r2.n_cases
    assert cont_all['run'].cells == ('1', '2')
    for entity in BIDS_ENTITY_KEYS:
        if entity in cont_all and len(set(cont_all[entity])) > 1:
            assert entity not in cont_all.info
        else:
            assert entity in cont_all.info
    for run in cont_all['run'].cells:
        run_ds = cont_all.sub(cont_all['run'] == run)
        assert run_ds[0, 'epoch_time'] == pytest.approx(0.0)
        assert run_ds[0, 'events'][0, 'epoch_time'] == pytest.approx(0.0)

    cont_epochs = e.load_epochs()
    assert isinstance(cont_epochs['mag'], Datalist)
    assert cont_epochs.n_cases == cont_all.n_cases
    assert all(events[0, 'epoch_time'] == pytest.approx(y.time.tmin + 0.1, abs=0.002) for events, y in cont_epochs.zip('events', 'mag'))


@requires_mne_sample_data
def test_sample_eeg(samples_experiment):
    set_log_level('warning', 'mne')

    root = samples_experiment(2, 1, 1, pick='eeg')

    class Experiment(Pipeline):

        raw = {
            'av-ref': RawReReference('raw'),
        }

    e = Experiment(root)

    # average reference
    raw = e.load_raw(raw='av-ref')
    assert raw.info['custom_ref_applied'] == True


@requires_mne_sample_data
def test_load_trf(samples_experiment):
    "load_trf, caching, and the separable TRFJob"
    import pickle
    from eelbrain import BoostingResult
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4)
    e = SampleTRF(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='1-40', inv='')

    # compute
    res = e.load_trf('imp', 0, 0.1)
    assert isinstance(res, BoostingResult)

    # cache hit
    options = e._trf_options('imp', 0., 0.1, 'boosting', None, None, None, False, {})
    assert e._resolve_derivative('trf', options=options).is_valid()

    # path
    path = Path(e.load_trf('imp', 0, 0.1, path_only=True))
    assert path.exists()

    # data-carrying, picklable job reproduces the result
    job = e.load_trf_job('imp', 0, 0.1)
    job = pickle.loads(pickle.dumps(job))
    res2 = job.fit()
    assert isinstance(res2, BoostingResult)

    # external execution re-incorporated into the cache
    spec = e._trf_job_spec('imp', 0, 0.1)
    assert Path(spec.path) == path
    assert spec.is_done
    path.unlink()
    spec.ctx.manifest_path.unlink(missing_ok=True)
    assert not spec.is_done
    result = pickle.loads(pickle.dumps(spec.make_job())).fit()  # "off-host"
    spec.save_result(result)
    assert spec.is_done
    assert path.exists()


@requires_mne_sample_data
def test_predictor_subset_fingerprint(samples_experiment):
    "Editing an unused predictor-file column does not invalidate a cached TRF; editing a used one does"
    import os
    from eelbrain import BoostingResult, Dataset, Var, save
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4)
    e = SampleTRF(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='1-40', inv='')
    samplingrate = 1 / e.load_epochs(reject=False)['mag'].time.tstep

    pdir = Path(root) / 'derivatives' / 'predictors'
    pdir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(root) / 'derivatives' / 'eelbrain' / 'cache' / 'predictor'
    mtime = [1_700_000_000]

    def write(stim, value, unused):
        # a NUTS Dataset predictor with a bool mask and an extra column ('unused') the term ignores
        ds = Dataset({'time': Var([0., .1, .2, .3, .4]), 'value': Var(value), 'mask': Var(np.array([True, True, True, True, False])), 'unused': Var(unused)})
        path = pdir / f'{stim}~word.pickle'
        save.pickle(ds, path)
        mtime[0] += 1  # ensure the quick (mtime) fingerprint changes between writes
        os.utime(path, (mtime[0], mtime[0]))

    def read_reference(stim):
        return json.loads((ref_dir / f'{stim}~word-value-mask.json').read_text())

    ones = [1., 1., 1., 1., 1.]
    for stim in ('auditory', 'visual'):
        write(stim, ones, [0., 0., 0., 0., 0.])

    # bare key = intercept: unit impulse at each time stamp
    x = e.load_predictor('auditory~word', 0.1)
    assert x.sum() == 5.

    res = e.load_trf('word-value-mask', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(res, BoostingResult)
    options = e._trf_options('word-value-mask', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    ctx = e._resolve_derivative('trf', options=options)
    assert ctx.is_valid()

    # dependent manifests store only the small version identity, never the data
    for code in ('auditory~word-value-mask', 'visual~word-value-mask'):
        fingerprint = ctx._manifest().dependencies[code]['fingerprint']
        assert 'data' not in fingerprint
        assert set(fingerprint['version']) == {'uid', 'serial'}
    version_0 = read_reference('auditory')['version']
    assert version_0['serial'] == 0

    # editing only the unused column (new mtime, same relevant data) keeps the TRF valid
    write('auditory', ones, [9., 9., 9., 9., 9.])
    assert e._resolve_derivative('trf', options=options).is_valid()
    reference = read_reference('auditory')
    assert reference['version'] == version_0  # same data → same version
    assert reference['source']['mtime'] == mtime[0]  # refreshed for the fast path

    # editing a used column (value) invalidates the TRF and bumps the serial
    write('auditory', [2., 2., 2., 2., 2.], [9., 9., 9., 9., 9.])
    assert not e._resolve_derivative('trf', options=options).is_valid()
    assert read_reference('auditory')['version'] == {'uid': version_0['uid'], 'serial': 1}

    # a deleted reference is recreated with a new uid → dependents rebuild, never stale-accept
    e.load_trf('word-value-mask', 0, 0.1, samplingrate=samplingrate)
    assert e._resolve_derivative('trf', options=options).is_valid()
    reference = read_reference('auditory')
    (ref_dir / reference['data_file']).unlink()
    (ref_dir / 'auditory~word-value-mask.json').unlink()
    write('auditory', [2., 2., 2., 2., 2.], [9., 9., 9., 9., 9.])  # touch to force a quick-fingerprint mismatch
    assert not e._resolve_derivative('trf', options=options).is_valid()
    version_new = read_reference('auditory')['version']
    assert version_new['serial'] == 0
    assert version_new['uid'] != version_0['uid']


@requires_mne_sample_data
@pytest.mark.slow
def test_load_trf_source(samples_experiment):
    "load_trf in source space"
    from eelbrain import BoostingResult
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4, mris=True)
    e = SampleTRF(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='1-40', src='ico-2', parc='ac')
    res = e.load_trf('imp', 0, 0.1)
    assert isinstance(res, BoostingResult)
    assert e._resolve_derivative('trf', options=e._trf_options('imp', 0., 0.1, 'boosting', None, None, None, False, {})).is_valid()


@requires_mne_sample_data
def test_load_trf_filepredictor(samples_experiment):
    "load_trf with a UTSPredictor: per-stimulus predictor dependency edges"
    from eelbrain import BoostingResult, NDVar, UTS, save
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4)
    e = SampleTRF(root)
    e.set(subject='R0000', epoch='target', epoch_rejection='', raw='1-40', inv='')

    # match the predictor sampling to the data's natural (decimated) rate so the
    # samplingrate is an integer ratio of the raw rate and needs no resampling
    tstep = e.load_epochs(reject=False)['mag'].time.tstep
    samplingrate = 1 / tstep

    # write a predictor file per stimulus (one for each 'modality' cell)
    pdir = Path(root) / 'derivatives' / 'predictors'
    pdir.mkdir(parents=True, exist_ok=True)
    uts = UTS(0, tstep, 60)
    rng = np.random.RandomState(0)
    predictor_ndvars = {stim: NDVar(rng.normal(size=60), uts, name='env') for stim in ('auditory', 'visual')}
    for stim, ndvar in predictor_ndvars.items():
        save.pickle(ndvar, pdir / f'{stim}~env.pickle')

    # load_predictor shapes one stimulus' file into an NDVar at the requested tstep
    x = e.load_predictor('auditory~env', tstep)
    assert isinstance(x, NDVar)
    assert x.time.tstep == tstep
    assert x.name == 'auditory~env'

    # compute
    res = e.load_trf('env', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(res, BoostingResult)

    # the per-stimulus predictor file edges are recorded in the manifest
    options = e._trf_options('env', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    ctx = e._resolve_derivative('trf', options=options)
    assert ctx.is_valid()
    assert {'auditory~env', 'visual~env'} <= set(ctx._manifest().dependencies)

    # the manifest stores only the small version identity, never the data
    for code in ('auditory~env', 'visual~env'):
        fingerprint = ctx._manifest().dependencies[code]['fingerprint']
        assert 'data' not in fingerprint
        assert set(fingerprint['version']) == {'uid', 'serial'}

    # re-saving identical data (new mtime) keeps the TRF valid: the deep
    # comparison against the reference copy absorbs the file-stat drift
    import os
    save.pickle(predictor_ndvars['auditory'], pdir / 'auditory~env.pickle')
    os.utime(pdir / 'auditory~env.pickle', (1_700_000_000, 1_700_000_000))
    assert e._resolve_derivative('trf', options=options).is_valid()

    # editing a predictor file invalidates the cached TRF
    save.pickle(NDVar(rng.normal(size=60), uts, name='env'), pdir / 'auditory~env.pickle')
    assert not e._resolve_derivative('trf', options=options).is_valid()


def _subject_predictor_path(root, subject, desc, task='sample', run='', per_event=False):
    entities = f'sub-{subject}'
    if not per_event:
        entities += f'_task-{task}'
        if run:
            entities += f'_run-{run}'
    path = Path(root) / 'derivatives' / 'subject-predictors' / f'sub-{subject}' / f'{entities}_desc-{desc}.pickle'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@requires_mne_sample_data
def test_load_trf_subject_predictor(samples_experiment):
    "load_trf with a SubjectUTSPredictor (sequence mode): one recording-long file per recording"
    from eelbrain import BoostingResult, NDVar, UTS, save
    from eelbrain._experiment.trf.model import TRFModelError
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=2, n_segments=4)
    e = SampleTRF(root)
    e.set(epoch='target', epoch_rejection='', raw='1-40', inv='')

    subjects = ('R0000', 'R0001')
    rng = np.random.RandomState(0)
    e.set(subject='R0000')
    ds = e.load_epochs(reject=False)
    tstep = ds['mag'].time.tstep
    samplingrate = 1 / tstep
    # the file must span the whole recording, since each trial is cut out at its
    # own recording time (sequence mode); size it from the trial sample range
    sfreq = ds.info['raw.samplingrate']
    span = (ds['sample'].max() - ds['sample'].min()) / sfreq + 2.
    n_samples = int(span / tstep)
    uts = UTS(-0.5, tstep, n_samples)

    # a different recording-long predictor per subject (no session/acquisition
    # in this dataset, so no ses-/acq- entities in the path)
    predictor_ndvars = {s: NDVar(rng.normal(size=n_samples), uts, name='envseq') for s in subjects}
    for subject, ndvar in predictor_ndvars.items():
        assert 'sampling' not in ndvar.info
        save.pickle(ndvar, _subject_predictor_path(root, subject, 'envseq'))

    # load_predictor returns each subject's own file (it bypasses the assembly path)
    for subject in subjects:
        x = e.load_predictor('envseq', tstep, subject=subject)
        assert isinstance(x, NDVar)
        assert x.name == 'envseq'
        assert_array_equal(x.x, predictor_ndvars[subject].x)

    # a term with a stimulus is rejected in sequence mode
    with pytest.raises(TRFModelError):
        e.load_predictor('auditory~envseq', tstep, subject='R0000')

    # filter_x='continuous' determines whether to filter from metadata supplied
    # by the predictor configuration, not from the raw predictor file
    job = e.load_trf_job('envseq', 0, 0.1, samplingrate=samplingrate, filter_x='continuous', subject='R0000')
    assert job.xs[0].info['sampling'] == 'continuous'

    # compute per subject; the (stimulus-free) predictor edge is in the manifest
    options = e._trf_options('envseq', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    for subject in subjects:
        e.set(subject=subject)
        res = e.load_trf('envseq', 0, 0.1, samplingrate=samplingrate)
        assert isinstance(res, BoostingResult)
        ctx = e._resolve_derivative('trf', options=options)
        assert ctx.is_valid()
        assert 'envseq@task-sample' in set(ctx._manifest().dependencies)

    # editing R0000's file invalidates only R0000's TRF; R0001 stays valid
    save.pickle(NDVar(rng.normal(size=n_samples), uts, name='envseq'), _subject_predictor_path(root, 'R0000', 'envseq'))
    e.set(subject='R0000')
    assert not e._resolve_derivative('trf', options=options).is_valid()
    e.set(subject='R0001')
    assert e._resolve_derivative('trf', options=options).is_valid()


@requires_mne_sample_data
def test_load_trf_continuous_predictor(samples_experiment):
    "Continuous (ContinuousEpoch) predictors: per-event UTSPredictor and per-event SubjectUTSPredictor"
    from eelbrain import BoostingResult, NDVar, UTS, save
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4)
    e = SampleTRF(root)
    e.set(subject='R0000', epoch='cont', epoch_rejection='', raw='1-40', inv='')

    # ContinuousEpoch: one case per continuous segment, response is a Datalist
    ds = e.load_epochs(reject=False, decim=5, keep_mne=True)
    assert ds.info['nested_events'] == 'events'
    assert ds.n_cases > 1
    assert isinstance(ds['epochs'], Datalist)
    assert isinstance(ds['mag'], Datalist)
    assert ds[0, 'epoch_time'] == pytest.approx(0.0)
    assert ds[1, 'epoch_time'] > 0
    epoch_tmin = ds['epochs'][0].times[0]
    for epoch_time, events, epochs_i, y_i in ds.zip('epoch_time', 'events', 'epochs', 'mag'):
        assert events[0, 'epoch_time'] == pytest.approx(epoch_time)
        assert epochs_i.times[0] == pytest.approx(epoch_time + epoch_tmin)
        assert y_i.time.tmin == pytest.approx(epochs_i.times[0])
    tstep = ds['mag'][0].time.tstep
    samplingrate = 1 / tstep
    stimuli = ('auditory', 'visual')  # the 'modality' stim_var cells

    rng = np.random.RandomState(0)
    # keep the per-event predictor shorter than the epoch's pad_end (0.1 s), so
    # the last event's predictor fits inside its segment
    stim_uts = UTS(0, tstep, 10)

    # flat per-stimulus files for a UTSPredictor, placed at each event's time
    pdir = Path(root) / 'derivatives' / 'predictors'
    pdir.mkdir(parents=True, exist_ok=True)
    for stim in stimuli:
        save.pickle(NDVar(rng.normal(size=stim_uts.nsamples), stim_uts, name='env'), pdir / f'{stim}~env.pickle')

    res = e.load_trf('env', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(res, BoostingResult)
    job = e.load_trf_job('env', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(job.y, Datalist)
    assert isinstance(job.xs[0], Datalist)
    assert all(x_i.time == y_i.time for x_i, y_i in zip(job.xs[0], job.y))
    # one predictor-file edge per stimulus, enumerated from the nested events
    options = e._trf_options('env', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    deps = set(e._resolve_derivative('trf', options=options)._manifest().dependencies)
    assert {'auditory~env', 'visual~env'} <= deps

    # per-event SubjectUTSPredictor: one subject-specific file per stimulus
    for stim in stimuli:
        save.pickle(NDVar(rng.normal(size=stim_uts.nsamples), stim_uts, name='envp'), _subject_predictor_path(root, 'R0000', f'{stim}~envp', per_event=True))
    res = e.load_trf('envp', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(res, BoostingResult)
    job = e.load_trf_job('envp', 0, 0.1, samplingrate=samplingrate)
    assert all(x_i.time == y_i.time for x_i, y_i in zip(job.xs[0], job.y))
    options = e._trf_options('envp', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    deps = set(e._resolve_derivative('trf', options=options)._manifest().dependencies)
    assert {'auditory~envp', 'visual~envp'} <= deps

    # sequence-mode SubjectUTSPredictor on the same ContinuousEpoch: one
    # recording-long file, cut into per-segment chunks by recording time
    sfreq = ds.info['raw.samplingrate']
    span = (ds['sample'].max() - ds['sample'].min()) / sfreq + 6.
    n_samples = int(span / tstep)
    save.pickle(NDVar(rng.normal(size=n_samples), UTS(-0.5, tstep, n_samples), name='envseq'), _subject_predictor_path(root, 'R0000', 'envseq'))
    res = e.load_trf('envseq', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(res, BoostingResult)
    job = e.load_trf_job('envseq', 0, 0.1, samplingrate=samplingrate, filter_x='continuous')
    assert all(x_i.info['sampling'] == 'continuous' for x_i in job.xs[0])
    assert all(x_i.time == y_i.time for x_i, y_i in zip(job.xs[0], job.y))


@requires_mne_sample_data
def test_load_trf_continuous_predictor_multiple_runs(samples_experiment):
    "Pooled ContinuousEpoch TRFs load each run's SubjectUTSPredictor files"
    from eelbrain import BoostingResult, NDVar, UTS, save
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4, n_runs=2)
    e = SampleTRF(root)
    e.set(subject='R0000', epoch='cont', epoch_rejection='', raw='1-40', inv='')

    ds = e.load_epochs(reject=False, decim=5)
    assert ds['run'].cells == ('1', '2')
    assert isinstance(ds['mag'], Datalist)
    tstep = ds['mag'][0].time.tstep
    samplingrate = 1 / tstep
    rng = np.random.RandomState(1)

    # Sequence predictors use a separate recording-long file for each run.
    tmax = max(y.time.tmax for y in ds['mag']) + 1
    sequence_uts = UTS(-0.5, tstep, int(np.ceil((tmax + 0.5) / tstep)))
    sequences = {}
    for run in ds['run'].cells:
        sequence = NDVar(rng.normal(size=sequence_uts.nsamples), sequence_uts, name='envseq')
        sequences[run] = sequence
        save.pickle(sequence, _subject_predictor_path(root, 'R0000', 'envseq', run=run))

    sequence_job = e.load_trf_job('envseq', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(sequence_job.y, Datalist)
    assert isinstance(sequence_job.xs[0], Datalist)
    for run, x_i, y_i in zip(ds['run'], sequence_job.xs[0], sequence_job.y):
        sequence = sequences[run]
        i_start = sequence.time._array_index(y_i.time.tmin)
        assert_array_equal(x_i.x, sequence.x[i_start:i_start + y_i.time.nsamples])
        assert x_i.time == y_i.time

    sequence_result = e.load_trf('envseq', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(sequence_result, BoostingResult)
    sequence_options = e._trf_options('envseq', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    sequence_ctx = e._resolve_derivative('trf', options=sequence_options)
    sequence_deps = set(sequence_ctx._manifest().dependencies)
    assert {'envseq@task-sample_run-1', 'envseq@task-sample_run-2'} <= sequence_deps

    # Per-event predictors are subject/stimulus-specific and shared across runs.
    stimuli = ('auditory', 'visual')
    stimulus_uts = UTS(0, tstep, 10)
    per_event = {}
    for stim in stimuli:
        predictor = NDVar(rng.normal(size=stimulus_uts.nsamples), stimulus_uts, name='envp')
        per_event[stim] = predictor
        save.pickle(predictor, _subject_predictor_path(root, 'R0000', f'{stim}~envp', per_event=True))

    per_event_job = e.load_trf_job('envp', 0, 0.1, samplingrate=samplingrate)
    for events, x_i, y_i in zip(ds['events'], per_event_job.xs[0], per_event_job.y):
        stim = events[0, 'modality']
        predictor = per_event[stim]
        i_start = x_i.time._array_index(events[0, 'epoch_time'])
        assert_array_equal(x_i.x[i_start:i_start + predictor.time.nsamples], predictor.x)
        assert x_i.time == y_i.time

    per_event_result = e.load_trf('envp', 0, 0.1, samplingrate=samplingrate)
    assert isinstance(per_event_result, BoostingResult)
    per_event_options = e._trf_options('envp', 0., 0.1, 'boosting', None, None, samplingrate, False, {})
    per_event_deps = set(e._resolve_derivative('trf', options=per_event_options)._manifest().dependencies)
    assert {'auditory~envp', 'visual~envp'} <= per_event_deps

    # Fixed-length pooled epochs likewise compute sequence offsets separately
    # within each run rather than from one global sample origin.
    e.set(epoch='target')
    target_ds = e.load_epochs(reject=False)
    target_job = e.load_trf_job('envseq', 0, 0.1, samplingrate=samplingrate)
    sample_0 = {run: target_ds[target_ds['run'] == run, 'sample'][0] for run in target_ds['run'].cells}
    sfreq = target_ds.info['raw.samplingrate']
    for i, (run, sample_i) in enumerate(target_ds.zip('run', 'sample')):
        sequence = sequences[run]
        offset = (sample_i - sample_0[run]) / sfreq
        offset = tstep * round(offset / tstep)
        i_start = sequence.time._array_index(offset + target_job.y.time.tmin)
        assert_array_equal(target_job.xs[0].x[i], sequence.x[i_start:i_start + target_job.y.time.nsamples])

    # Any constituent recording changing invalidates the pooled ContinuousEpoch fit.
    e.set(epoch='cont')
    save.pickle(NDVar(rng.normal(size=sequence_uts.nsamples), sequence_uts, name='envseq'), _subject_predictor_path(root, 'R0000', 'envseq', run='1'))
    assert not e._resolve_derivative('trf', options=sequence_options).is_valid()


@requires_mne_sample_data
def test_load_trfs(samples_experiment):
    "load_trfs: per-subject and group assembly in sensor space"
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=2, n_segments=4)
    e = SampleTRF(root)
    e.set(epoch='target', epoch_rejection='', raw='1-40', inv='')

    # single subject -> 1-case Dataset with metrics and kernel
    ds = e.load_trfs('R0000', 'imp', 0, 0.1)
    assert isinstance(ds, Dataset)
    assert ds.n_cases == 1
    assert ds[0, 'subject'] == 'R0000'
    assert ds[0, 'epoch'] == 'target'
    assert ds.info['xs'] == ['imp']
    for key in ('r', 'z', 'residual', 'det', 'imp'):
        assert isinstance(ds[key], NDVar)

    # group -> one case per subject
    ds_all = e.load_trfs('all', 'imp', 0, 0.1)
    assert ds_all.n_cases == 2
    assert sorted(ds_all['subject'].cells) == ['R0000', 'R0001']

    # scale='original' rescales the kernel
    ds_scaled = e.load_trfs('R0000', 'imp', 0, 0.1, scale='original')
    assert (ds_scaled[0, 'imp'].x != ds[0, 'imp'].x).any()

    # trfs=False loads only the metrics
    ds_metrics = e.load_trfs('R0000', 'imp', 0, 0.1, trfs=False)
    assert ds_metrics.info['xs'] == []
    assert 'imp' not in ds_metrics
    assert isinstance(ds_metrics['r'], NDVar)


@requires_mne_sample_data
def test_load_trfs_collection(samples_experiment):
    "load_trfs over an EpochCollection: one case per member epoch"
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment, SampleTRF

    class SampleTRFCollection(SampleTRF):
        epochs = {**SampleExperiment.epochs, 'avc': EpochCollection(('auditory', 'visual'))}

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=1, n_segments=4)
    e = SampleTRFCollection(root)
    e.set(subject='R0000', epoch='avc', epoch_rejection='', raw='1-40', inv='')

    ds = e.load_trfs('R0000', 'imp', 0, 0.1)
    assert ds.n_cases == 2
    assert sorted(ds['epoch'].cells) == ['auditory', 'visual']
    assert ds.info['xs'] == ['imp']
    assert all(s == 'R0000' for s in ds['subject'])


@requires_mne_sample_data
@pytest.mark.slow
def test_load_trfs_source(samples_experiment):
    "load_trfs in source space: group morph to the common brain plus smoothing"
    from eelbrain._experiment.tests.sample_experiment import SampleTRF

    set_log_level('warning', 'mne')
    root = samples_experiment(n_subjects=2, n_segments=4, mris=True)
    e = SampleTRF(root)
    e.set(epoch='target', epoch_rejection='', raw='1-40', src='ico-2', parc='ac', inv='free-6-MNE')

    ds = e.load_trfs('all', 'imp', 0, 0.1, smooth=0.005)
    assert ds.n_cases == 2
    assert sorted(ds['subject'].cells) == ['R0000', 'R0001']
    # all subjects morphed onto the common brain, so kernels share one source space
    assert ds[0, 'imp'].source.subject == 'fsaverage'
    assert ds[1, 'imp'].source.subject == 'fsaverage'
    assert ds[0, 'imp'].source == ds[1, 'imp'].source
