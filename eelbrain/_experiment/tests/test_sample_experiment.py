# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Test MneExperiment using mne-python sample data"""
from pathlib import Path
from os.path import join, exists
import pytest
import shutil
from warnings import catch_warnings, filterwarnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from eelbrain import *
from eelbrain.pipeline import *
from eelbrain._exceptions import DefinitionError
from eelbrain.testing import TempDir, assert_dataobj_equal, requires_mne_sample_data


@requires_mne_sample_data
def test_sample():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 3, 2, mris=True)

    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)

    assert e.get('raw') == '1-40'
    assert e.get('subject') == 'R0000'
    assert e.get('subject', subject='R0002') == 'R0002'

    # events
    e.set('R0001', rej='')
    ds = e.load_selected_events(epoch='target')
    assert ds.n_cases == 39
    ds = e.load_selected_events(epoch='auditory')
    assert ds.n_cases == 20
    ds = e.load_selected_events(epoch='av')
    assert ds.n_cases == 39

    # evoked cache invalidated by change in bads
    e.set('R0001', rej='', epoch='target')
    ds = e.load_evoked(ndvar=False)
    assert ds[0, 'evoked'].info['bads'] == []
    e.make_bad_channels(['MEG 0331'])
    ds = e.load_evoked(ndvar=False)
    assert ds[0, 'evoked'].info['bads'] == ['MEG 0331']

    e.set(rej='man', model='modality')
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
    _ = e._report_subject_info(ds, '')

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
            'v_shift': SecondaryEpoch('visual', vars={'shift': 'Var([0.0], repeat=len(side))'}),
            'a_shift': SecondaryEpoch('auditory', vars={'shift': 'Var([0.1], repeat=len(side))'}),
            'av_shift': SuperEpoch(('v_shift', 'a_shift'), post_baseline_trigger_shift='shift', post_baseline_trigger_shift_max=0.1, post_baseline_trigger_shift_min=0.0),
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
    v = ds.sub("epoch=='v_shift'", 'meg')
    v_target = e.load_epochs(baseline=True, epoch='visual')['meg'].sub(time=(-0.1, v.time.tstop))
    assert_almost_equal(v.x, v_target.x)
    a = ds.sub("epoch=='a_shift'", 'meg').sub(time=(-0.1, 0.099))
    a_target = e.load_epochs(baseline=True, epoch='auditory')['meg'].sub(time=(0, 0.199))
    assert_almost_equal(a.x, a_target.x, decimal=20)

    # duplicate subject
    class BadExperiment(SampleExperiment):
        groups = {'group': ('R0001', 'R0002', 'R0002')}
    with pytest.raises(DefinitionError):
        BadExperiment(root)

    # non-existing subject
    class BadExperiment(SampleExperiment):
        groups = {'group': ('R0001', 'R0003', 'R0002')}
    with pytest.raises(DefinitionError):
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
            'apply-ica': RawApplyICA('tsss', 'ica'),
            **SampleExperiment.raw,
        }
    e = Experiment(root)
    ica_path = e.make_ica(raw='ica')
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

    # rename subject
    # --------------
    src = Path(e.get('raw-dir', subject='R0001'))
    dst = Path(e.get('raw-dir', subject='R0003', match=False))
    shutil.move(src, dst)
    for path in dst.glob('*.fif'):
        shutil.move(path, dst / path.parent / path.name.replace('R0001', 'R0003'))
    # check subject list
    e = SampleExperiment(root)
    assert list(e) == ['R0000', 'R0002', 'R0003']
    # check that cached test got deleted
    assert e.get('raw') == '1-40'
    with pytest.raises(IOError):
        e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False)
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False, make=True)
    assert res.df == 2
    assert res.p.min() == pytest.approx(.143, abs=.001)
    assert res.difference.max() == pytest.approx(4.47e-13, 1e-15)

    # remove subject
    # --------------
    shutil.rmtree(dst)
    # check cache
    e = SampleExperiment(root)
    assert list(e) == ['R0000', 'R0002']
    # check that cached test got deleted
    assert e.get('raw') == '1-40'
    with pytest.raises(IOError):
        e.load_test('a>v', 0.05, 0.2, 0.05, samples=20, data='sensor', baseline=False)

    # label_events
    # ------------
    class Experiment(SampleExperiment):
        def label_events(self, ds):
            SampleExperiment.label_events(self, ds)
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
    # change parc definition

    class Experiment(SampleExperiment):
        parcs = {
            'ac': SubParc('aparc', ('transversetemporal', 'superiortemporal')),
        }
    e = Experiment(root)
    assert len(e.glob('annot-file', True, parc='ac')) == 0
    labels = e.load_annot(parc='ac', mrisubject='fsaverage')
    assert len(labels) == 6


@requires_mne_sample_data
@pytest.mark.slow
def test_sample_source():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 3, 2, mris=True)  # TODO: use sample MRI which already has forward solution
    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)

    # source space tests
    e.set(src='ico-4', rej='', epoch='auditory')
    # These two tests are only identical if the evoked has been cached before the first test is loaded
    resp = e.load_test('left=right', 0.05, 0.2, 0.05, samples=100, parc='ac', make=True)
    resm = e.load_test('left=right', 0.05, 0.2, 0.05, samples=100, mask='ac', make=True)
    assert_dataobj_equal(resp.t, resm.t)
    # ROI tests
    e.set(epoch='target')
    ress = e.load_test('left=right', 0.05, 0.2, 0.05, samples=100, data='source.rms', parc='ac', make=True)
    res = ress.res['transversetemporal-lh']
    assert res.p.min() == pytest.approx(0.429, abs=.001)
    ress = e.load_test('twostage', 0.05, 0.2, 0.05, samples=100, data='source.rms', parc='ac', make=True)
    res = ress.res['transversetemporal-lh']
    assert res.samples == -1
    assert res.tests['intercept'].p.min() == 1 / 7


@requires_mne_sample_data
def test_sample_sessions():
    set_log_level('warning', 'mne')
    from eelbrain._experiment.tests.sample_experiment_sessions import SampleExperiment

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 1, 2)

    class Experiment(SampleExperiment):

        raw = {
            'ica': RawICA('raw', ('sample1', 'sample2'), 'fastica', max_iter=1),
            **SampleExperiment.raw,
        }

    root = join(tempdir, 'SampleExperiment')
    e = Experiment(root)
    # bad channels
    e.make_bad_channels('0111')
    assert e.load_bad_channels() == ['MEG 0111']
    assert e.load_bad_channels(session='sample2') == []
    e.show_bad_channels()
    e.merge_bad_channels()
    assert e.load_bad_channels(session='sample2') == ['MEG 0111']
    e.show_bad_channels()

    # rejection
    for _ in e:
        for epoch in ('target1', 'target2'):
            e.set(epoch=epoch)
            e.make_epoch_selection(auto=2e-12)

    ds = e.load_evoked('R0000', epoch='target2')
    e.set(session='sample1')
    ds2 = e.load_evoked('R0000')
    assert_dataobj_equal(ds2, ds, decimal=19)

    # super-epoch
    ds1 = e.load_epochs(epoch='target1')
    ds2 = e.load_epochs(epoch='target2')
    ds_super = e.load_epochs(epoch='super')
    assert_dataobj_equal(ds_super['meg'], combine((ds1['meg'], ds2['meg'])))
    # evoked
    dse_super = e.load_evoked(epoch='super', model='modality%side')
    target = ds_super.aggregate('modality%side', drop=('i_start', 't_edf', 'T', 'index', 'trigger', 'session', 'interpolate_channels', 'epoch'))
    assert_dataobj_equal(dse_super, target, 19)

    # conflicting session and epoch settings
    rej_path = join(root, 'meg', 'R0000', 'epoch selection', 'sample2_1-40_target2-man.pickled')
    e.set(epoch='target2', raw='1-40')
    assert not exists(rej_path)
    e.set(session='sample1')
    e.make_epoch_selection(auto=2e-12)
    assert exists(rej_path)

    # ica
    e.set('R0000', raw='ica')
    with catch_warnings():
        filterwarnings('ignore', "FastICA did not converge", UserWarning)
        assert e.make_ica() == join(root, 'meg', 'R0000', 'R0000 ica-ica.fif')


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
    assert e.get('raw') == '1-40'

    e.make_epoch_selection(auto={'mag': 2e-12, 'grad': 5e-11, 'eeg': 1.5e-4})
    ds = e.load_selected_events(reject='keep')
    assert ds['accept'].sum() == 69
