# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join
from warnings import catch_warnings, filterwarnings

import pytest

from eelbrain import datasets, set_log_level
from eelbrain.pipeline import MneExperiment, PrimaryEpoch, RawFilter, RawICA, TTestRelated
from eelbrain.testing import TempDir, path, requires_mne_sample_data


class Experiment(MneExperiment):

    meg_system = 'neuromag306mag'
    stim_channel = 'STI 014'

    sessions = 'sample'

    visits = ('', '1')

    raw = {
        '1-40': RawFilter('raw', 1, 40, method='iir'),
        'ica': RawICA('raw', 'sample', 'fastica', max_iter=1),
        'ica1-40': RawFilter('ica', 1, 40, method='iir'),
    }

    variables = {
        'event': {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'},
        'side': {(1, 3): 'left', (2, 4): 'right'},
        'modality': {(1, 2): 'auditory', (3, 4): 'visual'}
    }

    epochs = {
        'target': PrimaryEpoch('sample', "event == 'target'", tmax=0.15, decim=4),
    }

    tests = {
        'side': TTestRelated('side', 'left', 'right'),
    }


def test_visit_patterns():
    e = Experiment()
    assert e._glob_pattern('fwd-file', True, session='sample') == path('/eelbrain-cache/raw/*/sample*-*-*-fwd.fif')


@requires_mne_sample_data
def test_mne_experiment_visit():
    set_log_level('warning', 'mne')
    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 3, 1, n_visits=2)
    root = join(tempdir, 'SampleExperiment')

    e = Experiment(root)
    ds0 = e.load_events(epoch='target', visit='')
    ds1 = e.load_events(epoch='target', visit='1')
    assert ds0[1, 'trigger'] != ds1[1, 'trigger']
    for _ in e:
        e.make_epoch_selection(auto=2e-12)
    e.set(visit='')
    for _ in e:
        e.make_epoch_selection(auto=2e-12)
    res0 = e.load_test('side', 0.05, 0.15, 0.05, data='sensor', make=True)
    e.set(visit='1')
    with pytest.raises(IOError):
        e.load_test('side', 0.05, 0.15, 0.05, data='sensor')
    res1 = e.load_test('side', 0.05, 0.15, 0.05, data='sensor', make=True)
    assert res1.p.min() != res0.p.min()

    # ica
    e.set(raw='ica')
    with catch_warnings():
        filterwarnings('ignore', "FastICA did not converge", UserWarning)
        assert e.make_ica(visit='') == join(root, 'meg', 'R0000', 'R0000 ica-ica.fif')
        assert e.make_ica(visit='1') == join(root, 'meg', 'R0000', 'R0000 1 ica-ica.fif')
