# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import Dataset, Factor, Var
from eelbrain._exceptions import DefinitionError
from eelbrain.pipeline import *
from eelbrain.testing import assert_dataobj_equal, TempDir


SUBJECT = 'CheeseMonger'
SUBJECTS = [f'R{i:04}' for i in (1, 11, 111, 1111)]
SAMPLINGRATE = 1000.
TRIGGERS = np.tile(np.arange(1, 5), 2)
I_START = np.arange(1001, 1441, 55)


@pytest.fixture
def root_dir():
    tempdir = TempDir()
    for subject in SUBJECTS:
        sdir = os.path.join(tempdir, 'meg', subject)
        os.makedirs(sdir)
    return tempdir


class BaseExperiment(MneExperiment):

    sessions = 'file'

    raw = {
        '0-40': RawFilter('raw', None, 40, method='iir'),
        '1-40': RawFilter('raw', 1, 40, method='iir'),
        'ica': RawICA('raw', 'file'),
    }


class EventExperiment(MneExperiment):

    trigger_shift = 0.03

    sessions = 'cheese'

    raw = {
        '0-40': RawFilter('raw', None, 40, method='iir'),
        '1-40': RawFilter('raw', 1, 40, method='iir'),
    }

    variables = {
        'kind': {(1, 2, 3, 4): 'cheese', (11, 12, 13, 14): 'pet'},
        'name': {1: 'Leicester', 2: 'Tilsit', 3: 'Caerphilly', 4: 'Bel Paese'},
        'backorder': {(1, 4): 'no', (2, 3): 'yes'},
        'taste': {(1, 2): 'good', 'default': 'bad'},
    }

    epochs = {
        'cheese': {'sel': "kind == 'cheese'", 'tmin': -0.2},
        'cheese-leicester': {'base': 'cheese', 'tmin': -0.1, 'sel': "name == 'Leicester'"},
        'cheese-tilsit': {'base': 'cheese', 'sel': "name == 'Tilsit"},
    }

    defaults = {'model': 'name'}


class EventExperimentTriggerShiftDict(EventExperiment):
    "Test trigger shift as dictionary"
    trigger_shift = {SUBJECT: 0.04}


def gen_triggers():
    raw = Var([], info={'sfreq': SAMPLINGRATE})
    ds = Dataset(info={'subject': SUBJECT, 'session': 'cheese', 'raw': raw, 'sfreq': SAMPLINGRATE})
    ds['trigger'] = Var(TRIGGERS)
    ds['i_start'] = Var(I_START)
    return ds


def assert_inv_works(e, inv, args, make_kw, apply_kw):
    e.reset()
    e.set(inv=inv)
    method, make_kw_, apply_kw_ = e._inv_params()
    assert make_kw_ == make_kw
    assert apply_kw_ == apply_kw
    e.reset()
    e.set_inv(*args)
    assert e.get('inv') == inv
    method, make_kw_, apply_kw_ = e._inv_params()
    assert make_kw_ == make_kw
    assert apply_kw_ == apply_kw


def test_mne_experiment_templates():
    "Test MneExperiment template formatting"
    tempdir = TempDir()
    e = BaseExperiment(tempdir, False)

    # Don't create dirs without root
    assert e.get('raw-file', mkdir=True).endswith('-raw.fif')

    # model
    assert e.get('model', model='a % b') == 'a%b'
    assert e.get('model', model='b % a') == 'a%b'
    with pytest.raises(ValueError):
        e.set(model='a*b')
    with pytest.raises(ValueError):
        e.set(model='log(a)')

    # compounds
    assert e.get('src_kind') == '0-40 bestreg free-3-dSPM'
    e.set_inv('fixed')
    assert e.get('src_kind') == '0-40 bestreg fixed-3-dSPM'
    e.set(cov='noreg')
    assert e.get('src_kind') == '0-40 noreg fixed-3-dSPM'
    e.set(raw='1-40')
    assert e.get('src_kind') == '1-40 noreg fixed-3-dSPM'
    e.set(src='ico-5')
    assert e.get('src_kind') == '1-40 noreg ico-5 fixed-3-dSPM'
    e.set(src='ico-4')
    assert e.get('src_kind') == '1-40 noreg fixed-3-dSPM'

    # find terminal field names
    assert e.find_keys('raw-file') == ['root', 'subject', 'session', 'visit']
    assert e.find_keys('evoked-file', False) == ['subject', 'raw', 'epoch', 'visit', 'model', 'rej', 'equalize_evoked_count']

    assert_inv_works(e, 'free-3-MNE',
                     ('free', 3, 'MNE'),
                     {'loose': 1, 'depth': 0.8},
                     {'method': 'MNE', 'lambda2': 1 / 9})
    assert_inv_works(e, 'free-3-dSPM-0.2-pick_normal',
                     ('free', 3, 'dSPM', .2, True),
                     {'loose': 1, 'depth': 0.2},
                     {'method': 'dSPM', 'lambda2': 1 / 9, 'pick_ori': 'normal'})
    assert_inv_works(e, 'fixed-2-MNE-0.2',
                     ('fixed', 2, 'MNE', .2),
                     {'fixed': True, 'depth': 0.2},
                     {'method': 'MNE', 'lambda2': 1 / 4})
    assert_inv_works(e, 'loose.2-2-MNE-pick_normal',
                     (0.2, 2, 'MNE', 0.8, True),
                     {'loose': 0.2, 'depth': 0.8},
                     {'method': 'MNE', 'lambda2': 1 / 4, 'pick_ori': 'normal'})
    assert_inv_works(e, 'loose.5-3-sLORETA',
                     (0.5, 3, 'sLORETA'),
                     {'loose': 0.5, 'depth': 0.8},
                     {'method': 'sLORETA', 'lambda2': 1 / 9})
    assert_inv_works(e, 'fixed-1-MNE-0',
                     ('fixed', 1, 'MNE', 0),
                     {'fixed': True, 'depth': None},
                     {'method': 'MNE', 'lambda2': 1})
    assert_inv_works(e, 'fixed-MNE-0',
                     ('fixed', 0, 'MNE', 0),
                     {'fixed': True, 'depth': None},
                     {'method': 'MNE', 'lambda2': 0})
    # redundant values
    assert e.get('inv', inv='fixed-1-MNE-0.8') == 'fixed-1-MNE'
    assert e.get('inv', inv='fixed-0-MNE-0.8') == 'fixed-MNE'

    with pytest.raises(ValueError):
        e.set_inv('free', -3, 'dSPM')
    with pytest.raises(ValueError):
        e.set(inv='free-3-mne')
    with pytest.raises(ValueError):
        e.set(inv='free-3-MNE-2')

    assert e.find_keys('test-file', False) == ['analysis', 'group', 'epoch', 'visit', 'test', 'test_options', 'test_dims']
    assert e._glob_pattern('test-file', True, group='all') == os.path.join(tempdir, 'eelbrain-cache', 'test', '* all', '* *.pickled')


def test_test_experiment(root_dir):
    "Test event labeling with the EventExperiment subclass of MneExperiment"
    e = EventExperiment()

    # test defaults
    assert e.get('session') == 'cheese'
    assert e.get('model') == 'name'

    # test event labeling
    ds = e._label_events(gen_triggers())
    name = Factor([e.variables['name'][t] for t in TRIGGERS], name='name')
    assert_dataobj_equal(ds['name'], name)
    tgt = ds['trigger'].as_factor(e.variables['backorder'], 'backorder')
    assert_dataobj_equal(ds['backorder'], tgt)
    tgt = ds['trigger'].as_factor(e.variables['taste'], 'taste')
    assert_dataobj_equal(ds['taste'], tgt)
    assert_array_equal(ds['i_start'], I_START)
    assert_array_equal(ds['subject'] == SUBJECT, True)

    # epochs
    assert e._epochs['cheese'].tmin == -0.2
    assert e._epochs['cheese-leicester'].tmin == -0.1
    assert e._epochs['cheese-tilsit'].tmin == -0.2

    # tests
    e = EventExperiment(root_dir)
    # add test
    EventExperiment.tests['aov'] = ANOVA('backorder * taste * subject')
    e = EventExperiment(root_dir)
    e.set(test='aov')
    assert e.get('model') == 'backorder%taste'
    # remove test
    del EventExperiment.tests['aov']
    e = EventExperiment(root_dir)


class FileExperiment(MneExperiment):

    groups = {
        'gsub': Group(SUBJECTS[1:]),
        'gexc': SubGroup('all', SUBJECTS[0]),
        'gexc2': SubGroup('gexc', SUBJECTS[-1:]),
    }

    sessions = 'file'


class FileExperimentDefaults(FileExperiment):

    defaults = {'session': 'file',
                'group': 'gsub'}


def test_file_handling(root_dir):
    "Test MneExperiment with actual files"
    e = FileExperiment(root_dir)
    assert e.get('subject') == SUBJECTS[0]
    assert [s for s in e.iter(group='all')] == SUBJECTS
    assert [s for s in e.iter(group='gsub')] == SUBJECTS[1:]
    assert [s for s in e.iter(group='gexc')] == SUBJECTS[1:]
    assert [s for s in e.iter(group='gexc2')] == SUBJECTS[1:-1]
    assert e.get('subject') == SUBJECTS[1]
    assert e.get('subject', group='all') == SUBJECTS[1]
    e.set(SUBJECTS[0])
    assert e.get('subject') == SUBJECTS[0]
    assert e.get('subject', group='gsub') == SUBJECTS[1]

    e = FileExperimentDefaults(root_dir)
    assert e.get('group'), 'gsub'
    assert e.get('subject') == SUBJECTS[1]


class VisitExperiment(BaseExperiment):

    visits = ('', '1')


def test_visit():
    tempdir = TempDir()
    for subject in SUBJECTS:
        sdir = os.path.join(tempdir, 'meg', subject)
        os.makedirs(sdir)
    e = VisitExperiment(tempdir, raw='ica')

    assert e.get('ica-file') == os.path.join(tempdir, 'meg', 'R0001', 'R0001 ica-ica.fif')
    e.set(visit='1')
    assert e.get('ica-file') == os.path.join(tempdir, 'meg', 'R0001', 'R0001 1 ica-ica.fif')


# definition checks
def test_bad_definitions():
    "Test invalid definitions raise errors"
    # bad raw
    class BadExperiment(BaseExperiment):
        raw = {'bad raw': RawFilter('raw', 1, 40), **BaseExperiment.raw}

    with pytest.raises(DefinitionError):
        BadExperiment()

    # bad vardef
    with pytest.raises(DefinitionError):
        class BadExperiment(BaseExperiment):
            tests = {'badvars': ANOVA('badvar * subject', vars=((), 'badvar = urk'))}
