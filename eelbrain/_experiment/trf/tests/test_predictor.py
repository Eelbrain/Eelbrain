import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain import Dataset, Factor, NDVar, UTS, Var
from eelbrain._experiment.trf.model import TRFModelError, parse_term
from eelbrain._experiment.trf.nodes import Recording, find_bids_recordings
from eelbrain._experiment.trf.predictor import EventPredictor, NUTSPredictor, SubjectUTSPredictor, UTSPredictor


def test_predictor_sampling():
    assert UTSPredictor().sampling == 'continuous'
    uts_predictor = UTSPredictor(sampling='discrete')
    assert uts_predictor.sampling == 'discrete'
    sequence_predictor = SubjectUTSPredictor(sampling='discrete')
    contents = NDVar(np.zeros(2), UTS(0, 0.1, 2))
    x = sequence_predictor._prepare_sequence(contents, 0.1, parse_term('envseq'))
    assert x.info['sampling'] == 'discrete'
    with pytest.raises(TRFModelError):
        sequence_predictor._prepare_sequence(contents, 0.1, parse_term('envseq-value-step'))

    predictor = NUTSPredictor()
    assert predictor._sampling() == 'discrete'
    assert predictor._sampling('step') == 'continuous'


def test_subject_uts_predictor_identity():
    "SubjectUTSPredictor file identity depends on sequence versus per-event mode"
    from pathlib import Path

    # per_event=False: bare code, no stimulus allowed
    p = SubjectUTSPredictor()
    assert p.per_event is False
    assert p._key_fields == ('subject', 'session', 'task', 'acquisition', 'run')
    assert p._as_dict() == {'type': 'SubjectUTSPredictor', 'resample': None, 'sampling': 'continuous', 'per_event': False}
    term = parse_term('envseq')
    state = {'subject': 'R0001', 'session': '', 'task': 'sample', 'acquisition': '', 'run': ''}
    assert p._path(term, state, Path('/root')) == Path('/root/derivatives/subject-predictors/sub-R0001/sub-R0001_task-sample_desc-envseq.pickle')
    assert p._reference_stem(term, state) == 'sub-R0001_task-sample_desc-envseq'

    # session + acquisition + run add entities in canonical BIDS order
    state2 = {'subject': 'R0001', 'session': '02', 'task': 'story', 'acquisition': 'hi', 'run': '3'}
    assert p._reference_stem(term, state2) == 'sub-R0001_ses-02_task-story_acq-hi_run-3_desc-envseq'

    # per_event=True: the stimulus becomes part of the file identity
    pe = SubjectUTSPredictor(per_event=True)
    assert pe.per_event is True
    assert pe._key_fields == ('subject', 'session', 'acquisition')
    stim_term = parse_term('auditory~envp')
    assert pe._path(stim_term, state, Path('/root')).name == 'sub-R0001_desc-auditory~envp.pickle'
    assert pe._reference_stem(stim_term, state) == 'sub-R0001_desc-auditory~envp'
    assert pe._reference_stem(stim_term, state2) == 'sub-R0001_ses-02_acq-hi_desc-auditory~envp'
    # distinct stimuli map to distinct reference copies
    assert pe._reference_stem(parse_term('visual~envp'), state) != pe._reference_stem(stim_term, state)


def test_case_recordings():
    "BIDS entities varying by case are columns; invariant entities are in info"
    ds = Dataset(
        {'run': Factor(['1', '2'])},
        info=dict(subject='R0001', session='', task='sample', acquisition=''),
    )
    recordings = find_bids_recordings(ds)
    assert recordings[0] == Recording('R0001', '', 'sample', '', '1')
    assert recordings[1] == Recording('R0001', '', 'sample', '', '2')
    del ds.info['task']
    with pytest.raises(KeyError):
        find_bids_recordings(ds)


def _continuous_events(stim_var='stimulus'):
    "Two events in a later segment: 'a' at epoch_time=2, 'b' at epoch_time=2.5"
    ds = Dataset()
    ds['epoch_time'] = Var([2.0, 2.5])
    ds[stim_var] = Factor(['a', 'b'])
    return ds


def test_uts_predictor_generate_continuous():
    "UTSPredictor._generate_continuous places per-stimulus predictors at their epoch_time"
    uts = UTS(2, 0.1, 10)  # 1 s output on the ContinuousEpoch-wide clock
    tstep = uts.tstep
    # per-stimulus predictors: 3-sample ramps starting at their own t=0
    cache = {
        'a': NDVar(np.array([1., 2., 3.]), UTS(0, tstep, 3), name='x'),
        'b': NDVar(np.array([4., 5., 6.]), UTS(0, tstep, 3), name='x'),
    }
    term = parse_term('env')
    x = UTSPredictor()._generate_continuous(uts, _continuous_events(), 'stimulus', term, cache)
    assert x.time == uts
    expected = np.zeros(10)
    expected[0:3] = [1, 2, 3]  # 'a' at epoch_time=2
    expected[5:8] = [4, 5, 6]  # 'b' at epoch_time=2.5 -> sample 5
    assert_array_equal(x.x, expected)
    assert x.info['sampling'] == 'continuous'


def test_uts_predictor_generate_continuous_out_of_bounds():
    "A stimulus predictor running past the segment raises"
    uts = UTS(2, 0.1, 4)
    cache = {'a': NDVar(np.zeros(3), UTS(0, 0.1, 3), name='x'), 'b': NDVar(np.zeros(3), UTS(0, 0.1, 3), name='x')}
    with pytest.raises(ValueError):
        UTSPredictor()._generate_continuous(uts, _continuous_events(), 'stimulus', term=parse_term('env'), cache=cache)


def test_event_predictor_generate_continuous():
    "EventPredictor._generate_continuous puts a unit impulse at each event's epoch_time"
    uts = UTS(2, 0.1, 10)
    x = EventPredictor()._generate_continuous(uts, _continuous_events(), parse_term('imp'))
    assert x.time == uts
    expected = np.zeros(10)
    expected[0] = 1.
    expected[5] = 1.
    assert_array_equal(x.x, expected)


def test_nuts_predictor_generate_continuous():
    "NUTSPredictor._generate_continuous shifts each stimulus' event table by epoch_time"
    uts = UTS(2, 0.1, 10)
    # per-stimulus NUTS data: one impulse each, at time 0 within the stimulus
    cache = {
        'a': Dataset({'time': Var([0.0]), 'value': Var([2.0])}),
        'b': Dataset({'time': Var([0.0]), 'value': Var([3.0])}),
    }
    term = parse_term('word-value')
    x = NUTSPredictor()._generate_continuous(uts, _continuous_events(), 'stimulus', term, cache)
    assert x.time == uts
    expected = np.zeros(10)
    expected[0] = 2.0   # 'a' impulse shifted to epoch_time=2
    expected[5] = 3.0   # 'b' impulse shifted to epoch_time=2.5
    assert_array_equal(x.x, expected)
