import json
from pathlib import Path
from types import SimpleNamespace

from eelbrain import Pipeline
from eelbrain.pipeline import RawICA, RawMaxwell
from eelbrain._experiment.preprocessing import ica_input_name


def _write_dataset_description(root: Path, name: str) -> None:
    (root / 'dataset_description.json').write_text(json.dumps({'Name': name, 'BIDSVersion': '1.10.0'}))


def test_ignore_runs_excludes_ica_and_head_position_recordings(tmp_path):
    """ignore_runs must also be honored by non-epoch derivatives that
    concatenate across runs, not just by epoch/event aggregation.

    Regression coverage for the same Pipeline.__init__ bug as
    test_epoch_run's ignore_runs check: Pipeline._recordings (built from
    find_matching_paths) used to include every run regardless of
    ignore_runs, so ICAInput (which concatenates all of a subject's runs
    for the ICA fit once the ICA step follows a RawMaxwell step, see
    RawICA._concatenate_runs) and CanonicalHeadPositionDerivative would
    have silently used the ignored run's recording too.

    Only entity discovery (Pipeline.__init__: find_matching_paths,
    _recordings, _runs_for) and the resulting node wiring are under test
    here, not any actual raw data; nothing in this test reads a raw file's
    content, so the source files only need BIDS-valid names, not
    MNE-readable content.
    """
    _write_dataset_description(tmp_path, 'ignore-runs-test')
    meg_dir = tmp_path / 'sub-01' / 'meg'
    meg_dir.mkdir(parents=True)
    for run in ('1', '2', '3'):
        (meg_dir / f'sub-01_task-test_run-{run}_meg.fif').touch()

    class IgnoreRunExperiment(Pipeline):
        stim_channel = 'STI 014'
        ignore_entities = {'ignore_runs': ('2',)}
        raw = {
            'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True, st_overlap=False),
            'ica': RawICA('tsss', method='fastica', n_components=1),
        }

    experiment = IgnoreRunExperiment(tmp_path)
    assert experiment._runs == ('1', '3')
    assert experiment._recordings == {('01', '', 'test', '', run) for run in ('1', '3')}

    ica_node = experiment._derivatives._nodes[ica_input_name('ica')]
    assert ica_node.pipe._concatenate_runs is True
    ctx = SimpleNamespace(state={'subject': '01', 'session': '', 'acquisition': ''})
    states = ica_node._source_states(ctx, ['test'])
    assert {state['run'] for state in states} == {'1', '3'}


def test_recordings_preserve_runless_recording_alongside_numbered_runs(tmp_path):
    """A recording without a run entity must not disappear from
    Pipeline._recordings merely because another recording in the same
    dataset has a numbered run.

    Regression test for a bug in the initial fix for ignore_runs (see
    test_ignore_runs_excludes_ica_and_head_position_recordings above):
    filtering matching_paths with ``runs=self._runs`` (self._runs being
    only the explicit run labels found, e.g. ('1', '2')) excluded every
    recording that has no run entity at all, whenever any numbered run
    existed elsewhere in the dataset. For a RawICA step that does *not*
    follow a RawMaxwell step (so it does not concatenate runs, and
    ICAInput._source_states keys directly on the ambient run state), that
    made the run-less subject's own recording undiscoverable, raising
    FileMissingError even though nothing about that subject was ignored.
    """
    _write_dataset_description(tmp_path, 'mixed-runs-test')
    for subject, run in [('01', '1'), ('01', '2'), ('02', None)]:
        name = f'sub-{subject}_task-test_run-{run}_meg.fif' if run else f'sub-{subject}_task-test_meg.fif'
        meg_dir = tmp_path / f'sub-{subject}' / 'meg'
        meg_dir.mkdir(parents=True, exist_ok=True)
        (meg_dir / name).touch()

    class MixedRunExperiment(Pipeline):
        stim_channel = 'STI 014'
        raw = {
            'ica': RawICA('raw', 'test', method='fastica', n_components=1),
        }

    experiment = MixedRunExperiment(tmp_path)
    assert experiment._runs == ('1', '2')
    assert experiment._recordings == {
        ('01', '', 'test', '', '1'),
        ('01', '', 'test', '', '2'),
        ('02', '', 'test', '', ''),
    }

    ica_node = experiment._derivatives._nodes[ica_input_name('ica')]
    assert ica_node.pipe._concatenate_runs is False
    ctx = SimpleNamespace(state={'subject': '02', 'session': '', 'acquisition': '', 'run': ''})
    assert ica_node._source_states(ctx, ['test']) == [{'task': 'test', 'run': ''}]
