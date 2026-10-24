import json
from pathlib import Path

import pytest

from eelbrain import Pipeline
from eelbrain.pipeline import PrimaryEpoch
from eelbrain._exceptions import ConfigurationError


def _write_dataset_description(root: Path, name: str) -> None:
    (root / 'dataset_description.json').write_text(json.dumps({'Name': name, 'BIDSVersion': '1.10.0'}))


def test_ignore_runs_excludes_recordings(tmp_path):
    """Ignored runs are excluded from the recordings index (used for run aggregation, ICA and head-position inputs)"""
    _write_dataset_description(tmp_path, 'ignore-runs-test')
    meg_dir = tmp_path / 'sub-01' / 'meg'
    meg_dir.mkdir(parents=True)
    for run in ('1', '2', '3'):
        (meg_dir / f'sub-01_task-test_run-{run}_meg.fif').touch()

    class IgnoreRunExperiment(Pipeline):
        stim_channel = 'STI 014'
        ignore_entities = {'ignore_runs': ('2',)}

    experiment = IgnoreRunExperiment(tmp_path)
    assert experiment._runs == ('1', '3')
    assert experiment._recordings == {('01', '', 'test', '', run) for run in ('1', '3')}
    assert experiment._runs_for == {('01', '', 'test', ''): ['1', '3']}

    class PinnedRunExperiment(IgnoreRunExperiment):
        epochs = {'r2': PrimaryEpoch('test', tmax=0.3, run='2')}

    with pytest.raises(ConfigurationError, match="run='2' is not available"):
        PinnedRunExperiment(tmp_path)


def test_recordings_preserve_runless_recording_alongside_numbered_runs(tmp_path):
    """A recording without a run entity is kept when other recordings have numbered runs"""
    _write_dataset_description(tmp_path, 'mixed-runs-test')
    for subject, run in [('01', '1'), ('01', '2'), ('02', None)]:
        name = f'sub-{subject}_task-test_run-{run}_meg.fif' if run else f'sub-{subject}_task-test_meg.fif'
        meg_dir = tmp_path / f'sub-{subject}' / 'meg'
        meg_dir.mkdir(parents=True, exist_ok=True)
        (meg_dir / name).touch()

    class MixedRunExperiment(Pipeline):
        stim_channel = 'STI 014'

    experiment = MixedRunExperiment(tmp_path)
    assert experiment._runs == ('1', '2')
    assert experiment._recordings == {
        ('01', '', 'test', '', '1'),
        ('01', '', 'test', '', '2'),
        ('02', '', 'test', '', ''),
    }
    assert experiment._runs_for[('02', '', 'test', '')] == ['']


def test_pinned_run_without_runs(tmp_path):
    """An epoch pinned to a run is rejected when the dataset has no run entity"""
    _write_dataset_description(tmp_path, 'no-runs-test')
    meg_dir = tmp_path / 'sub-01' / 'meg'
    meg_dir.mkdir(parents=True)
    (meg_dir / 'sub-01_task-test_meg.fif').touch()

    class PinnedRunExperiment(Pipeline):
        stim_channel = 'STI 014'
        epochs = {'r1': PrimaryEpoch('test', tmax=0.3, run='1')}

    with pytest.raises(ConfigurationError, match="run='1' is not available"):
        PinnedRunExperiment(tmp_path)
