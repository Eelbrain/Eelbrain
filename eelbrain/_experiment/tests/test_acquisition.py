import json
from pathlib import Path

import mne
import numpy as np

from eelbrain import Pipeline
from eelbrain.pipeline import PrimaryEpoch
from eelbrain._experiment.pathing import ica_file_path, rej_file_path, test_basename as result_basename
from eelbrain._experiment.preprocessing import raw_input_name


def _make_raw(path, triggers: tuple[int, ...]) -> None:
    info = mne.create_info(['MEG 001', 'STI 014'], 100., ['mag', 'stim'])
    data = np.zeros((2, 200))
    for index, trigger in enumerate(triggers, 1):
        data[1, index * 20] = trigger
    raw = mne.io.RawArray(data, info, verbose='error')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path, overwrite=True, verbose='error')


def test_acquisition_derivative_paths():
    state = {
        'subject': '01',
        'session': '',
        'task': 'test',
        'acquisition': 'highres',
        'run': '1',
        'raw': 'ica',
        'epoch': 'test',
        'epoch_rejection': 'manual',
    }
    assert ica_file_path(state, 'ica', datatype='meg') == Path('derivatives/mne/sub-01/meg/sub-01_acq-highres_run-1_desc-ica_ica.fif')
    assert rej_file_path(state, datatype='meg') == Path('derivatives/mne/sub-01/meg/sub-01_acq-highres_run-1_raw-ica_epoch-test_rej-manual_epoch.pickle')
    assert result_basename(state, datatype='meg') == 'acq-highres_run-1_meg'


def test_acquisitions_are_independent_analysis_branches(tmp_path):
    (tmp_path / 'dataset_description.json').write_text(json.dumps({'Name': 'acquisition-test', 'BIDSVersion': '1.10.0'}))
    meg_dir = tmp_path / 'sub-01' / 'meg'
    for acquisition, triggers in {'a': (1,), 'b': (1, 2)}.items():
        for run in ('1', '2'):
            path = meg_dir / f'sub-01_task-test_acq-{acquisition}_run-{run}_meg.fif'
            _make_raw(path, triggers)

    class AcquisitionExperiment(Pipeline):
        stim_channel = 'STI 014'
        epochs = {
            'test': PrimaryEpoch('test', tmin=0, tmax=0.01, baseline=False),
        }

    experiment = AcquisitionExperiment(tmp_path)
    assert experiment.get_field_values('acquisition') == ['a', 'b']
    assert experiment._recordings == {
        ('01', '', 'test', acquisition, run)
        for acquisition in ('a', 'b')
        for run in ('1', '2')
    }
    assert experiment._runs_for == {
        ('01', '', 'test', acquisition): ['1', '2']
        for acquisition in ('a', 'b')
    }

    event_paths = {}
    for acquisition, n_events in {'a': 2, 'b': 4}.items():
        experiment.set(epoch='test', epoch_rejection='', acquisition=acquisition)
        events = experiment.load_selected_events()
        assert events.n_cases == n_events

        experiment.set(run='1')
        raw_ctx = experiment._resolve_derivative(raw_input_name('raw'), options={'noise': False})
        assert raw_ctx.node.path(raw_ctx).name == f'sub-01_task-test_acq-{acquisition}_run-1_meg.fif'
        event_paths[acquisition] = experiment._resolve_derivative('events').artifact_path

    assert event_paths['a'] != event_paths['b']
