from eelbrain._experiment.epochs import PrimaryEpoch
from eelbrain._experiment.variable_def import LabelVar
from eelbrain.pipeline import RawMaxwell, RawFilter, RawICA
from eelbrain import MneExperiment


class PhantomExperiment(MneExperiment):
    ignore_entities = {
        'ignore_subjects': ['noIPG'],
        'ignore_sessions': ['220426'],
    }
    raw = {
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True),
        '1-40': RawFilter('tsss', 1, 40),
        'ica': RawICA('tsss', 'dip13', method='fastica', n_components=0.95),
        'ica1-40': RawFilter('ica', 1, 40),
    }
    variables = {
        'pulse_phase': LabelVar('trigger', {
            (1, 2, 3, 4, 5): 'early',
            (6, 7, 8, 9): 'late',
        }),
    }
    epochs = {
        'early_pulse': PrimaryEpoch('dip13', "pulse_phase == 'early'"),
        'late_pulse': PrimaryEpoch('dip13', "pulse_phase == 'late'"),
    }


if __name__ == '__main__':
    e = PhantomExperiment('D:\\sfb_meg_phantom')
    e.set(subject='mdtpc', task='dip13mov')
    print(e.get('task'))
    epochs = e.load_epochs(ndvar=False)
    evoked = e.load_evoked(ndvar=False)
    print(evoked)
