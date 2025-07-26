from eelbrain._experiment.epochs import PrimaryEpoch
from eelbrain._experiment.variable_def import LabelVar
from eelbrain.pipeline import RawMaxwell, RawFilter, RawICA
from eelbrain import MneExperiment


class SomatomotorExperiment(MneExperiment):
    ignore_entities = {
        'ignore_subjects': ['noIPG'],
        'ignore_sessions': ['mri'],
    }
    raw = {
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=.9, st_only=True),
        '1-40': RawFilter('tsss', 1, 40),
        'ica': RawICA('tsss', 'somatomotor', method='fastica', n_components=0.95),
        'ica1-40': RawFilter('ica', 1, 40),
    }
    variables = {
        'event': LabelVar('trigger', {
            (1,): 'Finger',
            (2,): 'NULL',
            (3,): 'somatosensory',
        }),
    }
    epochs = {
        'finger': PrimaryEpoch('somatomotor', "event == 'Finger'", samplingrate=251.005),
        'not_null': PrimaryEpoch('somatomotor', "event != 'NULL'", samplingrate=251.005),
    }


if __name__ == '__main__':
    e = SomatomotorExperiment('D:\\somatomotor')
    # print(e.load_events())
    # e.set(subject='mdtpc', task='dip13mov')
    # print(e.get('task'))
    epochs = e.load_epochs()
    # evoked = e.load_evoked(ndvar=False)
    print(epochs)