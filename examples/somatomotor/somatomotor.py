# skip test: data unavailable
from eelbrain.pipeline import *


class Somatomotor(Pipeline):

    ignore_entities = {
        'ignore_runs': ('2', '3'),
        'ignore_sessions': 'mri',
    }

    raw = {
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
        '1-40': RawFilter('tsss', 1, 40),
        'ica': RawICA('1-40', 'somatomotor', n_components=0.99),
    }

    variables = {
        'is_null': LabelVar('trigger', {2: 'true', (1, 3): 'false'}),
        'stimulus': LabelVar('trigger', {2: 'null', 1: 'finger', 3: 'somatosensory'}),
    }

    epochs = {
        'not_null': PrimaryEpoch('somatomotor', "is_null == 'false'", samplingrate=251.005),
        'finger': SecondaryEpoch('not_null', "stimulus == 'finger'"),
        'somatosensory': SecondaryEpoch('not_null', "stimulus == 'somatosensory'"),
    }

    tests = {
        '=0': TTestOneSample(),
        'connection': TTestRelated('stimulus', 'finger', 'somatosensory'),
        'anova': ANOVA('stimulus * subject'),
    }


root = '/mnt/d/Data/Somatomotor'
e = Somatomotor(root)
e.set(rej='')
# print(e.load_evoked(subjects=-1, data='meg'))
print(e.load_test('connection', 0.3, 0.5, 0.05, data='meg', baseline=False, epoch='not_null', make=True))
