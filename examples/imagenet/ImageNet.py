# skip test: data unavailable
from eelbrain.pipeline import MneExperiment, RawFilter, RawICA, LabelVar, PrimaryEpoch, SecondaryEpoch, TTestOneSample, TTestRelated, ANOVA


class ImageNet(MneExperiment):

    preload = True

    ignore_entities = {
        'ignore_subjects': ('02', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 'emptyroom'),
        'ignore_sessions': ('ImageNet02', 'ImageNet03', 'ImageNet04', 'MRI'),
        'ignore_runs': ('02'),
    }

    raw = {
        '1-40': RawFilter('raw', 1, 40),
        'ica': RawICA('1-40', 'ImageNet', n_components=0.99),
    }

    variables = {
        'position': LabelVar('trigger', {1: 'begin', 2: 'end', (3, 4): 'middle'}),
        'event': LabelVar('trigger', {(1, 2): 'unused', 3: 'resp', 4: 'stim_on'}),
    }

    epochs = {
        'used': PrimaryEpoch('ImageNet', "position == 'middle'", samplingrate=200),
        'resp': SecondaryEpoch('used', "event == 'resp'"),
        'stim_on': SecondaryEpoch('used', "event == 'stim_on'"),
        'cov': SecondaryEpoch('used', tmax=0),
    }

    tests = {
        '=0': TTestOneSample(),
        'connection': TTestRelated('event', 'stim_on', 'resp'),
        'anova': ANOVA('event * subject'),
    }


root = '/mnt/d/Data/ds005810'
e = ImageNet(root)
e.set(rej='', epoch='used')
# e.make_ica()
# e.make_epoch_selection(auto=True)
# e.load_raw(preload=True)
# print(e.load_raw(preload=True))
# print(e.load_evoked_stc(subjects=-1))
print(e.load_test('connection', 0.3, 0.5, 0.05, data='source', baseline=False, epoch='used', make=True))
