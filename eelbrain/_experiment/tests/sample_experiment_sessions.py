# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# flake8: noqa
"""
A copy of sample_experiment.py with multiple sessions.
To produce the data directory for this experiment use (make sure
that the directory you specify exists)::

    >>> from eelbrain import datasets
    >>> datasets.setup_samples_experiment('~/Data', n_segments=2, n_sessions=2, name='SampleExperimentSessions')

Then you can use::

    >>> from sample_experiment_sessions import SampleExperiment, ROOT
    >>> e = SampleExperiment(ROOT)

"""
from eelbrain.pipeline import *


ROOT = "~/Data/SampleExperimentSessions"


class SampleExperiment(MneExperiment):

    meg_system = 'neuromag306mag'
    stim_channel = 'STI 014'

    sessions = ('sample1', 'sample2')

    raw = {
        '0-40': RawFilter('raw', None, 40, method='iir'),
        '1-40': RawFilter('raw', 1, 40, method='iir'),
    }

    variables = {
        'event': {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'},
        'side': {(1, 3): 'left', (2, 4): 'right'},
        'modality': {(1, 2): 'auditory', (3, 4): 'visual'}
    }

    epochs = {
        'target1': PrimaryEpoch('sample1', "event == 'target'", decim=5),
        'target2': PrimaryEpoch('sample2', "event == 'target'", decim=5),
        'super': SuperEpoch(('target1', 'target2')),
    }
