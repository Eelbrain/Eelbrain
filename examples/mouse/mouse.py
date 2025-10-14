# skip test: data unavailable
from eelbrain.pipeline import *


class Mouse(Pipeline):

    # Name of the experimental session(s), used to locate *-raw.fif files
    sessions = 'CAT'

    # Pre-processing pipeline: each entry in `raw` specifies one processing step. The first parameter
    # of each entry specifies the source (another processing step or 'raw' for raw input data).
    raw = {
        # Maxwell filter as first step (taking input from raw data, 'raw')
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
        # Band-pass filter data between 1 and 40 Hz (taking Maxwell-filtered data as input, 'tsss)
        '1-40': RawFilter('tsss', 1, 40),
        # Perform ICA on filtered data
        'ica': RawICA('1-40', 'CAT', n_components=0.99),
    }

    # Variables determine how event triggeres are mapped to meaningful labels. Events are represented
    # as data-table in which each row corresponds to one event (i.e., one trigger). Each variable
    # defined here adds one column in that data-table, assigning a label or value to each event.
    variables = {
        # The first parameter specifies the source variable (here the trigger values),
        # the second parameter a mapping from source to target labels/values
        'stimulus': LabelVar('trigger', {(162, 163): 'target', (166, 167): 'prime'}),
        'prediction': LabelVar('trigger', {(162, 166): 'expected', (163, 167): 'unexpected'}),
    }

    # Epochs specify how to extract time-locked data segments ("epochs") from the continuous data.
    epochs = {
        # A PrimaryEpoch definition extracts epochs directly from continuous data. The first argument
        # specifies the recording session from which to extract the data (here: 'CAT'). The second
        # argument specifies which events to extract the data from (here: all events at which the
        # 'stimulus' variable, defined above, has a value of either 'prime' or 'target').
        'word': PrimaryEpoch('CAT', "stimulus.isin(('prime', 'target'))", samplingrate=200),
        # A secondary epoch inherits its properties from the base epoch ("word") unless they are
        # explicitly modified (here, selecting a subset of events)
        'prime': SecondaryEpoch('word', "stimulus == 'prime'"),
        'target': SecondaryEpoch('word', "stimulus == 'target'"),
        # The 'cov' epoch defines the data segments used to compute the noise covariance matrix for
        # source localization
        'cov': SecondaryEpoch('prime', tmax=0),
    }

    # Tests define contrasts or comparisons based on variables.
    # In the example, the following would test for the difference between expected and unexpected words in
    # target words:
    # >>> result = mouse.load_test(test='surprise', epoch='target')
    tests = {
        '=0': TTestOneSample(),
        'surprise': TTestRelated('prediction', 'unexpected', 'expected'),
        'anova': ANOVA('prediction * subject'),
    }

    # Define masks for permutation tests in anatomically constrained regions. Fro exmample:
    # >>> result = mouse.load_test(test='surprise', epoch='target', mask='lateraltemporal')
    parcs = {
        'STG': SubParc('aparc', ('transversetemporal', 'superiortemporal')),
        'IFG': SubParc('aparc', ('parsopercularis', 'parsorbitalis', 'parstriangularis')),
        'lateraltemporal': SubParc('aparc', (
            'transversetemporal', 'superiortemporal', 'bankssts',
            'middletemporal', 'inferiortemporal')),
    }


# Initialize the pipeline instance with the root path where the data is located
root = '~/Data/Mouse'
e = Mouse(root)
