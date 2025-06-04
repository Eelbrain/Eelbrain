from eelbrain.pipeline import MneExperiment, PrimaryEpoch, SecondaryEpoch
from eelbrain.testing import path


class Experiment(MneExperiment):

    sessions = ('cheese', 'pets')

    epochs = {
        'cheese': PrimaryEpoch('cheese'),
        'hard-cheese': SecondaryEpoch('cheese', "texture == 'hard'"),
        'avians': PrimaryEpoch('pets', "has_wings"),
    }


def test_experiment():
    e = Experiment()

    # epoch/session interaction
    assert e._glob_pattern('fwd-file', True) == path('/eelbrain-cache/raw/*/*-*-*-fwd.fif')
    assert e._glob_pattern('fwd-file', True, session='pets') == path('/eelbrain-cache/raw/*/pets-*-*-fwd.fif')
    assert e._glob_pattern('fwd-file', True, epoch='hard-cheese') == path('/eelbrain-cache/raw/*/cheese-*-*-fwd.fif')
