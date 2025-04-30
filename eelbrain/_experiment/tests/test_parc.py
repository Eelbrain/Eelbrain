# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain.pipeline import CombinationParc, SubParc


def test_combination_parc():
    parc = CombinationParc(
        'aparc',
        {'STG301': "split(transversetemporal + superiortemporal, 3)[:2]"}
    )
    assert parc._base_labels() == {'transversetemporal', 'superiortemporal'}
    parc = CombinationParc(
        'aparc',
        {
            'STG301': "split(transversetemporal + superiortemporal, 3)[:2]",
            'MTG301': "split(middletemporal, 3)[:2]",
        }
    )
    assert parc._base_labels() == {'transversetemporal', 'superiortemporal', 'middletemporal'}


def test_sub_parc():
    parc = SubParc('aparc', ('transversetemporal', 'superiortemporal'))
    assert parc._base_labels() == {'transversetemporal', 'superiortemporal'}
