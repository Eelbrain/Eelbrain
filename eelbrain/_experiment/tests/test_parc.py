# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain._exceptions import ConfigurationError
from eelbrain.pipeline import CombinationParc, IndividualSeededParc, SubParc


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


def test_seeded_parc_uses_stored_name():
    parc = IndividualSeededParc({'seed-lh': {'R0001': (-54, 10, -8)}})
    parc._store_name('seeded')

    assert parc.name == 'seeded'
    assert 'name' not in parc._as_dict()
    with pytest.raises(ConfigurationError, match="Parcellation seeded not defined for subject R0002"):
        parc._seeds_for_subject('R0002')
