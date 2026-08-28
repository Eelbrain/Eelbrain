# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne
import pytest

from eelbrain._exceptions import ConfigurationError
from eelbrain._experiment.preprocessing import RawSource


def test_raw_source_rename_channels():
    "rename_channels renames the montage and builtin adjacency, not the data"
    rename = {'A1': 'Fp1', 'A2': 'Fz', 'A3': 'Cz'}
    raw = RawSource(montage='biosemi16', rename_channels=rename, adjacency='biosemi16')
    # Montage uses data names
    for data_name, montage_name in rename.items():
        assert data_name in raw.montage.ch_names
        assert montage_name not in raw.montage.ch_names
    # Builtin adjacency is resolved to an edge list with data names
    assert isinstance(raw.adjacency, list)
    adjacency_names = {name for pair in raw.adjacency for name in pair}
    assert 'A1' in adjacency_names
    assert 'Fp1' not in adjacency_names
    # Fp2 is not renamed and keeps its montage name
    assert 'Fp2' in adjacency_names
    # Renamed and original adjacency describe the same graph
    _, ch_names = mne.channels.read_ch_adjacency('biosemi16')
    reverse = {data_name: montage_name for data_name, montage_name in rename.items()}
    original = RawSource(montage='biosemi16', adjacency='biosemi16')
    assert original.adjacency == 'biosemi16'
    renamed_back = sorted(tuple(sorted((reverse.get(a, a), reverse.get(b, b)))) for a, b in raw.adjacency)
    coo = mne.channels.read_ch_adjacency('biosemi16')[0].tocoo()
    expected = sorted({tuple(sorted((ch_names[min(i, j)], ch_names[max(i, j)]))) for i, j in zip(coo.row, coo.col) if i != j})
    assert renamed_back == expected

    # rename_channels requires a montage
    with pytest.raises(ConfigurationError):
        RawSource(rename_channels=rename)
    # rename_channels values need to be in the montage
    with pytest.raises(ConfigurationError):
        RawSource(montage='biosemi16', rename_channels={'A1': 'NoSuchChannel'})
