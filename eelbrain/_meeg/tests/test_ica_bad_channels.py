# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tests for finding defective channels through gaps in ICA component maps."""
import numpy as np
import pytest

from eelbrain import NDVar, Scalar, Sensor
from eelbrain._meeg import find_channel_gaps
from eelbrain._meeg.ica_bad_channels import SMOOTHNESS_DEFAULT, _map_smoothness, _neighbor_matrix


SMOOTHNESS = SMOOTHNESS_DEFAULT['eeg']


def _sensor():
    return Sensor.from_montage('biosemi64')


def _dipole_map(sensor: Sensor, r0: np.ndarray, p: np.ndarray) -> np.ndarray:
    "Field of a current dipole at ``r0`` with moment ``p``, normalized to unit maximum"
    d = sensor.locations - r0
    v = (d * p).sum(1) / (d ** 2).sum(1) ** 1.5
    return v / np.abs(v).max()


def _dipole_maps(sensor: Sensor, n: int, seed: int = 0) -> np.ndarray:
    "``n`` smooth topographies from random current dipoles inside the head"
    rng = np.random.RandomState(seed)
    radius = np.linalg.norm(sensor.locations, axis=1).mean()
    maps = []
    while len(maps) < n:
        direction = rng.uniform(-1, 1, 3)
        r0 = direction / np.linalg.norm(direction) * rng.uniform(0.2, 0.5) * radius
        p = rng.uniform(-1, 1, 3)
        maps.append(_dipole_map(sensor, r0, p / np.linalg.norm(p)))
    return np.array(maps)


def _spike_map(sensor: Sensor, channel: str, seed: int = 0) -> np.ndarray:
    "Component loading on a single channel (channel noise rather than a field pattern)"
    rng = np.random.RandomState(seed)
    x = 0.01 * rng.randn(len(sensor))
    x[sensor.names.index(channel)] = 1.
    return x


def _components(sensor: Sensor, maps: np.ndarray, seed: int = 0) -> NDVar:
    "Component maps with arbitrary per-component scale, as in an ICA decomposition"
    scale = np.random.RandomState(seed).uniform(0.1, 10, (len(maps), 1))
    return NDVar(maps * scale, (Scalar('component', np.arange(len(maps))), sensor), 'components')


def test_neighbor_matrix():
    "Neighbor matrix is symmetric and consistent with the adjacency graph"
    sensor = _sensor()
    matrix, degree = _neighbor_matrix(sensor)
    edges = sensor.adjacency()
    assert np.array_equal(matrix, matrix.T)
    assert matrix.diagonal().sum() == 0
    assert matrix.sum() == 2 * len(edges)
    assert np.array_equal(degree, np.bincount(edges.ravel(), minlength=len(sensor)))


def test_map_smoothness():
    "Smoothness separates field patterns from single-channel noise"
    sensor = _sensor()
    matrix, degree = _neighbor_matrix(sensor)
    r = _map_smoothness(_dipole_maps(sensor, 15), matrix, degree)
    assert r.min() > SMOOTHNESS
    r = _map_smoothness(_spike_map(sensor, 'Pz')[None], matrix, degree)
    assert abs(r[0]) < 0.2
    # constant map -> 0 rather than nan
    r = _map_smoothness(np.ones((1, len(sensor))), matrix, degree)
    assert r[0] == 0


def test_find_channel_gaps():
    "A channel that is 0 in all component maps is detected"
    sensor = _sensor()
    maps = np.vstack([_dipole_maps(sensor, 15), _spike_map(sensor, 'C4')[None]])
    maps[:, sensor.names.index('Pz')] = 0
    result = find_channel_gaps(_components(sensor, maps), smoothness=SMOOTHNESS, ch_type='eeg')
    assert [channel.name for channel in result.channels] == ['Pz']
    channel = result.channels[0]
    assert channel.consistency == 1
    assert channel.n_evidence == channel.n_testable > 1
    assert channel.gap < 0.01
    # the single-channel component is not a realistic field pattern
    assert not result.solid[-1]
    assert result.solid[:-1].mean() > 0.8


def test_find_channel_gaps_polarity_reversal():
    "A channel on the null line of every component is not flagged"
    sensor = _sensor()
    # dipoles below Cz with tangential moments produce a polarity reversal at Cz
    radial = sensor.locations[sensor.names.index('Cz')]
    radial = radial / np.linalg.norm(radial)
    tangential = np.cross(radial, [1., 0, 0])
    tangential /= np.linalg.norm(tangential)
    other = np.cross(radial, tangential)
    maps = []
    for angle in np.linspace(0, np.pi, 12, endpoint=False):
        p = np.cos(angle) * tangential + np.sin(angle) * other
        maps.append(_dipole_map(sensor, 0.5 * radial * np.linalg.norm(sensor.locations[0]), p))
    result = find_channel_gaps(_components(sensor, np.array(maps)), smoothness=SMOOTHNESS, ch_type='eeg')
    i = sensor.names.index('Cz')
    # Cz has a gap in every component, but is never testable
    assert result.n_testable[i] == 0
    assert not result.channels


def test_find_channel_gaps_partial():
    "gap_ratio is the sensitivity floor: a partially attenuated channel is not detected"
    sensor = _sensor()
    maps = _dipole_maps(sensor, 15)
    maps[:, sensor.names.index('Pz')] *= 0.5
    result = find_channel_gaps(_components(sensor, maps), smoothness=SMOOTHNESS, ch_type='eeg')
    assert not result.channels


def test_find_channel_gaps_variance():
    "Components with a negligible variance contribution are excluded"
    sensor = _sensor()
    maps = _dipole_maps(sensor, 15)
    maps[:, sensor.names.index('Pz')] = 0
    components = _components(sensor, maps)
    source_variance = np.ones(15)
    source_variance[5:] = 1e-9
    result = find_channel_gaps(components, source_variance, SMOOTHNESS, ch_type='eeg')
    assert not result.solid[5:].any()
    assert result.solid.sum() < find_channel_gaps(components, smoothness=SMOOTHNESS).solid.sum()
    assert [channel.name for channel in result.channels] == ['Pz']


def test_find_channel_gaps_no_adjacency():
    "Sensor adjacency is required"
    sensor = _sensor()
    sensor = Sensor(sensor.locations, sensor.names, adjacency='none')
    components = _components(sensor, _dipole_maps(sensor, 5))
    with pytest.raises(RuntimeError):
        find_channel_gaps(components)
