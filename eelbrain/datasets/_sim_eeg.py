# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Simulate EEG data"""
import numpy as np
import scipy.signal
import scipy.spatial

from .._data_obj import Dataset, Factor, Var, NDVar, Sensor, UTS
from .._ndvar import gaussian, powerlaw_noise
from .. import _info


def _topo(sensor, center):
    i = sensor.names.index(center)
    loc = sensor.locations[i]
    dists = scipy.spatial.distance.cdist([loc], sensor.locations)[0]
    radius = sensor._sphere_fit[1].mean()
    dists /= radius
    topo = 1.0 - dists
    return NDVar(topo, (sensor,))


def simulate_erp(
        n_trials: int = 80,
        seed: int = 0,
        snr: float = 0.2,
        sensor: Sensor = None,
        time: UTS = None,
        short_long: bool = False,
) -> Dataset:
    """Simulate event-related EEG data

    Parameters
    ----------
    n_trials
        Number of trials (default 100).
    seed
        Random seed.
    snr
        Signal-to-noise ratio.
    sensor
        Sensor dimension.
    time
        Time dimension.
    short_long
        Create bimodal distribution of word length for categorial analysis.

    Examples
    --------
    Compare with kiloword::

        ys = datasets.simulate_erp()['eeg']
        ys -= ys.mean(sensor=['M1', 'M2'])
        import mne
        path = mne.datasets.kiloword.data_path()
        y = load.mne.epochs_ndvar(path + '/kword_metadata-epo.fif')
        plot.TopoButterfly([y, ys])

    """
    assert n_trials % 2 == 0

    if sensor is None:
        sensor = Sensor.from_montage('standard_alphabetic')
        sensor.set_adjacency(connect_dist=1.66)
    if time is None:
        time = UTS(-0.100, 0.005, 140)

    # Generate random values for the independent variable (cloze probability)
    rng = np.random.RandomState(seed)
    cloze_x = np.concatenate([
        rng.uniform(0, 0.3, n_trials // 2),
        rng.uniform(0.8, 1.0, n_trials // 2),
    ])
    # Word length (number of characters)
    if short_long:
        assert not n_trials % 4
        n_condition = n_trials // 4
        params = [(4, 1.5), (8, 2), (4, 1.5), (8, 2)]
        n_chars = np.concatenate([rng.normal(loc, scale, n_condition) for loc, scale in params])
        n_chars = np.clip(n_chars, 2, None)
    else:
        rng.shuffle(cloze_x)
        n_chars = rng.uniform(3, 7, n_trials)
    cloze = Var(cloze_x)
    n_chars = Var(np.round(n_chars)).astype(int)

    # Generate topography
    n400_topo = -2.0 * _topo(sensor, 'Cz')
    # Generate timing
    n400_timecourse = gaussian(0.400, 0.034, time)
    # Put all the dimensions together to simulate the EEG signal
    surprisal = -cloze.log(2)
    signal = surprisal * 0.25 * n400_timecourse * n400_topo

    # add early responses:
    # 130 ms
    tc = gaussian(0.130, 0.025, time)
    topo = _topo(sensor, 'O1') + _topo(sensor, 'O2') - 0.5 * _topo(sensor, 'Cz')
    signal += (n_chars * 0.5) * tc * topo
    # 195 ms
    amp = Var(rng.normal(0.8, 1, n_trials))
    tc = gaussian(0.195, 0.015, time)
    topo = 1.2 * _topo(sensor, 'F3') + _topo(sensor, 'F4')
    signal += amp * tc * topo
    # 270
    amp = Var(rng.normal(1., 1, n_trials))
    tc = gaussian(0.270, 0.050, time)
    topo = _topo(sensor, 'O1') + _topo(sensor, 'O2')
    signal += amp * tc * topo
    # 280
    amp = Var(rng.normal(-1, 1, n_trials))
    tc = gaussian(0.280, 0.030, time)
    topo = _topo(sensor, 'Pz')
    signal += amp * tc * topo
    # 600
    amp = Var(rng.normal(0.5, 1, n_trials))
    tc = gaussian(0.590, 0.100, time)
    topo = -_topo(sensor, 'Fz')
    signal += amp * tc * topo

    # Add noise
    noise = powerlaw_noise(signal, 1, rng)
    noise = noise.smooth('sensor', 0.02, 'gaussian')
    noise *= (signal.std() / noise.std() / snr)
    signal += noise

    # Data scale
    signal *= 1e-6
    signal.info = _info.for_eeg()

    # Store EEG data in a Dataset with trial information
    ds = Dataset()
    ds['eeg'] = signal
    ds['cloze'] = Var(cloze_x)
    ds['predictability'] = Factor(cloze_x > 0.5, labels={True: 'high', False: 'low'})
    ds['n_chars'] = Var(n_chars)
    if short_long:
        ds['length'] = Factor(['short', 'long'], repeat=n_condition, tile=2)
    return ds
