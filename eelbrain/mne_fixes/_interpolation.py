# Mostly retaining MNE-Python functions to compensate for API changes
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.linalg import pinv

import mne

try:
    from mne.forward import _map_meg_or_eeg_channels
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "Eelbrain's MEG channel interpolation relies on the private MNE-Python "
        "function mne.forward._map_meg_or_eeg_channels, which is unavailable in "
        f"the installed MNE-Python version ({mne.__version__}). Please report "
        "this at https://github.com/christianbrodbeck/Eelbrain/issues"
    ) from error


def _calc_g(cosang, stiffness=4, n_legendre_terms=50):
    """Spherical spline g function (Perrin et al., 1989), as in MNE-Python."""
    factors = [
        (2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return legval(cosang, [0] + factors)


def _make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Spherical spline interpolation matrix mapping good to bad sensors.

    Implementation of Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF.
    (1989). Spherical splines for scalp potential and current density mapping.
    Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """
    pos_from = pos_from / np.linalg.norm(pos_from, axis=1, keepdims=True)
    pos_to = pos_to / np.linalg.norm(pos_to, axis=1, keepdims=True)
    n_from = len(pos_from)

    G_from = _calc_g(pos_from @ pos_from.T)
    G_to_from = _calc_g(pos_to @ pos_from.T)
    if alpha is not None:
        G_from.flat[::n_from + 1] += alpha

    C = np.block([[G_from, np.ones((n_from, 1))], [np.ones((1, n_from)), 0.]])
    interpolation = np.hstack([G_to_from, np.ones((len(pos_to), 1))]) @ pinv(C)[:, :-1]
    return interpolation


# mne 0.10 function
def map_meg_channels(inst, picks_good, picks_bad, mode):
    info_from = mne.pick_info(inst.info, picks_good, copy=True)
    info_to = mne.pick_info(inst.info, picks_bad, copy=True)
    return _map_meg_or_eeg_channels(info_from, info_to, mode=mode, origin='auto')


# private in 0.9.0 (Epochs method)
def get_channel_positions(self, picks=None):
    """Gets channel locations from info

    Parameters
    ----------
    picks : array-like of int | None
        Indices of channels to include. If None (default), all meg and eeg
        channels that are available are returned (bad channels excluded).
    """
    if picks is None:
        picks = mne.pick_types(self.info, meg=True, eeg=True)
    chs = self.info['chs']
    pos = np.array([chs[k]['loc'][:3] for k in picks])
    n_zero = np.sum(np.sum(np.abs(pos), axis=1) == 0)
    if n_zero > 1:  # XXX some systems have origin (0, 0, 0)
        raise ValueError(f'Could not extract channel positions for {n_zero} channels')
    return pos
