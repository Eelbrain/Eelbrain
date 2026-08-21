"""Detect abnormal-power time windows in continuous EEG data.

Python port of the window-detection part of the ``clean_rawdata`` MATLAB/EEGLAB
toolbox (the channel-cleaning functions ``clean_flatlines``/``clean_channels*``
are not part of this port).

Reference
---------
Christian Kothe, SCCN/UCSD — https://github.com/sccn/clean_rawdata
"""
from __future__ import annotations

import mne
import numpy as np
from scipy.special import gamma as _gamma
from scipy.special import gammaincinv as _gammaincinv


def fit_eeg_distribution(
        X: np.ndarray,
        min_clean_fraction: float = 0.25,
        max_dropout_fraction: float = 0.1,
        quants: tuple[float, float] = (0.022, 0.6),
        step_sizes: tuple[float, float] = (0.01, 0.01),
        shape_range: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Estimate the mean and std of clean EEG from potentially contaminated data.

    Fits a truncated generalized Gaussian via a grid search minimizing KL
    divergence, as in Kothe & Jung (2016). Used internally by
    :func:`clean_windows`.

    Parameters
    ----------
    X
        1-D array of RMS amplitude values (one per window, per channel).
    min_clean_fraction
        Minimum fraction of ``X`` assumed to be clean EEG.
    max_dropout_fraction
        Maximum fraction of ``X`` allowed to be sensor-dropout (near-zero)
        artifacts.
    quants
        Lower and upper quantile limits (``q_low``, ``q_high``) of the
        truncated distribution.
    step_sizes
        Grid-search step sizes for (``lower_offset``, ``width``).
    shape_range
        Candidate beta (shape) values for the generalized Gaussian. Defaults
        to ``np.arange(1.7, 3.65, 0.15)``.

    Returns
    -------
    mu, sigma, alpha, beta : float
        Location, scale (std), scale parameter, and shape parameter of the
        fitted distribution.
    """
    if shape_range is None:
        shape_range = np.arange(1.7, 3.65, 0.15)

    X = np.sort(X.ravel().astype(float))
    n = len(X)

    q = np.asarray(quants, dtype=float)
    lower_min = q.min()
    max_width = np.diff(q)[0]
    min_width = min_clean_fraction * max_width

    # z-bounds and rescale constants for each beta value
    zbounds = []
    rescale = []
    for b in shape_range:
        s = np.sign(q - 0.5)
        z = s * _gammaincinv(1.0 / b, s * (2 * q - 1)) ** (1.0 / b)
        zbounds.append(z)
        rescale.append(b / (2 * _gamma(1.0 / b)))

    # build shifted data matrix: rows=quantile range, cols=lower-offset variants
    lower_offsets = np.arange(lower_min, lower_min + max_dropout_fraction + 1e-9, step_sizes[0])
    col_starts = np.round(n * lower_offsets).astype(int)
    n_rows = int(round(n * max_width))
    # guard against going out of bounds
    col_starts = np.clip(col_starts, 0, n - n_rows - 1)
    # shape: (n_rows, n_offsets)
    Xmat = np.column_stack([X[s: s + n_rows] for s in col_starts])
    X1 = Xmat[0, :]                 # minimum in each column
    Xmat = Xmat - X1                # shift so minimum is 0

    opt_val = np.inf
    opt_beta_val = shape_range[0]

    width_range = np.round(n * np.arange(max_width, min_width - 1e-9, -step_sizes[1])).astype(int)
    width_range = width_range[width_range >= 2]

    for m in width_range:
        nbins = max(1, int(round(3 * np.log2(1 + m / 2))))
        Xm = Xmat[m - 1, :]  # (n_offsets,) — range in each column at width m
        safe = Xm > 0
        if not np.any(safe):
            continue
        # scale to [0, nbins]
        H = Xmat[:m, safe] * nbins / Xm[safe]  # (m, n_safe)

        # histogram all columns at once
        H_int = np.clip(np.floor(H).astype(int), 0, nbins - 1)
        log_q = np.zeros((nbins, H_int.shape[1]))
        for bi in range(nbins):
            log_q[bi] = np.sum(H_int == bi, axis=0)
        log_q = np.log(log_q + 0.01)

        for bi, b in enumerate(shape_range):
            bounds = zbounds[bi]
            x_bins = bounds[0] + (np.arange(0.5, nbins) / nbins) * np.diff(bounds)[0]
            p = np.exp(-np.abs(x_bins) ** b) * rescale[bi]
            p = p / p.sum()                             # (nbins,)

            kl = (p[:, None] * (np.log(p[:, None]) - log_q)).sum(axis=0) + np.log(m)
            idx = np.argmin(kl)
            min_val = kl[idx]
            if min_val < opt_val:
                opt_val = min_val
                opt_beta_val = b
                opt_bounds = bounds
                safe_cols = np.where(safe)[0]
                opt_lu = np.array([X1[safe_cols[idx]], X1[safe_cols[idx]] + Xmat[m - 1, safe_cols[idx]]])

    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)[0]
    mu = opt_lu[0] - opt_bounds[0] * alpha
    sig = np.sqrt(alpha ** 2 * _gamma(3.0 / opt_beta_val) / _gamma(1.0 / opt_beta_val))
    return float(mu), float(sig), float(alpha), float(opt_beta_val)


def _bad_runs(bad_mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end]`` runs of ``True`` in ``bad_mask`` (both inclusive)

    Robust to runs touching either edge of the array.
    """
    padded = np.concatenate([[True], bad_mask, [True]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0][:-1]
    ends = np.where(edges == -1)[0][1:] - 1  # inclusive index of the last bad sample
    return list(zip(starts, ends))


def clean_windows(
        raw: mne.io.BaseRaw,
        max_bad_channels: float = 0.2,
        zthresholds: tuple[float, float] = (-3.5, 5.0),
        window_len: float = 1.0,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        truncate_quant: tuple[float, float] = (0.022, 0.6),
        step_sizes: tuple[float, float] = (0.01, 0.01),
        picks: str | list[str] | None = "eeg",
) -> list[tuple[str, float, float]]:
    """Find time windows with abnormally high or low EEG power.

    For each channel and window, the RMS amplitude is z-scored relative to a
    robust estimate of the clean EEG distribution (fitted via
    :func:`fit_eeg_distribution`). A window counts as bad when more than
    ``max_bad_channels`` channels exceed ``zthresholds`` simultaneously; the
    channels reported for such a window are those that individually exceed
    ``zthresholds`` in it (which can be more than ``max_bad_channels``).

    Parameters
    ----------
    raw
        Continuous MNE Raw object (high-passed ≥ 1 Hz recommended).
    max_bad_channels
        Max fraction (< 1) or count (≥ 1) of bad channels allowed per
        retained window (default 0.2, i.e. 20%).
    zthresholds
        ``(z_low, z_high)`` power z-score tolerances. Windows with a channel
        outside this range count toward ``max_bad_channels``.
    window_len
        Window length in seconds.
    window_overlap
        Fraction of overlap between successive windows.
    max_dropout_fraction
        Max fraction of windows allowed to have near-zero amplitude.
    min_clean_fraction
        Min fraction of windows expected to be artifact-free.
    truncate_quant
        Quantile range for the truncated GGD fit.
    step_sizes
        Grid-search step sizes for distribution fitting.
    picks
        Channel selection passed to :meth:`~mne.io.Raw.get_data`.

    Returns
    -------
    bad_intervals
        ``(channel, tmin, tmax)`` triples, one per channel per detected bad
        time interval, raw-local (0 at ``raw``'s own first sample; ``tmax``
        exclusive).
    """
    ch_names = raw.copy().pick(picks).ch_names
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=picks).astype(float)  # (C, S)
    C, S = data.shape

    if 0 < max_bad_channels < 1:
        max_bad_channels = int(round(C * max_bad_channels))
    else:
        max_bad_channels = int(max_bad_channels)

    N = int(window_len * sfreq)
    step = int(N * (1 - window_overlap))
    offsets = np.arange(0, S - N + 1, max(1, step))

    # per-channel z-scored RMS over windows
    wz = np.zeros((C, len(offsets)))
    for ci in range(C):
        rms = np.array([np.sqrt(np.mean(data[ci, o: o + N] ** 2)) for o in offsets])
        mu, sig, _, _ = fit_eeg_distribution(
            rms, min_clean_fraction, max_dropout_fraction, truncate_quant, step_sizes,
        )
        wz[ci] = (rms - mu) / max(sig, 1e-12)

    # a window counts as bad when more than max_bad_channels channels are
    # simultaneously out of range (sorting discards channel identity, which
    # is only needed for the count, not for the per-channel attribution below)
    swz = np.sort(wz, axis=0)
    remove_mask = np.zeros(len(offsets), dtype=bool)
    z_low, z_high = zthresholds
    if z_high > 0:
        remove_mask |= swz[-(max_bad_channels + 1), :] > z_high
    if z_low < 0:
        remove_mask |= swz[max_bad_channels, :] < z_low
    if not remove_mask.any():
        return []

    # channels individually out of range, restricted to windows that count as bad
    exceeds = np.zeros_like(wz, dtype=bool)
    if z_high > 0:
        exceeds |= wz > z_high
    if z_low < 0:
        exceeds |= wz < z_low
    exceeds &= remove_mask

    bad_intervals = []
    for ci in range(C):
        bad_mask = np.zeros(S, dtype=bool)
        for wi in np.where(exceeds[ci])[0]:
            o = offsets[wi]
            bad_mask[o: o + N] = True
        for start, end in _bad_runs(bad_mask):
            bad_intervals.append((ch_names[ci], start / sfreq, (end + 1) / sfreq))
    return bad_intervals
