# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Find defective channels through gaps in ICA component maps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._data_obj import NDVar, Sensor


# Minimum spatial smoothness for a component map to count as "solid" (a realistic field
# pattern rather than channel noise), by channel type. Calibrated on ICA decompositions of
# sample_audvis_trunc_raw.fif: magnetometer maps are almost always smooth (r = 0.80-1.00,
# median 0.96), EEG maps span r = -0.12-0.98. Planar gradiometer maps are spatial
# derivatives and never exceed r = 0.70 with respect to the neuromag306planar adjacency
# graph (which additionally makes every channel adjacent to its co-located partner, so that
# only 47% of edges connect sensors with the same gradient orientation); gradiometers are
# therefore disabled by default, and their threshold is set just below the maximum that
# occurs in practice so that enabling them errs towards finding no solid component.
SMOOTHNESS_DEFAULT = {'mag': 0.85, 'grad': 0.60, 'eeg': 0.80}
CH_TYPE_DEFAULT = {'mag': True, 'grad': False, 'eeg': True}
GAP_RATIO_DEFAULT = 0.5  # |w[c]| <= this * mean(|w[neighbors]|)
MIN_COMPONENTS_DEFAULT = 2  # number of components that need to show the gap
CONSISTENCY_DEFAULT = 0.5  # n_evidence / n_testable
_SALIENCE = 0.15  # min(|w[neighbors]|) >= this * max(|w|); applies to every neighbor, so it
# is a stronger requirement than the same value applied to the neighborhood mean would be
_SIGN_CONSISTENCY = 0.8  # |mean(w[neighbors])| >= this * mean(|w[neighbors]|)
_MIN_VARIANCE = 0.001  # relative variance contribution floor
_MIN_NEIGHBORS = 3


@dataclass
class ChannelGap:
    """A channel that is missing from ICA component maps.

    Parameters
    ----------
    index
        Index of the channel in the sensor dimension.
    name
        Name of the channel.
    n_evidence
        Number of components in which the channel is a gap.
    n_testable
        Number of components in which the channel could be evaluated, i.e. in which every one
        of its neighbors carries a salient field, of consistent polarity.
    consistency
        ``n_evidence / n_testable``.
    gap
        Median relative weight over the testable components, i.e. the channel's weight in the
        polarity of its neighborhood divided by the average absolute neighbor weight: ~1 for a
        normal channel, ~0 for a channel that records nothing, and < 0 for a channel whose
        polarity is reversed relative to its neighbors.
    gap_components
        Components in which the channel is a gap, smallest gap first.
    no_gap_components
        Components in which the channel could be evaluated but is not a gap, smallest gap
        first.
    """
    index: int
    name: str
    n_evidence: int
    n_testable: int
    consistency: float
    gap: float
    gap_components: list[int]
    no_gap_components: list[int]


@dataclass
class ChannelGapResult:
    """Result of :func:`find_channel_gaps`.

    Parameters
    ----------
    ch_type
        Channel type that was analyzed.
    channels
        Channels that were flagged, ranked by consistency and evidence.
    smoothness
        Spatial smoothness of each component map ``(n_component,)``.
    solid
        Which components were used ``(n_component,)``.
    n_testable
        Number of testable components per channel ``(n_sensor,)``.
    n_evidence
        Number of components with a gap per channel ``(n_sensor,)``.
    consistency
        ``n_evidence / n_testable`` per channel ``(n_sensor,)``.
    """
    ch_type: str
    channels: list[ChannelGap]
    smoothness: np.ndarray
    solid: np.ndarray
    n_testable: np.ndarray
    n_evidence: np.ndarray
    consistency: np.ndarray


def neighbor_matrix(sensor: Sensor) -> tuple[np.ndarray, np.ndarray]:
    """Dense neighbor matrix and degree vector for a sensor dimension

    Parameters
    ----------
    sensor
        Sensor dimension; its adjacency needs to be defined.

    Returns
    -------
    matrix
        Symmetric ``(n_sensor, n_sensor)`` array which is 1 where two sensors are neighbors.
    degree
        Number of neighbors per sensor ``(n_sensor,)``.

    Notes
    -----
    Based on the adjacency graph rather than :meth:`Sensor.neighbors`, which is degenerate
    for co-located planar gradiometers (the distance to the partner sensor is 0, so no
    sensor qualifies as a neighbor).
    """
    n = len(sensor)
    matrix = np.zeros((n, n))
    edges = sensor.adjacency()
    matrix[edges[:, 0], edges[:, 1]] = 1
    matrix[edges[:, 1], edges[:, 0]] = 1
    return matrix, matrix.sum(1)


def map_smoothness(
        w: np.ndarray,
        matrix: np.ndarray,
        degree: np.ndarray,
) -> np.ndarray:
    """Correlation across sensors between each component map and its neighbor mean

    Parameters
    ----------
    w
        Component maps ``(n_component, n_sensor)``.
    matrix
        Neighbor matrix from :func:`neighbor_matrix`.
    degree
        Number of neighbors per sensor (0 replaced by 1 to avoid division by zero).

    Returns
    -------
    smoothness
        Correlation coefficient for each component ``(n_component,)``; 0 for constant maps.
    """
    neighbor_mean = (w @ matrix) / degree
    x = w - w.mean(1, keepdims=True)
    y = neighbor_mean - neighbor_mean.mean(1, keepdims=True)
    norm = np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (x * y).sum(1) / norm
    return np.nan_to_num(r)


def find_channel_gaps(
        components: NDVar,
        source_variance: np.ndarray = None,
        smoothness: float = 0.9,
        gap_ratio: float = GAP_RATIO_DEFAULT,
        min_components: int = MIN_COMPONENTS_DEFAULT,
        min_consistency: float = CONSISTENCY_DEFAULT,
        ch_type: str = None,
) -> ChannelGapResult:
    """Find channels that are missing from ICA component maps

    A channel that does not record any signal appears as a gap in component maps: its weight
    is ~0 where its neighbors carry a strong field. A weight of ~0 in a single component is
    not diagnostic, because the channel could be located on the null line of a polarity
    reversal. Two properties make it diagnostic: the weight is ~0 in multiple *solid*
    components (components reflecting realistic field patterns rather than channel noise),
    and it is ~0 while the surrounding channels share the same polarity.

    Parameters
    ----------
    components
        Component maps, with ``component`` and ``sensor`` dimensions. The sensor dimension
        needs to have adjacency defined.
    source_variance
        Variance of each component's source time course ``(n_component,)``. Used with the
        norm of the component map to estimate each component's contribution to the data
        variance, in order to exclude numerically degenerate components. If unspecified,
        this screen is skipped.
    smoothness
        Minimum correlation between a component map and its neighbor mean for the component
        to be considered solid (see :data:`SMOOTHNESS_DEFAULT` for values by channel type).
    gap_ratio
        Maximum relative weight for a channel to count as a gap (see :class:`ChannelGap`).
        Since the relative weight is signed, this detects channels that record nothing (~0) as
        well as channels whose polarity is reversed relative to all their neighbors (< 0).
        This is also the sensitivity floor: a channel whose gain is merely attenuated, to more
        than ~0.2-0.3 of normal, is not detected.
    min_components
        Minimum number of components in which a channel needs to be a gap (at least 1: a
        channel that is a gap in no component is not evidence of anything).
    min_consistency
        Minimum fraction of the testable components in which a channel needs to be a gap.
    ch_type
        Channel type, for labeling the result.

    Returns
    -------
    result
        Flagged channels along with the underlying statistics.

    Notes
    -----
    Both ``min_components`` and ``min_consistency`` are needed: on real data, healthy
    channels reach 2 components with a gap, and are separated from defective channels only
    by the fraction of testable components (0.67-1.00 for defective channels, ≤0.25 for
    healthy channels in ICA decompositions of ``sample_audvis_trunc_raw.fif``).

    Channels that are already excluded as bad are not part of the ICA decomposition, and
    hence can not be evaluated; an empty result does not imply that all excluded channels
    were rightly excluded.
    """
    if min_components < 1:
        raise ValueError(f"{min_components=}: a channel needs to be a gap in at least one component")
    w = components.get_data(('component', 'sensor')).astype(np.float64)
    sensor = components.get_dim('sensor')
    matrix, degree = neighbor_matrix(sensor)
    safe_degree = np.where(degree > 0, degree, 1.)

    # solid components: realistic field patterns rather than channel noise
    smoothness_by_component = map_smoothness(w, matrix, safe_degree)
    scale = np.sqrt((w ** 2).mean(1))  # arbitrary per-component scale
    solid = (smoothness_by_component >= smoothness) & (scale > 0)
    if source_variance is not None:
        variance = source_variance * (w ** 2).sum(1)
        total = variance.sum()
        if total > 0:
            solid &= variance / total >= _MIN_VARIANCE

    n_sensor = len(sensor)
    n_testable = np.zeros(n_sensor, int)
    n_evidence = np.zeros(n_sensor, int)
    consistency = np.zeros(n_sensor)
    channels = []
    result = ChannelGapResult(ch_type, channels, smoothness_by_component, solid, n_testable, n_evidence, consistency)
    if not solid.any():
        return result

    # normalize each map to unit RMS to remove the arbitrary per-component scale
    ws = w[solid] / scale[solid, None]
    abs_ws = np.abs(ws)
    neighbor_mean = (ws @ matrix) / safe_degree  # signed: polarity
    neighbor_abs = (abs_ws @ matrix) / safe_degree  # magnitude: reference level
    peak = abs_ws.max(1, keepdims=True)

    # Every neighbor needs to carry a strong field, not just the neighborhood on average:
    # at the edge of the sensor layout, where neighbors lie on one side only, their mean is
    # a biased reference.
    neighbor_min = np.zeros_like(abs_ws)
    for j in np.flatnonzero(degree):
        neighbor_min[:, j] = abs_ws[:, matrix[j] > 0].min(1)

    salient = neighbor_min >= _SALIENCE * peak  # neighborhood carries a real field
    uniform = np.abs(neighbor_mean) >= _SIGN_CONSISTENCY * neighbor_abs  # no polarity reversal
    testable = salient & uniform & (degree >= _MIN_NEIGHBORS)
    # Weight in the polarity of the neighborhood: ~1 for a channel that follows the field,
    # ≤ ~0 for a channel that records noise
    gap = np.sign(neighbor_mean) * ws / np.where(neighbor_abs > 0, neighbor_abs, 1.)
    evidence = testable & (gap <= gap_ratio)

    n_testable[:] = testable.sum(0)
    n_evidence[:] = evidence.sum(0)
    consistency[:] = n_evidence / np.maximum(n_testable, 1)
    flagged = (n_evidence >= min_components) & (consistency >= min_consistency)

    solid_components = np.flatnonzero(solid)
    for i in np.flatnonzero(flagged):
        gap_index = np.flatnonzero(evidence[:, i])
        no_gap_index = np.flatnonzero(testable[:, i] & ~evidence[:, i])
        gap_index = gap_index[np.argsort(gap[gap_index, i])]
        no_gap_index = no_gap_index[np.argsort(gap[no_gap_index, i])]
        channels.append(ChannelGap(
            index=int(i),
            name=sensor.names[i],
            n_evidence=int(n_evidence[i]),
            n_testable=int(n_testable[i]),
            consistency=float(consistency[i]),
            gap=float(np.median(gap[testable[:, i], i])),
            gap_components=[int(c) for c in solid_components[gap_index]],
            no_gap_components=[int(c) for c in solid_components[no_gap_index]],
        ))
    channels.sort(key=lambda channel: (channel.consistency, channel.n_evidence), reverse=True)
    return result
