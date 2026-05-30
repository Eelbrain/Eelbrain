"""Channel-type metadata shared across the rejection/selection GUIs.

Unit display and scaling are handled by the general framework in
:mod:`eelbrain.plot._base` (:func:`scaled_unit`, :class:`AxisScale`), which
maps a data unit to its display unit. This module only holds the
channel-type-specific information that the plotting framework does not: which
MNE channels make up each type, the color used to draw them, the data unit each
type carries, and conservative single-trial default amplitudes.
"""
from ..plot._base import scaled_unit

# pick_types kwargs per channel type
CH_TYPE_PICK_KWARGS = {
    'mag':  {'meg': 'mag'},
    'grad': {'meg': 'grad'},
    'eeg':  {'meg': False, 'eeg': True},
}
# Colors for each channel type in butterfly plots
CH_TYPE_COLORS = {
    'mag':  'steelblue',
    'grad': 'forestgreen',
    'eeg':  'firebrick',
}
# SI data unit each channel type carries (matches NDVar.info['unit'] set by
# eelbrain._io.fiff; planar gradiometer data is in T/m, displayed as fT/cm).
CH_TYPE_DATA_UNIT = {
    'mag':  'T',
    'grad': 'T/m',
    'eeg':  'V',
}
# Conservative single-trial y-axis limits, in SI units.
CH_TYPE_DEFAULT_VLIM_SI = {
    'mag':  2000e-15,   # 2000 fT
    'grad': 30e-12,     # 300 fT/cm
    'eeg':  50e-6,      # 50 µV
}


def ch_type_scale(ch_type: str) -> tuple:
    """Return ``(display_unit, scale)`` for a channel type; ``display = scale * SI``."""
    return scaled_unit(CH_TYPE_DATA_UNIT.get(ch_type, ch_type))
