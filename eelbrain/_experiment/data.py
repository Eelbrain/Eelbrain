# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Data description shared by epoch, source, and statistics derivatives."""
from __future__ import annotations

import re

import mne


class DataSpec:
    """Internal description of the data going into an analysis

    Combines the data *space* (sensor vs source, determined by the ``inv``
    state) with the data *kind* (sensor type and/or aggregation). Instances are
    composed at the :class:`Pipeline` boundary from the user-facing ``data``
    argument and the ``inv`` state; see ``Pipeline._resolve_data``.

    Parameters
    ----------
    string
        Internal string describing the data: ``'sensor'``/``'source'`` (or a
        specific sensor type ``'meg'``/``'mag'``/``'grad'``/``'eeg'``), with an
        optional ``.mean``/``.rms`` aggregation suffix.
    """
    RE = re.compile(r"^(source|sensor|meg|mag|grad|eeg)(?:\.(mean|rms))?$")
    source = False
    sensor = False

    def __init__(self, string: str):
        self.string = string
        m = self.RE.match(string)
        if m is None:
            raise ValueError(f"data={string!r}: invalid data description")
        dim, self.aggregate = m.groups()
        if dim in ('meg', 'mag'):
            self._to_ndvar = ('mag',)
            self.y_name = 'meg'  # mag NDVars are keyed 'meg' (see .load_epochs())
            self.sensor = True
        elif dim in ('grad', 'eeg'):
            self._to_ndvar = (dim,)
            self.y_name = dim
            self.sensor = True
        elif dim == 'sensor':
            self._to_ndvar = None
            self.y_name = 'meg'
            self.sensor = True
        elif dim == 'source':
            self._to_ndvar = None
            self.y_name = 'src'
            self.source = True
        else:
            raise RuntimeError(f"{string=} ({dim=})")

    @classmethod
    def coerce(cls, obj):
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, dict):
            # canonical form from _cache_form_()
            return cls(obj['string'])
        else:
            return cls(obj)

    def _cache_form_(self) -> str:
        """Canonical form for cache keys/fingerprints/manifests"""
        return self.string

    def __repr__(self):
        return f"DataSpec({self.string!r})"

    def __eq__(self, other):
        return isinstance(other, DataSpec) and self.string == other.string

    def _testnd_parc(self, disconnect_labels: bool) -> str | None:
        if self.source and not self.aggregate:
            return 'source' if disconnect_labels else None
        if disconnect_labels:
            raise TypeError(f"{disconnect_labels=}: invalid for data={self.string!r}")
        return None

    def data_to_ndvar(self, info: mne.Info) -> list[str]:
        assert self.sensor
        if self._to_ndvar is None:
            return info.get_channel_types(unique=True, only_data_chs=True)
        else:
            return self._to_ndvar
