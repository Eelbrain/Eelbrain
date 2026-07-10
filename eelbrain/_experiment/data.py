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
    string : str
        Internal string describing the data: ``'sensor'``/``'source'`` (or a
        specific sensor type ``'meg'``/``'mag'``/``'grad'``/``'eeg'``), with an
        optional ``.mean``/``.rms`` aggregation suffix.
    time : bool
        Whether the base data contains a time axis.
    morph : bool
        If loading source space data, whether the data is morphed to the common
        brain.
    """
    RE = re.compile(r"^(source|sensor|meg|mag|grad|eeg)(?:\.(mean|rms))?$")
    source = False
    sensor = False
    aggregate = None  # None, 'mean', or 'rms'

    def __init__(self, string, time=True, morph=False):
        self.time = bool(time)
        self.morph = bool(morph)
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
            self.y_name = 'srcm' if self.morph else 'src'
            self.source = True
        else:
            raise RuntimeError(f"{string=} ({dim=})")

        dims = []
        if self.source and not self.aggregate:
            dims.append('source')
        elif self.sensor and not self.aggregate:
            dims.append('sensor')
        if self.time:
            dims.append('time')
        self.dims = tuple(dims)

        # whether parc is used from subjects or from common-brain
        if self.source and not self.aggregate:
            self.parc_level = 'common'
        elif self.source:
            self.parc_level = 'individual'
        else:
            self.parc_level = None

    @classmethod
    def coerce(cls, obj, time=True, morph=False):
        if isinstance(obj, cls):
            if obj.time == time and obj.morph == morph:
                return obj
            else:
                return cls(obj.string, time, morph)
        elif isinstance(obj, dict):
            # canonical form from _cache_form_(); complete, so time/morph args are ignored
            return cls(obj['string'], obj.get('time', True), obj.get('morph', False))
        else:
            return cls(obj, time, morph)

    def _cache_form_(self) -> dict:
        """Simple canonical form for cache keys/fingerprints/manifests (see :meth:`~.derivative_cache.DerivativeRegistry.canonicalize`); :func:`normalize_data_option` parses it back."""
        return {'string': self.string, 'time': self.time, 'morph': self.morph if self.source else False}

    def __repr__(self):
        args = [repr(self.string)]
        if not self.time:
            args.append('time=False')
        if self.source and self.morph:
            args.append('morph=True')
        return f"DataSpec({', '.join(args)})"

    def __eq__(self, other):
        if not isinstance(other, DataSpec):
            return False
        elif self.string != other.string or self.time != other.time:
            return False
        elif self.source:
            return self.morph == other.morph
        return True

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


def normalize_data_option(ctx, value) -> DataSpec:
    """:class:`~.derivative_cache.OptionSpec` normalizer for ``data`` options holding a :class:`DataSpec`.

    Accepts a :class:`DataSpec` (returned unchanged — unlike
    :meth:`DataSpec.coerce`, whose ``time``/``morph`` arguments are
    authoritative and would rebuild the spec), its canonical dict form from
    :meth:`DataSpec._cache_form_` (an offline-reconstructed request), or a
    plain data string. Idempotent, as :class:`~.derivative_cache.OptionSpec`
    requires.
    """
    if isinstance(value, DataSpec):
        return value
    return DataSpec.coerce(value)
