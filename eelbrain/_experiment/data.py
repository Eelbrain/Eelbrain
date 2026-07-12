# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Data description shared by epoch, source, and statistics derivatives."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mne

if TYPE_CHECKING:
    from .._data_obj import Dataset


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

    Notes
    -----
    ``DataSpec`` is the single source of truth for the keys under which response
    NDVars are stored in loaded :class:`Dataset`\\ s. Sensor NDVars are keyed by
    their MNE channel type (``'mag'``/``'grad'``/``'eeg'``) and source estimates
    by ``'src'``. Consumers retrieve the response NDVar key from a loaded dataset
    via :meth:`response_key`.
    """
    RE = re.compile(r"^(source|sensor|meg|mag|grad|eeg)(?:\.(mean|rms))?$")
    source = False
    sensor = False

    def __init__(self, string: str):
        self.string = string
        m = self.RE.match(string)
        if m is None:
            raise ValueError(f"data={string!r}: invalid data description")
        self.space, self.aggregate = m.groups()
        self.source = self.space == 'source'
        self.sensor = not self.source

    @classmethod
    def coerce(cls, obj):
        if isinstance(obj, cls):
            return obj
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
        """parc parameter for testnd test"""
        if self.source and not self.aggregate:
            return 'source' if disconnect_labels else None
        if disconnect_labels:
            raise TypeError(f"{disconnect_labels=}: invalid for data={self.string!r}")
        return None

    def find_ndvar_channel_types(self, info: mne.Info) -> list[str]:
        """NDVar keys for the sensor data in ``info``.

        Sensor NDVars are keyed by their MNE channel type
        (``'mag'``/``'grad'``/``'eeg'``), so these are both the channel types
        passed to the loader and the keys under which the NDVars are stored.
        """
        assert self.sensor
        if self.space in ('eeg', 'mag', 'grad'):
            return [self.space]
        channel_types = info.get_channel_types(unique=True, only_data_chs=True)
        if self.space == 'sensor':
            return channel_types
        elif self.space == 'meg':
            return [ch_type for ch_type in ('mag', 'grad') if ch_type in channel_types]
        raise RuntimeError(f"{self} with {channel_types=}")

    def response_key(self, ds: Dataset) -> str:
        """Key of the response NDVar in a loaded dataset ``ds``.

        Source data is keyed ``'src'`` and a specific sensor type by its channel
        type; only the all-sensor description is resolved against the data,
        preferring ``'mag'`` when several sensor types are present.

        Parameters
        ----------
        ds
            Loaded dataset carrying the response NDVar(s). Only used to
            disambiguate the all-sensor description, which needs the loaded
            channel types in ``ds.info['sensor_types']``.
        """
        if self.source:
            return 'src'
        elif self.space in ('eeg', 'mag', 'grad'):
            return self.space
        types = ds.info['sensor_types']
        return 'mag' if 'mag' in types else types[0]
