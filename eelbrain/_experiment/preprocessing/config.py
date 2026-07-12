# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Raw preprocessing configurations.

:class:`RawPipe` is the configuration base class for the raw pipeline; its
subclasses implement a specific preprocessing step by overriding
:meth:`RawPipe._make` and expose user-configurable parameters. Users add these
objects to :class:`~pipeline.Pipeline`. The graph nodes that build and load the
concrete artifacts from these configurations live in
:mod:`._experiment.preprocessing.nodes`.
"""
from __future__ import annotations
import fnmatch
import logging
from pathlib import Path
from typing import Any
import warnings
from collections.abc import Mapping, Sequence

import mne
from mne_bids import BIDSPath
from scipy import signal

from ... import load
from ..._data_obj import NDVar, Sensor
from ..._exceptions import ConfigurationError
from ..._io.fiff import KIT_NEIGHBORS
from ..._io.txt import read_adjacency
from ..._ndvar import filter_data
from ..._text import enumeration
from ..._utils import user_activity
from ..derivative_cache import Request
from ..configuration import Configuration, ConfigurationDict, sequence_arg, typed_arg
from ..exceptions import FileMissingError
from ..pathing import ica_file_path

MNE_VERBOSITY = 'WARNING'
LOG = logging.getLogger(__name__)


class RawPipe(Configuration):
    """Base class for raw-pipeline configurations."""
    DICT_ATTRS = ()

    def _can_resolve(self, pipes: Mapping[str, RawPipe]) -> bool:
        """Determine whether this pipe's dependencies are available in ``pipes``."""
        raise NotImplementedError

    def _get_adjacency(self, data: str) -> str | list[tuple[str, str]] | None:
        raise NotImplementedError

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
    ) -> str | None:
        raise NotImplementedError

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        """Assemble bad channels list from sources"""
        raise NotImplementedError


def raw_node_name(raw: str) -> str:
    return f'raw@{raw}'


def raw_bad_channels_input_name(raw: str) -> str:
    return f'raw-input-bads@{raw}'


def raw_input_name(raw: str) -> str:
    return f'raw-input@{raw}'


def ica_input_name(raw: str) -> str:
    return f'ica-input@{raw}'


class RawSource(RawPipe):
    """Raw data source

    Parameters
    ----------
    sysname
        Used to determine sensor positions (not needed for KIT files, or when a
        montage is specified).
    rename_channels
        Rename channels based on a ``{from: to}`` dictionary. This happens
        *after* calling the ``reader``, and *before* applying the ``montage``.
        Useful to convert system-specific channel names to those of a standard montages.
    montage
        Name of a montage that is applied to raw data to set sensor positions
        (see :meth:`mne.io.Raw.set_montage`).
    adjacency
        Adjacency between sensors. Can be specified as:

        - ``'auto'`` to use :func:`mne.channels.find_ch_adjacency`
        - Pre-defined adjacency (one of :func:`mne.channels.get_builtin_ch_adjacencies`)
        - Path to load adjacency from a file
        - ``"none"`` for no connections
        - ``"grid"`` for grid connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.

        If unspecified, it is inferred from ``sysname`` if possible.
    ...
    """
    DICT_ATTRS = ('sysname', 'rename_channels', 'montage', 'adjacency', 'kwargs')

    def __init__(
            self,
            sysname: str = None,
            rename_channels: dict = None,
            montage: str = None,
            adjacency: str | list[tuple[str, str]] | Path = None,
            **kwargs,
    ):
        RawPipe.__init__(self)
        if isinstance(adjacency, str):
            if adjacency not in ('auto', 'grid', 'none') and adjacency not in mne.channels.get_builtin_ch_adjacencies():
                adjacency = Path(adjacency)
        if isinstance(adjacency, Path):
            adjacency = read_adjacency(adjacency)
        self.sysname = sysname
        self.rename_channels = typed_arg(rename_channels, dict)
        self.montage = montage
        self.adjacency = adjacency
        self.kwargs = kwargs

    def _can_resolve(self, pipes: dict[str, RawPipe]) -> bool:
        return True

    def _normalize_channel_names(self, raw: mne.io.BaseRaw, bad_chs: list[str]) -> list[str]:
        """Validate and normalize channel names against the raw file's sensor layout."""
        sensor = load.mne.sensor_dim(raw.info, adjacency=self.adjacency)
        return sensor._normalize_sensor_names(bad_chs)

    def _detect_flat_channels(self, path: BIDSPath, raw: mne.io.BaseRaw, flat: float = None) -> list[str] | None:
        """Detect flat channels; returns None if the operation should be skipped."""
        if flat is None:
            if path.datatype == 'meg':
                flat = 1e-14
            elif path.datatype == 'eeg':
                return None
            else:
                raise NotImplementedError(f"{path.datatype=}")
        elif flat == 0:
            return None
        bad_chs = list(raw.info['bads'])
        sysname = self._get_sysname(raw.info, path.subject, path.datatype)
        raw_ndvar = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=self.adjacency)
        bad_chs.extend(raw_ndvar.sensor.names[raw_ndvar.std('time') < flat])
        return bad_chs

    def _as_dict(self) -> dict:
        out = RawPipe._as_dict(self)
        if isinstance(self.montage, mne.channels.DigMontage):
            out['montage'] = Sensor.from_montage(self.montage)
        return out

    def _get_adjacency(self, data: str) -> str | list[tuple[str, str]] | None:
        if data == 'eog':
            return None
        else:
            return self.adjacency

    def _get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
    ) -> str | None:
        if data == 'eog':
            return None
        elif isinstance(self.sysname, str):
            return self.sysname
        elif isinstance(self.sysname, dict):
            for k, v in self.sysname.items():
                if fnmatch.fnmatch(subject, k):
                    return v
        kit_system_id = info.get('kit_system_id')
        return KIT_NEIGHBORS.get(kit_system_id)


class CachedRawPipe(RawPipe):
    _bad_chs_affect_cache: bool = False
    DICT_ATTRS = ('source',)

    def __init__(self, source: str, cache: bool = True):
        RawPipe.__init__(self)
        self.source = source
        self.cache = cache

    def _can_resolve(self, pipes: Mapping[str, RawPipe]) -> bool:
        return self.source in pipes

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return info

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        return ctx.load(raw_node_name(self.source), options={'noise': noise}, view='bads')


class RawFilter(CachedRawPipe):
    """Filter raw pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    l_freq
        Low cut-off frequency in Hz.
    h_freq
        High cut-off frequency in Hz.
    cache
        Cache the resulting raw files (default ``True``).
    n_jobs
        Parameter for :meth:`mne.io.Raw.filter`; Values other than 1 are slower
        in most cases due to added overhead except for very large files.
    ...
        :meth:`mne.io.Raw.filter` parameters.

    See Also
    --------
    Pipeline.raw
    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('l_freq', 'h_freq', 'n_jobs', 'kwargs')

    def __init__(
            self,
            source: str,
            l_freq: float = None,
            h_freq: float = None,
            cache: bool = True,
            n_jobs: str | int | None = 1,
            **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.kwargs = kwargs
        self.n_jobs = n_jobs

    def _filter_ndvar(self, ndvar, **kwargs):
        return filter_data(ndvar, self.l_freq, self.h_freq, **self.kwargs, **kwargs)

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: filtering for %s...", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        raw.filter(self.l_freq, self.h_freq, **self.kwargs, n_jobs=self.n_jobs, verbose=MNE_VERBOSITY)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        if self.l_freq and self.l_freq > (info['highpass'] or 0):
            with info._unlock():
                info['highpass'] = float(self.l_freq)
        if self.h_freq and self.h_freq < (info['lowpass'] or info['sfreq']):
            with info._unlock():
                info['lowpass'] = float(self.h_freq)
        return info


class RawFilterElliptic(CachedRawPipe):
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('low_stop', 'low_pass', 'high_pass', 'high_stop', 'gpass', 'gstop')

    def __init__(self, source, low_stop, low_pass, high_pass, high_stop, gpass, gstop):
        CachedRawPipe.__init__(self, source)
        self.low_stop = low_stop
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.high_stop = high_stop
        self.gpass = gpass
        self.gstop = gstop

    def _sos(self, sfreq):
        nyq = sfreq / 2.
        low_stop = self.low_stop
        low_pass = self.low_pass
        high_pass = self.high_pass
        high_stop = self.high_stop
        gpass = self.gpass
        gstop = self.gstop
        if high_stop is None:
            assert low_stop is not None
            assert high_pass is None
        else:
            high_stop /= nyq
            high_pass /= nyq

        if low_stop is None:
            assert low_pass is None
        else:
            low_pass /= nyq
            low_stop /= nyq

        if low_stop is None:
            btype = 'lowpass'
            wp, ws = high_pass, high_stop
        elif high_stop is None:
            btype = 'highpass'
            wp, ws = low_pass, low_stop
        else:
            btype = 'bandpass'
            wp, ws = (low_pass, high_pass), (low_stop, high_stop)
        order, wn = signal.ellipord(wp, ws, gpass, gstop)
        return signal.ellip(order, gpass, gstop, wn, btype, output='sos')

    def _filter_ndvar(self, ndvar):
        axis = ndvar.get_axis('time')
        sos = self._sos(1. / ndvar.time.tstep)
        x = signal.sosfilt(sos, ndvar.x, axis)
        return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: filtering for %s...", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        sos = self._sos(raw.info['sfreq'])
        for i in picks:
            raw._data[i] = signal.sosfilt(sos, raw._data[i])
        low, high = self.low_pass, self.high_pass
        with raw.info._unlock():
            if high and raw.info['lowpass'] > high:
                raw.info['lowpass'] = float(high)
            if low and raw.info['highpass'] < low:
                raw.info['highpass'] = float(low)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        low, high = self.low_pass, self.high_pass
        if high and high < (info['lowpass'] or info['sfreq']):
            with info._unlock():
                info['lowpass'] = float(high)
        if low and low > (info['highpass'] or 0):
            with info._unlock():
                info['highpass'] = float(low)
        return info


class RawICA(CachedRawPipe):
    """ICA raw pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    task
        Task(s) to use for estimating ICA components. Can be omitted (``None``)
        when the experiment has exactly one task, or when the ICA step occurs
        after a :class:`RawMaxwell` step (in which case all tasks are used, see
        Notes).
    method
        Method for ICA decomposition (default: ``'extended-infomax'``; see
        :class:`mne.preprocessing.ICA`).
    random_state
        Set the random state for ICA decomposition to make results reproducible
        (default 0, see :class:`mne.preprocessing.ICA`).
    fit_kwargs
        A dictionary with keyword arguments that should be passed to
        :meth:`mne.preprocessing.ICA.fit`. This includes
        ``reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}`` unless
        a different value for ``reject`` is specified here.
    cache : bool
        Cache the resulting raw files (default ``False``).
    ...
        Additional parameters for :class:`mne.preprocessing.ICA`.

    See Also
    --------
    Pipeline.raw
    RawApplyICA

    Notes
    -----
    This preprocessing step estimates one set of ICA components per subject,
    using the data specified in the ``task`` parameter. If the experiment has
    exactly one task, ``task`` can be omitted. The selected
    components are then removed from all data tasks during this preprocessing
    step, regardless of whether they were used to estimate the components or
    not.

    When the ICA step occurs after a :class:`RawMaxwell` step, ``task`` can be
    omitted even with multiple tasks: all tasks and runs available for each
    subject/session/acquisition are concatenated for the fit. This is safe because Maxwell
    filtering maps every recording to a common head position. Run concatenation
    applies to any ICA step after a :class:`RawMaxwell` step (also with an
    explicit ``task``); without a preceding :class:`RawMaxwell` step a single
    run is used.

    Use :meth:`Pipeline.make_ica_selection` for each subject to
    select ICA components that should be removed. The arguments to that function
    determine what data is used to visualize the component time courses.

    This step merges bad channels from all tasks.

    Examples
    --------
    Some ICA examples::

        class Experiment(Pipeline):

            raw = {
                '1-40': RawFilter('raw', 1, 40),
                # Extended infomax with PCA preprocessing
                'ica': RawICA('1-40', n_components=0.99),
                # Fast ICA
                'fastica': RawICA('1-40', 'task', 'fastica', n_components=0.9),
                # Change thresholds for data rejection using fit_kwargs
                'ica-rej': RawICA('1-40', 'task', 'fastica', fit_kwargs=dict(
                    reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 500e-6},
                )),
            }

    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('task', 'kwargs', 'fit_kwargs')

    run: str | Sequence[str] = None
    # Whether to concatenate all runs per subject/session/acquisition for the ICA fit.
    # Resolved during pipeline assembly (True when the step is after RawMaxwell).
    _concatenate_runs: bool = False

    def __init__(
            self,
            source: str,
            task: str | Sequence[str] | None = None,
            method: str = 'extended-infomax',
            random_state: int = 0,
            fit_kwargs: dict[str, Any] = None,
            cache: bool = False,
            **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.task = sequence_arg('task', task, allow_none=True)
        self.method = method
        self.random_state = random_state
        self.kwargs = {'method': method, 'random_state': random_state, **kwargs}
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}

    def path(self, ctx: Request) -> Path:
        return ctx.root / ica_file_path(ctx.state, self.name, self._concatenate_runs, datatype=ctx.datatype)

    def _load_ica(
            self,
            ctx: Request,
    ) -> mne.preprocessing.ICA:
        ica_path = self.path(ctx)
        if not ica_path.exists():
            raise FileMissingError(f"ICA file {ica_path.name} does not exist for raw={self.name!r}. Run e.make_ica() to create it.")
        return mne.preprocessing.read_ica(ica_path)

    @staticmethod
    def _check_ica_channels(
            ica: mne.preprocessing.ICA,
            info: mne.Info,
            return_missing: bool = False,  # return channels present in the data but missing from the ICA
    ) -> bool | tuple:
        "Check whether `ica` and `info` contain the same channels"
        # Compare channel presence, not bad-status (exclude=[]): a currently-bad channel that
        # is still in the data is not "missing" from the ICA.
        picks = mne.pick_types(info, meg=True, eeg=True, ref_meg=False, exclude=[])
        raw_ch_names = [info.ch_names[i] for i in picks]
        if return_missing:
            raw_set = set(raw_ch_names)
            ica_set = set(ica.ch_names)
            if ica_set - raw_set:
                raise RuntimeError(f"ICA contains channels not present in data: {enumeration(sorted(ica_set - raw_set))}")
            else:
                return tuple(raw_set - ica_set)
        else:
            return raw_ch_names == ica.ch_names

    def _fit_ica(
            self,
            raw: mne.io.BaseRaw,
            subject: str,
            raw_name: str,
    ) -> mne.preprocessing.ICA:
        LOG.info("Raw %s: computing ICA decomposition for %s", raw_name, subject)
        kwargs = self.kwargs.copy()
        kwargs.setdefault('max_iter', 256)
        if kwargs['method'] == 'extended-infomax':
            kwargs['method'] = 'infomax'
            kwargs['fit_params'] = {'extended': True}

        ica = mne.preprocessing.ICA(**kwargs)
        fit_kwargs = {'reject': {'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}, **self.fit_kwargs}
        with user_activity:
            ica.fit(raw, **fit_kwargs)
        return ica

    def _apply_ica(
            self,
            raw: mne.io.BaseRaw,
            ica: mne.preprocessing.ICA,
            bad_channels: list[str],
            raw_name: str,
            log: logging.Logger | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.debug("Raw %s: applying ICA...", raw_name)
        raw.info['bads'] = [ch for ch in bad_channels if ch in raw.ch_names]
        missing = self._check_ica_channels(ica, raw.info, return_missing=True)
        if missing:
            # Channels excluded from the ICA fit (e.g. bad at fit time) are not in
            # ica.ch_names. Keep them in the data marked as bad — ica.apply leaves them
            # untouched and they remain available for downstream interpolation — rather than
            # dropping them outright.
            raw.info['bads'] = sorted(set(raw.info['bads']).union(missing))
        ica.apply(raw)
        return raw

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        bads = set()
        bads.update(ctx.load(ica_input_name(self.name), view='bads'))
        # Task that has not been used for ICA fit
        if noise:
            bads.update(ctx.load(raw_node_name(self.source), options={'noise': True}, view='bads'))
        elif ctx.state['task'] not in self.task:
            bads.update(ctx.load(raw_node_name(self.source), view='bads'))
        return sorted(bads)


class RawApplyICA(CachedRawPipe):
    """Apply ICA estimated in a :class:`RawICA` pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    ica
        Name of the :class:`RawICA` pipe from which to load the ICA components.
    cache
        Cache the resulting raw files (default ``False``).

    See Also
    --------
    Pipeline.raw

    Notes
    -----
    This pipe inherits bad channels from the ICA.

    Examples
    --------
    Estimate ICA components with 1-40 Hz band-pass filter and apply the ICA
    to data that is high pass filtered at 0.1 Hz::

        class Experiment(Pipeline):

            raw = {
                '1-40': RawFilter('raw', 1, 40),
                'ica': RawICA('1-40', 'task', 'extended-infomax', n_components=0.99),
                '0.1-40': RawFilter('raw', 0.1, 40),
                '0.1-40-ica': RawApplyICA('0.1-40', 'ica'),
            }

    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('ica_source',)

    def __init__(
            self,
            source: str,
            ica: str,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.ica_source = ica

    def _can_resolve(self, pipes: Mapping[str, RawPipe]) -> bool:
        return CachedRawPipe._can_resolve(self, pipes) and self.ica_source in pipes

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None

    def _collect_bads(
            self,
            ctx: Request,
            *,
            noise: bool = False,
    ) -> list[str]:
        bads = set()
        bads.update(ctx.load(raw_node_name(self.source), options={'noise': noise}, view='bads'))
        bads.update(ctx.load(raw_node_name(self.ica_source), view='bads'))
        return sorted(bads)


class RawMaxwell(CachedRawPipe):
    """Maxwell filter raw pipe.

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    bad_condition
        How to deal with ill-conditioned SSS matrices; by default, an error is
        raised, which might prevent the process to complete for some subjects.
        Set to ``'warning'`` to proceed anyways.
    cache
        Cache the resulting raw files (default ``True``).
    flat
        Threshold for marking flat channels as bad (default 1e-14).
    ...
        :func:`mne.preprocessing.maxwell_filter` parameters.

    See Also
    --------
    Pipeline.raw

    Notes
    -----
    For empty room recordings, there is no ``dev_head_t`` information, ``coord_frame = 'meg'`` will be used automatically.
    Flat channels are automatically marked as bad with a threshold of parameter ``flat``.
    """

    _bad_chs_affect_cache = True
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('bad_condition', 'flat', 'kwargs')

    def __init__(
        self,
        source: str,
        bad_condition: str = 'error',
        cache: bool = True,
        flat: float = 1e-14,
        **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.kwargs = kwargs
        self.bad_condition = bad_condition
        self.flat = flat

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
            calibration: Path | None = None,
            cross_talk: Path | None = None,
            destination: mne.transforms.Transform | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: computing Maxwell filter for %s", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        if noise:
            coord_frame = 'meg'
            destination = None
        else:
            coord_frame = 'head'

        with user_activity:
            # find bad channels
            noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw, calibration=calibration, cross_talk=cross_talk, bad_condition=self.bad_condition, coord_frame=coord_frame)
            raw.info['bads'] = sorted(raw.info['bads'] + noisy_chs + flat_chs)
            # Maxwell filter
            kwargs = self.kwargs
            st_duration = kwargs.get('st_duration')
            if st_duration is not None and kwargs.get('st_overlap', True):
                # MNE's overlapping tSSS uses a Hann window of round(st_duration * sfreq) samples with 50% overlap, which only satisfies the constant-overlap-add constraint for an even sample count; nudge st_duration up by one sample when it would be odd
                n_samples = int(round(st_duration * raw.info['sfreq']))
                if n_samples % 2:
                    kwargs = {**kwargs, 'st_duration': (n_samples + 1) / raw.info['sfreq']}
            return mne.preprocessing.maxwell_filter(raw, calibration=calibration, cross_talk=cross_talk, destination=destination, bad_condition=self.bad_condition, coord_frame=coord_frame, verbose=MNE_VERBOSITY, **kwargs)

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None


class RawOversampledTemporalProjection(CachedRawPipe):
    """Oversampled temporal projection: see :func:`mne.preprocessing.oversampled_temporal_projection`"""
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + ('duration',)

    def __init__(
            self,
            source: str,
            duration: float = 10.0,
            cache: bool = True,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.duration = duration

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        logger = log or LOG
        logger.info("Raw %s: computing oversampled temporal projection for %s", raw_name, path.fpath if not noise else path.find_empty_room().fpath)
        with user_activity:
            return mne.preprocessing.oversampled_temporal_projection(raw, self.duration)


class Reference(Configuration):
    """Re-reference EEG data after epoching and channel interpolation

    Used as a value in :attr:`Pipeline.references` and selected through the
    ``reference`` state. Also the base class for :class:`RawReReference`, which
    applies the same operation to continuous raw data.

    Parameters
    ----------
    reference
        New reference: ``'average'`` (default) or one or several electrode
        names.
    add
        Reconstruct reference channels with given names and set them to 0.
    drop
        Drop these channels after applying the reference.

    See Also
    --------
    Pipeline.references
    """
    DICT_ATTRS = ('reference', 'add', 'drop')

    def __init__(
            self,
            reference: str | Sequence[str] = 'average',
            add: str | Sequence[str] = None,
            drop: str | Sequence[str] = None,
    ):
        if isinstance(reference, str):
            self.reference = reference
        else:
            self.reference = sequence_arg('reference', reference, allow_none=False, sequence_type=list)
        self.add = sequence_arg('add', add, sequence_type=list)
        self.drop = sequence_arg('drop', drop, sequence_type=list)

    def _apply_reference(
            self,
            inst: mne.io.BaseRaw | mne.BaseEpochs,
            montage: str | mne.channels.DigMontage | None = None,
    ) -> mne.io.BaseRaw | mne.BaseEpochs:
        """Apply the reference to a :class:`~mne.io.BaseRaw` or :class:`~mne.Epochs`."""
        if self.add:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The locations of multiple reference channels are ignored', module='mne')
                inst = mne.add_reference_channels(inst, self.add, copy=False)
            if montage:
                inst.set_montage(montage)
        inst.set_eeg_reference(self.reference)
        if self.drop:
            inst = inst.drop_channels(self.drop)
        return inst

    def _prepare_source_data(
            self,
            inst: mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked,
            montage: str | mne.channels.DigMontage | None = None,
    ) -> None:
        """Prepare an EEG instance for source localization in-place.

        Reconstructs implicit reference channels (:attr:`add`) as zeros and adds
        an average-reference *projection*. Unlike :meth:`_apply_reference`, this
        never applies a direct reference, because MNE requires the average
        reference as a projection (``custom_ref_applied`` must be ``False``) for
        inverse modeling. A no-op for data without EEG channels and for data
        that already carries an average-reference projection.
        """
        if self.reference != 'average' or self.drop:
            raise NotImplementedError(f"{self} for source localization; only an average reference (optionally with add=...) is supported.")
        if self.add:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The locations of multiple reference channels are ignored', module='mne')
                mne.add_reference_channels(inst, self.add, copy=False)
            if montage:
                inst.set_montage(montage)
            # add_reference_channels marks a custom reference; adding the
            # average-reference projection resets that flag, which is required
            # for inverse modeling (custom_ref_applied must be False).
            inst.set_eeg_reference('average', projection=True)
        elif not inst.info['custom_ref_applied'] and mne.pick_types(inst.info, meg=False, eeg=True, ref_meg=False, exclude=[]).size:
            # Ensure an average-reference projection is present (required by MNE
            # for inverse modeling). set_eeg_reference(projection=True) is
            # idempotent: it adds the projection if missing and otherwise leaves
            # the data untouched (warning suppressed). Skipped when a custom
            # reference is applied, so custom-referenced data still raises in MNE.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'An average reference projection was already added', module='mne')
                inst.set_eeg_reference('average', projection=True)


class RawReReference(Reference, CachedRawPipe):
    """Re-reference EEG data as preprocessing step

    For most workflows, it is recommended to re-reference after epoching
    using :ref:`state-reference`.

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    reference
        New reference: ``'average'`` (default) or one or several electrode
        names.
    add
        Reconstruct reference channels with given names and set them to 0.
    drop
        Drop these channels after applying the reference.
    cache
        Cache the resulting raw files (default ``False``).

    See Also
    --------
    Pipeline.raw
    """
    DICT_ATTRS = CachedRawPipe.DICT_ATTRS + Reference.DICT_ATTRS

    def __init__(
            self,
            source: str,
            reference: str | Sequence[str] = 'average',
            add: str | Sequence[str] = None,
            drop: str | Sequence[str] = None,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        Reference.__init__(self, reference, add, drop)

    def _make(
            self,
            raw: mne.io.BaseRaw,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
            source_pipe: RawSource | None = None,
    ) -> mne.io.BaseRaw:
        return self._apply_reference(raw, montage=source_pipe.montage if source_pipe else None)

    def _make_info(
            self,
            info: mne.Info,
            *,
            path: BIDSPath,
            noise: bool = False,
            raw_name: str = None,
            log: logging.Logger | None = None,
    ) -> mne.Info | None:
        return None


class RawPipeGraph(Mapping[str, RawPipe]):
    """Resolved raw-pipeline graph with convenience lineage lookups."""

    def __init__(
            self,
            pipes: dict[str, RawPipe],
            source_names: dict[str, str | None],
            root_source_names: dict[str, str],
            ica_names: dict[str, str | None],
            lineages: dict[str, tuple[str, ...]],
    ):
        self._pipes = ConfigurationDict('raw pipe', pipes)
        self._source_names = source_names
        self._root_source_names = root_source_names
        self._ica_names = ica_names
        self._lineages = lineages

    def __getitem__(self, item: str) -> RawPipe:
        return self._pipes[item]

    def __iter__(self):
        return iter(self._pipes)

    def __len__(self) -> int:
        return len(self._pipes)

    def source_name(self, raw_name: str) -> str | None:
        """Return the immediate upstream raw name for ``raw_name``."""
        return self._source_names[raw_name]

    def source_pipe(self, raw_name: str) -> RawPipe | None:
        """Return the immediate upstream raw pipe for ``raw_name``."""
        source_name = self.source_name(raw_name)
        if source_name is None:
            return None
        return self[source_name]

    def root_source_name(self, raw_name: str) -> str:
        """Return the source raw name at the root of ``raw_name``."""
        return self._root_source_names[raw_name]

    def root_source_pipe(self, raw_name: str) -> RawSource:
        """Return the source raw pipe at the root of ``raw_name``."""
        pipe = self[self.root_source_name(raw_name)]
        assert isinstance(pipe, RawSource)
        return pipe

    def ica_name(self, raw_name: str) -> str:
        """Return the ICA raw name associated with ``raw_name``."""
        ica_name = self._ica_names[raw_name]
        if ica_name is None:
            raise ValueError(f"{raw_name=} does not involve ICA")
        return ica_name

    def ica_pipe(self, raw_name: str) -> RawICA:
        """Return the ICA raw pipe associated with ``raw_name``."""
        ica_name = self.ica_name(raw_name)
        pipe = self[ica_name]
        assert isinstance(pipe, RawICA)
        return pipe

    def lineage_names(self, raw_name: str) -> tuple[str, ...]:
        """Return the raw-step names from source to ``raw_name``."""
        return self._lineages[raw_name]

    def lineage_pipes(self, raw_name: str) -> tuple[RawPipe, ...]:
        """Return the raw-step pipes from source to ``raw_name``."""
        return tuple(self[name] for name in self.lineage_names(raw_name))


def assemble_raw_pipes(
        raw: dict[str, RawPipe],
        tasks: tuple[str],
) -> RawPipeGraph:
    """Resolve raw-pipe dependencies and bind pipe names."""
    pending = dict(raw)
    resolved = {}
    source_names = {}
    root_source_names = {}
    ica_names = {}
    lineages = {}
    for name, pipe in pending.items():
        pipe._store_name(name)
    while pending:
        n_pending = len(pending)
        for key in list(pending):
            if pending[key]._can_resolve(resolved):
                pipe = pending.pop(key)
                if isinstance(pipe, RawICA):
                    after_maxwell = any(isinstance(resolved[name], RawMaxwell) for name in lineages[pipe.source])
                    pipe._concatenate_runs = after_maxwell
                    if pipe.task is None:
                        if len(tasks) == 1 or after_maxwell:
                            pipe.task = tasks
                        else:
                            raise ConfigurationError(f"RawICA {key!r} has task=None but the experiment has {len(tasks)} tasks. Specify task explicitly, or place the ICA step after a RawMaxwell step to use all tasks. Available tasks: {', '.join(tasks)}.")
                    missing = set(pipe.task).difference(tasks)
                    if missing:
                        raise ConfigurationError(f"RawICA {key!r} lists one or more non-exising tasks: {', '.join(missing)}. Available tasks: {', '.join(tasks)}.")
                if isinstance(pipe, RawSource):
                    source_names[key] = None
                    root_source_names[key] = key
                    ica_names[key] = None
                    lineages[key] = (key,)
                else:
                    source_names[key] = pipe.source
                    root_source_names[key] = root_source_names[pipe.source]
                    if isinstance(pipe, RawICA):
                        ica_names[key] = key
                    elif isinstance(pipe, RawApplyICA):
                        ica_names[key] = pipe.ica_source
                    else:
                        ica_names[key] = ica_names[pipe.source]
                    lineages[key] = (*lineages[pipe.source], key)
                resolved[key] = pipe
        if len(pending) == n_pending:
            raise ConfigurationError(f"Unable to resolve source for raw {enumeration(pending)}, circular dependency?")
    return RawPipeGraph(raw, source_names, root_source_names, ica_names, lineages)
