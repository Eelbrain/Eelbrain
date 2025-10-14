# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pre-processing operations based on NDVars"""
from __future__ import annotations
import warnings
from copy import deepcopy
import fnmatch
from itertools import chain
import logging
from os import makedirs, remove
from os.path import basename, dirname, exists, getmtime, join, splitext
from pathlib import Path
from typing import Any, Collection, Dict, List, Literal, Sequence, Tuple, Union

import mne
from scipy import signal
from mne_bids import BIDSPath, read_raw_bids, mark_channels
import pandas as pd

from .. import load
from .._data_obj import NDVar, Sensor
from .._exceptions import DefinitionError
from .._io.fiff import KIT_NEIGHBORS
from .._io.txt import read_adjacency
from .._ndvar import filter_data
from .._text import enumeration
from .._utils import ask, deprecate_kwarg, user_activity
from ..mne_fixes import CaptureLog
from ..mne_fixes._version import MNE_VERSION, V0_19, V0_24
from .definitions import log_dict_change, tuple_arg, typed_arg
from .exceptions import FileMissingError

AddBadsArg = Union[bool, Sequence[str]]
PreloadArg = Union[bool, Literal[-1]]


class RawPipe:
    name: str = None  # set on linking
    log: logging.Logger = None

    def _can_link(self, pipes: Dict[str, RawPipe]) -> bool:
        raise NotImplementedError

    def _link(
            self,
            name: str,
            pipes: Dict[str, RawPipe],
            cache_dir: str,
            deriv_dir: str,
            log: logging.Logger
    ) -> RawPipe:
        raise NotImplementedError

    def _link_base(
            self,
            name: str,
            log: logging.Logger,
    ) -> RawPipe:
        out = deepcopy(self)
        out.name = name
        out.log = log
        return out

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = {arg: getattr(self, arg) for arg in chain(args, ('name',))}
        out['type'] = self.__class__.__name__
        return out

    @staticmethod
    def _normalize_dict(state: dict) -> None:
        pass

    def cache(self, path: BIDSPath) -> mne.io.BaseRaw:
        "Make sure the file exists and is up to date"
        raise NotImplementedError

    def get_adjacency(self, data: str) -> Union[str, List[Tuple[str, str]], Path]:
        raise NotImplementedError

    def get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str
    ) -> str | None:
        raise NotImplementedError

    def get_path(
            self,
            path: BIDSPath,
            file_type: Literal['raw', 'bads', 'cache', 'ica'] = 'raw',
    ) -> str:
        # Caller of this method should check whether raw exists or not.
        # Channels file will be created if it doesn't exist.
        if file_type == 'raw':
            return str(path.fpath)
        elif file_type == 'bads':
            bads_path = path.fpath.parent / path.copy().update(suffix='channels', extension='.tsv').basename
            if not exists(bads_path):
                self.log.info("No channels.tsv found for %s, creating an empty one.", path.fpath)
                makedirs(bads_path.parent, exist_ok=True)
                pd.DataFrame(columns=['name', 'type', 'units', 'low_cutoff', 'high_cutoff', 'description', 'sampling_frequency', 'status', 'status_description']).to_csv(bads_path, sep='\t', index=False)
            return str(bads_path)
        elif file_type == 'cache':
            return join(
                self.cache_dir,
                'raw',
                get_subject_session(path.basename),
                f'{splitext(path.basename)[0]}_raw-{self.name}.fif'
            )
        else:
            return join(
                self.deriv_dir,
                'ica',
                f'{remove_task(splitext(path.basename)[0])}_raw-{self.name}_ica.fif'
            )

    def load(
            self,
            path: BIDSPath,
            add_bads: AddBadsArg = True,
            preload: PreloadArg = False,  # -1: info only, data will not be needed
            raw: mne.io.BaseRaw = None,
    ) -> mne.io.BaseRaw:
        # raw
        if raw is None:
            if preload == -1:
                preload = False
            raw = self._load(path, preload)
        # bad channels
        if isinstance(add_bads, Sequence):
            raw.info['bads'] = list(add_bads)
        elif add_bads is False:
            raw.info['bads'] = []
        elif add_bads is not True:
            raise TypeError(f"{add_bads=}")
        return raw

    def _load(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        return read_raw_bids(
            path,
            extra_params={'preload': preload},
            verbose='critical',
        )

    def load_bad_channels(
            self,
            path: BIDSPath,
            existing: Collection[str] = None,
    ) -> list[str]:
        raise NotImplementedError

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs,
            redo: bool,
    ) -> None:
        raise NotImplementedError

    def make_bad_channels_auto(
            self,
            path: BIDSPath,
            flat: float,
            redo: bool,
    ) -> None:
        raise NotImplementedError

    def mtime(
            self,
            path: BIDSPath,
            bad_chs: bool = True,
    ) -> float | None:
        "Modification time of anything influencing the output of load"
        raise NotImplementedError


class RawSource(RawPipe):
    """Raw data source

    Parameters
    ----------
    filename
        Pattern for filenames. The pattern should contain the fields
        ``{subject}`` and ``{recording}`` (which internally is expanded to
        ``session`` and, if applicable, ``visit``;
        default ``'{subject}_{recording}-raw.fif'``).
    reader
        Function for reading data (default is :func:`mne.io.read_raw_fif`).
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
        Ajacency between sensors. Can be specified as:

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
        Additional parameters for the ``reader`` function.

    See Also
    --------
    Pipeline.raw

    Examples
    --------
    If you would load the EEG data like this::

        form pathlib import Path

        subject = '0001'
        raw = mne.io.read_raw_brainvision(
            f'/data/eeg/{subject}/raw/task_{subject}.vhdr',
            eog=['EOG horizontal left', 'EOG horizontal right', 'EOG vertical below', 'EOG vertical above'],
        )
        raw.rename_channels({'FZ': 'Fz', 'OZ': 'Oz', 'CZ': 'Cz', 'Ref right mastoid': 'A1'})
        raw.set_montage('standard_1020')

    Then the equivalent :class:`RawSource` is::

        raw = {
            'raw': RawSource(
                'raw/{recording}_{subject}.vhdr',
                reader=mne.io.read_raw_brainvision,
                rename_channels={'FZ': 'Fz', 'OZ': 'Oz', 'CZ': 'Cz', 'Ref right mastoid': 'A1'},
                eog=['EOG horizontal left', 'EOG horizontal right', 'EOG vertical below', 'EOG vertical above'],
                montage='standard_1020'),
            ...
        }

    """

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            sysname: str = None,
            rename_channels: dict = None,
            montage: str = None,
            adjacency: Union[str, List[Tuple[str, str]], Path] = None,
            **kwargs,
    ):
        RawPipe.__init__(self)
        if isinstance(adjacency, str):
            if adjacency not in ('auto', 'grid', 'none') and adjacency not in mne.channels.get_builtin_ch_adjacencies():
                adjacency = Path(adjacency)
        if isinstance(adjacency, Path):
            adjacency = read_adjacency(adjacency)
        # TODO: Using MNE-BIDS to read raw limits the file formats, add more readers later
        # self.filename = typed_arg(filename, str)
        # self.reader = reader
        self.sysname = sysname
        self.rename_channels = typed_arg(rename_channels, dict)
        self.montage = montage
        self.adjacency = adjacency
        self._kwargs = kwargs
        # TODO: Should we keep this?
        # if MNE_VERSION < V0_19 and reader is mne.io.read_raw_cnt:
        #     self._read_raw_kwargs = {'montage': None, **kwargs}
        # else:
        #     self._read_raw_kwargs = kwargs

    def _can_link(self, pipes: Dict[str, RawPipe]) -> bool:
        return True

    def _link(
            self,
            name: str,
            pipes: Dict[str, RawPipe],
            cache_dir: str,
            deriv_dir: str,
            log: logging.Logger,
    ) -> RawPipe:
        return RawPipe._link_base(self, name, log)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = RawPipe._as_dict(self, args)
        out.update(self._kwargs)
        if self.rename_channels:
            out['rename_channels'] = self.rename_channels
        if self.montage:
            if isinstance(self.montage, mne.channels.DigMontage):
                out['montage'] = Sensor.from_montage(self.montage)
            else:
                out['montage'] = self.montage
        if self.adjacency is not None:
            out['connectivity'] = self.adjacency
        return out

    def cache(self, path: BIDSPath) -> None:
        "Make sure the file exists and is up to date"
        raw_path = self.get_path(path)
        if not exists(raw_path):
            raise FileMissingError(f"Raw input file does not exist at expected location {raw_path}")
        return None

    def _load(self, path: BIDSPath, preload):
        raw: mne.io.BaseRaw = read_raw_bids(
            path,
            extra_params={'preload': preload},
            verbose='critical',
        )
        if self.rename_channels:
            if rename := {k: v for k, v in self.rename_channels.items() if k in raw.ch_names}:
                raw.rename_channels(rename)
        if self.montage:
            raw.set_montage(self.montage)
        # Empty room may have missing raw.info['dig'], find alternative ways to do this when encountered
        # if not raw.info['dig'] and self._dig_sessions is not None and self._dig_sessions[subject]:
        #     dig_recording = self._dig_sessions[subject][recording]
        #     if dig_recording != recording:
        #         dig_raw = self._load(subject, dig_recording, False)
        #         raw.set_montage(mne.channels.DigMontage(dig=dig_raw.info['dig']))
        return raw

    def get_adjacency(self, data: str) -> Union[str, List[Tuple[str, str]], Path]:
        if data == 'eog':
            return None
        else:
            return self.adjacency

    def get_sysname(
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

    def load_bad_channels(
            self,
            path: BIDSPath,
            existing: Collection[str] = None,
    ) -> list[str]:
        bads_path = self.get_path(path, 'bads')
        channels_df = pd.read_csv(bads_path, sep='\t')
        if 'status' not in channels_df.columns.tolist():
            self.log.info("Generating bad_channels for %s", path.fpath)
            self.make_bad_channels_auto(path)
        bad_chs = channels_df.query('status == "bad"')['name'].tolist()
        if existing is not None:
            bad_chs = [ch for ch in bad_chs if ch in existing]
        return bad_chs

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs,
            redo: bool,
    ) -> None:
        bads_path = self.get_path(path, 'bads')
        channels_df = pd.read_csv(bads_path, sep='\t')
        if 'status' not in channels_df.columns.tolist():
            old_bads = None
        else:
            old_bads = channels_df.query('status == "bad"')['name'].tolist()
        # find new bad channels
        if isinstance(bad_chs, (str, int)):
            bad_chs = (bad_chs,)
        raw = self.load(path, add_bads=False)
        sensor = load.mne.sensor_dim(raw.info, adjacency=self.adjacency)
        new_bads = sensor._normalize_sensor_names(bad_chs)
        # update with old bad channels
        if old_bads is not None and not redo:
            new_bads = sorted(set(old_bads).union(new_bads))
        # print change
        print(f"{old_bads} -> {new_bads}")
        if new_bads == old_bads:
            return
        # write new bad channels
        mark_channels(path, ch_names=new_bads, status='bad')

    def make_bad_channels_auto(
            self,
            path: BIDSPath,
            flat: float = None,
            redo: bool = False,
    ) -> None:
        raw = self.load(path, add_bads=False)
        bad_chs: list[str] = raw.info['bads']
        if flat is None:
            flat = 1e-14  # May need a setting to exclude the EEG reference?
        if flat:
            sysname = self.get_sysname(raw.info, path.entities['subject'], None)
            raw = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=self.adjacency)
            bad_chs.extend(raw.sensor.names[raw.std('time') < flat])
        self.make_bad_channels(path, bad_chs, redo)

    def mtime(
            self,
            path: BIDSPath,
            bad_chs: bool = True,
    ) -> float | None:
        raw_path = self.get_path(path)
        if not exists(raw_path):
            raise FileMissingError(f"Raw input file does not exist at expected location {raw_path}")
        bads_path = self.get_path(path, 'bads')
        if bad_chs:
            return max(getmtime(raw_path), getmtime(bads_path))
        else:
            return getmtime(raw_path)


class CachedRawPipe(RawPipe):
    _bad_chs_affect_cache: bool = False
    source: RawPipe = None  # set on linking

    def __init__(self, source, cache=True):
        RawPipe.__init__(self)
        self._source_name = source
        self._cache = cache

    def _can_link(self, pipes: Dict[str, RawPipe]) -> bool:
        return self._source_name in pipes

    def _link(
            self,
            name: str,
            pipes: Dict[str, RawPipe],
            cache_dir: str,
            deriv_dir: str,
            log: logging.Logger,
    ) -> RawPipe:
        if self._source_name not in pipes:
            raise DefinitionError(f"{self.__class__.__name__} {name!r} source {self._source_name!r} does not exist")
        out = RawPipe._link_base(self, name, log)
        out.source = pipes[self._source_name]
        out.cache_dir = cache_dir
        out.deriv_dir = deriv_dir
        return out

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = RawPipe._as_dict(self, args)
        out['source'] = self._source_name
        return out

    def cache(
            self,
            path: BIDSPath,
            load: bool = False,
            preload: PreloadArg = False,
    ) -> mne.io.BaseRaw | None:
        "Make sure the cache is up to date"
        cache_path = self.get_path(path, 'cache')
        if exists(cache_path):
            mtime = self.mtime(path, self._bad_chs_affect_cache)
            if mtime and getmtime(cache_path) >= mtime:
                if load:
                    return mne.io.read_raw_fif(cache_path, preload=preload, verbose='critical')
                else:
                    return None
        from .. import __version__
        # make sure the target directory exists
        makedirs(dirname(cache_path), exist_ok=True)
        # generate new raw
        with CaptureLog(cache_path[:-3] + 'log') as logger:
            logger.info(f"eelbrain {__version__}")
            logger.info(f"mne {mne.__version__}")
            logger.info(repr(self._as_dict()))
            raw = self._make(path, True)
        # save
        try:
            raw.save(cache_path, overwrite=True, verbose='critical')
        except BaseException:
            # clean up potentially corrupted file
            if exists(cache_path):
                remove(cache_path)
            raise
        return raw

    def get_adjacency(self, data: str) -> Union[str, List[Tuple[str, str]], Path]:
        return self.source.get_adjacency(data)

    def get_sysname(
            self,
            info: mne.Info,
            subject: str,
            data: str,
    ) -> str | None:
        return self.source.get_sysname(info, subject, data)

    def load(
            self,
            path: BIDSPath,
            add_bads: AddBadsArg = True,
            preload: PreloadArg = False,  # -1: info only, data will not be needed
            raw: mne.io.BaseRaw = None,
    ) -> mne.io.BaseRaw:
        if raw is not None:
            pass
        elif self._cache:
            raw = self.cache(path, load=True, preload=preload)
        elif preload == -1:
            raw = self._make_info(path)
        else:
            raw = self._make(path, preload)
        return RawPipe.load(self, path, add_bads, preload, raw)

    def load_bad_channels(
            self,
            path: BIDSPath,
            existing: Collection[str] = None,
    ) -> list[str]:
        return self.source.load_bad_channels(path, existing)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
    ) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _make_info(self, path: BIDSPath) -> mne.io.BaseRaw:
        return self.source.load(path, preload=-1)

    def make_bad_channels(
            self,
            path: BIDSPath,
            bad_chs,
            redo: bool,
    ) -> None:
        self.source.make_bad_channels(path, bad_chs, redo)

    def make_bad_channels_auto(self, *args, **kwargs) -> None:
        self.source.make_bad_channels_auto(*args, **kwargs)

    def mtime(
            self,
            path: BIDSPath,
            bad_chs: bool = True,
    ) -> float | None:
        return self.source.mtime(path, bad_chs or self._bad_chs_affect_cache)


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
    def __init__(
            self,
            source: str,
            l_freq: float = None,
            h_freq: float = None,
            cache: bool = True,
            n_jobs: Union[str, int, None] = 1,
            **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.args = (l_freq, h_freq)
        self.kwargs = kwargs
        self.n_jobs = n_jobs
        # mne backwards compatibility (fir_design default change 0.15 -> 0.16)
        if 'use_kwargs' in kwargs:
            self._use_kwargs = kwargs.pop('use_kwargs')
        else:
            self._use_kwargs = kwargs

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'args', 'kwargs'])

    def filter_ndvar(self, ndvar, **kwargs):
        return filter_data(ndvar, *self.args, **self._use_kwargs, **kwargs)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True)
        self.log.info("Raw %s: filtering for %s...", self.name, path.fpath)
        raw.filter(*self.args, **self._use_kwargs, n_jobs=self.n_jobs)
        return raw

    def _make_info(self, path: BIDSPath) -> mne.io.BaseRaw:
        raw = super()._make_info(path)
        l_freq, h_freq = self.args
        if l_freq and l_freq > (raw.info['highpass'] or 0):
            with raw.info._unlock():
                raw.info['highpass'] = float(l_freq)
        if h_freq and h_freq < (raw.info['lowpass'] or raw.info['sfreq']):
            with raw.info._unlock():
                raw.info['lowpass'] = float(h_freq)
        return raw


class RawFilterElliptic(CachedRawPipe):

    def __init__(self, source, low_stop, low_pass, high_pass, high_stop, gpass, gstop):
        CachedRawPipe.__init__(self, source)
        self.args = (low_stop, low_pass, high_pass, high_stop, gpass, gstop)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'args'])

    def _sos(self, sfreq):
        nyq = sfreq / 2.
        low_stop, low_pass, high_pass, high_stop, gpass, gstop = self.args
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

    def filter_ndvar(self, ndvar):
        axis = ndvar.get_axis('time')
        sos = self._sos(1. / ndvar.time.tstep)
        x = signal.sosfilt(sos, ndvar.x, axis)
        return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)

    def _make(
            self,
            path: BIDSPath,
            preload: bool,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True)
        self.log.info("Raw %s: filtering for %s...", self.name, path.fpath)
        # filter data
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        sos = self._sos(raw.info['sfreq'])
        for i in picks:
            raw._data[i] = signal.sosfilt(sos, raw._data[i])
        # update info
        low, high = self.args[1], self.args[2]
        if high and raw.info['lowpass'] > high:
            raw.info['lowpass'] = float(high)
        if low and raw.info['highpass'] < low:
            raw.info['highpass'] = float(low)
        return raw


class RawICA(CachedRawPipe):
    """ICA raw pipe

    Parameters
    ----------
    source
        Name of the raw pipe to use for input data.
    task
        Session(s) to use for estimating ICA components.
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
    using the data specified in the ``task`` parameter. The selected
    components are then removed from all data sessions during this preprocessing
    step, regardless of whether they were used to estimate the components or
    not.

    Use :meth:`Pipeline.make_ica_selection` for each subject to
    select ICA components that should be removed. The arguments to that function
    determine what data is used to visualize the component time courses.
    For example, to determine which components load strongly on empty room data,
    use ``e.make_ica_selection(task='emptyroom')`` (assuming an
    ``'emptyroom'`` task is present).

    This step merges bad channels from all sessions.

    Examples
    --------
    Some ICA examples::

        class Experiment(Pipeline):

            raw = {
                '1-40': RawFilter('raw', 1, 40),
                # Extended infomax with PCA preprocessing
                'ica': RawICA('1-40', 'extended-infomax', n_components=0.99),
                # Fast ICA
                'fastica': RawICA('1-40', 'task', 'fastica', n_components=0.9),
                # Change thresholds for data rejection using fit_kwargs
                'ica-rej': RawICA('1-40', 'task', 'fastica', fit_kwargs=dict(
                    reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 500e-6},
                )),
            }

    """

    def __init__(
            self,
            source: str,
            task: Union[str, Sequence[str]],
            method: str = 'extended-infomax',
            random_state: int = 0,
            fit_kwargs: Dict[str, Any] = None,
            cache: bool = False,
            **kwargs,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.task = tuple_arg('task', task, allow_none=False)
        self.kwargs = {'method': method, 'random_state': random_state, **kwargs}
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, [*args, 'task', 'kwargs'])
        if self.fit_kwargs:
            out['fit_kwargs'] = self.fit_kwargs
        return out

    def load_bad_channels(
            self,
            path: BIDSPath,
            existing: Collection[str] = None,
    ) -> list[str]:
        bad_chs = set()
        for task in self.task:
            path_ = path.copy().update(task=task)
            bad_chs.update(self.source.load_bad_channels(path_, existing))
        return sorted(bad_chs)

    def load_ica(self, path: BIDSPath) -> mne.preprocessing.ICA:
        ica_path = self.get_path(path, 'ica')
        if not exists(ica_path):
            raise FileMissingError(f"ICA file {basename(ica_path)} does not exist for raw={self.name!r}. Run e.make_ica_selection() to create it.")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Version 0.23 introduced max_iter', DeprecationWarning)
            return mne.preprocessing.read_ica(ica_path)

    @staticmethod
    def _check_ica_channels(
            ica: mne.preprocessing.ICA,
            raw: mne.io.BaseRaw,
            raise_on_mismatch: bool = False,
            raw_name: str = None,
            subject: str = None,
            return_missing: bool = False,  # if ICA is only missing channels, retrun those
    ) -> bool | tuple:
        picks = mne.pick_types(raw.info, meg=True, eeg=True, ref_meg=False)
        raw_ch_names = [raw.ch_names[i] for i in picks]
        names_match = ica.ch_names == raw_ch_names
        if return_missing:
            assert raise_on_mismatch
            if names_match:
                return ()
            ica_set = set(ica.ch_names)
            raw_set = set(raw_ch_names)
            if not ica_set - raw_set:
                if missing := raw_set - ica_set:
                    return tuple(missing)
        if raise_on_mismatch and not names_match:
            raise RuntimeError(f"The ICA channel names do not match the data channels for raw={raw_name!r}, {subject=}. Have the bad channels changed since the ICA was computed? Try to revert the data channels, or recompute the ICA using Pipeline.make_ica().\nData: {', '.join(raw_ch_names)}\nICA:  {', '.join(ica.ch_names)}")
        return names_match

    def load_concatenated_source_raw(
            self,
            path: BIDSPath,
            tasks: list[str],
            runs: list[str],
    ) -> mne.io.BaseRaw:
        bad_channels = self.load_bad_channels(path)
        path_list = []
        for task in tasks:
            path_ = path.copy().update(task=task)
            if not runs:
                path_list.append(path_)
                continue
            for run in runs:
                path_list.append(path_.copy().update(run=run))
        raw = self.source.load(path_list[0], bad_channels)
        for path_ in path_list[1:]:
            raw_ = self.source.load(path_, bad_channels)
            raw.append(raw_)
        return raw

    def make_ica(
            self,
            path: BIDSPath,
            run: list[str],
    ) -> str:
        ica_path = self.get_path(path, 'ica')
        bad_channels = self.load_bad_channels(path)
        if exists(ica_path):
            raw = self.source.load(path.copy().update(task=self.task[0]), bad_channels, preload=-1)
            ica = mne.preprocessing.read_ica(ica_path)
            # equal channel names in different raw is guaranteed here
            if not self._check_ica_channels(ica, raw):
                self.log.info("Raw %s for subject=%r: ICA channels mismatch data channels, recomputing ICA...", self.name, path.entities['subject'])
            else:
                mtimes = [self.source.mtime(path.copy().update(task=task), self._bad_chs_affect_cache) for task in self.task]
                if all(mtimes) and getmtime(ica_path) > max(mtimes):
                    return ica_path
                # ICA file is newer than raw
                command = ask(f"The input for the ICA of {path.entities['subject']} seems to have changed since the ICA was generated.", {'delete': 'delete and recompute the ICA', 'ignore': 'Keep using the old ICA'}, help="This message indicates that the modification date of the raw input data or of the bad channels file is more recent than that of the ICA file. If the data actually changed, ICA components might not be valid anymore and should be recomputed. If the change is spurious (e.g., the raw file was modified in a way that does not affect the ICA) load and resave the ICA file to stop seeing this message.")
                if command == 'ignore':
                    return ica_path
                elif command == 'delete':
                    remove(ica_path)
                else:
                    raise RuntimeError(f"{command=}")

        raw = self.load_concatenated_source_raw(path, self.task, run)
        self.log.info("Raw %s: computing ICA decomposition for %s", self.name, path.entities['subject'])
        kwargs = self.kwargs.copy()
        kwargs.setdefault('max_iter', 256)
        if MNE_VERSION > V0_19 and kwargs['method'] == 'extended-infomax':
            kwargs['method'] = 'infomax'
            kwargs['fit_params'] = {'extended': True}

        ica = mne.preprocessing.ICA(**kwargs)
        # reject presets from meeg-preprocessing
        fit_kwargs = {'reject': {'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}, **self.fit_kwargs}
        with user_activity:
            ica.fit(raw, **fit_kwargs)
        makedirs(dirname(ica_path), exist_ok=True)
        if MNE_VERSION >= V0_24:
            ica.save(ica_path, overwrite=True)
        else:
            ica.save(ica_path)
        return ica_path

    def _make(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True)
        return self._apply(raw, path, self.name)

    def _apply(
            self,
            raw: mne.io.BaseRaw,
            path: BIDSPath,
            raw_name: str,
    ) -> mne.io.BaseRaw:
        self.log.debug("Raw %s: applying ICA for %s...", raw_name, path.fpath)
        raw.info['bads'] = self.load_bad_channels(path, existing=raw.ch_names)
        ica = self.load_ica(path)
        missing = self._check_ica_channels(ica, raw, raise_on_mismatch=True, raw_name=raw_name, subject=path.entities['subject'], return_missing=True)
        if missing:
            raw.drop_channels(missing)
        ica.apply(raw)
        return raw

    def mtime(
            self,
            path: BIDSPath,
            bad_chs: bool = True,
    ) -> float | None:
        mtime = CachedRawPipe.mtime(self, path, bad_chs or self._bad_chs_affect_cache)
        if mtime:
            ica_path = self.get_path(path, 'ica')
            if exists(ica_path):
                return max(mtime, getmtime(ica_path))


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
    ica_source = None  # set on linking

    def __init__(
            self,
            source: str,
            ica: str,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self._ica_source = ica

    def _can_link(self, pipes: Dict[str, RawPipe]) -> bool:
        return CachedRawPipe._can_link(self, pipes) and self._ica_source in pipes

    def _link(
            self,
            name: str,
            pipes: Dict[str, RawPipe],
            cache_dir: str,
            deriv_dir: str,
            log: logging.Logger,
    ) -> RawPipe:
        out = CachedRawPipe._link(self, name, pipes, cache_dir, deriv_dir, log)
        out.ica_source = pipes[self._ica_source]
        return out

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, args)
        out['ica_source'] = self._ica_source
        return out

    def load_bad_channels(
            self,
            path: BIDSPath,
            existing: Collection[str] = None,
    ) -> list[str]:
        return self.ica_source.load_bad_channels(path, existing)

    def _make(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True)
        return self.ica_source._apply(raw, path, self.name)

    def mtime(
            self,
            path: BIDSPath,
            bad_chs: bool = True,
    ) -> float | None:
        mtime = CachedRawPipe.mtime(self, path, bad_chs)
        if mtime:
            ica_mtime = self.ica_source.mtime(path, bad_chs)
            if ica_mtime:
                return max(mtime, ica_mtime)


class RawMaxwell(CachedRawPipe):
    """Maxwell filter raw pipe

    Parameters
    ----------
    source : str
        Name of the raw pipe to use for input data.
    bad_condition : str
        How to deal with ill-conditioned SSS matrices; by default, an error is
        raised, which might prevent the process to complete for some subjects.
        Set to ``'warning'`` to proceed anyways.
    cache : bool
        Cache the resulting raw files (default ``True``).
    ...
        :func:`mne.preprocessing.maxwell_filter` parameters.

    See Also
    --------
    Pipeline.raw
    """

    _bad_chs_affect_cache = True

    def __init__(self, source, bad_condition='error', cache=True, **kwargs):
        CachedRawPipe.__init__(self, source, cache)
        self.kwargs = kwargs
        self.bad_condition = bad_condition

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'kwargs'])

    def _make(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path)
        self.log.info("Raw %s: computing Maxwell filter for %s", self.name, path.fpath)
        with user_activity:
            return mne.preprocessing.maxwell_filter(raw, bad_condition=self.bad_condition, **self.kwargs)


class RawOversampledTemporalProjection(CachedRawPipe):
    """Oversampled temporal projection: see :func:`mne.preprocessing.oversampled_temporal_projection`"""

    def __init__(
            self,
            source: str,
            duration: float = 10.0,
            cache: bool = True,
    ):
        CachedRawPipe.__init__(self, source, cache)
        self.duration = duration

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'duration'])

    def _make(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path)
        self.log.info("Raw %s: computing oversampled temporal projection for %s", self.name, path.fpath)
        with user_activity:
            return mne.preprocessing.oversampled_temporal_projection(raw, self.duration)


class RawUpdateBadChannels(CachedRawPipe):

    def __init__(
            self,
            source: str,
            bad_channels: Dict[str, Sequence[str]],
    ):
        CachedRawPipe.__init__(self, source, False)
        self.bad_channels = bad_channels
        self._pattern_keys = [key for key in bad_channels if ('*' in key or '?' in key)]

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        return CachedRawPipe._as_dict(self, [*args, 'bad_channels'])

    def _make(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        return self.source.load(path, preload=preload)

    def load_bad_channels(
            self,
            path: BIDSPath,
            existing: Collection[str] = None,
    ) -> list[str]:
        bad_channels = self.source.load_bad_channels(path, existing)
        subject = path.entities['subject']
        if subject in self.bad_channels:
            key = subject
        else:
            for key in self._pattern_keys:
                if fnmatch.fnmatch(subject, key):
                    break
            else:
                raise KeyError(subject)
        return sorted(set(bad_channels).union(self.bad_channels[key]))


class RawReReference(CachedRawPipe):
    """Re-reference EEG data

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
    def __init__(
            self,
            source: str,
            reference: Union[str, Sequence[str]] = 'average',
            add: Union[str, Sequence[str]] = None,
            drop: Union[str, Sequence[str]] = None,
            cache: bool = False,
    ):
        CachedRawPipe.__init__(self, source, cache)
        if not isinstance(reference, str):
            reference = tuple_arg('reference', reference, allow_none=False)
        self.reference = reference
        self.add = tuple_arg('add', add)
        self.drop = tuple_arg('drop', drop)

    def _as_dict(self, args: Sequence[str] = ()) -> dict:
        out = CachedRawPipe._as_dict(self, [*args, 'reference'])
        if self.add:
            out['add'] = self.add
        if self.drop:
            out['drop'] = self.drop
        return out

    @staticmethod
    def _normalize_dict(state: dict) -> None:
        if not isinstance(state['reference'], str):
            state['reference'] = tuple_arg('reference', state['reference'])
        for key in ['add', 'drop']:
            if key in state:
                state[key] = tuple_arg(key, state[key])

    def _make(
            self,
            path: BIDSPath,
            preload: PreloadArg,
    ) -> mne.io.BaseRaw:
        raw = self.source.load(path, preload=True)
        if self.add:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The locations of multiple reference channels are ignored', module='mne')
                raw = mne.add_reference_channels(raw, self.add, copy=False)
            # apply new channel position
            pipe = self.source
            while not isinstance(pipe, RawSource):
                pipe = pipe.source
            if pipe.montage:
                raw.set_montage(pipe.montage)
        raw.set_eeg_reference(list(self.reference))
        if self.drop:
            raw = raw.drop_channels(self.drop)
        return raw

    def _make_info(self, path: BIDSPath) -> mne.io.BaseRaw:
        if self.add or self.drop:
            return self._make(path, False)
        else:
            return super()._make_info(path)


def assemble_pipeline(
        raw: dict[str, RawPipe],
        tasks: tuple[str],
        cache_dir: str,
        deriv_dir: str,
        log: logging.Logger,
) -> dict[str, RawPipe]:
    "Assemble preprocessing pipeline form a definition in a dict"
    linked_raw = {}
    while raw:
        n = len(raw)
        for key in list(raw):
            if raw[key]._can_link(linked_raw):
                pipe = raw.pop(key)._link(key, linked_raw, cache_dir, deriv_dir, log)
                if isinstance(pipe, RawICA):
                    missing = set(pipe.task).difference(tasks)
                    if missing:
                        raise DefinitionError(f"RawICA {key!r} lists one or more non-exising tasks: {', '.join(missing)}")
                linked_raw[key] = pipe
        if len(raw) == n:
            raise DefinitionError(f"Unable to resolve source for raw {enumeration(raw)}, circular dependency?")
    return linked_raw


###############################################################################
# Comparing pipelines
######################


def compare_pipelines(
        old: Dict[str, Dict],
        new: Dict[str, Dict],
        log: logging.Logger,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return a tuple of raw keys for which definitions changed

    Parameters
    ----------
    old
        A {name: params} dict for the previous preprocessing pipeline.
    new
        Current pipeline.
    log
        Logger for logging changes.

    Returns
    -------
    bad_raw : {str: str}
        ``{pipe_name: status}`` dictionary. Status can be 'new', 'removed' or
        'changed'.
    bad_ica : {str: str}
        Same as ``bad_raw`` but only for RawICA pipes (for which ICA files
        might have to be removed).
    """
    out = {}  # status:  good, changed, new, removed
    to_check = []  # need to check whether source is still valid
    keys = set(new).union(old)
    for key in keys:
        new_dict = new.get(key)
        old_dict = old.get(key)
        if new_dict is None:
            out[key] = 'removed'
        elif old_dict is None:
            out[key] = 'new'
        elif new_dict == old_dict:
            if key == 'raw':
                out[key] = 'good'
            else:
                to_check.append(key)
            continue
        else:
            out[key] = 'changed'
        log_dict_change(log, 'raw', key, old_dict, new_dict)

    # secondary changes
    while to_check:
        n = len(to_check)
        for key in tuple(to_check):
            parents = [new[key][k] for k in ('source', 'ica_source') if k in new[key]]
            if any(p not in out for p in parents):
                continue
            elif all(out[p] == 'good' for p in parents):
                out[key] = 'good'
            else:
                out[key] = 'changed'
                log.warning(f"  raw {key} parent changed")
            to_check.remove(key)
        if len(to_check) == n:
            raise RuntimeError("Queue not decreasing")

    bad_raw = {k: v for k, v in out.items() if v != 'good'}
    bad_ica = {k: v for k, v in bad_raw.items() if new.get(k, old.get(k))['type'] == 'RawICA'}
    return bad_raw, bad_ica


def ask_to_delete_ica_files(
        raw: mne.io.BaseRaw,
        status: str,
        filenames: list[str],
) -> None:
    "Ask whether outdated ICA files should be removed and act accordingly"
    if status == 'new':
        msg = ("The definition for raw=%r has been added, but ICA-files "
               "already exist. These files might not correspond to the new "
               "settings and should probably be deleted." % (raw,))
    elif status == 'removed':
        msg = ("The definition for raw=%r has been removed. The corresponsing "
               "ICA files should probably be deleted:" % (raw,))
    elif status == 'changed':
        msg = ("The definition for raw=%r has changed. The corresponding ICA "
               "files should probably be deleted." % (raw,))
    else:
        raise RuntimeError("status=%r" % (status,))
    command = ask(
        "%s Delete %i files?" % (msg, len(filenames)),
        (('abort', 'abort to fix the raw definition and try again'),
         ('delete', 'delete the invalid files'),
         ('ignore', 'pretend that the files are valid; you will not be warned again')))

    if command == 'delete':
        for filename in filenames:
            remove(filename)
    elif command == 'abort':
        raise RuntimeError("User abort")
    elif command != 'ignore':
        raise RuntimeError("command=%r" % (command,))


def normalize_dict(raw: dict) -> None:
    "Normalize pipeline state with latest pipeline classes"
    for key, params in raw.items():
        pipe_class = globals()[params['type']]
        pipe_class._normalize_dict(params)


def remove_task(fname: str) -> str:
    parts = fname.split('_')
    filtered_parts = [part for part in parts if not part.startswith('task-')]
    return '_'.join(filtered_parts)


def remove_subject(fname: str) -> str:
    parts = fname.split('_')
    filtered_parts = [part for part in parts if not part.startswith('sub-')]
    return '_'.join(filtered_parts)


def get_subject_session(fname: str) -> str:
    parts = fname.split('_')
    filtered_parts = [part for part in parts if (part.startswith('sub-') or part.startswith('ses-'))]
    return '_'.join(filtered_parts)
