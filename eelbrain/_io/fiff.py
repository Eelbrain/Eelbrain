# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""I/O for MNE"""
from collections.abc import Iterable
import fnmatch
from functools import reduce
from itertools import chain, zip_longest
from logging import getLogger
from math import floor
import operator
import os
from pathlib import Path
import re
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union
import warnings

import mne
from mne.source_estimate import _BaseSourceEstimate
from mne.io.constants import FIFF
from mne.io.kit.constants import KIT
import numpy as np

from .. import _info
from .._types import PathArg
from .._data_obj import Dimension, Factor, Var, NDVar, Dataset, Case, Sensor, Space, SourceSpace, VolumeSourceSpace, UTS, _matrix_graph
from .._info import BAD_CHANNELS
from .._text import n_of
from .._utils import deprecate_kwarg, ui, as_list
from ..mne_fixes import MNE_EVOKED, MNE_RAW, MNE_VOLUME_STC
from ..mne_fixes._channels import _adjacency_id


try:  # mne >= 0.19
    from mne.io.kit.constants import KIT_NEIGHBORS
except ImportError:
    KIT_NEIGHBORS = {
        KIT.SYSTEM_NYU_2008: 'KIT-157',
        KIT.SYSTEM_NYU_2009: 'KIT-157',
        KIT.SYSTEM_NYU_2010: 'KIT-157',
        KIT.SYSTEM_NYUAD_2011: 'KIT-208',
        KIT.SYSTEM_NYUAD_2012: 'KIT-208',
        KIT.SYSTEM_NYUAD_2014: 'KIT-208',
        KIT.SYSTEM_UMD_2004: 'KIT-UMD-1',
        KIT.SYSTEM_UMD_2014_07: 'KIT-UMD-2',
        KIT.SYSTEM_UMD_2014_12: 'KIT-UMD-3',
    }

BaselineArg = Optional[Tuple[Optional[float], Optional[float]]]
AdjacencyArg = Union[str, Sequence[Tuple[str, str]], np.ndarray]
DataArg = Literal['eeg', 'mag', 'grad']
PicksArg = Any

TOPOGRAPHIC_CH_TYPES = ('mag', 'grad', 'planar1', 'planar2', 'eeg')
DATA_CH_TYPES = ('csd', 'seeg', 'ecog', 'dbs', 'eog', 'emg', 'ecg', 'resp')


def mne_neighbor_files():
    neighbor_dir = Path(mne.channels.__file__).parent / 'data' / 'neighbors'
    return [re.match(r'([\w-]+)_neighb', path.stem).group(1) for path in neighbor_dir.glob('*_neighb.mat')]


def mne_raw(
        path: PathArg = None,
        proj: Union[bool, str] = False,
        **kwargs,
) -> mne.io.BaseRaw:
    """Load a :class:`mne.io.Raw` object

    Parameters
    ----------
    path
        path to a raw fiff or sqd file. If no path is supplied, a file can be
        chosen from a file dialog.
    proj
        Add projections from a separate file to the Raw object.
        **``False``**: No proj file will be added.
        **``True``**: ``'{raw}*proj.fif'`` will be used.
        ``'{raw}'`` will be replaced with the raw file's path minus '_raw.fif',
        and '*' will be expanded using fnmatch. If multiple files match the
        pattern, a ValueError will be raised.
        **``str``**: A custom path template can be provided, ``'{raw}'`` and
        ``'*'`` will be treated as with ``True``.
    **kwargs
        Additional keyword arguments are forwarded to :class:`mne.io.Raw`
        initialization.
    """
    if path is None:
        path = ui.ask_file("Pick a raw data file", "Pick a raw data file",
                           [('Functional image file (*.fif)', '*.fif'),
                            ('KIT Raw File (*.sqd,*.con', '*.sqd;*.con')])
        if not path:
            return

    if isinstance(path, str):
        path = Path(path)
    if isinstance(path, Path):
        ext = path.suffix.lower()
        if ext.startswith('.fif'):
            raw = mne.io.read_raw_fif(path, **kwargs)
        elif ext in ('.sqd', '.con'):
            raw = mne.io.read_raw_kit(path, **kwargs)
        else:
            raise ValueError(f"{path}: Unknown extension")
    elif isinstance(path, Iterable):
        # MNE Raw supports list of file-names
        raw = mne.io.read_raw_fif(path, **kwargs)
    else:
        raise TypeError("path=%r" % (path,))

    if proj:
        if proj is True:
            proj = '{raw}*proj.fif'

        if '{raw}' in proj:
            raw_file = raw.filenames[0]
            raw_root, _ = os.path.splitext(raw_file)
            raw_root = raw_root.rstrip('raw')
            proj = proj.format(raw=raw_root)

        if '*' in proj:
            head, tail = os.path.split(proj)
            names = fnmatch.filter(os.listdir(head), tail)
            if len(names) == 1:
                proj = os.path.join(head, names[0])
            else:
                if len(names) == 0:
                    err = "No file matching %r"
                else:
                    err = "Multiple files matching %r"
                raise ValueError(err % proj)

        # add the projections to the raw file
        proj = mne.read_proj(proj)
        raw.add_proj(proj, remove_existing=True)

    return raw


def events(
        raw: Union[PathArg, mne.io.BaseRaw] = None,
        merge: int = None,
        proj: Union[bool, str] = False,
        name: str = None,
        bads: List = None,
        stim_channel: Union[str, Sequence[str]] = None,
        events: str = None,
        annotations: bool = None,
        feature: Literal['onset', 'offset', 'step'] = 'onset',
        **kwargs,
) -> Dataset:
    """
    Load events from a raw fiff file.

    Parameters
    ----------
    raw
        The raw fiff file from which to extract events (if raw and events are
        both ``None``, a file dialog will be displayed to select a raw file).
    merge
        Merge stimulus channel steps occurring in neighboring samples.
        The integer value indicates across how many samples events should
        be merged, and the sign indicates in which direction they should be
        merged (negative means towards the earlier event, positive towards
        the later event). By default, this parameter is based on the data:
        -1 for KIT data, 0 otherwise.
        The main reason for merging events is an artifact from analog event
        recording systems. If events are recorded in an analog channel,
        event onsets can be blurred (what should be ``0, 0, 0, 1, 1, 1, ...``,
        can look like ``0, 0, 0, 0.5, 1, 1, ...``; ``merge=-1`` would
        turn the latter into the former).
    proj
        Path to the projections file that will be loaded with the raw file.
        ``'{raw}'`` will be expanded to the raw file's path minus extension.
        With ``proj=True``, ``'{raw}_*proj.fif'`` will be used,
        looking for any projection file starting with the raw file's name.
        If multiple files match the pattern, a ValueError will be raised.
    name
        A name for the Dataset. If ``None``, the raw filename will be used.
    bads
        Specify additional bad channels in the raw data file (these are added
        to the ones that are already defined in the raw file).
    stim_channel
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will default to
        'STI 014'.
    events
        If events are stored in a fiff file separate from the Raw object, the
        path to the events file can be supplied here. The events in the Dataset
        will reflect the events in the events file rather than the raw file.
    annotations
        Generate events from annotations instead of the stim channel (by
        default, annotations are used when present).
    **
        Keyword arguments for loading the raw file (see
        :func:`mne.io.read_raw_kit` or :func:`mne.io.read_raw_kit`).

    Returns
    -------
    events
        A Dataset with the following variables:
         - *i_start*: the index of the event in the raw file.
         - *trigger*: the event value/id.
         - *event*: the event label (only for events from annotations).
        The Dataset's info dictionary contains the following values:
         - *raw*: the mne Raw object.

    """
    if (raw is None and events is None) or isinstance(raw, (str, Path)):
        raw = mne_raw(raw, proj=proj, **kwargs)

    if bads is not None and raw is not None:
        raw.info['bads'].extend(bads)

    if name is None and raw is not None:
        raw_path = raw.filenames[0]
        if isinstance(raw_path, str):
            name = os.path.basename(raw_path)
        else:
            name = None

    labels = None
    if events is None:
        if annotations is None:
            regex = re.compile('(bad|edge)', re.IGNORECASE)
            index = [not regex.match(desc) for desc in raw.annotations.description]
            if any(index):
                annotations = raw.annotations[index]

        if annotations:
            events, event_ids = mne.events_from_annotations(raw)
        else:
            if merge is None:
                if raw.info.get('kit_system_id') is None:
                    merge = 0
                else:
                    merge = -1
            # Use mne.find_stim_steps() becaue mne.find_events() merges forward (i.e. coerces merge to be ≥ 0); loop through channels because find_stim_steps() only uses the first channel
            if stim_channel is None:
                stim_channels = [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True)]
            elif isinstance(stim_channel, str):
                stim_channels = [stim_channel]
            else:
                stim_channels = stim_channel

            event_list = []
            for channel_i in stim_channels:
                events = mne.find_stim_steps(raw, merge=merge, stim_channel=channel_i)
                if feature == 'onset':
                    events = events[np.flatnonzero(events[:, 2])]
                elif feature == 'offset':
                    events = events[~np.flatnonzero(events[:, 2])]
                elif feature != 'step':
                    raise ValueError(f"{feature=}")
                event_list.append(events)
            if len(event_list):
                events = np.concatenate(event_list, axis=0)
                events = events[np.argsort(events[:, 0])]
            else:
                events = np.empty((0, 3))
            event_ids = getattr(raw, 'event_id', None)

        if event_ids:
            labels = {event_id: label for label, event_id in event_ids.items()}
    else:
        events = mne.read_events(events)

    ds = Dataset({
        'i_start': Var(events[:, 0]),
        'trigger': Var(events[:, 2]),
    }, name, info={'raw': raw})
    if labels is not None:
        ds['event'] = Factor(ds['trigger'], labels=labels)
    return ds


def _guess_ndvar_data_type(info: mne.Info) -> str:
    "Guess which type of data to extract from an mne object"
    data_types = info.get_channel_types(unique=True)
    for ch_type in chain(TOPOGRAPHIC_CH_TYPES, DATA_CH_TYPES):
        if ch_type in data_types:
            return ch_type
    if data_types[0] == 'stim' and len(data_types) > 1:
        return data_types[1]
    return data_types[0]


def _picks(info, data, exclude) -> np.ndarray:
    meg = eeg = False
    kwargs = {'stim': False, 'ref_meg': False}
    if data is None:
        data = _guess_ndvar_data_type(info)
    if data == 'eeg':
        eeg = True
    elif data == 'eeg&eog':
        eeg = True
        kwargs['eog'] = True
    elif data in ['grad', 'mag', 'planar1', 'planar2']:
        meg = data
    elif data is True:
        meg = True
        eeg = True
    else:
        kwargs[data] = True
    return mne.pick_types(info, meg, eeg, exclude=exclude, **kwargs)


def _ndvar_epochs_reject(data, reject):
    if reject:
        if not np.isscalar(reject):
            raise TypeError(f"{reject=}: must be scalar (rejection threshold)")
        reject = {data: reject}
    else:
        reject = None
    return reject


def _sensor_info(
        data: str,
        vmax: float,
        mne_info: mne.Info,
        user_info: dict = None,
        mult: float = 1,
):
    if data == 'eeg' or data == 'eog' or data == 'eeg&eog':
        info = _info.for_eeg(vmax, mult)
        summary_vmax = 0.1 * vmax if vmax else None
        info['summary_info'] = _info.for_eeg(summary_vmax, mult)
    elif data == 'mag':
        info = _info.for_meg(vmax, mult)
        summary_vmax = 0.1 * vmax if vmax else None
        info['summary_info'] = _info.for_meg(summary_vmax, mult)
    elif data in ['grad', 'planar1', 'planar2']:
        info = _info.for_meg(vmax, mult, 'T/cm', '∆U')
        summary_vmax = 0.1 * vmax if vmax else None
        info['summary_info'] = _info.for_meg(summary_vmax, mult, 'T/cm', '∆U')
    else:
        info = {}

    if data in TOPOGRAPHIC_CH_TYPES:
        info['proj'] = 'z root'
    info['samplingrate'] = mne_info['sfreq']
    if user_info:
        info.update(user_info)
    return info


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def epochs(
        ds: Dataset,
        tmin: float = -0.1,
        tmax: float = None,
        baseline: BaselineArg = None,
        decim: int = 1,
        mult: float = 1,
        proj: bool = False,
        data: DataArg = None,
        reject: float = None,
        exclude: Union[str, Sequence[str]] = 'bads',
        info: dict = None,
        name: str = None,
        raw: mne.io.BaseRaw = None,
        sensors: Sensor = None,
        i_start: str = 'i_start',
        tstop: float = None,
        sysname: str = None,
        adjacency: AdjacencyArg = None,
) -> NDVar:
    """
    Load epochs as :class:`NDVar`.

    Parameters
    ----------
    ds
        Dataset containing a variable which defines epoch cues (i_start).
    tmin
        First sample to include in the epochs in seconds (Default is -0.1).
    tmax
        Last sample to include in the epochs in seconds (Default 0.6; use
        ``tstop`` instead to specify index exclusive of last sample).
    baseline
        Time interval for baseline correction. ``(tmin, tmax)`` tuple in
        seconds, or ``None`` to use all the data (e.g., ``(None, 0)`` uses all
        the data from the beginning of the epoch up to ``t = 0``). Set to
        ``None`` for no baseline correction (default).
    decim
        Downsample the data by this factor when importing. ``1`` means no
        downsampling. Note that this function does not low-pass filter
        the data. The data is downsampled by picking out every
        n-th sample (see `Wikipedia <http://en.wikipedia.org/wiki/Downsampling>`_).
    mult
        multiply all data by a constant.
    proj
        mne.Epochs kwarg (subtract projections when loading data)
    data
        Which data channels data to include (default based on channels in data).
    reject
        Threshold for rejecting epochs (peak to peak). Requires a for of
        mne-python which implements the Epochs.model['index'] variable.
    exclude
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    info
        Entries for the ndvar's info dict.
    name
        name for the new NDVar.
    raw
        Raw file providing the data; if ``None``, ``ds.info['raw']`` is used.
    sensors
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor instance can be supplied through this kwarg.
    i_start
        name of the variable containing the index of the events.
    tstop
        Alternative to ``tmax``. While ``tmax`` specifies the last samples to
        include, ``tstop`` specifies the sample before which to stop (standard
        Python indexing convention).
        For example, at 100 Hz the epoch with ``tmin=-0.1, tmax=0.4`` will have
        51 samples, while the epoch specified with ``tmin=-0.1, tstop=0.4`` will
        have 50 samples.
    sysname
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency
        Adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.

    Returns
    -------
    epochs
        The data epochs as (case, sensor, time) data.
    """
    if raw is None:
        raw = ds.info['raw']

    if data is None:
        data = _guess_ndvar_data_type(raw.info)
    picks = _picks(raw.info, data, exclude)
    reject = _ndvar_epochs_reject(data, reject)

    epochs_ = mne_epochs(ds, tmin, tmax, baseline, i_start, raw, decim=decim, picks=picks, reject=reject, proj=proj, tstop=tstop)
    ndvar = epochs_ndvar(epochs_, name, data, 'bads', mult, info, sensors, None, sysname, adjacency)

    if len(epochs_) == 0:
        raise RuntimeError(f"No events left in {raw.filenames[0]}")
    return ndvar


def add_epochs(ds, tmin=-0.1, tmax=0.6, baseline=None, decim=1, mult=1,
               proj=False, data=None, reject=None, exclude='bads', info=None,
               name="meg", raw=None, sensors=None, i_start='i_start',
               sysname=None, tstop=None):
    """
    Load epochs and add them to a dataset as :class:`NDVar`.

    Unless the ``reject`` argument is specified, ``ds``
    is modified in place. With ``reject``, a subset of ``ds`` is returned
    containing only those events for which data was loaded.

    Parameters
    ----------
    ds : Dataset
        Dataset containing a variable which defines epoch cues (i_start) and to
        which the epochs are added.
    tmin : scalar
        First sample to include in the epochs in seconds (Default is -0.1).
    tmax : scalar
        Last sample to include in the epochs in seconds (Default 0.6; use
        ``tstop`` instead to specify index exclusive of last sample).
    baseline : (float, float) | None
        Time interval for baseline correction. ``(tmin, tmax)`` tuple in
        seconds, or ``None`` to use all the data (e.g., ``(None, 0)`` uses all
        the data from the beginning of the epoch up to ``t = 0``). Set to
        ``None`` for no baseline correction (default).
    decim : int
        Downsample the data by this factor when importing. ``1`` means no
        downsampling. Note that this function does not low-pass filter
        the data. The data is downsampled by picking out every
        n-th sample (see `Wikipedia <http://en.wikipedia.org/wiki/Downsampling>`_).
    mult : scalar
        multiply all data by a constant.
    proj : bool
        mne.Epochs kwarg (subtract projections when loading data)
    data : 'eeg' | 'mag' | 'grad'
        Which data channels data to include (default based on channels in data).
    reject : None | scalar
        Threshold for rejecting epochs (peak to peak). Requires a for of
        mne-python which implements the Epochs.model['index'] variable.
    exclude : list of string | str
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    info : None | dict
        Entries for the ndvar's info dict.
    name : str
        name for the new NDVar.
    raw : None | mne Raw
        Raw file providing the data; if ``None``, ``ds.info['raw']`` is used.
    sensors : None | Sensor
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor instance can be supplied through this kwarg.
    i_start : str
        name of the variable containing the index of the events.
    sysname : str
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    tstop : scalar
        Alternative to ``tmax``: While ``tmax`` specifies the last samples to
        include, ``tstop`` can be used to specify the epoch time excluding the
        last time point (i.e., standard Python/Eelbrain indexing convention).
        For example, at 100 Hz the epoch with ``tmin=-0.1, tmax=0.4`` will have
        51 samples, while the epoch specified with ``tmin=-0.1, tstop=0.4`` will
        have 50 samples.

    Returns
    -------
    ds : Dataset
        Dataset containing the epochs. If no events are rejected, ``ds`` is the
        same object as the input ``ds`` argument, otherwise a copy of it.
    """
    if raw is None:
        raw = ds.info['raw']

    if data is None:
        data = _guess_ndvar_data_type(raw.info)
    picks = _picks(raw.info, data, exclude)
    reject = _ndvar_epochs_reject(data, reject)

    epochs_ = mne_epochs(ds, tmin, tmax, baseline, i_start, raw, decim=decim,
                         picks=picks, reject=reject, proj=proj, tstop=tstop)
    ds = _trim_ds(ds, epochs_)
    ds[name] = epochs_ndvar(epochs_, name, data, mult=mult, info=info,
                            sensors=sensors, sysname=sysname)
    return ds


def add_mne_epochs(ds, tmin=-0.1, tmax=None, baseline=None, target='epochs', **kwargs):
    """
    Load epochs and add them to a dataset as :class:`mne.Epochs`.

    If, after loading, the Epochs contain fewer cases than the Dataset, a copy
    of the Dataset is made containing only those events also contained in the
    Epochs. Note that the Epochs are always loaded with ``preload==True``.

    If the Dataset's info dictionary contains a 'bad_channels' entry, those bad
    channels are added to the epochs.


    Parameters
    ----------
    ds : Dataset
        Dataset with events from a raw fiff file (i.e., created by
        load.mne.events).
    tmin : scalar
        First sample to include in the epochs in seconds (Default is -0.1).
    tmax : scalar
        Last sample to include in the epochs in seconds (Default 0.6; use
        ``tstop`` instead to specify index exclusive of last sample).
    baseline : (float, float) | None
        Time interval for baseline correction. ``(tmin, tmax)`` tuple in
        seconds, or ``None`` to use all the data (e.g., ``(None, 0)`` uses all
        the data from the beginning of the epoch up to ``t = 0``). Set to
        ``None`` for no baseline correction (default).
    target : str
        Name for the Epochs object in the Dataset.
    ...
        See :func:`~eelbrain.load.mne.mne_epochs`.
    """
    epochs_ = mne_epochs(ds, tmin, tmax, baseline, **kwargs)
    ds = _trim_ds(ds, epochs_)
    ds[target] = epochs_
    return ds


def _mne_events(ds=None, i_start='i_start', trigger='trigger'):
    """Convert events from a Dataset into mne events"""
    if isinstance(i_start, str):
        i_start = ds[i_start]

    n_events = len(i_start)

    if isinstance(trigger, str):
        trigger = ds[trigger]
    elif trigger is None:
        trigger = np.ones(n_events)

    events = np.empty((n_events, 3), dtype=np.int32)
    events[:, 0] = i_start.x
    events[:, 1] = 0
    events[:, 2] = trigger
    return events


def mne_epochs(ds, tmin=-0.1, tmax=None, baseline=None, i_start='i_start',
               raw=None, drop_bad_chs=True, picks=None, reject=None, tstop=None,
               decim=1, **kwargs):
    """Load epochs as :class:`mne.Epochs`.

    Parameters
    ----------
    ds : Dataset
        Dataset containing a variable which defines epoch cues (i_start).
    tmin : scalar
        First sample to include in the epochs in seconds (Default is -0.1).
    tmax : scalar
        Last sample to include in the epochs in seconds (Default 0.6; use
        ``tstop`` instead to specify index exclusive of last sample).
    baseline : (float, float) | None
        Time interval for baseline correction. ``(tmin, tmax)`` tuple in
        seconds, or ``None`` to use all the data (e.g., ``(None, 0)`` uses all
        the data from the beginning of the epoch up to ``t = 0``). Set to
        ``None`` for no baseline correction (default).
    i_start : str
        name of the variable containing the index of the events.
    raw : None | mne Raw
        If None, ds.info['raw'] is used.
    drop_bad_chs : bool
        Drop all channels in raw.info['bads'] form the Epochs. This argument is
        ignored if the picks argument is specified.
    picks, reject
        :class:`mne.Epochs` parameters.
    tstop : scalar
        Alternative to ``tmax``. While ``tmax`` specifies the last samples to
        include, ``tstop`` specifies the sample before which to stop (standard
        Python indexing convention).
        For example, at 100 Hz the epoch with ``tmin=-0.1, tmax=0.4`` will have
        51 samples, while the epoch specified with ``tmin=-0.1, tstop=0.4`` will
        have 50 samples.
    ...
        :class:`mne.Epochs` parameters.
    """
    if baseline is False:
        baseline = None
    if raw is None:
        raw = ds.info['raw']
    if tmax is None:
        if tstop is None:
            tmax = 0.6
        else:
            sfreq = raw.info['sfreq'] / decim
            start_index = int(round(tmin * sfreq))
            stop_index = int(round(tstop * sfreq))
            tmax = tmin + (stop_index - start_index - 1) / sfreq
    elif tstop is not None:
        raise TypeError(f"tmax and tstop can not both be specified at the same time, got tmax={tmax}, tstop={tstop}")

    if drop_bad_chs and picks is None and raw.info['bads']:
        picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, ref_meg=False)

    events = _mne_events(ds=ds, i_start=i_start)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The events passed to the Epochs constructor', RuntimeWarning)
        epochs = mne.Epochs(raw, events, None, tmin, tmax, baseline, picks, preload=True, reject=reject, decim=decim, **kwargs)
    if reject is None and len(epochs) != len(events):
        getLogger('eelbrain').warning("%s: MNE generated only %i Epochs for %i events. The raw file might end before the end of the last epoch.", raw.filenames[0], len(epochs), len(events))

    #  add bad channels from ds
    if BAD_CHANNELS in ds.info:
        invalid = []
        for ch_name in ds.info[BAD_CHANNELS]:
            if ch_name in raw.info['bads']:
                pass
            elif ch_name not in epochs.ch_names:
                invalid.append(ch_name)
            elif ch_name not in epochs.info['bads']:
                epochs.info['bads'].append(ch_name)
        if invalid:
            raise ValueError(f"{n_of(len(invalid), 'invalid channel')} in ds.info[{BAD_CHANNELS!r}]: {', '.join(invalid)}")

    return epochs


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def sensor_dim(
        info: mne.Info,
        picks: np.ndarray = None,
        sysname: str = None,
        adjacency: AdjacencyArg = None,
) -> Sensor:
    """Create a :class:`Sensor` dimension from an :class:`mne.Info` object.

    Parameters
    ----------
    info
        Measurement info dictionary (or mne-python object that has a ``.info``
        attribute that contains measurement info).
    picks
        Channel picks (integer array, as used in mne-python).
        By default, all MEG and EEG channels are included.
    sysname
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency : str | list of (str, str) | array of int, (n_edges, 2)
        Sensor adjacency (adjacency graph). Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.

    Returns
    -------
    sensor_dim : Sensor
        Sensor dimension object.
    """
    if not isinstance(info, mne.Info):
        info_ = getattr(info, 'info', info)
        if not isinstance(info_, mne.Info):
            raise TypeError(f"{info=}: no mne.Info object")
        info = info_

    if picks is None:
        picks = mne.pick_types(info, meg=True, eeg=True, ref_meg=False, exclude=())
    else:
        picks = np.asarray(picks, int)

    chs = [info['chs'][i] for i in picks]
    ch_locs = [ch['loc'][:3] for ch in chs]
    ch_names = [ch['ch_name'] for ch in chs]

    # use KIT system ID if available
    sysname = KIT_NEIGHBORS.get(info.get('kit_system_id'), sysname)
    if sysname and sysname.startswith('neuromag'):
        ch_unit = {ch['unit'] for ch in chs}
        if len(ch_unit) > 1:
            raise RuntimeError(f"More than one channel kind for sysname={sysname!r}: {tuple(ch_unit)}")
        ch_unit = ch_unit.pop()
        if ch_unit == FIFF.FIFF_UNIT_T_M:
            sysname = 'neuromag306planar'
        elif ch_unit == FIFF.FIFF_UNIT_T:
            sysname = 'neuromag306mag'
        elif ch_unit == FIFF.FIFF_UNIT_V:
            sysname = 'neuromag306eeg'
            if adjacency is None:
                adjacency = 'auto'
        else:
            raise ValueError(f"Unknown channel unit for {sysname=}: {ch_unit!r}")

    if adjacency is None:
        adjacency = sysname or 'auto'

    ch_type = None
    if adjacency == 'auto':
        if sysname and sysname.startswith('KIT-'):
            adjacency = sysname
        else:
            ch_types = info.get_channel_types(picks, unique=True)
            if len(ch_types) == 1:
                ch_type = ch_types[0]
                if ch_type not in ['mag', 'grad', 'eeg']:
                    adjacency = 'none'
            else:
                adjacency = 'none'

    if adjacency in ('grid', 'none'):
        pass
    elif isinstance(adjacency, str):
        if adjacency == 'auto':
            adjacency = _adjacency_id(info, ch_type)
        if adjacency is None:
            c_matrix, adj_ch_names = mne.channels.find_ch_adjacency(info, ch_type)
            if any(ch not in adj_ch_names for ch in ch_names):
                raise NotImplementedError("Adjacency fot this data type")
        else:
            c_matrix, adj_ch_names = mne.channels.read_ch_adjacency(adjacency)
            # fix channel names
            if adjacency.startswith('neuromag'):
                if (' ' not in adj_ch_names[0]) and (' ' in ch_names[0]):
                    adj_ch_names = [f'{n[:3]} {n[3:]}' for n in adj_ch_names]
            elif adjacency == 'ctf275':
                suffix = ch_names[0][-5:]
                adj_ch_names = [f'{name}{suffix}' for name in adj_ch_names]
            elif adjacency.startswith('bti'):
                adj_ch_names = [f'MEG {name[1:]:0>3}' for name in adj_ch_names]

        # fix channel order
        if adj_ch_names != ch_names:
            if set(ch_names).issubset(adj_ch_names):
                index = np.array([adj_ch_names.index(name) for name in ch_names])
                c_matrix = c_matrix[index][:, index]
            else:
                missing = [name for name in ch_names if name not in adj_ch_names]
                unused = [name for name in adj_ch_names if name not in ch_names]
                raise IndexError(
                    f"{adjacency=} is missing channels {', '.join(missing)}\n"
                    f"Unused channels in adjacency: {', '.join(unused)}\n"
                    f"If the builtin sensor adjacency is inappropriate for this dataset, consider specifying the sensor adjacency manually."
                )

        adjacency = _matrix_graph(c_matrix)
    elif adjacency in (None, False):
        adjacency = 'none'

    return Sensor(ch_locs, ch_names, sysname, adjacency=adjacency)


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def variable_length_epochs(
        events: Dataset,
        tmin: Union[float, Sequence[float]],
        tmax: Union[float, Sequence[float]] = None,
        baseline: BaselineArg = None,
        allow_truncation: bool = False,
        data: DataArg = None,
        exclude: Union[str, Sequence[str]] = 'bads',
        sysname: str = None,
        adjacency: AdjacencyArg = None,
        tstop: Union[float, Sequence[float]] = None,
        name: str = None,
        **kwargs,
) -> List[NDVar]:
    """Load data epochs where each epoch has a different length

    Parameters
    ----------
    events
        Dataset containing events and an :class:`mne.io.Raw` data object,
        as returned by :func:`eelbrain.load.mne.events`.
    tmin
        First sample to include in each epoch in seconds, relative to event time.
        Can be :class:`str` referencing a variable in ``events``.
    tmax
        Last sample to include in each epoch in seconds, relative to event time.
        Can be :class:`str` referencing a variable in ``events``.
    baseline
        Time interval for baseline correction. ``(tmin, tmax)`` tuple in
        seconds, or ``None`` to use all the data (e.g., ``(None, 0)`` uses all
        the data from the beginning of the epoch up to ``t = 0``). Set to
        ``None`` for no baseline correction (default).
    allow_truncation
        If a ``tmax`` value falls outside the data available in ``raw``,
        automatically truncate the epoch (by default this raises a
        ``ValueError``).
    data
        Which data channels data to include (default based on channels in data).
    exclude
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    sysname
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag306',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency : str | list of (str, str) | array of int, (n_edges, 2)
        Adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.
    tstop
        Alternative to ``tmax``. While ``tmax`` specifies the last samples to
        include, ``tstop`` specifies the sample before which to stop (standard
        Python indexing convention).
        For example, at 100 Hz the epoch with ``tmin=-0.1, tmax=0.4`` will have
        51 samples, while the epoch specified with ``tmin=-0.1, tstop=0.4`` will
        have 50 samples.
    name
        Name for the NDVar.
    ...
        :class:`mne.Epochs` parameters.

    Returns
    -------
    epochs
        List of data epochs of shape.
    """
    epochs_ = variable_length_mne_epochs(events, tmin, tmax, baseline, allow_truncation, tstop=tstop, **kwargs)
    return [epochs_ndvar(epoch, name, data, exclude, sysname=sysname, adjacency=adjacency)[0] for epoch in epochs_]


def variable_length_mne_epochs(
        events: Dataset,
        tmin: Union[float, Sequence[float], str],
        tmax: Union[float, Sequence[float], str] = None,
        baseline: BaselineArg = None,
        allow_truncation: bool = False,
        tstop: Union[float, Sequence[float]] = None,
        picks: PicksArg = None,
        decim: int = 1,
        **kwargs,
) -> List[mne.Epochs]:
    """Load mne Epochs where each epoch has a different length

    Parameters
    ----------
    events
        Dataset containing events and an :class:`mne.io.Raw` data object,
        as returned by :func:`eelbrain.load.mne.events`.
    tmin
        First sample to include in each epoch in seconds, relative to event time.
        Can be :class:`str` referencing a variable in ``events``.
    tmax
        Last sample to include in each epoch in seconds, relative to event time.
        Can be :class:`str` referencing a variable in ``events``.
    baseline
        Time interval for baseline correction. ``(tmin, tmax)`` tuple in
        seconds, or ``None`` to use all the data (e.g., ``(None, 0)`` uses all
        the data from the beginning of the epoch up to ``t = 0``). Set to
        ``None`` for no baseline correction (default).
    allow_truncation
        If a ``tmin`` or ``tmax`` value falls outside the data available in
        ``raw``, automatically truncate the epoch (by default this raises a
        ``ValueError``).
    tstop
        Alternative to ``tmax``. While ``tmax`` specifies the last samples to
        include, ``tstop`` specifies the sample before which to stop (standard
        Python indexing convention).
        For example, at 100 Hz the epoch with ``tmin=-0.1, tmax=0.4`` will have
        51 samples, while the epoch specified with ``tmin=-0.1, tstop=0.4`` will
        have 50 samples.
    ...
        :class:`mne.Epochs` parameters.
    """
    if baseline is False:
        baseline = None
    raw = events.info['raw']
    if tmax is None:
        if tstop is None:
            raise TypeError(f"{tmax=}, {tstop=}: must specify at least one")
        if isinstance(tstop, str):
            tstop = events.eval(tstop)
        n = len(tstop)
    else:
        if isinstance(tmax, str):
            tmax = events.eval(tmax)
        n = len(tmax)
    if np.isscalar(tmin):
        tmin = np.repeat(tmin, n)
    else:
        tmin = np.asarray(tmin)
    if tmax is None:
        sfreq = raw.info['sfreq'] / decim
        start_index = np.round(tmin * sfreq).astype(int)
        stop_index = np.round(np.asarray(tstop) * sfreq).astype(int)
        tmax = tmin + (stop_index - start_index - 1) / sfreq
    elif np.isscalar(tmax):
        tmax = np.repeat(tmax, n)
    if picks is None and raw.info['bads']:
        picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, ref_meg=False, exclude=[])
    events_array = _mne_events(events)
    # Load epochs
    out = []
    for i, (tmin_i, tmax_i) in enumerate(zip(tmin, tmax)):
        i_min = events_array[i, 0] + floor(tmin_i * raw.info['sfreq'])
        if raw.first_samp > i_min:
            if allow_truncation:
                tmin_i = (raw.first_samp - events_array[i, 0]) / raw.info['sfreq']
            else:
                missing = (i_min - raw.first_samp) / raw.info['sfreq']
                raise ValueError(f"{tmin[i]=} is outside of data range by {missing:g} s")
        i_max = events_array[i, 0] + floor(tmax_i * raw.info['sfreq'])
        if raw.last_samp < i_max:
            if allow_truncation:
                tmax_i = (raw.last_samp - events_array[i, 0]) / raw.info['sfreq']
            else:
                missing = (i_max - raw.last_samp) / raw.info['sfreq']
                raise ValueError(f"{tmax[i]=} is outside of data range by {missing:g} s")
        epochs_i = mne.Epochs(raw, events_array[i:i + 1], None, tmin_i, tmax_i, baseline, picks, preload=True, decim=decim, **kwargs)
        out.append(epochs_i)
    return out


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def raw_ndvar(
        raw: Union[mne.io.BaseRaw, PathArg],
        i_start: Union[int, Sequence[int]] = None,
        i_stop: Union[int, Sequence[int]] = None,
        decim: int = 1,
        reset_tmin: bool = False,
        data: str = None,
        exclude: Union[str, Sequence[str]] = 'bads',
        sysname: str = None,
        adjacency: AdjacencyArg = None,
) -> Union[NDVar, List[NDVar]]:
    """Raw data as NDVar

    Parameters
    ----------
    raw
        Raw instance, or path of a raw FIFF file..
    i_start
        Start sample (see notes; default is the beginning of the ``raw``).
    i_stop
        Stop sample (see notes; default is end of the ``raw``).
    decim
        Downsample the data by this factor when importing. ``1`` (default)
        means no downsampling. Note that this function does not low-pass filter
        the data. The data is downsampled by picking out every n-th sample.
    reset_tmin
        Set the time axis of each :class:`NDVar` in the output to begin at 0.
        By default, the NDVars wll retain time information from the raw data.
    data
        The kind of data to include (default based on data).
    exclude
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    sysname
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag306',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency
        Adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.

    Returns
    -------
    data : NDVar | list of NDVar
        Data (sensor or source space). If ``i_start`` and ``i_stop`` are scalar
        then a single NDVar is returned, if they are lists then a list of NDVars
        is returned.

    Notes
    -----
    ``i_start`` and ``i_stop`` are interpreted as event indexes (from
    :func:`mne.find_events`), i.e. relative to ``raw.first_samp``.
    """
    if not isinstance(raw, MNE_RAW):
        raw = mne_raw(raw)
    if raw.filenames and raw.filenames[0]:
        name = os.path.basename(raw.filenames[0])
    else:
        name = None
    start_scalar = i_start is None or isinstance(i_start, int)
    stop_scalar = i_stop is None or isinstance(i_stop, int)
    if start_scalar != stop_scalar:
        raise TypeError(f"i_start and i_stop must either both be scalar or both iterable, got: \n{i_start=}\n{i_stop=}")
    elif start_scalar:
        i_start = (i_start,)
        i_stop = (i_stop,)
        scalar = True
    else:
        scalar = False

    # event index to raw index
    i_start = [i if i is None else i - raw.first_samp for i in i_start]
    i_stop = [i if i is None else i - raw.first_samp for i in i_stop]

    # target dimension
    if data is None:
        data = _guess_ndvar_data_type(raw.info)
    picks = _picks(raw.info, data, exclude)
    dim = sensor_dim(raw, picks, sysname, adjacency)
    info = _sensor_info(data, None, raw.info)

    out = []
    for start, stop in zip(i_start, i_stop):
        x = raw[picks, start:stop][0]

        if decim != 1:
            x = x[:, ::decim]
        if reset_tmin:
            tmin = 0
        elif start is None:
            tmin = raw.first_samp / raw.info['sfreq']
        else:
            tmin = (raw.first_samp + start) / raw.info['sfreq']
        time = UTS(tmin, float(decim) / raw.info['sfreq'], x.shape[1])
        out.append(NDVar(x, (dim, time), name, info))

    if scalar:
        return out[0]
    else:
        return out


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def epochs_ndvar(
        epochs: Union[mne.BaseEpochs, PathArg],
        name: str = None,
        data: DataArg = None,
        exclude: Union[str, Sequence[str]] = 'bads',
        mult: float = 1,
        info: dict = None,
        sensors: Sensor = None,
        vmax: float = None,
        sysname: str = None,
        adjacency: AdjacencyArg = None,
        proj: bool = True,
):
    """
    Convert an :class:`mne.Epochs` object to an :class:`NDVar`.

    Parameters
    ----------
    epochs
        The epochs object or path to an epochs FIFF file.
    name
        Name for the NDVar.
    data
        Which data channels data to include (default based on channels in data).
    exclude
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    mult
        multiply all data by a constant.
    info : None | dict
        Additional contents for the info dictionary of the NDVar.
    sensors
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor can be supplied through this kwarg.
    vmax
        Set a default range for plotting.
    sysname
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag306',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency : str | list of (str, str) | array of int, (n_edges, 2)
        Adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.
    proj
        Add projectors (only applies when ``epochs`` is a path).
    """
    if isinstance(epochs, (str, Path)):
        epochs = mne.read_epochs(epochs, proj=proj)

    if data is None:
        data = _guess_ndvar_data_type(epochs.info)
    picks = _picks(epochs.info, data, exclude)
    info_ = _sensor_info(data, vmax, epochs.info, info, mult)

    x = epochs.get_data(copy=False)
    if len(picks) < x.shape[1]:
        x = x[:, picks]

    if mult != 1:
        x *= mult

    sensor = sensors or sensor_dim(epochs, picks, sysname, adjacency)
    time = UTS(epochs.times[0], 1. / epochs.info['sfreq'], len(epochs.times))
    return NDVar(x, ('case', sensor, time), info=info_, name=name)


def evoked_ndvar(evoked, name=None, data=None, exclude='bads', vmax=None,
                 sysname=None, adjacency=None):
    """
    Convert one or more mne :class:`Evoked` objects to an :class:`NDVar`.

    Parameters
    ----------
    evoked : str | Evoked | list of Evoked
        The Evoked to convert to NDVar. Can be a string designating a file
        path to a evoked fiff file containing only one evoked.
    name : str
        Name of the NDVar.
    data : 'eeg' | 'mag' | 'grad'
        Which data channels data to include (default based on channels in data).
    exclude : list of string | string
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    vmax : None | scalar
        Set a default range for plotting.
    sysname : str
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag306',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency : str | list of (str, str) | array of int, (n_edges, 2)
        adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.

    Notes
    -----
    If evoked objects have different channels, the intersection is used (i.e.,
    only the channels present in all objects are retained).
    """
    if isinstance(evoked, str):
        evoked = mne.read_evokeds(evoked)

    if isinstance(evoked, MNE_EVOKED):
        case_out = False
        evoked = (evoked,)
    elif isinstance(evoked, (tuple, list)):
        case_out = True
    else:
        raise TypeError("evoked=%s" % repr(evoked))

    # data type to load
    if data is None:
        data_set = {_guess_ndvar_data_type(e.info) for e in evoked}
        if len(data_set) > 1:
            raise ValueError(f"Different Evoked objects contain different data types: {', '.join(data_set)}")
        data = data_set.pop()

    # MEG system
    kit_sys_ids = {e.info.get('kit_system_id') for e in evoked}
    kit_sys_ids -= {None}
    if len(kit_sys_ids) > 1:
        raise ValueError("Evoked objects from different KIT systems can not be "
                         "combined because they have different sensor layouts")
    elif kit_sys_ids:
        sysname = KIT_NEIGHBORS.get(kit_sys_ids.pop(), sysname)

    if data == 'mag':
        info = _info.for_meg(vmax)
    elif data == 'eeg':
        info = _info.for_eeg(vmax)
    elif data in ('grad', 'planar1', 'planar2'):
        info = _info.for_meg(vmax, unit='T/cm')
    else:
        raise ValueError(f"{data=}")

    e0 = evoked[0]
    if len(evoked) == 1:
        picks = _picks(e0.info, data, exclude)
        x = e0.data[picks]
        if case_out:
            x = x[None, :]
        first, last, sfreq = e0.first, e0.last, e0.info['sfreq']
    else:
        # timing:  round sfreq because precision is lost by FIFF format
        timing_set = {(e.first, e.last, e.info['sfreq']) for e in evoked}
        if len(timing_set) == 1:
            first, last, sfreq = timing_set.pop()
        else:
            raise ValueError(f"Evoked objects have different timing information (first, last, sfreq): {', '.join(map(str, timing_set))}")

        # find excluded channels
        ch_sets = [set(e.info['ch_names']) for e in evoked]
        all_chs = set.union(*ch_sets)
        common = set.intersection(*ch_sets)
        exclude = set.union(*list(map(set, (e.info['bads'] for e in evoked))))
        exclude.update(all_chs.difference(common))
        exclude = list(exclude)

        # get data
        x = []
        for e in evoked:
            picks = _picks(e.info, data, exclude)
            x.append(e.data[picks])

    sensor = sensor_dim(e0, picks, sysname, adjacency)
    time = UTS.from_int(first, last, sfreq)
    if case_out:
        dims = ('case', sensor, time)
    else:
        dims = (sensor, time)
    return NDVar(x, dims, name, info)


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def forward_operator(
        fwd: Union[str, mne.Forward],
        src: str,
        subjects_dir: PathArg = None,
        parc: str = 'aparc',
        sysname: str = None,
        adjacency: AdjacencyArg = None,
        name: str = None,
) -> NDVar:
    """Load forward operator as :class:`NDVar`

    Parameters
    ----------
    fwd
        MNE Forward solution, or path to forward solution.
    src
        Tag describing the source space. Should be a source space type indicator
        (one of ``ico|oct|vol``), followed by a number indicating spacing
        (e.g., "ico-4" for a surface source space based on 4-fold icosahedral
        subdivision; "vol-7" for volume source space with 7 mm grid).
    subjects_dir
        Location of the MRI subjects directory.
    parc
        Parcellation to load (corresponding to existing annot files; default
        'aparc').
    sysname
        Name of the sensor system to load sensor adjacency (e.g. 'neuromag',
        inferred automatically for KIT data converted with a recent version of
        MNE-Python).
    adjacency
        adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection [i, j] with i < j. If the array's dtype is uint32,
          property checks are disabled to improve efficiency.
        - ``"grid"`` to use adjacency in the sensor names

        If unspecified, it is inferred from ``sysname`` if possible.
    name
        Name the NDVar (default is the filename if a path is provided,
        otherwise "fwd").

    Returns
    -------
    fwd : NDVar
        NDVar ``(sensor, source)`` containing the gain matrix.
    """
    is_vol = src.startswith('vol')
    if isinstance(fwd, str):
        if name is None:
            name = os.path.basename(fwd)
        fwd = mne.read_forward_solution(fwd)
        mne.convert_forward_solution(fwd, force_fixed=not is_vol, use_cps=True, copy=False)
    elif name is None:
        name = 'fwd'
    sensor = sensor_dim(fwd['info'], sysname=sysname, adjacency=adjacency)
    assert np.all(sensor.names == fwd['sol']['row_names'])
    subject = fwd['src'][0]['subject_his_id']
    if is_vol:
        source = VolumeSourceSpace.from_file(subjects_dir, subject, src, parc, source_spaces=fwd['src'])
        x = fwd['sol']['data'].reshape((len(sensor), len(source), 3))
        dims = (sensor, source, Space('RAS'))
    else:
        source = SourceSpace.from_file(subjects_dir, subject, src, parc, source_spaces=fwd['src'])
        x = fwd['sol']['data']
        dims = (sensor, source)
    return NDVar(x, dims, name)


def inverse_operator(inv, src, subjects_dir=None, parc='aparc', name=None):
    """Load inverse operator as :class:`NDVar`

    Parameters
    ----------
    inv : str | mne.minimum_norm.InverseOperator
        MNE inverse operator, or path to inverse operator.
    src : str
        Tag describing the source space (e.g., "ico-4").
    subjects_dir : str
        Location of the MRI subjects directory.
    parc : str
        Parcellation to load (corresponding to existing annot files; default
        'aparc').
    name : str
        Name the NDVar (default is the filename if a path is provided,
        otherwise "inv").

    Returns
    -------
    inv : NDVar  (source, sensor)
        NDVar containing the inverse operator.
    """
    if isinstance(inv, str):
        if name is None:
            name = os.path.basename(inv)
        inv = mne.minimum_norm.read_inverse_operator(inv)
    elif name is None:
        name = 'inv'
    sensor = sensor_dim(inv['info'], _picks(inv['info'], True, 'bads'))
    assert np.all(sensor.names == inv['eigen_fields']['col_names'])
    subject = inv['src'][0]['subject_his_id']
    source = SourceSpace.from_file(subjects_dir, subject, src, parc, source_spaces=inv['src'])
    inv = mne.minimum_norm.prepare_inverse_operator(inv, 1, 1., 'MNE')
    k = mne.minimum_norm.inverse._assemble_kernel(inv, None, 'MNE', False)[0]
    return NDVar(k, (source, sensor), {}, name)


@deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
def stc_ndvar(
        stc: Union[_BaseSourceEstimate, Sequence[_BaseSourceEstimate], PathArg],
        subject: str,
        src: str,
        subjects_dir: PathArg = None,
        method: str = None,
        fixed: bool = None,
        name: str = None,
        check: bool = True,
        parc: Optional[str] = '',
        adjacency: str = None,
        sss_filename: str = '{subject}-{src}-src.fif',
):
    """
    Convert one or more :class:`mne.SourceEstimate` objects to an :class:`NDVar`.

    Parameters
    ----------
    stc : SourceEstimate | list of SourceEstimates | str
        The source estimate object(s) or a path to an stc file. Volum and vector
        source estimates are supported.
    subject
        MRI subject (used for loading MRI in PySurfer plotting)
    src
        The kind of source space used (e.g., 'ico-4').
    subjects_dir
        The path to the subjects_dir (needed to locate the source space file).
    method : 'MNE' | 'dSPM' | 'sLORETA' | 'eLORETA'
        Source estimation method (optional, used for generating info).
    fixed
        Source estimation orientation constraint (optional, used for generating
        info).
    name
        Ndvar name.
    check
        If multiple stcs are provided, check if all stcs have the same times
        and vertices.
    parc
        Name of a parcellation to add to the source space. ``None`` to add no
        parcellation. The default is ``aparc`` for surface source-spaces and
        none for volume source spaces.
    adjacency : 'link-midline'
        Modify source space adjacency to link medial sources of the two
        hemispheres across the midline.
    sss_filename
        Filename template for the MNE source space file.
    """
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir)
    if isinstance(stc, Path):
        stc = str(stc)
    if isinstance(stc, str):
        stc = mne.read_source_estimate(stc)

    # construct data array
    if isinstance(stc, _BaseSourceEstimate):
        case = False
        stcs = None
    else:
        case = True
        stcs = stc
        stc = stcs[0]

    vertices = as_list(stc.vertices)  # not always list for mne < 0.21
    if stcs and check:
        times = stc.times
        for stc_ in stcs[1:]:
            assert np.array_equal(stc_.times, times)
            stc_vertices = as_list(stc_.vertices)
            for v1, v0 in zip_longest(stc_vertices, vertices):
                assert np.array_equal(v1, v0)

    if stcs:
        x = np.array([s.data for s in stcs])
    else:
        x = stc.data

    # Construct NDVar Dimensions
    time = UTS(stc.tmin, stc.tstep, stc.times.size)
    if isinstance(stc, MNE_VOLUME_STC):
        if parc == '':
            parc = None
        ss = VolumeSourceSpace(vertices, subject, src, subjects_dir, parc, filename=sss_filename)
        is_vector = stc.data.ndim == 3
    elif isinstance(stc, (mne.SourceEstimate, mne.VectorSourceEstimate)):
        if parc == '':
            parc = 'aparc'
        ss = SourceSpace(vertices, subject, src, subjects_dir, parc, filename=sss_filename)
        is_vector = isinstance(stc, mne.VectorSourceEstimate)
    else:
        raise TypeError(f"{stc=}")
    # Apply adjacency modification
    if isinstance(adjacency, str):
        if adjacency == 'link-midline':
            ss._link_midline()
        elif adjacency != '':
            raise ValueError(f"{adjacency=}")
    elif adjacency is not None:
        raise TypeError(f"{adjacency=}")
    # assemble dims
    dims = [ss, time]
    if is_vector:
        dims.insert(1, Space('RAS'))
    if case:
        dims.insert(0, Case)

    # find the right measurement info
    info = {}
    if fixed is False:
        info['meas'] = 'Activation'
        if method:
            info['unit'] = method
    elif fixed is True:
        info['meas'] = 'Current Estimate'
        if method == 'MNE':
            info['unit'] = 'A'
        elif method:
            info['unit'] = method
    elif fixed is not None:
        raise ValueError(f"{fixed=}")

    return NDVar(x, dims, name, info)


def ndvar_stc(
        ndvar: NDVar,
) -> (
    mne.source_estimate._BaseSourceEstimate,
    Tuple[int, ...],  # target shape
    List[Dimension],  # Dimensions
):
    """Convert an NDVar with source space data to an :mod:`mne` object

    Parameters
    ----------
    ndvar
        Data in source space.
    """
    source_dim = ndvar.get_dim('source')
    is_vector_stc = ndvar.has_dim('space')
    if is_vector_stc:
        dim_names = ndvar.get_dimnames(first=('source', 'space'))
    else:
        dim_names = ndvar.get_dimnames(first='source')
    if 'case' in dim_names:
        case_axis = dim_names.index('case')
    else:
        case_axis = None
    dims = ndvar.get_dims(dim_names)
    source_shape = [len(dim) for dim in dims]
    compressed_shape = (
        *source_shape[:1 + is_vector_stc],
        reduce(operator.mul, source_shape[1 + is_vector_stc:], 1),
    )
    data = ndvar.get_data(dim_names).reshape(compressed_shape)
    # Whether to use STC time axis for time
    use_time_axis = len(dim_names) == 2 + is_vector_stc and dim_names[-1] == 'time'
    if use_time_axis:
        time_dim = ndvar.get_dim('time')
        tmin = time_dim.tmin
        tstep = time_dim.tstep
    else:
        tmin = 0
        tstep = 1
    target_shape = (-1, *source_shape[1:])
    # Initialize appropriate MNE STC object
    if isinstance(source_dim, SourceSpace):
        if is_vector_stc:
            stc = mne.VectorSourceEstimate(data, source_dim.vertices, tmin, tstep, source_dim.subject)
        else:
            stc = mne.SourceEstimate(data, source_dim.vertices, tmin, tstep, source_dim.subject)
    elif isinstance(source_dim, VolumeSourceSpace):
        assert len(source_dim.vertices) == 1
        vertices = source_dim.vertices[0]
        if is_vector_stc:
            stc = mne.VolVectorSourceEstimate(data, vertices, tmin, tstep, source_dim.subject)
        else:
            stc = mne.VolSourceEstimate(data, vertices, tmin, tstep, source_dim.subject)
    return stc, target_shape, dims[1:], case_axis


def _trim_ds(ds, epochs):
    """Trim a Dataset to account for rejected epochs.

    If no epochs were rejected, the original ds is rturned.

    Parameters
    ----------
    ds : Dataset
        Dataset that was used to construct epochs.
    epochs : Epochs
        Epochs loaded with mne_epochs()
    """
    if len(epochs) < ds.n_cases:
        ds = ds.sub(epochs.selection)
        ds.info['epochs.selection'] = epochs.selection

    return ds
