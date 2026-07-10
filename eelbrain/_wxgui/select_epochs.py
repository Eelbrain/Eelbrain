"""GUI for rejecting epochs

File format
-----------
Required variables:

 - ``trigger`` :class:`Var` (int)
 - ``accept`` :class:`Var` (boolean)
 - ``rej_tag`` :class:`Factor`

 Optional:

 - ``interpolate_channels`` :class:`Datalist` (lists)

"""

# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizaes Document
#  - listens to Document changes
#  - issues commands to Model
from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
import math
import os
import re
import time

from matplotlib.transforms import Bbox
import mne
import numpy as np
from scipy.spatial.distance import cdist
import wx

from .. import _meeg as meeg
from .. import _text
from .. import load, save, plot, fmtxt
from .._data_obj import Dataset, Factor, NDVar, Var, Datalist, combine
from .._info import BAD_CHANNELS, INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS
from .._ndvar import neighbor_correlation
from .._utils.parse import FLOAT_PATTERN, POS_FLOAT_PATTERN, INT_PATTERN
from .._utils.numpy_utils import FULL_SLICE, INT_TYPES
from ..mne_fixes import MNE_EPOCHS
from ..plot._base import AxisData, DataLayer, PlotType, AxisScale, find_fig_vlims, find_fig_cmaps
from ..plot._nuts import PltBinNuts
from ..plot._topo import AxTopomap
from ..plot._utsnd import AxButterflyEpoch
from ._ch_types import CH_TYPE_PICK_KWARGS, CH_TYPE_COLORS, CH_TYPE_DEFAULT_VLIM_SI, ch_type_scale
from .app import get_app
from .frame import EelbrainDialog
from .mpl_canvas import FigureCanvasPanel
from .frame import NavigableFrame
from .history import Action, FileDocument, FileModel, FileFrame
from .text import HTMLFrame
from .utils import Icon, REValidator
from . import ID


# IDs
TOPO_PLOT = -2
OUT_OF_RANGE = -3

# Minimum number of sensors required to compute a topomap projection; with fewer the after-rejection topomap is blanked
MIN_TOPO_SENSORS = 4

# For unit-tests
TEST_MODE = False

# Default peak-to-peak / absolute rejection thresholds, in SI units.
# Peak amplitudes from mne_epochs.get_data() are in SI: T for mag, T/m for grad, V for eeg.
_THRESHOLD_DEFAULT_SI = {
    'mag':  2000e-15,   # 2000 fT
    'grad': 2000e-15,   # 20 fT/cm
    'eeg':  150e-6,     # 150 µV
}


def _epoch_list_to_ranges(elist):
    out = []
    i = 0
    i_last = len(elist) - 1
    start = None
    while i <= i_last:
        cur = elist[i]
        if i < i_last and elist[i + 1] - cur == 1:
            if start is None:
                start = cur
        elif start is None:
            out.append(fmtxt.Link(cur, 'epoch:%i' % cur))
        else:
            out.append(fmtxt.Link(start, 'epoch:%i' % start) + '-' + fmtxt.Link(cur, 'epoch:%i' % cur))
            start = None
        i += 1
    return out


def format_epoch_list(l, head="Epochs by channel:"):
    d = meeg.channel_listlist_to_dict(l)
    if not d:
        return "None."
    out = fmtxt.List(head)
    for ch in sorted(d, key=lambda x: -len(d[x])):
        item = fmtxt.FMText(f"{ch} ({len(d[ch])}): ")
        item += fmtxt.delim_list(_epoch_list_to_ranges(d[ch]))
        out.add_item(item)
    return out


def _ask(caption: str, message: str, style: int, answer: bool = None):
    "Intercept GUI prompts for tests"
    if answer is True:
        return wx.ID_OK
    elif answer is False:
        return wx.ID_NO
    app = get_app()
    return app.message_box(message, caption, style)


class ChangeAction(Action):
    """Action objects are kept in the history and can do and undo themselves

    Parameters
    ----------
    desc : str
        Description of the action
        list of (i, name, old, new) tuples
    """

    def __init__(
            self,
            desc: str,
            index: int | slice | np.ndarray | None = None,
            old_accept: bool | Var | np.ndarray | None = None,
            new_accept: bool | Var | np.ndarray | None = None,
            old_tag: str | Factor | None = None,
            new_tag: str | Factor | None = None,
            old_path: str | None = None,
            new_path: str | None = None,
            old_bad_chs: list | None = None,
            new_bad_chs: list | None = None,
            old_interpolate: list[str] | Datalist | None = None,
            new_interpolate: list[str] | Datalist | None = None,
    ) -> None:
        self.desc = desc
        self.index = index
        self.old_path = old_path
        self.old_accept = old_accept
        self.old_tag = old_tag
        self.new_path = new_path
        self.new_accept = new_accept
        self.new_tag = new_tag
        self.old_bad_chs = old_bad_chs
        self.new_bad_chs = new_bad_chs
        self.old_interpolate = old_interpolate
        self.new_interpolate = new_interpolate

    def do(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.new_accept, self.new_tag,
                         self.new_interpolate)
        if self.new_path is not None:
            doc.set_path(self.new_path)
        if self.new_bad_chs is not None:
            doc.set_bad_channels(self.new_bad_chs)

    def undo(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.old_accept, self.old_tag,
                         self.old_interpolate)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)
        if self.new_bad_chs is not None:
            doc.set_bad_channels(self.old_bad_chs)


class Document(FileDocument):
    """Represent the current state of the Document

    Data can be accesses through attributes, but should only be changed through
    the set_...() methods.

    Parameters
    ----------
    path : None | str
        Default location of the epoch selection file (used for save
        command). If the file exists, it is loaded as initial state.

    Attributes
    ----------
    n_epochs : int
        The number of epochs.
    epochs : NDVar
        The raw epochs.
    accept : Var of bool
        Case status.
    tag : Factor
        Case tag.
    trigger : Var of int
        Case trigger value.
    blink : Datalist | None
        Case eye tracker artifact data.
    """

    def __init__(
            self,
            ds: Dataset | mne.BaseEpochs,
            data: str = 'meg',
            accept: str = 'accept',
            blink: str = 'blink',
            tag: str = 'rej_tag',
            trigger: str = 'trigger',
            path: str | None = None,
            bad_chs: list | None = None,
            allow_interpolation: bool = True,
    ) -> None:
        FileDocument.__init__(self, path)
        # ``long_epochs`` mode: variable-length epochs supplied as a Datalist of
        # mne.Epochs (one or more per case) rather than a single equal-length
        # mne.Epochs. It is a read-only, continuous single-column display.
        self.long_epochs = False
        if isinstance(ds, MNE_EPOCHS):
            mne_epochs = ds
            ds = Dataset()
            mne_epochs.load_data()
            ds[data] = mne_epochs
            ds[trigger] = Var(mne_epochs.events[:, 2])
            mne_epochs_list = [mne_epochs]
        elif not isinstance(data, str):
            raise TypeError(f"{data=}; must be a string key into ds")
        else:
            value = ds.get(data)
            if isinstance(value, MNE_EPOCHS):
                mne_epochs_list = [value]
            elif isinstance(value, Datalist) and len(value) and all(isinstance(e, MNE_EPOCHS) for e in value):
                self.long_epochs = True
                mne_epochs_list = list(value)
            else:
                raise TypeError(f"{ds[data]=}; must be an mne.BaseEpochs instance or a Datalist of mne.BaseEpochs")
            for e in mne_epochs_list:
                e.load_data()

        # channel types present (from the first epochs object)
        ch_types = []
        for ch_type, pick_kwargs in CH_TYPE_PICK_KWARGS.items():
            if len(mne.pick_types(mne_epochs_list[0].info, ref_meg=False, exclude='bads', **pick_kwargs)):
                ch_types.append(ch_type)
        if not ch_types:
            raise RuntimeError("No data channels found in MNE Epochs")

        if self.long_epochs:
            # one (sensor x time) NDVar per channel type, per epoch
            self.epoch_data = []
            for e in mne_epochs_list:
                per_type = [(ct, load.mne.epochs_ndvar(e, data=ct)) for ct in ch_types]
                for ci in range(len(e)):
                    self.epoch_data.append([(ct, nd[ci]) for ct, nd in per_type])
            self.epochs_by_type = []
            data = self.epochs = self.epoch_data[0][0][1]  # representative for sensor/time access
            n = self.n_epochs = len(self.epoch_data)
            type_ndvars = self.epoch_data[0]
            # per-type display vmax over all epochs
            self.type_vmax = {}
            for j, ct in enumerate(ch_types):
                vals = np.concatenate([self.epoch_data[i][j][1].x.ravel() for i in range(n)])
                vmax = float(np.percentile(np.abs(vals), 99.5))
                self.type_vmax[ct] = vmax if vmax > 0 else 1.0
        else:
            self.epoch_data = None
            epochs_by_type = [(ct, load.mne.epochs_ndvar(mne_epochs_list[0], data=ct)) for ct in ch_types]
            self.epochs_by_type = epochs_by_type
            data = epochs_by_type[0][1]  # primary NDVar (backward compat: bad channels, time axis)
            self.n_epochs = n = len(data)
            type_ndvars = epochs_by_type
            self.type_vmax = {}
            for ct, nd in epochs_by_type:
                vmax = float(np.percentile(np.abs(nd.x), 99.5))
                self.type_vmax[ct] = vmax if vmax > 0 else 1.0

        if not isinstance(accept, str):
            raise TypeError("accept needs to be a string")
        if accept not in ds:
            x = np.ones(n, dtype=bool)
            ds[accept] = Var(x)
        accept = ds[accept]

        if not isinstance(tag, str):
            raise TypeError("tag needs to be a string")
        if tag in ds:
            tag = ds[tag]
        else:
            tag = Factor([''], repeat=n, name=tag)
            ds.add(tag)

        if not isinstance(trigger, str):
            raise TypeError("trigger needs to be a string")
        self._trigger_key = trigger
        if trigger in ds:
            trigger = ds[trigger]
        else:
            raise KeyError(f"ds does not contain a variable named {trigger!r}. The trigger parameters needs to point to a variable in ds containing trigger values.")

        if INTERPOLATE_CHANNELS in ds:
            interpolate = ds[INTERPOLATE_CHANNELS]
            if not allow_interpolation and any(interpolate):
                raise ValueError("Dataset contains channel interpolation information but interpolation is turned off")
        else:
            interpolate = Datalist([[]] * ds.n_cases, INTERPOLATE_CHANNELS, 'strlist')

        # time-windowed bad-channel interpolation (long-epoch mode)
        if INTERPOLATE_WINDOWS in ds:
            self.interpolate_windows = list(ds[INTERPOLATE_WINDOWS])
        else:
            self.interpolate_windows = [[] for _ in range(n)]

        if isinstance(blink, str):
            if ds is not None:
                blink = ds.get(blink, None)
        elif blink is True:
            if 'edf' in ds.info:
                tmin = data.time.tmin
                tmax = data.time.tmax
                _, blink = load.eyelink.artifact_epochs(ds, tmin, tmax, esacc=False)
            else:
                wx.MessageBox("No eye tracker data was found in ds.info['edf']. Use load.eyelink.add_edf(ds) to add an eye tracker file to a Dataset ds.", "Eye Tracker Data Not Found")
                blink = None
        elif blink is not None:
            raise TypeError("blink needs to be a string or None")

        if blink is not None:
            raise NotImplementedError("Frame.SetPage() needs to be updated to use blink information")

        # options
        self.allow_interpolation = allow_interpolation

        # data
        self.epochs = data
        self.accept = accept
        self.tag = tag
        self.interpolate = interpolate
        self.trigger = trigger
        self.blink = blink
        self.bad_channels = []  # list of int
        self.good_channels = None
        self.epochs_selection = ds.info.get('epochs.selection')

        # Channel-type metadata (derived from a representative per-type NDVar)
        self.ch_type_names = list(ch_types)
        self._ch_name_to_type = {
            name: ct
            for ct, ndvar in type_ndvars
            for name in ndvar.sensor.names
        }

        # cache
        self._good_sensor_indices = {}

        # publisher
        self.callbacks.register_key('case_change')
        self.callbacks.register_key('bad_chs_change')

        # finalize
        if bad_chs is not None:
            self.set_bad_channels_by_name(bad_chs)

        if path and os.path.exists(path):
            accept, tag, interpolate, bad_chs, windows = self.read_rej_file(path)
            self.accept[:] = accept
            self.tag[:] = tag
            self.interpolate[:] = interpolate
            self.interpolate_windows = windows
            self.set_bad_channels(bad_chs)
            self.saved = True

    @property
    def bad_channel_names(self):
        return [self.epochs.sensor.names[i] for i in self.bad_channels]

    def get_channel_colors(self, epoch):
        """Return per-channel color dict {sensor_index: color} for ``epoch``.

        Used to color channels by type in butterfly plots. Returns ``None``
        when fewer than two channel types are present.
        """
        if len(self.ch_type_names) < 2:
            return None
        colors = {}
        for i, name in enumerate(epoch.sensor.names):
            ch_type = self._ch_name_to_type.get(name)
            if ch_type is not None:
                colors[i] = CH_TYPE_COLORS[ch_type]
        return colors or None

    def iter_good_epochs(self):
        "All cases, only good channels"
        if self.bad_channels:
            epochs = self.epochs.sub(sensor=self.good_channels)
        else:
            epochs = self.epochs

        bad_set = set(self.bad_channel_names)
        interpolate = [set(chs) - bad_set for chs in self.interpolate]

        if any(interpolate):
            for epoch, chs in zip(epochs, interpolate):
                if chs:
                    yield epoch.sub(sensor=epoch.sensor.index(exclude=chs))
                else:
                    yield epoch
        else:
            yield from epochs

    def good_sensor_index(self, case):
        "Index of non-interpolated sensor relative to good sensors"
        if self.interpolate[case]:
            key = frozenset(self.interpolate[case])
            if key in self._good_sensor_indices:
                return self._good_sensor_indices[key]
            else:
                out = np.ones(len(self.epochs.sensor), bool)
                out[self.epochs.sensor._array_index(self.interpolate[case])] = False
                if self.good_channels is not None:
                    out = out[self.good_channels]
                self._good_sensor_indices[key] = out
                return out

    def get_epoch(self, case, name):
        if self.bad_channels:
            return self.epochs.sub(case=case, sensor=self.good_channels, name=name)
        else:
            return self.epochs.sub(case=case, name=name)

    def get_epoch_by_type(self, case, name):
        """Return ``[(ch_type, NDVar)]`` for one epoch across all channel types.

        Bad channels (by name) are excluded from each type's NDVar.
        """
        bad_names = set(self.bad_channel_names)
        result = []
        if self.long_epochs:
            items = self.epoch_data[case]  # already (sensor x time) per type
        else:
            items = [(ct, nd.sub(case=case)) for ct, nd in self.epochs_by_type]
        for ch_type, ndvar in items:
            if bad_names:
                good = [i for i, n in enumerate(ndvar.sensor.names) if n not in bad_names]
                if len(good) < len(ndvar.sensor):
                    sub = ndvar.sub(sensor=good, name=name)
                else:
                    sub = ndvar.sub(name=name)
            else:
                sub = ndvar.sub(name=name)
            result.append((ch_type, sub))
        return result

    def windows_in_range(self, case, tmin, tmax):
        """Bad-channel interpolation windows of ``case`` overlapping ``[tmin, tmax)``."""
        return [w for w in self.interpolate_windows[case] if w.tmin < tmax and w.tmax > tmin]

    def get_grand_average(self):
        "Grand average of all accepted epochs"
        return self.epochs.sub(case=self.accept.x, sensor=self.good_channels,
                               name="Grand Average").mean('case')

    def get_grand_averages_by_type(self):
        "Grand average per channel type, excluding bad channels"
        accept = self.accept.x
        bad_names = set(self.bad_channel_names)
        result = []
        for ch_type, ndvar in self.epochs_by_type:
            if bad_names:
                good = [i for i, n in enumerate(ndvar.sensor.names) if n not in bad_names]
                if len(good) < len(ndvar.sensor):
                    ndvar = ndvar.sub(sensor=good)
            result.append((ch_type, ndvar.sub(case=accept, name="Grand Average").mean('case')))
        return result

    def set_bad_channels(self, indexes):
        """Set the channels to treat as bad (i.e., exclude)

        Parameters
        ----------
        bad_chs : collection of int
            Indices of channels to treat as bad.
        """
        indexes = sorted(indexes)
        if indexes == self.bad_channels:
            return
        self.bad_channels = indexes
        if indexes:
            self.good_channels = np.setdiff1d(np.arange(len(self.epochs.sensor)),
                                              indexes, True)
        else:
            self.good_channels = None
        self._good_sensor_indices.clear()
        self.callbacks.callback('bad_chs_change')

    def set_bad_channels_by_name(self, names):
        self.set_bad_channels(self.epochs.sensor._array_index(names))

    def set_case(self, index, state, tag, interpolate):
        if state is not None:
            self.accept[index] = state
        if tag is not None:
            self.tag[index] = tag
        if interpolate is not None:
            self.interpolate[index] = interpolate

        self.callbacks.callback('case_change', index)

    def set_path(self, path):
        """Set the path

        Parameters
        ----------
        path : str
            Path under which to save. The extension determines the way file
            (*.pickle -> pickled Dataset; *.txt -> tsv)
        """
        root, ext = os.path.splitext(path)
        if ext == '':
            path = root + '.txt'
        FileDocument.set_path(self, path)

    def read_rej_file(
            self,
            path: str,
            answer: bool = None,  # User answer to dialogs (for tests)
    ):
        "Read a file making sure it is compatible"
        _, ext = os.path.splitext(path)
        if ext.startswith('.pickle'):
            ds = load.unpickle(path)
        elif ext == '.txt':
            ds = load.tsv(path, delimiter='\t')
        else:
            raise ValueError(f"Unknown file extension for rejections: {path}")

        # fix
        if 'tag' in ds and 'rej_tag' not in ds:
            ds['rej_tag'] = ds.pop('tag')

        # check file contents
        needed = [self._trigger_key, 'accept', 'rej_tag']
        if INTERPOLATE_CHANNELS in ds:
            needed.append(INTERPOLATE_CHANNELS)
        if INTERPOLATE_WINDOWS in ds:
            needed.append(INTERPOLATE_WINDOWS)
        missing = set(needed).difference(ds)
        if missing:
            raise OSError(f"{path} is not a valid epoch rejection file. It is missing the following keys: {', '.join(missing)}")

        # check file
        if ds.n_cases > self.n_epochs:
            cmd = _ask("Truncate the file?", f"The File contains more events than the data (file: {ds.n_cases}, data: {self.n_epochs}). Truncate the file?", wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT | wx.ICON_WARNING, answer)
            if cmd == wx.ID_OK:
                ds = ds[:self.n_epochs]
            else:
                raise OSError("Unequal number of cases")
        elif ds.n_cases < self.n_epochs:
            cmd = _ask("Load partial file?", f"The rejection file contains fewer epochs than the data (file: {ds.n_cases}, data: {self.n_epochs}). Load anyways (epochs missing from the file will be accepted)?", wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT | wx.ICON_WARNING, answer)
            if cmd == wx.ID_OK:
                n_missing = self.n_epochs - ds.n_cases
                tail = Dataset(info=ds.info)
                tail[self._trigger_key] = Var(self.trigger[-n_missing:])
                tail['accept'] = Var([True], repeat=n_missing)
                tail['rej_tag'] = Factor([''], repeat=n_missing)
                if INTERPOLATE_CHANNELS in ds:
                    tail[INTERPOLATE_CHANNELS] = Datalist([[]] * n_missing)
                if INTERPOLATE_WINDOWS in ds:
                    tail[INTERPOLATE_WINDOWS] = Datalist([[] for _ in range(n_missing)], INTERPOLATE_WINDOWS)
                ds = combine((ds[needed], tail))
            else:
                raise OSError("Unequal number of cases")

        if not np.all(ds[self._trigger_key] == self.trigger):
            cmd = _ask("Ignore trigger mismatch?", "The file contains different triggers from the data. Ignore mismatch and proceed?", wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT, answer)
            if cmd == wx.ID_OK:
                ds[self._trigger_key] = self.trigger
            else:
                raise OSError("Trigger mismatch")

        accept = ds['accept']
        if 'rej_tag' in ds:
            tag = ds['rej_tag']
        else:
            tag = Factor([''], repeat=self.n_epochs, name='rej_tag')

        if INTERPOLATE_CHANNELS in ds:
            interpolate = ds[INTERPOLATE_CHANNELS]
            if not self.allow_interpolation and any(interpolate):
                cmd = _ask("Clear Channel Interpolation Instructions?", "The file contains channel interpolation instructions, but interpolation is disabled is the current session. Drop interpolation instructions?", wx.OK | wx.CANCEL | wx.CANCEL_DEFAULT, answer)
                if cmd == wx.ID_OK:
                    for l in interpolate:
                        del l[:]
                else:
                    raise RuntimeError("File with interpolation when interpolation is disabled")
        else:
            interpolate = Datalist([[]] * self.n_epochs, INTERPOLATE_CHANNELS, 'strlist')

        if INTERPOLATE_WINDOWS in ds:
            windows = list(ds[INTERPOLATE_WINDOWS])
        else:
            windows = [[] for _ in range(self.n_epochs)]

        if BAD_CHANNELS in ds.info:
            bad_channels = self.epochs.sensor._array_index(ds.info[BAD_CHANNELS])
        else:
            bad_channels = []

        return accept, tag, interpolate, bad_channels, windows

    def save(self):
        # find dest path
        _, ext = os.path.splitext(self.path)

        # create Dataset to save
        info = {BAD_CHANNELS: self.bad_channel_names, 'epochs.selection': self.epochs_selection}
        ds = Dataset([self.trigger, self.accept, self.tag, self.interpolate], info=info)
        if any(self.interpolate_windows):
            ds[INTERPOLATE_WINDOWS] = Datalist(self.interpolate_windows, INTERPOLATE_WINDOWS)

        if ext.startswith('.pickle'):
            save.pickle(ds, self.path)
        elif ext == '.txt':
            ds.save_txt(self.path)
        else:
            raise ValueError(f"Unsupported extension: {ext!r}")


class Model(FileModel):
    """Manages a document as well as its history"""

    def clear(self):
        desc = "Clear"
        index = np.logical_not(self.doc.accept.x)
        old_tag = self.doc.tag[index]
        action = ChangeAction(desc, index, False, True, old_tag, 'clear')
        logger = getLogger(__name__)
        logger.info("Clearing %i rejections" % index.sum())
        self.history.do(action)

    def load(
            self,
            path: str,
            answer: bool = None,  # User answer to dialogs (for tests)
    ):
        try:
            new_accept, new_tag, new_interpolate, new_bad_chs, new_windows = self.doc.read_rej_file(path, answer)
        except Exception as error:
            if answer is not None:
                raise
            caption = f"Error reading {os.path.basename(path)}"
            msg = f"{error}\nFor more details see Terminal output."
            app = get_app()
            app.message_box(msg, caption, wx.ICON_ERROR)
            raise

        # create load action
        action = ChangeAction("Load File", FULL_SLICE, self.doc.accept, new_accept,
                              self.doc.tag, new_tag, self.doc.path, path,
                              self.doc.bad_channels, new_bad_chs,
                              self.doc.interpolate, new_interpolate)
        self.history.do(action)
        # interpolation windows are not part of the undo history (read-only mode)
        self.doc.interpolate_windows = new_windows
        self.history.register_save()

    def set_bad_channels(self, bad_channels, desc="Set bad channels"):
        "Set bad channels with a list of int"
        action = ChangeAction(desc, old_bad_chs=self.doc.bad_channels,
                              new_bad_chs=bad_channels)
        self.history.do(action)

    def set_case(self, i, state, tag=None, desc="Manual Change"):
        old_accept = self.doc.accept[i]
        if tag is None:
            old_tag = None
        else:
            old_tag = self.doc.tag[i]
        action = ChangeAction(desc, i, old_accept, state, old_tag, tag)
        self.history.do(action)

    def set_interpolation(self, case, ch_names):
        action = ChangeAction(f"Epoch {case} interpolate {', '.join(ch_names)!r}",
                              case, old_interpolate=self.doc.interpolate[case],
                              new_interpolate=ch_names)
        self.history.do(action)

    def set_range(self, start, stop, new_accept):
        if start < 0:
            start += len(self.doc.accept)

        if stop <= 0:
            stop += len(self.doc.accept)

        if stop <= start:
            return

        old_accept = self.doc.accept[start: stop].copy()
        action = ChangeAction("Set Range", slice(start, stop), old_accept,
                              new_accept)
        self.history.do(action)

    def threshold(self, threshold=2e-12, method='abs'):
        """Find epochs based on a threshold criterion

        Parameters
        ----------
        threshold : scalar
            The threshold value. Examples: 1.25e-11 to detect saturated
            channels; 2e-12: for conservative MEG rejection.
        method : 'abs' | 'p2p'
            How to apply the threshold. With "abs", the threshold is applied to
            absolute values. With 'p2p' the threshold is applied to
            peak-to-peak values.

        Returns
        -------
        sub_threshold : array of bool
            True for all epochs in which the criterion is not reached (i.e.,
            epochs that should be accepted).
        """
        args = ', '.join(map(str, (threshold, method)))
        logger = getLogger(__name__)
        logger.info(f"Auto-reject trials: {args}")

        if method == 'abs':
            x = [x.abs().max(('time', 'sensor')) for x in
                 self.doc.iter_good_epochs()]
        elif method == 'p2p':
            x = [(x.max('time') - x.min('time')).max('sensor') for x in
                 self.doc.iter_good_epochs()]
        else:
            raise ValueError(f"Invalid method: {method!r}")
        return np.array(x) < threshold

    def threshold_multi(self, type_thresholds, method='abs'):
        """Find epochs based on per-channel-type threshold criteria.

        Parameters
        ----------
        type_thresholds : list of (ch_type, threshold_si, display_str)
            As returned by ``ThresholdDialog.GetThresholds()``.
        method : 'abs' | 'p2p'

        Returns
        -------
        sub_threshold : array of bool
            True for epochs where no channel type exceeds its threshold.
        """
        logger = getLogger(__name__)
        logger.info("Auto-reject trials (multi-type): %s", type_thresholds)
        type_thresh_dict = {ct: thresh for ct, thresh, _ in type_thresholds}
        bad_names = set(self.doc.bad_channel_names)
        n = self.doc.n_epochs
        above = np.zeros(n, bool)
        for ct, ndvar in self.doc.epochs_by_type:
            if ct not in type_thresh_dict:
                continue
            thresh = type_thresh_dict[ct]
            # Remove globally bad channels
            if bad_names:
                good = [i for i, nm in enumerate(ndvar.sensor.names) if nm not in bad_names]
                if len(good) < len(ndvar.sensor):
                    ndvar = ndvar.sub(sensor=good)
            # ndvar.x shape: (n_cases, n_sensors, n_times)
            x = ndvar.x
            if method == 'abs':
                peaks = np.abs(x).max(axis=(1, 2))
            elif method == 'p2p':
                peaks = (x.max(axis=2) - x.min(axis=2)).max(axis=1)
            else:
                raise ValueError(f"Invalid method: {method!r}")
            above |= peaks >= thresh
        return ~above

    def toggle_interpolation(self, case, ch_name):
        old_interpolate = self.doc.interpolate[case]
        new_interpolate = old_interpolate[:]
        if ch_name in new_interpolate:
            new_interpolate.remove(ch_name)
            desc = "Don't interpolate %s for %i" % (ch_name, case)
        else:
            new_interpolate.append(ch_name)
            desc = "Interpolate %s for %i" % (ch_name, case)
        action = ChangeAction(desc, case, old_interpolate=old_interpolate,
                              new_interpolate=new_interpolate)
        self.history.do(action)

    def update_bad_chs(self, bad_chs, interp, desc):
        if interp is None:
            index = old_interp = new_interp = None
        else:
            new_interp = self.doc.interpolate[:]
            new_interp._update_listlist(interp)
            index = np.flatnonzero(new_interp != self.doc.interpolate)
            if len(index):
                old_interp = self.doc.interpolate[index]
                new_interp = new_interp[index]
            else:
                index = old_interp = new_interp = None

        # find changed bad channels
        old_bad_chs = self.doc.bad_channel_names
        if bad_chs and any(ch not in old_bad_chs for ch in bad_chs):
            new_bad_chs = sorted(set(old_bad_chs).union(bad_chs))
        else:
            new_bad_chs = old_bad_chs = None

        action = ChangeAction(desc, index, old_bad_chs=old_bad_chs, new_bad_chs=new_bad_chs,
                              old_interpolate=old_interp, new_interpolate=new_interp)
        self.history.do(action)

    def update_rejection(self, new_accept, mark_good, mark_bad, desc, new_tag):
        # find changes
        if not mark_good:
            index = np.invert(new_accept)
        elif not mark_bad:
            index = new_accept
        else:
            index = None

        if index is None:
            index = new_accept != self.doc.accept.x
        else:
            np.logical_and(index, new_accept != self.doc.accept.x, index)
        index = np.flatnonzero(index)

        # construct action
        old_accept = self.doc.accept[index]
        new_accept = new_accept[index]
        old_tag = self.doc.tag[index]
        action = ChangeAction(desc, index, old_accept, new_accept, old_tag,
                              new_tag)
        self.history.do(action)


class Frame(NavigableFrame, FileFrame):
    """Epoch rejection GUI

    Exclude bad epochs and interpolate or remove bad channels.

    * Use the `Bad Channels` button in the toolbar to exclude channels from
      analysis (use the `GA` button to plot the grand average and look for
      channels that are consistently bad).
    * Click the `Threshold` button to automatically reject epochs in which the
      signal exceeds a certain threshold.
    * Click on an epoch plot to toggle rejection of that epoch.
    * Click on a channel in the topo-map to mark that channel in the epoch
      plots.
    * Press ``i`` on the keyboard to toggle channel interpolation for the
      channel that is closest to the cursor along the y-axis.
    * Press ``shift-i`` on the keyboard to edit a list of interpolated channels
      for the epoch under the cursor.


    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    right-arrow go to the next page
    left-arrow  go to the previous page
    b           butterfly plot of the epoch under the pointer
    c           pairwise sensor correlation plot or the current epoch
    t           topomap plot of the epoch/time point under the pointer
    i           interpolate the channel nearest to the pointer on the y-axis
    shift-i     open dialog to enter channels for interpolation
    =========== ============================================================
    """
    _doc_name = 'epoch selection'
    _title = "Select Epochs"

    def __init__(
            self,
            parent: wx.Frame | None,
            model: Model,
            nplots: int | tuple[int, int] | None,
            topo: bool | None,
            vlim: float | None,
            color,
            lw: float,
            mark: Sequence | None,
            mcolor,
            mlw: float,
            antialiased: bool,
            pos: tuple[int, int] | None,
            size: tuple[int, int] | None,
            allow_interpolation: bool,
            read_only: bool = False,
    ) -> None:
        """View object of the epoch selection GUI

        Parameters
        ----------
        parent : wx.Frame
            Parent window.
        others :
            See TerminalInterface constructor.
        """
        super().__init__(parent, pos, size, model)
        # long (variable-length) epochs: continuous single-column read-only browser
        self.long_epochs = self.doc.long_epochs
        if self.long_epochs:
            read_only = True
        self.allow_interpolation = allow_interpolation
        self.read_only = read_only

        # bind events
        self.doc.callbacks.subscribe('case_change', self.CaseChanged)
        self.doc.callbacks.subscribe('bad_chs_change', self.ShowPage)

        # setup figure canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.set_facecolor('white')
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.025,
                                    top=.975, wspace=.1, hspace=.25)

        # Toolbar
        tb = self.InitToolbar()
        tb.AddSeparator()

        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        self.page_choice = wx.Choice(tb, -1)
        tb.AddControl(self.page_choice)
        tb.Bind(wx.EVT_CHOICE, self.OnPageChoice)

        # --> forward / backward
        self.AddNavigationButtons(tb)
        tb.AddSeparator()

        if not read_only:
            # --> Bad Channels
            # button = wx.Button(tb, ID.SET_BAD_CHANNELS, "Bad Channels")
            # button.Bind(wx.EVT_BUTTON, self.OnSetBadChannels)
            # tb.AddControl(button)

            # --> Thresholding
            button = wx.Button(tb, ID.THRESHOLD, "Threshold")
            button.Bind(wx.EVT_BUTTON, self.OnThreshold)
            tb.AddControl(button)

        # right-most part
        tb.AddStretchableSpace()

        # Grand-average plot
        button = wx.Button(tb, ID.GRAND_AVERAGE, "GA")
        button.SetHelpText("Plot the grand average of all accepted epochs")
        button.Bind(wx.EVT_BUTTON, self.OnPlotGrandAverage)
        tb.AddControl(button)

        # Info
        tb.AddTool(wx.ID_INFO, 'Info', Icon("actions/info"))
        self.Bind(wx.EVT_TOOL, self.OnInfo, id=wx.ID_INFO)

        # --> Help
        self.InitToolbarTail(tb)
        tb.Realize()

        self.CreateStatusBar()

        # check plot parameters
        if mark:
            mark, invalid = self.doc.epochs.sensor._normalize_sensor_names(mark, missing='return')
            if invalid:
                desc = ', '.join(invalid)
                msg = f"Some channels specified to mark do not exist in the data and will be ignored: {desc}"
                wx.CallLater(1, wx.MessageBox, msg, "Invalid Channels in Mark", wx.OK | wx.ICON_WARNING)
        else:
            mark = []

        # Per-channel-type vlims for normalized butterfly display (multi-type only)
        self._type_vlims = {}  # {ch_type: vmax_si} for normalising each type to ~[-1, 1]
        if len(self.doc.ch_type_names) > 1:
            self._type_vlims = dict(self.doc.type_vmax)
        # Persisted per-type display vlims (in SI), falling back to defaults; these
        # give a stable y-axis limit across sessions rather than scaling to the data.
        self._auto_vlim = self.config.ReadBool('VLim/auto', False)
        self._type_display_vlims = {}
        for ch_type in self.doc.ch_type_names:
            saved = self.config.ReadFloat(f'VLim/vlim_{ch_type}', -1.0)
            if saved > 0:
                self._type_display_vlims[ch_type] = saved
            else:
                self._type_display_vlims[ch_type] = CH_TYPE_DEFAULT_VLIM_SI[ch_type]

        # setup plot parameters
        plot_list = ((self.doc.epochs,),)
        cmaps = find_fig_cmaps(plot_list)
        # Single-type: start from the persisted / default limit rather than the data
        if vlim is None and not self._type_vlims:
            vlim = self._type_display_vlims[self.doc.ch_type_names[0]]
        self._vlims = find_fig_vlims(plot_list, vlim, None, cmaps)
        self._mark = mark
        self._bfly_kwargs = {'color': color, 'lw': lw, 'mlw': mlw,
                             'antialiased': antialiased, 'vlims': self._vlims,
                             'mcolor': mcolor}
        self._topo_kwargs = {'vlims': self._vlims, 'mcolor': 'red', 'mmarker': 'x'}

        # transient artists for windowed-interpolation highlights (long mode)
        self._window_handles = []

        if self.long_epochs:
            self._set_layout_long(topo)
        else:
            self._SetLayout(nplots, topo)

        # Bind Events ---
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        self.canvas.mpl_connect('motion_notify_event', self.OnPointerMotion)

        # plot objects
        self._current_page_i = None
        self._epoch_idxs = None
        self._case_plots = None
        self._case_axes = None
        self._case_segs = None
        self._axes_by_idx = None
        self._topo_axes = []        # list of axes (before/after-rejection pair per channel type)
        self._topo_plots = []       # list of AxTopomap, parallel to _topo_axes
        self._topo_specs = []       # list of (ch_type, kind) where kind in {'all', 'rejected'}
        self._topo_interp_handles = []  # transient blue-x marks for the hovered epoch's rejected channels
        self._topo_plot_info_str = None
        self._case_segs_by_type = None  # list of [(ch_type, NDVar)] per visible epoch
        self._bfly_vlim = None          # normalized vlim last applied (multi-type only)

        # Finalize
        self.ShowPage(0)
        self.UpdateTitle()

    def CanBackward(self):
        return bool(self._current_page_i > 0)

    def CanForward(self):
        return bool(self._current_page_i < self._n_pages - 1)

    def CanSave(self) -> bool:
        if self.read_only:
            return False
        return super().CanSave()

    def UpdateTitle(self) -> None:
        super().UpdateTitle()
        if self.read_only:
            self.SetTitle(self.GetTitle() + ' (read-only)')

    def CaseChanged(self, index):
        "Update the states of the segments on the current page"
        if isinstance(index, INT_TYPES):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.doc.n_epochs
            index = range(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # update epoch plots
        axes = []
        for idx in index:
            if idx in self._axes_by_idx:
                ax = self._axes_by_idx[idx]
                ax_idx = ax.ax_idx
                h = self._case_plots[ax_idx]
                h.set_state(self.doc.accept[idx])
                # interpolated channels
                ch_index = h.epoch.sensor.channel_idx
                h.set_marked(INTERPOLATE_CHANNELS, [ch_index[ch] for ch in
                                                    self.doc.interpolate[idx]
                                                    if ch in ch_index])

                axes.append(ax)

        self.canvas.redraw(axes)
        self.canvas.store_canvas()

    def GoToEpoch(self, i):
        if self.long_epochs:
            for page, rows in enumerate(self._segs_by_page):
                if any(self._rows_spec[r][0] == i for r in rows):
                    break
            else:
                raise ValueError(f"Epoch not found: {i!r}")
        else:
            for page, epochs in enumerate(self._segs_by_page):
                if i in epochs:
                    break
            else:
                raise ValueError(f"Epoch not found: {i!r}")
        if page != self._current_page_i:
            self.SetPage(page)

    def MakeToolsMenu(self, menu):
        app = wx.GetApp()
        item = menu.Append(wx.ID_ANY, "Set Bad Channels",
                           "Specify bad channels for the whole file")
        app.Bind(wx.EVT_MENU, self.OnSetBadChannels, item)
        item = menu.Append(wx.ID_ANY, "Set Rejection for Range",
                           "Set the rejection status for a range of epochs")
        app.Bind(wx.EVT_MENU, self.OnRejectRange, item)
        item = menu.Append(wx.ID_ANY, "Find Epochs by Threshold",
                           "Find epochs based in a specific threshold")
        app.Bind(wx.EVT_MENU, self.OnThreshold, item)
        item = menu.Append(wx.ID_ANY, "Find Bad Channels",
                           "Find bad channels using different criteria")
        app.Bind(wx.EVT_MENU, self.OnFindNoisyChannels, item)
        menu.AppendSeparator()
        item = menu.Append(wx.ID_ANY, "Plot Grand Average",
                           "Plot the grand average of all accepted epochs "
                           "(does not perform single-epoch channel "
                           "interpolation)")
        app.Bind(wx.EVT_MENU, self.OnPlotGrandAverage, item)
        item = menu.Append(wx.ID_ANY, "Info")
        app.Bind(wx.EVT_MENU, self.OnInfo, item)

    def OnBackward(self, event):
        "Turn the page backward"
        self.SetPage(self._current_page_i - 1)

    def OnCanvasClick(self, event):
        "Called by mouse clicks"
        ax = event.inaxes
        logger = getLogger(__name__)
        if ax:
            logger.debug("Canvas click at ax.ax_idx=%i", ax.ax_idx)
            if ax.ax_idx >= 0:
                if self.read_only:
                    return
                idx = ax.epoch_idx
                state = not self.doc.accept[idx]
                tag = "manual"
                desc = "Epoch %i %s" % (idx, state)
                self.model.set_case(idx, state, tag, desc)
            elif ax.ax_idx == TOPO_PLOT:
                topo = self._topo_plots[getattr(ax, 'topo_idx', 0)]
                ch_locs = topo.sensors.locations
                sensor_i = np.argmin(cdist(ch_locs, [[event.xdata, event.ydata]]))
                sensor = topo.sensors.sensors.names[sensor_i]
                if sensor in self._mark:
                    self._mark.remove(sensor)
                else:
                    self._mark.append(sensor)
                self.SetPlotStyle(mark=self._mark)
        else:
            logger.debug("Canvas click outside axes")

    def OnCanvasKey(self, event):
        # GUI Control events
        if event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
            return
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
            return
        elif event.key == 'u':
            if self.CanUndo():
                self.OnUndo(None)
            return
        elif event.key == 'U':
            if self.CanRedo():
                self.OnRedo(None)
            return

        # plotting
        ax = event.inaxes
        if ax is None or ax.ax_idx == TOPO_PLOT:
            return
        elif ax.ax_idx > 0 and ax.epoch_idx == OUT_OF_RANGE:
            return
        elif event.key == 't':
            self.PlotTopomap(ax.ax_idx, event.xdata)
        elif event.key == 'b':
            self.PlotButterfly(ax.ax_idx)
        elif event.key == 'c':
            self.PlotCorrelation(ax.ax_idx)
        elif event.key == 'i':
            if not self.read_only:
                self.ToggleChannelInterpolation(ax, event)
        elif event.key == 'I':
            if not self.read_only:
                self.OnSetInterpolation(ax.epoch_idx)

    def OnFindNoisyChannels(self, event):
        if self.read_only:
            return
        dlg = FindNoisyChannelsDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            # Find bad channels
            flat, flat_average, mincorr = dlg.GetValues()
            if flat:
                flats = meeg.find_flat_epochs(self.doc.epochs, flat)
            else:
                flats = None

            if flat_average:
                flats_av = meeg.find_flat_evoked(self.doc.epochs, flat_average)
            else:
                flats_av = None

            if mincorr:
                noisies = meeg.find_noisy_channels(self.doc.epochs, mincorr)
            else:
                noisies = None

            # Apply
            if dlg.do_apply.GetValue():
                has_flats = flats and any(flats)
                has_noisies = noisies and any(noisies)
                if has_flats and has_noisies:
                    interp = flats[:]
                    interp._update_listlist(noisies)
                elif has_flats:
                    interp = flats
                elif has_noisies:
                    interp = noisies
                else:
                    interp = None
                self.model.update_bad_chs(flats_av, interp, "Find noisy channels")

            # Show Report
            if dlg.do_report.GetValue():
                doc = fmtxt.Section("Noisy Channels")
                doc.append(f"Total of {len(self.doc.epochs)} epochs.")
                if flat_average:
                    sec = doc.add_section(f"Flat in the average (<{flat_average})")
                    if flats_av:
                        sec.add_paragraph(', '.join(flats_av))
                    else:
                        sec.add_paragraph("None.")
                if flat:
                    sec = doc.add_section(f"Flat Channels (<{flat})")
                    sec.add_paragraph(format_epoch_list(flats))
                if mincorr:
                    sec = doc.add_section(f"Neighbor correlation < {mincorr}")
                    sec.add_paragraph(format_epoch_list(noisies))
                InfoFrame(self, "Noisy Channels", doc.get_html())

            dlg.StoreConfig()
        dlg.Destroy()

    def OnForward(self, event):
        "Turn the page forward"
        self.SetPage(self._current_page_i + 1)

    def OnInfo(self, event):
        doc = fmtxt.Section(f"{self.doc.n_epochs} Epochs")

        # rejected epochs
        rejected = np.invert(self.doc.accept.x)
        sec = doc.add_section(_text.n_of(rejected.sum(), 'epoch') + ' rejected')
        if np.any(rejected):
            para = fmtxt.delim_list(fmtxt.Link(epoch, f"epoch:{epoch}") for epoch in np.flatnonzero(rejected))
            sec.add_paragraph(para)

        # bad channels
        heading = _text.n_of(len(self.doc.bad_channels), "bad channel", True)
        sec = doc.add_section(heading.capitalize())
        if self.doc.bad_channels:
            sec.add_paragraph(', '.join(self.doc.bad_channel_names))

        # interpolation
        if self.long_epochs:
            # total interpolated time per channel across all epochs
            duration = {}
            for windows in self.doc.interpolate_windows:
                for w in windows:
                    duration[w.channel] = duration.get(w.channel, 0.0) + (w.tmax - w.tmin)
            if duration:
                sec = doc.add_section("Interpolated time windows")
                items = fmtxt.List("Channel (total interpolated duration):")
                for ch in sorted(duration, key=lambda c: -duration[c]):
                    items.add_item(f"{ch}: {duration[ch]:g} s")
                sec.add_paragraph(items)
            else:
                doc.add_section("No time-windowed channel interpolation")
        elif any(self.doc.interpolate):
            sec = doc.add_section("Interpolate channels")
            sec.add_paragraph(format_epoch_list(self.doc.interpolate, "Interpolation by epoch:"))
        else:
            doc.add_section("No channels interpolated in individual epochs")

        InfoFrame(self, "Rejection Info", doc.get_html())

    def OnPageChoice(self, event):
        "Called by the page Choice control"
        page = event.GetSelection()
        self.SetPage(page)

    def OnPlotGrandAverage(self, event):
        self.PlotGrandAverage()

    def OnPointerMotion(self, event):
        "Update view on mouse pointer movement"
        ax = event.inaxes
        if not ax:
            return self.SetStatusText("")
        elif ax.ax_idx == TOPO_PLOT:
            return self.SetStatusText(self._topo_plot_info_str)
        elif ax.epoch_idx == OUT_OF_RANGE:
            return self.SetStatusText("")

        # compose status text
        x = ax.xaxis.get_major_formatter().format_data(event.xdata)
        if self._type_vlims:
            y_parts = []
            for ch_type in self.doc.ch_type_names:
                if ch_type not in self._type_display_vlims:
                    continue
                y_si = event.ydata * self._type_display_vlims[ch_type]
                display_unit, scale = ch_type_scale(ch_type)
                y_parts.append(f'{y_si * scale:.4g} {display_unit}')
            y = ' / '.join(y_parts)
        else:
            y = ax.yaxis.get_major_formatter().format_data(event.ydata)
        status = f"Epoch {ax.epoch_idx},  x = {x} s,  y = {y}"
        if ax.ax_idx >= 0:  # single trial plot
            if self.long_epochs:
                interp = sorted({w.channel for w in self.doc.windows_in_range(ax.epoch_idx, event.xdata, event.xdata + 1e-9)})
            else:
                interp = self.doc.interpolate[ax.epoch_idx]
            if interp:
                status += f",  interpolate {', '.join(interp)}"
        self.SetStatusText(status)

        # update topomaps (before / after channel rejection) at the pointer time
        if self._plot_topo:
            self._update_topomaps(ax.epoch_idx, ax.ax_idx, event.xdata)
            marked = ', '.join(self._mark)
            self._topo_plot_info_str = (f"Topomap: Epoch {ax.epoch_idx},  t = {x} s,  marked: {marked}")

    def OnRejectRange(self, event):
        if self.read_only:
            return
        dlg = RejectRangeDialog(self)
        if dlg.ShowModal() == wx.ID_OK:
            start = int(dlg.first.GetValue())
            stop = int(dlg.last.GetValue()) + 1
            state = bool(dlg.action.GetSelection())
            self.model.set_range(start, stop, state)
            dlg.StoreConfig()
        dlg.Destroy()

    def OnSetBadChannels(self, event):
        if self.read_only:
            return
        default_value = ', '.join(self.doc.bad_channel_names)
        dlg = wx.TextEntryDialog(self, "Please enter bad channel names separated by comma (e.g., \"MEG 003, MEG 010\"):", "Set Bad Channels", default_value)
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names_in = filter(None, (s.strip() for s in dlg.GetValue().split(',')))
                    names = self.doc.epochs.sensor._normalize_sensor_names(names_in)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        bad_channels = self.doc.epochs.sensor._array_index(names)
        self.model.set_bad_channels(bad_channels)

    def OnSetInterpolation(self, epoch):
        "Show Dialog for channel interpolation for this epoch (index)"
        if self.read_only:
            return
        old = self.doc.interpolate[epoch]
        dlg = wx.TextEntryDialog(self, "Please enter channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set "
                                 "Channels for Interpolation", ', '.join(old))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names = filter(None, (s.strip() for s in dlg.GetValue().split(',')))
                    new = self.doc.epochs.sensor._normalize_sensor_names(names)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        if new != old:
            self.model.set_interpolation(epoch, new)

    def OnSetLayout(self, event):
        if self.long_epochs:
            dlg = LongLayoutDialog(self, self._rows_per_page, self._seconds_per_row, self._plot_topo)
            if dlg.ShowModal() == wx.ID_OK:
                self.config.WriteInt('Layout/n_rows', dlg.rows_per_page)
                self.config.WriteFloat('Layout/seconds_per_row', dlg.seconds_per_row)
                self.config.Flush()
                self._set_layout_long(dlg.topo)
                self.ShowPage(0)
            dlg.Destroy()
            return
        dlg = LayoutDialog(self, self._rows, self._columns, self._plot_topo)
        if dlg.ShowModal() == wx.ID_OK:
            self.SetLayout(dlg.layout, dlg.topo)

    def OnSetMarkedChannels(self, event):
        "Mark is represented in sensor names"
        dlg = wx.TextEntryDialog(self, "Please enter channel names separated by "
                                 "comma (e.g., \"MEG 003, MEG 010\"):", "Set Marked"
                                 "Channels", ', '.join(self._mark))
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    names_in = filter(None, (s.strip() for s in dlg.GetValue().split(',')))
                    names = self.doc.epochs.sensor._normalize_sensor_names(names_in)
                    break
                except ValueError as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        self.SetPlotStyle(mark=names)

    def OnSetVLim(self, event):
        if self._type_vlims:
            # Multi-type: per-channel-type dialog
            type_scales = {ct: ch_type_scale(ct) for ct in self.doc.ch_type_names}
            dlg = VLimDialog(self, type_scales, self._type_display_vlims, self._auto_vlim, CH_TYPE_DEFAULT_VLIM_SI)
            if dlg.ShowModal() == wx.ID_OK:
                self._auto_vlim = dlg.GetAuto()
                self._type_display_vlims = dlg.GetVLims()
                self.config.WriteBool('VLim/auto', self._auto_vlim)
                for ch_type, vlim_si in self._type_display_vlims.items():
                    self.config.WriteFloat(f'VLim/vlim_{ch_type}', vlim_si)
                self.config.Flush()
                self.ShowPage()
            dlg.Destroy()
        else:
            # Single-type: text entry in the data's display unit
            ch_type = self.doc.ch_type_names[0]
            display_unit, scale = ch_type_scale(ch_type)
            vlim_si = tuple(self._vlims.values())[0][1]
            prompt = f"New Y-axis limit ({display_unit}):" if display_unit else "New Y-axis limit:"
            dlg = wx.TextEntryDialog(self, prompt, "Set Y-Axis Limit", f'{vlim_si * scale:g}')
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                try:
                    vlim = abs(float(value)) / scale
                except Exception as exception:
                    msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                           wx.OK | wx.ICON_ERROR)
                    msg.ShowModal()
                    msg.Destroy()
                    raise
                self.SetVLim(vlim)
                # Persist so the limit is stable across sessions
                self._type_display_vlims[ch_type] = vlim
                self.config.WriteFloat(f'VLim/vlim_{ch_type}', vlim)
                self.config.Flush()
            dlg.Destroy()

    def OnThreshold(self, event):
        if self.read_only:
            return
        type_scales = {ct: ch_type_scale(ct) for ct in self.doc.ch_type_names}
        dlg = ThresholdDialog(self, type_scales, _THRESHOLD_DEFAULT_SI)
        if dlg.ShowModal() == wx.ID_OK:
            method = dlg.GetMethod()
            if dlg.type_rows:
                # Multi-type: per-channel-type thresholds
                type_thresholds = dlg.GetThresholds()
                if not type_thresholds:
                    dlg.Destroy()
                    return
                sub_threshold = self.model.threshold_multi(type_thresholds, method)
                threshold_desc = ', '.join(f'{s} ({ct})' for ct, _, s in type_thresholds)
            else:
                threshold = dlg.GetThreshold()
                sub_threshold = self.model.threshold(threshold, method)
                threshold_desc = str(threshold)
            mark_below = dlg.GetMarkBelow()
            mark_above = dlg.GetMarkAbove()
            if mark_below or mark_above:
                self.model.update_rejection(sub_threshold, mark_below,
                                            mark_above, f"Threshold-{method}",
                                            f"{method}_{threshold_desc}")
            if dlg.do_report.GetValue():
                rejected = np.invert(sub_threshold)
                doc = fmtxt.Section("Threshold")
                doc.append("%s at %s:  reject %i of %i epochs:" %
                           (method, threshold_desc, rejected.sum(), len(rejected)))
                if np.any(rejected):
                    para = fmtxt.delim_list(fmtxt.Link(epoch, "epoch:%i" % epoch) for epoch in np.flatnonzero(rejected))
                    doc.add_paragraph(para)
                InfoFrame(self, "Rejection Info", doc.get_html())
            dlg.StoreConfig()
        dlg.Destroy()

    def OnUpdateUISetMarkedChannels(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(True)

    def PlotCorrelation(self, ax_index):
        epoch_idx = self._epoch_idxs[ax_index]
        seg = self._case_segs[ax_index]
        name = 'Epoch %i Neighbor Correlation' % epoch_idx
        plot.Topomap(neighbor_correlation(seg, name=name), sensorlabels='name')

    def PlotButterfly(self, ax_index):
        epoch = self._get_ax_data(ax_index)
        plot.TopoButterfly(epoch, vmax=self._vlims)

    def PlotGrandAverage(self):
        if self.long_epochs:
            wx.MessageBox("Grand average is not available for variable-length epochs.", "Grand Average Unavailable", wx.OK | wx.ICON_INFORMATION)
            return
        if len(self.doc.epochs_by_type) > 1:
            for ch_type, epoch in self.doc.get_grand_averages_by_type():
                plot.TopoButterfly(epoch, title=f"Grand Average – {ch_type}")
        else:
            plot.TopoButterfly(self.doc.get_grand_average())

    def PlotTopomap(self, ax_index, time):
        tseg = self._get_ax_data(ax_index, time)
        plot.Topomap(tseg, vmax=self._vlims, sensorlabels='name', w=8,
                     title=tseg.name)

    def SetLayout(self, nplots=(6, 6), topo=True):
        """Determine the layout of the Epochs canvas

        Parameters
        ----------
        nplots : int | tuple of 2 int
            Number of epoch plots per page. Can be an ``int`` to produce a
            square layout with that many epochs, or an ``(n_rows, n_columns)``
            tuple.
        topo : bool
            Show a topomap plot of the time point under the mouse cursor.
        """
        self._SetLayout(nplots, topo)
        self.ShowPage(0)

    def _SetLayout(
            self,
            nplots: int | tuple[int, int] | None,
            topo: bool,
    ):
        if topo is None:
            topo = self.config.ReadBool('Layout/show_topo', True)
        else:
            topo = bool(topo)
            self.config.WriteBool('Layout/show_topo', topo)

        # the topomaps occupy whole grid cells (the last ``n_reserve``), one cell
        # per map (all + rejected, per channel type), so epochs keep full cell size
        n_topo = 2 * len(self.doc.ch_type_names)

        if nplots is None:
            nrow = self.config.ReadInt('Layout/n_rows', 6)
            ncol = self.config.ReadInt('Layout/n_cols', 6)
            nax = ncol * nrow
            reserve = n_topo if topo else 0
            if reserve >= nax:
                topo = False
                reserve = 0
            n_per_page = nax - reserve
        else:
            if isinstance(nplots, int):
                if nplots < 1:
                    raise ValueError(f"{nplots=}: needs to be >= 1")
                reserve = n_topo if topo else 0
                nax = nplots + reserve
                nrow = int(math.ceil(math.sqrt(nax)))
                ncol = int(math.ceil(nax / nrow))
                n_per_page = nplots
            else:
                nrow, ncol = nplots
                nax = ncol * nrow
                if nax < 1:
                    raise ValueError(f"{nplots=}: Need at least one plot.")
                reserve = n_topo if topo else 0
                if reserve >= nax:
                    topo = False
                    reserve = 0
                n_per_page = nax - reserve
            self.config.WriteInt('Layout/n_rows', nrow)
            self.config.WriteInt('Layout/n_cols', ncol)
        self.config.Flush()

        self._plot_topo = topo
        self._n_reserve = reserve  # grid cells reserved for topomaps (one per map)

        # prepare segments
        n = self.doc.n_epochs
        self._rows = nrow
        self._columns = ncol
        self._n_per_page = n_per_page
        self._n_pages = n_pages = int(math.ceil(n / n_per_page))

        # get a list of IDS for each page
        self._segs_by_page = []
        for i in range(n_pages):
            start = i * n_per_page
            stop = min((i + 1) * n_per_page, n)
            self._segs_by_page.append(np.arange(start, stop))

        # update page selector
        pages = []
        for i in range(n_pages):
            istart = self._segs_by_page[i][0]
            if i == n_pages - 1:
                pages.append('%i: %i..%i' % (i, istart, self.doc.n_epochs))
            else:
                pages.append('%i: %i...' % (i, istart))
        self.page_choice.SetItems(pages)

    def SetPlotStyle(self, **kwargs):
        """Select channels to mark in the butterfly plots.

        Parameters
        ----------
        color : None | matplotlib color
            Color for primary data (default is black).
        lw : scalar
            Linewidth for normal sensor plots.
        mark : None | str | list of str
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        mlw : scalar
            Line width for marked sensor plots.
        antialiased : bool
            Perform Antialiasing on epoch plots (associated with a minor speed
            cost).
        """
        self._SetPlotStyle(**kwargs)
        self.ShowPage()

    def _SetPlotStyle(self, **kwargs):
        "See .SetPlotStyle()"
        for key, value in kwargs.items():
            if key == 'vlims':
                raise TypeError(f"{key!r} is an invalid keyword argument for this function")
            elif key == 'mark':
                self._mark = value
            elif key in self._bf_kwargs:
                self._bfly_kwargs[key] = value
            else:
                raise KeyError(repr(key))

    def SetVLim(self, vlim):
        """Set the value limits (butterfly plot y axes and topomap colormaps)

        Parameters
        ----------
        vlim : scalar | (scalar, scalar)
            For symmetric limits the positive vmax, for asymmetric limits a
            (vmin, vmax) tuple.
        """
        for p in self._case_plots:
            p.set_ylim(vlim)
        for topo in self._topo_plots:
            topo.set_vlim(vlim)

        if np.isscalar(vlim):
            vlim = (-vlim, vlim)
        for key in self._vlims:
            self._vlims[key] = vlim
        self.canvas.draw()
        self.canvas.store_canvas()

    def _page_change(self, page):
        "Perform operations common to page change events"
        self._current_page_i = page
        self.page_choice.Select(page)
        if self.long_epochs:
            self._epoch_idxs = [self._rows_spec[r][0] for r in self._segs_by_page[page]]
        else:
            self._epoch_idxs = self._segs_by_page[page]

    def _get_bfly_vlim(self):
        """Normalized butterfly y-axis limit for the combined multi-type display.

        In auto mode: fitted to the largest value visible on the current page.
        In fixed mode: derived from ``_type_display_vlims`` (per-type SI limits).
        """
        if self._auto_vlim:
            return max(1.0, max(float(np.abs(h.epoch.x).max()) for h in self._case_plots))
        return 1.0

    def SetPage(self, page):
        "Change the page that is displayed without redrawing"
        if self.long_epochs:
            # rows differ in time range/epoch between pages: full redraw
            self.ShowPage(page)
            return
        self._page_change(page)

        self._case_segs = []
        self._case_segs_by_type = []
        self._axes_by_idx = {}

        for i, epoch_idx in enumerate(self._epoch_idxs):
            ax = self._case_axes[i]
            h = self._case_plots[i]
            case_by_type = self.doc.get_epoch_by_type(epoch_idx, 'Epoch %i' % epoch_idx)
            display_case = self._get_display_epoch(case_by_type)
            h.set_data(display_case, epoch_idx)
            h.set_state(self.doc.accept[epoch_idx])
            chs = [display_case.sensor.channel_idx[ch]
                   for ch in self.doc.interpolate[epoch_idx]
                   if ch in display_case.sensor.channel_idx]
            h.set_marked(INTERPOLATE_CHANNELS, chs)
            h.set_visible()

            # store objects
            ax.epoch_idx = epoch_idx
            self._case_segs.append(case_by_type[0][1])   # primary type for _get_ax_data
            self._case_segs_by_type.append(case_by_type)
            self._axes_by_idx[epoch_idx] = ax

        # hide lines axes without data
        if len(self._epoch_idxs) < len(self._case_axes):
            for i in range(len(self._epoch_idxs), len(self._case_plots)):
                self._case_plots[i].set_visible(False)
                self._case_axes[i].epoch_idx = OUT_OF_RANGE

        # Enforce consistent y-axis limits across all epoch plots (multi-type)
        if self._type_vlims and self._case_plots:
            self._bfly_vlim = self._get_bfly_vlim()
            for h in self._case_plots:
                h.set_ylim(self._bfly_vlim)
            if self._plot_topo:
                for (ch_type, _), topo in zip(self._topo_specs, self._topo_plots):
                    if ch_type in self._type_display_vlims:
                        topo.set_vlim(self._bfly_vlim * self._type_display_vlims[ch_type])

        self.canvas.draw()
        self.canvas.store_canvas()

    def _get_display_epoch(self, case_by_type):
        """Build a combined ``sensor × time`` NDVar for butterfly display.

        For single channel type: returns the case NDVar unchanged.
        For multiple types: normalises each type by its display vlim
        (``_type_display_vlims``) so that the user-specified amplitude maps to
        ±1 on the shared y-axis, then concatenates the sensor dimensions.  The returned NDVar is only used for drawing and
        is not stored on the Document.

        Also accepts a plain NDVar and returns it unchanged.
        """
        if isinstance(case_by_type, NDVar):
            return case_by_type
        if len(case_by_type) == 1:
            return case_by_type[0][1]
        from .._data_obj import Sensor as SensorDim
        arrays, names, locs = [], [], []
        for ch_type, case_ndvar in case_by_type:
            vmax = self._type_display_vlims.get(ch_type, 1.0)
            arrays.append(case_ndvar.x / vmax)
            names.extend(case_ndvar.sensor.names)
            locs.append(case_ndvar.sensor.locs)
        combined_sensor = SensorDim(np.vstack(locs), names, adjacency='none')
        combined_x = np.concatenate(arrays, axis=0)  # sensor is dim 0 for single-case
        return NDVar(combined_x, (combined_sensor, case_by_type[0][1].time))

    def ShowPage(self, page=None):
        "Dislay a specific page (start counting with 0)"
        if self.long_epochs:
            return self._show_page_long(page)
        wx.BeginBusyCursor()
        logger = getLogger(__name__)
        t0 = time.time()
        if page is not None:
            self._page_change(page)

        self.figure.clf()
        self._topo_interp_handles = []  # cleared by clf(); drop stale references
        nrow = self._rows
        ncol = self._columns

        # formatters
        t_formatter, t_locator, t_label = self.doc.epochs.time._axis_format(True, True)
        y_scale = AxisScale(self.doc.epochs, True)

        # butterfly kwargs: for multi-type normalized data autoscale the y-axis
        bfly_kwargs = dict(self._bfly_kwargs)
        if self._type_vlims:
            bfly_kwargs['vlims'] = {}

        # segment plots
        self._case_plots = []
        self._case_axes = []
        self._case_segs = []
        self._case_segs_by_type = []
        self._axes_by_idx = {}
        mark = None
        for i, epoch_idx in enumerate(self._epoch_idxs):
            case_by_type = self.doc.get_epoch_by_type(epoch_idx, 'Epoch %i' % epoch_idx)
            display_case = self._get_display_epoch(case_by_type)
            if mark is None:
                mark = [display_case.sensor.channel_idx[ch] for ch in self._mark
                        if ch in display_case.sensor.channel_idx]
            state = self.doc.accept[epoch_idx]
            ax = self.figure.add_subplot(nrow, ncol, i + 1, xticks=[0], yticks=[])
            channel_colors = self.doc.get_channel_colors(display_case)
            h = AxButterflyEpoch(ax, display_case, mark, state, epoch_idx,
                                 channel_colors=channel_colors, **bfly_kwargs)
            # mark interpolated channels
            if self.doc.interpolate[epoch_idx]:
                chs = [display_case.sensor.channel_idx[ch]
                       for ch in self.doc.interpolate[epoch_idx]
                       if ch in display_case.sensor.channel_idx]
                h.set_marked(INTERPOLATE_CHANNELS, chs)
            # mark eye tracker artifacts
            if self.doc.blink is not None:
                PltBinNuts(ax, self.doc.blink[epoch_idx],
                           color=(0.99, 0.76, 0.21))
            # formatters
            if t_locator is not None:
                ax.xaxis.set_major_locator(t_locator)
            ax.xaxis.set_major_formatter(t_formatter)
            if not self._type_vlims:
                ax.yaxis.set_major_formatter(y_scale.formatter)

            # store objects
            ax.ax_idx = i
            ax.epoch_idx = epoch_idx
            self._case_plots.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case_by_type[0][1])   # primary type for _get_ax_data
            self._case_segs_by_type.append(case_by_type)
            self._axes_by_idx[epoch_idx] = ax

        # Enforce a consistent y-axis limit across all epoch plots (multi-type)
        if self._type_vlims and self._case_plots:
            self._bfly_vlim = self._get_bfly_vlim()
            for h in self._case_plots:
                h.set_ylim(self._bfly_vlim)

        # topomaps (before / after channel rejection) in the last reserved cells
        self._topo_axes = []
        self._topo_plots = []
        self._topo_specs = []
        if self._plot_topo:
            bboxes = []
            for k in range(self._n_reserve):
                cell = nrow * ncol - self._n_reserve + 1 + k
                placeholder = self.figure.add_subplot(nrow, ncol, cell)
                bboxes.append(placeholder.get_position())
                self.figure.delaxes(placeholder)
            self._create_topomaps(bboxes)

        self.canvas.draw()
        self.canvas.store_canvas()

        dt = time.time() - t0
        logger.debug('Page draw took %.1f seconds.', dt)
        wx.EndBusyCursor()

    # -- long (variable-length) epoch mode -----------------------------------

    def _set_layout_long(self, topo):
        "Layout for the continuous single-column long-epoch browser"
        if topo is None:
            topo = self.config.ReadBool('Layout/show_topo', True)
        else:
            topo = bool(topo)
            self.config.WriteBool('Layout/show_topo', topo)
        self._plot_topo = topo
        self._n_reserve = 0  # long mode uses a bottom strip, not reserved cells
        self._rows_per_page = max(1, self.config.ReadInt('Layout/n_rows', 6))
        seconds_per_row = self.config.ReadFloat('Layout/seconds_per_row', 10.0)
        self._seconds_per_row = seconds_per_row if seconds_per_row > 0 else 10.0
        self.config.Flush()
        self._build_rows()

    def _build_rows(self):
        "Tile each epoch into fixed-width rows; a new epoch always starts a new row"
        spr = self._seconds_per_row
        rows = []
        for epoch_idx in range(self.doc.n_epochs):
            uts = self.doc.epoch_data[epoch_idx][0][1].time
            t = uts.tmin
            tstop = uts.tstop
            while t < tstop - uts.tstep / 2:
                t_stop = min(t + spr, tstop)
                rows.append((epoch_idx, t, t_stop))
                t = t_stop
        self._rows_spec = rows
        n_rows = len(rows)
        rpp = self._rows_per_page
        self._rows = rpp
        self._columns = 1
        self._n_pages = max(1, math.ceil(n_rows / rpp))
        self._segs_by_page = [np.arange(i * rpp, min((i + 1) * rpp, n_rows)) for i in range(self._n_pages)]
        # page selector labels
        pages = []
        for rows_on_page in self._segs_by_page:
            epoch_idx, t_start, _ = self._rows_spec[rows_on_page[0]]
            pages.append(f"epoch {epoch_idx} ({t_start:g} s)")
        self.page_choice.SetItems(pages)

    def _show_page_long(self, page=None):
        "Draw a page of the continuous single-column long-epoch browser"
        wx.BeginBusyCursor()
        logger = getLogger(__name__)
        t0 = time.time()
        if page is not None:
            self._page_change(page)
        row_idxs = self._segs_by_page[self._current_page_i]

        self.figure.clf()
        self._window_handles = []
        self._topo_interp_handles = []  # cleared by clf(); drop stale references
        rpp = self._rows_per_page

        if self._plot_topo:
            gs = self.figure.add_gridspec(rpp + 1, 1, height_ratios=[1] * rpp + [1.4])
        else:
            gs = self.figure.add_gridspec(rpp, 1)

        bfly_kwargs = dict(self._bfly_kwargs)
        if self._type_vlims:
            bfly_kwargs['vlims'] = {}

        self._case_plots = []
        self._case_axes = []
        self._case_segs = []
        self._case_segs_by_type = []
        self._axes_by_idx = {}
        mark = None
        for i, row_idx in enumerate(row_idxs):
            epoch_idx, t_start, t_stop = self._rows_spec[row_idx]
            case_by_type = self.doc.get_epoch_by_type(epoch_idx, 'Epoch %i' % epoch_idx)
            row_by_type = [(ct, nd.sub(time=(t_start, t_stop))) for ct, nd in case_by_type]
            display_row = self._get_display_epoch(row_by_type)
            if mark is None:
                mark = [display_row.sensor.channel_idx[ch] for ch in self._mark
                        if ch in display_row.sensor.channel_idx]
            state = self.doc.accept[epoch_idx]
            ax = self.figure.add_subplot(gs[i, 0], xticks=[0], yticks=[])
            channel_colors = self.doc.get_channel_colors(display_row)
            label = f'Epoch {epoch_idx}: {t_start:g}–{t_stop:g} s'
            h = AxButterflyEpoch(ax, display_row, mark, state, label,
                                 channel_colors=channel_colors, **bfly_kwargs)
            # highlight the time windows in which channels are interpolated
            self._draw_window_highlights(ax, epoch_idx, display_row)
            # formatters
            t_formatter, t_locator, _ = display_row.time._axis_format(True, True)
            if t_locator is not None:
                ax.xaxis.set_major_locator(t_locator)
            ax.xaxis.set_major_formatter(t_formatter)
            if not self._type_vlims:
                ax.yaxis.set_major_formatter(AxisScale(display_row, True).formatter)

            ax.ax_idx = i
            ax.epoch_idx = epoch_idx
            self._case_plots.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(row_by_type[0][1])
            self._case_segs_by_type.append(row_by_type)
            self._axes_by_idx.setdefault(epoch_idx, ax)

        # consistent y-axis limit across rows (multi-type normalized display)
        if self._type_vlims and self._case_plots:
            self._bfly_vlim = self._get_bfly_vlim()
            for h in self._case_plots:
                h.set_ylim(self._bfly_vlim)

        # persistent before/after-rejection hover-topomaps in a bottom strip
        self._topo_axes = []
        self._topo_plots = []
        self._topo_specs = []
        if self._plot_topo and self._case_segs_by_type:
            placeholder = self.figure.add_subplot(gs[rpp, 0])
            strip = placeholder.get_position()
            self.figure.delaxes(placeholder)
            # square maps flush to the right edge of the strip
            n_topo = 2 * len(self._case_segs_by_type[0])
            fig_w, fig_h = self.figure.get_size_inches()
            size = min(strip.height * fig_h / fig_w, strip.width / n_topo)
            x_start = strip.x0 + strip.width - size * n_topo
            bboxes = [Bbox.from_bounds(x_start + k * size, strip.y0, size, strip.height) for k in range(n_topo)]
            self._create_topomaps(bboxes)

        self.canvas.draw()
        self.canvas.store_canvas()
        logger.debug('Page draw took %.1f seconds.', time.time() - t0)
        wx.EndBusyCursor()

    def _draw_window_highlights(self, ax, epoch_idx, display_row):
        "Recolor interpolated channels' traces over their bad time windows"
        t_start = display_row.time.tmin
        t_stop = display_row.time.tstop
        channel_idx = display_row.sensor.channel_idx
        for w in self.doc.windows_in_range(epoch_idx, t_start, t_stop):
            if w.channel not in channel_idx:
                continue
            a = max(w.tmin, t_start)
            b = min(w.tmax, t_stop)
            seg = display_row.sub(time=(a, b), sensor=w.channel)
            if seg.time.nsamples == 0:
                continue
            handles = ax.plot(seg.time.times, seg.x, color='b', lw=1.2, ls=':', zorder=8)
            self._window_handles.extend(handles)

    # -- topomaps (before / after channel rejection) -------------------------

    def _rejected_channels(self, epoch_idx, t):
        "Channels rejected (interpolated) in ``epoch_idx`` at time ``t``"
        if self.long_epochs:
            return {w.channel for w in self.doc.windows_in_range(epoch_idx, t, t + 1e-9)}
        return set(self.doc.interpolate[epoch_idx])

    def _create_topomaps(self, bboxes):
        """Create the before/after-rejection topomaps, one per ``bbox``.

        ``bboxes`` holds one position per map, ordered ``(type0 all, type0
        rejected, type1 all, ...)``. The first map of each pair shows all
        channels; the second excludes the channels rejected at the cursor time
        (and is blanked when too few channels remain to interpolate). Both keep
        the full sensor layout so that rejected channels can be marked with a
        blue x in either map.
        """
        self._topo_axes = []
        self._topo_plots = []
        self._topo_specs = []
        if not self._case_segs_by_type:
            return
        first_cbt = self._case_segs_by_type[0]
        specs = [(ch_type, kind) for ch_type, _ in first_cbt for kind in ('all', 'rejected')]
        case_by_type = dict(first_cbt)
        mark_topo = [ch for ch in self._mark if ch not in self.doc.bad_channel_names] if self._mark else None
        multi = bool(self._type_vlims)
        t_init = min(max(0.1, first_cbt[0][1].time.tmin), first_cbt[0][1].time.tmax)
        for slot, ((ch_type, kind), bbox) in enumerate(zip(specs, bboxes)):
            case_ndvar = case_by_type[ch_type]
            topo_kwargs = dict(self._topo_kwargs, vlims={}) if multi else dict(self._topo_kwargs)
            ax = self.figure.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height])
            ax.ax_idx = TOPO_PLOT
            ax.topo_idx = slot
            ax.set_axis_off()
            # construct with the full sensor layout so every rejected channel
            # has a marker location, even where excluded from the image
            layers = AxisData([DataLayer(case_ndvar.sub(time=t_init), PlotType.IMAGE)])
            topo = AxTopomap(ax, layers, mark=mark_topo, **topo_kwargs)
            if multi and self._bfly_vlim is not None and ch_type in self._type_display_vlims:
                topo.set_vlim(self._bfly_vlim * self._type_display_vlims[ch_type])
            # label overlapping the bottom of the map (saves vertical space)
            label = ch_type if kind == 'all' else f'{ch_type} −rej'
            ax.text(0.5, 0.0, label, transform=ax.transAxes, ha='center', va='bottom', fontsize=8, zorder=10, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))
            self._topo_axes.append(ax)
            self._topo_plots.append(topo)
            self._topo_specs.append((ch_type, kind))
        self._topo_plot_info_str = ""
        # initial images and marks for the first visible epoch
        self._update_topomaps(self._epoch_idxs[0], 0, t_init, redraw=False)

    def _set_topo_blank(self, ax, blank):
        "White out a topomap axes when too few electrodes remain to interpolate"
        # blitting restores a cached background, so the axes patch is made
        # opaque-white to paint over any previously drawn map
        ax.patch.set_visible(blank)
        if blank:
            ax.patch.set_facecolor('white')
        for artist in (*ax.images, *ax.collections, *ax.lines):
            artist.set_visible(not blank)

    def _update_topomaps(self, epoch_idx, ax_idx, t, redraw=True):
        "Update both topomaps and the blue-x rejected-channel marks at time ``t``"
        if not self._topo_plots:
            return
        case_by_type = dict(self._case_segs_by_type[ax_idx])
        rejected = self._rejected_channels(epoch_idx, t)
        while self._topo_interp_handles:
            self._topo_interp_handles.pop().remove()
        for (ch_type, kind), ax, topo in zip(self._topo_specs, self._topo_axes, self._topo_plots):
            ndvar = case_by_type.get(ch_type)
            if ndvar is None:
                continue
            if kind == 'rejected' and rejected:
                keep = [n for n in ndvar.sensor.names if n not in rejected]
            else:
                keep = list(ndvar.sensor.names)
            if len(keep) < MIN_TOPO_SENSORS:
                self._set_topo_blank(ax, True)
                continue
            self._set_topo_blank(ax, False)
            tseg = ndvar.sub(time=t, sensor=keep) if len(keep) < len(ndvar.sensor) else ndvar.sub(time=t)
            topo.set_data([tseg])
            # mark rejected channels with a blue x
            if rejected:
                smap = topo.sensors
                idx = [smap.sensors._array_index(ch) for ch in rejected if ch in smap.sensors.names]
                if idx:
                    self._topo_interp_handles.append(smap.ax.scatter(smap.locs[idx, 0], smap.locs[idx, 1], s=20, c='blue', marker='x'))
        if redraw:
            self.canvas.redraw(self._topo_axes)

    def ToggleChannelInterpolation(self, ax, event):
        if self.read_only:
            return
        if not self.allow_interpolation:
            wx.MessageBox("Interpolation is disabled for this session",
                          "Interpolation disabled", wx.OK)
            return
        plt = self._case_plots[ax.ax_idx]
        locs = plt.epoch.sub(time=event.xdata).x
        sensor = np.argmin(np.abs(locs - event.ydata))
        sensor_name = plt.epoch.sensor.names[sensor]
        self.model.toggle_interpolation(ax.epoch_idx, sensor_name)

    def _get_ax_data(self, ax_index, time=None):
        if ax_index >= 0:
            epoch_idx = self._epoch_idxs[ax_index]
            epoch_name = 'Epoch %i' % epoch_idx
            seg = self._case_segs[ax_index]
            sensor_idx = self.doc.good_sensor_index(epoch_idx)
        else:
            raise ValueError(f"Invalid ax_index: {ax_index}")

        if time is not None:
            if time is True:
                time = min(max(0.1, seg.time.tmin), seg.time.tmax)
            name = '%s, %i ms' % (epoch_name, 1e3 * time)
            return seg.sub(time=time, sensor=sensor_idx, name=name)
        elif sensor_idx is None:
            return seg
        else:
            return seg.sub(sensor=sensor_idx)


class FindNoisyChannelsDialog(EelbrainDialog):
    def __init__(self, parent, *args, **kwargs):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Find Noisy Channels",
                                *args, **kwargs)
        # load config
        config = parent.config
        flat = config.ReadFloat("FindNoisyChannels/flat", 1e-13)
        flat_average = config.ReadFloat("FindNoisyChannels/flat_average", 1e-14)
        corr = config.ReadFloat("FindNoisyChannels/mincorr", 0.35)
        do_apply = config.ReadBool("FindNoisyChannels/do_apply", True)
        do_report = config.ReadBool("FindNoisyChannels/do_report", True)

        # construct layout
        sizer = wx.BoxSizer(wx.VERTICAL)

        # flat channels
        sizer.Add(wx.StaticText(self, label="Threshold for flat channels:"))
        msg = "Invalid entry for flat channel threshold: {value}. Please specify a number > 0."
        validator = REValidator(POS_FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, value=str(flat), validator=validator)
        ctrl.SetHelpText("A channel that does not deviate from 0 by more than this value is considered flat.")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.flat_ctrl = ctrl

        # flat channels in average
        sizer.Add(wx.StaticText(self, label="Threshold for flat channels in average:"))
        msg = "Invalid entry for average flat channel threshold: {value}. Please specify a number > 0."
        validator = REValidator(POS_FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, value=str(flat_average), validator=validator)
        ctrl.SetHelpText("A channel that does not deviate from 0 by more than "
                         "this value in the average of all epochs is "
                         "considered flat.")
        sizer.Add(ctrl)
        self.flat_average_ctrl = ctrl

        # Correlation
        sizer.Add(wx.StaticText(self, label="Threshold for channel to neighbor correlation:"))
        msg = "Invalid entry for channel neighbor correlation: {value}. Please specify a number > 0."
        validator = REValidator(POS_FLOAT_PATTERN, msg, False)
        ctrl = wx.TextCtrl(self, value=str(corr), validator=validator)
        ctrl.SetHelpText("A channel is considered noisy if the average of the correlation with its neighbors is smaller than this value.")
        sizer.Add(ctrl)
        self.mincorr_ctrl = ctrl

        # output
        sizer.AddSpacer(4)
        sizer.Add(wx.StaticText(self, label="Output"))
        self.do_report = wx.CheckBox(self, wx.ID_ANY, "Show Report")
        self.do_report.SetValue(do_report)
        sizer.Add(self.do_report)
        self.do_apply = wx.CheckBox(self, wx.ID_ANY, "Apply")
        self.do_apply.SetValue(do_apply)
        sizer.Add(self.do_apply)

        # default button
        sizer.AddSpacer(4)
        btn = wx.Button(self, wx.ID_DEFAULT, "Default Settings")
        sizer.Add(btn, border=2)
        btn.Bind(wx.EVT_BUTTON, self.OnSetDefault)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btn.Bind(wx.EVT_BUTTON, self.OnOK)
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetValues(self):
        return (float(self.flat_ctrl.GetValue()),
                float(self.flat_average_ctrl.GetValue()),
                float(self.mincorr_ctrl.GetValue()))

    def OnOK(self, event):
        if self.do_report.GetValue() or self.do_apply.GetValue():
            event.Skip()
        else:
            wx.MessageBox("Specify at least one action (report or apply)", "No Command Selected", wx.ICON_EXCLAMATION)

    def OnSetDefault(self, event):
        self.flat_ctrl.SetValue('1e-13')
        self.flat_average_ctrl.SetValue('1e-14')
        self.mincorr_ctrl.SetValue('0.35')

    def StoreConfig(self):
        config = self.Parent.config
        config.WriteFloat("FindNoisyChannels/flat", float(self.flat_ctrl.GetValue()))
        config.WriteFloat("FindNoisyChannels/flat_average", float(self.flat_average_ctrl.GetValue()))
        config.WriteFloat("FindNoisyChannels/mincorr", float(self.mincorr_ctrl.GetValue()))
        config.WriteBool("FindNoisyChannels/do_report", self.do_report.GetValue())
        config.WriteBool("FindNoisyChannels/do_apply", self.do_apply.GetValue())
        config.Flush()


class LayoutDialog(EelbrainDialog):

    def __init__(self, parent, rows, columns, topo):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Select-Epochs Layout")
        # result attributes
        self.topo = None
        self.layout = None

        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, wx.ID_ANY, "Layout: number of rows and columns (e.g., '5 7')\nor number of epochs (e.g., '35'):"))
        self.text = wx.TextCtrl(self, wx.ID_ANY, "%i %i" % (rows, columns))
        self.text.SelectAll()
        sizer.Add(self.text)

        self.topo_ctrl = wx.CheckBox(self, wx.ID_ANY, "Topographic map")
        self.topo_ctrl.SetValue(topo)
        sizer.Add(self.topo_ctrl)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.Bind(wx.EVT_BUTTON, self.OnOk, id=wx.ID_OK)
        self.SetSizer(sizer)
        sizer.Fit(self)

    def OnOk(self, event):
        self.topo = self.topo_ctrl.GetValue()
        value = self.text.GetValue()
        m = re.match(r"(\d+)\s*(\d*)", value)
        if m:
            rows_str, columns_str = m.groups()
            rows = int(rows_str)
            if columns_str:
                columns = int(columns_str)
                n_plots = rows * columns
                self.layout = (rows, columns)
            else:
                self.layout = n_plots = rows

            if n_plots >= self.topo + 1:
                event.Skip()
                return
            else:
                wx.MessageBox("Layout does not have enough plots", "Invalid Layout", wx.OK | wx.ICON_ERROR, self)
        else:
            wx.MessageBox(f"Invalid layout string: {value}", "Invalid Layout", wx.OK | wx.ICON_ERROR, self)
        self.text.SetFocus()
        self.text.SelectAll()


class LongLayoutDialog(EelbrainDialog):
    "Layout dialog for the continuous long-epoch browser"

    def __init__(self, parent, rows_per_page, seconds_per_row, topo):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Select-Epochs Layout")
        self.rows_per_page = None
        self.seconds_per_row = None
        self.topo = None

        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, wx.ID_ANY, "Rows per page:"))
        validator = REValidator(INT_PATTERN, "Invalid number of rows: {value}. Need an integer.")
        self.rows_ctrl = wx.TextCtrl(self, wx.ID_ANY, str(rows_per_page), validator=validator)
        sizer.Add(self.rows_ctrl)

        sizer.Add(wx.StaticText(self, wx.ID_ANY, "Seconds per row:"))
        validator = REValidator(POS_FLOAT_PATTERN, "Invalid duration: {value}. Need a number > 0.", False)
        self.seconds_ctrl = wx.TextCtrl(self, wx.ID_ANY, f'{seconds_per_row:g}', validator=validator)
        sizer.Add(self.seconds_ctrl)

        self.topo_ctrl = wx.CheckBox(self, wx.ID_ANY, "Topographic map")
        self.topo_ctrl.SetValue(topo)
        sizer.Add(self.topo_ctrl)

        button_sizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.Bind(wx.EVT_BUTTON, self.OnOk, id=wx.ID_OK)
        self.SetSizer(sizer)
        sizer.Fit(self)

    def OnOk(self, event):
        try:
            rows_per_page = int(self.rows_ctrl.GetValue())
            seconds_per_row = float(self.seconds_ctrl.GetValue())
        except ValueError:
            wx.MessageBox("Invalid layout values", "Invalid Layout", wx.OK | wx.ICON_ERROR, self)
            return
        if rows_per_page < 1 or seconds_per_row <= 0:
            wx.MessageBox("Rows per page must be >= 1 and seconds per row > 0", "Invalid Layout", wx.OK | wx.ICON_ERROR, self)
            return
        self.rows_per_page = rows_per_page
        self.seconds_per_row = seconds_per_row
        self.topo = self.topo_ctrl.GetValue()
        event.Skip()


class RejectRangeDialog(EelbrainDialog):

    def __init__(self, parent):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Reject Epoch Range")

        # load config
        config = parent.config
        first = config.ReadInt("RejectRange/first", 0)
        last = config.ReadInt("RejectRange/last", 0)
        action = config.ReadInt("Threshold/action", 0)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Range
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, wx.ID_ANY, "First:"))
        validator = REValidator(INT_PATTERN, "Invalid entry for First: {value}. Need an integer.")
        self.first = wx.TextCtrl(self, wx.ID_ANY, str(first), validator=validator)
        hsizer.Add(self.first)
        hsizer.Add(wx.StaticText(self, wx.ID_ANY, "Last:"))
        validator = REValidator(INT_PATTERN, "Invalid entry for Last: {value}. Need an integer.")
        self.last = wx.TextCtrl(self, wx.ID_ANY, str(last), validator=validator)
        hsizer.Add(self.last)
        sizer.Add(hsizer)

        # Action
        self.action = wx.RadioBox(self, wx.ID_ANY, "Action",
                                  choices=('Reject', 'Accept'),
                                  style=wx.RA_SPECIFY_ROWS)
        self.action.SetSelection(action)
        sizer.Add(self.action)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def StoreConfig(self):
        config = self.Parent.config
        config.WriteInt("RejectRange/first", int(self.first.GetValue()))
        config.WriteInt("RejectRange/last", int(self.last.GetValue()))
        config.WriteInt("RejectRange/action", self.action.GetSelection())
        config.Flush()


class ThresholdDialog(EelbrainDialog):
    """Threshold-criterion rejection dialog.

    When ``type_scales`` is non-empty the dialog shows one row per channel type
    (checkbox + type label + value field + unit label), mirroring
    ``select_components.FindNoisyEpochsDialog``.  Otherwise the legacy
    single-threshold layout is used.

    Parameters
    ----------
    type_scales : {ch_type: (display_unit, scale)}
        Display unit and display-per-SI scale for each channel type (from
        :func:`._ch_types.ch_type_scale`).
    default_thresholds_si : {ch_type: float}
        Default threshold for each channel type, in SI units.
    """

    _methods = (('absolute', 'abs'),
                ('peak-to-peak', 'p2p'))

    def __init__(self, parent, type_scales=None, default_thresholds_si=None):
        title = "Threshold Criterion Rejection"
        wx.Dialog.__init__(self, parent, wx.ID_ANY, title)
        choices = tuple(m[0] for m in self._methods)
        method_tags = tuple(m[1] for m in self._methods)

        config = parent.config
        method = config.Read("Threshold/method", "p2p")
        if method not in method_tags:
            method = "p2p"
        mark_above = config.ReadBool("Threshold/mark_above", True)
        mark_below = config.ReadBool("Threshold/mark_below", False)
        do_report = config.ReadBool("Threshold/do_report", True)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, wx.ID_ANY, "Mark epochs based on a threshold criterion"))

        ctrl = wx.RadioBox(self, wx.ID_ANY, "Method", choices=choices)
        ctrl.SetSelection(method_tags.index(method))
        sizer.Add(ctrl)
        self.method_ctrl = ctrl

        # --- threshold input ---
        type_scales = type_scales or {}
        default_thresholds_si = default_thresholds_si or {}
        self.type_rows = []  # (ch_type, enabled_ctrl, threshold_ctrl, display_unit, scale)
        if type_scales:
            # Multi-type: one row per channel type
            grid = wx.FlexGridSizer(rows=len(type_scales), cols=4, vgap=3, hgap=5)
            for ch_type, (display_unit, scale) in type_scales.items():
                threshold_si = config.ReadFloat(f"Threshold/threshold_si_{ch_type}", default_thresholds_si.get(ch_type, 0))
                threshold = threshold_si * scale
                enabled = config.ReadBool(f"Threshold/enabled_{ch_type}", True)
                enabled_ctrl = wx.CheckBox(self, label='')
                enabled_ctrl.SetValue(enabled)
                grid.Add(enabled_ctrl, flag=wx.ALIGN_CENTER_VERTICAL)
                grid.Add(wx.StaticText(self, label=ch_type), flag=wx.ALIGN_CENTER_VERTICAL)
                validator = REValidator(POS_FLOAT_PATTERN, "Invalid entry: {value}. Please specify a number > 0.", False)
                threshold_ctrl = wx.TextCtrl(self, value=f'{threshold:g}', validator=validator, style=wx.TE_RIGHT)
                threshold_ctrl.SetHelpText(f"Reject epochs where {ch_type} signal exceeds this value at any sensor")
                grid.Add(threshold_ctrl, flag=wx.ALIGN_CENTER_VERTICAL)
                grid.Add(wx.StaticText(self, label=display_unit), flag=wx.ALIGN_CENTER_VERTICAL)
                self.type_rows.append((ch_type, enabled_ctrl, threshold_ctrl, display_unit, scale))
            sizer.Add(grid, flag=wx.ALL, border=5)
            self.threshold_ctrl = None  # legacy attribute unused in multi-type mode
        else:
            # Legacy single-threshold field (no MNE channel-type info available)
            threshold = config.ReadFloat("Threshold/threshold", 2e-12)
            msg = "Invalid entry for threshold: {value}. Need a floating point number."
            validator = REValidator(FLOAT_PATTERN, msg, False)
            ctrl = wx.TextCtrl(self, wx.ID_ANY, str(threshold), validator=validator)
            ctrl.SetHelpText("Threshold value (positive scalar)")
            ctrl.SelectAll()
            sizer.Add(ctrl)
            self.threshold_ctrl = ctrl

        # output
        sizer.AddSpacer(4)
        sizer.Add(wx.StaticText(self, label="Output"))
        self.mark_above = wx.CheckBox(self, wx.ID_ANY, "Reject epochs exceeding the threshold")
        self.mark_above.SetValue(mark_above)
        sizer.Add(self.mark_above)
        self.mark_below = wx.CheckBox(self, wx.ID_ANY, "Accept epochs below the threshold")
        self.mark_below.SetValue(mark_below)
        sizer.Add(self.mark_below)
        self.do_report = wx.CheckBox(self, wx.ID_ANY, "Show Report")
        self.do_report.SetValue(do_report)
        sizer.Add(self.do_report)

        button_sizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btn.Bind(wx.EVT_BUTTON, self.OnOK)
        button_sizer.AddButton(btn)
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetMarkAbove(self):
        return self.mark_above.IsChecked()

    def GetMarkBelow(self):
        return self.mark_below.IsChecked()

    def GetMethod(self):
        index = self.method_ctrl.GetSelection()
        return self._methods[index][1]

    def GetThreshold(self):
        """Return threshold as float (legacy single-type mode only)."""
        return float(self.threshold_ctrl.GetValue())

    def GetThresholds(self):
        """Return ``[(ch_type, threshold_si, display_str), ...]`` for enabled types."""
        result = []
        for ch_type, enabled_ctrl, threshold_ctrl, display_unit, scale in self.type_rows:
            if not enabled_ctrl.GetValue():
                continue
            threshold_display = float(threshold_ctrl.GetValue())
            threshold_si = threshold_display / scale
            result.append((ch_type, threshold_si, f'{threshold_display:g} {display_unit}'))
        return result

    def OnOK(self, event):
        if not (self.mark_above.GetValue() or self.mark_below.GetValue() or self.do_report.GetValue()):
            wx.MessageBox("Specify at least one action (create report or reject or accept epochs)",
                          "No Command Selected", wx.ICON_EXCLAMATION)
        else:
            event.Skip()

    def StoreConfig(self):
        config = self.Parent.config
        config.Write("Threshold/method", self.GetMethod())
        config.WriteBool("Threshold/mark_above", self.GetMarkAbove())
        config.WriteBool("Threshold/mark_below", self.GetMarkBelow())
        config.WriteBool("Threshold/do_report", self.do_report.GetValue())
        if self.type_rows:
            for ch_type, enabled_ctrl, threshold_ctrl, _, scale in self.type_rows:
                config.WriteBool(f"Threshold/enabled_{ch_type}", enabled_ctrl.GetValue())
                config.WriteFloat(f"Threshold/threshold_si_{ch_type}", float(threshold_ctrl.GetValue()) / scale)
        else:
            config.WriteFloat("Threshold/threshold", self.GetThreshold())
        config.Flush()


class VLimDialog(EelbrainDialog):
    """Y-axis limit dialog for multi-channel-type butterfly plots.

    Shows one row per channel type (type label | value field | unit) plus an
    "Automatic" checkbox that disables the fields and lets the frame fit the
    limit to each page's data instead.
    """

    def __init__(self, parent, type_scales, type_vlims_si, auto, default_vlims_si):
        wx.Dialog.__init__(self, parent, wx.ID_ANY, "Set Y-Axis Limits")
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Auto checkbox
        self.auto_ctrl = wx.CheckBox(self, label="Automatic (fit to current page)")
        self.auto_ctrl.SetValue(auto)
        sizer.Add(self.auto_ctrl, flag=wx.ALL, border=5)

        # Per-type grid: [type] [value] [unit]
        grid = wx.FlexGridSizer(rows=len(type_scales), cols=3, vgap=3, hgap=6)
        self._rows = []  # (ch_type, text_ctrl, scale, default_si)
        for ch_type, (display_unit, scale) in type_scales.items():
            default_si = default_vlims_si.get(ch_type, 1.0)
            current_si = type_vlims_si.get(ch_type, default_si)
            grid.Add(wx.StaticText(self, label=ch_type), flag=wx.ALIGN_CENTER_VERTICAL)
            validator = REValidator(POS_FLOAT_PATTERN, "Invalid value: {value}. Need a number > 0.", False)
            ctrl = wx.TextCtrl(self, value=f'{current_si * scale:g}', validator=validator,
                               style=wx.TE_RIGHT)
            ctrl.Enable(not auto)
            grid.Add(ctrl, flag=wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)
            grid.Add(wx.StaticText(self, label=display_unit), flag=wx.ALIGN_CENTER_VERTICAL)
            self._rows.append((ch_type, ctrl, scale, default_si))
        grid.AddGrowableCol(1)
        sizer.Add(grid, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        self.auto_ctrl.Bind(wx.EVT_CHECKBOX, self._on_auto)

        btn = wx.Button(self, wx.ID_DEFAULT, "Defaults")
        btn.Bind(wx.EVT_BUTTON, self._on_defaults)
        sizer.Add(btn, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=8)

        button_sizer = wx.StdDialogButtonSizer()
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        sizer.Add(button_sizer, flag=wx.ALL, border=5)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def _on_auto(self, event):
        enable = not self.auto_ctrl.GetValue()
        for _, ctrl, _, _ in self._rows:
            ctrl.Enable(enable)

    def _on_defaults(self, event):
        for ch_type, ctrl, scale, default_si in self._rows:
            ctrl.SetValue(f'{default_si * scale:g}')

    def GetAuto(self):
        return self.auto_ctrl.GetValue()

    def GetVLims(self):
        """Return ``{ch_type: vlim_si}`` from the current field values."""
        return {ch_type: float(ctrl.GetValue()) / scale
                for ch_type, ctrl, scale, default_si in self._rows}


class InfoFrame(HTMLFrame):

    def OpenURL(self, url):
        m = re.match(r'^(\w+):(\w+)$', url)
        if m:
            kind, address = m.groups()
            if kind == 'epoch':
                self.Parent.GoToEpoch(int(address))
            else:
                raise NotImplementedError(f"{kind} URL")
        else:
            raise ValueError(f"Invalid link URL: {url!r}")
