"""GUI for visualizing raw data and selecting bad channels"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizes Document
#  - listens to Document changes
#  - issues commands to Model

import mne
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import wx

from .._colorspaces import UNAMBIGUOUS_COLORS
from .._data_obj import Dataset, Factor, NDVar, UTS
from .._io.fiff import _picks, sensor_dim as _sensor_dim, _sensor_info
from .._ndvar import neighbor_correlation
from .._types import PathArg
from .._utils.parse import INT_PATTERN, POS_FLOAT_PATTERN
from .. import plot
from ..plot._base import AxisData, DataLayer, PlotType
from ..plot._topo import AxTopomap
from .frame import EelbrainDialog, NavigableFrame
from .history import Action, FileDocument, FileModel, FileFrame
from .mpl_canvas import FigureCanvasPanel
from .utils import REValidator
from ._ch_types import CH_TYPE_COLORS, CH_TYPE_DEFAULT_VLIM_SI, ch_type_scale
from .select_epochs import VLimDialog


DEFAULT_WINDOW = 30.0  # seconds

_EVENT_COLORS = [c for c in UNAMBIGUOUS_COLORS.values()]

TEST_MODE = False


def _minmax_envelope(
        times: np.ndarray,
        data: np.ndarray,
        n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce data to <= ``2 * n_bins`` points/channel, preserving per-bin min & max.

    Parameters
    ----------
    times
        Time axis matching the last dimension of ``data``.
    data
        Data array, ``(n_channels, n_samples)``.
    n_bins
        Number of bins (pixel columns); each bin contributes its minimum and
        maximum, so spikes/noise remain visible after reduction.
    """
    n_samples = data.shape[1]
    bin_size = n_samples // n_bins
    if bin_size <= 1:  # already at or below target resolution
        return times, data
    n_full = n_samples // bin_size
    n_used = n_full * bin_size
    d = data[:, :n_used].reshape(data.shape[0], n_full, bin_size)
    env = np.empty((data.shape[0], 2 * n_full), data.dtype)
    env[:, 0::2] = d.min(axis=2)
    env[:, 1::2] = d.max(axis=2)
    t = times[:n_used].reshape(n_full, bin_size).mean(axis=1)
    return np.repeat(t, 2), env


class ChangeAction(Action):

    def __init__(self, desc: str, old_bad: frozenset, new_bad: frozenset):
        self.desc = desc
        self.old_bad = old_bad
        self.new_bad = new_bad

    def do(self, doc):
        doc.set_bad_channels(self.new_bad)

    def undo(self, doc):
        doc.set_bad_channels(self.old_bad)


class Document(FileDocument):
    """Raw M/EEG data paired with a BIDS channels.tsv for bad-channel selection.

    Parameters
    ----------
    path
        Path to the BIDS channels.tsv file.
    raw
        MNE Raw instance with sensor position information.
    events
        Optional Dataset with event information. See ``t_column`` for the
        time column. A ``'duration'`` column (seconds) causes events to be
        shown as filled rectangles. Any :class:`Factor` columns can be
        selected for color-coding events in the toolbar dropdown.
    t_column
        Name of the column in ``events`` that holds event onset times in
        seconds from the start of the recording.  ``None`` (default) tries
        ``'onset'`` first (BIDS standard), then ``'time'``.
    sysname
        Sensor system name for adjacency lookup.
    adjacency
        Sensor adjacency specification.
    decim
        Decimation factor used when building the downsampled NDVar for
        neighbor-correlation computation.  ``None`` (default) picks a value
        based on ``raw.info['lowpass']``: targets a sample rate of at least
        3× the low-pass cutoff (≥ 100 Hz), falling back to 200 Hz if the
        raw has not been low-pass filtered.
    """

    def __init__(
            self,
            path: PathArg,
            raw: mne.io.BaseRaw,
            events: Dataset | None = None,
            t_column: str | None = None,
            sysname: str = None,
            adjacency=None,
            decim: int | None = None,
    ):
        FileDocument.__init__(self, path)
        self.raw = raw
        self.events = events
        sfreq = raw.info['sfreq']

        # Load channels.tsv
        channels_df = pd.read_csv(path, sep='\t')
        self._channels_df = channels_df
        tsv_channel_names = set(channels_df['name'].tolist())
        initial_bad = frozenset(
            channels_df.loc[channels_df['status'] == 'bad', 'name'].tolist()
        )

        # Resolve decimation factor from lowpass filter if not given
        if decim is None:
            lowpass = raw.info['lowpass']
            if 0 < lowpass < sfreq / 2:
                target_sfreq = max(3 * lowpass, 100.0)
            else:
                target_sfreq = 200.0
            decim = max(1, int(round(sfreq / target_sfreq)))
        self.decim = decim

        # Resolve time column for events
        if t_column is None and events is not None:
            t_column = 'onset' if 'onset' in events else 'time'
        self._t_column = t_column

        # Build per-type NDVars (downsampled for NC computation)
        ndvars_by_type = []

        for ch_type in ('eeg', 'mag', 'grad'):
            all_picks = _picks(raw.info, ch_type, exclude=[])
            # Restrict to channels present in the channels.tsv
            picks = np.array([p for p in all_picks if raw.ch_names[p] in tsv_channel_names])
            if len(picks) == 0:
                continue

            # Build sensor dimension
            sensor = _sensor_dim(raw, picks, sysname, adjacency)
            info = _sensor_info(ch_type, None, raw.info)

            # Extract downsampled data (sensor × time) for NC computation
            x_full = raw[picks, :][0][:, ::decim]
            time = UTS(0.0, decim / sfreq, x_full.shape[1])
            ndvar = NDVar(x_full, (sensor, time), name=ch_type, info=info)

            ndvars_by_type.append((ch_type, ndvar, picks))

        if not ndvars_by_type:
            raise RuntimeError(
                "No displayable channel types (EEG/MEG) found in the raw data "
                "that are also listed in the channels.tsv file."
            )
        self.ndvars_by_type = ndvars_by_type  # [(ch_type, ndvar, picks_array), ...]

        # Compute static neighbor correlations once
        self.nc_static = {}
        for ch_type, ndvar, picks in ndvars_by_type:
            try:
                self.nc_static[ch_type] = neighbor_correlation(ndvar)
            except Exception:
                self.nc_static[ch_type] = None

        self.bad_channels: set[str] = set()
        self.saved = True
        self.callbacks.register_key('bad_chs_change')
        if initial_bad:
            self.bad_channels = set(initial_bad)

    @property
    def all_channel_names(self) -> set[str]:
        names = set()
        for ch_type, ndvar, picks in self.ndvars_by_type:
            names.update(ndvar.sensor.names)
        return names

    def set_bad_channels(self, names):
        self.bad_channels = set(names)
        self.callbacks.callback('bad_chs_change')

    def toggle_bad(self, name: str) -> frozenset:
        new_bad = self.bad_channels.copy()
        if name in new_bad:
            new_bad.remove(name)
        else:
            new_bad.add(name)
        return frozenset(new_bad)

    def compute_nc_dynamic(self, ch_type: str, ndvar: NDVar) -> NDVar | None:
        """NC recomputed with bad channels omitted, for a smooth interpolated map."""
        static = self.nc_static.get(ch_type)
        bad = self.bad_channels
        if static is None:
            return None
        good = [n for n in ndvar.sensor.names if n not in bad]
        if len(good) == len(ndvar.sensor) or len(good) < 2:
            return static
        try:
            return neighbor_correlation(ndvar.sub(sensor=good))
        except Exception:
            return static

    def save(self):
        """Write current bad-channel status back to the channels.tsv file."""
        df = self._channels_df.copy()
        df['status'] = df['name'].apply(
            lambda n: 'bad' if n in self.bad_channels else 'good'
        )
        df.to_csv(self.path, sep='\t', index=False)


class Model(FileModel):

    def __init__(self, doc: Document):
        FileModel.__init__(self, doc)

    def toggle_bad(self, channel_name: str):
        old_bad = frozenset(self.doc.bad_channels)
        new_bad = self.doc.toggle_bad(channel_name)
        self.history.do(ChangeAction(f"Toggle {channel_name}", old_bad, new_bad))

    def set_bad_channels(self, names):
        old_bad = frozenset(self.doc.bad_channels)
        new_bad = frozenset(names)
        if old_bad == new_bad:
            return
        self.history.do(ChangeAction("Set bad channels", old_bad, new_bad))

    def clear(self):
        old_bad = frozenset(self.doc.bad_channels)
        if old_bad:
            self.history.do(ChangeAction("Clear bad channels", old_bad, frozenset()))


class Frame(NavigableFrame, FileFrame):
    """GUI for selecting bad channels in continuous M/EEG recordings.

    * Click on a line in the butterfly plot to toggle that channel as bad.
    * Click on a sensor dot in a topomap to toggle that channel as bad.
    * Bad channels are shown as red dotted lines in butterfly plots and
      as red ``x`` marks in topomaps.
    * The cursor topomap updates in real time as you move the mouse.
    * The "NC (clean)" topomap recomputes neighbor correlation after each
      bad-channel change.

    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    left        scroll backward one window
    right       scroll forward one window
    alt+left    jump to beginning
    alt+right   jump to end
    t           enlarge the topomap under the pointer (with channel names)
    =========== ============================================================
    """

    _doc_name = 'channel selection'
    _name = 'SelectChannels'
    _title = 'Select Channels'
    _wildcard = "BIDS channels file (*_channels.tsv)|*_channels.tsv"

    def __init__(
            self,
            model: Model,
            parent: wx.Frame = None,
            pos: tuple[int, int] = None,
            size: tuple[int, int] = None,
    ):
        FileFrame.__init__(self, parent, pos, size, model)

        raw = self.doc.raw
        sfreq = raw.info['sfreq']
        self._sfreq = sfreq

        # Time-navigation state
        self.window_size = self.config.ReadFloat('window_size', DEFAULT_WINDOW)
        self.t_start = 0.0
        self._window_samples = int(self.window_size * sfreq)

        # Display reduction state (configured via the Layout dialog)
        self._envelope = self.config.ReadBool('Display/envelope', True)
        # decim_auto: pick the factor to draw ~1 sample per horizontal pixel
        self._decim_auto = self.config.ReadBool('Display/decim_auto', True)
        self._decim = max(1, self.config.ReadInt('Display/decim', self.doc.decim))

        # VLim state: stored in SI units; auto=True recomputes from data each window
        self._auto_vlim = self.config.ReadBool('VLim/auto', False)
        self._type_display_vlims: dict[str, float] = {}
        for ch_type, _, _ in self.doc.ndvars_by_type:
            saved = self.config.ReadFloat(f'VLim/vlim_{ch_type}', -1.0)
            self._type_display_vlims[ch_type] = saved if saved > 0 else CH_TYPE_DEFAULT_VLIM_SI[ch_type]
        self._max_t_start = max(0.0, (raw.n_times - self._window_samples) / sfreq)

        # Plot handles (populated by _plot)
        self._butterfly_axes: list = []
        self._butterfly_lc: dict[str, LineCollection] = {}  # ch_type → LineCollection
        self._butterfly_names: dict[str, list] = {}  # ch_type → [ch_name, ...]
        self._cursor_topos: list[AxTopomap] = []
        self._static_nc_topos: list[AxTopomap] = []
        self._dynamic_nc_topos: list[AxTopomap] = []
        self._topo_axes: list = []
        self._cursor_cbars: list = []        # Colorbar per cursor topo (rescales with cursor)
        self._cursor_cbar_axes: list = []    # their axes, for blitted redraws
        self._events_ax = None
        self._event_artists: list = []   # handles for current event markers
        self._cursor_t: float | None = None

        # Canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.set_facecolor('white')
        self.figure.subplots_adjust(0, 0, 1, 1, 0, 0)

        # Scrollbar (horizontal, below canvas)
        self.scrollbar = wx.ScrollBar(self, style=wx.SB_HORIZONTAL)
        self.scrollbar.Bind(wx.EVT_SCROLL, self.OnScroll)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.scrollbar, 0, wx.EXPAND)
        self.SetSizer(sizer)

        # Toolbar
        tb = self.InitToolbar(can_open=False)
        tb.AddSeparator()
        self.AddNavigationButtons(tb)
        tb.AddSeparator()

        btn = wx.Button(tb, wx.ID_ANY, "Bad Channels")
        btn.Bind(wx.EVT_BUTTON, self.OnSetBadChannels)
        tb.AddControl(btn)

        btn = wx.Button(tb, wx.ID_ANY, "Clear")
        btn.SetHelpText("Remove all bad-channel marks")
        btn.Bind(wx.EVT_BUTTON, self.OnClearBadChannels)
        tb.AddControl(btn)

        # Event coloring dropdown
        self._events_colorby: str | None = None
        self._events_colorby_choice = None
        if self.doc.events is not None:
            factor_cols = [k for k, v in self.doc.events.items() if isinstance(v, Factor)]
            if factor_cols:
                self._events_colorby = factor_cols[0]
                tb.AddSeparator()
                tb.AddControl(wx.StaticText(tb, label="Color events by:"))
                choice = wx.Choice(tb, choices=factor_cols)
                choice.SetSelection(0)
                choice.Bind(wx.EVT_CHOICE, self.OnColorByChoice)
                self._events_colorby_choice = choice
                tb.AddControl(choice)

        btn = wx.Button(tb, wx.ID_ANY, "VLim")
        btn.SetHelpText("Set y-axis limits for butterfly plots")
        btn.Bind(wx.EVT_BUTTON, self.OnSetVLim)
        tb.AddControl(btn)

        tb.AddStretchableSpace()
        self.InitToolbarTail(tb)
        tb.Realize()
        self.CreateStatusBar()

        # Subscribe to document changes
        self.doc.callbacks.subscribe('bad_chs_change', self._update_bad_channels)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        self.canvas.mpl_connect('motion_notify_event', self.OnPointerMotion)
        # Re-layout and re-capture the blit background after a resize (the stored
        # background and the square-topo layout both depend on the figure size).
        self._resize_pending = False
        self.canvas.Bind(wx.EVT_SIZE, self.OnCanvasResize)

        self._update_scrollbar()
        self._plot()
        self.UpdateTitle()

    # --- Navigation ---

    def CanBackward(self) -> bool:
        return bool(self.t_start > 0.0)

    def CanDown(self) -> bool:
        return False

    def CanForward(self) -> bool:
        return bool(self.t_start < self._max_t_start)

    def CanUp(self) -> bool:
        return False

    def OnBackward(self, event):
        self.SetWindowStart(self.t_start - self.window_size)

    def OnForward(self, event):
        self.SetWindowStart(self.t_start + self.window_size)

    def OnScroll(self, event):
        pos = self.scrollbar.GetThumbPosition()
        self.SetWindowStart(pos / self._sfreq, _update_scrollbar=False)

    def SetWindowStart(self, t: float, _update_scrollbar: bool = True):
        t = float(np.clip(t, 0.0, self._max_t_start))
        if t == self.t_start:
            return
        self.t_start = t
        self._cursor_t = None
        self._update_window()
        if _update_scrollbar:
            self._update_scrollbar()

    def _update_scrollbar(self):
        pos = int(self.t_start * self._sfreq)
        n = self.doc.raw.n_times
        self.scrollbar.SetScrollbar(pos, self._window_samples, n, self._window_samples)

    def OnCanvasResize(self, event):
        # Let matplotlib resize the figure first, then re-plot once the new size
        # has propagated. A full re-layout (rather than a blitted redraw) is
        # required because the stale blit background causes jumbled artifacts.
        event.Skip()
        if not self._resize_pending:
            self._resize_pending = True
            wx.CallAfter(self._do_resize)

    def _do_resize(self):
        self._resize_pending = False
        if self:  # window may have been destroyed before the deferred call
            self._plot()

    # --- Plotting ---

    def _axis_width_px(self) -> int:
        """Width of the butterfly axes in pixels (they span 95% of the figure)."""
        return max(1, int(self.figure.get_figwidth() * self.figure.dpi * 0.95))

    def _effective_decim(self) -> int:
        """Decimation factor actually applied to the display.

        In ``decim_auto`` mode this targets ~1 sample per horizontal pixel for
        the current window (``sfreq * window_size / width_px``); otherwise it is
        the user-specified :attr:`_decim`.
        """
        if self._decim_auto:
            return max(1, round(self._sfreq * self.window_size / self._axis_width_px()))
        return self._decim

    def _window_lines(
            self,
            picks: np.ndarray,
            scale: float,
            start: int,
            stop: int,
            times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Window data prepared for display: optional decimation, then envelope.

        Parameters
        ----------
        picks
            Channel indices into ``raw``.
        scale
            Multiplier converting SI units to display units.
        start
            First sample of the window.
        stop
            Sample after the last sample of the window.
        times
            Time axis for ``raw[:, start:stop]``.
        """
        d = self.doc.raw[picks, start:stop][0]
        decim = self._effective_decim()
        if decim > 1:
            d = d[:, ::decim]
            times = times[::decim]
        d = d * scale
        if self._envelope:
            return _minmax_envelope(times, d, self._axis_width_px())
        return times, d

    def _line_styles(self, ch_type: str) -> tuple[list, list]:
        """Per-channel ``(colors, linestyles)`` for a butterfly LineCollection."""
        bad = self.doc.bad_channels
        type_color = CH_TYPE_COLORS.get(ch_type, 'k')
        colors, linestyles = [], []
        for ch_name in self._butterfly_names[ch_type]:
            is_bad = ch_name in bad
            colors.append('red' if is_bad else type_color)
            linestyles.append(':' if is_bad else '-')
        return colors, linestyles

    def _plot(self):
        """Create the initial figure layout with butterfly, events, and topo axes."""
        self.figure.clf()
        doc = self.doc
        raw = doc.raw
        sfreq = self._sfreq
        n_types = len(doc.ndvars_by_type)

        has_events = doc.events is not None
        topo_h_frac = 0.22
        events_h_frac = 0.07 if has_events else 0.0
        bf_h_frac = 1.0 - topo_h_frac - events_h_frac
        bf_per_type = bf_h_frac / n_types

        start = int(self.t_start * sfreq)
        stop = min(start + self._window_samples, raw.n_times)
        times = raw.times[start:stop]
        t_end = self.t_start + self.window_size

        # --- Butterfly axes (one per channel type, stacked top-to-bottom) ---
        self._butterfly_axes = []
        self._butterfly_lc = {}
        self._butterfly_names = {}

        for i_type, (ch_type, ndvar, picks) in enumerate(doc.ndvars_by_type):
            bottom = events_h_frac + topo_h_frac + (n_types - 1 - i_type) * bf_per_type
            ax = self.figure.add_axes(
                (0.05, bottom, 0.95, bf_per_type - 0.005),
                frameon=True,
            )
            display_unit, scale = ch_type_scale(ch_type)
            ax.set_xlim(self.t_start, t_end)
            ax.set_ylabel(f"{display_unit or ch_type}", fontsize=8)
            show_xticks = (i_type == n_types - 1)
            ax.tick_params(labelbottom=show_xticks, labelsize=7)
            ax.ch_type = ch_type
            ax.i_type = i_type
            self._butterfly_names[ch_type] = list(ndvar.sensor.names)

            # Fetch, decimate, scale and (optionally) envelope the window data
            t_env, display_data = self._window_lines(picks, scale, start, stop, times)

            # Determine y-axis limits
            if self._auto_vlim:
                abs_max = float(np.percentile(np.abs(display_data), 99))
                vlim_display = abs_max if abs_max > 0 else 1.0
            else:
                vlim_display = self._type_display_vlims[ch_type] * scale
            ax.set_ylim(-vlim_display, vlim_display)

            segments = [np.column_stack([t_env, display_data[j]]) for j in range(display_data.shape[0])]
            colors, linestyles = self._line_styles(ch_type)
            lc = LineCollection(segments, linewidths=0.4, colors=colors, linestyles=linestyles, antialiased=False)
            ax.add_collection(lc)

            self._butterfly_axes.append(ax)
            self._butterfly_lc[ch_type] = lc

        if self._butterfly_axes:
            self._butterfly_axes[-1].set_xlabel("Time (s)", fontsize=8)

        # --- Events axis ---
        self._events_ax = None
        self._event_artists = []
        if has_events:
            bottom = topo_h_frac
            events_ax = self.figure.add_axes(
                (0.05, bottom, 0.95, events_h_frac - 0.005),
                frameon=True,
            )
            events_ax.set_ylabel('Events')
            events_ax.set_xlim(self.t_start, t_end)
            events_ax.set_ylim(0, 1)
            events_ax.set_yticks([])
            events_ax.tick_params(labelbottom=False, bottom=False)
            self._events_ax = events_ax
            self._draw_events()

        # --- Topo row: square (1:1) topomaps. The dynamic cursor maps are
        #     left-aligned, the neighbor-correlation (raw + clean) maps are
        #     right-aligned. Each channel type contributes one cursor map on the
        #     left and an NC pair (sharing one colorbar) on the right. ---
        fig_w = self.figure.get_figwidth()
        fig_h = self.figure.get_figheight()
        cbar_w = 0.006       # colorbar axis width (figure fraction)
        cbar_pad = 0.004     # gap between a map and its colorbar
        cbar_label_w = 0.045  # space right of a colorbar for its tick/unit labels
        nc_inner_gap = 0.008  # gap between the NC raw and NC clean maps of a type
        group_gap = 0.02     # gap between channel-type groups
        middle_gap = 0.03    # minimum gap between the left and right groups

        # Topomaps are square; their height is the constraining dimension, but
        # cap the width so the left and right groups never overlap.
        topo_h_max = topo_h_frac * 0.85
        square_w = topo_h_max * fig_h / fig_w
        # 3 maps per type span the width (1 cursor + 2 NC); the rest is overhead
        overhead = (
            n_types * (cbar_pad + cbar_w + cbar_label_w)  # left: one colorbar per cursor map
            + n_types * (nc_inner_gap + cbar_pad + cbar_w + cbar_label_w)  # right: NC pair + colorbar
            + 2 * (n_types - 1) * group_gap
            + middle_gap
        )
        topo_w = min(square_w, (1.0 - overhead) / (3 * n_types))
        topo_h = topo_w * fig_w / fig_h  # keep square in physical space
        topo_bottom = (topo_h_frac - topo_h) / 2

        # Left-aligned cursor maps: (map_x, cbar_x) per channel type
        cursor_map_x, cursor_cbar_x = [], []
        x = 0.0
        for _ in range(n_types):
            cursor_map_x.append(x)
            cursor_cbar_x.append(x + topo_w + cbar_pad)
            x += topo_w + cbar_pad + cbar_w + cbar_label_w + group_gap

        # Right-aligned NC maps: (raw_x, clean_x, cbar_x) per channel type. The
        # unit reserves cbar_label_w right of the colorbar so tick/unit labels
        # are not clipped at the figure edge (or by the next group).
        nc_unit_w = 2 * topo_w + nc_inner_gap + cbar_pad + cbar_w + cbar_label_w
        nc_total_w = n_types * nc_unit_w + (n_types - 1) * group_gap
        nc_raw_x, nc_clean_x, nc_cbar_x = [], [], []
        x = 1.0 - nc_total_w
        for _ in range(n_types):
            nc_raw_x.append(x)
            nc_clean_x.append(x + topo_w + nc_inner_gap)
            nc_cbar_x.append(x + 2 * topo_w + nc_inner_gap + cbar_pad)
            x += nc_unit_w + group_gap

        self._cursor_topos = []
        self._static_nc_topos = []
        self._dynamic_nc_topos = []
        self._topo_axes = []
        self._cursor_cbars = []
        self._cursor_cbar_axes = []

        topo_groups = [
            ('Cursor', self._cursor_topos),
            ('Neighbor corr raw', self._static_nc_topos),
            ('Neighbor corr clean', self._dynamic_nc_topos),
        ]

        for i_type, (ch_type, ndvar, picks) in enumerate(doc.ndvars_by_type):
            sensor = ndvar.sensor
            display_unit, scale = ch_type_scale(ch_type)
            nc_raw_topo = None  # provides the shared color scale/colorbar for the NC pair
            for j_group, (label, topo_list) in enumerate(topo_groups):
                if j_group == 0:
                    map_x = cursor_map_x[i_type]
                elif j_group == 1:
                    map_x = nc_raw_x[i_type]
                else:
                    map_x = nc_clean_x[i_type]
                ax = self.figure.add_axes(
                    (map_x, topo_bottom, topo_w, topo_h),
                )
                ax.ch_type = ch_type
                ax.topo_group = j_group
                self._topo_axes.append(ax)

                # Initial data for this topo; cursor in display units, NC unitless
                if j_group == 0:
                    d = raw[picks, 0:1][0][:, 0] * scale
                    topo_ndvar = NDVar(d, (sensor,), name=ch_type, info=ndvar.info)
                    cbar_label = display_unit or ch_type
                    interpolation = 'linear'
                    vlims = {ndvar.info.get('meas'): (-vlim_display, vlim_display)}
                else:
                    # NC raw uses the static map; NC clean omits bad channels
                    if j_group == 1:
                        topo_ndvar = doc.nc_static.get(ch_type)
                    else:
                        topo_ndvar = doc.compute_nc_dynamic(ch_type, ndvar)
                    if topo_ndvar is None:
                        topo_ndvar = NDVar(np.zeros(len(sensor)), (sensor,), name=ch_type)
                    cbar_label = 'r'
                    interpolation = 'nearest'
                    vlims = {'r': (-1, 1)}

                layers = AxisData([DataLayer(topo_ndvar, PlotType.IMAGE)])
                p = AxTopomap(ax, layers, vlims=vlims, interpolation=interpolation, clip='even')
                ax.text(0.5, 0.0, f"{label} ({ch_type})", transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=7)

                if j_group == 0:
                    # Cursor topo: own colorbar that rescales with the cursor
                    cbar_ax = self.figure.add_axes(
                        (cursor_cbar_x[i_type], topo_bottom, cbar_w, topo_h),
                    )
                    cbar = self._add_topo_colorbar(p, cbar_ax, cbar_label)
                    self._cursor_cbars.append(cbar)
                    self._cursor_cbar_axes.append(cbar_ax)
                elif j_group == 1:
                    # NC raw defines the shared color scale; colorbar drawn with the clean topo
                    nc_raw_topo = p
                else:
                    # NC clean: share the raw NC color scale and a single colorbar
                    cbar_ax = self.figure.add_axes(
                        (nc_cbar_x[i_type], topo_bottom, cbar_w, topo_h),
                    )
                    self._add_topo_colorbar(nc_raw_topo, cbar_ax, cbar_label)

                # Mark bad channels with red ×
                if doc.bad_channels:
                    self._mark_bad_on_topo(p, sensor, doc.bad_channels)

                topo_list.append(p)

        self.canvas.store_canvas()
        self.canvas.draw()

    def _add_topo_colorbar(self, topo_plot: AxTopomap, cbar_ax, label: str):
        """Attach a thin vertical colorbar (with unit label) to a topomap."""
        cbar = self.figure.colorbar(topo_plot.plots[0].im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=6)
        cbar.set_label(label, fontsize=7)
        cbar.outline.set_linewidth(0.5)
        self._set_cbar_ticks(cbar)
        return cbar

    def _set_cbar_ticks(self, cbar):
        """Label only the min, center, and max of the colorbar range."""
        vmin, vmax = cbar.mappable.get_clim()
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])

    def _mark_bad_on_topo(self, topo_plot: AxTopomap, sensor, bad: set[str]):
        """Add red × marks at bad channel positions on a topomap."""
        bad_idx = [i for i, n in enumerate(sensor.names) if n in bad]
        if bad_idx:
            topo_plot.sensors.mark_sensors(bad_idx, color='red', marker='x', size=30, zorder=10)

    def _update_window(self):
        """Update butterfly and events display for the current time window."""
        raw = self.doc.raw
        sfreq = self._sfreq
        t_start = self.t_start
        t_end = t_start + self.window_size

        start = int(t_start * sfreq)
        stop = min(start + self._window_samples, raw.n_times)
        times = raw.times[start:stop]

        for i_type, (ch_type, ndvar, picks) in enumerate(self.doc.ndvars_by_type):
            ax = self._butterfly_axes[i_type]
            ax.set_xlim(t_start, t_end)
            _, scale = ch_type_scale(ch_type)
            t_env, display_data = self._window_lines(picks, scale, start, stop, times)
            segments = [np.column_stack([t_env, display_data[j]]) for j in range(display_data.shape[0])]
            self._butterfly_lc[ch_type].set_segments(segments)
            if self._auto_vlim:
                abs_max = float(np.percentile(np.abs(display_data), 99))
                vlim_display = abs_max if abs_max > 0 else 1.0
                ax.set_ylim(-vlim_display, vlim_display)

        if self._events_ax is not None:
            self._events_ax.set_xlim(t_start, t_end)
            self._draw_events()

        # Full redraw + re-capture the blit background: the new window changed
        # the butterfly/events axes, so a stale background would otherwise be
        # restored (showing the previous page) on the next cursor-topo blit.
        self.canvas.draw()
        self.canvas.store_canvas()

    def _draw_events(self):
        """Redraw event markers for the current time window."""
        ax = self._events_ax
        if ax is None or self.doc.events is None:
            return

        # Remove previous event markers
        for h in self._event_artists:
            try:
                h.remove()
            except ValueError:
                pass
        self._event_artists = []

        events = self.doc.events
        t_col = self.doc._t_column
        if t_col is None or t_col not in events:
            return

        t_start = self.t_start
        t_end = t_start + self.window_size
        times = events[t_col].x
        has_duration = 'duration' in events
        colorby = self._events_colorby

        # Build per-cell color map
        cell_color: dict[str, object] = {}
        if colorby and colorby in events and isinstance(events[colorby], Factor):
            factor = events[colorby]
            for k, cell in enumerate(factor.cells):
                cell_color[cell] = _EVENT_COLORS[k % len(_EVENT_COLORS)]

        for i in range(events.n_cases):
            t = float(times[i])
            dur = float(events['duration'].x[i]) if has_duration else 0.0
            # Skip if entirely outside window
            if t + dur < t_start or t > t_end:
                continue
            color = cell_color.get(events[colorby][i] if colorby else '', (0.5, 0.5, 0.5)) if cell_color else (0.5, 0.5, 0.5)
            if has_duration and dur > 0:
                h = ax.axvspan(t, t + dur, alpha=0.4, color=color, linewidth=0)
            else:
                h = ax.axvline(t, color=color, alpha=0.7, linewidth=0.8)
            self._event_artists.append(h)

    def _update_bad_channels(self):
        """Called by Document when bad channels change; updates line styles and topos."""
        bad = self.doc.bad_channels

        # Update butterfly line colors and styles
        for ch_type, ndvar, picks in self.doc.ndvars_by_type:
            lc = self._butterfly_lc.get(ch_type)
            if lc is None:
                continue
            colors, linestyles = self._line_styles(ch_type)
            lc.set_color(colors)
            lc.set_linestyle(linestyles)

        # Update sensor marks on all topos
        for p, (ch_type, ndvar, picks) in zip(
                self._cursor_topos + self._static_nc_topos + self._dynamic_nc_topos,
                list(self.doc.ndvars_by_type) * 3,
        ):
            sensor = ndvar.sensor
            p.sensors.mark_sensors(None)  # clear
            if bad:
                self._mark_bad_on_topo(p, sensor, bad)

        # Recompute and update dynamic NC topos
        for i_type, (ch_type, ndvar, picks) in enumerate(self.doc.ndvars_by_type):
            nc_dyn = self.doc.compute_nc_dynamic(ch_type, ndvar)
            if nc_dyn is not None and i_type < len(self._dynamic_nc_topos):
                self._dynamic_nc_topos[i_type].set_data([nc_dyn])

        self.canvas.draw()
        # Refresh the blit background so cursor-topo / window updates stay valid
        self.canvas.store_canvas()

    def _update_cursor_topo(self, t: float):
        """Update cursor topomaps to show data at time t."""
        if t == self._cursor_t:
            return
        self._cursor_t = t
        raw = self.doc.raw
        t_idx = int(np.clip(int(t * self._sfreq), 0, raw.n_times - 1))

        redraw_axes = []
        for i_type, (ch_type, ndvar, picks) in enumerate(self.doc.ndvars_by_type):
            if i_type >= len(self._cursor_topos):
                break
            _, scale = ch_type_scale(ch_type)
            d = raw[picks, t_idx:t_idx + 1][0][:, 0] * scale
            cursor_ndvar = NDVar(d, (ndvar.sensor,), name=ch_type, info=ndvar.info)
            self._cursor_topos[i_type].set_data([cursor_ndvar])
            redraw_axes.append(self._topo_axes[i_type * 3])

        if redraw_axes:
            self.canvas.redraw(redraw_axes)

    def SetVLim(self, vlim_si_dict: dict):
        """Apply y-axis limits (SI units) to butterfly axes."""
        self._type_display_vlims.update(vlim_si_dict)
        for i_type, (ch_type, _, _) in enumerate(self.doc.ndvars_by_type):
            if ch_type not in vlim_si_dict:
                continue
            _, scale = ch_type_scale(ch_type)
            vlim_display = vlim_si_dict[ch_type] * scale
            self._butterfly_axes[i_type].set_ylim(-vlim_display, vlim_display)
        self.canvas.draw()
        self.canvas.store_canvas()

    # --- Event handlers ---

    def OnSetVLim(self, event):
        type_scales = {ch_type: ch_type_scale(ch_type) for ch_type, _, _ in self.doc.ndvars_by_type}
        dlg = VLimDialog(self, type_scales, self._type_display_vlims, self._auto_vlim, CH_TYPE_DEFAULT_VLIM_SI)
        if dlg.ShowModal() == wx.ID_OK:
            self._auto_vlim = dlg.GetAuto()
            self._type_display_vlims = dlg.GetVLims()
            self.config.WriteBool('VLim/auto', self._auto_vlim)
            for ch_type, vlim_si in self._type_display_vlims.items():
                self.config.WriteFloat(f'VLim/vlim_{ch_type}', vlim_si)
            self.config.Flush()
            if self._auto_vlim:
                self._update_window()  # recompute from current window data
            else:
                self.SetVLim(self._type_display_vlims)
        dlg.Destroy()

    def OnCanvasClick(self, event):
        if event.button != 1 or not event.inaxes:
            return
        ax = event.inaxes
        ch_name = None

        if hasattr(ax, 'topo_group'):
            # Click in a topo axis — find nearest sensor
            ch_type = getattr(ax, 'ch_type', None)
            if ch_type is None:
                return
            for ch_t, ndvar, picks in self.doc.ndvars_by_type:
                if ch_t != ch_type:
                    continue
                from ..plot._sensors import SENSORMAP_FRAME
                sensor = ndvar.sensor
                locs = sensor.get_locs_2d('default', frame=SENSORMAP_FRAME)
                dists = np.hypot(locs[:, 0] - event.xdata, locs[:, 1] - event.ydata)
                i_nearest = int(np.argmin(dists))
                if dists[i_nearest] < 0.12:
                    ch_name = sensor.names[i_nearest]
                break

        elif hasattr(ax, 'ch_type') and not hasattr(ax, 'topo_group') and not getattr(ax, 'is_events', False):
            # Click in butterfly axis — find nearest line at the clicked time point
            ch_type = ax.ch_type
            for ch_t, ndvar, picks in self.doc.ndvars_by_type:
                if ch_t != ch_type:
                    continue
                raw = self.doc.raw
                t_idx = int(np.clip(int(event.xdata * self._sfreq), 0, raw.n_times - 1))
                d = raw[picks, t_idx:t_idx + 1][0][:, 0]
                _, scale = ch_type_scale(ch_type)
                d_display = d * scale
                i_nearest = int(np.argmin(np.abs(d_display - event.ydata)))
                ch_name = ndvar.sensor.names[i_nearest]
                break

        if ch_name is not None:
            self.model.toggle_bad(ch_name)

    def OnCanvasKey(self, event):
        key = event.key
        if key == 'right' and self.CanForward():
            self.OnForward(None)
        elif key == 'left' and self.CanBackward():
            self.OnBackward(None)
        elif key == 'alt+right':
            self.SetWindowStart(self._max_t_start)
        elif key == 'alt+left':
            self.SetWindowStart(0.0)
        elif key in ('t', 'T') and event.inaxes is not None and hasattr(event.inaxes, 'topo_group'):
            self._plot_topomap_popup(event.inaxes)

    def _plot_topomap_popup(self, ax):
        """Pop up an enlarged topomap (with channel names) for the topo under the cursor."""
        ch_type = getattr(ax, 'ch_type', None)
        group = ax.topo_group
        entry = next(((ct, nd, pk) for ct, nd, pk in self.doc.ndvars_by_type if ct == ch_type), None)
        if entry is None:
            return
        ch_type, ndvar, picks = entry
        if group == 0:
            t = self._cursor_t if self._cursor_t is not None else self.t_start
            raw = self.doc.raw
            t_idx = int(np.clip(int(t * self._sfreq), 0, raw.n_times - 1))
            _, scale = ch_type_scale(ch_type)
            d = raw[picks, t_idx:t_idx + 1][0][:, 0] * scale
            data = NDVar(d, (ndvar.sensor,), name=ch_type, info=ndvar.info)
            title = f"Cursor ({ch_type}) t={t:.3f} s"
        elif group == 1:
            data = self.doc.nc_static.get(ch_type)
            title = f"Neighbor corr raw ({ch_type})"
        else:
            data = self.doc.compute_nc_dynamic(ch_type, ndvar)
            title = f"Neighbor corr clean ({ch_type})"
        if data is None:
            return
        plot.Topomap(data, sensorlabels='name', axw=9, title=title)

    def OnPointerMotion(self, event):
        if not event.inaxes:
            return
        ax = event.inaxes
        if hasattr(ax, 'ch_type') and not hasattr(ax, 'topo_group') and not getattr(ax, 'is_events', False):
            t = event.xdata
            if t is not None and self.t_start <= t <= self.t_start + self.window_size:
                self._update_cursor_topo(t)
            unit = ch_type_scale(ax.ch_type)[0] or ''
            self.SetStatusText(f"t = {t:.3f} s,  y = {event.ydata:.2f} {unit}")
        elif hasattr(ax, 'topo_group'):
            self.SetStatusText("Click a sensor to toggle bad")

    def OnColorByChoice(self, event):
        choice = event.GetEventObject()
        factor_cols = [k for k, v in self.doc.events.items() if isinstance(v, Factor)]
        self._events_colorby = factor_cols[choice.GetSelection()]
        self._draw_events()
        self.canvas.draw()

    def OnSetBadChannels(self, event):
        default = ', '.join(sorted(self.doc.bad_channels))
        dlg = wx.TextEntryDialog(
            self,
            "Enter bad channel names, comma-separated:",
            "Set Bad Channels",
            default,
        )
        if dlg.ShowModal() == wx.ID_OK:
            names = [s.strip() for s in dlg.GetValue().split(',') if s.strip()]
            unknown = [n for n in names if n not in self.doc.all_channel_names]
            if unknown:
                wx.MessageBox(
                    f"Unknown channels: {', '.join(unknown)}",
                    "Invalid Entry",
                    wx.OK | wx.ICON_ERROR,
                )
            else:
                self.model.set_bad_channels(names)
        dlg.Destroy()

    def OnClearBadChannels(self, event):
        self.model.clear()

    def OnSetLayout(self, event):
        dlg = LayoutDialog(self, self.window_size, self._envelope, self._decim_auto, self._decim, self._sfreq, self._axis_width_px())
        if dlg.ShowModal() == wx.ID_OK:
            self._envelope = dlg.envelope
            self._decim_auto = dlg.decim_auto
            self._decim = dlg.decim
            self.config.WriteBool('Display/envelope', self._envelope)
            self.config.WriteBool('Display/decim_auto', self._decim_auto)
            self.config.WriteInt('Display/decim', self._decim)
            if dlg.window_size != self.window_size:
                self.window_size = dlg.window_size
                self._window_samples = int(self.window_size * self._sfreq)
                self._max_t_start = max(0.0, (self.doc.raw.n_times - self._window_samples) / self._sfreq)
                self.t_start = float(np.clip(self.t_start, 0.0, self._max_t_start))
                self.config.WriteFloat('window_size', self.window_size)
                self._update_scrollbar()
                self._plot()  # sample count and x-limits change: full re-layout
            else:
                self._update_window()  # only the displayed point density changes
            self.config.Flush()
        dlg.Destroy()

    def OnUpdateUISetLayout(self, event):
        event.Enable(True)

    def OnClose(self, event):
        if super().OnClose(event):
            self.doc.callbacks.remove('bad_chs_change', self._update_bad_channels)
            self.config.WriteFloat('window_size', self.window_size)
            self.config.WriteBool('Display/envelope', self._envelope)
            self.config.WriteBool('Display/decim_auto', self._decim_auto)
            self.config.WriteInt('Display/decim', self._decim)
            self.config.Flush()


class LayoutDialog(EelbrainDialog):
    "Display/layout options for the select-channels GUI"

    def __init__(
            self,
            parent: wx.Window,
            window_size: float,
            envelope: bool,
            decim_auto: bool,
            decim: int,
            sfreq: float,
            width_px: int,
    ):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Select-Channels Layout")
        self.window_size: float | None = None
        self.envelope: bool | None = None
        self.decim_auto: bool | None = None
        self.decim: int | None = None
        self._sfreq = sfreq
        self._width_px = width_px

        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, wx.ID_ANY, "Window length (s):"))
        validator = REValidator(POS_FLOAT_PATTERN, "Invalid window length: {value}. Need a number > 0.", False)
        self.window_ctrl = wx.TextCtrl(self, wx.ID_ANY, f'{window_size:g}', validator=validator)
        self.window_ctrl.Bind(wx.EVT_TEXT, self.OnRefresh)
        sizer.Add(self.window_ctrl)

        sizer.AddSpacer(8)
        self.envelope_ctrl = wx.CheckBox(self, wx.ID_ANY, "Reduce to screen resolution (min/max)")
        self.envelope_ctrl.SetValue(envelope)
        self.envelope_ctrl.SetToolTip("Draw the per-pixel minimum and maximum (fast). Uncheck to draw every channel sample, which shows whether a peak comes from one channel or many.")
        sizer.Add(self.envelope_ctrl)

        sizer.AddSpacer(8)
        sizer.Add(wx.StaticText(self, wx.ID_ANY, f"Downsample data; original sampling rate: {sfreq:g} Hz"))
        self.auto_ctrl = wx.CheckBox(self, wx.ID_ANY, "Fit to display (~1 sample per pixel)")
        self.auto_ctrl.SetValue(decim_auto)
        self.auto_ctrl.SetToolTip("Pick the decimation factor automatically so the displayed window has about one sample per horizontal pixel.")
        self.auto_ctrl.Bind(wx.EVT_CHECKBOX, self.OnRefresh)
        sizer.Add(self.auto_ctrl)
        sizer.Add(wx.StaticText(self, wx.ID_ANY, "Decimation factor:"))
        validator = REValidator(INT_PATTERN, "Invalid decimation factor: {value}. Need an integer >= 1.")
        self.decim_ctrl = wx.TextCtrl(self, wx.ID_ANY, str(decim), validator=validator)
        self.decim_ctrl.Bind(wx.EVT_TEXT, self.OnRefresh)
        sizer.Add(self.decim_ctrl)
        self.rate_label = wx.StaticText(self, wx.ID_ANY, "")
        sizer.Add(self.rate_label)

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
        self._refresh()

    def _effective_decim(self) -> int:
        """Decimation factor implied by the current field values."""
        if self.auto_ctrl.GetValue():
            try:
                window = float(self.window_ctrl.GetValue())
            except ValueError:
                return 1
            if window <= 0:
                return 1
            return max(1, round(self._sfreq * window / self._width_px))
        try:
            return max(1, int(self.decim_ctrl.GetValue()))
        except ValueError:
            return 1

    def _refresh(self):
        """Enable/disable the manual field and update the effective-rate label."""
        auto = self.auto_ctrl.GetValue()
        self.decim_ctrl.Enable(not auto)
        decim = self._effective_decim()
        suffix = " (auto)" if auto else ""
        self.rate_label.SetLabel(f"Effective display rate: {self._sfreq / decim:g} Hz (decim {decim}){suffix}")

    def OnRefresh(self, event):
        self._refresh()

    def OnOk(self, event):
        try:
            window_size = float(self.window_ctrl.GetValue())
            decim = int(self.decim_ctrl.GetValue())
        except ValueError:
            wx.MessageBox("Invalid layout values", "Invalid Layout", wx.OK | wx.ICON_ERROR, self)
            return
        if window_size <= 0 or decim < 1:
            wx.MessageBox("Window length must be > 0 and decimation factor >= 1", "Invalid Layout", wx.OK | wx.ICON_ERROR, self)
            return
        self.window_size = window_size
        self.envelope = self.envelope_ctrl.GetValue()
        self.decim_auto = self.auto_ctrl.GetValue()
        self.decim = decim
        event.Skip()
