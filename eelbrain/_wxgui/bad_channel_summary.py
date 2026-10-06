"""Summary window for marking channels found by the ICA bad channel tools as bad

Shared by the pipeline GUI (Bad-Chs in the ICA task: every recording with an ICA) and the
ICA GUI (Find Bad Channels: the one recording on display).
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import numpy as np
import wx

from .._data_obj import NDVar
from .._meeg.ica_bad_channels import ChannelGapResult


# Default minimum share of a channel's variance due to one ICA component for marking the channel as bad (Bad-Chs in the ICA task)
CHANNEL_VARIANCE_DEFAULT = 0.5


# Bad channel evidence of one recording (see PipelineFrame._ica_bad_channel_candidates)
CandidateList = list[tuple[str, int, float]]  # (ch_name, component, variance_fraction) per component loading on a single channel
GapList = list[tuple[str, int, int]]  # (ch_name, n_evidence, n_testable) per channel missing from component maps
FlatList = list[tuple[str, str]]  # (ch_name, ch_type) per flat channel
RecordingResult = tuple[tuple[str, ...], CandidateList, GapList, FlatList]  # (combo, candidates, gaps, flat)
# Summary row: (combo, ch_name, component, variance_fraction, gap, flat); component and variance_fraction are
# None for a channel no component loads on, gap is (n_evidence, n_testable) or None for a channel that is not a gap
SummaryRow = tuple[tuple[str, ...], str, int | None, float | None, tuple[int, int] | None, bool]
# Channels to mark as bad, grouped by recording: [(combo, names), ...]
Additions = list[tuple[tuple[str, ...], list[str]]]


def bad_channel_evidence(
        candidates: Sequence[tuple[int, str, np.ndarray, float]],
        gap_results: Sequence[tuple[NDVar, ChannelGapResult]],
        flat: FlatList,
) -> tuple[CandidateList, GapList, FlatList]:
    """Reduce the findings of the ICA document's bad channel tools to the summary's form

    Parameters
    ----------
    candidates
        Components loading on a single channel (see
        :meth:`~eelbrain._wxgui.select_components.Document.single_channel_components`).
    gap_results
        Channels missing from component maps, per channel type (see
        :meth:`~eelbrain._wxgui.select_components.Document.channel_gaps`).
    flat
        Flat channels (see :meth:`~eelbrain._wxgui.select_components.Document.flat_channels`).
    """
    candidate_list = [(ch_name, component, variance_fraction) for component, ch_name, _, variance_fraction in candidates]
    gaps = [(channel.name, channel.n_evidence, channel.n_testable) for _, result in gap_results for channel in result.channels]
    return candidate_list, gaps, list(flat)


def _channel_summary_rows(results: Sequence[RecordingResult]) -> list[SummaryRow]:
    """One row per channel for the bad channel summary

    Parameters
    ----------
    results
        ``(combo, candidates, gaps, flat)`` per recording (see
        :meth:`PipelineFrame._ica_bad_channel_candidates`). A channel that several
        components load on is listed with the one that explains most of its variance.
        Within a recording, flat channels come first, then channels missing from component
        maps, then the rest by variance fraction, descending.
    """
    rows = []
    for combo, candidates, gaps, flat in results:
        best = {}  # {ch_name: (component, variance_fraction)}
        for ch_name, component, variance_fraction in candidates:
            if ch_name not in best or variance_fraction > best[ch_name][1]:
                best[ch_name] = (component, variance_fraction)
        gap_by_name = {ch_name: (n_evidence, n_testable) for ch_name, n_evidence, n_testable in gaps}
        flat_names = [ch_name for ch_name, _ in flat]  # a list: the order of ties is the channel order
        for ch_name in (*gap_by_name, *flat_names):
            best.setdefault(ch_name, (None, None))
        for ch_name, (component, variance_fraction) in sorted(best.items(), key=lambda item: (item[0] in flat_names, item[0] in gap_by_name, item[1][1] or 0), reverse=True):
            rows.append((combo, ch_name, component, variance_fraction, gap_by_name.get(ch_name), ch_name in flat_names))
    return rows


def _flat_in_every_recording(results: Sequence[RecordingResult]) -> list[str]:
    """EEG channels that are flat in every recording: most likely the reference rather than defective

    Parameters
    ----------
    results
        ``(combo, candidates, gaps, flat)`` per recording (see :func:`_channel_summary_rows`).
    """
    if not results:
        return []
    flat_sets = [{ch_name for ch_name, ch_type in flat if ch_type == 'eeg'} for _, _, _, flat in results]
    return sorted(set.intersection(*flat_sets))


def _bad_channels_above(
        rows: Sequence[SummaryRow],
        threshold: float,
        exclude: Collection[str] = (),
) -> Additions:
    """Channels to mark as bad at ``threshold``, grouped by recording: ``[(combo, names), ...]``

    Parameters
    ----------
    rows
        Summary rows (see :func:`_channel_summary_rows`).
    threshold
        Minimum variance fraction for a channel to count as bad; a channel that is flat or
        missing from component maps counts regardless.
    exclude
        Channels that are never marked, whatever the evidence (e.g. the EEG reference,
        which is flat by design).
    """
    by_combo = {}
    for combo, ch_name, _, variance_fraction, gap, flat in rows:
        if ch_name in exclude:
            continue
        if flat or gap is not None or variance_fraction >= threshold:
            by_combo.setdefault(combo, []).append(ch_name)
    return list(by_combo.items())


class BadChannelSummaryFrame(wx.Frame):
    """Channels that are flat, missing from ICA maps or dominated by one component, with an Apply button to mark them as bad

    A regular window rather than a dialog, so that the user can go back to the host
    window and inspect the ICAs in question before applying.

    Parameters
    ----------
    parent
        Host window (pipeline GUI or ICA GUI).
    results
        ``(combo, candidates, gaps, flat)`` per recording that was analyzed (see
        :func:`_channel_summary_rows`).
    key_fields
        Column labels for the ``combo`` of each recording (empty for a single recording).
    apply
        Called as ``apply(additions, recompute)`` when the user clicks Apply, with the
        channels to mark as bad grouped by recording (``[(combo, names), ...]``) and
        whether to queue the new ICA decompositions right away; the host writes the bad
        channels and deletes the ICAs.
    threshold
        Initial minimum variance fraction for marking a channel as bad.
    """

    def __init__(
            self,
            parent: wx.Window,
            results: Sequence[RecordingResult],
            key_fields: Sequence[str],
            apply: Callable[[Additions, bool], None],
            threshold: float = CHANNEL_VARIANCE_DEFAULT,
    ) -> None:
        super().__init__(parent, title="Bad Channels")
        self._rows = _channel_summary_rows(results)
        self._n_recordings = len(results)
        self._apply = apply

        self._threshold_value = threshold  # last valid entry, kept while the text is not a number

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(wx.StaticText(panel, label="Mark a channel as bad when one component explains at least"), flag=wx.ALIGN_CENTER_VERTICAL)
        self._threshold = wx.SpinCtrlDouble(panel, min=0, max=100, initial=100 * threshold, inc=5)
        self._threshold.SetDigits(0)
        # arrows and typed text; the selection is updated once the control has finished
        # processing the change, so that its text is current when it is read
        self._threshold.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        self._threshold.Bind(wx.EVT_TEXT, self._on_change)
        hbox.Add(self._threshold, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=6)
        hbox.Add(wx.StaticText(panel, label="% of its variance"), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=4)
        vbox.Add(hbox, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=12)
        # wrapped to the window width in _on_size
        self._texts = [wx.StaticText(panel, label="A channel that is flat, or missing from component maps (Gap: components with a gap / components in which the channel could be evaluated), is marked regardless of the threshold. A flat EEG channel can be the reference rather than defective: channels listed under Do not mark are left alone whatever the evidence.")]
        vbox.Add(self._texts[0], flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=12)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(wx.StaticText(panel, label="Do not mark (comma-separated):"), flag=wx.ALIGN_CENTER_VERTICAL)
        self._exclude = wx.TextCtrl(panel, value=', '.join(_flat_in_every_recording(results)))
        self._exclude.SetToolTip("Pre-filled with the EEG channels that are flat in every recording, which is most likely the reference")
        self._exclude.Bind(wx.EVT_TEXT, self._on_change)
        hbox.Add(self._exclude, proportion=1, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=6)
        vbox.Add(hbox, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, border=12)

        self._list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_NONE)
        columns = [*((field.title(), 90) for field in key_fields), ("Channel", 100), ("Flat", 50), ("Gap", 80), ("Component", 90), ("Variance", 80)]
        for i, (label, width) in enumerate(columns):
            self._list.InsertColumn(i, label, width=width)
        for combo, ch_name, component, variance_fraction, gap, flat in self._rows:
            flat_str = 'flat' if flat else ''
            gap_str = '' if gap is None else f"{gap[0]} / {gap[1]}"
            component_str = '' if component is None else f"#{component}"
            variance_str = '' if variance_fraction is None else f"{variance_fraction:.0%}"
            values = (*combo, ch_name, flat_str, gap_str, component_str, variance_str)
            idx = self._list.InsertItem(self._list.GetItemCount(), values[0])
            for col, value in enumerate(values[1:], 1):
                self._list.SetItem(idx, col, value)
        vbox.Add(self._list, proportion=1, flag=wx.EXPAND)

        self._texts.append(wx.StaticText(panel, label="Applying adds the channels to the bad channels of their recordings and deletes those recordings' ICA files, which were estimated with these channels included."))
        vbox.Add(self._texts[1], flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=12)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self._recompute = wx.CheckBox(panel, label="Re-compute the ICA now")
        self._recompute.SetValue(True)
        self._recompute.SetToolTip("Start computing the new ICA decompositions right away; otherwise they need to be computed before component selection can continue")
        hbox.Add(self._recompute, flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.AddStretchSpacer()
        self._apply_btn = wx.Button(panel, label="Apply")
        self._apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
        hbox.Add(self._apply_btn, flag=wx.ALIGN_CENTER_VERTICAL)
        vbox.Add(hbox, flag=wx.EXPAND | wx.ALL, border=12)
        panel.SetSizer(vbox)
        self._panel = panel
        self._labels = [text.GetLabel() for text in self._texts]
        panel.Bind(wx.EVT_SIZE, self._on_size)
        self.CreateStatusBar()
        self.SetSize((640, 500))
        self._update_selection()

    def _on_size(self, event) -> None:
        "Re-wrap the explanatory texts to the window width"
        width = self._panel.GetClientSize().width - 24
        for text, label in zip(self._texts, self._labels):
            text.SetLabel(label)
            text.Wrap(width)
        self._panel.Layout()
        event.Skip()

    def get_threshold(self) -> float:
        "Minimum share of a channel's variance due to one component, as a fraction"
        try:
            self._threshold_value = float(self._threshold.GetTextValue()) / 100
        except ValueError:  # text being edited is not a number
            pass
        return self._threshold_value

    def exclude(self) -> list[str]:
        "Channels listed under Do not mark"
        return [name.strip() for name in self._exclude.GetValue().split(',') if name.strip()]

    def bad_channels(self) -> Additions:
        """Channels to mark as bad with the current settings, grouped by recording: ``[(combo, names), ...]``"""
        return _bad_channels_above(self._rows, self.get_threshold(), self.exclude())

    def _on_change(self, event) -> None:
        wx.CallAfter(self._update_selection)

    def _update_selection(self) -> None:
        "Colour the rows that will be marked and update the status bar and Apply button"
        additions = self.bad_channels()
        selected = {(combo, ch_name) for combo, names in additions for ch_name in names}
        # an explicit colour: wx.NullColour does not clear a colour once one was set (macOS)
        default = self._list.GetTextColour()
        for i, (combo, ch_name, *_) in enumerate(self._rows):
            self._list.SetItemTextColour(i, wx.RED if (combo, ch_name) in selected else default)
        self._list.Refresh()
        n_channels = sum(len(names) for _, names in additions)
        self.SetStatusText(f"{n_channels} bad channels in {len(additions)} of {self._n_recordings} recordings")
        self._apply_btn.Enable(bool(additions))

    def _on_apply(self, event) -> None:
        self._apply(self.bad_channels(), self._recompute.GetValue())
        self.Close()
