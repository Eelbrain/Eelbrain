"""Pipeline supervisor GUI launched by ``eelbrain-gui``."""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import wx

from .. import load
from .._data_obj import Dataset
from .._exceptions import ConfigurationError, DataError
from .._experiment.derivative_cache import ALLOW_PROTECTED_OVERWRITE, JobSpec, ProtectedArtifactError
from .._experiment.epoch_rejection import ChannelModelRejection, ManualRejection
from .._experiment.epochs import PrimaryEpoch
from .._experiment.exceptions import FileMissingError, ICAChannelsChangedError, ICAMissingError
from .._experiment.pathing import MRI_SDIR
from .._experiment.preprocessing import REINDEX_ICA, RawICA, RawSource, ica_input_name, raw_bad_channels_input_name, raw_input_name
from .._utils.mne_utils import is_fake_mri
from .frame import EelbrainFrame
from .select_components import Document as ICADocument
from .utils import StaleICADialog, TracebackDialog

if TYPE_CHECKING:
    from .._experiment.pipeline import Pipeline


def _launch_coreg_subprocess(
        mrisubject: str,
        subjects_dir: str,
        inst: str,
        trans: str | None = None,
        on_close: Callable | None = None,
) -> None:
    """Launch mne.gui.coregistration in a subprocess to avoid Qt/wx event loop conflict."""
    kwargs = f'subject={mrisubject!r}, {subjects_dir=}, {inst=}, block=True'
    if trans is not None:
        kwargs += f', {trans=}'
    proc = subprocess.Popen([
        sys.executable, '-c',
        f'import mne; mne.gui.coregistration({kwargs})',
    ])
    if on_close is not None:
        threading.Thread(target=lambda: (proc.wait(), on_close()), daemon=True).start()


class _AbortRequested(Exception):
    """Raised when the user clicks Abort in the stale-ICA dialog."""


_USER_ERROR_TYPES = (ConfigurationError, DataError, FileMissingError, FileNotFoundError)


def _format_user_error(error: Exception) -> tuple[str, str] | None:
    """Return a dialog title/message for expected pipeline failures."""
    if isinstance(error, ICAMissingError):
        return "ICA not computed", f"The ICA has not been computed yet, so the requested data is not available.\n\n{error}"
    if isinstance(error, FileMissingError):
        return "Missing input", f"A required input file is missing.\n\n{error}"
    if isinstance(error, FileNotFoundError):
        path = error.filename or str(error)
        return "Missing file", f"A required file is missing:\n\n{path}"
    if isinstance(error, DataError):
        return "Data error", str(error)
    if isinstance(error, ConfigurationError):
        return "Configuration error", str(error)
    return None


def _error_dialog_args(error: Exception) -> tuple[str, str, str | None]:
    """Return ``(tb, title, message)`` for :meth:`PipelineFrame._show_error`.

    ``message`` is ``None`` for unexpected errors, selecting the bug-report
    presentation.
    """
    tb = ''.join(traceback.format_exception(error))
    dialog = _format_user_error(error)
    if dialog is None:
        return tb, "Error", None
    return tb, *dialog


def _timed_rows(
        combos: Iterable,
        log: logging.Logger,
        label: str,
) -> Iterator:
    """Yield ``combos``, logging at DEBUG level how long the consumer spends on each one.

    :meth:`PipelineFrame._iter_rows` builds its table by looping over
    :meth:`PipelineFrame._iter_combos`, so the interval between two yields is the work
    that goes into one table row, whichever task's branch produced it. Visible on the
    terminal with ``eelbrain-gui --debug``.

    Parameters
    ----------
    combos
        Key-field combinations from :meth:`Pipeline.iter`, one per table row.
    log
        Pipeline logger to write the timings to.
    label
        Task name, identifying which table the rows belong to.
    """
    t = time.time()
    for i, combo in enumerate(combos):
        yield combo
        now = time.time()
        log.debug(f"Pipeline GUI {label}: row {i} {combo} in {now - t:.3f} s")
        t = now


class BadChannelsDialog(wx.Dialog):
    """Editable comma-separated bad-channel entry with live validation.

    The OK button is disabled while the entry contains channel names that are
    not present in the recording, and a status message lists the offenders.
    """

    def __init__(self, parent, sensor, current_bads: list[str]) -> None:
        super().__init__(parent, title="Set Bad Channels")
        self._sensor = sensor
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(wx.StaticText(self, label="Bad channels (comma-separated):"), flag=wx.LEFT | wx.RIGHT | wx.TOP, border=12)
        self._text = wx.TextCtrl(self, value=', '.join(current_bads), size=(400, -1))
        self._text.Bind(wx.EVT_TEXT, self._on_text)
        vbox.Add(self._text, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=12)
        self._status = wx.StaticText(self, label="")
        self._status.SetForegroundColour(wx.RED)
        vbox.Add(self._status, flag=wx.EXPAND | wx.ALL, border=12)
        buttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        self._ok_button = self.FindWindowById(wx.ID_OK)
        vbox.Add(buttons, flag=wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=12)
        self.SetSizerAndFit(vbox)
        self._validate()

    def _parse(self) -> list[str]:
        return [name for name in (part.strip() for part in self._text.GetValue().split(',')) if name]

    def _on_text(self, event):
        self._validate()

    def _validate(self) -> None:
        missing = [ch for ch in self._parse() if ch not in self._sensor.names]
        if missing:
            self._status.SetLabel(f"Not in data: {', '.join(sorted(missing))}")
            self._ok_button.Disable()
        else:
            self._status.SetLabel("")
            self._ok_button.Enable()

    def get_bad_channels(self) -> list[str]:
        return self._parse()


# Row standing in for the common brain in the MRI table; not a subject
COMMON_BRAIN_ROW = '(common brain)'
# Status of the common brain row without a reconstruction. Distinct from
# MRITask.missing_status because the row is not a subject: it is neither counted
# nor coloured like one, and _on_mri_activated offers to download fsaverage for it.
COMMON_BRAIN_MISSING = 'missing'
# Statuses written by the compute queue while a row is in flight: the artifact
# is not there yet, so they count as missing in the status bar
TRANSIENT_STATUS = ('queued', '⟳')
# Status of a row whose artifact has not been looked at yet: the first pass of a
# refresh lays out the table, the second fills these in (see PipelineFrame._refresh_thread)
LOADING = '…'
# Status of a row whose artifact could not be inspected or computed; always shown red
ERROR = 'error'
# Stands in for a detail column that has no value because the artifact is missing
PLACEHOLDER = '—'
GREY = wx.Colour(150, 150, 150)


@dataclass(frozen=True)
class Layout:
    """Column geometry of the table for one task and Raw selection.

    Resolved once per table by :meth:`Task.layout` and cached by
    :class:`PipelineFrame`, so that the rows built off the main thread and the
    columns they are written into can never disagree.

    Attributes
    ----------
    key_fields
        State fields identifying a row; their column values form its ``combo``.
    columns
        ``(title, width)`` for every column, in order.
    status_col
        Index of the Status column.
    """
    key_fields: tuple[str, ...]
    columns: tuple[tuple[str, int], ...]
    status_col: int


class Task:
    """Per-task presentation rules for the pipeline table.

    One instance per entry in the task dropdown, holding everything that varies
    between tasks: which toolbar controls apply, the column layout, the status
    vocabulary, the row colouring and the status-bar summary. Building the rows
    and acting on a double-click stay in :class:`PipelineFrame`, which
    dispatches on :attr:`name`.

    Attributes
    ----------
    name
        Internal key; also identifies the task in a job's scope (see
        :meth:`PipelineFrame._table_scope`).
    label
        Entry in the task dropdown.
    detail_columns
        ``(title, width)`` for the columns after Status.
    subject_width
        Width of the leading Subject column.
    status_width
        Width of the Status column.
    missing_status
        Status of a row whose artifact has not been made yet; ``None`` for a
        task whose artifact always exists.
    done_status
        Status of a row whose artifact is available.
    unit
        Noun for the status-bar count.
    recording_unit
        Noun for the status-bar count when the table has one row per recording
        rather than per subject, i.e. when a key field beyond ``subject``
        varies; ``None`` for tasks that always have one row per subject.
    summary
        Status-bar phrase after the count.
    missing_note
        Noun for the ``(N missing …)`` suffix; ``None`` omits the suffix, ``''``
        gives a bare ``(N missing)``.
    shows_raw
        Whether the Raw dropdown applies.
    shows_epoch
        Whether the Epoch and Rejection dropdowns apply.
    compute_label
        Label for the compute button; ``None`` for tasks that only inspect.
    compute_tooltip
        Tooltip for the compute button.
    """
    name: str
    label: str
    detail_columns: tuple[tuple[str, int], ...] = ()
    subject_width: int = 180
    status_width: int = 110
    missing_status: str | None = None
    done_status: str | None = None
    unit: str = 'subjects'
    recording_unit: str | None = None
    summary: str = ''
    missing_note: str | None = None
    shows_raw: bool = False
    shows_epoch: bool = False
    compute_label: str | None = None
    compute_tooltip: str | None = None

    def __repr__(self) -> str:
        return f'<Task {self.name}>'

    # ------------------------------------------------------------------
    # Availability and toolbar

    @staticmethod
    def available(pipeline: Pipeline) -> bool:
        """Whether this task applies to ``pipeline`` at all."""
        return True

    @staticmethod
    def raw_choices(pipeline: Pipeline) -> list[str]:
        """Raw pipes to offer in the Raw dropdown."""
        return list(pipeline.get_field_values('raw'))

    def computable(
            self,
            pipeline: Pipeline,
            epoch_rejection: str | None,
    ) -> bool:
        """Whether the compute button applies to the current selection."""
        return self.compute_label is not None

    # ------------------------------------------------------------------
    # Columns

    def key_fields(
            self,
            pipeline: Pipeline,
            raw_name: str | None,
    ) -> tuple[str, ...]:
        """State fields identifying a row; their column values form its ``combo``."""
        return ('subject',)

    def extra_columns(self, key_fields: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        """``(title, width)`` for the columns between Subject and Status."""
        return tuple((field.title(), 90) for field in key_fields[1:])

    def layout(
            self,
            pipeline: Pipeline,
            raw_name: str | None,
    ) -> Layout:
        """Resolve the column geometry; the single derivation of all three parts."""
        key_fields = self.key_fields(pipeline, raw_name)
        extra = self.extra_columns(key_fields)
        columns = (('Subject', self.subject_width), *extra, ('Status', self.status_width), *self.detail_columns)
        return Layout(key_fields, columns, 1 + len(extra))

    # ------------------------------------------------------------------
    # Row appearance

    def row_colour(
            self,
            row: tuple[str, ...],
            layout: Layout,
    ) -> wx.Colour | None:
        """Text colour for a row, or ``None`` for the default."""
        return None

    def result_columns(self, result: object) -> tuple[str, ...]:
        """Detail-column values describing a freshly computed artifact."""
        raise NotImplementedError(f"{self.name} is not computable")

    def missing_row(
            self,
            combo: tuple[str, ...],
            layout: Layout,
            status: str | None = None,
    ) -> tuple[str, ...]:
        """Row for a key combination with no artifact to show: status plus placeholders.

        Also the row of the first refresh pass, whose status is not known yet
        (``status=LOADING``; see :meth:`PipelineFrame._refresh_thread`).

        Parameters
        ----------
        combo
            Key-field column values the row is prefixed with.
        layout
            Column geometry the row has to fit. Any column between the key
            fields and Status gets a placeholder too, so that the row is as
            wide as the table even for a task whose leading columns are not all
            key fields (MRI, Coregistration).
        status
            Status to show; defaults to :attr:`missing_status`.
        """
        n_lead = layout.status_col - len(combo)
        n_detail = len(layout.columns) - layout.status_col - 1
        return (*combo, *(PLACEHOLDER,) * n_lead, self.missing_status if status is None else status, *(PLACEHOLDER,) * n_detail)

    # ------------------------------------------------------------------
    # Status bar

    def counts_row(self, row: tuple[str, ...]) -> bool:
        """Whether a row counts towards the status-bar total."""
        return True

    def counts_done(self, status: str) -> bool:
        return status == self.done_status

    def counts_missing(self, status: str) -> bool:
        return status == self.missing_status or (self.compute_label is not None and status in TRANSIENT_STATUS)

    def status_bar(
            self,
            rows: list[tuple[str, ...]],
            layout: Layout,
    ) -> str:
        """Summary of the whole table for the status bar."""
        rows = [row for row in rows if self.counts_row(row)]
        n_done = sum(1 for row in rows if self.counts_done(row[layout.status_col]))
        # a task cached per recording shows one row per key-field combination, so
        # counting them as subjects would overstate the total
        unit = self.recording_unit if self.recording_unit and len(layout.key_fields) > 1 else self.unit
        msg = f"{n_done} / {len(rows)} {unit} · {self.summary}"
        if self.missing_note is not None:
            n_missing = sum(1 for row in rows if self.counts_missing(row[layout.status_col]))
            if n_missing:
                noun = f" {self.missing_note}" if self.missing_note else ''
                msg += f"  ({n_missing} missing{noun})"
        n_error = sum(1 for row in rows if row[layout.status_col] == ERROR)
        if n_error:
            msg += f"  ({n_error} error)"
        return msg


class BadChannelsTask(Task):
    name = 'bad_chs'
    label = "Bad channels"
    detail_columns = (('N bad', 90),)
    done_status = 'done'
    summary = "bad channels defined"
    recording_unit = 'recordings'
    shows_raw = True

    @staticmethod
    def available(pipeline: Pipeline) -> bool:
        return any(isinstance(pipe, RawSource) for pipe in pipeline._raw.values())

    def key_fields(
            self,
            pipeline: Pipeline,
            raw_name: str | None,
    ) -> tuple[str, ...]:
        # bad channels are stored per recording; show one row per combination of
        # the key fields that vary in this experiment
        fields = ['subject']
        for field, values in (('session', pipeline._sessions), ('task', pipeline._tasks), ('run', pipeline._runs)):
            if len(values) > 1:
                fields.append(field)
        return tuple(fields)


class ICATask(Task):
    name = 'ica'
    label = "ICA"
    detail_columns = (('Components', 110), ('Rejected', 90))
    missing_status = 'no ICA'
    done_status = 'selected'
    summary = "ICA selected"
    missing_note = 'ICA file'
    recording_unit = 'recordings'
    shows_raw = True
    compute_label = "Make ICA"
    compute_tooltip = "Compute ICA for all subjects with missing files"

    @staticmethod
    def available(pipeline: Pipeline) -> bool:
        return any(isinstance(pipe, RawICA) for pipe in pipeline._raw.values())

    @staticmethod
    def raw_choices(pipeline: Pipeline) -> list[str]:
        return [name for name, pipe in pipeline._raw.items() if isinstance(pipe, RawICA)]

    def key_fields(
            self,
            pipeline: Pipeline,
            raw_name: str | None,
    ) -> tuple[str, ...]:
        # ICA is cached per (subject, session[, run]); show one row per
        # combination of the key fields that vary in this experiment.
        fields = ['subject']
        if len(pipeline._sessions) > 1:
            fields.append('session')
        if raw_name and not pipeline._raw[raw_name]._concatenate_runs and len(pipeline._runs) > 1:
            fields.append('run')
        return tuple(fields)

    def row_colour(
            self,
            row: tuple[str, ...],
            layout: Layout,
    ) -> wx.Colour | None:
        # an ICA without a single rejected component is almost always an oversight
        if row[layout.status_col] == self.done_status and row[layout.status_col + 2] == '0':
            return wx.RED
        return None

    def result_columns(self, ica: mne.preprocessing.ICA) -> tuple[str, ...]:
        return str(ica.n_components_), str(len(ica.exclude))


class EpochRejectionTask(Task):
    name = 'epoch_rej'
    label = "Epoch rejection"
    detail_columns = (('N total', 90), ('N rejected', 90))
    missing_status = 'missing'
    done_status = 'done'
    summary = "epoch rejection done"
    shows_raw = True
    shows_epoch = True
    compute_label = "Compute rejection"
    compute_tooltip = "Compute rejection files for all subjects with missing files"

    @staticmethod
    def available(pipeline: Pipeline) -> bool:
        return any(rej is not None for rej in pipeline._epoch_rejection.values())

    def computable(
            self,
            pipeline: Pipeline,
            epoch_rejection: str | None,
    ) -> bool:
        # a ManualRejection is edited in its own GUI, not computed in bulk
        return isinstance(pipeline._epoch_rejection.get(epoch_rejection), ChannelModelRejection)

    def result_columns(self, ds: Dataset) -> tuple[str, ...]:
        return str(ds.n_cases), str(int((~ds['accept']).sum()))


class MRITask(Task):
    name = 'mri'
    label = "MRI"
    status_width = 130
    missing_status = 'no MRI'
    done_status = 'ok'
    summary = "MRI available"
    missing_note = ''

    def extra_columns(self, key_fields: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        # display-only, so it is not a key field: a row is still named by subject
        return (('MRI subject', 170),)

    def counts_row(self, row: tuple[str, ...]) -> bool:
        return row[0] != COMMON_BRAIN_ROW

    def counts_done(self, status: str) -> bool:
        return status in (self.done_status, 'template')

    def row_colour(
            self,
            row: tuple[str, ...],
            layout: Layout,
    ) -> wx.Colour | None:
        if row[layout.status_col] == self.missing_status:
            return wx.RED
        elif row[0] == COMMON_BRAIN_ROW:
            return GREY
        return None


class CoregTask(Task):
    name = 'coreg'
    label = "Coregistration"
    subject_width = 140
    missing_status = 'missing'
    done_status = 'ok'
    unit = 'sessions'
    summary = "coregistration done"
    missing_note = ''

    def key_fields(
            self,
            pipeline: Pipeline,
            raw_name: str | None,
    ) -> tuple[str, ...]:
        return ('subject', 'session')

    def extra_columns(self, key_fields: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        # Session is a key field; MRI subject is display-only
        return (('Session', 80), ('MRI subject', 140))

    def row_colour(
            self,
            row: tuple[str, ...],
            layout: Layout,
    ) -> wx.Colour | None:
        return wx.RED if row[layout.status_col] == self.missing_status else None


# Order determines the task dropdown
TASKS = (BadChannelsTask(), ICATask(), EpochRejectionTask(), MRITask(), CoregTask())
TASKS_BY_NAME = {task.name: task for task in TASKS}


class PipelineFrame(EelbrainFrame):
    """Top-level window for inspecting and running pipeline setup tasks.

    Shows per-subject status for ICA selection or epoch rejection, and opens
    the corresponding sub-GUI on double-click.
    """

    def __init__(self, pipeline) -> None:
        super().__init__(parent=None, title=f"Pipeline: {pipeline.root}")
        self._pipeline = pipeline
        self._refresh_token = None  # replaced each refresh; threads compare identity
        # Serializes the two background threads' use of the pipeline: the refresh walk,
        # and the worker's data loading and saving. Deliberately not held across job(),
        # which is the long part and needs no pipeline access, so a refresh never waits
        # for a fit -- only for a load or a save. A refresh that stops to ask about a
        # stale ICA does hold it until the user answers, which stalls the worker between
        # jobs. Never taken on the main thread: that would freeze the window while the
        # worker loads, and since the worker waits on the main thread for its own
        # stale-ICA dialog, a main-thread waiter could deadlock. Main-thread pipeline
        # use (opening a sub-GUI, writing bad channels) is therefore still unguarded.
        self._pipeline_lock = threading.Lock()
        self._compute_token = None  # replaced each compute run; threads compare identity
        # Whether a worker thread is alive. Distinct from _compute_token, which Stop
        # clears at once for the UI while the worker keeps going until its current job
        # returns: during that window no second worker may start.
        self._worker_active = False
        # One job queue for every computable task: jobs are computed sequentially by a
        # single worker thread, so that only one thread at a time uses the pipeline.
        # Entries carry their own scope, because the user can switch tasks and queue more
        # rows while a batch is running.
        self._job_queue = []  # [(scope, combo, spec), ...] waiting to be computed
        self._job_in_progress = None  # (scope, combo) the worker popped and is computing
        self._job_queue_lock = threading.Lock()
        self._n_done = self._n_total = 0  # progress of the current run
        self._n_loading = 0  # rows of the displayed table whose status is still to be filled in
        # Job specs for the rows currently displayed, keyed by (scope, combo). Minted
        # during refresh (where pipeline.iter() sets the state), cleared for the new table
        # by _populate_table and filled in row by row by _fill_row. The scope is part of
        # the key because a combo only names a row within one table (see
        # :meth:`_table_scope`), and a lookup can outlive the table it was minted for
        # (_on_ica_bad_channels runs when a separate window closes). A spec holds no data,
        # only the state it was resolved with, so it stays usable after the inputs change:
        # make_job() re-resolves dependencies live at that point.
        self._job_specs: dict[tuple[tuple, tuple], JobSpec] = {}
        self._tasks: list[Task] = []  # dropdown entries, in order
        # Column geometry of the displayed table, read by everything that addresses a
        # row by column index. Installed by _setup_columns, first through the
        # _on_task_changed() below and again whenever the task or Raw pipe changes.
        # Part of a job's scope, so a row built for one layout is never written into
        # another.
        self._layout: Layout

        self._init_ui()
        self._populate_tasks()
        # MRI and Coregistration are always offered, so the dropdown is never empty and
        # a task is selected for the rest of the window's life
        self._task_choice.SetSelection(0)
        self._on_task_changed(None)

        # Width: fit the widest toolbar
        # Height: fill the usable display (wx.Fit() doesn't help here because the
        # ListCtrl uses proportion=1 and its content is populated asynchronously).
        display = wx.GetClientDisplayRect()
        self.SetSize((800, display.height - 80))
        self.Centre()
        self.Bind(wx.EVT_CLOSE, self._on_close)

    # ------------------------------------------------------------------
    # UI construction

    def _init_ui(self):
        self._panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Toolbar row
        toolbar = wx.BoxSizer(wx.HORIZONTAL)

        toolbar.Add(
            wx.StaticText(self._panel, label="Task:"),
            flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=8,
        )
        self._task_choice = wx.Choice(self._panel)
        self._task_choice.Bind(wx.EVT_CHOICE, self._on_task_changed)
        toolbar.Add(
            self._task_choice,
            flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=6,
        )

        # Extra controls shown only in epoch-rejection mode
        self._epoch_rejection_label = wx.StaticText(self._panel, label="Rejection:")
        self._epoch_rejection_choice = wx.Choice(self._panel)
        self._epoch_rejection_choice.Bind(wx.EVT_CHOICE, self._on_epoch_rejection_changed)
        self._epoch_label = wx.StaticText(self._panel, label="Epoch:")
        self._epoch_choice = wx.Choice(self._panel)
        self._epoch_choice.Bind(wx.EVT_CHOICE, self._on_state_changed)
        self._raw_label = wx.StaticText(self._panel, label="Raw:")
        self._raw_choice = wx.Choice(self._panel)
        self._raw_choice.Bind(wx.EVT_CHOICE, self._on_raw_changed)

        for widget, border in [
            (self._epoch_rejection_label, 14),
            (self._epoch_rejection_choice, 4),
            (self._epoch_label, 10),
            (self._epoch_choice, 4),
            (self._raw_label, 10),
            (self._raw_choice, 4),
        ]:
            toolbar.Add(widget, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=border)

        toolbar.AddStretchSpacer()

        # Compute button + progress; label and visibility follow the task
        self._compute_btn = wx.Button(self._panel, label="", style=wx.BU_EXACTFIT)
        self._compute_btn.Bind(wx.EVT_BUTTON, self._on_compute)
        toolbar.Add(self._compute_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)

        self._progress_gauge = wx.Gauge(self._panel, style=wx.GA_HORIZONTAL | wx.GA_SMOOTH)
        self._progress_gauge.SetMinSize((100, -1))
        toolbar.Add(self._progress_gauge, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=4)

        self._progress_label = wx.StaticText(self._panel, label="")
        toolbar.Add(self._progress_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=8)

        self._refresh_btn = wx.Button(self._panel, label="↺", style=wx.BU_EXACTFIT)
        self._refresh_btn.SetToolTip("Refresh status")
        self._refresh_btn.Bind(wx.EVT_BUTTON, self._on_refresh)
        toolbar.Add(self._refresh_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=8)

        vbox.Add(toolbar, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=6)
        vbox.Add(wx.StaticLine(self._panel), flag=wx.EXPAND)

        # Subject table
        self._list = wx.ListCtrl(
            self._panel,
            style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_NONE,
        )
        self._list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_item_activated)
        self._list.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self._on_item_right_click)
        vbox.Add(self._list, proportion=1, flag=wx.EXPAND)

        self._panel.SetSizer(vbox)
        self.CreateStatusBar()

        for w in (self._epoch_rejection_label, self._epoch_rejection_choice,
                  self._epoch_label, self._epoch_choice,
                  self._raw_label, self._raw_choice,
                  self._compute_btn, self._progress_gauge, self._progress_label):
            w.Hide()

    # ------------------------------------------------------------------
    # Task / state population

    def _populate_tasks(self):
        # The task selects the operation; the raw pipe is chosen separately via
        # the Raw dropdown (filtered to the sources/ICA stages for the task).
        self._tasks = [task for task in TASKS if task.available(self._pipeline)]
        for task in self._tasks:
            self._task_choice.Append(task.label)

    def _current_task(self) -> Task:
        """Task selected in the dropdown; always set (see :meth:`__init__`)."""
        return self._tasks[self._task_choice.GetSelection()]

    def _raw_name(self) -> str | None:
        """Raw pipe the current table is for; ``None`` for a task without a Raw dropdown.

        The dropdown keeps its selection while hidden, so the task decides whether
        that selection means anything.
        """
        return self._raw_choice.GetStringSelection() if self._current_task().shows_raw else None

    def _table_scope(self) -> tuple:
        """Identity of the table on display: ``(task, epoch_rejection, epoch, raw, layout)``.

        Every choice that selects which rows are shown, and the arguments
        :meth:`_iter_combos` and :meth:`_iter_rows` need to produce them. A row ``combo`` only names a
        row within one scope, so a job minted for one table is never applied to a
        row of another: the Raw, Epoch and Epoch-rejection choices stay enabled
        while a computation runs, and switching one of them (or the number of ICA
        key-field columns that follows from Raw) leaves the same combo pointing at
        an unrelated recording. The layout comes from :meth:`_setup_columns` rather
        than being re-derived off the main thread, so a row can never come back the
        wrong width.
        """
        task = self._current_task()
        epoch_rejection = self._current_epoch_rejection() if task.shows_epoch else None
        epoch_name = self._epoch_choice.GetStringSelection() if task.shows_epoch else None
        return task, epoch_rejection, epoch_name, self._raw_name(), self._layout

    def _populate_epoch_choices(self):
        previous = self._epoch_choice.GetStringSelection()
        self._epoch_choice.Clear()
        for name, epoch in self._pipeline._epochs.items():
            if isinstance(epoch, PrimaryEpoch):
                self._epoch_choice.Append(name)
        self._restore_selection(self._epoch_choice, previous, 0)

    def _populate_raw_choices(self, task: Task) -> None:
        """Fill the Raw dropdown with the pipes relevant to ``task``."""
        previous = self._raw_choice.GetStringSelection()
        self._raw_choice.Clear()
        for name in task.raw_choices(self._pipeline):
            self._raw_choice.Append(name)
        default = self._raw_choice.FindString('raw')
        self._restore_selection(self._raw_choice, previous, default if default != wx.NOT_FOUND else 0)

    def _populate_epoch_rejection_choices(self):
        previous = self._epoch_rejection_choice.GetStringSelection()
        self._epoch_rejection_choice.Clear()
        for name, rej in self._pipeline._epoch_rejection.items():
            if rej is not None:
                self._epoch_rejection_choice.Append(name)
        self._restore_selection(self._epoch_rejection_choice, previous, 0)

    @staticmethod
    def _restore_selection(choice: wx.Choice, previous: str, default: int):
        """Re-select ``previous`` if still present, else fall back to ``default``."""
        if not choice.GetCount():
            return
        index = choice.FindString(previous) if previous else wx.NOT_FOUND
        choice.SetSelection(index if index != wx.NOT_FOUND else default)

    def _current_epoch_rejection(self) -> str | None:
        return self._epoch_rejection_choice.GetStringSelection() or None

    def _update_compute_button(self):
        """Show the compute button, labelled for the task that can use it."""
        task = self._current_task()
        computable = task.computable(self._pipeline, self._current_epoch_rejection())
        if computable:
            self._compute_btn.SetLabel("Stop" if self._compute_token is not None else task.compute_label)
            self._compute_btn.SetToolTip(task.compute_tooltip)
            # Computing waits until every row is in, so that one click queues every missing row; stopping never waits
            self._compute_btn.Enable(self._compute_token is not None or not self._n_loading)
        self._compute_btn.Show(computable)
        self._panel.Layout()

    def _on_epoch_rejection_changed(self, event):
        self._update_compute_button()
        self._start_refresh()

    # ------------------------------------------------------------------
    # Event handlers

    def _on_task_changed(self, event):
        task = self._current_task()
        # Belt and braces: the task dropdown is disabled while a computation runs, so
        # there is normally nothing to stop, and _setup_columns clears the rows anyway
        self._stop_compute()
        for widget in (self._epoch_rejection_label, self._epoch_rejection_choice, self._epoch_label, self._epoch_choice):
            widget.Show(task.shows_epoch)
        self._raw_label.Show(task.shows_raw)
        self._raw_choice.Show(task.shows_raw)
        if task.shows_epoch:
            self._populate_epoch_rejection_choices()
            self._populate_epoch_choices()
        if task.shows_raw:
            self._populate_raw_choices(task)
        self._update_compute_button()
        self._setup_columns()
        self._start_refresh()

    def _on_state_changed(self, event):
        self._start_refresh()

    def _on_raw_changed(self, event):
        # Switching the ICA raw pipe can change run concatenation, and with it the
        # per-row key fields, so reinstall the columns before refreshing.
        self._setup_columns()
        self._start_refresh()

    def _on_refresh(self, event):
        self._start_refresh()

    def _on_item_activated(self, event):
        """Row double-click"""
        self._activate_row(event.GetIndex())

    def _on_item_right_click(self, event):
        """Row right-click: context menu (Bad channels task only)."""
        if self._current_task().name != 'bad_chs' or self._n_loading:
            return  # while rows are loading, the refresh thread is using the pipeline (see _activate_row)
        idx = event.GetIndex()
        if idx == wx.NOT_FOUND:
            return
        menu = wx.Menu()
        set_item = menu.Append(wx.ID_ANY, "Set Bad Channels")
        plot_item = menu.Append(wx.ID_ANY, "Plot continuous data")
        self.Bind(wx.EVT_MENU, lambda event: self._set_bad_channels_dialog(idx), set_item)
        self.Bind(wx.EVT_MENU, lambda event: self._activate_row(idx), plot_item)
        self._list.PopupMenu(menu)
        self.Unbind(wx.EVT_MENU, source=set_item)
        self.Unbind(wx.EVT_MENU, source=plot_item)
        menu.Destroy()

    def _set_bad_channels_dialog(self, idx: int) -> None:
        """Edit a row's bad channels via a text dialog with live validation."""
        pipeline = self._pipeline
        raw_name = self._raw_name()
        state = dict(zip(self._layout.key_fields, self._row_combo(idx)))
        wx.BeginBusyCursor()
        try:
            pipeline.set(raw=raw_name, **state)
            source_name = pipeline._raw.root_source_name(raw_name)
            source_pipe = pipeline._raw.root_source_pipe(raw_name)
            raw = pipeline._load_derivative(raw_input_name(source_name), options={'noise': False})
            sensor = load.mne.sensor_dim(raw.info, adjacency=source_pipe.adjacency)
            current_bads = pipeline.load_bad_channels()
        except _USER_ERROR_TYPES as error:
            self._show_error(*_error_dialog_args(error))
            return
        finally:
            wx.EndBusyCursor()
        dlg = BadChannelsDialog(self, sensor, current_bads)
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            bad_chs = dlg.get_bad_channels()
        finally:
            dlg.Destroy()
        try:
            pipeline.make_bad_channels(bad_chs, redo=True, raw=raw_name, **state)
        except _USER_ERROR_TYPES as error:
            self._show_error(*_error_dialog_args(error))
            return
        self._start_refresh()

    def _activate_row(self, idx: int) -> None:
        """Perform the double-click action for the row at ``idx``."""
        if self._n_loading:
            # The refresh thread is still setting the pipeline's state row by row (see
            # _iter_rows), which an action on the main thread would interleave with
            return
        subject = self._list.GetItemText(idx, 0)
        task = self._current_task()
        # Loop so that after the user incorporates a stale ICA we can retry the
        # action through the same try, keeping the _USER_ERROR_TYPES handler
        # around the retry; every other path falls through to the return.
        while True:
            try:
                self._activate_item(idx, subject, task)
            except ICAMissingError:
                dlg = wx.MessageDialog(self, f"The ICA for {subject} has not been computed yet, so raw data at the ICA stage cannot be displayed. Compute the ICA first (Make ICA in the ICA task).", "ICA Not Computed", wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
            except _USER_ERROR_TYPES as error:
                self._show_error(*_error_dialog_args(error))
            except ICAChannelsChangedError as error:
                if self._ask_ica_channels_changed(error):
                    Path(error.path).unlink()
                    self._start_refresh()
                else:
                    wx.CallAfter(wx.GetApp().ExitMainLoop)
            except ProtectedArtifactError as error:
                # A stale ICA dependency surfaced while building the requested
                # artifact (e.g. make_epoch_rejection); route it through the
                # same dialog used during refresh. A single row is involved, so
                # "Apply to all" is not offered here.
                choice, _ = self._ask_stale_ica(subject, error)
                if choice == StaleICADialog.INCORPORATE:
                    self._pipeline.load_ica(raw=self._raw_name(), accept_stale=True)
                    continue  # manifest now matches; retry the action
                elif choice == StaleICADialog.ABORT:
                    wx.CallAfter(wx.GetApp().ExitMainLoop)
                else:  # DELETE / IGNORE / dismissed: the action can not proceed
                    if choice == StaleICADialog.DELETE:
                        Path(error.path).unlink()
                    self._start_refresh()
            return

    def _activate_item(self, idx: int, subject: str, task: Task) -> None:
        """Perform the action for a double-clicked row."""
        wx.BeginBusyCursor()
        try:
            if task.name == 'bad_chs':
                raw_name = self._raw_name()
                state = dict(zip(self._layout.key_fields, self._row_combo(idx)))
                frame = self._pipeline.make_bad_channels_selection(raw=raw_name, **state)
                if frame is not None:
                    doc = frame.model.doc
                    doc.callbacks.subscribe(
                        'saved',
                        lambda: wx.CallAfter(self._start_refresh),
                    )
            elif task.name == 'ica':
                raw_name = self._raw_name()
                scope = self._table_scope()
                combo = self._row_combo(idx)
                state = dict(zip(self._layout.key_fields, combo))
                frame = self._pipeline.make_ica_selection(raw=raw_name, **state)
                if frame is not None:
                    doc = frame.model.doc
                    # enables the ICA GUI to add bad channels it finds
                    doc.bad_channels_callback = partial(self._on_ica_bad_channels, raw_name, state, scope, combo, doc)
                    doc.callbacks.subscribe(
                        'saved',
                        lambda: wx.CallAfter(self._update_ica_row, scope, combo, doc),
                    )
            elif task.name == 'epoch_rej':
                name = self._current_epoch_rejection()
                if name is not None:
                    # opens an editable GUI for ManualRejection, read-only for an
                    # automatically generated rejection
                    self._pipeline.make_epoch_rejection(
                        subject=subject,
                        epoch_rejection=name,
                        epoch=self._epoch_choice.GetStringSelection(),
                        raw=self._raw_name(),
                    )
                    # Epoch rejection has no in-memory object to read from,
                    # so do a targeted single-subject refresh instead.
                    self._start_refresh()
            elif task.name == 'mri':
                self._on_mri_activated(idx, subject)
            elif task.name == 'coreg':
                self._on_coreg_activated(idx)
        finally:
            wx.EndBusyCursor()

    def _on_ica_bad_channels(
            self,
            raw_name: str,
            state: dict[str, str],
            scope: tuple,
            combo: tuple,
            doc: ICADocument,
            names: Sequence[str],
            recompute: bool,
    ):
        """Add bad channels found in the ICA GUI.

        The ICA was estimated with these channels included, so it is deleted; the ICA GUI
        closes itself after invoking this.

        Parameters
        ----------
        raw_name
            ICA raw step the decomposition belongs to.
        state
            Subject and key fields of the recording, from its row.
        scope
            Table the row belongs to, captured when this ICA GUI was opened: by the
            time the user gets here the pipeline GUI may show a different one, and
            ``combo`` would then name an unrelated recording.
        combo
            Row combo, for queueing the recompute against the right row.
        doc
            Document of the ICA GUI that found the channels; its file is the one
            invalidated here.
        names
            Channels to add to the bad channels.
        recompute
            Whether to queue the new ICA decomposition right away.
        """
        self._pipeline.set(raw=raw_name, **state)
        spec = self._pipeline._job_spec(ica_input_name(raw_name))
        # ICA combines bad channels across tasks/runs
        node = spec.ctx.node
        for source_state in node._source_states(spec.ctx, node.pipe.task):
            self._pipeline.make_bad_channels(names, raw=raw_name, **{**state, **source_state})
        Path(doc.path).unlink(missing_ok=True)
        if recompute:
            wx.CallAfter(self._queue_jobs, scope, [(combo, spec)])
        else:
            wx.CallAfter(self._start_refresh)

    def _on_mri_activated(self, row_idx: int, subject: str):
        """Handle double-click on an MRI row."""
        mrisubject = self._list.GetItemText(row_idx, 1)
        status = self._list.GetItemText(row_idx, 2)
        subjects_dir = str(self._pipeline.root / MRI_SDIR)
        common_brain = self._pipeline.get('common_brain')

        if subject == COMMON_BRAIN_ROW:
            if status == COMMON_BRAIN_MISSING:
                if mrisubject == 'fsaverage':
                    dlg = wx.MessageDialog(
                        self,
                        f"fsaverage is not yet present in {subjects_dir}.\n\n"
                        "Download it now from the MNE dataset repository?",
                        "Download fsaverage?",
                        wx.YES_NO | wx.ICON_QUESTION,
                    )
                    if dlg.ShowModal() == wx.ID_YES:
                        self._fetch_fsaverage()
                    dlg.Destroy()
                else:
                    wx.MessageBox(
                        f"Common brain '{mrisubject}' has no FreeSurfer reconstruction "
                        "in the FreeSurfer subjects directory.",
                        "MRI not found", wx.OK | wx.ICON_INFORMATION, self,
                    )
        elif status == MRITask.missing_status:
            dlg = wx.MessageDialog(
                self,
                f"To create a scaled template brain from {common_brain}, switch to the Coregistration task.",
                f"No FreeSurfer reconstruction found for {mrisubject}",
                wx.OK | wx.ICON_INFORMATION,
            )
            dlg.ShowModal()
            dlg.Destroy()

    def _on_coreg_activated(self, row_idx: int):
        """Handle double-click on a Coregistration row."""
        subject = self._list.GetItemText(row_idx, 0)
        session = self._list.GetItemText(row_idx, 1)
        mrisubject = self._list.GetItemText(row_idx, 2)
        subjects_dir_path = self._pipeline.root / MRI_SDIR
        subjects_dir = str(subjects_dir_path)
        pipeline = self._pipeline
        # If the subject has no FreeSurfer reconstruction, fall back to the
        # template brain so the coreg GUI can open and the user can use its
        # "Scale MRI" feature to create a subject-specific brain.
        if not (subjects_dir_path / mrisubject / 'surf' / 'lh.pial').exists():
            mrisubject = pipeline.get('common_brain')
        with pipeline._temporary_state:
            kw = dict(subject=subject, raw='raw')
            if session:
                kw['session'] = session
            pipeline.set(**kw)
            raw_ctx = pipeline._resolve_derivative(raw_input_name('raw'))
            inst = str(raw_ctx.node.path(raw_ctx))
            trans_ctx = pipeline._resolve_derivative('trans-input')
            trans = str(trans_ctx.node.path(trans_ctx)) if trans_ctx.node.exists(trans_ctx) else None
        _launch_coreg_subprocess(mrisubject, subjects_dir, inst, trans,
                                 on_close=lambda: wx.CallAfter(self._start_refresh))

    # ------------------------------------------------------------------
    # Table management

    def _setup_columns(self) -> None:
        """Install the columns for the current task and cache their geometry."""
        self._layout = self._current_task().layout(self._pipeline, self._raw_name())
        self._list.ClearAll()
        for i, (label, width) in enumerate(self._layout.columns):
            self._list.InsertColumn(i, label, width=width)

    def _row(self, idx: int) -> tuple[str, ...]:
        """All column values of a row."""
        return tuple(self._list.GetItemText(idx, c) for c in range(len(self._layout.columns)))

    def _row_combo(self, idx: int) -> tuple:
        """Key-field column values of a row, identifying it within the task."""
        return tuple(self._list.GetItemText(idx, c) for c in range(len(self._layout.key_fields)))

    def _find_row(self, combo: tuple) -> int:
        """Row index whose leading columns match ``combo``, or -1."""
        for i in range(self._list.GetItemCount()):
            if all(self._list.GetItemText(i, c) == val for c, val in enumerate(combo)):
                return i
        return -1

    def _set_row_colour(self, idx: int, row: tuple[str, ...]) -> None:
        """Apply the task's colour rule to a row, clearing it when none applies.

        Rows without a rule are left with no explicit colour rather than being
        painted the default one, so that the list can still invert them when they
        are selected.
        """
        colour = wx.RED if row[self._layout.status_col] == ERROR else self._current_task().row_colour(row, self._layout)
        self._list.SetItemTextColour(idx, wx.NullColour if colour is None else colour)

    def _set_row_result(
            self,
            idx: int,
            status: str,
            values: tuple[str, ...],
    ) -> None:
        """Write the Status and detail columns of a row, and recolour it."""
        self._list.SetItem(idx, self._layout.status_col, status)
        for col, value in enumerate(values, self._layout.status_col + 1):
            self._list.SetItem(idx, col, value)
        self._set_row_colour(idx, self._row(idx))

    def _populate_table(self, rows: list[tuple[str, ...]], token: object) -> None:
        """Install the rows of a new table, with their status still to be filled in."""
        if token is not self._refresh_token:
            return
        # Clears the specs of the previous table; _fill_row adds this table's as its rows
        # resolve. Both run on the main thread and behind the token guard, so a raw/task
        # switch can never leave a spec from the previous table behind.
        self._job_specs = {}
        self._list.DeleteAllItems()
        self._n_loading = len(rows)
        for row in rows:
            idx = self._list.InsertItem(self._list.GetItemCount(), row[0])
            for col, val in enumerate(row[1:], 1):
                self._list.SetItem(idx, col, val)
            self._set_row_colour(idx, row)
        self._refresh_status_bar()

    def _fill_row(
            self,
            token: object,
            scope: tuple,  # see :meth:`_table_scope`
            index: int,
            combo: tuple[str, ...],
            row: tuple[str, ...],
            spec: JobSpec | None,
    ) -> None:
        """Replace one loading row with the status and details the refresh found for it.

        ``index`` is where :meth:`_populate_table` put the row: the second pass of the
        refresh resolves the rows the first pass found, in order, and ``token``
        guarantees the table on display is still the one they were found for. The combo
        is verified all the same, so a row can never be given another recording's status.

        Parameters
        ----------
        token
            Refresh the row belongs to; a row of a superseded refresh is dropped.
        scope
            Table the row belongs to (see :meth:`_table_scope`); with ``combo``, the
            key of its job spec.
        index
            Position of the row in the table.
        combo
            Key-field column values identifying the row.
        row
            All column values of the row.
        spec
            Job spec for the row's artifact; ``None`` if it can not be computed.
        """
        if token is not self._refresh_token:
            return
        if index >= self._list.GetItemCount() or self._row_combo(index) != combo:
            return
        if spec is not None:
            self._job_specs[scope, combo] = spec
        for col, value in enumerate(row):
            self._list.SetItem(index, col, value)
        self._set_row_colour(index, row)
        self._n_loading -= 1
        self._refresh_status_bar()

    def _update_ica_row(
            self,
            scope: tuple,  # see :meth:`_table_scope`
            combo: tuple,
            doc: ICADocument,
    ) -> None:
        """Update a single ICA row from the already-in-memory document (no disk I/O)."""
        i = self._displayed_row(scope, combo)
        if i != -1:
            task = scope[0]  # the scope matched, so this is the task on display
            self._set_row_result(i, task.done_status, task.result_columns(doc.ica))
        self._refresh_status_bar()

    def _refresh_status_bar(self):
        """Recompute the status bar summary from the current table contents.

        While the second pass of a refresh is still filling rows in, the summary would
        undercount, so the progress of that pass is shown instead; that is called once
        per row, so it is counted rather than read off the table.
        """
        n = self._list.GetItemCount()
        if self._n_loading:
            self.SetStatusText(f"Loading… {n - self._n_loading} / {n}")
        else:
            rows = [self._row(i) for i in range(n)]
            self.SetStatusText(self._current_task().status_bar(rows, self._layout))
            self._update_compute_button()

    # ------------------------------------------------------------------
    # Background status refresh

    def _start_refresh(self) -> None:
        scope = self._table_scope()
        task, epoch_rejection, epoch_name, _, _ = scope
        token = object()
        self._refresh_token = token
        self._list.DeleteAllItems()
        self._n_loading = 0
        self.SetStatusText("Loading…")
        if self._compute_token is None:  # during a computation the button is Stop, which a refresh must not block
            self._compute_btn.Disable()

        if task.shows_epoch:
            if epoch_rejection is None:
                self.SetStatusText("No epoch rejection defined")
                return
            if not epoch_name:
                self.SetStatusText("No epochs defined")
                return

        threading.Thread(
            target=self._refresh_thread,
            args=(token, scope),
            daemon=True,
        ).start()

    def _refresh_thread(
            self,
            token: object,
            scope: tuple,  # see :meth:`_table_scope`
    ) -> None:
        """Fill the table in two passes: which rows it has, then what is in them.

        The first pass reads only the pipeline's state model and the BIDS dataset, so
        the table is on screen before any artifact is opened; the second pass posts
        every row as soon as its status resolves, which for the ICA task takes a raw
        file read per recording. Both passes hold the pipeline lock, so neither ever
        walks the pipeline while the compute worker is using it.
        """
        task, _, _, _, layout = scope
        log = self._pipeline._log
        n_filled = 0
        first_error = None  # shown once the pass is over, so that it does not stall the rest
        t_start = time.time()
        try:
            with self._pipeline_lock:
                t_locked = time.time()
                combos = list(self._iter_combos(scope))
            # A refresh that is queued behind a running computation waits for the lock,
            # so the two intervals are logged separately (see eelbrain-gui --debug)
            log.debug(f"Pipeline GUI {task.name}: {len(combos)} rows in {time.time() - t_locked:.3f} s, after waiting {t_locked - t_start:.3f} s for the pipeline")
            if token is not self._refresh_token:
                return  # the table was replaced while the rows were being determined
            wx.CallAfter(self._populate_table, [task.missing_row(combo, layout, LOADING) for combo in combos], token)

            t_start = time.time()
            with self._pipeline_lock:
                t_locked = time.time()
                for index, (combo, row, spec, error) in enumerate(self._iter_rows(token, scope, combos)):
                    wx.CallAfter(self._fill_row, token, scope, index, combo, row, spec)
                    n_filled += 1
                    if error is not None and first_error is None:
                        first_error = error
        except _AbortRequested:
            return  # app exit already scheduled
        except Exception as error:
            wx.CallAfter(self._show_error, *_error_dialog_args(error))
            wx.CallAfter(self._end_refresh, token)
            return
        log.debug(f"Pipeline GUI {task.name}: {n_filled} row details in {time.time() - t_locked:.3f} s, after waiting {t_locked - t_start:.3f} s for the pipeline")
        if first_error is not None:
            wx.CallAfter(self._show_error, *_error_dialog_args(first_error))

    def _end_refresh(self, token: object) -> None:
        """Settle the table of a refresh whose second pass failed: rows it never filled in are shown as errors.

        Parameters
        ----------
        token
            Refresh that ended; one that was superseded has nothing to settle.
        """
        if token is not self._refresh_token or not self._n_loading:
            return
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, self._layout.status_col) == LOADING:
                self._set_row_result(i, ERROR, ())
        self._n_loading = 0
        self._refresh_status_bar()

    def _show_error(self, tb: str, title: str = "Error", message: str | None = None):
        self.SetStatusText("Error")
        dlg = TracebackDialog(self, tb, title, message)
        dlg.ShowModal()
        dlg.Destroy()

    def _on_close(self, event):
        if self._compute_token is not None:
            dlg = wx.MessageDialog(
                self,
                "A computation is in progress. "
                "Closing this window will cancel it and the current job's "
                "progress will be lost.\n\nClose anyway?",
                "Cancel computation?",
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            )
            confirmed = dlg.ShowModal() == wx.ID_YES
            dlg.Destroy()
            if not confirmed:
                event.Veto()
                return
            self._compute_token = None  # let the thread wind down
        event.Skip()  # proceed with normal close

    # ------------------------------------------------------------------
    # Background computation (one queue for every computable task)

    def _on_compute(self, event):
        """Compute button: queue every missing row of the current task, or stop."""
        if self._compute_token is not None:
            self._stop_compute()
            return
        scope = self._table_scope()
        task, epoch_rejection = scope[:2]
        if not task.computable(self._pipeline, epoch_rejection):
            return
        self._queue_jobs(scope, self._missing_jobs(scope))

    def _missing_jobs(self, scope: tuple) -> list[tuple[tuple, JobSpec]]:
        """``(combo, spec)`` for the displayed rows whose artifact has not been computed yet.

        Rows without a job spec (not computable) are skipped.
        """
        missing = scope[0].missing_status
        combos = [self._row_combo(i) for i in range(self._list.GetItemCount()) if self._list.GetItemText(i, self._layout.status_col) == missing]
        return [(combo, self._job_specs[scope, combo]) for combo in combos if (scope, combo) in self._job_specs]

    def _queue_jobs(self, scope: tuple, jobs: list[tuple[tuple, JobSpec]]) -> None:
        """Add rows to the computation queue

        Jobs are computed sequentially by a single worker thread. When a computation is
        already running the jobs are appended to the queue instead of starting a second
        thread, so that only one thread at a time uses the pipeline.

        Parameters
        ----------
        scope
            Table the rows belong to (see :meth:`_table_scope`).
        jobs
            ``(row combo, job spec)`` pairs to compute. The caller supplies the
            spec, because the row it belongs to is not always the row currently
            displayed under that combo.
        """
        new = [(scope, combo, spec) for combo, spec in jobs]
        if not new:
            return
        with self._job_queue_lock:
            # A combo only names a row within one scope, so dedupe on both. The job the
            # worker is computing right now has left the queue but is not done, so it has
            # to be counted too, or it gets computed a second time.
            queued = {(scope_, combo_) for scope_, combo_, _ in self._job_queue}
            if self._job_in_progress is not None:
                queued.add(self._job_in_progress)
            new = [entry for entry in new if (entry[0], entry[1]) not in queued]
            if not new:
                return
            self._job_queue.extend(new)
            self._n_total += len(new)
        # show the rows as waiting: their artifact is gone (or was never made)
        n_detail = len(self._layout.columns) - self._layout.status_col - 1
        for _, combo, _ in new:
            i = self._displayed_row(scope, combo)
            if i != -1:
                self._set_row_result(i, 'queued', (PLACEHOLDER,) * n_detail)
        if self._compute_token is None:
            self._start_compute()
        else:
            self._update_progress()  # the running worker picks the new jobs up

    def _update_progress(self) -> None:
        """Show the progress of the queue, whose total grows as jobs are added."""
        self._progress_gauge.SetRange(max(self._n_total, 1))
        self._progress_gauge.SetValue(self._n_done)
        self._progress_label.SetLabel(f"{self._n_done} / {self._n_total}")

    def _start_compute(self) -> None:
        """Start the worker thread that computes the queued jobs.

        A no-op while a worker is still alive: after Stop it keeps running until its
        current job returns, and a second thread would put two of them on the pipeline
        at once. Its exit path calls :meth:`_drain_queue`, which starts the queue then.
        """
        if self._worker_active:
            return
        token = object()
        self._compute_token = token
        self._n_done = 0
        with self._job_queue_lock:
            self._n_total = len(self._job_queue)

        self._update_compute_button()
        self._update_progress()
        self._progress_gauge.Show()
        self._progress_label.Show()
        self._refresh_btn.Disable()
        self._task_choice.Disable()
        self._panel.Layout()

        self._worker_active = True
        threading.Thread(target=self._compute_thread, args=(token,), daemon=True).start()

    def _finish_compute_ui(self):
        """Restore toolbar controls after computation ends or is cancelled."""
        self._update_compute_button()
        self._progress_gauge.Hide()
        self._progress_label.Hide()
        self._refresh_btn.Enable()
        self._task_choice.Enable()
        self._panel.Layout()

    def _stop_compute(self):
        """Cancel a running compute thread and immediately restore the UI."""
        if self._compute_token is None:
            return
        self._compute_token = None
        with self._job_queue_lock:
            self._job_queue.clear()
        missing = self._current_task().missing_status
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, self._layout.status_col) in TRANSIENT_STATUS:
                self._list.SetItem(i, self._layout.status_col, missing)
        self._finish_compute_ui()

    def _compute_thread(self, token):
        try:
            self._compute_queued_jobs(token)
        finally:
            # Before the CallAfter, so _drain_queue sees the worker as gone and is
            # free to start the next one.
            self._worker_active = False
            wx.CallAfter(self._on_compute_done, token)

    def _compute_queued_jobs(self, token) -> None:
        while True:
            if token is not self._compute_token:
                break
            with self._job_queue_lock:
                if not self._job_queue:
                    break
                scope, combo, spec = self._job_queue.pop(0)
                self._job_in_progress = (scope, combo)
            task = scope[0]
            wx.CallAfter(self._on_job_computing, token, scope, combo)
            try:
                result = self._compute_job(task, spec, combo)
                # Compute the columns before counting the job as done, so that a
                # failure here goes through the error branch exactly once.
                values = None if result is None else task.result_columns(result)
                self._n_done += 1
                if values is None:  # user declined; the artifact is still missing
                    wx.CallAfter(self._on_job_skipped, token, scope, combo)
                else:
                    wx.CallAfter(self._on_job_computed, token, scope, combo, values)
            except _AbortRequested:
                # The app is exiting; drop the rest so _drain_queue does not start a
                # fresh worker on them while the main loop is being torn down.
                with self._job_queue_lock:
                    self._job_queue.clear()
                break
            except Exception as error:
                self._n_done += 1
                wx.CallAfter(self._on_job_error, token, scope, combo, *_error_dialog_args(error))
            finally:
                with self._job_queue_lock:
                    self._job_in_progress = None

    def _compute_job(
            self,
            task: Task,
            spec: JobSpec,
            combo: tuple,
    ):
        """Compute and cache one job, or ``None`` when the user declined (worker thread).

        A stale ICA file may hold manual component selections, so it is never
        overwritten silently: ``make_job()`` raises and the user decides here. Only
        the ICA task prompts; another task can surface the same error from a nested
        ICA dependency, which re-resolving *its* request cannot fix, so that is
        reported as a plain error instead.

        Everything that reaches the pipeline -- loading the job's data, and saving its
        result -- runs under :attr:`_pipeline_lock`, so it never overlaps a refresh
        walk. The computation itself does not: a :class:`Job` carries its own data, so
        the hour a fit takes is time the refresh thread can use. The lock is also
        dropped for the stale-ICA dialog, which waits on the main thread.

        Parameters
        ----------
        task
            Task the job belongs to.
        spec
            Host-side handle for the artifact.
        combo
            Row combo, used to name the recording in the stale-ICA dialog.
        """
        try:
            with self._pipeline_lock:
                job = spec.make_job()
        except ProtectedArtifactError as error:
            if task.name != 'ica':
                raise
            choice, _ = self._ask_stale_ica(combo[0], error)
            if choice == StaleICADialog.ABORT:
                wx.CallAfter(wx.GetApp().ExitMainLoop)
                raise _AbortRequested()
            elif choice == StaleICADialog.INCORPORATE:
                with self._pipeline_lock:
                    return spec.with_controls(REINDEX_ICA).ctx.load()
            elif choice != StaleICADialog.DELETE:
                return None  # IGNORE, or dialog dismissed: leave the file alone
            Path(error.path).unlink()
            spec = spec.with_controls(ALLOW_PROTECTED_OVERWRITE)
            with self._pipeline_lock:
                job = spec.make_job()
        result = job()
        with self._pipeline_lock:
            return spec.save_result(job, result)

    def _displayed_row(self, scope: tuple, combo: tuple) -> int:
        """Row index for a job, or -1 when its table is not the one on display.

        The same combo can name a row in more than one table, so the scope has to
        match before a row is touched (see :meth:`_table_scope`).
        """
        if scope != self._table_scope():
            return -1
        return self._find_row(combo)

    def _on_job_computing(self, token, scope, combo):
        """Mark a row with ⟳ while its artifact is computed."""
        if token is not self._compute_token:
            return
        i = self._displayed_row(scope, combo)
        if i != -1:
            self._list.SetItem(i, self._layout.status_col, '⟳')

    def _on_job_skipped(self, token, scope, combo):
        """Restore a row after the user declined to compute it.

        Only the ICA task can get here (only its :exc:`pipeline.ProtectedArtifactError` is
        offered to the user), and only by leaving the existing file alone, so the
        row ends up as it would after a refresh: ``'stale'``, with the detail
        columns left at the placeholder ``_queue_jobs`` wrote.
        """
        if token is not self._compute_token:
            return
        i = self._displayed_row(scope, combo)
        if i != -1:
            self._list.SetItem(i, self._layout.status_col, 'stale')
        self._update_progress()
        self._refresh_status_bar()

    def _on_job_computed(self, token, scope, combo, values):
        """Update a row after a successful computation."""
        if token is not self._compute_token:
            return
        i = self._displayed_row(scope, combo)
        if i != -1:
            self._set_row_result(i, scope[0].done_status, values)
        self._update_progress()
        self._refresh_status_bar()

    def _on_job_error(self, token, scope, combo, tb, title, message):
        """Mark a row as errored and show the error dialog, then continue."""
        if token is not self._compute_token:
            return
        i = self._displayed_row(scope, combo)
        if i != -1:
            self._list.SetItem(i, self._layout.status_col, ERROR)
        self._update_progress()
        self._show_error(tb, f"{title}: {' '.join(combo)}", message)

    def _on_compute_done(self, token):
        """Called when the compute thread exits (finished or cancelled)."""
        if token is self._compute_token:
            self._compute_token = None
            self._finish_compute_ui()
            self._refresh_status_bar()
        # Also for a canceled worker, whose UI _stop_compute already restored: jobs
        # queued while it was finishing could not start a thread of their own.
        self._drain_queue()

    def _drain_queue(self) -> None:
        """Start the worker if jobs were queued in the window before it exited."""
        if self._compute_token is not None:
            return
        with self._job_queue_lock:
            pending = bool(self._job_queue)
        if pending:
            self._start_compute()

    def _ask_stale_ica(self, subject: str, error: ProtectedArtifactError, allow_apply_to_all: bool = False) -> tuple[str | None, bool]:
        """Show StaleICADialog and return ``(choice, apply_to_all)``.

        Safe to call from any thread: when called off the main thread the
        dialog is shown via ``CallAfter`` and this blocks until the user
        decides.
        """
        def show() -> tuple[str | None, bool]:
            dlg = StaleICADialog(
                self, subject,
                error.message or str(error),
                error.reason or '',
                allow_apply_to_all=allow_apply_to_all,
            )
            dlg.ShowModal()
            result = (dlg.choice, dlg.apply_to_all)
            dlg.Destroy()
            return result

        if wx.IsMainThread():
            return show()
        result = [None]
        ready = threading.Event()

        def run():
            result[0] = show()
            ready.set()

        wx.CallAfter(run)
        ready.wait()
        return result[0]

    def _ask_ica_channels_changed(self, error: ICAChannelsChangedError) -> bool:
        """Prompt when bad channels changed since the ICA was created.

        Parameters
        ----------
        error
            The error, listing the bad channels then and now.

        Returns ``True`` to delete the ICA, ``False`` to abort.
        """
        dlg = wx.MessageDialog(
            self,
            "Bad channels have changed since creating the ICA. Delete ICA or abort?",
            "Bad channels changed",
            wx.YES_NO | wx.ICON_WARNING,
        )
        dlg.SetExtendedMessage(f"When the ICA was created: {', '.join(error.bads_before) or 'none'}\nNow: {', '.join(error.bads_after) or 'none'}")
        dlg.SetYesNoLabels("Delete ICA", "Abort")
        delete = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        return delete

    def _handle_stale_ica(
            self,
            combo: tuple[str, ...],
            scope: tuple,  # see :meth:`_table_scope`
            error: ProtectedArtifactError,
            choice: str | None,
    ) -> tuple[str, ...]:
        """Apply a stale-ICA ``choice`` during refresh, returning a table row tuple.

        ``combo`` holds the leading key-field columns (subject and any
        session/run columns) that the row is prefixed with.
        """
        task, _, _, raw_name, layout = scope
        if choice == StaleICADialog.ABORT:
            wx.CallAfter(wx.GetApp().ExitMainLoop)
            raise _AbortRequested()
        elif choice == StaleICADialog.DELETE:
            Path(error.path).unlink()
            return task.missing_row(combo, layout)
        elif choice == StaleICADialog.INCORPORATE:
            ica = self._pipeline.load_ica(raw=raw_name, accept_stale=True)
            return (*combo, task.done_status, *task.result_columns(ica))
        elif choice == StaleICADialog.IGNORE:
            ica = mne.preprocessing.read_ica(error.path)
            return (*combo, 'stale', *task.result_columns(ica))
        else:  # dialog dismissed without a choice
            return task.missing_row(combo, layout, 'stale')

    def _fetch_fsaverage(self):
        """Download fsaverage to the experiment's FreeSurfer subjects directory in a thread."""
        subjects_dir = self._pipeline.root / MRI_SDIR
        self._progress_gauge.SetRange(1)  # non-zero range required for Pulse() to animate
        self._progress_gauge.Show()
        self._progress_label.SetLabel("Downloading fsaverage…")
        self._progress_label.Show()
        self._refresh_btn.Disable()
        self._task_choice.Disable()
        self._panel.Layout()
        self.SetStatusText("Downloading fsaverage…")
        self._download_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_download_timer, self._download_timer)
        self._download_timer.Start(100)

        def run():
            try:
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
                wx.CallAfter(self._finish_fsaverage_download, None)
            except Exception:
                wx.CallAfter(self._finish_fsaverage_download, traceback.format_exc())

        threading.Thread(target=run, daemon=True).start()

    def _on_download_timer(self, event):
        self._progress_gauge.Pulse()

    def _finish_fsaverage_download(self, error_tb):
        self._download_timer.Stop()
        self._progress_gauge.Hide()
        self._progress_label.Hide()
        self._refresh_btn.Enable()
        self._task_choice.Enable()
        self._panel.Layout()
        if error_tb:
            self._show_error(error_tb)
        else:
            self._start_refresh()

    def _iter_combos(
            self,
            scope: tuple,  # see :meth:`_table_scope`
    ) -> Iterator[tuple[str, ...]]:
        """Iterate the table's rows, setting the pipeline state for each one.

        First pass of a refresh: which rows the table has follows from the pipeline's
        state model and, for the tasks whose rows are recordings, from which files the
        BIDS dataset actually holds -- neither of which requires opening an artifact.
        The second pass (:meth:`_iter_rows`) is over the list this produced.
        """
        task, _, _, raw_name, layout = scope
        pipeline = self._pipeline
        combos = pipeline.iter(list(layout.key_fields), **self._iter_state(scope))
        # the tasks that show one row per recording skip recordings that were never acquired
        source_name = pipeline._raw.root_source_name(raw_name) if task.name == 'bad_chs' else 'raw'
        skip_missing_recordings = task.name in ('bad_chs', 'coreg')
        for combo in combos:
            if skip_missing_recordings:
                raw_ctx = pipeline._resolve_derivative(raw_input_name(source_name))
                if not raw_ctx.node.exists(raw_ctx):
                    continue
            yield combo
        # Common brain row at the bottom of the MRI table; not a subject, and outside the
        # iteration, so the pipeline state is the one it was left in
        if task.name == 'mri' and pipeline.get('common_brain'):
            yield (COMMON_BRAIN_ROW,)

    def _iter_state(self, scope: tuple) -> dict[str, str]:
        """State every row of the table is resolved under, besides its key fields."""
        task, epoch_rejection, epoch_name, raw_name, _ = scope
        if task.name == 'epoch_rej':
            return {'raw': raw_name, 'epoch': epoch_name, 'epoch_rejection': epoch_rejection}
        elif task.name == 'coreg':
            return {'raw': 'raw'}
        return {}

    def _iter_rows(
            self,
            token: object,
            scope: tuple,  # see :meth:`_table_scope`
            combos: Sequence[tuple[str, ...]],
    ) -> Iterator[tuple[tuple[str, ...], tuple[str, ...], JobSpec | None, Exception | None]]:
        """Yield ``(combo, row, job spec, error)`` for every row of the table.

        Second pass of a refresh, over the rows the first pass found: this is where a
        row's artifact is inspected, which for the ICA task means validating it against
        the raw data it was estimated from -- one raw file per recording. Rows are
        therefore yielded one at a time, so that the caller can show each as soon as it
        resolves. ``spec`` is ``None`` for a row whose artifact the compute queue cannot
        make. A row whose inspection raises is yielded with status :data:`ERROR` and the
        exception as ``error``, so that one broken artifact does not hide the rest of
        the table; ``error`` is ``None`` otherwise.

        Parameters
        ----------
        token
            Refresh this pass belongs to; the pass stops once it is superseded.
        scope
            Table the rows belong to (see :meth:`_table_scope`).
        combos
            Rows to resolve, as :meth:`_iter_combos` yielded them.
        """
        task, epoch_rejection, _, raw_name, layout = scope
        pipeline = self._pipeline
        constants = self._iter_state(scope)
        bulk_choice = None  # set once the user ticks "Apply to all" in the stale-ICA dialog
        with pipeline._temporary_state:
            for combo in _timed_rows(combos, pipeline._log, task.name):
                if token is not self._refresh_token:
                    return
                if combo != (COMMON_BRAIN_ROW,):
                    pipeline.set(**constants, **dict(zip(layout.key_fields, combo)))
                spec = error = None
                try:
                    if task.name == 'bad_chs':
                        source_name = pipeline._raw.root_source_name(raw_name)
                        bads_ctx = pipeline._resolve_derivative(raw_bad_channels_input_name(source_name))
                        try:
                            bads = bads_ctx.load()  # seeds a missing derivatives channels.tsv, so the only failure is bad data
                        except DataError:  # EEG channels without positions
                            row = task.missing_row(combo, layout, ERROR)
                        else:
                            row = (*combo, task.done_status, str(len(bads)))

                    elif task.name == 'ica':
                        ctx = pipeline._resolve_derivative(ica_input_name(raw_name))
                        spec = JobSpec(ctx)
                        status = ctx.load(view='status')
                        if status == 'ok':
                            try:
                                ica = ctx.load()
                                row = (*combo, task.done_status, *task.result_columns(ica))
                            except ProtectedArtifactError as stale:  # not ``error``: the name is unbound after the block, but yielded below
                                if bulk_choice is None:
                                    choice, apply_to_all = self._ask_stale_ica(combo[0], stale, allow_apply_to_all=True)
                                    if apply_to_all:
                                        bulk_choice = choice
                                else:
                                    choice = bulk_choice
                                row = self._handle_stale_ica(combo, scope, stale, choice)
                        elif status == 'missing-ica':
                            row = task.missing_row(combo, layout)
                        else:
                            row = task.missing_row(combo, layout, 'no data')

                    elif task.name == 'epoch_rej':
                        rej = pipeline._epoch_rejection[epoch_rejection]
                        node_name = 'epoch-rejection-input' if isinstance(rej, ManualRejection) else 'epoch-rejection-channel-model'
                        rej_ctx = pipeline._resolve_derivative(node_name)
                        if isinstance(rej, ManualRejection):
                            path = rej_ctx.node.path(rej_ctx)  # an input, with no resolved artifact path
                        else:
                            spec = JobSpec(rej_ctx)
                            # Existence, not spec.is_done: validating (or rebuilding) every
                            # subject's rejection file on each refresh would be far too expensive.
                            path = spec.path
                        if path.exists():
                            ds = load.unpickle(path)
                            row = (*combo, task.done_status, *task.result_columns(ds))
                        else:
                            row = task.missing_row(combo, layout)

                    elif task.name == 'mri':
                        is_common_brain = combo == (COMMON_BRAIN_ROW,)
                        mrisubject = pipeline.get('common_brain' if is_common_brain else 'mrisubject')
                        mri_dir = pipeline.root / MRI_SDIR / mrisubject
                        if not (mri_dir / 'surf' / 'lh.pial').exists():
                            status = COMMON_BRAIN_MISSING if is_common_brain else task.missing_status
                        elif not is_common_brain and is_fake_mri(mri_dir):
                            status = 'template'
                        else:
                            status = task.done_status
                        row = (*combo, mrisubject, status)

                    elif task.name == 'coreg':
                        mrisubject = pipeline.get('mrisubject')
                        trans_ctx = pipeline._resolve_derivative('trans-input')
                        has_trans = trans_ctx.node.exists(trans_ctx)
                        row = (*combo, mrisubject, task.done_status if has_trans else task.missing_status)

                except _AbortRequested:
                    raise
                except Exception as exc:
                    row, spec, error = task.missing_row(combo, layout, ERROR), None, exc
                yield combo, row, spec, error
