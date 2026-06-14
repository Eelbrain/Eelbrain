"""Pipeline supervisor GUI launched by ``eelbrain-gui``."""
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

import mne
import wx

from .. import load
from .._exceptions import ConfigurationError, DataError
from .._experiment.derivative_cache import ProtectedArtifactError
from .._experiment.epoch_rejection import ChannelModelRejection, ManualRejection
from .._experiment.epochs import PrimaryEpoch
from .._experiment.exceptions import FileMissingError
from .._experiment.pathing import MRI_SDIR
from .._experiment.preprocessing import RawICA, RawSource, ica_input_name, raw_bad_channels_input_name, raw_input_name
from .._utils.mne_utils import is_fake_mri
from .frame import EelbrainFrame
from .utils import StaleICADialog, TracebackDialog


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
    """Raised in the refresh thread when the user clicks Abort."""


_USER_ERROR_TYPES = (ConfigurationError, DataError, FileMissingError, FileNotFoundError)


def _format_user_error(error: Exception) -> tuple[str, str] | None:
    """Return a dialog title/message for expected pipeline failures."""
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


def _user_error_dialog(error: Exception) -> tuple[str, str]:
    dialog = _format_user_error(error)
    if dialog is None:
        raise TypeError(f"{error!r} is not a user-facing GUI error")
    return dialog


class PipelineFrame(EelbrainFrame):
    """Top-level window for inspecting and running pipeline setup tasks.

    Shows per-subject status for ICA selection or epoch rejection, and opens
    the corresponding sub-GUI on double-click.
    """

    def __init__(self, pipeline) -> None:
        super().__init__(parent=None, title=f"Pipeline: {pipeline.root}")
        self._pipeline = pipeline
        self._refresh_token = None  # replaced each refresh; threads compare identity
        self._compute_token = None  # replaced each make-ICA run; threads compare identity
        self._tasks = []  # list of (task_type, task_key)
        self._bad_chs_iter_fields: list[str] = []  # session/task/run columns for bad_chs

        self._init_ui()
        self._populate_tasks()
        if self._task_choice.GetCount():
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
        self._raw_choice.Bind(wx.EVT_CHOICE, self._on_state_changed)

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

        # Make ICA button + progress (ICA tasks only)
        self._make_ica_btn = wx.Button(self._panel, label="Make ICA", style=wx.BU_EXACTFIT)
        self._make_ica_btn.SetToolTip("Compute ICA for all subjects with missing files")
        self._make_ica_btn.Bind(wx.EVT_BUTTON, self._on_make_ica)
        toolbar.Add(self._make_ica_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)

        # Compute-rejection button (automatic epoch rejection only)
        self._make_rej_btn = wx.Button(self._panel, label="Compute rejection", style=wx.BU_EXACTFIT)
        self._make_rej_btn.SetToolTip("Compute rejection files for all subjects with missing files")
        self._make_rej_btn.Bind(wx.EVT_BUTTON, self._on_make_rejection)
        toolbar.Add(self._make_rej_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=6)

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
        vbox.Add(self._list, proportion=1, flag=wx.EXPAND)

        self._panel.SetSizer(vbox)
        self.CreateStatusBar()

        for w in (self._epoch_rejection_label, self._epoch_rejection_choice,
                  self._epoch_label, self._epoch_choice,
                  self._raw_label, self._raw_choice,
                  self._make_ica_btn, self._make_rej_btn,
                  self._progress_gauge, self._progress_label):
            w.Hide()

    # ------------------------------------------------------------------
    # Task / state population

    def _populate_tasks(self):
        # The task selects the operation; the raw pipe is chosen separately via
        # the Raw dropdown (filtered to the sources/ICA stages for the task).
        if any(isinstance(pipe, RawSource) for pipe in self._pipeline._raw.values()):
            self._tasks.append(('bad_chs', None))
            self._task_choice.Append("Bad channels")

        if any(isinstance(pipe, RawICA) for pipe in self._pipeline._raw.values()):
            self._tasks.append(('ica', None))
            self._task_choice.Append("ICA")

        if any(rej is not None for rej in self._pipeline._epoch_rejection.values()):
            self._tasks.append(('epoch_rej', None))
            self._task_choice.Append("Epoch rejection")

        self._tasks.append(('mri', 'mri'))
        self._task_choice.Append("MRI")
        self._tasks.append(('coreg', 'coreg'))
        self._task_choice.Append("Coregistration")

    def _current_task(self) -> tuple[str | None, str | None]:
        idx = self._task_choice.GetSelection()
        if idx == wx.NOT_FOUND or idx >= len(self._tasks):
            return None, None
        return self._tasks[idx]

    def _populate_epoch_choices(self):
        self._epoch_choice.Clear()
        for name, epoch in self._pipeline._epochs.items():
            if isinstance(epoch, PrimaryEpoch):
                self._epoch_choice.Append(name)
        if self._epoch_choice.GetCount():
            self._epoch_choice.SetSelection(0)

    def _populate_raw_choices(self, task_type: str):
        """Fill the Raw dropdown with the pipes relevant to ``task_type``."""
        self._raw_choice.Clear()
        if task_type == 'ica':
            names = [name for name, pipe in self._pipeline._raw.items() if isinstance(pipe, RawICA)]
        else:  # bad_chs / epoch_rej: any raw stage (source derived where needed)
            names = list(self._pipeline.get_field_values('raw'))
        for name in names:
            self._raw_choice.Append(name)
        default = self._raw_choice.FindString('raw')
        self._raw_choice.SetSelection(default if default != wx.NOT_FOUND else 0)

    def _populate_epoch_rejection_choices(self):
        self._epoch_rejection_choice.Clear()
        for name, rej in self._pipeline._epoch_rejection.items():
            if rej is not None:
                self._epoch_rejection_choice.Append(name)
        if self._epoch_rejection_choice.GetCount():
            self._epoch_rejection_choice.SetSelection(0)

    def _current_epoch_rejection(self) -> str | None:
        return self._epoch_rejection_choice.GetStringSelection() or None

    def _update_rejection_button(self):
        """Show the 'Compute rejection' button only for an automatic rejection."""
        task_type, _ = self._current_task()
        name = self._current_epoch_rejection()
        is_auto = (task_type == 'epoch_rej' and name is not None
                   and isinstance(self._pipeline._epoch_rejection[name], ChannelModelRejection))
        self._make_rej_btn.Show(is_auto)
        self._panel.Layout()

    def _on_epoch_rejection_changed(self, event):
        self._update_rejection_button()
        self._start_refresh()

    # ------------------------------------------------------------------
    # Event handlers

    def _on_task_changed(self, event):
        task_type, task_key = self._current_task()
        self._stop_compute()
        if task_type == 'bad_chs':
            p = self._pipeline
            extra = []
            if len(p._sessions) > 1:
                extra.append('session')
            if len(p._tasks) > 1:
                extra.append('task')
            if len(p._runs) > 1:
                extra.append('run')
            self._bad_chs_iter_fields = extra
        show_epoch = task_type == 'epoch_rej'
        show_raw = task_type in ('epoch_rej', 'bad_chs', 'ica')
        self._epoch_rejection_label.Show(show_epoch)
        self._epoch_rejection_choice.Show(show_epoch)
        self._epoch_label.Show(show_epoch)
        self._epoch_choice.Show(show_epoch)
        self._raw_label.Show(show_raw)
        self._raw_choice.Show(show_raw)
        if show_epoch:
            self._populate_epoch_rejection_choices()
            self._populate_epoch_choices()
        if show_raw:
            self._populate_raw_choices(task_type)
        self._make_ica_btn.Show(task_type == 'ica')
        self._update_rejection_button()
        self._panel.Layout()
        self._setup_columns(task_type)
        self._start_refresh()

    def _on_state_changed(self, event):
        self._start_refresh()

    def _on_refresh(self, event):
        self._start_refresh()

    def _on_item_activated(self, event):
        """Row double-click"""
        idx = event.GetIndex()
        subject = self._list.GetItemText(idx, 0)
        task_type, task_key = self._current_task()
        if task_type is None:
            return
        wx.BeginBusyCursor()
        try:
            if task_type == 'bad_chs':
                raw_name = self._raw_choice.GetStringSelection()
                state = {'subject': subject}
                for col, field in enumerate(self._bad_chs_iter_fields, start=1):
                    state[field] = self._list.GetItemText(idx, col)
                frame = self._pipeline.make_bad_channels_selection(raw=raw_name, **state)
                if frame is not None:
                    doc = frame.model.doc
                    doc.callbacks.subscribe(
                        'saved',
                        lambda: wx.CallAfter(self._start_refresh),
                    )
            elif task_type == 'ica':
                raw_name = self._raw_choice.GetStringSelection()
                frame = self._pipeline.make_ica_selection(subject=subject, raw=raw_name)
                if frame is not None:
                    doc = frame.model.doc
                    doc.callbacks.subscribe(
                        'saved',
                        lambda: wx.CallAfter(self._update_ica_row, subject, doc),
                    )
            elif task_type == 'epoch_rej':
                name = self._current_epoch_rejection()
                if name is not None:
                    # opens an editable GUI for ManualRejection, read-only for an
                    # automatically generated rejection
                    self._pipeline.make_epoch_rejection(
                        subject=subject,
                        epoch_rejection=name,
                        epoch=self._epoch_choice.GetStringSelection(),
                        raw=self._raw_choice.GetStringSelection(),
                    )
                    # Epoch rejection has no in-memory object to read from,
                    # so do a targeted single-subject refresh instead.
                    self._start_refresh()
            elif task_type == 'mri':
                self._on_mri_activated(idx, subject)
            elif task_type == 'coreg':
                self._on_coreg_activated(idx)
        except _USER_ERROR_TYPES as error:
            self._show_user_error(*_user_error_dialog(error))
        finally:
            wx.EndBusyCursor()

    def _on_mri_activated(self, row_idx: int, subject: str):
        """Handle double-click on an MRI row."""
        mrisubject = self._list.GetItemText(row_idx, 1)
        status = self._list.GetItemText(row_idx, 2)
        subjects_dir = str(self._pipeline.root / MRI_SDIR)
        common_brain = self._pipeline.get('common_brain')

        if subject == '(common brain)':
            if status == 'missing':
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
        elif status == 'no MRI':
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

    def _setup_columns(self, task_type):
        self._list.ClearAll()
        if task_type == 'bad_chs':
            cols = [('Subject', 180)]
            for f in self._bad_chs_iter_fields:
                cols.append((f.title(), 90))
            cols += [('Status', 110), ('N bad', 90)]
        elif task_type == 'ica':
            cols = [('Subject', 180), ('Status', 110), ('Components', 110), ('Rejected', 90)]
        elif task_type == 'mri':
            cols = [('Subject', 180), ('MRI subject', 170), ('Status', 130)]
        elif task_type == 'coreg':
            cols = [('Subject', 140), ('Session', 80), ('MRI subject', 140), ('Status', 110)]
        else:
            cols = [('Subject', 180), ('Status', 110), ('N total', 90), ('N rejected', 90)]
        for i, (label, width) in enumerate(cols):
            self._list.InsertColumn(i, label, width=width)

    def _populate_table(self, rows: list[tuple[str, ...]], token: object) -> None:
        if token is not self._refresh_token:
            return
        task_type, _ = self._current_task()
        self._list.DeleteAllItems()
        grey = wx.Colour(150, 150, 150)
        for row in rows:
            idx = self._list.InsertItem(self._list.GetItemCount(), row[0])
            for col, val in enumerate(row[1:], 1):
                self._list.SetItem(idx, col, val)
            if task_type == 'bad_chs':
                status = row[-2]  # status is always second-to-last
                if status in ('no data', 'no file'):
                    self._list.SetItemTextColour(idx, grey)
            elif task_type == 'ica':
                if row[1] == 'selected' and row[3] == '0':
                    self._list.SetItemTextColour(idx, wx.RED)
            elif task_type == 'mri':
                if row[2] == 'no MRI':
                    self._list.SetItemTextColour(idx, wx.RED)
                elif row[0] == '(common brain)':
                    self._list.SetItemTextColour(idx, grey)
            elif task_type == 'coreg':
                if row[3] == 'missing':
                    self._list.SetItemTextColour(idx, wx.RED)
        self._refresh_status_bar()

    def _update_ica_row(self, subject: str, doc) -> None:
        """Update a single ICA row from the already-in-memory document (no disk I/O)."""
        n_comp = doc.ica.n_components_
        n_excl = len(doc.ica.exclude)
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 0) == subject:
                self._list.SetItem(i, 1, 'selected')
                self._list.SetItem(i, 2, str(n_comp))
                self._list.SetItem(i, 3, str(n_excl))
                colour = wx.RED if n_excl == 0 else wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOXTEXT)
                self._list.SetItemTextColour(i, colour)
                break
        self._refresh_status_bar()

    def _refresh_status_bar(self):
        """Recompute the status bar summary from the current table contents."""
        task_type, _ = self._current_task()
        n = self._list.GetItemCount()
        if task_type == 'bad_chs':
            n_done = sum(1 for i in range(n) if self._list.GetItemText(i, 1) == 'done')
            n_missing = sum(1 for i in range(n) if self._list.GetItemText(i, 1) == 'no file')
            msg = f"{n_done} / {n} subjects · bad channels defined"
            if n_missing:
                msg += f"  ({n_missing} missing channels.tsv)"
            self.SetStatusText(msg)
            return
        if task_type == 'ica':
            n_ok = sum(1 for i in range(n) if self._list.GetItemText(i, 1) == 'selected')
            n_missing = sum(1 for i in range(n) if self._list.GetItemText(i, 1) == 'no ICA')
            msg = f"{n_ok} / {n} subjects · ICA selected"
            if n_missing:
                msg += f"  ({n_missing} missing ICA file)"
        elif task_type == 'epoch_rej':
            n_ok = sum(1 for i in range(n) if self._list.GetItemText(i, 1) == 'done')
            msg = f"{n_ok} / {n} subjects · epoch rejection done"
        elif task_type == 'mri':
            # exclude the common brain row from subject counts
            subject_rows = [i for i in range(n) if self._list.GetItemText(i, 0) != '(common brain)']
            n_sub = len(subject_rows)
            n_ok = sum(1 for i in subject_rows if self._list.GetItemText(i, 2) in ('ok', 'template'))
            n_missing = sum(1 for i in subject_rows if self._list.GetItemText(i, 2) == 'no MRI')
            msg = f"{n_ok} / {n_sub} subjects · MRI available"
            if n_missing:
                msg += f"  ({n_missing} missing)"
        elif task_type == 'coreg':
            n_ok = sum(1 for i in range(n) if self._list.GetItemText(i, 3) == 'ok')
            n_missing = sum(1 for i in range(n) if self._list.GetItemText(i, 3) == 'missing')
            msg = f"{n_ok} / {n} sessions · coregistration done"
            if n_missing:
                msg += f"  ({n_missing} missing)"
        else:
            msg = ""
        self.SetStatusText(msg)

    # ------------------------------------------------------------------
    # Background status refresh

    def _start_refresh(self) -> None:
        task_type, task_key = self._current_task()
        if task_type is None:
            return
        token = object()
        self._refresh_token = token
        self._list.DeleteAllItems()
        self.SetStatusText("Loading…")

        epoch_name = (self._epoch_choice.GetStringSelection()
                      if task_type == 'epoch_rej' else None)
        raw_name = (self._raw_choice.GetStringSelection()
                    if task_type in ('epoch_rej', 'bad_chs', 'ica') else None)

        if task_type == 'epoch_rej':
            # carry the selected rejection name through as task_key
            task_key = self._current_epoch_rejection()
            if task_key is None:
                self.SetStatusText("No epoch rejection defined")
                return
            if not epoch_name:
                self.SetStatusText("No epochs defined")
                return

        threading.Thread(
            target=self._refresh_thread,
            args=(token, task_type, task_key, epoch_name, raw_name),
            daemon=True,
        ).start()

    def _refresh_thread(
            self,
            token: object,
            task_type: str,
            task_key: str,
            epoch_name: str | None,
            raw_name: str | None,
    ) -> None:
        try:
            rows = self._compute_rows(token, task_type, task_key, epoch_name, raw_name)
        except _AbortRequested:
            return  # app exit already scheduled
        except _USER_ERROR_TYPES as error:
            wx.CallAfter(self._show_user_error, *_user_error_dialog(error))
            return
        except Exception:
            tb = traceback.format_exc()
            wx.CallAfter(self._show_error, tb)
            return
        wx.CallAfter(self._populate_table, rows, token)

    def _show_user_error(self, title: str, message: str):
        self.SetStatusText("Error")
        wx.MessageBox(message, title, wx.OK | wx.ICON_ERROR, self)

    def _show_error(self, tb: str):
        self.SetStatusText("Error")
        dlg = TracebackDialog(self, tb)
        dlg.ShowModal()
        dlg.Destroy()

    def _on_close(self, event):
        if self._compute_token is not None:
            dlg = wx.MessageDialog(
                self,
                "ICA computation is in progress. "
                "Closing this window will cancel it and the current subject's "
                "progress will be lost.\n\nClose anyway?",
                "Cancel ICA computation?",
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
    # Make-ICA background computation

    def _on_make_ica(self, event):
        if self._compute_token is not None:
            self._stop_compute()
            return

        task_type, _ = self._current_task()
        if task_type != 'ica':
            return
        raw_name = self._raw_choice.GetStringSelection()

        subjects = [
            self._list.GetItemText(i, 0)
            for i in range(self._list.GetItemCount())
            if self._list.GetItemText(i, 1) == 'no ICA'
        ]
        if not subjects:
            return

        # Invalidate any running refresh so both threads don't touch the
        # pipeline concurrently.
        self._refresh_token = object()

        token = object()
        self._compute_token = token
        n_total = len(subjects)

        self._make_ica_btn.SetLabel("Stop")
        self._progress_gauge.SetRange(n_total)
        self._progress_gauge.SetValue(0)
        self._progress_gauge.Show()
        self._progress_label.SetLabel(f"0 / {n_total}")
        self._progress_label.Show()
        self._refresh_btn.Disable()
        self._task_choice.Disable()
        self._panel.Layout()

        threading.Thread(
            target=self._make_ica_thread,
            args=(token, raw_name, subjects),
            daemon=True,
        ).start()

    def _finish_compute_ui(self):
        """Restore toolbar controls after computation ends or is cancelled."""
        self._make_ica_btn.SetLabel("Make ICA")
        self._make_rej_btn.SetLabel("Compute rejection")
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
        task_type, _ = self._current_task()
        missing = 'no ICA' if task_type == 'ica' else 'missing'
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 1) == '⟳':
                self._list.SetItem(i, 1, missing)
        self._finish_compute_ui()

    def _make_ica_thread(self, token, raw_name, subjects):
        pipeline = self._pipeline
        n_done = 0
        n_total = len(subjects)
        for subject in subjects:
            if token is not self._compute_token:
                break
            wx.CallAfter(self._on_subject_computing, token, subject)
            try:
                # make_ica computes and saves the ICA file; it also leaves the
                # pipeline context set to this subject so ctx.load() works below.
                pipeline.make_ica(subject=subject, raw=raw_name)
                ctx = pipeline._resolve_derivative(ica_input_name(raw_name))
                ica = ctx.load()
                n_done += 1
                wx.CallAfter(
                    self._on_subject_computed, token, subject,
                    str(ica.n_components_), str(len(ica.exclude)),
                    n_done, n_total,
                )
            except _USER_ERROR_TYPES as error:
                title, message = _user_error_dialog(error)
                n_done += 1
                wx.CallAfter(self._on_subject_user_error, token, subject, title, message, n_done, n_total)
            except Exception:
                tb = traceback.format_exc()
                n_done += 1
                wx.CallAfter(self._on_subject_error, token, subject, tb, n_done, n_total)
        wx.CallAfter(self._on_make_ica_done, token)

    def _on_subject_computing(self, token, subject):
        """Mark a subject's row with ⟳ while its ICA is being computed."""
        if token is not self._compute_token:
            return
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 0) == subject:
                self._list.SetItem(i, 1, '⟳')
                break

    def _on_subject_computed(self, token, subject, n_comp, n_excl, n_done, n_total):
        """Update a row after successful ICA computation."""
        if token is not self._compute_token:
            return
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 0) == subject:
                self._list.SetItem(i, 1, 'selected')
                self._list.SetItem(i, 2, n_comp)
                self._list.SetItem(i, 3, n_excl)
                colour = (wx.RED if n_excl == '0'
                          else wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOXTEXT))
                self._list.SetItemTextColour(i, colour)
                break
        self._progress_gauge.SetValue(n_done)
        self._progress_label.SetLabel(f"{n_done} / {n_total}")
        self._refresh_status_bar()

    def _on_subject_user_error(self, token, subject, title, message, n_done, n_total):
        """Mark a row after an expected input/configuration failure."""
        if token is not self._compute_token:
            return
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 0) == subject:
                self._list.SetItem(i, 1, 'error')
                break
        self._progress_gauge.SetValue(n_done)
        self._progress_label.SetLabel(f"{n_done} / {n_total}")
        wx.MessageBox(f"{subject}: {message}", title, wx.OK | wx.ICON_ERROR, self)

    def _on_subject_error(self, token, subject, tb, n_done, n_total):
        """Mark a row as errored after a failed ICA computation."""
        if token is not self._compute_token:
            return
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 0) == subject:
                self._list.SetItem(i, 1, 'error')
                break
        self._progress_gauge.SetValue(n_done)
        self._progress_label.SetLabel(f"{n_done} / {n_total}")
        # Show the traceback so the user knows what went wrong, then continue.
        dlg = TracebackDialog(self, tb)
        dlg.ShowModal()
        dlg.Destroy()

    def _on_make_ica_done(self, token):
        """Called when the make-ICA thread exits (finished or cancelled)."""
        if token is not self._compute_token:
            return  # _stop_compute already cleaned up
        self._compute_token = None
        self._finish_compute_ui()
        self._refresh_status_bar()

    # ------------------------------------------------------------------
    # Compute-rejection background computation (automatic rejection)

    def _on_make_rejection(self, event):
        if self._compute_token is not None:
            self._stop_compute()
            return

        task_type, _ = self._current_task()
        name = self._current_epoch_rejection()
        if task_type != 'epoch_rej' or name is None:
            return
        if not isinstance(self._pipeline._epoch_rejection[name], ChannelModelRejection):
            return

        epoch_name = self._epoch_choice.GetStringSelection()
        raw_name = self._raw_choice.GetStringSelection()
        subjects = [
            self._list.GetItemText(i, 0)
            for i in range(self._list.GetItemCount())
            if self._list.GetItemText(i, 1) == 'missing'
        ]
        if not subjects:
            return

        self._refresh_token = object()
        token = object()
        self._compute_token = token
        n_total = len(subjects)

        self._make_rej_btn.SetLabel("Stop")
        self._progress_gauge.SetRange(n_total)
        self._progress_gauge.SetValue(0)
        self._progress_gauge.Show()
        self._progress_label.SetLabel(f"0 / {n_total}")
        self._progress_label.Show()
        self._refresh_btn.Disable()
        self._task_choice.Disable()
        self._panel.Layout()

        threading.Thread(
            target=self._make_rejection_thread,
            args=(token, name, epoch_name, raw_name, subjects),
            daemon=True,
        ).start()

    def _make_rejection_thread(self, token, name, epoch_name, raw_name, subjects):
        pipeline = self._pipeline
        n_done = 0
        n_total = len(subjects)
        for subject in subjects:
            if token is not self._compute_token:
                break
            wx.CallAfter(self._on_subject_computing, token, subject)
            try:
                pipeline.set(subject=subject, epoch_rejection=name, epoch=epoch_name, raw=raw_name)
                ctx = pipeline._resolve_derivative('epoch-rejection-channel-model')
                rej_ds = ctx.load()
                n_rej = int((~rej_ds['accept']).sum())
                n_done += 1
                wx.CallAfter(self._on_subject_rejection_computed, token, subject, str(rej_ds.n_cases), str(n_rej), n_done, n_total)
            except _USER_ERROR_TYPES as error:
                title, message = _user_error_dialog(error)
                n_done += 1
                wx.CallAfter(self._on_subject_user_error, token, subject, title, message, n_done, n_total)
            except Exception:
                tb = traceback.format_exc()
                n_done += 1
                wx.CallAfter(self._on_subject_error, token, subject, tb, n_done, n_total)
        wx.CallAfter(self._on_make_rejection_done, token)

    def _on_subject_rejection_computed(self, token, subject, n_epochs, n_rej, n_done, n_total):
        """Update a row after a successful rejection computation."""
        if token is not self._compute_token:
            return
        for i in range(self._list.GetItemCount()):
            if self._list.GetItemText(i, 0) == subject:
                self._list.SetItem(i, 1, 'done')
                self._list.SetItem(i, 2, n_epochs)
                self._list.SetItem(i, 3, n_rej)
                break
        self._progress_gauge.SetValue(n_done)
        self._progress_label.SetLabel(f"{n_done} / {n_total}")
        self._refresh_status_bar()

    def _on_make_rejection_done(self, token):
        """Called when the compute-rejection thread exits (finished or cancelled)."""
        if token is not self._compute_token:
            return  # _stop_compute already cleaned up
        self._compute_token = None
        self._finish_compute_ui()
        self._refresh_status_bar()

    def _handle_stale_ica(self, subject: str, error: ProtectedArtifactError, pipeline, raw_name: str) -> tuple:
        """Show StaleICADialog on the main thread; block until the user decides.

        Returns a table row tuple for the subject.
        """
        result = [None]
        ready = threading.Event()

        def show():
            dlg = StaleICADialog(
                self, subject,
                error.message or str(error),
                error.reason or '',
            )
            dlg.ShowModal()
            result[0] = dlg.choice
            dlg.Destroy()
            ready.set()

        wx.CallAfter(show)
        ready.wait()
        choice = result[0]

        if choice == StaleICADialog.ABORT:
            wx.CallAfter(wx.GetApp().ExitMainLoop)
            raise _AbortRequested()
        elif choice == StaleICADialog.DELETE:
            Path(error.path).unlink()
            return (subject, 'no ICA', '—', '—')
        elif choice == StaleICADialog.INCORPORATE:
            ica = pipeline.load_ica(raw=raw_name, accept_stale=True)
            return (subject, 'selected', str(ica.n_components_), str(len(ica.exclude)))
        elif choice == StaleICADialog.IGNORE:
            ica = mne.preprocessing.read_ica(error.path)
            return (subject, 'stale', str(ica.n_components_), str(len(ica.exclude)))
        else:  # dialog dismissed without a choice
            return (subject, 'stale', '—', '—')

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

    def _compute_rows(
            self,
            token: object,
            task_type: str,
            task_key: str,
            epoch_name: str | None,
            raw_name: str | None,
    ) -> list[tuple[str, ...]]:
        pipeline = self._pipeline
        rows = []

        if task_type == 'bad_chs':
            source_name = pipeline._raw.root_source_name(raw_name)
            extra = self._bad_chs_iter_fields
            iter_fields = ('subject',) + tuple(extra)
            iter_arg = iter_fields[0] if len(iter_fields) == 1 else list(iter_fields)
            for combo in pipeline.iter(iter_arg):
                if token is not self._refresh_token:
                    break
                if isinstance(combo, str):
                    combo = (combo,)
                raw_ctx = pipeline._resolve_derivative(raw_input_name(source_name))
                if not raw_ctx.node.exists(raw_ctx):
                    rows.append(combo + ('no data', '—'))
                    continue
                bads_ctx = pipeline._resolve_derivative(raw_bad_channels_input_name(source_name))
                tsv_path = bads_ctx.node.path(bads_ctx)
                if not tsv_path.exists():
                    rows.append(combo + ('no file', '—'))
                else:
                    bads = bads_ctx.load()
                    rows.append(combo + ('done', str(len(bads))))

        elif task_type == 'ica':
            for subject in pipeline:
                if token is not self._refresh_token:
                    break
                ctx = pipeline._resolve_derivative(ica_input_name(raw_name))
                status = ctx.load(view='status')
                if status == 'ok':
                    try:
                        ica = ctx.load()
                        rows.append((subject, 'selected',
                                     str(ica.n_components_), str(len(ica.exclude))))
                    except ProtectedArtifactError as error:
                        row = self._handle_stale_ica(subject, error, pipeline, task_key)
                        rows.append(row)
                elif status == 'missing-ica':
                    rows.append((subject, 'no ICA', '—', '—'))
                else:
                    rows.append((subject, 'no data', '—', '—'))

        elif task_type == 'epoch_rej':
            rej = pipeline._epoch_rejection[task_key]
            node_name = 'epoch-rejection-input' if isinstance(rej, ManualRejection) else 'epoch-rejection-channel-model'
            for subject in pipeline.iter(
                    raw=raw_name, epoch=epoch_name, epoch_rejection=task_key):
                if token is not self._refresh_token:
                    break
                rej_ctx = pipeline._resolve_derivative(node_name)
                path = rej_ctx.node.path(rej_ctx)
                if path.exists():
                    ds = load.unpickle(path)
                    n_rej = int((~ds['accept']).sum())
                    rows.append((subject, 'done',
                                 str(ds.n_cases), str(n_rej)))
                else:
                    rows.append((subject, 'missing', '—', '—'))

        elif task_type == 'mri':
            subjects_dir = pipeline.root / MRI_SDIR
            for subject in pipeline:
                if token is not self._refresh_token:
                    break
                mrisubject = pipeline.get('mrisubject')
                has_recon = (subjects_dir / mrisubject / 'surf' / 'lh.pial').exists()
                if has_recon:
                    status = 'template' if is_fake_mri(subjects_dir / mrisubject) else 'ok'
                else:
                    status = 'no MRI'
                rows.append((subject, mrisubject, status))
            # Common brain row at the bottom
            common_brain = pipeline.get('common_brain')
            if common_brain:
                has_cb = (subjects_dir / common_brain / 'surf' / 'lh.pial').exists()
                rows.append(('(common brain)', common_brain, 'ok' if has_cb else 'missing'))

        elif task_type == 'coreg':
            raw_input = raw_input_name('raw')
            for subject, session in pipeline.iter(('subject', 'session'), raw='raw'):
                if token is not self._refresh_token:
                    break
                raw_ctx = pipeline._resolve_derivative(raw_input)
                if not raw_ctx.node.exists(raw_ctx):
                    continue
                mrisubject = pipeline.get('mrisubject')
                trans_ctx = pipeline._resolve_derivative('trans-input')
                has_trans = trans_ctx.node.exists(trans_ctx)
                rows.append((subject, session, mrisubject, 'ok' if has_trans else 'missing'))

        return rows
