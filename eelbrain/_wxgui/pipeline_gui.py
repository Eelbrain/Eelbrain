"""Pipeline supervisor GUI launched by ``eelbrain-gui``."""
import threading

import wx

from .. import load
from .._experiment.epochs import PrimaryEpoch
from .._experiment.preprocessing import RawICA, ica_input_name
from .frame import EelbrainFrame


class PipelineFrame(EelbrainFrame):
    """Top-level window for inspecting and running pipeline setup tasks.

    Shows per-subject status for ICA selection or epoch rejection, and opens
    the corresponding sub-GUI on double-click.
    """

    def __init__(self, pipeline):
        super().__init__(parent=None, title=f"Pipeline: {pipeline.root}")
        self._pipeline = pipeline
        self._refresh_token = None  # replaced each refresh; threads compare identity
        self._tasks = []  # list of (task_type, task_key)

        self._init_ui()
        self._populate_tasks()
        if self._task_choice.GetCount():
            self._task_choice.SetSelection(0)
            self._on_task_changed(None)

        self.SetSize((600, 440))
        self.Centre()

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
        self._epoch_label = wx.StaticText(self._panel, label="Epoch:")
        self._epoch_choice = wx.Choice(self._panel)
        self._epoch_choice.Bind(wx.EVT_CHOICE, self._on_state_changed)
        self._raw_label = wx.StaticText(self._panel, label="Raw:")
        self._raw_choice = wx.Choice(self._panel)
        self._raw_choice.Bind(wx.EVT_CHOICE, self._on_state_changed)

        for widget, border in [
            (self._epoch_label, 14),
            (self._epoch_choice, 4),
            (self._raw_label, 10),
            (self._raw_choice, 4),
        ]:
            toolbar.Add(widget, flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=border)

        toolbar.AddStretchSpacer()
        refresh_btn = wx.Button(self._panel, label="↺", style=wx.BU_EXACTFIT)
        refresh_btn.SetToolTip("Refresh status")
        refresh_btn.Bind(wx.EVT_BUTTON, self._on_refresh)
        toolbar.Add(refresh_btn, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=8)

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

        for w in (self._epoch_label, self._epoch_choice,
                  self._raw_label, self._raw_choice):
            w.Hide()

    # ------------------------------------------------------------------
    # Task / state population

    def _populate_tasks(self):
        for name, pipe in self._pipeline._raw.items():
            if isinstance(pipe, RawICA):
                self._tasks.append(('ica', name))
                self._task_choice.Append(f"ICA: {name}")

        for name, rej in self._pipeline._artifact_rejection.items():
            if rej.get('kind') == 'manual':
                self._tasks.append(('epoch_rej', name))
                self._task_choice.Append(f"Epoch rejection: {name}")

    def _current_task(self):
        idx = self._task_choice.GetSelection()
        if idx == wx.NOT_FOUND or idx >= len(self._tasks):
            return None, None
        return self._tasks[idx]

    def _populate_epoch_raw_choices(self):
        self._epoch_choice.Clear()
        for name, epoch in self._pipeline._epochs.items():
            if isinstance(epoch, PrimaryEpoch):
                self._epoch_choice.Append(name)
        if self._epoch_choice.GetCount():
            self._epoch_choice.SetSelection(0)

        self._raw_choice.Clear()
        for raw in self._pipeline.get_field_values('raw'):
            self._raw_choice.Append(raw)
        default = self._raw_choice.FindString('raw')
        self._raw_choice.SetSelection(default if default != wx.NOT_FOUND else 0)

    # ------------------------------------------------------------------
    # Event handlers

    def _on_task_changed(self, event):
        task_type, _ = self._current_task()
        show_extra = task_type == 'epoch_rej'
        for w in (self._epoch_label, self._epoch_choice,
                  self._raw_label, self._raw_choice):
            w.Show(show_extra)
        if show_extra:
            self._populate_epoch_raw_choices()
        self._panel.Layout()
        self._setup_columns(task_type)
        self._start_refresh()

    def _on_state_changed(self, event):
        self._start_refresh()

    def _on_refresh(self, event):
        self._start_refresh()

    def _on_item_activated(self, event):
        subject = self._list.GetItemText(event.GetIndex(), 0)
        task_type, task_key = self._current_task()
        if task_type is None:
            return
        wx.BeginBusyCursor()
        try:
            if task_type == 'ica':
                frame = self._pipeline.make_ica_selection(subject=subject, raw=task_key)
                if frame is not None:
                    doc = frame.model.doc
                    doc.callbacks.subscribe(
                        'saved',
                        lambda: wx.CallAfter(self._update_ica_row, subject, task_key, doc),
                    )
            elif task_type == 'epoch_rej':
                self._pipeline.make_epoch_selection(
                    subject=subject,
                    rej=task_key,
                    epoch=self._epoch_choice.GetStringSelection(),
                    raw=self._raw_choice.GetStringSelection(),
                )
                # Epoch rejection has no in-memory object to read from,
                # so do a targeted single-subject refresh instead.
                self._start_refresh()
        finally:
            wx.EndBusyCursor()

    # ------------------------------------------------------------------
    # Table management

    def _setup_columns(self, task_type):
        self._list.ClearAll()
        if task_type == 'ica':
            cols = [('Subject', 180), ('Status', 110), ('Components', 110), ('Rejected', 90)]
        else:
            cols = [('Subject', 180), ('Status', 110), ('N total', 90), ('N rejected', 90)]
        for i, (label, width) in enumerate(cols):
            self._list.InsertColumn(i, label, width=width)

    def _populate_table(self, rows, token):
        if token is not self._refresh_token:
            return
        task_type, _ = self._current_task()
        self._list.DeleteAllItems()
        for row in rows:
            idx = self._list.InsertItem(self._list.GetItemCount(), row[0])
            for col, val in enumerate(row[1:], 1):
                self._list.SetItem(idx, col, val)
            if task_type == 'ica' and row[3] == '0':
                self._list.SetItemTextColour(idx, wx.RED)
        self._refresh_status_bar()

    def _update_ica_row(self, subject, task_key, doc):
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
        if task_type == 'ica':
            n_ok = sum(
                1 for i in range(n)
                if self._list.GetItemText(i, 1) == 'selected'
            )
            n_missing = sum(
                1 for i in range(n)
                if self._list.GetItemText(i, 1) == 'no ICA'
            )
            msg = f"{n_ok} / {n} subjects · ICA selected"
            if n_missing:
                msg += f"  ({n_missing} missing ICA file)"
        elif task_type == 'epoch_rej':
            n_ok = sum(
                1 for i in range(n)
                if self._list.GetItemText(i, 1) == 'done'
            )
            msg = f"{n_ok} / {n} subjects · epoch rejection done"
        else:
            msg = ""
        self.SetStatusText(msg)

    # ------------------------------------------------------------------
    # Background status refresh

    def _start_refresh(self):
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
                    if task_type == 'epoch_rej' else None)

        threading.Thread(
            target=self._refresh_thread,
            args=(token, task_type, task_key, epoch_name, raw_name),
            daemon=True,
        ).start()

    def _refresh_thread(self, token, task_type, task_key, epoch_name, raw_name):
        try:
            rows = self._compute_rows(token, task_type, task_key, epoch_name, raw_name)
        except Exception as exc:
            wx.CallAfter(self.SetStatusText, f"Error: {exc}")
            return
        wx.CallAfter(self._populate_table, rows, token)

    def _compute_rows(self, token, task_type, task_key, epoch_name, raw_name):
        pipeline = self._pipeline
        rows = []

        if task_type == 'ica':
            for subject in pipeline:
                if token is not self._refresh_token:
                    break
                ctx = pipeline._resolve_derivative(ica_input_name(task_key))
                status = ctx.load(view='status')
                if status == 'ok':
                    ica = ctx.load()
                    rows.append((subject, 'selected',
                                 str(ica.n_components_), str(len(ica.exclude))))
                elif status == 'missing-ica':
                    rows.append((subject, 'no ICA', '—', '—'))
                else:
                    rows.append((subject, 'no data', '—', '—'))

        elif task_type == 'epoch_rej':
            for subject in pipeline.iter(
                    raw=raw_name, epoch=epoch_name, rej=task_key):
                if token is not self._refresh_token:
                    break
                rej_ctx = pipeline._resolve_derivative('rej-input')
                path = rej_ctx.node.path(rej_ctx)
                if path.exists():
                    ds = load.unpickle(path)
                    n_rej = int((~ds['accept']).sum())
                    rows.append((subject, 'done',
                                 str(ds.n_cases), str(n_rej)))
                else:
                    rows.append((subject, 'missing', '—', '—'))

        return rows
