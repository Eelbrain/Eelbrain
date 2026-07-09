'''Some WxPython utilities'''
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import platform
import re

import mne
import wx
from wx.lib.dialogs import ScrolledMessageDialog

import eelbrain
from eelbrain._wxgui import icons

# store icons once loaded for repeated access
_cache: dict[str, wx.Bitmap] = {}
_iconcache: dict[str, wx.Icon] = {}


def Icon(path: str, asicon: bool = False) -> wx.Bitmap | wx.Icon:
    if asicon:
        if path not in _iconcache:
            _iconcache[path] = icons.catalog[path].GetIcon()
        return _iconcache[path]
    else:
        if path not in _cache:
            _cache[path] = icons.catalog[path].GetBitmap()
        return _cache[path]


def show_text_dialog(parent: wx.Window, text: str, caption: str) -> ScrolledMessageDialog:
    "Create and show a ScrolledMessageDialog"
    style = wx.CAPTION | wx.CLOSE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU
    dlg = ScrolledMessageDialog(parent, text, caption, style=style)
    font = wx.Font(12, wx.MODERN, wx.NORMAL, wx.NORMAL, False, 'Inconsolata')
    dlg.text.SetFont(font)

    n_lines = dlg.text.GetNumberOfLines()
    line_text = dlg.text.GetLineText(0)
    w, h = dlg.text.GetTextExtent(line_text)
    dlg.text.SetSize((w + 100, (h + 3) * n_lines + 50))

    dlg.Fit()
    dlg.Show()
    return dlg


class TracebackDialog(wx.Dialog):
    """Modal dialog showing a full exception traceback with a copy button.

    Without ``message``, the error is presented as an unexpected bug, with
    version info and instructions for filing an issue; pass ``message`` (and
    ``title``) to present an expected error with a specific explanation
    instead, while keeping the traceback available for copying.
    """

    def __init__(
            self,
            parent: wx.Window,
            tb: str,
            title: str = "Error",
            message: str | None = None,
    ) -> None:
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._tb = tb
        bug_report = message is None
        if bug_report:
            message = "An unexpected error occurred. Make sure you are using the latest version of Eelbrain and MNE-Python. Check whether a corresponding issue exists, and if not, submit a new issue including the information below, at https://github.com/Eelbrain/Eelbrain/issues"

        vbox = wx.BoxSizer(wx.VERTICAL)

        header = wx.StaticText(self, label=message)
        header.Wrap(660)
        vbox.Add(header, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        mono = wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE))

        if bug_report:
            # Version/platform section
            self._version_info = (
                f"OS:          {platform.platform()}\n"
                f"Eelbrain:    {eelbrain.__version__}\n"
                f"MNE-Python:  {mne.__version__}"
            )
            version_text = wx.TextCtrl(
                self, value=self._version_info,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP,
            )
            version_text.SetFont(mono)
            vbox.Add(version_text, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

            version_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            copy_version_btn = wx.Button(self, label="Copy Version Info")
            copy_version_btn.Bind(wx.EVT_BUTTON, self._on_copy_version)
            version_btn_sizer.Add(copy_version_btn)
            vbox.Add(version_btn_sizer, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Traceback section
        tb_text = wx.TextCtrl(
            self, value=tb,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP | wx.HSCROLL,
        )
        tb_text.SetFont(mono)
        vbox.Add(tb_text, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        copy_tb_btn = wx.Button(self, label="Copy Traceback")
        copy_tb_btn.Bind(wx.EVT_BUTTON, self._on_copy_tb)
        btn_sizer.Add(copy_tb_btn, flag=wx.RIGHT, border=8)
        btn_sizer.AddStretchSpacer()
        close_btn = wx.Button(self, label="Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(0))
        btn_sizer.Add(close_btn)
        vbox.Add(btn_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        self.SetSizerAndFit(vbox)
        self.SetSize((700, 500))

    def _copy(self, text: str) -> None:
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(text))
            wx.TheClipboard.Close()

    def _on_copy_version(self, event: wx.CommandEvent) -> None:
        self._copy(self._version_info)

    def _on_copy_tb(self, event: wx.CommandEvent) -> None:
        self._copy(self._tb)


class StaleICADialog(wx.Dialog):
    """Ask the user how to handle a stale ICA file.

    After :meth:`ShowModal` returns, read :attr:`choice` for the user's
    decision: one of the :attr:`DELETE`, :attr:`INCORPORATE`, or :attr:`IGNORE`
    class constants, or ``None`` if the dialog was dismissed. When
    ``allow_apply_to_all`` is set, :attr:`apply_to_all` reports whether the
    "Apply to all" checkbox was ticked.
    """

    ABORT = 'abort'
    DELETE = 'delete'
    INCORPORATE = 'incorporate'
    IGNORE = 'ignore'

    def __init__(
            self,
            parent: wx.Window,
            subject: str,
            message: str,
            instructions: str = '',
            allow_apply_to_all: bool = False,
    ) -> None:
        super().__init__(
            parent, title=f"Stale ICA: {subject}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.choice: str | None = None
        self.apply_to_all: bool = False

        vbox = wx.BoxSizer(wx.VERTICAL)

        msg_label = wx.StaticText(self, label=message)
        msg_label.Wrap(540)
        vbox.Add(msg_label, flag=wx.ALL, border=12)

        if instructions:
            instr = wx.TextCtrl(self, value=instructions, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP)
            instr.SetMinSize((-1, 100))
            vbox.Add(instr, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=12)

        help_text = (
            "Choose how to proceed:\n"
            "• Delete: remove the stale ICA file; it will need to be recomputed.\n"
            "• Incorporate: keep the existing ICA and update its record to match the current pipeline.\n"
            "• Ignore: load the existing ICA for display only, without changing its record on disk.\n"
            "• Abort: quit the application without making any changes."
        )
        help_label = wx.StaticText(self, label=help_text)
        help_label.Wrap(540)
        vbox.Add(help_label, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=12)

        self._apply_all_cb: wx.CheckBox | None = None
        if allow_apply_to_all:
            self._apply_all_cb = wx.CheckBox(self, label="Apply this choice to all remaining stale ICA files")
            self._apply_all_cb.SetToolTip("Use the button you click for every other stale ICA file in this refresh, without asking again.")
            vbox.Add(self._apply_all_cb, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=12)

        vbox.Add(wx.StaticLine(self), flag=wx.EXPAND)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        del_btn = wx.Button(self, label="Delete")
        del_btn.SetToolTip("Delete the stale ICA file. The ICA will need to be recomputed.")
        del_btn.Bind(wx.EVT_BUTTON, lambda e: self._choose(self.DELETE))
        btn_sizer.Add(del_btn, flag=wx.RIGHT, border=8)

        inc_btn = wx.Button(self, label="Incorporate")
        inc_btn.SetToolTip("Accept the existing ICA and update its record to match the current pipeline state.")
        inc_btn.Bind(wx.EVT_BUTTON, lambda e: self._choose(self.INCORPORATE))
        btn_sizer.Add(inc_btn, flag=wx.RIGHT, border=8)

        ign_btn = wx.Button(self, label="Ignore")
        ign_btn.SetToolTip("Load the ICA for display only, without modifying its record on disk.")
        ign_btn.Bind(wx.EVT_BUTTON, lambda e: self._choose(self.IGNORE))
        btn_sizer.Add(ign_btn, flag=wx.RIGHT, border=8)

        btn_sizer.AddStretchSpacer()

        abort_btn = wx.Button(self, label="Abort")
        abort_btn.SetToolTip("Quit the application immediately.")
        abort_btn.Bind(wx.EVT_BUTTON, lambda e: self._choose(self.ABORT))
        btn_sizer.Add(abort_btn, border=8)

        vbox.Add(btn_sizer, flag=wx.ALL, border=12)

        self.SetSizerAndFit(vbox)
        self.SetMinSize((420, -1))

    def _choose(self, choice: str) -> None:
        self.choice = choice
        if self._apply_all_cb is not None:
            self.apply_to_all = self._apply_all_cb.GetValue()
        self.EndModal(0)


class FloatValidator(wx.Validator):

    def __init__(self, parent: wx.Window, attr: str) -> None:
        wx.Validator.__init__(self)
        self.parent = parent
        self.attr = attr
        self.value: float | None = None

    def Clone(self) -> 'FloatValidator':
        return FloatValidator(self.parent, self.attr)

    def Validate(self, parent: wx.Window) -> bool:
        ctrl = self.GetWindow()
        value = ctrl.GetValue()
        try:
            self.value = float(value)
        except ValueError:
            msg = wx.MessageDialog(self.parent, f"Can not convert {value!r} to float", "Invalid Entry", wx.OK | wx.ICON_ERROR)
            msg.ShowModal()
            msg.Destroy()
            return False
        else:
            return True

    def TransferToWindow(self) -> bool:
        ctrl = self.GetWindow()
        ctrl.SetValue(str(getattr(self.parent, self.attr)))
        ctrl.SelectAll()
        return True

    def TransferFromWindow(self) -> bool:
        if self.value is None:
            return False
        else:
            setattr(self.parent, self.attr, self.value)
            return True


class REValidator(wx.Validator):
    "Ensure that the value of a text field matches a regular expression"

    def __init__(self, pattern: str, message: str, can_be_empty: bool = False) -> None:
        wx.Validator.__init__(self)
        self.pattern = re.compile(pattern)
        self.message = message
        self.can_be_empty = bool(can_be_empty)

    def Clone(self) -> 'REValidator':
        return REValidator(self.pattern, self.message, self.can_be_empty)

    def Validate(self, win: wx.Window) -> bool:
        ctrl = self.GetWindow()
        text = ctrl.GetValue()

        if len(text.strip()) == 0 and self.can_be_empty:
            return True

        if self.pattern.match(text):
            return True

        wx.MessageBox(self.message.format(value=text), "Error")
        ctrl.SetBackgroundColour("pink")
        ctrl.SetFocus()
        ctrl.Refresh()
        return False

    def TransferToWindow(self) -> bool:
        return True

    def TransferFromWindow(self) -> bool:
        return True
