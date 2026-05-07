'''Some WxPython utilities'''
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import re

import wx
from wx.lib.dialogs import ScrolledMessageDialog

from eelbrain._wxgui import icons

# store icons once loaded for repeated access
_cache = {}
_iconcache = {}


def Icon(path, asicon=False):
    if asicon:
        if path not in _iconcache:
            _iconcache[path] = icons.catalog[path].GetIcon()
        return _iconcache[path]
    else:
        if path not in _cache:
            _cache[path] = icons.catalog[path].GetBitmap()
        return _cache[path]


def show_text_dialog(parent, text, caption):
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

    Intended for surfacing unexpected errors to the user with enough context
    to file a bug report.
    """

    def __init__(self, parent, tb: str):
        super().__init__(parent, title="Error", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._tb = tb

        vbox = wx.BoxSizer(wx.VERTICAL)

        text = wx.TextCtrl(
            self, value=tb,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP | wx.HSCROLL,
        )
        text.SetFont(wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE)))
        vbox.Add(text, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        copy_btn = wx.Button(self, label="Copy Traceback")
        copy_btn.Bind(wx.EVT_BUTTON, self._on_copy)
        btn_sizer.Add(copy_btn, flag=wx.RIGHT, border=8)
        btn_sizer.AddStretchSpacer()
        close_btn = wx.Button(self, label="Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(0))
        btn_sizer.Add(close_btn)
        vbox.Add(btn_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)

        self.SetSizerAndFit(vbox)
        self.SetSize((700, 400))

    def _on_copy(self, event):
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(self._tb))
            wx.TheClipboard.Close()


class StaleICADialog(wx.Dialog):
    """Ask the user how to handle a stale ICA file.

    After :meth:`ShowModal` returns, read :attr:`choice` for the user's
    decision: one of the :attr:`DELETE`, :attr:`INCORPORATE`, or :attr:`IGNORE`
    class constants, or ``None`` if the dialog was dismissed.
    """

    ABORT = 'abort'
    DELETE = 'delete'
    INCORPORATE = 'incorporate'
    IGNORE = 'ignore'

    def __init__(self, parent, subject: str, message: str, instructions: str = ''):
        super().__init__(
            parent, title=f"Stale ICA: {subject}",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.choice = None

        vbox = wx.BoxSizer(wx.VERTICAL)

        msg_label = wx.StaticText(self, label=message)
        msg_label.Wrap(540)
        vbox.Add(msg_label, flag=wx.ALL, border=12)

        if instructions:
            instr = wx.TextCtrl(self, value=instructions, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_WORDWRAP)
            instr.SetMinSize((-1, 100))
            vbox.Add(instr, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=12)

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

    def _choose(self, choice: str):
        self.choice = choice
        self.EndModal(0)


class FloatValidator(wx.Validator):

    def __init__(self, parent, attr):
        wx.Validator.__init__(self)
        self.parent = parent
        self.attr = attr
        self.value = None

    def Clone(self):
        return FloatValidator(self.parent, self.attr)

    def Validate(self, parent):
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

    def TransferToWindow(self):
        ctrl = self.GetWindow()
        ctrl.SetValue(str(getattr(self.parent, self.attr)))
        ctrl.SelectAll()
        return True

    def TransferFromWindow(self):
        if self.value is None:
            return False
        else:
            setattr(self.parent, self.attr, self.value)
            return True


class REValidator(wx.Validator):
    "Ensure that the value of a text field matches a regular expression"

    def __init__(self, pattern, message, can_be_empty=False):
        wx.Validator.__init__(self)
        self.pattern = re.compile(pattern)
        self.message = message
        self.can_be_empty = bool(can_be_empty)

    def Clone(self):
        return REValidator(self.pattern, self.message, self.can_be_empty)

    def Validate(self, win):
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
#         else:
#             ctrl.SetBackgroundColour(
#                 wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
#             ctrl.Refresh()
#             return True

    def TransferToWindow(self):
        return True

    def TransferFromWindow(self):
        return True
