import wx

from .._utils import IS_OSX
from . import ID
from .utils import Icon

FOCUS_UI_UPDATE_FUNC_NAMES = {
    wx.ID_COPY: 'CanCopy',
    ID.COPY_AS_PNG: 'CanCopyPNG',
    wx.ID_CUT: 'CanCut',
    wx.ID_PASTE: 'CanPaste',
}


class EelbrainWindow:
    # Frame subclass to support UI Update

    def OnWindowIconize(self, event: wx.CommandEvent) -> None:
        self.Iconize()

    def OnWindowZoom(self, event: wx.CommandEvent) -> None:
        self.Maximize()

    def OnUpdateUIBackward(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUIClear(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUIClose(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(True)

    def OnUpdateUIDown(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(True)

    def OnUpdateUIDrawCrosshairs(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)
        event.Check(False)

    def OnUpdateUIForward(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUIFocus(self, event: wx.UpdateUIEvent) -> None:
        func_name = FOCUS_UI_UPDATE_FUNC_NAMES[event.GetId()]
        win = self.FindFocus()
        func = getattr(win, func_name, None)
        if func is None:
            func = getattr(self, func_name, None)
            if func is None:
                event.Enable(False)
                return
        event.Enable(func())

    def OnUpdateUIOpen(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUIRedo(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUISave(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUISaveAs(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUISetLayout(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUISetMarkedChannels(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUISetVLim(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUISetTime(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUITools(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(hasattr(self, 'MakeToolsMenu'))

    def OnUpdateUIUndo(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)

    def OnUpdateUIUp(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(False)


class NavigableFrame:
    """Mixin for frames with backward/forward (and optionally up/down) navigation.

    Subclasses must implement ``CanBackward`` and ``CanForward``, and
    ``CanUp``/``CanDown`` if ``AddNavigationButtons`` is called with
    ``up_down=True``.  The mixin overrides the ``OnUpdateUI*`` stubs from
    ``EelbrainWindow`` and provides ``AddNavigationButtons`` to wire up toolbar
    buttons in one call instead of duplicating four lines per frame.
    """

    def CanBackward(self) -> bool:
        raise NotImplementedError

    def CanForward(self) -> bool:
        raise NotImplementedError

    def CanUp(self) -> bool:
        raise NotImplementedError

    def CanDown(self) -> bool:
        raise NotImplementedError

    def AddNavigationButtons(self, tb: wx.ToolBar, *, up_down: bool = False) -> None:
        """Add backward/forward (and optionally up/down) buttons to *tb*.

        Also binds the corresponding ``EVT_TOOL`` events to ``self.On*``.
        """
        if up_down:
            tb.AddTool(wx.ID_UP, "Up", Icon("tango/actions/go-up"))
            tb.AddTool(wx.ID_DOWN, "Down", Icon("tango/actions/go-down"))
            self.Bind(wx.EVT_TOOL, self.OnUp, id=wx.ID_UP)
            self.Bind(wx.EVT_TOOL, self.OnDown, id=wx.ID_DOWN)
        tb.AddTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        tb.AddTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)

    def OnUpdateUIBackward(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(self.CanBackward())

    def OnUpdateUIDown(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(self.CanDown())

    def OnUpdateUIForward(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(self.CanForward())

    def OnUpdateUISetLayout(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(True)

    def OnUpdateUISetVLim(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(True)

    def OnUpdateUIUp(self, event: wx.UpdateUIEvent) -> None:
        event.Enable(self.CanUp())


class EelbrainFrame(wx.Frame, EelbrainWindow):
    _allow_user_set_title = False

    def __init__(
            self,
            parent: wx.Window | None = None,
            id: int = wx.ID_ANY,
            title: str = "",
            pos: wx.Point | tuple[int, int] = wx.DefaultPosition,
            *args,
            **kwargs,
    ) -> None:
        wx.Frame.__init__(self, parent, id, title, pos, *args, **kwargs)
        if not IS_OSX:
            from .app import get_app
            self.SetMenuBar(get_app().CreateMenu(self))
        self._title = self.GetTitle()

    def OnClear(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnCopy(self, event: wx.CommandEvent) -> None:
        win = wx.Window.FindFocus()
        if hasattr(win, 'CanCopy'):
            return win.Copy()
        elif hasattr(self, 'CanCopy'):
            return self.Copy()
        else:
            event.Skip()

    def OnDrawCrosshairs(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnOpen(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnRedo(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSave(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSaveAs(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSetVLim(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSetLayout(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSetMarkedChannels(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSetTime(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnSetWindowTitle(self, event: wx.CommandEvent) -> None:
        dlg = wx.TextEntryDialog(self, f"New title for '{self._title}':", "Set Window Title", value=self._title)
        if dlg.ShowModal() == wx.ID_OK:
            self._title = dlg.GetValue()
            self.SetTitle(self._title)
        dlg.Destroy()

    def OnUndo(self, event: wx.CommandEvent) -> None:
        raise RuntimeError(str(self))

    def OnWindowClose(self, event: wx.CommandEvent) -> None:
        self.Close()

    def SetTitleSuffix(self, suffix: str) -> None:
        self.SetTitle(self._title + suffix)


class EelbrainDialog(wx.Dialog, EelbrainWindow):

    pass
