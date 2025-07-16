from logging import getLogger
import packaging.version
import select
import sys
from threading import Thread
from time import sleep
import webbrowser

import wx
from wx.adv import TaskBarIcon, TBI_DOCK

from .._utils import IS_OSX, IS_WINDOWS
from ..plot._base import CONFIG
from .about import AboutFrame
from .frame import FOCUS_UI_UPDATE_FUNC_NAMES, EelbrainFrame
from .utils import Icon
from . import ID


APP = None  # hold the App instance
JUMPSTART_TIME = 250  # ms


def wildcard(filetypes):
    if filetypes:
        return '|'.join(map('|'.join, filetypes))
    else:
        return ""


class App(wx.App):
    _pt_thread = None
    about_frame = None
    _result = None
    _bash_ui_from_mainloop = None
    _ipython = None
    using_prompt_toolkit = False

    def OnInit(self):
        self.SetAppName("Eelbrain")
        self.SetAppDisplayName("Eelbrain")

        # register in IPython
        if CONFIG['prompt_toolkit'] and 'IPython' in sys.modules and packaging.version.parse(sys.modules['IPython'].__version__) >= packaging.version.parse('5'):
            import IPython.core.error

            if CONFIG['prompt_toolkit'] == 'eelbrain':
                import IPython.terminal.pt_inputhooks
                import IPython.core.pylabtools

                IPython.terminal.pt_inputhooks.register('eelbrain', self.pt_inputhook)
                IPython.core.pylabtools.backend2gui.clear()  # prevent pylab from initializing event-loop
            shell = IPython.get_ipython()
            if shell is not None:
                self._pt_thread = self._pt_thread_win if IS_WINDOWS else self._pt_thread_linux
                try:
                    shell.enable_gui(CONFIG['prompt_toolkit'])
                except IPython.core.error.UsageError:
                    print(f"Prompt-toolkit does not seem to be supported by the current IPython shell ({shell.__class__.__name__}); The Eelbrain GUI needs to block Terminal input to work. Use eelbrain.gui.run() to start GUI interaction.")
                else:
                    self.using_prompt_toolkit = True
                    self._ipython = shell
                    getLogger('Eelbrain').debug("Initialized prompt_toolkit with %s", CONFIG['prompt_toolkit'])

        self.SetExitOnFrameDelete(not self.using_prompt_toolkit)

        if IS_OSX:
            self.dock_icon = DockIcon(self)
            self.menubar = self.CreateMenu(self)
            # list windows in Window menu
            self.window_menu_window_items = []
            self.Bind(wx.EVT_MENU_OPEN, self.OnMenuOpened)
        else:
            self.dock_icon = None
            self.menu_bar = None
            self.window_menu_window_items = None

        return True

    def CreateMenu(self, t):
        """Create Menubar

        Parameters
        ----------
        t : App | EelbrainFrame
            Object to which the menu will be attached; on macOS ``self``, on
            other systems the specific :class:`EelbrainFrame`.
        """
        menu_bar = wx.MenuBar()

        # File Menu
        m = file_menu = wx.Menu()
        m.Append(wx.ID_OPEN, '&Open... \tCtrl+O')
        m.AppendSeparator()
        m.Append(wx.ID_CLOSE, '&Close Window \tCtrl+W')
        m.Append(wx.ID_SAVE, "Save \tCtrl+S")
        m.Append(wx.ID_SAVEAS, "Save As... \tCtrl+Shift+S")
        menu_bar.Append(file_menu, "File")

        # Edit Menu
        m = edit_menu = wx.Menu()
        m.Append(ID.UNDO, '&Undo \tCtrl+Z')
        m.Append(ID.REDO, '&Redo \tCtrl+Shift+Z')
        m.AppendSeparator()
        m.Append(wx.ID_CUT, 'Cut \tCtrl+X')
        m.Append(wx.ID_COPY, 'Copy \tCtrl+C')
        m.Append(ID.COPY_AS_PNG, 'Copy as PNG \tCtrl+Shift+C')
        m.Append(wx.ID_PASTE, 'Paste \tCtrl+V')
        m.AppendSeparator()
        m.Append(wx.ID_CLEAR, 'Cle&ar')
        menu_bar.Append(edit_menu, "Edit")

        # Tools Menu
        # updated by the active GUI
        if IS_OSX or hasattr(t, 'MakeToolsMenu'):
            tools_menu = wx.Menu()
            if not IS_OSX:
                t.MakeToolsMenu(tools_menu)
            menu_bar.Append(tools_menu, "Tools")

        # View Menu
        m = view_menu = wx.Menu()
        m.Append(ID.SET_VLIM, "Set Axis Limits... \tCtrl+l", "Change the current figure's axis limits")
        m.Append(ID.LINK_TIME_AXES, "Link Time Axes", "Synchronize the time displayed on figures")
        m.Append(ID.SET_MARKED_CHANNELS, "Mark Channels...", "Mark specific channels in plots")
        m.Append(ID.DRAW_CROSSHAIRS, "Draw &Crosshairs", "Draw crosshairs under the cursor", kind=wx.ITEM_CHECK)
        m.AppendSeparator()
        m.Append(ID.SET_LAYOUT, "&Set Layout... \tCtrl+Shift+l", "Change the page layout")
        menu_bar.Append(view_menu, "View")

        # Go Menu
        m = go_menu = wx.Menu()
        m.Append(wx.ID_FORWARD, '&Forward \tCtrl+]', 'Go One Page Forward')
        m.Append(wx.ID_BACKWARD, '&Back \tCtrl+[', 'Go One Page Back')
        m.Append(ID.TIME, '&Time... \tCtrl+t', 'Go to time...')
        if not self.using_prompt_toolkit:
            m.AppendSeparator()
            m.Append(ID.YIELD_TO_TERMINAL, '&Yield to Terminal \tAlt+Ctrl+Q')
        menu_bar.Append(go_menu, "Go")

        # Window Menu
        m = window_menu = wx.Menu()
        m.Append(ID.WINDOW_MINIMIZE, '&Minimize \tCtrl+M')
        m.Append(ID.WINDOW_ZOOM, '&Zoom')
        m.Append(ID.SET_TITLE, '&Set Title')
        m.AppendSeparator()
        m.Append(ID.WINDOW_TILE, '&Tile')
        m.AppendSeparator()
        menu_bar.Append(window_menu, "Window")

        # Help Menu
        m = help_menu = wx.Menu()
        m.Append(ID.HELP_EELBRAIN, 'Eelbrain Help')
        m.Append(ID.HELP_PYTHON, "Python Help")
        m.AppendSeparator()
        m.Append(wx.ID_ABOUT, '&About Eelbrain')
        menu_bar.Append(help_menu, self.GetMacHelpMenuTitleName() if IS_OSX else 'Help')

        # Menu Bar
        wx.MenuBar.MacSetCommonMenuBar(menu_bar)

        # Bind Menu Commands
        t.Bind(wx.EVT_MENU, self.OnAbout, id=wx.ID_ABOUT)
        t.Bind(wx.EVT_MENU, t.OnOpen, id=wx.ID_OPEN)
        t.Bind(wx.EVT_MENU, t.OnClear, id=wx.ID_CLEAR)
        t.Bind(wx.EVT_MENU, t.OnWindowClose, id=wx.ID_CLOSE)
        t.Bind(wx.EVT_MENU, t.OnCopy, id=wx.ID_COPY)
        t.Bind(wx.EVT_MENU, self.OnCopyAsPNG, id=ID.COPY_AS_PNG)
        t.Bind(wx.EVT_MENU, self.OnCut, id=wx.ID_CUT)
        t.Bind(wx.EVT_MENU, t.OnDrawCrosshairs, id=ID.DRAW_CROSSHAIRS)
        t.Bind(wx.EVT_MENU, self.OnOnlineHelp, id=ID.HELP_EELBRAIN)
        t.Bind(wx.EVT_MENU, self.OnOnlineHelp, id=ID.HELP_PYTHON)
        t.Bind(wx.EVT_MENU, self.OnPaste, id=wx.ID_PASTE)
        t.Bind(wx.EVT_MENU, t.OnRedo, id=ID.REDO)
        t.Bind(wx.EVT_MENU, t.OnSave, id=wx.ID_SAVE)
        t.Bind(wx.EVT_MENU, t.OnSaveAs, id=wx.ID_SAVEAS)
        t.Bind(wx.EVT_MENU, t.OnSetLayout, id=ID.SET_LAYOUT)
        t.Bind(wx.EVT_MENU, t.OnSetMarkedChannels, id=ID.SET_MARKED_CHANNELS)
        t.Bind(wx.EVT_MENU, t.OnSetVLim, id=ID.SET_VLIM)
        t.Bind(wx.EVT_MENU, t.OnSetTime, id=ID.TIME)
        t.Bind(wx.EVT_MENU, self.OnLinkTimeAxes, id=ID.LINK_TIME_AXES)
        t.Bind(wx.EVT_MENU, t.OnUndo, id=ID.UNDO)
        t.Bind(wx.EVT_MENU, t.OnWindowIconize, id=ID.WINDOW_MINIMIZE)
        t.Bind(wx.EVT_MENU, self.OnWindowTile, id=ID.WINDOW_TILE)
        t.Bind(wx.EVT_MENU, t.OnWindowZoom, id=ID.WINDOW_ZOOM)
        t.Bind(wx.EVT_MENU, t.OnSetWindowTitle, id=ID.SET_TITLE)
        t.Bind(wx.EVT_MENU, self.OnQuit, id=wx.ID_EXIT)
        t.Bind(wx.EVT_MENU, self.OnYieldToTerminal, id=ID.YIELD_TO_TERMINAL)

        # UI-update concerning frames
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIBackward, id=wx.ID_BACKWARD)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIClear, id=wx.ID_CLEAR)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIClose, id=wx.ID_CLOSE)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIDown, id=wx.ID_DOWN)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIDrawCrosshairs, id=ID.DRAW_CROSSHAIRS)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIForward, id=wx.ID_FORWARD)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIOpen, id=wx.ID_OPEN)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIRedo, id=ID.REDO)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUISave, id=wx.ID_SAVE)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUISaveAs, id=wx.ID_SAVEAS)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUISetLayout, id=ID.SET_LAYOUT)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUISetMarkedChannels, id=ID.SET_MARKED_CHANNELS)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUISetVLim, id=ID.SET_VLIM)
        t.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUILinkTimeAxes, id=ID.LINK_TIME_AXES)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUISetTime, id=ID.TIME)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUITools, id=ID.TOOLS)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIUndo, id=ID.UNDO)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIUp, id=wx.ID_UP)

        # UI-update concerning focus
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIFocus, id=wx.ID_COPY)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIFocus, id=ID.COPY_AS_PNG)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIFocus, id=wx.ID_CUT)
        t.Bind(wx.EVT_UPDATE_UI, t.OnUpdateUIFocus, id=wx.ID_PASTE)

        return menu_bar

    def _pt_thread_win(self, context):
        # On Windows, select.poll() is not available
        while context._input_is_ready is None or not context.input_is_ready():
            sleep(0.020)
        wx.CallAfter(self.ExitMainLoop, True)

    def _pt_thread_linux(self, context):
        # getLogger('Eelbrain').debug("PTK: poll")
        poll = select.poll()
        poll.register(context.fileno(), select.POLLIN)
        poll.poll(-1)
        wx.CallAfter(self.ExitMainLoop, True)

    def pt_inputhook(self, context):
        """prompt_toolkit inputhook"""
        # prompt_toolkit.eventloop.inputhook.InputHookContext
        Thread(target=self._pt_thread, args=(context,)).start()
        self.MainLoop()

    def jumpstart(self):
        wx.CallLater(JUMPSTART_TIME, self.ExitMainLoop)
        self.MainLoop()

    def _get_active_frame(self):
        win = wx.Window.FindFocus()
        win_parent = wx.GetTopLevelParent(win)
        if win_parent:
            return win_parent
        for w in wx.GetTopLevelWindows():
            if hasattr(w, 'IsActive') and w.IsActive():
                return w
        return wx.GetActiveWindow()

    def _get_parent_gui(self):
        frame = self._get_active_frame()
        if frame is None:
            return
        while True:
            if hasattr(frame, 'MakeToolsMenu'):
                return frame
            elif frame.Parent is not None:
                frame = frame.Parent
            else:
                return

    def _bash_ui(self, func, *args):
        "Launch a modal dialog based on terminal input"
        # Create fake frame to prevent dialog from sticking
        if not self.GetTopWindow():
            self.SetTopWindow(wx.Frame(None))
        # Run dialog
        self._bash_ui_from_mainloop = self.using_prompt_toolkit or self.IsMainLoopRunning()
        if self._bash_ui_from_mainloop:
            return func(*args)
        else:
            wx.CallAfter(func, *args)
            print("Please switch to the Python Application to provide input.")
            self.MainLoop()
            return self._result

    def _bash_ui_finalize(self, result):
        if self._bash_ui_from_mainloop:
            self.Yield()
            return result
        else:
            self._result = result
            self.ExitMainLoop()

    def ask_for_dir(self, title="Select Folder", message="Please Pick a Folder",
                    must_exist=True):
        return self._bash_ui(self._ask_for_dir, title, message, must_exist)

    def _ask_for_dir(self, title, message, must_exist):
        style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        if must_exist:
            style = style | wx.DD_DIR_MUST_EXIST

        dialog = wx.DirDialog(None, message, name=title,
                              style=style)
        dialog.SetTitle(title)
        if dialog.ShowModal() == wx.ID_OK:
            result = dialog.GetPath()
        else:
            result = False
        dialog.Destroy()
        return self._bash_ui_finalize(result)

    def ask_for_file(self, title, message, filetypes, directory, mult):
        return self._bash_ui(self._ask_for_file, title, message, filetypes, directory, mult)

    def _ask_for_file(self, title, message, filetypes, directory, mult):
        """Return path(s) or False.

        Parameters
        ----------
        ...
        directory : str
            Path to initial directory.

        Returns
        -------
        result : list | str | None
            Paths(s) or False.
        """
        style = wx.FD_OPEN
        if mult:
            style = style | wx.FD_MULTIPLE
        dialog = wx.FileDialog(None, message, directory, wildcard=wildcard(filetypes), style=style)
        dialog.SetTitle(title)
        if dialog.ShowModal() == wx.ID_OK:
            if mult:
                result = dialog.GetPaths()
            else:
                result = dialog.GetPath()
        else:
            result = False
        dialog.Destroy()
        return self._bash_ui_finalize(result)

    def ask_for_string(self, title, message, default='', parent=None):
        return self._bash_ui(self._ask_for_string, title, message, default, parent)

    def _ask_for_string(self, title, message, default, parent):
        dialog = wx.TextEntryDialog(parent, message, title, default)
        if dialog.ShowModal() == wx.ID_OK:
            result = dialog.GetValue()
        else:
            result = False
        dialog.Destroy()
        return self._bash_ui_finalize(result)

    def ask_saveas(self, title, message, filetypes, defaultDir, defaultFile):
        return self._bash_ui(self._ask_saveas, title, message, filetypes, defaultDir, defaultFile)

    def _ask_saveas(self, title, message, filetypes, defaultDir, defaultFile):
        # setup file-dialog
        dialog = wx.FileDialog(None, message, wildcard=wildcard(filetypes), style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        dialog.SetTitle(title)
        if defaultDir:
            dialog.SetDirectory(defaultDir)
        if defaultFile:
            dialog.SetFilename(defaultFile)
        # get result
        if dialog.ShowModal() == wx.ID_OK:
            result = dialog.GetPath()
        else:
            result = False
        dialog.Destroy()
        return self._bash_ui_finalize(result)

    def message_box(self, message, caption, style, parent=None):
        return self._bash_ui(self._message_box, message, caption, style, parent)

    def _message_box(self, message, caption, style, parent):
        dialog = wx.MessageDialog(parent, message, caption, style)
        result = dialog.ShowModal()
        dialog.Destroy()
        return self._bash_ui_finalize(result)

    def ExitMainLoop(self, event_with_pt=True):
        if event_with_pt or not self.using_prompt_toolkit:
            # with prompt-toolkit, this leads to hanging when terminating the
            # interpreter
            wx.App.ExitMainLoop(self)

    def Attach(self, obj, desc, default_name, parent):
        if self._ipython is None:
            self.message_box("Attach Unavailable", "The attach command requires running from within IPython 5 or later", wx.ICON_ERROR | wx.OK, parent)
            return
        name = self.ask_for_string("Attach", f"Variable name for {desc} in terminal:", default_name, parent)
        if name:
            self._ipython.user_global_ns[name] = obj

    def OnAbout(self, event):
        if not self.about_frame:
            self.about_frame = AboutFrame(None)
        self.about_frame.Show()
        self.about_frame.Raise()

    def OnClear(self, event):
        frame = self._get_active_frame()
        frame.OnClear(event)

    def OnCopy(self, event):
        win = wx.Window.FindFocus()
        if hasattr(win, 'CanCopy'):
            return win.Copy()
        win = self._get_active_frame()
        if hasattr(win, 'CanCopy'):
            return win.Copy()
        getLogger('Eelbrain').debug("App.OnCopy() call but neither focus nor frame have CanCopy()")
        event.Skip()

    def OnCopyAsPNG(self, event):
        wx.Window.FindFocus().CopyAsPNG()

    def OnCut(self, event):
        win = wx.Window.FindFocus()
        win.Cut()

    def OnDrawCrosshairs(self, event):
        frame = self._get_active_frame()
        frame.OnDrawCrosshairs(event)

    def OnLinkTimeAxes(self, event):
        from ..plot._base import TimeSlicer
        from .._data_obj import UTS

        figures = []
        for window in wx.GetTopLevelWindows():
            eelfigure = getattr(window, '_eelfigure', None)
            if eelfigure and isinstance(eelfigure, TimeSlicer) and isinstance(eelfigure._time_dim, UTS):
                figures.append(eelfigure)
        if len(figures) >= 2:
            f0 = figures[0]
            for figure in figures[1:]:
                f0.link_time_axis(figure)

    def OnMenuOpened(self, event):
        "Update the window-specific Tools menu"
        menu = event.GetMenu()
        if menu.GetTitle() == 'Tools':
            for item in menu.GetMenuItems():
                menu.Remove(item)
            frame = self._get_parent_gui()
            if frame:
                frame.MakeToolsMenu(menu)

    def OnOnlineHelp(self, event):
        "Called from the Help menu to open external resources"
        event_id = event.GetId()
        if event_id == ID.HELP_EELBRAIN:
            webbrowser.open("https://pythonhosted.org/eelbrain/")
        elif event_id == ID.HELP_PYTHON:
            webbrowser.open("http://docs.python.org/2.7/")
        else:
            raise RuntimeError("Invalid help ID")

    def OnOpen(self, event):
        frame = self._get_active_frame()
        frame.OnOpen(event)

    def OnPaste(self, event):
        win = wx.Window.FindFocus()
        win.Paste()

    def OnQuit(self, event):
        getLogger('Eelbrain').debug("App.OnQuit()")
        for win in wx.GetTopLevelWindows():
            if not win.Close():
                return
        getLogger('Eelbrain').debug("App.ExitMainLoop()")
        self.ExitMainLoop()

    def OnRedo(self, event):
        frame = self._get_active_frame()
        frame.OnRedo(event)

    def OnSave(self, event):
        frame = self._get_active_frame()
        frame.OnSave(event)

    def OnSaveAs(self, event):
        frame = self._get_active_frame()
        frame.OnSaveAs(event)

    def OnSetVLim(self, event):
        frame = self._get_active_frame()
        frame.OnSetVLim(event)

    def OnSetLayout(self, event):
        frame = self._get_active_frame()
        frame.OnSetLayout(event)

    def OnSetMarkedChannels(self, event):
        frame = self._get_active_frame()
        frame.OnSetMarkedChannels(event)

    def OnSetTime(self, event):
        frame = self._get_active_frame()
        frame.OnSetTime(event)

    def OnSetWindowTitle(self, event):
        frame = self._get_active_frame()
        frame.OnSetWindowTitle(event)

    def OnUndo(self, event):
        frame = self._get_active_frame()
        frame.OnUndo(event)

    def OnUpdateUIBackward(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIBackward'):
            frame.OnUpdateUIBackward(event)
        else:
            event.Enable(False)

    def OnUpdateUIClear(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIClear'):
            frame.OnUpdateUIClear(event)
        else:
            event.Enable(False)

    def OnUpdateUIClose(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIClose'):
            frame.OnUpdateUIClose(event)
        else:
            event.Enable(False)

    def OnUpdateUIDown(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIDown'):
            frame.OnUpdateUIDown(event)
        else:
            event.Enable(False)

    def OnUpdateUIDrawCrosshairs(self, event):
        frame = self._get_active_frame()
        if isinstance(frame, EelbrainFrame):
            frame.OnUpdateUIDrawCrosshairs(event)
        else:
            event.Enable(False)
            event.Check(False)

    def OnUpdateUIForward(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIForward'):
            frame.OnUpdateUIForward(event)
        else:
            event.Enable(False)

    def OnUpdateUIFocus(self, event):
        func_name = FOCUS_UI_UPDATE_FUNC_NAMES[event.GetId()]
        win = wx.Window.FindFocus()
        func = getattr(win, func_name, None)
        if func is None:
            win = self._get_active_frame()
            func = getattr(win, func_name, None)
            if func is None:
                event.Enable(False)
                return
        event.Enable(func())

    def OnUpdateUILinkTimeAxes(self, event):
        n = 0
        for window in wx.GetTopLevelWindows():
            eelfigure = getattr(window, '_eelfigure', None)
            if eelfigure:
                n += hasattr(eelfigure, 'link_time_axis')
                if n >= 2:
                    event.Enable(True)
                    return
        event.Enable(False)

    def OnUpdateUIOpen(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIOpen'):
            frame.OnUpdateUIOpen(event)
        else:
            event.Enable(False)

    def OnUpdateUIRedo(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIRedo'):
            frame.OnUpdateUIRedo(event)
        else:
            event.Enable(False)

    def OnUpdateUISave(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUISave'):
            frame.OnUpdateUISave(event)
        else:
            event.Enable(False)

    def OnUpdateUISaveAs(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUISaveAs'):
            frame.OnUpdateUISaveAs(event)
        else:
            event.Enable(False)

    def OnUpdateUISetLayout(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUISetLayout'):
            frame.OnUpdateUISetLayout(event)
        else:
            event.Enable(False)

    def OnUpdateUISetMarkedChannels(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUISetMarkedChannels'):
            frame.OnUpdateUISetMarkedChannels(event)
        else:
            event.Enable(False)

    def OnUpdateUISetVLim(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUISetVLim'):
            frame.OnUpdateUISetVLim(event)
        else:
            event.Enable(False)

    def OnUpdateUISetTime(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUISetTime'):
            frame.OnUpdateUISetTime(event)
        else:
            event.Enable(False)

    def OnUpdateUISetWindowTitle(self, event):
        frame = self._get_active_frame()
        event.Enable(getattr(frame, '_allow_user_set_title', False))

    def OnUpdateUITools(self, event):
        event.Enable(bool(self._get_parent_gui))

    def OnUpdateUIUndo(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIUndo'):
            frame.OnUpdateUIUndo(event)
        else:
            event.Enable(False)

    def OnUpdateUIUp(self, event):
        frame = self._get_active_frame()
        if frame and hasattr(frame, 'OnUpdateUIUp'):
            frame.OnUpdateUIUp(event)
        else:
            event.Enable(False)

    def OnWindowClose(self, event):
        frame = self._get_active_frame()
        if frame:
            frame.Close()

    def OnWindowIconize(self, event):
        frame = self._get_active_frame()
        if frame:
            frame.Iconize()

    def OnWindowRaise(self, event):
        id_ = event.GetId()
        window = wx.FindWindowById(id_)
        window.Raise()

    def OnWindowTile(self, event):
        frames = sorted(wx.GetTopLevelWindows(), key=lambda x: x.Position[0])
        dx, dy = wx.DisplaySize()
        x = 0
        y = 0
        y_next = 0
        for frame in frames:
            sx, sy = frame.Size
            if x and x + sx > dx:
                if y_next > dy:
                    return
                x = 0
                y = y_next
            frame.Position = (x, y)
            y_next = max(y_next, y + sy)
            x += sx

    def OnWindowZoom(self, event):
        frame = self._get_active_frame()
        if frame:
            frame.Maximize()

    def OnYieldToTerminal(self, event):
        self.ExitMainLoop()


def get_app(jumpstart=False):
    global APP
    if APP is None:
        try:
            APP = App()
        except SystemExit as exc:
            if exc.code.startswith("This program needs access to the screen"):
                raise SystemExit(
                    f"{exc.code} \n\n"
                    f"If you are using an iPython terminal: make sure you are "
                    f"running a framework build by launching IPython with:\n\n"
                    f"    $ eelbrain\n\n"
                    f"If you are using a Jupyter notebook, prefix the notebook with\n\n"
                    f"    %matplotlib inline\n\n"
                    f"and restart the kernel.")
            else:
                raise

        if jumpstart and IS_OSX:
            # Give wx a chance to initialize the GUI backend
            APP.OnAbout(None)
            wx.CallLater(JUMPSTART_TIME, APP.about_frame.Close)
            wx.CallLater(JUMPSTART_TIME, APP.ExitMainLoop)
            APP.MainLoop()

    return APP


def needs_jumpstart():
    return APP is None and IS_OSX


def run(block=False):
    """Hand over command to the GUI (quit the GUI to return to the terminal)

    Parameters
    ----------
    block : bool
        Block the Terminal even if the GUI is capable of being run in parallel.
        Control returns to the Terminal when the user quits the GUI application.
        This is also useful to prevent plots from closing at the end of a
        script.
    """
    app = get_app()
    if app.using_prompt_toolkit:
        if block:
            app.MainLoop()
    else:
        if not app.IsMainLoopRunning():
            print("Starting GUI. Quit the Python application to return to the "
                  "shell...")
            app.MainLoop()


class DockIcon(TaskBarIcon):
    # http://stackoverflow.com/a/38249390/166700
    def __init__(self, app):
        TaskBarIcon.__init__(self, iconType=TBI_DOCK)
        self.app = app

        # Set the image
        self.SetIcon(Icon('eelbrain256', True), "Eelbrain")
        self.imgidx = 1

    def CreatePopupMenu(self):
        if not self.app.using_prompt_toolkit:
            menu = wx.Menu()
            menu.Append(ID.YIELD_TO_TERMINAL, '&Yield to Terminal')
            return menu
