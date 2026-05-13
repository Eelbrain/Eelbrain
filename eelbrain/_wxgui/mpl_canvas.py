'''
Created on Dec 21, 2010

Mpl examples:
http://matplotlib.sourceforge.net/examples/user_interfaces/index.html


'''
from logging import getLogger
import math
import tempfile

import numpy as np
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.figure import Figure
import wx

from .app import get_app
from .frame import EelbrainFrame, EelbrainDialog
from .help import show_help_txt
from .utils import FloatValidator, Icon
from . import ID


class FigureCanvasPanel(FigureCanvasWxAgg):
    """wx.Panel with a matplotlib figure

    Notes
    -----
    Subclass of mpl's Canvas to allow for more interaction with Eelbrain (such
    as copying the contents to the clipboard).
    """

    def __init__(self, parent, *args, **kwargs):
        """wx.Panel with a matplotlib figure

        Parameters
        ----------
        figsize : tuple
            Figure dimensions (width, height) in inches
        dpi : int
            Dots per inch.
        facecolor : mpl color
            The figure patch facecolor; defaults to rc ``figure.facecolor``
        edgecolor : mpl color
            The figure patch edge color; defaults to rc ``figure.edgecolor``
        linewidth : scalar
            The figure patch edge linewidth; the default linewidth of the frame
        frameon : bool
            If ``False``, suppress drawing the figure frame
        subplotpars :
            A :class:`SubplotParams` instance, defaults to rc
        tight_layout : bool | dict
            If ``False`` use ``subplotpars``; if ``True`` adjust subplot
            parameters using :meth:`tight_layout` with default padding.
            When providing a dict containing the keys `pad`, `w_pad`, `h_pad`
            and `rect`, the default :meth:`tight_layout` paddings will be
            overridden. Defaults to rc ``figure.autolayout``.
        """
        self.figure = Figure(*args, **kwargs)
        self.original_size_inches = self.figure.get_size_inches()
        FigureCanvasWxAgg.__init__(self, parent, wx.ID_ANY, self.figure)
        self.Bind(wx.EVT_ENTER_WINDOW, self.ChangeCursor)
        self._background = None
        self.hidpi = self.GetContentScaleFactor()

    def _on_key_down(self, event):
        # Override to avoid system chime
        KeyEvent("key_press_event", self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()

    def _on_key_up(self, event):
        # Override to avoid system chime
        KeyEvent("key_release_event", self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()

    def SizeToFigure(self):
        "Adjust the Panel size to the size implied when the figure was created"
        w, h = self.original_size_inches
        dpi = self.figure.dpi / self.hidpi
        w = math.ceil(w * dpi)
        h = math.ceil(h * dpi)
        self.SetSize((w, h))

    def CanCopy(self):
        return True

    def bufferHasChanged(self):
        return True

    def ChangeCursor(self, event):
        "http://matplotlib.sourceforge.net/examples/user_interfaces/wxcursor_demo.html"
        self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))

    def Copy(self):
        # By default, copy PDF
        if not wx.TheClipboard.Open():
            getLogger('eelbrain').debug("Failed to open clipboard")
            return
        try:
            path = tempfile.mktemp('.pdf')
            self.figure.savefig(path)
            # copy path
            do = wx.FileDataObject()
            do.AddFile(path)
            wx.TheClipboard.SetData(do)
        finally:
            wx.TheClipboard.Close()
            wx.TheClipboard.Flush()

    @staticmethod
    def CanCopyPNG():
        return True

    def CopyAsPNG(self):
        self.Copy_to_Clipboard()

    def _to_matplotlib_event(self, event: wx.MouseEvent):
        """Convert wxPython event to Matplotlib event

        - Sets axes and position but ignores source
        - cf. matplotlib.backends.backend_wx._FigureCanvasWxBase
        """
        x = event.GetX() * self.hidpi
        y = (self.GetSize().GetHeight() - event.GetY()) * self.hidpi
        return MouseEvent('wx-event', self.figure.canvas, x, y, guiEvent=event)

    def redraw(self, axes=()):
        # https://matplotlib.org/stable/users/explain/animations/blitting.html
        if not axes:
            return
        elif self._background is None:
            raise RuntimeError("Background not captured")

        self.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
        self.blit(self.figure.bbox)

    def store_canvas(self):
        self._background = self.copy_from_bbox(self.figure.bbox)


class CanvasFrame(EelbrainFrame):
    # after:
    # http://matplotlib.sourceforge.net/examples/user_interfaces/embedding_in_wx2.html
    _plot_name = "CanvasFrame"
    _allow_user_set_title = True

    def __init__(self, parent=None, title="Matplotlib Frame", pos=wx.DefaultPosition, eelfigure=None, statusbar=True, toolbar=True, mpl_toolbar=False, **kwargs):
        EelbrainFrame.__init__(self, parent, -1, title, pos)
        self._pos_arg = pos

        # set up the canvas
        self.canvas = FigureCanvasPanel(self, **kwargs)
        self.figure = self.canvas.figure

        if statusbar:
            self.CreateStatusBar()

        if toolbar:
            tb = self.CreateToolBar(wx.TB_HORIZONTAL)
            tb.SetToolBitmapSize(size=(32, 32))
            self.FillToolBar(tb, eelfigure)
            tb.Realize()

        if mpl_toolbar:
            self.add_mpl_toolbar()

        # Fit frame size to the intended figure size
        self.canvas.SizeToFigure()
        frame_size = self.ClientToWindowSize(self.canvas.GetSize())
        self.SetSize(frame_size)

        self._eelfigure = eelfigure
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def store_canvas(self):
        self.canvas.store_canvas()

    def FillToolBar(self, tb, eelfigure):
        "Subclass should call this after adding their own items"
        if hasattr(self.Parent, 'attach'):
            tb.AddTool(ID.ATTACH, "Attach", Icon("actions/attach"))
            self.Bind(wx.EVT_TOOL, self.OnAttach, id=ID.ATTACH)

        tb.AddTool(wx.ID_SAVE, "Save", Icon("tango/actions/document-save"))
        self.Bind(wx.EVT_TOOL, self.OnSaveAs, id=wx.ID_SAVE)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUISave, id=wx.ID_SAVE)

        # intermediate, custom part
        if eelfigure is None:
            self._fill_toolbar(tb)
        else:
            eelfigure._fill_toolbar(tb)
            self._plot_name = eelfigure.__class__.__name__
            self.__doc__ = eelfigure.__doc__

        # right-most part
        tb.AddStretchableSpace()
        tb.AddTool(ID.ATTACH, "Attach", Icon("actions/attach"))
        self.Bind(wx.EVT_TOOL, self.OnAttach, id=ID.ATTACH)
        if self.__doc__:
            tb.AddTool(wx.ID_HELP, 'Help', Icon("tango/apps/help-browser"))
            self.Bind(wx.EVT_TOOL, self.OnHelp, id=wx.ID_HELP)

    def _fill_toolbar(self, tb):
        pass

    def Show(self, show=True):
        if self._pos_arg == wx.DefaultPosition:
            rect = self.GetRect()
            display_w, display_h = wx.DisplaySize()
            dx = -rect.Left if rect.Right > display_w else 0
            dy = -rect.Top + 22 if rect.Bottom > display_h else 0
            if dx or dy:
                rect.Offset(dx, dy)
                self.SetRect(rect)
        EelbrainFrame.Show(self, show)

    def add_mpl_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if 0:  # wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.Sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()

    def OnAttach(self, event):
        get_app().Attach(self._eelfigure, f"{self._plot_name} plot", 'p', self)

    def OnClose(self, event):
        # remove circular reference
        if getattr(self, '_eelfigure', None):
            self._eelfigure._frame = None
            del self._eelfigure
        event.Skip()

    def OnDrawCrosshairs(self, event):
        self._eelfigure.draw_crosshairs(event.IsChecked())

    def OnHelp(self, event):
        show_help_txt(self.__doc__, self, self._plot_name)

    def OnSave(self, event):
        self.OnSaveAs(event)

    def OnSaveAs(self, event):
        default_file = f"{self.GetTitle().replace(': ', ' - ')}.pdf"
        dlg = wx.FileDialog(self, "If no file type is selected below, it is inferred from the extension.",
                            defaultFile=default_file,
                            wildcard="Any (*.*)|*.*|PDF (*.pdf)|*.pdf|PNG (*.png)|*.png",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            self.figure.savefig(dlg.GetPath())
        dlg.Destroy()

    def OnSetTime(self, event):
        dlg = SetTimeDialog(self, self._eelfigure.get_time())
        if dlg.ShowModal() == wx.ID_OK:
            self._eelfigure.set_time(dlg.time)
        dlg.Destroy()

    def OnSetVLim(self, event):
        if self._eelfigure._can_set_vlim:
            vlim = self._eelfigure.get_vlim()
            ylim = None
        else:
            vlim = None
            ylim = self._eelfigure.get_ylim() if self._eelfigure._can_set_ylim else None
        xlim = self._eelfigure.get_xlim() if self._eelfigure._can_set_xlim else None
        dlg = AxisLimitsDialog(vlim, ylim, xlim, self)
        if dlg.ShowModal() == wx.ID_OK:
            if vlim is not None:
                self._eelfigure.set_vlim(*dlg.vlim)
            if ylim is not None:
                self._eelfigure.set_ylim(*dlg.ylim)
            if xlim is not None:
                self._eelfigure.set_xlim(*dlg.xlim)
        dlg.Destroy()

    def OnUpdateUIDrawCrosshairs(self, event):
        event.Enable(True)
        event.Check(self._eelfigure._draw_crosshairs)

    def OnUpdateUISave(self, event):
        event.Enable(True)

    def OnUpdateUISaveAs(self, event):
        event.Enable(True)

    def OnUpdateUISetVLim(self, event):
        event.Enable(self._eelfigure._can_set_xlim or self._eelfigure._can_set_ylim)

    def OnUpdateUISetTime(self, event):
        event.Enable(self._eelfigure._can_set_time)


class TestCanvas(CanvasFrame):
    "This is a minimal CanvasFrame subclass"

    def __init__(self, effect=10, mpl_toolbar=True):
        CanvasFrame.__init__(self, title="Test MPL Frame", mpl_toolbar=mpl_toolbar)
        self.plot()
        self.Show(effect)

    def plot(self):
        self.axes = self.figure.add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)


class LimitsValidator(wx.Validator):
    """Validator for axis limits

    Parameters
    ----------
    parent : wx Dialog
        Parent.
    attr : str
        Attribute on ``parent`` to operate on.
    single : 'symmetric' | 'offset'
        How to interpret input when only one value is provided;
        ``symmetric``: ``[-v, v]``
        ``offset``: ``[v, v + old_range]``
    """

    def __init__(self, parent, attr, single='symmetric'):
        wx.Validator.__init__(self)
        self.parent = parent
        self.attr = attr
        self.value = None
        self.single = single

    def Clone(self):
        return LimitsValidator(self.parent, self.attr, self.single)

    def Validate(self, parent):
        ctrl = self.GetWindow()
        value = ctrl.GetValue()
        if not value.strip():
            self.TransferToWindow()
            return False

        try:
            values = value.replace(',', ' ').split()
            if len(values) == 1:
                if self.single == 'symmetric':
                    upper = float(values[0])
                    lower = -upper
                elif self.single == 'offset':
                    lower = float(values[0])
                    old_lower, old_upper = getattr(self.parent, self.attr)
                    upper = lower + old_upper - old_lower
                else:
                    raise RuntimeError(f'{self.single=}')
            elif len(values) == 2:
                lower, upper = map(float, values)
            else:
                raise ValueError("Wrong number of values")
        except Exception as exception:
            msg = wx.MessageDialog(
                self.parent, str(exception), "Invalid Entry",
                wx.OK | wx.ICON_ERROR)
            msg.ShowModal()
            msg.Destroy()
            return False
        else:
            self.value = (lower, upper)
            return True

    def TransferToWindow(self):
        ctrl = self.GetWindow()
        vmin, vmax = getattr(self.parent, self.attr)
        if self.single == 'symmetric':
            use_single = -vmin == vmax
        elif self.single == 'offset':
            use_single = False
        else:
            raise RuntimeError(f'{self.single=}')
        if use_single:
            ctrl.SetValue(f'{vmax:g}')
        else:
            ctrl.SetValue(f'{vmin:g} {vmax:g}')
        ctrl.SelectAll()
        return True

    def TransferFromWindow(self):
        if self.value is None:
            return False
        else:
            setattr(self.parent, self.attr, self.value)
            return True


class AxisLimitsDialog(EelbrainDialog):
    """Dialog to set axis limits

    Parameters
    ----------
    ylim : None | tuple
        ``(bottom, top)`` if the y-limits can be adjusted, else ``None``.
    xlim : None | tuple
        ``(left, right)`` if the x-limits can be adjusted, else ``None``.
    ... :
        Wx-Python Dialog parameters.
    """

    def __init__(self, vlim, ylim, xlim, parent, *args, **kwargs):
        EelbrainDialog.__init__(self, parent, wx.ID_ANY, "Set Axis Limits", *args, **kwargs)
        self.vlim = vlim
        self.ylim = ylim
        self.xlim = xlim

        # Layout
        mainsizer = wx.BoxSizer(wx.VERTICAL)

        sizer = wx.GridSizer(2, 5, 5)
        # v-limit
        if vlim is not None:
            sizer.Add(wx.StaticText(self, label="Value Limits:"))
            self.v_text = wx.TextCtrl(self, validator=LimitsValidator(self, 'vlim'))
            sizer.Add(self.v_text)
            self.v_text.SetFocus()
        else:
            self.v_text = None
        # y-axis
        if ylim is not None:
            sizer.Add(wx.StaticText(self, label="Y-Axis Limits:"))
            self.y_text = wx.TextCtrl(self, validator=LimitsValidator(self, 'ylim'))
            sizer.Add(self.y_text)
            self.y_text.SetFocus()
        else:
            self.y_text = None
        # x-axis
        if xlim is not None:
            sizer.Add(wx.StaticText(self, label="X-Axis Limits:"))
            self.x_text = wx.TextCtrl(self, validator=LimitsValidator(self, 'xlim', 'offset'))
            sizer.Add(self.x_text)
            if self.y_text is None:
                self.x_text.SetFocus()
        else:
            self.x_text = None

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        button_sizer.Realize()

        # finalize
        mainsizer.Add(sizer, flag=wx.ALL, border=10)
        # mainsizer.AddSpacer(10)
        mainsizer.Add(button_sizer, flag=wx.ALL | wx.ALIGN_RIGHT, border=10)
        self.SetSizer(mainsizer)
        mainsizer.Fit(self)


class SetTimeDialog(EelbrainDialog):

    def __init__(self, parent, current_time):
        EelbrainDialog.__init__(self, parent, title="Set Time")
        self.time = current_time

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(wx.StaticText(self, label="New time:"))
        self.text_ctrl = wx.TextCtrl(self, validator=FloatValidator(self, 'time'))
        sizer.Add(self.text_ctrl)
        self.text_ctrl.SetFocus()

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        button_sizer.Realize()

        # finalize
        mainsizer = wx.BoxSizer(wx.VERTICAL)
        mainsizer.Add(sizer, flag=wx.ALL, border=10)
        # mainsizer.AddSpacer(10)
        mainsizer.Add(button_sizer, flag=wx.ALL | wx.ALIGN_RIGHT, border=10)
        self.SetSizer(mainsizer)
        mainsizer.Fit(self)
