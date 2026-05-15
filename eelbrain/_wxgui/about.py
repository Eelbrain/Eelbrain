"About dialog for Eelbrain"

import wx

from .frame import EelbrainFrame
from .utils import Icon


class AboutFrame(EelbrainFrame):
    def __init__(self, parent):
        from .. import __version__

        EelbrainFrame.__init__(self, parent, -1, "", style=wx.CLOSE_BOX)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        img = Icon('eelbrain256')
        bmp = wx.StaticBitmap(self, -1, img)
        sizer.Add(bmp, 0, wx.LEFT | wx.TOP | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL,
                  40)

        txt = "Eelbrain"
        text = wx.StaticText(self, -1, txt)
        font = text.GetFont()
        font.SetFaceName(".LucidaGrandeUI")
        font.SetPointSize(30)
        text.SetFont(font)
        sizer.Add(text, 0, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)

        txt = f"Version {__version__}"
        text = wx.StaticText(self, -1, txt)
        font = text.GetFont()
        font.SetFaceName(".LucidaGrandeUI")
        font.SetPointSize(12)
        text.SetFont(font)
        text.SetForegroundColour(wx.Colour(120, 120, 120))
        sizer.Add(text, 0, wx.LEFT | wx.BOTTOM | wx.RIGHT | wx.ALIGN_CENTER_HORIZONTAL, 10)

        sizer.Add(300, 40)
        self.Fit()
        self.CenterOnScreen(wx.HORIZONTAL)
