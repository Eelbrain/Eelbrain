# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import urllib.parse

import wx
import wx.html

from .frame import EelbrainFrame


class TextFrame(EelbrainFrame):
    "Read-only text frame, shows itself"

    def __init__(self, parent, title, text, *args, **kwargs):
        super().__init__(parent, title=title, *args, **kwargs)
        self.text = wx.TextCtrl(self, wx.ID_ANY, text, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.Show()


class HTMLWindow(wx.html.HtmlWindow):

    def OnLinkClicked(self, link):
        url = link.GetHref()
        self.Parent.OpenURL(url)


class HTMLFrame(EelbrainFrame):

    def __init__(self, parent, title, text, **kwargs):
        EelbrainFrame.__init__(self, parent, title=title, **kwargs)
        self.text = HTMLWindow(self, wx.ID_ANY, style=wx.VSCROLL)
        self.text.SetPage(text)
        self.Show()

    def OpenURL(self, url):
        raise NotImplementedError(f"{url=}")


class HTML2Frame(EelbrainFrame):

    def __init__(self, parent, title, text, **kwargs):
        import wx.html2

        EelbrainFrame.__init__(self, parent, title=title, **kwargs)
        self.webview = wx.html2.WebView.New(self)
        self.Bind(wx.html2.EVT_WEBVIEW_NAVIGATING, self.OnNavigating, self.webview)
        self.webview.SetPage(text, 'start-url')
        self.Show()

    def OnNavigating(self, evt):
        url = urllib.parse.unquote(evt.GetURL())
        if url != 'file:///start-url':
            self.OpenURL(url)

    def OpenURL(self, url):
        raise NotImplementedError(f"{url=}")
