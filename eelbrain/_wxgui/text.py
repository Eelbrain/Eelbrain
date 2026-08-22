# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
import tempfile
import urllib.parse

import wx
import wx.html

from .. import fmtxt
from .frame import EelbrainFrame


class TextFrame(EelbrainFrame):
    "Read-only text frame, shows itself"

    def __init__(self, parent: wx.Window, title: str, text: str, *args, **kwargs) -> None:
        super().__init__(parent, title=title, *args, **kwargs)
        self.text = wx.TextCtrl(self, wx.ID_ANY, text, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.Show()


class HTMLWindow(wx.html.HtmlWindow):

    def OnLinkClicked(self, link: wx.html.HtmlLinkInfo) -> None:
        url = link.GetHref()
        self.Parent.OpenURL(url)


class HTMLFrame(EelbrainFrame):

    def __init__(self, parent: wx.Window, title: str, text: str, **kwargs) -> None:
        EelbrainFrame.__init__(self, parent, title=title, **kwargs)
        self.text = HTMLWindow(self, wx.ID_ANY, style=wx.VSCROLL)
        self.text.SetPage(text)
        self.Show()

    def OpenURL(self, url: str) -> None:
        raise NotImplementedError(f"{url=}")


class HTML2Frame(EelbrainFrame):
    """Frame for displaying an FMText document with WebView

    WebView is missing from wxWidgets builds with ``wxUSE_WEBVIEW=0`` (e.g., conda-forge on Linux). Where it is unavailable, the document is displayed with :mod:`wx.html`, which renders the same document without CSS.
    """

    def __init__(self, parent: wx.Window, title: str, doc: fmtxt.FMTextElement, **kwargs) -> None:
        EelbrainFrame.__init__(self, parent, title=title, **kwargs)
        try:
            from wx import html2  # not 'import wx.html2', which shadows wx in this scope

            self.webview = html2.WebView.New(self)
        except (ImportError, NotImplementedError):
            self.webview = None
            self.text = HTMLWindow(self, wx.ID_ANY, style=wx.VSCROLL)
            # wx.html can not display embedded images, so images are written to files; they are decoded while the page is parsed and the files are not needed afterwards
            with tempfile.TemporaryDirectory(prefix='eelbrain-report-') as temp_dir:
                root = Path(temp_dir)
                (root / 'images').mkdir()
                path = root / 'report.html'
                path.write_bytes(fmtxt.make_html_doc(doc, root, 'images').encode('ascii', 'xmlcharrefreplace'))
                self.text.LoadPage(path.as_uri())
        else:
            self.Bind(html2.EVT_WEBVIEW_NAVIGATING, self.OnNavigating, self.webview)
            self.webview.SetPage(fmtxt.make_html_doc(doc), 'start-url')
        self.Show()

    def OnNavigating(self, evt: wx.CommandEvent) -> None:
        url = urllib.parse.unquote(evt.GetURL())
        # Ignore internal WebView lifecycle URLs (page-load base URL and blank-page events).
        # The base URL passed to SetPage() may arrive as 'start-url' or 'file:///start-url'
        # depending on the platform/wxpython version.
        if url in ('about:blank', 'start-url', 'file:///start-url'):
            return
        self.OpenURL(url)

    def OpenURL(self, url: str) -> None:
        raise NotImplementedError(f"{url=}")
