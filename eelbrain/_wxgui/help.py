# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Help Viewer"""
from __future__ import annotations

import wx

from ..fmtxt import make_html_doc, Section, Code
from .text import HTMLFrame


def show_help_txt(text: str, parent: wx.Window, title: str = "") -> HelpFrame:
    """Show help frame with text in monospaced font"""
    s = Section(title, Code(text))
    html = make_html_doc(s, None)
    return HelpFrame(parent, title, html)


class HelpFrame(HTMLFrame):

    def __init__(self, parent: wx.Window, title: str, html: str, **kwargs) -> None:
        display_w, display_h = wx.DisplaySize()
        x = 0
        y = 25
        w = min(700, display_w)
        h = min(2000, display_h - y - 100)
        HTMLFrame.__init__(self, parent, title, html, pos=(x, y), size=(w, h), **kwargs)
