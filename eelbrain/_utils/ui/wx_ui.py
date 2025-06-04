"""WxPython-based implementation of the Eelbrain ui functions."""
from typing import Optional

from ..._wxgui import wx, get_app


def ask_saveas(title, message, filetypes, defaultDir, defaultFile):
    """See eelbrain.ui documentation"""
    app = get_app()
    return app.ask_saveas(title, message, filetypes, defaultDir, defaultFile)


def ask_dir(title="Select Folder", message="Please Pick a Folder", must_exist=True):
    app = get_app()
    return app.ask_for_dir(title, message, must_exist)


def ask_file(title, message, filetypes, directory, mult):
    app = get_app()
    return app.ask_for_file(title, message, filetypes, directory, mult)


def ask(
        title: str = "Overwrite File?",
        message: str = "Duplicate filename. Do you want to overwrite?",
        cancel: bool = False,
        default: Optional[bool] = True,  # True=YES, False=NO, None=Nothing
):
    style = wx.YES_NO | wx.ICON_QUESTION
    if cancel:
        style = style | wx.CANCEL
    if default:
        style = style | wx.YES_DEFAULT
    elif default is False:
        style = style | wx.NO_DEFAULT
    dialog = wx.MessageDialog(None, message, title, style)
    answer = dialog.ShowModal()
    if answer == wx.ID_NO:
        return False
    elif answer == wx.ID_YES:
        return True
    else:
        return None


def ask_color(default=(0, 0, 0)):
    dlg = wx.ColourDialog(None)
    dlg.GetColourData().SetChooseFull(True)
    if dlg.ShowModal() == wx.ID_OK:
        data = dlg.GetColourData()
        out = data.GetColour().Get()
        out = tuple([o / 255. for o in out])
    else:
        out = False
    dlg.Destroy()
    return out


def ask_str(message, title, default=''):
    app = get_app()
    return app.ask_for_string(title, message, default)


def message(title, message="", icon='i'):
    style = wx.OK
    if icon == 'i':
        style = style | wx.ICON_INFORMATION
    elif icon == '?':
        style = style | wx.ICON_QUESTION
    elif icon == '!':
        style = style | wx.ICON_EXCLAMATION
    elif icon == 'error':
        style = style | wx.ICON_ERROR
    elif icon is None:
        pass
    else:
        raise ValueError("Invalid icon argument: %r" % icon)
    dlg = wx.MessageDialog(None, message, title, style)
    dlg.ShowModal()


def copy_file(path):
    if wx.TheClipboard.Open():
        try:
            data_object = wx.FileDataObject()
            data_object.AddFile(path)
            wx.TheClipboard.SetData(data_object)
        except BaseException:
            wx.TheClipboard.Close()
            raise
        else:
            wx.TheClipboard.Close()


def copy_text(text):
    if wx.TheClipboard.Open():
        try:
            data_object = wx.TextDataObject(text)
            wx.TheClipboard.SetData(data_object)
        except BaseException:
            wx.TheClipboard.Close()
            raise
        else:
            wx.TheClipboard.Close()
