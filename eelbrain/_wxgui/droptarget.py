import logging

import wx


def filename_repr(filenames: list[str] | str | None) -> str:
    if filenames:
        if isinstance(filenames, str):
            filenames = [filenames]

        try:
            filenames = tuple(map(str, filenames))
        except BaseException:
            pass

        logging.debug(filenames)

        if len(filenames) == 1:
            string = repr(filenames[0])
        else:
            string = repr(filenames)

        return string
    else:
        return ''


class FilenameDropTarget(wx.FileDropTarget):
    """
    File drop target: http://wiki.wxpython.org/DragAndDrop

    (!) apparently the FileDropTarget can only be used once; when I saved the
    instance as an Editor attribute and SetDropTarget it to a second editor
    this caused a segmentation fault.

    """

    def __init__(self, text_target: wx.TextCtrl) -> None:
        super().__init__()
        self.text_target = text_target

    def OnDropFiles(self, x: int, y: int, filenames: list[str]) -> None:
        msg = f"DROP! {filenames!r}"
        logging.info(msg)
        if len(filenames) == 1:
            filenames = f"'{filenames[0]}'"
        self.text_target.ReplaceSelection(str(filenames))


class TextDropTarget(wx.TextDropTarget):
    def __init__(self, text_target: wx.TextCtrl) -> None:
        super().__init__()
        self.text_target = text_target

    def OnDropText(self, x: int, y: int, text: str) -> None:
        msg = f"DROP! {text!r}"
        logging.info(msg)
        self.text_target.ReplaceSelection(text)


class StringDropTarget(wx.DropTarget):
    """
    DropTarget for multiple data types

    based on:
    http://www.wiki.wxpython.org/DragAndDrop#wxDataObjectComposite

    """

    def __init__(self, target: wx.TextCtrl) -> None:
        super().__init__()
        self.target = target

        self.do = wx.DataObjectComposite()
        self.filedo = wx.FileDataObject()
        self.textdo = wx.TextDataObject()
        self.do.Add(self.filedo)
        self.do.Add(self.textdo)
        self.SetDataObject(self.do)

    def OnData(self, x: int, y: int, d: wx.DragResult) -> wx.DragResult:
        if self.GetData():
            df = self.do.GetReceivedFormat().GetType()

            if df in [wx.DF_UNICODETEXT, wx.DF_TEXT]:
                string = self.textdo.GetText()
            elif df == wx.DF_FILENAME:
                filenames = self.filedo.GetFilenames()
                string = filename_repr(filenames)

            msg = f"OnData! {string!r}"
            logging.info(msg)
            self.target.ReplaceSelection(string)
        return d


def set_for_strings(window: wx.Window) -> None:
    drop_target = StringDropTarget(window)
    window.SetDropTarget(drop_target)
