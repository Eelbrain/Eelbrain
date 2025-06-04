import logging

import wx


def filename_repr(filenames):
    if filenames:
        if isinstance(filenames, str):
            filenames = [filenames]

#        if 'wxMSW' in wx.PlatformInfo:
            # does windows require separate handling for to backslash?
            # on os-x, if the path contains a backslash, it is inserted as '\\'
#            conv = []
#            for name in filenames:
#                try:
#                    conv.append('r%r' % str(name))
#                except:
#                    conv.append(repr(name))
#
#            if len(filenames) == 1:
#                string = filenames[0]
#            else:
#                string = '[' + ', '.join(filenames) + ']'
#
#        else:
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

    def __init__(self, text_target):
        wx.FileDropTarget.__init__(self)
        self.text_target = text_target

    def OnDropFiles(self, x, y, filenames):
        msg = "DROP! %r" % filenames
        logging.info(msg)
        if len(filenames) == 1:
            filenames = "'%s'" % (filenames[0])
        self.text_target.ReplaceSelection(str(filenames))


class TextDropTarget(wx.TextDropTarget):
    def __init__(self, text_target):
        wx.TextDropTarget.__init__(self)
        self.text_target = text_target

    def OnDropText(self, x, y, text):
        msg = "DROP! %r" % text
        logging.info(msg)
        self.text_target.ReplaceSelection(text)


class StringDropTarget(wx.DropTarget):
    """
    DropTarget for multiple data types

    based on:
    http://www.wiki.wxpython.org/DragAndDrop#wxDataObjectComposite

    """

    def __init__(self, target):
        wx.DropTarget.__init__(self)
        self.target = target

        self.do = wx.DataObjectComposite()  # the dataobject that gets filled with the appropriate data
        self.filedo = wx.FileDataObject()
        self.textdo = wx.TextDataObject()
#        self.bmpdo = wx.BitmapDataObject()
        self.do.Add(self.filedo)
        self.do.Add(self.textdo)
#        self.do.Add(self.bmpdo)
        self.SetDataObject(self.do)

    def OnData(self, x, y, d):
        if self.GetData():
            df = self.do.GetReceivedFormat().GetType()
#            data = self.GetData()

            if df in [wx.DF_UNICODETEXT, wx.DF_TEXT]:
                string = self.textdo.GetText()
            elif df == wx.DF_FILENAME:
                filenames = self.filedo.GetFilenames()
                string = filename_repr(filenames)

            msg = "OnData! %r" % string
            logging.info(msg)
            self.target.ReplaceSelection(string)
        return d


def set_for_strings(window):
    drop_target = StringDropTarget(window)
    window.SetDropTarget(drop_target)
