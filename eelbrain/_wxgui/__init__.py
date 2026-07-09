"""Import wx through this module to modify up logging"""
from warnings import filterwarnings

filterwarnings('ignore', message='Not importing directory .*', module='wx.*')
import wx

# filter unnecessary warnings
from .. import _config
if _config.SUPPRESS_WARNINGS:
    filterwarnings('ignore', category=wx.wxPyDeprecationWarning)
    filterwarnings('ignore', 'NewId()', category=DeprecationWarning)
    filterwarnings('ignore', module='(traitsui|pyface|apptools)', category=DeprecationWarning)
    filterwarnings('ignore', 'invalid escape sequence', DeprecationWarning)  # tvtk
    wx.Log.EnableLogging(False)

from .utils import Icon, show_text_dialog
from .app import needs_jumpstart, get_app, run
from . import history, select_channels, select_epochs, select_components, load_stcs
