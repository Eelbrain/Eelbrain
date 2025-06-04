import wx
from eelbrain import gui
from eelbrain.testing import gui_test, TempDir
from eelbrain._io.tests.test_stc_dataset import _create_fake_files


@gui_test
def test_load_stcs():
    tmp = TempDir()
    _create_fake_files(tmp)
    frame = gui.load_stcs()
    assert frame.dir_ctl.GetPath() == ""
    frame.dir_ctl.SetPath(tmp)
    evt = wx.FileDirPickerEvent()
    evt.SetPath(tmp)
    frame.OnDirChange(evt)
    assert frame.loader is not None
    assert len(frame.factor_name_ctrls) == 2
    assert "verb" in frame.loader.levels[1]
    frame.Close()


@gui_test
def test_failed_load():
    tmp = TempDir()
    frame = gui.load_stcs()
    frame.dir_ctl.SetPath(tmp)
    evt = wx.FileDirPickerEvent()
    evt.SetPath(tmp)
    frame.OnDirChange(evt)
    assert "No .stc" in frame.status.GetStatusText()
    frame.Close()
