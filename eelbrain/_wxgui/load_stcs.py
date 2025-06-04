"""GUI to detect and load stc files and experimental conditions

Prompts user for experiment design information, and upon submission
loads stcs into an ``eelbrain.Dataset`` via a ``DatasetSTCLoader``
instance.
"""

import os
import wx

from . import get_app
from .frame import EelbrainFrame
from .._io.stc_dataset import DatasetSTCLoader


TEST_MODE = False


class STCLoaderFrame(EelbrainFrame):

    add_params = dict(proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

    def __init__(self, parent):
        super().__init__(parent, wx.ID_ANY, "Find and Load STCs")
        self.loader = None
        self.factor_name_ctrls = []
        self.InitUI()
        self.Show()
        self.Center()
        self.Raise()

    def InitUI(self):
        self.SetMinSize((500, -1))
        self.status = self.CreateStatusBar(1)
        self.sizer = wx.BoxSizer(wx.VERTICAL)  # top-level sizer
        data_title = TitleSizer(self, "MEG/MRI Information")
        self.sizer.Add(data_title, **self.add_params)
        dir_label = wx.StaticText(self, label=".stc Data Directory")
        self.dir_ctl = wx.DirPickerCtrl(self)
        if self.dir_ctl.HasTextCtrl():
            self.dir_ctl.SetTextCtrlProportion(5)
            self.dir_ctl.SetPickerCtrlProportion(1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(dir_label, 0, wx.BOTTOM, 2)
        vsizer.Add(self.dir_ctl, 0, wx.EXPAND)
        self.sizer.Add(vsizer, **self.add_params)
        self.mri_panel = MRIPanel(self)
        self.sizer.Add(self.mri_panel, **self.add_params)
        self.design_title = TitleSizer(self, "Experiment Design")
        self.sizer.Add(self.design_title, **self.add_params)
        self.design_title.ctl.Hide()
        self.factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.factor_sizer, **self.add_params)
        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.attach_ds = wx.Button(self, wx.ID_ANY, "Return Dataset")
        self.launch_stats = wx.Button(self, wx.ID_ANY, "Launch Stats GUI")
        self.attach_ds.Disable()
        self.launch_stats.Disable()
        bottom_sizer.Add(self.attach_ds, 0)
        bottom_sizer.Add(self.launch_stats, 0)
        self.sizer.Add(bottom_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.sizer.Layout()
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        self.Bind(wx.EVT_DIRPICKER_CHANGED, self.OnDirChange, self.dir_ctl)
        self.Bind(wx.EVT_BUTTON, self.OnAttachDataset, self.attach_ds)
        self.Bind(wx.EVT_BUTTON, self.OnLaunchStats, self.launch_stats)

    def OnDirChange(self, dir_picker_evt):
        """Create dataset loader and display level/factor names"""
        self.factor_sizer.Clear(True)
        path = dir_picker_evt.GetPath()
        try:
            self.loader = DatasetSTCLoader(path)
            self.status.SetStatusText("")
        except ValueError as err:
            self.status.SetStatusText(str(err))
            return
        self.DisplayLevels(self.loader.levels)
        self.attach_ds.Enable()
        self.launch_stats.Enable()

    def DisplayLevels(self, levels):
        """Show level names and factor name input for each factor"""
        self.design_title.ctl.Show()
        self.factor_name_ctrls = []
        for i, lvls in enumerate(levels):
            panel = FactorPanel(self, lvls, i)
            # self.factor_sizer.Add(panel, 0, wx.EXPAND | wx.RIGHT, 30)
            self.factor_sizer.Add(panel, 0, wx.EXPAND | wx.RIGHT, 30)
            self.factor_name_ctrls.append(panel.factor_ctl)
        self.factor_sizer.Layout()
        self.sizer.Layout()
        self.sizer.Fit(self)

    def _get_factor_names(self):
        return [c.GetValue() for c in self.factor_name_ctrls]

    def _get_stc_kwargs(self):
        kw = dict()
        kw["subjects_dir"] = self.mri_panel.mri_dir.GetPath()
        kw["subject"] = self.mri_panel.mri_subj.GetValue()
        kw["src"] = self.mri_panel.mri_src.GetValue()
        return kw

    def _get_dataset(self):
        names = self._get_factor_names()
        self.loader.set_factor_names(names)
        stc_kw = self._get_stc_kwargs()
        return self.loader.make_dataset(**stc_kw)

    def OnAttachDataset(self, evt):
        ds = self._get_dataset()
        get_app().Attach(ds, "Dataset from STCs", "ds", self)
        self.Close()

    def OnLaunchStats(self, evt):
        raise NotImplementedError()


class FactorPanel(wx.Panel):
    """Panel to display a factor and its level names"""
    def __init__(self, parent, levels, idx, editable=True):
        super(FactorPanel, self).__init__(parent)
        if editable:
            self.factor_ctl = wx.TextCtrl(self, value="factor_%d" % idx)
        else:
            self.factor_ctl = wx.StaticText(self, label="factor_%d" % idx)
            self.factor_ctl.SetFont(wx.Font.Bold(self.factor_ctl.GetFont()))
        level_names = ["- " + i for i in levels]
        level_ctl = wx.StaticText(self)
        level_ctl.SetLabel("\n".join(level_names))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.factor_ctl)
        sizer.Add(level_ctl, 0, wx.EXPAND | wx.TOP, 10)
        sizer.Layout()
        self.SetSizer(sizer)


class MRIPanel(wx.Panel):
    """Panel containing MRI input fields (dir, subject, src)"""
    def __init__(self, parent):
        super(MRIPanel, self).__init__(parent)
        sizer = wx.StaticBoxSizer(wx.VERTICAL, self, "")
        # check for subjects dir in environment
        sdir = os.environ.get("SUBJECTS_DIR")
        if sdir is None:
            sdir = "/Applications/freesurfer/subjects"
        # directory control for MRI subjects dir, default freesurfer
        dir_ctl = wx.DirPickerCtrl(self, path=sdir)
        if dir_ctl.HasTextCtrl():
            dir_ctl.SetTextCtrlProportion(5)
            dir_ctl.SetPickerCtrlProportion(1)
        # text control for MRI subject, default 'fsaverage'
        subj_ctl = wx.TextCtrl(self, value="fsaverage")
        # dropdown to choose source space sampling, default ico-4
        srcs = ["ico-%d" % i for i in range(2, 7)] + ["oct-%d" % i for i in range(2, 7)] + ["all"]
        src_ctl = wx.ComboBox(self, choices=srcs, value="ico-4")
        # attach controls to panel for use in loader
        self.mri_dir = dir_ctl
        self.mri_subj = subj_ctl
        self.mri_src = src_ctl
        sizer.Add(wx.StaticText(self, label="MRI Directory"), 0, wx.BOTTOM, 2)
        sizer.Add(dir_ctl, 0, wx.EXPAND | wx.BOTTOM, 5)
        hs = wx.BoxSizer(wx.HORIZONTAL)
        for label, ctl in zip(("Subject", "Source Space"),
                              (subj_ctl, src_ctl)):
            vs = wx.BoxSizer(wx.VERTICAL)
            vs.Add(wx.StaticText(self, label=label), 0)
            vs.Add(ctl, 0)
            hs.Add(vs, 0, wx.RIGHT, 15)
        sizer.Add(hs, 0, wx.BOTTOM, 5)
        sizer.Layout()
        self.SetSizer(sizer)


class TitleSizer(wx.BoxSizer):
    def __init__(self, parent, title):
        super().__init__(wx.HORIZONTAL)
        self.ctl = wx.StaticText(parent, label=title)
        font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.ctl.SetFont(font)
        self.Add(self.ctl)
