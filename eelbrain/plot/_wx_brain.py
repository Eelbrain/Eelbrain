"""Embedd Mayavi in Eelbrain

for testing:

src = datasets.get_mne_sample(src='ico', sub=[0])['src']
brain = plot.brain.brain(src.source, mask=False,hemi='lh',views='lat')
"""
from logging import getLogger
from typing import Optional

from mayavi.core.ui.api import SceneEditor, MlabSceneModel
import numpy as np
from traits.api import HasTraits, Instance
from traitsui.api import View, Item, HGroup, VGroup
from tvtk.api import tvtk
from tvtk.pyface.toolkit import toolkit_object

from .._utils import IS_OSX
from .._wxgui import wx, ID, Icon
from .._wxgui.app import get_app
from .._wxgui.frame import EelbrainFrame
from .._wxgui.mpl_canvas import AxisLimitsDialog, SetTimeDialog
from ._brain_object import Brain


SCENE_NAME = 'scene_%i'
SURFACES = ('inflated', 'pial', 'smoothwm')

# undecorated scene
Scene = toolkit_object('scene:Scene')


class MayaviView(HasTraits):

    view = Instance(View)

    def __init__(self, width, height, n_rows, n_columns):
        HasTraits.__init__(self)

        n_scenes = n_rows * n_columns
        if n_scenes < 1:
            raise ValueError(f"{n_rows=}, {n_columns=}")

        self.scenes = [MlabSceneModel() for _ in range(n_scenes)]
        for i, scene in enumerate(self.scenes):
            self.add_trait(SCENE_NAME % i, scene)

        if n_rows == n_columns == 1:
            item = Item(SCENE_NAME % 0, editor=SceneEditor(scene_class=Scene), resizable=True, show_label=False)
            self.view = View(item, width=width, height=height, resizable=True)
        else:
            rows = []
            for row in range(n_rows):
                columns = []
                for column in range(n_columns):
                    i = row * n_columns + column
                    item = Item(SCENE_NAME % i, editor=SceneEditor(scene_class=Scene), resizable=True, show_label=False)
                    columns.append(item)
                rows.append(HGroup(*columns))
            self.view = View(VGroup(*rows))

        self.figures = [scene.mayavi_scene for scene in self.scenes]


class BrainFrame(EelbrainFrame):
    _allow_user_set_title = True

    def __init__(
            self,
            parent: Optional[wx.Window],
            brain: Brain,
            title: str,
            width: int,
            height: int,
            n_rows: int,
            n_columns: int,
            surf: str,
            pos: wx.Position = None,
    ):
        pos_ = wx.DefaultPosition if pos is None else pos
        EelbrainFrame.__init__(self, parent, wx.ID_ANY, f"Brain: {title}", pos_)

        # toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))
        tb.AddTool(wx.ID_SAVE, "Save", Icon("tango/actions/document-save"))
        self.Bind(wx.EVT_TOOL, self.OnSaveAs, id=wx.ID_SAVE)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUISave, id=wx.ID_SAVE)
        # color-bar
        tb.AddTool(ID.PLOT_COLORBAR, "Plot Colorbar", Icon("plot/colorbar"))
        tb.Bind(wx.EVT_TOOL, self.OnPlotColorBar, id=ID.PLOT_COLORBAR)
        # surface
        self._surf_selector = wx.Choice(tb, choices=[name.capitalize() for name in SURFACES], name='Surface')
        if surf in SURFACES:
            self._surf_selector.SetSelection(SURFACES.index(surf))
        tb.AddControl(self._surf_selector, "Surface")
        self._surf_selector.Bind(
            wx.EVT_CHOICE, self.OnChoiceSurface, source=self._surf_selector)
        # view
        tb.AddTool(ID.VIEW_LATERAL, "Lateral View", Icon('brain/lateral'))
        self.Bind(wx.EVT_TOOL, self.OnSetView, id=ID.VIEW_LATERAL)
        tb.AddTool(ID.VIEW_MEDIAL, "Medial View", Icon('brain/medial'))
        self.Bind(wx.EVT_TOOL, self.OnSetView, id=ID.VIEW_MEDIAL)
        tb.AddTool(ID.SMOOTHING, "Smoothing Steps", Icon('brain/smoothing'))
        self.Bind(wx.EVT_TOOL, self.OnSetSmoothing, id=ID.SMOOTHING)
        # attach
        tb.AddStretchableSpace()
        tb.AddTool(ID.ATTACH, "Attach", Icon("actions/attach"))
        self.Bind(wx.EVT_TOOL, self.OnAttach, id=ID.ATTACH)
        tb.Realize()

        self.mayavi_view = MayaviView(width, height, n_rows, n_columns)
        self._n_rows = n_rows
        self._n_columns = n_columns
        # Use traits to create a panel, and use it as the content of this
        # wx frame.
        self.ui = self.mayavi_view.edit_traits(parent=self, view=self.mayavi_view.view, kind='subpanel')
        self.panel = self.ui.control
        # Hide the toolbar (the edit_traits command assigns scene_editor)
        for scene in self.mayavi_view.scenes:
            scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        if IS_OSX:
            # Bug in Mayavi/VTK: the render_window has twice the size of its panel, so it's only partially visible; this moves the relevant part into the center; see also Brain.set_parallel_view()
            figure = self.mayavi_view.figures[0]
            figure.scene.camera.window_center = [0.5, 0.5]

        self.SetImageSize(width, height)

        self.figure = self.mayavi_view.figures
        self._brain = brain
        self.Bind(wx.EVT_CLOSE, self.OnClose)  # remove circular reference

        # replace key bindings
        self.panel.Unbind(wx.EVT_KEY_DOWN)
        for child in self.panel.Children[0].Children:
            panel = child.Children[0]
            panel.Unbind(wx.EVT_CHAR)
            panel.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

    def CanCopy(self):
        return True

    def Copy(self):
        ss = self._brain.screenshot('rgba', True)
        ss = np.round(ss * 255).astype(np.uint8)
        h, w, _ = ss.shape
        image = wx.ImageFromDataWithAlpha(
            w, h, ss[:, :, :3].tostring(), ss[:, :, 3].tostring())
        bitmap = image.ConvertToBitmap()
        data = wx.BitmapDataObject(bitmap)
        if not wx.TheClipboard.Open():
            getLogger('eelbrain').debug("Failed to open clipboard")
            return
        try:
            wx.TheClipboard.SetData(data)
        finally:
            wx.TheClipboard.Close()
            wx.TheClipboard.Flush()

    def OnAttach(self, event):
        get_app().Attach(self._brain, "Brain plot", 'brain', self)

    def OnChoiceSurface(self, event):
        self._brain._set_surf(SURFACES[event.GetSelection()])

    def OnClose(self, event):
        event.Skip()
        if self._brain is not None:
            self._brain._surfer_close()
            # remove circular references
            self._brain._frame = None
            self._brain = None

    def OnKeyDown(self, event):
        if self._brain is None:
            return  # plot is closed
        key = chr(event.GetUnicodeKey())
        if key == '.':
            self._brain._nudge_time(1)
        elif key == ',':
            self._brain._nudge_time(-1)
        else:
            event.Skip()

    def OnPlotColorBar(self, event):
        if self._brain._has_data():
            self._brain.plot_colorbar()
        elif self._brain._has_annot() or self._brain._has_labels():
            self._brain.plot_legend()

    def OnSave(self, event):
        self.OnSaveAs(event)

    def OnSaveAs(self, event):
        default_file = '%s.png' % self.GetTitle().replace(': ', ' - ')
        dlg = wx.FileDialog(self, "If no file type is selected below, it is "
                                  "inferred from the extension.",
                            defaultFile=default_file,
                            wildcard="Any (*.*)|*.*|PNG (*.png)|*.png",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            # no antialiasing (leads to loss of alpha channel)
            self._brain.save_image(dlg.GetPath(), 'rgba')
        dlg.Destroy()

    def OnSetSmoothing(self, event):
        props = self._brain.get_data_properties()
        old_value = props['smoothing_steps']
        dlg = wx.TextEntryDialog(self, "Data overlay smoothing steps:",
                                 "Smoothing Steps", str(old_value))
        value = None
        while True:
            if dlg.ShowModal() != wx.ID_OK:
                break
            try:
                value = int(dlg.GetValue())
                if value < 1:
                    raise ValueError("Needs to be at least 1")
                elif value > 100:
                    raise ValueError("Values > 100 take too long")
            except Exception as exception:
                msg = wx.MessageDialog(self, str(exception), "Invalid Entry",
                                       wx.OK | wx.ICON_ERROR)
                msg.ShowModal()
                msg.Destroy()
            else:
                break
        dlg.Destroy()
        if value is not None and value != old_value:
            self._brain.set_data_smoothing_steps(value)

    def OnSetView(self, event):
        if event.Id == ID.VIEW_LATERAL:
            views = ('lateral', 'medial')
        elif event.Id == ID.VIEW_MEDIAL:
            views = ('medial', 'lateral')
        else:
            return

        for row, view in zip(self._brain.brain_matrix, views):
            for b in row:
                b.show_view(view)
        # Re-adjust position to account for mayavi bug
        self._brain.set_parallel_view(scale=True)

    def OnSetVLim(self, event):
        vlim = self._brain.get_vlim()
        dlg = AxisLimitsDialog(vlim, None, None, self)
        if dlg.ShowModal() == wx.ID_OK:
            self._brain.set_vlim(*dlg.vlim)
        dlg.Destroy()

    def OnSetTime(self, event):
        current_time = self._brain.get_time()
        dlg = SetTimeDialog(self, current_time)
        if dlg.ShowModal() == wx.ID_OK:
            self._brain.set_time(dlg.time)
        dlg.Destroy()

    def OnUpdateUISave(self, event):
        event.Enable(True)

    def OnUpdateUISaveAs(self, event):
        event.Enable(True)

    def OnUpdateUISetVLim(self, event):
        event.Enable(self._brain._has_data())

    def OnUpdateUISetTime(self, event):
        event.Enable(self._brain._has_data())

    def SetImageSize(self, width, height):
        if self._n_columns == 1 and self._n_rows == 1:
            width += 2
            height += 2
        else:
            width += self._n_columns * 2 + 4
            height += self._n_rows * 2 + 4
        self.SetClientSize((width, height))
