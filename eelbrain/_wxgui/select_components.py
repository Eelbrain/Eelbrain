"""GUI for selecting topographical components"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizaes Document
#  - listens to Document changes
#  - issues commands to Model
from collections import defaultdict
from itertools import repeat
from math import ceil
from operator import itemgetter
import re
from typing import Sequence, Tuple, Union

import mne
import matplotlib.figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from scipy import linalg
import seaborn
import wx
from wx.lib.scrolledpanel import ScrolledPanel

from .. import load, plot, fmtxt
from .._colorspaces import UNAMBIGUOUS_COLORS
from .._data_obj import Dataset, Factor, NDVar, Categorial, Scalar, combine
from .._io.fiff import _picks
from .._ndvar import concatenate, neighbor_correlation
from .._types import PathArg
from .._utils.numpy_utils import INT_TYPES
from .._utils.parse import FLOAT_PATTERN, POS_FLOAT_PATTERN
from .._utils.system import IS_OSX
from ..mne_fixes._version import MNE_VERSION, V0_24
from ..plot._base import DISPLAY_UNIT, UNIT_FORMAT, AxisData, DataLayer, PlotType
from ..plot._topo import AxTopomap
from .frame import EelbrainDialog
from .history import Action, FileDocument, FileModel, FileFrame, FileFrameChild
from .mpl_canvas import FigureCanvasPanel
from .text import HTML2Frame as HTMLFrame
from .utils import Icon, REValidator
from . import ID


COLOR = {True: (.5, 1, .5), False: (1, .3, .3)}
LINE_COLOR = {True: 'k', False: (1, 0, 0)}
TOPO_ARGS = {
    'interpolation': 'linear',  # interpolation that does not assume continuity
    'clip': 'even',
}

# For unit-tests
TEST_MODE = False


class ChangeAction(Action):
    """Action objects are kept in the history and can do and undo themselves

    Parameters
    ----------
    desc : str
        Description of the action
        list of (i, name, old, new) tuples
    """
    def __init__(self, desc, index=None, old_accept=None, new_accept=None,
                 old_path=None, new_path=None):
        self.desc = desc
        self.index = index
        self.old_path = old_path
        self.old_accept = old_accept
        self.new_path = new_path
        self.new_accept = new_accept

    def do(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.new_accept)
        if self.new_path is not None:
            doc.set_path(self.new_path)

    def undo(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.old_accept)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)


class Document(FileDocument):
    """Represents data for the current state of the Document

    (Data can be accessed, but should only be modified through the Model)

    Parameters
    ----------
    path
        Path to the ICA file.
    data
        Dataset containing 'epochs' (mne Epochs), 'index' (Var describing
        epochs) and variables describing cases in epochs, used to plot
        condition averages.
    """
    def __init__(
            self,
            path: PathArg,
            data: Union[Dataset, mne.BaseEpochs],
            sysname: str,
            adjacency: Union[str, Sequence] = None,
            drop_epochs_std: float = None,  # drop epochs with high signal (e.g. 10)
            decim: int = None,
    ):
        FileDocument.__init__(self, path)
        self._ndvar_args = dict(sysname=sysname, adjacency=adjacency)
        self.saved = True
        self._explained_variance = {}

        if isinstance(data, mne.io.BaseRaw):
            events = mne.make_fixed_length_events(data)
            if decim is None:
                decim = int(round(data.info['sfreq'] / 100))
            ds = Dataset({'epochs': mne.Epochs(data, events, 1, 0, 1, baseline=None, proj=False, decim=decim, preload=True)})
        elif isinstance(data, mne.BaseEpochs):
            ds = Dataset({'epochs': data})
        elif isinstance(data, Dataset):
            ds = data
        else:
            raise TypeError(f'{data=}')

        # Exclude extreme epochs
        epochs_ndvar = self.as_ndvar(ds['epochs'])
        if drop_epochs_std:
            variance = epochs_ndvar.var('sensor').max('time')
            index = np.arange(len(variance))
            while True:
                sub_variance = variance[index]
                sub_variance -= sub_variance.mean()
                sub_variance /= sub_variance.std()
                print(f"Max variance: {sub_variance.max()} STD...")
                if sub_variance.max() > drop_epochs_std:
                    mask = sub_variance <= drop_epochs_std
                    print(f"Excluding {np.sum(~mask)} epochs > {drop_epochs_std} STD: {index[~mask]}...")
                    index = index[mask]
                else:
                    break
            ds = ds[index]
            epochs_ndvar = epochs_ndvar[index]

        # Read ICA
        self.ica = ica = mne.preprocessing.read_ica(path)
        self.accept = np.ones(self.ica.n_components_, bool)
        self.accept[ica.exclude] = False
        self.epochs = epochs = ds['epochs']
        self.epochs_ndvar = epochs_ndvar
        self.ds = ds
        # for 3d-data, pick magnetometers
        picks = _picks(ica.info, None, 'bads')

        # components
        data = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_])
        ic_dim = Scalar('component', np.arange(len(data)))
        self.components = NDVar(data[:, picks], (ic_dim, self.epochs_ndvar.sensor), 'components', {'meas': 'component', 'cmap': 'xpolar'})

        # sources
        data = ica.get_sources(epochs).get_data(copy=False)
        self.sources = NDVar(data, ('case', ic_dim, self.epochs_ndvar.time), 'sources', {'meas': 'component', 'cmap': 'xpolar'})

        # find unique epoch labels
        if 'index' in ds:
            labels = map(str, ds['index'])
            if 'epoch' in ds:
                labels = map(' '.join, zip(ds['epoch'], labels))
        else:
            labels = map(str, range(len(epochs)))
        self.epoch_labels = tuple(labels)

        # properties which are not modified by ICA
        # global mean
        if ica.noise_cov is None:  # revert standardization
            global_mean = ica.pca_mean_ * ica.pre_whitener_[:, 0]
        else:
            global_mean = np.dot(linalg.pinv(ica.pre_whitener_), ica.pca_mean_)
        self.global_mean = NDVar(global_mean[picks], (self.epochs_ndvar.sensor,))
        # pre-ICA signal range
        self.pre_ica_min = self.epochs_ndvar.min('sensor')
        self.pre_ica_max = self.epochs_ndvar.max('sensor')
        self.pre_ica_range_scale = (self.pre_ica_max.mean() - self.pre_ica_min.mean())
        self.pre_ica_min /= self.pre_ica_range_scale
        self.pre_ica_max /= self.pre_ica_range_scale

        # publisher
        self.callbacks.register_key('case_change')

    def as_ndvar(self, epochs):
        if isinstance(epochs, mne.BaseEpochs):
            return load.mne.epochs_ndvar(epochs, **self._ndvar_args)
        elif isinstance(epochs, mne.Evoked):
            return load.mne.evoked_ndvar(epochs, **self._ndvar_args)
        raise TypeError(f"{epochs=}")

    def apply(self, inst):
        if isinstance(inst, list):
            return [self.ica.apply(i.copy()) for i in inst]
        else:
            return self.ica.apply(inst.copy())

    def set_case(self, index, state):
        self.accept[index] = state
        self.ica.exclude = list(np.flatnonzero(np.invert(self.accept)))
        self.callbacks.callback('case_change', index)

    def save(self):
        if MNE_VERSION >= V0_24:
            self.ica.save(self.path, overwrite=True)
        else:
            self.ica.save(self.path)

    def explained_variance(self, component: int, format: bool = False):
        if component not in self._explained_variance:
            self._explained_variance[component] = self.ica.get_explained_variance_ratio(self.epochs, components=component)
        desc_dict = self._explained_variance[component]
        if format:
            if len(desc_dict) == 1:
                return ''.join(f'{v:.1%}' for v in desc_dict.values())
            else:
                return ', '.join([f'{k}: {v:.1%}' for k, v in desc_dict.items()])
        return desc_dict


class Model(FileModel):
    """Manages a document with its history"""

    def __init__(self, doc: Document):
        FileModel.__init__(self, doc)

    def set_case(self, index, state, desc="Manual Change"):
        old_accept = self.doc.accept[index]
        action = ChangeAction(desc, index, old_accept, state)
        self.history.do(action)

    def toggle(self, case):
        old_accept = self.doc.accept[case]
        action = ChangeAction("Manual toggle", case, old_accept, not old_accept)
        self.history.do(action)

    def clear(self):
        action = ChangeAction("Clear", slice(None), self.doc.accept.copy(), True)
        self.history.do(action)


class ContextMenu(wx.Menu):
    "Helper class for Menu to store component ID"
    def __init__(self, i_comp: int = None, i_epoch: int = None):
        wx.Menu.__init__(self)
        self.i_comp = i_comp
        self.i_epoch = i_epoch


class SharedToolsMenu:  # Frame mixin
    # set by FileFrame:
    doc = None
    config = None
    # MakeToolsMenu() might be called before __init__
    butterfly_baseline = ID.BASELINE_NONE
    last_model = ""

    def AddToolbarButtons(self, tb):
        button = wx.Button(tb, label="Rare Events")
        button.Bind(wx.EVT_BUTTON, self.OnFindRareEvents)
        tb.AddControl(button)
        button = wx.Button(tb, label="Noisy Epochs")
        button.Bind(wx.EVT_BUTTON, self.OnFindNoisyEpochs)
        tb.AddControl(button)
        button = wx.Button(tb, label="PSD")
        button.Bind(wx.EVT_BUTTON, self.OnPlotPSD)
        tb.AddControl(button)

    def MakeToolsMenu(self, menu):
        app = wx.GetApp()

        # Artifact detection helpers
        item = menu.Append(wx.ID_ANY, "Find Rare Events", "Find components with major loading on a small number of epochs")
        app.Bind(wx.EVT_MENU, self.OnFindRareEvents, item)
        item = menu.Append(wx.ID_ANY, "Find Noisy Epochs", "Find epochs with strong signal")
        app.Bind(wx.EVT_MENU, self.OnFindNoisyEpochs, item)
        item = menu.Append(wx.ID_ANY, "Find Bad Channels", "Find components that are likely due to bad channels")
        app.Bind(wx.EVT_MENU, self.OnFindBadChannels, item)
        menu.AppendSeparator()

        # plotting
        item = menu.Append(wx.ID_ANY, "Butterfly Plot Grand Average", "Plot the grand average of all epochs")
        app.Bind(wx.EVT_MENU, self.OnPlotGrandAverage, item)
        item = menu.Append(wx.ID_ANY, "Butterfly Plot by Category", "Separate butterfly plots for different model cells")
        app.Bind(wx.EVT_MENU, self.OnPlotButterfly, item)
        # Baseline submenu
        blmenu = wx.Menu()
        blmenu.AppendRadioItem(ID.BASELINE_CUSTOM, "Baseline Period")
        blmenu.AppendRadioItem(ID.BASELINE_GLOABL_MEAN, "Global Mean")
        blmenu.AppendRadioItem(ID.BASELINE_NONE, "No Baseline Correction")
        blmenu.Check(self.butterfly_baseline, True)
        blmenu.Bind(wx.EVT_MENU, self.OnSetButterflyBaseline, id=ID.BASELINE_CUSTOM)
        blmenu.Bind(wx.EVT_MENU, self.OnSetButterflyBaseline, id=ID.BASELINE_GLOABL_MEAN)
        blmenu.Bind(wx.EVT_MENU, self.OnSetButterflyBaseline, id=ID.BASELINE_NONE)
        menu.AppendSubMenu(blmenu, "Baseline")

    def OnFindNoisyEpochs(self, event):
        unit = self.doc.epochs_ndvar.info.get('unit', '<unknown unit>')
        if unit in DISPLAY_UNIT:
            display_unit = DISPLAY_UNIT[unit]
            scale_factor = 1 / UNIT_FORMAT[display_unit]
        else:
            display_unit = unit
            scale_factor = None
        dlg = FindNoisyEpochsDialog(self, unit=display_unit)
        rcode = dlg.ShowModal()
        dlg.Destroy()
        if rcode != wx.ID_OK:
            return
        threshold = float(dlg.threshold.GetValue())
        threshold_desc = f'{threshold:g} {display_unit}'
        if scale_factor:
            threshold *= scale_factor
        apply_rejection = dlg.apply_rejection.GetValue()
        sort_by_component = dlg.sort_by_component.GetValue()
        max_ch_ratio = dlg.max_ch_ratio.GetValue()
        if max_ch_ratio:
            max_ch_ratio = float(max_ch_ratio)
        else:
            max_ch_ratio = 0
        dlg.StoreConfig()

        # compute and rank
        if apply_rejection:
            epochs = self.doc.as_ndvar(self.doc.apply(self.doc.epochs))
        else:
            epochs = self.doc.epochs_ndvar
        peaks = epochs.extrema(('time', 'sensor')).abs().x

        # collect output
        res = [(i, peak) for i, peak in enumerate(peaks) if peak >= threshold]  # epoch, value
        if len(res) == 0:
            wx.MessageBox(f"No epochs with signals exceeding {threshold_desc} were found.", "No Noisy Epochs Found", style=wx.ICON_INFORMATION)
            return

        if sort_by_component:
            res_by_component = defaultdict(list)
            # Find contribution of each component
            component_magnitude = self.doc.components.abs().sum('sensor')
            if apply_rejection:
                component_magnitude.x *= self.doc.accept
            magnitude = self.doc.sources.abs().sum('time') * component_magnitude
            for i, peak in res:
                magnitude_i = magnitude[i]
                c_max = magnitude_i.argmax()
                ratio = magnitude_i[c_max] / magnitude_i.sum()
                res_by_component[c_max].append((i, peak, ratio))
            # Sort epochs by ratio
            for res_list in res_by_component.values():
                res_list.sort(key=itemgetter(2), reverse=True)
            # Sort components by max ratio
            max_ratio = {component: values[0][2] for component, values in res_by_component.items()}
            sorted_components = sorted(max_ratio, key=lambda c: max_ratio[c], reverse=True)
            res_by_component = {c: res_by_component[c] for c in sorted_components}
        else:
            res_by_component = None

        # format output
        doc = fmtxt.Section("Noisy epochs")
        doc.add_paragraph(f"Epochs with signal exceeding {threshold_desc}")
        if sort_by_component:
            doc.add_paragraph("Sorted by dominant component")
        doc.append(fmtxt.linebreak)
        if sort_by_component:
            for component, values in res_by_component.items():
                # test whether this is a single noisy channel
                channel_values = np.sort(np.abs(self.doc.components[component].x))
                max_channel_ratio = channel_values[-1] / channel_values[-2]
                if max_ch_ratio and max_channel_ratio > max_ch_ratio:
                    continue
                # plot component map
                figure = matplotlib.figure.Figure(figsize=(1, 1))
                canvas = FigureCanvasAgg(figure)
                axes = figure.add_subplot()
                plot.Topomap(self.doc.components[component], axes=axes)
                image = fmtxt.Image(f'#{component}', 'jpg')
                canvas.print_jpeg(image)
                # Component properties
                # sec = doc.add_section(f"#{component}")
                sec = doc
                heading = fmtxt.FMTextElement(f"#{component}", 'h2')
                table = fmtxt.Table('lll', rules=False)
                table.cells(image, heading, f'{max_channel_ratio:.1f}')
                sec.add_paragraph(table)
                # sec.add_paragraph([image, f"Ch 1/2 ratio: {max_channel_ratio:.1f}", fmtxt.linebreak])
                # add links to epochs
                by_ratio = defaultdict(list)
                for i, peak, ratio in values:
                    by_ratio[f'{ratio:.0%}'].append(i)
                for ratio, epochs in by_ratio.items():
                    sec.append(f'{ratio}: ')
                    for i in epochs:
                        sec.append([fmtxt.Link(self.doc.epoch_labels[i], f'component:{component} epoch:{i}'), ', '])
                    sec.append(fmtxt.linebreak)
        else:
            for i, peak in res:
                doc.append(fmtxt.Link(self.doc.epoch_labels[i], f'epoch:{i}'))
                doc.append(f": {peak:g}")
                doc.append(fmtxt.linebreak)
        InfoFrame(self, "Noisy Epochs", doc, 300)

    def OnFindRareEvents(self, event):
        dlg = FindRareEventsDialog(self)
        rcode = dlg.ShowModal()
        dlg.Destroy()
        if rcode != wx.ID_OK:
            return
        threshold = float(dlg.threshold.GetValue())
        dlg.StoreConfig()

        # compute and rank
        y = self.doc.sources.max('time') - self.doc.sources.min('time')
        z = (y - y.mean('case')) / y.std('case')
        z_max = z.max('case').x
        components_ranked = np.argsort(z_max)[::-1]

        # collect output
        res = []
        for c in components_ranked:
            if z_max[c] < threshold:
                break
            z_epochs = z.x[:, c]
            idx = np.flatnonzero(z_epochs >= threshold)
            rank = np.argsort(z_epochs[idx])[::-1]
            res.append((c, z_max[c], idx[rank]))

        if len(res) == 0:
            wx.MessageBox("No rare events were found.", "No Rare Events Found", style=wx.ICON_INFORMATION)
            return

        # format output
        doc = fmtxt.Section("Rare Events")
        doc.add_paragraph(f"Components that disproportionally affect a small number of epochs (z-scored peak-to-peak > {threshold:g}). Epochs are ranked by peak-to-peak.")
        doc.append(fmtxt.linebreak)
        hash_char = {True: fmtxt.FMTextElement('# ', 'font', {'color': 'green'}),
                     False: fmtxt.FMTextElement('# ', 'font', {'color': 'red'})}
        for c, ft, epochs in res:
            doc.append(hash_char[self.doc.accept[c]])
            doc.append(f"{c} ({ft:.1f}):  ")
            doc.append(fmtxt.delim_list((fmtxt.Link(self.doc.epoch_labels[e], f'component:{c} epoch:{e}') for e in epochs)))
            doc.append(fmtxt.linebreak)
        InfoFrame(self, "Rare Events", doc, 500)

    def OnFindBadChannels(self, event):
        nc_before = neighbor_correlation(concatenate(self.doc.epochs_ndvar))
        if self.doc.accept.all():
            nc_after = None
        else:
            epochs = self.doc.as_ndvar(self.doc.apply(self.doc.epochs))
            nc_after = neighbor_correlation(concatenate(epochs))

        # Find ICA components that load on a single channel
        candidates = []
        for i, component_map in enumerate(self.doc.components):
            abs_comp = abs(component_map.x)
            argsort = np.argsort(abs_comp)
            if abs_comp[argsort[-1]] > abs_comp[argsort[-2]] * 3:
                ch_name = self.doc.epochs_ndvar.sensor.names[argsort[-1]]
                # Explained variance
                explained_desc = self.doc.explained_variance(i, format=True)
                explained_variance = max(self.doc.explained_variance(i).values())
                # Loading by epoch
                max_loadings = self.doc.sources[:, i].extrema('time').abs().x
                # Store
                candidates.append([i, ch_name, max_loadings, explained_desc, explained_variance])
        candidates = sorted(candidates, key=itemgetter(-1), reverse=True)

        # format output
        doc = fmtxt.Section("Bad Channels")

        # Neighbor correlation map
        for nc, desc in [[nc_before, 'raw data'], [nc_after, 'cleaned']]:
            if nc is None:
                continue
            figure = matplotlib.figure.Figure(figsize=(4, 3))
            axes = figure.add_axes((0.1, 0.1, .7, 0.8))
            p = plot.Topomap(nc, axes=axes, vmax=1, interpolation='linear')
            p.plot_colorbar(right_of=axes, ticks=3)
            image = fmtxt.Image(f'Neighbor correlation {desc}', 'jpg')
            canvas = FigureCanvasAgg(figure)
            canvas.print_jpeg(image)
            doc.append(image)

        # Candidate components
        for component, ch_name, max_loadings, explained_variance, _ in candidates:
            # plot component map
            figure = matplotlib.figure.Figure(figsize=(1, 1))
            canvas = FigureCanvasAgg(figure)
            axes = figure.add_subplot()
            plot.Topomap(self.doc.components[component], axes=axes, interpolation='linear')
            image = fmtxt.Image(f'#{component}', 'jpg')
            canvas.print_jpeg(image)

            # Text desc
            component_link = fmtxt.Link(f"#{component}", f'component:{component}')
            desc = fmtxt.FMText([ch_name, fmtxt.linebreak, component_link, fmtxt.linebreak, explained_variance])
            table = fmtxt.Table('lll', rules=False)
            doc.add_paragraph(table)

            # Loadings
            binrange = [0, max_loadings.max()]
            figure = matplotlib.figure.Figure(figsize=(2, 1))
            canvas = FigureCanvasAgg(figure)
            axes = figure.add_subplot()
            seaborn.histplot(max_loadings, binrange=binrange, bins=40, ax=axes)
            histogram = fmtxt.Image(f'#{component}', 'jpg')
            canvas.print_jpeg(histogram)

            table.cells(image, desc, histogram)

        InfoFrame(self, "Bad Channels", doc, 500)

    def OnPlotButterfly(self, event):
        self.PlotConditionAverages(self)

    def OnPlotGrandAverage(self, event):
        self.PlotEpochButterfly()

    def OnPlotPSD(self, event):
        self.PlotPSD()

    def OnSetButterflyBaseline(self, event):
        self.butterfly_baseline = event.GetId()

    def PlotConditionAverages(self, parent):
        "Prompt for model and plot condition averages"
        factors = [n for n, v in self.doc.ds.items() if
                   isinstance(v, Factor)]
        if len(factors) == 0:
            wx.MessageBox("The dataset that describes the epochs does not "
                          "contain any Factors that could be used to plot the "
                          "data by condition.", "No Factors in Dataset",
                          style=wx.ICON_ERROR)
            return
        elif len(factors) == 1:
            default = factors[0]
        else:
            default = self.last_model or factors[0]
        msg = "Specify the model (available factors: %s)" % ', '.join(factors)

        plot_model = None
        dlg = wx.TextEntryDialog(parent, msg, "Plot by Condition", default)
        while plot_model is None:
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                use = [s.strip() for s in value.replace(':', '%').split('%')]
                invalid = [f for f in use if f not in factors]
                if invalid:
                    wx.MessageBox("The following are not valid factor names: %s"
                                  % (', '.join(invalid)), "Invalid Entry",
                                  wx.ICON_ERROR)
                else:
                    plot_model = '%'.join(use)
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        self.last_model = value

        ds = self.doc.ds.aggregate(plot_model, drop_bad=True)
        titles = [' '.join(ds[i, f] for f in use) + ' (n=%i)' % ds[i, 'n'] for
                  i in range(ds.n_cases)]
        self._PlotButterfly(ds['epochs'], titles)

    def PlotEpochButterfly(self, i_epoch: int = None):
        if i_epoch is None:
            self._PlotButterfly(self.doc.epochs.average(), "Epochs Average")
        else:
            name = f"Epoch {self.doc.epoch_labels[i_epoch]}"
            self._PlotButterfly(self.doc.epochs[i_epoch], name)

    def _PlotButterfly(self, epoch, title):
        original = self.doc.as_ndvar(epoch)
        clean = self.doc.as_ndvar(self.doc.apply(epoch))
        if self.butterfly_baseline == ID.BASELINE_CUSTOM:
            if original.time.tmin >= 0:
                wx.MessageBox(f"The data displayed does not have a baseline period (tmin={original.time.tmin}). Change the baseline through the Tools menu.", "No Baseline Period", style=wx.ICON_ERROR)
                return
            original -= original.mean(time=(None, 0))
            clean -= clean.mean(time=(None, 0))
        elif self.butterfly_baseline == ID.BASELINE_GLOABL_MEAN:
            original -= self.doc.global_mean
            clean -= self.doc.global_mean

        if original.has_case:
            if isinstance(title, str):
                title = repeat(title, len(original))
            vmax = 1.1 * max(abs(original.min()), original.max())
            for data, title_ in zip(zip(original, clean), title):
                plot.TopoButterfly(data, vmax=vmax, title=title_, axtitle=("Original", "Cleaned"))
        else:
            plot.TopoButterfly([original, clean], title=title, axtitle=("Original", "Cleaned"))

    def PlotPSD(self):
        ds_original = Dataset({'psd': self.doc.as_ndvar(self.doc.epochs).fft().mean('sensor')})
        ds_original[:, 'data'] = 'Source'
        ds_clean = Dataset({'psd': self.doc.as_ndvar(self.doc.apply(self.doc.epochs)).fft().mean('sensor')})
        ds_clean[:, 'data'] = 'Cleaned'
        ds = combine((ds_original, ds_clean))
        colors = {'Source': 'red', 'Cleaned': 'blue'}
        plot.UTSStat('psd', 'data', data=ds, error=np.std, w=8, title="Spectrum (±1 STD)", colors=colors)


class Frame(SharedToolsMenu, FileFrame):
    """GIU for selecting ICA sensor-space components

    Component Selection
    ===================

    * Click on components topographies to select/deselect them.
    * Right-click for a context-menu.

    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    t           topomap plot of the component under the pointer
    a           array-plot of the source time course of the component
    s           plot sources, starting with the component under the cursor
    f           plot the frequency spectrum for the component under the
                pointer
    b           butterfly plot of grand average (original and cleaned)
    B           butterfly plot of condition averages
    =========== ============================================================
    """
    _doc_name = 'component selection'
    _title = 'Select Components'
    _wildcard = "ICA fiff file (*-ica.fif)|*.fif"

    def __init__(
            self,
            model: Model,
            parent: wx.Frame = None,
            pos: Tuple[int, int] = None,
            size: Tuple[int, int] = None,
    ):
        FileFrame.__init__(self, parent, pos, size, model)
        SharedToolsMenu.__init__(self)
        self.source_frame = None

        # setup layout
        self.ax_size = 200
        figsize = (10, 10)
        self.SetMinSize((400, 400))

        # setup scrolled panel
        panel = ScrolledPanel(self)
        self.panel = panel

        # setup figure canvas
        self.canvas = FigureCanvasPanel(panel, figsize=figsize)
        self.canvas.figure.subplots_adjust(0, 0, 1, 1, 0, 0)
        panel.SetupScrolling(False, scrollToTop=False, scrollIntoView=False)

        # sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 0)
        panel.SetSizer(sizer)
        self.canvas_sizer = sizer

        # Toolbar
        tb = self.InitToolbar(can_open=False)
        tb.AddSeparator()
        # buttons
        button = wx.Button(tb, ID.SHOW_SOURCES, "Sources")
        button.Bind(wx.EVT_BUTTON, self.OnShowSources)
        tb.AddControl(button)
        SharedToolsMenu.AddToolbarButtons(self, tb)
        # tail
        tb.AddStretchableSpace()
        self.InitToolbarTail(tb)
        tb.Realize()

        self.CreateStatusBar()

        # Bind Events ---
        self.doc.callbacks.subscribe('case_change', self.CaseChanged)
        self.panel.Bind(wx.EVT_SIZE, self.OnPanelResize)
        self.canvas.mpl_connect('axes_enter_event', self.OnPointerEntersAxes)
        self.canvas.mpl_connect('axes_leave_event', self.OnPointerEntersAxes)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        # re-Bind right click
        self.canvas.Unbind(wx.EVT_RIGHT_DOWN)
        self.canvas.Unbind(wx.EVT_RIGHT_UP)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)

        # Finalize
        self.plot()
        self.UpdateTitle()
        self.canvas.SetFocus()

    def plot(self):
        n = self.doc.ica.n_components_
        fig = self.canvas.figure
        fig.clf()

        panel_w = self.panel.GetSize()[0]
        n_h = max(2, panel_w // self.ax_size)
        n_v = int(ceil(n / n_h))

        # adjust canvas size
        size = (self.ax_size * n_h, self.ax_size * n_v)
        self.canvas_sizer.SetItemMinSize(self.canvas, size)

        # plot
        axes = tuple(fig.add_subplot(n_v, n_h, i) for i in range(1, n + 1))
        # bgs = tuple(ax.patch)
        for i, ax, c, accept in zip(range(n), axes, self.doc.components, self.doc.accept):
            layers = AxisData([DataLayer(c, PlotType.IMAGE)])
            AxTopomap(ax, layers, **TOPO_ARGS)
            ax.text(0.5, 1, "# %i" % i, ha='center', va='top')
            p = Rectangle((0, 0), 1, 1, color=COLOR[accept], zorder=-1)
            ax.add_patch(p)
            ax.i = i
            ax.background = p

        self.axes = axes
        self.n_h = n_h
        self.canvas.store_canvas()
        self.Layout()

    def CaseChanged(self, index):
        "Update the state of the segments on the current page"
        if isinstance(index, INT_TYPES):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self.doc.components)
            index = range(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # update epoch plots
        axes = []
        for idx in index:
            ax = self.axes[idx]
            ax.background.set_color(COLOR[self.doc.accept[idx]])
            axes.append(ax)

        if IS_OSX:
            try:
                self.canvas.redraw(axes)
            except AttributeError:
                self.canvas.draw()
        else:
            self.canvas.draw()  # FIXME: optimize on non-macOS systems

    def FindTopComponent(self, i_epoch: int, only_accepted: bool = False):
        components = self.doc.components
        sources = self.doc.sources.sub(case=i_epoch)
        if only_accepted:
            components = components.sub(component=self.doc.accept)
            sources = sources.sub(component=self.doc.accept)
        comp_power = (components ** 2).sum('sensor')
        source_power = (sources ** 2).sum('time')
        epoch_comp_power = comp_power * source_power
        top_component = epoch_comp_power.argmax()
        self.GoToComponentEpoch(component=top_component)

    def FindTopEpoch(self, i_comp: int):
        source = self.doc.sources.sub(component=i_comp)
        y = source - source.mean()
        y **= 2
        ss = y.sum('time')  # ndvar has epoch as index
        self.GoToComponentEpoch(i_comp, ss.argmax())

    def GoToComponentEpoch(self, component: int = None, epoch: int = None):
        if not self.source_frame:
            self.ShowSources(0)
        self.source_frame.GoToComponentEpoch(component, epoch)

    def MakeToolsMenu(self, menu):
        app = wx.GetApp()
        # show sources
        item = menu.Append(wx.ID_ANY, "Source Viewer", "Open a source time course viewer window")
        app.Bind(wx.EVT_MENU, self.OnShowSources, item)
        # shared menu
        menu.AppendSeparator()
        SharedToolsMenu.MakeToolsMenu(self, menu)

    def OnCanvasClick(self, event):
        "Called by mouse clicks"
        if event.button == 1:
            if event.inaxes:
                self.model.toggle(event.inaxes.i)

    def OnCanvasKey(self, event):
        if not event.inaxes:
            return
        if event.key == 't':
            self.PlotCompTopomap(event.inaxes.i)
        elif event.key == 'a':
            self.PlotCompSourceArray(event.inaxes.i)
        elif event.key == 's':
            self.ShowSources(event.inaxes.i)
        elif event.key == 'f':
            self.PlotCompFFT(event.inaxes.i)
        elif event.key == 'b':
            self.PlotEpochButterfly()
        elif event.key == 'B':
            self.PlotConditionAverages(self)

    def OnFindTopAcceptedComponent(self, event):
        self.FindTopComponent(event.EventObject.i_epoch, only_accepted=True)

    def OnFindTopComponent(self, event):
        self.FindTopComponent(event.EventObject.i_epoch)

    def OnFindTopEpoch(self, event):
        self.FindTopEpoch(event.EventObject.i_comp)

    def OnPanelResize(self, event):
        w, h = event.GetSize()
        n_h = w // self.ax_size
        if n_h >= 2 and n_h != self.n_h:
            self.plot()

    def OnPlotCompFFT(self, event):
        self.PlotCompFFT(event.EventObject.i_comp)

    def OnPlotCompSourceArray(self, event):
        self.PlotCompSourceArray(event.EventObject.i_comp)

    def OnPlotCompTopomap(self, event):
        self.PlotCompTopomap(event.EventObject.i_comp)

    def OnPlotEpoch(self, event):
        self.PlotEpochButterfly(event.EventObject.i_epoch)

    def OnPointerEntersAxes(self, event):
        try:
            sb = self.GetStatusBar()
        except RuntimeError:
            return  # can be called after the window closes (Windows)
        if event.inaxes:
            sb.SetStatusText(f"#{event.inaxes.i} of {len(self.doc.components)} ICA Components")
        else:
            sb.SetStatusText(f"{len(self.doc.components)} ICA Components")

    def OnRankEpochs(self, event):
        i_comp = event.EventObject.i_comp
        source = self.doc.sources.sub(component=i_comp)
        y = source - source.mean()
        y /= y.std()
        y **= 2
        ss = y.sum('time').x  # ndvar has epoch as index

        # sort
        sort = np.argsort(ss)[::-1]

        # doc
        lst = fmtxt.List(f"Epochs SS loading in descending order for component {i_comp}")
        for i in sort:
            link = fmtxt.Link(self.doc.epoch_labels[i], f'component:{i_comp} epoch:{i}')
            lst.add_item(link + f': {ss[i]:.1f}')
        doc = fmtxt.Section(f"#{i_comp} Ranked Epochs", lst)
        InfoFrame(self, f"Component {i_comp} Epoch SS", doc, 200)

    def _context_menu(self, i_comp: int = None, i_epoch: int = None):
        menu = ContextMenu(i_comp, i_epoch)
        if i_comp is not None:
            item = menu.Append(wx.ID_ANY, "Top Epoch")
            self.Bind(wx.EVT_MENU, self.OnFindTopEpoch, item)
            item = menu.Append(wx.ID_ANY, "Rank Epochs")
            self.Bind(wx.EVT_MENU, self.OnRankEpochs, item)
            item = menu.Append(wx.ID_ANY, "Plot Topomap")
            self.Bind(wx.EVT_MENU, self.OnPlotCompTopomap, item)
            item = menu.Append(wx.ID_ANY, "Plot Source Array")
            self.Bind(wx.EVT_MENU, self.OnPlotCompSourceArray, item)
            item = menu.Append(wx.ID_ANY, "Plot Source FFT")
            self.Bind(wx.EVT_MENU, self.OnPlotCompFFT, item)
        if i_comp is not None and i_epoch is not None:
            menu.AppendSeparator()
        if i_epoch is not None:
            item = menu.Append(wx.ID_ANY, "Top Component")
            self.Bind(wx.EVT_MENU, self.OnFindTopComponent, item)
            item = menu.Append(wx.ID_ANY, "Top Accepted Component")
            self.Bind(wx.EVT_MENU, self.OnFindTopAcceptedComponent, item)
            item = menu.Append(wx.ID_ANY, "Plot Epoch")
            self.Bind(wx.EVT_MENU, self.OnPlotEpoch, item)
        return menu

    def OnRightDown(self, event):
        mpl_event = self.canvas._to_matplotlib_event(event)
        if not mpl_event.inaxes:
            return
        menu = self._context_menu(mpl_event.inaxes.i)
        pos = self.panel.CalcScrolledPosition(event.Position)
        self.PopupMenu(menu, pos)
        menu.Destroy()

    def OnShowSources(self, event):
        self.ShowSources(0)

    def OnUpdateUIOpen(self, event):
        event.Enable(False)

    def PlotCompFFT(self, i_comp):
        plot.UTSStat(self.doc.sources.sub(component=i_comp).fft(), error=np.std, w=8, title=f"# {i_comp} Spectrum (±1 STD)", legend=False)

    def PlotCompSourceArray(self, i_comp):
        x = self.doc.sources.sub(component=i_comp)
        dim = Categorial('epoch', self.doc.epoch_labels)
        x = NDVar(x.x, (dim,) + x.dims[1:], x.info, x.name)
        plot.Array(x, w=10, h=10,
                   title='# %i' % i_comp, axtitle=False, interpolation='none')

    def PlotCompTopomap(self, i_comp):
        plot.Topomap(self.doc.components[i_comp], sensorlabels='name', axw=9, title=f'# {i_comp}')

    def ShowSources(self, i_first):
        if self.source_frame:
            self.source_frame.Raise()
        else:
            self.source_frame = SourceFrame(self, i_first)


class SourceFrame(SharedToolsMenu, FileFrameChild):
    """Component source time course display for selecting ICA components.

    * Click on components topographies to select/deselect them.
    * Right-click for a context-menu.

    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    arrows      scroll through components/epochs
    alt+arrows  scroll to beginning/end
    t           topomap plot of the component under the pointer
    a           array-plot of the source time course of the component under
                the pointer
    f           plot the frequency spectrum for the component under the
                pointer
    b           butterfly plot of the original and cleaned data (of the
                epoch under the pointer, or of the grand average if the
                pointer is over other elements)
    B           Butterfly plot of condition averages
    =========== ============================================================
    """
    _doc_name = 'component selection'
    _title = 'ICA Source Time Course'
    _wildcard = "ICA fiff file (*-ica.fif)|*.fif"

    def __init__(
            self,
            parent: Frame,
            i_first: int,
    ):
        FileFrameChild.__init__(self, parent, None, None, parent.model)
        SharedToolsMenu.__init__(self)

        # prepare canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.subplots_adjust(0, 0, 1, 1, 0, 0)
        self.figure.set_facecolor('white')

        # attributes
        self.parent = parent
        self.model = parent.model
        self.doc = parent.model.doc
        self.n_comp_actual = self.n_comp = self.config.ReadInt('layout_n_comp', 10)
        self.n_comp_in_ica = len(self.doc.components)
        self.i_first = i_first
        self.n_epochs = self.config.ReadInt('layout_n_epochs', 20)
        self.i_first_epoch = 0
        self.pad_time = 0  # need to pad x-axis when showing fewer epochs than fit on axis)
        self.n_epochs_in_data = len(self.doc.sources)
        self.y_scale = self.config.ReadFloat('y_scale', 10)  # scale factor for y axis
        self._marked_component_i = None
        self._marked_component_h = None
        self._marked_epoch_i = None
        self._marked_epoch_h = None
        self.show_range = True  # show axis with pre/post ICA data range

        # Toolbar
        tb = self.InitToolbar(can_open=False)
        tb.AddSeparator()
        self.up_button = tb.AddTool(wx.ID_UP, "Up", Icon("tango/actions/go-up"))
        self.down_button = tb.AddTool(wx.ID_DOWN, "Down", Icon("tango/actions/go-down"))
        self.back_button = tb.AddTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.next_button = tb.AddTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        tb.AddSeparator()
        SharedToolsMenu.AddToolbarButtons(self, tb)
        tb.AddStretchableSpace()
        self.InitToolbarTail(tb)
        tb.Realize()

        # event bindings
        self.doc.callbacks.subscribe('case_change', self.CaseChanged)
        self.Bind(wx.EVT_TOOL, self.OnUp, id=wx.ID_UP)
        self.Bind(wx.EVT_TOOL, self.OnDown, id=wx.ID_DOWN)
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        # re-Bind mouse click
        self.canvas.Unbind(wx.EVT_LEFT_DOWN)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.canvas.Unbind(wx.EVT_RIGHT_DOWN)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)

        self._plot()
        self.UpdateTitle()
        self.Show()

    def _get_source_data(self):
        "Return ``(source_data, epoch-labels)`` tuple for current page"
        n_comp = self.n_comp
        n_comp_actual = self.n_comp_actual
        epoch_index = slice(self.i_first_epoch, self.i_first_epoch + self.n_epochs)
        data = self.doc.sources.sub(case=epoch_index, component=slice(self.i_first, self.i_first + n_comp))
        y = data.get_data(('component', 'case', 'time')).reshape((n_comp_actual, -1))
        if y.base is not None and data.x.base is not None:
            y = y.copy()
        start = n_comp - 1 + self.show_range
        stop = -1 + (n_comp - n_comp_actual) + self.show_range
        y += np.arange(start * self.y_scale, stop * self.y_scale, -self.y_scale)[:, None]
        # pad epoch labels for x-axis
        epoch_labels = self.doc.epoch_labels[epoch_index]
        if len(epoch_labels) < self.n_epochs:
            epoch_labels += ('',) * (self.n_epochs - len(epoch_labels))
        return y, epoch_labels

    def _pad(self, y):
        "Pad time-axis when data contains fewer epochs than the x-axis"
        if self.pad_time:
            return np.pad(y, (0, self.pad_time), 'constant')
        else:
            return y

    def _get_raw_range(self):
        epoch_index = slice(self.i_first_epoch, self.i_first_epoch + self.n_epochs)
        y_min = self._pad(self.doc.pre_ica_min[epoch_index].x.ravel())
        y_max = self._pad(self.doc.pre_ica_max[epoch_index].x.ravel())
        return y_min, y_max

    def _get_clean_range(self):
        epoch_index = slice(self.i_first_epoch, self.i_first_epoch + self.n_epochs)
        epochs = self.doc.epochs[epoch_index]
        y_clean = self.doc.as_ndvar(self.doc.apply(epochs))
        y_min = y_clean.min('sensor').x.ravel()
        y_max = y_clean.max('sensor').x.ravel()
        y_min /= self.doc.pre_ica_range_scale
        y_max /= self.doc.pre_ica_range_scale
        return self._pad(y_min), self._pad(y_max)

    def _plot(self):
        # partition figure
        self.figure.clf()
        figheight = self.figure.get_figheight()
        n_comp = self.n_comp
        # make sure there are no empty lines
        if self.i_first and self.n_comp_in_ica - self.i_first < n_comp:
            self.i_first = max(0, self.n_comp_in_ica - n_comp)
        # further layout-relevant properties
        n_comp_actual = min(self.n_comp_in_ica - self.i_first, n_comp)
        self.n_comp_actual = n_comp_actual
        elen = len(self.doc.sources.time)

        # layout
        n_rows = n_comp + self.show_range
        axheight = 1 / (n_rows + 0.5)  # 0.5 = bottom space for epoch labels

        # topomaps
        ax_size_in = axheight * figheight
        axwidth = ax_size_in / self.figure.get_figwidth()
        left = axwidth / 2
        self.topo_plots = []
        self.topo_labels = []
        for i in range(n_comp_actual):
            i_comp = self.i_first + i
            ax = self.figure.add_axes((left, 1 - (i + 1) * axheight, axwidth, axheight))
            layers = AxisData([DataLayer(self.doc.components[i_comp], PlotType.IMAGE)])
            p = AxTopomap(ax, layers, **TOPO_ARGS)
            text = ax.text(0, 0.5, "# %i" % i_comp, va='center', ha='right', color='k')
            ax.i = i
            ax.i_comp = i_comp
            self.topo_plots.append(p)
            self.topo_labels.append(text)

        # source time course data
        y, xtick_labels = self._get_source_data()

        # axes
        left = 1.5 * axwidth
        bottom = 1 - n_rows * axheight
        xticks = np.arange(elen / 2, elen * self.n_epochs, elen)
        ax = self.figure.add_axes((left, bottom, 1 - left, 1 - bottom), frameon=False, yticks=(), xticks=xticks, xticklabels=xtick_labels)
        ax.tick_params(bottom=False)
        ax.i = -1
        ax.i_comp = None

        # store canvas before plotting lines
        self.canvas.draw()

        # plot epochs
        self.lines = ax.plot(y.T, color=LINE_COLOR[True], clip_on=False)
        # line color
        reject_color = LINE_COLOR[False]
        for i in range(n_comp_actual):
            if not self.doc.accept[i + self.i_first]:
                self.lines[i].set_color(reject_color)
        # data pre/post range
        if self.show_range:
            pre_color = UNAMBIGUOUS_COLORS['orange']
            post_color = UNAMBIGUOUS_COLORS['bluish green']
            ax.text(-10, 0.1, 'Range: Raw', va='bottom', ha='right', color=pre_color)
            ax.text(-10, -0.1, 'Cleaned', va='top', ha='right', color=post_color)
            # raw
            ys_raw = self._get_raw_range()
            self.y_range_pre_lines = [ax.plot(yi, color=pre_color, clip_on=False)[0] for yi in ys_raw]
            # cleaned
            ys_clean = self._get_clean_range()
            self.y_range_post_lines = [ax.plot(yi, color=post_color, clip_on=False)[0] for yi in ys_clean]
        # axes limits
        self.ax_tc_ylim = (-0.5 * self.y_scale, (n_rows - 0.5) * self.y_scale)
        ax.set_ylim(self.ax_tc_ylim)
        ax.set_xlim((0, y.shape[1]))
        # epoch demarcation
        for x in range(elen, elen * self.n_epochs, elen):
            ax.axvline(x, ls='--', c='k')

        self.ax_tc = ax
        self.canvas.draw()

    def _plot_update_raw_range(self):
        y_min, y_max = self._get_raw_range()
        for line, data in zip(self.y_range_pre_lines, (y_min, y_max)):
            line.set_ydata(data)

    def _plot_update_clean_range(self):
        y_min, y_max = self._get_clean_range()
        for line, data in zip(self.y_range_post_lines, (y_min, y_max)):
            line.set_ydata(data)

    def _event_i_comp(self, event):
        if event.inaxes:
            if event.inaxes.i_comp is None:
                i_in_axes = ceil(event.ydata / self.y_scale + 0.5)
                if i_in_axes == 1 and self.show_range:
                    return
                i_comp = int(self.i_first + self.n_comp + self.show_range - i_in_axes)
                if i_comp < self.n_comp_in_ica:
                    return i_comp
            else:
                return event.inaxes.i_comp

    def _event_i_epoch(self, event):
        if event.inaxes is not None and event.inaxes.i_comp is None:
            i_epoch = self.i_first_epoch + int(event.xdata // len(self.doc.sources.time))
            if 0 <= i_epoch < len(self.doc.epochs):
                return i_epoch

    def CanBackward(self):
        return self.i_first_epoch > 0

    def CanDown(self):
        return self.i_first + self.n_comp < self.n_comp_in_ica

    def CanForward(self):
        return self.i_first_epoch + self.n_epochs < self.n_epochs_in_data

    def CanUp(self):
        return self.i_first > 0

    def CaseChanged(self, index):
        "Update the states of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.doc.n_epochs
            index = range(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # filter to visible epochs
        i_last = self.i_first + self.n_comp_actual
        index = [i_comp for i_comp in index if self.i_first <= i_comp <= i_last]
        # update epoch plots
        if index:
            for i_comp in index:
                self.lines[i_comp - self.i_first].set_color(LINE_COLOR[self.doc.accept[i_comp]])
            self._plot_update_clean_range()
            self.canvas.draw()

    def GoToComponentEpoch(self, component: int = None, epoch: int = None):
        if component is not None:
            self._marked_component_i = component
            self.SetFirstComponent(component // self.n_comp * self.n_comp)
        if epoch is not None:
            self._marked_epoch_i = epoch
            self.SetFirstEpoch(epoch // self.n_epochs * self.n_epochs)
        self.Raise()

    def OnBackward(self, event):
        "Turn the page backward"
        self.SetFirstEpoch(self.i_first_epoch - self.n_epochs)

    def OnCanvasKey(self, event):
        if event.key is None:
            return
        elif event.key == 'alt+down':
            self.SetFirstComponent(self.n_comp_in_ica - self.n_comp)
        elif event.key == 'down':
            if self.CanDown():
                self.OnDown(None)
        elif event.key == 'alt+up':
            self.SetFirstComponent(0)
        elif event.key == 'up':
            if self.CanUp():
                self.OnUp(None)
        elif event.key == 'alt+right':
            self.SetFirstEpoch(((self.n_epochs_in_data - 1) // self.n_epochs) * self.n_epochs)
        elif event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
        elif event.key == 'alt+left':
            self.SetFirstEpoch(0)
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
        elif event.key == 'B':
            self.parent.PlotConditionAverages(self)
        elif event.key == 'b':
            self.PlotEpochButterfly(self._event_i_epoch(event))
        elif not event.inaxes:
            return
        # component-specific plots
        i_comp = self._event_i_comp(event)
        if i_comp is None:  # source time course axes
            return
        elif event.key in 'tT':
            self.parent.PlotCompTopomap(i_comp)
        elif event.key == 'a':
            self.parent.PlotCompSourceArray(i_comp)
        elif event.key == 'f':
            self.parent.PlotCompFFT(i_comp)

    def OnClose(self, event):
        if super(SourceFrame, self).OnClose(event):
            self.doc.callbacks.remove('case_change', self.CaseChanged)
            self.config.WriteInt('layout_n_comp', self.n_comp)
            self.config.WriteInt('layout_n_epochs', self.n_epochs)
            self.config.WriteFloat('y_scale', self.y_scale)
            self.config.Flush()

    def OnDown(self, event):
        "Turn the page backward"
        self.SetFirstComponent(self.i_first + self.n_comp)

    def OnForward(self, event):
        "Turn the page forward"
        self.SetFirstEpoch(self.i_first_epoch + self.n_epochs)

    def OnLeftDown(self, event):
        "Called by mouse clicks"
        mpl_event = self.canvas._to_matplotlib_event(event)
        i_comp = self._event_i_comp(mpl_event)
        if i_comp is None:
            return
        self.model.toggle(i_comp)

    def OnRightDown(self, event):
        mpl_event = self.canvas._to_matplotlib_event(event)
        i_comp = self._event_i_comp(mpl_event)
        i_epoch = self._event_i_epoch(mpl_event)
        if i_comp is None and i_epoch is None:
            return
        menu = self.parent._context_menu(i_comp, i_epoch)
        self.PopupMenu(menu, event.Position)
        menu.Destroy()

    def OnSetLayout(self, event):
        caption = "Set ICA Source Layout"
        msg = "Number of components and epochs (e.g., '10 20')"
        default = '%i %i' % (self.n_comp, self.n_epochs)
        dlg = wx.TextEntryDialog(self, msg, caption, default)
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                try:
                    n_comp, n_epochs = map(int, value.split())
                except Exception:
                    wx.MessageBox("Invalid entry: %r. Need two integers \n"
                                  "(e.g., '10 20').", "Invalid Entry",
                                  wx.OK | wx.ICON_ERROR)
                else:
                    dlg.Destroy()
                    break
            else:
                dlg.Destroy()
                return

        self.n_comp = n_comp
        self.n_epochs = n_epochs
        self._plot()

    def OnSetVLim(self, event):
        dlg = wx.TextEntryDialog(self, "Y-axis scale:", "Y-Axis Scale",
                                 "%g" % (10. / self.y_scale,))
        value = None
        while True:
            if dlg.ShowModal() != wx.ID_OK:
                break
            error = None
            try:
                value = float(dlg.GetValue())
                if value <= 0:
                    error = f"{value}: must be > 0"
            except Exception as exception:
                error = str(exception)

            if error:
                msg = wx.MessageDialog(self, error, "Invalid Entry", wx.OK | wx.ICON_ERROR)
                msg.ShowModal()
                msg.Destroy()
            else:
                break
        dlg.Destroy()
        if value is not None:
            self.y_scale = 10. / value
            # redraw
            self.SetFirstEpoch(self.i_first_epoch)

    def OnUp(self, event):
        "Turn the page backward"
        self.SetFirstComponent(self.i_first - self.n_comp)

    def OnUpdateUIBackward(self, event):
        event.Enable(self.CanBackward())

    def OnUpdateUIDown(self, event):
        event.Enable(self.CanDown())

    def OnUpdateUIForward(self, event):
        event.Enable(self.CanForward())

    def OnUpdateUISetLayout(self, event):
        event.Enable(True)

    def OnUpdateUISetVLim(self, event):
        event.Enable(True)

    def OnUpdateUIUp(self, event):
        event.Enable(self.CanUp())

    def SetFirstComponent(self, i_first):
        if i_first < 0:
            i_first = 0
        elif i_first >= self.n_comp_in_ica:
            i_first = self.n_comp_in_ica - 1
        n_rows = self.n_comp + self.show_range

        # marked component
        if self._marked_component_h is not None:
            self._marked_component_h.remove()
            self._marked_component_h = None
        if self._marked_component_i is not None:
            i_from_top = self._marked_component_i - i_first
            i_from_bottom = n_rows - 1 - i_from_top
            if 0 <= i_from_bottom < n_rows:
                bottom = (i_from_bottom - 0.5) * self.y_scale
                self._marked_component_h = self.ax_tc.axhspan(bottom, bottom + self.y_scale, edgecolor='yellow', facecolor='yellow')

        n_comp_actual = min(self.n_comp_in_ica - i_first, self.n_comp)
        for i in range(n_comp_actual):
            p = self.topo_plots[i]
            i_comp = i_first + i
            p.set_data([self.doc.components[i_comp]], True)
            p.ax.i_comp = i_comp
            self.topo_labels[i].set_text("# %i" % i_comp)
            self.lines[i].set_color(LINE_COLOR[self.doc.accept[i_comp]])

        if n_comp_actual < self.n_comp:
            empty_data = self.doc.components[0].copy()
            empty_data.x.fill(0)
            for i in range(n_comp_actual, self.n_comp):
                p = self.topo_plots[i]
                p.set_data([empty_data])
                p.ax.i_comp = -1
                self.topo_labels[i].set_text("")
                self.lines[i].set_color('white')

        self.i_first = i_first
        self.n_comp_actual = n_comp_actual
        self.SetFirstEpoch(self.i_first_epoch)

    def SetFirstEpoch(self, i_first_epoch):
        self.i_first_epoch = i_first_epoch

        # marked epoch
        if self._marked_epoch_h is not None:
            self._marked_epoch_h.remove()
            self._marked_epoch_h = None
        if self._marked_epoch_i is not None:
            i = self._marked_epoch_i - i_first_epoch
            if 0 <= i < self.n_epochs:
                elen = len(self.doc.sources.time)
                bottom = -0.5 * self.y_scale
                height = (self.n_comp + self.show_range) * self.y_scale
                self._marked_epoch_h = Rectangle((i * elen, bottom), elen, height, edgecolor='yellow', facecolor='yellow')
                self.ax_tc.add_patch(self._marked_epoch_h)

        # update data
        y, tick_labels = self._get_source_data()
        if i_first_epoch + self.n_epochs > self.n_epochs_in_data:
            elen = len(self.doc.sources.time)
            n_missing = self.i_first_epoch + self.n_epochs - self.n_epochs_in_data
            pad_time = elen * n_missing
        else:
            pad_time = 0
        self.pad_time = pad_time

        if self.n_comp_actual < self.n_comp:
            pad_comp = self.n_comp - self.n_comp_actual
        else:
            pad_comp = 0

        if pad_time or pad_comp:
            y = np.pad(y, ((0, pad_comp), (0, pad_time)), 'constant')

        for line, data in zip(self.lines, y):
            line.set_ydata(data)

        if self.show_range:
            self._plot_update_raw_range()
            self._plot_update_clean_range()

        self.ax_tc.set_xticklabels(tick_labels)
        self.ax_tc.set_ylim(self.ax_tc_ylim)
        self.canvas.draw()


class FindNoisyEpochsDialog(EelbrainDialog):
    _default_thresholds = {'µV': 100, 'fT': 1000}

    def __init__(self, parent, unit, **kwargs):
        self.unit = unit
        super(FindNoisyEpochsDialog, self).__init__(parent, wx.ID_ANY, "Find Bad Epochs", **kwargs)
        config = parent.config
        threshold = config.ReadFloat(f"FindNoisyEpochsDialog/threshold_{unit}", self._default_threshold())
        apply_rejection = config.ReadBool("FindNoisyEpochsDialog/apply_rejection", True)
        sort_by_component = config.ReadBool("FindNoisyEpochsDialog/sort_by_component", True)
        max_ch_ratio = config.Read("FindNoisyEpochsDialog/max_ch_ratio", '')

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Threshold
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer.Add(wx.StaticText(self, label="Threshold for bad epochs: "))
        validator = REValidator(POS_FLOAT_PATTERN, "Invalid entry: {value}. Please specify a number > 0.", False)
        self.threshold = ctrl = wx.TextCtrl(self, value=f'{threshold:g}', validator=validator, style=wx.TE_RIGHT)
        ctrl.SetHelpText("Find epochs in which the signal exceeds this value at any sensor")
        ctrl.SelectAll()
        h_sizer.Add(ctrl)
        h_sizer.Add(wx.StaticText(self, label=unit))
        sizer.Add(h_sizer)

        # Apply rejection before finding noisy epochs
        self.apply_rejection = ctrl = wx.CheckBox(self, label="Apply ICA rejection")
        ctrl.SetValue(apply_rejection)
        sizer.Add(ctrl)

        # Sort noisy epochs by component
        self.sort_by_component = ctrl = wx.CheckBox(self, label="Sort by ICA component")
        ctrl.SetValue(sort_by_component)
        sizer.Add(ctrl)

        # Filter by ch 1/2 ratio
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer.Add(wx.StaticText(self, label="Max channel ratio filter: "))
        validator = REValidator(FLOAT_PATTERN, "Invalid entry: {value}. Please specify a number ≥ 0.", True)
        self.max_ch_ratio = ctrl = wx.TextCtrl(self, value=max_ch_ratio, validator=validator, style=wx.TE_RIGHT)
        ctrl.SetHelpText("Filter components that are due to bad channels through the first / second channel ratio")
        ctrl.SelectAll()
        h_sizer.Add(ctrl)
        sizer.Add(h_sizer)

        # default button
        btn = wx.Button(self, wx.ID_DEFAULT, "Default Settings")
        sizer.Add(btn, border=2)
        btn.Bind(wx.EVT_BUTTON, self.OnSetDefault)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def _default_threshold(self):
        return self._default_thresholds.get(self.unit, 1)

    def OnSetDefault(self, event):
        self.threshold.SetValue(f'{self._default_threshold()}')

    def StoreConfig(self):
        config = self.Parent.config
        config.WriteFloat(f"FindNoisyEpochsDialog/threshold_{self.unit}", float(self.threshold.GetValue()))
        config.WriteBool("FindNoisyEpochsDialog/apply_rejection", self.apply_rejection.GetValue())
        config.WriteBool("FindNoisyEpochsDialog/sort_by_component", self.sort_by_component.GetValue())
        config.Write("FindNoisyEpochsDialog/max_ch_ratio", self.max_ch_ratio.GetValue())
        config.Flush()


class FindRareEventsDialog(EelbrainDialog):
    def __init__(self, parent, *args, **kwargs):
        super(FindRareEventsDialog, self).__init__(parent, wx.ID_ANY, "Find Rare Events", *args, **kwargs)
        config = parent.config
        threshold = config.ReadFloat("FindRareEvents/threshold", 2.)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Threshold
        sizer.Add(wx.StaticText(self, label="Threshold for rare epochs\n(z-scored peak-to-peak value):"))
        validator = REValidator(POS_FLOAT_PATTERN, "Invalid entry: {value}. Please specify a number > 0.", False)
        ctrl = wx.TextCtrl(self, value=str(threshold), validator=validator)
        ctrl.SetHelpText("Epochs whose z-scored peak-to-peak value exceeds  this value are considered rare")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.threshold = ctrl

        # default button
        btn = wx.Button(self, wx.ID_DEFAULT, "Default Settings")
        sizer.Add(btn, border=2)
        btn.Bind(wx.EVT_BUTTON, self.OnSetDefault)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def OnSetDefault(self, event):
        self.threshold.SetValue('2')

    def StoreConfig(self):
        config = self.Parent.config
        config.WriteFloat("FindRareEvents/threshold", float(self.threshold.GetValue()))
        config.Flush()


class InfoFrame(HTMLFrame):

    def __init__(
            self,
            parent: Union[wx.Window, SharedToolsMenu],
            title: str,
            doc: fmtxt.FMTextElement,
            w: int,
            h: int = -1,
    ):
        pos, size = self.find_pos(w, h)
        style = wx.MINIMIZE_BOX | wx.MAXIMIZE_BOX | wx.RESIZE_BORDER | wx.CAPTION | wx.CLOSE_BOX | wx.FRAME_FLOAT_ON_PARENT | wx.FRAME_TOOL_WINDOW
        html_doc = fmtxt.make_html_doc(doc)
        HTMLFrame.__init__(self, parent, title, html_doc, pos=pos, size=size, style=style)

    @staticmethod
    def find_pos(w: int, h: int):
        display_w, display_h = wx.DisplaySize()
        h_max = display_h - 44
        h = h_max if h <= 0 else min(h, h_max)
        pos = (display_w - w, int(round((display_h - h) / 2)))
        return pos, (w, h)

    def OpenURL(self, url):
        component = epoch = None
        for part in url.split():
            m = re.match(r'^epoch:(\d+)$', part)
            if m:
                epoch = int(m.group(1))
                continue
            m = re.match(r'^component:(\d+)$', part)
            if m:
                component = int(m.group(1))
                continue
            raise ValueError(f"url={url!r}")
        self.Parent.GoToComponentEpoch(component, epoch)
