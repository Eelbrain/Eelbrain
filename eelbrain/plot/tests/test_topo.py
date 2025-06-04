# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from matplotlib.backend_bases import KeyEvent
import numpy as np
import wx

from eelbrain import datasets, plot, testnd
from eelbrain._utils import IS_WINDOWS
from eelbrain.testing import requires_mne_sample_data, hide_plots
from eelbrain.testing.matplotlib import assert_titles_visible


@hide_plots
def test_plot_topomap():
    "Test plot.Topomap"
    ds = datasets.simulate_erp(8, 0)
    ds['long'] = ds['n_chars'] > 4
    ds['topo'] = ds['eeg'].mean(time=(0.300, 0.500))

    p = plot.Topomap('topo', data=ds)
    # contours
    p.add_contour(0, '0', 'V')
    p.add_contour(0.5e-6, '#00FF00', 'V')
    # colormap with limits
    assert p.get_vlim() == (-2e-06, 2e-06)
    cmap = plot.soft_threshold_colormap('jet', 1e-6, 4e-6, symmetric=True)
    p.set_cmap(cmap)
    assert p.get_vlim() == (-4e-06, 4e-06)
    p.close()

    p = plot.Topomap('topo', data=ds, vmax=0.5e-6, w=2)
    p.close()
    p = plot.Topomap('topo', 'predictability', data=ds, axw=2)
    assert_titles_visible(p)
    p.close()
    p = plot.Topomap('topo', 'predictability % long', data=ds, axw=2, columns=2)
    assert_titles_visible(p)
    p.close()
    p = plot.Topomap('topo', 'predictability % long', data=ds, axw=2, columns=2, title='Topomap Plot Title')
    assert_titles_visible(p)
    p.close()

    # axtitles from array
    index = np.array([1, 3, 2])
    p = plot.Topomap(ds[index, 'topo'], '.case', rows=1, axh=2, h=2.4, axtitle=index)
    p.close()


@requires_mne_sample_data
@hide_plots
def test_plot_topomap_mne():
    "Test plot.Topomap with MNE data"
    ds = datasets.get_mne_sample(sub=[0, 1], sns=True)
    p = plot.Topomap(ds['meg'].summary(time=(.1, .12)), proj='left')
    p.close()
    # grad
    ds = datasets.get_mne_sample(sub=[0], sns='planar1')
    plot.Topomap('meg.sub(time=.1)', data=ds)


@hide_plots
def test_plot_topo_butterfly():
    "Test plot.TopoButterfly"
    ds = datasets.get_uts(utsnd=True)

    # single row
    p = plot.TopoButterfly('utsnd', data=ds)
    p.set_time(0.2)
    # t keypress on topomap
    x, y = p.topo_axes[0].transAxes.transform((.5, .5))
    event = KeyEvent('test', p.canvas, 't', x, y, wx.KeyEvent())
    p._on_key_press(event)
    p.close()

    p = plot.TopoButterfly('utsnd', data=ds, vmax=2, w=6, t=0.5)
    assert p.axes[0].get_ylim() == (-2.0, 2.0)
    assert p._time_fixed
    p.close()

    # multiple rows
    p = plot.TopoButterfly('utsnd', 'A%B', data=ds, w=6)
    if not IS_WINDOWS:
        assert (*p.figure.get_size_inches(),) == (6, 12)
    # t keypress on topomaps
    for ax in p.topo_axes:
        x, y = ax.transAxes.transform((.5, .5))
        event = KeyEvent('test', p.canvas, 't', x, y, wx.KeyEvent())
        p._on_key_press(event)
    p.close()

    p = plot.TopoButterfly('utsnd', mark=[1, 2], data=ds)
    p.close()

    p = plot.TopoButterfly('utsnd', mark=['1', '2'], data=ds)
    p.set_vlim(2)
    assert p.get_vlim() == (-2.0, 2.0)
    p.set_ylim(-1, 1)
    assert p.get_ylim() == (-1.0, 1.0)
    p.close()


@hide_plots
def test_plot_array():
    "Test plot.TopoArray"
    ds = datasets.get_uts(utsnd=True)
    p = plot.TopoArray('utsnd', data=ds)
    assert repr(p) == "<TopoArray: utsnd>"
    p.set_topo_t(0, 0.2)
    p.close()
    p = plot.TopoArray('utsnd', data=ds, vmax=0.2, w=2)
    p.close()
    p = plot.TopoArray('utsnd', 'A%B', data=ds, axw=4)
    assert repr(p) == "<TopoArray: utsnd ~ A x B>"
    p.close()

    # results
    res = testnd.TTestIndependent('utsnd', 'A', data=ds, pmin=0.05, tstart=0.1, tstop=0.3, samples=2)
    p = plot.TopoArray(res)
    assert repr(p) == "<TopoArray: a0, a1, a0 - a1>"
    p.set_topo_t(0, 0.)
    p.close()
