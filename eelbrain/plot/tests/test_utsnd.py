# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np

from eelbrain import datasets, plot, testnd, cwt_morlet
from eelbrain.plot._figure import Figure
from eelbrain.plot._utsnd import AxButterflyEpoch
from eelbrain.testing import requires_mne_sample_data
from eelbrain.testing import hide_plots


@hide_plots
def test_plot_butterfly():
    "Test plot.Butterfly"
    ds = datasets.get_uts(utsnd=True)
    p = plot.Butterfly('utsnd', data=ds)
    p.close()
    p = plot.Butterfly('utsnd', 'A%B', data=ds)
    p.close()

    # other y-dim
    stc = datasets.get_mne_stc(True, subject='fsaverage')
    p = plot.Butterfly(stc)
    p.close()

    # _ax_bfly_epoch (used in GUI, not part of a figure)
    fig = Figure(1)
    ax = AxButterflyEpoch(fig.axes[0], ds[0, 'utsnd'])
    fig.show()
    ax.set_data(ds[1, 'utsnd'])
    fig.draw()

    # source space
    stc = datasets.get_mne_stc(True)
    p = plot.Butterfly(stc, '.source.hemi')
    p.close()


@hide_plots
def test_plot_array():
    "Test plot.Array"
    ds = datasets.get_uts(utsnd=True)
    p = plot.Array('utsnd', data=ds)
    # contours
    p.add_contour(1, 'k')
    p.add_contour(2, 'blue')
    p.close()
    p = plot.Array('utsnd', data=ds, contours=6)
    p.close()
    # data parameters
    p = plot.Array('utsnd', 'A%B', data=ds)
    assert p._layout.nax == 4
    assert (p._layout.rows, p._layout.columns) == (2, 2)
    p.close()
    p = plot.Array('utsnd', 'A', sub='B=="b1"', data=ds)
    assert p._layout.nax == 2
    p.close()

    # Scalar dimension
    sgram = abs(cwt_morlet(ds[0, 'uts'], np.arange(3, 7, 0.1) ** 2))
    p = plot.Array(sgram)
    labels = [l.get_text() for l in p.figure.axes[0].get_yticklabels()]
    assert labels == ['9', '12', '15', '19', '23', '27', '32', '38', '44']
    p.close()

    sgram = abs(cwt_morlet(ds[0, 'uts'], np.arange(3, 3.5, 0.01) ** 2))
    p = plot.Array(sgram)
    labels = [l.get_text() for l in p.figure.axes[0].get_yticklabels()]
    assert labels == ['9.0', '9.4', '9.7', '10.0', '10.4', '10.8', '11.1', '11.4', '11.8', '12.2']
    p.close()

    # From test results
    res1 = testnd.TTestIndependent('utsnd', 'A', 'a1', 'a0', data=ds, tstart=0, samples=10)
    res2 = testnd.TTestIndependent('utsnd', 'B', 'b1', 'b0', data=ds, tstart=0, samples=10)
    plot.TopoArray([res1, res2])

    # Categorial dimension
    ds = datasets._get_continuous()
    p = plot.Array(ds['x2'], interpolation='none')
    assert len(p.figure.axes[0].get_yticks()) == 2


@requires_mne_sample_data
@hide_plots
def test_plot_mne_evoked():
    "Test plotting evoked from the mne sample dataset"
    evoked = datasets.get_mne_evoked()
    p = plot.Array(evoked)
    p.close()


@requires_mne_sample_data
@hide_plots
def test_plot_mne_epochs():
    "Test plotting epochs from the mne sample dataset"
    epochs = datasets.get_mne_epochs()

    # grand average
    p = plot.Array(epochs)
    p.close()

    # with model
    p = plot.Array(epochs, np.arange(2).repeat(8))
    p.close()


@hide_plots
def test_plot_results():
    "Test plotting test results"
    ds = datasets.get_uts(True)

    # ANOVA
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, samples=0, pmin=0.05)
    p = plot.Array(res)
    p.close()
    res = testnd.ANOVA('utsnd', 'A*B*rm', data=ds, samples=2, pmin=0.05)
    p = plot.Array(res)
    p.close()

    # Correlation
    res = testnd.Correlation('utsnd', 'Y', 'rm', data=ds)
    p = plot.Array(res)
    p.close()
    res = testnd.Correlation('utsnd', 'Y', 'rm', data=ds, samples=10, pmin=0.05)
    p = plot.Array(res)
    p.close()
