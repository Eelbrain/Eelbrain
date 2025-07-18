# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest
from eelbrain import Factor, datasets, plot
from eelbrain.testing import hide_plots


@hide_plots
@pytest.mark.parametrize("barplot", [plot.Barplot, plot.BarplotHorizontal])
def test_barplots(barplot):
    "Test plot.Barplot"
    ds = datasets.get_uv()

    # one category
    barplot('fltvar', data=ds, test=False)
    barplot('fltvar', 'A', data=ds, test=False)
    barplot('fltvar', 'A%B', data=ds, test=False)
    barplot('fltvar', match='rm', data=ds, test=False)
    barplot('fltvar', 'A', match='rm', data=ds, test=False)
    barplot('fltvar', 'A%B', match='rm', data=ds, test=False)

    # cells
    barplot('fltvar', 'A%B', cells=(('a2', 'b2'), ('a1', 'b1')), data=ds, test=False)
    barplot('fltvar', 'A%B', match='rm', cells=(('a2', 'b2'), ('a1', 'b1')), data=ds, test=False)


@hide_plots
def test_barplot():
    "Test plot.Barplot"
    ds = datasets.get_uv()

    # average
    plot.Barplot('fltvar', data=ds)
    plot.Barplot('fltvar', match='rm', data=ds)

    # one category
    p = plot.Barplot('fltvar + 2', 'A', data=ds)
    assert p.get_ylim()[0] == 0
    p = plot.Barplot('fltvar - 2', 'A', 'rm', data=ds)
    assert p.get_ylim()[1] == 0

    # multiple categories
    plot.Barplot('fltvar', 'A%B', match='rm', data=ds)
    plot.Barplot('fltvar', 'A%B', match='rm', data=ds, within_subject_error=False)
    # test against 0
    plot.Barplot('fltvar', 'A%B', match='rm', test=0, data=ds)
    plot.Barplot('fltvar', 'A%B', match='rm', test=0, tail=1, data=ds)

    # cells
    plot.Barplot('fltvar', 'A%B', cells=(('a2', 'b2'), ('a1', 'b1')), data=ds)
    plot.Barplot('fltvar', 'A%B', match='rm', cells=(('a2', 'b2'), ('a1', 'b1')), data=ds)

    # Fixed top
    p = plot.Barplot('fltvar', 'A%B', data=ds, top=2, test_markers=False)
    ax = p.axes[0]
    assert ax.get_ylim()[1] == 2


@hide_plots
def test_boxplot():
    "Test plot.Boxplot"
    ds = datasets.get_uv(nrm=True)
    plot.Boxplot('fltvar', 'A%B', match='rm', data=ds)
    # boxplot args
    plot.Boxplot('fltvar', 'A%B', match='rm', data=ds, showmeans=True)

    # one category
    plot.Boxplot('fltvar', data=ds, test=False)
    plot.Boxplot('fltvar', data=ds)
    plot.Boxplot('fltvar', match='rm', data=ds)
    plot.Boxplot('fltvar', 'A%B', match='rm', data=ds, label_fliers=True)

    # cells
    plot.Boxplot('fltvar', 'A%B', cells=(('a2', 'b2'), ('a1', 'b1')), data=ds)
    plot.Boxplot('fltvar', 'A%B', match='rm', cells=(('a2', 'b2'), ('a1', 'b1')), data=ds)

    # many pairwise significances
    ds['fltvar'][ds.eval("A%B==('a1','b1')")] += 1
    ds['fltvar'][ds.eval("A%B==('a2','b2')")] -= 1
    ds['C'] = Factor('qw', repeat=10, tile=4)
    plot.Boxplot('fltvar', 'A%B%C', data=ds)

    # long labels
    ds['A'].update_labels({'a1': 'a very long label', 'a2': 'another very long label'})
    p = plot.Boxplot('fltvar', 'A%B', data=ds)
    labels = p._ax.get_xticklabels()
    bbs = [l.get_window_extent() for l in labels]
    for i in range(len(bbs) - 1):
        assert bbs[i].x1 < bbs[i + 1].x0

    # nested rm
    plot.Boxplot('fltvar', 'A%B', match='nrm', data=ds)
    plot.Boxplot('fltvar', 'A%B', match='nrm', data=ds, sub="nrm != 's001'")


@hide_plots
def test_scatter():
    "Test plot.Scatter()"
    ds = datasets.get_uv()

    plot.Scatter('fltvar', 'fltvar2', data=ds)
    plot.Scatter('fltvar', 'fltvar2', 'A', data=ds)
    plot.Scatter('fltvar', 'fltvar2', 'A%B', data=ds)

    # colormap
    p = plot.Scatter('fltvar', 'fltvar2', 'intvar', data=ds)
    p.plot_colorbar()


@hide_plots
def test_histogram():
    "Test plot.Histogram"
    ds = datasets.get_uts()
    plot.Histogram('Y', 'A%B', ds=ds)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds, density=True)
    plot.Histogram('Y', 'A%B', ds=ds, test=True)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds, test=True)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds, density=True, test=True)


@hide_plots
def test_regression():
    "Test plot.Regression"
    ds = datasets.get_uv()

    plot.Regression('fltvar', 'fltvar2', data=ds)
    plot.Regression('fltvar', 'fltvar2', 'A', data=ds)
    plot.Regression('fltvar', 'fltvar2', 'A%B', data=ds)


@hide_plots
def test_timeplot():
    "Test plot.Timeplot"
    ds = datasets.get_loftus_masson_1994()
    ds['cat'] = Factor([int(s) > 5 for s in ds['subject']], labels={True: 'a', False: 'b'})

    plot.Timeplot('n_recalled', 'exposure', 'subject', data=ds)
    plot.Timeplot('n_recalled', 'exposure', 'cat', data=ds)
    plot.Timeplot('n_recalled', 'exposure', 'cat', 'subject', data=ds, x_jitter=True)
