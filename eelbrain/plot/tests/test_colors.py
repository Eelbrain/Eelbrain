# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain import datasets, plot
from eelbrain.testing import hide_plots


def test_generate_colors():
    "Test functions for assigning colors to cells"
    ds = datasets.get_uv()
    cells_1 = ('A', 'B')
    cells_2 = ('a', 'b', 'c')

    colors = plot.colors_for_oneway(cells_1)
    assert len(colors) == len(cells_1)

    colors = plot.colors_for_twoway(cells_1, cells_2)
    assert len(colors) == len(cells_1) * len(cells_2)

    colors = plot.colors_for_twoway(cells_2, cells_1)
    assert len(colors) == len(cells_1) * len(cells_2)

    # colors_for_categorial()
    f = ds['A']
    colors = plot.colors_for_categorial(f)
    for cell in f.cells:
        assert cell in colors

    i = ds.eval("A%B")
    colors = plot.colors_for_categorial(i)
    for cell in i.cells:
        assert cell in colors

    i = ds.eval("A%B%rm")
    colors = plot.colors_for_categorial(i)
    for cell in i.cells:
        assert cell in colors

    with pytest.raises(TypeError):
        plot.colors_for_categorial("A%B")


def test_generate_colormaps():
    cmap = plot.two_step_colormap('black', 'red', name='oneside')
    assert not cmap.symmetric
    cmap = plot.two_step_colormap('black', 'red', 'transparent', 'blue', 'black', name='red-blue')
    assert cmap.symmetric
    cmap = plot.two_step_colormap('black', (1, 0, 0.3), 'transparent', (0.3, 0, 1), 'black', name='red-blue-2')
    assert cmap.symmetric


def ticks(p):
    return [t.get_text() for t in p._colorbar.ax.get_xticklabels()]


@hide_plots
def test_plot_colorbar():
    "Test plot.ColorBar()"
    p = plot.ColorBar('jet', -1, 1)
    p.close()
    p = plot.ColorBar('jet', -1, 1, ticks=[-1, 0, 1])
    assert ticks(p) == ['−1', '0', '1']
    p = plot.ColorBar('jet', -1, 1, orientation='vertical')
    p.close()
    p = plot.ColorBar('jet', -1, 1, label_position='top')
    p.close()
    p = plot.ColorBar('jet', -1, 1, orientation='vertical', label_position='right')
    p.close()
    p = plot.ColorBar('xpolar-a', -3, 3, clipmin=0, unit='t', ticks=[0, 2, 3])
    assert ticks(p) == ['0', '2', '3']
    p.close()

    p = plot.ColorBar('xpolar-a', -3e-6, 3e-6, unit='µV')
    assert ticks(p) == ['-3', '-1.5', '0', '1.5', '3']
    assert p._colorbar.ax.get_xlabel() == 'µV'

    # soft-thresholded colormap
    cmap = plot.soft_threshold_colormap('xpolar-a', 0.2, 2.0)
    p = plot.ColorBar(cmap)
    p.close()
    cmap = plot.soft_threshold_colormap('xpolar-a', 1, 3)
    p = plot.ColorBar(cmap, clipmin=0, unit='t', ticks=[0, 2, 3])
    assert ticks(p) == ['0', '2', '3']
    p.close()
    cmap = plot.soft_threshold_colormap('xpolar-a', 0.5, 3)
    p = plot.ColorBar(cmap, clipmin=0, unit='t', ticks=[0, 2, 3])
    assert ticks(p) == ['0', '2', '3']
    p.close()


@hide_plots
def test_plot_colors():
    "Test plotting color schemes"
    cells_1 = ('A', 'B')
    cells_2 = ('a', 'b', 'c')
    colors = plot.colors_for_oneway(cells_1)
    colors2way = plot.colors_for_twoway(cells_1, cells_2)
    labels = {k: f'Long label {k}' for k in cells_2}

    p = plot.ColorList(colors)
    assert p._figtitle is None
    w0, h0 = p.figure.get_size_inches()
    p.close()

    p = plot.ColorList(colors, labels={'A': 'A' * 50, 'B': 'Bbb'})
    w, h = p.figure.get_size_inches()
    assert h == h0
    assert w > w0
    p.close()

    p = plot.ColorList(colors2way)
    p.close()
    p = plot.ColorGrid(cells_1, cells_2, colors2way)
    p.close()
    p = plot.ColorGrid(cells_2, cells_1, colors2way)
    p.close()
    p = plot.ColorGrid(cells_1, cells_2, colors2way, labels=labels, w=1.2, h=1.5)
    p.close()
