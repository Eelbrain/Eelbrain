# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import Factor, datasets, boosting, epoch_impulse_predictor, plot
from eelbrain.testing import hide_plots


@hide_plots
def test_datasplit():
    ds = datasets.get_uts()
    ds['x'] = epoch_impulse_predictor('uts', data=ds)

    trf = boosting('uts', 'x', 0, 0.100, data=ds, partitions=4, test=0)
    p = trf.splits.plot()
    assert p.figure.axes[0].get_xlabel() == 'Time [s]'
    p.close()

    trf = boosting('uts', 'x', 0, 0.100, data=ds, partitions=4)
    p = trf.splits.plot()
    p.close()


@hide_plots
def test_preview_partitions():
    # trial durations in seconds; x-axis in seconds
    trials = [180, 240, 204, 192, 228]
    p = plot.preview_partitions(trials, partitions=5, test=0, split='time')
    ax = p.figure.axes[0]
    assert ax.get_xlabel() == 'Time [s]'
    assert ax.get_xlim() == (0, 1044)
    # min_section in seconds (default 15 s): boundaries within 2.4 s of a trial boundary snap to it
    assert_array_equal(p.splits.split_segments[:, 1] / 100, [180, 208.8, 420, 624, 816, 835.2, 1044])
    p.close()
    p = plot.preview_partitions(trials, partitions=5, test=0, split='time', min_section=2)
    assert_array_equal(p.splits.split_segments[:, 1] / 100, [180, 208.8, 417.6, 420, 624, 626.4, 816, 835.2, 1044])
    p.close()
    # trials specified by number: no time information, each case is labeled at its center
    p = plot.preview_partitions(6, partitions=3, test=0)
    ax = p.figure.axes[0]
    assert ax.get_xlabel() == 'Case'
    assert_array_equal(ax.get_xticks(), [500, 1500, 2500, 3500, 4500, 5500])
    assert [label.get_text() for label in ax.get_xticklabels()] == ['0', '1', '2', '3', '4', '5']
    p.close()
    p = plot.preview_partitions(20, partitions=5, test=0)  # not every case is labeled
    ax = p.figure.axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == [str(i) for i in range(0, 20, 2)]
    p.close()
    p = plot.preview_partitions(np.int64(20), partitions=5, test=0)  # numpy integer
    assert p.figure.axes[0].get_xlabel() == 'Case'
    p.close()
    # default partitions: one per trial
    p = plot.preview_partitions(6, test=0)
    assert p.splits.n_partitions == 6
    assert p.splits.partitions_arg is None
    p.close()
    # NDVar data
    ds = datasets.get_uts()
    p = plot.preview_partitions('uts', data=ds, partitions=4)
    assert p.figure.axes[0].get_xlabel() == 'Time [s]'
    p.close()
    # continuous data without time information: a nominal number of samples
    p = plot.preview_partitions(partitions=4, test=0)
    assert p.figure.axes[0].get_xlabel() == 'Sample'
    assert_array_equal(p.splits.split_segments, [[0, 250], [250, 500], [500, 750], [750, 1000]])
    p.close()
    # a single trial is continuous data
    p = plot.preview_partitions(1, partitions=4, test=0)
    assert p.figure.axes[0].get_xlabel() == 'Sample'
    assert_array_equal(p.splits.split_segments, [[0, 250], [250, 500], [500, 750], [750, 1000]])
    p.close()
    p = plot.preview_partitions(3, test=0, split='within-cases')
    assert p.figure.axes[0].get_xlabel() == 'Case'
    assert p.splits.partitions_arg is None
    assert len(p.splits.split_segments) == 15
    p.close()
    # walk-forward: windows at the end of every trial; unused data is grayed out
    p = plot.preview_partitions(ds['uts'][0], partitions=4, test=0, scheme='walk-forward')
    assert_array_equal(p.splits.split_segments[:, 1], [85, 90, 95, 100])
    ax = p.figure.axes[0]
    assert sum(abs(patch.get_facecolor()[0] - 0.85) < 1e-6 for patch in ax.patches) == 2  # the last split uses all data
    p.close()
    p = plot.preview_partitions(2, partitions=4, test=0, scheme='walk-forward')
    assert p.figure.axes[0].get_xlabel() == 'Case'
    assert len(p.splits.split_segments) == 8
    p.close()
    # the windows are exact for small proportions
    p = plot.preview_partitions(1, partitions=3, validate=0.004, test=0, scheme='walk-forward')
    assert_array_equal(p.splits.split_segments, [[0, 992], [992, 996], [996, 1000]])
    p.close()
    # a single split needs no ticks to number the splits
    p = plot.preview_partitions(2, partitions=2, test=0, scheme='walk-forward')
    assert len(p.figure.axes[0].get_yticks()) == 0
    p.close()
    # model: a line above the plot indicates the cell of each trial
    p = plot.preview_partitions(12, partitions=3, test=0, model=Factor('aabb', tile=3))
    lines_a, lines_b = p.figure.axes[0].collections
    assert_array_equal([line[:, 0] / 1000 for line in lines_a.get_segments()], [[0, 1], [1, 2], [4, 5], [5, 6], [8, 9], [9, 10]])
    assert_array_equal([line[:, 0] / 1000 for line in lines_b.get_segments()], [[2, 3], [3, 4], [6, 7], [7, 8], [10, 11], [11, 12]])
    assert [text.get_text() for text in p.legend.get_texts()] == ['Training', 'Validation', 'a', 'b']
    p.close()
    # cells that share names with the plot's roles
    p = plot.preview_partitions(12, partitions=3, test=0, model=Factor(['train', 'test'], tile=6))
    assert [text.get_text() for text in p.legend.get_texts()] == ['Training', 'Validation', 'train', 'test']
    ax = p.figure.axes[0]
    assert ax.patches[0].get_facecolor() != tuple(ax.collections[0].get_colors()[0])
    p.close()
