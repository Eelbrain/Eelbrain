# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from numpy.testing import assert_array_equal

from eelbrain import datasets, boosting, epoch_impulse_predictor, plot
from eelbrain.testing import hide_plots


@hide_plots
def test_datasplit():
    ds = datasets.get_uts()
    ds['x'] = epoch_impulse_predictor('uts', data=ds)

    trf = boosting('uts', 'x', 0, 0.100, data=ds, partitions=4, test=0)
    p = trf.splits.plot()
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
    # trials specified by number: no time information
    p = plot.preview_partitions(20, partitions=5, test=0)
    assert p.figure.axes[0].get_xlabel() == 'Case'
    p.close()
    # NDVar data
    ds = datasets.get_uts()
    p = plot.preview_partitions('uts', data=ds, partitions=4)
    assert p.figure.axes[0].get_xlabel() == 'Time [s]'
    p.close()
    # continuous data without time information: one sample per partition, labeled at its center
    p = plot.preview_partitions(partitions=4, test=0)
    ax = p.figure.axes[0]
    assert ax.get_xlabel() == 'Partition'
    assert_array_equal(ax.get_xticks(), [0.5, 1.5, 2.5, 3.5])
    assert [label.get_text() for label in ax.get_xticklabels()] == ['0', '1', '2', '3']
    assert_array_equal(p.splits.split_segments, [[0, 1], [1, 2], [2, 3], [3, 4]])
    p.close()
    # walk-forward: windows at the end of every trial; unused data is grayed out
    p = plot.preview_partitions(ds['uts'][0], partitions=3, test=0, split='walk-forward')
    assert_array_equal(p.splits.split_segments[:, 1], [85, 90, 95, 100])
    ax = p.figure.axes[0]
    assert sum(abs(patch.get_facecolor()[0] - 0.85) < 1e-6 for patch in ax.patches) == 2  # the last split uses all data
    p.close()
    p = plot.preview_partitions(2, partitions=3, test=0, split='walk-forward')
    assert p.figure.axes[0].get_xlabel() == 'Sample'
    assert len(p.splits.split_segments) == 8
    p.close()
