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
