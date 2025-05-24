# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import datasets, plot
from eelbrain.testing import hide_plots


@hide_plots
def test_map2d():
    "Test plot.SensorMap2d"
    ds = datasets.get_uts(utsnd=True)
    p = plot.SensorMap(ds['utsnd'])

    # plot attributes
    p.set_label_color('g')
    p.set_label_text('index')

    # adjacency
    p.show_adjacency()
    p.show_adjacency(None)

    # mark sensors
    p.mark_sensors([1, 2])
    p.mark_sensors([0])
    p.remove_markers()

    p.close()


@hide_plots
def test_multi():
    "Test plot.SensorMaps"
    ds = datasets.get_uts(utsnd=True)
    p = plot.SensorMaps(ds['utsnd'])
    roi = [1, 2]
    p.set_selection(roi)
    roi2 = p.get_selection()

    test_range = np.arange(3)
    assert_array_equal(test_range[roi2], test_range[roi], "ROI changed after set/get")

    p.close()
