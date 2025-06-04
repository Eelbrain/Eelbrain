# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import sys

import numpy as np
import pytest

from eelbrain import datasets, plot, _info, NDVar
from eelbrain.testing import hide_plots, requires_mne_testing_data


@hide_plots
@requires_mne_testing_data
@pytest.mark.mayavi
def test_plot_brain():
    """Test plot.brain plots"""
    if sys.platform.startswith('win'):
        pytest.xfail("Hangs on Appveyor")
    stc = datasets.get_mne_stc(True, 'oct-4')

    # size
    b = plot.brain.brain(stc.source, hemi='rh', w=400, h=300, mask=False)
    assert b.screenshot().shape in ((300, 400, 3), (600, 800, 3))
    if sys.platform == 'linux':
        pytest.xfail("Brain.set_size() on Linux/Travis")
    b.set_size(200, 150)
    assert b.screenshot().shape in ((150, 200, 3), (300, 400, 3))
    b.close()
    # both hemispheres [broken upstream?]
    # b = plot.brain.brain(stc.source, w=600, h=300, mask=False)
    # assert b.screenshot().shape in ((300, 600, 3), (600, 1200, 3))
    # b.set_size(400, 150)
    # assert b.screenshot().shape in ((150, 400, 3), (300, 800, 3))
    # b.close()

    # standard plot
    y = stc.sub(time=0.090)
    brain = plot.brain.brain(y, mask=False, hemi='lh')
    # labels (int)
    labels = y.astype(int)
    brain.add_ndvar_annotation(labels)
    brain.add_ndvar_annotation(labels, 'jet')
    brain.close()

    # plot shortcuts
    p = plot.brain.dspm(stc, mask=False)
    cb = p.plot_colorbar()
    cb.close()
    p.close()

    p = plot.brain.dspm(stc, hemi='lh', mask=False)
    cb = p.plot_colorbar()
    cb.close()
    p.close()

    p = plot.brain.cluster(stc, hemi='rh', views='parietal', mask=False)
    cb = p.plot_colorbar()
    cb.close()
    p.close()

    image = plot.brain.bin_table(stc, tstart=0.05, tstop=0.07, tstep=0.01, surf='white', mask=False)
    print(repr(image))
    print(image)

    # plot p-map
    pmap = NDVar(np.random.uniform(0, 1, stc.shape), stc.dims, _info.for_p_map(stc.info))
    p = plot.brain.p_map(pmap, stc, mask=False)
    cb = p.plot_colorbar()
    cb.close()
    p.close()

    # mask
    stcm = stc.mask(stc < 11)
    p = plot.brain.brain(stcm, mask=False)
    p.close()


@hide_plots
@requires_mne_testing_data
@pytest.mark.mayavi
def test_sequence_plotter():
    if sys.platform.startswith('win'):
        pytest.xfail("Hangs on Appveyor")
    stc = datasets.get_mne_stc(True)
    stc_mask = stc > 5

    # time dimension
    y = stc.sub(time=(0.000, 0.200)).bin(0.050)
    y_mask = stc_mask.sub(time=(0.000, 0.200)).bin(0.050, func='max')
    sp = plot.brain.SequencePlotter()
    sp.set_brain_args(surf='white', mask=False)  # the source space required for mask is not in the test dataset
    sp.add_ndvar(y, vmax=10)
    sp.add_ndvar(y_mask)
    sp.plot_table(view='lateral')
    # test internals
    assert sp._get_frame_labels(True) == ['0 - 50 ms', '50 - 100', '100 - 150', '150 - 200']

    # separate NDVars
    sp = plot.brain.SequencePlotter()
    sp.set_brain_args(mask=False)
    sp.add_ndvar(stc.mean(time=(0.150, 0.250)), label='normal')
    sp.add_ndvar(stc.mean(time=(0.150, 0.250)), cmap='jet', label='jet')
    sp.add_ndvar(stc_mask.max(time=(0.150, 0.250)), label='mask')
    sp.plot_table(view='lateral', orientation='vertical')
    # test internals
    assert sp._get_frame_labels(True) == ['normal', 'jet', 'mask']
