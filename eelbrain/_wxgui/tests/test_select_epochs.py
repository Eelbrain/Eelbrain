# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join

import mne
from numpy.testing import assert_array_equal
import numpy as np
import pytest

from eelbrain import gui, load, save, set_log_level
from eelbrain.testing import TempDir, gui_test
from eelbrain._wxgui.select_epochs import Document, Model


@gui_test
def test_select_epochs():
    "Test Select-Epochs GUI Document"
    set_log_level('warning', 'mne')

    data_path = mne.datasets.testing.data_path()
    raw_path = join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
    raw = mne.io.Raw(raw_path, preload=True).pick_types('mag', stim=True)
    ds = load.mne.events(raw, stim_channel='STI 014')
    ds['meg'] = load.mne.epochs(ds, tmax=0.1)
    # 25 cases
    arange = np.arange(25)
    def false_at(index): return np.isin(arange, index, invert=True)

    tempdir = TempDir()
    path = join(tempdir, 'rej.pickle')

    # Test Document
    # =============
    # create a file
    doc = Document(ds, 'meg')
    doc.set_path(path)
    doc.set_case(1, False, 'tag', None)
    doc.set_case(slice(22, 24), False, 'tag', None)
    doc.set_case(2, None, None, ['2'])
    doc.set_bad_channels([1])
    # check modifications
    assert_array_equal(doc.accept, false_at([1, 22, 23]))
    assert doc.tag[1] == 'tag'
    assert doc.interpolate[1] == []
    assert doc.interpolate[2] == ['2']
    assert doc.bad_channels == [1]
    # save
    doc.save()

    # check the file
    ds_ = load.unpickle(path)
    assert doc.epochs.sensor._array_index(ds_.info['bad_channels']) == [1]

    # load the file
    doc = Document(ds, 'meg', path=path)
    # modification checks
    assert_array_equal(doc.accept, false_at([1, 22, 23]))
    assert doc.tag[1] == 'tag'
    assert doc.interpolate[1] == []
    assert doc.interpolate[2] == ['2']
    assert doc.bad_channels == [1]

    # Test Model
    # ==========
    doc = Document(ds, 'meg', path=path)
    model = Model(doc)

    # accept
    assert_array_equal(doc.accept, false_at([1, 22, 23]))
    model.set_case(0, False, None, None)
    assert_array_equal(doc.accept, false_at([0, 1, 22, 23]))
    model.history.undo()
    assert_array_equal(doc.accept, false_at([1, 22, 23]))
    model.history.redo()
    assert_array_equal(doc.accept, false_at([0, 1, 22, 23]))

    # interpolate
    model.toggle_interpolation(2, '2')
    model.toggle_interpolation(2, '3')
    assert doc.interpolate[2] == ['3']
    model.toggle_interpolation(2, '4')
    assert doc.interpolate[2] == ['3', '4']
    model.toggle_interpolation(2, '3')
    assert doc.interpolate[2] == ['4']
    model.toggle_interpolation(3, '3')
    assert doc.interpolate[2] == ['4']
    assert doc.interpolate[3] == ['3']
    model.history.undo()
    model.history.undo()
    assert doc.interpolate[2] == ['3', '4']
    assert doc.interpolate[3] == []
    model.history.redo()
    assert doc.interpolate[2] == ['4']

    # bad channels
    model.set_bad_channels([1])
    model.set_bad_channels([1, 10])
    assert doc.bad_channels == [1, 10]
    model.history.undo()
    assert doc.bad_channels == [1]
    model.history.redo()
    assert doc.bad_channels == [1, 10]

    # reload to reset
    model.load(path)
    assert_array_equal(doc.accept, false_at([1, 22, 23]))
    assert doc.tag[1] == 'tag'
    assert doc.interpolate[1] == []
    assert doc.interpolate[2] == ['2']
    assert doc.bad_channels == [1]

    # load truncated file
    rej_ds = load.unpickle(path)
    save.pickle(rej_ds[:23], path)
    with pytest.raises(IOError):
        model.load(path, answer=False)
    model.load(path, answer=True)
    assert_array_equal(doc.accept, false_at([1, 22]))

    # Test GUI
    # ========
    frame = gui.select_epochs(ds, nplots=9)
    assert not frame.CanBackward()
    assert frame.CanForward()
    frame.OnForward(None)
    frame.SetVLim(1e-12)
