# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join

import mne
from numpy.testing import assert_array_equal
import numpy as np
import pytest

from eelbrain import gui, load, save, set_log_level
from eelbrain._data_obj import Dataset, Datalist, Var
from eelbrain._info import INTERPOLATE_WINDOWS
from eelbrain._meeg import BadChannelWindow
from eelbrain.testing import TempDir, gui_test, requires_mne_testing_data
from eelbrain._wxgui.select_epochs import Document, Model


@gui_test
@requires_mne_testing_data
def test_select_epochs():
    "Test Select-Epochs GUI Document"
    set_log_level('warning', 'mne')

    data_path = mne.datasets.testing.data_path(download=False)
    raw_path = join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
    raw = mne.io.Raw(raw_path, preload=True).pick_types('mag', stim=True)
    ds = load.mne.events(raw, stim_channel='STI 014')
    ds['meg'] = load.mne.mne_epochs(ds, tmax=0.1)
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
    # before/after-rejection topomaps (single MEG type -> one pair), each
    # occupying a full reserved grid cell (epochs keep full cell height)
    assert [kind for _, kind in frame._topo_specs] == ['all', 'rejected']
    assert len(frame._topo_plots) == 2
    assert frame._n_reserve == 2
    epoch_h = frame._case_axes[0].get_position().height
    topo_h = frame._topo_axes[0].get_position().height
    assert abs(topo_h - epoch_h) < 0.02  # topomap as tall as an epoch cell
    background = frame.canvas._background
    frame.OnForward(None)
    assert frame.canvas._background is not background
    background = frame.canvas._background
    frame.SetVLim(1e-12)
    assert frame.canvas._background is not background

    # read-only mode
    ro_path = join(tempdir, 'rej_ro.pickle')
    save.pickle(rej_ds, ro_path)
    frame = gui.select_epochs(ds, path=ro_path, nplots=9, read_only=True)
    assert frame.read_only is True
    assert frame.CanSave() is False
    assert '(read-only)' in frame.GetTitle()
    # editing is a no-op in read-only mode
    n_bad = len(frame.doc.bad_channels)
    frame.OnSetBadChannels(None)
    assert len(frame.doc.bad_channels) == n_bad


def _variable_length_ds():
    "Dataset with ragged epochs (different lengths) and windowed interpolation"
    set_log_level('warning', 'mne')
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    sfreq = 100.
    rng = np.random.RandomState(0)

    def make_epoch(n_times):
        info = mne.create_info(ch_names, sfreq, 'eeg')
        data = rng.standard_normal((1, len(ch_names), n_times)) * 1e-5
        epochs = mne.EpochsArray(data, info, tmin=-0.1, verbose='error')
        epochs.set_montage('standard_1020')
        return epochs

    epochs_list = [make_epoch(300), make_epoch(400)]  # 3 s and 4 s
    ds = Dataset()
    ds['epochs'] = Datalist(epochs_list, 'epochs')
    ds['value'] = Var([1, 2])
    ds[INTERPOLATE_WINDOWS] = Datalist([
        [BadChannelWindow('C3', 0.5, 1.5)],
        [],
    ], INTERPOLATE_WINDOWS)
    return ds


@gui_test
def test_select_epochs_long():
    "Select-Epochs GUI for long, variable-length epochs with windowed interpolation"
    ds = _variable_length_ds()

    # Document
    doc = Document(ds, 'epochs', trigger='value')
    assert doc.long_epochs is True
    assert doc.n_epochs == 2
    assert len(doc.epoch_data) == 2
    # epoch 0 has 300 samples, epoch 1 has 400
    assert doc.epoch_data[0][0][1].time.nsamples == 300
    assert doc.epoch_data[1][0][1].time.nsamples == 400
    assert doc.interpolate_windows[0][0].channel == 'C3'
    assert not doc.interpolate_windows[1]
    assert [w.channel for w in doc.windows_in_range(0, 1.0, 1.2)] == ['C3']
    assert doc.windows_in_range(0, 2.0, 2.5) == []
    assert doc.windows_in_range(1, 0.0, 4.0) == []

    # windows are read back from a rejection file (as the pipeline supplies them)
    tempdir = TempDir()
    rej_path = join(tempdir, 'rej.pickle')
    rej_ds = Dataset()
    rej_ds['value'] = ds['value']
    rej_ds[:, 'accept'] = True
    rej_ds[:, 'rej_tag'] = ''
    rej_ds[INTERPOLATE_WINDOWS] = ds[INTERPOLATE_WINDOWS]
    save.pickle(rej_ds, rej_path)
    doc_file = Document(_variable_length_ds(), 'epochs', trigger='value', path=rej_path)
    assert doc_file.interpolate_windows[0][0].channel == 'C3'
    assert not doc_file.interpolate_windows[1]

    # Frame (read-only continuous browser)
    frame = gui.select_epochs(ds, 'epochs', trigger='value', topo=True)
    assert frame.long_epochs is True
    assert frame.read_only is True
    assert '(read-only)' in frame.GetTitle()

    # exercise multi-row tiling and paging
    frame._seconds_per_row = 1.
    frame._rows_per_page = 4
    frame._build_rows()
    # epoch 0 -> 3 rows, epoch 1 -> 4 rows; each epoch starts a new row
    assert len(frame._rows_spec) == 7
    assert frame._rows_spec[0][0] == 0
    assert frame._rows_spec[3][0] == 1  # epoch 1 starts on its own row
    assert frame._n_pages == 2

    frame.ShowPage(0)
    assert frame.CanForward()
    assert not frame.CanBackward()
    # the C3 window (0.5-1.5 s) overlaps rows on the first page -> highlight artists
    assert len(frame._window_handles) >= 1

    # before/after-rejection topomaps (single EEG type -> one pair)
    assert [kind for _, kind in frame._topo_specs] == ['all', 'rejected']
    assert len(frame._topo_plots) == 2
    # row 0 spans [-0.1, 0.9) s; C3 is rejected at 0.7 s, marked with a blue x
    frame._update_topomaps(0, 0, 0.7)
    assert len(frame._topo_interp_handles) == 2  # one mark per topomap
    # outside the window no channel is rejected -> no marks
    frame._update_topomaps(0, 0, 0.0)
    assert len(frame._topo_interp_handles) == 0

    # too few electrodes remain to interpolate -> the after-rejection map is blanked
    from eelbrain._meeg import BadChannelWindow as _BCW
    frame.doc.interpolate_windows[0] = [_BCW(ch, 0.5, 1.5) for ch in ('C3', 'C4', 'F3', 'F4', 'P3')]
    frame._update_topomaps(0, 0, 0.7)
    all_ax, rej_ax = frame._topo_axes
    assert all_ax.patch.get_visible() is False   # 'all' map still shown
    assert rej_ax.patch.get_visible() is True    # 'rejected' map blanked (white patch)
    assert not any(im.get_visible() for im in rej_ax.images)

    frame.OnForward(None)
    assert frame.CanBackward()
