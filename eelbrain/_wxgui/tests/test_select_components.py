# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join
from warnings import catch_warnings, filterwarnings

import mne
from eelbrain import gui, load
from eelbrain.testing import gui_test, TempDir, requires_mne_testing_data
from eelbrain._wxgui import ID
from eelbrain._wxgui.select_components import AddBadChannelsDialog, ComponentMapDialog, FindBadChannelsDialog, HelpDialog, YScaleDialog, _FIND_BAD_CHANNELS_HELP, _find_bad_channels_help


@gui_test
@requires_mne_testing_data
def test_select_components():
    "Test Select-Epochs GUI Document"
    tempdir = TempDir()
    path = join(tempdir, 'test-ica.fif')

    data_path = mne.datasets.testing.data_path(download=False)
    raw_path = join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
    raw = mne.io.Raw(raw_path, preload=True)
    ds = load.mne.events(raw, stim_channel='STI 014')
    ds['epochs'] = load.mne.mne_epochs(ds, tmax=0.1)
    ica = mne.preprocessing.ICA(0.95, max_iter=1)
    with catch_warnings():
        filterwarnings('ignore', 'FastICA did not converge')
        ica.fit(raw)
    ica.save(path)

    frame = gui.select_components(path, ds)
    frame.model.toggle(1)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(path)
    assert ica.exclude == [1]

    frame.OnUndo(None)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(path)
    assert ica.exclude == []

    # tools
    frame.ShowBadChannels()
    dlg = FindBadChannelsDialog(frame, frame.doc.components_by_type)
    assert [ch_type for ch_type, _, _ in dlg.type_rows] == [ch_type for ch_type, _ in frame.doc.components_by_type]
    ch_type, components = frame.doc.components_by_type[0]
    map_dlg = ComponentMapDialog(dlg, ch_type, components)
    map_dlg.Destroy()
    help_dlg = HelpDialog(dlg, "Find Bad Channels", _find_bad_channels_help())
    help_dlg.Destroy()
    # tooltips and the help dialog use the same strings
    assert dlg.gap_ratio.GetToolTipText() == _FIND_BAD_CHANNELS_HELP['gap_ratio'][1]
    help_doc = str(_find_bad_channels_help())
    assert all(label in help_doc for label, _ in _FIND_BAD_CHANNELS_HELP.values())
    # min_components is a count: 0 and fractions would crash the report
    pattern = dlg.min_components.GetValidator().pattern
    assert pattern.match('2')
    assert not pattern.match('0')
    assert not pattern.match('2.5')
    dlg.Destroy()

    # layout and scale: one text box per value
    scale_dlg = YScaleDialog(frame, 5, 8, 2., 3., frame.doc.continuous)
    assert scale_dlg.GetValues() == (5, 8, 2., 3.)
    scale_dlg.Destroy()

    # adding bad channels is only offered when a host application can write them
    assert frame.doc.bad_channels_callback is None
    frame.AddBadChannels(['MEG 0113'])  # no-op without a callback
    bad_dlg = AddBadChannelsDialog(frame, ['MEG 0113'])
    assert bad_dlg.recompute.GetValue()
    bad_dlg.Destroy()

    # plotting
    for i in [ID.BASELINE_NONE, ID.BASELINE_GLOABL_MEAN, ID.BASELINE_CUSTOM]:
        frame.butterfly_baseline = i
        frame.OnPlotGrandAverage(None)

    frame.Close()
