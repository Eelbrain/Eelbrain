# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join
from warnings import catch_warnings, filterwarnings

import mne
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eelbrain import gui, load
from eelbrain.testing import gui_test, TempDir, requires_mne_testing_data
from eelbrain._wxgui import ID
from eelbrain._wxgui.select_components import ComponentMapDialog, FindBadChannelsDialog, HelpDialog, YScaleDialog, _FIND_BAD_CHANNELS_HELP, _find_bad_channels_help


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
    frame.ShowBadChannels(channel_ratio=1.)  # every component is a candidate: renders the diagnostics table
    # mixing in data units reconstructs the channel data from the sources
    doc = frame.doc
    ica_full = mne.preprocessing.read_ica(path)
    applied = doc.as_ndvar(ica_full.apply(ds['epochs'].copy(), n_pca_components=ica_full.n_components_, verbose=False))
    reconstruction = np.einsum('kc,nkt->nct', doc.mixing.x, doc.sources.x) + doc.global_mean.x[:, None]
    assert_allclose(reconstruction, applied.x, rtol=1e-6, atol=1e-6 * np.abs(applied.x).max())
    # share of a channel's variance due to one component
    ch_name = doc.epochs_ndvar.sensor.names[5]
    variance_fraction = doc.channel_variance_fraction(2, ch_name)
    contribution = doc.mixing[2, ch_name] * doc.sources[:, 2]
    assert variance_fraction == pytest.approx(contribution.var() / doc.epochs_ndvar.sub(sensor=ch_name).var())
    assert 0 <= variance_fraction
    # components loading on a single channel, ranked by that share
    candidates = doc.single_channel_components(channel_ratio=1.)
    assert len(candidates) == len(doc.components)
    assert all(a[-1] >= b[-1] for a, b in zip(candidates, candidates[1:]))
    for component, ch_name, max_loadings, variance_fraction in candidates:
        assert ch_name == doc.epochs_ndvar.sensor.names[np.argmax(abs(doc.components[component].x))]
        assert variance_fraction == doc.channel_variance_fraction(component, ch_name)
        assert max_loadings.shape == (len(doc.epochs_ndvar),)
    assert doc.single_channel_components(channel_ratio=1000.) == []
    # flat channels: per channel type standard deviation threshold (SI units)
    assert doc.flat_channels() == []
    names = list(doc.epochs_ndvar.sensor.names)
    std = doc.epochs_ndvar.x.std(axis=(0, 2))
    ch_types = doc.epochs.get_channel_types(picks=names)
    i_quietest_mag = min((i for i, ch_type in enumerate(ch_types) if ch_type == 'mag'), key=std.__getitem__)
    assert doc.flat_channels({'mag': 1.01 * std[i_quietest_mag]}) == [(names[i_quietest_mag], 'mag')]
    # channels missing from component maps, with the default channel types
    gap_results, skipped = doc.channel_gaps()
    assert [result.ch_type for _, result in gap_results] == ['mag', 'eeg']
    assert [ch_type for ch_type, _ in skipped] == ['grad']
    assert [result.ch_type for _, result in doc.channel_gaps({'grad': 0.6})[0]] == ['grad']
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

    # the summary window for marking channels as bad: this recording only
    summary = frame.ShowBadChannelSummary()
    assert len(summary._rows) == len({ch_name for _, ch_name, *_ in summary._rows})  # one row per channel
    assert [combo for combo, *_ in summary._rows] == [()] * len(summary._rows)
    assert summary.exclude() == []
    assert all(combo == () for combo, _ in summary.bad_channels())
    summary.Destroy()
    # adding bad channels is only offered when a host application can write them
    assert frame.doc.bad_channels_callback is None
    frame.AddBadChannels([((), ['MEG 0113'])], True)  # no-op without a callback

    # plotting
    for i in [ID.BASELINE_NONE, ID.BASELINE_GLOABL_MEAN, ID.BASELINE_CUSTOM]:
        frame.butterfly_baseline = i
        frame.OnPlotGrandAverage(None)

    frame.Close()
