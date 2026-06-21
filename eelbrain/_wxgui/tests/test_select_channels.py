# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join

import mne
import numpy as np
import pandas as pd

from eelbrain import gui, set_log_level
from eelbrain.testing import TempDir, gui_test
from eelbrain._wxgui.select_channels import _minmax_envelope


def test_minmax_envelope():
    "Min/max envelope preserves per-bin extremes and target length"
    times = np.arange(100, dtype=float)
    data = np.zeros((2, 100))
    data[0] = np.arange(100)
    data[1, 37] = 999.0  # spike that must survive reduction
    t_env, env = _minmax_envelope(times, data, 10)
    assert env.shape == (2, 20)
    assert t_env.shape == (20,)
    assert env[1].max() == 999.0  # spike preserved as a bin maximum
    assert np.all(env[:, 0::2] <= env[:, 1::2])  # min <= max within each bin
    # passthrough when already at/below target resolution
    t2, env2 = _minmax_envelope(times, data, 1000)
    assert env2.shape == data.shape


def _eeg_raw(n_times=6000, sfreq=100.):
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    rng = np.random.RandomState(0)
    info = mne.create_info(ch_names, sfreq, 'eeg')
    data = rng.standard_normal((len(ch_names), n_times)) * 1e-5
    data[4, 1500] += 5e-4  # spike on C3
    raw = mne.io.RawArray(data, info, verbose='error')
    raw.set_montage('standard_1020')
    return raw, ch_names


@gui_test
def test_select_channels():
    "Smoke-test the Select-Channels GUI Frame (downsampling + LineCollection)"
    set_log_level('warning', 'mne')
    raw, ch_names = _eeg_raw()

    tempdir = TempDir()
    path = join(tempdir, 'sub-01_channels.tsv')
    status = ['bad' if n == 'F3' else 'good' for n in ch_names]
    pd.DataFrame({'name': ch_names, 'status': status}).to_csv(path, sep='\t', index=False)

    frame = gui.select_channels(raw, path)
    red = (1.0, 0.0, 0.0, 1.0)

    # one LineCollection per channel type, one segment per channel
    assert set(frame._butterfly_lc) == {'eeg'}
    lc = frame._butterfly_lc['eeg']
    assert frame._butterfly_names['eeg'] == ch_names
    assert len(lc.get_segments()) == len(ch_names)
    # bad channel from the file is drawn red
    assert tuple(lc.get_colors()[ch_names.index('F3')]) == red

    # envelope and auto-decim are on by default
    assert frame._envelope is True
    assert frame._decim_auto is True
    # auto decim targets ~1 sample per horizontal pixel for the window
    assert frame._effective_decim() == max(1, round(frame._sfreq * frame.window_size / frame._axis_width_px()))

    # scrolling keeps a valid blit background and one segment per channel
    assert frame.CanForward()
    frame.OnForward(None)
    assert frame.canvas._background is not None
    assert len(frame._butterfly_lc['eeg'].get_segments()) == len(ch_names)

    # toggling a bad channel recolors the matching segment
    frame.model.toggle_bad('C3')
    assert tuple(frame._butterfly_lc['eeg'].get_colors()[ch_names.index('C3')]) == red
    frame.model.toggle_bad('C3')
    assert tuple(frame._butterfly_lc['eeg'].get_colors()[ch_names.index('C3')]) != red

    # full resolution: envelope off, no decimation
    frame._envelope = False
    frame._decim_auto = False
    frame._decim = 1
    frame._update_window()
    n_full = frame._butterfly_lc['eeg'].get_segments()[0].shape[0]

    # manual decimation reduces the point count
    frame._decim = 4
    frame._update_window()
    n_decim = frame._butterfly_lc['eeg'].get_segments()[0].shape[0]
    assert n_decim < n_full

    # auto decim also reduces relative to full resolution
    frame._decim_auto = True
    frame._update_window()
    assert frame._butterfly_lc['eeg'].get_segments()[0].shape[0] <= n_full
