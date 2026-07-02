# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from eelbrain._experiment.preprocessing.nodes import RawSourceInput


def test_read_raw_applies_bids_channels(tmp_path):
    "RawSourceInput._apply_bids_channels applies channels.tsv channel metadata"
    bids_path = BIDSPath(
        root=tmp_path,
        subject='01',
        task='test',
        datatype='meg',
        suffix='meg',
        extension='.fif',
    )
    bids_path.fpath.parent.mkdir(parents=True)
    info = mne.create_info(['MEG 001', 'EOG 001'], 100, ['mag', 'misc'])
    data = np.zeros((2, 100))
    raw = mne.io.RawArray(data, info, verbose='error')
    raw.save(bids_path.fpath, overwrite=True, verbose='error')

    channels_path = bids_path.copy().update(suffix='channels', extension='.tsv').fpath
    channels = pd.DataFrame({
        'name': ['MEG 001', 'EOG 001'],
        'type': ['MEGMAG', 'EOG'],
        'units': ['T', 'V'],
        'status': ['good', 'bad'],
    })
    channels.to_csv(channels_path, sep='\t', index=False)

    raw_read = RawSourceInput._read_raw(bids_path, preload=False)
    RawSourceInput._apply_bids_channels(bids_path, raw_read)

    assert raw_read.get_channel_types(picks=['EOG 001']) == ['eog']
    assert raw_read.info['bads'] == []
