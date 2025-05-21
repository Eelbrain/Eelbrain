# make mne backwards-compatibility
import packaging.version

import mne


MNE_VERSION = packaging.version.parse(mne.__version__)
V0_19 = packaging.version.parse('0.19')
V0_22 = packaging.version.parse('0.22')
V0_24 = packaging.version.parse('0.24')
V1 = packaging.version.parse('1')

# 0.22 renamed read_ch_adjacency → read_ch_adjacency
if MNE_VERSION < V0_22:
    mne.channels.find_ch_adjacency = mne.channels.find_ch_adjacency
    mne.channels.read_ch_adjacency = mne.channels.read_ch_adjacency
    mne.spatial_src_adjacency = mne.spatial_src_adjacency
