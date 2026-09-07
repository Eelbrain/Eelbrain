# make mne backwards-compatibility
import packaging.version

import mne


MNE_VERSION = packaging.version.parse(mne.__version__)
V1 = packaging.version.parse('1')
# Head movement compensation relies on fixes to find_bad_channels_maxwell and maxwell_filter
MNE_SUPPORTS_HEAD_POS = MNE_VERSION >= packaging.version.parse('1.13.0.dev0')

assert MNE_VERSION > V1
