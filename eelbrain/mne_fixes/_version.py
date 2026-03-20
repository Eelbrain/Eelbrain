# make mne backwards-compatibility
import packaging.version

import mne


MNE_VERSION = packaging.version.parse(mne.__version__)
V1 = packaging.version.parse('1')

assert MNE_VERSION > V1
