"""
Tools for loading data.

The following submodules are available:

fiff:
    Load mne fiff files to datasets and as mne objects (requires mne-python)

txt:
    Load datasets and vars from text files

"""

from .._io import txt
from . import besa
from . import fiff
from . import mne

from .._io.txt import tsv
from .._io.cnd import read_cnd as cnd
from .._io.pickle import unpickle, update_subjects_dir, convert_pickle_protocol
from .._io.sphere import load_sphere as sphere_audio
from .._io.wav import load_wav as wav
