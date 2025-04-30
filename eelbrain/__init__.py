# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Accessible tool for statistical analysis of MEG and EEG data.

For the full documentation see <http://eelbrain.readthedocs.io>.



  "The human brain is like an enormous fish. It's flat
   and slimy, and has gills through which it can see."
                                               - Dr. Quat



"""
# import this first for a chance to reverse mne's non-standard logging practice
from . import mne_fixes

from ._config import configure
from ._celltable import Celltable
from ._data_obj import Datalist, Dataset, Var, Factor, Interaction, Model, NDVar, Case, Categorial, Scalar, Sensor, SourceSpace, VolumeSourceSpace, UTS, Space, choose, combine, align, align1, cellname, shuffled_index
from ._experiment import MneExperiment
from ._mne import complete_source_space, labels_from_clusters, morph_source_space, resample_ico_source_space, xhemi
from ._ndvar import Butterworth, concatenate, convolve, correlation_coefficient, cross_correlation, cwt_morlet, dss, filter_data, find_intervals, find_peaks, frequency_response, gaussian, label_operator, maximum, minimum, neighbor_correlation, normalize_in_cells, powerlaw_noise, psd_welch, rename_dim, resample, segment, set_connectivity, set_parc, set_time, set_tmin
from ._ndvar.edge_detector import edge_detector
from ._ndvar.gammatone import gammatone_bank
from ._ndvar.uts import pad
from ._stats.testnd import NDTest, MultiEffectNDTest
from ._trf._boosting import boosting, BoostingResult
from ._trf._predictors import epoch_impulse_predictor, event_impulse_predictor
from ._utils import set_log_level

from . import datasets
from . import gui
from . import load
from . import plot
from . import report
from . import save
from . import table
from . import test
from . import testnd

from .fmtxt import Report


__version__ = '0.40.4'
