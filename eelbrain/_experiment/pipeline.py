# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pipeline class to manage data from a experiment"""
import copy
from datetime import datetime
from itertools import chain, product
import logging
import os
from os.path import exists
from pathlib import Path
from typing import Any, Literal
from collections.abc import Sequence

import numpy as np
import mne
from mne.minimum_norm import apply_inverse_raw
import mne_bids
from mne_bids import BIDSPath, get_entity_vals

from .. import fmtxt
from .. import gui
from .. import load
from .. import plot
from .. import save
from .._data_obj import CellArg, Datalist, Dataset, Factor, Var, NDVar, SourceSpace, VolumeSourceSpace, assert_is_legal_dataset_key, combine
from .._exceptions import ConfigurationError, OldVersionError
from .._info import BAD_CHANNELS
from .._names import INTERPOLATE_CHANNELS
from .._meeg import new_rejection_ds
from .._mne import find_source_subject, label_from_annot
from ..mne_fixes import suppress_mne_warning
from .._ndvar import concatenate, cwt_morlet, neighbor_correlation
from .._text import enumeration
from .._types import PathArg
from .._utils import ask, subp, keydefaultdict, log_level, ScreenHandler
from .._utils.mne_utils import is_fake_mri
from .covariance import CovDerivative, EpochCovariance, RawCovariance, cov_node_name
from .derivative_cache import ALLOW_PROTECTED_OVERWRITE, DerivativeRegistry, ProtectedArtifactError
from .configuration import sequence_arg
from .epochs import (
    EpochBase, EpochsDerivative,
    EvokedDerivative, EvokedGroupDatasetDerivative, PrimaryEpoch, RejectionInput,
    SecondaryEpoch, SuperEpoch, assemble_epochs, decim_param,
)
from .events import EventsDerivative, SELECTED_EVENTS, SelectedEventsDerivative
from .exceptions import FileMissingError
from .experiment import StateModel
from .groups import assemble_groups
from .pathing import (
    deriv_dir, epoch_basename, ica_file_path,
    join_stem_parts, mri_dir, mri_sdir, raw_basename, raw_dir, rej_file_path,
    results_dir, src_file_path, trans_file_path,
)
from .parc import SEEDED_PARC_RE, AnnotDerivative, CombinationParc, EelbrainParc, FreeSurferParc, FSAverageParc, IndividualSeededParc, LabelParc, Parcellation, SeededParc, VolumeParc, _resolve_parc
from .preprocessing import (
    ICAInput, RawBadChannelsInput, RawDerivative, RawPipe, RawSource, RawSourceInput, RawICA,
    REINDEX_ICA, assemble_raw_pipes, get_ica_pipe, get_ica_pipe_name,
    ica_input_name, raw_bad_channels_input_name, raw_node_name,
)
from .reports import (
    CoregReportDerivative, EEGReportDerivative, EEGSensorsReportDerivative,
    LMReportDerivative, ROIReportDerivative, SourceReportDerivative,
)
from .results import EvokedTestDataDerivative, MovieDerivative, TestResultDerivative
from .source import (
    BemInput, EpochsStcDerivative, EpochsStcGroupDatasetDerivative,
    EvokedStcDerivative, EvokedStcGroupDatasetDerivative, FwdDerivative,
    InvDerivative, ROIData, SourceMorphDerivative, SrcDerivative, TransInput,
    InverseSolution, MinimumNormInverseSolution, _drop_unknown_labels, _source_parc, eval_src,
)
from .test_def import (
    Test,
    TestDims, validate_tests,
)
from .two_stage import TwoStageDataDerivative, TwoStageLevel1Derivative, TwoStageLevel2Derivative, TwoStageTest
from .variable_def import Variables, apply_vardef, label_groups as label_groups_var

BIDS_PATH_KEYS = ('datatype', 'suffix', 'extension', 'subject', 'session', 'task', 'acquisition', 'run', 'split')

# Allowable parameters
COV_PARAMS = {'epoch', 'method', 'reg', 'keep_sample_mean', 'reg_eval_win_pad'}
# Argument types
BaselineArg = bool | tuple[float | None, float | None]
DataArg = str | TestDims
PMinArg = Literal['tfce'] | float | None
SubjectArg = str | Literal[1, -1]


def guess_y(ds, default=None):
    "Given a dataset, guess the dependent variable"
    for y in ('srcm', 'src', 'meg', 'eeg'):
        if y in ds:
            return y
    if default is not None:
        return default
    raise RuntimeError(f"Could not find data in {ds}")


class Pipeline(StateModel):
    """Analyze an MEG or EEG experiment

    Parameters
    ----------
    root : str | None
        the root directory for the experiment (usually the directory
        containing the 'meg' and 'mri' directories). The experiment can be
        initialized without the root for testing purposes.
    find_subjects : bool
        Automatically look for subjects in the MEG-directory (default
        True). Set ``find_subjects=False`` to initialize the experiment
        without any files.
    ...
        Initial state parameters.

    Notes
    -----
    .. seealso::
        Guide on using :ref:`experiment-class-guide`.
    """
    _repr_args = ('root',)
    path_version: int = 2
    screen_log_level: str | int = logging.INFO
    cache_inv: bool = True  # Whether to cache inverse solution
    # moderate speed gain for loading source estimates (34 subjects: 20 vs 70 s)
    # hard drive space ~ 100 mb/file
    cache_policy_overrides: dict[str, str | bool] = {}

    # datatype and extension are usually inferred from a BIDS dataset; override here if needed
    datatype: str = None
    extension: str = None

    ignore_entities: dict[str, list[str]] = {}
    preload: bool = False

    # Raw preprocessing pipeline
    raw: dict[str, RawPipe] = {}

    # Load events from a subset of available stim channels
    stim_channel: str | Sequence[str] = None
    # merge adjacent events in the stimulus channel
    merge_triggers: int = None
    # add this value to all trigger times (in seconds); global shift, or {subject: shift, (subject, session): shift} dictionary
    trigger_shift: float | dict[str | tuple, float] = 0

    # variables for automatic labeling {name: {trigger: label, triggers: label}}
    variables: dict[str, Any] = {}

    # named epochs
    epochs: dict[str, EpochBase] = {}

    # Rejection
    # =========
    # eog_sns: The sensors to plot separately in the rejection GUI. The default
    # is the two MEG sensors closest to the eyes.
    _eog_sns = {
        'KIT-157': ('MEG 143', 'MEG 151'),
        'KIT-NYU-2019': ('MEG 014', 'MEG 146'),
        'KIT-208': ('MEG 087', 'MEG 130'),
        'KIT-UMD-1': ('MEG 042', 'MEG 025'),
        'KIT-UMD-2': ('MEG 042', 'MEG 025'),
        'KIT-UMD-3': ('MEG 042', 'MEG 025'),
        'KIT-BRAINVISION': ('HEOGL', 'HEOGR', 'VEOGb'),
        'neuromag306mag': ('MEG 0121', 'MEG 1411'),
    }
    #
    # artifact_rejection dict:
    #
    # kind : 'manual' | 'make'
    #     How the rejection is derived:
    #     'manual': manually create a rejection file (use the selection GUI
    #     through .make_epoch_selection())
    #     'make' a rejection file is created by the user
    # interpolation : bool
    #     enable by-epoch channel interpolation
    #
    # For manual rejection
    # ^^^^^^^^^^^^^^^^^^^^
    _artifact_rejection = {
        '': {'kind': None},
        'man': {'kind': 'manual', 'interpolation': True},
    }
    artifact_rejection = {}
    _artifact_rejection_default = 'man'

    # groups can be defined as subject lists: {'group': ('member1', 'member2', ...)}
    # or by exclusion: {'group': {'base': 'all', 'exclude': ('member1', 'member2')}}
    groups = {}

    # MEG-system (used as ``sysname`` to infer adjacency; for usage search `get_sysname`).
    meg_system = None

    # kwargs for regularization of the covariance matrix
    _covs = {
        'auto': EpochCovariance('cov', 'auto'),
        'bestreg': EpochCovariance('cov', 'best'),
        'reg': EpochCovariance('cov', 'diagonal_fixed'),
        'noreg': EpochCovariance('cov', 'empirical'),
        'emptyroom': RawCovariance(),
        'ad_hoc': RawCovariance(method='ad_hoc'),
    }

    # MRI subject names: {subject: mrisubject} mappings
    # selected with e.set(mri=dict_name)
    # default is identity (mrisubject = subject)
    mri_subjects = {'': keydefaultdict(lambda s: s)}

    # Parcellations
    _default_parcs = {
        'aparc.a2005s': FreeSurferParc(),
        'aparc.a2009s': FreeSurferParc(),
        'aparc': FreeSurferParc(),
        'aparc.DKTatlas': FreeSurferParc(),
        'cortex': LabelParc(('cortex',), ('lateral', 'medial')),
        'PALS_B12_Brodmann': FSAverageParc(),
        'PALS_B12_Lobes': FSAverageParc(),
        'PALS_B12_OrbitoFrontal': FSAverageParc(),
        'PALS_B12_Visuotopic': FSAverageParc(),
        # Volume
        'aparc+aseg': VolumeParc(),
        # Combinations
        'lobes': EelbrainParc(True, ('lateral', 'medial')),
        'lobes-op': CombinationParc('lobes', {'occipitoparietal': "occipital + parietal"}, ('lateral', 'medial')),
        'lobes-ot': CombinationParc('lobes', {'occipitotemporal': "occipital + temporal"}, ('lateral', 'medial')),
    }
    parcs: dict[str, Parcellation] = {}

    # Frequencies:  lowbound, highbound, step
    _freqs = {'gamma': {'frequencies': np.arange(25, 50, 2),
                        'n_cycles': 5}}
    freqs = {}

    # specify defaults for specific fields (e.g. specify the initial subject
    # name)
    defaults = {}

    # model order: list of factors in the order in which models should be built
    # (default for factors not in this list is alphabetic)
    _model_order = []

    # Tests
    # -----
    # Tests imply a model which is set automatically
    tests: dict[str, Test] = {}
    _empty_test = False  # for TRFExperiment
    # plotting
    # --------
    _brain_plot_defaults = {'surf': 'inflated'}
    brain_plot_defaults = {}

    def __init__(
            self,
            root: PathArg,
            **state,
    ):
        ########################################################################
        # Checks
        ########
        if root is None:
            raise AttributeError("Pipeline subclasses must have root.")
        self.root = root = Path(root).absolute().expanduser()
        # BIDS entities
        # ignore task `noise` by default
        ignore_entities = copy.deepcopy(self.ignore_entities)
        ignore_tasks = ignore_entities.get('ignore_tasks', [])
        if 'noise' not in ignore_tasks:
            ignore_entities['ignore_tasks'] = list(ignore_tasks) + ['noise']

        self._subjects = tuple(get_entity_vals(root, 'subject', **ignore_entities))
        self._sessions = tuple(get_entity_vals(root, 'session', **ignore_entities))
        self._tasks = tuple(get_entity_vals(root, 'task', **ignore_entities))
        self._acquisitions = tuple(get_entity_vals(root, 'acquisition', **ignore_entities))
        self._runs = tuple(get_entity_vals(root, 'run', **ignore_entities))
        self._splits = tuple(get_entity_vals(root, 'split', **ignore_entities))

        if self.datatype is not None:
            if self.datatype not in ('meg', 'eeg'):
                raise ConfigurationError(f"`datatype` must be 'meg' or 'eeg', not {self.datatype!r}.")
            if not isinstance(self.extension, str):
                raise TypeError(f"{self.__class__.__name__}.extension={self.extension!r} with {self.__class__.__name__}.datatype={self.datatype!r}; extension needs to be specified (e.g., '.fif').")
            self._datatype = self.datatype
            extensions = (self.extension,)
        else:
            datatypes = tuple(mne_bids.get_datatypes(root))
            if 'meg' in datatypes and 'eeg' in datatypes:
                raise ConfigurationError(f"Can't infer datatype. Both MEG and EEG data found in {root}.")
            elif 'meg' in datatypes:
                self._datatype = 'meg'
                extensions = ('.fif',)
            elif 'eeg' in datatypes:
                self._datatype = 'eeg'
                data_extensions = {path.extension for path in mne_bids.find_matching_paths(root, datatypes='eeg', suffixes='eeg', extensions=['.edf', '.vhdr', '.set', '.bdf', '.fif'])}
                if len(data_extensions) == 0:
                    raise FileMissingError(f"No EEG data files found in {root}.")
                elif len(data_extensions) > 1:
                    raise ConfigurationError(f"Multiple EEG data file types found in {root}: {enumeration(sorted(data_extensions))}.")
                extensions = tuple(data_extensions)
            else:
                raise ConfigurationError(f"Can't infer datatype. No MEG or EEG data found in {root}.")
        self._bids_path = BIDSPath(root=root)
        StateModel.__init__(self)

        ########################################################################
        # Logger
        ########
        # log-file
        self._log = log = logging.Logger(self.__class__.__name__, logging.DEBUG)
        log_file = root / 'derivatives' / 'eelbrain' / f'{self.__class__.__name__}.log'
        os.makedirs(log_file.parent, exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(levelname)-8s %(asctime)s %(message)s", "%m-%d %H:%M")  # %(name)-12s
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        log.addHandler(handler)

        # terminal log
        handler = ScreenHandler()
        self._screen_log_level = log_level(self.screen_log_level)
        handler.setLevel(self._screen_log_level)
        log.addHandler(handler)
        self._screen_log_handler = handler

        ########################################################################
        # Experiment arguments
        ######################
        # groups
        self._groups = assemble_groups(self.groups, set(self._subjects))

        # mri_subjects
        self._mri_subjects = self.mri_subjects.copy()

        # preprocessing
        self._raw = assemble_raw_pipes({'raw': RawSource(), **self.raw}, self._tasks)
        raw_pipe: RawSource = self._raw['raw']

        # legacy adjacency determination
        if raw_pipe.sysname is None:
            if self.meg_system is not None:
                raw_pipe.sysname = self.meg_system

        # variables
        self._variables = Variables(self.variables)
        self._variables._check_trigger_vars()

        # epochs
        self._epochs = assemble_epochs(self.epochs)

        # epoch rejection
        artifact_rejection = {}
        for name, params in chain(self._artifact_rejection.items(), self.artifact_rejection.items()):
            if params['kind'] in ('manual', 'make', None):
                artifact_rejection[name] = params.copy()
            elif params['kind'] == 'ica':
                raise ValueError(f"kind={params['kind']!r} in artifact_rejection {name!r}; The ICA option has been removed, use the RawICA raw pipe instead.")
            else:
                raise ValueError(f"kind={params['kind']!r} in artifact_rejection {name!r}")
        self._artifact_rejection = artifact_rejection

        # parcellations
        # make : can be made if non-existent
        # morph_from_fraverage : can be morphed from fsaverage to other subjects
        for name, parc in self.parcs.items():
            if not isinstance(parc, Parcellation):
                raise TypeError(f"parcs[{name!r}]={parc!r}: need Parcellation")
        self._parcs = {**self._default_parcs, **self.parcs}
        for name, parc in self._parcs.items():
            parc._store_name(name)
        parc_values = [*self._parcs.keys(), '']

        # frequency
        freqs = {}
        for name, f in chain(self._freqs.items(), self.freqs.items()):
            if name in freqs:
                raise ValueError(f"Frequency {name} defined twice")
            elif 'frequencies' not in f:
                raise KeyError(f"Frequency values missing for {name}")
            elif 'n_cycles' not in f:
                raise KeyError(f"Number of cycles not defined for {name}")
            freqs[name] = f
        self._freqs = freqs

        # tests
        validate_tests(self.tests)

        ########################################################################
        # Experiment class setup
        ########################
        # epoch
        epoch_keys = sorted(self._epochs)
        for default_epoch in epoch_keys:
            if isinstance(self._epochs[default_epoch], PrimaryEpoch):
                break
        else:
            default_epoch = None
        self._register_field('epoch', epoch_keys, default_epoch, repr=True)

        # Register BIDS_PATH_KEYS
        self._register_field('subject', self._subjects, repr=True)
        self._register_field('session', self._sessions or None, repr=True)
        self._register_field('task', self._tasks, depends_on=('epoch',), slave_handler=self._update_task, repr=True)
        self._register_field('acquisition',  self._acquisitions or None, repr=True)
        self._register_field('run', self._runs or None, repr=True)
        self._register_field('split', self._splits or None, repr=True)
        self._register_field('datatype', (self._datatype,), repr=True)
        self._register_field('suffix', (self._datatype,), repr=True)
        self._register_field('extension', extensions, repr=True)
        self._register_field('equalize_evoked_count', ('', 'eq'), allow_empty=True)
        self._register_constant('common_brain', 'fsaverage')
        self._register_field('hemi', ('lh', 'rh'))

        self._register_field('mri', sorted(self._mri_subjects), allow_empty=True)
        self._register_field('group', self._groups.keys(), 'all', post_set_handler=self._post_set_group)

        # raw
        raw_default = sorted(self.raw)[0] if self.raw else None
        self._register_field('raw', sorted(self._raw), default=raw_default, repr=True)
        self._register_field('rej', self._artifact_rejection.keys(), self._artifact_rejection_default, allow_empty=True)

        # cov
        self._register_field('cov', sorted(self._covs))
        self._register_field('inv', default='free-3-dSPM', eval_handler=self._eval_inv)
        self._register_field('model', eval_handler=self._eval_model)
        self._register_field('test', sorted(self.tests), post_set_handler=self._post_set_test, allow_empty=self._empty_test, repr=False)
        self._register_field('parc', parc_values, 'aparc', eval_handler=self._eval_parc, allow_empty=True)
        self._register_field('freq', self._freqs.keys())
        self._register_field('src', default='ico-4', eval_handler=eval_src)
        self._register_field('adjacency', ('', 'link-midline'), allow_empty=True)

        # # slave fields
        self._register_field('mrisubject', depends_on=('mri', 'subject'), slave_handler=self._update_mrisubject, repr=False)

        # Initialize dependency tree
        self._init_derivative_registry()

        ########################################################################
        # Finalize
        ##########
        # log package versions
        from .. import __version__
        log.info("*** %s initialized with root %s on %s ***", self.__class__.__name__, root, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        level = logging.DEBUG if any('dev' in v for v in (__version__, mne.__version__)) else logging.INFO
        log.log(level, "Using eelbrain %s, mne %s.", __version__, mne.__version__)

        # register experimental features
        self._subclass_init()

        # set initial values
        self.set(**state)
        self._store_state()

        ########################################################################
        # Cached analysis defaults
        ##########################
        self._stim_channel = sequence_arg(f'{self.__class__.__name__}.stim_channel', self.stim_channel)
        self._raw_samplingrate = {}  # {(subject, recording): samplingrate}
        with self._temporary_state:
            for subject, session, task, acquisition, run in self.iter(('subject', 'session', 'task', 'acquisition', 'run'), group='all', raw='raw'):
                key = (subject, session, task, acquisition, run)
                raw_path = self._bids_path.fpath

                if not raw_path.exists():
                    continue

                pipe = self._raw[self.get('raw')]
                self._raw_samplingrate[key] = pipe._load_info(self._bids_path, self._raw).get('sfreq')

        # check for digitizer data differences
        # ====================================
        # Coordinate frames:
        # MEG (markers)  ==raw-file==>  head shape  ==trans-file==>  MRI
        #
        #  - raw files with identical head shapes can share trans-file (head-mri)
        #  - raw files with identical MEG markers (and head shape) can share
        #    forward solutions
        #  - SuperEpochs currently need to have a single forward solution,
        #    hence marker positions need to be the same between sub-epochs
            # if subjects_with_raw_changes:
            #     log.info("Raw input files new or changed, checking digitizer data")
            # for subject, session in subjects_with_raw_changes:
            #     # find unique digitizer datasets
            #     dev_head_t = None
            #     for task, acquisition, run in self.iter(('task', 'acquisition', 'run'), subject=subject, session=session):
            #         raw = self.load_raw(False)
            #         _dev_head_t = raw.info['dev_head_t']
            #         if _dev_head_t is None:
            #             raise FileDeficientError(f"The raw file {self._bids_path.basename} is missing dev_head_t information")
            #         if dev_head_t is None:
            #             dev_head_t = _dev_head_t
            #         if dev_head_t != _dev_head_t:
            #             raise FileDeficientError(f"Raw file {self._bids_path.basename} has dev_head_t that is different from other files.")

    def _restore_state(self, state=-1, discard_tip=True):
        StateModel._restore_state(self, state=state, discard_tip=discard_tip)
        self._update_bids_path()

    def _subclass_init(self):
        "Allow subclass to register experimental features"

    def _init_derivative_registry(self):
        self._derivative_state_fields = tuple(self._terminal_fields)
        self._derivatives = DerivativeRegistry(self.root, self._log)
        result_args = (
            self.tests,
            self._epochs,
            self._parcs,
            self._groups,
        )
        brain_report_args = (*result_args, self._mri_subjects, self.get('common_brain'), {**self._brain_plot_defaults, **self.brain_plot_defaults})

        # Register inputs
        for raw_name, pipe in self._raw.items():
            self._derivatives.register(RawBadChannelsInput(raw_name, pipe, self._raw))
            if isinstance(pipe, RawSource):
                self._derivatives.register(RawSourceInput(raw_name, pipe, self._raw))
            if isinstance(pipe, RawICA):
                self._derivatives.register(ICAInput(raw_name, pipe, self._raw, self._runs))
        self._derivatives.register(TransInput())
        self._derivatives.register(BemInput())
        self._derivatives.register(RejectionInput(self.root, self._artifact_rejection, self._epochs))

        # Register derivatives
        for raw_name, pipe in self._raw.items():
            if not isinstance(pipe, RawSource):
                self._derivatives.register(RawDerivative(raw_name, pipe, self._raw))
        self._derivatives.register(EventsDerivative(
            self.trigger_shift,
            sequence_arg(f'{self.__class__.__name__}.stim_channel', self.stim_channel),
            self.merge_triggers,
            self._variables,
            self._groups,
            self.preload,
            type(self).fix_events,
            type(self).label_events,
            self.__class__.__name__,
            len(self._tasks) > 1,
            len(self._sessions) > 1,
        ))
        self._derivatives.register(SelectedEventsDerivative(self._raw, self._epochs, self._artifact_rejection))
        self._derivatives.register(EpochsDerivative(self._raw, self._epochs))
        self._derivatives.register(EvokedDerivative(self._raw, self._epochs))
        self._derivatives.register(EvokedGroupDatasetDerivative(self._raw, self._groups))
        self._derivatives.register(EvokedTestDataDerivative(self.tests, self._epochs, self._groups))
        self._derivatives.register(TwoStageDataDerivative(self.tests, self._epochs, self._groups))
        self._derivatives.register(TwoStageLevel1Derivative(self.tests))
        self._derivatives.register(AnnotDerivative(self._parcs, tuple(self.get_field_values('hemi'))))
        self._derivatives.register(EpochsStcDerivative(self._raw, self._epochs))
        self._derivatives.register(EvokedStcDerivative(self._raw, self._epochs))
        self._derivatives.register(EpochsStcGroupDatasetDerivative(self._mri_subjects, self.get('common_brain'), self._groups))
        self._derivatives.register(EvokedStcGroupDatasetDerivative(self._mri_subjects, self.get('common_brain'), self._groups))
        self._derivatives.register(TestResultDerivative(*result_args))
        self._derivatives.register(TwoStageLevel2Derivative(*result_args))
        self._derivatives.register(SourceReportDerivative(*brain_report_args))
        self._derivatives.register(ROIReportDerivative(*brain_report_args))
        self._derivatives.register(EEGReportDerivative(*result_args))
        self._derivatives.register(EEGSensorsReportDerivative(*result_args))
        self._derivatives.register(LMReportDerivative(*brain_report_args))
        self._derivatives.register(CoregReportDerivative(self._raw))
        self._derivatives.register(MovieDerivative(*result_args))
        for cov_name, cov in self._covs.items():
            self._derivatives.register(CovDerivative(cov_name, cov))
        self._derivatives.register(SrcDerivative())
        self._derivatives.register(SourceMorphDerivative())
        self._derivatives.register(FwdDerivative())
        self._derivatives.register(InvDerivative())

    def _load_derivative(
            self,
            name: str,  # Registered derivative name.
            cache: bool | None = None,  # Explicit cache override for this load.
            state: dict[str, Any] | None = None,  # State overrides before resolving the derivative.
            options: dict[str, Any] | None = None,
            *,
            controls: frozenset[str] | set[str] | tuple[str, ...] = (),
    ):
        state_ = self._derivative_state(state)
        options_ = {} if options is None else dict(options)
        return self._derivatives.resolve(name, state=state_, options=options_, controls=controls).load(cache)

    def _derivative_state(
            self,
            state: dict[str, Any] | None = None,
            **extra_state,
    ) -> dict[str, Any]:
        merged_state = {}
        if state:
            merged_state.update(state)
        if extra_state:
            merged_state.update(extra_state)
        with self._temporary_state:
            explicit_none = {key: value for key, value in merged_state.items() if value is None}
            if merged_state:
                self.set(**{key: value for key, value in merged_state.items() if value is not None})
            out = {field: self.get(field) for field in self._derivative_state_fields}
            out.update(explicit_none)
            out['root'] = self.root
            out['common_brain'] = self.get('common_brain')
            return out

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        return self.iter()

    def get(
            self,
            temp: str,
            vmatch: bool = True,
            match: bool = True,
            mkdir: bool = False,
            **state,
    ):
        if not match:
            vmatch = False
        if mkdir:
            raise TypeError("Pipeline.get(..., mkdir=True) is no longer supported; create directories at the explicit path site")
        return StateModel.get(self, temp, vmatch=vmatch, **state)

    def _process_subject_arg(
            self,
            subjects: SubjectArg | None,
            kwargs: dict[str, str],
    ) -> tuple[str | None, str | None]:
        """Process subject arg for methods that work on groups and subjects

        Parameters
        ----------
        subjects
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        kwargs
            Additional state parameters to set.

        Returns
        -------
        subject : None | str
            Subject name if the value specifies a subject, None otherwise.
            One of ``subject`` and ``group`` will always be a ``str``,
            the other always ``None``.
        group : None | str
            Group name if the value specifies a group, None otherwise.
        """
        if subjects is None:  # default:
            subjects = -1 if 'group' in kwargs else 1

        if isinstance(subjects, int):
            if subjects == 1:
                return self.get('subject', **kwargs), None
            elif subjects == -1:
                return None, self.get('group', **kwargs)
            else:
                raise ValueError(f"{subjects=}")
        elif isinstance(subjects, str):
            if subjects in self.get_field_values('group'):
                if 'group' in kwargs:
                    if kwargs['group'] != subjects:
                        raise ValueError(f"group={kwargs['group']!r} inconsistent with subject={subjects!r}")
                    self.set(**kwargs)
                else:
                    self.set(group=subjects, **kwargs)
                return None, subjects
            else:
                return self.get('subject', subject=subjects, **kwargs), None
        else:
            raise TypeError(f"{subjects=}")

    def get_field_values(self, field, exclude=(), **state):
        """Find values for a field taking into account exclusion

        Parameters
        ----------
        field : str
            Field for which to find values.
        exclude : list of str
            Exclude these values.
        ...
            State parameters.
        """
        if state:
            self.set(**state)
        if isinstance(exclude, str):
            exclude = (exclude,)

        if field == 'mrisubject':
            subjects = StateModel.get_field_values(self, 'subject')
            mri_subjects = self._mri_subjects[self.get('mri')]
            mrisubjects = sorted(mri_subjects[s] for s in subjects)
            if exclude:
                mrisubjects = [s for s in mrisubjects if s not in exclude]
            common_brain = self.get('common_brain')
            if common_brain and (not exclude or common_brain not in exclude):
                mrisubjects.insert(0, common_brain)
            mrisubjects = ['sub-' + s for s in mrisubjects if (s != common_brain and not s.startswith('sub-'))]
            return mrisubjects
        else:
            return StateModel.get_field_values(self, field, exclude)

    def iter(self, fields='subject', exclude=None, values=None, progress_bar=None, **state):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : sequence | str
            Field(s) over which should be iterated.
        exclude : dict  {str: iterator over str}
            Exclude values from iteration (``{field: values_to_exclude}``).
        values : dict  {str: iterator over str}
            Fields with custom values to iterate over (instead of the
            corresponding field values) with {name: (sequence of values)}
            entries.
        progress_bar : str
            Message to show in the progress bar.
        ...
            State parameters.
        """
        return StateModel.iter(self, fields, exclude, values, progress_bar, **state)

    def iter_range(self, start=None, stop=None, field='subject'):
        """Iterate through a range on a field with ordered values.

        Parameters
        ----------
        start : None | str
            Start value (inclusive). With ``None``, begin at the first value.
        stop : None | str
            Stop value (inclusive). With ``None``, end with the last value.
        field : str
            Name of the field.

        Returns
        -------
        iterator over value : str
            Current field value.
        """
        values = self.get_field_values(field)
        if start is not None:
            start = values.index(start)
        if stop is not None:
            stop = values.index(stop) + 1
        values = values[start:stop]

        with self._temporary_state:
            for value in values:
                self._restore_state(discard_tip=False)
                self.set(**{field: value})
                yield value

    def fix_events(self, ds):
        """Modify event order or timing

        Parameters
        ----------
        ds : Dataset
            A Dataset containing events (with variables as returned by
            :func:`load.mne.events`).

        Returns
        -------
        ds : Dataset
            Should return the modified events dataset. Needs to contain at least
            the ``i_start`` and ``trigger`` columns.


        See Also
        --------
        label_events : Add event labels

        Notes
        -----
        Override this method in subclasses to change the event structure or
        timing. This method is called *before* adding other variables.

        The subject and session the events are from can be determined with
        ``ds.info['subject']`` and ``ds.info['session']``.

        Calling the original (super-class) method is not necessary.

        Examples
        --------
        Drop the last event from subject ``S01``::

            class Experiment(Pipeline):

                def fix_events(self, ds):
                    if ds.info['subject'] == 'S01':
                        return ds[:-1]
                    else:
                        return ds
        """
        return ds

    def label_events(self, ds):
        """Add event labels to events loaded from raw files

        Parameters
        ----------
        ds : Dataset
            A Dataset containing events (with variables as returned by
            :func:`load.mne.events`).

        Returns
        -------
        ds : Dataset
            Should return the modified events dataset.

        See Also
        --------
        fix_events : Change event order or timing

        Notes
        -----
        Override this method in subclasses to add event labels more flexibly
        than through the :attr:`variables` attribute. This method is applied
        *after* adding other variables.

        The subject and session the events are from can be determined with
        ``ds.info['subject']`` and ``ds.info['session']``.

        Calling the original (super-class) method is not necessary.

        Examples
        --------
        Add a label whenever trigger 2 follows trigger 1::

            class Experiment(Pipeline):

                def label_events(self, ds):
                    # assign 'no' to all events
                    ds[:, 'new'] = 'no'
                    # assign 'yes' to events where trigger 2 follows trigger 1
                    for i in range(1, ds.n_cases):
                        if ds[i, 'trigger'] == 2 and ds[i-1, 'trigger'] == 1:
                            ds[i, 'new'] = 'yes'
                    return ds

        Add events based on separate files. This assumes that the events in
        the recording only indicate trial onsets, and separate files contain
        events listed relative to these trial onsets::

            class Experiment(Pipeline):

                def label_events(self, ds):
                    samplingrate = ds.info['sfreq']
                    new_events = []
                    # loop through trials
                    for i_start, trigger in ds.zip('i_start', 'trigger'):
                        # load the event file, assuming that the trigger in the
                        # data was used to indicate the trial ID
                        trial_events = load.tsv(f'/files/trial_{trigger}.txt')
                        # assuming trial_events has a column called 'time' (in
                        # seconds), we infer the event's sample in the raw file
                        trial_i_start = i_start + trial_events['time'] * samplingrate
                        trial_events['i_start'] = Var(trial_i_start.astype(int))
                        # events also need a trigger column
                        trial_events[:, 'trigger'] = trigger
                        # collect all trials
                        new_events.append(trial_events)
                    # combine the trials to a single dataset
                    return combine(new_events)
        """
        return ds

    def label_subjects(self, ds):
        """Label the subjects in ds

        Creates a boolean :class:`Var` in ``ds`` for each group marking group
        membership.

        Parameters
        ----------
        ds : Dataset
            A Dataset with 'subject' entry.
        """
        subject = ds['subject']
        for name, subjects in self._groups.items():
            ds[name] = Var(subject.isin(subjects))

    def label_groups(self, subject, groups):
        """Generate Factor for group membership

        Parameters
        ----------
        subject : Factor
            A Factor with subjects.
        groups : list of str | {str: str} dict
            Groups which to label (raises an error if group membership is not
            unique). To use labels other than the group names themselves, use
            a ``{group: label}`` dict.

        Returns
        -------
        group : Factor
            A :class:`Factor` that labels the group for each subject.
        """
        return label_groups_var(subject, groups, self._groups)

    def load_annot(self, **state):
        """Load a parcellation (from an annot file)

        Returns
        -------
        labels : list of Label
            Labels in the parcellation (output of
            :func:`mne.read_labels_from_annot`).
        ...
            State parameters.
        """
        return self._load_derivative('annot', state=state)

    def load_bad_channels(self, noise: bool = False, **kwargs):
        """Load bad channels

        Parameters
        ----------
        noise
            Load bad channels for empty-room noise recording instead of the subject recording.
        ...
            State parameters.

        Returns
        -------
        bad_chs : list of str
            Bad channels.
        """
        raw_name = self.get('raw', **kwargs)
        return self._load_derivative(raw_bad_channels_input_name(raw_name), state={**kwargs, 'raw': raw_name}, options={'noise': noise})

    def load_cov(self, **kwargs):
        """Load the covariance matrix

        Parameters
        ----------
        ...
            State parameters.
        """
        cov_name = self.get('cov', **kwargs)
        return self._load_derivative(cov_node_name(cov_name), state={**kwargs, 'cov': cov_name})

    @suppress_mne_warning
    def load_epochs(
            self,
            subjects: SubjectArg = None,
            baseline: BaselineArg = False,
            ndvar: bool | Literal['both'] = True,
            add_bads: bool | list = True,
            reject: bool | Literal['keep'] = True,
            cat: Sequence[CellArg] = None,
            samplingrate: int = None,
            decim: int = None,
            pad: float = 0,
            data_raw: bool = False,
            data: str = 'sensor',
            trigger_shift: bool = True,
            tmin: float = None,
            tmax: float = None,
            tstop: float = None,
            interpolate_bads: Literal[True, False, 'keep'] = False,
            **state,
    ) -> Dataset:
        """
        Load a :class:`Dataset` with epochs for a given epoch definition

        Parameters
        ----------
        subjects
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        baseline
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to not apply baseline
            correction.
        ndvar
            Convert epochs to :class:`NDVar` (using keys ``'meg'`` for MEG data and
            ``'eeg'`` for EEG data in the returned :class:`Dataset`).
            With ``ndvar=False``, include :class:`mne.Epochs` with key ``'epochs'``.
            Use ``'both'`` to include both NDVar and :class:`mne.Epochs`.
        add_bads
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the bad channels file. Alternatively,
            a list of bad channels can be specified.
        reject
            Reject bad trials. If ``True`` (default), bad trials are removed
            from the Dataset. Set to ``False`` to ignore the trial rejection.
            Set ``reject='keep'`` to load the rejection (added it to the events
            as ``'accept'`` variable), but keep bad trails.
        cat
            Only load data for these cells (cells of model).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        pad : scalar
            Pad the epochs with this much time (in seconds; e.g. for spectral
            analysis).
        data_raw
            Keep the :class:`mne.io.Raw` instance in ``ds.info['raw']``
            (default False).
        data
            Data to load; 'sensor' to load all sensor data (default);
            'sensor.rms' to return RMS over sensors. Only applies to NDVar
            output.
        trigger_shift
            Apply post-baseline trigger-shift if it applies to the epoch
            (default True).
        tmin
            Override the epoch's ``tmin`` parameter.
        tmax
            Override the epoch's ``tmax`` parameter.
        tstop
            Override the epoch's ``tmax`` parameter as exclusive ``tstop``.
        interpolate_bads
            Interpolate channels marked as bad for the whole recording (useful
            when comparing topographies across subjects; default False).
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-rej`: which trials to use

        """
        data = TestDims.coerce(data)
        if not data.sensor:
            raise ValueError(f"data={data.string!r}; load_evoked is for loading sensor data")
        if data.sensor is not True:
            if not ndvar:
                raise ValueError(f"data={data.string!r} with ndvar=False")
            if interpolate_bads:
                raise ValueError(f"{interpolate_bads=} with data={data.string!r}")
        if isinstance(ndvar, str) and ndvar != 'both':
            raise ValueError(f"{ndvar=}")

        subject, group = self._process_subject_arg(subjects, state)
        options = {
            'baseline': baseline,
            'ndvar': ndvar,
            'add_bads': add_bads,
            'reject': reject,
            'cat': cat,
            'samplingrate': samplingrate,
            'decim': decim,
            'pad': pad,
            'data_raw': data_raw,
            'data': data,
            'trigger_shift': trigger_shift,
            'tmin': tmin,
            'tmax': tmax,
            'tstop': tstop,
            'interpolate_bads': interpolate_bads,
        }
        if group is None:
            if subject is not None:
                state['subject'] = subject
            return self._load_derivative('epochs', state=state, options=options)

        epoch_name = self.get('epoch')
        dss = [
            self._load_derivative('epochs', options=options)
            for _ in self.iter(group=group, progress_bar=f"Load {epoch_name}")
        ]
        return combine(dss)

    def load_epochs_stc(
            self,
            subjects: str | int = None,
            baseline: BaselineArg = True,
            src_baseline: BaselineArg = False,
            cat: Sequence[CellArg] = None,
            keep_epochs: bool | str = False,
            morph: bool = None,
            data_raw: bool = False,
            samplingrate: int = None,
            decim: int = None,
            pad: float = 0,
            ndvar: bool = True,
            reject: bool | str = True,
            **state):
        """Load a Dataset with stcs for single epochs

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
            Warning: loading single trial data for multiple subjects at once
            uses a lot of memory, which can lead to a periodically unresponsive
            terminal).
        baseline
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        cat
            Only load data for these cells (cells of model).
        keep_epochs : bool | 'ndvar' | 'both'
            Keep the sensor space data in the Dataset that is returned (default
            False; True to keep :class:`mne.Epochs` object; ``'ndvar'`` to keep
            :class:`NDVar`; ``'both'`` to keep both).
        morph
            Morph the source estimates to the common brain
            (default ``False``, except when loading multiple subjects and ``ndvar=True``).
        data_raw
            Keep the :class:`mne.io.Raw` instance in ``ds.info['raw']``
            (default False).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        pad
            Pad the epoch's data by extending ``tmin`` and ``tmax`` (specify
            ``pad`` time in seconds).
        ndvar
            Add the source estimates as :class:`NDVar` named "src" instead of a list of
            :class:`mne.SourceEstimate` objects named "stc" (default True).
        reject : bool | 'keep'
            Reject bad trials. If ``True`` (default), bad trials are removed
            from the Dataset. Set to ``False`` to ignore the trial rejection.
            Set ``reject='keep'`` to load the rejection (added it to the events
            as ``'accept'`` variable), but keep bad trails.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-rej`: which trials to use
             - :ref:`state-cov`: covariance matrix for inverse solution
             - :ref:`state-src`: source space
             - :ref:`state-inv`: inverse solution

        Returns
        -------
        epochs_dataset : Dataset
            Dataset containing single trial data (epochs).
        """
        self._current_source_parc(**state)
        subject, group = self._process_subject_arg(subjects, state)
        options = {
            'baseline': baseline,
            'src_baseline': src_baseline,
            'cat': cat,
            'keep_epochs': keep_epochs,
            'morph': morph,
            'data_raw': data_raw,
            'samplingrate': samplingrate,
            'decim': decim,
            'pad': pad,
            'ndvar': ndvar,
            'reject': reject,
        }
        if group is not None:
            return self._load_derivative('epochs-stc-group-dataset', options=options)
        else:
            return self._load_derivative('epochs-stc', options=options)

    def load_events(
            self,
            subject: str = None,
            add_bads: bool | list[str] = True,
            data_raw: bool = False,
            **kwargs,
    ) -> Dataset:
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject
            Subject for which to load events (default is the current subject
            in the experiment's state).
        add_bads
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the bad channels file. Alternatively,
            a list of bad channels can be specified.
        data_raw
            Keep the :class:`mne.io.Raw` instance in ``ds.info['raw']``
            (default False).
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window

        """
        state = dict(kwargs)
        if subject is not None:
            state['subject'] = subject
        ds = self._load_derivative('events', state=state)
        if data_raw:
            ds.info['raw'] = self.load_raw(add_bads, preload=self.preload, **state)
        return ds

    def load_evoked(
            self,
            subjects: str | int = None,
            baseline: BaselineArg = False,
            ndvar: bool | int = True,
            cat: Sequence[CellArg] = None,
            samplingrate: int = None,
            decim: int = None,
            data_raw: bool = False,
            data: DataArg = 'sensor',
            **state):
        """
        Load a Dataset with condition average responses for each subject.

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        baseline
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to not apply baseline
            correction.
        ndvar : bool | 2
            Convert the :class:`mne.Evoked` objects to an :class:`NDVar` (the
            name in the Dataset is ``'meg'`` or ``'eeg'``). With
            ``ndvar=False``, the :class:`mne.Evoked` objects are added as
            ``'evoked'``. ``2`` to add both.
        cat
            Only load data for these cells (cells of model).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        data_raw
            Keep the :class:`mne.io.Raw` instance in ``ds.info['raw']``
            (default False).
        data
            Data to load; 'sensor' to load all sensor data (default);
            'sensor.rms' to return RMS over sensors. Only applies to NDVar
            output.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-rej`: which trials to use
             - :ref:`state-model`: how to group trials into conditions
             - :ref:`state-equalize_evoked_count`: control number of trials per cell

        Notes
        -----
        Channel interpolation: Bad channels are always interpolated. When
        loading data for a single subject, bad channels are marked as
        bad/excluded. When loading group level data, datasets are merged using
        interpolated data.
        """
        data = TestDims.coerce(data)
        if not data.sensor:
            raise ValueError(f"data={data.string!r}; load_evoked is for loading sensor data")
        elif data.sensor is not True and not ndvar:
            raise ValueError(f"data={data.string!r} with ndvar=False")
        subject, group = self._process_subject_arg(subjects, state)
        epoch_name = self.get('epoch')
        epoch = self._epochs[epoch_name]
        if baseline is True:
            baseline = epoch.baseline
        options = {
            'baseline': baseline,
            'ndvar': ndvar,
            'cat': cat,
            'samplingrate': samplingrate,
            'decim': decim,
            'data_raw': data_raw,
            'data': data,
        }
        if group is not None:
            state['group'] = group
            return self._load_derivative('evoked-group-dataset', state=state, options=options)
        state['group'] = None
        if subject is not None:
            state['subject'] = subject
        return self._load_derivative('evoked', state=state, options=options)

    def load_epochs_stf(
            self,
            subjects: str | int = None,
            baseline: BaselineArg = True,
            morph: bool = None,
            keep_stc: bool = False,
            **state):
        """Load frequency space single trial data

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        baseline
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        morph
            Morph the source estimates to the common brain
            (default ``False``, except when loading multiple subjects and ``ndvar=True``).
        keep_stc
            Keep the source timecourse data in the Dataset that is returned
            (default False).
        ...
            State parameters.
        """
        ds = self.load_epochs_stc(subjects, baseline, ndvar=True, morph=morph, **state)
        name = 'srcm' if 'srcm' in ds else 'src'

        # apply morlet transformation
        freq_params = self._freqs[self.get('freq')]
        freq_range = freq_params['frequencies']
        ds['stf'] = cwt_morlet(ds[name], freq_range, use_fft=True, n_cycles=freq_params['n_cycles'], output='complex')

        if not keep_stc:
            del ds[name]

        return ds

    def load_evoked_stf(
            self,
            subjects: str | int = None,
            baseline: BaselineArg = True,
            morph: bool = None,
            keep_stc: bool = False,
            **state):
        """Load frequency space evoked data

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        baseline
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        morph
            Morph the source estimates to the common brain
            (default ``False``, except when loading multiple subjects and ``ndvar=True``).
        keep_stc
            Keep the source timecourse data in the Dataset that is returned
            (default False).
        ...
            State parameters.
        """
        ds = self.load_evoked_stc(subjects, baseline, morph=morph, **state)
        name = 'srcm' if 'srcm' in ds else 'src'

        # apply morlet transformation
        freq_params = self._freqs[self.get('freq')]
        freq_range = freq_params['frequencies']
        ds['stf'] = cwt_morlet(ds[name], freq_range, use_fft=True, n_cycles=freq_params['n_cycles'], zero_mean=False, out='magnitude')

        if not keep_stc:
            del ds[name]

        return ds

    def load_evoked_stc(
            self,
            subjects: str | int = None,
            baseline: BaselineArg = True,
            src_baseline: BaselineArg = False,
            cat: Sequence[CellArg] = None,
            keep_evoked: bool = False,
            morph: bool = None,
            data_raw: bool = False,
            samplingrate: int = None,
            decim: int = None,
            ndvar: bool = True,
            **state):
        """Load evoked source estimates.

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        baseline
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        cat
            Only load data for these cells (cells of model).
        keep_evoked
            Keep the sensor space data in the Dataset that is returned (default
            False).
        morph
            Morph the source estimates to the common brain
            (default ``False``, except when loading multiple subjects and ``ndvar=True``).
        data_raw
            Keep the :class:`mne.io.Raw` instance in ``ds.info['raw']``
            (default False).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        ndvar
            Add the source estimates as NDVar named "src" instead of a list of
            :class:`mne.SourceEstimate` objects named "stc" (default True).
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-rej`: which trials to use
             - :ref:`state-model`: how to group trials into conditions
             - :ref:`state-equalize_evoked_count`: control number of trials per cell
             - :ref:`state-cov`: covariance matrix for inverse solution
             - :ref:`state-src`: source space
             - :ref:`state-inv`: inverse solution

        """
        self._current_source_parc(**state)
        subject, group = self._process_subject_arg(subjects, state)
        options = {
            'baseline': baseline,
            'src_baseline': src_baseline,
            'cat': cat,
            'keep_evoked': keep_evoked,
            'morph': morph,
            'data_raw': data_raw,
            'samplingrate': samplingrate,
            'decim': decim,
            'ndvar': ndvar,
        }
        if group is not None:
            state['group'] = group
            return self._load_derivative('evoked-stc-group-dataset', state=state, options=options)
        state['group'] = None
        if subject is not None:
            state['subject'] = subject
        return self._load_derivative('evoked-stc', state=state, options=options)

    def load_induced_stc(
            self,
            subjects: str | int = None,
            frequencies: float | Sequence[float] = None,
            n_cycles: float | Sequence[float] = None,
            pad: float = 0.250,
            baseline: BaselineArg = True,
            cat: Sequence[CellArg] = None,
            morph: bool = False,
            decim: int = 1,
            **state,
    ) -> Dataset:
        """Morlet wavelet induced power and phase in source space.

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        frequencies
            Frequencies for which to compute induced activity.
        n_cycles
            Number of cycles in each wavelet. Fixed number or one per frequency.
        pad
            Pad the epochs data to avoid edge effects in wavelet representation
            (specified in seconds; default 0.250).
        baseline
            Baseline for the epochs, ``True`` to use the epoch's baseline
            specification (default).
        cat
            Only load data for these cells (cells of model).
        morph
            Morph the source estimates to the common_brain (default False).
        decim
            Decimate time-frequency representation (cumulative with epoch
            decimation factor).
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-rej`: which trials to use
             - :ref:`state-model`: how to group trials into conditions
             - :ref:`state-equalize_evoked_count`: control number of trials per cell
             - :ref:`state-cov`: covariance matrix for inverse solution
             - :ref:`state-src`: source space
             - :ref:`state-inv`: inverse solution
        """
        self._current_source_parc(**state)
        subject, group = self._process_subject_arg(subjects, state)
        if frequencies is None:
            frequencies = np.logspace(2, 5, 10, base=2)
        elif not np.isscalar(frequencies):
            frequencies = np.asarray(frequencies)

        if n_cycles is None:
            n_cycles = frequencies / 3
        elif not np.isscalar(n_cycles):
            n_cycles = np.asarray(n_cycles)

        epoch_name = self.get('epoch')
        epoch = self._epochs[epoch_name]
        if group is not None:
            dss = []
            for _ in self.iter(group=group, progress_bar=f"Load induced {epoch_name}"):
                ds = self.load_induced_stc(None, frequencies, n_cycles, pad, baseline, cat, morph, decim)
                dss.append(ds)
            return combine(dss)

        # 1 subject
        ds = self.load_epochs_stc(1, baseline, False, cat, morph=morph, pad=pad)
        # conditions
        model = self.get('model') or None
        stc = ds['srcm' if morph else 'src']
        cwt = cwt_morlet(stc, frequencies, False, n_cycles, True, 'complex', decim)
        if pad:
            cwt = cwt.sub(time=(epoch.tmin, epoch.tmax + cwt.time.tstep / 10))
        cwt.x = (cwt.x * cwt.x.conj()).real
        ds['power'] = cwt
        return ds.aggregate(model, drop_bad=True)

    def load_fwd(
            self,
            surf_ori: bool = True,
            ndvar: bool = False,
            **state):
        """Load the forward solution

        Parameters
        ----------
        surf_ori
            Force surface orientation (default True; only applies if
            ``ndvar=False``, :class:`NDVar` forward operators are alsways
            surface based).
        ndvar
            Return forward solution as :class:`NDVar` (default is
            :class:`mne.forward.Forward`).
        ...
            State parameters.

        Returns
        -------
        forward_operator : mne.forward.Forward | NDVar
            Forward operator.
        """
        with self._temporary_state:
            if state:
                self.set(**state)
            state_ = self._derivative_state()
            fwd = self._load_derivative('fwd')
            fwd_file = self._derivatives.resolve('fwd', state=state_).artifact_path
            src = self.get('src')
            if ndvar:
                parc = self._current_source_parc()
                if parc:
                    self.make_annot()
                fwd = load.mne.forward_operator(fwd_file, src, mri_sdir(state_), parc, adjacency=False)
                if parc:
                    fwd = _drop_unknown_labels(fwd)
                return fwd
            if surf_ori:
                mne.convert_forward_solution(fwd, surf_ori, copy=False)
            return fwd

    def load_ica(
            self,
            accept_stale: bool = False,
            **state,
    ) -> mne.preprocessing.ICA:
        """Load the mne-python ICA object

        Parameters
        ----------
        accept_stale
            Accept an existing ICA file even when Eelbrain can not confirm
            that it was created from the current data and ICA settings, for
            example after changing the raw preprocessing used to estimate the
            ICA. This rewrites the bookkeeping for that file instead of
            raising :class:`ProtectedArtifactError`. Use this only when you
            intentionally want to keep the existing file on your own
            responsibility instead of reverting those changes or recomputing
            the ICA. When Eelbrain detects a mismatch, the error message names
            the raw step and setting that changed so you can decide whether to
            revert that change.
        ...
            State parameters.

        Returns
        -------
        ICA object for the current :ref:`state-raw` setting.
        """
        raw_name = get_ica_pipe_name(self._raw, self.get('raw', **state))
        return self._derivatives.resolve(
            ica_input_name(raw_name),
            state=self._derivative_state({**state, 'raw': raw_name}),
            controls={REINDEX_ICA} if accept_stale else (),
        ).load()

    def load_inv(
            self,
            fiff: Any = None,
            ndvar: bool = False,
            **state,
    ) -> mne.minimum_norm.InverseOperator | NDVar:
        """Load the inverse operator

        Parameters
        ----------
        fiff : Raw | Epochs | Evoked | ...
            Object which provides the mne info dictionary (default: load the
            raw file).
        ndvar
            Return the inverse operator as NDVar (default is
            :class:`mne.minimum_norm.InverseOperator`). The NDVar representation
            does not take into account any direction selectivity (loose/free
            orientation) or noise normalization properties.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-rej`: which trials to use
             - :ref:`state-cov`: covariance matrix for inverse solution
             - :ref:`state-src`: source space
             - :ref:`state-inv`: inverse solution

        """
        with self._temporary_state:
            if state:
                self.set(**state)
            inv = self._load_derivative('inv', cache=self.cache_inv, options={'fiff': fiff})

            if ndvar:
                parc = self._current_source_parc()
                inv = load.mne.inverse_operator(inv, self.get('src'), mri_sdir(self._derivative_state()), parc)
                if parc:
                    inv = _drop_unknown_labels(inv)
            return inv

    def _prepare_inv(
            self,
            fiff: Any,
            morph: bool,
    ):
        # load inv
        parc = self._current_source_parc()
        # make sure annotation exists
        if parc:
            self.make_annot()

        inv = self.load_inv(fiff)

        # determine whether initial source-space can be restricted
        subjects_dir = str(mri_sdir(self._derivative_state()))
        mrisubject = self.get('mrisubject')
        is_scaled = find_source_subject(mrisubject, subjects_dir)
        if parc and (is_scaled or not morph):
            label = label_from_annot(inv['src'], mrisubject, subjects_dir, parc)
        else:
            label = None

        return inv, label, subjects_dir, mrisubject, is_scaled, parc

    def load_label(
            self,
            label: str,
            **kwargs,
    ) -> mne.Label | mne.BiHemiLabel:
        """Retrieve a label as mne Label object

        Parameters
        ----------
        label : str
            Name of the label. If the label name does not end in '-bh' or '-rh'
            the combination of the labels ``label + '-lh'`` and
            ``label + '-rh'`` is returned.
        ...
            State parameters.
        """
        labels = self._load_labels(label, **kwargs)
        if label in labels:
            return labels[label]
        elif not label.endswith(('-lh', '-rh')):
            return labels[label + '-lh'] + labels[label + '-rh']
        else:
            raise ValueError(f"Label {label!r} could not be found in parc {self.get('parc')!r}.")

    def _load_labels(self, regexp=None, **kwargs):
        """Load labels from an annotation file."""
        self.make_annot(**kwargs)
        subjects_dir = str(mri_sdir(self._derivative_state()))
        labels = mne.read_labels_from_annot(self.get('mrisubject'), self.get('parc'), regexp=regexp, subjects_dir=subjects_dir)
        return {label.name: label for label in labels}

    def load_source_morph(self, **state):
        """Load the source morph from mrisubject to common_brain

        Parameters
        ----------
        ...
            State parameters.

        Notes
        -----
        For scaled template brains, no geometric morphing is required for the
        internal NDVar code paths: :func:`eelbrain.morph_source_space` handles
        that case directly from source-space metadata. In that situation this
        method still returns a trivial identity :class:`mne.SourceMorph` for
        compatibility with public STC-based workflows.
        """
        return self._load_derivative('source-morph', state=state)

    def load_neighbor_correlation(
            self,
            subjects: SubjectArg = None,
            epoch: str = None,
            add_bads: bool = True,
            return_data: bool = False,
            **state,
    ) -> NDVar | Dataset | tuple[NDVar, NDVar]:
        """Load sensor neighbor correlation

        Parameters
        ----------
        subjects
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        epoch
            Epoch to use for computing neighbor-correlation (by default, the
            whole task is used).
        add_bads
            Reject bad channels first.
        return_data
            Return the data from which the correlation is calculated. Only
            possible when loading neighbor-correlation for a single subject.

        Returns
        -------
        data : NDVar
            Data from which the correlation is calculated (only retuned with
            ``return_data=True``).
        nc : NDVar | Dataset
            Sensor neighbor-correlation as :class:`NDVar` for a single subject
            or as :class:`Dataset` for multiple subjects.
        """
        subject, group = self._process_subject_arg(subjects, state)
        if group is not None:
            if return_data:
                raise ValueError(f"{return_data=} when loading data for group")
            if state:
                self.set(**state)
            lines = [(subject, self.load_neighbor_correlation(1, epoch, add_bads)) for subject in self]
            return Dataset.from_caselist(['subject', 'nc'], lines)
        if epoch:
            if epoch is True:
                epoch = self.get('epoch')
            epoch_params = self._epochs[epoch]
            if len(epoch_params.tasks) != 1:
                raise ValueError(f"{epoch=}: epoch has multiple tasks")
            ds = self.load_epochs(add_bads=add_bads, epoch=epoch, reject=False, decim=1, **state)
            key = ds.info['sensor_types'][0]
            data = concatenate(ds[key])
        else:
            data = self.load_raw(ndvar=True, add_bads=add_bads, **state)
        n_corr = neighbor_correlation(data)
        if return_data:
            return data, n_corr
        else:
            return n_corr

    def load_raw(
            self,
            add_bads: bool | Sequence[str] = True,
            preload: bool = False,
            ndvar: bool = False,
            samplingrate: int = None,
            decim: int = None,
            tstart: float = None,
            tstop: float = None,
            noise: bool = False,
            **kwargs,
    ) -> mne.io.Raw | NDVar:
        """
        Load a raw file as mne Raw object.

        Parameters
        ----------
        add_bads
            Add bad channel information to the bad channels text file (default
            ``True``).
        preload
            Load raw data into memory (default ``False``; see
            :func:`mne.io.read_raw_fif` parameter).
        ndvar
            Load as NDVar instead of mne Raw object (default ``False``).
        samplingrate
            Samplingrate in Hz for the analysis.
        decim
            Decimate data (default 1, i.e. no decimation; value other than 1
            implies ``preload=True``)
        tstart
            Crop the raw data. After cropping the time axis will be reset, i.e.,
            the ``tstart`` will be set to ``t = 0``.
        tstop
            Crop the raw data.
        noise
            Load corresponding empty-room data instead of current subject's task data (default ``False``).
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
        """
        raw_name = self.get('raw', **kwargs)
        raw = self._load_derivative(raw_node_name(raw_name), state={**kwargs, 'raw': raw_name}, options={'add_bads': add_bads, 'preload': preload, 'noise': noise})
        if decim and decim > 1:
            assert samplingrate is None, "samplingrate and decim can't both be specified"
            samplingrate = int(round(raw.info['sfreq'] / decim))
        if tstart or tstop:
            raw = raw.crop(tstart or 0, tstop, False)
        if samplingrate or preload:
            raw.load_data()
        if samplingrate:
            raw.resample(samplingrate)

        if ndvar:
            pipe = self._raw[raw_name]
            data = TestDims('sensor')
            data_kind = data.data_to_ndvar(raw.info)[0]
            sysname = pipe._get_sysname(raw.info, self.get('subject'), data_kind, self._raw)
            adjacency = pipe._get_adjacency(data_kind, self._raw)
            raw = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=adjacency)

        return raw

    def _current_source_parc(self, **state):
        return _source_parc({
            'src': self.get('src', **state),
            'parc': self.get('parc', **state),
        })

    def load_raw_stc(
            self,
            morph: bool = False,
            ndvar: bool = True,
            samplingrate: int = None,
            tstart: float = None,
            tstop: float = None,
            **kwargs,
    ) -> mne.SourceEstimate | mne.VectorSourceEstimate | mne.VolSourceEstimate | NDVar:
        """
        Load a raw file as mne Raw object.

        Parameters
        ----------
        morph
            Morph the source estimates to the common_brain (default False).
        ndvar
            Load as NDVar instead of mne Raw object (default ``True``).
        samplingrate
            Samplingrate in Hz for the analysis.
        tstart
            Crop the raw data. After cropping the time axis will be reset, i.e.,
            the ``tstart`` will be set to ``t = 0``.
        tstop
            Crop the raw data.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-session`: from which session to load raw data
             - :ref:`state-raw`: preprocessing pipeline
        """
        raw = self.load_raw(samplingrate=samplingrate, tstart=tstart, tstop=tstop, **kwargs)
        inv, label, mri_sdir, mrisubject, is_scaled, parc = self._prepare_inv(raw, morph)
        solution = InverseSolution._coerce(self.get('inv'))
        stc = apply_inverse_raw(raw, inv, label=label, **solution._apply_kw)

        if ndvar:
            src = self.get('src')
            return solution._to_ndvar(stc, mrisubject, src, mri_sdir, parc=parc, adjacency=self.get('adjacency'))
        else:
            return stc

    def load_selected_events(
            self,
            subjects: SubjectArg = None,
            reject: bool | Literal['keep'] = True,
            add_bads: bool | list[str] = True,
            index: bool | str = True,
            data_raw: bool = False,
            vardef: str = None,
            cat: Sequence[CellArg] = None,
            **kwargs,
    ) -> Dataset:
        """
        Load events and return a subset based on epoch and rejection

        Parameters
        ----------
        subjects
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        reject
            Reject bad trials. If ``True`` (default), bad trials are removed
            from the Dataset. Set to ``False`` to ignore the trial rejection.
            Set ``reject='keep'`` to load the rejection (added it to the events
            as ``'accept'`` variable), but keep bad trails.
        add_bads
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the bad channels file. Alternatively,
            a list of bad channels can be specified.
        index
            Index the Dataset before rejection (provide index name as str).
        data_raw
            Keep the :class:`mne.io.Raw` instance in ``ds.info['raw']``
            (default False).
        vardef
            Name of a test defining additional variables to add to the returned
            Dataset.
        cat
            Only load data for these cells (cells of the current ``model``).
        ...
            State parameters.

        Notes
        -----
        When trial rejection is set to automatic, not rejection is performed
        because no epochs are loaded.
        """
        if reject not in (True, False, 'keep'):
            raise ValueError(f"{reject=}")
        if index is True:
            index = 'index'
        elif index and not isinstance(index, str):
            raise TypeError(f"{index=}")
        state = dict(kwargs)
        subject, group = self._process_subject_arg(subjects, state)

        if group is not None:
            if data_raw:
                raise ValueError(f"{data_raw=}: can't keep raw when combining subjects")
            return combine([self.load_selected_events(subjects=subject_, reject=reject, index=index, vardef=vardef, cat=cat, **state) for subject_ in self.iter(group=group)])
        elif subject is None:
            raise RuntimeError(f"{subject=}, {group=}")

        options = {
            'reject': reject,
            'add_bads': add_bads,
            'index': index,
            'data_raw': data_raw,
            'cat': cat,
        }
        ds = self._load_derivative(SELECTED_EVENTS, options=options)
        apply_vardef(ds, vardef, self.tests, self._groups)
        return ds

    def load_src(
            self,
            add_geom: bool = False,
            ndvar: bool = False,
            **state,
    ) -> mne.SourceSpaces | SourceSpace | VolumeSourceSpace:
        """Load the current source space

        Parameters
        ----------
        add_geom
            Parameter for :func:`mne.read_source_spaces`.
        ndvar
            Return as NDVar Dimension object (default False).
        ...
            State parameters.

        Examples
        --------
        Plot a volume source space with :mod:`mayavi`::

            from mayavi import mlab

            src = e.load_src(mrisubject='fsaverage', src='vol-7', ndvar=True)
            mlab.points3d(*src.coordinates.T)
            mlab.show()
        """
        src_spaces = self._load_derivative('src', state=state)
        if ndvar:
            with self._temporary_state:
                src = self.get('src', **state)
                subjects_dir = str(mri_sdir(self._derivative_state()))
                mri_subject = self.get('mrisubject')
                if src.startswith('vol'):
                    return VolumeSourceSpace.from_file(subjects_dir, mri_subject, src)
                parc = self.get('parc')
                return SourceSpace.from_file(subjects_dir, mri_subject, src, parc)
        if add_geom:
            return mne.read_source_spaces(src_file_path(self._derivative_state(state)), add_geom)
        return src_spaces

    def load_test(
            self,
            test: str,
            tstart: float = None,
            tstop: float = None,
            pmin: PMinArg = None,
            disconnect_labels: bool = False,
            samples: int = 10000,
            data: str = 'source',
            baseline: BaselineArg = True,
            smooth: float = None,
            src_baseline: BaselineArg = None,
            samplingrate: int = None,
            return_data: bool = False,
            make: bool = False,
            **state,
    ):
        """Create and load spatio-temporal cluster test results

        Parameters
        ----------
        test
            Test for which to create a report (entry in Pipeline.tests.
        tstart
            Beginning of the time window for the test in seconds
            (default is the beginning of the epoch).
        tstop
            End of the time window for the test in seconds
            (default is the end of the epoch).
        pmin
            Kind of test.
        disconnect_labels
            For ``data='source'``, disconnect cluster adjacency across labels in
            the current ``parc`` state. The default is to run one masked
            whole-brain source test.
        samples
            Number of random permutations of the data used to determine cluster
            *p*-values (default 10'000). If the test is already cached with a
            number ≥ ``samples`` the cached version is returned, otherwise the
            test is recomputed.
        data
            Data to test, for example:

            - ``'source'`` spatio-temporal test in source space.
            - ``'sensor'`` spatio-temporal test in sensor space (MEG).
            - ``'eeg'`` spatio-temporal test in EEG sensor space.
            - ``'source.mean'`` ROI mean time course.
            - ``'sensor.rms'`` RMS across sensors.

        baseline
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        smooth
            Smooth data in space before test (value in [m] STD of Gaussian).
        src_baseline
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        return_data
            Return the data along with the test result (see below).

            .. Warning::
                Single trial data (i.e., two-stage tests) take up a lot of
                memory and it might not be possible to load all data at once.
                Instead, loop through subjects and collect summary statistics.

        make
            If the target file does not exist, create it (could take a long
            time depending on the test; if False, raise an IOError).
        ...
            State parameters (Use the ``group`` state parameter to select the
            subject group for which to perform the test).

        Returns
        -------
        ds : Dataset | dict (if return_data==True)
            Data that forms the basis of the test (for ROI tests, a
            ``{roi: dataset}`` dictionary).
        res : NDTest | ROITestResult
            Test result for the specified test (for ROIs tests,
            an :class:`~_experiment.ROITestResult` object).
        """
        self.set(test=test, **state)
        data = TestDims.coerce(data, morph=True)
        if data.source:
            self._current_source_parc(**state)
        data._testnd_parc(disconnect_labels)
        options = {
            'data': data,
            'samples': samples,
            'test': test,
            'tstart': tstart,
            'tstop': tstop,
            'pmin': pmin,
            'disconnect_labels': disconnect_labels,
            'baseline': baseline,
            'src_baseline': src_baseline,
            'smooth': smooth,
            'samplingrate': samplingrate,
        }
        test_obj = self.tests[test]
        result_node = 'two-stage-level-2' if isinstance(test_obj, TwoStageTest) else 'test-result'
        data_node = 'two-stage-data' if isinstance(test_obj, TwoStageTest) else 'evoked-test-data'
        result_state = self._derivative_state()
        handle = self._derivatives.resolve(result_node, state=result_state, options=options)
        dst = handle.artifact_path
        desc = self._derivatives.describe_artifact_path(dst)

        if handle.is_valid():
            try:
                res = handle.load()
            except OldVersionError:
                res = None
            else:
                if not return_data:
                    return res
        elif not make and dst.exists():
            raise OSError(f"The requested test is outdated: {desc}. Set make=True to perform the test.")
        else:
            res = None

        if res is None and not make:
            raise OSError(f"The requested test is not cached: {desc}. Set make=True to perform the test.")
        if res is None:
            res = handle.load()
            if not return_data:
                return res

        data_options = {key: value for key, value in options.items() if key != 'disconnect_labels'}
        data_state = self._derivative_state()
        res_data = self._derivatives.resolve(data_node, state=data_state, options=data_options).load()
        if isinstance(res_data, ROIData):
            res_data = res_data.label_data
        return res_data, res

    def make_annot(self, **state):
        """Ensure that annot files for the current parcellation exist."""
        self._load_derivative('annot', state=state)

    def make_bad_channels(
        self,
        bad_chs: tuple[str] | str | int = (),
        redo: bool = False,
        noise: bool = False,
        **kwargs: Any,
    ) -> None:
        """Write the bad channel definition file for a raw file

        If the file already exists, new bad channels are added to the old ones.
        In order to replace the old file with only the new values, set
        ``redo=True``.

        Parameters
        ----------
        bad_chs
            Names of the channels to set as bad. Numerical entries are
            interpreted as "MEG XXX". If bad_chs contains entries not present
            in the raw data, a ValueError is raised.
        redo
            If the file already exists, replace it (instead of adding).
        noise
            If True, make bad channels for the empty-room recording instead of the current subject's recording.
        ...
            State parameters.

        See Also
        --------
        make_bad_channels_auto : find bad channels automatically
        load_bad_channels : load the current bad_channels file
        merge_bad_channels : merge bad channel definitions for all tasks
        """
        pipe = self._raw[self.get('raw', **kwargs)]
        bids_path = self._bids_path
        pipe._make_bad_channels(bids_path, bad_chs, redo=redo, noise=noise, pipes=self._raw)

    def make_bad_channels_auto(
        self,
        flat: float = None,
        redo: bool = False,
        noise: bool = False,
        **state: Any,
    ) -> None:
        """Automatically detect bad channels

        Works on ``raw='raw'``

        Parameters
        ----------
        flat
            Threshold for detecting flat channels: channels with ``std < flat``
            are considered bad (default 1e-14 for MEG and 0 for EEG).
        redo
            If the file already exists, replace it (instead of adding).
        noise
            If True, make bad channels for the empty-room recording instead of the current subject's recording.
        ...
            State parameters.
        """
        if state:
            self.set(**state)
        pipe = self._raw['raw']
        bids_path = self._bids_path
        pipe._make_bad_channels_auto(bids_path, flat, redo=redo, noise=noise, pipes=self._raw)

    def make_bad_channels_neighbor_correlation(
            self,
            r: float,
            epoch: str = None,
            add_bads: bool = True,
            save: bool = True,
            **state,
    ) -> (NDVar, list[str]):
        """Iteratively exclude bad channels based on low average neighbor-correlation

        Parameters
        ----------
        r
            Minimum admissible neighbor correlation. Any channel whose average
            correlation with its neighbors is below this value is added to the
            list of bad channels (e.g., 0.3).
        epoch
            Epoch to use for computing neighbor-correlation (by default, the
            whole task is used).
        add_bads
            Reject bad channels first.
        save
            Save the bad channels to the bad channel specification file. Set
            ``save=False`` to examine the result without actually changing the
            bad channels.
        ...
            State parameters.

        Returns
        -------
        neighbor_correlation
            Head-map with the neighbor correlation for each sensor.
        bad_channels
            Channels that are excluded based on criteria.

        Notes
        -----
        Algorithm:

        1. Load the corresponding data
        2. Calculate the pairwise correlation between each neighboring sensor pair
        3. Assign to each sensor the average correlation with its neighbors
        4. If the sensor with the lowest correlation is < ``r``, exclude it and
           go back to 2.

        .. warning::
            Data is loaded for the currently specified ``raw`` setting, but bad
            channels apply to all ``raw`` settings equally. Hence, when using this
            method with multiple subjects, it is important to set ``raw`` to the
            same value.
        """
        data, full_nc = self.load_neighbor_correlation(1, epoch, add_bads, return_data=True, **state)
        bad_chs = []
        nc = full_nc
        while nc.min() < r:
            sensor = nc.argmin()
            bad_chs.append(sensor)
            # Recalculate correlations without the bad channel
            new_index = nc.sensor.index(exclude=sensor)
            data = data.sub(sensor=new_index)
            nc = neighbor_correlation(data)
            # Update full head map
            full_index = full_nc.sensor.index(exclude=bad_chs)
            full_nc[full_index] = nc

        if save and bad_chs:
            self.make_bad_channels(bad_chs)
        return full_nc, bad_chs

    @suppress_mne_warning
    def make_ica_selection(
            self,
            epoch: str = None,
            samplingrate: float = None,
            decim: int = None,
            task: str | Sequence[str] = None,
            **state,
    ):
        """Select ICA components to remove through a GUI

        Parameters
        ----------
        epoch
            Load data from this :ref:`state-epoch` for visualization during
            component selection (does not affect the ICA components themselvs).
            If unspecified, the default is to load the data form the entire
            :ref:`state-task` that the ICA is based on.
        samplingrate
            Samplingrate in Hz for the visualization (set to a lower value to
            improve GUI performance; for raw data, the default is ~100 Hz, for
            epochs the default is the epoch setting).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        task
            One or more tasks for which to plot the raw data (this parameter
            can not be used together with ``epoch``; default is the task used
            for ICA estimation).
        ...
            State parameters.

        Notes
        -----
        Computing ICA decomposition can take a while. In order to precompute
        the decomposition for all subjects before doing the selection use
        :meth:`.make_ica()` in a loop as in::

            >>> for subject in e:
            ...     e.make_ica()
            ...
        """
        debug = state.pop('debug', False)
        # ICA
        path = self.make_ica(**state)
        # display data
        subject = self.get('subject')
        pipe = get_ica_pipe(self._raw, self.get('raw', **state))
        bads = pipe._load_bad_channels(self._bids_path, pipes=self._raw)
        with self._temporary_state:
            if epoch is None:
                if task is None:
                    task = pipe.task
                raw = pipe._load_concatenated_source_raw(self._bids_path, task, self._runs, self._raw)
                decim = decim_param(samplingrate, decim, None, raw.info, minimal=True)
                info = raw.info
                display_data = raw
            elif task is not None:
                raise TypeError(f"{task=} with {epoch=}")
            else:
                ds = self.load_epochs(ndvar=False, epoch=epoch, reject=False, raw=pipe.source, samplingrate=samplingrate, decim=decim, add_bads=bads)
                if isinstance(ds['epochs'], Datalist):  # variable-length epoch
                    data = np.concatenate([epoch.get_data()[0] for epoch in ds['epochs']], axis=1)  # n_epochs, n_channels, n_times
                    raw = mne.io.RawArray(data, ds[0, 'epochs'].info)
                    events = mne.make_fixed_length_events(raw)
                    ds = Dataset({'epochs': mne.Epochs(raw, events, 1, 0, 1, baseline=None, proj=False, preload=True)})
                info = ds['epochs'].info
                decim = None
                display_data = ds
        data = TestDims('sensor')
        data_kind = data.data_to_ndvar(info)[0]
        sysname = pipe._get_sysname(info, subject, data_kind, self._raw)
        adjacency = pipe._get_adjacency(data_kind, self._raw)
        frame = gui.select_components(path, display_data, sysname, adjacency, decim, debug)
        if debug:
            return frame

    def make_ica(self, **state):
        """Compute ICA decomposition for a :class:`pipeline.RawICA` preprocessing step

        Parameters
        ----------
        ...
            State parameters.

        Returns
        -------
        path : str
            Path to the ICA file.

        Notes
        -----
        ICA decomposition can take some time. This function can be used to
        precompute ICA decompositions for all subjects after trial pre-rejection
        has been completed::

            >>> for subject in e:
            ...     e.make_ica()

        If an existing ICA file is stale, that means Eelbrain can still see
        the file but can no longer confirm that it was created from the
        current data and ICA settings. You will be asked whether to overwrite
        it or incorporate it as-is. The error message explains which raw step
        and setting changed so you can decide whether to revert that change.

        """
        raw_name = get_ica_pipe_name(self._raw, self.get('raw', **state))
        self._raw[raw_name]
        handle = self._derivatives.resolve(ica_input_name(raw_name), state=self._derivative_state({**state, 'raw': raw_name}))
        ctx = handle
        try:
            handle.node.materialize(ctx)
        except ProtectedArtifactError as error:
            command = ask(
                f"ICA file {Path(error.path).name} is stale. How should it be handled?",
                {
                    'overwrite': 'recompute ICA and overwrite the existing file',
                    'incorporate': 'keep the existing file and rewrite its manifest to the current pipeline state',
                    'abort': 'keep the existing file and abort',
                },
                help="This ICA file may contain manual component selections, so Eelbrain does not replace it automatically when the current data and settings no longer match.",
            )
            if command == 'overwrite':
                handle.node.materialize(ctx, allow_protected_overwrite=True)
            elif command == 'incorporate':
                handle.node.materialize(ctx, allow_protected_reindex=True)
            elif command != 'abort':
                raise RuntimeError(f"{command=}")
            else:
                raise RuntimeError("User aborted ICA overwrite")
        return str(ica_file_path(ctx.state, raw=raw_name))

    def make_mov_ga_dspm(self, subjects=None, baseline=True, src_baseline=False,
                         fmin=2, surf=None, views=None, hemi=None, time_dilation=4.,
                         foreground=None, background=None, smoothing_steps=None,
                         dst=None, redo=False, **state):
        """Make a grand average movie from dSPM values (requires PySurfer 0.6)

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        fmin : scalar
            Minimum dSPM value to draw (default 2). fmax is 3 * fmin.
        surf : str
            Surface on which to plot data.
        views : str | tuple of str
            View(s) of the brain to include in the movie.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot.
        time_dilation : scalar
            Factor by which to slow the passage of time. For example, with
            ``time_dilation=4`` (the default) a segment of data for 500 ms will
            last 2 s.
        foreground : mayavi color
            Figure foreground color (i.e., the text color).
        background : mayavi color
            Figure background color.
        smoothing_steps : None | int
            Number of smoothing steps if data is spatially undersampled (pysurfer
            ``Brain.add_data()`` argument).
        dst : str (optional)
            Path to save the movie. The default is a file in the results
            folder with a name determined based on the input data. Plotting
            parameters (``view`` and all subsequent parameters) are not
            included in the filename. "~" is expanded to the user's home
            folder.
        redo : bool
            Make the movie even if the target file exists already.
        ...
            State parameters.
        """
        state['model'] = ''
        subject, group = self._process_subject_arg(subjects, state)
        data = TestDims("source", morph=bool(group))
        brain_kwargs = self._surfer_plot_kwargs(surf, views, foreground, background, smoothing_steps, hemi)
        self.set(equalize_evoked_count='')

        if dst is not None:
            dst = os.path.expanduser(dst)

        options = {
            'dst': dst,
            'data': data,
            'single_subject': group is None,
            'movie_kind': 'ga-dspm',
            'subject': subject,
            'baseline': baseline,
            'src_baseline': src_baseline,
            'fmin': fmin,
            'brain_kwargs': brain_kwargs,
            'time_dilation': time_dilation,
        }
        handle_state = self._derivative_state()
        if not redo and self._derivatives.resolve('movie', state=handle_state, options=options).is_valid():
            return
        self._load_derivative('movie', options=options, controls={ALLOW_PROTECTED_OVERWRITE})

    def make_mov_ttest(self, subjects=None, model='', c1=None, c0=None, p=0.05,
                       baseline=True, src_baseline=False, disconnect_labels=False,
                       surf=None, views=None, hemi=None, time_dilation=4.,
                       foreground=None, background=None, smoothing_steps=None,
                       dst=None, redo=False, **state):
        """Make a t-test movie (requires PySurfer 0.6)

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        model : None | str
            Model on which the conditions c1 and c0 are defined. The default
            (``''``) is the grand average.
        c1 : None | str | tuple
            Test condition (cell in model). If None, the grand average is
            used and c0 has to be a scalar.
        c0 : str | scalar
            Control condition (cell on model) or scalar against which to
            compare c1.
        p : 0.1 | 0.05 | 0.01 | .001
            Maximum p value to draw.
        baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        disconnect_labels : bool
            Disconnect cluster adjacency across labels from the current
            ``parc`` state.
        surf : str
            Surface on which to plot data.
        views : str | tuple of str
            View(s) of the brain to include in the movie.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot.
        time_dilation : scalar
            Factor by which to slow the passage of time. For example, with
            ``time_dilation=4`` (the default) a segment of data for 500 ms will
            last 2 s.
        foreground : mayavi color
            Figure foreground color (i.e., the text color).
        background : mayavi color
            Figure background color.
        smoothing_steps : None | int
            Number of smoothing steps if data is spatially undersampled (pysurfer
            ``Brain.add_data()`` argument).
        dst : str (optional)
            Path to save the movie. The default is a file in the results
            folder with a name determined based on the input data. Plotting
            parameters (``view`` and all subsequent parameters) are not
            included in the filename. "~" is expanded to the user's home
            folder.
        redo : bool
            Make the movie even if the target file exists already.
        ...
            State parameters.
        """
        if p == 0.1:
            pmid = 0.05
            pmin = 0.01
        elif p == 0.05:
            pmid = 0.01
            pmin = 0.001
        elif p == 0.01:
            pmid = 0.001
            pmin = 0.001
        elif p == 0.001:
            pmid = 0.0001
            pmin = 0.00001
        else:
            raise ValueError(f"p={p}")

        data = TestDims("source", morph=True)
        brain_kwargs = self._surfer_plot_kwargs(surf, views, foreground, background,
                                                smoothing_steps, hemi)
        surf = brain_kwargs['surf']
        if model:
            if not c1:
                raise ValueError("If x is specified, c1 needs to be specified; "
                                 "got c1=%s" % repr(c1))
            elif c0:
                cat = (c1, c0)
            else:
                cat = (c1,)
        elif c1 or c0:
            raise ValueError("If x is not specified, c1 and c0 should not be "
                             "specified either; got c1=%s, c0=%s"
                             % (repr(c1), repr(c0)))
        else:
            cat = None

        state.update(model=model)
        self._current_source_parc(**state)
        with self._temporary_state:
            subject, group = self._process_subject_arg(subjects, state)
            if dst is not None:
                dst = os.path.expanduser(dst)

            options = {
                'dst': dst,
                'data': data,
                'single_subject': group is None,
                'movie_kind': 'ttest',
                'subject': subject,
                'group': group,
                'baseline': baseline,
                'src_baseline': src_baseline,
                'disconnect_labels': disconnect_labels,
                'cat': cat,
                'p': p,
                'pmin': pmin,
                'pmid': pmid,
                'surf': surf,
                'time_dilation': time_dilation,
                'cluster_state': state,
            }
            handle_state = self._derivative_state()
            if not redo and self._derivatives.resolve('movie', state=handle_state, options=options).is_valid():
                return
        self._load_derivative('movie', options=options, controls={ALLOW_PROTECTED_OVERWRITE})

    def export_mrat_evoked(self, **kwargs):
        """Export the sensor data FIF files needed for MRAT sensor analysis

        Parameters
        ----------
        ...
            State parameters.

        Examples
        --------
        To produce evoked files for all subjects in the experiment:

        >>> experiment.set(model='factor1%factor2')
        >>> for _ in experiment:
        >>>     experiment.export_mrat_evoked()
        ...
        """
        ds = self.load_evoked(ndvar=False, **kwargs)
        state = self._derivatives.resolve('evoked', state=self._derivative_state(kwargs)).state
        root = deriv_dir(state) / 'mrat' / state['raw'] / f"{epoch_basename(state)}_epoch-{state['epoch']}_rej-{state['rej']}_model-{state['model']}_count-{state['equalize_evoked_count']}"

        # create fiffs
        model = state['model']
        factors = [f.strip() for f in model.split('%')]
        for case in ds.itercases():
            condition = '_'.join(case[f] for f in factors)
            path = root / condition / f"{condition}_{state['subject']}-ave.fif"
            path.parent.mkdir(parents=True, exist_ok=True)
            evoked = case['evoked']
            evoked.save(path)

    def export_mrat_stcs(self, **kwargs):
        """Export the STC files needed for the MRAT analysis tool

        Parameters
        ----------
        ...
            State parameters.

        Examples
        --------
        To produce stc files for all subjects in the experiment:

        >>> experiment.set_inv('free')
        >>> experiment.set(model='factor1%factor2')
        >>> for _ in experiment:
        >>>     experiment.export_mrat_stcs()
        ...
        """
        ds = self.load_evoked_stc(morph=True, ndvar=False, **kwargs)
        state = self._derivatives.resolve(
            'evoked-stc',
            state=self._derivative_state(kwargs),
            options={'morph': True, 'ndvar': False},
        ).state
        kind = '_'.join((state['raw'], state['cov'], state['mri'], state['src'], state['inv']))
        root = deriv_dir(state) / 'mrat' / kind / f"{epoch_basename(state)}_epoch-{state['epoch']}_rej-{state['rej']}_model-{state['model']}_count-{state['equalize_evoked_count']}"

        # save condition info
        info_file = deriv_dir(state) / 'mrat' / f"{state['subject']} info.txt"
        info_file.parent.mkdir(parents=True, exist_ok=True)
        ds.save_txt(info_file)

        # create stcs
        model = state['model']
        factors = [f.strip() for f in model.split('%')]
        for case in ds.itercases():
            condition = '_'.join(case[f] for f in factors)
            path = root / condition / f"{condition}_{state['subject']}"
            path.parent.mkdir(parents=True, exist_ok=True)
            stc = case['stcm']
            stc.save(path)

    def make_plot_annot(self, surf='inflated', redo=False, **state):
        """Create a figure for the contents of an annotation file

        Parameters
        ----------
        surf : str
            FreeSurfer surface on which to plot the annotation.
        redo : bool
            If the target file already exists, overwrite it.
        ...
            State parameters.
        """
        with self._temporary_state:
            if state:
                self.set(**state)
            if is_fake_mri(mri_dir(self._derivative_state())):
                self.set(mrisubject=self.get('common_brain'), match=False)

            export_state = self._derivative_state()
            stem = join_stem_parts(
                f"parc-{export_state['parc']}",
                f"mrisubject-{export_state['mrisubject']}",
                f"surf-{surf}",
            )
            dst = results_dir(export_state) / 'source-annot' / f'{stem}.png'
            if not redo and dst.exists():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)

            brain = self.plot_annot(surf=surf, axw=600)
            brain.save_image(dst, 'rgba', True)
            legend = brain.plot_legend(show=False)
            legend.save(dst.with_suffix('.pdf'), facecolor="none")
            brain.close()
            legend.close()

    def make_plot_label(self, label, surf='inflated', redo=False, **state):
        with self._temporary_state:
            if state:
                self.set(**state)
            if is_fake_mri(mri_dir(self._derivative_state())):
                self.set(mrisubject=self.get('common_brain'), match=False)

            dst = self._make_plot_label_dst(surf, label)
            if not redo and dst.exists():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)

            brain = self.plot_label(label, surf=surf)
            brain.save_image(dst, 'rgba', True)

    def make_plots_labels(self, surf='inflated', redo=False, **state):
        self.set(**state)
        with self._temporary_state:
            if is_fake_mri(mri_dir(self._derivative_state())):
                self.set(mrisubject=self.get('common_brain'), match=False)

            labels = tuple(self._load_labels().values())
            dsts = [self._make_plot_label_dst(surf, label.name) for label in labels]
        if not redo and all(exists(dst) for dst in dsts):
            return

        brain = self.plot_brain(surf, None, 'split', ['lat', 'med'], w=1200)
        for label, dst in zip(labels, dsts):
            brain.add_label(label)
            brain.save_image(dst, 'rgba', True)
            brain.remove_labels(hemi='lh')

    def _make_plot_label_dst(self, surf, label):
        state = self._derivative_state()
        directory = results_dir(state) / 'source-labels' / join_stem_parts(
            f"parc-{state['parc']}",
            f"mrisubject-{state['mrisubject']}",
            f"surf-{surf}",
        )
        return directory / f'{join_stem_parts(label)}.png'

    def make_epoch_selection(
            self,
            samplingrate: int = None,
            data: str = 'sensor',
            auto: float | dict = None,
            overwrite: bool = None,
            decim: int = None,
            **state):
        """Open :func:`gui.select_epochs` for manual epoch selection

        The GUI is opened with the correct file name; if the corresponding
        file exists, it is loaded, and upon saving the correct path is
        the default.

        Parameters
        ----------
        samplingrate
            Samplingrate in Hz for the visualization (set to a lower value to
            improve GUI performance; the default is the epoch setting).
        data
            For data with multiple channel types, specify the channel type to
            display (``mag``, ``planar1``, ``planar2``).
        auto : scalar (optional)
            Perform automatic rejection instead of showing the GUI by supplying
            a an absolute threshold (for example, ``1e-12`` to reject any epoch
            in which the absolute of at least one channel exceeds 1 picotesla).
            If a rejection file already exists also set ``overwrite=True``.
            When working with data from multiple sensor types, use a dictionary
            to set levels for all types,
            e.g. ``{'mag': 2e-12, 'grad': 5e-11, 'eeg': 1.5e-4}``.
        overwrite : bool
            If ``auto`` is specified and a rejection file already exists,
            overwrite the old file. The default is to raise an :exc:`IOError` if
            the file exists (``None``). Set to ``False`` to quietly keep the
            exising file.
        decim
            Data decimation factor (alternative to ``samplingrate``).
        ...
            State parameters.


        Notes
        -----
        By default, the epoch selection is different for each primary epoch and
        for each preprocessing setting (``raw``). To share the same epoch
        selection, create the corresponding selection file for each target
        preprocessing setting.
        """
        rej = self.get('rej', **state)
        rej_args = self._artifact_rejection[rej]
        if rej_args['kind'] != 'manual':
            raise ValueError(f"{rej=}; Epoch rejection is not manual")

        if data == 'grad':
            raise NotImplementedError(f"{data=} visualization of gradiometer vector data is not implemented; use data='planar1' and data='planar2'")
        data = TestDims.coerce(data)
        assert data.sensor is True

        epoch = self._epochs[self.get('epoch')]
        if not isinstance(epoch, PrimaryEpoch):
            if isinstance(epoch, SecondaryEpoch):
                raise ValueError(f"The current epoch {epoch.name!r} inherits selections from {epoch.sel_epoch!r}. To access a rejection file for this epoch, call `e.set(epoch={epoch.sel_epoch!r})` and then call `e.make_epoch_selection()` again.")
            elif isinstance(epoch, SuperEpoch):
                raise ValueError(f"The current epoch {epoch.name!r} inherits selections from these other epochs: {epoch.sub_epochs!r}. To access selections for these epochs, call `e.make_epoch_selection(epoch=epoch)` for each.")
            else:
                raise ValueError(f"The current epoch {epoch.name!r} is not a primary epoch and inherits selections from other epochs. Generate trial rejection for these epochs.")

        path = rej_file_path(self._derivative_state(task=epoch.task))
        path.parent.mkdir(parents=True, exist_ok=True)

        if auto is not None and overwrite is not True and exists(path):
            if overwrite is False:
                return
            elif overwrite is None:
                raise OSError(self.format("A rejection file already exists for {subject}, epoch {epoch}, rej {rej}. Set the overwrite parameter to specify how to handle existing files."))
            else:
                raise TypeError(f"{overwrite=}")

        ds = self.load_epochs(ndvar=True, data=data, reject=False, trigger_shift=False, samplingrate=samplingrate, decim=decim)
        if data._to_ndvar is None:
            ch_types = ['meg', 'mag', 'grad', 'planar1', 'planar2', 'eeg']
            ch_types = [t for t in ch_types if t in ds]
            if len(ch_types) > 1 and not auto:
                raise NotImplementedError(f"Found multiple channel types: {enumeration(ch_types)}. Rejection GUI for multiple channel types is not implemented. Use the `data` parameter to visualize one channel type at a time.")
            elif not ch_types:
                raise RuntimeError("No data found")
            y_name = ch_types.pop()
        else:
            y_name = data.y_name

        if auto is not None:
            if isinstance(auto, dict):
                auto_dict = auto
                missing = {key for key in auto_dict if key not in ds}
                if missing:
                    raise ValueError(f"{auto=}: channel types {enumeration(missing)} not in data")
            else:
                auto_dict = {y_name: auto}
            # create rejection
            rej_ds = new_rejection_ds(ds)
            rej_ds[:, 'accept'] = True
            for key, threshold in auto_dict.items():
                rej_ds['accept'] &= ds[key].abs().max(('sensor', 'time')) <= threshold
            # create description for info
            args = [f"{auto=}"]
            if overwrite is True:
                args.append("overwrite=True")
            if samplingrate is not None:
                args.append(f"{samplingrate=}")
            if decim is not None:
                args.append(f"{decim=}")
            rej_ds.info['desc'] = f"Created with {self.__class__.__name__}.make_epoch_selection({', '.join(args)})"
            # save
            save.pickle(rej_ds, path)
            # print info
            n_rej = rej_ds.eval("sum(accept == False)")
            desc = self.format("{subject}, epoch {epoch}")
            self._log.info(f"make_epoch_selection: {n_rej} of {rej_ds.n_cases} epochs rejected with threshold {auto} for {desc}")
            return

        vlim = {'meg': 2e-12, 'mag': 2e-12, 'eeg': 1.5e-4, 'planar1': 5e-11, 'planar2': 5e-11}[y_name]
        eog_sns = self._eog_sns.get(ds[y_name].sensor.sysname, ())
        # don't mark eog sns if it is bad
        bad_channels = self.load_bad_channels()
        eog_sns = [c for c in eog_sns if c not in bad_channels]

        gui.select_epochs(ds, y_name, path=path, vlim=vlim, mark=eog_sns)

    def make_report(
            self,
            test: str,
            disconnect_labels: bool = False,
            pmin: str = None,
            tstart: float = None,
            tstop: float = None,
            samples: int = 10000,
            baseline: BaselineArg = True,
            src_baseline: BaselineArg = None,
            include: float = 0.2,
            redo: bool = False,
            **state,
    ):
        """Create an HTML report on spatio-temporal clusters

        Parameters
        ----------
        test
            Test for which to create a report (entry in Pipeline.tests).
        disconnect_labels
            Disconnect source-space cluster adjacency across labels from the
            current ``parc`` state instead of running one masked whole-brain
            source test.
        pmin
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart
            Beginning of the time window for the test in seconds
            (default is the beginning of the epoch).
        tstop
            End of the time window for the test in seconds
            (default is the end of the epoch).
        samples
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 10,000).
        baseline
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        include : 0 < scalar <= 1
            Create plots for all clusters with p-values smaller or equal this value.
        redo
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        ...
            State parameters.

        See Also
        --------
        load_test : load corresponding data and tests
        """
        if samples < 1:
            raise ValueError(f"{samples=}: needs to be > 0")
        elif include <= 0 or include > 1:
            raise ValueError(f"{include=}: needs to be 0 < include <= 1")

        self.set(**state)
        self._current_source_parc(**state)
        data = TestDims('source', morph=True)
        options = {
            'data': data,
            'samples': samples,
            'test': test,
            'baseline': baseline,
            'src_baseline': src_baseline,
            'disconnect_labels': disconnect_labels,
            'pmin': pmin,
            'tstart': tstart,
            'tstop': tstop,
            'include': include,
        }
        handle_state = self._derivative_state()
        if not redo and self._derivatives.resolve('source-report', state=handle_state, options=options).is_valid():
            return
        self._load_derivative('source-report', options=options, controls={ALLOW_PROTECTED_OVERWRITE})

    def make_report_rois(self, test, pmin=None, tstart=None, tstop=None,
                         samples=10000, baseline=True, src_baseline=False,
                         redo=False, **state):
        """Create an HTML report on ROI time courses

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in Pipeline.tests).
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : scalar
            Beginning of the time window for the test in seconds
            (default is the beginning of the epoch).
        tstop : scalar
            End of the time window for the test in seconds
            (default is the end of the epoch).
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters.
        baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        redo : bool
            If the target file already exists, delete and recreate it.
        ...
            State parameters.

        See Also
        --------
        load_test : load corresponding data and tests (use ``data="source.mean"``)
        """
        self.set(test=test, **state)
        test_obj = self.tests[test]
        if samples < 1:
            raise ValueError("Need samples > 0 to run permutation test.")
        elif isinstance(test_obj, TwoStageTest):
            raise NotImplementedError("ROI analysis not implemented for two-stage tests")

        self._current_source_parc(**state)
        data = TestDims('source.mean')
        options = {
            'data': data,
            'samples': samples,
            'test': test,
            'baseline': baseline,
            'src_baseline': src_baseline,
            'pmin': pmin,
            'tstart': tstart,
            'tstop': tstop,
        }
        handle_state = self._derivative_state()
        if not redo and self._derivatives.resolve('roi-report', state=handle_state, options=options).is_valid():
            return
        self._load_derivative('roi-report', options=options, controls={ALLOW_PROTECTED_OVERWRITE})

    def make_report_coreg(self, file_name=None, **state):
        """Create HTML report with plots of the MEG/MRI coregistration

        Parameters
        ----------
        file_name : str
            Where to save the report (default is in the root/methods director).
        ...
            State parameters.
        """
        self.set(**state)
        if file_name is not None:
            file_name = os.path.expanduser(file_name)
        self._load_derivative(
            'coreg-report',
            options={'dst': file_name},
            controls={ALLOW_PROTECTED_OVERWRITE},
        )

    def make_src(self, **state):
        """Make the source space

        Parameters
        ----------
        ...
            State parameters.
        """
        self._load_derivative('src', state=state)

    def merge_bad_channels(self):
        """Merge bad channel definitions for different tasks

        Load the bad channel definitions for all tasks of the current
        subject and save the union for all tasks.

        See Also
        --------
        make_bad_channels : set bad channels for a single task
        """
        n_chars = max(map(len, self._tasks))
        # collect bad channels
        bads = set()
        tasks = []
        with self._temporary_state:
            # ICARaw merges bad channels dynamically, so explicit merge needs to
            # be performed lower in the hierarchy
            self.set(raw='raw')
            for task in self.iter('task'):
                if exists(self._raw['raw']._raw_path(self._bids_path)):
                    bads.update(self.load_bad_channels())
                    tasks.append(task)
                else:
                    print("%%-%is: skipping, raw file missing" % n_chars % task)
            # update bad channel files
            for task in tasks:
                print(task.ljust(n_chars), end=': ')
                self.make_bad_channels(bads, task=task)

    def next(self, field='subject'):
        """Change field to the next value

        Parameters
        ----------
        field : str | list of str
            The field for which the value should be changed (default 'subject').
            Can also contain multiple fields, e.g. ``['subject', 'session']``.

        Example
        -------
        >>> raw_01 = e.load_raw()  # raw for S01
        >>> e.next()
        subject: S01 -> S03
        >>> raw_03 = e.load_raw()  # raw for S03
        >>> e.next()
        subject: S03 -> S04

        """
        if isinstance(field, str):
            current = self.get(field)
            values = self.get_field_values(field)
            def fmt(x): return x
        else:
            current = tuple(self.get(f) for f in field)
            values = list(product(*(self.get_field_values(f) for f in field)))
            def fmt(x): return '/'.join(x)

        # find the index of the next value
        if current in values:
            idx = values.index(current) + 1
            if idx == len(values):
                idx = -1
        else:
            for idx in range(len(values)):
                if values[idx] > current:
                    break
            else:
                idx = -1

        # set the next value
        if idx == -1:
            next_ = values[0]
            print(f"The last {fmt(field)} was reached; rewinding to {fmt(next_)}")
        else:
            next_ = values[idx]
            print(f"{fmt(field)}: {fmt(current)} -> {fmt(next_)}")

        if isinstance(field, str):
            self.set(**{field: next_})
        else:
            self.set(**dict(zip(field, next_)))

    def plot_annot(
            self,
            parc: str = None,
            surf: str = None,
            views: str | Sequence[str] = None,
            hemi: str = None,
            borders: bool | int = False,
            alpha: float = 0.7,
            w: int = None,
            h: int = None,
            axw: int = None,
            axh: int = None,
            foreground: Any = None,
            background: Any = None,
            seeds: bool = False,
            **state,
    ):
        """Plot the annot file on which the current parcellation is based

        Parameters
        ----------
        parc
            Parcellation to plot. If None (default), use parc from the current
            state.
        surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
            Freesurfer surface to use as brain geometry.
        views
            One or several views to show in the figure. The options are:
            ``'lateral', 'medial', 'ventral', 'dorsal', 'rostral', 'parietal',
            'frontal', 'caudal'``.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot (default includes hemisphere with more
            than one label in the annot file).
        borders
            Show only label borders (PySurfer Brain.add_annotation() argument).
        alpha
            Alpha of the annotation (1=opaque, 0=transparent, default 0.7).
        axw
            Figure width per hemisphere.
        foreground : mayavi color
            Figure foreground color (i.e., the text color).
        background : mayavi color
            Figure background color.
        seeds
            Plot seeds as points (only applies to seeded parcellations).
        ...
            State parameters.

        Returns
        -------
        brain : Brain
            PySurfer Brain with the parcellation plot.
        legend : ColorList
            ColorList figure with the legend.
        """
        if parc is not None:
            state['parc'] = parc
        self.set(**state)

        self.make_annot()

        parc_name, parc = self._get_parc()
        if seeds:
            if not isinstance(parc, SeededParc):
                raise ValueError(f"seeds=True is only valid for seeded parcellation, not for parc={parc_name!r}")
            # if seeds are defined on a scaled common-brain, we need to plot the scaled brain:
            plot_on_scaled_common_brain = isinstance(parc, IndividualSeededParc)
        else:
            plot_on_scaled_common_brain = False

        state_ = self._derivative_state()
        subjects_dir = str(mri_sdir(state_))
        if (not plot_on_scaled_common_brain) and is_fake_mri(mri_dir(state_)):
            subject = self.get('common_brain')
        else:
            subject = self.get('mrisubject')

        kwa = self._surfer_plot_kwargs(surf, views, foreground, background, None, hemi)
        brain = plot.brain.annot(parc_name, subject, borders=borders, alpha=alpha, w=w, h=h, axw=axw, axh=axh, subjects_dir=subjects_dir, **kwa)
        if seeds:
            from mayavi import mlab

            seeds = parc._seeds_for_subject(subject)
            seed_points = {hemi: [np.atleast_2d(coords) for name, coords in seeds.items() if name.endswith(hemi)] for hemi in ('lh', 'rh')}
            plot_points = {hemi: np.vstack(points).T if len(points) else None for hemi, points in seed_points.items()}
            for hemisphere in brain.brains:
                if plot_points[hemisphere.hemi] is None:
                    continue
                x, y, z = plot_points[hemisphere.hemi]
                mlab.points3d(x, y, z, figure=hemisphere._f, color=(1, 0, 0), scale_factor=10)
            brain.set_parallel_view(scale=True)

        return brain

    def plot_brain(self, common_brain=True, **brain_kwargs):
        """Plot the brain model

        Parameters
        ----------
        common_brain : bool
            If the current mrisubject is a scaled MRI, use the common_brain
            instead.
        ... :
            :class:`~plot._brain_object.Brain` options as keyword arguments.
        """
        from ..plot._brain_object import Brain

        brain_args = self._surfer_plot_kwargs()
        brain_args.update(brain_kwargs)
        state_ = self._derivative_state()
        brain_args['subjects_dir'] = str(mri_sdir(state_))
        if 'hemi' not in brain_args:
            brain_args['hemi'] = self.get('hemi')

        # find subject
        if common_brain and is_fake_mri(mri_dir(state_)):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)
        else:
            mrisubject = self.get('mrisubject')

        return Brain(mrisubject, **brain_args)

    def plot_coregistration(
            self,
            surfaces: str | list | dict = 'auto',
            meg: tuple[str, ...] = ('helmet', 'sensors'),
            dig: bool = True,
            parallel: bool = True,
            **state):
        """Plot the coregistration (Head shape and MEG helmet)

        Parameters
        ----------
        surfaces
            :func:`mne.viz.plot_alignment` parameter.
        meg
            :func:`mne.viz.plot_alignment` parameter.
        dig
            Plot the digitization points (default True; 'fiducials' to plot
            fiducial points only).
        parallel
            Set parallel view.
        ...
            State parameters.

        Notes
        -----
        Uses :func:`mne.viz.plot_alignment`
        """
        self.set(**state)
        with self._temporary_state:
            raw = self.load_raw(raw='raw')
        state_ = self._derivative_state()
        fig = mne.viz.plot_alignment(raw.info, trans_file_path(state_), self.get('mrisubject'), mri_sdir(state_), surfaces, meg=meg, dig=dig, interaction='terrain')
        if parallel:
            fig.plotter.enable_parallel_projection()
        return fig

    def plot_whitened_gfp(self, s_start=None, s_stop=None, run=None):
        """Plot the GFP of the whitened evoked to evaluate the the covariance matrix

        Parameters
        ----------
        s_start : str
            Subject at which to start (default is the first subject).
        s_stop: str
            Subject at which to stop (default is the last subject).
        run : bool
            Run the GUI after plotting (default depends on environment).
        """
        gfps = []
        subjects = []
        with self._temporary_state:
            self.set(model='')
            for subject in self.iter_range(s_start, s_stop):
                cov = self.load_cov()
                picks = np.arange(len(cov.ch_names))
                ds = self.load_evoked(baseline=True, ndvar=False)
                whitened_evoked = mne.whiten_evoked(ds[0, 'evoked'], cov, picks)
                gfp = whitened_evoked.data.std(0)

                gfps.append(gfp)
                subjects.append(subject)

        colors = plot.colors_for_oneway(subjects)
        title = f"Whitened Global Field Power ({self.get('cov')})"
        fig = plot._base.Figure(1, title, h=7, run=run)
        ax = fig.axes[0]
        for subject, gfp in zip(subjects, gfps):
            ax.plot(whitened_evoked.times, gfp, label=subject, color=colors[subject])
        ax.legend(loc='right')
        fig.show()
        return fig

    def plot_evoked(
            self,
            subjects: SubjectArg = None,
            data: DataArg = None,
            separate: bool = False,
            baseline: BaselineArg = True,
            ylim: Literal['same', 'different'] = 'same',
            name: str = None,
            h: float = 2.5,
            run: bool = None,
            **kwargs):
        """Plot evoked sensor data

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group. Default is current subject
            (or group if ``group`` is specified).
        data
            By default, plot sensor data and source estimates; set to ``meg``/
            ``eeg``/``source`` to plot only one.
        separate
            When plotting a group, plot all subjects separately instead or the group
            average (default False).
        baseline
            Apply baseline correction using this period. True to use the epoch's
            baseline specification (default).
        ylim
            Use the same or different y-axis limits for different subjects
            (default 'same').
        name
            Name to display as window title (default is subject epoch model).
        h
            Height per plot.
        run
            Run the GUI after plotting (default in accordance with plotting
            default).
        ...
            State parameters.
        """
        subject, group = self._process_subject_arg(subjects, kwargs)
        if data is None:
            sns = src = True
        else:
            data = TestDims.coerce(data)
            sns, src = bool(data.sensor), bool(data.source)
        model = self.get('model') or None
        epoch = self.get('epoch')
        if model:
            model_name = f"~{model}"
        elif subject or separate:
            model_name = "Average"
        else:
            model_name = "Grand Average"
        is_vector_data = src and self.get('inv').startswith('vec')
        is_volume_source_space = src and self.get('src').startswith('vol')
        if is_vector_data and not is_volume_source_space:
            raise NotImplementedError("Vector data currently can only be plotted for volume source space")

        if separate and not subject:
            if src:
                raise NotImplementedError(f"{separate=} for source estimates")
            plots = []
            vlim = []
            for subject in self.iter(group=group):
                ds = self.load_evoked(baseline=baseline)
                y = guess_y(ds)
                title = f"{subject} {epoch} {model_name}"
                p = plot.TopoButterfly(y, model, data=ds, axh=h, name=title, run=False)
                plots.append(p)
                vlim.append(p.get_vlim())

            if ylim.startswith('s'):
                vlim = np.array(vlim)
                vmax = np.abs(vlim, out=vlim).max()
                for p in plots:
                    p.set_vlim(vmax)
            elif not ylim.startswith('d'):
                raise ValueError(f"{ylim=}")

            if run or plot._base.do_autorun():
                gui.run()

        if subject:
            title = name or f"{subject} {epoch} {model_name}"
            subject_arg = subject
            src_key = 'src'
        else:
            title = name or f"{group} {epoch} {model_name}"
            subject_arg = group
            src_key = 'srcm'

        if src:
            ds = self.load_evoked_stc(subject_arg, baseline=baseline, keep_evoked=sns)
            out = [ds]
            if model:
                x = ds.eval(model)
                ys = [ds[src_key].mean(case=x == cell) for cell in x.cells]
            else:
                ys = [ds[src_key]]
            for y in ys:
                if is_volume_source_space:
                    plots = plot.GlassBrain.butterfly(y, w=2 * h, h=h, name=title)
                else:
                    plots = plot.brain.butterfly(y, w=2 * h, h=h, name=title)
                out.extend(plots)
            right_of = out[2]
        else:
            ds = self.load_evoked(subject_arg, baseline=baseline)
            out = [ds]
            right_of = None
        if sns:
            key = 'meg' if 'meg' in ds else 'eeg'
            p = plot.TopoButterfly(key, model, data=ds, axh=h, w=2.5 * h, name=title, right_of=right_of, run=run)
            if right_of:
                p.link_time_axis(right_of)
            out.append(p)
        return out

    def plot_label(self, label, surf=None, views=None, w=600):
        """Plot a label"""
        if isinstance(label, str):
            label = self.load_label(label)
        title = label.name
        hemi = 'split' if isinstance(label, mne.BiHemiLabel) else label.hemi
        kwargs = self._surfer_plot_kwargs(surf, views, hemi=hemi)
        brain = self.plot_brain(title=title, w=w, **kwargs)
        brain.add_label(label, alpha=0.75)
        return brain

    def plot_raw(self, decim=10, xlim=5, add_bads=True, subtract_mean=False, **state):
        """Plot raw sensor data

        Parameters
        ----------
        decim : int
            Decimate data for faster plotting (default 10).
        xlim : scalar
            Number of seconds to display (default 5 s).
        add_bads : bool | list
            Add bad channel information to the bad channels text file (default
            True).
        subtract_mean : bool
            Subtract the mean from each channel (useful when plotting raw data
            recorded with DC offset).
        ...
            State parameters.
        """
        raw = self.load_raw(add_bads, ndvar=True, decim=decim, **state)
        state_ = self._derivative_state(state)
        name = join_stem_parts(raw_basename(state_), f'raw-{state_["raw"]}')
        if raw.info['meas'] == 'V':
            vmax = 1.5e-4
        elif raw.info['meas'] == 'B':
            vmax = 2e-12
        else:
            vmax = None
        if subtract_mean:
            raw -= raw.mean('time')
        return plot.TopoButterfly(raw, w=0, h=3, xlim=xlim, vmax=vmax, name=name)

    def run_mne_analyze(self, modal=False):
        """Run mne_analyze

        Parameters
        ----------
        modal : bool
            Causes the shell to block until mne_analyze is closed.

        Notes
        -----
        Sets the current directory to raw-dir, and sets the SUBJECT and
        SUBJECTS_DIR to current values
        """
        state_ = self._derivative_state()
        subp.run_mne_analyze(str(raw_dir(state_)), self.get('mrisubject'),
                             str(mri_sdir(state_)), modal)

    def run_mne_browse_raw(self, modal=False):
        """Run mne_analyze

        Parameters
        ----------
        modal : bool
            Causes the shell to block until mne_browse_raw is closed.

        Notes
        -----
        Sets the current directory to raw-dir, and sets the SUBJECT and
        SUBJECTS_DIR to current values
        """
        state_ = self._derivative_state()
        subp.run_mne_browse_raw(str(raw_dir(state_)), self.get('mrisubject'), str(mri_sdir(state_)), modal)

    def set(self, subject=None, match=True, **state):
        """
        Set variable values.

        Parameters
        ----------
        subject : str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        match : bool
            For fields with pre-defined values, only allow valid values (default
            ``True``).
        ...
            State parameters.
        """
        if subject is not None:
            if 'group' not in state:
                if subject not in self._field_values['subject'] and subject in self._groups['all']:
                    old = self.get('group')
                    print(f"group: {old} --> all ({subject} not in {old})")
                    state['group'] = 'all'
                else:
                    state['subject'] = subject
                    subject = None
        StateModel.set(self, match, **state)
        if subject is not None:
            StateModel.set(self, match, subject=subject)
        self._update_bids_path()

    def _post_set_group(self, _, group):
        if group == '*' or group not in self._groups:
            return
        group_members = self._groups[group]
        self._field_values['subject'] = group_members
        subject = self.get('subject')
        if subject != '*' and subject not in group_members and group_members:
            self.set(group_members[0])

    def set_inv(
            self,
            ori: str = 'free',
            snr: float = 3,
            method: str = 'dSPM',
            depth: float = 0.8,
            pick_normal: bool = False,
            **state):
        """Set the type of inverse solution used for source estimation

        Parameters
        ----------
        ori : 'free' | 'fixed' | 'vec' | float ]0, 1]
            Orientation constraint (default ``'free'``; use a number between 0
            and 1 to specify a loose constraint).

            At each source point, ...

            - ``free``: ... estimate a current dipole with arbitrary direction.
              For further analysis, only the magnitude of the current is
              retained, while the direction is ignored. This is good for
              detecting changes in neural current strength when current
              direction is variable (for example, due to anatomical differences
              between subjects).
            - ``fixed``: ... estimate current flow orthogonal to the cortical
              surface. The sign of the estimates indicates current direction
              relative to the surface (positive for current out of the brain).
            - ``vec``: ... estimate a current vector with arbitrary direction,
              and return this current as 3 dimensional vector.
            - loose (``float``): ... estimate a current dipole with arbitrary
              direction. Then, multiple the two components parallel to the
              surface with this number, and retain the magnitude.

        snr
            SNR estimate used for regularization (``λ = 1 / snr``). Larger λ
            (smaller SNR) correspond to spatially smoother and weaker current
            estimates. 3 is recommended for averaged responses, 1 for raw or
            single trial data. Set to 0 for unregularized inverse solution
            (``λ = 0``).
        method : 'MNE' | 'dSPM' | 'sLORETA' | 'eLORETA'
            Noise normalization method. ``MNE`` uses unnormalized current
            estimates. ``dSPM`` [1]_ (default) ``sLORETA`` [2]_ and eLORETA [3]_
            normalize each the estimate at each source with an estimate of the
            noise at that source (default ``'dSPM'``).
        depth
            Depth weighting [4]_ (``0`` to disable depth weighting).
            See :func:`mne.minimum_norm.make_inverse_operator`.
        pick_normal
            Estimate a free orientation current vector, then pick the component
            orthogonal to the cortical surface and discard the parallel
            components.
        ...
            State parameters.

        Notes
        -----
        Can also be set through the ``inv`` state parameter (see :ref:`state-inv`).
        To determine the string corresponding to a given set of parameters,
        use :meth:`Pipeline.inv_str`.

        .. warning::
            Free and loose orientation inverse solutions have a non-zero
            expected value. In that case, when source localizing condition
            averages, the number of trials affects the expected value.
            For designs with unequal number of trials per cell,
            be sure to use :ref:`state-equalize_evoked_count` appropriately.

        References
        ----------
        .. [1] Dale A, Liu A, Fischl B, Buckner R. (2000)
               Dynamic statistical parametric mapping: combining fMRI and MEG
               for high-resolution imaging of cortical activity.
               Neuron, 26:55-67.
               `10.1016/S0896-6273(00)81138-1
               <https://doi.org/10.1016/S0896-6273(00)81138-1>`_
        .. [2] Pascual-Marqui RD (2002),
               Standardized low resolution brain electromagnetic tomography
               (sLORETA): technical details.
               Methods Find. Exp. Clin. Pharmacology, 24(D):5-12.
        .. [3] Pascual-Marqui RD (2007).
               Discrete, 3D distributed, linear imaging methods of electric
               neuronal activity. Part 1: exact, zero error localization.
               `arXiv:0710.3341 <https://arxiv.org/abs/0710.3341>`_
        .. [4] Lin F, Witzel T, Ahlfors S P, Stufflebeam S M, Belliveau J W,
               Hämäläinen M S. (2006) Assessing and improving the spatial accuracy
               in MEG source localization by depth-weighted minimum-norm estimates.
               NeuroImage, 31(1):160–171.
               `10.1016/j.neuroimage.2005.11.054
               <https://doi.org/10.1016/j.neuroimage.2005.11.054>`_

        """
        self.set(inv=self.inv_str(ori, snr, method, depth, pick_normal), **state)

    @staticmethod
    def inv_str(
            ori: str = 'free',
            snr: float = 3,
            method: str = 'dSPM',
            depth: float = 0.8,
            pick_normal: bool = False,
    ):
        "Construct inv string from settings; see :meth:`.set_inv`"
        return MinimumNormInverseSolution(ori, snr, method, depth, pick_normal)._string()

    @classmethod
    def _eval_inv(cls, inv):
        return InverseSolution._coerce(inv)._string()

    def _eval_model(self, model):
        if model == '':
            return model
        elif len(model) > 1 and '*' in model:
            raise ValueError("model=%r; To specify interactions, use '%' instead of '*'")

        factors = [v.strip() for v in model.split('%')]

        # find order value for each factor
        ordered_factors = {}
        unordered_factors = []
        for factor in sorted(factors):
            assert_is_legal_dataset_key(factor)
            if factor in self._model_order:
                ordered_factors[self._model_order.index(factor)] = factor
            else:
                unordered_factors.append(factor)

        # recompose
        model = [ordered_factors[v] for v in sorted(ordered_factors)]
        if unordered_factors:
            model.extend(unordered_factors)
        return '%'.join(model)

    def _update_mrisubject(self, fields):
        subject = fields['subject']
        mri = fields['mri']
        if subject == '*' or mri == '*':
            return '*'
        mrisubject = self._mri_subjects[mri][subject]
        if mrisubject == self.get('common_brain') or mrisubject.startswith('sub-'):
            return mrisubject
        return 'sub-' + mrisubject

    def _update_task(self, fields):
        epoch = fields['epoch']
        if epoch in self._epochs:
            epoch = self._epochs[epoch]
            return epoch.tasks[0]
        elif not epoch or epoch == '*':
            return  # don't force task
        return '*'  # if a named epoch is not in _epochs it might be a removed epoch

    def _eval_parc(self, parc):
        if not parc:
            return ''
        if parc in self._parcs:
            if isinstance(self._parcs[parc], SeededParc):
                raise ValueError(f"Seeded parc set without size, use e.g. parc='{parc}-25'")
            else:
                return parc
        m = SEEDED_PARC_RE.match(parc)
        if m:
            name = m.group(1)
            if isinstance(self._parcs.get(name), SeededParc):
                return parc
            else:
                raise ValueError(f"{parc=}: No parcellation named '{parc}' and no seeded parcellation named '{name}'")
        else:
            raise ValueError(f"{parc=}")

    def _get_parc(self):
        """Parc information

        Returns
        -------
        parc : str
            The current parc setting.
        params : dict | None
            The parc definition (``None`` for ``parc=''``).
        """
        return _resolve_parc(self._parcs, self.get('parc'))

    def _post_set_test(self, _, test):
        if test != '*' and test in self.tests:  # with vmatch=False, test object might not be availale
            test_obj = self.tests[test]
            if test_obj.model is not None:
                self.set(model=test_obj.model)

    def show_bad_channels(
            self,
            tasks: bool | str | Sequence[str] = None,
            **state,
    ):
        """List bad channels

        Parameters
        ----------
        tasks
            By default, bad channels for the current task are shown. Set
            ``tasks`` to ``True`` to show bad channels for all tasks, or
            a list of task names to show bad channeles for these tasks.
        ...
            State parameters.

        Notes
        -----
        ICA Raw pipes merge bad channels from different tasks (by combining
        the bad channels from all tasks).
        """
        if state:
            self.set(**state)

        if tasks is True:
            use_tasks = self._tasks
        elif tasks:
            use_tasks = [tasks] if isinstance(tasks, str) else tasks
        else:
            use_tasks = None

        if use_tasks is None:
            bad_channels = {subject: self.load_bad_channels() for subject in self}
            list_tasks = False
        else:
            bad_channels = {key: self.load_bad_channels() for key in self.iter(('subject', 'task'), values={'task': use_tasks})}
            # whether they are equal between tasks
            bad_by_s = {}
            for (subject, task), bads in bad_channels.items():
                if subject in bad_by_s:
                    if bad_by_s[subject] != bads:
                        list_tasks = True
                        break
                else:
                    bad_by_s[subject] = bads
            else:
                bad_channels = bad_by_s
                list_tasks = False

        # table
        task_desc = ', '.join(use_tasks) if use_tasks else self.get('task')
        caption = f"Bad channels in {task_desc}"
        if list_tasks:
            subjects = sorted({subject for subject, _ in bad_channels})
            t = fmtxt.Table('l' * (1 + len(use_tasks)), caption=caption)
            t.cells('Subject', *use_tasks)
            t.midrule()
            for subject in subjects:
                t.cell(subject)
                for task in use_tasks:
                    t.cell(', '.join(bad_channels[subject, task]))
        else:
            if use_tasks:
                caption += " (all tasks equal)"
            t = fmtxt.Table('ll', caption=caption)
            t.cells('Subject', 'Bad channels')
            t.midrule()
            for subject in sorted(bad_channels):
                t.cells(subject, ', '.join(bad_channels[subject]))
        return t

    def show_dependencies(
            self,
            name: str,
            options: dict[str, Any] | None = None,
            *,
            max_line_length: int | None = None,
            return_str: bool = False,
            **state,
    ) -> str | None:
        """Show the dependency tree for one registered input or derivative.

        Parameters
        ----------
        name
            Registered dependency node name, for example ``'evoked'``,
            ``'test-result'`` or ``'fwd'``.
        options
            Load options for the requested node.
        max_line_length
            Maximum line length for the formatted tree. By default, infer the
            current terminal width.
        return_str
            Return the formatted tree instead of printing it.
        ...
            State parameters for resolving the requested node.
        """
        state_ = self._derivative_state(state)
        options_ = {} if options is None else dict(options)
        tree = self._derivatives.dependency_tree(name, state=state_, options=options_, max_line_length=max_line_length)
        if return_str:
            return tree
        print(tree)
        return None

    def show_raw_info(self, **state):
        """Display the selected pipeline for raw processing

        See Also
        --------
        show_subjects : list presence of raw input file by subject
        """
        raw = self.get('raw', **state)
        pipe = source_pipe = self._raw[raw]
        pipeline = [pipe]
        while not isinstance(source_pipe, RawSource):
            source_pipe = self._raw[source_pipe.source]
            pipeline.insert(0, source_pipe)
        print(f"Preprocessing pipeline: {' --> '.join(p.name for p in pipeline)}")

        # pipe-specific
        if isinstance(pipe, RawICA):
            rows = []
            for subject in self:
                try:
                    ica = self.load_ica()
                    rows.append((subject, ica.n_components_, len(ica.exclude)))
                except FileMissingError:
                    path = self._bids_path.copy()
                    if all(path.copy().update(task=task).fpath.exists() for task in pipe.task):
                        rows.append((subject, "No ICA-file", -1))
                    else:
                        rows.append((subject, "No data", -1))

            n_selected = [row[-1] for row in rows]
            mark_unselected = any(n_selected) and not all(n_selected)

            table = fmtxt.Table('lrr' + 'r' * mark_unselected)
            table.cells('Subject', 'n components', 'reject')
            if mark_unselected:
                table.cell('*')
            table.midrule()
            for subject, n, n_selected in rows:
                table.cells(subject, n)
                if not isinstance(n, str):
                    table.cell(n_selected)
                    if mark_unselected and n_selected == 0:
                        table.cell('*')
                table.endline()
            print()
            print(table)

    def show_reg_params(self, asds=False, **kwargs):
        """Show the covariance matrix regularization parameters

        Parameters
        ----------
        asds : bool
            Return a dataset with the parameters (default False).
        ...
            State parameters.
        """
        if kwargs:
            self.set(**kwargs)
        subjects = []
        reg = []
        for subject in self:
            handle = self._derivatives.resolve(cov_node_name(self.get('cov')), state=self._derivative_state())
            path = handle.artifact_path.with_suffix('.info.txt')
            if exists(path):
                with open(path) as fid:
                    text = fid.read()
                reg.append(float(text.strip()))
            else:
                reg.append(float('nan'))
            subjects.append(subject)
        ds = Dataset()
        ds['subject'] = Factor(subjects)
        ds['reg'] = Var(reg)
        if asds:
            return ds
        else:
            print(ds)

    def show_rej_info(self, flagp=None, asds=False, bads=False, **state):
        """Information about artifact rejection

        Parameters
        ----------
        flagp : scalar
            Flag entries whose percentage of good trials is lower than this
            number.
        asds : bool
            Return a Dataset with the information (default is to print it).
        bads : bool
            Display bad channel names (not just number of bad channels).

        See Also
        --------
        .show_raw_info : Display the number of ICA components rejected
        """
        # TODO: include ICA raw preprocessing pipes
        if state:
            self.set(**state)
        raw_name = self.get('raw')
        epoch_name = self.get('epoch')
        rej_name = self.get('rej')
        rej = self._artifact_rejection[rej_name]
        has_epoch_rejection = rej['kind'] is not None
        has_interp = rej.get('interpolation')

        # format bad channels
        if bads:
            bads_fmt = ', '.join
        else:
            bads_fmt = len

        bads_in_rej = False
        subjects = []
        n_events = []
        n_good = []
        bad_chs = []
        n_interp = []
        for subject in self:
            subjects.append(subject)
            try:
                bads_raw = self.load_bad_channels()
            except FileMissingError:  # raw file is missing
                bad_chs.append(('NaN', 'NaN'))
                if has_epoch_rejection:
                    n_good.append(float('nan'))
                if has_interp:
                    n_interp.append(float('nan'))
                n_events.append(np.nan)
                continue

            try:
                ds = self.load_selected_events(reject='keep')
            except FileMissingError:  # rejection file is missing
                ds = self.load_selected_events(reject=False)
                bad_chs.append((bads_fmt(bads_raw), 'NaN'))
                if has_epoch_rejection:
                    n_good.append(float('nan'))
                if has_interp:
                    n_interp.append(float('nan'))
            else:
                bads_rej = set(ds.info[BAD_CHANNELS]).difference(bads_raw)
                if bads_rej:
                    bads_in_rej = True
                bad_chs.append((bads_fmt(bads_raw), bads_fmt(bads_rej)))
                if has_epoch_rejection:
                    n_good.append(ds['accept'].sum())
                if has_interp:
                    n_interp.append(np.mean([len(chi) for chi in ds[INTERPOLATE_CHANNELS]]))
            n_events.append(ds.n_cases)
        has_interp = has_interp and any(n_interp)
        caption = f"Rejection info for raw={raw_name}, epoch={epoch_name}, rej={rej_name}. Percent is rounded to one decimal."

        if bads_in_rej:
            caption += " Bad channels: defined in bad_channels file and in rej-file."
            bad_chs = [f'{bads_raw} + {bads_rej}' for bads_raw, bads_rej in bad_chs]
        else:
            bad_chs = [f'{bads_raw}' for bads_raw, bads_rej in bad_chs]

        if bads:
            bad_chs = [s.replace('MEG ', '') for s in bad_chs]

        if has_interp:
            caption += " ch_interp: average number of channels interpolated per epoch, rounded to one decimal."
        out = Dataset(caption=caption)
        out['subject'] = Factor(subjects)
        out['n_events'] = Var(n_events)
        if has_epoch_rejection:
            out['n_good'] = Var(n_good)
            out['percent'] = Var(np.round(100 * out['n_good'] / out['n_events'], 1))
        if flagp:
            out['flag'] = Factor(out['percent'] < flagp, labels={False: '', True: '*'})
        out['bad_channels'] = Factor(bad_chs)
        if has_interp:
            out['ch_interp'] = Var(np.round(n_interp, 1))

        if asds:
            return out
        else:
            print(out)

    def show_subjects(
            self,
            raw: bool = False,
            mri: bool = None,
            mrisubject: bool = False,
            caption: str | bool = True,
            asds: bool = False,
            **state,
    ):
        """Create a Dataset with subject information

        Parameters
        ----------
        raw
            Display which raw input files exist.
        mri
            Add a column specifying whether the subject is using a scaled MRI
            or whether it has its own MRI.
        mrisubject
            Add a column showing the MRI subject corresponding to each subject.
        caption
            Caption for the table (For True, use the default "Subject in group
            {group}".
        asds
            Return the table as Dataset instead of an FMTxt Table.
        ...
            State parameters.
        """
        if isinstance(mri, str):
            state['mri'] = mri
            mri = True
        elif mri is None:
            mri = exists(mri_sdir(self._derivative_state()))
        if state:
            self.set(**state)

        # caption
        if caption is True:
            caption = self.format("Subjects in group {group}")

        subject_list = []
        mri_list = []
        mrisubject_list = []
        raw_list = []
        datatype = self.get('datatype')
        suffix = self.get('suffix')
        for subject in self.iter():
            subject_list.append(subject)
            mrisubject_ = self.get('mrisubject')
            mrisubject_list.append(mrisubject_)
            if raw:
                query = BIDSPath(
                    subject=subject,
                    datatype=datatype,
                    suffix=suffix,
                    root=self.root,
                )
                matches = query.match()
                basenames = [match.basename for match in matches]
                raw_list.append(', '.join(basenames))
            if mri:
                subject_mri_dir = str(mri_dir(self._derivative_state()))
                if not exists(subject_mri_dir):
                    mri_list.append('*missing')
                elif is_fake_mri(subject_mri_dir):
                    subjects_dir = str(mri_sdir(self._derivative_state()))
                    info = mne.coreg.read_mri_cfg(mrisubject_, subjects_dir)
                    cell = f"{info['subject_from']} * {info['scale']!s}"
                    mri_list.append(cell)
                else:
                    mri_list.append(mrisubject_)

        ds = Dataset(caption=caption)
        ds['subject'] = Factor(subject_list)
        if mri:
            ds['mri'] = Factor(mri_list)
        if mrisubject:
            ds['mrisubject'] = Factor(mrisubject_list)
        if raw:
            ds['raw_files'] = Factor(raw_list)

        if asds:
            return ds
        else:
            return ds.as_table(midrule=True, count=True)

    def _surfer_plot_kwargs(self, surf=None, views=None, foreground=None, background=None, smoothing_steps=None, hemi=None):
        out = self._brain_plot_defaults.copy()
        out.update(self.brain_plot_defaults)
        if views:
            out['views'] = views
        else:
            parc, p = self._get_parc()
            if p is not None and p.views:
                out['views'] = p.views

        if surf:
            out['surf'] = surf
        if foreground:
            out['foreground'] = foreground
        if background:
            out['background'] = background
        if smoothing_steps:
            out['smoothing_steps'] = smoothing_steps
        if hemi:
            out['hemi'] = hemi
        return out

    def _update_bids_path(self):
        keys = {
            k: v for k, v in self._fields.items()
            if (k in BIDS_PATH_KEYS) and v and ('*' not in v)
        }
        self._bids_path.update(**keys)
