# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pipeline class to manage data from an experiment"""
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
import copy
from datetime import datetime
from itertools import product
import logging
import os
from os.path import exists
from pathlib import Path
from typing import Any, Literal

import numpy as np
import mne
import mne_bids
from mne_bids import find_matching_paths, get_entity_vals

from .. import fmtxt
from .. import gui
from .. import load
from .. import plot
from .. import save
from .._data_obj import CellArg, Datalist, Dataset, Factor, Var, NDVar, SourceSpace, VolumeSourceSpace, assert_is_legal_dataset_key, combine
from .._exceptions import ConfigurationError, DimensionMismatchError
from .._info import BAD_CHANNELS, INTERPOLATE_CHANNELS
from .._meeg import new_rejection_ds
from ..mne_fixes import suppress_mne_warning
from .._ndvar import concatenate, neighbor_correlation
from .._stats.testnd import NDTest
from .._text import enumeration
from .._types import PathArg
from .._utils import ask, keydefaultdict, log_level, ScreenHandler
from .._utils.mne_utils import is_fake_mri
from .covariance import CovDerivative, EpochCovariance, RawCovariance
from .derivative_cache import DerivativeRegistry, ProtectedArtifactError, Request, _format_size
from .configuration import Configuration, ConfigurationDict, sequence_arg
from .epochs import (
    EpochBase, EpochsDerivative, RecordingEpochsDerivative, EvokedDerivative,
    EvokedGroupDatasetDerivative, PrimaryEpoch, SecondaryEpoch,
    SuperEpoch, assemble_epochs, decim_param,
)
from .epoch_rejection import ChannelModelRejection, ChannelModelRejectionDerivative, EpochRejection, ManualRejection, RejectionInput
from .events import EpochEventsDerivative, EventsDerivative, EventsInput, LabeledEventsDerivative, SelectedEventsDerivative
from .exceptions import FileMissingError, ICAChannelsChangedError
from .logging import CACHE_EVENT_COLUMNS, StructuredFormatter
from .state_model import StateModel
from .statistics.nodes import ROITestResult
from .groups import assemble_groups
from .pathing import (
    LOG_DIR, MRI_SDIR, bids_path, join_stem_parts, mri_dir, raw_basename,
    src_file_path, trans_file_path,
)
from .parc import SEEDED_PARC_RE, AnnotDerivative, CombinationParc, EelbrainParc, FreeSurferParc, FSAverageParc, IndividualSeededParc, LabelParc, Parcellation, SeededParc, VolumeParc, _resolve_parc
from .preprocessing import (
    CachedRawPipe, ICAInput, MaxwellCalibrationInput, MaxwellCrosstalkInput, CanonicalHeadPositionDerivative, RawBadChannelsInput, RawDerivative, RawHeadPositionDerivative, RawPipe, RawSource, RawSourceDerivative, RawSourceInput, RawICA, RawMaxwell, Reference,
    REINDEX_ICA, assemble_raw_pipes, ica_input_name, raw_bad_channels_input_name, raw_node_name, raw_input_name,
)
from .data import DataSpec
from .source import (
    BemInput, EpochsStcDerivative,
    EvokedStcDerivative, EvokedStcGroupDatasetDerivative, FwdDerivative,
    InvDerivative, ROIData, SourceMorphDerivative, SrcDerivative, TransInput,
    MinimumNormInverseSolution, _drop_unknown_labels, _source_parc, eval_src,
)
from .statistics import EvokedTestDataDerivative, TestResultDerivative, TwoStageDataDerivative, TwoStageLevel1Derivative, TwoStageLevel2Derivative, TwoStageTest
from .statistics.config import Test, validate_tests
from .trf import Boosting, Estimator, Model, NUTSPredictor, PredictorInput, TRFDatasetDerivative, TRFDerivative, TRFGroupDatasetDerivative, TRFJob, TRFJobSpec, UTSPredictor, filter_predictor
from .trf.model import parse_term
from .variable_def import Variables, apply_vardef, label_groups


# Allowable parameters
COV_PARAMS = {'epoch', 'method', 'reg', 'keep_sample_mean', 'reg_eval_win_pad'}
# Argument types
BaselineArg = bool | tuple[float | None, float | None]
DataArg = str | DataSpec
PMinArg = Literal['tfce'] | float | None
SubjectArg = str | Literal[1, -1]


def _session_log_file(log_dir: Path, name: str, initialized: datetime) -> Path:
    """Determine the log-file name for a new pipeline session."""
    date = initialized.strftime('%Y-%m-%d')
    prefix = f'{name}-{date}-'
    sessions = []
    if log_dir.exists():
        for path in log_dir.iterdir():
            if path.suffix != '.log' or not path.stem.startswith(prefix):
                continue
            session, separator, time = path.stem[len(prefix):].partition('-')
            if separator and session.isdecimal() and len(time) == 4 and time.isdecimal():
                sessions.append(int(session))
    session = max(sessions, default=0) + 1
    time = initialized.strftime('%H%M')
    return log_dir / f'{prefix}{session}-{time}.log'


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
    screen_log_level: str | int = logging.INFO
    cache_inv: bool = True  # Whether to cache inverse solution
    # moderate speed gain for loading source estimates (34 subjects: 20 vs 70 s)
    # hard drive space ~ 100 mb/file
    # Whether to persist sensor-space epochs per recording to disk
    cache_epochs: bool = False

    # datatype and extension are usually inferred from a BIDS dataset; override here if needed
    datatype: str = None
    extension: str = None
    # default sensor-space data kind for analyses (load_test/load_trf) when ``data`` is
    # unspecified and the analysis is in sensor space; ``None`` infers from datatype
    # ('eeg' for EEG, 'meg' for MEG).
    default_data: str = None

    ignore_entities: dict[str, list[str]] = {}
    preload: bool = False

    # Raw preprocessing pipeline
    raw: dict[str, RawPipe] = {}

    # Load events from a subset of available stim channels
    stim_channel: str | Sequence[str] = None
    # merge adjacent events in the stimulus channel
    merge_triggers: int = None
    # add this value to all trigger times (in seconds); global shift, or {subject: shift, (subject, session): shift} dictionary
    trigger_shift: float | dict[str | tuple[str, str], float] = 0

    # variables for automatic labeling {name: {trigger: label, triggers: label}}
    variables: dict[str, Any] = {}
    # Cache the output of label_events(). Set to False if label_events() reads
    # from external files whose changes should trigger cache invalidation.
    cache_event_labels: bool = True

    # named epochs
    epochs: dict[str, EpochBase] = {}

    # predictors for TRF models, selected through the 'code' argument of
    # load_predictor (e.g. {'gammatone': UTSPredictor(resample='bin')})
    predictors: dict[str, Configuration] = {}

    # estimators for TRF fitting, selected through the 'estimator' argument of
    # load_trf (e.g. {'ncrf': NCRF()}); 'boosting' (Boosting()) is always available
    estimators: dict[str, Estimator] = {}
    # named TRF models, for abbreviations in model strings passed to load_trf
    models: dict[str, str] = {}
    # events Dataset column(s) identifying the stimulus for file predictors; a
    # single name, or a {key: column} mapping for multiple stimuli per event
    stim_var: str = 'stimulus'

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
    # epoch_rejection: named EpochRejection configurations, selected through the 'epoch_rejection' state.
    epoch_rejection = {}

    # references: named Reference configurations, selected through the
    # 'reference' state. Applied to epochs after channel interpolation (EEG only).
    # A built-in 'average' entry (Reference('average')) is always available and
    # can be overridden here (e.g. to reconstruct an implicit reference channel
    # with Reference('average', add='Cz')).
    references = {}

    # groups can be defined as subject lists: {'group': ('member1', 'member2', ...)}
    # or by exclusion: {'group': {'base': 'all', 'exclude': ('member1', 'member2')}}
    groups = {}

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

    # specify defaults for specific fields (e.g. specify the initial subject)
    defaults = {}

    # model order: list of factors in the order in which models should be built
    # (default for factors not in this list is alphabetic)
    _model_order = []

    # Tests
    # -----
    tests: dict[str, Test] = {}

    # plotting
    # --------
    _brain_plot_defaults = {'surf': 'inflated'}
    brain_plot_defaults = {}

    def __init__(
            self,
            root: PathArg,
            screen_log_level: str | int = None,
            **state,
    ):
        ########################################################################
        # Checks
        ########
        if root is None:
            raise TypeError("Pipeline subclasses must have root.")
        root = Path(root).expanduser().absolute()
        self.root: Path = root

        # BIDS entities
        ignore_entities = copy.deepcopy(self.ignore_entities)
        ignore_tasks = ignore_entities.get('ignore_tasks', [])
        if 'noise' not in ignore_tasks:
            ignore_entities['ignore_tasks'] = [*ignore_tasks,  'noise']

        self._subjects = tuple(get_entity_vals(root, 'subject', **ignore_entities))
        self._sessions = tuple(get_entity_vals(root, 'session', **ignore_entities))
        self._tasks = tuple(get_entity_vals(root, 'task', **ignore_entities))
        self._runs = tuple(get_entity_vals(root, 'run', **ignore_entities))
        if self.datatype is not None:
            if self.datatype not in ('meg', 'eeg'):
                raise ConfigurationError(f"`datatype` must be 'meg' or 'eeg', not {self.datatype!r}.")
            if not isinstance(self.extension, str):
                raise TypeError(f"{self.__class__.__name__}.extension={self.extension!r} with {self.__class__.__name__}.datatype={self.datatype!r}; extension needs to be specified (e.g., '.fif').")
            datatype = self.datatype
            extensions = (self.extension,)
        else:
            datatypes = tuple(mne_bids.get_datatypes(root))
            if 'meg' in datatypes and 'eeg' in datatypes:
                raise ConfigurationError(f"Can't infer datatype. Both MEG and EEG data found in {root}.")
            elif 'meg' in datatypes:
                datatype = 'meg'
                extensions = ('.fif',)
            elif 'eeg' in datatypes:
                datatype = 'eeg'
                data_extensions = {path.extension for path in mne_bids.find_matching_paths(root, datatypes='eeg', suffixes='eeg', extensions=['.edf', '.vhdr', '.set', '.bdf', '.fif'])}
                if len(data_extensions) == 0:
                    raise FileMissingError(f"No EEG data files found in {root}.")
                elif len(data_extensions) > 1:
                    raise ConfigurationError(f"Multiple EEG data file types found in {root}: {enumeration(sorted(data_extensions))}.")
                extensions = tuple(data_extensions)
            else:
                raise ConfigurationError(f"Can't infer datatype. No MEG or EEG data found in {root}.")
        self._raw_extension = extensions[0]
        self._datatype = datatype

        acquisitions = tuple(get_entity_vals(root, 'acquisition', **ignore_entities))
        matching_paths = tuple(
            path
            for path in find_matching_paths(root, subjects=self._subjects, sessions=self._sessions, tasks=self._tasks, datatypes=datatype, suffixes=datatype, extensions=extensions, ignore_nosub=True)
            if not path.acquisition or path.acquisition in acquisitions
        )
        self._acquisitions = tuple(sorted({path.acquisition or '' for path in matching_paths}))

        # Recordings index: existing (subject, session, task, acquisition, run) combinations of source
        # recordings, from a single find_matching_paths scan. Scoped to the raw datatype /
        # suffix / extension and to ``sub-*`` directories (ignore_nosub) so it never
        # descends into ``derivatives`` / ``sourcedata`` (where non-BIDS names would fail
        # to parse). Absent entities are recorded as ''. Snapshot at init time; used for
        # recording-existence checks in preprocessing nodes and for the
        # per-(subject, session, task, acquisition) run lists below.
        self._recordings: frozenset[tuple[str, str, str, str, str]] = frozenset(
            (path.subject or '', path.session or '', path.task or '', path.acquisition or '', path.run or '')
            for path in matching_paths
        )
        # Per-(subject, session, task, acquisition) run lists; used for combine-all epoch aggregation.
        # Runs can vary by subject.
        runs_seen: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        if self._runs:
            for subject, session, task, acquisition, run in self._recordings:
                runs_seen[(subject, session, task, acquisition)].add(run)
        self._runs_for: dict[tuple[str, str, str, str], list[str]] = {key: sorted(runs) for key, runs in runs_seen.items()}

        StateModel.__init__(self)

        ########################################################################
        # Logger
        ########
        # log-file
        # A dedicated Logger instance per experiment (not via getLogger, which returns a
        # singleton keyed by name): keeps each instance's handlers and log level isolated,
        # even for two live experiments on the same root. ``parent`` is None, so records
        # never propagate to the root logger (no double-logging via host configuration).
        self._log = log = logging.Logger(self.__class__.__name__, logging.DEBUG)
        initialized = datetime.now()
        log_file = _session_log_file(root / LOG_DIR, self.__class__.__name__, initialized)
        os.makedirs(log_file.parent, exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = StructuredFormatter("%(levelname)-8s %(asctime)s %(message)s", "%m-%d %H:%M")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        log.addHandler(handler)

        # terminal log
        handler = ScreenHandler()
        self._screen_log_level = log_level(screen_log_level or self.screen_log_level)
        handler.setLevel(self._screen_log_level)
        log.addHandler(handler)
        self._screen_log_handler = handler

        ########################################################################
        # Experiment arguments
        ######################
        # groups
        self._groups = ConfigurationDict('group', assemble_groups(self.groups, set(self._subjects)))

        # preprocessing
        self._raw = assemble_raw_pipes({'raw': RawSource(), **self.raw}, self._tasks)

        # variables
        self._variables = Variables(self.variables)
        self._variables._check_trigger_vars()

        # epochs
        self._epochs = ConfigurationDict('epoch', assemble_epochs(self.epochs, self._tasks))

        # epoch rejection; 'manual' is always available, '' selects no rejection
        epoch_rejection: dict[str, EpochRejection | None] = {'': None}
        for name, rejection in self.epoch_rejection.items():
            if not isinstance(name, str):
                raise TypeError(f"epoch_rejection[{name!r}]: name must be a string")
            elif not name:
                raise ValueError(f"epoch_rejection[{name!r}]: name can't be empty")
            elif not isinstance(rejection, EpochRejection):
                raise TypeError(f"epoch_rejection[{name!r}]={rejection!r}: need EpochRejection")
            epoch_rejection[name] = rejection
        self._epoch_rejection = ConfigurationDict('epoch_rejection', epoch_rejection)

        # epoch re-referencing; 'average' is always available and user-overridable
        references = {'': None, 'average': Reference('average')}
        for name, reference in self.references.items():
            if not isinstance(name, str):
                raise TypeError(f"references[{name!r}]: name must be a string")
            elif not name:
                raise ValueError(f"references[{name!r}]: name can't be empty")
            elif not isinstance(reference, Reference) or isinstance(reference, RawPipe):
                raise TypeError(f"references[{name!r}]={reference!r}: need Reference")
            elif name == 'average':
                if reference.reference != 'average':
                    raise ConfigurationError(f"references[{name!r}]={reference!r}: the standard average reference must be an average reference")
                elif reference.drop:
                    raise ConfigurationError(f"references[{name!r}]={reference!r}: the standard average reference can not drop channels")
            references[name] = reference
        self._references = ConfigurationDict('reference', references)

        # mri_subjects
        self._mri_subjects = self.mri_subjects.copy()

        # Sensor noise covariance estimates
        self._covs = ConfigurationDict('covariance', self._covs)
        for name, cov in self._covs.items():
            if not isinstance(cov, (RawCovariance, EpochCovariance)):
                raise TypeError(f"_covs[{name!r}]={cov!r}: need RawCovariance or EpochCovariance")
            cov._store_name(name)

        # parcellations
        # make : can be made if non-existent
        # morph_from_fraverage : can be morphed from fsaverage to other subjects
        for name, parc in self.parcs.items():
            if not isinstance(parc, Parcellation):
                raise TypeError(f"parcs[{name!r}]={parc!r}: need Parcellation")
        self._parcs = ConfigurationDict('parcellation', {**self._default_parcs, **self.parcs})
        for name, parc in self._parcs.items():
            parc._store_name(name)
        parc_values = [*self._parcs.keys(), '']

        # tests
        validate_tests(self.tests)
        for test_obj in self.tests.values():
            if test_obj.model:
                test_obj.model = self._eval_model(test_obj.model)
        self.tests = ConfigurationDict('test', self.tests)

        # TRF: named models, estimators, predictors, stimulus variables
        self._named_models: dict[str, Model] = ConfigurationDict('model')
        for name, value in self.models.items():
            self._named_models[name] = Model.coerce(value).initialize(self._named_models)
        estimators = {'boosting': Boosting(), **self.estimators}
        for name, estimator in estimators.items():
            if not isinstance(estimator, Estimator):
                raise TypeError(f"estimators[{name!r}]={estimator!r}: need Estimator")
            estimator._store_name(name)
        self._estimators = ConfigurationDict('estimator', estimators)
        self.predictors = ConfigurationDict('predictor', self.predictors)
        if not isinstance(self.stim_var, str):
            raise TypeError(f"{self.__class__.__name__}.stim_var={self.stim_var!r}")

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

        # Register BIDS entity fields
        self._register_field('subject', self._subjects, repr=True)
        self._register_field('session', self._sessions or None, repr=True)
        self._register_field('task', self._tasks, depends_on=('epoch',), slave_handler=self._update_task, repr=True)
        self._register_field('acquisition', self._acquisitions, repr=True, allow_empty=True, depends_on=('epoch', 'subject', 'session', 'task'), slave_handler=self._update_acquisition)
        self._register_field('run', self._runs, repr=True, depends_on=('epoch', 'subject', 'session', 'task', 'acquisition'), slave_handler=self._update_run)
        self._register_field('equalize_evoked_count', ('', 'eq'), allow_empty=True)
        self._register_field('common_brain', ('fsaverage',))

        self._register_field('mri', sorted(self._mri_subjects), allow_empty=True)
        self._register_field('group', self._groups.keys(), 'all', post_set_handler=self._post_set_group)

        # raw
        raw_default = sorted(self.raw)[0] if self.raw else None
        self._register_field('raw', sorted(self._raw), default=raw_default, repr=True)
        self._register_field('epoch_rejection', self._epoch_rejection.keys(), allow_empty=True)
        self._register_field('reference', self._references.keys(), allow_empty=True)

        # cov
        self._register_field('cov', sorted(self._covs))
        # inv determines the analysis space: a non-empty inverse means source space, inv='' means sensor space.
        self._register_field('inv', default='', eval_handler=self._eval_inv, allow_empty=True)
        # default sensor-space data kind for analyses (see .default_data)
        if self.default_data is None:
            self._default_data = 'eeg' if datatype == 'eeg' else 'meg'
        else:
            data = DataSpec(self.default_data)
            if not data.sensor or data.string.split('.', 1)[0] == 'sensor':
                raise ConfigurationError(f"{self.__class__.__name__}.default_data={self.default_data!r}; must be a specific sensor type ('mag', 'grad', or 'eeg').")
            self._default_data = self.default_data
        self._register_field('parc', parc_values, 'aparc', eval_handler=self._eval_parc, allow_empty=True)
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
        log.info("*** %s initialized with root %s on %s ***", self.__class__.__name__, root, initialized.strftime('%Y-%m-%d %H:%M:%S'))
        level = logging.DEBUG if any('dev' in v for v in (__version__, mne.__version__)) else logging.INFO
        log.log(level, "Using eelbrain %s, mne %s.", __version__, mne.__version__)
        # Legend for the tab-separated columns appended to cache-event log lines (DEBUG, file only).
        log.debug("Cache-event columns (tab-separated after the message): %s", '\t'.join(CACHE_EVENT_COLUMNS))

        # set initial values
        self.set(**state)
        self._store_state()

    def _repr_args(self) -> tuple[str, ...]:
        return (str(self.root),)

    def _init_derivative_registry(self):
        self._derivatives = DerivativeRegistry(self.root, self._log, self._datatype)
        result_args = (
            self.tests,
            self._epochs,
            self._parcs,
            self._groups,
        )
        # --- Inputs (externally managed files) and preprocessing ---
        maxwell_registered = False
        for raw_name, pipe in self._raw.items():
            if isinstance(pipe, RawSource):
                raw_input = RawSourceInput(raw_name, pipe, self._raw_extension)
                self._derivatives.register(raw_input)
                self._derivatives.register(RawBadChannelsInput(raw_name, pipe, self._raw_extension))
                self._derivatives.register(RawSourceDerivative(raw_name, pipe, self._raw_extension))
                self._derivatives.register(RawHeadPositionDerivative(raw_input.name))
                self._derivatives.register(CanonicalHeadPositionDerivative(self._recordings, self._tasks, self._runs))
            elif isinstance(pipe, CachedRawPipe):
                self._derivatives.register(RawDerivative(raw_name, pipe, self._raw, self._raw_extension))
                if isinstance(pipe, RawICA):
                    self._derivatives.register(ICAInput(raw_name, pipe, self._recordings, self._runs))
                elif isinstance(pipe, RawMaxwell) and not maxwell_registered:
                    self._derivatives.register(MaxwellCalibrationInput())
                    self._derivatives.register(MaxwellCrosstalkInput())
                    maxwell_registered = True
            else:
                raise TypeError(f"Unknown raw pipe {pipe}")
        self._derivatives.register(TransInput())
        self._derivatives.register(BemInput())
        self._derivatives.register(RejectionInput(self.root, self._epoch_rejection, self._epochs))
        self._derivatives.register(ChannelModelRejectionDerivative(self._epochs, self._epoch_rejection))

        # --- Predictors and TRFs ---
        self._derivatives.register(PredictorInput(self.root, self.predictors))
        self._derivatives.register(TRFDerivative(self.root, self._estimators, self.predictors, self.stim_var, self._raw))
        self._derivatives.register(TRFDatasetDerivative(self.root, self._estimators, self._epochs))
        self._derivatives.register(TRFGroupDatasetDerivative(self._mri_subjects, self._groups))

        # --- Sensor-space: events → epochs → evoked ---
        self._derivatives.register(EventsInput(self._raw_extension))
        self._derivatives.register(EventsDerivative(
            self.trigger_shift,
            sequence_arg(f'{self.__class__.__name__}.stim_channel', self.stim_channel),
            self.merge_triggers,
            self.preload,
            self.fix_events,
            self.__class__.__name__,
        ))
        self._derivatives.register(LabeledEventsDerivative(
            self.label_events,
            self.__class__.__name__,
            len(self._tasks) > 1,
            len(self._sessions) > 1,
            self._variables,
            self._groups,
            self.cache_event_labels,
        ))
        self._derivatives.register(SelectedEventsDerivative(self._epochs, self._epoch_rejection))
        self._derivatives.register(EpochEventsDerivative(self._epochs, self._runs_for))
        self._derivatives.register(RecordingEpochsDerivative(self._raw, self._epochs, self._references, self.cache_epochs))
        self._derivatives.register(EpochsDerivative(self._raw, self._epochs, self._runs_for))
        self._derivatives.register(EvokedDerivative(self._raw, self._epochs))
        self._derivatives.register(EvokedGroupDatasetDerivative(self._raw, self._groups))

        # --- Source-space infrastructure ---
        self._derivatives.register(CovDerivative(self._covs, self._raw, self._references, self._recordings))
        self._derivatives.register(SrcDerivative())
        self._derivatives.register(SourceMorphDerivative())
        self._derivatives.register(FwdDerivative(self._raw, self._references, self._recordings))
        self._derivatives.register(InvDerivative(self._raw, self._references, self._recordings, self.cache_inv))
        self._derivatives.register(AnnotDerivative(self._parcs))

        # --- Source-space: epochs/evoked projected to source space ---
        self._derivatives.register(EpochsStcDerivative(self._raw, self._epochs, self._references))
        self._derivatives.register(EvokedStcDerivative(self._raw, self._epochs, self._references))
        self._derivatives.register(EvokedStcGroupDatasetDerivative(self._mri_subjects, self._groups))

        # --- Statistical tests ---
        self._derivatives.register(EvokedTestDataDerivative(self.tests, self._epochs, self._groups))
        self._derivatives.register(TwoStageDataDerivative(self.tests, self._epochs, self._groups))
        self._derivatives.register(TwoStageLevel1Derivative(self.tests))
        self._derivatives.register(TestResultDerivative(*result_args))
        self._derivatives.register(TwoStageLevel2Derivative(*result_args))

    def _resolve_derivative(
            self,
            name: str,
            options: dict[str, Any] | None = None,
            controls: frozenset[str] | set[str] | tuple[str, ...] = (),
    ) -> Request:
        return self._derivatives.resolve(name, state=self.state, options=options, controls=controls)

    def _load_derivative(
            self,
            name: str,  # Registered derivative name.
            options: dict[str, Any] | None = None,
            view: str | None = None,
            *,
            controls: frozenset[str] | set[str] | tuple[str, ...] = (),
    ) -> Any:
        return self._resolve_derivative(name, options=options, controls=controls).load(view=view)

    def clean_cache(
            self,
            dry_run: bool = False,
            delete: bool = False,
            revalidate: bool = True,
    ) -> fmtxt.Table | None:
        """Report and delete invalid or stale cache files (garbage collection).

        Parameters
        ----------
        dry_run
            Only scan and report; delete nothing.
        delete
            Delete stale files without asking for confirmation.
        revalidate
            Detect stale artifacts by re-validating each cached request
            against the current pipeline configuration. Set to ``False`` for a
            faster scan restricted to structurally invalid files.

        Returns
        -------
        report_table
            Per-category summary of the scan (file counts and sizes).
        """
        report = self._derivatives.scan_cache(revalidate=revalidate)
        deletable = report.deletable()
        total_size = report.total_size()
        table = report.summary()
        if report.errors:
            self._log.debug("Cache scan errors:\n%s", '\n'.join(f"{path}: {error}" for path, error in report.errors))
        if dry_run or not deletable:
            return table
        print(table)
        while not delete:
            command = ask(
                f"Delete {len(deletable)} cache files ({_format_size(total_size)})?",
                {'delete': 'permanently delete the listed files', 'list': 'List files that will be deleted', 'abort': 'keep everything'},
                help="Deleted artifacts are rebuilt automatically when they are requested again. Files categorized as unverifiable or unknown are always kept.",
            )
            if command == 'list':
                print(report.file_table())
            elif command == 'delete':
                delete = True
            else:
                return None
        report.collect()
        return None

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        return self.iter()

    def _process_subject_arg(
            self,
            subjects: SubjectArg | None,
            kwargs: dict[str, str],
    ) -> tuple[str | None, str | None]:
        """Determine subject or group for analysis and update state

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

    def get_field_values(
            self,
            field: str,
            exclude: Iterable[str] = (),
            **state,
    ) -> list[str]:
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

    def iter_range(
            self,
            start: str | None = None,
            stop: str | None = None,
            field: str = 'subject',
    ) -> Iterator[str]:
        """Iterate through a range on a field with ordered values.

        Parameters
        ----------
        start
            Start value (inclusive). With ``None``, begin at the first value.
        stop
            Stop value (inclusive). With ``None``, end with the last value.
        field
            Name of the field.

        Returns
        -------
        Iterator[str]
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

    def fix_events(self, ds: Dataset) -> Dataset:
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
        timing. This method only applies to events derived from M/EEG raw data files,
        and not to events from BIDS ``events.tsv`` sidecar files,
        and is called *before* adding other variables.

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

    def label_events(self, ds: Dataset) -> Dataset:
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
                    # assign 'yes' to events where value 2 follows value 1
                    for i in range(1, ds.n_cases):
                        if ds[i, 'value'] == 2 and ds[i-1, 'value'] == 1:
                            ds[i, 'new'] = 'yes'
                    return ds

        Add events based on separate files. This assumes that the events in
        the recording only indicate trial onsets, and separate files contain
        events listed relative to these trial onsets::

            class Experiment(Pipeline):

                def label_events(self, ds):
                    samplingrate = ds.info['raw.samplingrate']
                    new_events = []
                    # loop through trials
                    for sample, value in ds.zip('sample', 'value'):
                        # load the event file, assuming that the trigger value
                        # in the data was used to indicate the trial ID
                        trial_events = load.tsv(f'/files/trial_{value}.txt')
                        # assuming trial_events has a column called 'onset' (in
                        # seconds), we infer the event's sample in the raw file
                        trial_sample = sample + trial_events['onset'] * samplingrate
                        trial_events['sample'] = Var(trial_sample.astype(int))
                        # events also need a value column
                        trial_events[:, 'value'] = value
                        # collect all trials
                        new_events.append(trial_events)
                    # combine the trials to a single dataset
                    return combine(new_events)
        """
        return ds

    def label_subjects(self, ds: Dataset) -> None:
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
        return label_groups(subject, groups, self._groups)

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
        self.set(**state)
        return self._load_derivative('annot')

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
        return self._load_derivative(raw_node_name(raw_name), options={'noise': noise}, view='bads')

    def load_cov(self, **kwargs):
        """Load the covariance matrix

        Parameters
        ----------
        ...
            State parameters.
        """
        return self._load_derivative('cov', **kwargs)

    def _resolve_data(
            self,
            data: DataArg,
    ) -> DataSpec:
        """Resolve the ``data`` argument into a :class:`DataSpec` for analysis.

        :class:`DataSpec` parses the value; this method supplies the default for
        ``data=None`` and checks the parsed *space* against the ``inv`` state
        (``inv=''`` → sensor, non-empty → source). Must be called after
        ``**state`` has been applied so that ``self.get('inv')`` is current.

        Parameters
        ----------
        data
            Data kind: ``None`` (the datatype default in the current space), a
            sensor type (``'meg'``/``'mag'``/``'grad'``/``'eeg'``) or
            ``'source'``, optionally with a ``'.mean'``/``'.rms'`` aggregation.
        """
        source_space = bool(self.get('inv'))
        if data is None:
            data = 'source' if source_space else self._default_data
        spec = DataSpec.coerce(data)
        if source_space and not spec.source:
            raise ValueError(f"data={data!r} is sensor-space data, but the analysis is in source space (inv={self.get('inv')!r}); set inv='' for sensor-space analysis")
        if not source_space and spec.source:
            raise ValueError(f"data={data!r} is source-space data, but the analysis is in sensor space (inv=''); set a non-empty inverse for source-space analysis")
        if spec.sensor and spec.string.split('.', 1)[0] == 'sensor':
            raise ValueError(f"data={data!r}: 'sensor' selects all sensor types; specify a single type ('mag', 'grad', 'eeg') for analysis")
        return spec

    @suppress_mne_warning
    def load_epochs(
            self,
            baseline: BaselineArg = True,
            ndvar: bool | str = True,
            reject: bool | Literal['keep'] = True,
            samplingrate: int = None,
            decim: int = None,
            pad: float = 0,
            tmin: float = None,
            tmax: float = None,
            tstop: float = None,
            interpolate_bads: Literal[True, False, 'keep'] = False,
            src_baseline: BaselineArg = False,
            morph: bool = None,
            keep_mne: bool = False,
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
            Apply baseline correction using this period. ``True`` (default) to
            use the epoch's baseline specification; ``False`` to not apply
            baseline correction.
        ndvar
            Data to convert to :class:`NDVar`. ``True`` (default) converts all
            sensor types (with keys ``'mag'``/``'grad'``/``'eeg'`` …); a sensor
            type (``'meg'``/``'mag'``/``'grad'``/``'eeg'``), optionally aggregated
            (e.g. ``'eeg.rms'``/``'eeg.mean'``), returns a single :class:`NDVar`;
            ``False`` returns :class:`mne.Epochs` with key ``'epochs'``. In source
            space (``inv`` set) the source estimates are returned as ``'src'``.
        reject
            Reject bad trials. If ``True`` (default), bad trials are removed
            from the Dataset. Set to ``False`` to ignore the trial rejection.
            Set ``reject='keep'`` to load the rejection (added it to the events
            as ``'accept'`` variable), but keep bad trails.
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        pad : scalar
            Pad the epochs with this much time (in seconds; e.g. for spectral
            analysis).
        tmin
            Override the epoch's ``tmin`` parameter (sensor space only).
        tmax
            Override the epoch's ``tmax`` parameter (sensor space only).
        tstop
            Override the epoch's ``tmax`` parameter as exclusive ``tstop``
            (sensor space only).
        interpolate_bads
            Interpolate channels marked as bad for the whole recording (useful
            when comparing topographies across subjects; default ``False``;
            sensor space only). ``True`` interpolates and includes those channels
            in the output; ``'keep'`` interpolates but leaves the channels marked
            as bad (so they remain excluded from NDVar output).
        src_baseline
            Apply baseline correction in source space using this period (source
            space only; ``True`` to use the epoch's baseline specification).
        morph
            Morph source estimates to the common brain (source space only;
            default ``False``, except when loading multiple subjects with
            ``ndvar=True``).
        keep_mne
            Also include the underlying :class:`mne.Epochs` (sensor space) or
            sensor-space data (source space) in the returned :class:`Dataset`.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-epoch_rejection`: which trials to use
             - :ref:`state-inv`: inverse solution (``inv=''`` for sensor space,
               a non-empty inverse for source space)

        """
        if self.get('inv', **state):  # source space
            if interpolate_bads or tmin is not None or tmax is not None or tstop is not None:
                raise ValueError("interpolate_bads/tmin/tmax/tstop are not available for source-space epochs; set inv='' for sensor space")
            if isinstance(ndvar, str):
                raise ValueError(f"{ndvar=}: a data-kind ndvar is only valid for sensor-space epochs; in source space use ndvar=True or ndvar=False")
            self._current_source_parc()
            options = {
                'baseline': baseline,
                'src_baseline': src_baseline,
                'keep_epochs': keep_mne,
                'morph': morph,
                'samplingrate': samplingrate,
                'decim': decim,
                'pad': pad,
                'ndvar': ndvar,
                'reject': reject,
            }
            return self._load_derivative('epochs-stc', options=options)

        # sensor space
        if isinstance(ndvar, str):
            data = self._resolve_data(ndvar)
            node_ndvar = 'both' if keep_mne else True
        elif ndvar:
            data = DataSpec('sensor')
            node_ndvar = 'both' if keep_mne else True
        else:
            data = DataSpec('sensor')
            node_ndvar = False
        options = {
            'baseline': baseline,
            'ndvar': node_ndvar,
            'reject': reject,
            'samplingrate': samplingrate,
            'decim': decim,
            'pad': pad,
            'data': data,
            'tmin': tmin,
            'tmax': tmax,
            'tstop': tstop,
            'interpolate_bads': bool(interpolate_bads),
            'reset_bads': interpolate_bads == True,
        }
        return self._load_derivative('epochs', options=options)

    def load_events(self, **state) -> Dataset:
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window

        """
        self.set(**state)
        return self._load_derivative('labeled-events')

    def load_predictor(
            self,
            code: str,
            tstep: float = 0.01,
            n_samples: int = None,
            tmin: float = None,
            filter_x: bool | Literal['continuous'] = False,
            name: str = None,
            **state,
    ) -> NDVar:
        """Load a file predictor as an :class:`NDVar`

        Reads the predictor file's relevant data and shapes it into a predictor
        on the requested time axis. Only file predictors
        (:class:`UTSPredictor`, :class:`NUTSPredictor`) can be loaded directly;
        an :class:`EventPredictor` is generated from the data and is only
        available through :meth:`load_trf`.

        Parameters
        ----------
        code
            Code of the predictor to load, using the pattern
            ``{stimulus}~{predictor}``. The ``predictor`` part selects a
            definition in :attr:`predictors`; additional ``-`` delimited items
            specify columns or a NUTS representation method (see
            :class:`NUTSPredictor`).
        tstep
            Time-step for the predictor (for :class:`NDVar` predictors the
            original ``tstep`` is used by default; for :class:`Dataset`
            predictors ``tstep`` determines the sampling of the output).
        n_samples
            Number of samples in the predictor (the default returns all
            available samples).
        tmin
            First sample time stamp (default is all available data).
        filter_x
            Filter the predictor with the same filters as the M/EEG data (i.e.
            the :class:`RawFilter` pipes of the current ``raw`` pipeline).
            ``True`` to filter all predictors; ``'continuous'`` to filter only
            time-continuous predictors (those with ``sampling='continuous'``,
            see :class:`FilePredictorBase`).
        name
            Reassign the name of the predictor :class:`NDVar`.
        ...
            State parameters.
        """
        self.set(**state)
        term = parse_term(code)
        predictor = self.predictors[term.predictor_key]
        if not isinstance(predictor, (UTSPredictor, NUTSPredictor)):
            raise NotImplementedError(f"{term.string}: load_predictor only supports file predictors; load {type(predictor).__name__} through load_trf")
        contents = self._load_derivative('predictor', options={'code': code})
        x = predictor._generate(contents, tmin, tstep, n_samples, term)
        x = filter_predictor(x, self._raw, self.get('raw'), filter_x)
        x.name = term.string if name is None else name
        return x

    def _trf_options(
            self,
            x: str,
            tstart: float,
            tstop: float,
            estimator: str,
            data: str | None,
            mask: str | None,
            samplingrate: int | None,
            filter_x: bool | str,
            state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize parameters for TRF nodes"""
        if state:
            self.set(**state)
        if mask is not None:
            raise NotImplementedError(f"{mask=}: source-space masking is not implemented yet")
        # Resolve the data kind against the analysis space (inv state) and the estimator.
        est = self._estimators[estimator]
        if est.requires_sensor_space:
            if inv := self.get('inv'):
                raise ValueError(f"{inv=} for {estimator=}: {estimator} uses sensor data and localizes internally; set inv='' for sensor-space analysis")
            elif data is not None:
                raise ValueError(f"{data=}: estimator {estimator!r} uses all sensor data; leave data unset")
            data_string = 'sensor'
        else:
            data_string = self._resolve_data(data).string
        model = Model.coerce(x).initialize(self._named_models).sorted()
        return {'x': model, 'tstart': float(tstart), 'tstop': float(tstop), 'estimator': estimator, 'data': data_string, 'mask': mask, 'samplingrate': samplingrate, 'filter_x': filter_x}

    def load_trf(
            self,
            x: str,
            tstart: float = 0.,
            tstop: float = 0.5,
            *,
            estimator: str = 'boosting',
            data: str = None,
            mask: str = None,
            samplingrate: int = None,
            filter_x: bool | Literal['continuous'] = False,
            path_only: bool = False,
            **state,
    ):
        """Load (or compute) the TRF for a model and the current subject

        Parameters
        ----------
        x
            Model (e.g. ``'gammatone + word'``).
        tstart
            Start of the TRF in seconds.
        tstop
            Stop of the TRF in seconds.
        estimator
            Name of the estimator in :attr:`estimators` (default ``'boosting'``).
            Estimator-specific parameters (``basis``, ``delta``, ``mu``, …) are
            set on the :class:`~eelbrain._experiment.trf.Estimator` object.
        data
            Sensor-space data *kind* to fit: a sensor type
            (``'meg'``/``'mag'``/``'grad'``/``'eeg'``), optionally aggregated
            (e.g. ``'eeg.rms'``/``'eeg.mean'``). The default (``None``) uses
            :attr:`default_data`. The analysis *space* is set by the ``inv``
            state (``inv=''`` for sensor space; a non-empty inverse for source
            space); in source space leave ``data`` unset. NCRF requires
            ``inv=''`` and leaves ``data`` unset.
        mask
            Parcellation to mask source-space data (not implemented yet).
        samplingrate
            Samplingrate in Hz for the analysis.
        filter_x
            Filter predictors like the M/EEG data (see :meth:`load_predictor`).
        path_only
            Return the path to the cache file instead of loading the TRF.
        ...
            State parameters.
        """
        options = self._trf_options(x, tstart, tstop, estimator, data, mask, samplingrate, filter_x, state)
        ctx = self._resolve_derivative('trf', options=options)
        if path_only:
            return ctx.artifact_path
        return ctx.load()

    def _trf_job_spec(
            self,
            x: str,
            tstart: float = 0.,
            tstop: float = 0.5,
            *,
            estimator: str = 'boosting',
            data: str = None,
            mask: str = None,
            samplingrate: int = None,
            filter_x: bool | Literal['continuous'] = False,
            **state,
    ) -> TRFJobSpec:
        "Host-side handle for one TRF fit (generate job, check whether done, save result)"
        options = self._trf_options(x, tstart, tstop, estimator, data, mask, samplingrate, filter_x, state)
        ctx = self._resolve_derivative('trf', options=options)
        return TRFJobSpec(ctx)

    def load_trf_job(
            self,
            x: str,
            tstart: float = 0.,
            tstop: float = 0.5,
            *,
            estimator: str = 'boosting',
            data: str = None,
            mask: str = None,
            samplingrate: int = None,
            filter_x: bool | Literal['continuous'] = False,
            **state,
    ) -> TRFJob:
        """Load the data and return a picklable :class:`TRFJob` for fitting this TRF elsewhere

        The returned job carries the M/EEG response and regressors, so it can be
        pickled and executed on a machine without access to the raw data. This is
        useful for distributed fitting and for inspecting the exact data used to
        estimate a model.

        Parameters
        ----------
        x
            Model (e.g. ``'gammatone + word'``).
        tstart
            Start of the TRF in seconds.
        tstop
            Stop of the TRF in seconds.
        estimator
            Name of the estimator in :attr:`estimators` (default ``'boosting'``).
        data
            Sensor-space data *kind* to fit (see :meth:`load_trf`).
        mask
            Parcellation to mask source-space data (not implemented yet).
        samplingrate
            Samplingrate in Hz for the analysis.
        filter_x
            Filter predictors like the M/EEG data (see :meth:`load_predictor`).
        ...
            State parameters.
        """
        return self._trf_job_spec(x, tstart, tstop, estimator=estimator, data=data, mask=mask, samplingrate=samplingrate, filter_x=filter_x, **state).make_job()

    def load_trfs(
            self,
            subjects: SubjectArg,
            x: str,
            tstart: float = 0.,
            tstop: float = 0.5,
            *,
            estimator: str = 'boosting',
            data: str = None,
            mask: str = None,
            samplingrate: int = None,
            filter_x: bool | Literal['continuous'] = False,
            scale: Literal['original'] = None,
            smooth: float = None,
            trfs: bool = True,
            **state,
    ) -> Dataset:
        """Load TRFs for a group (or subject) as a :class:`Dataset`

        Assembles the per-subject TRFs (see :meth:`load_trf`) into a group-level
        :class:`Dataset` with one case per subject (× member epoch for an
        :class:`EpochCollection`), holding the estimator's fit-quality metrics
        and the TRF kernels. Source-space data is morphed to the common brain so
        that subjects are comparable.

        Parameters
        ----------
        subjects : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject name or a
            group name such as ``'all'``. ``1`` to use the current subject;
            ``-1`` for the current group.
        x
            Model (e.g. ``'gammatone + word'``).
        tstart
            Start of the TRF in seconds.
        tstop
            Stop of the TRF in seconds.
        estimator
            Name of the estimator in :attr:`estimators` (default ``'boosting'``).
        data
            Response data to fit (see :meth:`load_trf`).
        mask
            Parcellation to mask source-space data (not implemented yet).
        samplingrate
            Samplingrate in Hz for the analysis.
        filter_x
            Filter predictors like the M/EEG data (see :meth:`load_predictor`).
        scale : 'original'
            Rescale the TRFs to the scale of the source data (the default is the
            scale based on normalized predictors and responses).
        smooth
            Smooth the TRFs and metric maps in space (STD of the Gaussian kernel
            in [m]; only for source data).
        trfs
            Include the TRF kernels. Set ``False`` to load only the fit metrics.
        ...
            State parameters.

        Returns
        -------
        trf_ds
            Dataset with ``subject``, ``epoch``, the estimator's fit metrics, and
            one :class:`NDVar` per TRF component. ``trf_ds.info['xs']`` lists the
            TRF component keys.
        """
        subject, group = self._process_subject_arg(subjects, state)
        trf_options = self._trf_options(x, tstart, tstop, estimator, data, mask, samplingrate, filter_x)
        options = {**trf_options, 'scale': scale, 'smooth': smooth, 'trfs': trfs}
        if group is not None:
            ds = self._load_derivative('trf-group-dataset', options=options)
        else:
            ds = self._load_derivative('trf-dataset', options=options)
        return ds

    def load_evoked(
            self,
            subjects: str | int = None,
            baseline: BaselineArg = True,
            ndvar: bool | str = True,
            cat: Sequence[CellArg] = None,
            samplingrate: int = None,
            decim: int = None,
            interpolate_bads: bool = False,
            src_baseline: BaselineArg = False,
            morph: bool = None,
            keep_mne: bool = False,
            model: str = '',
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
            Apply baseline correction using this period. ``True`` (default) to
            use the epoch's baseline specification; ``False`` to not apply
            baseline correction.
        ndvar
            Data to convert to :class:`NDVar`. ``True`` (default) converts all
            sensor types (with keys ``'mag'``/``'grad'``/``'eeg'`` …); a sensor
            type (``'meg'``/``'mag'``/``'grad'``/``'eeg'``), optionally aggregated
            (e.g. ``'eeg.rms'``/``'eeg.mean'``), returns a single :class:`NDVar`;
            ``False`` returns the :class:`mne.Evoked` objects as ``'evoked'``. In
            source space (``inv`` set) the source estimates are returned as
            ``'src'``.
        cat
            Only load data for these cells (cells of model).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        decim
            Data decimation factor (alternative to ``samplingrate``).
        interpolate_bads
            Interpolate channels marked as bad (useful when comparing topographies
            across subjects; default ``False``; sensor space only).
        src_baseline
            Apply baseline correction in source space using this period (source
            space only; ``True`` to use the epoch's baseline specification).
        morph
            Morph source estimates to the common brain (source space only;
            default ``False``, except when loading multiple subjects with
            ``ndvar=True``).
        keep_mne
            Also include the underlying :class:`mne.Evoked` (sensor space) or
            sensor-space data (source space) in the returned :class:`Dataset`.
        model
            How to group trials into conditions before averaging (e.g.
            ``'condition'`` or ``'a % b'``). The default (``''``) is the grand
            average across all trials.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch`: which events to use and time window
             - :ref:`state-epoch_rejection`: which trials to use
             - :ref:`state-equalize_evoked_count`: control number of trials per cell
             - :ref:`state-inv`: inverse solution (``inv=''`` for sensor space,
               a non-empty inverse for source space)

        Notes
        -----
        Channel interpolation: Bad channels are always interpolated. When
        loading data for a single subject, bad channels are marked as
        bad/excluded. When loading group level data, datasets are merged using
        interpolated data.
        """
        subject, group = self._process_subject_arg(subjects, state)
        model = self._eval_model(model)
        if inv := self.get('inv'):  # source space
            if interpolate_bads:
                raise ValueError("interpolate_bads not available for source-space data; set inv='' for sensor space")
            if isinstance(ndvar, str):
                raise ValueError(f"{ndvar=} with {inv=}: a data-kind ndvar is only valid for sensor-space evoked; in source space use ndvar=True or ndvar=False")
            self._current_source_parc()
            options = {
                'model': model,
                'baseline': baseline,
                'src_baseline': src_baseline,
                'cat': cat,
                'keep_evoked': keep_mne,
                'morph': morph,
                'samplingrate': samplingrate,
                'decim': decim,
                'ndvar': ndvar,
            }
            if group is not None:
                return self._load_derivative('evoked-stc-group-dataset', options=options)
            return self._load_derivative('evoked-stc', options=options)

        # sensor space
        if isinstance(ndvar, str):
            data = self._resolve_data(ndvar)
            node_ndvar = 'both' if keep_mne else True
        elif ndvar:
            data = DataSpec('sensor')
            node_ndvar = 'both' if keep_mne else True
        else:
            data = DataSpec('sensor')
            node_ndvar = False
        epoch_name = self.get('epoch')
        epoch = self._epochs[epoch_name]
        if baseline is True:
            baseline = epoch.baseline
        options = {
            'model': model,
            'baseline': baseline,
            'ndvar': node_ndvar,
            'cat': cat,
            'samplingrate': samplingrate,
            'decim': decim,
            'interpolate_bads': interpolate_bads,
            'data': data,
        }
        if group is not None:
            # Group data is merged in a common sensor space, so bad channels are always interpolated (the interpolate_bads argument only controls single-subject loads).
            options['interpolate_bads'] = True
            return self._load_derivative('evoked-group-dataset', options=options)
        return self._load_derivative('evoked', options=options)

    def load_fwd(
            self,
            surf_ori: bool = True,
            ndvar: bool = False,
            **state,
    ) -> mne.forward.Forward | NDVar:
        """Load the forward solution

        Parameters
        ----------
        surf_ori
            Force surface orientation (default True; only applies if
            ``ndvar=False``, :class:`NDVar` forward operators are always
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
        self.set(**state)
        fwd = self._load_derivative('fwd')
        if ndvar:
            src = self.get('src')
            parc = self._current_source_parc()
            if parc:
                self.make_annot()
            fwd = load.mne.forward_operator(fwd, src, self.root / MRI_SDIR, parc, adjacency=False)
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
        raw_name = self.get('raw', **state)
        ica_raw_name = self._raw.ica_name(raw_name)
        return self._derivatives.resolve(
            ica_input_name(ica_raw_name),
            state={**self.state, 'raw': ica_raw_name},
            controls={REINDEX_ICA} if accept_stale else (),
        ).load()

    def load_inv(
            self,
            ndvar: bool = False,
            **state,
    ) -> mne.minimum_norm.InverseOperator | NDVar:
        """Load the inverse operator

        Parameters
        ----------
        ndvar
            Return the inverse operator as NDVar (default is
            :class:`mne.minimum_norm.InverseOperator`). The NDVar representation
            does not take into account any direction selectivity (loose/free
            orientation) or noise normalization properties.
        ...
            Applicable :ref:`state-parameters`:

             - :ref:`state-raw`: preprocessing pipeline
             - :ref:`state-epoch_rejection`: which trials to use
             - :ref:`state-cov`: covariance matrix for inverse solution
             - :ref:`state-src`: source space
             - :ref:`state-inv`: inverse solution

        """
        self.set(**state)
        inv = self._load_derivative('inv')

        if ndvar:
            parc = self._current_source_parc()
            inv = load.mne.inverse_operator(inv, self.get('src'), self.root / MRI_SDIR, parc)
            if parc:
                inv = _drop_unknown_labels(inv)
        return inv

    def load_label(
            self,
            label: str,
            **kwargs,
    ) -> mne.Label | mne.BiHemiLabel:
        """Retrieve a label as mne Label object

        Parameters
        ----------
        label : str
            Name of the label. If the label name does not end in '-lh' or '-rh'
            the combination of the labels ``label + '-lh'`` and
            ``label + '-rh'`` is returned.
        ...
            State parameters.
        """
        labels = self.load_annot(**kwargs)
        labels = {label.name: label for label in labels}
        if label in labels:
            return labels[label]
        elif not label.endswith(('-lh', '-rh')):
            return labels[label + '-lh'] + labels[label + '-rh']
        else:
            raise ValueError(f"Label {label!r} could not be found in parc {self.get('parc')!r}.")

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
        self.set(**state)
        return self._load_derivative('source-morph')

    def load_neighbor_correlation(
            self,
            subjects: SubjectArg = None,
            epoch: str = None,
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
            lines = [(subject, self.load_neighbor_correlation(1, epoch)) for subject in self]
            return Dataset.from_caselist(['subject', 'nc'], lines)
        if epoch:
            if epoch is True:
                epoch = self.get('epoch')
            epoch_params = self._epochs[epoch]
            if len(epoch_params.tasks) != 1:
                raise ValueError(f"{epoch=}: epoch has multiple tasks")
            ds = self.load_epochs(epoch=epoch, reject=False, decim=1, **state)
            key = ds.info['sensor_types'][0]
            data = concatenate(ds[key])
        else:
            data = self.load_raw(ndvar=True, **state)
        n_corr = neighbor_correlation(data)
        if return_data:
            return data, n_corr
        else:
            return n_corr

    def load_raw(
            self,
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
        raw = self._load_derivative(raw_node_name(raw_name), options={'preload': preload, 'noise': noise})
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
            source_pipe = self._raw.root_source_pipe(raw_name)
            data = DataSpec('sensor')
            data_kind = data.find_ndvar_channel_types(raw.info)[0]
            sysname = source_pipe._get_sysname(raw.info, self.get('subject'), data_kind)
            adjacency = source_pipe._get_adjacency(data_kind)
            raw = load.mne.raw_ndvar(raw, sysname=sysname, adjacency=adjacency)

        return raw

    def _current_source_parc(self) -> str:
        """Ensure valid parc setting in state"""
        return _source_parc(self.state)

    def load_selected_events(
            self,
            subjects: SubjectArg = None,
            reject: bool | Literal['keep'] = True,
            vardef: str = None,
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
        vardef
            Name of a test defining additional variables to add to the returned
            Dataset.
        ...
            State parameters.

        Notes
        -----
        When trial rejection is set to automatic, not rejection is performed
        because no epochs are loaded.
        """
        if reject not in (True, False, 'keep'):
            raise ValueError(f"{reject=}")
        state = dict(kwargs)
        subject, group = self._process_subject_arg(subjects, state)

        if group is not None:
            return combine([self.load_selected_events(subjects=subject_, reject=reject, vardef=vardef, **state) for subject_ in self.iter(group=group)])
        elif subject is None:
            raise RuntimeError(f"{subject=}, {group=}")

        options = {'reject': reject}
        ds = self._load_derivative('epoch-events', options=options)
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
        self.set(**state)
        src_spaces = self._load_derivative('src')
        if ndvar:
            src = self.get('src')
            subjects_dir = self.root / MRI_SDIR
            mri_subject = self.get('mrisubject')
            if src.startswith('vol'):
                return VolumeSourceSpace.from_file(subjects_dir, mri_subject, src)
            parc = self.get('parc')
            return SourceSpace.from_file(subjects_dir, mri_subject, src, parc)
        if add_geom:
            return mne.read_source_spaces(self.root / src_file_path(self._fields), add_geom)
        return src_spaces

    def load_test(
            self,
            test: str,
            tstart: float = None,
            tstop: float = None,
            pmin: PMinArg = None,
            disconnect_labels: bool = False,
            samples: int = 10000,
            data: str = None,
            baseline: BaselineArg = True,
            smooth: float = None,
            src_baseline: BaselineArg = None,
            samplingrate: int = None,
            return_data: bool = False,
            **state,
    ) -> NDTest | ROITestResult | tuple[Dataset | ROIData, NDTest | ROITestResult]:
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
            Data *kind* to test (the analysis *space* is set by the ``inv``
            state: ``inv=''`` for sensor space, a non-empty inverse for source
            space). The default (``None``) uses :attr:`default_data` in sensor
            space, or the full source estimates in source space. Examples:

            - ``None`` with ``inv`` set: spatio-temporal test in source space.
            - ``None`` with ``inv=''``: spatio-temporal test in sensor space
              (using :attr:`default_data`, e.g. ``'meg'`` or ``'eeg'``).
            - ``'source.mean'`` with ``inv`` set: ROI mean time course.
            - ``'eeg.rms'`` with ``inv=''``: RMS across the EEG sensors.
            - ``'eeg'`` with ``inv=''``: spatio-temporal test of EEG sensors.

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
        ...
            State parameters (Use the ``group`` state parameter to select the
            subject group for which to perform the test).

        Returns
        -------
        ds : Dataset | ROIData
            Data that forms the basis of the test (for ROI tests, a
            ``{roi: dataset}`` dictionary).
        res : NDTest | ROITestResult
            Test result for the specified test (for ROIs tests,
            an :class:`~_experiment.ROITestResult` object).
        """
        test_obj = self.tests[test]
        self.set(**state)
        data = self._resolve_data(data)
        if data.source:
            self._current_source_parc()
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
        result_node = 'two-stage-level-2' if isinstance(test_obj, TwoStageTest) else 'test-result'
        result = self._load_derivative(result_node, options=options)
        if not return_data:
            return result
        elif isinstance(test_obj, TwoStageTest):
            raise NotImplementedError("Data for two-stage test")
        data_options = {key: value for key, value in options.items() if key != 'disconnect_labels'}
        data = self._load_derivative('evoked-test-data', options=data_options)
        return data, result

    def make_annot(self, **state) -> None:
        """Ensure that annot files for the current parcellation exist."""
        self.set(**state)
        self._resolve_derivative('annot').ensure()

    def make_bad_channels(
        self,
        bad_chs: Sequence[str] | str | int = (),
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
        """
        raw_name = self.get('raw', **kwargs)
        source_name = self._raw.root_source_name(raw_name)
        if isinstance(bad_chs, (str, int)):
            bad_chs = (bad_chs,)
        raw = self._load_derivative(raw_input_name(source_name), options={'noise': noise})
        bads_ctx = self._resolve_derivative(raw_bad_channels_input_name(source_name), options={'noise': noise})
        bads_ctx.node.write(bads_ctx, raw, bad_chs, redo, create=True)

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
        source_name = self._raw.root_source_name('raw')
        pipe = self._raw[source_name]
        raw_ctx = self._resolve_derivative(raw_input_name(source_name), options={'noise': noise, 'preload': True})
        raw = raw_ctx.load()
        bads_ctx = self._resolve_derivative(raw_bad_channels_input_name(source_name), options={'noise': noise})
        bids_path = raw_ctx.node._resolve_bids_path(raw_ctx)
        detected = pipe._detect_flat_channels(bids_path, raw, flat)
        if detected is not None:
            bads_ctx.node.write(bads_ctx, raw, detected, redo, create=True)

    def make_bad_channels_neighbor_correlation(
            self,
            r: float,
            epoch: str = None,
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
        data, full_nc = self.load_neighbor_correlation(1, epoch, return_data=True, **state)
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
        path = self.make_ica(**state)  # sets raw to ica-raw
        # display data
        subject = self.get('subject')
        ica_name = self.get('raw')
        pipe = self._raw.ica_pipe(ica_name)
        bads = self._load_derivative(raw_node_name(ica_name), options={'noise': False}, view='bads')
        labeled_events = None
        if epoch is None:
            if task is None:
                task = pipe.task
            else:
                task = sequence_arg('task', task)
            ctx = self._resolve_derivative(ica_input_name(ica_name))
            raw = ctx.node.load_concatenated_source_raw(ctx, task)
            decim = decim_param(samplingrate, decim, None, raw.info, minimal=True)
            info = raw.info
            display_data = raw
            # labeled events for the timeline; concatenate across tasks and shift
            # each task's onsets by the same offset used to append the raws
            if task:
                event_dss = []
                offset = 0.0  # seconds into the concatenated recording
                with self._temporary_state:
                    for state in ctx.node._source_states(ctx, task):
                        ds_t = self.load_events(raw=pipe.source, **state)
                        ds_t['onset'] = ds_t['onset'] + offset
                        event_dss.append(ds_t)
                        offset += (ds_t.info['raw.last_samp'] - ds_t.info['raw.first_samp'] + 1) / ds_t.info['raw.samplingrate']
                labeled_events = combine(event_dss, incomplete='fill in')
        elif task is not None:
            raise TypeError(f"{task=} with {epoch=}")
        else:
            with self._temporary_state:
                ds = self.load_epochs(ndvar=False, epoch=epoch, reject=False, raw=pipe.source, samplingrate=samplingrate, decim=decim)
                epochs = ds['epochs']
                if isinstance(epochs, Datalist):
                    for epoch_ in epochs:
                        epoch_.info['bads'] = bads
                else:
                    epochs.info['bads'] = bads
            if isinstance(ds['epochs'], Datalist):  # variable-length epoch
                data = np.concatenate([epoch.get_data()[0] for epoch in ds['epochs']], axis=1)  # n_epochs, n_channels, n_times
                raw = mne.io.RawArray(data, ds[0, 'epochs'].info)
                mne_events = mne.make_fixed_length_events(raw)
                ds = Dataset({'epochs': mne.Epochs(raw, mne_events, 1, 0, 1, baseline=None, proj=False, preload=True)})
            info = ds['epochs'].info
            decim = None
            display_data = ds
        data = DataSpec('sensor')
        data_kind = data.find_ndvar_channel_types(info)[0]
        source_pipe = self._raw.root_source_pipe(ica_name)
        sysname = source_pipe._get_sysname(info, subject, data_kind)
        adjacency = source_pipe._get_adjacency(data_kind)
        try:
            frame = gui.select_components(path, display_data, sysname, adjacency, decim, debug, events=labeled_events)
        except DimensionMismatchError as error:
            # The sensors no longer match those the ICA was estimated on, which
            # in this context means the bad channels have changed.
            raise ICAChannelsChangedError(path) from error
        return frame

    def make_bad_channels_selection(
            self,
            raw: str = None,
            **state,
    ):
        """GUI for selecting bad channels in continuous M/EEG recordings

        Opens :func:`eelbrain.gui.select_channels` for the current subject.
        The document is the Pipeline-specific ``*_channels.tsv`` file under
        the ``derivatives/mne/`` hierarchy (seeded from the BIDS source the
        first time it is written). Events come from labeled-events.

        Parameters
        ----------
        raw
            Which raw pipeline stage to display.  Defaults to the source raw.
        ...
            State parameters (e.g. ``subject``).
        """
        if raw is not None:
            state['raw'] = raw
        if state:
            self.set(**state)
        raw_name = self.get('raw')
        source_name = self._raw.root_source_name(raw_name)
        subject = self.get('subject')
        # Load raw at the requested pipeline stage (unprocessed input if source)
        raw_data = self._load_derivative(raw_node_name(raw_name), options={'preload': False, 'noise': False})
        # Bad channels are stored in the derivatives/mne hierarchy; ensure the
        # file exists (seeded from the BIDS source) so the GUI can read/write it
        bads_ctx = self._resolve_derivative(raw_bad_channels_input_name(source_name))
        bads_ctx.node.write(bads_ctx, raw_data, [], redo=False, create=True)
        channels_path = bads_ctx.node.path(bads_ctx)
        # Labeled events for the timeline
        events = self._load_derivative('labeled-events')
        # Sensor system info
        source_pipe = self._raw.root_source_pipe(raw_name)
        data_kind = DataSpec('sensor').find_ndvar_channel_types(raw_data.info)[0]
        sysname = source_pipe._get_sysname(raw_data.info, subject, data_kind)
        adjacency = source_pipe._get_adjacency(data_kind)
        return gui.select_channels(raw_data, channels_path, events=events, sysname=sysname, adjacency=adjacency)

    def make_ica(self, **state) -> Path:
        """Compute ICA decomposition for a :class:`pipeline.RawICA` preprocessing step

        Parameters
        ----------
        ...
            State parameters.

        Returns
        -------
        path : Path
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
        raw_name = self.get('raw', **state)
        ica_raw_name = self._raw.ica_name(raw_name)
        if ica_raw_name != raw_name:
            self.set(raw=ica_raw_name)
            print(f"raw: {raw_name} -> {ica_raw_name}")
        ctx = self._resolve_derivative(ica_input_name(ica_raw_name))
        try:
            ctx.node.materialize(ctx)
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
                ctx.node.materialize(ctx, allow_protected_overwrite=True)
            elif command == 'incorporate':
                ctx.node.materialize(ctx, allow_protected_reindex=True)
            elif command != 'abort':
                raise RuntimeError(f"{command=}")
            else:
                raise RuntimeError("User aborted ICA overwrite")
        return self._raw[ica_raw_name].path(ctx)

    def make_epoch_rejection(
            self,
            samplingrate: int | None = None,
            auto: float | dict | None = None,
            overwrite: bool | None = None,
            decim: int | None = None,
            **state,
    ):
        """Open :func:`gui.select_epochs` for the current epoch rejection

        For a :class:`ManualRejection` the GUI is opened for editing (with the
        correct file name; an existing file is loaded and is the default save
        path). For an automatically generated rejection (e.g.
        :class:`ChannelModelRejection`) the rejection is computed/cached and the
        GUI is opened **read-only** for inspection.

        Parameters
        ----------
        samplingrate
            Samplingrate in Hz for the visualization (set to a lower value to
            improve GUI performance; the default is the epoch setting).
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
        rej = self.get('epoch_rejection', **state)
        rej_args = self._epoch_rejection[rej]
        if rej_args is None:
            raise ValueError(f"epoch_rejection={rej!r}; no epoch rejection configured")

        epoch = self._epochs[self.get('epoch')]
        if not isinstance(epoch, PrimaryEpoch):
            if isinstance(epoch, SecondaryEpoch):
                raise ValueError(f"The current epoch {epoch.name!r} inherits selections from {epoch.sel_epoch!r}. To access a rejection file for this epoch, call `e.set(epoch={epoch.sel_epoch!r})` and then call `e.make_epoch_rejection()` again.")
            elif isinstance(epoch, SuperEpoch):
                raise ValueError(f"The current epoch {epoch.name!r} inherits selections from these other epochs: {epoch.sub_epochs!r}. To access selections for these epochs, call `e.make_epoch_rejection(epoch=epoch)` for each.")
            else:
                raise ValueError(f"The current epoch {epoch.name!r} is not a primary epoch and inherits selections from other epochs. Generate trial rejection for these epochs.")

        if isinstance(rej_args, ChannelModelRejection):
            # automatically generated: build+cache the rejection, then inspect read-only
            rej_ctx = self._resolve_derivative('epoch-rejection-channel-model')
            rej_ctx.load()
            path = rej_ctx.node.path(rej_ctx)
            ds = self._load_derivative('epochs', options={'reject': False, 'ndvar': False})
            return gui.select_epochs(ds, 'epochs', trigger='value', path=path, read_only=True)
        elif not isinstance(rej_args, ManualRejection):
            raise NotImplementedError(f"make_epoch_rejection for {type(rej_args).__name__}")

        rej_ctx = self._resolve_derivative('epoch-rejection-input')
        path = rej_ctx.node.path(rej_ctx)
        path.parent.mkdir(parents=True, exist_ok=True)

        if auto is not None and overwrite is not True and path.exists():
            if overwrite is False:
                return
            elif overwrite is None:
                raise OSError(self.format("A rejection file already exists for {subject}, epoch {epoch}, rej {rej}. Set the overwrite parameter to specify how to handle existing files."))
            else:
                raise TypeError(f"{overwrite=}")

        if auto is not None:
            ds = self._load_derivative('epochs', options={'reject': False, 'ndvar': True})
            ch_types = ['meg', 'mag', 'grad', 'planar1', 'planar2', 'eeg']
            ch_types = [t for t in ch_types if t in ds]
            if not ch_types:
                raise RuntimeError("No data found")
            y_name = ch_types[0]

            auto_dict: dict[str, float]
            if isinstance(auto, dict):
                auto_dict = auto
                if missing := set(ch_types).difference(auto_dict):
                    raise ValueError(f"{auto=}: channel types {enumeration(missing)} missing")
                elif unknown := set(auto_dict).difference(ch_types):
                    raise ValueError(f"{auto=}: channel types {enumeration(unknown)} not in data")
            elif len(ch_types) == 1:
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
            rej_ds.info['desc'] = f"Created with {self.__class__.__name__}.make_epoch_rejection({', '.join(args)})"
            # save
            save.pickle(rej_ds, path)
            # print info
            n_rej = rej_ds.eval("sum(accept == False)")
            desc = self.format("{subject}, epoch {epoch}")
            self._log.info(f"make_epoch_rejection: {n_rej} of {rej_ds.n_cases} epochs rejected with threshold {auto} for {desc}")
            return

        ds = self._load_derivative('epochs', options={'reject': False, 'ndvar': False})
        # eog_sns = self._eog_sns.get(ds[y_name].sensor.sysname, ())
        # don't mark eog sns if it is bad
        # bad_channels = self.load_bad_channels()
        # eog_sns = [c for c in eog_sns if c not in bad_channels]
        return gui.select_epochs(ds, 'epochs', trigger='value', path=path)

    def next(self, field: str | Sequence[str] = 'subject'):
        """Change field to the next value

        Parameters
        ----------
        field
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

        state_ = self._fields
        subjects_dir = str(self.root / MRI_SDIR)
        if (not plot_on_scaled_common_brain) and is_fake_mri(self.root / mri_dir(state_)):
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

    def plot_brain(
            self,
            common_brain: bool = True,
            hemi: str = 'split',
            **brain_kwargs,
    ):
        """Plot the brain model

        Parameters
        ----------
        common_brain
            If the current mrisubject is a scaled MRI, use the common_brain
            instead.
        hemi
            Which hemispheres to plot (one of ``'lh' | 'rh' | 'both' | 'split'``).
        ... :
            :class:`~plot._brain_object.Brain` options as keyword arguments.
        """
        from ..plot._brain_object import Brain

        brain_args = self._surfer_plot_kwargs(hemi=hemi)
        brain_args.update(brain_kwargs)
        state_ = self._fields
        brain_args['subjects_dir'] = str(self.root / MRI_SDIR)

        # find subject
        if common_brain and is_fake_mri(self.root / mri_dir(state_)):
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
        state_ = self._fields
        fig = mne.viz.plot_alignment(raw.info, self.root / trans_file_path(state_, datatype=self._datatype), self.get('mrisubject'), self.root / MRI_SDIR, surfaces, meg=meg, dig=dig, interaction='terrain')
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
            model: str = '',
            **kwargs):
        """Plot evoked sensor data

        Parameters
        ----------
        subjects
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
        model
            How to group trials into conditions before averaging. The default
            (``''``) plots the grand average.
        ...
            State parameters.
        """
        subject, group = self._process_subject_arg(subjects, kwargs)
        source_inv = self.get('inv')  # source space requires a configured inverse
        if data is None:
            sns = True
            src = bool(source_inv)  # only plot source estimates if an inverse is configured
        else:
            data = DataSpec.coerce(data)
            sns, src = bool(data.sensor), bool(data.source)
            if src and not source_inv:
                raise ValueError(f"data={data.string!r}: no inverse is configured (inv=''); set inv to plot source estimates")
        # response NDVar key(s) for the sensor plots are named by a DataSpec
        sensor_data = data if isinstance(data, DataSpec) else DataSpec('sensor')
        model = self._eval_model(model)
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
                ds = self.load_evoked(baseline=baseline, model=model)
                y = sensor_data.response_key(ds)
                title = f"{subject} {epoch} {model_name}"
                p = plot.TopoButterfly(y, model or None, data=ds, axh=h, name=title, run=False)
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
        else:
            title = name or f"{group} {epoch} {model_name}"
            subject_arg = group

        if src:
            ds = self.load_evoked(subject_arg, baseline=baseline, keep_mne=sns, inv=source_inv, model=model)
            out = [ds]
            if model:
                x = ds.eval(model)
                ys = [ds['src'].mean(case=x == cell) for cell in x.cells]
            else:
                ys = [ds['src']]
            for y in ys:
                if is_volume_source_space:
                    plots = plot.GlassBrain.butterfly(y, w=2 * h, h=h, name=title)
                else:
                    plots = plot.brain.butterfly(y, w=2 * h, h=h, name=title)
                out.extend(plots)
            right_of = out[2]
        else:
            ds = self.load_evoked(subject_arg, baseline=baseline, inv='', model=model)
            out = [ds]
            right_of = None
        if sns:
            key = sensor_data.response_key(ds)
            p = plot.TopoButterfly(key, model or None, data=ds, axh=h, w=2.5 * h, name=title, right_of=right_of, run=run)
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
        kwargs = self._surfer_plot_kwargs(surf, views)
        brain = self.plot_brain(hemi=hemi, title=title, w=w, **kwargs)
        brain.add_label(label, alpha=0.75)
        return brain

    def plot_raw(self, decim=10, xlim=5, subtract_mean=False, **state):
        """Plot raw sensor data

        Parameters
        ----------
        decim : int
            Decimate data for faster plotting (default 10).
        xlim : scalar
            Number of seconds to display (default 5 s).
        subtract_mean : bool
            Subtract the mean from each channel (useful when plotting raw data
            recorded with DC offset).
        ...
            State parameters.

        See Also
        --------
        make_bad_channels_selection : interactive plor for raw data
        """
        raw = self.load_raw(ndvar=True, decim=decim, **state)
        state_ = self._fields
        name = join_stem_parts(raw_basename(state_, datatype=self._datatype), f'raw-{state_["raw"]}')
        if raw.info['meas'] == 'V':
            vmax = 1.5e-4
        elif raw.info['meas'] == 'B':
            vmax = 2e-12
        else:
            vmax = None
        if subtract_mean:
            raw -= raw.mean('time')
        return plot.TopoButterfly(raw, w=0, h=3, xlim=xlim, vmax=vmax, name=name)

    def set(self, subject: str = None, **state):
        """
        Set variable values.

        Parameters
        ----------
        subject
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        ...
            Other state parameters.
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
        StateModel.set(self, **state)
        if subject is not None:
            StateModel.set(self, subject=subject)

    def _post_set_group(self, _: str, group: str) -> None:
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
            depth: float = 0,
            pick_normal: bool = False,
            **state,
    ):
        """Set the type of inverse solution used for source estimation

        Parameters
        ----------
        ori
            Orientation constraint (one of
            ``'free' | 'fixed' | 'vec' | float ]0, 1]``;
            default ``'free'``;
            use a number between 0 and 1 to specify a loose constraint).

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
            depth: float = 0,
            pick_normal: bool = False,
    ):
        "Construct inv string from settings; see :meth:`.set_inv`"
        return MinimumNormInverseSolution(ori, snr, method, depth, pick_normal)._string()

    @staticmethod
    def _eval_inv(inv: str):
        if inv == '':  # sensor space
            return ''
        return MinimumNormInverseSolution._from_string(inv)._string()

    def _eval_model(self, model: str) -> str:
        if model == '':
            return model
        elif len(model) > 1 and '*' in model:
            raise ValueError(f"{model=}; To specify interactions, use '%' instead of '*'")

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

    def _update_mrisubject(self, fields: dict) -> str:
        subject = fields['subject']
        mri = fields['mri']
        if subject == '*' or mri == '*':
            return '*'
        mrisubject = self._mri_subjects[mri][subject]
        if mrisubject == self.get('common_brain') or mrisubject.startswith('sub-'):
            return mrisubject
        return 'sub-' + mrisubject

    def _update_task(self, fields: dict) -> str | None:
        epoch = fields['epoch']
        if epoch in self._epochs:
            epoch = self._epochs[epoch]
            return epoch.tasks[0]
        elif not epoch or epoch == '*':
            return  # don't force task
        return '*'  # if a named epoch is not in _epochs it might be a removed epoch

    def _update_run(self, fields: dict) -> str | None:
        if not self._runs:
            return None
        acquisition = self._update_acquisition(fields)
        if acquisition is None:
            acquisition = fields.get('acquisition', '')
        epoch_name = fields['epoch']
        if epoch_name not in self._epochs:
            # No epoch set: constrain run to what's valid for the current task
            task = fields.get('task', '')
            runs = self._runs_for.get((fields['subject'], fields.get('session', ''), task, acquisition), ())
        else:
            epoch = self._epochs[epoch_name]
            if not isinstance(epoch, PrimaryEpoch):
                return None
            runs = self._runs_for.get((fields['subject'], fields.get('session', ''), epoch.task, acquisition), ())
            if epoch.run is not None:
                # Subject may lack run tags entirely (single untagged recording);
                # epoch.run='01' should then resolve to '' rather than a missing file.
                # any(runs) is False for ('',) since '' is falsy.
                return epoch.run if any(runs) else ''
        if len(runs) == 1:
            return runs[0]
        if runs and fields.get('run') not in runs:
            return runs[0]  # current run invalid for this subject/session/task; reset
        return None  # don't force run

    def _update_acquisition(self, fields: dict) -> str | None:
        subject = fields.get('subject', '')
        session = fields.get('session', '')
        task = fields.get('task', '')
        acquisitions = sorted({acquisition for subject_, session_, task_, acquisition, _ in self._recordings if (subject_, session_, task_) == (subject, session, task)})
        if len(acquisitions) == 1:
            return acquisitions[0]
        if acquisitions and fields.get('acquisition') not in acquisitions:
            return acquisitions[0]
        return None

    def _eval_parc(self, parc: str) -> str:
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

    def _get_parc(self) -> tuple[str, Parcellation | None]:
        """Return ``(parc_name, parc_definition)`` for the current parc state.

        ``parc_definition`` is ``None`` when ``parc=''``.
        """
        return _resolve_parc(self._parcs, self.get('parc'))

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

    def _show_dependencies(
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
        self.set(**state)
        tree = self._derivatives.dependency_tree(name, state=self.state, options=options, max_line_length=max_line_length)
        if return_str:
            return tree
        print(tree)
        return None

    def show_head_position_overview(
            self,
            tolerance: float = 1e-3,
            asds: bool = False,
            **state,
    ) -> 'fmtxt.Table | fmtxt.Section | Dataset':
        """Overview of head position (dev_head_t) across tasks, subjects and sessions.

        Examines the device-to-head transformation matrix for all source
        recordings and groups tasks with similar head position.
        Useful for deciding whether separate coregistration/forward solutions
        are needed, or whether ICA should be performed on combined or separate data.

        Labels are assigned per subject, starting from A for the first task
        encountered. Tasks with the same label share the same head position
        within ``tolerance``; labels are not comparable across subjects.

        Parameters
        ----------
        tolerance
            Maximum element-wise absolute difference in the ``dev_head_t``
            transformation matrix for two recordings to be considered as having
            the same head position. Default ``1e-3`` corresponds to approximately
            1 mm for translation (and roughly 0.06° for rotation), which is
            conservative enough to justify sharing a forward solution.
        asds
            Return a :class:`Dataset` instead of formatted output.
        ...
            State parameters.

        Returns
        -------
        fmtxt.Table | fmtxt.Section
            Table with tasks as rows and subjects as columns. Each cell contains
            a cluster label (A, B, C, ...) indicating the head position group;
            cells with the same label share the same head position within
            ``tolerance``. Missing recordings are shown as "—". When the
            experiment has multiple sessions with differing head positions, a
            :class:`fmtxt.Section` with one table per session is returned.
        Dataset
            (when ``asds=True``) Long-format Dataset with columns ``subject``,
            ``task`` (and ``session`` when multiple sessions exist), and
            ``label``.
        """
        if state:
            self.set(**state)

        tasks = self._tasks
        subjects = list(self.get_field_values('subject'))
        has_sessions = bool(self._sessions)
        has_runs = bool(self._runs)
        MISSING_MARK = '—'
        CHL_MARK = '†'

        # Collect dev_head_t for every (subject, session, task, run) in the selected acquisition.
        # session is '' when the dataset has no BIDS sessions, run is '' when no runs exist.
        source_name = self._raw.root_source_name('raw')
        node_name = raw_input_name(source_name)
        # Inner dicts are keyed by (task, run) pairs.
        data: dict[str, dict[str, dict[tuple, np.ndarray | None]]] = {}
        chl: dict[str, dict[str, dict[tuple, bool]]] = {}

        for subject, session, task, run in self.iter(('subject', 'session', 'task', 'run'), raw='raw'):
            key = (task, run)
            ctx = self._resolve_derivative(node_name, options={'noise': False})
            if ctx.exists():
                data.setdefault(session, {}).setdefault(subject, {})[key] = None
                chl.setdefault(session, {}).setdefault(subject, {})[key] = False
                info = self._load_derivative(node_name, view='info', options={'noise': False})
                head_t = info.get('dev_head_t')
                if head_t is not None:
                    data[session][subject][key] = head_t['trans'].copy()
                chl[session][subject][key] = bool(info.get('hpi_meas'))

        sessions = sorted(data.keys())
        task_order = {t: i for i, t in enumerate(tasks)}

        def _row_keys(session: str) -> list[tuple[str, str]]:
            """Ordered (task, run) pairs that appear in any subject for this session."""
            found = {key for subj_data in data[session].values() for key in subj_data}
            return sorted(found, key=lambda tr: (task_order.get(tr[0], 999), tr[1]))

        # Assign per-subject cluster labels within each session.
        # For each subject, compare transforms greedily: the first (task, run)
        # encountered gets 'A', subsequent ones close to an existing group get
        # that group's label, otherwise a new label is assigned.
        any_missing = False
        any_chl = False
        labels: dict[str, dict[str, dict[tuple, str]]] = {}
        for session in sessions:
            labels[session] = {}
            row_keys = _row_keys(session)
            for subject in subjects:
                subject_data = data[session].get(subject, {})
                representatives: list[tuple[str, np.ndarray]] = []
                next_char = ord('A')
                subject_labels: dict[tuple, str] = {}
                for key in row_keys:
                    if chl[session].get(subject, {}).get(key, False):
                        any_chl = True
                    if key not in subject_data:
                        subject_labels[key] = ''
                        continue
                    trans = subject_data[key]
                    if trans is None:
                        subject_labels[key] = MISSING_MARK
                        any_missing = True
                        continue
                    for rep_label, rep_trans in representatives:
                        if np.allclose(trans, rep_trans, atol=tolerance, rtol=0):
                            subject_labels[key] = rep_label
                            break
                    else:
                        label = chr(next_char)
                        next_char += 1
                        representatives.append((label, trans))
                        subject_labels[key] = label
                labels[session][subject] = subject_labels

        def _cell(session: str, subject: str, key: tuple) -> str:
            label = labels[session][subject].get(key, '')
            if chl[session].get(subject, {}).get(key, False):
                return label + CHL_MARK
            return label

        if asds:
            rows_subj, rows_ses, rows_task, rows_run, rows_label = [], [], [], [], []
            for session in sessions:
                row_keys = _row_keys(session)
                for subject in subjects:
                    for task, run in row_keys:
                        rows_subj.append(subject)
                        rows_ses.append(session)
                        rows_task.append(task)
                        rows_run.append(run)
                        rows_label.append(_cell(session, subject, (task, run)))
            ds = Dataset()
            ds['subject'] = Factor(rows_subj)
            if has_sessions:
                ds['session'] = Factor(rows_ses)
            ds['task'] = Factor(rows_task)
            if has_runs:
                ds['run'] = Factor(rows_run)
            ds['label'] = Factor(rows_label)
            return ds

        def _make_caption(session: str) -> str:
            parts = []
            row_keys = _row_keys(session)
            if len(row_keys) > 1:
                session_labels = labels[session]
                patterns = [
                    tuple(session_labels[s].get(key, MISSING_MARK) for key in row_keys)
                    for s in subjects
                ]
                counts = Counter(patterns)
                majority_pat, majority_n = counts.most_common(1)[0]
                pat_str = '–'.join(majority_pat)
                if majority_n == len(subjects):
                    if set(majority_pat) == {'A'}:
                        parts.append("All subjects have the same head position for all tasks.")
                    else:
                        parts.append(f"All subjects: {pat_str}.")
                else:
                    exceptions = [
                        f"{s} ({'–'.join(p)})"
                        for s, p in zip(subjects, patterns)
                        if p != majority_pat
                    ]
                    pattern = f"Majority ({majority_n}/{len(subjects)} subjects): {pat_str}. Exceptions: {', '.join(exceptions)}."
                    parts.append(pattern)
            if any_missing:
                parts.append(f"{MISSING_MARK}: no initial head position (dev_head_t).")
            if any_chl:
                parts.append(f"{CHL_MARK}: continuous head localization.")
            return ' '.join(parts)

        def _make_table(session: str, title: str = None) -> fmtxt.Table:
            row_keys = _row_keys(session)
            col_spec = 'l' + ('r' if has_runs else '') + 'c' * len(subjects)
            t = fmtxt.Table(col_spec, title=title, caption=_make_caption(session))
            t.cell('Task')
            if has_runs:
                t.cell('Run')
            for subject in subjects:
                t.cell(subject)
            t.midrule()
            prev_task = None
            for task, run in row_keys:
                t.cell(task if task != prev_task else '')
                prev_task = task
                if has_runs:
                    t.cell(run or '(no run)')
                for subject in subjects:
                    t.cell(_cell(session, subject, (task, run)))
            return t

        if not has_sessions or len(sessions) == 1:
            title = "Head position"
            if has_sessions and sessions[0]:
                title += f" (session: {sessions[0]})"
            return _make_table(sessions[0], title=title)

        # Multiple sessions: check whether patterns are identical across all sessions
        def _session_sig(session: str) -> dict[str, tuple]:
            sl = labels[session]
            row_keys = _row_keys(session)
            return {s: tuple(sl[s].get(key, MISSING_MARK) for key in row_keys) for s in subjects}

        sig0 = _session_sig(sessions[0])
        if all(_session_sig(ses) == sig0 for ses in sessions[1:]):
            return _make_table(sessions[0], title="Head position (identical across all sessions)")

        section = fmtxt.Section("Head position across tasks")
        for session in sessions:
            sub = section.add_section(f"Session: {session}")
            sub.append(_make_table(session))
        return section

    def show_raw_info(self, **state) -> fmtxt.Table | None:
        """Display the selected pipeline for raw processing

        See Also
        --------
        show_subjects : list presence of raw input file by subject
        """
        raw = self.get('raw', **state)
        pipe = self._raw[raw]
        pipeline = self._raw.lineage_pipes(raw)
        print(f"Preprocessing pipeline: {' --> '.join(p.name for p in pipeline)}")

        # pipe-specific
        if isinstance(pipe, RawICA):
            rows = []
            for subject in self:
                ctx = self._resolve_derivative(ica_input_name(raw))
                status = ctx.load(view='status')
                if status == 'ok':
                    ica = ctx.load()
                    rows.append((subject, ica.n_components_, len(ica.exclude)))
                elif status == 'missing-ica':
                    rows.append((subject, "No ICA-file", -1))
                elif status == 'missing-raw':
                    rows.append((subject, "No data", -1))
                else:
                    raise RuntimeError(f"{status=}")

            n_selected = [row[-1] for row in rows]
            mark_unselected = any(n_selected) and not all(n_selected)

            table = fmtxt.Table('lrr' + 'r' * mark_unselected)
            table.cells('Subject', 'n components', 'reject')
            if mark_unselected:
                table.cell('*')
                table.caption("*: ICA with no rejected components")
            table.midrule()
            for subject, n, n_selected in rows:
                table.cells(subject, n)
                if not isinstance(n, str):
                    table.cell(n_selected)
                    if mark_unselected and n_selected == 0:
                        table.cell('*')
                table.endline()

            return table

    def show_reg_params(self, **state):
        """Show the covariance matrix regularization parameters

        Parameters
        ----------
        ...
            State parameters.
        """
        cov = self.get('cov', **state)
        cov_config = self._covs[cov]
        if not isinstance(cov_config, EpochCovariance):
            raise ValueError(f"{cov=}: not an EpochCovariance")

        rows = []
        for subject in self:
            handle = self._resolve_derivative('cov')
            path = handle.artifact_path.with_suffix('.info.txt')
            if exists(path):
                with open(path) as fid:
                    text = fid.read()
                reg = float(text.strip())
            else:
                reg = float('nan')
            rows.append((subject, reg))
        ds = Dataset.from_caselist(['subject', 'reg'], rows)
        return ds

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
        rej_name = self.get('epoch_rejection')
        rej = self._epoch_rejection[rej_name]
        has_epoch_rejection = rej is not None
        has_interp = rej is not None and rej.interpolation

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
            caption += " Bad channels: defined in bad_channels file and in epoch-rejection file."
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
    ) -> Dataset | fmtxt.Table:
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
            Caption for the table (default "Subject in group {group}").
        asds
            Return the table as Dataset instead of an :class:`fmtxt.Table`.
        ...
            State parameters.
        """
        if isinstance(mri, str):
            state['mri'] = mri
            mri = True
        elif mri is None:
            mri = exists(self.root / MRI_SDIR)
        if state:
            self.set(**state)

        # caption
        if caption is True:
            caption = self.format("Subjects in group {group}")

        subject_list = []
        mri_list = []
        mrisubject_list = []
        raw_list = []  # {task: [] for task in self.get_field_values('task')}
        for subject in self.iter():
            subject_list.append(subject)
            mrisubject_ = self.get('mrisubject')
            mrisubject_list.append(mrisubject_)
            if raw:
                # for task in self.iter('task'):
                #     pass
                # FIXME: use ctx.node.exists()
                fixed_state = {k: v for k, v in self._fields.items() if not (isinstance(v, str) and '*' in v)}
                query = bids_path(self.root, fixed_state, self._raw_extension, datatype=self._datatype)
                matches = query.match()
                basenames = [match.basename for match in matches]
                raw_list.append(', '.join(basenames))
            if mri:
                subject_mri_dir = str(self.root / mri_dir(self._fields))
                if not exists(subject_mri_dir):
                    mri_list.append('*missing')
                elif is_fake_mri(subject_mri_dir):
                    subjects_dir = str(self.root / MRI_SDIR)
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

    def _surfer_plot_kwargs(
            self,
            surf: str | None = None,
            views: str | tuple[str, ...] | None = None,
            foreground=None,
            background=None,
            smoothing_steps: int | None = None,
            hemi: str | None = None,
    ) -> dict:
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
