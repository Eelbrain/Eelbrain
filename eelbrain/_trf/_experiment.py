# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Eelbrain :class:`Pipeline` extension for analyzing continuous response models"
from collections import defaultdict
import datetime
import fnmatch
from functools import partial
from glob import glob
from itertools import chain, repeat
from operator import attrgetter
import os
from os.path import exists, getmtime, join, relpath
from pathlib import Path
from pyparsing import ParseException
import re
import time
import types
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union
import sys
import warnings

import eelbrain
from eelbrain import (
    fmtxt, load, save, table, plot,
    MultiEffectNDTest, BoostingResult,
    Pipeline, Dataset, Datalist, Factor, NDVar, UTS,
    morph_source_space, rename_dim, boosting, combine, concatenate,
)
from eelbrain.pipeline import TTestOneSample, TTestRelated, TwoStageTest, RawFilter, RawSource
from eelbrain._experiment.definitions import FieldCode
from eelbrain._experiment.epochs import EpochCollection
from eelbrain._experiment.mne_experiment import DataArg, PMinArg, DefinitionError, FileMissingError, TestDims, Variables, guess_y, cache_valid
from eelbrain._experiment.parc import CombinationParc, SubParc
from eelbrain._data_obj import NDVarArg, isuv
from eelbrain._io.pickle import update_subjects_dir
from eelbrain._text import ms, n_of
from eelbrain._types import PathArg
from eelbrain._utils.mne_utils import is_fake_mri
from eelbrain._utils.notebooks import tqdm
from eelbrain._utils import ask
from filelock import FileLock
import numpy as np
from numpy import newaxis
from trftools.pipeline.estimator import Estimator

from .._ndvar import pad
from .._numpy_funcs import arctanh
from ._code import Code
from ._jobs import TRFsJob
from ._model import Comparison, Model, ModelExpression, StructuredModel, load_models, model_comparison_table, model_name_parser, save_models
from ._predictor import EventPredictor, FilePredictor, FilePredictorBase, MakePredictor, SessionPredictor
from ._results import DependentType, ResultCollection
from . import _trf_report as trf_report


DATA_DEFAULT = 'source'
FIT_METRICS = ('r', 'z', 'r1', 'z1', 'residual', 'det')
UV_FUNCTIONS = ('sum', 'mean', 'max')
FIT_METRIC_RE = re.compile(rf"^({'|'.join(FIT_METRICS)})(?:\.({'|'.join(UV_FUNCTIONS)}))?$")
# https://en.wikipedia.org/wiki/Fisher_transformation

# templates for files affected by models; is_public
TRF_TEMPLATES = (
    ('trf-file', False),
    ('trf-test-file', False),
    ('model-test-file', False),
    ('model-report-file', True),
)
NCRF_RE = re.compile(r'(ncrf)(?:-([\w\d]+))?$')

ComparisonArg = Union[str, Comparison, StructuredModel]
ModelArg = Union[str, Model]
ModelOrComparisonArg = Union[str, Model, Comparison, StructuredModel]
FilterXArg = Literal[True, False, 'continuous']


class NameTooLong(Exception):

    def __init__(self, name):
        Exception.__init__(self, "Name too long (%i characters), consider adding a shortened model name: %s" % (len(name), name))


class FilenameTooLong(Exception):

    def __init__(self, path):
        Exception.__init__(self, "Filename too long (%i characters), consider adding a shortened model name: %s" % (len(os.path.basename(path)), path))


def split_model(x):
    return [v.strip() for v in x.split('+')]


def difference_maps(dss):
    """Difference maps for model comparison"""
    diffs = {}
    subjects = None
    for x, x_ds in dss.items():
        ds = table.repmeas('z', 'model', 'subject', ds=x_ds)
        if subjects is None:
            subjects = ds['subject']
        else:
            assert np.all(ds['subject'] == subjects)
        diff = ds['test'] - ds['baseline']
        for hemi in ('lh', 'rh'):
            diffs[x, hemi] = diff.sub(source=hemi)
    return subjects, diffs


class ModelDescriber:

    def __init__(self, structured_models: Dict[str, StructuredModel]):
        self.abbreviations = {}
        for key, model in structured_models.items():
            model_key = tuple(sorted([term.string for term in model.terms]))
            self.abbreviations[model_key] = key
        self.ns = {len(model_key) for model_key in self.abbreviations}

    def describe(self, model: Union[Model, str]) -> str:
        model = Model.coerce(model)
        terms = []
        n_terms = len(model.terms)
        start = 0
        while start < n_terms:
            for stop in range(n_terms, start + 1, -1):
                if stop - start in self.ns:
                    key = tuple(sorted([term.code for term in model.terms[start:stop]]))
                    if key in self.abbreviations:
                        terms.append(self.abbreviations[key])
                        start = stop
                        break
            else:
                terms.append(model.terms[start].string)
                start += 1
        return ' + '.join(terms)


class TRFExperiment(Pipeline):
    """Pipeline for TRF analysis (see also: :class:`eelbrain.pipeline.Pipeline`)

    Setup attributes
    ----------------

    .. autoattribute:: TRFExperiment.stim_var


    Initialization
    --------------
    """
    stim_var: str = 'stimulus'
    """
    Event variable that identifies stimulus files for loading predictors
     
    ``stim_var`` is specified as event dataset column name. For example, with::
    
        stim_var = 'stimulus'
    
    and the following event dataset::
    
        >>> print(alice.load_events())
        #    i_start   trigger   stimulus  time     SOA      subject   duration
        -----------------------------------------------------------------------
        0    1863      1         stim_1    3.726    57.618   S01       58.541  
        1    30672     5         stim_2    61.344   60.898   S01       61.845  
        ...
    
    The model term ``gammatone-8`` will use the predictor files ``stim_1~gammatone-8`` for the first event, and ``stim_2~gammatone-8`` for the second.
    
    To specify multiple stimuli per event use a ``{key: stim_var}`` dictionary. For example, for a cocktail party experiment with two auditory sources (foreground and background)::

        stim_var = {'': 'fg', 'bg': 'bg', 'mix': 'mix'}

    with the following event dataset::

        #    i_start   trigger   T        SOA      subject   fg   bg   mix
        ------------------------------------------------------------------
        0    1863      1         3.726    57.618   S01       s1   s3   s13
        1    30672     5         61.344   60.898   S01       s2   s4   s24
        ...

    The "stream" is specified in model terms with ``~``:

     - 'gammatone' would use predictors based on the ``fg`` column: ``s1~gammatone``, ``s2~gammatone``, ...
     - 'bg~gammatone' would use predictors based on the ``bg`` column: ``s3~gammatone``, ``s4~gammatone``, ...
     - 'mix~gammatone' would use predictors based on the ``mix`` column: ``s13~gammatone``, ``s24~gammatone``, ...
    """
    predictors: Dict[str, Union[EventPredictor, FilePredictor, MakePredictor]] = {}

    _values = {
        # Predictors
        'predictor-dir': join('{deriv-dir}', 'predictors'),
        # TRF
        'trf-sdir': join('{cache-dir}', 'trf'),
        'trf-dir': join('{trf-sdir}', '{subject}'),
        'trf-file': join('{trf-dir}', '{analysis}', '{epoch_basename} {test_options}.pickle'),
        'trf-test-file': join('{cache-dir}', 'trf-test', '{analysis} {group}', '{folder}', '{epoch_basename} {test} {test_options}.pickle'),
        # model comparisons
        'model-test-file': join('{cache-dir}', 'model-test', '{analysis} {group}', '{folder}', '{epoch_basename} {test} {test_options}.pickle'),
        'model-res-dir': join('{deriv-dir}', 'results-models'),
        'model-report-file': join('{model-res-dir}', '{analysis} {group}', '{folder}', '{epoch_basename} {test} {test_options}.html'),
        # predictors
        'predictor-cache-dir': join('{cache-dir}', 'predictors'),
    }
    _artifact_rejection_default = ''

    models = {}
    _empty_test = True

    _parc_supersets = {}

    def _resolve_estimator(self, estimator: Union[str, Estimator, None]) -> Union[Estimator, None]:
        """Resolve estimator name to an Estimator instance (or None)."""
        if estimator is None:
            return None
        if isinstance(estimator, str):
            estimators = getattr(self, 'estimators', None)
            if not isinstance(estimators, dict):
                raise ValueError("estimator='...' requires experiment.estimators dict")
            est = estimators.get(estimator)
            if est is None:
                raise ValueError(f"estimator={estimator!r} not in {list(estimators.keys())}")
            estimator = est
        if not isinstance(estimator, Estimator):
            raise ValueError(f"estimator must be str or Estimator, got {type(estimator)!r}")
        return estimator

    def _apply_estimator_params(self, estimator: Union[Estimator, None], **kwargs) -> Dict[str, Any]:
        """Apply estimator params on top of explicit args and return updated kwargs."""
        if estimator is None:
            return kwargs
        effective = dict(estimator.parameters_for_partial())
        kwargs['delta'] = effective.get('delta', kwargs.get('delta'))
        kwargs['mindelta'] = effective.get('mindelta', kwargs.get('mindelta'))
        kwargs['error'] = effective.get('error', kwargs.get('error'))
        kwargs['basis'] = effective.get('basis', kwargs.get('basis'))
        kwargs['partitions'] = effective.get('partitions', kwargs.get('partitions'))
        kwargs['cv'] = effective.get('test', kwargs.get('cv'))
        kwargs['selective_stopping'] = effective.get('selective_stopping', kwargs.get('selective_stopping'))
        kwargs['partition_results'] = effective.get('partition_results', kwargs.get('partition_results'))
        return kwargs

    def _collect_invalid_files(self, invalid_cache, new_state, cache_state):
        rm = Pipeline._collect_invalid_files(self, invalid_cache, new_state, cache_state)

        # stimuli
        for var, subject in invalid_cache['variable_for_subject']:
            if var in self._stim_var.values():
                state = {'subject': subject}
                rm['trf-file'].add(state)
                for group, members in self._groups.items():
                    if subject in members:
                        state = {'group': group}
                        rm['trf-test-file'].add(state)
                        rm['model-test-file'].add(state)
                        rm['model-report-file'].add(state)

        # epochs are based on events
        for item in invalid_cache['events']:
            # Eelbrain versions may store 2-tuples or longer (e.g. include session/run)
            subject = item[0]
            recording = item[1]
            for epoch, params in self._epochs.items():
                sessions = getattr(params, 'sessions', None)
                if sessions is not None and recording not in sessions:
                    continue
                rm['trf-file'].add({'subject': subject, 'epoch': epoch})

        # group
        for group in invalid_cache['groups']:
            state = {'group': group}
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # raw
        for raw in invalid_cache['raw']:
            for analysis in (raw, f'{raw} *'):
                state = {'analysis': analysis}
                rm['trf-file'].add(state)
                rm['trf-test-file'].add(state)
                rm['model-test-file'].add(state)
                rm['model-report-file'].add(state)

        # epochs
        for epoch in invalid_cache['epochs']:
            state = {'epoch': epoch}
            rm['trf-file'].add(state)
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # cov
        for cov in invalid_cache['cov']:
            state = {'analysis': f'* {cov} *'}
            rm['trf-file'].add(state)
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        # parcs
        for parc in invalid_cache['parc']:
            for opt in (f'{parc} *', f'* {parc} *'):
                state = {'test_options': opt}
                rm['trf-file'].add(state)
            state = {'folder': f'{parc} masked'}
            rm['trf-test-file'].add(state)
            rm['model-test-file'].add(state)
            rm['model-report-file'].add(state)

        return rm

    @classmethod
    def _eval_inv(cls, inv):
        if NCRF_RE.match(inv):
            return inv
        else:
            return Pipeline._eval_inv(inv)

    def _post_set_inv(self, _, inv):
        if NCRF_RE.match(inv):
            inv = '*'
        Pipeline._post_set_inv(self, _, inv)

    @staticmethod
    def _update_inv_cache(fields):
        if NCRF_RE.match(fields['inv']):
            return fields['inv']
        return Pipeline._update_inv_cache(fields)

    def _subclass_init(self):
        # predictors
        for key in self.predictors:
            if not key.isidentifier():
                raise ValueError(f"{key!r}: invalid predictor key")
        # for model test
        self._field_values['test'] += ('',)
        # event variable that indicates stimuli
        if isinstance(self.stim_var, str):
            self._stim_var = {'': self.stim_var}
        elif isinstance(self.stim_var, dict):
            self._stim_var = self.stim_var.copy()
        else:
            raise TypeError(f"{self.__class__.__name__}.stim_var={self.stim_var!r}")
        # structured models
        for name in self.models:
            try:
                model_name_parser.parseString(name, True)
            except ParseException:
                raise DefinitionError(f"Invalid key in {self.__class__.__name__}.models: {name!r}")
            if re.match(r'^[\w\-+~]*-red\d*$', name):
                raise ValueError(f"Invalid key in {self.__class__.__name__}.models: {name!r} (-red* pattern is reservered)")
        self._structured_models = {k: StructuredModel.coerce(v) for k, v in self.models.items()}
        self._structured_model_names = {m: k for k, m in self._structured_models.items()}

        # Model names
        self._named_models: Dict[str, Model] = {}
        self._model_names: Dict[str, str] = {}
        self._model_names_file = join(self.get('cache-dir', mkdir=True), 'model-names.pickle')
        self._model_names_file_lock = FileLock(self._model_names_file + '.lock')
        self._model_names_mtime = 0
        with self._model_names_file_lock:
            self._refresh_model_names()

        # Parcellations: find parcellations with subset relationship
        self._parc_supersets = defaultdict(set, {k: set(v) for k, v in self._parc_supersets.items()})
        # automatic
        sub_parcs = {k: v for k, v in self._parcs.items() if isinstance(v, SubParc)}
        for key, parc in self._parcs.items():
            if not isinstance(parc, (SubParc, CombinationParc)):
                continue
            for src_key, src_parc in sub_parcs.items():
                if src_key == key:
                    continue
                elif parc.base == src_parc.base and all(l in src_parc.labels for l in parc._base_labels()):
                    self._parc_supersets[key].add(src_key)

    def _refresh_model_names(self):
        if exists(self._model_names_file):
            mtime = os.path.getmtime(self._model_names_file)
            if mtime != self._model_names_mtime:
                self._named_models = load_models(self._model_names_file)
                self._model_names = {model.sorted_key: name for name, model in self._named_models.items()}
                self._model_names_mtime = mtime
        self._last_model_name_refresh = time.time()

    def _register_model(self, model: Model) -> str:
        "Register a new model (generate a name)"
        model = model.without_randomization
        with self._model_names_file_lock:
            self._refresh_model_names()
            if model.sorted_key in self._model_names:
                return self._model_names[model.sorted_key]
            name = self._generate_model_name(model)
            self._named_models[name] = model
            save_models(self._named_models, self._model_names_file)
        self._model_names[model.sorted_key] = name
        return name

    def _generate_model_name(self, model: Model):
        assert not model.has_randomization
        if len(model.terms) == 1:
            if model.name not in self._named_models:
                return model.name
        for i in range(9999999):
            name = f"model{i}"
            if name not in self._named_models:
                return name
        raise RuntimeError("Ran out of model names...")

    def _find_model_files(
            self,
            name: str,
            trfs: bool = False,
            tests: bool = False,
    ) -> Iterator[str]:
        """Find all files associated with a model

        Will not find ``model (name$shift)``
        """
        assert trfs or tests
        if name not in self._named_models:
            raise ValueError(f"{name=}: not a named model")
        pattern = f"*{name}*"
        regex = re.compile(rf"(^| ){re.escape(name)}(-red\d+)?($|[. ])")  # -red for legacy names
        for temp, is_public in TRF_TEMPLATES:
            if temp.startswith('trf'):
                if not trfs:
                    continue
            if temp.startswith('model'):
                if not tests:
                    continue
            for path in self.glob(temp, True, test_options=pattern):
                if regex.search(path):
                    yield path

    def _repair_model_names(self):
        """Infer model names from TRFs"""
        # TRFs store only legal key version
        # Check for missing models
        model_files = defaultdict(list)
        models = defaultdict(set)
        glob_files = list(self.glob('trf-file', True))
        for path in tqdm(glob_files, "Scanning TRF-files"):
            parameters = self._parse_trf_path(path)
            key = parameters['model']
            # store files
            model_files[key].append(path)
            # extract model
            trf = load.unpickle(path)
            model = Model.from_string(trf.x)
            models[key].add(model.sorted())
        # Check for files whose model keys are not listed
        missing = set(model_files) - set(self._named_models)
        if missing:
            self._log.error("Found TRF files for non-existing models:")
            for model in sorted(missing):
                n = len(model_files[model])
                self._log.error(f"  {model}: {n} files")
            raise NotImplementedError("This cannot be repaired automatically")
        # Check for model keys that have conflicting models in TRFs
        conflicts = {key: model_set for key, model_set in models.items() if len(model_set) > 1}
        if conflicts:
            for key, model_set in conflicts.items():
                self._log.error(f"Model {key}: found multiple interpretations in TRF files")
                for terms in sorted([model.name for model in model_set]):
                    self._log.error(f"  {terms}")
            raise NotImplementedError("This cannot be repaired automatically")
        # Check for models that differ in experiment and in files
        conflicts = []
        for key, model_set in models.items():
            model = next(iter(model_set))
            exp_model = self._named_models[key]
            if model.sorted_key != exp_model.dataset_based_key:
                conflicts.append((key, model, exp_model))
                self._log.error(f"Model {key}: mismatching definitions")
                self._log.error(f"  Exp : {exp_model.dataset_based_key}")
                self._log.error(f"  TRFs: {model.sorted_key}")
        if conflicts:
            to_replace = []
            message = []
            for key, trf_model, exp_model in conflicts:
                # Infer dataset key -> term name
                term_names = {term.string: set() for term in trf_model.terms}
                for model in self._named_models.values():
                    for term in model.terms:
                        term_key = Dataset.as_key(term.string)
                        if term_key in term_names:
                            term_names[term_key].add(term.string)
                if conflicts := {term_key: names for term_key, names in term_names.items() if len(names) != 1}:
                    self._log.error(f"Cannot repair {key} due to ambiguous term keys:")
                    for term_key, names in conflicts.items():
                        self._log.error(f"  {term_key} -> {', '.join(names)}")
                    continue
                term_names = {term_key: names.pop() for term_key, names in term_names.items()}
                new_model = Model.from_string([term_names[term.string] for term in trf_model.terms])
                to_replace.append((key, exp_model, new_model))
                self._log.error(f'{key}:')
                self._log.error(f'  Experiment: {exp_model.sorted().name}')
                self._log.error(f'  TRF-files:  {new_model.sorted().name}')
            if to_replace:
                if ask('Replace experiment models with models inferred from TRFs?', {'yes': 'Replace model names', 'no': 'Abort without changing anything'}, default='no') != 'yes':
                    self._log.error('No changes made')
                    return
                self._log.error('Updating model names...')
                for key, exp_model, new_model in to_replace:
                    self._named_models[key] = new_model
                    self._model_names[new_model.sorted_key] = key
                with self._model_names_file_lock:
                    save_models(self._named_models, self._model_names_file)
                self._log.error('Done.')

    def _remove_model(self, name, files=None):
        """Remove a named model and delete all associated files

        .. warning::
            Only use this command when only a single instance of this
            TRFExperiment class exists.
            A separate, previously created instance of the same TRFExperiment
            class would retain a reference to this model name and might use it
            again, leading to conflicts later.
        """
        if files is None:
            files = list(self._find_model_files(name, trfs=True, tests=True))
        if files:
            while True:
                command = ask(f"Remove model {name} and delete {len(files)} files?", {"remove": "confirm", 'show': 'list files'}, allow_empty=True)
                if command == 'remove':
                    break
                elif command == 'show':
                    prefix = os.path.commonprefix(files)
                    n_prefix = len(prefix)
                    print(f"At {prefix}:")
                    print('\n'.join(f'  {path[n_prefix:]}' for path in files))
                else:
                    raise RuntimeError("Model deletion aborted")

        for path in files:
            os.remove(path)

        with self._model_names_file_lock:
            self._refresh_model_names()
            model = self._named_models.pop(name)
            del self._model_names[model.sorted_key]
            save_models(self._named_models, self._model_names_file)

    # Stimuli
    #########
    def add_predictors(
            self,
            ds: Dataset,
            model: str,
            filter_x: FilterXArg = False,
            y: str = None,
    ):
        """Add all predictor variables in a given model to a :class:`Dataset`

        Parameters
        ----------
        ds
            Dataset with the dependent measure.
        model
            Model for which to load predictors.
        filter_x
            Filter predictors with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        y
            :class:`UTS` or :class:`NDVar` to match time axis to.
        """
        x = self._coerce_model(model)
        for term in x.terms:
            code = Code.coerce(term.string)  # TODO: use parse result
            self.add_predictor(ds, code, filter_x, y)

    def add_predictor(
            self,
            ds: Dataset,
            code: str,
            filter_x: FilterXArg = False,
            y: Union[UTS, NDVarArg] = None,
    ):
        """Add predictor variable to a :class:`Dataset`

        Parameters
        ----------
        ds
            Dataset with the dependent measure.
        code
            Predictor to add. Suffix demarcated by ``$`` for shuffling.
        filter_x
            Filter predictors with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        y
            :class:`UTS` or :class:`NDVar` to match time axis to.
        """
        if isinstance(y, UTS):
            time = y
        else:
            if y is None:
                y = ds[guess_y(ds)]
            elif isinstance(y, str):
                y = ds[y]
            if isinstance(y, NDVar):
                time = y.time
            else:
                time = [yi.time for yi in y]
        is_variable_time = isinstance(time, list)
        code = Code.coerce(code)
        code.seed(ds.info['subject'])
        directory = Path(self.get('predictor-dir'))

        try:
            predictor = self.predictors[code.lookahead()]
        except KeyError:
            raise code.error(f"predictor undefined in {self.__class__.__name__}", 0)

        if isinstance(predictor, SessionPredictor):
            if not is_variable_time:
                raise NotImplementedError(f"SessionPredictor for fixed duration epochs")
            x = self.load_predictor(code, filter_x=filter_x, name=code.key)
            onset_times = ds['time'] - ds[0, 'time']
            ds[code.key] = predictor._epoch_for_data(x, time, onset_times)
            return

        # register that predictor is identified
        code.next()

        # which Dataset variable indicates the stimulus?
        if not code.stim:
            stim_var = self._stim_var['']
        elif code.stim in self._stim_var:  # backwards compatibility
            stim_var = self._stim_var[code.stim or '']
        else:
            stim_var = code.stim

        # For nested events
        events_key = ds.info.get('nested_events')
        if events_key:
            if filter_x:
                raise ValueError(f"{filter_x=}: not available for {predictor.__class__.__name__}")
            assert is_variable_time
            xs = []
            assert not code._shuffle_done
            for uts, sub_ds in zip(time, ds[events_key]):
                code._shuffle_done = False  # each iteration will register shuffle
                if isinstance(predictor, EventPredictor):
                    x = predictor._generate_continuous(uts, sub_ds, code)
                elif isinstance(predictor, FilePredictorBase):
                    x = predictor._generate_continuous(uts, sub_ds, stim_var, code, directory)
                else:
                    raise RuntimeError(predictor)
                xs.append(x)
            ds[code.key] = xs
            return

        if isinstance(predictor, EventPredictor):
            if filter_x:
                raise ValueError(f"{filter_x=}: not available for {predictor.__class__.__name__}")
            elif is_variable_time:
                raise NotImplementedError("EventPredictor not implemented for variable-time epochs")
            ds[code.key] = predictor._generate(time, ds, code)
            code.assert_done()
            return

        # load predictors (cache for same stimulus unless they are randomized)
        if code.has_randomization or is_variable_time:
            time_dims = time if is_variable_time else repeat(time, ds.n_cases)
            xs = [self.load_predictor(code.with_stim(stim), time.tstep, time.nsamples, time.tmin, filter_x, code.key) for stim, time in zip(ds[stim_var], time_dims)]
        else:
            x_cache = {stim: self.load_predictor(code.with_stim(stim), time.tstep, time.nsamples, time.tmin, filter_x, code.key) for stim in ds[stim_var].cells}
            xs = [x_cache[stim] for stim in ds[stim_var]]

        if not is_variable_time:
            xs = combine(xs)
        ds[code.key] = xs

    def prune_models(
            self,
            prune_tests: bool = False,
    ):
        """Remove internal models that have no corresponding files

        Parameters
        ----------
        prune_tests
            Remove cached test files if corresponding TRFs have been removed
            previously.

        See Also
        --------
        .remove_model
        .show_models
        """
        models = list(self._named_models)
        for model in models:
            for _ in self._find_model_files(model, trfs=True, tests=not prune_tests):
                break
            else:
                if prune_tests:
                    for path in self._find_model_files(model, tests=True):
                        os.unlink(path)
                self._remove_model(model, files=[])

    def trf_job(self, x: str, priority: bool = False, **kwargs):
        """Create a batch-job for computing TRFs

        Parameters
        ----------
        x
            Model or comparison. If a comparison, compute TRFs for both models
            that are implied in the comparison.
        priority
            Prioritize job over other jobs that have been scheduled earlier.
        ...
            For more arguments see :meth:`.load_trf`. For example, use the
            ``group`` parameter to control for which participants to compute the
            TRFs.
        """
        comparison = self._coerce_comparison(x, True)
        return TRFsJob(comparison, self, priority, **kwargs)

    def load_predictor(
            self,
            code: str,
            tstep: float = None,
            n_samples: int = None,
            tmin: float = None,
            filter_x: FilterXArg = False,
            name: str = None,
            **state,
    ):
        """Load predictor NDVar

        Parameters
        ----------
        code
            Code for the predictor to load (using the pattern
            ``{stimulus}~{code}${randomization}``)
        tstep
            Time-step for the predictor (for :class:`NDVar` predictors, the
            default is the original ``tstep``; for :class:`Dataset` predictors
            ``tstep`` needs to be specified).
        n_samples
            Number of samples in the predictor (the default returns all
            available samples).
        tmin
            First sample time stamp (default is all abailable data).
        filter_x
            Filter predictors with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        name
            Reassign the name of the predictor :class:`NDVar`.


        See Also
        --------
        .load_predictors : load multiple predictors simultaneously
        """
        code = Code.coerce(code)
        try:
            predictor = self.predictors[code.lookahead()]
        except KeyError:
            raise code.error(f"predictor undefined in {self.__class__.__name__}", 0)

        # if called without add_predictor
        if state:
            self.set(**state)
        if code._seed is None:
            code.seed(self.get('subject'))

        if isinstance(predictor, FilePredictorBase):
            directory = Path(self.get('predictor-dir'))
            if isinstance(predictor, FilePredictor):
                x = predictor._generate(tmin, tstep, n_samples, code, directory)
            elif isinstance(predictor, SessionPredictor):
                x = predictor._generate(tmin, tstep, n_samples, code, directory, self.get('subject'), self.get('recording'))
            code.register_string_done()
            code.assert_done()
        elif isinstance(predictor, MakePredictor):
            x = self._make_predictor(code, tstep, n_samples, tmin)
        elif isinstance(predictor, EventPredictor):
            raise ValueError(f"{code.string!r}: can't load {predictor.__class__.__name__} without data; use {self.__class__.__name__}.add_predictor()")
        else:
            raise code.error(f"Unknown predictor type {predictor}", 0)

        if isinstance(filter_x, str):
            if filter_x == 'continuous':
                filter_x = x.info['sampling'] == 'continuous'
            else:
                raise ValueError(f"{filter_x=}")

        if filter_x:
            raw = self.get('raw')
            pipe = self._raw[raw]
            pipes = []
            while not isinstance(pipe, RawSource):
                if isinstance(pipe, RawFilter):
                    pipes.append(pipe)
                pipe = pipe.source
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'filter_length ', RuntimeWarning)
                for pipe in reversed(pipes):
                    x = pipe.filter_ndvar(x, pad='edge')

        if name is None:
            x.name = code.string
        else:
            x.name = name
        return x

    def load_predictors(self, stim, model, tstep=0.01, n_samples=None, tmin=0.):
        """Multiple predictors corresponding to ``model`` in a list

        See Also
        --------
        .load_predictor : load a single predictor
        """
        model = self._coerce_model(model)
        out = []
        for term in model.terms:
            y = self.load_predictor(f'{stim}~{term.string}', tstep, n_samples, tmin)
            out.append(y)
        return out

    def _make_predictor(self, code, tstep=0.01, n_samples=None, tmin=0., seed=False):
        "Wrapper for .make_predictor() with caching"
        if code.has_randomization:
            cache_path = None
        else:
            cache_dir = Path(self.get('predictor-cache-dir', mkdir=True))
            cache_path = cache_dir / f'{code.string} {tmin:g} {tstep:g}.pickle'
        # load/generate predictor
        if cache_path and exists(cache_path):
            x = load.unpickle(cache_path)
        else:
            x = self.make_predictor(code, tstep, n_samples, tmin, seed)
            code.assert_done()
        # cache
        if cache_path:
            save.pickle(x, cache_path)
        # match time axis
        x = pad(x, tmin, nsamples=n_samples)
        # check time
        if n_samples is None:
            target_time = UTS(tmin, tstep, x.time.nsamples)
        else:
            target_time = UTS(tmin, tstep, n_samples)
        if x.time != target_time:
            raise code.error(f"Predictor time {x.time} does not match requested time {target_time}")
        return x

    def make_predictor(self, code, tstep=0.01, n_samples=None, tmin=0., seed=False):
        raise NotImplementedError

    def _post_process_trf(
            self,
            trf: BoostingResult,
            data: TestDims,
            parc: str = None,  # current trf parcellation
            to_parc: str = None,  # change trf parcellation to this
    ):
        "Apply post-processing to a TRF"
        if not data.source:
            return trf
        if data.morph:
            # Make sure common brain source space exists
            common_brain = self.get('common_brain')
            with self._temporary_state:
                self.make_src(mrisubject=common_brain)
                if parc:
                    self.make_annot(parc=parc, mrisubject=common_brain)
            # Cached morph matrix
            if is_fake_mri(self.get('mri-dir')):
                source_morph = None
            else:
                source_morph = self.load_source_morph()
            # Apply morph
            trf._morph(common_brain, source_morph=source_morph)
        if to_parc:
            self.make_annot(parc=to_parc)
            trf._set_parc(to_parc)
        return trf

    # TRF
    #####
    def load_trf(
            self,
            x: ModelArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            backward: bool = False,
            make: bool = False,
            path_only: bool = False,
            partition_results: bool = False,
            morph: bool = False,
            estimator: Union[str, Estimator, None] = None,
            **state,
    ) -> Union[BoostingResult, str]:
        """TRF estimated with boosting

        Parameters
        ----------
        x
            One or more predictor variables, joined with '+'.
        tstart
            Start of the TRF in s (default 0).
        tstop
            Stop of the TRF in s (default 0.5).
        basis
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions
            Number of partitions used for cross-validation in boosting (default
            is the number of epochs; -1 to concatenate data).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta
            Boosting delta.
        mindelta
            Boosting parameter.
        filter_x
            Filter ``x`` with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        selective_stopping
            Stop boosting each predictor separately.
        cv
            Cross-validation.
        data : 'source' | 'meg' | 'eeg'
            Analyze source-space data (default) or sensor space data.
        backward
            Fit a backward model (reconstruct the stimulus from the brain response).
            Note that in this case latencies are relative to the brain response time:
            to reconstruct ``x`` using the 500 ms response following it,
            set ``backward=True, tstart=-0.500, tstop=0``.
        make
            If the TRF does not exist, estimate it
            (the default is to raise an IOError; this is because estimating new TRFs
            can take a substantial amount of time).
        path_only
            Return the path instead of loading the TRF.
        partition_results
            Keep results for each test-partition (TRFs and model evaluation).
        morph
            Morph source space data to the FSAverage brain.
        estimator : str | Estimator | None
            Named estimator key from ``experiment.estimators`` (e.g. ``'boosting'``),
            or an :class:`~trftools.pipeline.estimator.Estimator` instance. If given,
            its parameters override the explicit delta/basis/partitions/cv/... arguments
            and determine the cache path and fitting call.
        ...
            State parameters.

        Returns
        -------
        Estimated model (or path, if ``path_only`` is True).
        """
        data = TestDims.coerce(data, morph=morph)
        x = self._coerce_model(x)
        # Resolve estimator (str -> instance) and apply effective params
        estimator = self._resolve_estimator(estimator)
        if estimator is not None:
            data, mask, state = estimator.normalize_trf_args(self, data, mask, state)
        effective = self._apply_estimator_params(
            estimator,
            delta=delta,
            mindelta=mindelta,
            error=error,
            basis=basis,
            partitions=partitions,
            cv=cv,
            selective_stopping=selective_stopping,
            partition_results=partition_results,
        )
        delta = effective['delta']
        mindelta = effective['mindelta']
        error = effective['error']
        basis = effective['basis']
        partitions = effective['partitions']
        cv = effective['cv']
        selective_stopping = effective['selective_stopping']
        partition_results = effective['partition_results']
        # check epoch
        epoch = self._epochs[self.get('epoch', **state)]
        if isinstance(epoch, EpochCollection):
            raise ValueError(f"epoch={epoch.name!r} (use .load_trfs() to load multiple TRFs from a collection epoch)")
        # check cache
        dst = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, **state)
        if path_only:
            return dst
        elif exists(dst) and cache_valid(getmtime(dst), self._epochs_mtime()):
            try:
                res = load.unpickle(dst)
            except:
                print(f"Error unpickling {dst}")
                raise
            if data.source:
                update_subjects_dir(res, self.get('mri-sdir'), 4)
            # TRFs from before storing n_samples
            if isinstance(res, BoostingResult) and res.n_samples is None:
                self._log.info("Recovering missing n_samples...")
                meg = self.load_epochs(samplingrate=samplingrate)['meg']
                res.n_samples = meg.shape[0] * meg.shape[meg.get_axis('time')]
                save.pickle(res, dst)
            # check x
            if not backward and hasattr(res, 'x'):  # not NCRF
                # res.x are from a Dataset (except variable length epochs)
                res_keys = [res.x] if isinstance(res.x, str) else res.x
                res_keys = sorted(Dataset.as_key(x) for x in res_keys)
                x_keys = sorted(Dataset.as_key(term) for term in x.term_names)
                if res_keys != x_keys:
                    res_model = Model.from_string(res_keys)
                    x_model = Model.from_string(x_keys)
                    table = model_comparison_table(x_model, res_model, 'x', 'cached result')
                    res_key = '+'.join(res_keys)
                    for name, model in self._named_models.items():
                        if res_key == model.dataset_based_key:
                            table.caption(f"Looks like {name}")
                            break
                    else:
                        table.caption("Model not recognized")
                    raise RuntimeError(f"Result x mismatch:\n{dst}\n{table}")

            if partition_results and res.partition_results is None:
                self._log.info("Refitting TRF (cached TRF exists, but without partition_results; interrupt process to cancel)...")
            else:
                return self._post_process_trf(res, data, mask)

        # try to load from superset parcellation
        if not data.source:
            pass
        elif not mask:
            assert self.get('src')[:3] == 'vol'
        elif mask in self._parc_supersets:
            for super_parc in self._parc_supersets[mask]:
                try:
                    res = self.load_trf(x, tstart, tstop, basis, error, partitions, samplingrate, super_parc, delta, mindelta, filter_x, selective_stopping, cv, data, backward, partition_results=partition_results)
                except IOError:
                    pass
                else:
                    return self._post_process_trf(res, data, super_parc, mask)

        # make it if make=True
        if not make:
            model_desc = ModelDescriber(self._structured_models).describe(x)
            raise IOError(f"TRF {relpath(dst, self.get('root'))} does not exist (model {model_desc!r}); set make=True to compute it.")

        self._log.info("Computing TRF:  %s %s %s %s", self.get('subject'), data.string, '->' if backward else '<-', x.name)
        func = self._trf_job(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, partition_results, estimator=estimator, **state)
        if func is None:
            res = load.unpickle(dst)  # _trf_job() created a link from an equivalent result (NCRF)
        else:
            res = func()
            save.pickle(res, dst)
        return self._post_process_trf(res, data, mask)

    def _locate_trf(
            self,
            x: ModelArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            backward: bool = False,
            allow_new: bool = False,  # This is a new TRF (i.e., generate new model names)
            **state):
        "Return path of the corresponding trf-file"
        # FIXME: if filter_x == 'continuous': check whether all x are same type
        self._set_trf_options(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, state=state, allow_new=allow_new)

        path = self.get('trf-file', mkdir=True)
        if len(os.path.basename(path)) > 255:
            raise FilenameTooLong(path)
        elif exists(path) and not cache_valid(getmtime(path), self._epochs_mtime()):
            os.remove(path)
        return path

    def _trf_job(
            self,
            x: ModelArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            backward: bool = False,
            partition_results: bool = False,
            estimator: Estimator = None,
            **state,
    ) -> Optional[Callable]:
        "Return function to create TRF result"
        data = TestDims.coerce(data)
        epoch = self.get('epoch', **state)
        assert not isinstance(self._epochs[epoch], EpochCollection)
        x = self._coerce_model(x)
        if not x:
            return

        if estimator is not None and getattr(estimator, 'name', None) == 'ncrf':
            return self._trf_job_ncrf_estimator(
                x,
                tstart,
                tstop,
                partitions,
                samplingrate,
                filter_x,
                data,
                estimator,
            )

        if data.source:
            inv = self.get('inv')
            m = NCRF_RE.match(inv)
            if m:
                data = TestDims('sensor')
            else:
                morph = is_fake_mri(self.get('mri-dir'))
                data = TestDims.coerce(data, morph=morph)
        else:
            inv = m = None

        # NCRF: Cross-validations
        ncrf_args = {'mu': 'auto'}
        if m:
            ncrf_tag = m.group(2) or ''  # group 2 is None when not present
        else:
            ncrf_tag = ''

        if ncrf_tag.isdigit():
            ncrf_args['mu'] = float(ncrf_tag) / 10000
        elif ncrf_tag == '50it':
            ncrf_args['n_iter'] = 50
        elif ncrf_tag == 'no_champ':
            ncrf_args.update(n_iter=1, n_iterf=1000, n_iterc=0)
        elif ncrf_tag:
            # find best mu from previous cross-validations
            with self._temporary_state:
                cv = self.load_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv=m.group(1))

            if ncrf_tag == 'l2':
                ncrf_args['mu'] = cv.cv_mu('l2')
            elif ncrf_tag == 'l2mu':
                ncrf_args['mu'] = cv.cv_mu('l2/mu')
            elif ncrf_tag == 'cv2':
                grade = 10
                cv_results = sorted(cv._cv_results, key=attrgetter('mu'))
                best_cv = min(cv_results, key=attrgetter('cross_fit'))
                i = cv_results.index(best_cv)
                ncrf_args['mu'] = np.logspace(np.log10(cv_results[i-1].mu), np.log10(cv_results[i+1].mu), grade+2)[1:-1]
            else:
                raise RuntimeError(f'inv={inv!r}')
            # check whether fit with mu exists
            if ncrf_tag.startswith('l2'):
                src_inv = None
                if ncrf_args['mu'] == cv.mu:
                    src_inv = 'dstrf'
                elif inv == 'dstrf-l2mu':
                    with self._temporary_state:
                        l2_trf = self.load_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv='dstrf-l2')
                    if ncrf_args['mu'] == l2_trf.mu:
                        src_inv = 'dstrf-l2'
                # if fit with mu exists, link it
                if src_inv is not None:
                    with self._temporary_state:
                        src = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv=src_inv)
                        dst = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, inv=inv)
                    os.link(src, dst)
                    return

        # load data
        if m:
            ds = self.load_epochs(samplingrate=samplingrate, data=data)
        elif data.source is True:
            # FIXME: in TRF-path, use mri-subject rather than mri value
            ds = self.load_epochs_stc(baseline=False, mask=mask, samplingrate=samplingrate, morph=data.morph)
        elif data.sensor:
            interpolate_bads = (data.sensor is True) and not backward
            ds = self.load_epochs(samplingrate=samplingrate, data=data, interpolate_bads=interpolate_bads)
        else:
            raise NotImplemented(f"data={data.string!r}")
        y = ds[data.y_name]
        is_variable_time = isinstance(y, Datalist)
        # load predictors
        xs = []
        for term in sorted(x.term_names):
            code = Code.coerce(term)
            self.add_predictor(ds, code, filter_x, data.y_name)
            xs.append(ds[code.key])

        # determine partitions for NCRF
        if m:
            assert partitions is None
            if is_variable_time:
                partitions = 1
            elif (y.time.nsamples * y.time.tstep) / tstop < 30:
                # make sure chunk size is at least 30 TRFs
                partitions = -1
            else:
                partitions = 1

        # reshape data
        if partitions is None:
            if not 3 <= ds.n_cases <= 10:
                # Default when estimator doesn't set partitions (e.g. NCRF with sensor space) or many cases
                partitions = min(5, max(2, ds.n_cases // 2))
        elif partitions < 0:
            partitions = None if partitions == -1 else -partitions
            y = concatenate(y)
            xs = [concatenate(x) for x in xs]

        if len(xs) == 1:
            xs = xs[0]
            if backward:
                y, xs = xs, y
        elif backward:
            raise ValueError("backward model with more than one stimulus")
        else:
            # check that all x have unique names
            names = [x_.name for x_ in xs]
            if len(set(names)) < len(names):
                raise ValueError(f"Multiple predictors with same name: {', '.join(names)}")
            if m and is_variable_time:
                # [[u1, u2], [v1, v2]] -> [[u1, v1], [u2, v2]]
                xs = list(zip(*xs))

        if m:
            y0 = y[0] if is_variable_time else y
            fwd = self.load_fwd(ndvar=True)
            cov = self.load_cov()
            # NCRF assumes data has subset of sensors in covariance
            if set(y0.sensor.names).difference(cov.ch_names):
                if is_variable_time:
                    y = [yi.sub(sensor=cov.ch_names) for yi in y]
                else:
                    y = y.sub(sensor=cov.ch_names)
            from ncrf import fit_ncrf
            if estimator is not None:
                ncrf_args = {**ncrf_args, **estimator.parameters_for_partial()}
            ncrf_args.setdefault('normalize', True)
            ncrf_args.setdefault('in_place', True)
            return partial(fit_ncrf, y, xs, fwd, cov, tstart, tstop, **ncrf_args)
        boosting_args = {
            'delta': delta,
            'mindelta': mindelta,
            'error': error,
            'basis': basis,
            'partitions': partitions,
            'test': cv,
            'selective_stopping': selective_stopping,
            'partition_results': partition_results,
        }
        return partial(boosting, y, xs, tstart, tstop, 'inplace', **boosting_args)

    def _trf_job_ncrf_estimator(
            self,
            x: Model,
            tstart: float,
            tstop: float,
            partitions: int,
            samplingrate: int,
            filter_x: FilterXArg,
            data: DataArg,
            estimator: Estimator,
    ) -> Callable:
        "Return NCRF fit job for estimator-driven entry points."
        ds = self.load_epochs(samplingrate=samplingrate, data=data)
        y = ds[data.y_name]
        is_variable_time = isinstance(y, Datalist)

        xs = []
        for term in sorted(x.term_names):
            code = Code.coerce(term)
            self.add_predictor(ds, code, filter_x, data.y_name)
            xs.append(ds[code.key])

        if partitions is None:
            if is_variable_time:
                partitions = 1
            elif (y.time.nsamples * y.time.tstep) / tstop < 30:
                # Keep chunk size large enough for stable NCRF fits.
                partitions = -1
            else:
                partitions = 1

        if partitions < 0:
            partitions = None if partitions == -1 else -partitions
            y = concatenate(y)
            xs = [concatenate(x_) for x_ in xs]

        if len(xs) == 1:
            xs = xs[0]
        else:
            names = [x_.name for x_ in xs]
            if len(set(names)) < len(names):
                raise ValueError(f"Multiple predictors with same name: {', '.join(names)}")
            if is_variable_time:
                xs = list(zip(*xs))

        y0 = y[0] if is_variable_time else y
        fwd = self.load_fwd(ndvar=True)
        cov = self.load_cov()
        if set(y0.sensor.names).difference(cov.ch_names):
            if is_variable_time:
                y = [yi.sub(sensor=cov.ch_names) for yi in y]
            else:
                y = y.sub(sensor=cov.ch_names)

        from ncrf import fit_ncrf

        ncrf_args = {'mu': 'auto', **estimator.parameters_for_partial()}
        ncrf_args.setdefault('normalize', True)
        ncrf_args.setdefault('in_place', True)
        return partial(fit_ncrf, y, xs, fwd, cov, tstart, tstop, **ncrf_args)

    def load_trfs(
            self,
            subject: Union[str, int],
            x: ModelArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            backward: bool = False,
            make: bool = False,
            scale: str = None,
            smooth: float = None,
            smooth_time: float = None,
            vardef: Union[None, str, Variables] = None,
            permutations: int = 1,
            vector_as_norm: bool = False,
            trfs: bool = True,
            partition_results: bool = False,
            **state):
        """Load TRFs for the group in a Dataset (see ``.load_trf()``)

        Parameters
        ----------
        subject : str | 1 | -1
            Subject(s) for which to load data. Can be a single subject
            name or a group name such as ``'all'``. ``1`` to use the current
            subject; ``-1`` for the current group.
        x : str
            One or more predictor variables, joined with '+'.
        tstart
            Start of the TRF in s (default 0).
        tstop
            Stop of the TRF in s (default 0.5).
        basis
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions
            Number of partitions used for cross-validation in boosting. A
            positive number to divide epochs evenly (e.g., ``partitions=5`` to
            group ``epochs[0::5]``, ``epochs[1::5]``, ..., ``epochs[4::5]``.
            A negative number to concatenate all epochs and then divide the
            resulting time series into ``abs(partitions)`` contiguous, equal
            length segments.
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta
            Boosting delta.
        mindelta
            Boosting parameter.
        filter_x
            Filter ``x`` with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        selective_stopping
            Stop boosting each predictor separately.
        cv
            Use cross-validation.
        data : 'source' | 'meg' | 'eeg'
            Analyze source-space data (default) or sensor space data.
        backward
            Fit a backward model (reconstruct the stimulus from the brain response).
            Note that in this case latencies are relative to the brain response time:
            to reconstruct ``x`` using the 500 ms response following it,
            set ``backward=True, tstart=-0.500, tstop=0``.
        make
            If some of the TRFs do not exist, estimate those TRFs
            (the default is to raise an IOError; this is because estimating new TRFs
            can take a substantial amount of time).
        scale : 'original'
            Rescale TRFs to the scale of the source data (default is the scale
            based on normalized predictors and responses).
        smooth
            Smooth TRFs (spatial smoothing, in [m] STD of Gaussian).
        smooth_time
            Smooth TRFs temporally.
        vardef : str
            Add variables for a given test.
        permutations
            When loading a partially permuted model, average the result
            of ``permutations`` different permutations.
        vector_as_norm
            For vector data, return the norm at each time point instead of the
            vector.
        trfs
            Load TRFs. If TRFs are not needed, setting ``trfs=False`` can speed
            up loading for complex model.
        partition_results
            Keep results for each test-partition (TRFs and model evaluation).
            Partition results are currently not available in the TRFs dataset,
            but setting ``partition_results=True`` will make sure that the TRFs
            that are computed contain the partition-specific results.
        ...
            Experiment state parameters.

        Returns
        -------
        trf_ds : Dataset
            Dataset with the following variables: ``subject``, ``r``
            (correlation map) and one NDVar for each component of the TRF.
            ``trf_ds.info['xs']`` is a tuple of the names of all TRF components.
        """
        data = TestDims.coerce(data)
        subject, group = self._process_subject_arg(subject, state)
        x = self._coerce_model(x)

        # group data
        if group is not None:
            dss = []
            for _ in self.iter(group=group, progress_bar="Load TRFs"):
                ds = self.load_trfs(1, x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, scale, None, None, vardef, permutations, vector_as_norm, trfs)
                dss.append(ds)
            ds = combine(dss, to_list=True)
            self._smooth_trfs(data, ds, smooth, smooth_time)
            return ds

        # collection epochs
        epoch = self._epochs[self.get('epoch')]
        if isinstance(epoch, EpochCollection):
            dss = []
            with self._temporary_state:
                for sub_epoch in epoch.collect:
                    ds = self.load_trfs(1, x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, scale, None, None, None, permutations, vector_as_norm, trfs, epoch=sub_epoch)
                    dss.append(ds)
            ds = combine(dss)
            self._smooth_trfs(data, ds, smooth, smooth_time)
            self._add_vars(ds, self._variables, group_only=True)
            self._add_vars(ds, vardef, group_only=True)
            return ds

        # single subject and epoch
        if permutations == 1:
            xs = (x,)
        elif not x.has_randomization:
            raise ValueError(f"{permutations=} for model without randomization ({x.name})")
        else:
            xs = x.multiple_permutations(permutations)

        if data.source:
            inv = self.get('inv')
            is_ncrf = bool(NCRF_RE.match(inv))
            is_vector_data = is_ncrf or inv.startswith('vec')
        else:
            is_vector_data = is_ncrf = False

        # load result(s)
        h = r = z = r1 = z1 = residual = det = tstep = res_partitions = mu = None
        for x_ in xs:
            res = self.load_trf(x_, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, morph=True)
            # Fit metrics
            if tstep is None:  # (first iteration)
                if is_ncrf:
                    tstep = res.tstep
                    mu = res.mu
                else:
                    res_partitions = res.partitions
                    r = res.r
                    z = arctanh(r, info={'unit': 'z(r)'})
                    if is_vector_data:
                        r1 = res.r_l1
                        z1 = arctanh(r1, info={'unit': 'z(r)'})
                    residual = res.residual
                    det = res.proportion_explained
                    one_h = res.h_source if isinstance(res.h_source, NDVar) else res.h_source[0]
                    tstep = one_h.time.tstep
            else:
                if is_ncrf:
                    mu = 0  # how to combine multiple mu?
                else:
                    assert res.partitions == res_partitions
                    r += res.r
                    z += arctanh(res.r)
                    if r1 is not None:
                        r1 += res.r_l1
                        z1 += arctanh(res.r_l1)
                    residual += res.residual
                    det += res.proportion_explained
            # TRFs
            if trfs:
                if scale is None:
                    res_h = res.h
                elif scale == 'original':
                    res_h = res.h_scaled
                else:
                    raise ValueError(f"{scale=}")
                # make sure h is a tuple
                if isinstance(res_h, NDVar):
                    res_h = [res_h]
                # morph to average brain
                if is_ncrf:
                    res_h = [morph_source_space(h, 'fsaverage') for h in res_h]
                # for vector results, the average norm is relevant
                if is_vector_data and vector_as_norm:
                    res_h = [hi.norm('space') for hi in res_h]
                # combine results
                if h is None:
                    h = res_h
                else:
                    for hi, res_hi in zip(h, res_h):
                        hi += res_hi
        if not trfs:
            h = ()

        # average
        if permutations > 1:
            for hi in h:
                hi /= permutations
            if not is_ncrf:
                r /= permutations
                z /= permutations
                if is_vector_data:
                    r1 /= permutations
                    z1 /= permutations
                residual /= permutations
                det /= permutations

        # output Dataset
        x_keys = [Dataset.as_key(term) for term in x.term_names]
        ds = Dataset(info={'xs': x_keys, 'x_names': x.term_names, 'samplingrate': 1 / tstep, 'partitions': partitions or res_partitions}, name=self._x_desc(x))
        ds['subject'] = Factor([subject], random=True)
        ds[:, 'epoch'] = epoch.name
        if is_ncrf:
            ds[:, 'mu'] = mu
        else:
            ds[:, 'r'] = r
            ds[:, 'residual'] = residual
            ds[:, 'det'] = det
            ds[:, 'z'] = z
            if is_vector_data:
                ds['r1'] = r1[newaxis]
                ds['z1'] = z1[newaxis]

        # add kernel to dataset
        for hi in h:
            ds[Dataset.as_key(hi.name)] = hi[newaxis]

        self._smooth_trfs(data, ds, smooth, smooth_time)
        self._add_vars(ds, self._variables, group_only=True)
        self._add_vars(ds, vardef, group_only=True)
        return ds

    @staticmethod
    def _smooth_trfs(data, ds, smooth, smooth_time):
        # TODO: optimize smoothing (cache smoother Dimension._gaussian_smoother())
        if smooth:
            if data.source is not True:
                raise ValueError(f"smooth={smooth!r} with data={data.string!r}")
            keys = [key for key in ('residual', 'det', 'r', 'z', 'r1', 'z1') if key in ds]
            for key in chain(ds.info['xs'], keys):
                if key not in ds:
                    continue
                ds[key] = ds[key].smooth('source', smooth, 'gaussian')
        if smooth_time:
            for key in ds.info['xs']:
                if key not in ds:
                    continue
                ds[key] = ds[key].smooth('time', smooth_time)

    def _locate_missing_trfs(
            self,
            x: ModelArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            backward: bool = False,
            partition_results: bool = False,
            permutations: int = 1,
            existing: bool = False,
            **state,
    ) -> List[Tuple[str, Tuple[dict, dict], dict]]:
        "Return ``(path, state, args)`` for ._trf_job() for each missing trf-file"
        data = TestDims.coerce(data)
        x = self._coerce_model(x)
        if not x:
            return []
        if state:
            self.set(**state)

        out = []
        args = (x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, partition_results)

        # multiple permutations
        if permutations > 1 and x.has_randomization:
            args = args[1:]
            for xi in x.multiple_permutations(permutations):
                out.extend(self._locate_missing_trfs(xi, *args, existing=existing))
            return out

        # EpochCollection: separate TRF for member epochs
        epoch = self._epochs[self.get('epoch')]
        if isinstance(epoch, EpochCollection):
            with self._temporary_state:
                for epoch_ in epoch.collect:
                    out.extend(self._locate_missing_trfs(*args, existing=existing, epoch=epoch_))
            return out

        # one model, one epoch
        for _ in self:
            path = self._locate_trf(*args[:-1], allow_new=True)
            if not existing:
                if os.path.exists(path):
                    continue  # TRF exists for requested mask
                # Check whether TRF exists for superset parc
                super_exists = False
                for super_parc in self._parc_supersets.get(mask, ()):
                    spath = self._locate_trf(x, tstart, tstop, basis, error, partitions, samplingrate, super_parc, delta, mindelta, filter_x, selective_stopping, cv, data, backward)
                    if os.path.exists(spath):
                        super_exists = True
                        break
                if super_exists:
                    continue
            out.append((path, self._copy_state(), args))
        return out

    def _assert_fsaverage_sym_exists(self):
        fsa_sym_dir = Path(self.get('mri-sdir')) / 'fsaverage_sym'
        if not fsa_sym_dir.exists():
            raise FileMissingError("The fsaverage_sym brain is needed for xhemi tests and is missing from the MRI directory")

    def _xhemi_parc(self):
        parc = self.get('parc')
        with self._temporary_state:
            if parc == 'lobes':
                parc = 'cortex'
                self.set(parc=parc)
            self.make_annot(mrisubject='fsaverage_sym')
            self.make_src()
        return parc

    def load_trf_test(
            self,
            x: ModelArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            term: str = None,
            terms: Union[str, Sequence[str]] = None,
            permutations: int = 1,
            make: bool = False,
            scale: str = None,
            smooth: float = None,
            smooth_time: float = None,
            pmin: PMinArg = 'tfce',
            samples: int = 10000,
            test: Union[str, bool] = True,
            return_data: bool = False,
            xhemi: bool = False,
            xhemi_smooth: float = 0.005,
            **state,
    ):
        """Load TRF test result

        Parameters
        ----------
        x : str
            Model for which to load TRFs. By default, all TRFs in the model are
            tested; use ``term`` to test only TRFs of specific predictors.
        tstart
            Start of the TRF in s (default 0).
        tstop
            Stop of the TRF in s (default 0.5).
        basis
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions
            Number of partitions used for cross-validation in boosting (default
            is the number of epochs; -1 to concatenate data).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta
            Boosting delta.
        mindelta
            Boosting parameter.
        filter_x
            Filter ``x`` with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        selective_stopping
            Stop boosting each predictor separately.
        cv
            Use cross-validation.
        data : 'source' | 'meg' | 'eeg'
            Analyze source-space data (default) or sensor space data.
        term
            TRF to test (by default all TRFs in the model).
            Mutually exclusive with the ``terms`` parameter.
        terms
            TRFs to test (by default all TRFs in the model).
            Multiple TRFs can be specified as list or with a :mod:`fnmatch`
            pattern.
            Mutually exclusive with the ``term`` parameter.
        permutations
            When testing against a partially permuted model, average the result
            of ``permutations`` different permutations as baseline model.
        make
            If a TRF does not exist, make it (the default is to raise an
            IOError).
        scale : 'original'
            Rescale TRFs to the scale of the source data (default is the scale
            based on normalized predictors and responses).
        smooth
            Smooth data in space before test (value in [m] STD of Gaussian).
        smooth_time
            Smooth TRFs temporally.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        samples
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 10,000).
        test : True | str
            Test to perform (default ``True`` is test against 0).
        return_data
            Return the data along with the test result (see below).
        xhemi
            Test between hemispheres.
        xhemi_smooth
            Smooth TRFs before morphing to the other hemisphere (Gaussian std
            in [m]; default 0.005 (5 mm)).
        ...
            Experiment state parameters.

        Returns
        -------
        ds : Dataset (if return_data==True)
            Data that forms the basis of the test.
        res : ResultCollection
            Test results for the specified test.
        """
        data = TestDims.coerce(data)
        if state:
            self.set(**state)
        if data.source:
            inv = self.get('inv')
            is_vector_data = inv.startswith('vec')
        elif xhemi:
            raise ValueError(f"{xhemi=} for data={data.string!r}")
        else:
            is_vector_data = False
        # determine whether baseline model is needed:
        if test is True and is_vector_data and not xhemi:
            assert not cv
            compare_with_baseline_model = True
        else:
            assert permutations == 1
            compare_with_baseline_model = False
        # vector data can not be tested against 0
        if compare_with_baseline_model:
            raise NotImplementedError

        # which terms to test
        model = self._coerce_model(x)
        if terms is None and term is None:
            terms = [term_i.string for term_i in model.terms]
        elif term is None:
            if isinstance(terms, str):
                term_names = [term_i.string for term_i in model.terms]
                term_list = [term_i for term_i in term_names if fnmatch.fnmatch(term_i, terms)]
                if not term_list:
                    raise ValueError(f"{terms=}: not matching TRF among {', '.join(term_names)}")
                terms = term_list
            else:
                terms = list(terms)
        elif terms is None:
            terms = [term]
        else:
            raise TypeError(f"{term=}, {terms=}")
        return_one = term is not None

        if xhemi:
            if xhemi_smooth % 0.001:
                raise ValueError(f'{xhemi_smooth=}; parameter in [m] needs to be integer number of [mm]')
            test_options = (f'xhemi-abs-{xhemi_smooth * 1000:.0f}',)
        else:
            test_options = ()

        if xhemi:
            test_obj = TTestRelated('hemi', 'lh', 'rh') if test is True else self.tests[test]
            parc = self._xhemi_parc()
        else:
            test_obj = TTestOneSample() if test is True else self.tests[test]
            parc = None
        if isinstance(test_obj, TwoStageTest):
            raise NotImplementedError

        out = ResultCollection()
        ds_out = trf_ds = trf_res = lms = None
        desc = 'X-Hemi ' if xhemi else ''
        for term_i in tqdm(terms, f"{desc}TRF-Tests for {model.name}", leave=False):
            self._set_trf_options(model, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, pmin=pmin, is_group_result=True, scale=scale, smooth_source=smooth, smooth_time=smooth_time, test=test, test_options=[term_i, *test_options], permutations=permutations)

            # check if cached
            dst = self.get('trf-test-file', mkdir=True)
            if self._result_file_mtime(dst, data):
                res = load.unpickle(dst)
                if res.samples >= samples or res.samples == -1:
                    if data.source:
                        update_subjects_dir(res, self.get('mri-sdir'), 2)
                else:
                    res = None
            else:
                res = None

            # load data
            if trf_ds is None and (res is None or return_data):
                if xhemi:
                    trf_ds, trf_res = self.load_trf_test(model, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, term_i, None, permutations, make, scale, smooth, smooth_time, pmin, test=test, return_data=True)
                    if test is True:
                        ds_out = Dataset(info=trf_ds.info)
                        ds_out['subject'] = trf_ds['subject'].tile(2)
                        ds_out['hemi'] = Factor(('lh', 'rh'), repeat=trf_ds.n_cases)
                    else:
                        ds_out = trf_ds.copy()
                elif isinstance(test_obj, TwoStageTest):
                    # stage 1
                    lms = {y: [] for y in terms}
                    ds_out = []
                    for subject in self.iter():
                        ds = self.load_trfs(1, model, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make, scale=scale, smooth=smooth, smooth_time=smooth_time, vardef=test_obj.vars, permutations=permutations)
                        for term_j in terms:
                            key = Dataset.as_key(term_j)
                            lms[term_j].append(test_obj.make_stage_1(key, ds, subject))
                        if return_data:
                            ds_out.append(ds)
                    if return_data:
                        ds_out = combine(ds_out)
                else:
                    trf_ds = ds_out = self.load_trfs(-1, model, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, make=make, scale=scale, smooth=smooth, smooth_time=smooth_time, vardef=test_obj.vars, permutations=permutations, vector_as_norm=True)
            # do test
            if res is None:
                key = Dataset.as_key(term_i)
                test_kwargs = self._test_kwargs(samples, pmin, None, None, data, None)
                if xhemi:
                    # data
                    y = trf_ds[key].abs()
                    if xhemi_smooth:
                        y = y.smooth('source', xhemi_smooth, 'gaussian')
                    lh, rh = eelbrain.xhemi(y, parc=parc)
                    if test is True:
                        y = combine((lh, rh))
                    else:
                        y = lh - rh
                    # mask
                    mask_lh, mask_rh = eelbrain.xhemi(trf_res[key].p <= 0.05, parc=parc)
                    np.maximum(mask_lh.x, mask_rh.x, mask_lh.x)
                    mask = mask_lh > .5
                    y *= mask
                    # test
                    res = self._make_test(y, ds_out, test_obj, test_kwargs)
                    if return_data:
                        ds_out[key] = y
                elif isinstance(test_obj, TwoStageTest):
                    # stage 2
                    res = test_obj.make_stage_2(lms[term_i], test_kwargs)
                else:
                    res = self._make_test(key, trf_ds, test_obj, test_kwargs)
                save.pickle(res, dst)
            out[term_i] = res

        if return_one:
            out = out[term]
        if return_data:
            return ds_out, out
        else:
            return out

    def _set_trf_options(
            self,
            x: ModelOrComparisonArg,
            tstart: float,
            tstop: float,
            basis: float,
            error: str,
            partitions: int,
            samplingrate: Optional[int],
            mask: Union[bool, str],
            delta: float,
            mindelta: float,
            filter_x: FilterXArg,
            selective_stopping: int,
            cv: bool,
            data: DataArg,
            backward: bool = False,
            pmin=None,
            is_group_result: bool = False,
            metric: str = None,
            scale: str = None,
            smooth_source: float = None,
            smooth_time: float = None,
            is_public: bool = False,
            test: Union[str, bool] = None,
            test_options: Union[str, Sequence[str]] = None,
            permutations: int = 1,
            by_subject: bool = False,
            public_name: str = None,
            state: dict = None,  # avoid _set_trf_options(**state) because _set_trf_options could catch invalid state parameters like `scale`
            allow_new: bool = False,  # This is a new TRF (i.e., generate new model names)
    ):
        if metric and not FIT_METRIC_RE.match(metric):
            raise ValueError(f'{metric=}')
        data = TestDims.coerce(data)

        if test:
            if state is None:
                state = {}
            if test is True:
                state['test'] = ''
            elif isinstance(test, str):
                state['test'] = test
                state['match'] = False
            else:
                raise TypeError(f"{test=}")

        if state:
            self.set(**state)
        dstrf = self.get('inv') == 'dstrf'

        if dstrf:
            # make sure we are receiving default values
            assert basis == 0.050
            basis = None
            assert error == 'l1'
            error = None
            assert partitions is None
            partitions = -1
            assert mask is None
            assert delta == 0.005
            assert mindelta is None
            assert selective_stopping == 0
            assert data.source is True
            assert backward is False

        # model description
        if public_name is not None:
            assert is_public
            x_name = public_name
        else:
            x_name = self._x_desc(x, is_public, allow_new)

        # TRF method
        trf_options = [] if dstrf else ['boosting']
        # basis
        if basis:
            trf_options.append(f'h{ms(basis)}')
        if error:
            trf_options.append(error)
        # cross-validation
        if partitions is None:
            trf_options.append('seg')
        elif partitions > 0:
            trf_options.append(f'{partitions}ptns')
        elif partitions < 0:
            trf_options.append(f'con{-partitions}ptns')
        # backward model
        if backward:
            trf_options.append('backward')
        # filter regressors
        if filter_x is True:
            trf_options.append('filtx')
        elif filter_x:
            trf_options.append(f'filtx={filter_x}')
        # delta
        assert 0. < delta < 1.
        if delta != 0.005 or mindelta is not None:
            if delta != 0.005:
                delta = str(delta)[2:]
            else:
                delta = ''
            if mindelta is not None:
                assert 0. < mindelta < delta
                mindelta = '>' + str(mindelta)[2:]
            else:
                mindelta = ''
            trf_options.append(delta + mindelta)
        if selective_stopping:
            assert isinstance(selective_stopping, int)
            trf_options.append(f'ss{selective_stopping}')
        if cv:
            trf_options.append('cv')
        if scale is not None:
            assert scale in ('original',)
            trf_options.append(scale)
        if smooth_source:
            mm = smooth_source * 1000.
            assert int(mm) == mm
            assert mm < 50.
            trf_options.append(f"s{int(mm)}mm")
        if smooth_time:
            trf_options.append(smooth_time if isinstance(smooth_time, str) else f"s{ms(smooth_time)}")
        # mask
        src = self.get('src')
        if mask:
            if not isinstance(mask, str):
                raise TypeError(f"{mask=}")
            elif data.source is not True:
                raise ValueError(f"{mask=} with data={data.string!r}")
            elif src.startswith('vol'):
                raise ValueError(f"{mask=} with {src=}")
        else:
            assert mask is None
            if not dstrf and data.source is True and not src.startswith('vol'):
                raise ValueError(f"{mask=} with {src=}")

        options = [x_name]
        # whether TRF or test (for backwards compatibility)
        if is_group_result:
            folder = trf_options
        else:
            options.extend(trf_options)
            if mask:
                options.insert(0, mask)
            folder = ()

        if metric:
            options.append(metric)
        if test_options:
            if isinstance(test_options, str):
                options.append(test_options)
            else:
                options.extend(test_options)
        if permutations != 1:
            options.append(f'{permutations}-pmts')
        if by_subject:
            options.append('subjects')

        if samplingrate is None:
            epoch = self._epochs[self.get('epoch')]
            if epoch.samplingrate is None:
                # FIXME: samplingrate is needed for filename to make sure we don't create redundant TRFs; could read in raw samplingrate during initialization and store it if all files have the same samplingrate
                raise NotImplementedError(f"{samplingrate=} with epoch {self.get('epoch')} ({epoch.samplingrate=}); set samplingrate parameter either when loading TRF or on epoch definition")
            samplingrate = epoch.samplingrate

        self._set_analysis_options(data, False, False, pmin, tstart, tstop, None, mask, samplingrate, options, folder)

    def _parse_trf_test_options(self, test_options: FieldCode):
        # FIXME:  is invalid for group result which has different options order
        code = FieldCode.coerce(test_options)
        out = self._parse_test_options(code)
        # mask
        if code.lookahead_1 in self._parcs:
            out['mask'] = code.next()
        elif code.lookahead(2).startswith('model'):
            out['mask'] = f"*{code.next()}*"
        # model
        model = code.next()
        if code.lookahead_1.startswith('('):
            assert '$' not in model
            out['rand'] = code.next()
        elif '$' in model:
            model, rand = model.split('$')
            out['rand'] = '$' + rand
        out['model'] = model
        # trf-options
        out['trf_options'] = ' '.join(code)
        return out

    def _parse_predictor_path(self, filename: str):
        "Parse a predictor filename into components"
        path = Path(filename)
        assert path.suffix == '.pickle'
        name = path.stem
        stimulus, predictor = name.split('~')
        if ' ' in stimulus:
            subject, stimulus = stimulus.split(' ')
        else:
            subject = None
        return {'subject': subject, 'stimulus': stimulus, 'predictor': predictor}

    def _parse_trf_path(self, filename: str):
        """Parse a TRF filename into components

        Notes
        -----
        template is ``{trf-sdir}/{subject}/{analysis}/{epoch} {test_options}.pickle``
        """
        path = Path(filename)
        epoch, test_options = path.stem.split(' ', 1)
        code = FieldCode(test_options)
        out = self._parse_trf_test_options(code)
        out['subject'] = path.parent.parent.name
        out['analysis'] = path.parent.name
        out['epoch'] = epoch
        return out

    def _coerce_model(self, x: Union[str, Model]) -> Model:
        if isinstance(x, Model):
            return x
        elif x in self._structured_models:
            return self._structured_models[x].model
        elif x in self._named_models:
            return self._named_models[x]
        return ModelExpression.from_string(x).initialize(self._structured_models)

    def _coerce_comparison(
            self,
            x: ComparisonArg,
            cv: bool,
    ) -> Union[Comparison, StructuredModel]:
        # May return a StructuredModel because that includes information about how to test each term
        if isinstance(x, str):
            return Comparison.coerce(x, cv, self._structured_models)
        elif not isinstance(x, (StructuredModel, Comparison)):
            raise TypeError(f"{x=}: need comparison")
        return x

    def _coerce_model_or_comparison(
            self,
            x: Union[ComparisonArg, Model, str],
            cv: bool,
    ) -> Union[Model, StructuredModel, Comparison]:
        # Combination of the two above, preferring simpler types
        if isinstance(x, str):
            if x in self._structured_models:
                return self._structured_models[x]
            elif x in self._named_models:
                return self._named_models[x]
            comparison = Comparison.coerce(x, True, self._structured_models)
            if isinstance(comparison, StructuredModel):
                return comparison.model
            return comparison
        elif isinstance(x, (Model, StructuredModel, Comparison)):
            return x
        else:
            raise TypeError(f"{x=}: need model or comparison")

    def _x_desc(
            self,
            x: ModelOrComparisonArg,
            is_public: bool = False,
            allow_new: bool = False,
    ):
        "Description for x"
        if isinstance(x, Model):
            if not x:
                return '0'
            elif is_public:
                return x.name
            elif x.sorted_key in self._model_names:
                return self._model_names[x.sorted_key]
            elif x.without_randomization.sorted_key in self._model_names:
                xrand = x.randomized_component
                xrand_desc = self._x_desc(xrand.without_randomization)
                rand = {term.shuffle_string for term in xrand.terms}
                if len(rand) != 1:
                    raise NotImplementedError(f"{len(rand)} randomization schemes in {x=}")
                rand_desc = f'{xrand_desc}{rand.pop()}'
                if xrand == x:
                    return rand_desc
                base_name = self._model_names[x.without_randomization.sorted_key]
                return f'{base_name} ({rand_desc})'
            elif time.time() - self._last_model_name_refresh > 5:
                self._refresh_model_names()
                return self._x_desc(x, allow_new=allow_new)
            elif allow_new:
                self._register_model(x)
                return self._x_desc(x)
            else:
                raise RuntimeError(f"{x=}: previously unused model")
        elif isinstance(x, Comparison):
            if is_public:
                return x.name
            else:
                x_desc = partial(self._x_desc, allow_new=allow_new)
                return x.compose_name(x_desc, path=True)
        elif isinstance(x, StructuredModel):
            assert is_public  # all internal names should be Model-based
            if x in self._structured_model_names:
                return self._structured_model_names[x]
            elif x.public_name:
                return x.public_name
            else:
                raise RuntimeError(f"{x=}: has no public name")
        else:
            raise TypeError(f"{x=}")

    def load_model_test(
            self,
            x: ComparisonArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            backward: bool = False,
            permutations: int = 1,
            metric: str = 'z',
            smooth: float = None,
            test: str = None,
            return_data: bool = False,
            pmin: PMinArg = 'tfce',
            xhemi: bool = False,
            xhemi_mask: bool = True,
            make: bool = False,
            parameter: str = None,
            compare_to: Any = None,
            tail: int = None,
            partition_results: bool = False,
            **state,
    ):
        """Test comparing model fit between two models

        Parameters
        ----------
        x
            Comparison, or dictionary of named comparisons (see
            :ref:`trf-experiment-comparisons`).
        tstart
            Start of the TRF in s (default 0).
        tstop
            Stop of the TRF in s (default 0.5).
        basis
            Response function basis window width in [s] (default 0.050).
        error : 'l1' | 'l2'
            Error function.
        partitions
            Number of partitions used for cross-validation in boosting (default
            is the number of epochs; -1 to concatenate data).
        samplingrate
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        mask
            Parcellation to mask source space data (only applies when
            ``y='source'``).
        delta
            Boosting delta.
        mindelta
            Boosting parameter.
        filter_x
            Filter ``x`` with the same filter as the M/EEG data.
            ``filter_x=True`` to filter all predictors; ``filter_x=False``
            ``filter_x='continuous'`` to filter time-continuous predictors, but
            not discrete predictors (see :class:`FilePredictor` ``sampling``
            parameter).
        selective_stopping
            Stop boosting each predictor separately.
        cv
            Use cross-validation.
        data : 'source' | 'meg' | 'eeg'
            Analyze source-space data (default) or sensor space data.
        backward
            Fit a backward model (reconstruct the stimulus from the brain response).
            Note that in this case latencies are relative to the brain response time:
            to reconstruct ``x`` using the 500 ms response following it,
            set ``backward=True, tstart=-0.500, tstop=0``.
        permutations
            When testing against a partially permuted model, average the result
            of ``permutations`` different permutations as baseline model.
        metric
            Fit metric to use for test:

            - ``r``:   Pearson correlation
            - ``z``:   z-transformed correlation
            - ``r1``:  1 correlation
            - ``z1``:  z-transformed l1 correlation
            - ``residual``: Residual form model fit
            - ``det``: Proportion of the explained variability

        smooth
            Smooth data in space before test (value in [m] STD of Gaussian).
        test
            Hypothesis to test (default is ``x`` against ``0``).
        return_data
            Return the data the test is performed on alongside the test (if the
            test is cached this incerases loading times).
        pmin
            ``pmin`` value for test.
        xhemi
            Test between hemispheres.
        xhemi_mask
            When doing ``xhemi`` test, mask data with region that is significant
            in at least one hemisphere.
        make
            If the TRFs required for the test do not exist, estimate those TRFs
            (the default is to raise an IOError; this is because estimating new TRFs
            can take a substantial amount of time).
        parameter
            Instead of comparing two models, use ``parameter`` and
            ``compare_to`` to compare the fit of the same model when using
            different parameters. Set ``parameter`` to the name of the parameter
            which differs between models and ``compare_to`` to the baseline
            value. Example:
            ``tstop=1.000, parameter='tstop', compare_to=0.500, tail=1``
            Will test whether a TRF length of 1000 ms is better than 500 ms.
        compare_to
            The value of ``parameter`` in the test condition (the control
            condition will use the standard argument value).
        tail
            Tailedness for ``parameter`` test (default 0, i.e. two-tailed).
        partition_results
            See :meth:`.load_trf`.
        ...
            State parameters.

        Returns
        -------
        ds : Dataset | dict (if return_data==True)
            Dataset with values of the test and baseline models that forms the
            basis of the test.
        res : NDTest | ResultCollection
            Test result.
        """
        data = TestDims.coerce(data, time=False)
        comparison = self._coerce_comparison(x, cv)
        if parameter is not None:
            if isinstance(comparison, StructuredModel):
                comparison = Comparison(self._coerce_model(x), Model(()), tail=tail or 0)
            if test is not None:
                raise TypeError(f"{test=} for {parameter=}")
            test_desc = f'{parameter}={compare_to}'
        elif tail is not None:
            raise TypeError(f"{tail=}: argument only applies to parameter-tests")
        else:
            test_desc = True if test is None else test

        # Load multiple tests for a comparison group
        if isinstance(comparison, StructuredModel):
            if state:
                self.set(**state)
            ress = {comp.test_term_name: self.load_model_test(comp, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, permutations, metric, smooth, test, return_data, pmin, xhemi, xhemi_mask, make, parameter, compare_to, tail, partition_results) for comp in comparison.comparisons(cv)}
            if return_data:
                dss = {key: res[0] for key, res in ress.items()}
                ress = ResultCollection({key: res[1] for key, res in ress.items()})
                return dss, ress
            else:
                return ResultCollection(ress)

        if xhemi:
            self._assert_fsaverage_sym_exists()
            if xhemi_mask:
                test_options = 'xhemi.05'
            else:
                test_options = 'xhemi'
        elif xhemi_mask is not True:
            raise ValueError(f"{xhemi_mask=}; parameter is invalid unless xhemi=True")
        else:
            test_options = None

        self._set_trf_options(comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, pmin, True, metric, test=test_desc, smooth_source=smooth, test_options=test_options, permutations=permutations, state=state, allow_new=True)
        dst = self.get('model-test-file', mkdir=True)
        if self._result_file_mtime(dst, data):
            res = load.unpickle(dst)
            if data.source:
                update_subjects_dir(res, self.get('mri-sdir'), 1)
        else:
            res = None

        y, to_uv = FIT_METRIC_RE.match(metric).groups()
        if to_uv:  # make sure variables that don't affect tests are default
            if xhemi:
                raise ValueError(f"{xhemi=} with {metric=}")
            elif pmin != 'tfce':
                raise ValueError(f"{pmin=} with {metric=}")

        if return_data or res is None:
            # load data
            group = self.get('group')
            vardef = None if test is None else self._tests[test].vars
            x1_permutations = permutations if comparison.x1.has_randomization else 1
            ds1 = self.load_trfs(group, comparison.x1, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, make, trfs=False, vardef=vardef, permutations=x1_permutations, partition_results=partition_results)

            if comparison.x0.terms or parameter is not None:
                if parameter is not None:
                    kwargs = dict(zip(('tstart', 'tstop', 'basis', 'error', 'partitions', 'samplingrate', 'mask', 'delta', 'mindelta', 'filter_x', 'selective_stopping', 'cv'), (tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv)))
                    if parameter not in kwargs:
                        raise ValueError(f'{parameter=}: must be one of {set(kwargs)}')
                    kwargs[parameter] = compare_to
                    ds0 = self.load_trfs(group, comparison.x1, **kwargs, data=data, backward=backward, trfs=False, make=make, vardef=vardef, permutations=permutations, partition_results=partition_results)
                    if comparison.x0.terms:
                        ds1_0 = self.load_trfs(group, comparison.x0, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, trfs=False, make=make, vardef=vardef, permutations=x1_permutations, partition_results=partition_results)
                        ds1[y] -= ds1_0[y]
                        ds0_0 = self.load_trfs(group, comparison.x0, **kwargs, data=data, backward=backward, trfs=False, make=make, vardef=vardef, permutations=permutations, partition_results=partition_results)
                        ds0[y] -= ds0_0[y]
                else:
                    ds0 = self.load_trfs(group, comparison.x0, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, trfs=False, make=make, vardef=vardef, permutations=permutations, partition_results=partition_results)
                # restructure data
                assert np.all(ds1['subject'] == ds0['subject'])
                keep = tuple([k for k in ds1 if isuv(ds1[k]) and np.all(ds1[k] == ds0[k])])
                if test is None:
                    ds = combine((ds1[keep], ds0[keep]))
                else:
                    ds = ds1[keep]
                dss = [ds1, ds0]
            else:
                ds0 = None
                ds = ds1
                dss = [ds1]

            if test is None:
                if xhemi:
                    test_obj = TTestRelated('hemi', 'lh', 'rh')
                elif ds0 is None:
                    test_obj = TTestOneSample(comparison.tail)
                else:
                    test_obj = TTestRelated('model', 'test', 'baseline', comparison.tail)
            else:
                test_obj = self._tests[test]

            # smooth data (before xhemi morph)
            if smooth:
                for ds_i in dss:
                    ds_i[y] = ds_i[y].smooth('source', smooth, 'gaussian')

            if xhemi:
                if ds0 is not None:
                    ds1[y] -= ds0[y]
                lh, rh = eelbrain.xhemi(ds1[y], parc=self._xhemi_parc())
                if test is None:
                    ds[y] = combine((lh, rh))
                    ds['hemi'] = Factor(('lh', 'rh'), repeat=ds1.n_cases)
                else:
                    ds[y] = lh - rh

                # mask
                if xhemi_mask:
                    parc = self._xhemi_parc()
                    with self._temporary_state:
                        base_res = self.load_model_test(comparison, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, backward, permutations, metric, smooth, test, pmin=pmin, make=make, partition_results=partition_results)
                    if isinstance(base_res, MultiEffectNDTest):
                        raise NotImplementedError("xhemi_mask for multi-effect tests")
                    mask_lh, mask_rh = eelbrain.xhemi(base_res.p <= 0.05, parc=parc)
                    np.maximum(mask_lh.x, mask_rh.x, mask_lh.x)
                    ds[y] *= mask_lh > .5
            elif ds0 is None:
                pass
            elif test is None:
                ds[y] = combine((ds1[y], ds0[y]))
                ds['model'] = Factor(('test', 'baseline'), repeat=ds1.n_cases)
            elif isinstance(ds1[y], Datalist):
                ds[y] = Datalist([i1 - i0 for i1, i0 in zip(ds1[y], ds0[y])])
            else:
                ds[y] = ds1[y] - ds0[y]

            # test
            if res is None:
                # test arguments
                kwargs = self._test_kwargs(10000, pmin, None, None, data, None)
                res = self._make_test(y, ds, test_obj, kwargs, to_uv=to_uv)
                save.pickle(res, dst)

            if return_data:
                return ds, res
        return res

    def _locate_model_test_trfs(
            self,
            x: Union[StructuredModel, Comparison],
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            partition_results: bool = False,
            data: DataArg = DATA_DEFAULT,
            permutations: int = 1,
            existing: bool = False,
            **state,
    ) -> List[Tuple[str, Tuple[dict, dict], dict]]:
        """Find required jobs for a report

        Returns
        -------
        trf_jobs
            List of ``(path, state, args)`` tuples for missing TRFs.
        """
        if state:
            self.set(**state)

        if isinstance(x, StructuredModel):
            models = {m for comp in x.comparisons(cv) for m in comp.models}
        else:
            models = x.models

        missing = []
        for model in models:
            missing.extend(
                self._locate_missing_trfs(model, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, False, partition_results, permutations, existing))
        return missing

    def make_model_test_report(
            self,
            x: ComparisonArg,
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str ='l1',
            partitions: int = None,
            samplingrate: float = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            permutations: int = 1,
            metric: str = 'z',
            smooth: float = None,
            surf: str = None,
            views: Union[str, Sequence[str]] = None,
            make: bool = False,
            path_only: bool = False,
            public_name: str = None,
            test: bool = True,
            by_subject: bool = False,
            **state,
    ) -> Optional[str]:
        """Generate report for model comparison

        Parameters
        ----------
        by_subject
            Generate a report with each subject's data.

        Returns
        -------
        str
            Path to the report (only returned with ``path_only=True`` or if the
            report is newly created.
        """
        data = TestDims.coerce(data)
        if data.source is not True:
            raise NotImplementedError("Model-test report for data other than source space")
        x = self._coerce_comparison(x, cv)
        self._set_trf_options(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, metric=metric, smooth_source=smooth, is_group_result=True, is_public=True, test=test, permutations=permutations, by_subject=by_subject, public_name=public_name, state=state)
        dst = self.get('model-report-file', mkdir=True)
        if path_only:
            return dst
        elif exists(dst):
            return
        self._log.info("Make TRF-report: %s", relpath(dst, self.get('model-res-dir')))

        ds, res = self.load_model_test(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, permutations, metric, smooth, test, True, 'tfce', make=make)

        if isinstance(x, StructuredModel):
            comparisons = x.comparisons(cv)
            dss, ress = ds, res
            ds = dss[comparisons[0].test_term_name]
            res = ress[comparisons[0].test_term_name]
        else:
            comparisons = (x,)
            dss = {x.test_term_name: ds}
            ress = {x.test_term_name: res}

        if data.source:
            inv = self.get('inv')
            is_vector_data = inv.startswith('vec')
        else:
            is_vector_data = False

        # Report
        if public_name is None:
            public_name = self._x_desc(x, is_public=True)
        report = fmtxt.Report(public_name)
        report.add_paragraph(self._report_methods_brief(dst))

        if is_vector_data:
            if by_subject:
                raise NotImplementedError
            for comp in comparisons:
                section = trf_report.vsource_tfce_result(ress[comp.test_term_name], comp.test_term_name, f"{comp.x1_only} > {comp.x0_only}")
                report.append(section)
        else:
            surfer_kwargs = self._surfer_plot_kwargs(surf, views)
            if by_subject:
                subjects, diffs = difference_maps(dss)
                if 'hemi' in surfer_kwargs:
                    hemis = (surfer_kwargs.pop('hemi'),)
                else:
                    hemis = ('lh', 'rh')

                for hemi in hemis:
                    section = report.add_section(hemi)
                    brain = plot.brain.brain(ds[metric].source, w=220, h=150, hemi=hemi, **surfer_kwargs)
                    # brain.set_parallel_view(*BRAIN_VIEW[1:])
                    for x_ in comparisons:
                        subsection = section.add_section(x_.test_term_name)
                        row = []
                        for subject, dmap in zip(subjects, diffs[x_.test_term_name, hemi]):
                            brain.add_ndvar(dmap, remove_existing=True)
                            brain.add_text(0, 0, subject, 'subject', (0, 0, 0), font_size=30, justification='left')
                            brain.texts_dict['subject']['text'].property.font_size = 28
                            row.append(brain.image())
                        subsection.append(fmtxt.Figure(row))
                    brain.close()
                report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
                report.save_html(dst)
                return
            brain = None
            for comp in comparisons:
                section, brain = trf_report.source_tfce_result(ress[comp.test_term_name], surfer_kwargs, comp.test_term_name, f"{comp.x1_only} > {comp.x0_only}", brain)
                report.append(section)
            brain.close()

        # Info section
        if test is True:
            test = TTestRelated('model', 'test', 'baseline', comparisons[0].tail)
        sec = report.add_section("Info")
        info = self._report_test_info(sec, ds, test, res, data, model=False)
        # info:
        info.add_item(f"Mask: {mask}")
        # Info: model
        model_info = fmtxt.List("Predictor model")
        if isinstance(x, StructuredModel):
            model_info.add_item("Incremental model improvement for each term")
            model_info.add_item(x.public_name)
        elif isinstance(x, Comparison):
            if x.common_base:
                model_info.add_item("Common base:  " + x.common_base.name)
                model_info.add_item("Test model:  + " + x.x1_only.name)
                model_info.add_item("Baseline model:  + " + x.x0_only.name)
            else:
                model_info.add_item("Test model:  " + x.x1.name)
                model_info.add_item("Baseline model:  " + x.x0.name)
        if permutations > 1:
            model_info.add_item(f"Tests against {permutations} permutations.")
        info.add_item(model_info)
        # Info: reverse correlation method
        trf_info = fmtxt.List("TRF estimation using boosting")
        trf_info.add_item(f"TRF {ms(tstart)} - {ms(tstop)} ms at {ds.info['samplingrate']:g} Hz")
        if basis:
            trf_info.add_item(f"Basis of {ms(basis)} ms Hamming windows")
        trf_info.add_item(f"Error function: {error}")
        trf_info.add_item(f"∆ = {delta}")
        if mindelta is not None:
            trf_info.add_item(f"min-∆ = {mindelta}")
        if ds.info['partitions'] == -1:
            trf_info.add_item(f"Fitted to continuous data with 10 partitions")
        else:
            trf_info.add_item(f"Fitted to segmented data with {ds.info['partitions']} partitions")
        if filter_x:
            trf_info.add_item("Regressors filtered like data")
        if selective_stopping:
            trf_info.add_item(f"Selective stopping after {n_of(selective_stopping, 'failure')}")
        info.add_item(trf_info)
        # Signature
        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst)
        return dst

    def _merge_parcs(
            self,
            src_1: str,  # parcellation to merge
            src_2: str,  # parcellation to merge
            dst: str,  # name of the parcellation forming the union of src_1 and src_2
            verbose: bool = False,
    ):
        """Merge results from two complementary parcellations

        Parameters
        ----------
        src_1
            Name of the parcellation to merge.
        src_2
            Name of the parcellation to merge.
        dst
            Name of the new (combined) parcellation.
        verbose
            Print file names as files are being processed.

        Notes
        -----
        After merging, the source TRFs will be renamed to ``*.backup.pickle``.
        In order to delete the backup files, use::

            e.rm('trf-file', True, test_options='*.backup')
        """
        # make sure parcellations are compatible
        parc_1 = self._parcs[src_1]
        parc_2 = self._parcs[src_2]
        parc_dst = self._parcs[dst]
        assert isinstance(parc_1, SubParc)
        assert isinstance(parc_2, SubParc)
        assert isinstance(parc_dst, SubParc)
        assert parc_1.base == parc_2.base == parc_dst.base
        labels_1 = set(parc_1.labels)
        labels_2 = set(parc_2.labels)
        labels_dst = set(parc_dst.labels)
        assert not labels_1.intersection(labels_2)
        assert labels_1.union(labels_2) == labels_dst
        # target for backup of merged TRFs
        trf_sdir = Path(self.get('trf-sdir'))
        backup_dir = Path(self.get('cache-dir')) / 'trf-backup'
        # move legacy backup files from old backup scheme
        files = list(self.glob('trf-file', True))
        backup_files = [Path(p) for p in files if p.endswith('backup.pickle')]
        if backup_files:
            print(f"Moving {len(backup_files)} old backup files")
            for src_path in backup_files:
                dst_path = backup_dir / src_path.relative_to(trf_sdir)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                src_path.rename(dst_path)
        # find files
        # '{trf-dir}/{analysis}/{epoch_basename} {test_options}.pickle'
        # mask is in test_options
        combine = {}
        dst_exist = 0
        trf_dir = Path(self.get('trf-sdir'))
        for path_1 in trf_dir.glob(f'*/*/* {src_1} *.pickle'):
            path_2 = path_1.parent / path_1.name.replace(f' {src_1} ', f' {src_2} ')
            assert path_2 != path_1
            path_dst = path_1.parent / path_1.name.replace(f' {src_1} ', f' {dst} ')
            if path_2.exists():
                dst_exist += path_dst.exists()
                combine[path_dst] = (path_1, path_2)
        # display
        if not combine:
            print("No files found for merging")
            return
        prompt = f"Merge {len(combine)} file pairs?"
        if dst_exist:
            prompt = f"{prompt}\nWARNING: {dst_exist} target files already exist and will be overwritten"
        while True:
            command = ask(prompt, {'yes': 'merge files', 'show': 'list the files that would be merged'}, allow_empty=True)
            if command == 'yes':
                break
            elif command == 'show':
                for path_dst, (path_1, path_2) in combine.items():
                    print(path_dst.relative_to(trf_dir))
                    print(f'├ {path_1.relative_to(trf_dir)}')
                    print(f'⎿ {path_2.relative_to(trf_dir)}')
                continue
            return
        # make sure target parc exists
        subjects = set()
        for path_dst in combine:
            info = self._parse_trf_path(path_dst)
            subjects.add(info['subject'])
        for subject in subjects:
            self.make_annot(parc=dst, subject=subject)
        # merge
        mri_sdir = self.get('mri-sdir')
        for path_dst, paths_src in combine.items():
            if verbose:
                for path in paths_src:
                    print(f"src: {path}")
                print(f"dst: {path_dst}")
            trfs = [load.unpickle(path) for path in paths_src]
            for trf in trfs:
                update_subjects_dir(trf, mri_sdir, 4)
            res = concatenate(trfs, 'source')
            res._set_parc(dst)
            save.pickle(res, path_dst)
            for src_path in paths_src:
                dst_path = backup_dir / src_path.relative_to(trf_sdir)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                src_path.rename(dst_path)

    def invalidate(
            self,
            regressor: str,
            backup: Union[bool, PathArg] = False,
            verbose: bool = False,
    ):
        """Remove cache and result files when input data becomes invalid

        Parameters
        ----------
        regressor
            Regressor that became invalid. Can be a key in :attr:`.predictors`,
            or a pattern matching one or more regressor names,
            using ``*`` and ``?`` for pattern matching (see examples).
        backup
            Instead of deleting invalidated files, copy them to this directory.
            Can be an absolute path, or relative to experiment root. ``True`` to
            use ``eelbrain-cache-backup``.
        verbose
            Display terms and models that are affected.

        Notes
        -----
        Deletes TRFs and tests. Corresponding predictor files are not affected.

        Examples
        --------
        The examples assume ``e`` is a :class:`TRFExperiment` instance.
        Invalidate all ``gammatone`` predictors::

        >>> e.invalidate('gammatone')

        Invalidate a single predictor::

        >>> e.invalidate('gammatone-8')

        Invalidate multiple predictors using a wild card::

        >>> e.invalidate('gammatone-*')
        """
        # patterns
        if regressor in self.predictors:
            reg_re_term = re.compile(rf"^{regressor}(-\S+)?$")
        else:
            reg_re = fnmatch.translate(regressor)
            reg_re_term = re.compile(rf"^{reg_re}$")

        # find all named models that contain term
        terms = set()
        models = set()
        for name, model in self._named_models.items():
            for term in model.terms_without_randomization:
                if reg_re_term.match(term.code):
                    terms.add(term.code)
                    models.add(name)
                elif reg_re_term.match(term.string):
                    terms.add(term.string)
                    models.add(name)
        if verbose:
            print(f"Terms: {', '.join(sorted(terms))}")
            print(f"Models: {', '.join(sorted(models))}")
        files = set()  # avoid duplicate paths when model name contains regressor name
        counts = defaultdict(lambda: 0)
        for name in models:
            n = len(files)
            files.update(self._find_model_files(name, trfs=True, tests=True))
            counts[name] = len(files) - n

        # cached regressor files (only MakePredictors are cached)
        cache_dir = self.get('predictor-cache-dir', mkdir=True)
        files.update(glob(join(cache_dir, f'*~{regressor} *.pickle')))

        if not files:
            print("No files affected")
            return

        options = {
            'yes': 'delete files',
            'no': 'return without doing anything',
            'files': 'list files to be deleted',
            'models': 'list models including predictor',
        }
        verb = 'moving' if backup else 'deleting'
        while True:
            command = ask(f"Invalidate {regressor} regressor, {verb} {len(files)} files?", options, default='no')
            if command == 'yes':
                print(f"{verb.capitalize()} {len(files)} files...")
                if backup:
                    cache_dir = Path(self.get('cache-dir'))
                    if backup is True:
                        backup_dir = cache_dir.parent / 'eelbrain-cache-backup'
                    else:
                        backup_dir = Path(backup)
                        if not backup_dir.is_absolute():
                            backup_dir = cache_dir.parent / backup_dir
                    # check for existing targets
                    sources = [Path(path) for path in files]
                    targets = [backup_dir / path.relative_to(cache_dir) for path in sources]
                    exist = [path for path in targets if path.exists()]
                    if exist and ask(f'{len(exist)} backup target files already exist, overwrite?', {'yes': 'overwrite backup files', 'no': 'abort'}, default='no') == 'no':
                        return
                    # move files
                    backup_dir.mkdir(exist_ok=True)
                    for source, target in zip(sources, targets):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source.rename(target)
                else:
                    for path in files:
                        os.remove(path)
            elif command == 'no':
                pass
            elif command == 'files':
                print(f"Terms: {', '.join(sorted(terms))}")
                print(f"Models: {', '.join(sorted(models))}")
                paths = sorted(files)
                prefix = os.path.commonprefix(paths)
                print(f"Files in {prefix}:")
                for path in paths:
                    print(relpath(path, prefix))
                continue
            elif command == 'models':
                print(f"Terms: {', '.join(sorted(terms))}\n")
                describer = ModelDescriber(self._structured_models)
                t = fmtxt.Table('lll')
                t.cells('files', 'id', 'model')
                t.midrule()
                for model in models:
                    desc = describer.describe(self._named_models[model])
                    t.cells(counts[model], model, desc)
                print(t, end='\n\n')
                continue
            return

    # Source estimation
    ###################
    def load_psf(self, mask=True, **state):
        """Load inverse point spread function
        
        brain = plot.brain.brain(psf.source, mask=False, hemi='lh')
        brain.add_ndvar(psf[:, 'transversetemporal-lh'][:, 0])
        """
        if isinstance(mask, str):
            state['parc'] = mask
            mask = True
        if state:
            self.set(**state)
        inv_op = self.load_inv(ndvar=True, mask=mask)
        inv_op.source.subject = 'fsaverage'
        inv_op = rename_dim(inv_op, 'source', 'source_to')
        fwd_op = self.load_fwd(ndvar=True, mask=mask).sub(sensor=inv_op.sensor)
        fwd_op.source.subject = 'fsaverage'
        fwd_op = rename_dim(fwd_op, 'source', 'source_from')
        psf = inv_op.dot(fwd_op)
        return psf

    def remove_model(self, model):
        """Remove a named model and delete all associated files

        See Also
        --------
        .clean_models
        .show_models
        """
        if model in self.models:
            raise ValueError(f"{model!r} is an explicitly defined model; remove it from .models")
        self._remove_model(model)

    def show_localization_test(
            self,
            x: ModelArg,
            terms: Union[Sequence[str], Dict[str, str]],  # list of terms, or {label: term} dict
            test_masks: Sequence[NDVar] = None,  # [lh, rh]
            brain_view: Union[str, Sequence[float]] = None,
            axw: float = 4,
            surf: str = 'inflated',
            cortex: Any = ((1.00,) * 3, (.4,) * 3),
            return_data: bool = False,
            metric: str = 'det',
            **kwargs,
    ):
        """Test for localization difference of two terms in a model"""
        if not isinstance(terms, dict):
            terms = {term: term for term in terms}
        # trf_kwargs = {key: value for key, value in kwargs.items() if key not in ('metric',)}
        # load data
        ress = {}
        trf_dss = {}
        for term in ['', *terms.values()]:
            if term:
                comp = self._coerce_comparison(f'{x} | {term}', cv=True)
                model = comp.x0
                ress[term] = self.load_model_test(comp, metric=metric, **kwargs)
            else:
                model = x
            trf_dss[term] = self.load_trfs(-1, model, **kwargs, trfs=False)
        data = trf_report.CompareLocalization(terms, ress, trf_dss, metric, test_masks)
        if return_data:
            return data
        return data.report(brain_view, axw, surf, cortex)

    def show_model_test(
            self,
            x: Union[str, Dict[str, str]],
            brain_view: Union[str, Sequence[float]] = None,
            axw: float = None,
            surf: str = 'inflated',
            cortex: Any = ((1.00,) * 3, (.4,) * 3),
            sig: bool = True,
            heading: fmtxt.FMTextArg = None,
            caption: fmtxt.FMTextArg = None,
            vmax: float = None,
            cmap: str = None,
            alpha: float = 1.,
            view: Union[str, Sequence[str]] = 'lateral',
            xhemi: bool = False,
            data: DataArg = DATA_DEFAULT,
            **test_args,
    ) -> fmtxt.Section:
        """Document section for one or several model tests

        Parameters
        ----------
        x
            Test contrast, or dictionary with labeled contrasts
            ``{label: contrast}``.
        brain_view
            Crop brain view to pre-specified view, or set arguments for
            :meth:`~eelbrain.plot._brain_object.Brain.set_parallel_view`
            ``(forward [mm], up [mm], scale)`` (default scale is 95 for inflated
            surface, 75 otherwise).
        axw
            Brain axes width.
        surf
            FreeSurfer brain surface to plot.
        cortex
            Brain color scheme.
        sig
            Mask by significance (default ``True``)
        heading
            Heading for report section.
        caption
            Caption for the model test table.
        vmax
            Colormap range.
        cmap
            Colormap.
        alpha
            Alpha of the colormap.
        view
            Parameter for brain plot.
        xhemi
            Test lateralization.
        data : 'source' | 'meg' | 'eeg'
            Analyze source-space data (default) or sensor space data.
        ...
            Additional parameters for :meth:`.load_model_test`.

        Notes
        -----
        Surface source space only.
        """
        data = TestDims.coerce(data, time=False)
        test_args['data'] = data
        ress_hemi = None
        if isinstance(x, dict):
            ress = ResultCollection({k: self.load_model_test(m, **test_args) for k, m in x.items()})
            if any(isinstance(res, ResultCollection) for res in ress.values()):
                raise ValueError(f"{x=} contains a contrast that yielded multiple tests; to compare a model against 0, use 'model > 0'")
            if xhemi:
                ress_hemi = ResultCollection({k: self.load_model_test(m, xhemi=True, **test_args) for k, m in x.items()})
        else:
            ress = self.load_model_test(x, **test_args)
            if not isinstance(ress, dict):
                ress = ResultCollection({x: ress})
            if xhemi:
                ress_hemi = self.load_model_test(x, xhemi=True, **test_args)
                if not isinstance(ress_hemi, dict):
                    ress_hemi = ResultCollection({x: ress_hemi})
        if ress.dependent_type is DependentType.UNIVARIATE:
            return trf_report.uv_result(ress, ress_hemi, heading, caption)
        elif data.source is True:
            return trf_report.source_results(ress, ress_hemi, heading, brain_view, axw, surf, cortex, sig, vmax, cmap, alpha, view, caption)
        elif data.sensor is True:
            return trf_report.sensor_results(ress, heading, axw, vmax, cmap, caption)
        else:
            raise NotImplementedError(f'data={data.string!r}')

    def show_trf_test(
            self,
            x: ModelArg,
            # report plot parameters
            xlim: Tuple[float, float] = None,
            times: Sequence[float] = None,
            brain_view: Union[str, Sequence[float]] = None,
            axw: float = None,
            surf: str = 'inflated',
            cortex: Any = ((1.00,) * 3, (.4,) * 3),
            heading: str = None,
            vmax: float = None,
            cmap: str = None,
            labels: Dict[str, str] = None,
            rasterize: bool = None,
            # TRF model parameters
            tstart: float = 0,
            tstop: float = 0.5,
            basis: float = 0.050,
            error: str = 'l1',
            partitions: int = None,
            samplingrate: int = None,
            mask: str = None,
            delta: float = 0.005,
            mindelta: float = None,
            filter_x: FilterXArg = False,
            selective_stopping: int = 0,
            cv: bool = True,
            data: DataArg = DATA_DEFAULT,
            terms: Union[str, Sequence[str]] = None,
            permutations: int = 1,
            make: bool = False,
            scale: str = None,
            smooth: float = None,
            smooth_time: float = None,
            pmin: PMinArg = 'tfce',
            samples: int = 10000,
            test: Union[str, bool] = True,
            **state,
    ) -> fmtxt.Section:
        """Show mass-univariate test of TRFs"""
        ress = self.load_trf_test(x, tstart, tstop, basis, error, partitions, samplingrate, mask, delta, mindelta, filter_x, selective_stopping, cv, data, None, terms, permutations, make, scale, smooth, smooth_time, pmin, samples, test, **state)
        return trf_report.source_trfs(ress, heading, brain_view, axw, surf, cortex, vmax, xlim, times, cmap, labels, rasterize)

    def show_contamination(self, threshold=2e-12, separate=False, absolute=False, samplingrate=None, asds=False, **state):
        """Table of data exceeding threshold in epochs

        Determine the amount of time during which the absolute value from at
        least one sensor exceeds ``threshold``.

        Parameters
        ----------
        threshold : scalar
            Absolute threshold.
        separate : bool
            Include separate statistics for each epoch (default False).
        absolute : bool
            List absolute number of samples exceeding threshold (default is
            the percentage).
        samplingrate : int
            Samplingrate in Hz for the analysis (default is specified in epoch
            definition).
        asds : bool
            Return results as :class:`Dataset` instead of a table.
        ...
            State parameters.

        Returns
        -------
        table : fmtxt.Table | Dataset
            Result, either as table for display (default) or, with
            ``asds=True``, as  :class:`Dataset`.
        """
        if absolute:
            def agg(x):
                return x.sum()
        else:
            def agg(x):
                return int(round(x.mean() * 100.))

        lines = []
        for subject in self.iter(**state):
            ds = self.load_epochs(samplingrate=samplingrate)
            meg_abs = ds['meg'].extrema('sensor').abs()
            line = [subject, agg(meg_abs > threshold)]
            if separate:
                line.extend(agg(epoch > threshold) for epoch in meg_abs)
            lines.append(line)
        headings = ['Subject', 'Total']
        if separate:
            n_entries = max(map(len, lines))
            n_epochs = n_entries - 2
            headings.extend('Ep %i' % i for i in range(n_epochs))
            for line in lines:
                for _ in range(n_entries - len(line)):
                    line.append(np.nan)
        else:
            n_epochs = 0
        if asds:
            return Dataset.from_caselist(headings, lines)

        table = fmtxt.Table('lr' + 'r' * n_epochs)
        table.cells(*headings)
        table.midrule()
        for line in lines:
            table.cells(*line)
        return table

    def show_cached_trfs(
            self,
            model: str = None,
            term: str = None,
            raw_names: bool = False,
            keys: Sequence[str] = ('analysis', 'epoch', 'time_window', 'samplingrate', 'model', 'mask'),
            mask: str = None,
            rm: bool = False,
            mtime: Literal['min', 'max'] = None,
            subject: str = None,
            group: str = None,
            return_paths: bool = False,
            **state,
    ):
        """List cached TRFs and how much space they take

        Parameters
        ----------
        model
            String to fnmatch the model.
        term
            Include all models with this term (can be fnmatch pattern).
        raw_names
            Show model names as they are used in paths, instead of descriptive names.
        keys
            Keys which to use to group TRFs in the table.
        mask
            Only show TRFs matching this mask. Empty string (``''``) to match
            TRFs without mask.
        rm
            After listing TRFs, prompt to delete them (nothing will be deleted
            before user confirmation).
        mtime
            Show the earliest or latest file modification time for each model.
        subject
            Show files for a single subjects.
        group
            Show files for a group.
        return_paths
            Return the paths of the relevant files instead of a descriptive table.
        ...
            Additional constraints (default is to use wildcard ``*`` for all).

        See Also
        --------
        .show_models

        Notes
        -----
        To delete TRFs corresponding to a specific model, use, for example::

            e.rm('trf-file', True, test_options='* model111 *')

        Note that to show model names as they occur in paths, use ``raw_names=True``.
        Some fields are embedded, e.g. ``raw`` in ``analysis``, so to delete files with ``raw='1-8'``, use::

            e.rm('trf-file', True, test_options='* model *', analysis='1-8 *')

        """
        if group:
            if subject:
                raise ValueError(f"{subject=}, {group=}: can only specify one at a time")
            subjects = self._groups[group]
        elif subject:
            subjects = (subject,)
        else:
            subjects = ()

        if not raw_names:
            describer = ModelDescriber(self._structured_models)
        else:
            describer = None
        if term:
            pattern = re.compile(fnmatch.translate(term))
            term_models = [name for name, model in self._named_models.items() if any(pattern.match(term.string) for term in model.terms)]
            if not term_models:
                raise ValueError(f"{term=} does not match any models")
        else:
            term_models = None

        ns = defaultdict(lambda: 0)
        sizes = defaultdict(lambda: 0.)  # in bytes
        mtimes = defaultdict(list)
        paths = []
        for path in self.glob('trf-file', True, **state):
            properties = self._parse_trf_path(path)
            if subjects and properties['subject'] not in subjects:
                continue
            model_key = properties['model']
            if term_models and properties['model'] not in term_models:
                continue
            if describer:
                properties['model'] = describer.describe(self._named_models.get(properties['model'], properties['model']))
            if model:
                match = fnmatch.fnmatch(model_key, model)
                if not match and model_key in self._named_models:
                    model_obj = self._named_models[model_key]
                    match = fnmatch.fnmatch(model_obj.name, model)
                if not match and properties['model'] != model_key:
                    match = fnmatch.fnmatch(properties['model'], model)
                if not match:
                    continue
            if mask and not fnmatch.fnmatch(properties.get('mask', ''), mask):
                continue
            key = tuple([properties.get(k, '') for k in keys])
            ns[key] += 1
            sizes[key] += os.stat(path).st_size
            if mtime:
                mtimes[key].append(os.path.getmtime(path))
            paths.append(path)
        if return_paths:
            return paths
        elif not paths:
            print("No cached TRFs found")
            return
        sorted_keys = sorted(ns)
        t = fmtxt.Table('l' * len(keys) + 'rr' + 'r'*bool(mtime))
        t.cells(*keys, 'n', 'size (MB)')
        if mtime:
            t.cell(f'{mtime} mtime')
        t.midrule()
        for key in sorted_keys:
            t.cells(*key)
            t.cell(ns[key])
            size_mb = round(sizes[key] / 1e6, 1)
            t.cell(size_mb)
            if mtime:
                func = {'min': min, 'max': max}[mtime]
                time_obj = datetime.datetime.fromtimestamp(func(mtimes[key]))
                t.cell(time_obj.strftime('%Y-%m-%d %H:%M:%S'))
        if not rm:
            return t
        # prompt to delete files
        print(t)
        command = ask(f"Delete {len(paths)} TRFs?", {'yes': 'delete files', 'no': "don't delete files (default)"}, allow_empty=True)
        if command != 'yes':
            return
        for path in paths:
            os.remove(path)

    def show_model_code(self, model: ModelArg) -> str:
        self._x_desc(self._coerce_model(model))

    def show_model_terms(self, x: str, cv: bool = True) -> fmtxt.Table:
        """Table showing terms in a model or comparison

        Parameters
        ----------
        x
            Model or comparison for which to show terms.
        cv
            Whether to use cross-validation for interpreting comparisons.
        """
        obj = self._coerce_model_or_comparison(x, cv)
        return obj.term_table()

    def show_models(
            self,
            term: str = None,
            stim: bool = True,
            rand: bool = True,
            model: str = None,
            sort: bool = False,
            files: bool = False,
            abbreviate: bool = True,
    ):
        """List models that contain a term that matches ``term``

        Parameters
        ----------
        term
            Fnmatch pattern for a terms.
        stim
            Also include terms with a stimulus prefix.
        rand
            Also show models that contain ``term`` randomized.
        model
            Pattern to display only certain models.
        sort
            Sort terms (default False).
        files
            List the number of files associated with the model.
        abbreviate
            Abbreviate models with names from :attr:`.models`.

        See Also
        --------
        .remove_model
        .show_cached_trfs

        Notes
        -----
        Initial column contains ``*`` for models explicitly defined in
        :attr:`.models`.
        """
        if term is None:
            pattern = None
        else:
            pattern = fnmatch.translate(term)
            if stim:
                pattern = r'(\w+\~)?' + pattern
            if rand:
                pattern += r'(\$.*)?'
            pattern = re.compile(pattern)
        model_pattern = model

        # Find possible model abbreviations
        if abbreviate:
            describer = ModelDescriber(self._structured_models)
        else:
            describer = None

        columns = 'lll'
        if files:
            columns += 'rr'
        t = fmtxt.Table(columns)
        t.cells('.', 'Name', 'Terms')
        if files:
            t.cells('trfs', 'tests')
        t.midrule()
        for name, model in self._named_models.items():
            if pattern is not None:
                if not any(pattern.match(t.string) for t in model.terms):
                    continue
            if model_pattern is not None:
                if not fnmatch.fnmatch(name, model_pattern):
                    continue
            t.cell('*' if name in self.models else '')
            t.cell(name)

            if abbreviate:
                t.cell(describer.describe(model))
            elif sort:
                t.cell(model.sorted_key)
            else:
                t.cell(model.name)

            if files:
                n = len(list(self._find_model_files(name, trfs=True)))
                t.cell(n or '')
                n = len(list(self._find_model_files(name, tests=True)))
                t.cell(n or '')
        return t

    def show_predictors(self, pattern: str = None):
        """List predictors that have been added to the :class:`TRFExperiment`

        Parameters
        ----------
        pattern
            :mod:`fnmatch` pattern for showing a subset of predictors.

        See Also
        --------
        :attr:`.predictors` : adding predictors to the :class:`TRFExperiment`
        """
        for prefix_ in sorted(self.predictors):
            predictor = self.predictors[prefix_]
            # Non-file-based
            if not isinstance(predictor, FilePredictorBase):
                if not pattern or fnmatch.fnmatch(prefix_, pattern):
                    print(f"{prefix_}: {type(predictor).__name__}")
                continue
            # File-based
            predictor_dir = Path(self.get('predictor-dir'))
            files_1 = predictor_dir.glob(f'*~{prefix_}.pickle')
            files_2 = predictor_dir.glob(f'*~{prefix_}-*.pickle')
            sub_predictors = set()
            for filename in chain(files_1, files_2):
                components = self._parse_predictor_path(filename)
                sub_predictors.add(components['predictor'])
            if pattern:
                sub_predictors = fnmatch.filter(sub_predictors, pattern)
                if not sub_predictors:
                    continue
            print(f"{prefix_}: {type(predictor).__name__}")
            for sub_predictor in sorted(sub_predictors):
                print(f"  {sub_predictor}")
