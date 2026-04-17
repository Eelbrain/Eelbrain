# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Two-stage test definitions and derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import load, save, testnd
from .._data_obj import Dataset, combine
from .._io.pickle import update_subjects_dir
from .._utils.parse import find_variables
from .derivative_cache import Dependency, Derivative, Request, UncachedDerivative
from .pathing import MRI_SDIR
from .results import RESULT_OPTION_DEFAULTS, ResultOutputDerivative, _epochs_stc_options, _evoked_stc_options
from .source import ROIData, roi_data_from_subject_datasets
from .test_def import ROITestResult, ResolvedTestNDSpec, Test, guess_y
from .variable_def import apply_vardef


class TwoStageTest(Test):
    """Two-stage test: T-test of regression coefficients

    Stage 1: fit a regression model to the data for each subject.
    Stage 2: test coefficients from stage 1 against 0 across subjects.

    Parameters
    ----------
    stage_1 : str
        Stage 1 model specification. Coding for categorial predictors uses 0/1 dummy
        coding.
    vars : dict
        Add new variables for the stage 1 model. This is useful for specifying
        coding schemes based on categorial variables.
        Each entry specifies a variable with the following schema:
        ``{name: definition}``. ``definition`` can be either a string that is
        evaluated in the events-:class:`Dataset`, or a
        ``(source_name, {value: code})``-tuple (see example below).
        ``source_name`` can also be an interaction, in which case cells are joined
        with spaces (``"f1_cell f2_cell"``).
    model : str
        This parameter can be supplied to perform stage 1 tests on condition
        averages. If ``model`` is not specified, the stage1 model is fit on single
        trial data.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    The first example assumes 2 categorical variables present in events,
    'a' with values 'a1' and 'a2', and 'b' with values 'b1' and 'b2'. These are
    recoded into 0/1 codes::

        TwoStageTest(
            "a_num + b_num + a_num * b_num + index + a_num * index",
            vars={
                'a_num': ('a', {'a1': 0, 'a2': 1}),
                'b_num': ('b', {'b1': 0, 'b2': 1}),
            }),

    The second test definition uses the "index" variable which is always present
    and specifies the chronological index of the events as an integer count.
    This variable can thus be used to test for a linear change over time. Due
    to the numeric nature of these variables interactions can be computed by
    multiplication::

        TwoStageTest("a_num + index + a_num * index",
                     vars={'a_num': ('a', {'a1': 0, 'a2': 1})

    Numerical variables can also defined using data-object methods (e.g.
    :meth:`Factor.label_length`) or from interactions::

        TwoStageTest('wordlength', vars={'wordlength': 'word.label_length()'})
        TwoStageTest("ab", vars={'ab': ('a%b', {'a1 b1': 0, 'a1 b2': 1, 'a2 b1': 1, 'a2 b2': 2})})
    """
    kind = 'two-stage'
    DICT_ATTRS = Test.DICT_ATTRS + ('stage_1',)

    def __init__(self, stage_1: str, vars: dict = None, model: str = None):
        Test.__init__(self, stage_1, model, vars=vars, depend_on=find_variables(stage_1))
        self.stage_1 = stage_1

    def make_stage_1(self, y, data, subject, sub=None):
        """Assumes that model has already been applied"""
        return testnd.LM(y, self.stage_1, sub=sub, data=data, samples=0, subject=subject)

    @staticmethod
    def make_stage_2(lms, kwargs):
        lm = testnd.LMGroup(lms)
        lm.compute_column_ttests(**kwargs)
        return lm

    def make(self, y, ds, force_permutation, kwargs):
        lms = [self.make_stage_1(y, ds, subject, f"subject=={subject!r}") for subject in ds['subject'].cells]
        return self.make_stage_2(lms, kwargs)


class ROI2StageResult(ROITestResult):
    """Test results for 2-stage tests in one or more ROIs

    Attributes
    ----------
    subjects : tuple of str
        Subjects included in the test.
    samples : int
        ``samples`` parameter used for permutation tests.
    res : {str: LMGroup} dict
        Test result for each ROI.
    n_trials_ds : Dataset
        Dataset describing how many trials were used in each condition per
        subject.
    """


@dataclass
class SubjectROILMResult:
    lms: dict[str, Any]
    n_trials_ds: Dataset


class TwoStageDataDerivative(UncachedDerivative[Dataset | ROIData]):
    """Prepared source-space data for two-stage level-1 fits.

    Options
    -------
    data
        Analysis data family. Sensor data is not supported.
    baseline
        Sensor-space baseline correction for upstream source estimates.
    src_baseline
        Source-space baseline correction.
    samplingrate
        Sampling rate override for upstream cached data.
    smooth
        Optional source-space smoothing.
    """
    name = 'two-stage-data'
    OPTION_DEFAULTS = {
        **RESULT_OPTION_DEFAULTS,
    }

    def __init__(self, tests: dict[str, Test], epochs: dict[str, Any], groups: dict[str, Any]):
        self.tests = tests
        self.epochs = epochs
        self.groups = groups

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(
            ctx,
            state_fields=('subject', 'epoch', 'raw', 'rej', 'model', 'equalize_evoked_count', 'test', 'cov', 'inv', 'src', 'mri', 'parc'),
            definitions={
                'test': self.tests[ctx.options['test']]._as_dict(),
                'epoch': self.epochs[ctx.state['epoch']]._as_dict(),
            },
        )

    def _subject_dependency(self, ctx: Request, subject: str) -> Dependency:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        samplingrate = ctx.options['samplingrate']
        if data.source is True:
            if test_obj.model is None or test_obj.vars:
                return Dependency(
                    'epochs-stc',
                    label=subject,
                    state={'subject': subject},
                    options=_epochs_stc_options(ctx, morph=data.morph, samplingrate=samplingrate),
                )
            return Dependency(
                'evoked-stc',
                label=subject,
                state={'subject': subject, 'model': test_obj.model},
                options=_evoked_stc_options(ctx, morph=data.morph, cat=None, samplingrate=samplingrate),
            )
        if test_obj.model is None or test_obj.vars:
            return Dependency(
                'epochs-stc',
                label=subject,
                state={'subject': subject},
                options=_epochs_stc_options(ctx, morph=None, samplingrate=samplingrate),
            )
        return Dependency(
            'evoked-stc',
            label=subject,
            state={'subject': subject, 'model': test_obj.model},
            options=_evoked_stc_options(ctx, morph=False, cat=None, samplingrate=samplingrate),
        )

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        data = ctx.options['data']
        if data.sensor:
            return ()
        return (self._subject_dependency(ctx, ctx.state['subject']),)

    def _load_subject_data(self, ctx: Request, subject: str) -> Dataset:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        samplingrate = ctx.options['samplingrate']

        if data.source is True:
            if test_obj.model is None or test_obj.vars:
                ds = ctx.load('epochs-stc', state={'subject': subject}, options=_epochs_stc_options(ctx, morph=data.morph, samplingrate=samplingrate))
                if test_obj.vars:
                    apply_vardef(ds, test_obj.vars, self.tests, self.groups)
                if test_obj.model is not None:
                    ds = ds.aggregate(
                        test_obj.model,
                        never_drop=(guess_y(ds),),
                        drop_bad=True,
                        equal_count=ctx.state['equalize_evoked_count'] == 'eq',
                        drop=('i_start', 't_edf', 'time', 'index', 'trigger'),
                    )
            else:
                ds = ctx.load('evoked-stc', state={'subject': subject, 'model': test_obj.model}, options=_evoked_stc_options(ctx, morph=data.morph, cat=None, samplingrate=samplingrate))
            if ctx.options['smooth']:
                ds[data.y_name] = ds[data.y_name].smooth('source', ctx.options['smooth'], 'gaussian')
            return ds

        if ctx.options['smooth']:
            raise TypeError(f"smooth={ctx.options['smooth']!r} for ROI two-stage tests")
        if test_obj.model is None:
            ds = ctx.load('epochs-stc', state={'subject': subject}, options=_epochs_stc_options(ctx, morph=None, samplingrate=samplingrate))
            if test_obj.vars:
                apply_vardef(ds, test_obj.vars, self.tests, self.groups)
            return ds
        if not test_obj.vars:
            return ctx.load('evoked-stc', state={'subject': subject, 'model': test_obj.model}, options=_evoked_stc_options(ctx, morph=False, cat=None, samplingrate=samplingrate))
        ds = ctx.load('epochs-stc', state={'subject': subject}, options=_epochs_stc_options(ctx, morph=None, samplingrate=samplingrate))
        apply_vardef(ds, test_obj.vars, self.tests, self.groups)
        return ds.aggregate(
            test_obj.model,
            never_drop=(guess_y(ds),),
            drop_bad=True,
            equal_count=ctx.state['equalize_evoked_count'] == 'eq',
            drop=('i_start', 't_edf', 'time', 'index', 'trigger'),
        )

    def build(self, ctx: Request) -> Dataset | ROIData:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        if not isinstance(test_obj, TwoStageTest):
            raise RuntimeError(f"{self.name!r} requires a TwoStageTest")
        if data.sensor:
            raise NotImplementedError(f"Two-stage test with data={data.string!r}")
        return self._load_subject_data(ctx, ctx.state['subject'])


class TwoStageLevel1Derivative(Derivative[Any]):
    """Cached first-stage LM fit for one subject."""
    name = 'two-stage-level-1'
    key_fields = (
        'subject', 'epoch', 'raw', 'rej', 'model', 'equalize_evoked_count',
        'test', 'cov', 'inv', 'src', 'mri', 'parc',
    )
    cache_suffix = '.pickle'
    OPTION_DEFAULTS = {
        **RESULT_OPTION_DEFAULTS,
    }

    def __init__(self, tests: dict[str, Test]):
        self.tests = tests

    def key(self, ctx: Request) -> dict[str, Any]:
        subject = ctx.state['subject']
        if subject in (None, '', '*'):
            raise RuntimeError(f"{self.name!r} requires an explicit subject")
        return super().key(ctx)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(
            ctx,
            state_fields=self.key_fields,
            definitions={'test': self.tests[ctx.options['test']]._as_dict()},
        )

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('two-stage-data', options=ctx.options_for('two-stage-data', *RESULT_OPTION_DEFAULTS)),)

    def build(self, ctx: Request):
        test_obj = self.tests[ctx.options['test']]
        if not isinstance(test_obj, TwoStageTest):
            raise RuntimeError(f"{self.name!r} requires a TwoStageTest")
        data = ctx.options['data']
        subject = ctx.state['subject']
        if data.source is True:
            ds = ctx.load('two-stage-data', options=ctx.options_for('two-stage-data', *RESULT_OPTION_DEFAULTS))
            return test_obj.make_stage_1(data.y_name, ds, subject)
        if data.sensor:
            raise NotImplementedError(f"Two-stage test with data={data.string!r}")
        ds = ctx.load('two-stage-data', options=ctx.options_for('two-stage-data', *RESULT_OPTION_DEFAULTS))
        roi_data = roi_data_from_subject_datasets([ds], data.source)
        return SubjectROILMResult(
            {label: test_obj.make_stage_1('label_tc', label_ds, subject) for label, label_ds in roi_data.label_data.items()},
            roi_data.n_trials_ds,
        )

    def load(self, ctx: Request, path: Path):
        value = load.unpickle(path)
        if ctx.options['data'].source:
            update_subjects_dir(value, ctx.registry.root / MRI_SDIR, 2)
        return value

    def save(self, ctx: Request, path: Path, value) -> None:
        save.pickle(value, path)


class TwoStageLevel2Derivative(ResultOutputDerivative):
    """Cached second-stage group result for two-stage tests."""
    name = 'two-stage-level-2'
    sampled_path = True
    cache_suffix = '.pickle'
    path = Derivative.path
    OPTION_DEFAULTS = {**RESULT_OPTION_DEFAULTS, 'disconnect_labels': False}
    VIEW_OPTION_DEFAULTS = {}

    def cache_label(self, ctx: Request) -> str:
        return self._path_stem(ctx) if ctx.options['samples'] is None else f"{self._path_stem(ctx)}_samples-{ctx.options['samples']}"

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        subjects = self.groups[ctx.state['group']]
        return tuple(
            Dependency('two-stage-level-1', label=subject, state={'subject': subject}, options=ctx.options_for('two-stage-level-1', *RESULT_OPTION_DEFAULTS))
            for subject in subjects
        )

    def build(self, ctx: Request):
        test_obj = self.tests[ctx.options['test']]
        if not isinstance(test_obj, TwoStageTest):
            raise RuntimeError(f"{self.name!r} requires a TwoStageTest")
        data = ctx.options['data']
        test_spec = ResolvedTestNDSpec.from_request(ctx, data)
        subjects = self.groups[ctx.state['group']]
        if data.source is not True and not isinstance(data.source, str):
            raise NotImplementedError(f"Two-stage test with data={data.string!r}")
        subject_results = [ctx.load('two-stage-level-1', state={'subject': subject}, options=ctx.options_for('two-stage-level-1', *RESULT_OPTION_DEFAULTS)) for subject in subjects]
        if data.source is True:
            return test_obj.make_stage_2(subject_results, test_spec.kwargs)

        label_lms = {}
        for subject_result in subject_results:
            for label, lm in subject_result.lms.items():
                label_lms.setdefault(label, []).append(lm)
        results = {
            label: test_obj.make_stage_2(lms, test_spec.kwargs)
            for label, lms in label_lms.items()
            if len(lms) > 2
        }
        n_trials_ds = combine([subject_result.n_trials_ds for subject_result in subject_results], incomplete='drop')
        return ROI2StageResult(subjects, ctx.options['samples'], n_trials_ds, None, results)

    def load(self, ctx: Request, path: Path):
        res = load.unpickle(path)
        if ctx.options['data'].source:
            update_subjects_dir(res, ctx.registry.root / MRI_SDIR, 2)
        return res

    def save(self, ctx: Request, path: Path, value) -> None:
        save.pickle(value, path)
