# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Derivatives for two-stage tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import load, save
from ..._data_obj import Dataset, combine
from ..._io.pickle import update_subjects_dir
from ..data import DataSpec
from ..derivative_cache import Dependency, Derivative, Request, UncachedDerivative
from ..pathing import MRI_SDIR
from ..source import ROIData, roi_data_from_dataset
from .config import ResolvedTestNDSpec, Test, TwoStageTest
from .nodes import RESULT_OPTION_DEFAULTS, RESULT_SOURCE_GROUP_KEY_FIELDS, ROITestResult, ResultOutputDerivative


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


def _validate_two_stage_options(
        name: str,
        test_obj: Test,
        data: DataSpec,
) -> None:
    """Requirements shared by every node in the two-stage chain."""
    if not isinstance(test_obj, TwoStageTest):
        raise RuntimeError(f"{name!r} requires a TwoStageTest")
    if not data.source:
        raise NotImplementedError(f"Two-stage test with data={data.string!r}")


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
    key_options = {
        **RESULT_OPTION_DEFAULTS,
    }

    def __init__(
            self,
            tests: dict[str, Test],
            epochs: dict[str, Any],
    ):
        self.tests = tests
        self.epochs = epochs

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # Match TwoStageLevel1Derivative
        data = ctx.options['data']
        fields = ['subject', 'session', 'acquisition', 'epoch', 'epoch_rejection', 'raw']
        if data.source:
            fields += ['equalize_evoked_count', 'inv', 'cov', 'src', 'parc', 'mrisubject', 'adjacency']
            if not data.aggregate:
                fields += ['common_brain']
        return tuple(fields)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'model': self.tests[ctx.options['test']].model,
            'epoch': self.epochs[ctx.state['epoch']],
        }

    def validate_options(self, ctx: Request) -> None:
        data = ctx.options['data']
        _validate_two_stage_options(self.name, self.tests[ctx.options['test']], data)
        if data.aggregate and (smooth := ctx.options['smooth']):
            raise TypeError(f"{smooth=} for ROI two-stage tests")

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        if data.source and not data.aggregate:
            if test_obj.model:
                dependency = Dependency(
                    'evoked-stc',
                    label='data',
                    options=ctx.options_for('evoked-stc', 'baseline', 'src_baseline', 'samplingrate', model=test_obj.model, morph=True),
                )
            else:
                dependency = Dependency(
                    'epochs-stc',
                    label='data',
                    options=ctx.options_for('epochs-stc', 'baseline', 'src_baseline', 'samplingrate', morph=True),
                )
        else:
            if test_obj.model:
                dependency = Dependency(
                    'evoked-stc',
                    label='data',
                    options=ctx.options_for('evoked-stc', 'baseline', 'src_baseline', 'samplingrate', model=test_obj.model),
                )
            else:
                dependency = Dependency(
                    'epochs-stc',
                    label='data',
                    options=ctx.options_for('epochs-stc', 'baseline', 'src_baseline', 'samplingrate'),
                )
        if not test_obj._test_vars:
            return dependency,
        # The shell the stage-1 predictors are resolved against, describing the same
        # cases as the data: the evoked event shell with a model, the events themselves
        # for single-trial data. ``reference`` matches EvokedStcDerivative.fixed_state.
        if test_obj.model:
            options = ctx.options_for('evoked', 'samplingrate', 'decim', model=test_obj.model)
            events = Dependency('evoked', label='events', view='shell', state={'reference': ''}, options=options)
        else:
            options = ctx.options_for('epoch-events', 'samplingrate', 'decim')
            events = Dependency('epoch-events', label='events', options=options)
        return dependency, events

    def dependency_fingerprint_override(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> dict[str, Any] | None:
        """Depend on the events that the stage-1 model reads, not the definitions behind them (see :meth:`Test._resolve_vars`)"""
        if dep.label != 'events':
            return None
        # Stage 1 fits one subject at a time, so there are no groups to resolve against
        return self.tests[ctx.options['test']]._resolve_vars(ctx.load(dep.label))

    def build(self, ctx: Request) -> Dataset | ROIData:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]

        ds = ctx.load('data')
        test_obj._resolve_vars(ds)
        if data.source and not data.aggregate:
            if ctx.options['smooth']:
                y = data.response_key(ds)
                ds[y] = ds[y].smooth('source', ctx.options['smooth'], 'gaussian')
        return ds


class TwoStageLevel1Derivative(Derivative[Any]):
    """Cached first-stage LM fit for one subject."""
    name = 'two-stage-level-1'
    cache_suffix = '.pickle'
    key_options = {
        **RESULT_OPTION_DEFAULTS,
    }

    def __init__(self, tests: dict[str, Test]):
        self.tests = tests

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # ``data`` is ``None`` until resolved, in which case the source superset is used (the artifact is never built for an unresolved request)
        data = ctx.options['data']
        fields = ['subject', 'session', 'acquisition', 'epoch', 'raw', 'epoch_rejection']
        if data is None or data.source:
            fields += ['equalize_evoked_count', 'cov', 'inv', 'src', 'mri', 'mrisubject', 'parc', 'common_brain', 'adjacency']
        return tuple(fields)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        # two-stage-data records the values the stage-1 model reads
        return {'test': self.tests[ctx.options['test']]._as_dict_without_vars()}

    def validate_options(self, ctx: Request) -> None:
        _validate_two_stage_options(self.name, self.tests[ctx.options['test']], ctx.options['data'])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('two-stage-data', options=ctx.options_for('two-stage-data', *RESULT_OPTION_DEFAULTS)),)

    def build(self, ctx: Request):
        test_obj = self.tests[ctx.options['test']]
        data = ctx.options['data']
        subject = ctx.state['subject']
        ds = ctx.load('two-stage-data')
        if not data.aggregate:
            return test_obj.make_stage_1(data.response_key(ds), ds, subject)
        roi_data = roi_data_from_dataset(ds, data.aggregate)
        return SubjectROILMResult(
            {label: test_obj.make_stage_1('label_tc', label_ds, subject) for label, label_ds in roi_data.label_data.items()},
            roi_data.n_trials_ds,
        )

    def load(self, ctx: Request, path: Path):
        value = load.unpickle(path)
        if ctx.options['data'].source:
            update_subjects_dir(value, ctx.root / MRI_SDIR, 2)
        return value

    def save(self, ctx: Request, path: Path, value) -> None:
        save.pickle(value, path)


class TwoStageLevel2Derivative(ResultOutputDerivative):
    """Cached second-stage group result for two-stage tests."""
    name = 'two-stage-level-2'
    key_fields = RESULT_SOURCE_GROUP_KEY_FIELDS
    cache_suffix = '.pickle'
    path = Derivative.path
    key_options = {**RESULT_OPTION_DEFAULTS, 'disconnect_labels': False}
    view_options = {}

    def cache_label(self, ctx: Request) -> str:
        return self._path_stem(ctx) if ctx.options['samples'] is None else f"{self._path_stem(ctx)}_samples-{ctx.options['samples']}"

    def validate_options(self, ctx: Request) -> None:
        _validate_two_stage_options(self.name, self.tests[ctx.options['test']], ctx.options['data'])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        subjects = self.groups[ctx.state['group']]
        return tuple(
            Dependency('two-stage-level-1', label=subject, state={'subject': subject}, options=ctx.options_for('two-stage-level-1', *RESULT_OPTION_DEFAULTS))
            for subject in subjects
        )

    def build(self, ctx: Request):
        test_obj = self.tests[ctx.options['test']]
        data = ctx.options['data']
        test_spec = ResolvedTestNDSpec.from_request(ctx)
        subjects = self.groups[ctx.state['group']]
        subject_results = [ctx.load(subject) for subject in subjects]
        if not data.aggregate:
            return test_obj.make_stage_2(subject_results, test_spec.kwargs)

        label_lms = {}
        for subject_result in subject_results:
            for label, lm in subject_result.lms.items():
                label_lms.setdefault(label, []).append(lm)
        dropped = sorted(label for label, lms in label_lms.items() if len(lms) <= 2)
        if dropped:
            ctx.registry.log.warning("Two-stage ROI test: dropping label(s) %s with data from 2 or fewer subjects (a group test needs more than 2)", ', '.join(dropped))
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
            update_subjects_dir(res, ctx.root / MRI_SDIR, 2)
        return res

    def save(self, ctx: Request, path: Path, value) -> None:
        save.pickle(value, path)
