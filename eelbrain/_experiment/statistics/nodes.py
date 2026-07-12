# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Result derivatives for statistical tests.

These derivatives orchestrate through other graph nodes, especially the
dataset-producing derivatives that correspond to the public ``Pipeline.load_x``
methods. Cache paths are internal derivative-owned artifacts; only end-product
exports own default public paths. They must not depend on injected facade
loaders or Pipeline-managed naming state, and naming-only path labels must not
be part of canonical cache identity.
"""

from __future__ import annotations

from inspect import getfullargspec
import logging
from pathlib import Path
from typing import Any, TypeVar

from ... import load
from ... import save
from ..._data_obj import Dataset
from ..._exceptions import ConfigurationError
from ..._io.pickle import update_subjects_dir
from ..._text import enumeration
from ..._stats.testnd import _MergedTemporalClusterDist
from ..data import DataSpec
from ..derivative_cache import Dependency, Derivative, OptionSpec, Request, UncachedDerivative
from ..pathing import (
    MRI_SDIR,
    join_stem_parts,
    report_export_path,
    test_basename,
    time_window_str,
)
from ..source import ROIData
from ..variable_def import apply_vardef
from .config import ResolvedTestNDSpec, Test

T = TypeVar('T')
USE_CTX = object()
RESULT_OPTION_DEFAULTS = {
    'samples': None,
    # normalize so that a request reconstructed from a manifest (offline
    # revalidation) re-parses the canonical dict form into a DataSpec
    'data': OptionSpec(None, DataSpec),
    'test': None,
    'tstart': None,
    'tstop': None,
    'pmin': None,
    'baseline': None,
    'src_baseline': None,
    'smooth': None,
    'samplingrate': None,
    'decim': None,
}

TEST_DATA_OPTION_NAMES = (
    'data',
    'test',
    'baseline',
    'src_baseline',
    'samplingrate',
    'smooth',
)

_RESULT_COMMON_KEY_FIELDS = (
    'session', 'acquisition', 'run', 'epoch', 'raw', 'epoch_rejection',
    'equalize_evoked_count',
)
RESULT_SENSOR_GROUP_KEY_FIELDS = (
    'group', *_RESULT_COMMON_KEY_FIELDS, 'reference',
)
RESULT_SOURCE_GROUP_KEY_FIELDS = (
    'group', *_RESULT_COMMON_KEY_FIELDS, 'cov', 'inv', 'src', 'mri',
    'mrisubject', 'parc', 'common_brain', 'adjacency',
)
RESULT_SOURCE_SUBJECT_KEY_FIELDS = (
    'subject', *_RESULT_COMMON_KEY_FIELDS, 'cov', 'inv', 'src', 'mri',
    'mrisubject', 'parc', 'common_brain', 'adjacency',
)


def _test_result_options(
        ctx: Request,
        *,
        data: DataSpec | object = USE_CTX,
) -> dict[str, Any]:
    if data is USE_CTX:
        data = ctx.options['data']
    out = {
        'data': data,
        'samples': ctx.options['samples'],
        'test': ctx.options['test'],
        'tstart': ctx.options['tstart'],
        'tstop': ctx.options['tstop'],
        'pmin': ctx.options['pmin'],
        'baseline': ctx.options['baseline'],
        'src_baseline': ctx.options['src_baseline'],
        'smooth': ctx.options['smooth'],
        'samplingrate': ctx.options['samplingrate'],
    }
    if 'disconnect_labels' in ctx.options:
        out['disconnect_labels'] = ctx.options['disconnect_labels']
    return out


def _validate_post_aggregation_test_vars(test_obj: Test):
    """Make sure user is not trying to base aggregation on test-specific vars"""
    model_vars = set(filter(None, (test_obj.model or '').split('%')))
    missing_model_vars = model_vars.intersection(test_obj.vars.vars)
    if missing_model_vars:
        vars_desc = enumeration(sorted(missing_model_vars))
        raise ConfigurationError(f"For evoked tests, Test.vars are computed after averaging. Model variable {vars_desc} can not be provided through Test.vars. Use TwoStageTest or Pipeline.variables instead.")


def sampled_artifact_path(path: str | Path, samples: int | None) -> Path:
    path = Path(path)
    if samples is None:
        return path
    return path.with_name(f"{path.stem}_samples-{samples}{path.suffix}")


class ROITestResult:
    """Test results for temporal tests in one or more ROIs

    Attributes
    ----------
    subjects : tuple of str
        Subjects included in the test.
    samples : int
        ``samples`` parameter used for permutation tests.
    res : {str: NDTest} dict
        Test result for each ROI.
    n_trials_ds : Dataset
        Dataset describing how many trials were used in each condition per
        subject.
    """

    def __init__(self, subjects, samples, n_trials_ds, merged_dist, res):
        self.subjects = subjects
        self.samples = samples
        self.n_trials_ds = n_trials_ds
        self.merged_dist = merged_dist
        self.res = res

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in getfullargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)


class ResultOutputDerivative(Derivative[T]):
    """Shared base for cached result/report/movie outputs.

    This is a :class:`~eelbrain._experiment.derivative_cache.Derivative`
    subclass with a fixed pattern:

    - Subclasses declare their cache identity through ``key_fields`` and
      ``key_options``.
    - :meth:`fingerprint` records configured test/epoch/parc definitions.
    - :meth:`path` chooses a user-facing export path, with optional
      ``samples``-specific disambiguation.
    - :meth:`load` returns that path, and :meth:`save` is a no-op, because
      subclasses normally create the final output file directly in
      :meth:`build` rather than serializing a separate in-memory artifact.

    Subclasses usually extend this template by overriding:

    - :meth:`dependencies` and :meth:`build` as ordinary derivative hooks
    - :meth:`_path_stem` or :meth:`_default_output_path` to customize export
      naming

    Options
    -------
    dst
        Optional explicit output path.
    samples
        Permutation/sample count stored in the cache identity.
    data
        Analysis data family to use (sensor, source, ROI, ...).
    test
        Test definition to run.
    tstart, tstop
        Optional time window for the analysis.
    pmin
        Cluster-forming threshold or ``'tfce'``.
    baseline
        Sensor-space baseline correction.
    src_baseline
        Source-space baseline correction.
    disconnect_labels
        Disconnect source-space cluster adjacency across labels from the
        current ``parc`` state.
    samplingrate
        Sampling rate override for upstream cached data.
    smooth
        Optional source-space smoothing.
    """
    cache_log_level = logging.INFO
    single_subject = False
    sampled_path = False
    key_options = RESULT_OPTION_DEFAULTS
    view_options = {'dst': None}

    def __init__(
            self,
            tests: dict[str, Test],
            epochs: dict[str, Any],
            parcs: dict[str, Any],
            groups: dict[str, tuple[str, ...] | list[str]],
    ):
        self.tests = tests
        self.epochs = epochs
        self.parcs = parcs
        self.groups = groups

    def _result_model(self, ctx: Request) -> str:
        """Model that groups trials for this result; derived from the test definition by default."""
        return self.tests[ctx.options['test']].model or ''

    def _path_context_parts(self, ctx: Request) -> list[str]:
        """Path-stem parts derived from analysis context/state."""
        data = ctx.options['data']
        parts = [f'data-{data.string}', f'raw-{ctx.state["raw"]}', f'rej-{ctx.state["epoch_rejection"]}']
        model = self._result_model(ctx)
        if model:
            parts.append(f'model-{model}')
        if ctx.state['equalize_evoked_count']:
            parts.append(f'count-{ctx.state["equalize_evoked_count"]}')
        if data.source:
            parts.extend((f'cov-{ctx.state["cov"]}', f'src-{ctx.state["src"]}', f'inv-{ctx.state["inv"]}', f'parc-{ctx.state["parc"]}'))
        return parts

    def _path_option_parts(self, ctx: Request) -> list[str]:
        """Path-stem parts derived from analysis options."""
        parts = []
        baseline = ctx.options['baseline']
        src_baseline = ctx.options['src_baseline']
        pmin = ctx.options['pmin']
        samplingrate = ctx.options['samplingrate']
        smooth = ctx.options['smooth']
        if baseline is False:
            parts.append('nobl')
        elif baseline not in (None, True):
            parts.append(f'bl-{time_window_str(baseline)}')
        if src_baseline is True:
            parts.append('srcbl')
        elif src_baseline not in (None, False):
            parts.append(f'srcbl-{time_window_str(src_baseline)}')
        if ctx.options.get('disconnect_labels', False):
            parts.append('disconnect-labels')
        if pmin == 'tfce':
            parts.append('tfce')
        elif pmin is not None:
            parts.append(f'p-{pmin}')
        if pmin is not None and ctx.options['data'].source and ctx.state['adjacency']:
            parts.append(f'adj-{ctx.state["adjacency"]}')
        if ctx.options['tstart'] is not None or ctx.options['tstop'] is not None:
            parts.append(f'tw-{time_window_str((ctx.options["tstart"], ctx.options["tstop"]))}')
        if samplingrate is not None:
            parts.append(f'sr-{samplingrate:g}Hz')
        if smooth:
            parts.append(f'sm-{int(round(smooth * 1000))}mm')
        return parts

    def _path_stem(self, ctx: Request) -> str:
        """Default export stem used by :meth:`path`."""
        return join_stem_parts(
            test_basename(ctx.state, datatype=ctx.datatype),
            f'epoch-{ctx.state["epoch"]}',
            f'test-{ctx.options["test"]}',
            self._path_context_parts(ctx),
            self._path_option_parts(ctx),
        )

    def _default_output_path(self, ctx: Request) -> Path:
        """Default user-facing export path used when ``dst`` is not set."""
        return ctx.root / report_export_path(ctx.state, self.name, self._path_stem(ctx), self.single_subject)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        out = {
            'test': self.tests[ctx.options['test']],
            'epoch': self.epochs[ctx.state['epoch']],
            'single_subject': self.single_subject,
        }
        if not self.single_subject:
            out['subjects'] = self.groups[ctx.state['group']]
        if ctx.options['data'].source and ctx.state['parc'] in self.parcs:
            out['parc'] = self.parcs[ctx.state['parc']]
        return out

    def path(
            self,
            ctx: Request,
    ) -> Path:
        dst = ctx.view_options['dst']
        path = Path(dst) if dst else self._default_output_path(ctx)
        if self.sampled_path:
            path = sampled_artifact_path(path, ctx.options['samples'])
        return path

    def load(
            self,
            ctx: Request,
            path: Path) -> T:
        return path

    def save(
            self,
            ctx: Request,
            path: Path,
            value: T,
    ) -> None:
        return


class EvokedTestDataDerivative(UncachedDerivative[Dataset | ROIData]):
    """Prepared test/report data for sensor, source, and ROI analyses.

    Options
    -------
    data
        Analysis data family to prepare.
    baseline
        Sensor-space baseline correction.
    src_baseline
        Source-space baseline correction.
    samplingrate
        Sampling rate override for upstream cached data.
    smooth
        Optional source-space smoothing.
    """
    name = 'evoked-test-data'
    key_options = {
        'data': OptionSpec(None, DataSpec),
        'test': None,
        'baseline': None,
        'src_baseline': None,
        'samplingrate': None,
        'decim': None,
        'smooth': None,
    }

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # Source-space fields identify the artifact only for source/ROI analyses
        # (see dependencies); a sensor test uses only evoked-group-dataset.
        fields = ('group', 'epoch', 'raw', 'session', 'acquisition', 'epoch_rejection', 'equalize_evoked_count')
        data = ctx.options['data']
        if data is None or data.source:
            fields += ('mri', 'cov', 'inv', 'src', 'parc', 'mrisubject', 'common_brain', 'adjacency')
        else:
            fields += ('reference',)
        return fields

    def __init__(self, tests: dict[str, Test], epochs: dict[str, Any], groups: dict[str, tuple[str, ...] | list[str]]):
        self.tests = tests
        self.epochs = epochs
        self.groups = groups

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'test': self.tests[ctx.options['test']],
            'epoch': self.epochs[ctx.state['epoch']],
            'subjects': tuple(self.groups[ctx.state['group']]),
        }

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        model = test_obj.model or ''
        if test_obj.vars:
            _validate_post_aggregation_test_vars(test_obj)

        if ctx.options['smooth']:
            if data.sensor:
                raise TypeError(f"smooth={ctx.options['smooth']!r} for sensor tests")
            if data.aggregate:
                raise TypeError(f"smooth={ctx.options['smooth']!r} for ROI tests")

        if data.sensor:
            if ctx.options['src_baseline']:
                raise TypeError(f"src_baseline={ctx.options['src_baseline']!r} for sensor tests")
            options = ctx.options_for('evoked', 'baseline', 'samplingrate', 'decim', 'data', model=model, cat=test_obj.cat, ndvar=True)
            return Dependency('evoked-group-dataset', options=options, label='dataset'),

        assert data.source
        morph = not data.aggregate
        options = ctx.options_for('evoked-stc-group-dataset', 'baseline', 'src_baseline', 'samplingrate', 'decim', 'data', ndvar=True, model=model, morph=morph, cat=test_obj.cat)
        return Dependency('evoked-stc-group-dataset', options=options, label='dataset'),

    def build(self, ctx: Request) -> Dataset | ROIData:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        ds = ctx.load('dataset')

        if data.source and data.aggregate:
            assert isinstance(ds, ROIData)
            apply_vardef(ds.n_trials_ds, test_obj.vars, self.tests, self.groups)
            for label_ds in ds.label_data.values():
                apply_vardef(label_ds, test_obj.vars, self.tests, self.groups)
            return ds

        assert isinstance(ds, Dataset)
        apply_vardef(ds, test_obj.vars, self.tests, self.groups)
        if data.sensor:
            return ds

        if smooth := ctx.options['smooth']:
            y = data.response_key(ds)
            ds[y] = ds[y].smooth('source', smooth, 'gaussian')
        return ds


class TestResultDerivative(ResultOutputDerivative):
    """Cached statistical test result.

    Uses the shared result-output options from
    :class:`ResultOutputDerivative`.
    """
    name = 'test-result'
    cache_suffix = '.pickle'
    path = Derivative.path
    key_options = {**RESULT_OPTION_DEFAULTS, 'disconnect_labels': False}
    view_options = {}

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        data = ctx.options['data']
        if data is None:
            raise RuntimeError(f"{self.name!r} requires the 'data' option")
        if data.source:
            return RESULT_SOURCE_GROUP_KEY_FIELDS
        return RESULT_SENSOR_GROUP_KEY_FIELDS

    def cache_label(self, ctx: Request) -> str:
        return join_stem_parts(self._path_stem(ctx), f'samples-{ctx.options["samples"]}') if ctx.options['samples'] is not None else self._path_stem(ctx)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES)),)

    def build(self, ctx: Request):
        test_obj = self.tests[ctx.options['test']]
        data = ctx.options['data']
        test_spec = ResolvedTestNDSpec.from_request(ctx)
        data_value = ctx.load('evoked-test-data')
        if isinstance(data_value, ROIData):
            subjects = list(self.groups[ctx.state['group']])
            n_per_label = {label: len(ds['subject'].cells) for label, ds in data_value.label_data.items()}
            do_mcc = len(data_value.label_data) > 1 and ctx.options['pmin'] not in (None, 'tfce') and len(set(n_per_label.values())) == 1
            label_results = {
                label: test_spec.make_result(self, 'label_tc', ds, test_obj, do_mcc)
                for label, ds in data_value.label_data.items()
            }
            merged_dist = _MergedTemporalClusterDist([res._cdist for res in label_results.values()]) if do_mcc else None
            return ROITestResult(subjects, ctx.options['samples'], data_value.n_trials_ds, merged_dist, label_results)
        if data.sensor and len(data_value.info['sensor_types']) > 1:
            desc = ', '.join(data_value.info['sensor_types'])
            raise RuntimeError(f"Data contains more than one sensor type ({desc}). Mass-univariate tests are not designed for multiple sensor types. Use the data argument to perform test on one sensor type.")
        return test_spec.make_result(self, test_spec.data.response_key(data_value), data_value, test_obj)

    def load(
            self,
            ctx: Request,
            path: Path):
        res = load.unpickle(path)
        if ctx.options['data'].source and not ctx.options['data'].aggregate:
            update_subjects_dir(res, ctx.root / MRI_SDIR, 2)
        return res

    def save(
            self,
            ctx: Request,
            path: Path,
            value,
    ) -> None:
        save.pickle(value, path)
