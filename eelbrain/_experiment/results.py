# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Result and movie derivatives.

These derivatives orchestrate through other graph nodes, especially the
dataset-producing derivatives that correspond to the public ``Pipeline.load_x``
methods. Cache paths are internal derivative-owned artifacts; only end-product
exports such as movies own default public paths. They must not depend on
injected facade loaders or Pipeline-managed naming state, and naming-only path
labels must not be part of canonical cache identity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

from .. import load
from .. import plot
from .. import save
from .. import testnd
from .._data_obj import Dataset
from .._exceptions import ConfigurationError
from .._io.pickle import update_subjects_dir
from .._text import enumeration
from .._stats.stats import ttest_t
from .._stats.testnd import _MergedTemporalClusterDist
from .derivative_cache import Dependency, Derivative, Request, UncachedDerivative
from .pathing import (
    epoch_basename,
    join_stem_parts,
    movie_export_path,
    mri_sdir,
    report_export_path,
    test_basename,
    time_window_str,
)
from .source import ROIData, roi_data_from_subject_datasets
from .test_def import ROITestResult, ResolvedTestNDSpec, Test, TestDims
from .variable_def import apply_vardef

T = TypeVar('T')
USE_CTX = object()
RESULT_OPTION_DEFAULTS = {
    'samples': None,
    'data': None,
    'test': None,
    'tstart': None,
    'tstop': None,
    'pmin': None,
    'baseline': None,
    'src_baseline': None,
    'smooth': None,
    'samplingrate': None,
}

TEST_DATA_OPTION_NAMES = (
    'data',
    'test',
    'baseline',
    'src_baseline',
    'samplingrate',
    'smooth',
)


def _group_request_state(ctx: Request, **state) -> dict[str, Any]:
    return {**ctx.state, **state, 'subject': None}


def _subject_request_state(ctx: Request, subject: str, **state) -> dict[str, Any]:
    return {**ctx.state, **state, 'subject': subject}


def _test_result_options(
        ctx: Request,
        *,
        data: TestDims | object = USE_CTX,
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


def _evoked_stc_options(
        ctx: Request,
        baseline=USE_CTX,
        src_baseline=USE_CTX,
        morph: bool = False,
        cat=None,
        data_raw: bool = False,
        samplingrate: int | None = None,
        decim: int | None = None,
        ndvar: bool = True,
) -> dict[str, Any]:
    if baseline is USE_CTX:
        baseline = ctx.options['baseline']
    if src_baseline is USE_CTX:
        src_baseline = ctx.options['src_baseline']
    return ctx.options_for(
        'evoked-stc',
        baseline=baseline,
        src_baseline=src_baseline,
        morph=morph,
        cat=cat,
        data_raw=data_raw,
        samplingrate=samplingrate,
        decim=decim,
        ndvar=ndvar,
        keep_evoked=False,
    )


def _epochs_stc_options(
        ctx: Request,
        baseline=USE_CTX,
        src_baseline=USE_CTX,
        cat=None,
        keep_epochs: bool | str = False,
        morph: bool | None = None,
        data_raw: bool = False,
        samplingrate: int | None = None,
        decim: int | None = None,
        ndvar: bool = True,
        reject: bool | str = True,
) -> dict[str, Any]:
    if baseline is USE_CTX:
        baseline = ctx.options['baseline']
    if src_baseline is USE_CTX:
        src_baseline = ctx.options['src_baseline']
    return ctx.options_for(
        'epochs-stc',
        baseline=baseline,
        src_baseline=src_baseline,
        cat=cat,
        keep_epochs=keep_epochs,
        morph=morph,
        data_raw=data_raw,
        samplingrate=samplingrate,
        decim=decim,
        ndvar=ndvar,
        reject=reject,
    )


def _validate_post_aggregation_test_vars(test_obj: Test, data_desc: str):
    model_vars = set(filter(None, (test_obj.model or '').split('%')))
    missing_model_vars = sorted(model_vars.intersection(test_obj.vars.vars))
    if missing_model_vars:
        vars_desc = enumeration(missing_model_vars)
        raise ConfigurationError(
            f"For evoked-backed {data_desc} tests, Test.vars must be computable from the post-aggregation dataset. "
            f"Model variable {vars_desc} can not be provided through Test.vars. Use TwoStageTest or Pipeline.variables instead."
        )


def _apply_post_aggregation_test_vars(ds, test_obj: Test, tests, groups, data_desc: str):
    if not test_obj.vars:
        return ds
    _validate_post_aggregation_test_vars(test_obj, data_desc)
    try:
        apply_vardef(ds, test_obj.vars, tests, groups)
    except Exception as error:
        raise ConfigurationError(f"For evoked-backed {data_desc} tests, Test.vars must be computable from the post-aggregation dataset. Use TwoStageTest or Pipeline.variables for trial-level variables ({error}).") from None
    return ds


class ResultOutputDerivative(Derivative[T]):
    """Shared base for cached result/report/movie outputs.

    This is a :class:`~eelbrain._experiment.derivative_cache.Derivative`
    subclass with a fixed pattern:

    - :meth:`key` encodes the logical analysis identity, independent of any
      explicit output destination.
    - :meth:`fingerprint` delegates to :meth:`Derivative.standard_fingerprint`
      using shared result-state and configured test/epoch/parc definitions.
    - :meth:`path` chooses a user-facing export path, with optional
      ``samples``-specific disambiguation.
    - :meth:`load` returns that path, and :meth:`save` is a no-op, because
      subclasses normally create the final output file directly in
      :meth:`build` rather than serializing a separate in-memory artifact.

    The underscored helper methods are grouped by the derivative hook they
    support:

    - ``_key_*`` helpers feed :meth:`key`
    - ``_fingerprint_*`` helpers feed :meth:`fingerprint`
    - ``_path_*`` helpers feed :meth:`path`

    Subclasses usually extend this template by overriding:

    - :meth:`dependencies` and :meth:`build` as ordinary derivative hooks
    - :meth:`_identity_extra` to add result-family-specific identity fields
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
    key_fields = ()
    cache_log_level = logging.INFO
    single_subject = False
    sampled_path = False
    OPTION_DEFAULTS = RESULT_OPTION_DEFAULTS
    VIEW_OPTION_DEFAULTS = {'dst': None}

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

    def _key_state_snapshot(
            self,
            ctx: Request,
            single_subject: bool,
    ) -> dict[str, Any]:
        """Canonical state subset used by :meth:`key`."""
        data = ctx.options['data']
        fields = ['epoch', 'raw', 'rej', 'model', 'equalize_evoked_count', 'test']
        if data and data.source:
            fields.extend(['cov', 'inv', 'src', 'mri', 'parc'])
        state = {field: ctx.state[field] for field in fields}
        if single_subject:
            state['subject'] = ctx.state['subject']
        else:
            state['subjects'] = tuple(self.groups[ctx.state['group']])
        return ctx.registry.canonicalize(state)

    def _fingerprint_state_fields(
            self,
            ctx: Request,
            single_subject: bool,
    ) -> tuple[str, ...]:
        """State keys forwarded to :meth:`Derivative.standard_fingerprint`."""
        data = ctx.options['data']
        fields = ['epoch', 'raw', 'rej', 'model', 'equalize_evoked_count', 'test']
        if data and data.source:
            fields.extend(['cov', 'inv', 'src', 'mri', 'parc'])
        if single_subject:
            fields.append('subject')
        return tuple(fields)

    def _key_analysis_options(self, ctx: Request) -> dict[str, Any]:
        """Canonical analysis options used by :meth:`key`."""
        data = ctx.options['data']
        return ctx.registry.canonicalize({
            'data': None if data is None else data.string,
            'baseline': ctx.options['baseline'],
            'src_baseline': ctx.options['src_baseline'],
            'disconnect_labels': ctx.options.get('disconnect_labels', False),
            'pmin': ctx.options['pmin'],
            'tstart': ctx.options['tstart'],
            'tstop': ctx.options['tstop'],
            'samplingrate': ctx.options['samplingrate'],
            'smooth': ctx.options['smooth'],
            'adjacency': ctx.state['adjacency'],
        })

    def _key_identity(self, ctx: Request) -> dict[str, Any]:
        """Stable logical identity shared by result-output cache keys."""
        return ctx.registry.canonicalize({
            'state': self._key_state_snapshot(ctx, self.single_subject),
            'options': self._key_analysis_options(ctx),
            'single_subject': self.single_subject,
            **self._identity_extra(ctx),
        })

    def _path_context_parts(self, ctx: Request) -> list[str]:
        """Path-stem parts derived from analysis context/state."""
        data = ctx.options['data']
        parts = [f'data-{data.string}', f'raw-{ctx.state["raw"]}', f'rej-{ctx.state["rej"]}']
        if ctx.state['model']:
            parts.append(f'model-{ctx.state["model"]}')
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
            test_basename(ctx.state),
            f'epoch-{ctx.state["epoch"]}',
            f'test-{ctx.options["test"]}',
            self._path_context_parts(ctx),
            self._path_option_parts(ctx),
        )

    def _default_output_path(self, ctx: Request) -> Path:
        """Default user-facing export path used when ``dst`` is not set."""
        return report_export_path(ctx.state, self.name, self._path_stem(ctx), self.single_subject)

    def _fingerprint_definitions(self, ctx: Request) -> dict[str, Any]:
        """Configured definitions embedded in :meth:`fingerprint`."""
        definitions = {
            'test': self.tests[ctx.options['test']]._as_dict(),
            'epoch': self.epochs[ctx.state['epoch']]._as_dict(),
        }
        if ctx.options['data'].source and ctx.state['parc'] in self.parcs:
            definitions['parc'] = self.parcs[ctx.state['parc']]._as_dict()
        return ctx.registry.canonicalize(definitions)

    def _identity_extra(self, ctx: Request) -> dict[str, Any]:
        """Extra identity fields shared by :meth:`key` and :meth:`fingerprint`."""
        return {}

    def key(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize({
            'identity': self._key_identity(ctx),
            'options': ctx.registry.canonicalize(ctx.options),
        })

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(
            ctx,
            state_fields=self._fingerprint_state_fields(ctx, self.single_subject),
            definitions=self._fingerprint_definitions(ctx),
            extra={
                'single_subject': self.single_subject,
                'subjects': None if self.single_subject else tuple(self.groups[ctx.state['group']]),
                **self._identity_extra(ctx),
            },
        )

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

    def provenance(
            self,
            ctx: Request,
            value: T,
    ) -> dict[str, Any]:
        return {'samples': ctx.options['samples'], 'single_subject': self.single_subject}


def sampled_artifact_path(path: str | Path, samples: int | None) -> Path:
    path = Path(path)
    if samples is None:
        return path
    return path.with_name(f"{path.stem}_samples-{samples}{path.suffix}")


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
    key_fields = (
        'group', 'epoch', 'raw', 'rej', 'model', 'equalize_evoked_count',
        'test', 'cov', 'inv', 'src', 'mri',
    )
    OPTION_DEFAULTS = {
        'data': None,
        'test': None,
        'baseline': None,
        'src_baseline': None,
        'samplingrate': None,
        'smooth': None,
    }

    def __init__(self, tests: dict[str, Test], epochs: dict[str, Any], groups: dict[str, tuple[str, ...] | list[str]]):
        self.tests = tests
        self.epochs = epochs
        self.groups = groups

    def _state_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = list(self.key_fields)
        if ctx.options['data'].source:
            fields.append('parc')
        return tuple(fields)

    def key(self, ctx: Request) -> dict[str, Any]:
        key = {field: ctx.state[field] for field in self._state_fields(ctx)}
        key['subjects'] = tuple(self.groups[ctx.state['group']])
        key['options'] = ctx.registry.canonicalize(ctx.options)
        return ctx.registry.canonicalize(key)

    def _sensor_evoked_options(self, ctx: Request, cat) -> dict[str, Any]:
        return ctx.options_for(
            'evoked',
            baseline=ctx.options['baseline'],
            ndvar=True,
            cat=cat,
            samplingrate=ctx.options['samplingrate'],
            decim=None,
            data_raw=False,
            data=ctx.options['data'],
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return self.standard_fingerprint(
            ctx,
            state_fields=self._state_fields(ctx),
            definitions={
                'test': self.tests[ctx.options['test']]._as_dict(),
                'epoch': self.epochs[ctx.state['epoch']]._as_dict(),
            },
            extra={'subjects': tuple(self.groups[ctx.state['group']])},
        )

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        samplingrate = ctx.options['samplingrate']
        subjects = self.groups[ctx.state['group']]

        if data.sensor:
            return (Dependency('evoked-group-dataset', options=self._sensor_evoked_options(ctx, test_obj.cat)),)

        if data.source is True:
            return (Dependency(
                'evoked-stc-group-dataset',
                options=_evoked_stc_options(ctx, morph=True, cat=test_obj.cat, samplingrate=samplingrate),
            ),)

        return tuple(
            Dependency(
                'evoked-stc',
                label=subject,
                state={'subject': subject},
                options=_evoked_stc_options(ctx, morph=False, cat=None, samplingrate=samplingrate),
            )
            for subject in subjects
        )

    def build(self, ctx: Request) -> Dataset | ROIData:
        data = ctx.options['data']
        test_obj = self.tests[ctx.options['test']]
        subjects = self.groups[ctx.state['group']]
        if test_obj.vars:
            _validate_post_aggregation_test_vars(test_obj, data.string)

        if data.sensor:
            ds = ctx.load('evoked-group-dataset', options=self._sensor_evoked_options(ctx, test_obj.cat))
            return _apply_post_aggregation_test_vars(ds, test_obj, self.tests, self.groups, data.string)

        samplingrate = ctx.options['samplingrate']
        if data.source is True:
            ds = ctx.load('evoked-stc-group-dataset', options=_evoked_stc_options(ctx, morph=True, cat=test_obj.cat, samplingrate=samplingrate))
            ds = _apply_post_aggregation_test_vars(ds, test_obj, self.tests, self.groups, data.string)
            if smooth := ctx.options['smooth']:
                ds[data.y_name] = ds[data.y_name].smooth('source', smooth, 'gaussian')
            return ds

        if ctx.options['smooth']:
            raise TypeError(f"smooth={ctx.options['smooth']!r} for ROI tests")

        dss = []
        for subject in subjects:
            ds = ctx.load(
                'evoked-stc',
                state={'subject': subject},
                options=_evoked_stc_options(ctx, morph=False, cat=None, samplingrate=samplingrate),
            )
            dss.append(_apply_post_aggregation_test_vars(ds, test_obj, self.tests, self.groups, data.string))
        return roi_data_from_subject_datasets(dss, data.source)


class TestResultDerivative(ResultOutputDerivative):
    """Cached statistical test result.

    Uses the shared result-output options from
    :class:`ResultOutputDerivative`.
    """
    name = 'test-result'
    sampled_path = True
    cache_suffix = '.pickle'
    path = Derivative.path
    OPTION_DEFAULTS = {**RESULT_OPTION_DEFAULTS, 'disconnect_labels': False}
    VIEW_OPTION_DEFAULTS = {}

    def cache_label(self, ctx: Request) -> str:
        return join_stem_parts(self._path_stem(ctx), f'samples-{ctx.options["samples"]}') if ctx.options['samples'] is not None else self._path_stem(ctx)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES)),)

    def build(self, ctx: Request):
        test_obj = self.tests[ctx.options['test']]
        data = ctx.options['data']
        test_spec = ResolvedTestNDSpec.from_request(ctx, data)
        data_value = ctx.load('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES))
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
        return test_spec.make_result(self, test_spec.data.y_name, data_value, test_obj)

    def load(
            self,
            ctx: Request,
            path: Path):
        res = load.unpickle(path)
        if ctx.options['data'].source is True:
            update_subjects_dir(res, mri_sdir(ctx.state), 2)
        return res

    def save(
            self,
            ctx: Request,
            path: Path,
            value,
    ) -> None:
        save.pickle(value, path)

    def validate(
            self,
            ctx: Request,
            path: Path,
            manifest,
    ) -> bool:
        return manifest.provenance.get('samples') == ctx.options['samples']

    def provenance(
            self,
            ctx: Request,
            value,
    ) -> dict[str, Any]:
        return {'samples': value.samples}


class MovieDerivative(ResultOutputDerivative[Path]):
    """Rendered movie export from source-space data.

    In addition to the shared result-output options, this derivative uses:

    - ``movie_kind`` to choose the movie recipe
    - ``single_subject`` / ``subject`` to choose subject vs group rendering
    - ``cat`` for condition selection
    - ``time_dilation`` for playback speed
    - ``fmin`` and ``brain_kwargs`` for ``ga-dspm`` movies
    - ``p``, ``pmin``, ``pmid``, ``surf``, and ``cluster_state`` for
      ``ttest`` movies
    """
    name = 'movie'
    OPTION_DEFAULTS = {
        **RESULT_OPTION_DEFAULTS,
        'disconnect_labels': False,
        'movie_kind': None,
        'time_dilation': 1.0,
        'single_subject': False,
        'subject': None,
        'cat': None,
        'fmin': None,
        'brain_kwargs': None,
        'p': None,
        'pmid': None,
        'surf': None,
        'cluster_state': None,
    }

    def _identity_extra(self, ctx: Request) -> dict[str, Any]:
        out = {'movie_kind': ctx.options['movie_kind'], 'time_dilation': ctx.options['time_dilation']}
        if ctx.options['movie_kind'] == 'ga-dspm':
            out.update({
                'fmin': ctx.options['fmin'],
                'brain_kwargs': ctx.registry.canonicalize(ctx.options['brain_kwargs']),
            })
        elif ctx.options['movie_kind'] == 'ttest':
            out.update({
                'cat': ctx.options['cat'],
                'p': ctx.options['p'],
                'pmin': ctx.options['pmin'],
                'pmid': ctx.options['pmid'],
                'surf': ctx.options['surf'],
                'cluster_state': ctx.registry.canonicalize(ctx.options['cluster_state'] or {}),
            })
        return out

    def _path_stem(self, ctx: Request) -> str:
        if ctx.options['movie_kind'] == 'ga-dspm':
            return join_stem_parts(
                epoch_basename(ctx.state),
                f'epoch-{ctx.state["epoch"]}',
                'ga-dspm',
                self._path_context_parts(ctx),
                self._path_option_parts(ctx),
                f'surf-{ctx.options["brain_kwargs"]["surf"]}',
                f'fmin-{ctx.options["fmin"]:g}',
            )
        cat = ctx.options['cat']
        if not cat:
            contrast = 'ga'
        elif len(cat) == 1:
            contrast = f'{cat[0]}'
        else:
            contrast = f'{cat[0]}-{cat[1]}'
        return join_stem_parts(
            epoch_basename(ctx.state),
            f'epoch-{ctx.state["epoch"]}',
            'ttest',
            self._path_context_parts(ctx),
            self._path_option_parts(ctx),
            f'contrast-{contrast}',
            f'p-{ctx.options["p"]}',
            f'surf-{ctx.options["surf"]}',
        )

    def _default_output_path(self, ctx: Request) -> Path:
        return movie_export_path(ctx.state, self._path_stem(ctx), ctx.options['single_subject'])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        kind = ctx.options['movie_kind']
        if kind == 'ga-dspm':
            if ctx.options['single_subject']:
                return (Dependency(
                    'evoked-stc',
                    state=_subject_request_state(ctx, ctx.options['subject']),
                    options=_evoked_stc_options(ctx),
                ),)
            return (Dependency(
                'evoked-stc-group-dataset',
                state=_group_request_state(ctx),
                options=_evoked_stc_options(ctx, morph=True),
            ),)
        if kind == 'ttest':
            if ctx.options['single_subject']:
                return (Dependency(
                    'epochs-stc',
                    state=_subject_request_state(ctx, ctx.options['subject']),
                    options=_epochs_stc_options(ctx, cat=ctx.options['cat']),
                ),)
            return (Dependency(
                'evoked-stc-group-dataset',
                state=_group_request_state(ctx),
                options=_evoked_stc_options(ctx, morph=True, cat=ctx.options['cat']),
            ),)
        return ()

    def build(self, ctx: Request) -> Path:
        kind = ctx.options['movie_kind']
        dst = self.path(ctx)
        if kind == 'ga-dspm':
            if ctx.options['single_subject']:
                ds = ctx.load('evoked-stc', state=_subject_request_state(ctx, ctx.options['subject']), options=_evoked_stc_options(ctx))
                y = ds['src']
            else:
                ds = ctx.load('evoked-stc-group-dataset', state=_group_request_state(ctx), options=_evoked_stc_options(ctx, morph=True))
                y = ds['srcm']
            brain = plot.brain.dspm(y, ctx.options['fmin'], ctx.options['fmin'] * 3, colorbar=False, **ctx.options['brain_kwargs'])
        elif kind == 'ttest':
            cluster_state = dict(ctx.options['cluster_state'] or {})
            if ctx.options['single_subject']:
                ds = ctx.load('epochs-stc', state=_subject_request_state(ctx, ctx.options['subject']), options=_epochs_stc_options(ctx, cat=ctx.options['cat']))
                y = 'src'
            else:
                ds = ctx.load('evoked-stc-group-dataset', state=_group_request_state(ctx), options=_evoked_stc_options(ctx, morph=True, cat=ctx.options['cat']))
                y = 'srcm'
            if cluster_state:
                cluster_state.update(samples=0, pmin=ctx.options['p'])
            if ctx.options['disconnect_labels']:
                cluster_state['parc'] = 'source'
            cat = ctx.options['cat']
            if ctx.state['model'] and cat and len(cat) == 2:
                c1, c0 = cat
                if ctx.options['single_subject']:
                    res = testnd.TTestIndependent(y, ctx.state['model'], c1, c0, data=ds, **cluster_state)
                else:
                    res = testnd.TTestRelated(y, ctx.state['model'], c1, c0, match='subject', data=ds, **cluster_state)
            elif cat:
                res = testnd.TTestOneSample(y, data=ds, **cluster_state)
            else:
                res = testnd.TTestOneSample(y, data=ds, **cluster_state)
            tmap = res.masked_parameter_map(None) if cluster_state else res.t
            brain = plot.brain.dspm(
                tmap,
                ttest_t(ctx.options['p'], res.df),
                ttest_t(ctx.options['pmin'], res.df),
                ttest_t(ctx.options['pmid'], res.df),
                surf=ctx.options['surf'],
            )
        else:
            raise RuntimeError(f"{kind=}")
        brain.save_movie(dst, ctx.options['time_dilation'])
        brain.close()
        return dst

    def provenance(
            self,
            ctx: Request,
            value: str,
    ) -> dict[str, Any]:
        return {'movie_kind': ctx.options['movie_kind']}
