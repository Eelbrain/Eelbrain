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

from itertools import product
import logging
from pathlib import Path
from typing import Any, TypeVar

from .. import load
from .. import plot
from .. import save
from .. import testnd
from .._data_obj import NDVar, Var, combine
from .._exceptions import OldVersionError
from .._io.pickle import update_subjects_dir
from .._stats.stats import ttest_t
from .._stats.testnd import _MergedTemporalClusterDist
from .derivative_cache import Dependency, Derivative, DerivativeContext
from .pathing import (
    epoch_basename,
    join_stem_parts,
    movie_export_path,
    mri_sdir,
    report_export_path,
    test_basename,
    time_window_str,
)
from .test_def import ROI2StageResult, ROITestResult, Test, TestDims, TwoStageTest

T = TypeVar('T')
USE_CTX = object()


def _group_request_state(ctx: DerivativeContext, **state) -> dict[str, Any]:
    return {**ctx.state, **state, 'group': ctx.get('group'), 'subject': None}


def _subject_request_state(ctx: DerivativeContext, subject: str, **state) -> dict[str, Any]:
    return {**ctx.state, **state, 'subject': subject, 'group': None}


def _test_result_options(
        ctx: DerivativeContext,
        *,
        data: TestDims | object = USE_CTX,
        parc: str | None | object = USE_CTX,
        mask: str | None | bool | object = USE_CTX,
        make: bool = False,
) -> dict[str, Any]:
    if data is USE_CTX:
        data = ctx.option('data')
    if parc is USE_CTX:
        parc = ctx.option('parc')
    if mask is USE_CTX:
        mask = ctx.option('mask')
    return {
        'data': data,
        'samples': ctx.option('samples'),
        'test': ctx.option('test'),
        'tstart': ctx.option('tstart'),
        'tstop': ctx.option('tstop'),
        'pmin': ctx.option('pmin'),
        'parc': parc,
        'mask': mask,
        'baseline': ctx.option('baseline'),
        'src_baseline': ctx.option('src_baseline'),
        'smooth': ctx.option('smooth'),
        'samplingrate': ctx.option('samplingrate'),
        '_allow_protected_overwrite': make,
    }


class ResultOutputDerivative(Derivative[T]):
    """Shared base for cached result/report/movie outputs.

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
    parc, mask
        Optional parcellation or source-space mask controls.
    baseline
        Sensor-space baseline correction.
    src_baseline
        Source-space baseline correction.
    samplingrate
        Sampling rate override for upstream cached data.
    smooth
        Optional source-space smoothing.
    """
    key_fields = ()
    cache_log_level = logging.INFO
    single_subject = False
    sampled_path = False

    def __init__(
            self,
            tests: dict[str, Test],
            epochs: dict[str, Any],
            parcs: dict[str, Any],
            groups: dict[str, tuple[str, ...] | list[str]],
            field_values: dict[str, tuple[str, ...] | list[str]],
            mri_subjects: dict[str, dict[str, str]],
            common_brain: str,
            cluster_criteria: dict[str, dict[str, Any]],
            brain_plot_defaults: dict[str, Any] | None = None,
    ):
        self.tests = tests
        self.epochs = epochs
        self.parcs = parcs
        self.groups = groups
        self.field_values = field_values
        self.mri_subjects = mri_subjects
        self.common_brain = common_brain
        self.cluster_criteria = cluster_criteria
        self.brain_plot_defaults = {} if brain_plot_defaults is None else brain_plot_defaults

    def _field_options(
            self,
            state: dict[str, Any],
            field: str,
            *,
            subject: str | None = None,
    ) -> list[Any]:
        value = state.get(field)
        if value not in (None, '', '*'):
            return [value]
        if field == 'subject':
            group = state.get('group')
            if group not in (None, '', '*'):
                return list(self.groups[group])
        if field == 'mrisubject':
            if subject is None:
                raise RuntimeError("mrisubject needs subject")
            mri = state.get('mri', '')
            value = self.mri_subjects[mri][subject]
            if value == self.common_brain or value.startswith('sub-'):
                return [value]
            return ['sub-' + value]
        return list(self.field_values.get(field, (value,)))

    def collect_states(
            self,
            state: dict[str, Any],
            fields: tuple[str, ...],
            **fixed,
    ) -> list[dict[str, Any]]:
        merged_state = {**state, **fixed}
        subjects = self._field_options(merged_state, 'subject') if 'subject' in fields else [merged_state.get('subject')]
        other_fields = [field for field in fields if field not in ('subject', 'mrisubject')]
        out = []
        for subject in subjects:
            options = [self._field_options(merged_state, field, subject=subject) for field in other_fields]
            for values in product(*options) if options else [()]:
                item = dict(merged_state)
                if 'subject' in fields:
                    item['subject'] = subject
                item.update(zip(other_fields, values))
                if 'mrisubject' in fields:
                    item['mrisubject'] = self._field_options(item, 'mrisubject', subject=item['subject'])[0]
                out.append({field: item[field] for field in fields})
        return out

    def state_snapshot(
            self,
            ctx: DerivativeContext,
            single_subject: bool,
    ) -> dict[str, Any]:
        data = ctx.option('data')
        fields = ['epoch', 'raw', 'rej', 'model', 'equalize_evoked_count', 'test']
        if data and data.source:
            fields.extend(['cov', 'inv', 'src', 'mri'])
        state = {field: ctx.get(field) for field in fields}
        if single_subject:
            state['subject'] = ctx.get('subject')
        else:
            state['group'] = ctx.get('group')
        return ctx.registry.canonicalize(state)

    def analysis_options(self, ctx: DerivativeContext) -> dict[str, Any]:
        mask = ctx.option('mask')
        if mask is True:
            mask = ctx.get('parc')
        data = ctx.option('data')
        return ctx.registry.canonicalize({
            'data': None if data is None else data.string,
            'baseline': ctx.option('baseline'),
            'src_baseline': ctx.option('src_baseline'),
            'pmin': ctx.option('pmin'),
            'tstart': ctx.option('tstart'),
            'tstop': ctx.option('tstop'),
            'parc': ctx.option('parc'),
            'mask': mask,
            'samplingrate': ctx.option('samplingrate'),
            'smooth': ctx.option('smooth'),
            'adjacency': ctx.get('adjacency'),
            'select_clusters': ctx.get('select_clusters'),
        })

    def cache_identity(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.canonicalize({
            'state': self.state_snapshot(ctx, self.single_subject),
            'options': self.analysis_options(ctx),
            'single_subject': self.single_subject,
            **self.extra_key(ctx),
        })

    def _context_parts(self, ctx: DerivativeContext) -> list[str]:
        data = ctx.option('data')
        parts = [f'data-{data.string}', f'raw-{ctx.get("raw")}', f'rej-{ctx.get("rej")}']
        if ctx.get('model'):
            parts.append(f'model-{ctx.get("model")}')
        if ctx.get('equalize_evoked_count'):
            parts.append(f'count-{ctx.get("equalize_evoked_count")}')
        if data.source:
            parts.extend((f'cov-{ctx.get("cov")}', f'src-{ctx.get("src")}', f'inv-{ctx.get("inv")}'))
        return parts

    def _option_parts(self, ctx: DerivativeContext) -> list[str]:
        parts = []
        baseline = ctx.option('baseline')
        src_baseline = ctx.option('src_baseline')
        pmin = ctx.option('pmin')
        parc = ctx.option('parc')
        mask = ctx.option('mask')
        samplingrate = ctx.option('samplingrate')
        smooth = ctx.option('smooth')
        if baseline is False:
            parts.append('nobl')
        elif baseline not in (None, True):
            parts.append(f'bl-{time_window_str(baseline)}')
        if src_baseline is True:
            parts.append('srcbl')
        elif src_baseline not in (None, False):
            parts.append(f'srcbl-{time_window_str(src_baseline)}')
        if parc:
            parts.append(f'parc-{parc}')
        elif mask:
            parts.append(f'mask-{ctx.get("parc") if mask is True else mask}')
        if pmin == 'tfce':
            parts.append('tfce')
        elif pmin is not None:
            parts.append(f'p-{pmin}')
        if pmin not in (None, 'tfce') and ctx.get('select_clusters'):
            parts.append(f'clusters-{ctx.get("select_clusters")}')
        if pmin is not None and ctx.option('data').source and ctx.get('adjacency'):
            parts.append(f'adj-{ctx.get("adjacency")}')
        if ctx.option('tstart') is not None or ctx.option('tstop') is not None:
            parts.append(f'tw-{time_window_str((ctx.option("tstart"), ctx.option("tstop")))}')
        if samplingrate is not None:
            parts.append(f'sr-{samplingrate:g}Hz')
        if smooth:
            parts.append(f'sm-{int(round(smooth * 1000))}mm')
        return parts

    def path_stem(self, ctx: DerivativeContext) -> str:
        return join_stem_parts(test_basename(ctx.state), f'epoch-{ctx.get("epoch")}', f'test-{ctx.option("test")}', self._context_parts(ctx), self._option_parts(ctx))

    def default_path(self, ctx: DerivativeContext) -> Path:
        return report_export_path(ctx.state, self.name, self.path_stem(ctx), self.single_subject)

    def subject_states(
            self,
            ctx: DerivativeContext,
            single_subject: bool,
    ) -> list[dict[str, Any]]:
        fields = (
            'subject', 'session', 'task', 'acquisition', 'run', 'split',
            'raw', 'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv',
        )
        if single_subject:
            return [{field: ctx.get(field) for field in fields}]
        return self.collect_states(ctx.state, fields)

    def _result_parc(self, ctx: DerivativeContext) -> str | None:
        parc = ctx.option('parc')
        if parc:
            return parc
        mask = ctx.option('mask')
        if mask is True:
            return ctx.get('parc')
        if isinstance(mask, str):
            return mask
        return None

    def definitions(self, ctx: DerivativeContext) -> dict[str, Any]:
        definitions = {
            'test': self.tests[ctx.get('test')]._as_dict(),
            'epoch': self.epochs[ctx.get('epoch')]._as_dict(),
        }
        parc = self._result_parc(ctx)
        if parc and parc in self.parcs:
            definitions['parc'] = self.parcs[parc]._as_dict()
        return ctx.registry.canonicalize(definitions)

    def _evoked_options(
            self,
            ctx: DerivativeContext,
            baseline=USE_CTX,
            ndvar: bool = True,
            cat=None,
            samplingrate=None,
            decim=None,
            data_raw: bool = False,
            vardef: str | None = None,
            data=None,
    ):
        if baseline is USE_CTX:
            baseline = ctx.option('baseline')
        data = TestDims.coerce(data)
        return {
            'baseline': baseline,
            'ndvar': ndvar,
            'cat': cat,
            'samplingrate': samplingrate,
            'decim': decim,
            'data_raw': data_raw,
            'vardef': vardef,
            'data': data,
        }

    def _evoked_stc_options(
            self,
            ctx: DerivativeContext,
            baseline=USE_CTX,
            src_baseline=USE_CTX,
            morph: bool = False,
            cat=None,
            mask=False,
            data_raw: bool = False,
            vardef: str | None = None,
            samplingrate: int | None = None,
            decim: int | None = None,
            ndvar: bool = True,
    ):
        if baseline is USE_CTX:
            baseline = ctx.option('baseline')
        if src_baseline is USE_CTX:
            src_baseline = ctx.option('src_baseline')
        return {
            'baseline': baseline,
            'src_baseline': src_baseline,
            'morph': morph,
            'cat': cat,
            'parc': ctx.option('parc'),
            'mask': mask,
            'data_raw': data_raw,
            'vardef': vardef,
            'samplingrate': samplingrate,
            'decim': decim,
            'ndvar': ndvar,
            'keep_evoked': False,
        }

    def _epochs_stc_options(
            self,
            ctx: DerivativeContext,
            baseline=USE_CTX,
            src_baseline=USE_CTX,
            cat=None,
            keep_epochs: bool | str = False,
            morph: bool | None = None,
            mask: bool | str = False,
            data_raw: bool = False,
            vardef: str | None = None,
            samplingrate: int | None = None,
            decim: int | None = None,
            ndvar: bool = True,
            reject: bool | str = True,
    ):
        if baseline is USE_CTX:
            baseline = ctx.option('baseline')
        if src_baseline is USE_CTX:
            src_baseline = ctx.option('src_baseline')
        return {
            'baseline': baseline,
            'src_baseline': src_baseline,
            'cat': cat,
            'keep_epochs': keep_epochs,
            'morph': morph,
            'parc': ctx.option('parc'),
            'mask': mask,
            'data_raw': data_raw,
            'vardef': vardef,
            'samplingrate': samplingrate,
            'decim': decim,
            'ndvar': ndvar,
            'reject': reject,
        }

    def load_group_evoked(
            self,
            ctx: DerivativeContext,
            baseline=USE_CTX,
            ndvar: bool = True,
            cat=None,
            samplingrate=None,
            decim=None,
            data_raw: bool = False,
            vardef: str | None = None,
            data=None,
            **state,
    ):
        options = self._evoked_options(ctx, baseline, ndvar, cat, samplingrate, decim, data_raw, vardef, data)
        return ctx.load('evoked-group-dataset', state=_group_request_state(ctx, **state), options=options)

    def load_group_evoked_stc(
            self,
            ctx: DerivativeContext,
            baseline=USE_CTX,
            src_baseline=USE_CTX,
            morph: bool = False,
            cat=None,
            mask=False,
            data_raw: bool = False,
            vardef: str | None = None,
            samplingrate: int | None = None,
            decim: int | None = None,
            ndvar: bool = True,
            **state,
    ):
        options = self._evoked_stc_options(ctx, baseline, src_baseline, morph, cat, mask, data_raw, vardef, samplingrate, decim, ndvar)
        return ctx.load('evoked-stc-group-dataset', state=_group_request_state(ctx, **state), options=options)

    def load_subject_evoked_stc(
            self,
            ctx: DerivativeContext,
            subject: str,
            baseline=USE_CTX,
            src_baseline=USE_CTX,
            morph: bool = False,
            cat=None,
            mask=False,
            data_raw: bool = False,
            vardef: str | None = None,
            samplingrate: int | None = None,
            decim: int | None = None,
            ndvar: bool = True,
            **state,
    ):
        options = self._evoked_stc_options(ctx, baseline, src_baseline, morph, cat, mask, data_raw, vardef, samplingrate, decim, ndvar)
        return ctx.load('evoked-stc', state=_subject_request_state(ctx, subject, **state), options=options)

    def load_subject_epochs_stc(
            self,
            ctx: DerivativeContext,
            subject: str,
            baseline=USE_CTX,
            src_baseline=USE_CTX,
            cat=None,
            keep_epochs: bool | str = False,
            morph: bool | None = None,
            mask: bool | str = False,
            data_raw: bool = False,
            vardef: str | None = None,
            samplingrate: int | None = None,
            decim: int | None = None,
            ndvar: bool = True,
            reject: bool | str = True,
            **state,
    ):
        options = self._epochs_stc_options(ctx, baseline, src_baseline, cat, keep_epochs, morph, mask, data_raw, vardef, samplingrate, decim, ndvar, reject)
        return ctx.load('epochs-stc', state=_subject_request_state(ctx, subject, **state), options=options)

    def result_desc(self, ctx: DerivativeContext) -> str:
        return ctx.registry.describe_artifact_path(self.path(ctx))

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {}

    def extra_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.extra_key(ctx)

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.canonicalize({
            **self.cache_identity(ctx),
            'samples': ctx.option('samples'),
        })

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'identity': self.cache_identity(ctx),
            'definitions': self.definitions(ctx),
            'options': self.analysis_options(ctx),
            'extra': ctx.registry.canonicalize(self.extra_fingerprint(ctx)),
        }

    def path(
            self,
            ctx: DerivativeContext,
    ) -> Path:
        dst = ctx.option('dst')
        path = Path(dst) if dst else self.default_path(ctx)
        if self.sampled_path:
            path = sampled_artifact_path(path, ctx.option('samples'))
        return path

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> T:
        return path

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: T,
    ) -> None:
        return

    def provenance(
            self,
            ctx: DerivativeContext,
            value: T,
    ) -> dict[str, Any]:
        return {'samples': ctx.option('samples'), 'single_subject': self.single_subject}


def sampled_artifact_path(path: str | Path, samples: int | None) -> Path:
    path = Path(path)
    if samples is None:
        return path
    return path.with_name(f"{path.stem}_samples-{samples}{path.suffix}")


class TestResultDerivative(ResultOutputDerivative):
    """Cached statistical test result.

    Uses the shared result-output options from
    :class:`ResultOutputDerivative`.
    """
    name = 'test-result'
    sampled_path = True
    cache_suffix = '.pickle'
    path = Derivative.path

    def cache_label(self, ctx: DerivativeContext) -> str:
        return join_stem_parts(self.path_stem(ctx), f'samples-{ctx.option("samples")}') if ctx.option('samples') is not None else self.path_stem(ctx)

    def load_test(
            self,
            ctx: DerivativeContext,
            return_data: bool,
            make: bool,
            *,
            data: TestDims | object = USE_CTX,
            parc: str | None | object = USE_CTX,
            mask: str | None | bool | object = USE_CTX,
    ):
        options = _test_result_options(ctx, data=data, parc=parc, mask=mask, make=make)
        handle = ctx.registry.resolve('test-result', state=ctx.state, options=options)
        dst = handle.artifact_path
        desc = ctx.registry.describe_artifact_path(dst)

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

        res_data, res = self.materialize_test(ctx, res, True, desc)
        return (res_data, res) if return_data else res

    def materialize_test(
            self,
            ctx: DerivativeContext,
            res,
            return_data: bool,
            desc: str | None = None,
    ):
        test_obj = self.tests[ctx.option('test')]
        data = ctx.option('data')
        mask = ctx.option('mask')
        parc = ctx.option('parc')
        pmin = ctx.option('pmin')
        smooth = ctx.option('smooth')
        samplingrate = ctx.option('samplingrate')
        baseline = ctx.option('baseline')
        src_baseline = ctx.option('src_baseline')

        parc_dim = None
        if data.source is True:
            if parc:
                mask = parc
                parc_dim = 'source'
            elif mask and pmin is None:
                parc_dim = 'source'
        elif isinstance(data.source, str):
            if not isinstance(parc, str):
                raise TypeError(f"parc needs to be set for ROI test (data={data.string!r})")
            if mask is not None:
                raise TypeError(f"{mask=}: invalid for data={data.string!r}")
        else:
            if parc is not None:
                raise TypeError(f"{parc=}: invalid for data={data.string!r}")
            if mask is not None:
                raise TypeError(f"{mask=}: invalid for data={data.string!r}")

        do_test = res is None
        test_kwargs = self.test_kwargs(ctx, data, parc_dim) if do_test else None

        if isinstance(test_obj, TwoStageTest):
            if smooth:
                raise NotImplementedError(f"{smooth=}: smoothing for two-stage tests")
            if isinstance(data.source, str):
                return self._make_test_rois_2stage(ctx, baseline, src_baseline, test_obj, test_kwargs, res, data, return_data, samplingrate)
            if data.source is True:
                return self._make_test_2stage(ctx, baseline, src_baseline, mask, test_obj, test_kwargs, res, data, return_data, samplingrate)
            raise NotImplementedError(f"Two-stage test with data={data.string!r}")
        if isinstance(data.source, str):
            if smooth:
                raise TypeError(f"{smooth=} for ROI tests")
            return self._make_test_rois(ctx, baseline, src_baseline, test_obj, pmin, test_kwargs, res, data, samplingrate)

        if data.sensor:
            res_data = self.load_group_evoked(ctx, baseline, True, test_obj.cat, samplingrate, None, False, test_obj.vars, data)
            if len(res_data.info['sensor_types']) > 1:
                desc_ = ', '.join(res_data.info['sensor_types'])
                raise RuntimeError(f"Data contains more than one sensor type ({desc_}). Mass-univariate tests are not designed for multiple sensor types. Use the data argument to perform test on one sensor type.")
        elif data.source:
            res_data = self.load_group_evoked_stc(ctx, baseline, src_baseline, True, test_obj.cat, mask, False, test_obj.vars, samplingrate, None, True)
            if smooth:
                res_data[data.y_name] = res_data[data.y_name].smooth('source', smooth, 'gaussian')
        else:
            raise ValueError(f"data={data.string!r}")

        if do_test:
            res = self.make_test(data.y_name, res_data, test_obj, test_kwargs)

        return (res_data, res) if return_data else (None, res)

    def load_spm(self, ctx: DerivativeContext):
        subject = ctx.get('subject')
        test_obj = self.tests[ctx.option('test')]
        if not isinstance(test_obj, TwoStageTest):
            raise NotImplementedError(f"Test kind {test_obj.__class__.__name__!r}")
        ds = self.load_subject_epochs_stc(ctx, subject, ctx.option('baseline'), ctx.option('src_baseline'), None, False, None, True, False, test_obj.vars)
        return testnd.LM('src', test_obj.stage_1, data=ds, samples=0, subject=subject)

    def test_kwargs(
            self,
            ctx: DerivativeContext,
            data,
            parc_dim: None | str,
    ):
        dims = data.dims if hasattr(data, 'dims') else data
        kwargs = {
            'samples': ctx.option('samples'),
            'tstart': ctx.option('tstart'),
            'tstop': ctx.option('tstop'),
            'parc': parc_dim,
        }
        pmin = ctx.option('pmin')
        if pmin == 'tfce':
            kwargs['tfce'] = True
        elif pmin is not None:
            kwargs['pmin'] = pmin
            criteria = self.cluster_criteria[ctx.get('select_clusters')]
            kwargs.update({'min' + dim: criteria[dim] for dim in dims if dim in criteria})
        return kwargs

    def make_test(
            self,
            y,
            ds,
            test: Test,
            kwargs: dict[str, Any],
            force_permutation: bool = False,
    ):
        test_obj = test if isinstance(test, Test) else self.tests[test]
        if isinstance(y, str):
            y = ds.eval(y)
        if isinstance(y, Var):
            return test_obj.make_uv(y, ds)
        if isinstance(y, list):
            dim = 'sensor' if y[0].has_dim('sensor') else 'source'
            return test_obj.make_uv(combine([getattr(yi, 'mean')(dim) for yi in y]), ds)
        if isinstance(y, NDVar) and y.has_dim('space'):
            return test_obj.make_vec(y, ds, force_permutation, kwargs)
        return test_obj.make(y, ds, force_permutation, kwargs)

    @staticmethod
    def _src_to_label_tc(ds, func):
        src = ds.pop('src')
        out = {}
        for label in src.source.parc.cells:
            if label.startswith('unknown-'):
                continue
            label_ds = ds.copy()
            label_ds['label_tc'] = getattr(src, func)(source=label)
            out[label] = label_ds
        return out

    def _subjects(self, ctx: DerivativeContext) -> list[str]:
        group = ctx.get('group')
        if group not in (None, '', '*'):
            return list(self.groups[group])
        return [ctx.get('subject')]

    def _make_test_rois(self, ctx: DerivativeContext, baseline, src_baseline, test_obj, pmin, test_kwargs, res, data, samplingrate):
        dss_list = []
        n_trials_dss = []
        labels = set()
        subjects = self._subjects(ctx)
        for subject in subjects:
            ds = self.load_subject_evoked_stc(ctx, subject, baseline, src_baseline, False, None, False, False, test_obj.vars, samplingrate)
            dss = self._src_to_label_tc(ds, data.source)
            n_trials_dss.append(ds)
            dss_list.append(dss)
            labels.update(dss.keys())

        label_dss = {label: [dss[label] for dss in dss_list if label in dss] for label in labels}
        label_data = {label: combine(dss, incomplete='drop') for label, dss in label_dss.items()}
        if res is not None:
            return label_data, res

        n_trials_ds = combine(n_trials_dss, incomplete='drop')
        n_per_label = {label: len(dss) for label, dss in label_dss.items()}
        do_mcc = len(labels) > 1 and pmin not in (None, 'tfce') and len(set(n_per_label.values())) == 1
        label_results = {label: self.make_test('label_tc', ds, test_obj, test_kwargs, do_mcc) for label, ds in label_data.items()}
        merged_dist = _MergedTemporalClusterDist([res_._cdist for res_ in label_results.values()]) if do_mcc else None
        res = ROITestResult(subjects, ctx.option('samples'), n_trials_ds, merged_dist, label_results)
        return label_data, res

    def _make_test_rois_2stage(self, ctx: DerivativeContext, baseline, src_baseline, test_obj, test_kwargs, res, data, return_data, samplingrate):
        lms = []
        res_data = []
        n_trials_dss = []
        subjects = self._subjects(ctx)
        for subject in subjects:
            if test_obj.model is None:
                ds = self.load_subject_epochs_stc(ctx, subject, baseline, src_baseline, None, False, None, True, False, test_obj.vars, samplingrate)
            else:
                ds = self.load_subject_evoked_stc(ctx, subject, baseline, src_baseline, False, None, True, False, test_obj.vars, samplingrate, model=test_obj.model)
            dss = self._src_to_label_tc(ds, data.source)
            if res is None:
                lms.append({label: test_obj.make_stage_1('label_tc', label_ds, subject) for label, label_ds in dss.items()})
                n_trials_dss.append(ds)
            if return_data:
                res_data.append(dss)

        if res is None:
            labels = set().union(*(subject_lms.keys() for subject_lms in lms))
            ress = {}
            for label in sorted(labels):
                label_lms = [subject_lms[label] for subject_lms in lms if label in subject_lms]
                if len(label_lms) > 2:
                    ress[label] = test_obj.make_stage_2(label_lms, test_kwargs)
            n_trials_ds = combine(n_trials_dss, incomplete='drop')
            res = ROI2StageResult(subjects, ctx.option('samples'), n_trials_ds, None, ress)

        if return_data:
            data_out = {}
            for label in res.keys():
                label_data = [subject_data[label] for subject_data in res_data if label in subject_data]
                data_out[label] = combine(label_data)
        else:
            data_out = None
        return data_out, res

    def _make_test_2stage(self, ctx: DerivativeContext, baseline, src_baseline, mask, test_obj, test_kwargs, res, data, return_data, samplingrate):
        lms = []
        res_data = []
        for subject in self._subjects(ctx):
            if test_obj.model is None:
                ds = self.load_subject_epochs_stc(ctx, subject, baseline, src_baseline, None, False, True, mask, False, test_obj.vars, samplingrate)
            else:
                ds = self.load_subject_evoked_stc(ctx, subject, baseline, src_baseline, True, None, mask, False, test_obj.vars, samplingrate, model=test_obj.model)
            if res is None:
                lms.append(test_obj.make_stage_1(data.y_name, ds, subject))
            if return_data:
                res_data.append(ds)

        if res is None:
            res = test_obj.make_stage_2(lms, test_kwargs)
        return (combine(res_data), res) if return_data else (None, res)

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        test_obj = self.tests[ctx.option('test')]
        data = ctx.option('data')
        parc = ctx.option('parc')
        mask = ctx.option('mask')
        pmin = ctx.option('pmin')
        samplingrate = ctx.option('samplingrate')
        baseline = ctx.option('baseline')
        src_baseline = ctx.option('src_baseline')

        if data.source is True:
            if parc:
                mask = parc
            elif mask and pmin is None:
                mask = mask
        elif isinstance(data.source, str):
            mask = False

        if isinstance(test_obj, TwoStageTest):
            deps = []
            for subject in self._subjects(ctx):
                if isinstance(data.source, str):
                    if test_obj.model is None:
                        deps.append(Dependency(
                            'epochs-stc',
                            label=subject,
                            state=_subject_request_state(ctx, subject),
                            options=self._epochs_stc_options(ctx, baseline, src_baseline, None, False, None, True, False, test_obj.vars, samplingrate),
                        ))
                    else:
                        deps.append(Dependency(
                            'evoked-stc',
                            label=subject,
                            state=_subject_request_state(ctx, subject, model=test_obj.model),
                            options=self._evoked_stc_options(ctx, baseline, src_baseline, False, None, True, False, test_obj.vars, samplingrate),
                        ))
                elif data.source is True:
                    if test_obj.model is None:
                        deps.append(Dependency(
                            'epochs-stc',
                            label=subject,
                            state=_subject_request_state(ctx, subject),
                            options=self._epochs_stc_options(ctx, baseline, src_baseline, None, False, True, mask, False, test_obj.vars, samplingrate),
                        ))
                    else:
                        deps.append(Dependency(
                            'evoked-stc',
                            label=subject,
                            state=_subject_request_state(ctx, subject, model=test_obj.model),
                            options=self._evoked_stc_options(ctx, baseline, src_baseline, True, None, mask, False, test_obj.vars, samplingrate),
                        ))
            return tuple(deps)

        if isinstance(data.source, str):
            return tuple(
                Dependency(
                    'evoked-stc',
                    label=subject,
                    state=_subject_request_state(ctx, subject),
                    options=self._evoked_stc_options(ctx, baseline, src_baseline, False, None, False, False, test_obj.vars, samplingrate),
                )
                for subject in self._subjects(ctx)
            )
        if data.sensor:
            return (Dependency(
                'evoked-group-dataset',
                state=_group_request_state(ctx),
                options=self._evoked_options(ctx, baseline, True, test_obj.cat, samplingrate, None, False, test_obj.vars, data),
            ),)
        if data.source:
            return (Dependency(
                'evoked-stc-group-dataset',
                state=_group_request_state(ctx),
                options=self._evoked_stc_options(ctx, baseline, src_baseline, True, test_obj.cat, mask, False, test_obj.vars, samplingrate, None, True),
            ),)
        return ()

    def build(self, ctx: DerivativeContext):
        _, res = self.materialize_test(ctx, None, False, self.result_desc(ctx))
        return res

    def load(
            self,
            ctx: DerivativeContext,
            path: Path):
        res = load.unpickle(path)
        if ctx.option('data').source is True:
            update_subjects_dir(res, mri_sdir(ctx.state), 2)
        return res

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value,
    ) -> None:
        save.pickle(value, path)

    def validate(
            self,
            ctx: DerivativeContext,
            path: Path,
            manifest,
    ) -> bool:
        return manifest.provenance.get('samples') == ctx.option('samples')

    def provenance(
            self,
            ctx: DerivativeContext,
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

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        out = {'movie_kind': ctx.option('movie_kind'), 'time_dilation': ctx.option('time_dilation')}
        if ctx.option('movie_kind') == 'ga-dspm':
            out.update({
                'fmin': ctx.option('fmin'),
                'brain_kwargs': ctx.registry.canonicalize(ctx.option('brain_kwargs')),
            })
        elif ctx.option('movie_kind') == 'ttest':
            out.update({
                'cat': ctx.option('cat'),
                'p': ctx.option('p'),
                'pmin': ctx.option('pmin'),
                'pmid': ctx.option('pmid'),
                'surf': ctx.option('surf'),
                'cluster_state': ctx.registry.canonicalize(ctx.option('cluster_state') or {}),
            })
        return out

    def path_stem(self, ctx: DerivativeContext) -> str:
        if ctx.option('movie_kind') == 'ga-dspm':
            return join_stem_parts(
                epoch_basename(ctx.state),
                f'epoch-{ctx.get("epoch")}',
                'ga-dspm',
                self._context_parts(ctx),
                self._option_parts(ctx),
                f'surf-{ctx.option("brain_kwargs")["surf"]}',
                f'fmin-{ctx.option("fmin"):g}',
            )
        cat = ctx.option('cat')
        if not cat:
            contrast = 'ga'
        elif len(cat) == 1:
            contrast = f'{cat[0]}'
        else:
            contrast = f'{cat[0]}-{cat[1]}'
        return join_stem_parts(
            epoch_basename(ctx.state),
            f'epoch-{ctx.get("epoch")}',
            'ttest',
            self._context_parts(ctx),
            self._option_parts(ctx),
            f'contrast-{contrast}',
            f'p-{ctx.option("p")}',
            f'surf-{ctx.option("surf")}',
        )

    def default_path(self, ctx: DerivativeContext) -> Path:
        return movie_export_path(ctx.state, self.path_stem(ctx), ctx.option('single_subject'))

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        kind = ctx.option('movie_kind')
        if kind == 'ga-dspm':
            if ctx.option('single_subject'):
                return (Dependency(
                    'evoked-stc',
                    state=_subject_request_state(ctx, ctx.option('subject')),
                    options=self._evoked_stc_options(ctx),
                ),)
            return (Dependency(
                'evoked-stc-group-dataset',
                state=_group_request_state(ctx),
                options=self._evoked_stc_options(ctx, morph=True),
            ),)
        if kind == 'ttest':
            if ctx.option('single_subject'):
                return (Dependency(
                    'epochs-stc',
                    state=_subject_request_state(ctx, ctx.option('subject')),
                    options=self._epochs_stc_options(ctx, cat=ctx.option('cat')),
                ),)
            return (Dependency(
                'evoked-stc-group-dataset',
                state=_group_request_state(ctx),
                options=self._evoked_stc_options(ctx, morph=True, cat=ctx.option('cat')),
            ),)
        return ()

    def build(self, ctx: DerivativeContext) -> Path:
        kind = ctx.option('movie_kind')
        dst = self.path(ctx)
        if kind == 'ga-dspm':
            if ctx.option('single_subject'):
                ds = ctx.load('evoked-stc', state=_subject_request_state(ctx, ctx.option('subject')), options=self._evoked_stc_options(ctx))
                y = ds['src']
            else:
                ds = ctx.load('evoked-stc-group-dataset', state=_group_request_state(ctx), options=self._evoked_stc_options(ctx, morph=True))
                y = ds['srcm']
            brain = plot.brain.dspm(y, ctx.option('fmin'), ctx.option('fmin') * 3, colorbar=False, **ctx.option('brain_kwargs'))
        elif kind == 'ttest':
            cluster_state = dict(ctx.option('cluster_state') or {})
            if ctx.option('single_subject'):
                ds = ctx.load('epochs-stc', state=_subject_request_state(ctx, ctx.option('subject')), options=self._epochs_stc_options(ctx, cat=ctx.option('cat')))
                y = 'src'
            else:
                ds = ctx.load('evoked-stc-group-dataset', state=_group_request_state(ctx), options=self._evoked_stc_options(ctx, morph=True, cat=ctx.option('cat')))
                y = 'srcm'
            if cluster_state:
                cluster_state.update(samples=0, pmin=ctx.option('p'))
            if ctx.get('model') and ctx.option('cat') and len(ctx.option('cat')) == 2:
                c1, c0 = ctx.option('cat')
                if ctx.option('single_subject'):
                    res = testnd.TTestIndependent(y, ctx.get('model'), c1, c0, data=ds, **cluster_state)
                else:
                    res = testnd.TTestRelated(y, ctx.get('model'), c1, c0, match='subject', data=ds, **cluster_state)
            elif ctx.option('cat'):
                res = testnd.TTestOneSample(y, data=ds, **cluster_state)
            else:
                res = testnd.TTestOneSample(y, data=ds, **cluster_state)
            tmap = res.masked_parameter_map(None) if cluster_state else res.t
            brain = plot.brain.dspm(
                tmap,
                ttest_t(ctx.option('p'), res.df),
                ttest_t(ctx.option('pmin'), res.df),
                ttest_t(ctx.option('pmid'), res.df),
                surf=ctx.option('surf'),
            )
        else:
            raise RuntimeError(f"{kind=}")
        brain.save_movie(dst, ctx.option('time_dilation'))
        brain.close()
        return dst

    def provenance(
            self,
            ctx: DerivativeContext,
            value: str,
    ) -> dict[str, Any]:
        return {'movie_kind': ctx.option('movie_kind')}
