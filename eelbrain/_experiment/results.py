# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Shared result-output derivatives and cache helpers."""

from __future__ import annotations

from pathlib import Path
from os.path import relpath
from typing import Any, TypeVar

from .. import load
from .. import plot
from .. import save
from .. import testnd
from .._io.pickle import update_subjects_dir
from .._stats.stats import ttest_t
from .derivative_cache import Artifact, Derivative, DerivativeContext, Input, file_fingerprint
from .preprocessing import CachedRawPipe
from .test_def import TestDims

T = TypeVar('T')
BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')


class RejectionInput(Input):
    name = 'rej-input'

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            rej = p._artifact_rejection[p.get('rej')]
            if rej['kind'] is None:
                return {'kind': 'none'}
            epoch = p._epochs[p.get('epoch')]
            return {
                'rej': p.get('rej'),
                'files': [
                    file_fingerprint(p.root, p.get('rej-file', epoch=e), 'rej-file')
                    for e in epoch.rej_file_epochs
                ],
            }


def sampled_artifact_path(path: str, samples: int | None) -> str:
    if samples is None:
        return path
    path_ = Path(path)
    return str(path_.with_name(f"{path_.stem}_samples-{samples}{path_.suffix}"))


def result_state_snapshot(p, single_subject: bool) -> dict[str, Any]:
    fields = (
        'analysis', 'test', 'test_options', 'test_dims', 'epoch', 'raw',
        'rej', 'cov', 'inv', 'src', 'mrisubject', 'model',
        'equalize_evoked_count', 'parc', 'folder', 'resname',
    )
    state = {field: p.get(field) for field in fields}
    if single_subject:
        state['subject'] = p.get('subject')
    else:
        state['group'] = p.get('group')
    return p._derivatives.canonicalize(state)


def result_subject_states(p, single_subject: bool) -> list[dict[str, Any]]:
    fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split',
        'raw', 'epoch', 'rej', 'cov', 'mrisubject', 'src', 'inv',
    )
    with p._temporary_state:
        if single_subject:
            return [{field: p.get(field) for field in fields}]
        return [{field: p.get(field) for field in fields} for _ in p]


def result_raw_dependency(p, state: dict[str, Any]) -> dict[str, Any]:
    pipe = p._raw[state['raw']]
    if isinstance(pipe, CachedRawPipe):
        handle = p._derivatives.resolve(pipe.raw_cache_node_name(), state=state, options={'noise': False})
    else:
        handle = p._derivatives.resolve(pipe.raw_meeg_input_name(), state=state, options={'add_bads': True, 'noise': False})
    return handle.describe_dependency()


def result_dependencies(
        p,
        data: TestDims,
        single_subject: bool = False,
) -> dict[str, Any]:
    subjects = []
    for state in result_subject_states(p, single_subject):
        subject_dependencies = {
            'state': {key: state[key] for key in BIDS_ENTITY_KEYS if state[key] is not None},
            'events': p._derivatives.resolve('events', state=state).describe_dependency(),
            'raw': result_raw_dependency(p, state),
            'rej': p._derivatives.resolve('rej-input', state=state).describe_dependency(),
        }
        if data.source:
            subject_dependencies['inv'] = p._derivatives.resolve('inv', state=state).describe_dependency()
        subjects.append(subject_dependencies)

    out = {'subjects': subjects}
    if data.source and data.parc_level:
        if single_subject:
            out['annot'] = p._annot_dependency(p.get('mrisubject'))
        elif data.parc_level == 'common':
            out['annot'] = p._annot_dependency(p.get('common_brain'))
        elif data.parc_level == 'individual':
            out['annot'] = [
                {
                    'subject': state['subject'],
                    'files': p._annot_dependency(state['mrisubject']),
                }
                for state in result_subject_states(p, False)
            ]
        else:
            raise RuntimeError(f"data={data.string!r}, parc_level={data.parc_level!r}")
    return p._derivatives.canonicalize(out)


def result_definitions(p) -> dict[str, Any]:
    definitions = {
        'test': p._tests[p.get('test')]._as_dict(),
        'epoch': p._epochs[p.get('epoch')]._as_dict(),
    }
    parc = p.get('parc')
    if parc and parc in p._parcs:
        definitions['parc'] = p._parcs[parc]._as_dict()
    return p._derivatives.canonicalize(definitions)


class ResultOutputDerivative(Derivative[T]):
    path_template = 'report-file'
    key_fields = ()
    single_subject = False
    sampled_path = False

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {}

    def extra_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.extra_key(ctx)

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        data = ctx.option('data')
        return ctx.registry.canonicalize({
            'state': result_state_snapshot(ctx.pipeline, self.single_subject),
            'samples': ctx.option('samples'),
            'single_subject': self.single_subject,
            'data': None if data is None else data.string,
            **self.extra_key(ctx),
        })

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        data = ctx.option('data')
        return {
            'data': None if data is None else data.string,
            'single_subject': self.single_subject,
            'definitions': result_definitions(ctx.pipeline),
            'dependencies': result_dependencies(ctx.pipeline, data, self.single_subject),
            'extra': ctx.registry.canonicalize(self.extra_fingerprint(ctx)),
        }

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> str:
        path = sampled_artifact_path(ctx.option('dst'), ctx.option('samples')) if self.sampled_path else ctx.option('dst')
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> T:
        return artifact.path

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: T,
    ) -> None:
        return

    def provenance(
            self,
            ctx: DerivativeContext,
            value: T,
    ) -> dict[str, Any]:
        return {'samples': ctx.option('samples'), 'single_subject': self.single_subject}


class TestResultDerivative(ResultOutputDerivative):
    name = 'test-result'
    path_template = 'test-file'
    sampled_path = True

    def build(self, ctx: DerivativeContext):
        _, res = ctx.pipeline._materialize_test_context(ctx, None, False, relpath(sampled_artifact_path(ctx.option('dst'), ctx.option('samples')), ctx.pipeline.get('test-dir')))
        return res

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ):
        res = load.unpickle(artifact.path)
        if ctx.option('data').source is True:
            update_subjects_dir(res, ctx.get('mri-sdir'), 2)
        return res

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value,
    ) -> None:
        save.pickle(value, artifact.path)

    def validate(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            manifest,
    ) -> bool:
        return manifest.provenance.get('samples') == ctx.option('samples')

    def provenance(
            self,
            ctx: DerivativeContext,
            value,
    ) -> dict[str, Any]:
        return {'samples': value.samples}


class MovieDerivative(ResultOutputDerivative[str]):
    name = 'movie'
    path_template = 'group-mov-file'

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'movie_kind': ctx.option('movie_kind')}

    def build(self, ctx: DerivativeContext) -> str:
        p = ctx.pipeline
        kind = ctx.option('movie_kind')
        dst = ctx.option('dst')
        if kind == 'ga-dspm':
            if ctx.option('single_subject'):
                ds = p._load_evoked_stc_context(ctx, ctx.option('subject'))
                y = ds['src']
            else:
                ds = p._load_evoked_stc_context(ctx, p.get('group'), morph=True)
                y = ds['srcm']
            brain = plot.brain.dspm(y, ctx.option('fmin'), ctx.option('fmin') * 3, colorbar=False, **ctx.option('brain_kwargs'))
        elif kind == 'ttest':
            cluster_state = dict(ctx.option('cluster_state') or {})
            if ctx.option('single_subject'):
                ds = p._load_epochs_stc_context(ctx, ctx.option('subject'), cat=ctx.option('cat'))
                y = 'src'
            else:
                ds = p._load_evoked_stc_context(ctx, ctx.option('group'), morph=True, cat=ctx.option('cat'))
                y = 'srcm'
            if cluster_state:
                cluster_state.update(samples=0, pmin=ctx.option('p'))
            if '{test_options}' in p.get('resname'):
                raise RuntimeError("Movie state not fully initialized")
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
