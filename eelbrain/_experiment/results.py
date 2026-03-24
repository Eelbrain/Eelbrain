# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Shared result-output derivatives and cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
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
from .test_def import Test
from .test_def import TestDims

T = TypeVar('T')
BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')
USE_CTX = object()


class RejectionInput(Input):
    name = 'rej-input'

    def __init__(
            self,
            root: str,
            artifact_rejection: dict[str, dict[str, Any]],
            epochs: dict[str, Any],
    ):
        self.root = root
        self.artifact_rejection = artifact_rejection
        self.epochs = epochs

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        rej = self.artifact_rejection[ctx.get('rej')]
        if rej['kind'] is None:
            return {'kind': 'none'}
        epoch = self.epochs[ctx.get('epoch')]
        return {
            'rej': ctx.get('rej'),
            'files': [
                file_fingerprint(self.root, ctx.path('rej-file', epoch=e), 'rej-file')
                for e in epoch.rej_file_epochs
            ],
        }


@dataclass
class ResultSupport:
    pipeline: Any
    raw_pipes: dict[str, Any]
    tests: dict[str, Test]
    epochs: dict[str, Any]
    parcs: dict[str, Any]

    def state_snapshot(
            self,
            ctx: DerivativeContext,
            single_subject: bool,
    ) -> dict[str, Any]:
        fields = (
            'analysis', 'test', 'test_options', 'test_dims', 'epoch', 'raw',
            'rej', 'cov', 'inv', 'src', 'mrisubject', 'model',
            'equalize_evoked_count', 'parc', 'folder', 'resname',
        )
        state = {field: ctx.get(field) for field in fields}
        if single_subject:
            state['subject'] = ctx.get('subject')
        else:
            state['group'] = ctx.get('group')
        return ctx.registry.canonicalize(state)

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

        p = self.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            return [{field: p.get(field) for field in fields} for _ in p]

    def raw_dependency(
            self,
            ctx: DerivativeContext,
            state: dict[str, Any],
    ) -> dict[str, Any]:
        pipe = self.raw_pipes[state['raw']]
        if isinstance(pipe, CachedRawPipe):
            handle = ctx.registry.resolve(pipe.raw_cache_node_name(), state=state, options={'noise': False})
        else:
            handle = ctx.registry.resolve(pipe.raw_meeg_input_name(), state=state, options={'add_bads': True, 'noise': False})
        return handle.describe_dependency()

    def dependencies(
            self,
            ctx: DerivativeContext,
            data: TestDims,
            single_subject: bool = False,
    ) -> dict[str, Any]:
        subjects = []
        for state in self.subject_states(ctx, single_subject):
            subject_dependencies = {
                'state': {key: state[key] for key in BIDS_ENTITY_KEYS if state[key] is not None},
                'events': ctx.registry.resolve('events', state=state).describe_dependency(),
                'raw': self.raw_dependency(ctx, state),
                'rej': ctx.registry.resolve('rej-input', state=state).describe_dependency(),
            }
            if data.source:
                subject_dependencies['inv'] = ctx.registry.resolve('inv', state=state).describe_dependency()
            subjects.append(subject_dependencies)

        out = {'subjects': subjects}
        if data.source and data.parc_level:
            parc = ctx.get('parc')
            if single_subject:
                out['annot'] = ctx.registry.resolve('annot', state={'mrisubject': ctx.get('mrisubject'), 'parc': parc}).describe_dependency()
            elif data.parc_level == 'common':
                out['annot'] = ctx.registry.resolve('annot', state={'mrisubject': ctx.get('common_brain'), 'parc': parc}).describe_dependency()
            elif data.parc_level == 'individual':
                out['annot'] = [
                    {
                        'subject': state['subject'],
                        'files': ctx.registry.resolve('annot', state={'mrisubject': state['mrisubject'], 'parc': parc}).describe_dependency(),
                    }
                    for state in self.subject_states(ctx, False)
                ]
            else:
                raise RuntimeError(f"data={data.string!r}, parc_level={data.parc_level!r}")
        return ctx.registry.canonicalize(out)

    def definitions(self, ctx: DerivativeContext) -> dict[str, Any]:
        definitions = {
            'test': self.tests[ctx.get('test')]._as_dict(),
            'epoch': self.epochs[ctx.get('epoch')]._as_dict(),
        }
        parc = ctx.get('parc')
        if parc and parc in self.parcs:
            definitions['parc'] = self.parcs[parc]._as_dict()
        return ctx.registry.canonicalize(definitions)

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
        p = self.pipeline
        if data is USE_CTX:
            data = ctx.option('data')
        if parc is USE_CTX:
            parc = ctx.option('parc')
        if mask is USE_CTX:
            mask = ctx.option('mask')
        return p._load_test(
            ctx.option('test'),
            ctx.option('tstart'),
            ctx.option('tstop'),
            ctx.option('pmin'),
            parc,
            mask,
            ctx.option('samples'),
            data,
            ctx.option('baseline'),
            ctx.option('src_baseline'),
            return_data,
            make,
            ctx.option('smooth'),
            ctx.option('samplingrate'),
        )

    def materialize_test(
            self,
            ctx: DerivativeContext,
            res,
            return_data: bool,
            desc: str | None = None,
    ):
        test_obj = self.tests[ctx.option('test')]
        return self.pipeline._materialize_test_request(
            test_obj,
            ctx.option('data'),
            ctx.option('baseline'),
            ctx.option('src_baseline'),
            ctx.option('mask'),
            ctx.option('parc'),
            ctx.option('pmin'),
            ctx.option('tstart'),
            ctx.option('tstop'),
            ctx.option('samples'),
            ctx.option('smooth'),
            ctx.option('samplingrate'),
            res,
            return_data,
            desc,
        )

    def load_spm(self, ctx: DerivativeContext):
        return self.pipeline._load_spm(ctx.option('baseline'), ctx.option('src_baseline'))

    def test_kwargs(
            self,
            ctx: DerivativeContext,
            data,
            parc_dim: None | str,
    ):
        return self.pipeline._test_kwargs(ctx.option('samples'), ctx.option('pmin'), ctx.option('tstart'), ctx.option('tstop'), data, parc_dim)

    def load_evoked_stc(
            self,
            ctx: DerivativeContext,
            subjects=None,
            morph: bool = False,
            cat=None,
    ):
        return self.pipeline.load_evoked_stc(subjects, ctx.option('baseline'), ctx.option('src_baseline'), morph=morph, cat=cat)

    def load_epochs_stc(
            self,
            ctx: DerivativeContext,
            subjects=None,
            cat=None,
    ):
        return self.pipeline.load_epochs_stc(subjects, ctx.option('baseline'), ctx.option('src_baseline'), cat=cat)

    def result_desc(self, ctx: DerivativeContext) -> str:
        dst = sampled_artifact_path(ctx.option('dst'), ctx.option('samples'))
        return relpath(dst, ctx.path('test-dir'))


def sampled_artifact_path(path: str, samples: int | None) -> str:
    if samples is None:
        return path
    path_ = Path(path)
    return str(path_.with_name(f"{path_.stem}_samples-{samples}{path_.suffix}"))


class ResultOutputDerivative(Derivative[T]):
    path_template = 'report-file'
    key_fields = ()
    single_subject = False
    sampled_path = False

    def __init__(self, support: ResultSupport):
        self.support = support

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {}

    def extra_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.extra_key(ctx)

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        data = ctx.option('data')
        return ctx.registry.canonicalize({
            'state': self.support.state_snapshot(ctx, self.single_subject),
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
            'definitions': self.support.definitions(ctx),
            'dependencies': self.support.dependencies(ctx, data, self.single_subject),
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
        _, res = self.support.materialize_test(ctx, None, False, self.support.result_desc(ctx))
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
        kind = ctx.option('movie_kind')
        dst = ctx.option('dst')
        if kind == 'ga-dspm':
            if ctx.option('single_subject'):
                ds = self.support.load_evoked_stc(ctx, ctx.option('subject'))
                y = ds['src']
            else:
                ds = self.support.load_evoked_stc(ctx, ctx.get('group'), morph=True)
                y = ds['srcm']
            brain = plot.brain.dspm(y, ctx.option('fmin'), ctx.option('fmin') * 3, colorbar=False, **ctx.option('brain_kwargs'))
        elif kind == 'ttest':
            cluster_state = dict(ctx.option('cluster_state') or {})
            if ctx.option('single_subject'):
                ds = self.support.load_epochs_stc(ctx, ctx.option('subject'), cat=ctx.option('cat'))
                y = 'src'
            else:
                ds = self.support.load_evoked_stc(ctx, ctx.option('group'), morph=True, cat=ctx.option('cat'))
                y = 'srcm'
            if cluster_state:
                cluster_state.update(samples=0, pmin=ctx.option('p'))
            if '{test_options}' in ctx.get('resname'):
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
