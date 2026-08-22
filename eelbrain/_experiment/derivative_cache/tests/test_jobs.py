from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from eelbrain._experiment.derivative_cache import (
    ALLOW_PROTECTED_OVERWRITE,
    Dependency,
    Derivative,
    DerivativeRegistry,
    ExternalArtifactDerivative,
    Job,
    JobInputsChangedError,
    JobSpec,
    ProtectedArtifactError,
    Request,
)
from eelbrain._experiment.derivative_cache.tests.test_derivative_cache import (
    DEFAULT_STATE,
    NarrowingDerivative,
    SourceInput,
    make_empty_registry,
    make_source_registry,
)
from eelbrain.testing import TempDir


# ---------------------------------------------------------------------------
# Test doubles: jobs and the nodes that make them
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _EchoJob(Job):
    "Trivial data-carrying job: the input is already loaded, so it needs no registry."
    text: str

    def __call__(self) -> str:
        return self.text.upper()


class JobDerivative(Derivative[str]):
    "Derivative whose build is expressed as make_job()(), like TRF and epoch rejection."
    name = 'job'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return self.make_job(ctx)()

    def make_job(self, ctx: Request) -> _EchoJob:
        with ctx._build_deps_context():
            return _EchoJob(ctx.load('source'))

    def load(self, ctx: Request, path: str) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: str, value: str) -> None:
        Path(path).write_text(value)


class ProtectedJobDerivative(JobDerivative):
    "Job derivative whose artifact lives outside cache-dir (user-visible, like the 'protected' node)"
    name = 'protected-job'

    def path(self, ctx: Request) -> Path:
        return ctx.registry.deriv_dir / 'mne' / f"{ctx.state['subject']}_protected-job.txt"


class RacyJobDerivative(JobDerivative):
    "Job derivative whose input is changed by another session while make_job reads it"
    name = 'racy-job'

    def __init__(self, source: SourceInput):
        self._source = source
        self.interrupt = None  # text another session writes mid-load, set per test

    def make_job(self, ctx: Request) -> _EchoJob:
        with ctx._build_deps_context():
            job = _EchoJob(ctx.load('source'))
        if self.interrupt is not None:
            self._source.source_path(ctx.state['subject']).write_text(self.interrupt)
        return job


class ExternalWriterDerivative(ExternalArtifactDerivative[str]):
    "Derivative whose build() writes the real artifact outside cache-dir, like 'src'"
    name = 'external-writer'
    key_fields = ('subject',)

    def path(self, ctx: Request) -> Path:
        return ctx.registry.deriv_dir / 'freesurfer' / f"{ctx.state['subject']}-external.txt"

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        path = self.path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('built')
        return 'built'

    def load(self, ctx: Request, path: str) -> str:
        return Path(path).read_text()


class FingerprintJobDerivative(Derivative[str]):
    "Job derivative whose own fingerprint (not its dependencies) tracks mutable state, like ICA bad channels"
    name = 'fingerprint-job'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, source: SourceInput):
        self._source = source

    def _source_text(self, ctx: Request) -> str:
        return self._source.source_path(ctx.state['subject']).read_text()

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'text': self._source_text(ctx)}

    def build(self, ctx: Request) -> str:
        return self.make_job(ctx)()

    def make_job(self, ctx: Request) -> _EchoJob:
        return _EchoJob(self._source_text(ctx))

    def load(self, ctx: Request, path: str) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: str, value: str) -> None:
        Path(path).write_text(value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_job_spec_round_trip():
    "A job computed off-host is re-united with its cache entry through JobSpec"
    root, registry, _source = make_source_registry()
    registry.register(JobDerivative())

    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    assert not spec.is_done
    assert spec.key == {'subject': 's1'}

    job = pickle.loads(pickle.dumps(spec.make_job()))  # "off-host"
    assert job.key == spec.key
    assert job.node == 'job'
    assert job.provenance is not None  # travels with the data, so it survives the round trip
    assert spec.save_result(job, job()) == 'ALPHA'
    assert spec.is_done
    assert Path(spec.path).read_text() == 'ALPHA'
    assert Path(spec.ctx.manifest_path).exists()

    # a fresh spec sees the cached artifact and reads it back
    assert JobSpec(registry.resolve('job', state=DEFAULT_STATE)).is_done
    assert registry.resolve('job', state=DEFAULT_STATE).load() == 'ALPHA'


def test_job_cache_logging():
    "The cache reports the work: a build in place, and generating a job for one computed elsewhere"
    root = TempDir()
    # As Pipeline builds it: a private instance with parent=None, so records only
    # reach these handlers via the registry the node was resolved through.
    log = logging.Logger('TestExperiment', logging.DEBUG)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    log.addHandler(handler)
    registry = DerivativeRegistry(root, log)
    source = SourceInput(root)
    source.source_path('s1').write_text('alpha')
    registry.register(source)
    registry.register(JobDerivative())

    # built in place: Request.load_artifact emits it
    assert registry.resolve('job', state=DEFAULT_STATE).load() == 'ALPHA'
    messages = [record.getMessage() for record in records]
    assert sum(message.startswith('Build job:') for message in messages) == 1

    # computed through a spec: JobSpec.make_job reports generating the job, exactly
    # once, and does not claim a build the cache did not perform
    Path(registry.resolve('job', state=DEFAULT_STATE).artifact_path).unlink()
    records.clear()
    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    job = spec.make_job()
    spec.save_result(job, job())
    messages = [record.getMessage() for record in records]
    assert sum(message.startswith('Generate job for job:') for message in messages) == 1
    assert not any(message.startswith('Build job:') for message in messages)


def test_job_result_records_make_time_inputs():
    "A result that comes back after its inputs changed is stale, not wrongly valid"
    root, registry, source = make_source_registry()
    registry.register(JobDerivative())

    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    job = spec.make_job()  # reads 'alpha'
    source.source_path('s1').write_text('changed')  # ... while the job is computed off-host
    spec.save_result(job, job())

    # the artifact is kept, and filed under the input it was computed from
    assert Path(spec.path).read_text() == 'ALPHA'
    assert JobSpec(registry.resolve('job', state=DEFAULT_STATE)).is_done is False
    assert registry.resolve('job', state=DEFAULT_STATE).load() == 'CHANGED'

    # a job whose inputs held still is valid, as before
    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    job = spec.make_job()
    spec.save_result(job, job())
    assert JobSpec(registry.resolve('job', state=DEFAULT_STATE)).is_done is True


def test_job_result_is_not_memoized_as_valid():
    "A result filed under a stale snapshot is not recorded as valid for the enclosing load"
    root, registry, source = make_source_registry()
    registry.register(JobDerivative())

    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    job = spec.make_job()  # reads 'alpha'
    source.source_path('s1').write_text('changed')  # ... while the job is computed off-host
    # The validity memo is shared across one nested load; the stale manifest this
    # writes must not enter it, or the artifact that needs rebuilding reads as valid.
    with registry._load_context():
        spec.save_result(job, job())
        assert registry.resolve('job', state=DEFAULT_STATE).is_valid() is False

    # a job whose inputs held still is memoized as valid, as an in-place build is
    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    job = spec.make_job()
    with registry._load_context():
        spec.save_result(job, job())
        assert registry.resolve('job', state=DEFAULT_STATE).is_valid() is True


def test_job_result_records_make_time_fingerprint():
    "The node's own fingerprint over mutable state is recorded as of make_job(), not save time"
    root, registry, source = make_source_registry()
    registry.register(FingerprintJobDerivative(source))

    spec = JobSpec(registry.resolve('fingerprint-job', state=DEFAULT_STATE))
    job = spec.make_job()  # fingerprints 'alpha'
    source.source_path('s1').write_text('changed')  # ... while the job is computed off-host
    spec.save_result(job, job())

    # filed under the fingerprint it was computed from, so it is stale, not wrongly valid
    assert Path(spec.path).read_text() == 'ALPHA'
    assert JobSpec(registry.resolve('fingerprint-job', state=DEFAULT_STATE)).is_done is False
    assert registry.resolve('fingerprint-job', state=DEFAULT_STATE).load() == 'CHANGED'


def test_job_refuses_inputs_that_change_during_the_load():
    "An input that changes while make_job reads it makes the job refuse, before anything is computed"
    root, registry, source = make_source_registry()
    node = RacyJobDerivative(source)
    registry.register(node)

    node.interrupt = 'changed'
    spec = JobSpec(registry.resolve('racy-job', state=DEFAULT_STATE))
    with pytest.raises(JobInputsChangedError):
        spec.make_job()
    # no job was returned, so nothing can be filed, and the spec can be retried
    assert not Path(spec.path).exists()

    node.interrupt = None
    job = spec.make_job()
    assert spec.save_result(job, job()) == 'CHANGED'
    assert JobSpec(registry.resolve('racy-job', state=DEFAULT_STATE)).is_done


def test_job_save_result_requires_a_matching_job():
    "save_result() files a result only against the job that computed it"
    root, registry, _source = make_source_registry()
    registry.register(JobDerivative())
    registry.register(FingerprintJobDerivative(_source))

    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    # a job the spec did not make carries no provenance to file the result under
    with pytest.raises(RuntimeError, match="provenance"):
        spec.save_result(_EchoJob('alpha'), 'ALPHA')

    # ... nor one made for a different node, whose key may well be identical
    other = JobSpec(registry.resolve('fingerprint-job', state=DEFAULT_STATE))
    other_job = other.make_job()
    assert other_job.key == spec.key
    with pytest.raises(RuntimeError, match="but this spec is for"):
        spec.save_result(other_job, other_job())

    job = spec.make_job()
    assert spec.save_result(job, job()) == 'ALPHA'


def test_external_writer_build_is_not_self_protected():
    "A build that writes its own external artifact is not blocked by the protected check"
    root, registry, _source = make_source_registry()
    registry.register(ExternalWriterDerivative())

    # first build: build() writes the external file, so it exists by the time the
    # manifest is written -- that must not read as an unauthorized overwrite
    ctx = registry.resolve('external-writer', state=DEFAULT_STATE)
    assert ctx.load() == 'built'
    assert Path(ctx.artifact_path).exists()
    assert Path(ctx.manifest_path).exists()
    assert registry.resolve('external-writer', state=DEFAULT_STATE).is_valid()


def test_job_result_does_not_clobber_protected_artifact():
    "save_result() refuses to overwrite a non-cache artifact without authorization, like load() does"
    root, registry, _source = make_source_registry()
    registry.register(ProtectedJobDerivative())

    spec = JobSpec(registry.resolve('protected-job', state=DEFAULT_STATE))
    job = spec.make_job()
    result = job()
    # a user-owned file appears at the external path while the job computes
    Path(spec.path).parent.mkdir(parents=True, exist_ok=True)
    Path(spec.path).write_text('user data')
    with pytest.raises(ProtectedArtifactError):
        spec.save_result(job, result)
    assert Path(spec.path).read_text() == 'user data'

    spec = JobSpec(registry.resolve('protected-job', state=DEFAULT_STATE, controls={ALLOW_PROTECTED_OVERWRITE}))
    job = spec.make_job()
    assert spec.save_result(job, job()) == 'ALPHA'


def test_build_after_make_job_records_current_inputs():
    "A build in place records current inputs, even on a request that made a job earlier"
    root, registry, source = make_source_registry()
    registry.register(JobDerivative())

    ctx = registry.resolve('job', state=DEFAULT_STATE)
    JobSpec(ctx).make_job()  # captures 'alpha'
    source.source_path('s1').write_text('changed')

    assert ctx.load() == 'CHANGED'  # built here and now, from the current input
    assert JobSpec(registry.resolve('job', state=DEFAULT_STATE)).is_done is True


def test_job_result_is_filed_under_its_own_job():
    "One spec can make several jobs; each result is filed under the data its own job holds"
    root, registry, source = make_source_registry()
    registry.register(JobDerivative())

    spec = JobSpec(registry.resolve('job', state=DEFAULT_STATE))
    first = spec.make_job()  # holds 'alpha'
    source.source_path('s1').write_text('changed')
    second = spec.make_job()  # holds 'changed'

    # the second result files under 'changed' and is valid
    assert spec.save_result(second, second()) == 'CHANGED'
    assert JobSpec(registry.resolve('job', state=DEFAULT_STATE)).is_done is True

    # re-sending the first job (which still holds 'alpha') files it under 'alpha',
    # so the artifact reads as stale rather than as a current result
    assert spec.save_result(first, first()) == 'ALPHA'
    assert JobSpec(registry.resolve('job', state=DEFAULT_STATE)).is_done is False
    assert registry.resolve('job', state=DEFAULT_STATE).load() == 'CHANGED'


def test_job_spec_with_controls():
    "with_controls() adds controls and carries everything else over unchanged"
    root, registry = make_empty_registry()
    registry.register(NarrowingDerivative(root))

    spec = JobSpec(registry.resolve('narrowing', state={'subject': 's1', 'mode': 'narrow'}, options={'alpha': 1}))
    with_control = spec.with_controls('a-control')

    assert with_control.ctx.controls == frozenset({'a-control'})
    assert spec.ctx.controls == frozenset()  # the original is untouched
    assert with_control.key == spec.key
    assert with_control.path == spec.path
    # the copy does not re-derive which options the caller provided, so the
    # inert-option warning does not fire again for the defaulted 'beta'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        spec.with_controls('another')
