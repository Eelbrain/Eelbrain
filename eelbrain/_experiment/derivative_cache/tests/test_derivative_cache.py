from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from eelbrain._experiment.derivative_cache import (
    ALLOW_PROTECTED_OVERWRITE,
    CACHE_DISAMBIGUATION_SUFFIX,
    ArtifactManifest,
    CachePolicy,
    Dependency,
    Derivative,
    GCCategory,
    OptionSpec,
    Request,
    DerivativeRegistry,
    Input,
    ProtectedArtifactError,
    UncachedDerivative,
    VersionedInput,
    compare_manifests,
    file_fingerprint,
)
from eelbrain._experiment.data import DataSpec
from eelbrain._experiment.logging import CacheInvalidation, StructuredFormatter
from eelbrain.testing import TempDir


DEFAULT_STATE = {'subject': 's1', 'mode': 'default'}
LOG = logging.getLogger('eelbrain.test.derivative_cache')


# ---------------------------------------------------------------------------
# Test doubles: pipelines and dependency-graph nodes
# ---------------------------------------------------------------------------

class _TemporaryState:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.state = None

    def __enter__(self):
        self.state = self.pipeline.state.copy()
        return self.pipeline

    def __exit__(self, exc_type, exc, tb):
        self.pipeline.state = self.state


class FakePipeline:
    def __init__(self, root):
        self.root = Path(root)
        self.state = {'subject': 's1', 'mode': 'default'}

    @property
    def _temporary_state(self):
        return _TemporaryState(self)

    def set(self, **kwargs):
        self.state.update(kwargs)

    def get(self, key, mkdir=False, **kwargs):
        state = dict(self.state)
        state.update(kwargs)
        if key in state:
            return state[key]

        if key == 'cache-dir':
            path = self.root / 'derivatives' / 'eelbrain' / 'cache'
            if mkdir:
                path.mkdir(parents=True, exist_ok=True)
            return str(path)
        if key == 'deriv-dir':
            return str(self.root / 'derivatives')
        if key == 'root':
            return str(self.root)

        subject = state['subject']
        if key == 'value-file':
            path = self.root / 'derivatives' / 'eelbrain' / 'cache' / subject / 'value.txt'
        elif key == 'summary-file':
            path = self.root / 'derivatives' / 'eelbrain' / 'cache' / subject / 'summary.txt'
        elif key == 'ephemeral-file':
            path = self.root / 'derivatives' / 'eelbrain' / 'cache' / subject / 'ephemeral.txt'
        elif key == 'protected-file':
            path = self.root / 'derivatives' / 'mne' / subject / 'protected.txt'
        else:
            raise KeyError(key)

        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def source_path(self, subject=None):
        if subject is None:
            subject = self.state['subject']
        path = self.root / 'inputs' / f'{subject}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class SourceInput(Input):
    name = 'source'
    key_fields = ('subject',)
    view_options = {'upper': False}

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def source_path(self, subject: str) -> Path:
        path = self.root / 'inputs' / f'{subject}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, ctx: Request) -> Path:
        return self.source_path(ctx.state['subject'])

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        path = self.source_path(ctx.state['subject'])
        return file_fingerprint(str(self.root), path, digest=True)

    def load(self, ctx: Request) -> str:
        value = self.source_path(ctx.state['subject']).read_text()
        if ctx.view_options['upper']:
            return value.upper()
        return value

    def load_view(self, ctx: Request, view: str) -> str:
        if view != 'echo':
            return super().load_view(ctx, view)
        return f"source:{self.load(ctx)}"


class ValueDerivative(Derivative[str]):
    name = 'value'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = 0
        self.load_calls = 0
        self.save_calls = 0

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        self.build_calls += 1
        return (self.root / 'inputs' / f"{ctx.state['subject']}.txt").read_text()

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        self.load_calls += 1
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        self.save_calls += 1
        Path(path).write_text(value)


class SummaryDerivative(Derivative[str]):
    name = 'summary'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = 0
        self.load_calls = 0

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('value'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        self.build_calls += 1
        return f"summary:{ctx.load('value')}"

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        self.load_calls += 1
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class ComparisonDerivative(Derivative[str]):
    name = 'comparison'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (
            Dependency('value', label='current'),
            Dependency('value', label='other', state={'subject': 's2'}),
        )

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return f"{ctx.load('value')} vs {ctx.load('value', subject='s2')}"

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class EphemeralDerivative(Derivative[str]):
    name = 'ephemeral'
    key_fields = ('subject',)
    cache_policy = CachePolicy.NEVER

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = 0

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        self.build_calls += 1
        return f"ephemeral-{self.build_calls}"

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class CollidingDerivative(Derivative[str]):
    name = 'colliding'
    key_fields = ('subject', 'mode')
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = {}

    def path(self, ctx: Request) -> Path:
        return self.root / 'derivatives' / 'eelbrain' / 'cache' / 'colliding' / 'shared.txt'

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject'], 'mode': ctx.state['mode']}

    def build(self, ctx: Request) -> str:
        key = (ctx.state['subject'], ctx.state['mode'])
        self.build_calls[key] = self.build_calls.get(key, 0) + 1
        return f"{ctx.state['subject']}:{ctx.state['mode']}:{self.build_calls[key]}"

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class OptionDerivative(Derivative[str]):
    name = 'optioned'
    key_fields = ('subject',)
    cache_suffix = '.txt'
    key_options = {'artifact': 0}
    view_options = {'view': 0}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.calls = []

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def build(self, ctx: Request) -> str:
        self.calls.append(('build', ctx.options['artifact'], ctx.view_options['view']))
        return f"artifact:{ctx.options['artifact']}"

    def load(self, ctx: Request, path: str) -> str:
        self.calls.append(('load', ctx.options['artifact'], ctx.view_options['view']))
        return Path(path).read_text()

    def apply_view_options(self, ctx: Request, value: str) -> str:
        self.calls.append(('view', ctx.options['artifact'], ctx.view_options['view']))
        return f"{value}|view:{ctx.view_options['view']}"

    def artifact_metadata(self, ctx: Request, value: str) -> dict[str, object]:
        return {'value': value}

    def load_view(self, ctx: Request, view: str) -> str:
        if view != 'echo':
            return super().load_view(ctx, view)
        value = ctx.load_artifact()
        self.calls.append(('named-view', ctx.options['artifact'], ctx.view_options['view']))
        return f"{value}|meta:{ctx.artifact_metadata['value']}"

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class SpecOptionDerivative(Derivative[str]):
    name = 'spec-optioned'
    key_fields = ('subject',)
    cache_suffix = '.txt'
    key_options = {
        'flag': OptionSpec(False, type=bool),
        'mode': OptionSpec(None, literal=('a', 'b', True)),
        'label': OptionSpec('', normalize=lambda value: value.lower()),
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = 0

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def build(self, ctx: Request) -> str:
        self.build_calls += 1
        return f"flag:{ctx.options['flag']}|mode:{ctx.options['mode']}|label:{ctx.options['label']}"

    def load(self, ctx: Request, path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class NarrowingDerivative(Derivative[str]):
    name = 'narrowing'
    key_fields = ('subject', 'mode')
    cache_suffix = '.txt'
    key_options = {'alpha': 0, 'beta': 0}

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def override_key_options(self, ctx: Request) -> tuple[str, ...] | None:
        if ctx.state['mode'] == 'narrow':
            return ('alpha',)
        return None

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def build(self, ctx: Request) -> str:
        return f"alpha:{ctx.options['alpha']}"

    def load(self, ctx: Request, path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class SelfViewDerivative(Derivative[str]):
    name = 'self-view'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path, declare_source: bool):
        self.root = Path(root)
        self.declare_source = declare_source

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if self.declare_source:
            return (Dependency('source'),)
        return ()

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return ctx.load(view='shell')

    def load_view(self, ctx: Request, view: str) -> str:
        if view != 'shell':
            return super().load_view(ctx, view)
        return f"shell:{ctx.load('source')}"

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class ApplyViewDependencyDerivative(Derivative[str]):
    name = 'apply-view-dependency'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path, declare_source: bool):
        self.root = Path(root)
        self.declare_source = declare_source

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if self.declare_source:
            return (Dependency('source', options={'upper': True}),)
        return ()

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return 'artifact'

    def apply_view_options(self, ctx: Request, value: str) -> str:
        return f"{value}:{ctx.load('source')}"

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class FingerprintOverrideDerivative(Derivative[str]):
    name = 'fingerprint-override'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path, dependencies: tuple[Dependency, ...], load_name: str):
        self.root = Path(root)
        self._dependencies = dependencies
        self.load_name = load_name

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return self._dependencies

    def dependency_fingerprint_override(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> dict[str, object] | None:
        return {'value': ctx.load(self.load_name)}

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return 'artifact'

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class ProtectedDerivative(Derivative[str]):
    name = 'protected'
    key_fields = ('subject',)

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, ctx: Request) -> str:
        return str(self.root / 'derivatives' / 'mne' / ctx.state['subject'] / 'protected.txt')

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return ctx.load('source')

    def load(
            self,
            ctx: Request,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: Request,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


# Instrumented doubles for the quick-fingerprint shortcut: ``CountingQuickInput``
# exposes a cheap quick fingerprint (file mtime) and counts how often its
# expensive full fingerprint is computed, so a test can assert the full one is
# skipped while the quick fingerprint still matches.
class CountingQuickInput(Input):
    """Input with a cheap quick fingerprint and an instrumented full fingerprint."""
    name = 'counting'
    key_fields = ('subject',)

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.full_calls = 0

    def path(self, ctx: Request) -> Path:
        path = self.root / 'inputs' / f"{ctx.state['subject']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        self.full_calls += 1
        return {'content': self.path(ctx).read_text()}

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict[str, object]:
        return {'mtime': self.path(ctx).stat().st_mtime_ns}

    def load(self, ctx: Request) -> str:
        return self.path(ctx).read_text()


class CountingValueDerivative(Derivative[str]):
    name = 'counting-value'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('counting'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'subject': ctx.state['subject']}

    def build(self, ctx: Request) -> str:
        return ctx.load('counting')

    def load(self, ctx: Request, path: str) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: str, value: str) -> None:
        Path(path).write_text(value)


# ---------------------------------------------------------------------------
# Registry builders
# ---------------------------------------------------------------------------

def make_empty_registry():
    root = TempDir()
    return root, DerivativeRegistry(root, LOG)


def make_source_registry():
    root, registry = make_empty_registry()
    source = SourceInput(root)
    source.source_path('s1').write_text('alpha')
    source.source_path('s2').write_text('beta')
    registry.register(source)
    return root, registry, source


def make_registry():
    root, registry, source = make_source_registry()
    pipeline = FakePipeline(root)
    value = ValueDerivative(root)
    summary = SummaryDerivative(root)
    comparison = ComparisonDerivative()
    ephemeral = EphemeralDerivative(root)
    protected = ProtectedDerivative(root)
    registry.register(value)
    registry.register(summary)
    registry.register(comparison)
    registry.register(ephemeral)
    registry.register(protected)
    return pipeline, registry, source, value, summary, comparison, ephemeral, protected, root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_manifest_roundtrip_ignores_unknown_fields():
    manifest = ArtifactManifest.from_dict({
        'schema_version': 1,
        'derivative': 'value',
        'derivative_version': 2,
        'key': {'subject': 's1'},
        'fingerprint': {'subject': 's1'},
        'dependencies': {'source': {'kind': 'input'}},
        'cache_policy': 'required',
        'software': {'mne': '1.0'},
        'serializer': 'obsolete',  # hypothetical future unknown field
    })
    assert manifest.derivative == 'value'


def test_registry_load_caches_derivative_and_writes_manifest():
    pipeline, registry, _, value, _, _, _, _, _root = make_registry()

    first = registry.resolve('value', state=DEFAULT_STATE).load()
    second = registry.resolve('value', state=DEFAULT_STATE).load()

    assert first == second == 'alpha'
    assert value.build_calls == 1


def test_restricted_state_get_is_checked():
    root, registry, _ = make_source_registry()

    class SneakyDerivative(ValueDerivative):
        name = 'sneaky'

        def build(self, ctx: Request) -> str:
            return str(ctx.state.get('mode'))

    class AbsentFieldDerivative(ValueDerivative):
        name = 'absent-field'

        def build(self, ctx: Request) -> str:
            return str(ctx.state.get('no-such-field', 'fallback'))

    registry.register(SneakyDerivative(root))
    registry.register(AbsentFieldDerivative(root))

    # .get() of an undeclared state field is checked like item access
    with pytest.raises(RuntimeError, match="not declared in this node's key_fields"):
        registry.resolve('sneaky', state=DEFAULT_STATE).load()

    # .get() of a field that is not part of state at all stays allowed
    assert registry.resolve('absent-field', state=DEFAULT_STATE).load() == 'fallback'


def test_dependency_key_change_invalidates_parent():
    """A dependency pointing to a different artifact must invalidate the parent.

    The dependency's fingerprint is configuration-only (the default, empty),
    so only its cache key distinguishes the two artifacts.
    """
    root, registry = make_empty_registry()

    class PlainDerivative(Derivative[str]):
        name = 'plain'
        key_fields = ('subject',)
        cache_suffix = '.txt'

        def build(self, ctx: Request) -> str:
            return ctx.state['subject']

        def load(self, ctx: Request, path: Path) -> str:
            return path.read_text()

        def save(self, ctx: Request, path: Path, value: str) -> None:
            path.write_text(value)

    class PassthroughDerivative(PlainDerivative):
        # key_fields intentionally empty: identity comes from the dependency key
        name = 'passthrough'
        key_fields = ()

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('plain', label='dep', state={'subject': ctx.state['pick']}),)

        def build(self, ctx: Request) -> str:
            return ctx.load('dep')

    registry.register(PlainDerivative())
    registry.register(PassthroughDerivative())

    assert registry.resolve('passthrough', state={'pick': 's1'}).load() == 's1'
    assert registry.resolve('passthrough', state={'pick': 's1'}).is_valid()
    assert not registry.resolve('passthrough', state={'pick': 's2'}).is_valid()
    assert registry.resolve('passthrough', state={'pick': 's2'}).load() == 's2'


def test_duplicate_dependency_labels_fail_before_build():
    root, registry, _ = make_source_registry()

    class DuplicateDepDerivative(ValueDerivative):
        name = 'duplicate-dep'

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('source'), Dependency('source'))

    derivative = DuplicateDepDerivative(root)
    registry.register(derivative)

    with pytest.raises(RuntimeError, match="Duplicate dependency label"):
        registry.resolve('duplicate-dep', state=DEFAULT_STATE).load()
    assert derivative.build_calls == 0


def test_unreadable_manifest_triggers_rebuild():
    pipeline, registry, _, value, _, _, _, _, _root = make_registry()
    handle = registry.resolve('value', state=DEFAULT_STATE)
    handle.load()

    # Corrupt JSON, e.g. from an interrupted write
    handle.manifest_path.write_text('{"derivative": "val')
    assert registry.read_manifest(handle.manifest_path) is None
    assert not handle.is_valid()
    assert registry.resolve('value', state=DEFAULT_STATE).load() == 'alpha'
    assert value.build_calls == 2

    # Structurally incompatible manifest (missing required fields)
    handle.manifest_path.write_text('{"schema_version": 1}')
    assert registry.read_manifest(handle.manifest_path) is None
    assert registry.resolve('value', state=DEFAULT_STATE).load() == 'alpha'
    assert value.build_calls == 3

    # Valid again after the rebuild rewrote the manifest
    assert registry.resolve('value', state=DEFAULT_STATE).is_valid()


def test_registry_logs_cache_events(caplog):
    _, registry, _, value, _, _, _, _, _root = make_registry()

    with caplog.at_level(logging.DEBUG, logger='eelbrain.test.derivative_cache'):
        registry.resolve('value', state=DEFAULT_STATE).load()
        registry.resolve('value', state=DEFAULT_STATE).load()

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith('Build value: value') for message in messages)
    assert any(message.startswith('Load cached value: value') for message in messages)
    assert value.save_calls == 1
    assert value.load_calls == 2

    # structured fields are attached for machine-readable consumption
    events = [event for event in (getattr(record, 'cache_event', None) for record in caplog.records) if event]
    assert any(event == {'event': 'build', 'derivative': 'value'} for event in events)
    assert any(event == {'event': 'cached', 'derivative': 'value'} for event in events)

    handle = registry.resolve('value', state=DEFAULT_STATE)
    cache_path = handle.artifact_path
    manifest_path = handle.manifest_path
    assert cache_path.exists()
    assert manifest_path.exists()
    assert cache_path.is_relative_to(registry.cache_dir / 'value')
    assert cache_path.suffix == '.txt'
    assert '_key-' in cache_path.name

    manifest = json.loads(manifest_path.read_text())
    assert manifest['derivative'] == 'value'
    assert manifest['key'] == {'subject': 's1'}
    assert manifest['dependencies']['source']['kind'] == 'input'


def test_recompute_logs_invalidation_reason(caplog):
    _, registry, source, value, _, _, _, _, _root = make_registry()

    registry.resolve('value', state=DEFAULT_STATE).load()  # initial build
    source.source_path('s1').write_text('changed')  # invalidate the source dependency

    with caplog.at_level(logging.DEBUG, logger='eelbrain.test.derivative_cache'):
        assert registry.resolve('value', state=DEFAULT_STATE).load() == 'changed'

    events = [getattr(record, 'cache_event', None) for record in caplog.records]
    recompute = next(event for event in events if event and event['event'] == 'recompute')
    assert recompute['derivative'] == 'value'
    assert recompute['category'] == 'dependencies'
    assert recompute['old'] != recompute['new']
    assert recompute['field']

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith('Recompute value (dependency changed') for message in messages)


def _fake_manifest(**overrides) -> ArtifactManifest:
    fields = dict(
        schema_version=1, derivative='value', derivative_version=1,
        key={'subject': 's1'}, fingerprint={'a': 1}, dependencies={},
        cache_policy='required', software={},
    )
    fields.update(overrides)
    return ArtifactManifest(**fields)


def test_compare_manifests_reports_first_difference():
    stored = _fake_manifest(fingerprint={'a': 1, 'b': 2})
    current = _fake_manifest(fingerprint={'a': 1, 'b': 3})

    invalidation = compare_manifests(stored, current)
    assert invalidation.category == 'fingerprint'
    assert invalidation.field() == 'b'
    assert (invalidation.old, invalidation.new) == (2, 3)
    assert invalidation.as_dict() == {'category': 'fingerprint', 'field': 'b', 'old': 2, 'new': 3}
    assert invalidation.message() == "fingerprint changed (b: 2 -> 3)"


def test_compare_manifests_categories():
    assert compare_manifests(_fake_manifest(), _fake_manifest()) is None
    assert compare_manifests(None, _fake_manifest()) == CacheInvalidation('missing_manifest')
    assert compare_manifests(_fake_manifest(key={'subject': 's1'}), _fake_manifest(key={'subject': 's2'})).category == 'key'
    assert compare_manifests(_fake_manifest(derivative_version=1), _fake_manifest(derivative_version=2)).category == 'version'


def test_structured_formatter_appends_tab_columns():
    formatter = StructuredFormatter('%(message)s')
    record = logging.LogRecord('x', logging.DEBUG, __file__, 1, 'hello', None, None)
    assert formatter.format(record) == 'hello'

    # full recompute event maps to all columns in fixed order
    record.cache_event = {'event': 'recompute', 'derivative': 'value', 'category': 'fingerprint', 'field': 'b', 'old': 2, 'new': 3}
    assert formatter.format(record).split('\t') == ['hello', 'recompute', 'value', 'fingerprint', 'b', '2', '3']

    # build/cached events leave the reason columns empty but keep the shared layout
    record.cache_event = {'event': 'build', 'derivative': 'value'}
    assert formatter.format(record).split('\t') == ['hello', 'build', 'value', '', '', '', '']


def test_dependency_change_invalidates_downstream_derivatives():
    pipeline, registry, _, value, summary, _, _, _, _root = make_registry()

    assert registry.resolve('summary', state=DEFAULT_STATE).load() == 'summary:alpha'
    assert value.build_calls == 1
    assert summary.build_calls == 1

    pipeline.source_path().write_text('changed')

    assert registry.resolve('summary', state=DEFAULT_STATE).load() == 'summary:changed'
    assert value.build_calls == 2
    assert summary.build_calls == 2


def test_non_key_state_does_not_invalidate_cache():
    _, registry, _, value, _, _, _, _, _root = make_registry()

    assert registry.resolve('value', state={'subject': 's1', 'mode': 'a'}).load() == 'alpha'
    assert registry.resolve('value', state={'subject': 's1', 'mode': 'b'}).load() == 'alpha'
    assert value.build_calls == 1


def test_generic_cache_path_uses_node_name_and_key():
    _, registry, _, _, _, _, _, _, _root = make_registry()

    a = registry.resolve('value', state={'subject': 's1', 'mode': 'a'}).artifact_path
    b = registry.resolve('value', state={'subject': 's1', 'mode': 'b'}).artifact_path
    c = registry.resolve('value', state={'subject': 's2', 'mode': 'a'}).artifact_path

    assert a == b
    assert a != c
    assert a.is_relative_to(registry.cache_dir / 'value')
    assert c.is_relative_to(registry.cache_dir / 'value')


def test_cache_label_substitutes_path_unsafe_characters():
    # Key-field values can contain characters that are illegal in Windows path
    # components (e.g. the '>' in a test named 'a>v'). These are replaced with '-'
    # rather than deleted, so operands stay separated in the readable slug while
    # the hash keeps distinct keys on distinct paths.
    _, registry, _, _, _, _, _, _, _root = make_registry()

    gt = registry.resolve('value', state={'subject': 'a>v'}).artifact_path
    lt = registry.resolve('value', state={'subject': 'a<v'}).artifact_path

    assert gt.name.startswith('subject-a-v_key-')
    assert not any(c in gt.name for c in '<>:"/\\|?*')
    # '>' and '<' map to the same readable slug but remain distinct cache entries
    assert gt.name.split('_key-')[0] == lt.name.split('_key-')[0]
    assert gt != lt


def test_cache_collision_sidecar_disambiguates_artifact_paths():
    root, registry = make_empty_registry()
    derivative = CollidingDerivative(root)
    registry.register(derivative)

    state_a = {'subject': 's1', 'mode': 'a'}
    state_b = {'subject': 's1', 'mode': 'b'}

    assert registry.resolve('colliding', state=state_a).load() == 's1:a:1'
    handle_a = registry.resolve('colliding', state=state_a)
    assert handle_a.artifact_path == handle_a.base_artifact_path

    assert registry.resolve('colliding', state=state_b).load() == 's1:b:1'
    handle_b = registry.resolve('colliding', state=state_b)
    assert handle_b.base_artifact_path == handle_a.base_artifact_path
    assert handle_b.artifact_path != handle_a.artifact_path
    assert handle_b.artifact_path.name == 'shared_alt-1.txt'

    sidecar_path = Path(f"{handle_a.base_artifact_path}.disambiguation.json")
    assert sidecar_path.exists()
    mapping = json.loads(sidecar_path.read_text())
    assert len(mapping) == 1

    assert registry.resolve('colliding', state=state_a).load() == 's1:a:1'
    assert registry.resolve('colliding', state=state_b).load() == 's1:b:1'
    assert derivative.build_calls == {('s1', 'a'): 1, ('s1', 'b'): 1}


def test_unique_cache_paths_do_not_create_disambiguation_sidecar():
    _, registry, _, value, _, _, _, _, _root = make_registry()

    assert registry.resolve('value', state=DEFAULT_STATE).load() == 'alpha'
    handle = registry.resolve('value', state=DEFAULT_STATE)

    assert not Path(f"{handle.base_artifact_path}.disambiguation.json").exists()
    assert value.build_calls == 1


def test_dependency_tree_formats_ascii_dependencies():
    _, registry, _, _, _, _, _, _, _root = make_registry()

    tree = registry.dependency_tree('comparison', state=DEFAULT_STATE)

    assert "comparison [derivative] {subject='s1'}" in tree
    assert "current -> value [derivative] {subject='s1'}" in tree
    assert "other -> value [derivative] {subject='s2'} [state: subject='s2']" in tree
    assert 'source [input]' in tree
    assert '├──' in tree
    assert '└──' in tree


def test_dependency_tree_respects_max_line_length():
    _, registry, _, _, _, _, _, _, _root = make_registry()

    tree = registry.dependency_tree('comparison', state=DEFAULT_STATE, max_line_length=44)
    lines = tree.splitlines()

    assert len(lines) > 4
    assert all(len(line) <= 44 for line in lines)
    assert "other -> value [derivative]" in tree
    assert "{subject='s2'} [state: subject='s2']" in tree


def test_uncached_derivative_rebuilds_every_time():
    _, registry, _, _, _, _, ephemeral, _, _root = make_registry()

    first = registry.resolve('ephemeral', state=DEFAULT_STATE).load()
    second = registry.resolve('ephemeral', state=DEFAULT_STATE).load()
    handle = registry.resolve('ephemeral', state=DEFAULT_STATE)

    assert first == 'ephemeral-1'
    assert second == 'ephemeral-2'
    assert ephemeral.build_calls == 2
    assert not handle.is_valid()
    with pytest.raises(TypeError, match="uncached derivative 'ephemeral'"):
        handle.artifact_path
    with pytest.raises(TypeError, match="uncached derivative 'ephemeral'"):
        handle.manifest_path


def test_registry_resolve_returns_request_for_input_and_derivative():
    _, registry, _, _, _, _, _, _, _root = make_registry()

    handle = registry.resolve('source', state=DEFAULT_STATE)
    assert isinstance(handle, Request)
    assert handle.describe_dependency()['name'] == 'source'
    assert handle.describe_dependency()['kind'] == 'input'
    with pytest.raises(TypeError, match="input 'source'"):
        _ = handle.artifact_path

    value_handle = registry.resolve('value', state=DEFAULT_STATE)
    assert isinstance(value_handle, Request)
    assert value_handle.describe_dependency()['name'] == 'value'
    assert value_handle.describe_dependency()['kind'] == 'derivative'
    # the manifest path is recorded relative to the cache dir (portable across a moved root)
    assert value_handle.describe_dependency()['manifest'] == value_handle.manifest_path.relative_to(registry.cache_dir).as_posix()

    assert registry.resolve('source', state=DEFAULT_STATE).load() == 'alpha'
    assert registry.resolve('source', state=DEFAULT_STATE, options={'upper': True}).load() == 'ALPHA'


def test_stale_external_artifact_is_protected():
    pipeline, registry, _, _, _, _, _, _, _root = make_registry()

    assert registry.resolve('protected', state=DEFAULT_STATE).load() == 'alpha'
    protected_path = Path(pipeline.get('protected-file'))
    manifest_path = Path(registry.manifest_path(protected_path, 'protected'))
    assert protected_path.exists()
    assert manifest_path.exists()
    # The manifest mirrors the artifact under the node directory, dropping the
    # artifact's top-level namespace ('mne').
    entity = Path(*protected_path.relative_to(pipeline.get('deriv-dir')).parts[1:])
    assert manifest_path == Path(f"{Path(pipeline.get('cache-dir')) / 'protected' / entity}.manifest.json")

    pipeline.source_path().write_text('changed')

    try:
        registry.resolve('protected', state=DEFAULT_STATE).load()
    except ProtectedArtifactError as error:
        assert error.derivative == 'protected'
        assert error.path == str(protected_path)
    else:
        raise AssertionError("Expected ProtectedArtifactError")

    assert protected_path.read_text() == 'alpha'
    assert registry.resolve('protected', state=DEFAULT_STATE, controls={ALLOW_PROTECTED_OVERWRITE}).load() == 'changed'
    assert protected_path.read_text() == 'changed'


def test_protected_artifact_requires_derivative_owned_reindexing():
    pipeline, registry, _, _, _, _, _, _, _root = make_registry()

    assert registry.resolve('protected', state=DEFAULT_STATE).load() == 'alpha'
    pipeline.source_path().write_text('changed')

    with pytest.raises(ProtectedArtifactError):
        registry.resolve('protected', state=DEFAULT_STATE, controls={'reindex_anything'}).load()


def test_runtime_code_does_not_use_private_get_node():
    """Make sure private API is not used"""
    experiment_dir = Path(__file__).resolve().parents[2]
    offenders = []
    for path in experiment_dir.glob('*.py'):
        if path.name == 'derivative_cache.py':
            continue
        if '._get_node(' in path.read_text():
            offenders.append(path.name)
    assert offenders == []


def test_request_splits_artifact_and_view_options():
    root, registry = make_empty_registry()
    derivative = OptionDerivative(root)
    registry.register(derivative)

    handle = registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 1, 'view': 2})

    assert handle.options == {'artifact': 1}
    assert handle.view_options == {'view': 2}
    # Artifact options are captured by the key, not the fingerprint.
    assert handle.key()['options'] == {'artifact': 1}
    assert 'options' not in handle.current_fingerprint()
    assert handle.options_for('optioned', artifact=4) == {'artifact': 4}
    assert handle.options_for('optioned', 'view', artifact=4) == {'view': 2, 'artifact': 4}
    with pytest.raises(TypeError, match="does not declare option"):
        handle.options_for('optioned', artifact=4, extra=5)


def test_registry_rejects_undeclared_options():
    root, registry = make_empty_registry()
    derivative = OptionDerivative(root)
    registry.register(derivative)

    with pytest.raises(TypeError, match="undeclared option"):
        registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 1, 'extra': 3})


def test_option_spec_validates_and_fills_defaults():
    root, registry = make_empty_registry()
    registry.register(SpecOptionDerivative(root))

    # defaults are exempt from validation ('mode' default None is not in literal)
    handle = registry.resolve('spec-optioned', state=DEFAULT_STATE)
    assert handle.options == {'flag': False, 'mode': None, 'label': ''}

    # type=bool coerce 1 -> True
    request = registry.resolve('spec-optioned', state=DEFAULT_STATE, options={'flag': 1})
    assert request.options['flag'] is True
    # literal matching is type-strict: 1 == True, but does not match literal True
    with pytest.raises(ValueError, match="must be one of"):
        registry.resolve('spec-optioned', state=DEFAULT_STATE, options={'mode': 1})
    request = registry.resolve('spec-optioned', state=DEFAULT_STATE, options={'mode': True})
    assert request.options['mode'] is True
    with pytest.raises(ValueError, match="must be one of"):
        registry.resolve('spec-optioned', state=DEFAULT_STATE, options={'mode': 'c'})


def test_option_spec_normalize_canonicalizes_cache_key():
    root, registry = make_empty_registry()
    derivative = SpecOptionDerivative(root)
    registry.register(derivative)

    first = registry.resolve('spec-optioned', state=DEFAULT_STATE, options={'label': 'ABC'})
    second = registry.resolve('spec-optioned', state=DEFAULT_STATE, options={'label': 'abc'})

    # the normalized value replaces the option for the whole request
    assert first.options['label'] == 'abc'
    # equivalent spellings share one cache key and one artifact
    assert first.key() == second.key()
    assert first.load() == second.load() == 'flag:False|mode:None|label:abc'
    assert derivative.build_calls == 1


def test_override_key_options_narrows_key():
    root, registry = make_empty_registry()
    registry.register(NarrowingDerivative(root))

    narrow = registry.resolve('narrowing', state={'subject': 's1', 'mode': 'narrow'}, options={'alpha': 1})
    assert narrow.key()['options'] == {'alpha': 1}
    wide = registry.resolve('narrowing', state={'subject': 's1', 'mode': 'wide'}, options={'beta': 2})
    assert wide.key()['options'] == {'alpha': 0, 'beta': 2}
    # a caller-set option that the node drops for this request triggers a warning
    with pytest.warns(UserWarning, match="no effect"):
        registry.resolve('narrowing', state={'subject': 's1', 'mode': 'narrow'}, options={'beta': 2})


def test_request_applies_view_options_after_build_and_load():
    root, registry = make_empty_registry()
    derivative = OptionDerivative(root)
    registry.register(derivative)

    first = registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 1, 'view': 2}).load()
    second = registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 1, 'view': 3}).load()

    assert first == 'artifact:1|view:2'
    assert second == 'artifact:1|view:3'
    assert derivative.calls == [
        ('build', 1, 2),
        ('load', 1, 2),
        ('view', 1, 2),
        ('load', 1, 3),
        ('view', 1, 3),
    ]


def test_request_loads_named_view_and_exposes_artifact_metadata():
    root, registry = make_empty_registry()
    derivative = OptionDerivative(root)
    registry.register(derivative)

    value = registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 2, 'view': 7}).load(view='echo')
    handle = registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 2, 'view': 7})
    manifest = json.loads(handle.manifest_path.read_text())

    assert value == 'artifact:2|meta:artifact:2'
    assert manifest['artifact_metadata'] == {'value': 'artifact:2'}
    assert derivative.calls == [
        ('build', 2, 7),
        ('load', 2, 7),
        ('named-view', 2, 7),
    ]


def test_load_view_obeys_declared_dependencies_during_build():
    root, registry, _ = make_source_registry()
    registry.register(SelfViewDerivative(root, True))

    value = registry.resolve('self-view', state=DEFAULT_STATE).load()

    assert value == 'shell:alpha'


def test_load_view_rejects_undeclared_dependencies_during_build():
    root, registry, _ = make_source_registry()
    registry.register(SelfViewDerivative(root, False))

    with pytest.raises(RuntimeError, match="not a declared dependency"):
        registry.resolve('self-view', state=DEFAULT_STATE).load()


def test_apply_view_options_obeys_declared_dependencies():
    root, registry, _ = make_source_registry()
    registry.register(ApplyViewDependencyDerivative(root, True))

    first = registry.resolve('apply-view-dependency', state=DEFAULT_STATE).load()
    second = registry.resolve('apply-view-dependency', state=DEFAULT_STATE).load()

    assert first == 'artifact:ALPHA'
    assert second == 'artifact:ALPHA'


def test_apply_view_options_rejects_undeclared_dependencies():
    root, registry, _ = make_source_registry()
    registry.register(ApplyViewDependencyDerivative(root, False))

    with pytest.raises(RuntimeError, match="not a declared dependency"):
        registry.resolve('apply-view-dependency', state=DEFAULT_STATE).load()


def test_dependency_fingerprint_override_obeys_declared_dependencies():
    root, registry, _ = make_source_registry()
    registry.register(FingerprintOverrideDerivative(root, (Dependency('source'),), 'source'))

    handle = registry.resolve('fingerprint-override', state=DEFAULT_STATE)
    value = handle.load()
    manifest = json.loads(handle.manifest_path.read_text())

    assert value == 'artifact'
    assert manifest['dependencies']['source']['fingerprint'] == {'value': 'alpha'}


def test_dependency_fingerprint_override_rejects_undeclared_dependencies():
    pipeline, registry, *_, _root = make_registry()
    root = pipeline.root
    registry.register(FingerprintOverrideDerivative(root, (Dependency('value'),), 'source'))

    with pytest.raises(RuntimeError, match="not a declared dependency"):
        registry.resolve('fingerprint-override', state=DEFAULT_STATE).load()


def test_request_loads_named_view_from_input():
    _, registry, _, _, _, _, _, _, _root = make_registry()

    value = registry.resolve('source', state=DEFAULT_STATE, options={'upper': True}).load(view='echo')

    assert value == 'source:ALPHA'


# ---------------------------------------------------------------------------
# Quick-fingerprint cache validity
#
# A matching quick fingerprint should let `_check_valid` reuse the stored
# fingerprint and skip the expensive full fingerprint (CountingQuickInput above).
# ---------------------------------------------------------------------------

def test_quick_fingerprint_skips_full_fingerprint_when_unchanged():
    import os

    root, registry = make_empty_registry()
    source = CountingQuickInput(root)
    registry.register(source)
    registry.register(CountingValueDerivative(root))
    source.path(registry.resolve('counting', state={'subject': 's1'})).write_text('hello')

    handle = registry.resolve('counting-value', state={'subject': 's1'})
    assert handle.load() == 'hello'

    # Re-validation with an unchanged quick fingerprint must not recompute the full one.
    source.full_calls = 0
    assert handle.is_valid()
    assert source.full_calls == 0

    # A changed quick fingerprint falls back to (and recomputes) the full fingerprint.
    os.utime(source.path(registry.resolve('counting', state={'subject': 's1'})), None)
    source.full_calls = 0
    assert handle.is_valid()  # spurious quick change: full fingerprint still matches
    assert source.full_calls == 1

    # The successful validation refreshed the manifest, restoring the quick path.
    source.full_calls = 0
    assert handle.is_valid()
    assert source.full_calls == 0

    source.path(registry.resolve('counting', state={'subject': 's1'})).write_text('changed')
    source.full_calls = 0
    assert not handle.is_valid()
    assert source.full_calls == 1


# ---------------------------------------------------------------------------
# Input key_fields + edge key-coverage validation
# ---------------------------------------------------------------------------

class _Leaf(Derivative[str]):
    """Minimal cached derivative for key-coverage tests."""
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def build(self, ctx: Request) -> str:
        return 'x'

    def load(self, ctx: Request, path: Path) -> str:
        return path.read_text()

    def save(self, ctx: Request, path: Path, value: str) -> None:
        path.write_text(value)


def test_input_requires_key_fields():
    root, registry = make_empty_registry()

    class NoFieldsInput(Input):
        name = 'no-fields'

        def path(self, ctx: Request) -> Path:
            return Path(self.root) / 'x'

    class OptOutInput(Input):
        name = 'opt-out'
        key_fields = ()

        def __init__(self, root):
            self.root = root

        def path(self, ctx: Request) -> Path:
            return Path(self.root) / 'x'

    with pytest.raises(TypeError, match='must declare key_fields'):
        registry.register(NoFieldsInput())
    # an explicit empty tuple opts out and is accepted
    registry.register(OptOutInput(root))


def test_input_read_restriction():
    root, registry = make_empty_registry()

    class RestrictedInput(Input):
        name = 'restricted'
        key_fields = ('subject',)

        def __init__(self, root):
            self.root = Path(root)

        def path(self, ctx: Request) -> Path:
            return self.root / f"{ctx.state['subject']}.txt"

        def fingerprint(self, ctx: Request) -> dict[str, object]:
            # reads an undeclared field in a cache-affecting method
            return {'mode': ctx.state['mode']}

        def load(self, ctx: Request):
            # load() is not restricted: reading an undeclared field is allowed
            return ctx.state['mode']

    node = RestrictedInput(root)
    registry.register(node)
    node.path(registry.resolve('restricted', state={'subject': 's1'})).write_text('data')

    with pytest.raises(RuntimeError, match="not declared in this node's key_fields"):
        registry.resolve('restricted', state={'subject': 's1', 'mode': 'a'}).current_fingerprint()
    # load() runs outside the check context, so the same read is allowed
    assert registry.resolve('restricted', state={'subject': 's1', 'mode': 'a'}).load() == 'a'


def _register_child_parent(registry, root, parent_cls):
    class Child(_Leaf):
        name = 'child'
        key_fields = ('subject', 'mode')
    registry.register(Child(root))
    registry.register(parent_cls(root))


def test_edge_key_coverage_violation():
    root, registry = make_empty_registry()

    class Parent(_Leaf):
        name = 'parent'
        key_fields = ('subject',)

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('child'),)

        def build(self, ctx: Request) -> str:
            return ctx.load('child')

    _register_child_parent(registry, root, Parent)
    with pytest.raises(RuntimeError, match=r"depends on state field\(s\).*'mode'"):
        registry.resolve('parent', state={'subject': 's1', 'mode': 'a'}).load()


def test_edge_key_coverage_pinned_on_edge():
    root, registry = make_empty_registry()

    class Parent(_Leaf):
        name = 'parent'
        key_fields = ('subject',)

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('child', state={'mode': 'fixed'}),)

        def build(self, ctx: Request) -> str:
            return ctx.load('child')

    _register_child_parent(registry, root, Parent)
    # 'mode' pinned on the edge → covered even though the parent does not key on it
    assert registry.resolve('parent', state={'subject': 's1', 'mode': 'a'}).load() == 'x'


def test_edge_key_coverage_parent_keys_field():
    root, registry = make_empty_registry()

    class Parent(_Leaf):
        name = 'parent'
        key_fields = ('subject', 'mode')

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('child'),)

        def build(self, ctx: Request) -> str:
            return ctx.load('child')

    _register_child_parent(registry, root, Parent)
    assert registry.resolve('parent', state={'subject': 's1', 'mode': 'a'}).load() == 'x'


def test_edge_key_coverage_dynamic_fields():
    root, registry = make_empty_registry()

    class Parent(_Leaf):
        name = 'parent'
        key_fields = ('subject',)

        def override_key_fields(self, ctx: Request):
            return ('subject', 'mode')

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('child'),)

        def build(self, ctx: Request) -> str:
            return ctx.load('child')

    _register_child_parent(registry, root, Parent)
    # override_key_fields makes 'mode' part of the parent's coverage
    assert registry.resolve('parent', state={'subject': 's1', 'mode': 'a'}).load() == 'x'


def test_edge_key_coverage_enforces_uncached_child():
    root, registry = make_empty_registry()

    class UncachedChild(UncachedDerivative[str]):
        name = 'child'
        key_fields = ('subject', 'mode')

        def build(self, ctx: Request) -> str:
            return 'x'

    class Parent(_Leaf):
        name = 'parent'
        key_fields = ('subject',)

        def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
            return (Dependency('child'),)

        def build(self, ctx: Request) -> str:
            return ctx.load('child')

    registry.register(UncachedChild())
    registry.register(Parent(root))
    # strict rule: an uncached child's declared key fields are enforced too
    with pytest.raises(RuntimeError, match=r"depends on state field\(s\).*'mode'"):
        registry.resolve('parent', state={'subject': 's1', 'mode': 'a'}).load()


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------

class ConfiguredDerivative(Derivative[str]):
    """Derivative whose fingerprint depends on a mutable configuration value."""
    name = 'configured'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path, config: str = 'a'):
        self.root = Path(root)
        self.config = config

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'config': self.config}

    def build(self, ctx: Request) -> str:
        return f"configured:{self.config}"

    def load(self, ctx: Request, path: Path) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: Path, value: str) -> None:
        Path(path).write_text(value)


class DownstreamDerivative(Derivative[str]):
    name = 'downstream'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('configured'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def build(self, ctx: Request) -> str:
        return f"downstream:{ctx.load('configured')}"

    def load(self, ctx: Request, path: Path) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: Path, value: str) -> None:
        Path(path).write_text(value)


class DirArtifactDerivative(Derivative[str]):
    """Derivative whose artifact is a directory containing several files."""
    name = 'dir-artifact'
    key_fields = ('subject',)
    cache_suffix = '.parts'

    def __init__(self, root: str | Path, config: str = 'a'):
        self.root = Path(root)
        self.config = config

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'config': self.config}

    def build(self, ctx: Request) -> str:
        return 'xy'

    def load(self, ctx: Request, path: Path) -> str:
        return ''.join((Path(path) / f'part-{i}.txt').read_text() for i in range(2))

    def save(self, ctx: Request, path: Path, value: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, part in enumerate(value):
            (path / f'part-{i}.txt').write_text(part)


class FakeVersionedInput(VersionedInput[str]):
    name = 'versioned'
    key_fields = ()  # single tracked source

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _source_path(self) -> Path:
        path = self.root / 'inputs' / 'versioned.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, ctx: Request) -> Path:
        return self._source_path()

    def _reference_stem(self, ctx: Request) -> str:
        return 'ref'

    def _source_fingerprint(self, ctx: Request) -> dict[str, object]:
        return file_fingerprint(str(self.root), self._source_path())

    def _current_data(self, ctx: Request) -> str:
        return self._source_path().read_text()

    def _data_equal(self, ctx: Request, stored: str, current: str) -> bool:
        return stored == current

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'version': self.reference_version(ctx)}

    def load(self, ctx: Request) -> str:
        return self._source_path().read_text()


class VersionedConsumerDerivative(Derivative[str]):
    name = 'versioned-consumer'
    key_fields = ()
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        return (Dependency('versioned'),)

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {}

    def build(self, ctx: Request) -> str:
        return ctx.load('versioned')

    def load(self, ctx: Request, path: Path) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: Path, value: str) -> None:
        Path(path).write_text(value)


class RichSpec:
    """Option value with a rich in-memory form and a simple canonical form."""

    def __init__(self, label: str):
        self.label = label

    def __eq__(self, other):
        return isinstance(other, RichSpec) and other.label == self.label

    def _cache_form_(self) -> dict:
        return {'label': self.label}


def _normalize_rich_spec(value):
    if isinstance(value, RichSpec):
        return value
    if isinstance(value, dict):
        return RichSpec(value['label'])
    return RichSpec(value)


class RichOptionDerivative(Derivative[str]):
    name = 'rich-option'
    key_fields = ('subject',)
    cache_suffix = '.txt'
    key_options = {'spec': OptionSpec(None, normalize=_normalize_rich_spec)}

    def __init__(self, root: str | Path, config: str = 'a'):
        self.root = Path(root)
        self.config = config

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return {'config': self.config, 'label': ctx.options['spec'].label}

    def build(self, ctx: Request) -> str:
        return ctx.options['spec'].label

    def load(self, ctx: Request, path: Path) -> str:
        return Path(path).read_text()

    def save(self, ctx: Request, path: Path, value: str) -> None:
        Path(path).write_text(value)


def make_gc_registry():
    root, registry, _ = make_source_registry()
    configured = ConfiguredDerivative(root)
    downstream = DownstreamDerivative(root)
    registry.register(configured)
    registry.register(downstream)
    return registry, configured, downstream, root


def _strip_resolve_fields(manifest_path: Path) -> None:
    """Rewrite a manifest as if it predated offline revalidation."""
    data = json.loads(manifest_path.read_text())
    del data['resolve_state']
    del data['resolve_options']
    manifest_path.write_text(json.dumps(data))


def _single_entry(report, category: GCCategory):
    entries = [entry for entry in report.entries if entry.category is category]
    assert len(entries) == 1, f"{category}: {report.entries}"
    return entries[0]


def test_manifest_resolve_fields_roundtrip():
    base = {
        'schema_version': 2,
        'derivative': 'value',
        'derivative_version': 1,
        'key': {'subject': 's1'},
        'fingerprint': {},
        'dependencies': {},
        'cache_policy': 'required',
        'software': {},
    }
    old = ArtifactManifest.from_dict(base)
    assert old.resolve_state is None
    assert old.resolve_options is None
    new = ArtifactManifest.from_dict({**base, 'resolve_state': {'subject': 's1'}, 'resolve_options': {}})
    assert new.resolve_state == {'subject': 's1'}
    assert ArtifactManifest.from_dict(new.to_dict()).resolve_state == {'subject': 's1'}
    assert compare_manifests(old, new) is None  # resolve fields never invalidate


def test_gc_clean_cache_is_empty():
    registry, configured, downstream, root = make_gc_registry()
    registry.resolve('downstream', state=DEFAULT_STATE).load()
    report = registry.scan_cache()
    assert report.entries == []
    assert report.errors == []
    assert report.scanned_manifests == 2
    # collect on a clean cache is a no-op
    registry.collect(report)
    assert registry.resolve('downstream', state=DEFAULT_STATE).is_valid()


def test_gc_dead_node_dir():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    ghost_dir = registry.cache_dir / 'ghost'
    (ghost_dir / 'sub-s1').mkdir(parents=True)
    (ghost_dir / 'sub-s1' / 'artifact.txt').write_text('x')
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.DEAD_NODE_DIR)
    assert entry.path == ghost_dir
    assert entry.size > 0
    registry.collect(report)
    assert not ghost_dir.exists()
    assert ctx.artifact_path.exists()


def test_gc_schema_mismatch():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    data = json.loads(ctx.manifest_path.read_text())
    data['schema_version'] = 1
    ctx.manifest_path.write_text(json.dumps(data))
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.SCHEMA)
    assert entry.path == ctx.artifact_path
    registry.collect(report)
    assert not ctx.artifact_path.exists()
    assert not ctx.manifest_path.exists()
    assert not (registry.cache_dir / 'configured').exists()  # emptied dirs are pruned


def test_gc_derivative_version_mismatch():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    configured.version = 2
    report = registry.scan_cache()
    _single_entry(report, GCCategory.DERIVATIVE_VERSION)
    registry.collect(report)
    assert not ctx.artifact_path.exists()


def test_gc_orphan_manifest():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    ctx.artifact_path.unlink()
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.ORPHAN_MANIFEST)
    assert entry.path == ctx.manifest_path
    registry.collect(report)
    assert not ctx.manifest_path.exists()


def test_gc_external_mirror():
    pipeline, registry, source, value, summary, comparison, ephemeral, protected, root = make_registry()
    ctx = registry.resolve('protected', state=DEFAULT_STATE)
    ctx.load()
    assert not registry.is_cache_artifact(ctx.artifact_path)
    # the mirror manifest of a live external artifact is preserved silently
    report = registry.scan_cache()
    assert report.entries == []
    registry.collect(report)
    assert ctx.manifest_path.exists()
    # once the external artifact is gone, the mirror is dead weight
    Path(ctx.artifact_path).unlink()
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.ORPHAN_MIRROR)
    assert entry.path == ctx.manifest_path
    registry.collect(report)
    assert not ctx.manifest_path.exists()


def test_gc_superseded_key():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    old_artifact = ctx.artifact_path
    configured.key_fields = ()  # definition change: subject no longer keys the artifact
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.SUPERSEDED_KEY)
    assert entry.path == old_artifact
    registry.collect(report)
    assert not old_artifact.exists()
    assert registry.resolve('configured', state=DEFAULT_STATE).load() == 'configured:a'


def test_gc_added_key_field_is_unverifiable():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    configured.key_fields = ('subject', 'mode')  # old manifests lack the new field's value
    report = registry.scan_cache()
    _single_entry(report, GCCategory.UNVERIFIABLE)
    assert report.errors
    registry.collect(report)
    assert ctx.artifact_path.exists()  # unverifiable files are never deleted


def test_gc_revalidation_stale():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    configured.config = 'b'
    assert registry.scan_cache(revalidate=False).entries == []
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.REVALIDATION_STALE)
    assert 'fingerprint' in entry.reason
    registry.collect(report)
    assert not ctx.artifact_path.exists()
    assert registry.resolve('configured', state=DEFAULT_STATE).load() == 'configured:b'


def test_gc_stale_dependency_propagation():
    registry, configured, downstream, root = make_gc_registry()
    downstream_ctx = registry.resolve('downstream', state=DEFAULT_STATE)
    downstream_ctx.load()
    # make the parent unverifiable on its own (like a rich-option node with a pre-migration manifest)
    _strip_resolve_fields(downstream_ctx.manifest_path)
    configured.config = 'b'
    report = registry.scan_cache()
    _single_entry(report, GCCategory.REVALIDATION_STALE)
    entry = _single_entry(report, GCCategory.STALE_DEPENDENCY)
    assert entry.path == downstream_ctx.artifact_path
    assert 'configured' in entry.reason
    registry.collect(report)
    assert not downstream_ctx.artifact_path.exists()
    assert not downstream_ctx.manifest_path.exists()


def test_gc_stale_dependency_after_child_rebuild():
    registry, configured, downstream, root = make_gc_registry()
    downstream_ctx = registry.resolve('downstream', state=DEFAULT_STATE)
    downstream_ctx.load()
    _strip_resolve_fields(downstream_ctx.manifest_path)
    configured.config = 'b'
    registry.resolve('configured', state=DEFAULT_STATE).load()  # rebuild the child in place
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.STALE_DEPENDENCY)
    assert entry.path == downstream_ctx.artifact_path
    registry.collect(report)
    assert not downstream_ctx.artifact_path.exists()
    assert registry.resolve('configured', state=DEFAULT_STATE).is_valid()


def test_gc_directory_artifact():
    root, registry = make_empty_registry()
    node = DirArtifactDerivative(root)
    registry.register(node)
    ctx = registry.resolve('dir-artifact', state=DEFAULT_STATE)
    assert ctx.load() == 'xy'
    assert ctx.artifact_path.is_dir()
    # files inside the directory artifact are not classified individually
    assert registry.scan_cache().entries == []
    node.config = 'b'
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.REVALIDATION_STALE)
    assert entry.path == ctx.artifact_path
    registry.collect(report)
    assert not ctx.artifact_path.exists()


def test_gc_stale_versioned_reference():
    root, registry = make_empty_registry()
    node = FakeVersionedInput(root)
    registry.register(node)
    node._source_path().write_text('one')
    ctx = registry.resolve('versioned')
    version_0 = node.reference_version(ctx)
    node._source_path().write_text('two!')
    version_1 = node.reference_version(ctx)
    assert version_1['serial'] == version_0['serial'] + 1
    reference_dir = registry.cache_dir / 'versioned'
    assert (reference_dir / 'ref.0.pickle').exists()
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.STALE_REFERENCE)
    assert entry.path == reference_dir / 'ref.0.pickle'
    registry.collect(report)
    assert not (reference_dir / 'ref.0.pickle').exists()
    assert (reference_dir / 'ref.json').exists()
    assert (reference_dir / 'ref.1.pickle').exists()


def test_gc_tmp_and_unknown_files():
    registry, configured, downstream, root = make_gc_registry()
    registry.resolve('configured', state=DEFAULT_STATE).load()
    node_dir = registry.cache_dir / 'configured'
    tmp_file = node_dir / 'manifest.json.tmp'
    tmp_file.write_text('partial')
    stray_file = node_dir / 'stray.bin'
    stray_file.write_text('who knows')
    report = registry.scan_cache()
    assert _single_entry(report, GCCategory.TMP).path == tmp_file
    assert _single_entry(report, GCCategory.UNKNOWN).path == stray_file
    file_table = str(report.file_table())
    assert 'Filename' in file_table
    assert str(tmp_file.relative_to(registry.cache_dir)) in file_table
    assert str(stray_file.relative_to(registry.cache_dir)) in file_table
    assert 'tmp' in file_table
    assert 'unknown' in file_table
    registry.collect(report)
    assert not tmp_file.exists()
    assert stray_file.exists()  # unknown files are never deleted


def test_gc_disambiguation_pruning():
    root, registry = make_empty_registry()
    colliding = CollidingDerivative(root)
    registry.register(colliding)
    base_ctx = registry.resolve('colliding', state={'subject': 's1', 'mode': 'default'})
    base_ctx.load()
    alt_ctx = registry.resolve('colliding', state={'subject': 's2', 'mode': 'default'})
    alt_ctx.load()
    sidecar_path = Path(f"{base_ctx.artifact_path}{CACHE_DISAMBIGUATION_SUFFIX}")
    assert sidecar_path.exists()
    assert registry.scan_cache().entries == []  # both variants live
    # simulate a variant that disappeared (e.g. collected in an earlier run)
    alt_ctx.artifact_path.unlink()
    alt_ctx.manifest_path.unlink()
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.STALE_DISAMBIGUATION)
    assert len(entry.prune_digests) == 1
    registry.collect(report)
    assert not sidecar_path.exists()  # last entry pruned → sidecar removed
    assert base_ctx.artifact_path.exists()


def test_gc_scan_is_readonly():
    root, registry = make_empty_registry()
    versioned = FakeVersionedInput(root)
    consumer = VersionedConsumerDerivative(root)
    registry.register(versioned)
    registry.register(consumer)
    versioned._source_path().write_text('one')
    registry.resolve('versioned-consumer').load()
    versioned._source_path().write_text('two!')
    cache_files = {path: path.stat().st_mtime_ns for path in registry.cache_dir.rglob('*') if path.is_file()}
    report = registry.scan_cache()
    _single_entry(report, GCCategory.REVALIDATION_STALE)
    after = {path: path.stat().st_mtime_ns for path in registry.cache_dir.rglob('*') if path.is_file()}
    assert after == cache_files  # no new reference pickle, no manifest refresh
    # a normal load rebuilds and mints the new reference
    assert registry.resolve('versioned-consumer').load() == 'two!'
    assert (registry.cache_dir / 'versioned' / 'ref.1.pickle').exists()


def test_gc_unverifiable_manifest_backfilled_on_use():
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    _strip_resolve_fields(ctx.manifest_path)
    report = registry.scan_cache()
    entry = _single_entry(report, GCCategory.UNVERIFIABLE)
    assert 'predates' in entry.reason
    registry.collect(report)
    assert ctx.artifact_path.exists()
    # a cache hit backfills the resolve context without rebuilding
    assert registry.resolve('configured', state=DEFAULT_STATE).load() == 'configured:a'
    data = json.loads(ctx.manifest_path.read_text())
    assert data['resolve_state'] == {'subject': 's1'}
    assert registry.scan_cache().entries == []


def test_gc_rich_option_round_trip():
    root, registry = make_empty_registry()
    node = RichOptionDerivative(root)
    registry.register(node)
    ctx = registry.resolve('rich-option', state=DEFAULT_STATE, options={'spec': RichSpec('A')})
    assert ctx.load() == 'A'
    report = registry.scan_cache()
    assert report.entries == []
    assert report.errors == []  # reconstruction re-parses the canonical form into a RichSpec
    node.config = 'b'
    report = registry.scan_cache()
    _single_entry(report, GCCategory.REVALIDATION_STALE)


def test_data_spec_cache_form_round_trip():
    for spec in (DataSpec('sensor'), DataSpec('source'), DataSpec('eeg.mean')):
        form = DerivativeRegistry.canonicalize(spec)
        assert isinstance(form, str)
        assert DataSpec.coerce(form) == spec


def test_gc_collect_logs_deletions(caplog):
    registry, configured, downstream, root = make_gc_registry()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)
    ctx.load()
    configured.config = 'b'
    report = registry.scan_cache()
    with caplog.at_level(logging.DEBUG, logger=LOG.name):
        registry.collect(report)
    debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
    assert any('Cache GC: removed' in message and 'revalidation_stale' in message for message in debug_messages)
    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert any(message.startswith('Cache GC: removed 2 files') for message in info_messages)


def test_pipeline_clean_cache(monkeypatch):
    """Pipeline.clean_cache: dry_run reports only; confirm=False deletes."""
    from eelbrain import Pipeline

    registry, configured, downstream, root = make_gc_registry()
    registry.resolve('downstream', state=DEFAULT_STATE).load()
    configured.config = 'b'

    class FakeExperiment:
        _derivatives = registry
        _log = LOG
        clean_cache = Pipeline.clean_cache

    experiment = FakeExperiment()
    ctx = registry.resolve('configured', state=DEFAULT_STATE)

    table = experiment.clean_cache(dry_run=True)
    assert 'revalidation_stale' in str(table)
    assert ctx.artifact_path.exists()  # dry run deletes nothing

    experiment.clean_cache(delete=True)
    assert not ctx.artifact_path.exists()
    assert registry.scan_cache().entries == []
