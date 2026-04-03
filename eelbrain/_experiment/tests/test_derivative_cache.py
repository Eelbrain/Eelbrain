from __future__ import annotations

import json
import logging
from pathlib import Path

from eelbrain._experiment.derivative_cache import (
    ArtifactManifest,
    CachePolicy,
    Dependency,
    Derivative,
    DerivativeContext,
    DerivativeHandle,
    DerivativeRegistry,
    Input,
    InputHandle,
    ProtectedArtifactError,
    file_fingerprint,
)
from eelbrain.testing import TempDir


DEFAULT_STATE = {'subject': 's1', 'mode': 'default'}


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
        self.cache_policy_overrides = {}

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
            path = self.root / 'derived' / subject / 'protected.txt'
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

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def source_path(self, subject: str) -> Path:
        path = self.root / 'inputs' / f'{subject}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        path = self.source_path(ctx.get('subject'))
        return file_fingerprint(str(self.root), path, 'source-file', digest=True)

    def load(self, ctx: DerivativeContext) -> str:
        value = self.source_path(ctx.get('subject')).read_text()
        if ctx.option('upper', False):
            return value.upper()
        return value


class ValueDerivative(Derivative[str]):
    name = 'value'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = 0
        self.load_calls = 0
        self.save_calls = 0

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        self.build_calls += 1
        return (self.root / 'inputs' / f"{ctx.get('subject')}.txt").read_text()

    def load(
            self,
            ctx: DerivativeContext,
            path: str) -> str:
        self.load_calls += 1
        return Path(path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
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

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('value'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        self.build_calls += 1
        return f"summary:{ctx.load('value')}"

    def load(
            self,
            ctx: DerivativeContext,
            path: str) -> str:
        self.load_calls += 1
        return Path(path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class ComparisonDerivative(Derivative[str]):
    name = 'comparison'
    key_fields = ('subject',)
    cache_suffix = '.txt'

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency('value', label='current'),
            Dependency('value', label='other', state={'subject': 's2'}),
        )

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        return f"{ctx.load('value')} vs {ctx.load('value', subject='s2')}"

    def load(
            self,
            ctx: DerivativeContext,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class EphemeralDerivative(Derivative[str]):
    name = 'ephemeral'
    key_fields = ('subject',)
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.txt'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.build_calls = 0

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        self.build_calls += 1
        return f"ephemeral-{self.build_calls}"

    def load(
            self,
            ctx: DerivativeContext,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class ProtectedDerivative(Derivative[str]):
    name = 'protected'
    key_fields = ('subject',)

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, ctx: DerivativeContext) -> str:
        return str(self.root / 'derived' / ctx.get('subject') / 'protected.txt')

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        return ctx.load('source')

    def load(
            self,
            ctx: DerivativeContext,
            path: str) -> str:
        return Path(path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            path: str,
            value: str,
    ) -> None:
        Path(path).write_text(value)


class ReindexableProtectedDerivative(ProtectedDerivative):
    name = 'reindexable-protected'

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        path = Path(self.path(ctx))
        text = path.read_text() if path.exists() else None
        return {
            'subject': ctx.get('subject'),
            'artifact_text': text,
        }

    def can_reindex_protected_artifact(
            self,
            ctx: DerivativeContext,
            path: str,
            manifest: ArtifactManifest,
            cache: bool | None = None,
    ) -> bool:
        current = dict(self.fingerprint(ctx))
        previous = dict(manifest.fingerprint)
        current.pop('artifact_text', None)
        previous.pop('artifact_text', None)
        return current == previous


def make_registry():
    root = TempDir()
    pipeline = FakePipeline(root)
    pipeline.source_path('s1').write_text('alpha')
    pipeline.source_path('s2').write_text('beta')
    registry = DerivativeRegistry(pipeline.root, logging.getLogger('eelbrain.test.derivative_cache'))
    source = SourceInput(root)
    value = ValueDerivative(root)
    summary = SummaryDerivative(root)
    comparison = ComparisonDerivative()
    ephemeral = EphemeralDerivative(root)
    protected = ProtectedDerivative(root)
    reindexable_protected = ReindexableProtectedDerivative(root)
    registry.register(source)
    registry.register(value)
    registry.register(summary)
    registry.register(comparison)
    registry.register(ephemeral)
    registry.register(protected)
    registry.register(reindexable_protected)
    return pipeline, registry, source, value, summary, comparison, ephemeral, protected, reindexable_protected


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
        'provenance': {'a': 1},
        'serializer': 'obsolete',
    })
    assert manifest.derivative == 'value'
    assert manifest.provenance == {'a': 1}


def test_registry_load_caches_derivative_and_writes_manifest():
    pipeline, registry, _, value, _, _, _, _, _ = make_registry()

    first = registry.load('value', state=DEFAULT_STATE)
    second = registry.load('value', state=DEFAULT_STATE)

    assert first == second == 'alpha'
    assert value.build_calls == 1


def test_registry_logs_cache_events(caplog):
    _, registry, _, value, _, _, _, _, _ = make_registry()

    with caplog.at_level(logging.DEBUG, logger='eelbrain.test.derivative_cache'):
        registry.load('value', state=DEFAULT_STATE)
        registry.load('value', state=DEFAULT_STATE)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith('Build value: value/') for message in messages)
    assert any(message.startswith('Load cached value: value/') for message in messages)
    assert value.save_calls == 1
    assert value.load_calls == 2

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


def test_dependency_change_invalidates_downstream_derivatives():
    pipeline, registry, _, value, summary, _, _, _, _ = make_registry()

    assert registry.load('summary', state=DEFAULT_STATE) == 'summary:alpha'
    assert value.build_calls == 1
    assert summary.build_calls == 1

    pipeline.source_path().write_text('changed')

    assert registry.load('summary', state=DEFAULT_STATE) == 'summary:changed'
    assert value.build_calls == 2
    assert summary.build_calls == 2


def test_non_key_state_does_not_invalidate_cache():
    _, registry, _, value, _, _, _, _, _ = make_registry()

    assert registry.load('value', state={'subject': 's1', 'mode': 'a'}) == 'alpha'
    assert registry.load('value', state={'subject': 's1', 'mode': 'b'}) == 'alpha'
    assert value.build_calls == 1


def test_generic_cache_path_uses_node_name_and_key():
    _, registry, _, _, _, _, _, _, _ = make_registry()

    a = registry.resolve('value', state={'subject': 's1', 'mode': 'a'}).artifact_path
    b = registry.resolve('value', state={'subject': 's1', 'mode': 'b'}).artifact_path
    c = registry.resolve('value', state={'subject': 's2', 'mode': 'a'}).artifact_path

    assert a == b
    assert a != c
    assert a.is_relative_to(registry.cache_dir / 'value')
    assert c.is_relative_to(registry.cache_dir / 'value')


def test_dependency_tree_formats_ascii_dependencies():
    _, registry, _, _, _, _, _, _, _ = make_registry()

    tree = registry.dependency_tree('comparison', state=DEFAULT_STATE)

    assert "comparison [derivative] {subject='s1'}" in tree
    assert "current -> value [derivative] {subject='s1'}" in tree
    assert "other -> value [derivative] {subject='s2'} [state: subject='s2']" in tree
    assert 'source [input]' in tree
    assert '├──' in tree
    assert '└──' in tree


def test_dependency_tree_respects_max_line_length():
    _, registry, _, _, _, _, _, _, _ = make_registry()

    tree = registry.dependency_tree('comparison', state=DEFAULT_STATE, max_line_length=44)
    lines = tree.splitlines()

    assert len(lines) > 4
    assert all(len(line) <= 44 for line in lines)
    assert "other -> value [derivative]" in tree
    assert "{subject='s2'} [state: subject='s2']" in tree


def test_disabled_by_default_derivative_skips_cache_by_default():
    _, registry, _, _, _, _, ephemeral, _, _ = make_registry()

    first = registry.load('ephemeral', state=DEFAULT_STATE)
    second = registry.load('ephemeral', state=DEFAULT_STATE)
    handle = registry.resolve('ephemeral', state=DEFAULT_STATE)

    assert first == 'ephemeral-1'
    assert second == 'ephemeral-2'
    assert ephemeral.build_calls == 2
    assert not handle.artifact_path.exists()
    assert not handle.manifest_path.exists()


def test_registry_resolve_returns_input_handle_and_loads_input():
    _, registry, _, _, _, _, _, _, _ = make_registry()

    handle = registry.resolve('source', state=DEFAULT_STATE)
    assert isinstance(handle, InputHandle)
    assert handle.describe_dependency()['kind'] == 'input'

    value_handle = registry.resolve('value', state=DEFAULT_STATE)
    assert isinstance(value_handle, DerivativeHandle)

    assert registry.load('source', state=DEFAULT_STATE) == 'alpha'
    assert registry.load('source', state=DEFAULT_STATE, options={'upper': True}) == 'ALPHA'


def test_stale_external_artifact_is_protected():
    pipeline, registry, _, _, _, _, _, _, _ = make_registry()

    assert registry.load('protected', state=DEFAULT_STATE) == 'alpha'
    protected_path = Path(pipeline.get('protected-file'))
    manifest_path = Path(registry.manifest_path(protected_path))
    assert protected_path.exists()
    assert manifest_path.exists()
    assert manifest_path.is_relative_to(Path(pipeline.get('cache-dir')) / 'manifests')

    pipeline.source_path().write_text('changed')

    try:
        registry.load('protected', state=DEFAULT_STATE)
    except ProtectedArtifactError as error:
        assert error.derivative == 'protected'
        assert error.path == str(protected_path)
    else:
        raise AssertionError("Expected ProtectedArtifactError")

    assert protected_path.read_text() == 'alpha'
    assert registry.load('protected', state=DEFAULT_STATE, options={'_allow_protected_overwrite': True}) == 'changed'
    assert protected_path.read_text() == 'changed'


def test_protected_artifact_can_reindex_manifest():
    pipeline, registry, _, _, _, _, _, _, _ = make_registry()

    assert registry.load('reindexable-protected', state=DEFAULT_STATE) == 'alpha'
    protected_path = Path(pipeline.get('protected-file'))
    manifest_path = Path(registry.manifest_path(protected_path))

    protected_path.write_text('user-edit')

    assert registry.load('reindexable-protected', state=DEFAULT_STATE) == 'user-edit'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['fingerprint']['artifact_text'] == 'user-edit'


def test_protected_artifact_can_be_incorporated_at_user_risk():
    pipeline, registry, _, _, _, _, _, _, _ = make_registry()

    assert registry.load('protected', state=DEFAULT_STATE) == 'alpha'
    pipeline.source_path().write_text('changed')

    assert registry.load('protected', state=DEFAULT_STATE, options={'_allow_protected_reindex': True}) == 'alpha'
    assert registry.load('protected', state=DEFAULT_STATE) == 'alpha'


def test_runtime_code_does_not_use_private_get_node():
    experiment_dir = Path(__file__).resolve().parents[1]
    offenders = []
    for path in experiment_dir.glob('*.py'):
        if path.name == 'derivative_cache.py':
            continue
        if '._get_node(' in path.read_text():
            offenders.append(path.name)
    assert offenders == []
