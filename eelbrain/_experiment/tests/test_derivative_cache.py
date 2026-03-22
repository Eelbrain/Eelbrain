from __future__ import annotations

import json
from pathlib import Path

from eelbrain._experiment.derivative_cache import (
    Artifact,
    ArtifactManifest,
    CachePolicy,
    Dependency,
    Derivative,
    DerivativeContext,
    DerivativeHandle,
    DerivativeRegistry,
    Input,
    InputHandle,
    MANIFEST_SUFFIX,
    ProtectedArtifactError,
    file_fingerprint,
)
from eelbrain.testing import TempDir


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
            path = self.root / 'cache'
            if mkdir:
                path.mkdir(parents=True, exist_ok=True)
            return str(path)
        if key == 'deriv-dir':
            return str(self.root / 'derived')
        if key == 'root':
            return str(self.root)

        subject = state['subject']
        if key == 'value-file':
            path = self.root / 'cache' / subject / 'value.txt'
        elif key == 'summary-file':
            path = self.root / 'cache' / subject / 'summary.txt'
        elif key == 'ephemeral-file':
            path = self.root / 'cache' / subject / 'ephemeral.txt'
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

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        path = ctx.pipeline.source_path(ctx.get('subject'))
        return file_fingerprint(ctx.pipeline.root, path, 'source-file', digest=True)

    def load(self, ctx: DerivativeContext) -> str:
        value = ctx.pipeline.source_path(ctx.get('subject')).read_text()
        if ctx.option('upper', False):
            return value.upper()
        return value


class ValueDerivative(Derivative[str]):
    name = 'value'
    path_template = 'value-file'
    key_fields = ('subject',)

    def __init__(self):
        self.build_calls = 0
        self.load_calls = 0
        self.save_calls = 0

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        self.build_calls += 1
        return ctx.pipeline.source_path(ctx.get('subject')).read_text()

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> str:
        self.load_calls += 1
        return Path(artifact.path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: str,
    ) -> None:
        self.save_calls += 1
        Path(artifact.path).write_text(value)


class SummaryDerivative(Derivative[str]):
    name = 'summary'
    path_template = 'summary-file'
    key_fields = ('subject',)

    def __init__(self):
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
            artifact: Artifact,
    ) -> str:
        self.load_calls += 1
        return Path(artifact.path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: str,
    ) -> None:
        Path(artifact.path).write_text(value)


class EphemeralDerivative(Derivative[str]):
    name = 'ephemeral'
    path_template = 'ephemeral-file'
    key_fields = ('subject',)
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT

    def __init__(self):
        self.build_calls = 0

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        self.build_calls += 1
        return f"ephemeral-{self.build_calls}"

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> str:
        return Path(artifact.path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: str,
    ) -> None:
        Path(artifact.path).write_text(value)


class ProtectedDerivative(Derivative[str]):
    name = 'protected'
    path_template = 'protected-file'
    key_fields = ('subject',)

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('source'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        return {'subject': ctx.get('subject')}

    def build(self, ctx: DerivativeContext) -> str:
        return ctx.load('source')

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> str:
        return Path(artifact.path).read_text()

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: str,
    ) -> None:
        Path(artifact.path).write_text(value)


class ReindexableProtectedDerivative(ProtectedDerivative):
    name = 'reindexable-protected'

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, object]:
        path = Path(ctx.path(self.path_template))
        text = path.read_text() if path.exists() else None
        return {
            'subject': ctx.get('subject'),
            'artifact_text': text,
        }

    def can_reindex_protected_artifact(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
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
    registry = DerivativeRegistry(pipeline)
    source = SourceInput()
    value = ValueDerivative()
    summary = SummaryDerivative()
    ephemeral = EphemeralDerivative()
    protected = ProtectedDerivative()
    reindexable_protected = ReindexableProtectedDerivative()
    registry.register(source)
    registry.register(value)
    registry.register(summary)
    registry.register(ephemeral)
    registry.register(protected)
    registry.register(reindexable_protected)
    return pipeline, registry, source, value, summary, ephemeral, protected, reindexable_protected


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
    pipeline, registry, _, value, _, _, _, _ = make_registry()

    first = registry.load('value')
    second = registry.load('value')

    assert first == second == 'alpha'
    assert value.build_calls == 1
    assert value.save_calls == 1
    assert value.load_calls == 2

    cache_path = Path(pipeline.get('value-file'))
    manifest_path = Path(f"{cache_path}{MANIFEST_SUFFIX}")
    assert cache_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest['derivative'] == 'value'
    assert manifest['key'] == {'subject': 's1'}
    assert manifest['dependencies']['source']['kind'] == 'input'


def test_dependency_change_invalidates_downstream_derivatives():
    pipeline, registry, _, value, summary, _, _, _ = make_registry()

    assert registry.load('summary') == 'summary:alpha'
    assert value.build_calls == 1
    assert summary.build_calls == 1

    pipeline.source_path().write_text('changed')

    assert registry.load('summary') == 'summary:changed'
    assert value.build_calls == 2
    assert summary.build_calls == 2


def test_non_key_state_does_not_invalidate_cache():
    _, registry, _, value, _, _, _, _ = make_registry()

    assert registry.load('value', state={'subject': 's1', 'mode': 'a'}) == 'alpha'
    assert registry.load('value', state={'subject': 's1', 'mode': 'b'}) == 'alpha'
    assert value.build_calls == 1


def test_disabled_by_default_derivative_skips_cache_by_default():
    pipeline, registry, _, _, _, ephemeral, _, _ = make_registry()

    first = registry.load('ephemeral')
    second = registry.load('ephemeral')

    assert first == 'ephemeral-1'
    assert second == 'ephemeral-2'
    assert ephemeral.build_calls == 2
    assert not Path(pipeline.get('ephemeral-file')).exists()
    assert not Path(f"{pipeline.get('ephemeral-file')}{MANIFEST_SUFFIX}").exists()


def test_registry_resolve_returns_input_handle_and_loads_input():
    _, registry, _, _, _, _, _, _ = make_registry()

    handle = registry.resolve('source')
    assert isinstance(handle, InputHandle)
    assert handle.describe_dependency()['kind'] == 'input'

    value_handle = registry.resolve('value')
    assert isinstance(value_handle, DerivativeHandle)

    assert registry.load('source') == 'alpha'
    assert registry.load('source', options={'upper': True}) == 'ALPHA'


def test_stale_external_artifact_is_protected():
    pipeline, registry, _, _, _, _, _, _ = make_registry()

    assert registry.load('protected') == 'alpha'
    protected_path = Path(pipeline.get('protected-file'))
    manifest_path = Path(registry.manifest_path(protected_path))
    assert protected_path.exists()
    assert manifest_path.exists()
    assert manifest_path.is_relative_to(Path(pipeline.get('cache-dir')) / 'manifests')

    pipeline.source_path().write_text('changed')

    try:
        registry.load('protected')
    except ProtectedArtifactError as error:
        assert error.derivative == 'protected'
        assert error.path == str(protected_path)
    else:
        raise AssertionError("Expected ProtectedArtifactError")

    assert protected_path.read_text() == 'alpha'
    assert registry.load('protected', options={'_allow_protected_overwrite': True}) == 'changed'
    assert protected_path.read_text() == 'changed'


def test_protected_artifact_can_reindex_manifest():
    pipeline, registry, _, _, _, _, _, _ = make_registry()

    assert registry.load('reindexable-protected') == 'alpha'
    protected_path = Path(pipeline.get('protected-file'))
    manifest_path = Path(registry.manifest_path(protected_path))

    protected_path.write_text('user-edit')

    assert registry.load('reindexable-protected') == 'user-edit'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['fingerprint']['artifact_text'] == 'user-edit'
