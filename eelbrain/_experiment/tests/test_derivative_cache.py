from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from eelbrain._experiment.derivative_cache import (
    ALLOW_PROTECTED_OVERWRITE,
    ArtifactManifest,
    CachePolicy,
    Dependency,
    Derivative,
    Request,
    DerivativeRegistry,
    Input,
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
    VIEW_OPTION_DEFAULTS = {'upper': False}

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def source_path(self, subject: str) -> Path:
        path = self.root / 'inputs' / f'{subject}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        path = self.source_path(ctx.state['subject'])
        return file_fingerprint(str(self.root), path, 'source-file', digest=True)

    def load(self, ctx: Request) -> str:
        value = self.source_path(ctx.state['subject']).read_text()
        if ctx.view_options['upper']:
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
    cache_policy = CachePolicy.DISABLED_BY_DEFAULT
    cache_suffix = '.txt'

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
    OPTION_DEFAULTS = {'artifact': 0}
    VIEW_OPTION_DEFAULTS = {'view': 0}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.calls = []

    def fingerprint(self, ctx: Request) -> dict[str, object]:
        return self.standard_fingerprint(ctx, state_fields=('subject',))

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


class ProtectedDerivative(Derivative[str]):
    name = 'protected'
    key_fields = ('subject',)

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, ctx: Request) -> str:
        return str(self.root / 'derived' / ctx.state['subject'] / 'protected.txt')

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
    registry.register(source)
    registry.register(value)
    registry.register(summary)
    registry.register(comparison)
    registry.register(ephemeral)
    registry.register(protected)
    return pipeline, registry, source, value, summary, comparison, ephemeral, protected, None


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

    first = registry.resolve('value', state=DEFAULT_STATE).load()
    second = registry.resolve('value', state=DEFAULT_STATE).load()

    assert first == second == 'alpha'
    assert value.build_calls == 1


def test_registry_logs_cache_events(caplog):
    _, registry, _, value, _, _, _, _, _ = make_registry()

    with caplog.at_level(logging.DEBUG, logger='eelbrain.test.derivative_cache'):
        registry.resolve('value', state=DEFAULT_STATE).load()
        registry.resolve('value', state=DEFAULT_STATE).load()

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

    assert registry.resolve('summary', state=DEFAULT_STATE).load() == 'summary:alpha'
    assert value.build_calls == 1
    assert summary.build_calls == 1

    pipeline.source_path().write_text('changed')

    assert registry.resolve('summary', state=DEFAULT_STATE).load() == 'summary:changed'
    assert value.build_calls == 2
    assert summary.build_calls == 2


def test_non_key_state_does_not_invalidate_cache():
    _, registry, _, value, _, _, _, _, _ = make_registry()

    assert registry.resolve('value', state={'subject': 's1', 'mode': 'a'}).load() == 'alpha'
    assert registry.resolve('value', state={'subject': 's1', 'mode': 'b'}).load() == 'alpha'
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


def test_cache_collision_sidecar_disambiguates_artifact_paths():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.derivative_cache'))
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
    _, registry, _, value, _, _, _, _, _ = make_registry()

    assert registry.resolve('value', state=DEFAULT_STATE).load() == 'alpha'
    handle = registry.resolve('value', state=DEFAULT_STATE)

    assert not Path(f"{handle.base_artifact_path}.disambiguation.json").exists()
    assert value.build_calls == 1


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

    first = registry.resolve('ephemeral', state=DEFAULT_STATE).load()
    second = registry.resolve('ephemeral', state=DEFAULT_STATE).load()
    handle = registry.resolve('ephemeral', state=DEFAULT_STATE)

    assert first == 'ephemeral-1'
    assert second == 'ephemeral-2'
    assert ephemeral.build_calls == 2
    assert not handle.artifact_path.exists()
    assert not handle.manifest_path.exists()


def test_registry_resolve_returns_request_for_input_and_derivative():
    _, registry, _, _, _, _, _, _, _ = make_registry()

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
    assert value_handle.describe_dependency()['manifest'] == str(value_handle.manifest_path)

    assert registry.resolve('source', state=DEFAULT_STATE).load() == 'alpha'
    assert registry.resolve('source', state=DEFAULT_STATE, options={'upper': True}).load() == 'ALPHA'


def test_stale_external_artifact_is_protected():
    pipeline, registry, _, _, _, _, _, _, _ = make_registry()

    assert registry.resolve('protected', state=DEFAULT_STATE).load() == 'alpha'
    protected_path = Path(pipeline.get('protected-file'))
    manifest_path = Path(registry.manifest_path(protected_path))
    assert protected_path.exists()
    assert manifest_path.exists()
    assert manifest_path.is_relative_to(Path(pipeline.get('cache-dir')) / 'manifests')

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
    pipeline, registry, _, _, _, _, _, _, _ = make_registry()

    assert registry.resolve('protected', state=DEFAULT_STATE).load() == 'alpha'
    pipeline.source_path().write_text('changed')

    with pytest.raises(ProtectedArtifactError):
        registry.resolve('protected', state=DEFAULT_STATE, controls={'reindex_anything'}).load()


def test_runtime_code_does_not_use_private_get_node():
    experiment_dir = Path(__file__).resolve().parents[1]
    offenders = []
    for path in experiment_dir.glob('*.py'):
        if path.name == 'derivative_cache.py':
            continue
        if '._get_node(' in path.read_text():
            offenders.append(path.name)
    assert offenders == []


def test_request_splits_artifact_and_view_options():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.derivative_cache'))
    derivative = OptionDerivative(root)
    registry.register(derivative)

    handle = registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 1, 'view': 2})

    assert handle.options == {'artifact': 1}
    assert handle.view_options == {'view': 2}
    assert handle.current_fingerprint()['options'] == {'artifact': 1}
    assert handle.options_for('optioned', artifact=4) == {'artifact': 4}
    assert handle.options_for('optioned', 'view', artifact=4) == {'view': 2, 'artifact': 4}
    with pytest.raises(TypeError, match="does not declare option"):
        handle.options_for('optioned', artifact=4, extra=5)


def test_registry_rejects_undeclared_options():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.derivative_cache'))
    derivative = OptionDerivative(root)
    registry.register(derivative)

    with pytest.raises(TypeError, match="undeclared option"):
        registry.resolve('optioned', state=DEFAULT_STATE, options={'artifact': 1, 'extra': 3})


def test_request_applies_view_options_after_build_and_load():
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.derivative_cache'))
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
    root = TempDir()
    registry = DerivativeRegistry(root, logging.getLogger('eelbrain.test.derivative_cache'))
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
