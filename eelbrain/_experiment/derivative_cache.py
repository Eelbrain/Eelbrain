"""Derivative-oriented cache primitives for :mod:`eelbrain._experiment`.

The cache is organized around registered dependency nodes.

- A derivative is *keyed* by the subset of pipeline state that selects which
  artifact it represents, such as subject, raw pipeline, epoch, or covariance.
- A derivative is *built* by computing that artifact from the current pipeline
  state.
- A derivative is *serialized* by saving the in-memory result to disk and
  loading it back later.
- A derivative is *fingerprinted* by recording the normalized settings and
  inputs that determine whether a cached artifact is still valid.

Within that framework:

- :class:`Input` declares one source node in the dependency graph.
- :class:`Derivative` declares one computed node in the dependency graph.
- :class:`NodeHandle` binds one node to the current pipeline state and cache
  options.
- :class:`DerivativeRegistry` resolves dependencies, normalizes state, and
  validates cached artifacts via sidecar manifests.

Manifests store the derivative fingerprint plus dependency fingerprints, so a
cache hit is valid when the artifact, its normalized key, and its dependency
graph still match the current pipeline configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Generic, TypeVar
from collections.abc import Callable

import mne

T = TypeVar('T')
MANIFEST_SUFFIX = '.manifest.json'
MANIFEST_SCHEMA_VERSION = 1


class CachePolicy(str, Enum):
    """How strongly the cache engine should prefer persistence for a derivative."""

    REQUIRED = 'required'
    OPTIONAL = 'optional'
    DISABLED_BY_DEFAULT = 'disabled_by_default'


@dataclass(frozen=True)
class InputFingerprint:
    """Portable description of one non-derivative input."""

    kind: str
    path: str | None
    exists: bool
    size: int | None = None
    mtime: float | None = None
    digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactManifest:
    """Sidecar metadata used to validate one cached artifact."""

    schema_version: int
    derivative: str
    derivative_version: int
    key: dict[str, Any]
    fingerprint: dict[str, Any]
    dependencies: dict[str, Any]
    cache_policy: str
    software: dict[str, str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        allowed = {field_.name for field_ in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)


@dataclass
class DerivativeContext:
    """Bound state/options environment for resolving one node request."""

    registry: DerivativeRegistry
    state: dict[str, Any]
    options: dict[str, Any]

    @property
    def pipeline(self) -> Any:
        return self.registry.pipeline

    def option(self, key: str, default=None):
        return self.options.get(key, default)

    def get(self, key: str):
        with self.pipeline._temporary_state:
            if self.state:
                self.pipeline.set(**self.state)
            return self.pipeline.get(key)

    def path(
            self,
            template: str,
            mkdir: bool = False,
            **extra_state,
    ) -> str:
        with self.pipeline._temporary_state:
            if self.state:
                self.pipeline.set(**self.state)
            if extra_state:
                self.pipeline.set(**extra_state)
            return self.pipeline.get(template, mkdir=mkdir)

    def load(
            self,
            name: str,
            state: dict[str, Any] | None = None,
            **extra_state,
    ):
        merged_state = dict(self.state)
        if state:
            merged_state.update(state)
        if extra_state:
            merged_state.update(extra_state)
        return self.registry.load(name, state=merged_state, options=self.options)


@dataclass(frozen=True)
class Dependency:
    """One edge in the derivative graph.

    ``name`` refers to either a registered :class:`Derivative` or a registered
    :class:`Input`, and the registry determines which kind of node it is.
    ``state`` can override which keyed instance should be resolved.
    """

    name: str
    state: Callable[[DerivativeContext], dict[str, Any]] | None = None


class DependencyNode:
    """Shared graph interface for derivatives and non-derivative inputs."""

    name: str

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return ()

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        raise NotImplementedError


class Input(DependencyNode):
    """Behavioral interface for one non-derivative dependency."""


class Derivative(DependencyNode, Generic[T]):
    """Behavioral interface for one cacheable derivative."""

    path_template: str
    key_fields: tuple[str, ...]
    cache_policy: CachePolicy = CachePolicy.REQUIRED
    version: int = 1

    def build(self, ctx: DerivativeContext) -> T:
        raise NotImplementedError

    def load(self, ctx: DerivativeContext, path: str) -> T:
        raise NotImplementedError

    def save(self, ctx: DerivativeContext, path: str, value: T) -> None:
        raise NotImplementedError

    def validate(
            self,
            ctx: DerivativeContext,
            path: str,
            manifest: ArtifactManifest,
    ) -> bool:
        return True

    def provenance(
            self,
            ctx: DerivativeContext,
            value: T,
    ) -> dict[str, Any]:
        return {}


class NodeHandle:
    """A dependency node plus its bound context for one concrete request."""

    def __init__(
            self,
            node: DependencyNode,
            ctx: DerivativeContext,
    ):
        self.node = node
        self.ctx = ctx

    @property
    def registry(self) -> DerivativeRegistry:
        return self.ctx.registry

    @property
    def state(self) -> dict[str, Any]:
        return self.ctx.state

    @property
    def options(self) -> dict[str, Any]:
        return self.ctx.options

    def dependency_fingerprints(
            self,
            cache: bool | None = None,
    ) -> dict[str, Any]:
        return self.registry.dependency_fingerprints(self.node, self.ctx, cache)

    def current_fingerprint(self) -> dict[str, Any]:
        return self.registry.canonicalize(self.node.fingerprint(self.ctx))

    def describe_dependency(
            self,
            cache: bool | None = None,
    ) -> dict[str, Any]:
        out = {
            'fingerprint': self.current_fingerprint(),
            'dependencies': self.dependency_fingerprints(cache),
        }
        if isinstance(self, DerivativeHandle):
            out['kind'] = 'derivative'
            out['key'] = self.key()
        else:
            out['kind'] = 'input'
        return out


class DerivativeHandle(NodeHandle, Generic[T]):
    """A derivative plus its bound context for one concrete load request."""

    node: Derivative[T]

    @property
    def derivative(self) -> Derivative[T]:
        return self.node

    def path(self, mkdir: bool = False) -> str:
        return self.ctx.pipeline.get(self.derivative.path_template, mkdir=mkdir, **self.state)

    def manifest_path(self) -> str:
        return f"{self.path()}{MANIFEST_SUFFIX}"

    def key(self) -> dict[str, Any]:
        return self.registry.normalize_state(self.derivative.key_fields, self.state)

    def _manifest(self) -> ArtifactManifest | None:
        return self.registry.read_manifest(self.manifest_path())

    def _is_valid(
            self,
            manifest: ArtifactManifest,
            cache: bool | None = None,
    ) -> bool:
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            return False
        if manifest.derivative != self.derivative.name:
            return False
        if manifest.derivative_version != self.derivative.version:
            return False
        if manifest.key != self.key():
            return False
        if manifest.fingerprint != self.current_fingerprint():
            return False
        if manifest.dependencies != self.dependency_fingerprints(cache):
            return False
        if not self.derivative.validate(self.ctx, self.path(), manifest):
            return False
        return True

    def load(self, cache: bool | None = None) -> T:
        use_cache = self.registry.should_cache(self.derivative, cache)
        path = self.path()
        if use_cache:
            manifest = self._manifest()
            if manifest and Path(path).exists() and self._is_valid(manifest, cache):
                return self.derivative.load(self.ctx, path)

        value = self.derivative.build(self.ctx)
        if not use_cache:
            return value

        path = self.path(mkdir=True)
        self.derivative.save(self.ctx, path, value)
        manifest = ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=self.derivative.name,
            derivative_version=self.derivative.version,
            key=self.key(),
            fingerprint=self.current_fingerprint(),
            dependencies=self.dependency_fingerprints(cache),
            cache_policy=self.derivative.cache_policy.value,
            software={
                'eelbrain_cache_schema': str(MANIFEST_SCHEMA_VERSION),
                'mne': mne.__version__,
            },
            provenance=self.registry.canonicalize(self.derivative.provenance(self.ctx, value)),
        )
        self.registry.write_manifest(self.manifest_path(), manifest)
        return self.derivative.load(self.ctx, path)


class InputHandle(NodeHandle):
    """An input plus its bound context for one concrete fingerprint request."""

    def __init__(
            self,
            input_: Input,
            ctx: DerivativeContext,
    ):
        super().__init__(input_, ctx)

    input: Input

    @property
    def input(self) -> Input:
        return self.node


class DerivativeRegistry:
    """Registry and resolver for dependency nodes bound to one pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self._nodes: dict[str, DependencyNode] = {}

    def register(self, node: DependencyNode) -> None:
        if node.name in self._nodes:
            raise RuntimeError(f"Dependency node {node.name!r} already registered")
        if not isinstance(node, (Derivative, Input)):
            raise TypeError(f"Unsupported node type: {type(node)!r}")
        self._nodes[node.name] = node

    def _resolve_state(
            self,
            state: dict[str, Any] | None = None,
            **extra_state,
    ) -> dict[str, Any]:
        merged_state = {}
        if state:
            merged_state.update(state)
        if extra_state:
            merged_state.update(extra_state)
        return merged_state

    def _get_node(self, name: str) -> DependencyNode:
        try:
            return self._nodes[name]
        except KeyError:
            raise RuntimeError(f"Unknown dependency {name!r}") from None

    def resolve(
            self,
            name: str,  # Registered node name.
            state: dict[str, Any] | None = None,  # Base state for this node instance.
            options: dict[str, Any] | None = None,
            **extra_state,  # Additional state overrides merged on top of ``state``.
    ) -> NodeHandle:
        node = self._get_node(name)
        ctx = DerivativeContext(self, self._resolve_state(state, **extra_state), options or {})
        if isinstance(node, Derivative):
            return DerivativeHandle(node, ctx)
        return InputHandle(node, ctx)

    def load(
            self,
            name: str,  # Registered derivative name.
            cache: bool | None = None,  # Explicit cache override for this load.
            state: dict[str, Any] | None = None,  # Base state for this derivative instance.
            options: dict[str, Any] | None = None,
            **extra_state,  # Additional state overrides merged on top of ``state``.
    ):
        handle = self.resolve(name, state=state, options=options, **extra_state)
        if not isinstance(handle, DerivativeHandle):
            raise TypeError(f"{name!r} is an input node and can not be loaded through the cache registry")
        return handle.load(cache)

    def should_cache(self, derivative: Derivative, cache: bool | None) -> bool:
        if cache is not None:
            return cache
        override = getattr(self.pipeline, 'cache_policy_overrides', {}).get(derivative.name)
        if override is not None:
            return override not in ('off', False)
        return derivative.cache_policy != CachePolicy.DISABLED_BY_DEFAULT

    def normalize_state(self, fields: tuple[str, ...], state: dict[str, Any]) -> dict[str, Any]:
        with self.pipeline._temporary_state:
            if state:
                self.pipeline.set(**state)
            normalized = {}
            for key in fields:
                value = self.pipeline.get(key)
                normalized[key] = self.canonicalize(value)
            return normalized

    def dependency_fingerprints(
            self,
            node: DependencyNode,  # Node whose dependencies are being fingerprinted.
            ctx: DerivativeContext,  # Bound state/options for the current load.
            cache: bool | None,  # Explicit cache override propagated to dependencies.
    ) -> dict[str, Any]:
        out = {}
        for dep in node.dependencies(ctx):
            handle = self.resolve(
                dep.name,
                state=self._resolve_state(ctx.state, **(dep.state(ctx) if dep.state else {})),
                options=ctx.options,
            )
            out[dep.name] = handle.describe_dependency(cache)
        return out

    def read_manifest(self, path: str) -> ArtifactManifest | None:
        manifest_path = Path(path)
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text())
        return ArtifactManifest.from_dict(data)

    def write_manifest(self, path: str, manifest: ArtifactManifest) -> None:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))

    @staticmethod
    def canonicalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): DerivativeRegistry.canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [DerivativeRegistry.canonicalize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(DerivativeRegistry.canonicalize(v) for v in value)
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                return repr(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)


def file_fingerprint(root: str, path: str | Path, kind: str, digest: bool = False, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fingerprint one input path using a project-relative location when possible."""

    path = Path(path)
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    if not path.exists():
        out = InputFingerprint(kind, relative, False, metadata=metadata or {})
    else:
        stat = path.stat()
        sha1 = None
        if digest and path.is_file():
            sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
        out = InputFingerprint(kind, relative, True, stat.st_size, stat.st_mtime, sha1, metadata or {})
    return asdict(out)
