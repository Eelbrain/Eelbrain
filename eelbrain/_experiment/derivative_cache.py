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

Artifacts inside ``cache-dir`` keep sidecar manifests and can be rebuilt
automatically when they go stale. Artifacts stored elsewhere are treated as
user-managed outputs: their manifests are mirrored under
``cache-dir/manifests`` and they are not overwritten without an explicit
opt-in from the caller.
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


@dataclass(frozen=True)
class Artifact:
    """Resolved storage location for one cached derivative artifact."""

    path: str
    manifest_path: str


class ProtectedArtifactError(RuntimeError):
    """Refuse to overwrite a stale artifact that lives outside ``cache-dir``."""

    def __init__(
            self,
            derivative: str,
            path: str,
    ):
        self.derivative = derivative
        self.path = path
        super().__init__(
            f"Stale artifact for derivative {derivative!r} at {path!r} lives outside "
            "the cache directory and will not be overwritten automatically."
        )


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
            options: dict[str, Any] | None = None,
            **extra_state,
    ):
        merged_state = dict(self.state)
        if state:
            merged_state.update(state)
        if extra_state:
            merged_state.update(extra_state)
        merged_options = dict(self.options)
        if options:
            merged_options.update(options)
        return self.registry.load(name, state=merged_state, options=merged_options)


@dataclass(frozen=True)
class Dependency:
    """One edge in the derivative graph.

    ``name`` refers to either a registered :class:`Derivative` or a registered
    :class:`Input`, and the registry determines which kind of node it is.
    ``label`` names this edge in dependency manifests; use it when the same
    node can appear more than once with different state.
    ``state`` can override which keyed instance should be resolved.
    ``options`` can override load-time options for the dependency.
    """

    name: str
    label: str | None = None
    state: Callable[[DerivativeContext], dict[str, Any]] | None = None
    options: Callable[[DerivativeContext], dict[str, Any]] | None = None


class DependencyNode(Generic[T]):
    """Shared dependency/fingerprint interface for graph nodes."""

    name: str

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return ()

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        raise NotImplementedError

    def dependency_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.fingerprint(ctx)


class Input(DependencyNode[T]):
    """Behavioral interface for one non-derivative dependency."""

    def load(self, ctx: DerivativeContext):
        raise NotImplementedError


class Derivative(DependencyNode[T]):
    """Behavioral interface for one cacheable derivative."""

    path_template: str
    key_fields: tuple[str, ...]
    cache_policy: CachePolicy = CachePolicy.REQUIRED
    version: int = 1

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> str:
        return ctx.path(self.path_template, mkdir=mkdir)

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.normalize_state(self.key_fields, ctx.state)

    def build(self, ctx: DerivativeContext) -> T:
        raise NotImplementedError

    def load(self, ctx: DerivativeContext, artifact: Artifact) -> T:
        raise NotImplementedError

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: T,
    ) -> None:
        raise NotImplementedError

    def validate(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            manifest: ArtifactManifest,
    ) -> bool:
        return True

    def can_reindex_protected_artifact(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            manifest: ArtifactManifest,
            cache: bool | None = None,
    ) -> bool:
        return False

    def provenance(
            self,
            ctx: DerivativeContext,
            value: T,
    ) -> dict[str, Any]:
        return {}


class NodeHandle(Generic[T]):
    """A dependency node plus its bound context for one concrete request."""

    def __init__(
            self,
            node: DependencyNode[T],
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

    def current_dependency_fingerprint(self) -> dict[str, Any]:
        return self.registry.canonicalize(self.node.dependency_fingerprint(self.ctx))

    def describe_dependency(
            self,
            cache: bool | None = None,
    ) -> dict[str, Any]:
        out = {
            'fingerprint': self.current_dependency_fingerprint(),
            'dependencies': self.dependency_fingerprints(cache),
        }
        if isinstance(self, DerivativeHandle):
            out['kind'] = 'derivative'
            out['key'] = self.key()
        else:
            out['kind'] = 'input'
        return out

    def load(self, cache: bool | None = None) -> T:
        raise NotImplementedError


class DerivativeHandle(NodeHandle[T]):
    """A derivative plus its bound context for one concrete load request."""

    node: Derivative[T]

    @property
    def derivative(self) -> Derivative[T]:
        return self.node

    def artifact(self, mkdir: bool = False) -> Artifact:
        path = self.derivative.path(self.ctx, mkdir=mkdir)
        return Artifact(path, self.registry.manifest_path(path))

    def _is_protected_artifact(self, artifact: Artifact) -> bool:
        return (
            Path(artifact.path).exists()
            and not self.registry.is_cache_artifact(artifact.path)
        )

    def key(self) -> dict[str, Any]:
        return self.derivative.key(self.ctx)

    def _manifest(self) -> ArtifactManifest | None:
        return self.registry.read_manifest(self.artifact().manifest_path)

    def _is_valid(
            self,
            manifest: ArtifactManifest,
            artifact: Artifact,
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
        if not self.derivative.validate(self.ctx, artifact, manifest):
            return False
        return True

    def _build_manifest(
            self,
            value: T,
            cache: bool | None = None,
    ) -> ArtifactManifest:
        return ArtifactManifest(
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

    def is_valid(self, cache: bool | None = None) -> bool:
        artifact = self.artifact()
        manifest = self._manifest()
        if manifest is None or not Path(artifact.path).exists():
            return False
        return self._is_valid(manifest, artifact, cache)

    def load(self, cache: bool | None = None) -> T:
        use_cache = self.registry.should_cache(self.derivative, cache)
        artifact = self.artifact()
        if use_cache:
            manifest = self._manifest()
            if manifest and Path(artifact.path).exists() and self._is_valid(manifest, artifact, cache):
                return self.derivative.load(self.ctx, artifact)
            if self._is_protected_artifact(artifact) and not self.ctx.option('_allow_protected_overwrite', False):
                if (
                        manifest
                        and self.derivative.can_reindex_protected_artifact(self.ctx, artifact, manifest, cache)
                ):
                    value = self.derivative.load(self.ctx, artifact)
                    self.registry.write_manifest(artifact.manifest_path, self._build_manifest(value, cache))
                    return value
                raise ProtectedArtifactError(self.derivative.name, artifact.path)

        value = self.derivative.build(self.ctx)
        if not use_cache:
            return value

        artifact = self.artifact(mkdir=True)
        self.derivative.save(self.ctx, artifact, value)
        self.registry.write_manifest(artifact.manifest_path, self._build_manifest(value, cache))
        return self.derivative.load(self.ctx, artifact)


class InputHandle(NodeHandle[T]):
    """An input plus its bound context for one concrete load request."""

    def __init__(
            self,
            input_: Input[T],
            ctx: DerivativeContext,
    ):
        super().__init__(input_, ctx)

    input: Input[T]

    @property
    def input(self) -> Input[T]:
        return self.node

    def load(self, cache: bool | None = None) -> T:
        return self.input.load(self.ctx)


class DerivativeRegistry:
    """Registry and resolver for dependency nodes bound to one pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self._nodes: dict[str, DependencyNode[Any]] = {}

    def register(self, node: DependencyNode[Any]) -> None:
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

    def _get_node(self, name: str) -> DependencyNode[Any]:
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
    ) -> NodeHandle[Any]:
        node = self._get_node(name)
        ctx = DerivativeContext(self, self._resolve_state(state, **extra_state), options or {})
        if isinstance(node, Derivative):
            return DerivativeHandle(node, ctx)
        elif isinstance(node, Input):
            return InputHandle(node, ctx)
        else:
            raise TypeError(f"{node=}: Unsupported node type {type(node)!r}")

    def load(
            self,
            name: str,  # Registered node name.
            cache: bool | None = None,  # Explicit cache override for derivative loads.
            state: dict[str, Any] | None = None,  # Base state for this node instance.
            options: dict[str, Any] | None = None,
            **extra_state,  # Additional state overrides merged on top of ``state``.
    ):
        handle = self.resolve(name, state=state, options=options, **extra_state)
        return handle.load(cache)

    def should_cache(self, derivative: Derivative, cache: bool | None) -> bool:
        if cache is not None:
            return cache
        override = getattr(self.pipeline, 'cache_policy_overrides', {}).get(derivative.name)
        if override is not None:
            return override not in ('off', False)
        return derivative.cache_policy != CachePolicy.DISABLED_BY_DEFAULT

    def is_cache_artifact(self, path: str | Path) -> bool:
        cache_dir = Path(self.pipeline.get('cache-dir')).resolve()
        artifact_path = Path(path).resolve()
        try:
            artifact_path.relative_to(cache_dir)
        except ValueError:
            return False
        else:
            return True

    def manifest_path(self, path: str | Path) -> str:
        artifact_path = Path(path)
        if self.is_cache_artifact(artifact_path):
            return f"{artifact_path}{MANIFEST_SUFFIX}"

        manifest_root = Path(self.pipeline.get('cache-dir')) / 'manifests'
        resolved_path = artifact_path.resolve()
        for label, root in (
                ('deriv-dir', Path(self.pipeline.get('deriv-dir'))),
                ('root', Path(self.pipeline.get('root'))),
        ):
            try:
                relative = resolved_path.relative_to(root.resolve())
            except ValueError:
                continue
            return str(manifest_root / label / relative) + MANIFEST_SUFFIX

        digest = hashlib.sha1(str(resolved_path).encode()).hexdigest()
        return str(manifest_root / 'external' / digest) + MANIFEST_SUFFIX

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
            node: DependencyNode[Any],  # Node whose dependencies are being fingerprinted.
            ctx: DerivativeContext,  # Bound state/options for the current load.
            cache: bool | None,  # Explicit cache override propagated to dependencies.
    ) -> dict[str, Any]:
        out = {}
        for dep in node.dependencies(ctx):
            options = dict(ctx.options)
            if dep.options:
                options.update(dep.options(ctx))
            key = dep.label or dep.name
            if key in out:
                raise RuntimeError(f"Duplicate dependency label {key!r} for node {node.name!r}")
            handle = self.resolve(
                dep.name,
                state=self._resolve_state(ctx.state, **(dep.state(ctx) if dep.state else {})),
                options=options,
            )
            out[key] = handle.describe_dependency(cache)
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
