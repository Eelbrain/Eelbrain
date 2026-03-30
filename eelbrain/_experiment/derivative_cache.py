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
- :class:`DerivativeRegistry` resolves dependencies and validates cached
  artifacts via sidecar manifests.

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
import logging
from pathlib import Path
import shutil
from typing import Any, Generic, TypeVar

import mne

T = TypeVar('T')
MANIFEST_SUFFIX = '.manifest.json'
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CACHE_LABEL = 'artifact'
MAX_CACHE_LABEL_LEN = 96


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


class ProtectedArtifactError(RuntimeError):
    """Refuse to replace a stale user-managed artifact automatically."""

    def __init__(
            self,
            derivative: str,
            path: Path,
            message: str | None = None,
            instructions: str | None = None,
    ):
        self.derivative = derivative
        self.path = str(path)
        self.message = message
        self.instructions = instructions
        text = message or (
            f"Existing artifact for derivative {derivative!r} at {self.path!r} does not match "
            "the current settings and was not replaced automatically."
        )
        if instructions:
            text += f" {instructions}"
        super().__init__(text)


def _slug_cache_path_part(value: Any) -> str:
    text = str(value)
    out = []
    pending_sep = False
    for char in text:
        if char.isalnum():
            out.append(char.lower())
            pending_sep = False
        elif out and not pending_sep:
            out.append('-')
            pending_sep = True
    return ''.join(out).strip('-') or DEFAULT_CACHE_LABEL


def _simple_cache_label(key: dict[str, Any]) -> str | None:
    parts = []
    for name, value in key.items():
        if value in (None, ''):
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{name}-{value}")
    if not parts:
        return None
    return '_'.join(parts)


def cache_artifact_path(
        cache_dir: str | Path,
        node_name: str,
        key: dict[str, Any],
        suffix: str,
        *,
        label: str | None = None,
) -> Path:
    """Build a cache path from derivative identity."""
    key_json = json.dumps(key, sort_keys=True, separators=(',', ':'), default=str)
    key_hash = hashlib.sha1(key_json.encode()).hexdigest()[:12]
    label_text = label or _simple_cache_label(key) or DEFAULT_CACHE_LABEL
    label_slug = _slug_cache_path_part(label_text)[:MAX_CACHE_LABEL_LEN].rstrip('-') or DEFAULT_CACHE_LABEL
    node_slug = _slug_cache_path_part(node_name)
    return Path(cache_dir) / node_slug / key_hash[:2] / f"{label_slug}_key-{key_hash}{suffix}"


@dataclass
class DerivativeContext:
    """Bound state/options environment for resolving one node request."""

    registry: DerivativeRegistry
    state: dict[str, Any]
    options: dict[str, Any]

    def option(self, key: str, default=None):
        return self.options.get(key, default)

    def get(self, key: str):
        return self.state[key]

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
        return self.registry.load(name, state=merged_state, options=self.options if options is None else options)


@dataclass(frozen=True)
class Dependency:
    """One edge in the derivative graph.

    Parameters
    ----------
    name
        Registered node name for the dependency. This can refer to either a
        registered :class:`Derivative` or a registered :class:`Input`.
    label
        Optional manifest label for this dependency edge. In most cases this
        can be omitted, and the dependency name is used as the manifest key.
        Set ``label`` when the same dependency name can appear more than once
        in one dependency set with different state or options, so each edge
        has a stable distinct name in the dependency manifest.
    state
        Optional state updates for this dependency. The mapping is merged on
        top of the parent state before resolving the dependency.
    options
        Optional complete options mapping for this dependency. When omitted,
        the dependency inherits the parent load options. When provided, this
        mapping replaces the parent load options for this dependency request.

    Notes
    -----
    :class:`Dependency` is a declarative description of one graph edge. It
    does not resolve anything by itself; the registry evaluates it when the
    parent node is fingerprinted or loaded.
    """

    name: str
    label: str | None = None
    state: dict[str, Any] | None = None
    options: dict[str, Any] | None = None


class DependencyNode(Generic[T]):
    """Base class for all registered dependency-graph nodes.

    Subclasses participate in the cache graph by providing three pieces of
    information:

    - a stable ``name`` used for registry lookup and dependency edges
    - a dependency list describing which other nodes this node needs
    - a fingerprint describing whether the current request still matches the
      artifact represented by this node

    Most users should subclass :class:`Input` or :class:`Derivative` rather
    than subclassing :class:`DependencyNode` directly.
    """

    name: str

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        """Describe other registered nodes that this node depends on.

        Override this when the node needs other inputs or derivatives.
        Return one :class:`Dependency` per edge in the dependency graph.

        Implementations should:

        - return only direct dependencies needed by this node
        - use ``state`` / ``options`` overrides on :class:`Dependency` when a
          dependency should be resolved with different state or options than
          the current request
        - keep the result deterministic for a given ``ctx``, since dependency
          manifests are part of cache validation

        The default implementation returns no dependencies.
        """
        return ()

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        """Describe the current version of this node's own inputs/settings.

        Subclasses must override this method.

        The fingerprint should contain all non-dependency information that
        makes this node's result stale, such as configuration parameters,
        source-file metadata, or definition snapshots. It should not duplicate
        dependency manifests; those are tracked separately through
        :meth:`dependencies`.
        """
        raise NotImplementedError

    def dependency_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        """Describe how this node should appear when used as a dependency.

        Override this only when the dependency-facing fingerprint should be
        smaller or different from the full artifact fingerprint. The default
        implementation reuses :meth:`fingerprint`.
        """
        return self.fingerprint(ctx)


class Input(DependencyNode[T]):
    """Base class for non-cacheable external inputs.

    Inputs represent artifacts that are not built by the cache system itself,
    such as raw source files, manually curated metadata, or external logs.
    They still participate in dependency manifests through
    :meth:`DependencyNode.fingerprint`.
    """

    def load(self, ctx: DerivativeContext):
        """Materialize the input for the current request.

        Subclasses must override this method.

        Implementations should load and return the input value described by
        ``ctx``. They should not write manifests or perform cache management;
        the registry handles that around derivatives only.
        """
        raise NotImplementedError


class Derivative(DependencyNode[T]):
    """Base class for one cache-managed derived artifact.

    A derivative is a named artifact family that can be keyed, built, loaded,
    saved, and validated. Subclasses normally override:

    - :meth:`path` to choose the artifact location
    - :meth:`fingerprint` to describe non-dependency staleness inputs
    - :meth:`build` to compute the artifact
    - :meth:`load` / :meth:`save` to serialize the artifact

    More specialized subclasses may additionally override :meth:`key`,
    :meth:`should_cache`, :meth:`validate`,
    :meth:`can_reindex_protected_artifact`, or :meth:`provenance`.
    """

    key_fields: tuple[str, ...]
    cache_policy: CachePolicy = CachePolicy.REQUIRED
    cache_suffix: str | None = None
    cache_log_level: int | None = logging.DEBUG
    version: int = 1

    def cache_label(self, ctx: DerivativeContext) -> str | None:
        """Return an optional readable label for the default cache path.

        Override this for cache-managed derivatives that benefit from a more
        readable stem than the default label derived from simple scalar key
        fields. The label is only for readability; the hash derived from
        :meth:`key` remains authoritative.
        """
        return _simple_cache_label(self.key(ctx))

    def cache_log_name(self, ctx: DerivativeContext) -> str:
        """Return the human-readable artifact name for cache log messages."""
        return self.name.replace('-', ' ')

    def cache_log_path(self, ctx: DerivativeContext, path: Path) -> str:
        """Return the displayed artifact path for cache log messages."""
        return ctx.registry.describe_artifact_path(path)

    def log_cache_hit(self, ctx: DerivativeContext, path: Path) -> None:
        """Emit the standard cache-hit message for this derivative."""
        self._log_cache_event(ctx, path, "Load cached")

    def log_cache_build(self, ctx: DerivativeContext, path: Path) -> None:
        """Emit the standard cache-build message for this derivative."""
        self._log_cache_event(ctx, path, "Build")

    def log_cache_reindex(self, ctx: DerivativeContext, path: Path) -> None:
        """Emit the standard manifest-reindex message for this derivative."""
        self._log_cache_event(ctx, path, "Reindex")

    def _log_cache_event(
            self,
            ctx: DerivativeContext,
            path: Path,
            action: str,
    ) -> None:
        if self.cache_log_level is None:
            return
        ctx.registry.log.log(self.cache_log_level, "%s %s: %s", action, self.cache_log_name(ctx), self.cache_log_path(ctx, path))

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> Path:
        """Return the concrete artifact path for this request.

        Subclasses may either override this method explicitly or declare
        :attr:`cache_suffix` to use the default cache path scheme based on the
        derivative name and :meth:`key`.

        The returned path identifies where the artifact itself lives. For
        artifacts inside ``cache-dir``, the registry writes the manifest next
        to the artifact. For public/export artifacts outside ``cache-dir``,
        the registry mirrors the manifest under ``cache-dir/manifests``.

        Implementations should derive the path from semantic state/options
        only. They should not perform dependency traversal or cache logic.
        ``mkdir`` is provided for compatibility; callers should not rely on
        ``path()`` to create directories.
        """
        if self.cache_suffix is None:
            raise NotImplementedError
        path = cache_artifact_path(
            ctx.registry.cache_dir,
            self.name,
            ctx.registry.canonicalize(self.key(ctx)),
            self.cache_suffix,
            label=self.cache_label(ctx),
        )
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        """Return the normalized key that identifies this artifact instance.

        Override this only when the default ``key_fields`` subset is not
        sufficient. The key should include only the state needed to distinguish
        different concrete artifacts for this derivative.
        """
        return canonical_state_subset(ctx.state, self.key_fields)

    def should_cache(
            self,
            ctx: DerivativeContext,
            cache: bool | None,
    ) -> bool:
        """Decide whether this request should read/write a cached artifact.

        Override this when cache behavior depends on the current request, not
        just on :attr:`cache_policy`. The return value should be deterministic
        for a given ``ctx`` and explicit ``cache`` override.
        """
        if cache is not None:
            return cache
        return self.cache_policy != CachePolicy.DISABLED_BY_DEFAULT

    def build(self, ctx: DerivativeContext) -> T:
        """Compute the in-memory artifact value for this request.

        Subclasses must override this method.

        Implementations should do the actual work of the derivative, typically
        by loading dependencies through ``ctx.load(...)`` and transforming
        them into the resulting artifact value. They should return the
        in-memory value and leave serialization to :meth:`save`.
        """
        raise NotImplementedError

    def load(self, ctx: DerivativeContext, path: Path) -> T:
        """Load a previously saved artifact from ``path``.

        Subclasses must override this method.

        Implementations should read ``path`` and return the in-memory value
        produced by :meth:`build`. They should not perform staleness checks;
        the registry calls this only after handling cache validation.
        """
        raise NotImplementedError

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: T,
    ) -> None:
        """Serialize ``value`` to ``path``.

        Subclasses must override this method.

        Implementations should write the artifact in a form that
        :meth:`load` can reconstruct. They should only write the artifact
        itself; the registry manages manifest files separately.
        """
        raise NotImplementedError

    def validate(
            self,
            ctx: DerivativeContext,
            path: Path,
            manifest: ArtifactManifest,
    ) -> bool:
        """Perform derivative-specific cache validation.

        Override this when path-local checks are needed in addition to key,
        fingerprint, and dependency validation, for example schema checks on
        the saved file itself. Return ``True`` when the artifact is still
        usable for the current request.
        """
        return True

    def can_reindex_protected_artifact(
            self,
            ctx: DerivativeContext,
            path: Path,
            manifest: ArtifactManifest,
            cache: bool | None = None,
    ) -> bool:
        """Allow manifest-only refresh for protected non-cache artifacts.

        Override this only for derivatives that store artifacts outside
        ``cache-dir`` and can safely keep the existing artifact even when the
        manifest is stale. Return ``True`` only when the artifact at ``path``
        is still valid and it is safe to rewrite just the manifest.
        """
        return False

    def provenance(
            self,
            ctx: DerivativeContext,
            value: T,
    ) -> dict[str, Any]:
        """Record optional extra provenance for the saved artifact.

        Override this to add human-readable or debugging metadata to the
        manifest after a successful build, for example dimensions, sample
        counts, or export destinations. The return value should be JSON-like
        and deterministic for the saved artifact.
        """
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

    def __init__(
            self,
            derivative: Derivative[T],
            ctx: DerivativeContext,
    ):
        super().__init__(derivative, ctx)
        self.artifact_path = Path(self.node.path(self.ctx))
        self.manifest_path = Path(self.registry.manifest_path(self.artifact_path))

    node: Derivative[T]
    artifact_path: Path
    manifest_path: Path

    def _is_protected_artifact(self) -> bool:
        return (
            self.artifact_path.exists()
            and not self.registry.is_cache_artifact(self.artifact_path)
        )

    def key(self) -> dict[str, Any]:
        return self.node.key(self.ctx)

    def _manifest(self) -> ArtifactManifest | None:
        return self.registry.read_manifest(self.manifest_path)

    def _is_valid(
            self,
            manifest: ArtifactManifest,
            cache: bool | None = None,
    ) -> bool:
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            return False
        if manifest.derivative != self.node.name:
            return False
        if manifest.derivative_version != self.node.version:
            return False
        if manifest.key != self.key():
            return False
        if manifest.fingerprint != self.current_fingerprint():
            return False
        if manifest.dependencies != self.dependency_fingerprints(cache):
            return False
        if not self.node.validate(self.ctx, self.artifact_path, manifest):
            return False
        return True

    def _build_manifest(
            self,
            value: T,
            cache: bool | None = None,
    ) -> ArtifactManifest:
        return ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=self.node.name,
            derivative_version=self.node.version,
            key=self.key(),
            fingerprint=self.current_fingerprint(),
            dependencies=self.dependency_fingerprints(cache),
            cache_policy=self.node.cache_policy.value,
            software={
                'eelbrain_cache_schema': str(MANIFEST_SCHEMA_VERSION),
                'mne': mne.__version__,
            },
            provenance=self.registry.canonicalize(self.node.provenance(self.ctx, value)),
        )

    def is_valid(self, cache: bool | None = None) -> bool:
        manifest = self._manifest()
        if manifest is None or not self.artifact_path.exists():
            return False
        return self._is_valid(manifest, cache)

    def load(self, cache: bool | None = None) -> T:
        use_cache = self.node.should_cache(self.ctx, cache)
        if use_cache:
            manifest = self._manifest()
            if manifest and self.artifact_path.exists() and self._is_valid(manifest, cache):
                self.node.log_cache_hit(self.ctx, self.artifact_path)
                return self.node.load(self.ctx, self.artifact_path)
            if self._is_protected_artifact() and not self.ctx.option('_allow_protected_overwrite', False):
                if (
                        manifest
                        and self.node.can_reindex_protected_artifact(self.ctx, self.artifact_path, manifest, cache)
                ):
                    value = self.node.load(self.ctx, self.artifact_path)
                    self.node.log_cache_reindex(self.ctx, self.artifact_path)
                    self.registry.write_manifest(self.manifest_path, self._build_manifest(value, cache))
                    return value
                if self.ctx.option('_allow_protected_reindex', False):
                    value = self.node.load(self.ctx, self.artifact_path)
                    self.node.log_cache_reindex(self.ctx, self.artifact_path)
                    self.registry.write_manifest(self.manifest_path, self._build_manifest(value, cache))
                    return value
                raise ProtectedArtifactError(self.node.name, self.artifact_path)

        if use_cache:
            self.node.log_cache_build(self.ctx, self.artifact_path)
        value = self.node.build(self.ctx)
        if not use_cache:
            return value

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.node.save(self.ctx, self.artifact_path, value)
        self.registry.write_manifest(self.manifest_path, self._build_manifest(value, cache))
        return self.node.load(self.ctx, self.artifact_path)


class InputHandle(NodeHandle[T]):
    """An input plus its bound context for one concrete load request."""

    def __init__(
            self,
            input_: Input[T],
            ctx: DerivativeContext,
    ):
        super().__init__(input_, ctx)

    node: Input[T]

    def load(self, cache: bool | None = None) -> T:
        return self.node.load(self.ctx)


class DerivativeRegistry:
    """Registry and resolver for dependency nodes bound to one experiment root."""

    def __init__(self, root: str | Path, log: logging.Logger):
        self.root = Path(root)
        self.log = log
        self.deriv_dir = self.root / 'derivatives'
        self.cache_dir = self.deriv_dir / 'eelbrain' / 'cache'
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

    def describe_artifact_path(self, path: str | Path) -> str:
        artifact_path = Path(path)
        if self.is_cache_artifact(artifact_path):
            return str(artifact_path.relative_to(self.cache_dir))
        try:
            return str(artifact_path.relative_to(self.root))
        except ValueError:
            return str(artifact_path)

    def _dependency_handles(
            self,
            node: DependencyNode[Any],
            ctx: DerivativeContext,
    ) -> list[tuple[Dependency, NodeHandle[Any]]]:
        out = []
        keys = set()
        for dep in node.dependencies(ctx):
            key = dep.label or dep.name
            if key in keys:
                raise RuntimeError(f"Duplicate dependency label {key!r} for node {node.name!r}")
            keys.add(key)
            options = ctx.options if dep.options is None else dep.options
            handle = self.resolve(
                dep.name,
                state=self._resolve_state(ctx.state, **(dep.state or {})),
                options=options,
            )
            out.append((dep, handle))
        return out

    @staticmethod
    def _tree_mapping_text(mapping: dict[str, Any] | None, *, values: bool = True) -> str | None:
        if not mapping:
            return None
        items = list(mapping.items())
        max_items = 6 if values else 8
        if values:
            parts = [f"{key}={value!r}" for key, value in items[:max_items]]
        else:
            parts = [str(key) for key, _ in items[:max_items]]
        if len(items) > max_items:
            parts.append(f"+{len(items) - max_items}")
        return ', '.join(parts)

    def _tree_request_id(self, handle: NodeHandle[Any]) -> str:
        return json.dumps({
            'name': handle.node.name,
            'state': self.canonicalize(handle.state),
            'options': self.canonicalize(handle.options),
        }, sort_keys=True, separators=(',', ':'))

    @staticmethod
    def _tree_line_width(max_line_length: int | None) -> int:
        if max_line_length is None:
            return shutil.get_terminal_size(fallback=(100, 24)).columns
        if max_line_length < 16:
            raise ValueError(f"{max_line_length=}: needs to be at least 16")
        return max_line_length

    @staticmethod
    def _clip_tree_segment(text: str, available: int) -> str:
        if len(text) <= available:
            return text
        if available <= 1:
            return '…'
        return text[:available - 1].rstrip() + '…'

    def _format_tree_line(
            self,
            first_prefix: str,
            continuation_prefix: str,
            segments: list[str],
            max_line_length: int,
    ) -> list[str]:
        lines = []
        current = first_prefix
        current_prefix = first_prefix
        for segment in segments:
            if len(current) + len(segment) <= max_line_length:
                current += segment
                continue
            if current != current_prefix:
                lines.append(current)
                current = continuation_prefix
                current_prefix = continuation_prefix
            available = max_line_length - len(current)
            current += self._clip_tree_segment(segment.lstrip(), available)
        lines.append(current)
        return lines

    def dependency_tree(
            self,
            name: str,
            state: dict[str, Any] | None = None,
            options: dict[str, Any] | None = None,
            max_line_length: int | None = None,
            **extra_state,
    ) -> str:
        """Format one resolved dependency request as an ASCII tree.

        Parameters
        ----------
        name
            Registered node name.
        state
            State for resolving the request.
        options
            Options for resolving the request.
        max_line_length
            Maximum line length for the formatted tree. By default, infer the
            current terminal width and wrap long node descriptions onto
            continuation lines.
        ...
            Additional state overrides merged on top of ``state``.
        """
        root = self.resolve(name, state=state, options=options, **extra_state)
        line_width = self._tree_line_width(max_line_length)
        seen = set()
        lines = []

        def append_node(
                handle: NodeHandle[Any],
                dep: Dependency | None,
                prefix: str,
                is_last: bool,
        ) -> None:
            first_prefix = ''
            continuation_prefix = '    '
            if dep is not None:
                first_prefix = prefix + ('└── ' if is_last else '├── ')
                continuation_prefix = prefix + ('    ' if is_last else '│   ')
            parts = []
            if dep and dep.label and dep.label != dep.name:
                parts.append(f"{dep.label} -> ")
            parts.append(handle.node.name)
            if isinstance(handle, DerivativeHandle):
                parts.append(' [derivative]')
                key_text = self._tree_mapping_text(self.canonicalize(handle.key()))
                if key_text:
                    parts.append(f" {{{key_text}}}")
            else:
                parts.append(' [input]')
            if dep and dep.state:
                state_text = self._tree_mapping_text(self.canonicalize(dep.state))
                if state_text:
                    parts.append(f" [state: {state_text}]")
            option_source = handle.options if dep is None else dep.options
            if option_source:
                option_text = self._tree_mapping_text(self.canonicalize(option_source), values=False)
                if option_text:
                    parts.append(f" [options: {option_text}]")

            request_id = self._tree_request_id(handle)
            if request_id in seen:
                parts.append(' [seen]')
                lines.extend(self._format_tree_line(first_prefix, continuation_prefix, parts, line_width))
                return

            seen.add(request_id)
            lines.extend(self._format_tree_line(first_prefix, continuation_prefix, parts, line_width))
            children = self._dependency_handles(handle.node, handle.ctx)
            child_prefix = continuation_prefix if dep is not None else prefix
            for i, (child_dep, child_handle) in enumerate(children):
                append_node(child_handle, child_dep, child_prefix, i == len(children) - 1)

        append_node(root, None, '', True)
        return '\n'.join(lines)

    def is_cache_artifact(self, path: str | Path) -> bool:
        cache_dir = self.cache_dir.resolve()
        artifact_path = Path(path).resolve()
        try:
            artifact_path.relative_to(cache_dir)
        except ValueError:
            return False
        else:
            return True

    def manifest_path(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        if self.is_cache_artifact(artifact_path):
            return Path(f"{artifact_path}{MANIFEST_SUFFIX}")

        manifest_root = self.cache_dir / 'manifests'
        resolved_path = artifact_path.resolve()
        for label, root in (
                ('deriv-dir', self.deriv_dir),
                ('root', self.root),
        ):
            try:
                relative = resolved_path.relative_to(root.resolve())
            except ValueError:
                continue
            return Path(f"{manifest_root / label / relative}{MANIFEST_SUFFIX}")

        digest = hashlib.sha1(str(resolved_path).encode()).hexdigest()
        return Path(f"{manifest_root / 'external' / digest}{MANIFEST_SUFFIX}")

    def dependency_fingerprints(
            self,
            node: DependencyNode[Any],  # Node whose dependencies are being fingerprinted.
            ctx: DerivativeContext,  # Bound state/options for the current load.
            cache: bool | None,  # Explicit cache override propagated to dependencies.
    ) -> dict[str, Any]:
        out = {}
        for dep, handle in self._dependency_handles(node, ctx):
            key = dep.label or dep.name
            out[key] = handle.describe_dependency(cache)
        return out

    def read_manifest(self, path: str | Path) -> ArtifactManifest | None:
        manifest_path = Path(path)
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text())
        return ArtifactManifest.from_dict(data)

    def write_manifest(self, path: str | Path, manifest: ArtifactManifest) -> None:
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

    root = Path(root)
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


def canonical_state_subset(
        state: dict[str, Any],
        fields: tuple[str, ...],
) -> dict[str, Any]:
    return {
        key: DerivativeRegistry.canonicalize(state[key])
        for key in fields
    }
