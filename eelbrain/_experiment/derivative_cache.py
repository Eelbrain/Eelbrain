"""Derivative-oriented cache primitives for :mod:`eelbrain._experiment`.

The goal of this module is providing a general interface for building
branching data analysis pipelines that allow caching intermediate results at
each processing step.

The cache is organized around a graph of registered dependency nodes.

- :class:`Input` declares one source node in the dependency graph.
- :class:`Derivative` declares one computed node in the dependency graph.
- :class:`Request` binds one node to specific analysis parameters given in a
  global pipeline state and local options.
- :class:`DerivativeRegistry` resolves dependencies and validates cached
  artifacts via sidecar manifests.

Within that framework:

- A derivative is *keyed* by the subset of pipeline state that selects which
  artifact it represents (such as subject, preprocessing applied, ...) and
  relevant analysis options that affect building of the artifact.
- A derivative is *built* by computing that artifact from the current pipeline
  state and options.
- A derivative is *serialized* by saving the in-memory result to disk and
  loading it back later.
- A derivative is *fingerprinted* by recording the normalized settings and
  inputs that determine whether a cached artifact is still valid.

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
from collections.abc import Callable

import mne
import numpy as np

from .._data_obj import Factor, Interaction, Var
from .configuration import Configuration
from .pathing import CACHE_DIR, DERIV_DIR

T = TypeVar('T')
MANIFEST_SUFFIX = '.manifest.json'
MANIFEST_SCHEMA_VERSION = 2
DEFAULT_CACHE_LABEL = 'artifact'
MAX_CACHE_LABEL_LEN = 96
# Hash prefix length for artifact path components (hex chars, i.e. 48 bits).
# Collisions at the path level are handled gracefully by the disambiguation
# sidecar, so a shorter prefix is acceptable in exchange for more readable paths.
CACHE_KEY_HASH_LEN = 12
CACHE_DISAMBIGUATION_SUFFIX = '.disambiguation.json'
ALLOW_PROTECTED_OVERWRITE = 'allow_protected_overwrite'


class CachePolicy(str, Enum):
    """How strongly the cache engine should prefer persistence for a derivative.

    REQUIRED
        Caching is always on. The artifact is always written and read from disk.
        Use for derivatives that are expensive to compute.
    OPTIONAL
        Caching is on by default but can be disabled by passing ``cache=False``
        to the caller. Use for derivatives that are cheap to rebuild but
        benefit from persistence across sessions.
    DISABLED_BY_DEFAULT
        Caching is opt-in: disabled unless the caller passes ``cache=True``
        explicitly. Use for derived values that are fast to compute or that
        should not accumulate on disk without explicit intent.
    NEVER
        Caching is permanently disabled and the derivative has no artifact
        path or manifest. Used by :class:`UncachedDerivative` subclasses that
        are always rebuilt on every request.
    """

    REQUIRED = 'required'
    OPTIONAL = 'optional'
    DISABLED_BY_DEFAULT = 'disabled_by_default'
    NEVER = 'never'


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
    artifact_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        # Unknown keys from newer schema versions are silently dropped so that
        # manifests written by a newer Eelbrain can still be read by an older
        # one. Structural validity (schema_version) is checked at the call site.
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


def _slug_cache_path_part(text: str) -> str:
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


def _cache_key_json(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(',', ':'), default=str)


def _full_cache_key_digest(key: dict[str, Any]) -> str:
    return hashlib.sha1(_cache_key_json(key).encode()).hexdigest()


def _cache_disambiguation_path(path: str | Path) -> Path:
    return Path(f"{Path(path)}{CACHE_DISAMBIGUATION_SUFFIX}")


def _disambiguated_cache_artifact_path(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


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
        Optional child request options passed to the target node when the
        dependency is resolved. Keys must be declared by the target node and
        can refer to either standard options from
        ``target.OPTION_DEFAULTS`` or view-only options from
        ``target.VIEW_OPTION_DEFAULTS``. The registry splits the mapping into
        artifact-affecting ``Request.options`` and post-load
        ``Request.view_options`` for the child request.
    view
        Optional dependency view name. Use this when the dependency should be
        validated through a reduced or specialized dependency fingerprint
        rather than the dependency node's default dependency-facing
        fingerprint. The named view is resolved through
        :meth:`DependencyNode.dependency_fingerprint` with ``view`` forwarded
        to the target node, which can expose multiple narrower fingerprint
        projections of the same artifact. This affects cache validation only;
        the dependency artifact itself is not loaded through this view.

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
    view: str | None = None


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
    The following attributes are relevant for both subclasses:

    Attributes
    ----------
    name
        Stable registry name for this node. Must be unique across all
        registered nodes and must not change once artifacts have been cached
        under that name.
    OPTION_DEFAULTS
        Options that affect how this node's artifact is built, or its cache
        identity. Keys declare the option names; values are their defaults.
        Options are node-local: they apply to this node only and do not
        propagate to dependencies unless the node explicitly forwards them
        via :meth:`Request.options_for`.
    VIEW_OPTION_DEFAULTS
        Options that only shape the returned value after the artifact has
        been built or loaded. They do not affect cache identity and are not
        forwarded to dependencies.
    fixed_state
        State entries that this node always forces to specific values,
        regardless of caller-provided state. Applied by the registry on top
        of any incoming state when this node is resolved, so conflicting
        caller state is overridden. Use this when the node name encodes a
        specific state value — e.g. a raw-processing node that always
        implies ``state['raw'] == raw_name`` — so that :class:`Dependency`
        declarations targeting this node need not redundantly repeat a
        ``state`` override. The counterpart to :attr:`Derivative.key_fields`:
        where ``key_fields`` declares polymorphism over state keys,
        ``fixed_state`` pins them.
    """

    name: str
    OPTION_DEFAULTS: dict[str, Any] = {}
    VIEW_OPTION_DEFAULTS: dict[str, Any] = {}
    fixed_state: dict[str, Any] = {}

    @classmethod
    def declared_options(cls) -> set[str]:
        """Return all option names declared by this node."""
        options = set(cls.OPTION_DEFAULTS)
        options.update(cls.VIEW_OPTION_DEFAULTS)
        return options

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        """Describe other registered nodes that this node depends on.

        Override this when the node needs other inputs or derivatives.
        Return one :class:`Dependency` per edge in the dependency graph.

        Implementations should:

        - return only direct dependencies needed by this node
        - prefer :attr:`fixed_state` on the target node over ``state``
          overrides on :class:`Dependency` for corrections that always apply
        - use ``state`` / ``options`` overrides on :class:`Dependency` for
          context-specific variations that the target node cannot anticipate
        - keep the result deterministic for a given ``ctx``, since dependency
          manifests are part of cache validation

        The default implementation returns no dependencies.
        """
        return ()

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        """Information that uniquely identifies an artifact as valid.

        Subclasses must override this method.

        The fingerprint should contain all non-dependency request information
        that makes this node's result stale, usually request ``state`` /
        ``options`` plus any configuration definitions or external file
        metadata that the node treats as part of its own validity.  All
        fingerprint values are passed through
        :meth:`~DerivativeRegistry.canonicalize` before storage, so
        :class:`~eelbrain._experiment.configuration.Configuration` objects,
        Eelbrain data types (:class:`~eelbrain.Var`, :class:`~eelbrain.Factor`,
        …), and other types handled by :meth:`~DerivativeRegistry.canonicalize`
        can be included directly without pre-serialization.  It should not
        duplicate dependency manifests; those are tracked separately through
        :meth:`dependencies`.

        For :class:`Derivative` subclasses, this is distinct from
        :meth:`Derivative.key`: the key chooses the cache slot/path for the
        artifact, while the fingerprint records the richer validity
        information that must match for that slot to be considered current.
        """
        raise NotImplementedError

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        """Describe how this node should appear when used as a dependency.

        Override this only when the dependency-facing fingerprint should be
        smaller or different from the full artifact fingerprint. ``view`` can
        be used to expose multiple named dependency fingerprints for the same
        node. The default implementation ignores ``view`` and reuses
        :meth:`fingerprint`.

        One use case is managed intermediate inputs, such as ICA files, where a
        change in excluded components invalidates dependent nodes, but does not
        signal an invalid ICA file artifact.
        """
        return self.fingerprint(ctx)

    def dependency_fingerprint_quick(self, ctx: Request, view: str | None = None) -> dict[str, Any] | None:
        """Cheap proxy for :meth:`dependency_fingerprint`, used as first-pass cache validity check.

        If this equals the stored quick fingerprint, the full
        :meth:`dependency_fingerprint` comparison is skipped and the dependency
        is considered unchanged. Return ``None`` to always fall back to the full
        comparison.

        Required invariant (one-directional): if this returns equal dicts,
        :meth:`dependency_fingerprint` must also be equal. The converse need
        not hold — a quick-fingerprint change may be spurious, in which case
        the full fingerprint will still match and the cache remains valid.
        """
        return None

    def load_view(
            self,
            ctx: Request,
            view: str,
    ):
        """Load one named dependency/user-facing view for this node.

        Override this to expose data views that bypass the :meth:`load` method.
        Views can be loaded through ``request.load(view=...)``.
        """
        raise ValueError(f"{self.name!r} does not define load view {view!r}")


class Input(DependencyNode[T]):
    """Base class for non-cacheable external inputs.

    Inputs represent artifacts that are not built by the cache system itself,
    such as raw source files, manually curated metadata, or external logs.
    They still participate in dependency manifests through
    :meth:`DependencyNode.fingerprint`.
    """

    def load(self, ctx: Request):
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
    - :meth:`fingerprint` to describe non-dependency request
      state/options/definitions that determine staleness;
      the helper method :meth:`standard_fingerprint` covers the common
      fingerprint shape without open-coding option handling.
    - :meth:`build` to compute the artifact
    - :meth:`load` / :meth:`save` to serialize the artifact

    The standard subclass contract is:

    - declare ``OPTION_DEFAULTS`` and ``VIEW_OPTION_DEFAULTS`` (inherited
      from :class:`DependencyNode`) for options affecting artifact identity
      or post-load shaping respectively
    - implement :meth:`build` to construct the artifact representation from
      state plus options
    - implement :meth:`load` / :meth:`save` for that artifact representation
    - optionally implement :meth:`apply_view_options` to transform the loaded
      artifact into the final return value



    Attributes
    ----------
    key_fields
        ``ctx.state`` fields that define the default artifact key and cache
        label when :meth:`key` is not overridden. Subclasses that override
        :meth:`key` may still use ``key_fields`` as a reference list of
        relevant state keys in their :meth:`fingerprint` logic, but that
        usage is a convention, not a framework contract.
    cache_policy
        Default caching mode used by :meth:`should_cache`.
    cache_suffix
        File suffix for the default :meth:`path` implementation. Leave
        ``None`` when overriding :meth:`path` directly.
    cache_log_level
        Log level for standard cache hit/build messages. Set to ``None``
        to suppress them.
    version
        Derivative-local schema version recorded in manifests. Increment
        when the serialization format changes incompatibly.
    """

    # ``ctx.state`` fields that define the default artifact key. Override
    # :meth:`key` directly when artifact identity is not just a state subset.
    key_fields: tuple[str, ...] = ()
    # Default cache behavior when callers do not pass an explicit cache=...
    cache_policy: CachePolicy = CachePolicy.REQUIRED
    # File suffix for the default :meth:`path` implementation.
    cache_suffix: str | None = None
    # Log level for standard cache hit/build messages. Set to None to silence.
    cache_log_level: int | None = logging.DEBUG
    # Derivative-local version recorded in manifests for compatibility checks.
    version: int = 1

    def cache_label(self, ctx: Request) -> str | None:
        """Return an optional readable label for the default cache path.

        Override this for cache-managed derivatives that benefit from a more
        readable stem than the default label derived from simple scalar key
        fields. The label is only for readability; the hash derived from
        :meth:`key` remains authoritative.
        """
        label_key = self.key(ctx) if not self.key_fields else canonical_state_subset(ctx.state, self.key_fields)
        return _simple_cache_label(label_key)

    def cache_log_path(self, ctx: Request, path: Path) -> str:
        """Return the displayed artifact path for cache log messages."""
        return ctx.registry.describe_artifact_path(path)

    def log_cache_hit(self, ctx: Request, path: Path) -> None:
        """Emit the standard cache-hit message for this derivative."""
        self._log_cache_event(ctx, path, "Load cached")

    def log_cache_build(self, ctx: Request, path: Path) -> None:
        """Emit the standard cache-build message for this derivative."""
        self._log_cache_event(ctx, path, "Build")

    def _log_cache_event(
            self,
            ctx: Request,
            path: Path,
            action: str,
    ) -> None:
        if self.cache_log_level is None:
            return
        ctx.registry.log.log(self.cache_log_level, "%s %s: %s", action, self.name, self.cache_log_path(ctx, path))

    def path(self, ctx: Request) -> Path:
        """Return the concrete artifact path for this request.

        Subclasses may either override this method explicitly or declare
        :attr:`cache_suffix` to use the default cache path scheme based on the
        derivative name and :meth:`key`.

        The returned path identifies where the artifact itself lives. For
        artifacts inside ``cache-dir``, the registry writes the manifest next
        to the artifact. For public/export artifacts outside ``cache-dir``,
        the registry mirrors the manifest under ``cache-dir/manifests``.

        Implementations should derive the path from semantic state/options
        only. They should not perform dependency traversal, create directories,
        or perform cache logic.
        """
        if self.cache_suffix is None:
            raise NotImplementedError
        key = ctx.registry.canonicalize(self.key(ctx))
        key_hash = _full_cache_key_digest(key)[:CACHE_KEY_HASH_LEN]
        label = self.cache_label(ctx) or DEFAULT_CACHE_LABEL
        label_slug = _slug_cache_path_part(label)[:MAX_CACHE_LABEL_LEN].rstrip('-') or DEFAULT_CACHE_LABEL
        node_slug = _slug_cache_path_part(self.name)
        return ctx.registry.cache_dir / node_slug / key_hash[:2] / f"{label_slug}_key-{key_hash}{self.cache_suffix}"

    def key(self, ctx: Request) -> dict[str, Any]:
        """The key used to generate a unique path for this artifact.

        Override this only when the default ``key_fields`` subset is not
        sufficient. The default implementation uses the configured
        ``key_fields`` subset of state and adds an ``options`` entry when the
        derivative's declared options also contribute to artifact identity.

        The key is used to resolve the artifact path and should stay focused
        on cache address/identity. It is narrower than :meth:`fingerprint`,
        which records the fuller set of non-dependency request
        state/options/definitions that make an existing artifact stale.
        """
        key = canonical_state_subset(ctx.state, self.key_fields)
        options = ctx.registry.canonicalize(ctx.options)
        if options:
            key['options'] = options
        return ctx.registry.canonicalize(key)

    def should_cache(
            self,
            cache: bool | None,
    ) -> bool:
        """Decide whether this request should read/write a cached artifact"""
        if self.cache_policy == CachePolicy.NEVER:
            return False
        if cache is not None:
            return cache
        return self.cache_policy != CachePolicy.DISABLED_BY_DEFAULT

    def build(self, ctx: Request) -> T:
        """Compute the artifact value for this request.

        Subclasses must override this method.

        Implementations should do the actual work of the derivative, typically
        by loading dependencies through ``ctx.load(...)`` and transforming
        them into the artifact representation saved by :meth:`save`.
        View-only shaping belongs in :meth:`apply_view_options`.
        """
        raise NotImplementedError

    def load(self, ctx: Request, path: Path) -> T:
        """Load the saved artifact representation from ``path``.

        Subclasses must override this method.

        Implementations should read ``path`` and return the in-memory value
        produced by :meth:`build`, before any view-only shaping. They should
        not perform staleness checks; the registry calls this only after
        handling cache validation.
        """
        raise NotImplementedError

    def save(
            self,
            ctx: Request,
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

    def artifact_metadata(
            self,
            ctx: Request,
            value: T,
    ) -> dict[str, Any]:
        """Return serializer metadata for this artifact.

        Override this when :meth:`load` or named views need small pieces of
        metadata about the serialized artifact beyond the main fingerprint,
        such as file lists or saved selection arrays.
        """
        return {}

    def apply_view_options(
            self,
            ctx: Request,
            value: T,
    ) -> T:
        """Apply view-only options to a built or loaded artifact value.

        Override this only when some options should affect the returned value
        without changing cache identity. The default implementation returns
        ``value`` unchanged.
        """
        return value

    def standard_fingerprint(
            self,
            ctx: Request,
            *,
            state_fields: tuple[str, ...] | None = None,
            definitions: dict[str, Any] | None = None,
            extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the common ``state`` / ``definitions`` / ``options`` shape.

        Parameters
        ----------
        ctx
            Bound request for the current derivative load.
        state_fields
            State keys from ``ctx.state`` that materially define the artifact.
            These keys are extracted directly from the request state and
            embedded under the ``"state"`` key after canonicalization.
            Default: ``self.key_fields``.
        definitions
            Optional configuration definitions embedded under the
            ``"definitions"`` key.  Values are passed through
            :meth:`~DerivativeRegistry.canonicalize`, so
            :class:`~eelbrain._experiment.configuration.Configuration` objects
            and other Eelbrain types can be passed directly without
            pre-serialization.  Use this for stable snapshots such as epoch
            definitions, test definitions, pipe definitions, or variable
            collections that explain what the derivative is building.
        extra
            Optional additional fingerprint fields merged at the top level
            after canonicalization. Use this for small derivative-specific
            values that do not fit naturally under ``state`` or
            ``definitions``.

        Returns
        -------
        fingerprint
            Canonicalized fingerprint mapping containing the selected
            ``state`` snapshot, any ``definitions`` snapshot, the current request
            ``options`` when non-empty, and any extra top-level fields.

        Notes
        -----
        This is the default helper for derivatives whose fingerprints mostly
        consist of semantic state, serialized configuration definitions,
        request options, and a few derivative-specific scalar values. If a
        relevant value is not a direct entry in ``ctx.state``, it should
        usually go into ``definitions`` or ``extra`` instead of being
        smuggled into ``state_fields`` through a custom mapping.
        """
        if state_fields is None:
            state_fields = self.key_fields
        out = {'state': canonical_state_subset(ctx.state, state_fields)}
        if definitions:
            out['definitions'] = ctx.registry.canonicalize(definitions)
        options = ctx.registry.canonicalize(ctx.options)
        if options:
            out['options'] = options
        if extra:
            out.update(ctx.registry.canonicalize(extra))
        return out


class UncachedDerivative(Derivative[T]):
    """Base class for derived values that should never persist to the cache.

    :attr:`cache_policy` is :attr:`CachePolicy.NEVER`, so no artifact path or
    manifest is created and ``build`` is called on every request.
    Subclasses must not declare ``key_fields`` or override :meth:`key`.
    """

    cache_policy = CachePolicy.NEVER
    cache_log_level = None

    def key(self, ctx: Request) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} is uncached; key() must not be called")

    def path(self, ctx: Request) -> Path:
        raise NotImplementedError(f"{type(self).__name__} is uncached; path() must not be called")


def _dep_entry_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare one describe_dependency entry with quick-fingerprint shortcut."""
    for key in ('name', 'kind', 'view'):
        if stored.get(key) != current.get(key):
            return False
    stored_quick = stored.get('quick_fingerprint')
    current_quick = current.get('quick_fingerprint')
    if stored_quick is not None and current_quick is not None and stored_quick == current_quick:
        return True
    if stored.get('fingerprint') != current.get('fingerprint'):
        return False
    return _dependencies_match(stored.get('dependencies', {}), current.get('dependencies', {}))


def _dependencies_match(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare dependency manifests, using quick fingerprints as a first-pass shortcut."""
    if stored.keys() != current.keys():
        return False
    return all(_dep_entry_matches(stored[k], current[k]) for k in stored)


class Request(Generic[T]):
    """One bound request for data from the dependency graph.

    Parameters
    ----------
    node
        The registered dependency-graph node for this request. This can be an
        :class:`Input` or a :class:`Derivative`.
    registry
        The registry that resolved this request and that should be used for
        dependency resolution, manifest handling, and canonicalization.
    state
        Fully resolved pipeline state for this request. State defines the
        semantic graph context for the request and propagates automatically to
        nested dependency loads unless a dependency overrides it.
    options
        Options declared by ``node`` that affect how the artifact is built.
        These options are node-local: they affect the target node, and only
        reach deeper dependencies when that node explicitly forwards them.
    view_options
        Options declared by ``node`` that only shape the returned value after
        the underlying artifact or input has been loaded.
    controls
        Explicit request controls that are not node options, such as
        permission to overwrite protected artifacts.

    Notes
    -----
    Derivative-only members such as :meth:`key`, :attr:`artifact_path`,
    :attr:`manifest_path`, and :meth:`is_valid` are available on the same
    object. They raise :class:`TypeError` when the request targets an input.
    """

    def __init__(
            self,
            node: DependencyNode[T],
            registry: DerivativeRegistry,
            state: dict[str, Any],
            options: dict[str, Any],
            view_options: dict[str, Any],
            controls: frozenset[str] | set[str] | tuple[str, ...] = (),
    ):
        self.node = node
        self.registry = registry
        self.root = registry.root
        self.state = state
        self.options = options
        self.view_options = view_options
        self.controls = frozenset(controls)
        self._key: dict[str, Any] | None = None
        self._base_artifact_path: Path | None = None
        self._artifact_path: Path | None = None
        self._manifest_path: Path | None = None
        self._artifact_metadata: dict[str, Any] | None = None
        # Populated during build() to enforce declared-dependency discipline.
        self._build_deps: dict[str, Callable] | None = None
        if isinstance(node, Derivative) and node.cache_policy != CachePolicy.NEVER:
            self._key = node.key(self)
            self._base_artifact_path = Path(node.path(self))
            self._artifact_path = Path(self.registry.resolve_cache_artifact_path(self._base_artifact_path, self._key))
            self._manifest_path = Path(self.registry.manifest_path(self._artifact_path))

    def has_control(self, control: str) -> bool:
        """Return whether this request includes one explicit execution control."""
        return control in self.controls

    def options_for(self, name: str, *keys: str, **overrides) -> dict[str, Any]:
        """Build a valid option mapping for a dependency node.

        ``keys`` names options from the current request that should be
        forwarded to the child request. ``overrides`` sets child options
        directly. Omitted options are not inherited.
        """
        node = self.registry._get_node(name)
        allowed = node.declared_options()
        forwarded = {}
        undeclared = sorted({*keys, *overrides}.difference(allowed))
        if undeclared:
            joined = ', '.join(repr(key) for key in undeclared)
            raise TypeError(f"{name!r} does not declare option(s): {joined}")
        for key in keys:
            if key in self.options:
                forwarded[key] = self.options[key]
            elif key in self.view_options:
                forwarded[key] = self.view_options[key]
            else:
                raise KeyError(f"Current request for {self.node.name!r} has no option {key!r} to forward to {name!r}")
        forwarded.update(overrides)
        return forwarded

    def dependency_fingerprints(self, cache: bool | None = None) -> dict[str, Any]:
        """Return the current dependency manifest fragment for this request."""
        return self.registry.dependency_fingerprints(self.node, self, cache)

    def current_fingerprint(self) -> dict[str, Any]:
        """Return the canonical current fingerprint for this node request."""
        return self.registry.canonicalize(self.node.fingerprint(self))

    def current_dependency_fingerprint(self, view: str | None = None) -> dict[str, Any]:
        """Return the canonical dependency-facing fingerprint for this request."""
        return self.registry.canonicalize(self.node.dependency_fingerprint(self, view))

    def current_dependency_fingerprint_quick(self, view: str | None = None) -> dict[str, Any] | None:
        """Return the canonical quick proxy fingerprint, or ``None`` if not supported."""
        result = self.node.dependency_fingerprint_quick(self, view)
        if result is None:
            return None
        return self.registry.canonicalize(result)

    def _resolve_quick_fingerprint(
            self,
            name: str,
            *,
            state: dict[str, Any] | None = None,
            options: dict[str, Any] | None = None,
            view: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a dependency node and return its canonicalized quick fingerprint."""
        handle = self.registry.resolve(name, state={**self.state, **(state or {})}, options=options)
        result = handle.node.dependency_fingerprint_quick(handle, view)
        if result is None:
            return None
        return self.registry.canonicalize(result)

    def describe_dependency(
            self,
            cache: bool | None = None,
            view: str | None = None,
    ) -> dict[str, Any]:
        """Describe this request for inclusion in another node's manifest."""
        out = {
            'name': self.node.name,
            'fingerprint': self.current_dependency_fingerprint(view),
            'dependencies': self.dependency_fingerprints(cache),
        }
        quick = self.current_dependency_fingerprint_quick(view)
        if quick is not None:
            out['quick_fingerprint'] = quick
        if view is not None:
            out['view'] = view
        if isinstance(self.node, Derivative) and self.node.cache_policy != CachePolicy.NEVER:
            out['kind'] = 'derivative'
            out['key'] = self.key()
            out['manifest'] = str(self.manifest_path)
        else:
            out['kind'] = 'input'
        return out

    def _require_derivative(self) -> Derivative[T]:
        if isinstance(self.node, Derivative):
            return self.node
        raise TypeError(f"Request for input {self.node.name!r} has no derivative artifact state")

    @property
    def base_artifact_path(self) -> Path:
        """Base artifact path before any cache-path disambiguation."""
        self._require_derivative()
        assert self._base_artifact_path is not None
        return self._base_artifact_path

    @property
    def artifact_path(self) -> Path:
        """Resolved artifact path for a derivative request."""
        self._require_derivative()
        assert self._artifact_path is not None
        return self._artifact_path

    @property
    def manifest_path(self) -> Path:
        """Resolved manifest path for a derivative request."""
        self._require_derivative()
        assert self._manifest_path is not None
        return self._manifest_path

    def key(self) -> dict[str, Any]:
        """Return the normalized derivative key for this request."""
        self._require_derivative()
        assert self._key is not None
        return self._key

    @property
    def artifact_metadata(self) -> dict[str, Any]:
        """Serializer metadata for the current derivative artifact, if any."""
        if self._artifact_metadata is not None:
            return self._artifact_metadata
        if not isinstance(self.node, Derivative):
            return {}
        manifest = self._manifest()
        if manifest is None:
            return {}
        return manifest.artifact_metadata

    def _manifest(self) -> ArtifactManifest | None:
        return self.registry.read_manifest(self.manifest_path)

    def _is_valid(
            self,
            manifest: ArtifactManifest,
            cache: bool | None = None,
    ) -> bool:
        derivative = self._require_derivative()
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            return False
        if manifest.derivative != derivative.name:
            return False
        if manifest.derivative_version != derivative.version:
            return False
        if manifest.key != self.key():
            return False
        if manifest.fingerprint != self.current_fingerprint():
            return False
        if not _dependencies_match(manifest.dependencies, self.dependency_fingerprints(cache)):
            return False
        return True

    def is_valid(self, cache: bool | None = None) -> bool:
        """Return whether the current derivative request already has a valid artifact."""
        self._require_derivative()
        manifest = self._manifest()
        if manifest is None or not self.artifact_path.exists():
            return False
        return self._is_valid(manifest, cache)

    def load_artifact(self, cache: bool | None = None) -> T:
        """Load or build the underlying derivative artifact without view shaping."""
        derivative = self._require_derivative()
        use_cache = derivative.should_cache(cache)
        if use_cache:
            manifest = self._manifest()
            if manifest and self.artifact_path.exists() and self._is_valid(manifest, cache):
                self._artifact_metadata = manifest.artifact_metadata
                derivative.log_cache_hit(self, self.artifact_path)
                return derivative.load(self, self.artifact_path)
            if self.artifact_path.exists() and not self.registry.is_cache_artifact(self.artifact_path) and not self.has_control(ALLOW_PROTECTED_OVERWRITE):
                raise ProtectedArtifactError(derivative.name, self.artifact_path)

        if use_cache:
            derivative.log_cache_build(self, self.artifact_path)

        def _make_dep_loader(dep: Dependency):
            dep_state = {**self.state, **dep.state} if dep.state else self.state

            def _load():
                # dep.view is resolved here on the fresh request (not via the
                # calling build()'s ctx), so it bypasses build-discipline
                # enforcement and correctly loads named views such as 'bads'.
                return self.registry.resolve(dep.name, state=dep_state, options=dep.options).load(cache=cache, view=dep.view)
            return _load

        self._build_deps = {
            dep.label or dep.name: _make_dep_loader(dep) for dep in self.node.dependencies(self)
        }
        try:
            artifact = derivative.build(self)
        finally:
            self._build_deps = None
        artifact_metadata = self.registry.canonicalize(derivative.artifact_metadata(self, artifact))
        self._artifact_metadata = artifact_metadata
        if not use_cache:
            return artifact

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        derivative.save(self, self.artifact_path, artifact)
        manifest = ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=derivative.name,
            derivative_version=derivative.version,
            key=self.key(),
            fingerprint=self.current_fingerprint(),
            dependencies=self.dependency_fingerprints(cache),
            cache_policy=derivative.cache_policy.value,
            software={
                'eelbrain_cache_schema': str(MANIFEST_SCHEMA_VERSION),
                'mne': mne.__version__,
            },
            artifact_metadata=artifact_metadata,
        )
        self.registry.write_manifest(self.manifest_path, manifest)
        self._artifact_metadata = manifest.artifact_metadata
        return derivative.load(self, self.artifact_path)

    def load(
            self,
            name: str | None = None,
            cache: bool | None = None,
            state: dict[str, Any] | None = None,
            options: dict[str, Any] | None = None,
            *,
            view: str | None = None,
            controls: frozenset[str] | set[str] | tuple[str, ...] = (),
    ):
        """Load this request, or a named dependency relative to this request's state.

        Parameters
        ----------
        name
            Registered node name to load as a dependency. When omitted or
            ``None``, the current request itself is materialized.
        cache
            Override the node's default :class:`CachePolicy`. ``True`` forces
            a cache read/write; ``False`` bypasses the cache and always
            rebuilds. ``None`` (the default) defers to the node's own policy.
            Only meaningful for :class:`Derivative` nodes; ignored for
            :class:`Input` nodes.
        state
            State overrides merged on top of the current request's state
            before resolving the dependency. Only valid when ``name`` is
            given.
        options
            Option overrides for the target node. Only valid when ``name`` is
            given; use :meth:`options_for` to forward options from the current
            request.
        view
            Named view to load instead of the node's default return value.
            Resolved through :meth:`DependencyNode.load_view` on the target
            node.
        controls
            Explicit execution controls forwarded to the nested load. Controls
            are never inherited implicitly; pass them only when the nested load
            genuinely requires them.

        Returns
        -------
        value
            The loaded artifact after view-option shaping, or the result of
            :meth:`DependencyNode.load_view` when ``view`` is given.

        Notes
        -----
        When loading the current request (``name`` is ``None``), only
        ``cache`` and ``view`` are accepted; passing ``state``, ``options``,
        or ``controls`` raises :class:`TypeError`.
        """
        if isinstance(name, str):
            if self._build_deps is not None:
                if name not in self._build_deps:
                    declared = sorted(self._build_deps)
                    raise RuntimeError(f"{self.node.name!r}.build() called ctx.load({name!r}) which is not a declared dependency. Declared: {declared}")
                if cache is not None or view is not None or state is not None or options is not None or controls:
                    raise TypeError(f"{self.node.name!r}.build() passed overrides to ctx.load({name!r}); declare cache, view, state, and options on the Dependency instead")
                return self._build_deps[name]()
            return self.registry.resolve(
                name,
                state={**self.state, **(state or {})},
                options=options,
                controls=controls,
            ).load(cache=cache, view=view)

        if state is not None or options is not None or controls:
            raise TypeError("Request.load() without a dependency name only accepts cache and view overrides")
        if view is not None:
            # Temporarily suspend build-discipline enforcement so that load_view
            # implementations (which are part of the node contract, not raw build
            # logic) are free to load dependencies with any options they need.
            build_deps = self._build_deps
            self._build_deps = None
            try:
                return self.node.load_view(self, view)
            finally:
                self._build_deps = build_deps
        if isinstance(self.node, Input):
            return self.node.load(self)

        derivative = self._require_derivative()
        artifact = self.load_artifact(cache)
        return derivative.apply_view_options(self, artifact)


class DerivativeRegistry:
    """Registry and resolver for dependency nodes bound to one experiment root."""

    def __init__(self, root: str | Path, log: logging.Logger):
        self.root = Path(root)
        self.log = log
        self.deriv_dir = self.root / DERIV_DIR
        self.cache_dir = self.root / CACHE_DIR
        self._nodes: dict[str, DependencyNode[Any]] = {}

    def register(self, node: DependencyNode[Any]) -> None:
        if node.name in self._nodes:
            raise RuntimeError(f"Dependency node {node.name!r} already registered")
        if not isinstance(node, (Derivative, Input)):
            raise TypeError(f"Unsupported node type: {type(node)!r}")
        self._nodes[node.name] = node

    def _get_node(self, name: str) -> DependencyNode[Any]:
        try:
            return self._nodes[name]
        except KeyError:
            raise RuntimeError(f"Unknown dependency {name!r}") from None

    def resolve(
            self,
            name: str,
            state: dict[str, Any] | None = None,
            options: dict[str, Any] | None = None,
            controls: frozenset[str] | set[str] | tuple[str, ...] = (),
    ) -> Request[Any]:
        node = self._get_node(name)
        node_options = {} if options is None else dict(options)
        undeclared = set(node_options).difference(node.declared_options())
        if undeclared:
            keys = ', '.join(repr(key) for key in sorted(undeclared))
            raise TypeError(f"{node.name!r} got undeclared option(s): {keys}")
        options = dict(node.OPTION_DEFAULTS)
        view_options = dict(node.VIEW_OPTION_DEFAULTS)
        for key, value in node_options.items():
            if key in options:
                options[key] = value
            else:
                view_options[key] = value
        return Request(
            node=node,
            registry=self,
            state={**(state or {}), **node.fixed_state},
            options=options,
            view_options=view_options,
            controls=controls,
        )

    def describe_artifact_path(self, path: str | Path) -> str:
        artifact_path = Path(path)
        if self.is_cache_artifact(artifact_path):
            return str(artifact_path.relative_to(self.cache_dir))
        try:
            return str(artifact_path.relative_to(self.root))
        except ValueError:
            return str(artifact_path)

    def _read_cache_disambiguation(self, path: str | Path) -> dict[str, str]:
        sidecar_path = _cache_disambiguation_path(path)
        if not sidecar_path.exists():
            return {}
        data = json.loads(sidecar_path.read_text())
        if not isinstance(data, dict):
            return {}
        return {str(key): value for key, value in data.items() if isinstance(value, str)}

    def _write_cache_disambiguation(self, path: str | Path, data: dict[str, str]) -> None:
        sidecar_path = _cache_disambiguation_path(path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(data, sort_keys=True, indent=2))

    def resolve_cache_artifact_path(
            self,
            path: str | Path,
            key: dict[str, Any],
    ) -> Path:
        artifact_path = Path(path)
        if not self.is_cache_artifact(artifact_path):
            return artifact_path

        canonical_key = self.canonicalize(key)
        digest = _full_cache_key_digest(canonical_key)
        mapping = self._read_cache_disambiguation(artifact_path)
        suffix = mapping.get(digest)
        if suffix is not None:
            return _disambiguated_cache_artifact_path(artifact_path, suffix)

        if not artifact_path.exists():
            return artifact_path

        manifest = self.read_manifest(self.manifest_path(artifact_path))
        if manifest is None or self.canonicalize(manifest.key) == canonical_key:
            return artifact_path

        used_suffixes = set(mapping.values())
        index = 1
        while True:
            suffix = f"_alt-{index}"
            if suffix not in used_suffixes:
                break
            index += 1

        mapping[digest] = suffix
        self._write_cache_disambiguation(artifact_path, mapping)
        return _disambiguated_cache_artifact_path(artifact_path, suffix)

    def _dependency_handles(
            self,
            node: DependencyNode[Any],
            ctx: Request,
    ) -> list[tuple[Dependency, Request[Any]]]:
        out = []
        keys = set()
        for dep in node.dependencies(ctx):
            key = dep.label or dep.name
            if key in keys:
                raise RuntimeError(f"Duplicate dependency label {key!r} for node {node.name!r}")
            keys.add(key)
            handle = self.resolve(
                dep.name,
                state={**ctx.state, **(dep.state or {})},
                options=dep.options,
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

    def _tree_request_id(self, handle: Request[Any], view: str | None = None) -> str:
        return json.dumps({
            'name': handle.node.name,
            'state': self.canonicalize(handle.state),
            'options': self.canonicalize({**handle.options, **handle.view_options}),
            'view': view,
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
        """
        root = self.resolve(name, state=state, options=options)
        line_width = self._tree_line_width(max_line_length)
        seen = set()
        lines = []

        def append_node(
                handle: Request[Any],
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
            if isinstance(handle.node, Derivative):
                if handle.node.cache_policy == CachePolicy.NEVER:
                    parts.append(' [uncached]')
                else:
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
            if dep and dep.view:
                parts.append(f" [view: {dep.view}]")
            option_source = {**handle.options, **handle.view_options} if dep is None else dep.options
            if option_source:
                option_text = self._tree_mapping_text(self.canonicalize(option_source), values=False)
                if option_text:
                    parts.append(f" [options: {option_text}]")

            request_id = self._tree_request_id(handle, dep.view if dep else None)
            if request_id in seen:
                parts.append(' [seen]')
                lines.extend(self._format_tree_line(first_prefix, continuation_prefix, parts, line_width))
                return

            seen.add(request_id)
            lines.extend(self._format_tree_line(first_prefix, continuation_prefix, parts, line_width))
            children = self._dependency_handles(handle.node, handle)
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
            ctx: Request,  # Bound state/options for the current load.
            cache: bool | None,  # Explicit cache override propagated to dependencies.
    ) -> dict[str, Any]:
        out = {}
        for dep, handle in self._dependency_handles(node, ctx):
            key = dep.label or dep.name
            out[key] = handle.describe_dependency(cache, dep.view)
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
        """Recursively convert ``value`` to a JSON-serializable, stable form.

        The output is suitable for fingerprints, keys, and manifests: dicts are
        sorted by key, sets are sorted, numpy scalars are unwrapped, and
        :class:`Path` objects become strings. Unrecognized types fall back to
        ``repr()``.

        Domain-specific types handled here (:class:`~eelbrain.Var`,
        :class:`~eelbrain.Factor`, :class:`~eelbrain.Interaction`,
        :class:`Configuration`) are an intentional coupling between the cache
        kernel and the Eelbrain data model; they allow fingerprints and keys to
        contain arbitrary data objects without callers having to pre-serialize
        them.
        """
        if isinstance(value, Var):
            return DerivativeRegistry.canonicalize(value.x.tolist())
        if isinstance(value, (Factor, Interaction)):
            return DerivativeRegistry.canonicalize(list(value))
        if isinstance(value, Configuration):
            return DerivativeRegistry.canonicalize(value._as_dict())
        if isinstance(value, np.ndarray):
            return DerivativeRegistry.canonicalize(value.tolist())
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


def file_fingerprint(
        root: str | Path,
        path: str | Path,
        kind: str,
        digest: bool = False,
        metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
