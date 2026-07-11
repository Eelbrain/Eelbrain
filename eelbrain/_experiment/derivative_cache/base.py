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

Dependency edges are validated for key-field coverage (see
:meth:`DerivativeRegistry._check_edge_key_coverage`): when a dependency's output
is sensitive to a state field, the depending node must either key on that field
too, or pin it on the edge (the way aggregation over a field is expressed).
Otherwise the parent would silently share one cache slot across different values
of that field.

Artifacts inside ``cache-dir`` keep sidecar manifests and can be rebuilt
automatically when they go stale. Artifacts stored elsewhere are treated as
user-managed outputs: their manifests are mirrored under the owning
derivative's node directory in the cache (e.g. an ICA at
``derivatives/mne/sub-01/...`` → ``cache-dir/<node-name>/sub-01/...``) and they
are not overwritten without an explicit opt-in from the caller.

Garbage collection
------------------

Stale artifacts left behind by changed definitions or removed nodes are found
and removed by :mod:`.garbage_collection`; the entry points are
:meth:`DerivativeRegistry.scan_cache` and :meth:`DerivativeRegistry.collect`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import pickle
import re
import shutil
import tomllib
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from uuid import uuid4
import warnings

import mne
import numpy as np

from ..._data_obj import Factor, Interaction, NDVar, Var
from ..configuration import Configuration
from ..logging import CacheInvalidation, diff_invalidation
from ..pathing import CACHE_DIR, DERIV_DIR, LOG_DIR

if TYPE_CHECKING:
    from .garbage_collection import GCReport

T = TypeVar('T')
MANIFEST_SUFFIX = '.manifest.json'
MANIFEST_SCHEMA_VERSION = 2
DEFAULT_CACHE_LABEL = 'artifact'
MAX_CACHE_LABEL_LEN = 96
# Characters that are illegal in path components on Windows (plus control chars).
# They are replaced with '-' rather than deleted so that semantically meaningful
# operators in labels (e.g. the '>' in a test named 'a>v') leave operands separated
# in the readable slug instead of being silently merged ('a>v' -> 'a-v', not 'av').
CACHE_PATH_UNSAFE = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')
CACHE_PATH_UNSAFE_REPLACEMENT = '-'
# Hash prefix length for artifact path components (hex chars, i.e. 64 bits).
# Collisions at the path level are handled gracefully by the disambiguation sidecar
CACHE_KEY_HASH_LEN = 16
CACHE_DISAMBIGUATION_SUFFIX = '.disambiguation.json'
ALLOW_PROTECTED_OVERWRITE = 'allow_protected_overwrite'

# Sentinel for undeclared key_fields
UNSET: Any = object()


def _toml_string(value: str) -> str:
    """Write a TOML basic string; JSON string escaping is compatible here."""
    return json.dumps(value, ensure_ascii=False)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via a temporary file so an interrupted write cannot leave a partial file."""
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _read_warning_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError:
        return []
    entries = data.get('warning', [])
    warnings_ = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = entry.get('item')
        category = entry.get('category')
        message = entry.get('message')
        if isinstance(item, str) and isinstance(category, str) and isinstance(message, str):
            warnings_.append({'item': item, 'category': category, 'message': message})
    return warnings_


def _write_warning_log(path: Path, header: str, warnings_: list[dict[str, str]]) -> None:
    lines = [
        'version = 1',
        f'header = {_toml_string(header.rstrip())}',
        '',
    ]
    for warning in warnings_:
        lines.extend((
            '[[warning]]',
            f'item = {_toml_string(warning["item"])}',
            f'category = {_toml_string(warning["category"])}',
            f'message = {_toml_string(warning["message"])}',
            '',
        ))
    path.write_text('\n'.join(lines))


class CachePolicy(str, Enum):
    """Whether artifacts for a derivative persist to the cache.

    REQUIRED
        The artifact is always written to and read from disk.
    NEVER
        Caching is permanently disabled: the derivative has no artifact path
        or manifest and is rebuilt on every request. Set by
        :class:`UncachedDerivative` subclasses, or at registration time for
        derivatives configured as uncached (e.g. ``Pipeline.cache_inv``).
    """

    REQUIRED = 'required'
    NEVER = 'never'


@dataclass(frozen=True)
class InputFingerprint:
    """Portable description of one non-derivative input."""

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
    # Canonical (state, options) that reproduce this request through
    # DerivativeRegistry.resolve(), enabling offline revalidation (garbage
    # collection). None (as opposed to {}) means the manifest predates these
    # fields and cannot be revalidated offline. Not compared by
    # compare_manifests, so adding them does not invalidate existing caches.
    resolve_state: dict[str, Any] | None = None
    resolve_options: dict[str, Any] | None = None

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
            reason: str | None = None,
    ):
        self.derivative = derivative
        self.path = str(path)
        self.message = message
        self.instructions = instructions
        self.reason = reason
        text = message or (
            f"Existing artifact for derivative {derivative!r} at {self.path!r} does not match "
            "the current settings and was not replaced automatically."
        )
        if instructions:
            text += f" {instructions}"
        super().__init__(text)


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


@dataclass(frozen=True)
class OptionSpec:
    """Declared specification for one node option.

    Use as a value in :attr:`DependencyNode.key_options` /
    :attr:`DependencyNode.view_options` in place of a plain default to
    normalize and validate the option value when a request is resolved. This
    happens before the cache key is computed, so equivalent spellings share
    one cached artifact and invalid values fail before build.

    The declared ``default`` itself is exempt from normalization and
    validation (checked by identity), so a ``None`` placeholder default does
    not need to satisfy ``type`` or ``literal``.

    Parameters
    ----------
    default
        Value used when the caller does not set the option.
    type
        Required type (or types) for the value. Booleans only match an
        explicit ``bool`` declaration, never ``int``, so an option declared as
        ``bool`` rejects ``1`` even though ``1 == True``.
    literal
        Exact allowed values. Matching is type-strict, so ``True`` in
        ``literal`` does not admit ``1``.
    normalize
        Called as ``normalize(value)`` before validation; the return
        value replaces the option value for the whole request (key,
        fingerprint, and build all see the normalized value). Must be
        idempotent, since child requests are normalized again when they are
        resolved.
    """

    default: Any
    type: type | tuple[type, ...] | None = None
    literal: tuple[Any, ...] | None = None
    normalize: Callable[[Any], Any] | None = None

    def validated(self, ctx: Request, name: str, value: Any) -> Any:
        """Normalize and validate one option value for ``ctx``."""
        if value is None and self.default is None:
            return value
        elif self.type and isinstance(value, self.type):
            return value
        elif self.normalize:
            value = self.normalize(value)
            if self.type:
                assert isinstance(value, self.type)
            return value
        elif self.type:
            if not isinstance(self.type, tuple):
                return self.type(value)
            # bool subclasses int; require an explicit bool declaration so that 1 does not pass as True
            valid = bool in self.type if isinstance(value, bool) else isinstance(value, self.type)
            if not valid:
                expected = ' | '.join(t.__name__ for t in self.type)
                raise TypeError(f"{ctx.node.name!r} option {name}={value!r}: expected {expected}, got {type(value).__name__}")
        elif self.literal is not None:
            if not any(value is allowed or (type(value) is type(allowed) and value == allowed) for allowed in self.literal):
                raise ValueError(f"{ctx.node.name!r} option {name}={value!r}: must be one of {self.literal}")
        return value


def _option_default(spec: Any) -> Any:
    """Default value of one ``key_options`` / ``view_options`` entry (plain default or :class:`OptionSpec`)."""
    return spec.default if isinstance(spec, OptionSpec) else spec


def _cache_entity_dir(key: dict[str, Any]) -> Path:
    """Directory grouping cache artifacts by BIDS subject/session entities.

    Per-subject derivatives are grouped under ``sub-<subject>/ses-<session>``
    (session omitted when absent), giving a browsable, BIDS-like layout and
    natural directory fan-out per subject. Group-level derivatives, which do not
    key on a subject, are grouped under ``group``.

    Parameters
    ----------
    key
        Canonical derivative key (see :meth:`Request.key`); subject and session
        are read from it so that grouping honors the node's actual identity
        fields rather than incidental pipeline state.
    """
    subject = key.get('subject')
    if subject in (None, ''):
        return Path('group')
    parts = [f'sub-{subject}']
    session = key.get('session')
    if session not in (None, ''):
        parts.append(f'ses-{session}')
    return Path(*(CACHE_PATH_UNSAFE.sub(CACHE_PATH_UNSAFE_REPLACEMENT, part) for part in parts))


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
        top of the parent state before resolving the dependency. This is also
        how a dependency's key field is satisfied when the parent does not key
        on it — pinning it here (the way aggregation over a field is expressed)
        covers it for the edge-coverage check.
    options
        Optional child request options passed to the target node when the
        dependency is resolved. Keys must be declared by the target node and
        can refer to either standard options from
        ``target.key_options`` or view-only options from
        ``target.view_options``. The registry splits the mapping into
        artifact-affecting ``Request.options`` and post-load
        ``Request.view_options`` for the child request.
    view
        Optional data view to load from the dependency. When this dependency
        is requested through ``ctx.load(...)`` inside the dependent node, the
        named view is loaded instead of the dependency's normal return value.
        The view name is also forwarded to
        :meth:`DependencyNode.dependency_fingerprint` so the manifest describes
        the same data view.

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
    key_fields
        State fields whose values determine this node's output. Reading any
        state field outside ``key_fields`` / ``fixed_state`` during
        :meth:`~Derivative.build`, :meth:`fingerprint`, or
        :meth:`dependencies` raises :class:`RuntimeError`. Use
        :meth:`override_key_fields` when a node's key depends on the state
        dynamically. An explicit empty tuple opts out (the
        node manages its own identity via a :meth:`Derivative.key` override).
        Mandatory for :class:`Input` nodes (only ``()`` opts out). ``key_fields``
        also feeds the edge-coverage check (see
        :meth:`DerivativeRegistry._check_edge_key_coverage`): every field a
        dependency keys on must be pinned on the edge or covered by the parent's
        key fields.
    key_options
        Options that affect how this node's artifact is built, and that enter
        the cache key. Keys declare the option names; values are their
        defaults, or :class:`OptionSpec` declarations that additionally
        normalize and validate the value when a request is resolved. Options
        are node-local: they apply to this node only and do not propagate to
        dependencies unless the node explicitly forwards them via
        :meth:`Request.options_for`. Which options actually enter the key
        can be narrowed per request via :meth:`override_key_options`.
    view_options
        Options that only shape the returned value after the artifact has
        been built or loaded. They do not affect cache identity. Like
        :attr:`key_options`, values are plain defaults or :class:`OptionSpec`
        declarations.
    fixed_state
        State entries that this node always forces to specific values,
        regardless of caller-provided state. Applied by the registry on top
        of any incoming state when this node is resolved, so conflicting
        caller state is overridden. Use this when the node name encodes a
        specific state value — e.g. a raw-processing node that always
        implies ``state['raw'] == raw_name`` — so that :class:`Dependency`
        declarations targeting this node need not redundantly repeat a
        ``state`` override. The counterpart to :attr:`key_fields`: where
        ``key_fields`` declares polymorphism over state keys, ``fixed_state``
        pins them.
    """

    name: str
    key_fields: tuple[str, ...] = UNSET
    key_options: dict[str, Any] = {}
    view_options: dict[str, Any] = {}
    fixed_state: dict[str, Any] = {}

    @classmethod
    def declared_options(cls) -> set[str]:
        """Return all option names declared by this node."""
        return {*cls.key_options, *cls.view_options}

    def override_key_fields(self, ctx: Request) -> tuple[str, ...] | None:
        """Dynamically choose the state fields in the cache key for this request.

        Override this when a node's identity fields depend on the request — for
        example a source/sensor node that only keys on ``src`` when it is
        in source space. Return the field names, or ``None`` to use the static
        :attr:`key_fields`.
        """
        return None

    def _get_key_fields(self, ctx: Request) -> tuple[str, ...]:
        fields = self.override_key_fields(ctx)
        if fields is None:
            assert self.key_fields is not UNSET
            return self.key_fields
        return fields

    def override_key_options(self, ctx: Request) -> tuple[str, ...] | None:
        """Dynamically choose the options that enter the cache key for this request.

        Override this to drop options that are inert in the current mode.
        Return the option names (a subset of :attr:`key_options`), or ``None``
        to use all of :attr:`key_options`.
        """
        return None

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

        Subclasses whose validity is fully determined by their key and
        dependency timestamps need not override this method.
        """
        return {}

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        """Describe how this node should appear when used as a dependency.

        Override this only when the dependency-facing fingerprint should be
        smaller or different from the full artifact fingerprint. ``view`` can
        be used to describe data returned through :meth:`load_view`. The
        default implementation ignores ``view`` and reuses :meth:`fingerprint`.

        One use case is managed intermediate inputs, such as ICA files, where a
        change in excluded components invalidates dependent nodes, but does not
        signal an invalid ICA file artifact.
        """
        return self.fingerprint(ctx)

    def dependency_fingerprint_override(
            self,
            ctx: Request,
            dep: Dependency,
            dep_ctx: Request,
    ) -> dict[str, Any] | None:
        """Return a complete local fingerprint for one dependency edge.

        ``ctx`` is the request for this node, whose artifact is being validated.
        Loading through ``ctx.load(...)`` is restricted to declared
        dependencies. ``dep`` is the declared edge from :meth:`dependencies`.
        ``dep_ctx`` is the resolved request for the dependency node.

        Override when this node depends on only part of a dependency. Return
        ``None`` to use the dependency node's own manifest entry.
        """
        return None

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

    def path(self, ctx: Request) -> Path:
        """Path to the artifact."""
        raise NotImplementedError

    def load_view(
            self,
            ctx: Request,
            view: str,
    ):
        """Load one named dependency/user-facing view for this node.

        Override this to expose data views that bypass the :meth:`load` method.
        Views can be loaded through ``request.load(view=...)``.

        Use views sparingly, to keep the dependency graph explicit. A view is
        appropriate for an alternate materialization of the same request, such
        for cheap metadata lookup. Prefer introducing a separate :class:`Input`,
        :class:`Derivative`, or :class:`UncachedDerivative` when the value has
        its own dependencies, or is used as a build input by other derivatives.

        Implementations acquire node data through ``ctx.load(...)`` just like
        :meth:`build`. Note that the declared-dependency restriction on
        ``ctx.load(...)`` is only enforced when the view is requested during
        the requesting node's own build; when the view is loaded through a
        dependency edge or directly, implementations are themselves
        responsible for reading only data that
        :meth:`dependency_fingerprint` covers for that view.
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

    def exists(self, ctx: Request) -> bool:
        """Return whether the input artifact for this request exists."""
        return self.path(ctx).exists()


class VersionedInput(Input[T]):
    """Input tracked through one canonical reference copy and a version identity.

    For inputs whose data is too large to embed in every dependent manifest
    (e.g. predictor time series): the node keeps a single canonical copy of the
    tracked data under ``cache-dir/<node-name>/``, together with a version
    identity ``{'uid': <uuid4 hex>, 'serial': <int>}``. Dependent manifests
    only store the small version identity (subclasses include
    :meth:`reference_version` in :meth:`~DependencyNode.fingerprint`), so the
    data exists once regardless of how many artifacts depend on it.

    Change detection is exact: when the cheap source fingerprint
    (typically a file stat) drifts, the current data is compared against the
    reference copy. Identical data refreshes the stored source fingerprint
    without changing the version, so dependents stay valid; changed data
    becomes the new reference with an incremented ``serial``, so dependents
    rebuild. ``uid`` is minted once when the reference is created, which makes
    the identity reset-safe: deleting and recreating a reference always
    changes it (a bare counter could climb back to a previously stored value
    with different data).
    """

    def _reference_stem(self, ctx: Request) -> str:
        """Stable identifier for the tracked data; sanitized for use as a file name."""
        raise NotImplementedError

    def _source_fingerprint(self, ctx: Request) -> dict[str, Any]:
        """Cheap fingerprint of the source (e.g. :func:`file_fingerprint`); the data is only compared when it drifts."""
        raise NotImplementedError

    def _current_data(self, ctx: Request) -> Any:
        """Load the tracked data from the source."""
        raise NotImplementedError

    def _data_equal(self, ctx: Request, stored: Any, current: Any) -> bool:
        """Exact comparison between the reference copy and the current data."""
        raise NotImplementedError

    def _reference_path(self, ctx: Request) -> Path:
        stem = CACHE_PATH_UNSAFE.sub(CACHE_PATH_UNSAFE_REPLACEMENT, self._reference_stem(ctx))
        return ctx.registry.cache_dir / self.name / f'{stem}.json'

    def reference_version(self, ctx: Request) -> dict[str, Any]:
        """Version identity of the tracked data, updating the reference when the source changed.

        May write to the reference (a data pickle plus a JSON pointing to it)
        even during a mere validity check; this parallels the in-place
        dependency-entry refresh the registry performs on parent manifests.
        Concurrent writers racing on the same change write identical content
        (the atomic replace picks one); concurrent creation can mint two
        ``uid`` values, costing at most one spurious rebuild, never a stale
        accept.
        """
        path = self._reference_path(ctx)
        # canonicalize so equality survives the JSON round-trip (tuples, key order)
        source = ctx.registry.canonicalize(self._source_fingerprint(ctx))
        reference = self._read_reference(path)
        if reference is not None and reference['source'] == source:
            return reference['version']
        data = self._current_data(ctx)
        if reference is not None:
            stored = self._read_reference_data(path, reference)
            if stored is None:  # data file lost → cannot compare, treat as new reference
                reference = None
            elif self._data_equal(ctx, stored, data):
                # only the source stat drifted → refresh it so later checks take the fast path
                if not ctx.registry._readonly:
                    reference['source'] = source
                    _atomic_write_text(path, json.dumps(reference, sort_keys=True, indent=2))
                return reference['version']
        if reference is None:
            version = {'uid': uuid4().hex, 'serial': 0}
        else:
            version = {'uid': reference['version']['uid'], 'serial': reference['version']['serial'] + 1}
        if ctx.registry._readonly:
            # Read-only validation (garbage-collection scan): report the
            # version that a write would produce — it mismatches any stored
            # dependent version, so dependents classify as stale — without
            # persisting anything.
            return version
        data_file = f'{path.stem}.{version["serial"]}.pickle'
        path.parent.mkdir(parents=True, exist_ok=True)
        # write the data first, then atomically replace the JSON (the source of
        # truth), so an interrupted write leaves the old reference intact
        data_path = path.parent / data_file
        tmp_path = data_path.with_name(f'{data_path.name}.tmp')
        tmp_path.write_bytes(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
        tmp_path.replace(data_path)
        _atomic_write_text(path, json.dumps({'source': source, 'version': version, 'data_file': data_file}, sort_keys=True, indent=2))
        return version

    @staticmethod
    def _read_reference(path: Path) -> dict[str, Any] | None:
        "Read the reference JSON; an unreadable or malformed reference counts as missing."
        if not path.exists():
            return None
        try:
            reference = json.loads(path.read_text())
            if isinstance(reference, dict) and {'source', 'version', 'data_file'} <= reference.keys():
                return reference
        except (OSError, ValueError):
            pass
        return None

    @staticmethod
    def _read_reference_data(path: Path, reference: dict[str, Any]) -> Any | None:
        try:
            return pickle.loads((path.parent / reference['data_file']).read_bytes())
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
            return None


class Derivative(DependencyNode[T]):
    """Base class for one cache-managed derived artifact.

    A derivative is a named artifact family that can be keyed, built, loaded,
    saved, and validated. Subclasses normally override:

    - :meth:`path` to choose the artifact location
    - :meth:`fingerprint` to describe configuration definitions that can
      change without the key changing (e.g. epoch parameters, pipe settings).
    - :meth:`build` to compute the artifact
    - :meth:`load` / :meth:`save` to serialize the artifact

    The standard subclass contract is:

    - declare ``key_options`` and ``view_options`` (inherited
      from :class:`DependencyNode`) for options affecting artifact identity
      or post-load shaping respectively
    - implement :meth:`build` to construct the artifact representation from
      state plus options
    - implement :meth:`load` / :meth:`save` for that artifact representation
    - optionally implement :meth:`apply_view_options` to transform the loaded
      artifact into the final return value


    Attributes
    ----------
    cache_policy
        Whether artifacts of this derivative persist to the cache.
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

    # Whether artifacts of this derivative persist to the cache.
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
        fields = self._get_key_fields(ctx)
        label_key = canonical_state_subset(ctx.state, fields) if fields else ctx.key()
        return _simple_cache_label(label_key)

    def cache_log_path(self, ctx: Request, path: Path) -> str:
        """Return the displayed artifact path for cache log messages."""
        return ctx.registry.describe_artifact_path(path)

    def log_cache_hit(self, ctx: Request, path: Path) -> None:
        """Emit the standard cache-hit message for this derivative."""
        self._log_cache_event(ctx, path, "Load cached", 'cached')

    def log_cache_build(self, ctx: Request, path: Path) -> None:
        """Emit the standard cache-build message for this derivative (no prior artifact)."""
        self._log_cache_event(ctx, path, "Build", 'build')

    def log_cache_recompute(self, ctx: Request, path: Path, reason: CacheInvalidation) -> None:
        """Emit the cache-recompute message, reporting why the cached artifact was invalid."""
        self._log_cache_event(ctx, path, "Recompute", 'recompute', reason=reason)

    def _log_cache_event(
            self,
            ctx: Request,
            path: Path,
            action: str,
            event: str,
            *,
            reason: CacheInvalidation | None = None,
    ) -> None:
        if self.cache_log_level is None:
            return
        detail = f" ({reason.message()})" if reason is not None else ""
        cache_event = {'event': event, 'derivative': self.name}
        if reason is not None:
            cache_event.update(reason.as_dict())
        ctx.registry.log.log(
            self.cache_log_level, "%s %s%s: %s", action, self.name, detail, self.cache_log_path(ctx, path),
            extra={'cache_event': cache_event},
        )

    def path(self, ctx: Request) -> Path:
        """Return the concrete artifact path for this request.

        Subclasses may either override this method explicitly or declare
        :attr:`cache_suffix` to use the default cache path scheme based on the
        derivative name and :meth:`key`.

        The returned path identifies where the artifact itself lives. For
        artifacts inside ``cache-dir``, the registry writes the manifest next
        to the artifact. For public/export artifacts outside ``cache-dir``,
        the registry mirrors the manifest under the derivative's node directory in ``cache-dir``.

        Implementations should derive the path from semantic state/options
        only. They should not perform dependency traversal, create directories,
        or perform cache logic.
        """
        if self.cache_suffix is None:
            raise NotImplementedError
        key = ctx.key()
        key_hash = _full_cache_key_digest(key)[:CACHE_KEY_HASH_LEN]
        label = self.cache_label(ctx) or DEFAULT_CACHE_LABEL
        label_clean = CACHE_PATH_UNSAFE.sub(CACHE_PATH_UNSAFE_REPLACEMENT, label.casefold())
        label_slug = label_clean[:MAX_CACHE_LABEL_LEN].strip('-_')
        return ctx.registry.cache_dir / self.name / _cache_entity_dir(key) / f"{label_slug}_key-{key_hash}{self.cache_suffix}"

    def key(self, ctx: Request) -> dict[str, Any]:
        """The key used to generate a unique path for this artifact.

        This is the framework assembler and should not need overriding.
        The key is used to resolve the artifact path and should stay focused
        on cache address/identity. It is narrower than :meth:`fingerprint`,
        which records the fuller set of non-dependency request
        state/options/definitions that make an existing artifact stale.
        """
        fields = self._get_key_fields(ctx)
        key = canonical_state_subset(ctx.state, fields)
        option_names = self.override_key_options(ctx)
        options = ctx.options if option_names is None else {name: ctx.options[name] for name in option_names}
        if options:
            key['options'] = options
        return key

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
        ``value`` unchanged. Any data loaded through ``ctx.load(...)`` must be
        declared in :meth:`dependencies`, just as for :meth:`build`.
        """
        return value


class UncachedDerivative(Derivative[T]):
    """Base class for derived values that should never persist to the cache.

    :attr:`cache_policy` is :attr:`CachePolicy.NEVER`, so no artifact path or
    manifest is created and ``build`` is called on every request. There is no
    cache key, so :meth:`key` and :meth:`path` are not used; subclasses still
    declare the state they depend on through :attr:`~DependencyNode.key_fields`
    purely for read enforcement, so reading an undeclared state field raises
    :class:`RuntimeError` exactly as for a cached derivative.
    """

    cache_policy = CachePolicy.NEVER
    cache_log_level = None

    def key(self, ctx: Request) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} is uncached; key() must not be called")

    def path(self, ctx: Request) -> Path:
        raise NotImplementedError(f"{type(self).__name__} is uncached; path() must not be called")


class ExternalArtifactDerivative(Derivative[T]):
    """Base class for derivatives whose artifact is materialized on disk by :meth:`build`.

    Some derivatives wrap external tools (e.g. FreeSurfer) that write their
    output directly into the conventional subjects-dir folder structure, and
    have no independent in-memory form the cache could serialize and reload on
    its own. For these nodes :meth:`build` is the writer: it runs the tools
    (and/or writes the value) and returns the in-memory value, while
    :meth:`load` re-reads the value from the real on-disk artifact. Unlike a
    plain :class:`Derivative`, :meth:`save` does not write the real artifact.

    Validity is tracked the normal way (manifest plus artifact existence), so
    :meth:`path` must point at something whose existence means "materialized".
    That anchor is either:

    - the real artifact file itself, when it is a single pipeline-generated
      file (e.g. a source space ``.fif``); or
    - a small stamp inside ``cache-dir``, when the real output is multiple
      files, or when a user-provided variant must stay a fingerprinted input
      rather than this node's writable artifact (e.g. parcellation ``*.annot``
      files, where anchoring on the user files outside ``cache-dir`` would route
      them through :class:`ProtectedArtifactError`).

    The :meth:`save` provided here materializes that stamp when, and only when,
    the anchor lives in ``cache-dir``; when :meth:`path` is the real external
    file, :meth:`build` already wrote it and there is nothing to do (writing
    would clobber the real artifact). Subclasses implement :meth:`build` and
    :meth:`load`.
    """

    def save(self, ctx: Request, path: Path, value: T) -> None:
        # build() materialized the real artifact via external tools. When path()
        # anchors on a cache-dir stamp, create it; when path() is the real
        # external file, it already exists and there is nothing to write.
        if ctx.registry.is_cache_artifact(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{self.name}\n")


class _RestrictedStateView(Mapping):
    """State view that enforces access only to declared key fields.

    Used during :meth:`Derivative.build`, :meth:`~DependencyNode.fingerprint`,
    :meth:`~DependencyNode.dependency_fingerprint`, and
    :meth:`~DependencyNode.dependencies` to ensure every state field that
    affects the artifact is declared in :attr:`~DependencyNode.key_fields` or
    :attr:`~DependencyNode.fixed_state`.

    This is a :class:`~collections.abc.Mapping`, not a :class:`dict`, so that
    bulk access goes through checked reads. Iteration — and hence ``keys()``,
    ``**ctx.state``, etc. yields only the declared fields, so ``**ctx.state``
    means "the state this node may depend on", never the complete pipeline
    state. Membership tests see the full state, since they do not read a
    value.
    """

    def __init__(self, state: dict[str, Any], allowed: frozenset[str]):
        self._state = state
        self._allowed = allowed

    def _check_allowed(self, key: str) -> None:
        if key not in self._allowed:
            raise RuntimeError(
                f"State field {key!r} is not declared in this node's key_fields (or fixed_state). If it affects this node's output, add it to key_fields."
            )

    def __getitem__(self, key: str) -> Any:
        self._check_allowed(key)
        return self._state[key]

    def get(self, key: str, default: Any = None) -> Any:
        # Reading a present field is checked like __getitem__; reading an absent
        # field stays allowed, matching plain dict semantics.
        if key in self._state:
            self._check_allowed(key)
        return self._state.get(key, default)

    def __contains__(self, key: object) -> bool:
        # Membership tests do not read a value, so they stay allowed.
        return key in self._state

    def __len__(self) -> int:
        return sum(1 for key in self._state if key in self._allowed)

    def __iter__(self):
        return (key for key in self._state if key in self._allowed)


def _dep_entry_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare one describe_dependency entry with quick-fingerprint shortcut.

    ``key`` participates because fingerprints often describe configuration
    only: a dependency that resolves to a different artifact (different cache
    key) must invalidate the parent even when its fingerprint is unchanged.
    """
    for key in ('name', 'kind', 'view', 'key'):
        if stored.get(key) != current.get(key):
            return False
    stored_quick = stored.get('quick_fingerprint')
    current_quick = current.get('quick_fingerprint')
    if stored_quick is not None and current_quick is not None and stored_quick == current_quick:
        return True
    if stored.get('fingerprint') != current.get('fingerprint'):
        return False
    return dependencies_match(stored.get('dependencies', {}), current.get('dependencies', {}))


def dependencies_match(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare dependency manifests, using quick fingerprints as a first-pass shortcut."""
    if stored.keys() != current.keys():
        return False
    return all(_dep_entry_matches(stored[k], current[k]) for k in stored)


def compare_manifests(stored: ArtifactManifest | None, current: ArtifactManifest) -> CacheInvalidation | None:
    """Return why ``stored`` is invalid relative to ``current``, or ``None`` if it matches.

    Composes the diff engine in :mod:`.logging` with manifest-level knowledge
    (the six validity checks and the quick-fingerprint dependency shortcut).
    """
    if stored is None:
        return CacheInvalidation('missing_manifest')
    if stored.schema_version != current.schema_version:
        return CacheInvalidation('schema', ('schema_version',), stored.schema_version, current.schema_version)
    if stored.derivative != current.derivative:
        return CacheInvalidation('derivative', ('derivative',), stored.derivative, current.derivative)
    if stored.derivative_version != current.derivative_version:
        return CacheInvalidation('version', ('derivative_version',), stored.derivative_version, current.derivative_version)
    if stored.key != current.key:
        return diff_invalidation('key', stored.key, current.key)
    if stored.fingerprint != current.fingerprint:
        return diff_invalidation('fingerprint', stored.fingerprint, current.fingerprint)
    if not dependencies_match(stored.dependencies, current.dependencies):
        return diff_invalidation('dependencies', stored.dependencies, current.dependencies, strip_quick=True)
    return None


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
            provided_key_options: frozenset[str] | set[str] | tuple[str, ...] = (),
    ):
        self.node = node
        self.registry = registry
        self.root = registry.root
        self.datatype = registry.datatype
        self._state = state
        self.state = state
        self.options = options
        self.view_options = view_options
        self.controls = frozenset(controls)
        # Key-tier options the caller explicitly set (for the inert-option warning).
        self._provided_key_options = frozenset(provided_key_options)
        # Normalize and validate OptionSpec-declared option values before the cache
        # key is computed below, so keys are canonical (equivalent spellings share
        # one artifact) and invalid values fail before build.
        for declared, values in ((node.key_options, self.options), (node.view_options, self.view_options)):
            for option, spec in declared.items():
                if isinstance(spec, OptionSpec):
                    values[option] = spec.validated(self, option, values[option])
        self._key: dict[str, Any] | None = None
        self._base_artifact_path: Path | None = None
        self._artifact_path: Path | None = None
        self._manifest_path: Path | None = None
        self._artifact_metadata: dict[str, Any] | None = None
        # Populated while derivative methods are constrained to declared dependencies.
        self._build_deps: dict[str, Dependency] | None = None
        self._build_deps_depth = 0
        # Restricted view for enforcement; None when there is nothing to enforce.
        # The readable set is this request's identity fields — from
        # override_key_fields() when defined, else the static key_fields — so a
        # node that keys dynamically need not also declare a redundant static
        # key_fields. A node that declares neither (e.g. a result node that
        # overrides key() and manages its own identity, or an Input with
        # key_fields=()) is not read-restricted. Inputs are restricted only in
        # their cache-affecting methods (fingerprint/dependencies); load() etc.
        # run outside the check context and may read arbitrary state.
        self._restricted_state: _RestrictedStateView | None = None
        if isinstance(node, (Derivative, Input)):
            read_fields = node._get_key_fields(self)
            allowed = frozenset(read_fields) | frozenset(node.fixed_state)
            if allowed:
                self._restricted_state = _RestrictedStateView(state, allowed)
        if isinstance(node, Derivative) and node.cache_policy != CachePolicy.NEVER:
            # Canonicalize here so key() implementations need not: a key that
            # only became canonical through the manifest JSON round-trip would
            # never equal its stored form and silently recompute on every run.
            self._key = registry.canonicalize(node.key(self))
            self._base_artifact_path = Path(node.path(self))
            self._artifact_path = Path(self.registry.resolve_cache_artifact_path(self._base_artifact_path, self._key))
            self._manifest_path = Path(self.registry.manifest_path(self._artifact_path, self.node.name))
            self._warn_inert_key_options()

    def _warn_inert_key_options(self) -> None:
        """Warn if the caller set a key option that is inert for this request.

        Only fires when the node narrows its key options via
        :meth:`~DependencyNode.override_key_options` and drops an option the
        caller explicitly set — i.e. the option has no effect in the current
        mode, so the caller likely expected an effect it will not get.
        """
        if not self._provided_key_options:
            return
        effective = self.node.override_key_options(self)
        if effective is None:
            return
        inert = self._provided_key_options.difference(effective)
        if inert:
            joined = ', '.join(repr(option) for option in sorted(inert))
            warnings.warn(f"{self.node.name!r}: option(s) {joined} were set but have no effect for this request (inert in the current mode); the result does not depend on them.", stacklevel=2)

    def has_control(self, control: str) -> bool:
        """Return whether this request includes one explicit execution control."""
        return control in self.controls

    @contextmanager
    def _state_check_context(self):
        """Restrict ``ctx.state`` to declared key_fields during cache-affecting calls.

        When active, any access to a state field not listed in
        :attr:`~Derivative.key_fields` or :attr:`~DependencyNode.fixed_state`
        raises :class:`RuntimeError`.  Safe to nest: the restriction is
        activated by the outermost call and deactivated only on its exit.
        A no-op for nodes that declare no identity fields (an
        :class:`UncachedDerivative` or :class:`Input` with ``key_fields=()``).
        """
        if self._restricted_state is None:
            yield
            return
        already_active = self.state is self._restricted_state
        if not already_active:
            self.state = self._restricted_state
        try:
            yield
        finally:
            if not already_active:
                self.state = self._state

    def exists(self) -> bool:
        """Return whether the input artifact for this request exists."""
        if isinstance(self.node, Input):
            return self.node.exists(self)
        else:
            raise TypeError(f"{self.node=} does not support exists()")

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

    def dependency_fingerprints(self, stored: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the current dependency manifest fragment for this request.

        ``stored`` is the dependency fragment from the previous manifest, if
        any. When given, each entry reuses its stored fingerprint where the
        cheap quick fingerprint still matches (see :meth:`describe_dependency`),
        so cache-validity checks skip the expensive recomputation.
        """
        return self.registry.dependency_fingerprints(self, stored)

    def current_fingerprint(self) -> dict[str, Any]:
        """Return the canonical current fingerprint for this node request."""
        with self._state_check_context():
            return self.registry.canonicalize(self.node.fingerprint(self))

    def _resolve_context(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Canonical ``(state, options)`` that reproduce this request through :meth:`DerivativeRegistry.resolve`.

        Stored in the manifest (``resolve_state`` / ``resolve_options``) so a
        request can be reconstructed offline for cache garbage collection. The
        state subset covers the effective key fields, which suffices to
        re-resolve the dependency subtree (see
        :meth:`DerivativeRegistry._check_edge_key_coverage`). ``fixed_state``
        is not stored; :meth:`DerivativeRegistry.resolve` re-applies it.
        """
        fields = self.node._get_key_fields(self)
        state = canonical_state_subset(self._state, fields)
        options = self.registry.canonicalize(self.options)
        return state, options

    def current_dependency_fingerprint(self, view: str | None = None) -> dict[str, Any]:
        """Return the canonical dependency-facing fingerprint for this request."""
        with self._state_check_context():
            return self.registry.canonicalize(self.node.dependency_fingerprint(self, view))

    def current_dependency_fingerprint_quick(self, view: str | None = None) -> dict[str, Any] | None:
        """Return the canonical quick proxy fingerprint, or ``None`` if not supported."""
        result = self.node.dependency_fingerprint_quick(self, view)
        if result is None:
            return None
        return self.registry.canonicalize(result)

    def describe_dependency(
            self,
            view: str | None = None,
            fingerprint_override: dict[str, Any] | None = None,
            stored: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Describe this request for inclusion in another node's manifest.

        ``stored`` is the matching entry from the previous manifest, if any.
        When it is given and the cheap ``quick_fingerprint`` still matches, the
        stored full fingerprint and sub-dependencies are reused instead of being
        recomputed, so a validity check does not pay for the expensive walk.
        """
        out: dict[str, Any] = {'name': self.node.name}
        if fingerprint_override is None:
            quick = self.current_dependency_fingerprint_quick(view)
            if quick is not None:
                out['quick_fingerprint'] = quick
            if stored is not None and quick is not None and stored.get('quick_fingerprint') == quick:
                out['fingerprint'] = stored.get('fingerprint')
                out['dependencies'] = stored.get('dependencies', {})
            else:
                out['fingerprint'] = self.current_dependency_fingerprint(view)
                out['dependencies'] = self.dependency_fingerprints(stored and stored.get('dependencies'))
        else:
            out['fingerprint'] = self.registry.canonicalize(fingerprint_override)

        if view is not None:
            out['view'] = view
        if isinstance(self.node, Derivative) and self.node.cache_policy != CachePolicy.NEVER:
            out['kind'] = 'derivative'
            out['key'] = self.key()
            out['manifest'] = self.manifest_path.relative_to(self.registry.cache_dir).as_posix()
        else:
            out['kind'] = 'input'
        return out

    def _require_derivative(self) -> Derivative[T]:
        if isinstance(self.node, Derivative):
            return self.node
        raise TypeError(f"Request for input {self.node.name!r} has no derivative artifact state")

    def _require_cached_derivative(self, attribute: str) -> None:
        derivative = self._require_derivative()
        if derivative.cache_policy is CachePolicy.NEVER:
            raise TypeError(f"Request for uncached derivative {derivative.name!r} has no {attribute}")

    @property
    def base_artifact_path(self) -> Path:
        """Base artifact path before any cache-path disambiguation."""
        self._require_cached_derivative('artifact path')
        assert self._base_artifact_path is not None
        return self._base_artifact_path

    @property
    def artifact_path(self) -> Path:
        """Resolved artifact path for a derivative request."""
        self._require_cached_derivative('artifact path')
        assert self._artifact_path is not None
        return self._artifact_path

    @property
    def manifest_path(self) -> Path:
        """Resolved manifest path for a derivative request."""
        self._require_cached_derivative('manifest path')
        assert self._manifest_path is not None
        return self._manifest_path

    def key(self) -> dict[str, Any]:
        """Return the normalized derivative key for this request."""
        self._require_cached_derivative('cache key')
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

    @property
    def declared_dependencies(self) -> dict[str, Dependency]:
        """Declared dependencies for the current build, keyed by label.

        Only available during :meth:`Derivative.build`; raises
        :class:`RuntimeError` if called outside a build context.
        """
        if self._build_deps is None:
            raise RuntimeError("declared_dependencies is only available during build()")
        return self._build_deps

    def _manifest(self) -> ArtifactManifest | None:
        return self.registry.read_manifest(self.manifest_path)

    def _check_valid(
            self,
            manifest: ArtifactManifest,
    ) -> CacheInvalidation | None:
        """Return why ``manifest`` is stale for this request, or ``None`` if valid."""
        derivative = self._require_derivative()
        resolve_state, resolve_options = self._resolve_context()
        current = ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=derivative.name,
            derivative_version=derivative.version,
            key=self.key(),
            fingerprint=self.current_fingerprint(),
            dependencies=self.dependency_fingerprints(stored=manifest.dependencies),
            cache_policy=derivative.cache_policy.value,
            software={},
            resolve_state=resolve_state,
            resolve_options=resolve_options,
        )
        reason = compare_manifests(manifest, current)
        if reason is None and not self.registry._readonly and (current.dependencies != manifest.dependencies or manifest.resolve_state is None):
            # A quick fingerprint drifted while the full fingerprint still
            # matched (e.g. a touched file). Persist the refreshed dependency
            # entries so future checks take the quick path again.
            current.software = manifest.software
            current.artifact_metadata = manifest.artifact_metadata
            self.registry.write_manifest(self.manifest_path, current)
        return reason

    def is_valid(self) -> bool:
        """Return whether the current derivative request already has a valid artifact."""
        derivative = self._require_derivative()
        if derivative.cache_policy is CachePolicy.NEVER:
            return False
        manifest = self._manifest()
        if manifest is None or not self.artifact_path.exists():
            return False
        return self._check_valid(manifest) is None

    def _dependency_map(self) -> dict[str, Dependency]:
        """Declared dependencies keyed by label, rejecting duplicate labels."""
        out: dict[str, Dependency] = {}
        for dep in self.node.dependencies(self):
            label = dep.label or dep.name
            if label in out:
                raise RuntimeError(f"Duplicate dependency label {label!r} for node {self.node.name!r}")
            out[label] = dep
        return out

    @contextmanager
    def _build_deps_context(self):
        if self._build_deps is None:
            self._build_deps = self._dependency_map()
        self._build_deps_depth += 1
        try:
            yield
        finally:
            self._build_deps_depth -= 1
            if self._build_deps_depth == 0:
                self._build_deps = None

    def load_artifact(self) -> T:
        """Load or build the underlying derivative artifact without view shaping."""
        derivative = self._require_derivative()
        use_cache = derivative.cache_policy is not CachePolicy.NEVER
        if use_cache:
            manifest = self._manifest()
            artifact_exists = self.artifact_path.exists()
            if manifest is None:
                reason = None if not artifact_exists else CacheInvalidation('missing_manifest')
            elif not artifact_exists:
                reason = CacheInvalidation('missing_artifact')
            else:
                reason = self._check_valid(manifest)
            if manifest is not None and artifact_exists and reason is None:
                self._artifact_metadata = manifest.artifact_metadata
                derivative.log_cache_hit(self, self.artifact_path)
                return derivative.load(self, self.artifact_path)
            if artifact_exists and not self.registry.is_cache_artifact(self.artifact_path) and not self.has_control(ALLOW_PROTECTED_OVERWRITE):
                raise ProtectedArtifactError(derivative.name, self.artifact_path)
            if reason is None:
                derivative.log_cache_build(self, self.artifact_path)
            else:
                derivative.log_cache_recompute(self, self.artifact_path, reason)

        with self._build_deps_context(), self.registry._node_warning_context(self), self._state_check_context():
            artifact = derivative.build(self)
        if not use_cache:
            self._artifact_metadata = self.registry.canonicalize(derivative.artifact_metadata(self, artifact))
            return artifact
        return self.save_artifact(artifact)

    def save_artifact(self, artifact: T) -> T:
        """Persist a built artifact and write its manifest; return the reloaded artifact.

        Shared by :meth:`load_artifact` and by off-host execution (an externally
        computed result re-united with its cache entry, e.g. via
        :meth:`~eelbrain._experiment.trf.job.TRFJobSpec.save_result`).
        """
        derivative = self._require_derivative()
        artifact_metadata = self.registry.canonicalize(derivative.artifact_metadata(self, artifact))
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        derivative.save(self, self.artifact_path, artifact)
        resolve_state, resolve_options = self._resolve_context()
        manifest = ArtifactManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            derivative=derivative.name,
            derivative_version=derivative.version,
            key=self.key(),
            fingerprint=self.current_fingerprint(),
            dependencies=self.dependency_fingerprints(),
            cache_policy=derivative.cache_policy.value,
            software={
                'eelbrain_cache_schema': str(MANIFEST_SCHEMA_VERSION),
                'mne': mne.__version__,
            },
            artifact_metadata=artifact_metadata,
            resolve_state=resolve_state,
            resolve_options=resolve_options,
        )
        self.registry.write_manifest(self.manifest_path, manifest)
        self._artifact_metadata = manifest.artifact_metadata
        return derivative.load(self, self.artifact_path)

    def load(
            self,
            name: str | None = None,
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
        ``view`` is accepted; passing ``state``, ``options``, or ``controls``
        raises :class:`TypeError`.
        """
        if isinstance(name, str):
            if self._build_deps is not None:
                if name not in self._build_deps:
                    declared = sorted(self._build_deps)
                    raise RuntimeError(f"{self.node.name!r} called ctx.load({name!r}) which is not a declared dependency. Declared: {declared}")
                if view is not None or state is not None or options is not None or controls:
                    raise TypeError(f"{self.node.name!r} passed overrides to ctx.load({name!r}); declare view, state, and options on the Dependency instead, and do not override controls here")
                dep = self._build_deps[name]
                child = self.registry.resolve(
                    name=dep.name,
                    state={**self._state, **dep.state} if dep.state else self._state,
                    options=dep.options,
                )
                self.registry._check_edge_key_coverage(self, dep, child)
                return child.load(view=dep.view)
            return self.registry.resolve(
                name,
                state={**self._state, **(state or {})},
                options=options,
                controls=controls,
            ).load(view=view)

        if state is not None or options is not None or controls:
            raise TypeError("Request.load() without a dependency name only accepts a view override")
        if view is not None:
            with self.registry._node_warning_context(self):
                return self.node.load_view(self, view)
        if isinstance(self.node, Input):
            with self.registry._node_warning_context(self):
                return self.node.load(self)

        derivative = self._require_derivative()
        with self._build_deps_context():
            artifact = self.load_artifact()
            return derivative.apply_view_options(self, artifact)


class DerivativeRegistry:
    """Registry and resolver for dependency nodes bound to one experiment root."""

    def __init__(self, root: str | Path, log: logging.Logger, datatype: str = 'meg'):
        self.root = Path(root)
        self.log = log
        self.datatype = datatype
        self.deriv_dir = self.root / DERIV_DIR
        self.cache_dir = self.root / CACHE_DIR
        self._nodes: dict[str, DependencyNode[Any]] = {}
        # When True, cache validation must not write anything (no manifest
        # refresh, no disambiguation sidecars, no VersionedInput references).
        # Set by _readonly_context() during a garbage-collection scan.
        self._readonly = False

    @contextmanager
    def _readonly_context(self):
        """Suppress all incidental cache writes during validation (used by :meth:`scan_cache`)."""
        already = self._readonly
        self._readonly = True
        try:
            yield
        finally:
            self._readonly = already

    def register(self, node: DependencyNode[Any]) -> None:
        if node.name in self._nodes:
            raise RuntimeError(f"Dependency node {node.name!r} already registered")
        if not isinstance(node, (Derivative, Input)):
            raise TypeError(f"Unsupported node type: {type(node)!r}")
        if isinstance(node, Input) and node.key_fields is UNSET:
            raise TypeError(f"Input {node.name!r} must declare key_fields (state fields that determine its content); use an empty tuple () only if its identity is fully option-based.")
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
        options = {name: _option_default(spec) for name, spec in node.key_options.items()}
        view_options = {name: _option_default(spec) for name, spec in node.view_options.items()}
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
            provided_key_options=frozenset(node_options).intersection(node.key_options),
        )

    @contextmanager
    def _node_warning_context(self, ctx: Request):
        """Capture warnings during one input load or derivative build, writing new ones once to an experiment log file."""
        node = ctx.node
        if isinstance(node, Input):
            path = node.path(ctx)
            if path.is_relative_to(self.root):
                path = path.relative_to(self.root)
            item = str(path)
        else:
            item = node.name
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            warnings.filterwarnings('ignore', r'unclosed file ', ResourceWarning)
            yield
        if not warning_list:
            return
        details_path = Path(self.root) / LOG_DIR / f'{node.name}-warnings.toml'
        details_path.parent.mkdir(parents=True, exist_ok=True)
        entries = _read_warning_log(details_path)
        seen = {(entry['item'], entry['category'], entry['message']) for entry in entries}
        new_entries = []
        for message in warning_list:
            category = message.category.__name__
            text = str(message.message)
            key = (item, category, text)
            if key in seen:
                continue
            seen.add(key)
            entry = {'item': item, 'category': category, 'message': text}
            entries.append(entry)
            new_entries.append(entry)
        if not new_entries:
            return
        _write_warning_log(details_path, f"Warnings emitted during {node.name}.\n", entries)
        count = len(new_entries)
        noun = 'warning was' if count == 1 else 'warnings were'
        self.log.warning("%s new %s issued during %s. Full details were written to %s. Previously recorded %s warnings will be suppressed in the terminal for this experiment.", count, noun, node.name, details_path, node.name)

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
        try:
            data = json.loads(sidecar_path.read_text())
        except (OSError, ValueError) as error:
            self.log.debug("Treating unreadable disambiguation sidecar %s as empty (%s)", sidecar_path, error)
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): value for key, value in data.items() if isinstance(value, str)}

    def _write_cache_disambiguation(self, path: str | Path, data: dict[str, str]) -> None:
        sidecar_path = _cache_disambiguation_path(path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(sidecar_path, json.dumps(data, sort_keys=True, indent=2))

    def resolve_cache_artifact_path(
            self,
            path: str | Path,
            key: dict[str, Any],  # Canonical derivative key (see Request.key()).
    ) -> Path:
        artifact_path = Path(path)
        if not self.is_cache_artifact(artifact_path):
            return artifact_path

        digest = _full_cache_key_digest(key)
        mapping = self._read_cache_disambiguation(artifact_path)
        suffix = mapping.get(digest)
        if suffix is not None:
            return _disambiguated_cache_artifact_path(artifact_path, suffix)

        if not artifact_path.exists():
            return artifact_path

        manifest = self.read_manifest(self.manifest_path(artifact_path))
        if manifest is None or manifest.key == key:
            return artifact_path

        used_suffixes = set(mapping.values())
        index = 1
        while True:
            suffix = f"_alt-{index}"
            if suffix not in used_suffixes:
                break
            index += 1

        if not self._readonly:
            mapping[digest] = suffix
            self._write_cache_disambiguation(artifact_path, mapping)
        return _disambiguated_cache_artifact_path(artifact_path, suffix)

    def _check_edge_key_coverage(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> None:
        """Validate that a dependency's key fields are determined by this edge.

        Every state field in the child's effective key must be pinned on the
        edge (``dep.state`` or the child's :attr:`~DependencyNode.fixed_state`)
        or covered by the parent's own identity (its effective key fields,
        :attr:`~DependencyNode.fixed_state`, or — for a cached parent — its
        cache key). A gap means the parent's cache slot does not distinguish
        values of a field the child's output depends on, so the parent artifact
        would silently share one slot across those values (see the module
        docstring on silent cache-slot sharing).

        Parameters
        ----------
        ctx
            Request for the parent node whose dependencies are being resolved.
        dep
            The dependency edge being validated.
        dep_ctx
            Resolved child request for ``dep``.
        """
        child = dep_ctx.node
        # collect child key fields
        fields = set(child._get_key_fields(dep_ctx))
        # collect parent key fields
        pinned = set(dep.state or ()) | set(child.fixed_state)
        parent = ctx.node
        parent_fields = parent._get_key_fields(ctx)
        coverage = set(parent_fields) | set(parent.fixed_state)
        missing = fields.difference(pinned | coverage)
        if missing:
            raise RuntimeError(f"{parent.name!r} depends on {child.name!r}, whose output depends on state field(s) {missing}, but {parent.name!r} neither keys or pins these on this edge. Fix by adding {missing} to {parent.name!r}.key_fields, or pin it via Dependency({child.name!r}, state=...).")

    def _dependency_handles(
            self,
            ctx: Request,
    ) -> list[tuple[Dependency, Request[Any]]]:

        out = []
        with ctx._build_deps_context():
            for dep in ctx._build_deps.values():
                request = self.resolve(
                    dep.name,
                    state={**ctx._state, **(dep.state or {})},
                    options=dep.options,
                )
                self._check_edge_key_coverage(ctx, dep, request)
                out.append((dep, request))
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
                    key_text = self._tree_mapping_text(handle.key())
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
            children = self._dependency_handles(handle)
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

    def manifest_path(self, artifact_path: Path, node_name: str | None = None) -> Path:
        if self.is_cache_artifact(artifact_path):
            return Path(f"{artifact_path}{MANIFEST_SUFFIX}")

        # External (user-managed) artifact: mirror the manifest under the owning derivative's node directory in the cache
        assert node_name is not None, "node_name is required for external artifacts"
        relative = artifact_path.relative_to(self.deriv_dir)
        relative = relative.relative_to(relative.parts[0])
        return self.cache_dir / node_name / relative.parent / f"{relative.name}{MANIFEST_SUFFIX}"

    def dependency_fingerprints(
            self,
            ctx: Request,  # Bound state/options for the current load.
            stored: dict[str, Any] | None = None,  # Previous manifest fragment, reused on quick-match.
    ) -> dict[str, Any]:
        out = {}
        with ctx._build_deps_context(), ctx._state_check_context():
            for dep, dep_ctx in self._dependency_handles(ctx):
                key = dep.label or dep.name
                fingerprint = ctx.node.dependency_fingerprint_override(ctx, dep, dep_ctx)
                stored_entry = stored.get(key) if stored else None
                out[key] = dep_ctx.describe_dependency(dep.view, fingerprint, stored_entry)
        return out

    def read_manifest(self, path: str | Path) -> ArtifactManifest | None:
        """Read a manifest, returning ``None`` when it is missing or unreadable.

        An unreadable manifest (corrupt JSON, e.g. from an interrupted write,
        or an incompatible structure from an old schema) means the artifact
        cannot be validated, which is equivalent to a missing manifest: the
        artifact will be rebuilt.
        """
        manifest_path = Path(path)
        if not manifest_path.exists():
            return None
        try:
            data = json.loads(manifest_path.read_text())
            return ArtifactManifest.from_dict(data)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self.log.debug("Treating unreadable manifest %s as missing (%s)", manifest_path, error)
            return None

    def write_manifest(self, path: str | Path, manifest: ArtifactManifest) -> None:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(manifest_path, json.dumps(manifest.to_dict(), sort_keys=True, indent=2))

    # --- Garbage collection -------------------------------------------------

    def scan_cache(self, revalidate: bool = True) -> GCReport:
        """Classify every file in the cache directory without modifying anything.

        Parameters
        ----------
        revalidate
            Reconstruct each cached request from its manifest and re-validate
            it against the current pipeline configuration (detects artifacts
            whose key is unchanged but whose configuration definitions
            changed). Set to ``False`` for a faster, structural-only scan.

        See Also
        --------
        collect : delete the files a scan flags
        """
        from .garbage_collection import scan_cache
        return scan_cache(self, revalidate)

    def collect(
            self,
            report: GCReport | None = None,
            revalidate: bool = True,
    ) -> GCReport:
        """Delete the cache files flagged by a garbage-collection scan.

        Parameters
        ----------
        report
            Scan result to act on; scans first when omitted.
        revalidate
            Passed to :meth:`scan_cache` when scanning here.

        See Also
        --------
        scan_cache : the scan and the classification categories
        """
        from .garbage_collection import collect
        return collect(self, report, revalidate)

    @staticmethod
    def canonicalize(value: Any) -> Any:
        """Recursively convert ``value`` to a JSON-serializable, stable form.

        The output is suitable for fingerprints, keys, and manifests: dicts are
        sorted by key, sets are sorted, numpy scalars are unwrapped, and
        :class:`Path` objects become strings. Unrecognized types fall back to
        ``repr()``.

        Domain-specific types handled here (:class:`~eelbrain.Var`,
        :class:`~eelbrain.NDVar`, :class:`~eelbrain.Factor`,
        :class:`~eelbrain.Interaction`, :class:`Configuration`) are an
        intentional coupling between the cache
        kernel and the Eelbrain data model; they allow fingerprints and keys to
        contain arbitrary data objects without callers having to pre-serialize
        them.

        Other rich objects can participate by defining ``_cache_form_()``,
        returning a simple (canonicalizable) representation of the object's
        identity. For objects used as key-tier option values, the owning node
        should declare the option with an :class:`OptionSpec` whose
        ``normalize`` accepts both the object and this form, so that a request
        reconstructed from a stored manifest (offline revalidation) re-parses
        the value into the rich object.
        """
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        cache_form = getattr(value, '_cache_form_', None)
        if cache_form is not None:
            return DerivativeRegistry.canonicalize(cache_form())
        if isinstance(value, Var):
            return DerivativeRegistry.canonicalize(value.x.tolist())
        if isinstance(value, NDVar):
            return {'name': value.name, 'dims': [repr(dim) for dim in value.dims], 'x': DerivativeRegistry.canonicalize(value.x)}
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
        return repr(value)


def file_fingerprint(
        root: str | Path,
        path: str | Path,
        digest: bool = False,
        metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fingerprint one input path using a project-relative location when possible.

    Parameters
    ----------
    root
        Project root directory; ``path`` is stored relative to it when possible, so that fingerprints remain valid when the project is moved.
    path
        The file (or directory) to fingerprint.
    digest
        Include a SHA-1 digest of the file's content (by default, only size and modification time are used).
    metadata
        Additional information to store in the fingerprint.
    """
    root = Path(root)
    path = Path(path)
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    if not path.exists():
        out = InputFingerprint(relative, False, metadata=metadata or {})
    else:
        stat = path.stat()
        sha1 = None
        if digest and path.is_file():
            sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
        out = InputFingerprint(relative, True, stat.st_size, stat.st_mtime, sha1, metadata or {})
    return asdict(out)


def canonical_state_subset(
        state: dict[str, Any],
        fields: tuple[str, ...],
) -> dict[str, Any]:
    return {
        key: DerivativeRegistry.canonicalize(state[key])
        for key in fields
    }
