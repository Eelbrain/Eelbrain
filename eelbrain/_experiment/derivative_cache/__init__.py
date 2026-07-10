"""Derivative-oriented cache primitives for :mod:`eelbrain._experiment`.

The framework (dependency graph, cache resolution and validation) lives in
:mod:`.base`; cache garbage collection lives in :mod:`.garbage_collection`.
See the module docstrings for details.
"""

from .base import (
    ALLOW_PROTECTED_OVERWRITE,
    CACHE_DISAMBIGUATION_SUFFIX,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_SUFFIX,
    ArtifactManifest,
    CachePolicy,
    Dependency,
    DependencyNode,
    Derivative,
    DerivativeRegistry,
    ExternalArtifactDerivative,
    Input,
    OptionSpec,
    ProtectedArtifactError,
    Request,
    UncachedDerivative,
    VersionedInput,
    canonical_state_subset,
    compare_manifests,
    dependencies_match,
    file_fingerprint,
)
from .garbage_collection import GC_KEPT_CATEGORIES, GCCategory, GCEntry, GCReport, _format_size
