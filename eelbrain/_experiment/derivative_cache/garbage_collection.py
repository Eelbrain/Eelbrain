"""Garbage collection for the derivative cache.

Stale files are rebuilt in place only when they are re-requested, so a changed
definition or a removed node leaves orphaned artifacts behind. The cache is
never pruned automatically. :meth:`DerivativeRegistry.scan_cache` classifies
every file under ``cache-dir`` (see :class:`GCCategory`) and
:meth:`GCReport.collect` deletes the ones that are safe to remove;
:meth:`~eelbrain.Pipeline.clean_cache` is the user-facing entry point.

Two kinds of staleness are detected. *Structural* invalidity (dead node
directories, outdated manifest schema or derivative version, orphaned manifests
and sidecars, superseded key variants, leftover ``.tmp`` files) is read
directly from the on-disk layout. *Revalidation* staleness — a still-current
cache key whose configuration changed — is found by reconstructing each
request from the manifest's stored ``resolve_state`` / ``resolve_options`` and
re-running the normal validity check; staleness then propagates along the
recorded dependency graph, so a changed upstream definition also collects
everything built from it. Manifests that cannot be reconstructed offline (e.g.
predating these fields) are reported but never deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
import shutil
from typing import Any
import warnings

from ... import fmtxt
from .base import CACHE_DISAMBIGUATION_SUFFIX, MANIFEST_SCHEMA_VERSION, MANIFEST_SUFFIX, ArtifactManifest, CachePolicy, DependencyNode, Derivative, DerivativeRegistry, Input, Request, VersionedInput, _disambiguated_cache_artifact_path


class GCCategory(str, Enum):
    """Classification of one cache path during garbage collection (see :meth:`DerivativeRegistry.scan_cache`)."""

    DEAD_NODE_DIR = 'dead_node_dir'                # top-level cache dir with no registered node
    SCHEMA = 'schema'                              # manifest schema_version is outdated
    DERIVATIVE_VERSION = 'derivative_version'      # manifest derivative_version differs from the registered node
    ORPHAN_MANIFEST = 'orphan_manifest'            # in-cache artifact is missing
    ORPHAN_MIRROR = 'orphan_mirror'                # mirror manifest whose external artifact is gone
    SUPERSEDED_KEY = 'superseded_key'              # the current request resolves to a different artifact path
    STALE_DISAMBIGUATION = 'stale_disambiguation'  # disambiguation sidecar with dead entries
    STALE_REFERENCE = 'stale_reference'            # superseded VersionedInput data pickle
    TMP = 'tmp'                                    # leftover from an interrupted atomic write
    REVALIDATION_STALE = 'revalidation_stale'      # key is current but the configuration changed
    STALE_DEPENDENCY = 'stale_dependency'          # a recorded dependency is stale or was rebuilt with a different fingerprint
    PROTECTED_STALE = 'protected_stale'            # user-managed input no longer validates — kept, reported only
    UNVERIFIABLE = 'unverifiable'                  # cannot be validated offline — kept, reported only
    UNKNOWN = 'unknown'                            # unclassified file — kept, reported only


# Categories that scan_cache reports but GCReport.collect() never deletes.
GC_KEPT_CATEGORIES = frozenset({GCCategory.PROTECTED_STALE, GCCategory.UNVERIFIABLE, GCCategory.UNKNOWN})
# Deletion phases for GCReport.collect(): plain files, artifact+manifest pairs, sidecar maintenance, whole dead trees.
_GC_DELETE_PHASE = {GCCategory.TMP: 0, GCCategory.STALE_REFERENCE: 0, GCCategory.STALE_DISAMBIGUATION: 2, GCCategory.DEAD_NODE_DIR: 3}


@dataclass
class GCEntry:
    """One cache path flagged by :meth:`DerivativeRegistry.scan_cache`."""

    path: Path                          # file or directory the entry refers to
    category: GCCategory
    node: str | None = None             # owning registered node, when known
    size: int = 0                       # bytes (recursive for directories), including the manifest
    reason: str | None = None           # short explanation (e.g. the cache invalidation)
    manifest_path: Path | None = None   # sidecar manifest removed together with the artifact
    prune_digests: tuple[str, ...] = ()  # stale disambiguation-sidecar entries (STALE_DISAMBIGUATION)


@dataclass
class GCReport:
    """Result of one cache garbage-collection scan."""

    registry: DerivativeRegistry = field(repr=False)
    entries: list[GCEntry] = field(default_factory=list)
    scanned_manifests: int = 0
    errors: list[tuple[Path, str]] = field(default_factory=list)  # scan-time exceptions (never raised)
    disambiguation_sidecars: list[Path] = field(default_factory=list)  # all sidecars seen, for post-deletion pruning

    def collect(self) -> None:
        """Delete the cache files flagged by this scan."""
        _collect(self)

    def deletable(self) -> list[GCEntry]:
        """The entries that :meth:`collect` will delete."""
        return [entry for entry in self.entries if entry.category not in GC_KEPT_CATEGORIES]

    def by_category(self) -> dict[GCCategory, list[GCEntry]]:
        """Entries grouped by category, in :class:`GCCategory` definition order."""
        out = {category: [] for category in GCCategory}
        for entry in self.entries:
            out[entry.category].append(entry)
        return {category: entries for category, entries in out.items() if entries}

    def total_size(self, deletable_only: bool = True) -> int:
        entries = self.deletable() if deletable_only else self.entries
        return sum(entry.size for entry in entries)

    def summary(self) -> fmtxt.Table:
        table = fmtxt.Table('lrr')
        table.cells('Category', 'Files', 'Size')
        table.midrule()
        for category, entries in self.by_category().items():
            label = category.value
            if category in GC_KEPT_CATEGORIES:
                label += ' (kept)'
            table.cells(label, len(entries), _format_size(sum(entry.size for entry in entries)))
        deletable = self.deletable()
        total_size = self.total_size()
        caption = []
        if deletable:
            table.midrule()
            table.cells('Total deletable', len(deletable), _format_size(total_size))
        else:
            caption.append("Nothing to delete.")
        if self.errors:
            caption.append(f"* {len(self.errors)} errors*")
        if caption:
            table.caption(' '.join(caption))
        return table

    def file_table(self) -> fmtxt.Table:
        """Table listing the individual files flagged by the scan."""
        table = fmtxt.Table('lrl')
        table.cells('Filename', 'Size', 'Category')
        table.midrule()
        for entry in self.entries:
            path = entry.path
            try:
                path = path.relative_to(self.registry.cache_dir)
            except ValueError:
                pass
            table.cells(str(path), _format_size(entry.size), entry.category.value)
        return table


@dataclass
class _ScannedManifest:
    """Bookkeeping for one parsed manifest during a garbage-collection scan."""

    manifest: ArtifactManifest
    manifest_path: Path
    artifact_path: Path | None  # adjacent in-cache artifact; None for lone (mirror/orphan) manifests
    node: str
    entry: GCEntry | None  # classification; None while the pair counts as valid


def _path_size(path: Path) -> int:
    """Size of a file, or the recursive size of a directory, in bytes (0 when inaccessible)."""
    try:
        if path.is_dir():
            return sum(child.stat().st_size for child in path.rglob('*') if child.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def _format_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024:
            break
        size /= 1024
    else:
        unit = 'TB'
    return f"{size:.0f} {unit}" if unit == 'B' else f"{size:.1f} {unit}"


def _reconstruct_request(registry: DerivativeRegistry, manifest: ArtifactManifest) -> Request[Any] | None:
    """Re-resolve the request that produced ``manifest`` for offline revalidation.

    Returns ``None`` when the manifest predates the stored resolve context.
    May raise for manifests that are no longer resolvable under the current
    definitions (undeclared options, non-round-trippable option values, …);
    callers classify such manifests as unverifiable.
    """
    if manifest.resolve_state is None or manifest.resolve_options is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # e.g. inert-key-option warnings for reconstructed defaults
        return registry.resolve(manifest.derivative, state=dict(manifest.resolve_state), options=dict(manifest.resolve_options))


def _same_path(a: Path, b: Path) -> bool:
    """Path equality tolerant of case-insensitive filesystems."""
    if a == b:
        return True
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False


def scan_cache(registry: DerivativeRegistry, revalidate: bool = True) -> GCReport:
    """Implementation of :meth:`DerivativeRegistry.scan_cache`."""
    report = GCReport(registry)
    if not registry.cache_dir.exists():
        return report
    scanned: dict[str, _ScannedManifest] = {}
    with registry._readonly_context():
        for child in sorted(registry.cache_dir.iterdir()):
            if child.name == '.DS_Store':
                continue
            elif child.name.endswith('.tmp'):
                report.entries.append(GCEntry(child, GCCategory.TMP, size=_path_size(child)))
            elif not child.is_dir():
                report.entries.append(GCEntry(child, GCCategory.UNKNOWN, size=_path_size(child)))
            elif child.name in registry._nodes:
                _scan_node_dir(registry, registry._nodes[child.name], child, report, scanned, revalidate)
            else:
                report.entries.append(GCEntry(child, GCCategory.DEAD_NODE_DIR, size=_path_size(child), reason='no registered node with this name'))
        _propagate_stale_dependencies(registry, report, scanned)
    return report


def _scan_node_dir(
        registry: DerivativeRegistry,
        node: DependencyNode[Any],
        node_dir: Path,
        report: GCReport,
        scanned: dict[str, _ScannedManifest],
        revalidate: bool,
) -> None:
    is_versioned_input = isinstance(node, VersionedInput)
    for dirpath, dirnames, filenames in os.walk(node_dir):
        directory = Path(dirpath)
        manifest_names = [name for name in filenames if name.endswith(MANIFEST_SUFFIX)]
        artifact_names = set()
        for manifest_name in sorted(manifest_names):
            artifact_name = manifest_name[:-len(MANIFEST_SUFFIX)]
            artifact_names.add(artifact_name)
            if artifact_name in dirnames:
                dirnames.remove(artifact_name)  # directory artifact: its contents belong to it
            _classify_manifest(registry, node, directory / manifest_name, report, scanned, revalidate)
        if is_versioned_input:
            keep, stale_references = _scan_reference_files(directory, filenames)
        else:
            keep, stale_references = set(), {}
        for name in sorted(set(filenames) - set(manifest_names) - artifact_names - keep):
            path = directory / name
            if name == '.DS_Store':
                continue
            elif name.endswith('.tmp'):
                report.entries.append(GCEntry(path, GCCategory.TMP, node=node.name, size=_path_size(path)))
            elif name.endswith(CACHE_DISAMBIGUATION_SUFFIX):
                _scan_disambiguation_sidecar(registry, node, path, report)
            elif name in stale_references:
                report.entries.append(GCEntry(path, GCCategory.STALE_REFERENCE, node=node.name, size=_path_size(path), reason=f"superseded by {stale_references[name]}"))
            else:
                report.entries.append(GCEntry(path, GCCategory.UNKNOWN, node=node.name, size=_path_size(path)))


def _scan_reference_files(directory: Path, filenames: list[str]) -> tuple[set[str], dict[str, str]]:
    """Classify VersionedInput reference files in one directory.

    Returns the file names to keep (reference JSONs and their live data
    pickles) and a mapping of superseded data pickles to the live
    ``data_file`` that replaced them. Pickles without a readable reference
    JSON are in neither (reported as unknown; never guess which is live).
    """
    keep: set[str] = set()
    stale: dict[str, str] = {}
    references: dict[str, str] = {}
    for name in filenames:
        if name.endswith('.json') and not name.endswith(MANIFEST_SUFFIX) and not name.endswith(CACHE_DISAMBIGUATION_SUFFIX):
            keep.add(name)  # the reference JSON itself is kept even when unreadable
            reference = VersionedInput._read_reference(directory / name)
            if reference is not None:
                references[name[:-len('.json')]] = reference['data_file']
    for name in filenames:
        match = re.match(r"^(?P<stem>.+)\.\d+\.pickle$", name)
        if match is None:
            continue
        live = references.get(match['stem'])
        if live is None:
            continue
        if name == live:
            keep.add(name)
        else:
            stale[name] = live
    return keep, stale


def _disambiguation_target_exists(base_path: Path, suffix: str) -> bool:
    variant = _disambiguated_cache_artifact_path(base_path, suffix)
    return variant.exists() or Path(f"{variant}{MANIFEST_SUFFIX}").exists()


def _scan_disambiguation_sidecar(registry: DerivativeRegistry, node: DependencyNode[Any], sidecar_path: Path, report: GCReport) -> None:
    report.disambiguation_sidecars.append(sidecar_path)
    base_path = sidecar_path.with_name(sidecar_path.name[:-len(CACHE_DISAMBIGUATION_SUFFIX)])
    mapping = registry._read_cache_disambiguation(base_path)
    if not mapping:
        report.entries.append(GCEntry(sidecar_path, GCCategory.STALE_DISAMBIGUATION, node=node.name, size=_path_size(sidecar_path), reason='empty or unreadable sidecar'))
        return
    dead = tuple(digest for digest, suffix in sorted(mapping.items()) if not _disambiguation_target_exists(base_path, suffix))
    if not dead:
        return
    all_dead = len(dead) == len(mapping)
    reason = 'no artifact left for any entry' if all_dead else f"{len(dead)} of {len(mapping)} entries have no artifact"
    report.entries.append(GCEntry(sidecar_path, GCCategory.STALE_DISAMBIGUATION, node=node.name, size=_path_size(sidecar_path) if all_dead else 0, reason=reason, prune_digests=dead))


def _classify_manifest(
        registry: DerivativeRegistry,
        node: DependencyNode[Any],
        manifest_path: Path,
        report: GCReport,
        scanned: dict[str, _ScannedManifest],
        revalidate: bool,
) -> None:
    report.scanned_manifests += 1
    artifact_path = manifest_path.with_name(manifest_path.name[:-len(MANIFEST_SUFFIX)])
    artifact_exists = artifact_path.exists()
    entry_path = artifact_path if artifact_exists else manifest_path
    size = _path_size(manifest_path) + (_path_size(artifact_path) if artifact_exists else 0)

    def classify(category: GCCategory, reason: str) -> GCEntry:
        entry = GCEntry(entry_path, category, node=node.name, size=size, reason=reason, manifest_path=manifest_path)
        report.entries.append(entry)
        return entry

    manifest = registry.read_manifest(manifest_path)
    if manifest is None:
        classify(GCCategory.UNVERIFIABLE, 'unreadable manifest')
        return

    def reconstruct() -> tuple[Request[Any] | None, str | None]:
        try:
            ctx = _reconstruct_request(registry, manifest)
        except Exception as error:
            report.errors.append((manifest_path, repr(error)))
            return None, f"cannot reconstruct request ({error})"
        if ctx is None:
            return None, 'manifest predates offline revalidation'
        if isinstance(ctx.node, Derivative) and ctx.node.cache_policy is CachePolicy.NEVER:
            return None, 'node is no longer a cached derivative'
        return ctx, None

    entry = None
    owner = registry._nodes.get(manifest.derivative)
    if owner is None:
        entry = classify(GCCategory.UNVERIFIABLE, f"manifest names unregistered derivative {manifest.derivative!r}")
    elif artifact_exists:
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            category = GCCategory.PROTECTED_STALE if isinstance(owner, Input) else GCCategory.SCHEMA
            entry = classify(category, f"manifest schema {manifest.schema_version} (current: {MANIFEST_SCHEMA_VERSION})")
        elif isinstance(owner, Derivative) and manifest.derivative_version != owner.version:
            entry = classify(GCCategory.DERIVATIVE_VERSION, f"derivative version {manifest.derivative_version} (current: {owner.version})")
        elif isinstance(owner, Input) and manifest.derivative_version != owner.version:
            entry = classify(GCCategory.PROTECTED_STALE, f"input version {manifest.derivative_version} (current: {owner.version})")
        else:
            ctx, failure = reconstruct()
            if ctx is None:
                entry = classify(GCCategory.UNVERIFIABLE, failure)
            elif isinstance(owner, Input):
                expected_manifest = registry.manifest_path(owner.path(ctx), owner.name)
                if not _same_path(expected_manifest, manifest_path):
                    entry = classify(GCCategory.PROTECTED_STALE, f"current request resolves to {registry.describe_artifact_path(owner.path(ctx))}")
                elif revalidate:
                    try:
                        valid = owner.is_valid(ctx)
                    except Exception as error:
                        report.errors.append((manifest_path, repr(error)))
                        entry = classify(GCCategory.UNVERIFIABLE, f"cannot validate ({error})")
                    else:
                        if not valid:
                            entry = classify(GCCategory.PROTECTED_STALE, 'input no longer matches its recorded provenance')
            elif not registry.is_cache_artifact(ctx.artifact_path):
                entry = classify(GCCategory.UNVERIFIABLE, 'unexpected file next to an external-artifact mirror manifest')
            elif not _same_path(ctx.artifact_path, artifact_path):
                entry = classify(GCCategory.SUPERSEDED_KEY, f"current request resolves to {registry.describe_artifact_path(ctx.artifact_path)}")
            elif revalidate:
                try:
                    invalidation = ctx._check_valid(manifest)
                except Exception as error:
                    report.errors.append((manifest_path, repr(error)))
                    entry = classify(GCCategory.UNVERIFIABLE, f"cannot validate ({error})")
                else:
                    if invalidation is not None:
                        entry = classify(GCCategory.REVALIDATION_STALE, invalidation.message())
    else:  # lone manifest: external-artifact mirror, or orphaned
        ctx, failure = reconstruct()
        if ctx is None:
            entry = classify(GCCategory.UNVERIFIABLE, failure)
        elif isinstance(owner, Input):
            expected_manifest = registry.manifest_path(owner.path(ctx), owner.name)
            if not _same_path(expected_manifest, manifest_path):
                entry = classify(GCCategory.ORPHAN_MIRROR, 'manifest no longer corresponds to the current input request')
            elif not owner.path(ctx).exists():
                entry = classify(GCCategory.ORPHAN_MIRROR, f"input {registry.describe_artifact_path(owner.path(ctx))} is missing")
            elif revalidate:
                try:
                    valid = owner.is_valid(ctx)
                except Exception as error:
                    report.errors.append((manifest_path, repr(error)))
                    entry = classify(GCCategory.UNVERIFIABLE, f"cannot validate ({error})")
                else:
                    if not valid:
                        entry = classify(GCCategory.PROTECTED_STALE, 'input no longer matches its recorded provenance')
        elif registry.is_cache_artifact(ctx.artifact_path):
            entry = classify(GCCategory.ORPHAN_MANIFEST, 'artifact is missing')
        elif not _same_path(ctx.manifest_path, manifest_path):
            entry = classify(GCCategory.ORPHAN_MIRROR, 'mirror no longer corresponds to the current request')
        elif not ctx.artifact_path.exists():
            entry = classify(GCCategory.ORPHAN_MIRROR, f"external artifact {registry.describe_artifact_path(ctx.artifact_path)} is missing")
        # else: live mirror of an existing external artifact — keep silently
    rel = manifest_path.relative_to(registry.cache_dir).as_posix()
    scanned[rel] = _ScannedManifest(manifest, manifest_path, artifact_path if artifact_exists else None, node.name, entry)


def _find_stale_dependency(
        registry: DerivativeRegistry,
        dependencies: dict[str, Any],
        scanned: dict[str, _ScannedManifest],
        deletable: set[str],
) -> str | None:
    """Name of the first recorded dependency that is stale, or ``None``.

    A dependency is stale when its recorded manifest is itself classified
    for deletion (it will rebuild with a new fingerprint, transitively
    invalidating this parent), or when it was already rebuilt and its
    current manifest no longer matches the recorded key/fingerprint. The
    fingerprint comparison only applies to plain edges whose node uses the
    default :meth:`DependencyNode.dependency_fingerprint` (identical to
    the fingerprint its own manifest stores); view edges and edges with a
    parent-side fingerprint override rely on the recorded-classification
    rule alone.
    """
    for label, entry in dependencies.items():
        if not isinstance(entry, dict):
            continue
        if entry.get('kind') == 'derivative':
            # recorded relative to the cache dir — the same key scanned/deletable use
            if manifest_path := entry.get('manifest'):
                if manifest_path in deletable:
                    return entry.get('name', label)
                item = scanned.get(manifest_path)
                if item is not None and entry.get('view') is None and isinstance(entry.get('dependencies'), dict):
                    child_node = registry._nodes.get(entry.get('name', ''))
                    if child_node is not None and type(child_node).dependency_fingerprint is DependencyNode.dependency_fingerprint:
                        if entry.get('key') != item.manifest.key or entry.get('fingerprint') != item.manifest.fingerprint:
                            return entry.get('name', label)
        nested = entry.get('dependencies')
        if isinstance(nested, dict):
            found = _find_stale_dependency(registry, nested, scanned, deletable)
            if found is not None:
                return found
    return None


def _propagate_stale_dependencies(registry: DerivativeRegistry, report: GCReport, scanned: dict[str, _ScannedManifest]) -> None:
    """Mark artifacts whose recorded dependency tree contains a stale artifact (to a fixpoint)."""
    deletable = {rel for rel, item in scanned.items() if item.entry is not None and item.entry.category not in GC_KEPT_CATEGORIES}
    changed = True
    while changed:
        changed = False
        for rel, item in scanned.items():
            if rel in deletable or item.artifact_path is None:
                continue  # already collected, or a lone/mirror manifest (never deleted by propagation)
            if item.entry is not None and item.entry.category is not GCCategory.UNVERIFIABLE:
                continue
            stale_child = _find_stale_dependency(registry, item.manifest.dependencies, scanned, deletable)
            if stale_child is None:
                continue
            reason = f"depends on stale {stale_child!r}"
            if item.entry is None:
                item.entry = GCEntry(item.artifact_path, GCCategory.STALE_DEPENDENCY, node=item.node, size=_path_size(item.artifact_path) + _path_size(item.manifest_path), reason=reason, manifest_path=item.manifest_path)
                report.entries.append(item.entry)
            else:  # upgrade an unverifiable pair (e.g. rich-option node): staleness is established through its dependencies
                item.entry.category = GCCategory.STALE_DEPENDENCY
                item.entry.reason = reason
            deletable.add(rel)
            changed = True


def _gc_remove_path(registry: DerivativeRegistry, path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as error:
        registry.log.warning("Cache GC: could not remove %s (%s)", path, error)
        return False


def _collect(report: GCReport) -> None:
    """Delete the cache files flagged in a garbage-collection report."""
    registry = report.registry
    touched_dirs: set[Path] = set()
    n_removed = 0
    counts: dict[str, int] = {}
    for entry in sorted(report.deletable(), key=lambda entry: _GC_DELETE_PHASE.get(entry.category, 1)):
        if entry.category is GCCategory.STALE_DISAMBIGUATION:
            continue  # sidecars are pruned against post-deletion reality below
        paths = [entry.path]
        if entry.manifest_path is not None and entry.manifest_path != entry.path:
            paths.append(entry.manifest_path)
        for path in paths:
            if _gc_remove_path(registry, path):
                n_removed += 1
                touched_dirs.add(path.parent)
                counts[entry.category.value] = counts.get(entry.category.value, 0) + 1
                detail = f"; {entry.reason}" if entry.reason else ''
                registry.log.debug("Cache GC: removed %s (%s%s)", registry.describe_artifact_path(path), entry.category.value, detail)
    for sidecar_path in report.disambiguation_sidecars:
        if not sidecar_path.exists():
            continue
        base_path = sidecar_path.with_name(sidecar_path.name[:-len(CACHE_DISAMBIGUATION_SUFFIX)])
        mapping = registry._read_cache_disambiguation(base_path)
        keep = {digest: suffix for digest, suffix in mapping.items() if _disambiguation_target_exists(base_path, suffix)}
        if keep == mapping and mapping:
            continue
        if keep:
            registry._write_cache_disambiguation(base_path, keep)
            registry.log.debug("Cache GC: pruned %i entries from %s", len(mapping) - len(keep), registry.describe_artifact_path(sidecar_path))
        elif _gc_remove_path(registry, sidecar_path):
            n_removed += 1
            touched_dirs.add(sidecar_path.parent)
            counts[GCCategory.STALE_DISAMBIGUATION.value] = counts.get(GCCategory.STALE_DISAMBIGUATION.value, 0) + 1
            registry.log.debug("Cache GC: removed %s (%s)", registry.describe_artifact_path(sidecar_path), GCCategory.STALE_DISAMBIGUATION.value)
    for directory in sorted(touched_dirs, key=lambda path: len(path.parts), reverse=True):
        while directory != registry.cache_dir and directory.is_relative_to(registry.cache_dir):
            try:
                directory.rmdir()  # only succeeds when empty
            except OSError:
                break
            directory = directory.parent
    if n_removed:
        summary = ', '.join(f"{name}: {count}" for name, count in sorted(counts.items()))
        registry.log.info("Cache GC: removed %i files (%s): %s", n_removed, _format_size(report.total_size()), summary)
    else:
        registry.log.info("Cache GC: nothing to remove")
