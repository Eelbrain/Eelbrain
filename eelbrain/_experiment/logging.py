"""Cache-event diagnostics and pipeline-internal logging.

This module owns how a cache event is *described* and *rendered*:

- the generic structure-diff engine (:func:`find_difference`,
  :func:`format_difference_path`),
- the structured invalidation reason it produces (:class:`CacheInvalidation`,
  :func:`diff_invalidation`),
- and the log rendering of those events (:data:`CACHE_EVENT_COLUMNS`,
  :class:`StructuredFormatter`).

Keeping the reason type next to the log-column schema makes their contract
explicit: :meth:`CacheInvalidation.as_dict` produces exactly the keys listed in
:data:`CACHE_EVENT_COLUMNS`. The recursive diff internals are private; consumers
use the small public surface above. The domain comparison that knows about
manifests lives in :mod:`.derivative_cache` (``compare_manifests``) and composes
this engine.
"""
from dataclasses import dataclass
import logging
from typing import Any


# ---------------------------------------------------------------------------
# Generic structure diff
# ---------------------------------------------------------------------------

def _strip_quick_fingerprints(obj: Any) -> Any:
    """Recursively remove ``quick_fingerprint`` keys so diffs show content changes only."""
    if isinstance(obj, dict):
        return {k: _strip_quick_fingerprints(v) for k, v in obj.items() if k != 'quick_fingerprint'}
    if isinstance(obj, list):
        return [_strip_quick_fingerprints(v) for v in obj]
    return obj


def _first_difference(
        old: Any,
        new: Any,
        path: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], Any, Any] | None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old).union(new), key=str):
            if key not in old:
                return (*path, str(key)), None, new[key]
            if key not in new:
                return (*path, str(key)), old[key], None
            diff = _first_difference(old[key], new[key], (*path, str(key)))
            if diff is not None:
                return diff
        return None
    if isinstance(old, list) and isinstance(new, list):
        for i in range(max(len(old), len(new))):
            if i >= len(old):
                return (*path, f'[{i}]'), None, new[i]
            if i >= len(new):
                return (*path, f'[{i}]'), old[i], None
            diff = _first_difference(old[i], new[i], (*path, f'[{i}]'))
            if diff is not None:
                return diff
        return None
    if old != new:
        return path, old, new
    return None


def _coarsen_diff(
        path: tuple[str, ...],
        old_val: Any,
        new_val: Any,
        old_root: Any,
        new_root: Any,
) -> tuple[tuple[str, ...], Any, Any]:
    """When a diff path ends in list indices, return the parent list instead.

    E.g. ``('bads', '[5]'), 'FT9', 'FT10'`` becomes ``('bads',), ['FT9', ...], ['FT10', ...]``.
    """
    trimmed = path
    while trimmed and trimmed[-1].startswith('['):
        trimmed = trimmed[:-1]
    if trimmed == path:
        return path, old_val, new_val

    def nav(data: Any, p: tuple[str, ...]) -> Any:
        for key in p:
            if data is None:
                return None
            if key.startswith('['):
                idx = int(key[1:-1])
                data = data[idx] if isinstance(data, list) and idx < len(data) else None
            elif isinstance(data, dict):
                data = data.get(key)
            else:
                return None
        return data

    old_parent = nav(old_root, trimmed)
    new_parent = nav(new_root, trimmed)
    if old_parent is None and new_parent is None:
        return path, old_val, new_val
    return trimmed, old_parent, new_parent


def find_difference(
        old: Any,
        new: Any,
        *,
        strip_quick: bool = False,
        coarsen: bool = True,
) -> tuple[tuple[str, ...], Any, Any] | None:
    """Locate the first place ``old`` and ``new`` differ.

    Returns ``(path, old_value, new_value)`` for the first differing field, or
    ``None`` if the structures are equal.

    Parameters
    ----------
    strip_quick
        Drop ``quick_fingerprint`` keys before comparing, so the diff reflects
        content changes rather than the quick-hash proxy.
    coarsen
        When the difference is inside a list, report the containing list
        (``('bads',)``) instead of the changed element (``('bads', '[5]')``).
    """
    if strip_quick:
        old = _strip_quick_fingerprints(old)
        new = _strip_quick_fingerprints(new)
    diff = _first_difference(old, new)
    if diff is None:
        return None
    if coarsen:
        return _coarsen_diff(*diff, old, new)
    return diff


def format_difference_path(path: tuple[str, ...], strip_prefix: tuple[str, ...] = ()) -> str:
    """Render a diff path as a dotted field label (``('a', '[0]', 'b') -> 'a[0].b'``)."""
    parts = list(path)
    if strip_prefix and tuple(parts[:len(strip_prefix)]) == strip_prefix:
        parts = parts[len(strip_prefix):]
    out = []
    for part in parts:
        if part.startswith('['):
            if out:
                out[-1] += part
            else:
                out.append(part)
        else:
            out.append(part)
    return '.'.join(out) or 'value'


# ---------------------------------------------------------------------------
# Structured invalidation reason
# ---------------------------------------------------------------------------

# Human-readable labels for the categories on :class:`CacheInvalidation`.
_INVALIDATION_LABELS = {
    'missing_manifest': 'no cache manifest',
    'missing_artifact': 'artifact missing',
    'schema': 'cache schema changed',
    'derivative': 'derivative changed',
    'version': 'derivative version changed',
    'key': 'key changed',
    'fingerprint': 'fingerprint changed',
    'dependencies': 'dependency changed',
}


@dataclass
class CacheInvalidation:
    """Why a cached artifact no longer matches the current request.

    ``category`` names which validity check failed; ``path``/``old``/``new``
    locate the first differing field for the ``key``/``fingerprint``/
    ``dependencies`` categories (empty for the structural categories).
    """

    category: str
    path: tuple[str, ...] = ()
    old: Any = None
    new: Any = None

    def field(self) -> str:
        return format_difference_path(self.path) if self.path else ''

    def message(self) -> str:
        base = _INVALIDATION_LABELS.get(self.category, self.category)
        if self.path:
            return f"{base} ({self.field()}: {self.old!r} -> {self.new!r})"
        return base

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {'category': self.category}
        if self.path:
            out['field'] = self.field()
            out['old'] = self.old
            out['new'] = self.new
        return out


def diff_invalidation(category: str, stored: Any, current: Any, *, strip_quick: bool = False) -> CacheInvalidation:
    """Build a :class:`CacheInvalidation` describing the first difference between two structures."""
    diff = find_difference(stored, current, strip_quick=strip_quick)
    if diff is None:
        return CacheInvalidation(category)
    return CacheInvalidation(category, *diff)


# ---------------------------------------------------------------------------
# Log rendering
# ---------------------------------------------------------------------------

# Fixed, positional columns appended (tab-separated) for records carrying
# ``cache_event``. Sharing one column order across entries keeps the structured
# tail compact and parseable as TSV instead of repeating keys on every line.
CACHE_EVENT_COLUMNS = ('event', 'derivative', 'category', 'field', 'old', 'new')


def _tsv_cell(value) -> str:
    if value is None:
        return ''
    return str(value).replace('\t', ' ').replace('\n', ' ')


class StructuredFormatter(logging.Formatter):
    """Formatter that appends tab-separated structured columns for cache events.

    Records emitted with ``extra={'cache_event': {...}}`` (see
    :meth:`Derivative._log_cache_event`) get the :data:`CACHE_EVENT_COLUMNS`
    values appended, tab-separated, after the human-readable message. Using a
    fixed column order shared across entries keeps the log file compact (no
    repeated keys) while remaining machine-parseable.
    """

    def format(self, record):
        msg = super().format(record)
        event = getattr(record, 'cache_event', None)
        if not event:
            return msg
        columns = '\t'.join(_tsv_cell(event.get(col)) for col in CACHE_EVENT_COLUMNS)
        return f"{msg}\t{columns}"
