"""Structured, displayable dependency tree for one resolved request.

:meth:`~eelbrain._experiment.derivative_cache.base.DerivativeRegistry.dependency_tree`
resolves one request and returns a :class:`DependencyTree`. Since dependency
edges are context-sensitive (they can branch on state and options, and may even
load upstream artifacts to enumerate their edges), the tree always describes
one concrete request, not the static node graph.

Rendering is separate from traversal: the tree renders as ASCII text in the
terminal (:meth:`DependencyTree.text`, also its ``repr``) and as a graphviz
flow chart in Jupyter notebooks (:meth:`DependencyTree.graph`, used by the
HTML display hook when the optional ``graphviz`` package and binary are
available; the chart is scaled down to fit the width of the output area).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import graphviz


def _mapping_text(mapping: dict[str, Any] | None, *, values: bool = True) -> str | None:
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


def _label_lines(mapping: dict[str, Any] | None, *, values: bool = True) -> list[str]:
    """Multi-line graph-node label for a mapping (one clipped item per line keeps node boxes narrow)."""
    if not mapping:
        return []
    items = list(mapping.items())
    max_items = 6 if values else 8
    if values:
        lines = [_clip_segment(f"{key}={value!r}", 40) for key, value in items[:max_items]]
    else:
        lines = [_clip_segment(', '.join(str(key) for key, _ in items[:max_items]), 40)]
    if len(items) > max_items:
        lines.append(f"+{len(items) - max_items}")
    return lines


def _line_width(max_line_length: int | None) -> int:
    if max_line_length is None:
        return shutil.get_terminal_size(fallback=(100, 24)).columns
    if max_line_length < 16:
        raise ValueError(f"{max_line_length=}: needs to be at least 16")
    return max_line_length


def _clip_segment(text: str, available: int) -> str:
    if len(text) <= available:
        return text
    if available <= 1:
        return '…'
    return text[:available - 1].rstrip() + '…'


def _format_line(
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
        current += _clip_segment(segment.lstrip(), available)
    lines.append(current)
    return lines


@dataclass
class DependencyTreeNode:
    """One resolved request in a :class:`DependencyTree`.

    Attributes
    ----------
    name
        Registered node name.
    kind
        ``'derivative'`` for cache-tracked derivatives, ``'uncached'`` for
        :class:`~eelbrain._experiment.derivative_cache.base.CachePolicy.NEVER`
        derivatives, ``'input'`` for inputs.
    label
        Edge label, when it differs from ``name``.
    key
        Resolved cache key (cached derivatives only).
    state
        Canonicalized state overrides declared on the edge leading here.
    options
        Canonicalized options: the effective request options for the root, the
        options declared on the edge for dependencies.
    view
        Data view loaded through the edge leading here (or requested for the
        root), if any.
    identity
        Stable id for the artifact/value this request resolves to, derived
        from the node's effective key (not the full request state), so
        requests that differ only in key-irrelevant state share one identity.
    seen
        This identity already appeared earlier in the tree; children are
        omitted.
    children
        Resolved dependency requests.
    """
    name: str
    kind: Literal['derivative', 'uncached', 'input']
    label: str | None
    key: dict[str, Any] | None
    state: dict[str, Any]
    options: dict[str, Any]
    view: str | None
    identity: str
    seen: bool = False
    children: list[DependencyTreeNode] = field(default_factory=list)

    def _segments(self) -> list[str]:
        parts = []
        if self.label:
            parts.append(f"{self.label} -> ")
        parts.append(self.name)
        if self.kind == 'uncached':
            parts.append(' [uncached]')
        elif self.kind == 'derivative':
            parts.append(' [derivative]')
            if key_text := _mapping_text(self.key):
                parts.append(f" {{{key_text}}}")
        else:
            parts.append(' [input]')
        if state_text := _mapping_text(self.state):
            parts.append(f" [state: {state_text}]")
        if self.view:
            parts.append(f" [view: {self.view}]")
        if option_text := _mapping_text(self.options, values=False):
            parts.append(f" [options: {option_text}]")
        if self.seen:
            parts.append(' [seen]')
        return parts


class DependencyTree:
    """Resolved dependency tree for one request, rendering as text or flow chart.

    In a terminal, the ``repr`` is an ASCII tree (:meth:`text`); in a Jupyter
    notebook the tree displays as a graphviz flow chart when the optional
    ``graphviz`` package and binary are installed (falling back to text
    otherwise). :meth:`graph` returns the underlying
    :class:`graphviz.Digraph` for customization or export.
    """

    def __init__(self, root: DependencyTreeNode):
        self.root = root

    def text(self, max_line_length: int | None = None) -> str:
        """Format the tree as ASCII text.

        Parameters
        ----------
        max_line_length
            Maximum line length for the formatted tree. By default, infer the
            current terminal width and wrap long node descriptions onto
            continuation lines.
        """
        line_width = _line_width(max_line_length)
        lines: list[str] = []

        def append_node(node: DependencyTreeNode, prefix: str, is_last: bool, is_root: bool) -> None:
            if is_root:
                first_prefix = ''
                continuation_prefix = '    '
            else:
                first_prefix = prefix + ('└── ' if is_last else '├── ')
                continuation_prefix = prefix + ('    ' if is_last else '│   ')
            lines.extend(_format_line(first_prefix, continuation_prefix, node._segments(), line_width))
            child_prefix = prefix if is_root else continuation_prefix
            for i, child in enumerate(node.children):
                append_node(child, child_prefix, i == len(node.children) - 1, False)

        append_node(self.root, '', True, True)
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self.text()

    def graph(self, rankdir: Literal['TB', 'LR'] = 'TB') -> graphviz.Digraph:
        """Render the tree as a graphviz flow chart (requires the optional ``graphviz`` package).

        Requests are deduplicated by identity, so a value used by several
        dependents appears as one node with several incoming edges.

        Parameters
        ----------
        rankdir
            Layout direction: ``'TB'`` (top-to-bottom, the default) or
            ``'LR'`` (left-to-right, more compact for deep chains with little
            branching).
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("Rendering the dependency graph requires the optional graphviz package (pip install graphviz, or mamba install python-graphviz)") from None
        dot = graphviz.Digraph('dependencies')
        dot.attr(rankdir=rankdir, nodesep='0.2', ranksep='0.3')
        dot.attr('node', fontname='Helvetica', fontsize='10', margin='0.1,0.05')
        dot.attr('edge', fontname='Helvetica', fontsize='9')
        node_ids: dict[str, str] = {}
        edges: set[tuple[str, str, str]] = set()

        def add_node(node: DependencyTreeNode) -> str:
            if node.identity in node_ids:
                return node_ids[node.identity]
            node_id = f'n{len(node_ids)}'
            node_ids[node.identity] = node_id
            label_lines = [node.name, *_label_lines(node.key)]
            if node is self.root:
                # The root has no incoming edge to carry its options and view
                if not node.key:
                    label_lines.extend(_label_lines(node.options, values=False))
                if node.view:
                    label_lines.append(f"view: {node.view}")
            label = '\n'.join(label_lines)
            if node.kind == 'derivative':
                dot.node(node_id, label, shape='box', style='rounded,filled', fillcolor='lightblue')
            elif node.kind == 'uncached':
                dot.node(node_id, label, shape='box', style='rounded,dashed')
            else:
                dot.node(node_id, label, shape='ellipse', style='filled', fillcolor='lightgray')
            return node_id

        def add_edges(node: DependencyTreeNode, node_id: str) -> None:
            for child in node.children:
                child_id = add_node(child)
                label_parts = []
                if child.label:
                    label_parts.append(child.label)
                label_parts.extend(_label_lines(child.state))
                label_parts.extend(_label_lines(child.options, values=False))
                if child.view:
                    label_parts.append(f"view: {child.view}")
                edge_label = '\n'.join(label_parts)
                edge = (node_id, child_id, edge_label)
                if edge not in edges:
                    edges.add(edge)
                    dot.edge(node_id, child_id, label=edge_label or None)
                if not child.seen:
                    add_edges(child, child_id)

        add_edges(self.root, add_node(self.root))
        return dot

    def _repr_html_(self) -> str | None:
        try:
            import graphviz
        except ImportError:
            return None
        try:
            svg = self.graph().pipe(format='svg', encoding='utf-8')
        except graphviz.ExecutableNotFound:
            return None
        # Scale a graph that is wider than the notebook output area down to fit
        # (the style attribute overrides the fixed width/height attributes)
        svg = svg[svg.index('<svg'):]
        return svg.replace('<svg', '<svg style="max-width:100%;height:auto"', 1)
