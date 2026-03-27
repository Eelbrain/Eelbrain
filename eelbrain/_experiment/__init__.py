# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Experiment pipeline architecture.

The :mod:`eelbrain._experiment` package is organized as a three-layer system
with strict one-way dependencies: higher layers use lower layers, but lower
layers are independent/unaware of higher layers. The layers are:

1. Abstract cache and dependency graph kernel
2. Domain-specific graph nodes and configuration
3. :class:`Pipeline` user interface

Higher layers may use lower layers, but lower layers must not reference higher
layers.

Cache kernel
------------
The cache kernel lives in :mod:`eelbrain._experiment.derivative_cache`. It is
fully generic and owns manifests, normalized keys, dependency traversal,
protected artifacts, and cache policy. It does not know experiment-specific
concepts such as raw pipes, epochs, tests, reports, or :class:`Pipeline`.

The graph is constructed from :class:`DependencyNode` instances, typically
typically using subclasses based on :class:`Input` and :class:`Derivative`.
The :class:`DerivativeRegistry` exposes access to artifact produces by this
pipeline through :meth:`DerivativeRegistry.load`.

Graph nodes and configuration
-----------------------------
Graph nodes implementing specific pipeline components live in modules such as
:mod:`eelbrain._experiment.preprocessing`,
:mod:`eelbrain._experiment.events`, :mod:`eelbrain._experiment.source`,
:mod:`eelbrain._experiment.parc`, :mod:`eelbrain._experiment.results`, and
:mod:`eelbrain._experiment.reports`.

Each node represents one managed file or artifact family and is initialized
with the configuration it needs up front. Cache derivatives own their internal
artifact paths directly from semantic state/options. End-product export
derivatives may additionally define default public paths for reports, movies,
and similar user-facing outputs.


Graph nodes often have corresponding configuration objects that expose
user settings.
Configuration objects may also serve as plugin-like extensions, with multiple different
configuration objects corresponding to one type of :class:`Derivative`.
For example, multiple :class:`RawPipe` subclasses define different preprocessing steps.
Such configuration objects may also include substantial domain behavior, including
implementation code for analysis steps, and are bound into graph nodes when
:class:`Pipeline` is initialized. They specify processing behavior and
node-specific configuration, but do not implement caching policy, manifests,
dependency traversal, or protected-artifact handling themselves. For example,
a :class:`RawPipe` subclass
1) provides the actual processing implementation through one of its methods, and
2) lets the user choose preprocessing options and parameters through
initialization parameters.

Graph-layer behavior consumed by nodes must itself be explicit-state and
pipeline-free. Lower layers may use pure helper functions in existing graph
modules, but must not capture :class:`Pipeline`, depend on temporary facade
state, or rely on :class:`FileTree` state-mutation helpers such as
``set()``/``format()``/``iter()``/``show_state()``.

Naming-only fields used for human-readable export paths must not live in
canonical derivative state or cache fingerprints. When a derivative needs
functionality that was previously implemented in ``Pipeline.load_x`` or
``Pipeline.show_x``, that behavior belongs in the dependency derivative or in
pure lower-layer helpers, not in the facade.

Cache behavior is always supplied by the lower layers.
Extending the system should work by adding configuration
objects such as new :class:`RawPipe` subclasses, not by editing cache-kernel
code.

Some configuration objects may result in multiple dependency nodes.
For example, preprocessing with ICA requires both ICA files from an
:class:`Input` node and projecting the data in a :class:`Derivative` node.


3. :class:`Pipeline`
--------------------
The primary user facade is :class:`Pipeline`, defined in
:mod:`eelbrain._experiment.mne_experiment`. It is the public API, composes the
graph from user configuration, and assembles a complete normalized state from
initialization, :meth:`Pipeline.set`, and :meth:`Pipeline.load_*` / :meth:`Pipeline.make_*`
calls. That assembled state becomes the key used to resolve the relevant graph
nodes. :class:`Pipeline` exposes convenience methods, but caching and artifact
management belong to the lower layers.

- :class:`Pipeline` normalizes state and derivative-specific options
- :class:`Pipeline` resolves the target node through :class:`DerivativeRegistry`
- the derivative’s ``.build(ctx)` loads data from its dependencies through ``ctx.load(...)``
  to create the result requested by the Pipeline

In other words, ``Pipeline.load_x`` and ``Pipeline.show_x`` are facades over
graph nodes, not execution backends.

Configuration objects such as :class:`RawPipe`, epoch definitions, test
definitions and related objects define
analysis behavior in an extensible way. They are supplied by the user through
:class:`Pipeline`.
"""

from .experiment import TreeModel, FileTree
from .mne_experiment import Pipeline
from .test_def import ROITestResult, ROI2StageResult
