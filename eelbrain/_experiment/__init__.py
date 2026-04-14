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
using :class:`Input` and :class:`Derivative` subclasses.
The :class:`DerivativeRegistry` exposes access to artifacts produced by this
pipeline through :meth:`DerivativeRegistry.load`.

Graph nodes and configuration
-----------------------------
Graph nodes implementing specific pipeline components live in modules such as
:mod:`eelbrain._experiment.preprocessing`,
:mod:`eelbrain._experiment.events`, and
:mod:`eelbrain._experiment.source`.

Each node represents one managed file or artifact family and is initialized
with the configuration it needs up front. Cache derivatives own their internal
artifact paths directly from semantic state/options. End-product export
derivatives may additionally define default public paths for reports, movies,
and similar user-facing outputs.

Graph nodes often have corresponding :class:`Configuration` objects that expose
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

Some configuration objects may govern to multiple dependency nodes.
For example, each configured :class:`RawPipe` produces its own raw derivative
node, and preprocessing with ICA requires an ICA :class:`Input`
node in addition to the raw derivative node that applies it.


3. :class:`Pipeline`
--------------------
The primary user facade is :class:`Pipeline`, defined in
:mod:`eelbrain._experiment.pipeline`. It is the public API, composes the
graph from user configuration, and assembles a complete normalized state from
initialization, :meth:`Pipeline.set`, and :meth:`Pipeline.load_*` / :meth:`Pipeline.make_*`
calls. That assembled state becomes the key used to resolve the relevant graph
nodes. :class:`Pipeline` exposes convenience methods, but caching and artifact
management belong to the lower layers.

- :class:`Pipeline` normalizes state and derivative-specific options
- :class:`Pipeline` resolves the target node through :class:`DerivativeRegistry`
- the derivative’s ``.build(ctx)` loads data from its dependencies through ``ctx.load(...)``
  to create the result requested by the Pipeline

In other words, :class:`Pipeline` methods like ``Pipeline.load_x`` and ``Pipeline.show_x``
are facades over graph nodes.

Configuration objects such as :class:`RawPipe`, epoch definitions, test
definitions and related objects define
analysis behavior in an extensible way. They are supplied by the user as
:class:`Pipeline` subclass attributes.
"""

from .state_model import StateModel
from .pipeline import Pipeline
from .test_def import ROITestResult
from .two_stage import ROI2StageResult, TwoStageTest
