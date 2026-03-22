# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Experiment pipeline architecture.

The :mod:`eelbrain._experiment` package is organized as a three-layer system
with strict one-way dependencies: higher layers use lower layers, and lower
layers must not reference higher layers.

The cache kernel lives in :mod:`eelbrain._experiment.derivative_cache`. It is
fully generic and owns manifests, normalized keys, dependency traversal,
protected artifacts, and cache policy. It does not know experiment-specific
concepts such as raw pipes, epochs, tests, reports, or :class:`Pipeline`.

Graph nodes live in modules such as :mod:`eelbrain._experiment.preprocessing`,
:mod:`eelbrain._experiment.events`, :mod:`eelbrain._experiment.source`,
:mod:`eelbrain._experiment.parc`, :mod:`eelbrain._experiment.results`, and
:mod:`eelbrain._experiment.reports`. These inputs and derivatives use the
cache kernel to manage concrete artifacts. Each node represents one managed
file or artifact family, has a path template plus logic for turning a key into
a concrete path, and is initialized with the configuration and behavior it
needs up front. Request-time state is supplied with the load/build request and
selects the keyed instance to materialize. Nodes may depend on the cache
kernel and on injected configuration or behavior objects, but they must not
depend on :class:`Pipeline`.

The user facade is :class:`Pipeline`, defined in
:mod:`eelbrain._experiment.mne_experiment`. It is the public API, composes the
graph from user configuration, and assembles state from initialization,
:meth:`Pipeline.set`, and :meth:`Pipeline.load_*` / :meth:`Pipeline.make_*`
calls. That assembled state becomes the key used to resolve the relevant graph
nodes. :class:`Pipeline` exposes convenience methods, but caching and artifact
management belong to the lower layers.

Configuration objects such as :class:`RawPipe`, epoch definitions, test
definitions, parcellations, covariance definitions, and related objects define
analysis behavior in an extensible way. They specify processing behavior and
node-specific configuration, but do not implement caching policy themselves.
For example, a :class:`RawPipe` subclass lets the user choose preprocessing
options and provide the actual processing implementation, while cache behavior
is supplied by the lower layers.
"""

from .experiment import TreeModel, FileTree
from .mne_experiment import Pipeline
from .test_def import ROITestResult, ROI2StageResult
