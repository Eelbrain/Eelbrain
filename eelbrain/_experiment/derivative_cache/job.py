# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Separable, picklable computing jobs

An expensive artifact is computed in two stages, so that a single artifact can
be computed on a machine that does not have the raw data:

- :class:`JobSpec` is host-side, internal machinery. It holds only a reference
  to the resolved request (node, state and options), generates the
  corresponding :class:`Job`, checks whether the artifact is already cached,
  and incorporates an externally computed result back into the cache.
- :class:`Job` is picklable and *data-carrying*: it holds all already-loaded
  inputs, so it can be shipped to another machine, computed there, and the
  result pickled back.


Collecting ``JobSpec``
----------------------
A :class:`JobSpec` carries no data, so specs can be collected in bulk across an iteration
without loading anything (:class:`Request` snapshots the state it is resolved
with, so a spec stays bound to the state it was created in)::

    specs = []
    for _ in pipeline.iter(['subject', 'session']):
        specs.append(pipeline._job_spec(ica_input_name('ica')))
    for spec in specs:
        if not spec.is_done:
            job = spec.make_job()
            spec.save_result(job, job())

Collecting the specs is free, but :attr:`JobSpec.is_done` is not: it re-derives
the current manifest, which walks the dependency fingerprints (file stats, bad
channels, ...).

Nodes participate by implementing :meth:`DependencyNode.make_job` (and, for
nodes that do not write through :meth:`Request.save_artifact`,
:meth:`DependencyNode.save_result`). A node's ``build`` should be expressed as
``self.make_job(ctx)()`` so that local and off-host execution share one code
path.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..logging import find_difference, format_difference_path
from .base import CachePolicy, Derivative, JobInputsChangedError, Request, file_fingerprint


@dataclass(frozen=True)
class JobProvenance:
    """What a job's inputs and target looked like when its data was read

    A job is computed from a snapshot: :meth:`DependencyNode.make_job` reads the
    inputs, and the result comes back an unbounded time later, by which point
    those inputs may have moved on. Writing the manifest from the *current*
    inputs would file the result under data it was not computed from, and the
    cache would then report the stale artifact as valid forever. So
    :meth:`JobSpec.make_job` records this on the job it returns, and the
    manifest is written from it when that job's result is saved.

    The recorded artifact is then correctly stale rather than wrongly valid: the
    ordinary validity check sees the recorded inputs differ from the current ones
    and schedules a rebuild, and no expensive result is thrown away in the
    meantime (reverting the input change makes it valid again).

    Attributes
    ----------
    dependencies
        Dependency fingerprints as of before the load, verified unchanged after
        it. Everything a job is computed from is loaded through
        ``ctx.load(...)``, so this covers all of it.
    fingerprint
        The node's own fingerprint, taken and verified the same way. Mostly
        definitions and state, which the request holds fixed -- but a node may
        fingerprint mutable state outside its dependencies (e.g. ICA bad
        channels), and the manifest must record what the job was computed from,
        not what that state became by the time the result was saved.
    artifact
        Fingerprint of the target artifact as of the start of
        :meth:`JobSpec.make_job` -- before the inputs are loaded, i.e. as close
        as possible to the moment the node authorizes replacing that specific
        file -- or ``None`` when there was no file. Only recorded for
        :attr:`CachePolicy.EXTERNAL`, whose artifact the cache does not own and
        must not clobber; a cache-owned artifact is always safe to overwrite.
    """
    dependencies: dict[str, Any]
    fingerprint: dict[str, Any]
    artifact: dict[str, Any] | None = None


@dataclass(frozen=True)
class Job:
    """Picklable, data-carrying unit of work computing one artifact

    Subclasses are frozen dataclasses holding already-loaded inputs, so a job
    can be pickled, executed on a machine without access to the raw data, and
    the result pickled back. Created host-side by
    :meth:`DependencyNode.make_job`, and re-united with its cache entry through
    :meth:`JobSpec.save_result`.

    Parameters
    ----------
    key
        Cache key of the artifact this job computes. A key is only unique
        within one node's cache namespace (two ICA steps over the same
        recording share one key), so correlating a result with the
        :class:`JobSpec` that generated the job takes ``(node, key)``.
    node
        Name of the registered node the job computes an artifact for.
    provenance
        What the inputs looked like when this job's data was read.

    Notes
    -----
    All three fields are stamped by :meth:`JobSpec.make_job`, not by the node's
    ``make_job`` implementation (and stay ``None`` on a job made for an
    in-place build, which never leaves its request). Keyword-only, so
    subclasses can declare positional fields of their own and still inherit
    them.
    """
    key: dict[str, Any] | None = field(default=None, kw_only=True)
    node: str | None = field(default=None, kw_only=True)
    provenance: JobProvenance | None = field(default=None, kw_only=True)

    def __call__(self):
        """Compute and return the artifact.

        Takes no arguments: everything the computation needs is a field, so the
        same call works here and on a machine that has nothing but the unpickled
        job. Reporting that the artifact is being computed is the cache's job,
        not this one's (see the module docstring).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class JobSpec:
    """Host-side handle for computing one artifact (internal)

    References the resolved request needed to (re)generate a data-carrying
    :class:`Job` and to incorporate an externally computed result into the
    cache. Cheap to create and to collect in bulk; not required to be
    picklable. Not thread-safe: :meth:`make_job` and :meth:`save_result` use
    the shared registry and the file system.

    Parameters
    ----------
    ctx
        Resolved request for the artifact (carries node, state and options).
    """
    ctx: Request

    @property
    def path(self) -> Path:
        "Target artifact path"
        return self.ctx.artifact_path

    @property
    def key(self) -> dict[str, Any]:
        "Cache key identifying the artifact"
        return self.ctx.key()

    @property
    def is_done(self) -> bool:
        "Whether a valid artifact already exists (may read the artifact, see module docstring)"
        return self.ctx.is_valid()

    def make_job(self) -> Job:
        """Load the data on the host and build a picklable :class:`Job`"""
        ctx = self.ctx
        # Fingerprint the EXTERNAL artifact before loading the inputs: the node's own
        # protection check runs at the start of its make_job, and a file another
        # session writes during the (potentially long) load was never covered by that
        # check, so it must not become the baseline that save_result may overwrite.
        artifact = None
        if ctx.node.cache_policy is CachePolicy.EXTERNAL and self.path.exists():
            artifact = file_fingerprint(ctx.root, self.path)
        # Fingerprint the inputs before they are read, and check afterward that they
        # held still: the job carries data read during the load, so fingerprints taken
        # after it would describe data the job does not hold -- and being equal to the
        # current state, they would mark that artifact valid rather than stale.
        provenance = JobProvenance(ctx.dependency_fingerprints(), ctx.current_fingerprint(), artifact)
        # For a derivative, the same contexts load_artifact wraps build() in, so its
        # loads are restricted to declared dependencies and key fields, and its
        # warnings are recorded, whether the artifact is computed in place or through
        # a job. An input has no in-place build to mirror, and may load beyond its
        # declared dependency edges (ICA loads bad channels and per-run source raws).
        if isinstance(ctx.node, Derivative):
            with ctx._build_deps_context(), ctx.registry._node_warning_context(ctx), ctx._state_check_context():
                job = ctx.node.make_job(ctx)
        else:
            job = ctx.node.make_job(ctx)
        # Check that the inputs are unchanged
        for before, after in ((provenance.dependencies, ctx.dependency_fingerprints()), (provenance.fingerprint, ctx.current_fingerprint())):
            difference = find_difference(before, after)
            if difference is not None:
                path, old, new = difference
                raise JobInputsChangedError(ctx.node.name, format_difference_path(path), old, new)

        ctx.node.log_job(ctx, self.path)
        return replace(job, key=self.key, node=ctx.node.name, provenance=provenance)

    def save_result(self, job: Job, result: Any) -> object:
        """Incorporate an externally computed result into the cache (artifact + manifest)

        Both halves are required: ``result`` is what came back, and ``job`` is
        what computed it, carrying the :class:`JobProvenance` the result is
        filed under. Pairing them here is what makes a late result safe -- the
        snapshot comes from the job that holds the data, so it stays right
        however many jobs this spec has made in the meantime, and a result can
        be filed long after the request has moved on.

        Parameters
        ----------
        job
            The job that produced ``result``, as returned by :meth:`make_job`.
        result
            The computed artifact value, i.e. ``job()``.
        """
        ctx = self.ctx
        if job.provenance is None:
            raise RuntimeError(f"{ctx.node.name!r}: this job carries no provenance -- only a job from JobSpec.make_job() can be saved")
        if (job.node, job.key) != (ctx.node.name, self.key):
            raise RuntimeError(f"job computes {job.node!r} {job.key} but this spec is for {ctx.node.name!r} {self.key}")
        return ctx.node.save_result(ctx, result, job.provenance)

    def with_controls(self, *controls: str) -> JobSpec:
        """Spec for a copy of the same request with additional controls

        Used to authorize an operation that the plain request refuses, such as
        recomputing a protected artifact (see :meth:`Request.with_controls`).

        Parameters
        ----------
        controls
            Request controls to add to those the current request carries.
        """
        return JobSpec(self.ctx.with_controls(*controls))
