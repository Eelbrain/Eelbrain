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
            spec.save_result(spec.make_job()())

Collecting the specs is free, but :attr:`JobSpec.is_done` is not: for an
artifact whose validity depends on its contents (an ICA file, whose manifest is
rebuilt from the saved object) it reads the artifact.

Nodes participate by implementing :meth:`DependencyNode.make_job` (and, for
nodes that do not write through :meth:`Request.save_artifact`,
:meth:`DependencyNode.save_result`). A node's ``build`` should be expressed as
``self.make_job(ctx)()`` so that local and off-host execution share one code
path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import CachePolicy, Request, file_fingerprint


@dataclass(frozen=True)
class JobProvenance:
    """What a job's inputs and target looked like when its data was loaded

    A job is computed from a snapshot: :meth:`DependencyNode.make_job` reads the
    inputs, and the result comes back an unbounded time later, by which point
    those inputs may have moved on. Writing the manifest from the *current*
    inputs would file the result under data it was not computed from, and the
    cache would then report the stale artifact as valid forever. So
    :meth:`JobSpec.make_job` records this, and the manifest is written from it.

    The recorded artifact is then correctly stale rather than wrongly valid: the
    ordinary validity check sees the recorded inputs differ from the current ones
    and schedules a rebuild, and no expensive result is thrown away in the
    meantime (reverting the input change makes it valid again).

    Attributes
    ----------
    dependencies
        Dependency fingerprints as of the load. Everything a job is computed
        from is loaded through ``ctx.load(...)``, so this covers all of it; the
        node's own ``fingerprint`` is definitions and state, which the request
        holds fixed anyway.
    artifact
        Fingerprint of the target artifact as of the start of
        :meth:`JobSpec.make_job` -- before the inputs are loaded, i.e. as close
        as possible to the moment the node authorizes replacing that specific
        file -- or ``None`` when there was no file. Only recorded for
        :attr:`CachePolicy.EXTERNAL`, whose artifact the cache does not own and
        must not clobber; a cache-owned artifact is always safe to overwrite.
    """
    dependencies: dict[str, Any]
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
        Cache key of the artifact this job computes; correlates a result with
        the :class:`JobSpec` that generated the job. Keyword-only, so
        subclasses can declare positional fields of their own and still inherit
        it.
    """
    key: dict[str, Any] | None = field(default=None, kw_only=True)

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
        """Load the data on the host and build a picklable :class:`Job`

        A job in hand is the host committing to computing this artifact, so this
        is where the cache-build message is emitted for a spec-driven
        computation -- the counterpart to :meth:`Request.load_artifact` emitting
        it for an artifact built in place. It is emitted only once
        :meth:`DependencyNode.make_job` has succeeded, so a refused job (e.g. a
        protected artifact) does not announce a build that never happens, and a
        retry with added controls does not announce it twice.

        The inputs as of this moment are recorded on the request as a
        :class:`JobProvenance`, so that :meth:`save_result` files the result
        under the data it was computed from however long it takes to come back.
        Recording is free overall: the manifest is written from these
        fingerprints instead of walking the dependencies a second time.
        """
        ctx = self.ctx
        # Fingerprint the EXTERNAL artifact before loading the inputs: the node's own
        # protection check runs at the start of its make_job, and a file another
        # session writes during the (potentially long) load was never covered by that
        # check, so it must not become the baseline that save_result may overwrite.
        artifact = None
        if ctx.node.cache_policy is CachePolicy.EXTERNAL and self.path.exists():
            artifact = file_fingerprint(ctx.root, self.path)
        job = ctx.node.make_job(ctx)
        ctx.node.log_cache_build(ctx, self.path)
        ctx._job_provenance = JobProvenance(ctx.dependency_fingerprints(), artifact)
        return job

    def save_result(self, result) -> object:
        "Incorporate an externally computed result into the cache (artifact + manifest)"
        return self.ctx.node.save_result(self.ctx, result)

    def with_controls(self, *controls: str) -> JobSpec:
        """Copy of the same request with additional controls

        Used to authorize an operation that the plain request refuses, such as
        recomputing a protected artifact, without going through mutable
        pipeline state. The request is copied rather than re-resolved, so
        already normalized options (and which of them the caller actually
        provided) carry over unchanged.

        Parameters
        ----------
        controls
            Request controls to add to those the current request carries.
        """
        ctx = self.ctx
        return JobSpec(Request(
            node=ctx.node,
            registry=ctx.registry,
            state=ctx._state,
            options=dict(ctx.options),
            view_options=dict(ctx.view_options),
            controls=ctx.controls.union(controls),
            provided_key_options=ctx._provided_key_options,
        ))
