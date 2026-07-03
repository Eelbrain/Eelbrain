# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Movie export derivatives.

These derivatives orchestrate through other graph nodes, especially the
dataset-producing derivatives that correspond to the public ``Pipeline.load_x``
methods. They share the result-output machinery in
:mod:`._experiment.statistics.nodes`; only end-product exports such as movies
own default public paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import plot
from .. import testnd
from .._stats.stats import ttest_t
from .derivative_cache import Dependency, Request
from .pathing import epoch_basename, join_stem_parts, movie_export_path
from .statistics.nodes import RESULT_OPTION_DEFAULTS, ResultOutputDerivative, _epochs_stc_options, _evoked_stc_options

_MOVIE_OPTION_DEFAULTS = {
    'time_dilation': 1.0,
    'single_subject': False,
    'subject': None,
}


class DSPMMovieDerivative(ResultOutputDerivative[Path]):
    """Rendered grand-average dSPM movie export.

    Options
    -------
    fmin
        Minimum value for the dSPM color scale.
    brain_kwargs
        Keyword arguments forwarded to :func:`~eelbrain.plot.brain.dspm`,
        including ``surf``.
    time_dilation
        Playback speed multiplier (``> 1`` slows down).
    single_subject
        Render one subject instead of the group average.
    subject
        Subject to render when ``single_subject=True``.
    """
    name = 'movie-dspm'
    OPTION_DEFAULTS = {**RESULT_OPTION_DEFAULTS, **_MOVIE_OPTION_DEFAULTS, 'fmin': None, 'brain_kwargs': None}

    def _identity_extra(self, ctx: Request) -> dict[str, Any]:
        return {
            'time_dilation': ctx.options['time_dilation'],
            'fmin': ctx.options['fmin'],
            'brain_kwargs': ctx.registry.canonicalize(ctx.options['brain_kwargs']),
        }

    def _path_stem(self, ctx: Request) -> str:
        return join_stem_parts(
            epoch_basename(ctx.state),
            f'epoch-{ctx.state["epoch"]}',
            'ga-dspm',
            self._path_context_parts(ctx),
            self._path_option_parts(ctx),
            f'surf-{ctx.options["brain_kwargs"]["surf"]}',
            f'fmin-{ctx.options["fmin"]:g}',
        )

    def _default_output_path(self, ctx: Request) -> Path:
        return ctx.root / movie_export_path(ctx.state, self._path_stem(ctx), ctx.options['single_subject'])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if ctx.options['single_subject']:
            return (Dependency('evoked-stc', state={'subject': ctx.options['subject']}, options=_evoked_stc_options(ctx)),)
        return (Dependency('evoked-stc-group-dataset', options=_evoked_stc_options(ctx, morph=True)),)

    def build(self, ctx: Request) -> Path:
        dst = self.path(ctx)
        if ctx.options['single_subject']:
            ds = ctx.load('evoked-stc')
            y = ds['src']
        else:
            ds = ctx.load('evoked-stc-group-dataset')
            y = ds['srcm']
        brain = plot.brain.dspm(y, ctx.options['fmin'], ctx.options['fmin'] * 3, colorbar=False, **ctx.options['brain_kwargs'])
        brain.save_movie(dst, ctx.options['time_dilation'])
        brain.close()
        return dst


class TTestMovieDerivative(ResultOutputDerivative[Path]):
    """Rendered t-test movie export.

    Options
    -------
    cat
        Condition subset; two conditions perform a contrast, one performs a
        one-sample test against zero.
    p
        Significance threshold for the cluster mask.
    pmid
        Midpoint of the t-value color scale.
    surf
        Surface for the brain rendering.
    cluster_state
        Extra keyword arguments forwarded to the cluster test.
    disconnect_labels
        Disconnect source-space cluster adjacency across parcellation labels.
    time_dilation
        Playback speed multiplier (``> 1`` slows down).
    single_subject
        Render one subject instead of the group.
    subject
        Subject to render when ``single_subject=True``.
    """
    name = 'movie-ttest'
    OPTION_DEFAULTS = {
        **RESULT_OPTION_DEFAULTS,
        **_MOVIE_OPTION_DEFAULTS,
        'disconnect_labels': False,
        'cat': None,
        'p': None,
        'pmid': None,
        'surf': None,
        'cluster_state': None,
    }

    def _identity_extra(self, ctx: Request) -> dict[str, Any]:
        return {
            'time_dilation': ctx.options['time_dilation'],
            'cat': ctx.options['cat'],
            'p': ctx.options['p'],
            'pmin': ctx.options['pmin'],
            'pmid': ctx.options['pmid'],
            'surf': ctx.options['surf'],
            'cluster_state': ctx.registry.canonicalize(ctx.options['cluster_state'] or {}),
        }

    def _path_stem(self, ctx: Request) -> str:
        cat = ctx.options['cat']
        if not cat:
            contrast = 'ga'
        elif len(cat) == 1:
            contrast = f'{cat[0]}'
        else:
            contrast = f'{cat[0]}-{cat[1]}'
        return join_stem_parts(
            epoch_basename(ctx.state),
            f'epoch-{ctx.state["epoch"]}',
            'ttest',
            self._path_context_parts(ctx),
            self._path_option_parts(ctx),
            f'contrast-{contrast}',
            f'p-{ctx.options["p"]}',
            f'surf-{ctx.options["surf"]}',
        )

    def _default_output_path(self, ctx: Request) -> Path:
        return ctx.root / movie_export_path(ctx.state, self._path_stem(ctx), ctx.options['single_subject'])

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if ctx.options['single_subject']:
            return (Dependency('epochs-stc', state={'subject': ctx.options['subject']}, options=_epochs_stc_options(ctx)),)
        return (Dependency('evoked-stc-group-dataset', options=_evoked_stc_options(ctx, morph=True, cat=ctx.options['cat'])),)

    def build(self, ctx: Request) -> Path:
        dst = self.path(ctx)
        cluster_state = dict(ctx.options['cluster_state'] or {})
        if ctx.options['single_subject']:
            ds = ctx.load('epochs-stc')
            y = 'src'
        else:
            ds = ctx.load('evoked-stc-group-dataset')
            y = 'srcm'
        if cluster_state:
            cluster_state.update(samples=0, pmin=ctx.options['p'])
        if ctx.options['disconnect_labels']:
            cluster_state['parc'] = 'source'
        cat = ctx.options['cat']
        if ctx.state['model'] and cat and len(cat) == 2:
            c1, c0 = cat
            if ctx.options['single_subject']:
                res = testnd.TTestIndependent(y, ctx.state['model'], c1, c0, data=ds, **cluster_state)
            else:
                res = testnd.TTestRelated(y, ctx.state['model'], c1, c0, match='subject', data=ds, **cluster_state)
        elif cat:
            res = testnd.TTestOneSample(y, data=ds, **cluster_state)
        else:
            res = testnd.TTestOneSample(y, data=ds, **cluster_state)
        tmap = res.masked_parameter_map(None) if cluster_state else res.t
        brain = plot.brain.dspm(
            tmap,
            ttest_t(ctx.options['p'], res.df),
            ttest_t(ctx.options['pmin'], res.df),
            ttest_t(ctx.options['pmid'], res.df),
            surf=ctx.options['surf'],
        )
        brain.save_movie(dst, ctx.options['time_dilation'])
        brain.close()
        return dst
