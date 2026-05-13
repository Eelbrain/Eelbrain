# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np

from .. import fmtxt
from .. import plot
from .. import test
from .. import testnd
from .._data_obj import Dataset, combine
from .._stats.stats import ttest_t
from .._text import ms
from ..fmtxt import Section, linebreak


def source_results(res, surfer_kwargs={}, title="Results", diff_cmap=None,
                   table_pmax=0.2, plot_pmax=0.05):
    "Only used for TRF-report"
    sec = Section(title)

    # raw difference
    brain = plot.brain.surfer_brain(res.difference, diff_cmap, **surfer_kwargs)
    cbar = brain.plot_colorbar(orientation='vertical', show=False)
    sec.add_figure("Correlation increase.", (brain.image('correlation'), cbar))
    brain.close()
    cbar.close()

    # test of difference
    if res._kind == 'cluster':
        clusters = res.find_clusters(table_pmax)
        pmax_repr = str(table_pmax)[1:]
        ctable = clusters.as_table(midrule=True, count=True, caption="All "
                                   "clusters with p<=%s." % pmax_repr)
        sec.append(ctable)

        clusters = res.find_clusters(plot_pmax, True)
        for cluster in clusters.itercases():
            # only plot relevant hemisphere
            sec.add_figure("Cluster %i: p=%.3f" % (cluster['id'], cluster['p']),
                           source_cluster_im(cluster['cluster'], surfer_kwargs),
                           {'class': 'float'})
        sec.append(linebreak)
    return sec


def source_cluster_im(ndvar, surfer_kwargs, mark_sources=None):
    """Plot ('source',) NDVar, only plot relevant hemi

    Parameters
    ----------
    ndvar : NDVar (source,)
        Source space data.
    surfer_kwargs : dict
        Keyword arguments for PySurfer plot.
    mark_sources : SourceSpace index
        Sources to mark on the brain plot (as SourceSpace index).
    """
    kwargs = surfer_kwargs.copy()
    if not ndvar.sub(source='lh').any():
        kwargs['hemi'] = 'rh'
    elif not ndvar.sub(source='rh').any():
        kwargs['hemi'] = 'lh'
    if ndvar.x.dtype.kind == 'b':
        brain = plot.brain.dspm(ndvar, 0, 1.5, **kwargs)
    elif ndvar.x.dtype.kind == 'i':  # map of cluster ids
        brain = plot.brain.surfer_brain(ndvar, 'jet', **kwargs)
    else:
        brain = plot.brain.cluster(ndvar, **kwargs)

    # mark sources on the brain
    if mark_sources is not None:
        mark_sources = np.atleast_1d(ndvar.source._array_index(mark_sources))
        i_hemi_split = np.searchsorted(mark_sources, ndvar.source.lh_n)
        lh_indexes = mark_sources[:i_hemi_split]
        if lh_indexes:
            lh_vertices = ndvar.source.lh_vertices[lh_indexes]
            brain.add_foci(lh_vertices, True, hemi='lh', color="gold")
        rh_indexes = mark_sources[i_hemi_split:]
        if rh_indexes:
            rh_vertices = ndvar.source.rh_vertices[rh_indexes - ndvar.source.lh_n]
            brain.add_foci(rh_vertices, True, hemi='rh', color="gold")

    out = brain.image(ndvar.name)
    brain.close()
    return out


def source_time_results(
        res: testnd.NDTest,
        data: Dataset,
        colors: dict = None,
        include: float = 0.1,
        surfer_kwargs: dict = {},
        title: str = None,
        parc: bool = True,
        y: str = None,
):
    report = Section(title)
    y = data[y or res.y]
    if parc is True:
        parc = res._first_cdist.parc
    model = res._plot_model()
    if parc and res._kind == 'cluster':
        source_bin_table(report, res, surfer_kwargs)

        # add subsections for individual labels
        title = "{tstart}-{tstop} p={p}{mark} {effect}"
        for label in y.source.parc.cells:
            section = report.add_section(label.capitalize())
            clusters = res.find_clusters(source=label)
            source_time_clusters(section, clusters, y, data, model, include, title, colors, res, surfer_kwargs)
    elif not parc and res._kind == 'cluster':
        source_bin_table(report, res, surfer_kwargs)
        clusters = res.find_clusters()
        clusters.sort('tstart')
        title = "{tstart}-{tstop} {location} p={p}{mark} {effect}"
        source_time_clusters(report, clusters, y, data, model, include, title, colors, res, surfer_kwargs)
    elif not parc and res._kind in ('raw', 'tfce'):
        section = report.add_section("P<=.05")
        source_bin_table(section, res, surfer_kwargs, 0.05)
        clusters = res.find_clusters(0.05, maps=True)
        clusters.sort('tstart')
        title = "{tstart}-{tstop} {location} p={p}{mark} {effect}"
        for cluster in clusters.itercases():
            source_time_cluster(section, cluster, y, model, data, title, colors, res.match, surfer_kwargs)

        # trend section
        section = report.add_section("Trend: p<=.1")
        source_bin_table(section, res, surfer_kwargs, 0.1)

        # not quite there section
        section = report.add_section("Anything: P<=.2")
        source_bin_table(section, res, surfer_kwargs, 0.2)
    elif parc and res._kind in ('raw', 'tfce'):
        title = "{tstart}-{tstop} p={p}{mark} {effect}"
        for label in y.source.parc.cells:
            section = report.add_section(label.capitalize())
            # TODO:  **sub is not implemented in find_clusters()
            clusters_sig = res.find_clusters(0.05, True, source=label)
            clusters_trend = res.find_clusters(0.1, True, source=label)
            clusters_trend = clusters_trend.sub("p>0.05")
            clusters_all = res.find_clusters(0.2, True, source=label)
            clusters_all = clusters_all.sub("p>0.1")
            clusters = combine((clusters_sig, clusters_trend, clusters_all))
            clusters.sort('tstart')
            source_time_clusters(section, clusters, y, data, model, include,
                                 title, colors, res, surfer_kwargs)
    else:
        raise RuntimeError
    return report


def source_bin_table(section, res, surfer_kwargs, pmin=1):
    caption = "All clusters in time bins. Each plot shows all sources that are part of a cluster at any time during the relevant time bin. Only the general minimum duration and source number criterion are applied."
    for effect, cdist in res._iter_cdists():
        ndvar = cdist.masked_parameter_map(pmin)
        if not ndvar.any():
            if effect:
                text = f'{effect}: nothing\n'
            else:
                text = 'Nothing\n'
            section.add_paragraph(text)
            continue
        elif effect:
            caption_ = f"{effect}: {caption}"
        else:
            caption_ = caption
        im = plot.brain.bin_table(ndvar, **surfer_kwargs)
        section.add_image_figure(im, caption_)


def source_time_lm(lm, pmin, surfer_kwargs):
    if pmin == 0.1:
        ps = (0.1, 0.01, 0.05)
    elif pmin == 0.05:
        ps = (0.05, 0.001, 0.01)
    elif pmin == 0.01:
        ps = (0.01, 0.0001, 0.001)
    elif pmin == 0.001:
        ps = (0.001, 0.00001, 0.0001)
    else:
        raise ValueError(f"{pmin=}")
    out = Section("SPMs")
    ts = [ttest_t(p, lm.df) for p in ps]
    for term in lm.column_names:
        im = plot.brain.dspm_bin_table(lm.t(term), *ts, summary='extrema',
                                       **surfer_kwargs)
        out.add_section(term, im)
    return out


def source_time_clusters(section, clusters, y, ds, model, include, title,
                         colors, res, surfer_kwargs):
    """Plot cluster with source and time dimensions

    Parameters
    ----------
    ...
    legend : None | fmtxt.Image
        Legend (if shared with other figures).

    Returns
    -------
    legend : fmtxt.Image
        Legend to share with other figures.
    """
    # compute clusters
    if clusters.n_cases == 0:
        section.append("No clusters found.")
        return

    section.append(
        clusters.as_table(midrule=True, count=True, caption="All clusters."))

    # plot individual clusters
    clusters = clusters.sub(f"p < {include}")
    # in non-threshold based tests, clusters don't have unique IDs
    add_cluster_im = 'cluster' not in clusters
    is_multi_effect_result = 'effect' in clusters
    for cluster in clusters.itercases():
        if add_cluster_im:
            if is_multi_effect_result:
                cluster['cluster'] = res.cluster(cluster['id'], cluster['effect'])
            else:
                cluster['cluster'] = res.cluster(cluster['id'])
        source_time_cluster(section, cluster, y, model, ds, title, colors,
                            res.match, surfer_kwargs)


def source_time_cluster(section, cluster, y, model, ds, title, colors, match,
                        surfer_kwargs):
    # cluster properties
    tstart_ms = ms(cluster['tstart'])
    tstop_ms = ms(cluster['tstop'])
    effect = cluster.get('effect', '')

    # section/title
    if title is not None:
        title_ = title.format(tstart=tstart_ms, tstop=tstop_ms,
                              p=f"{cluster['p']:.3f}", effect=effect,
                              location=cluster.get('location', ''),
                              mark=cluster['sig']).strip()
        while '  ' in title_:
            title_ = title_.replace('  ', ' ')
        section = section.add_section(title_)

    # description
    txt = section.add_paragraph("Id %i" % cluster['id'])
    if 'v' in cluster:
        txt.append(f", v={cluster['v']}")
    if 'p_parc' in cluster:
        txt.append(", corrected across all ROIs: ")
        txt.append(fmtxt.eq('p', cluster['p_parc'], 'mcc', '%s', drop0=True))
    txt.append('.')

    # add cluster image to report
    brain = plot.brain.cluster(cluster['cluster'].sum('time'), **surfer_kwargs)
    cbar = brain.plot_colorbar(orientation='vertical', show=False)
    caption = "Cluster"
    if effect:
        caption += 'effect of ' + effect
    caption += "%i - %i ms." % (tstart_ms, tstop_ms)
    section.add_figure(caption, (brain.image('cluster_spatial'), cbar))
    brain.close()
    cbar.close()
    # add cluster time course
    if effect:
        reduced_model = '%'.join(effect.split(' x '))
        if len(reduced_model) < len(model):
            colors_ = plot.colors_for_categorial(ds.eval(reduced_model))
            cluster_timecourse(section, cluster, y, 'source', reduced_model, ds,
                               colors_, match)
    cluster_timecourse(section, cluster, y, 'source', model, ds, colors, match)


def cluster_timecourse(section, cluster, y, dim, model, ds, colors, match):
    c_extent = cluster['cluster']
    cid = cluster['id']

    # cluster time course
    idx = c_extent.any('time')
    tc = y[idx].mean(dim)
    p = plot.UTSStat(tc, model, match=match, data=ds, legend=False, h=4,
                     colors=colors, show=False)
    # mark original cluster
    for ax in p.axes:
        ax.axvspan(cluster['tstart'], cluster['tstop'], color='r',
                   alpha=0.2, zorder=-2)

    if model:
        # legend
        legend_p = p.plot_legend(show=False)
        legend = legend_p.image("Legend")
        legend_p.close()
    else:
        p.axes[0].axhline(0, color='k')
    image_tc = p.image('cluster_%i_timecourse' % cid)
    p.close()

    # Barplot
    idx = (c_extent != 0)
    v = y.mean(idx)
    p = plot.Barplot(v, model, match, data=ds, corr=None, colors=colors, h=4,
                     show=False)
    image_bar = p.image('cluster_%i_barplot.png' % cid)
    p.close()

    # Boxplot
    p = plot.Boxplot(v, model, match, data=ds, corr=None, colors=colors, h=4,
                     show=False)
    image_box = p.image('cluster_%i_boxplot.png' % cid)
    p.close()

    if model:
        # compose figure
        section.add_figure("Time course in cluster area, and average value in cluster by condition, with pairwise t-tests.", [image_tc, image_bar, image_box, legend])
        # pairwise test table
        res = test.pairwise(v, model, match, data=ds, corr=None)
        section.add_figure("Pairwise t-tests of average value in cluster by condition", res)
    else:
        section.add_figure("Time course in cluster area, and average value in cluster.", [image_tc, image_bar, image_box])
