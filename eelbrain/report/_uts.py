# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from .. import plot
from .._data_obj import Dataset
from .._stats.testnd import _MergedTemporalClusterDist, NDTest
from .._text import ms
from ..fmtxt import Section


def time_results(
        res: NDTest,
        data: Dataset,
        colors: dict,
        title: str = 'Results',
        caption: str = "Timecourse",
        pairwise_pmax: float = 0.1,
        merged_dist: _MergedTemporalClusterDist = None,
):
    """Add time course with clusters

    Parameters
    ----------
    res
        Result of the temporal cluster test.
    data
        Data.
    colors
        Cell colors.
    title
        Section title.
    caption
        Time-course figure caption.
    pairwise_pmax
        Barplots for clusters with ``p <= pairwise_pmax``.
    merged_dist
        Merged cluster distribution for correcting p values across ROIs
    """
    if merged_dist:
        clusters = merged_dist.correct_cluster_p(res)
    else:
        clusters = res.find_clusters()
    if clusters.n_cases:
        idx = clusters.eval("p.argmin()")
        max_sig = clusters['sig'][idx]
        if max_sig:
            title += max_sig
    section = Section(title)

    # compose captions
    if clusters.n_cases:
        c_caption = f"Clusters in time window {res._desc_timewindow()} based on {res._desc_samples}."
        if 'p_parc' in clusters:
            c_caption += " p: p-value in ROI; p_parc: p-value corrected across ROIs."
        tc_caption = caption
    else:
        c_caption = f"No clusters found {res._desc_timewindow()}."
        tc_caption = ' '.join((caption, c_caption))

    # plotting arguments
    model = res._plot_model()
    sub = res._plot_sub()

    # add UTSStat plot
    p = plot.UTSStat(res.y, model, None, res.match, sub, data, colors=colors, legend=False, clusters=clusters, show=False)
    ax = p.axes[0]
    if res.tstart is not None:
        ax.axvline(res.tstart, color='k')
    if res.tstop is not None:
        ax.axvline(res.tstop, color='k')
    image = p.image('%s_cluster.png')
    legend_p = p.plot_legend(show=False)
    legend = legend_p.image("Legend")
    section.add_figure(tc_caption, [image, legend])
    p.close()
    legend_p.close()

    # add cluster table
    if clusters.n_cases:
        t = clusters.as_table(midrule=True, caption=c_caption)
        section.append(t)

    # pairwise plots
    model_ = model
    colors_ = colors
    if pairwise_pmax is not None:
        plots = []
        clusters_ = clusters.sub(f"p <= {pairwise_pmax}")
        clusters_.sort("tstart")
        for cluster in clusters_.itercases():
            cid = cluster['id']
            c_tstart = cluster['tstart']
            c_tstop = cluster['tstop']
            tw_str = f"{ms(c_tstart)} - {ms(c_tstop)} ms"
            if 'effect' in cluster:
                title = f"{cluster['effect']} {cid}{cluster['sig']}: {tw_str}"
                model_ = cluster['effect'].replace(' x ', '%')
                colors_ = colors if model_ == model else None
            else:
                title = f"Cluster {cid}{cluster['sig']}: {tw_str}"
            y_ = data[res.y].summary(time=(c_tstart, c_tstop))
            p = plot.Barplot(y_, model_, res.match, sub, data=data, corr=None, show=False, colors=colors_, title=title)
            plots.append(p.image())
            p.close()

        section.add_image_figure(plots, "Value in the time-window of the clusters with uncorrected pairwise t-tests.")

    return section
