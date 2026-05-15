# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np

from .. import fmtxt, plot
from .._data_obj import cellname
from .._stats.testnd import TTestIndependent, TTestRelated
from .._text import ms
from ..fmtxt import Section
from ._source import cluster_timecourse


def sensor_results(res, ds, color):
    report = Section("Results")
    if res._kind == 'cluster':
        p = plot.Topomap(res, show=False)
        report.add_figure("Significant clusters.", p)
        p.close()

        report.add_figure("All clusters.", res.clusters)
    else:
        raise NotImplementedError(f"Result kind {res._kind!r}")
    return report


def sensor_time_frequency_results(
        res: TTestIndependent | TTestRelated,
        p: float = 0.05,
        xlim: tuple[float, float] = None,
        vmax: float = None,
) -> fmtxt.FMText:
    if res.p.min() > p:
        return fmtxt.asfmtext(res)
    clusters = res.find_clusters(p, True)
    table_ = fmtxt.Table('lll')
    table_.cells('ID', 'Sensors', 'Difference in cluster')
    for cid, cluster_map, frequency_min, frequency_max, t_start, t_stop in clusters.zip('id', 'cluster', 'frequency_min', 'frequency_max', 'tstart', 'tstop'):
        table_.cell(cid)
        chs = cluster_map.sensor.names[cluster_map.any(('frequency', 'time'))]
        frequency_window = (frequency_min, frequency_max + 0.001)
        topo = res.difference.mean(frequency=frequency_window, time=(t_start, t_stop))
        frequency_desc = f'{frequency_min:.1f} - {frequency_max:.1f} Hz'
        plot_ = plot.Topomap(topo, mark=chs, mcolor='yellow', clip='circle', title=frequency_desc, vmax=vmax, h=2)
        table_.cell(plot_.image(close=True))
        diff = res.difference.mean(sensor=chs).mask(res.p.min(sensor=chs) > p, missing=True)
        ys = [res.c1_mean.mean(sensor=chs), res.c0_mean.mean(sensor=chs), diff]
        plot_ = plot.Array(ys, columns=3, xlim=xlim, vmax=vmax, h=2)
        table_.cell(plot_.image(close=True))
    return table_


def sensor_time_results(res, ds, colors, include=1):
    y = ds.eval(res.y)
    if res._kind in ('raw', 'tfce'):
        report = Section("Results")
        section = report.add_section("P<=.05")
        sensor_bin_table(section, res, 0.05)
        clusters = res.find_clusters(0.05, maps=True)
        clusters.sort('tstart')
        for cluster in clusters.itercases():
            sensor_time_cluster(section, cluster, y, res._plot_model(), ds,
                                colors, res.match)

        # trend section
        section = report.add_section("Trend: p<=.1")
        sensor_bin_table(section, res, 0.1)

        # not quite there section
        section = report.add_section("Anything: P<=.2")
        sensor_bin_table(section, res, 0.2)
    elif res._kind == 'cluster':
        report = Section("Clusters")
        sensor_bin_table(report, res)
        clusters = res.find_clusters(include, maps=True)
        clusters.sort('tstart')
        for cluster in clusters.itercases():
            sensor_time_cluster(report, cluster, y, res._plot_model(), ds,
                                colors, res.match)
    else:
        raise NotImplementedError(f"Result kind {res._kind!r}")
    return report


def sensor_bin_table(section, res, pmin=1):
    if pmin == 1:
        caption = "All clusters"
    else:
        caption = "p <= %.s" % pmin

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
        p = plot.TopomapBins(ndvar, show=False)
        section.add_image_figure(p, caption_)


def sensor_time_cluster(section, cluster, y, model, ds, colors, match='subject'):
    # cluster properties
    tstart_ms = ms(cluster['tstart'])
    tstop_ms = ms(cluster['tstop'])

    # section/title
    p_str = fmtxt.peq(cluster['p'], stars=True)
    title = f"{tstart_ms}-{tstop_ms} {p_str}"
    if 'effect' in cluster:
        title += f" {cluster['effect']}"
    section = section.add_section(title)

    # description
    paragraph = section.add_paragraph("Id %i" % cluster['id'])
    if 'v' in cluster:
        paragraph.append(f", v={cluster['v']}")

    # add cluster image to report
    topo = y.summary(time=(cluster['tstart'], cluster['tstop']))
    cluster_topo = cluster['cluster'].any('time')
    cluster_topo.info['contours'] = {0.5: (1, 1, 0)}
    if model:
        x = ds.eval(model)
        topos = [[topo[x == cell].summary('case', name=cellname(cell)),
                  cluster_topo] for cell in x.cells]
    else:
        topos = [[topo, cluster_topo]]
    p = plot.Topomap(topos, axh=3, rows=1, show=False)
    p.mark_sensors(np.flatnonzero(cluster_topo.x), c='y', marker='o')

    caption_ = ["Cluster"]
    if 'effect' in cluster:
        caption_.extend(('effect of', cluster['effect']))
    caption_.append("%i - %i ms." % (tstart_ms, tstop_ms))
    caption = ' '.join(caption_)
    section.add_image_figure(p, caption)
    p.close()

    cluster_timecourse(section, cluster, y, 'sensor', model, ds, colors, match)
