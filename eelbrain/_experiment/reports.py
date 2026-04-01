# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Report derivatives and shared report helpers.

Report nodes are the lower-level implementation of the public
``Pipeline.make_report*`` and ``Pipeline.show_*`` paths. They orchestrate by
loading dependency derivatives through ``ctx.load(...)`` and pure lower-layer
helpers. They own default public export paths directly from semantic
state/options and must not receive bound facade methods as execution backends.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

import mne

from .. import fmtxt
from .. import plot
from .. import report as _report
from .. import table
from .._data_obj import Dataset, Factor, align1
from .._utils.mne_utils import is_fake_mri
from .derivative_cache import Dependency, DerivativeContext, file_fingerprint
from .parc import IndividualSeededParc, SEEDED_PARC_RE
from .pathing import coreg_report_path, mri_dir, mri_sdir, trans_file_path
from .preprocessing import raw_node_name
from .results import ResultOutputDerivative, _group_request_state, _subject_request_state, _test_result_options
from .test_def import TwoStageTest


def report_methods_brief(path: str | Path):
    path = Path(path)
    items = [*path.parts[:-1], path.stem]
    return fmtxt.List('Methods brief', items[-3:])


def _subjects_dataset(node: ResultOutputDerivative, state: dict[str, Any]) -> Dataset:
    subjects = [item['subject'] for item in node.collect_states(state, ('subject',))]
    ds = Dataset(caption=f"Subjects in group {state['group']}")
    ds['subject'] = Factor(subjects)
    return ds


def _format_text(state: dict[str, Any], template: str) -> str:
    return template.format(**state)


def _evoked_kind(state: dict[str, Any]) -> str | None:
    parts = [state.get('rej'), state.get('equalize_evoked_count')]
    value = '_'.join(part for part in parts if part not in (None, ''))
    return value or None


def _show_state(state: dict[str, Any], hide: tuple[str, ...]):
    table_ = fmtxt.Table('lll')
    table_.cells('Key', '*', 'Value')
    table_.midrule()
    for key in sorted(state):
        if key in hide:
            continue
        value = state[key]
        if value not in (None, ''):
            table_.cells(key, '', repr(value))
    return table_


def _report_title(path: str | Path) -> str:
    return Path(path).stem


def _annot_state(node: ResultOutputDerivative, state: dict[str, Any]):
    parc = state['parc']
    if parc == '':
        return '', None
    if parc in node.parcs:
        return parc, node.parcs[parc]
    match = SEEDED_PARC_RE.match(parc)
    if match is None:
        raise ValueError(f"{parc=}: unknown parcellation")
    return parc, node.parcs[match.group(1)]


def _load_annot(node: ResultOutputDerivative, state: dict[str, Any], subject: str | None = None) -> list[mne.Label]:
    load_state = dict(state)
    if subject is not None:
        load_state['subject'] = subject
        load_state['mrisubject'] = node._field_options(load_state, 'mrisubject', subject=subject)[0]
    return mne.read_labels_from_annot(load_state['mrisubject'], load_state['parc'], 'both', subjects_dir=mri_sdir(load_state))


def _surfer_plot_kwargs(node: ResultOutputDerivative, state: dict[str, Any], surf=None, views=None, foreground=None, background=None, smoothing_steps=None, hemi=None) -> dict[str, Any]:
    out = dict(node.brain_plot_defaults)
    if views:
        out['views'] = views
    else:
        _, parc = _annot_state(node, state)
        if parc is not None and parc.views:
            out['views'] = parc.views
    if surf:
        out['surf'] = surf
    if foreground:
        out['foreground'] = foreground
    if background:
        out['background'] = background
    if smoothing_steps:
        out['smoothing_steps'] = smoothing_steps
    if hemi:
        out['hemi'] = hemi
    return out


def _plot_annot(node: ResultOutputDerivative, state: dict[str, Any], subject: str | None = None, axw: int | None = None):
    plot_state = dict(state)
    if subject is not None:
        plot_state['subject'] = subject
        plot_state['mrisubject'] = node._field_options(plot_state, 'mrisubject', subject=subject)[0]

    parc_name, parc = _annot_state(node, plot_state)
    plot_on_scaled_common_brain = isinstance(parc, IndividualSeededParc)
    if (not plot_on_scaled_common_brain) and is_fake_mri(mri_dir(plot_state)):
        subject_name = plot_state['common_brain']
    else:
        subject_name = plot_state['mrisubject']
    kwa = _surfer_plot_kwargs(node, plot_state)
    if axw is not None:
        kwa['axw'] = axw
    return plot.brain.annot(parc_name, subject_name, subjects_dir=mri_sdir(plot_state), **kwa)


def _report_subject_info(node: ResultOutputDerivative, state: dict[str, Any], ds, model):
    s_ds = _subjects_dataset(node, state)
    if 'n' in ds:
        if model:
            n_ds = table.repmeas('n', model, 'subject', data=ds)
        else:
            n_ds = ds
        n_ds_aligned = align1(n_ds, s_ds['subject'], 'subject')
        s_ds.update(n_ds_aligned)
    return s_ds.as_table(midrule=True, count=True, caption="All subjects included in the analysis with trials per condition")


def _report_test_info(node: ResultOutputDerivative, state: dict[str, Any], section, ds, test, res, data, include=None, model=True):
    test_obj = node.tests[test] if isinstance(test, str) else test
    info = fmtxt.List("Analysis:")
    epoch = _format_text(state, 'epoch = {epoch}')
    evoked_kind = _evoked_kind(state)
    if evoked_kind:
        epoch += f' {evoked_kind}'
    if model is True:
        model = state.get('model')
    if model:
        epoch += f" ~ {model}"
    info.add_item(epoch)
    if data.source:
        info.add_item(_format_text(state, "cov = {cov}"))
        info.add_item(_format_text(state, "inv = {inv}"))
    info.add_item(f"test = {test_obj.kind}  ({test_obj.desc})")
    if include is not None:
        info.add_item(f"Separate plots of all clusters with a p-value < {include}")
    section.append(info)
    info = res.info_list()
    section.append(info)
    section.append(_report_subject_info(node, state, ds, test_obj.model))
    section.append(_show_state(state, ('hemi', 'subject', 'mrisubject')))
    return info


def _report_parc_image(node: ResultOutputDerivative, state: dict[str, Any], section, caption, subjects=None):
    _, parc = _annot_state(node, state)
    if isinstance(parc, IndividualSeededParc):
        if subjects is None:
            raise RuntimeError("subjects needs to be specified for plotting individual parcellations")
        legend = None
        for subject in subjects:
            if all(label.name.startswith('unknown-') for label in _load_annot(node, state, subject)):
                section.add_image_figure("No labels", subject)
                continue
            brain = _plot_annot(node, state, subject, None)
            if legend is None:
                legend_plot = brain.plot_legend(show=False)
                legend = legend_plot.image('parc-legend')
                legend_plot.close()
            section.add_image_figure(brain.image('parc'), subject)
            brain.close()
        return

    brain = _plot_annot(node, {**state, 'mrisubject': state['common_brain']}, None, 500)
    legend = brain.plot_legend(show=False)
    section.add_image_figure([brain.image('parc'), legend.image('parc-legend')], caption)
    brain.close()
    legend.close()


def _coreg_subject_states(node: ResultOutputDerivative, ctx: DerivativeContext) -> list[dict[str, Any]]:
    return node.collect_states(ctx.state, ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'mrisubject'), raw='raw')


def _coreg_fingerprint(node: ResultOutputDerivative, ctx: DerivativeContext) -> dict[str, Any]:
    subjects = []
    for state in _coreg_subject_states(node, ctx):
        mri = file_fingerprint(state['root'], mri_dir(state), 'mri-dir', metadata={'mrisubject': state['mrisubject']})
        subjects.append({
            'state': {key: state[key] for key in ('subject', 'session', 'task', 'acquisition', 'run', 'split') if state[key] is not None},
            'mrisubject': state['mrisubject'],
            'raw': ctx.registry.resolve(raw_node_name('raw'), state={**ctx.state, **state, 'raw': 'raw'}, options={'add_bads': False, 'noise': False}).describe_dependency(),
            'trans': ctx.registry.resolve('trans-input', state={**ctx.state, **state}).describe_dependency(),
            'mri': mri,
        })
    return {'dependencies': ctx.registry.canonicalize({'subjects': subjects})}


def _build_coreg_report(node: ResultOutputDerivative, ctx: DerivativeContext) -> Path:
    from matplotlib import pyplot
    from mayavi import mlab

    dst = Path(ctx.option('dst'))
    title = 'Coregistration'
    if ctx.get('group') != 'all':
        title += ' ' + ctx.get('group')
    if ctx.get('mri'):
        title += ' ' + ctx.get('mri')

    report = fmtxt.Report(title)
    for state in _coreg_subject_states(node, ctx):
        subject = state['subject']
        mrisubject = state['mrisubject']
        raw = ctx.registry.load(raw_node_name('raw'), state={**ctx.state, **state, 'raw': 'raw'}, options={'add_bads': False, 'noise': False})
        fig = mne.viz.plot_alignment(raw.info, trans_file_path(state), mrisubject, mri_sdir(state), 'auto', meg=('helmet', 'sensors'), dig=True, interaction='terrain')
        fig.plotter.enable_parallel_projection()
        fig.scene.camera.parallel_projection = True
        fig.scene.camera.parallel_scale = .175
        mlab.draw(fig)

        mlab.view(90, 90, 1, figure=fig)
        im_front = fmtxt.Image.from_array(mlab.screenshot(figure=fig), 'front')
        mlab.view(0, 270, 1, roll=90, figure=fig)
        im_left = fmtxt.Image.from_array(mlab.screenshot(figure=fig), 'left')
        mlab.close(fig)

        if is_fake_mri(mri_dir(state)):
            bem_fig = None
        else:
            bem_fig = mne.viz.plot_bem(mrisubject, mri_sdir(state), brain_surfaces='white', show=False)

        if 'sub-' + subject == mrisubject:
            section_title = subject
            caption = f"Coregistration for subject {subject}."
        else:
            section_title = f"{subject} ({mrisubject})"
            caption = f"Coregistration for subject {subject} (MRI-subject {mrisubject})."
        section = report.add_section(section_title)
        if bem_fig is None:
            section.add_figure(caption, (im_front, im_left))
        else:
            section.add_figure(caption, (im_front, im_left, bem_fig))
            pyplot.close(bem_fig)

    report.sign()
    report.save_html(dst)
    return dst


def _append_source_report(node: ResultOutputDerivative, report, ctx: DerivativeContext) -> None:
    test_node = ctx.registry._get_node('test-result')
    ds, res = test_node.load_test(ctx, True, True)
    _report_test_info(node, ctx.state, report.add_section("Test Info"), ds, ctx.option('test'), res, ctx.option('data'), ctx.option('include'))
    parc = ctx.option('parc')
    mask = ctx.option('mask')
    if parc:
        section = report.add_section(parc)
        _report_parc_image(node, ctx.state, section, f"Labels in the {parc} parcellation.")
    elif mask:
        section = report.add_section(f"Whole Brain Masked by {mask}")
        _report_parc_image(node, ctx.state, section, f"Mask: {mask.capitalize()}")
    colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
    report.append(_report.source_time_results(res, ds, colors, ctx.option('include'), _surfer_plot_kwargs(node, ctx.state), parc=parc))


def _append_two_stage_report(node: ResultOutputDerivative, report, ctx: DerivativeContext) -> None:
    test_obj = node.tests[ctx.option('test')]
    test_node = ctx.registry._get_node('test-result')
    result = test_node.load_test(ctx, bool(test_obj.model), True)
    if test_obj.model:
        group_ds, rlm = result
    else:
        group_ds, rlm = None, result

    info_section = report.add_section("Test Info")
    parc = ctx.option('parc')
    mask = ctx.option('mask')
    if parc:
        section = report.add_section(parc)
        _report_parc_image(node, ctx.state, section, f"Labels in the {parc} parcellation.")
    elif mask:
        section = report.add_section(f"Whole Brain Masked by {mask}")
        _report_parc_image(node, ctx.state, section, f"Mask: {mask.capitalize()}")

    report.add_section("Design Matrix").append(rlm.design())
    for term in rlm.column_names:
        res = rlm.tests[term]
        ds = rlm.coefficients_dataset(term, long=True)
        report.append(_report.source_time_results(res, ds, None, ctx.option('include'), _surfer_plot_kwargs(node, ctx.state), term, y='coeff'))
    _report_test_info(node, ctx.state, info_section, group_ds or ds, test_obj, res, ctx.option('data'))


def _save_report(report, dst: Path, packages: tuple[str, ...], samples: int | None = None) -> Path:
    report.sign(packages)
    meta = None if samples is None else {'samples': samples}
    report.save_html(dst, meta=meta)
    return dst


class SourceReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for source-space test results.

    Uses the shared result-output options and adds ``include`` to control
    which clusters/terms receive separate plots.
    """
    name = 'source-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'include': ctx.option('include')}

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('test-result', options=_test_result_options(ctx)),)

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        report = fmtxt.Report(_report_title(dst))
        report.add_paragraph(report_methods_brief(dst))
        if isinstance(self.tests[ctx.option('test')], TwoStageTest):
            _append_two_stage_report(self, report, ctx)
        else:
            _append_source_report(self, report, ctx)
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'), ctx.option('samples'))


class ROIReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for ROI-based test results.

    Uses the shared result-output options.
    """
    name = 'roi-report'
    sampled_path = True

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('test-result', options=_test_result_options(ctx, mask=None)),)

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        test_node = ctx.registry._get_node('test-result')
        res_data, res = test_node.load_test(ctx, True, True, mask=None)
        labels_lh = []
        labels_rh = []
        for label in res.res.keys():
            if label.endswith('-lh'):
                labels_lh.append(label)
            elif label.endswith('-rh'):
                labels_rh.append(label)
            else:
                raise NotImplementedError(f"Label named {label!r}")
        labels_lh.sort()
        labels_rh.sort()
        report = fmtxt.Report(_report_title(dst))
        first_label = (labels_lh or labels_rh)[0]
        info_section = report.add_section("Test Info")
        _report_test_info(self, ctx.state, info_section, res.n_trials_ds, self.tests[ctx.option('test')], res.res[first_label], ctx.option('data'))
        section = report.add_section(ctx.option('parc'))
        _report_parc_image(self, ctx.state, section, f"ROIs in the {ctx.option('parc')} parcellation.", res.subjects)
        n_subjects = len(res.subjects)
        colors = plot.colors_for_categorial(res_data[first_label].eval(res.res[first_label]._plot_model()))
        for label in chain(labels_lh, labels_rh):
            res_i = res.res[label]
            ds = res_data[label]
            title = label[:-3].capitalize()
            caption = f"Mean in label {label}."
            n = len(ds['subject'].cells)
            if n < n_subjects:
                title += f' (n={n})'
                caption += f" Data from {n} of {n_subjects} subjects."
            section.append(_report.time_results(res_i, ds, colors, title, caption, merged_dist=res.merged_dist))
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'), ctx.option('samples'))


class EEGReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for sensor-space EEG test results.

    Uses the shared result-output options and adds ``include`` to control
    which clusters receive separate plots.
    """
    name = 'eeg-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'include': ctx.option('include')}

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('test-result', options=_test_result_options(ctx, parc=None, mask=None)),)

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        test_node = ctx.registry._get_node('test-result')
        ds, res = test_node.load_test(ctx, True, True, parc=None, mask=None)
        report = fmtxt.Report(_report_title(dst))
        info_section = report.add_section("Test Info")
        _report_test_info(self, ctx.state, info_section, ds, ctx.option('test'), res, ctx.option('data'), ctx.option('include'))
        sensor_map = plot.SensorMap(ds['eeg'], adjacency=True, show=False)
        image_conn = sensor_map.image("adjacency.png")
        info_section.add_figure("Sensor map with adjacency", image_conn)
        sensor_map.close()
        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.sensor_time_results(res, ds, colors, ctx.option('include')))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.option('samples'))


class EEGSensorsReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for a fixed list of EEG sensors.

    Uses the shared result-output options and adds ``sensors`` for the sensor
    names to plot.
    """
    name = 'eeg-sensors-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'sensors': tuple(ctx.option('sensors'))}

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        test_obj = self.tests[ctx.option('test')]
        return (Dependency('evoked-group-dataset', state=_group_request_state(ctx), options=self._evoked_options(ctx, ndvar=True, vardef=test_obj.vars, data=ctx.option('data'))),)

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        test_obj = self.tests[ctx.option('test')]
        test_node = ctx.registry._get_node('test-result')
        ds = ctx.load('evoked-group-dataset', state=_group_request_state(ctx), options=self._evoked_options(ctx, ndvar=True, vardef=test_obj.vars, data=ctx.option('data')))
        eeg = ds['eeg']
        sensors = ctx.option('sensors')
        missing = [sensor for sensor in sensors if sensor not in eeg.sensor.names]
        if missing:
            raise ValueError(f"The following sensors are not in the data: {missing}")
        report = fmtxt.Report(_report_title(dst))
        info_section = report.add_section("Test Info")
        sensor_map = plot.SensorMap(ds['eeg'], show=False)
        sensor_map.mark_sensors(sensors)
        info_section.add_figure("Sensor map", sensor_map)
        sensor_map.close()
        test_kwargs = test_node.test_kwargs(ctx, ('time', 'sensor'), None)
        results = [test_node.make_test(eeg.sub(sensor=sensor), ds, test_obj, test_kwargs) for sensor in sensors]
        colors = plot.colors_for_categorial(ds.eval(results[0]._plot_model()))
        for sensor, res in zip(sensors, results):
            report.append(_report.time_results(res, ds, colors, sensor, f"Signal at {sensor}."))
        _report_test_info(self, ctx.state, info_section, ds, ctx.option('test'), results[0], ctx.option('data'))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.option('samples'))


class LMReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for the first-stage subject LM used by two-stage tests.

    Uses the shared result-output options and adds ``mask`` for the optional
    source-space mask to plot.
    """
    name = 'lm-report'
    single_subject = True
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'mask': ctx.option('mask')}

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        test_obj = self.tests[ctx.option('test')]
        return (Dependency('epochs-stc', state=_subject_request_state(ctx, ctx.get('subject')), options=self._epochs_stc_options(ctx, ctx.option('baseline'), ctx.option('src_baseline'), None, False, None, True, False, test_obj.vars)),)

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        report = fmtxt.Report(_report_title(dst))
        report.append(_report.source_time_lm(ctx.registry._get_node('test-result').load_spm(ctx), ctx.option('pmin'), _surfer_plot_kwargs(self, ctx.state)))
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))


class CoregReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for MEG/MRI coregistration.

    Options
    -------
    dst
        Optional explicit output path.
    """
    name = 'coreg-report'
    key_fields = ()

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.canonicalize({'group': ctx.get('group'), 'mri': ctx.get('mri')})

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return _coreg_fingerprint(self, ctx)

    def path(
            self,
            ctx: DerivativeContext,
    ) -> Path:
        dst = ctx.option('dst')
        return Path(dst) if dst else coreg_report_path(ctx.state)

    def build(self, ctx: DerivativeContext) -> Path:
        return _build_coreg_report(self, ctx)
