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
from .derivative_cache import Dependency, Request, file_fingerprint
from .parc import IndividualSeededParc, SEEDED_PARC_RE
from .pathing import coreg_report_path, mri_dir, mri_sdir, trans_file_path
from .preprocessing import raw_node_name
from .results import (
    RESULT_OPTION_DEFAULTS,
    TEST_DATA_OPTION_NAMES,
    ResultOutputDerivative,
    _subject_request_state,
    _test_result_options,
    make_result_test,
    result_test_kwargs,
)
from .test_def import TestDims
from .two_stage import TwoStageTest


def _format_text(state: dict[str, Any], template: str) -> str:
    return template.format(**state)


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
    s_ds = Dataset(caption=f"Subjects in group {state['group']}")
    s_ds['subject'] = Factor([item['subject'] for item in node.collect_states(state, ('subject',))])
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
    evoked_kind = '_'.join(part for part in (state.get('rej'), state.get('equalize_evoked_count')) if part not in (None, '')) or None
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


def _coreg_subject_states(node: ResultOutputDerivative, ctx: Request) -> list[dict[str, Any]]:
    return node.collect_states(ctx.state, ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'mrisubject'), raw='raw')


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
    OPTION_DEFAULTS = {**RESULT_OPTION_DEFAULTS, 'include': None}

    def extra_key(self, ctx: Request) -> dict[str, Any]:
        return {'include': ctx.options['include']}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            return (
                Dependency('two-stage-level-2', options=ctx.options_for('two-stage-level-2', *RESULT_OPTION_DEFAULTS)),
                Dependency('two-stage-data', options=ctx.options_for('two-stage-data', *RESULT_OPTION_DEFAULTS)),
            )
        return (
            Dependency('test-result', options=_test_result_options(ctx)),
            Dependency('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES)),
        )

    def build(self, ctx: Request) -> Path:
        dst = self.path(ctx)
        report = fmtxt.Report(_report_title(dst))
        path_items = [*dst.parts[:-1], dst.stem]
        report.add_paragraph(fmtxt.List('Methods brief', path_items[-3:]))
        test_obj = self.tests[ctx.options['test']]
        if isinstance(test_obj, TwoStageTest):
            data_value = ctx.load('two-stage-data', options=ctx.options_for('two-stage-data', *RESULT_OPTION_DEFAULTS))
            rlm = ctx.load('two-stage-level-2', options=ctx.options_for('two-stage-level-2', *RESULT_OPTION_DEFAULTS))

            info_section = report.add_section("Test Info")
            parc = ctx.options['parc']
            mask = ctx.options['mask']
            if parc:
                section = report.add_section(parc)
                _report_parc_image(self, ctx.state, section, f"Labels in the {parc} parcellation.")
            elif mask:
                section = report.add_section(f"Whole Brain Masked by {mask}")
                _report_parc_image(self, ctx.state, section, f"Mask: {mask.capitalize()}")

            report.add_section("Design Matrix").append(rlm.design())
            for term in rlm.column_names:
                res = rlm.tests[term]
                ds = rlm.coefficients_dataset(term, long=True)
                report.append(_report.source_time_results(res, ds, None, ctx.options['include'], _surfer_plot_kwargs(self, ctx.state), term, y='coeff'))
            _report_test_info(self, ctx.state, info_section, data_value, test_obj, res, ctx.options['data'])
        else:
            data_value = ctx.load('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES))
            ds = data_value
            res = ctx.load('test-result', options=_test_result_options(ctx))
            _report_test_info(self, ctx.state, report.add_section("Test Info"), ds, ctx.options['test'], res, ctx.options['data'], ctx.options['include'])
            parc = ctx.options['parc']
            mask = ctx.options['mask']
            if parc:
                section = report.add_section(parc)
                _report_parc_image(self, ctx.state, section, f"Labels in the {parc} parcellation.")
            elif mask:
                section = report.add_section(f"Whole Brain Masked by {mask}")
                _report_parc_image(self, ctx.state, section, f"Mask: {mask.capitalize()}")
            colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
            report.append(_report.source_time_results(res, ds, colors, ctx.options['include'], _surfer_plot_kwargs(self, ctx.state), parc=parc))
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'), ctx.options['samples'])


class ROIReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for ROI-based test results.

    Uses the shared result-output options.
    """
    name = 'roi-report'
    sampled_path = True

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            return ()
        return (
            Dependency('test-result', options=_test_result_options(ctx, mask=None)),
            Dependency('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES)),
        )

    def build(self, ctx: Request) -> Path:
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            raise NotImplementedError("ROI report not implemented for two-stage tests")
        dst = self.path(ctx)
        roi_data = ctx.load('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES))
        res = ctx.load('test-result', options=_test_result_options(ctx, mask=None))
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
        _report_test_info(self, ctx.state, info_section, res.n_trials_ds, self.tests[ctx.options['test']], res.res[first_label], ctx.options['data'])
        section = report.add_section(ctx.options['parc'])
        _report_parc_image(self, ctx.state, section, f"ROIs in the {ctx.options['parc']} parcellation.", res.subjects)
        n_subjects = len(res.subjects)
        colors = plot.colors_for_categorial(roi_data.label_data[first_label].eval(res.res[first_label]._plot_model()))
        for label in chain(labels_lh, labels_rh):
            res_i = res.res[label]
            ds = roi_data.label_data[label]
            title = label[:-3].capitalize()
            caption = f"Mean in label {label}."
            n = len(ds['subject'].cells)
            if n < n_subjects:
                title += f' (n={n})'
                caption += f" Data from {n} of {n_subjects} subjects."
            section.append(_report.time_results(res_i, ds, colors, title, caption, merged_dist=res.merged_dist))
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'), ctx.options['samples'])


class EEGReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for sensor-space EEG test results.

    Uses the shared result-output options and adds ``include`` to control
    which clusters receive separate plots.
    """
    name = 'eeg-report'
    sampled_path = True
    OPTION_DEFAULTS = {**RESULT_OPTION_DEFAULTS, 'include': None}

    def extra_key(self, ctx: Request) -> dict[str, Any]:
        return {'include': ctx.options['include']}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            return ()
        return (
            Dependency('test-result', options=_test_result_options(ctx, parc=None, mask=None)),
            Dependency('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES)),
        )

    def build(self, ctx: Request) -> Path:
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            raise NotImplementedError("EEG report not implemented for two-stage tests")
        dst = self.path(ctx)
        ds = ctx.load('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES))
        res = ctx.load('test-result', options=_test_result_options(ctx, parc=None, mask=None))
        report = fmtxt.Report(_report_title(dst))
        info_section = report.add_section("Test Info")
        _report_test_info(self, ctx.state, info_section, ds, ctx.options['test'], res, ctx.options['data'], ctx.options['include'])
        sensor_map = plot.SensorMap(ds['eeg'], adjacency=True, show=False)
        image_conn = sensor_map.image("adjacency.png")
        info_section.add_figure("Sensor map with adjacency", image_conn)
        sensor_map.close()
        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.sensor_time_results(res, ds, colors, ctx.options['include']))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.options['samples'])


class EEGSensorsReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for a fixed list of EEG sensors.

    Uses the shared result-output options and adds ``sensors`` for the sensor
    names to plot.
    """
    name = 'eeg-sensors-report'
    sampled_path = True
    OPTION_DEFAULTS = {**RESULT_OPTION_DEFAULTS, 'sensors': ()}

    def extra_key(self, ctx: Request) -> dict[str, Any]:
        return {'sensors': tuple(ctx.options['sensors'])}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        self.tests[ctx.options['test']]
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            return ()
        return (Dependency('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES)),)

    def build(self, ctx: Request) -> Path:
        if isinstance(self.tests[ctx.options['test']], TwoStageTest):
            raise NotImplementedError("EEG sensor report not implemented for two-stage tests")
        dst = self.path(ctx)
        test_obj = self.tests[ctx.options['test']]
        ds = ctx.load('evoked-test-data', options=ctx.options_for('evoked-test-data', *TEST_DATA_OPTION_NAMES))
        eeg = ds['eeg']
        sensors = ctx.options['sensors']
        missing = [sensor for sensor in sensors if sensor not in eeg.sensor.names]
        if missing:
            raise ValueError(f"The following sensors are not in the data: {missing}")
        report = fmtxt.Report(_report_title(dst))
        info_section = report.add_section("Test Info")
        sensor_map = plot.SensorMap(ds['eeg'], show=False)
        sensor_map.mark_sensors(sensors)
        info_section.add_figure("Sensor map", sensor_map)
        sensor_map.close()
        test_kwargs = result_test_kwargs(self, ctx, ('time', 'sensor'), None)
        results = [make_result_test(self, eeg.sub(sensor=sensor), ds, test_obj, test_kwargs) for sensor in sensors]
        colors = plot.colors_for_categorial(ds.eval(results[0]._plot_model()))
        for sensor, res in zip(sensors, results):
            report.append(_report.time_results(res, ds, colors, sensor, f"Signal at {sensor}."))
        _report_test_info(self, ctx.state, info_section, ds, ctx.options['test'], results[0], ctx.options['data'])
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.options['samples'])


class LMReportDerivative(ResultOutputDerivative[Path]):
    """HTML report for the first-stage subject LM used by two-stage tests.

    Uses the shared result-output options and adds ``mask`` for the optional
    source-space mask to plot.
    """
    name = 'lm-report'
    single_subject = True
    sampled_path = True

    def extra_key(self, ctx: Request) -> dict[str, Any]:
        return {'mask': ctx.options['mask']}

    def _level_1_options(self, ctx: Request) -> dict[str, Any]:
        return ctx.options_for('two-stage-level-1', *RESULT_OPTION_DEFAULTS, data=TestDims.coerce('source', morph=False), smooth=None, parc=None, mask=True)

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        test_obj = self.tests[ctx.options['test']]
        if not isinstance(test_obj, TwoStageTest):
            return ()
        return (Dependency('two-stage-level-1', state=_subject_request_state(ctx, ctx.state['subject']), options=self._level_1_options(ctx)),)

    def build(self, ctx: Request) -> Path:
        test_obj = self.tests[ctx.options['test']]
        if not isinstance(test_obj, TwoStageTest):
            raise TypeError("LM report requires a TwoStageTest")
        dst = self.path(ctx)
        report = fmtxt.Report(_report_title(dst))
        lm = ctx.load('two-stage-level-1', state=_subject_request_state(ctx, ctx.state['subject']), options=self._level_1_options(ctx))
        report.append(_report.source_time_lm(lm, ctx.options['pmin'], _surfer_plot_kwargs(self, ctx.state)))
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
    OPTION_DEFAULTS = {}

    def key(self, ctx: Request) -> dict[str, Any]:
        return ctx.registry.canonicalize({'group': ctx.state['group'], 'mri': ctx.state['mri']})

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'subjects': ctx.registry.canonicalize([
                {
                    'state': {key: state[key] for key in ('subject', 'session', 'task', 'acquisition', 'run', 'split') if state[key] is not None},
                    'mrisubject': state['mrisubject'],
                    'mri': file_fingerprint(ctx.state['root'], mri_dir({**ctx.state, **state}), 'mri-dir', metadata={'mrisubject': state['mrisubject']}),
                }
                for state in _coreg_subject_states(self, ctx)
            ])
        }

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        deps = []
        for state in _coreg_subject_states(self, ctx):
            label = '_'.join(
                [state['subject'], *[f'{key}-{state[key]}' for key in ('session', 'task', 'acquisition', 'run', 'split') if state[key] is not None]]
            )
            deps.append(Dependency(
                raw_node_name('raw'),
                label=f'{label}:raw',
                state={**state, 'raw': 'raw'},
                options={'add_bads': False, 'noise': False},
            ))
            deps.append(Dependency('trans-input', label=f'{label}:trans', state=state))
        return tuple(deps)

    def path(
            self,
            ctx: Request,
    ) -> Path:
        dst = ctx.view_options['dst']
        return Path(dst) if dst else coreg_report_path(ctx.state)

    def build(self, ctx: Request) -> Path:
        from matplotlib import pyplot
        from mayavi import mlab

        dst = self.path(ctx)
        title = 'Coregistration'
        if ctx.state['group'] != 'all':
            title += ' ' + ctx.state['group']
        if ctx.state['mri']:
            title += ' ' + ctx.state['mri']

        report = fmtxt.Report(title)
        for state in _coreg_subject_states(self, ctx):
            subject = state['subject']
            mrisubject = state['mrisubject']
            raw = ctx.load(raw_node_name('raw'), state={**state, 'raw': 'raw'}, options={'add_bads': False, 'noise': False})
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
