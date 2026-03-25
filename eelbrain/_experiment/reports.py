# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Report derivatives and shared report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any
from collections.abc import Callable

import mne

from .. import fmtxt
from .. import plot
from .. import report as _report
from .. import table
from .._data_obj import align1
from .._utils.mne_utils import is_fake_mri
from .derivative_cache import Derivative, DerivativeContext, file_fingerprint
from .parc import IndividualSeededParc
from .pathing import mri_dir, mri_sdir
from .results import ResultOutputDerivative, ResultSupport
from .test_def import TwoStageTest


def report_methods_brief(path: str | Path):
    path = Path(path)
    items = [*path.parts[:-1], path.stem]
    return fmtxt.List('Methods brief', items[-3:])


@dataclass
class ReportSupport(ResultSupport):
    subjects_dataset: Callable[[dict[str, Any]], Any]
    format_text: Callable[[dict[str, Any], str], str]
    evoked_kind: Callable[[dict[str, Any]], str | None]
    show_state_for: Callable[[dict[str, Any], tuple[str, ...]], Any]
    report_title_for: Callable[[dict[str, Any]], str]
    annot_state_for: Callable[[dict[str, Any]], tuple[str, Any]]
    load_annot_for: Callable[[dict[str, Any], str | None], list[mne.Label]]
    plot_annot_for: Callable[[dict[str, Any], str | None, int | None], Any]
    surfer_plot_kwargs_for: Callable[[dict[str, Any]], dict[str, Any]]
    coreg_subject_states_for: Callable[[dict[str, Any]], list[dict[str, Any]]]
    plot_coregistration_for: Callable[[dict[str, Any]], Any]

    def report_subject_info(self, state: dict[str, Any], ds, model):
        """Table with subject information."""
        s_ds = self.subjects_dataset(state)
        if 'n' in ds:
            if model:
                n_ds = table.repmeas('n', model, 'subject', data=ds)
            else:
                n_ds = ds
            n_ds_aligned = align1(n_ds, s_ds['subject'], 'subject')
            s_ds.update(n_ds_aligned)
        return s_ds.as_table(midrule=True, count=True, caption="All subjects included in the analysis with trials per condition")

    def report_test_info(self, state: dict[str, Any], section, ds, test, res, data, include=None, model=True):
        """Add report metadata for one test result."""
        test_obj = self.tests[test] if isinstance(test, str) else test

        info = fmtxt.List("Analysis:")
        epoch = self.format_text(state, 'epoch = {epoch}')
        evoked_kind = self.evoked_kind(state)
        if evoked_kind:
            epoch += f' {evoked_kind}'
        if model is True:
            model = state.get('model')
        if model:
            epoch += f" ~ {model}"
        info.add_item(epoch)
        if data.source:
            info.add_item(self.format_text(state, "cov = {cov}"))
            info.add_item(self.format_text(state, "inv = {inv}"))
        info.add_item(f"test = {test_obj.kind}  ({test_obj.desc})")
        if include is not None:
            info.add_item(f"Separate plots of all clusters with a p-value < {include}")
        section.append(info)

        info = res.info_list()
        section.append(info)
        section.append(self.report_subject_info(state, ds, test_obj.model))
        section.append(self.show_state_for(state, ('hemi', 'subject', 'mrisubject')))
        return info

    def report_parc_image(self, state: dict[str, Any], section, caption, subjects=None):
        """Add a picture of the current parcellation."""
        _, parc = self.annot_state_for(state)
        if isinstance(parc, IndividualSeededParc):
            if subjects is None:
                raise RuntimeError("subjects needs to be specified for plotting individual parcellations")
            legend = None
            for subject in subjects:
                if all(label.name.startswith('unknown-') for label in self.load_annot_for(state, subject)):
                    section.add_image_figure("No labels", subject)
                    continue
                brain = self.plot_annot_for(state, subject, None)
                if legend is None:
                    legend_plot = brain.plot_legend(show=False)
                    legend = legend_plot.image('parc-legend')
                    legend_plot.close()
                section.add_image_figure(brain.image('parc'), subject)
                brain.close()
            return

        brain = self.plot_annot_for({**state, 'mrisubject': state['common_brain']}, None, 500)
        legend = brain.plot_legend(show=False)
        content = [brain.image('parc'), legend.image('parc-legend')]
        section.add_image_figure(content, caption)
        brain.close()
        legend.close()

    def report_title(self, state: dict[str, Any]) -> str:
        return self.report_title_for(state)

    def coreg_subject_states(self, ctx: DerivativeContext) -> list[dict[str, Any]]:
        return self.coreg_subject_states_for(ctx.state)

    def coreg_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        subjects = []
        for state in self.coreg_subject_states(ctx):
            mri = file_fingerprint(state['root'], mri_dir(state), 'mri-dir', metadata={'mrisubject': state['mrisubject']})
            subject_dependencies = {
                'state': {key: state[key] for key in ('subject', 'session', 'task', 'acquisition', 'run', 'split') if state[key] is not None},
                'mrisubject': state['mrisubject'],
                'raw': ctx.registry.resolve('raw', state={**ctx.state, **state, 'raw': 'raw'}, options={'add_bads': False, 'noise': False}).describe_dependency(),
                'trans': ctx.registry.resolve('trans-input', state={**ctx.state, **state}).describe_dependency(),
                'mri': mri,
            }
            subjects.append(subject_dependencies)
        return {'dependencies': ctx.registry.canonicalize({'subjects': subjects})}

    def build_coreg_report(self, ctx: DerivativeContext) -> Path:
        from matplotlib import pyplot
        from mayavi import mlab

        dst = Path(ctx.option('dst'))
        title = 'Coregistration'
        if ctx.get('group') != 'all':
            title += ' ' + ctx.get('group')
        if ctx.get('mri'):
            title += ' ' + ctx.get('mri')

        report = fmtxt.Report(title)
        for state in self.coreg_subject_states(ctx):
            subject = state['subject']
            mrisubject = state['mrisubject']
            fig = self.plot_coregistration_for(state)
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

    def append_source_report(self, report, ctx: DerivativeContext) -> None:
        ds, res = self.load_test(ctx, True, True)
        self.report_test_info(ctx.state, report.add_section("Test Info"), ds, ctx.option('test'), res, ctx.option('data'), ctx.option('include'))
        parc = ctx.option('parc')
        mask = ctx.option('mask')
        if parc:
            section = report.add_section(parc)
            self.report_parc_image(ctx.state, section, f"Labels in the {parc} parcellation.")
        elif mask:
            section = report.add_section(f"Whole Brain Masked by {mask}")
            self.report_parc_image(ctx.state, section, f"Mask: {mask.capitalize()}")

        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.source_time_results(res, ds, colors, ctx.option('include'), self.surfer_plot_kwargs_for(ctx.state), parc=parc))

    def append_two_stage_report(self, report, ctx: DerivativeContext) -> None:
        test_obj = self.tests[ctx.option('test')]
        return_data = bool(test_obj.model)
        result = self.load_test(ctx, return_data, True)
        if return_data:
            group_ds, rlm = result
        else:
            group_ds = None
            rlm = result

        info_section = report.add_section("Test Info")
        parc = ctx.option('parc')
        mask = ctx.option('mask')
        if parc:
            section = report.add_section(parc)
            self.report_parc_image(ctx.state, section, f"Labels in the {parc} parcellation.")
        elif mask:
            section = report.add_section(f"Whole Brain Masked by {mask}")
            self.report_parc_image(ctx.state, section, f"Mask: {mask.capitalize()}")

        report.add_section("Design Matrix").append(rlm.design())

        for term in rlm.column_names:
            res = rlm.tests[term]
            ds = rlm.coefficients_dataset(term, long=True)
            report.append(_report.source_time_results(res, ds, None, ctx.option('include'), self.surfer_plot_kwargs_for(ctx.state), term, y='coeff'))

        self.report_test_info(ctx.state, info_section, group_ds or ds, test_obj, res, ctx.option('data'))


def _save_report(report, dst: Path, packages: tuple[str, ...], samples: int | None = None) -> Path:
    report.sign(packages)
    meta = None if samples is None else {'samples': samples}
    report.save_html(dst, meta=meta)
    return dst


class SourceReportDerivative(ResultOutputDerivative[Path]):
    name = 'source-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'include': ctx.option('include')}

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        report = fmtxt.Report(self.support.report_title(ctx.state))
        report.add_paragraph(report_methods_brief(dst))
        if isinstance(self.support.tests[ctx.option('test')], TwoStageTest):
            self.support.append_two_stage_report(report, ctx)
        else:
            self.support.append_source_report(report, ctx)
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'), ctx.option('samples'))


class ROIReportDerivative(ResultOutputDerivative[Path]):
    name = 'roi-report'
    sampled_path = True

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        res_data, res = self.support.load_test(ctx, True, True, mask=None)
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
        report = fmtxt.Report(self.support.report_title(ctx.state))
        first_label = (labels_lh or labels_rh)[0]
        info_section = report.add_section("Test Info")
        self.support.report_test_info(ctx.state, info_section, res.n_trials_ds, self.support.tests[ctx.option('test')], res.res[first_label], ctx.option('data'))
        section = report.add_section(ctx.option('parc'))
        self.support.report_parc_image(ctx.state, section, f"ROIs in the {ctx.option('parc')} parcellation.", res.subjects)
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
    name = 'eeg-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'include': ctx.option('include')}

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        ds, res = self.support.load_test(ctx, True, True, parc=None, mask=None)
        report = fmtxt.Report(self.support.report_title(ctx.state))
        info_section = report.add_section("Test Info")
        self.support.report_test_info(ctx.state, info_section, ds, ctx.option('test'), res, ctx.option('data'), ctx.option('include'))
        sensor_map = plot.SensorMap(ds['eeg'], adjacency=True, show=False)
        image_conn = sensor_map.image("adjacency.png")
        info_section.add_figure("Sensor map with adjacency", image_conn)
        sensor_map.close()
        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.sensor_time_results(res, ds, colors, ctx.option('include')))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.option('samples'))


class EEGSensorsReportDerivative(ResultOutputDerivative[Path]):
    name = 'eeg-sensors-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'sensors': tuple(ctx.option('sensors'))}

    def build(self, ctx: DerivativeContext) -> Path:
        dst = self.path(ctx)
        test_obj = self.support.tests[ctx.option('test')]
        ds = self.support.load_evoked(ctx, ctx.get('group'), True, test_obj.vars)
        eeg = ds['eeg']
        sensors = ctx.option('sensors')
        missing = [sensor for sensor in sensors if sensor not in eeg.sensor.names]
        if missing:
            raise ValueError(f"The following sensors are not in the data: {missing}")
        report = fmtxt.Report(self.support.report_title(ctx.state))
        info_section = report.add_section("Test Info")
        sensor_map = plot.SensorMap(ds['eeg'], show=False)
        sensor_map.mark_sensors(sensors)
        info_section.add_figure("Sensor map", sensor_map)
        sensor_map.close()
        test_kwargs = self.support.test_kwargs(ctx, ('time', 'sensor'), None)
        results = [self.support.make_test(eeg.sub(sensor=sensor), ds, test_obj, test_kwargs) for sensor in sensors]
        colors = plot.colors_for_categorial(ds.eval(results[0]._plot_model()))
        for sensor, res in zip(sensors, results):
            report.append(_report.time_results(res, ds, colors, sensor, f"Signal at {sensor}."))
        self.support.report_test_info(ctx.state, info_section, ds, ctx.option('test'), results[0], ctx.option('data'))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.option('samples'))


class LMReportDerivative(ResultOutputDerivative[Path]):
    name = 'lm-report'
    single_subject = True
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'mask': ctx.option('mask')}

    def build(self, ctx: DerivativeContext) -> Path:
        report = fmtxt.Report(self.support.report_title(ctx.state))
        report.append(_report.source_time_lm(self.support.load_spm(ctx), ctx.option('pmin'), self.support.surfer_plot_kwargs_for(ctx.state)))
        return _save_report(report, self.path(ctx), ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))


class CoregReportDerivative(Derivative[Path]):
    name = 'coreg-report'
    path_template = 'report-file'
    key_fields = ()

    def __init__(self, support: ReportSupport):
        self.support = support

    def key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return ctx.registry.canonicalize({'group': ctx.get('group'), 'mri': ctx.get('mri')})

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return self.support.coreg_fingerprint(ctx)

    def path(
            self,
            ctx: DerivativeContext,
            mkdir: bool = False,
    ) -> Path:
        path = Path(ctx.option('dst'))
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def build(self, ctx: DerivativeContext) -> Path:
        return self.support.build_coreg_report(ctx)

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> Path:
        return path

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: Path,
    ) -> None:
        return
