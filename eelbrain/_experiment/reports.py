# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Report derivatives and shared report helpers."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

import mne

from .. import fmtxt
from .. import plot
from .. import report as _report
from .. import table
from .._data_obj import align1
from .._utils.mne_utils import is_fake_mri
from .derivative_cache import Artifact, Derivative, DerivativeContext, file_fingerprint
from .parc import IndividualSeededParc
from .preprocessing import raw_meeg_input_name
from .results import ResultOutputDerivative, ResultSupport
from .test_def import TwoStageTest


def report_methods_brief(path: str):
    path_ = Path(path)
    items = [*path_.parts[:-1], path_.stem]
    return fmtxt.List('Methods brief', items[-3:])


class ReportSupport(ResultSupport):
    def report_subject_info(self, ds, model):
        """Table with subject information."""
        s_ds = self.pipeline.show_subjects(asds=True)
        if 'n' in ds:
            if model:
                n_ds = table.repmeas('n', model, 'subject', data=ds)
            else:
                n_ds = ds
            n_ds_aligned = align1(n_ds, s_ds['subject'], 'subject')
            s_ds.update(n_ds_aligned)
        return s_ds.as_table(midrule=True, count=True, caption="All subjects included in the analysis with trials per condition")

    def report_test_info(self, section, ds, test, res, data, include=None, model=True):
        """Add report metadata for one test result."""
        p = self.pipeline
        test_obj = self.tests[test] if isinstance(test, str) else test

        info = fmtxt.List("Analysis:")
        epoch = p.format('epoch = {epoch}')
        evoked_kind = p.get('evoked_kind')
        if evoked_kind:
            epoch += f' {evoked_kind}'
        if model is True:
            model = p.get('model')
        if model:
            epoch += f" ~ {model}"
        info.add_item(epoch)
        if data.source:
            info.add_item(p.format("cov = {cov}"))
            info.add_item(p.format("inv = {inv}"))
        info.add_item(f"test = {test_obj.kind}  ({test_obj.desc})")
        if include is not None:
            info.add_item(f"Separate plots of all clusters with a p-value < {include}")
        section.append(info)

        info = res.info_list()
        section.append(info)
        section.append(self.report_subject_info(ds, test_obj.model))
        section.append(p.show_state(hide=('hemi', 'subject', 'mrisubject')))
        return info

    def report_parc_image(self, section, caption, subjects=None):
        """Add a picture of the current parcellation."""
        p = self.pipeline
        _, parc = p._get_parc()
        with p._temporary_state:
            if isinstance(parc, IndividualSeededParc):
                if subjects is None:
                    raise RuntimeError("subjects needs to be specified for plotting individual parcellations")
                legend = None
                for subject in p:
                    if all(label.name.startswith('unknown-') for label in p.load_annot()):
                        section.add_image_figure("No labels", subject)
                        continue
                    brain = p.plot_annot()
                    if legend is None:
                        legend_plot = brain.plot_legend(show=False)
                        legend = legend_plot.image('parc-legend')
                        legend_plot.close()
                    section.add_image_figure(brain.image('parc'), subject)
                    brain.close()
                return

            p.set(mrisubject=p.get('common_brain'))
            brain = p.plot_annot(axw=500)
        legend = brain.plot_legend(show=False)
        content = [brain.image('parc'), legend.image('parc-legend')]
        section.add_image_figure(content, caption)
        brain.close()
        legend.close()

    def report_title(self) -> str:
        return self.pipeline.format('{raw_basename}_{test_basename}_epoch-{epoch}_test-{test}_options-{test_options}')

    def coreg_subject_states(self, ctx: DerivativeContext) -> list[dict[str, Any]]:
        p = self.pipeline
        fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'mrisubject')
        with p._temporary_state:
            return [{field: p.get(field) for field in fields} for _ in p.iter(group=p.get('group'), raw='raw')]

    def coreg_fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        p = self.pipeline
        subjects = []
        for state in self.coreg_subject_states(ctx):
            with p._temporary_state:
                p.set(**state)
                mri = file_fingerprint(p.root, p.get('mri-dir'), 'mri-dir', metadata={'mrisubject': p.get('mrisubject')})
            subject_dependencies = {
                'state': {key: state[key] for key in ('subject', 'session', 'task', 'acquisition', 'run', 'split') if state[key] is not None},
                'mrisubject': state['mrisubject'],
                'raw': ctx.registry.resolve(raw_meeg_input_name('raw'), state={**state, 'raw': 'raw'}, options={'add_bads': False, 'noise': False}).describe_dependency(),
                'trans': ctx.registry.resolve('trans-input', state=state).describe_dependency(),
                'mri': mri,
            }
            subjects.append(subject_dependencies)
        return {'dependencies': ctx.registry.canonicalize({'subjects': subjects})}

    def build_coreg_report(self, ctx: DerivativeContext) -> str:
        from matplotlib import pyplot
        from mayavi import mlab

        p = self.pipeline
        dst = ctx.option('dst')
        title = 'Coregistration'
        if p.get('group') != 'all':
            title += ' ' + p.get('group')
        if p.get('mri'):
            title += ' ' + p.get('mri')

        report = fmtxt.Report(title)
        with p._temporary_state:
            for subject in p:
                mrisubject = p.get('mrisubject')
                fig = p.plot_coregistration()
                fig.scene.camera.parallel_projection = True
                fig.scene.camera.parallel_scale = .175
                mlab.draw(fig)

                mlab.view(90, 90, 1, figure=fig)
                im_front = fmtxt.Image.from_array(mlab.screenshot(figure=fig), 'front')

                mlab.view(0, 270, 1, roll=90, figure=fig)
                im_left = fmtxt.Image.from_array(mlab.screenshot(figure=fig), 'left')

                mlab.close(fig)

                if is_fake_mri(p.get('mri-dir')):
                    bem_fig = None
                else:
                    bem_fig = mne.viz.plot_bem(mrisubject, p.get('mri-sdir'), brain_surfaces='white', show=False)

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
        p = self.pipeline
        ds, res = self.load_test(ctx, True, True)
        self.report_test_info(report.add_section("Test Info"), ds, ctx.option('test'), res, ctx.option('data'), ctx.option('include'))
        parc = ctx.option('parc')
        mask = ctx.option('mask')
        if parc:
            section = report.add_section(parc)
            self.report_parc_image(section, f"Labels in the {parc} parcellation.")
        elif mask:
            section = report.add_section(f"Whole Brain Masked by {mask}")
            self.report_parc_image(section, f"Mask: {mask.capitalize()}")

        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.source_time_results(res, ds, colors, ctx.option('include'), p._surfer_plot_kwargs(), parc=parc))

    def append_two_stage_report(self, report, ctx: DerivativeContext) -> None:
        p = self.pipeline
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
            self.report_parc_image(section, f"Labels in the {parc} parcellation.")
        elif mask:
            section = report.add_section(f"Whole Brain Masked by {mask}")
            self.report_parc_image(section, f"Mask: {mask.capitalize()}")

        report.add_section("Design Matrix").append(rlm.design())

        for term in rlm.column_names:
            res = rlm.tests[term]
            ds = rlm.coefficients_dataset(term, long=True)
            report.append(_report.source_time_results(res, ds, None, ctx.option('include'), p._surfer_plot_kwargs(), term, y='coeff'))

        self.report_test_info(info_section, group_ds or ds, test_obj, res, ctx.option('data'))


def report_subject_info(p, ds, model):
    return ReportSupport(p, p._raw, p._tests, p._epochs, p._parcs).report_subject_info(ds, model)


def report_test_info(p, section, ds, test, res, data, include=None, model=True):
    return ReportSupport(p, p._raw, p._tests, p._epochs, p._parcs).report_test_info(section, ds, test, res, data, include, model)


def report_parc_image(p, section, caption, subjects=None):
    return ReportSupport(p, p._raw, p._tests, p._epochs, p._parcs).report_parc_image(section, caption, subjects)


def _save_report(report, dst: str, packages: tuple[str, ...], samples: int | None = None) -> str:
    report.sign(packages)
    meta = None if samples is None else {'samples': samples}
    report.save_html(dst, meta=meta)
    return dst


class SourceReportDerivative(ResultOutputDerivative[str]):
    name = 'source-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'include': ctx.option('include')}

    def build(self, ctx: DerivativeContext) -> str:
        dst = self.path(ctx)
        report = fmtxt.Report(self.support.report_title())
        report.add_paragraph(report_methods_brief(dst))
        if isinstance(self.support.tests[ctx.option('test')], TwoStageTest):
            self.support.append_two_stage_report(report, ctx)
        else:
            self.support.append_source_report(report, ctx)
        return _save_report(report, dst, ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'), ctx.option('samples'))


class ROIReportDerivative(ResultOutputDerivative[str]):
    name = 'roi-report'
    sampled_path = True

    def build(self, ctx: DerivativeContext) -> str:
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
        report = fmtxt.Report(self.support.report_title())
        first_label = (labels_lh or labels_rh)[0]
        info_section = report.add_section("Test Info")
        self.support.report_test_info(info_section, res.n_trials_ds, self.support.tests[ctx.option('test')], res.res[first_label], ctx.option('data'))
        section = report.add_section(ctx.option('parc'))
        self.support.report_parc_image(section, f"ROIs in the {ctx.option('parc')} parcellation.", res.subjects)
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


class EEGReportDerivative(ResultOutputDerivative[str]):
    name = 'eeg-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'include': ctx.option('include')}

    def build(self, ctx: DerivativeContext) -> str:
        dst = self.path(ctx)
        ds, res = self.support.load_test(ctx, True, True, parc=None, mask=None)
        report = fmtxt.Report(self.support.report_title())
        info_section = report.add_section("Test Info")
        self.support.report_test_info(info_section, ds, ctx.option('test'), res, ctx.option('data'), ctx.option('include'))
        sensor_map = plot.SensorMap(ds['eeg'], adjacency=True, show=False)
        image_conn = sensor_map.image("adjacency.png")
        info_section.add_figure("Sensor map with adjacency", image_conn)
        sensor_map.close()
        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.sensor_time_results(res, ds, colors, ctx.option('include')))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.option('samples'))


class EEGSensorsReportDerivative(ResultOutputDerivative[str]):
    name = 'eeg-sensors-report'
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'sensors': tuple(ctx.option('sensors'))}

    def build(self, ctx: DerivativeContext) -> str:
        dst = self.path(ctx)
        p = self.support.pipeline
        test_obj = self.support.tests[ctx.option('test')]
        ds = p.load_evoked(p.get('group'), ctx.option('baseline'), True, vardef=test_obj.vars)
        eeg = ds['eeg']
        sensors = ctx.option('sensors')
        missing = [sensor for sensor in sensors if sensor not in eeg.sensor.names]
        if missing:
            raise ValueError(f"The following sensors are not in the data: {missing}")
        report = fmtxt.Report(self.support.report_title())
        info_section = report.add_section("Test Info")
        sensor_map = plot.SensorMap(ds['eeg'], show=False)
        sensor_map.mark_sensors(sensors)
        info_section.add_figure("Sensor map", sensor_map)
        sensor_map.close()
        test_kwargs = self.support.test_kwargs(ctx, ('time', 'sensor'), None)
        results = [p._make_test(eeg.sub(sensor=sensor), ds, test_obj, test_kwargs) for sensor in sensors]
        colors = plot.colors_for_categorial(ds.eval(results[0]._plot_model()))
        for sensor, res in zip(sensors, results):
            report.append(_report.time_results(res, ds, colors, sensor, f"Signal at {sensor}."))
        self.support.report_test_info(info_section, ds, ctx.option('test'), results[0], ctx.option('data'))
        return _save_report(report, dst, ('eelbrain', 'mne', 'scipy', 'numpy'), ctx.option('samples'))


class LMReportDerivative(ResultOutputDerivative[str]):
    name = 'lm-report'
    single_subject = True
    sampled_path = True

    def extra_key(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {'mask': ctx.option('mask')}

    def build(self, ctx: DerivativeContext) -> str:
        p = self.support.pipeline
        report = fmtxt.Report(self.support.report_title())
        report.append(_report.source_time_lm(self.support.load_spm(ctx), ctx.option('pmin'), p._surfer_plot_kwargs()))
        return _save_report(report, self.path(ctx), ('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))


class CoregReportDerivative(Derivative[str]):
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
    ) -> str:
        path = ctx.option('dst')
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

    def build(self, ctx: DerivativeContext) -> str:
        return self.support.build_coreg_report(ctx)

    def load(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
    ) -> str:
        return artifact.path

    def save(
            self,
            ctx: DerivativeContext,
            artifact: Artifact,
            value: str,
    ) -> None:
        return
