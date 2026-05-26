from __future__ import annotations

from argparse import ArgumentParser
import contextlib
import logging
from operator import itemgetter
import os
from os.path import relpath, dirname
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import webbrowser

from .. import fmtxt, save
from .._config import configure
from .._experiment.test_def import TestDims
from .._types import PathArg
from .._utils import ScreenHandler
from .._utils.com import Notifier

from ._model import Comparison, StructuredModel


def make_jobs(job_file, open_in_browser=False):
    for job in read_job_file(job_file):
        path = job.execute()
        if path and open_in_browser:
            webbrowser.open(f"file://{path}")


class Job:
    "Single function to be executed"
    def __init__(self, path, desc=None):
        self.path = path
        self.desc = desc or path[:path.rfind('.')]

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.desc}>"

    def generate_job(self) -> Optional[Callable]:
        raise NotImplementedError

    def _execute(self):
        job = self.generate_job()
        if job:
            logger = logging.getLogger(__name__)
            logger.info("Starting %s", self.desc)
            save.pickle(job(), self.path)
            logger.info("Done     %s", self.desc)


class FuncJob(Job):
    "Job based on a pre-defined function"
    def __init__(
            self,
            path: str,
            job_loader: Callable,
            desc: str = None,
    ):
        Job.__init__(self, path, desc)
        self.job_loader = job_loader

    def generate_job(self) -> Optional[Callable]:
        return self.job_loader()


class TRFJob(Job):
    "Jop for a single TRF"
    def __init__(self, experiment, path, state, args):
        desc = f"{experiment.__class__.__name__} {relpath(path, experiment.get('trf-sdir'))}"
        Job.__init__(self, path, desc)
        self.experiment = experiment
        self.state = state
        self.args = args

    def generate_job(self):
        self.experiment._restore_state(self.state)
        return self.experiment._trf_job(*self.args)


class UserJob:
    """Baseclass for a user-request, able to generate Job instances"""
    # to be initialized
    trf_jobs = None
    missing_trfs = None
    test_path = None

    def __init__(
            self,
            name: str,
            priority: bool,
    ):
        self.name = str(name)
        self.priority = priority

    def __repr__(self):
        args = [self.name, *self._repr_items()]
        return f"<{self.__class__.__name__}: {', '.join(args)}>"

    def _repr_items(self) -> List[str]:
        return []

    def is_same(self, other: UserJob):
        return NotImplemented

    def init_sub_jobs(self):
        pass

    def has_followup_jobs(self):
        return False

    def cancel(self):
        self.trf_jobs = None
        self.missing_trfs.clear()


class FuncIterJob(UserJob):

    def __init__(
            self,
            name: str,
            jobs: Iterable,
            priority: bool = False,
    ):
        UserJob.__init__(self, name, priority)
        self._job_iter = jobs

    def is_same(self, other: UserJob):
        return type(self) is type(other) and self.name == other.name

    def init_sub_jobs(self):
        self.missing_trfs = set()
        self.trf_jobs = []
        for i, (path, job_loader) in enumerate(self._job_iter):
            if not callable(job_loader):
                raise TypeError(f"job_loader must be callable, got {job_loader!r}")
            job = FuncJob(path, job_loader, f'{self.name}-{i}')
            self.trf_jobs.append(job)
            self.missing_trfs.add(job.path)


class ExperimentJob(UserJob):
    """Baseclass for multi-TRF jobs (TRFs or model-comparison)"""
    def __init__(
            self,
            experiment,
            model: Union[Comparison, StructuredModel],
            priority: bool,
            options: Dict[str, Any],
            public_name: str = None,
            report: bool = False,
    ):
        # validate input
        from ._experiment import TRFExperiment
        if not isinstance(experiment, TRFExperiment):
            raise TypeError(f"{experiment=}")
        name = public_name or experiment._x_desc(model, True)
        UserJob.__init__(self, name, priority)

        if 'data' in options:
            options['data'] = TestDims.coerce(options['data'])

        self.model = model
        self.report = report
        self.experiment = experiment
        self.options = options
        self.public_model_name = public_name

    def _repr_items(self) -> List[str]:
        return [f'{k}={v!r}' for k, v in self.options.items()]

    def is_same(self, other: ExperimentJob):
        return (
            type(self) is type(other) and
            self.model == other.model and
            self.report == other.report and
            self.experiment.__class__.__name__ == other.experiment.__class__.__name__ and
            self.options == other.options)

    def init_test_path(self):
        raise NotImplementedError

    def init_sub_jobs(self):
        self.trf_jobs = self.generate_trf_jobs()
        self.missing_trfs = {job.path for job in self.trf_jobs}
        self.all_jobs_generated = True

    def _init_trf_jobs(self, existing: bool = False):
        raise NotImplementedError

    def execute(self):
        self.experiment.reset()
        return self._execute()

    def _execute(self):
        raise NotImplementedError

    def get_followup_jobs(self, log):
        raise NotImplementedError

    def generate_trf_jobs(self, existing: bool = False):
        """List of TRFJob

        Parameters
        ----------
        existing : bool
            Include jobs for TRFs that are already cached in pickle files
            (default ``False``).
        """
        self.experiment.reset()
        return [TRFJob(self.experiment, *args) for args in self._init_trf_jobs(existing)]


class TRFsJob(ExperimentJob):
    """Job for group of TRFs

    Parameters
    ----------
    model : str
        Model or regressors.
    report : bool
        Schedule a model-test report.
    experiment : Experiment
        The experiment instance providing access to the data.
    reduce_model : bool
        Reduce the model until it only contains predictors significant at the
        .05 level.
    parent : UserJob
        Parent job (for reduced model jobs).
    priority : bool
        Insert job at the beginning of the queue (default ``False``).
    reduction_tag : str
        Tag to use for reduced models (to distinguish different reduction
        algorithms, default ``'red'``).
    ...
        Model-test parameters.
    """
    def __init__(
            self,
            comparison: Union[Comparison, StructuredModel],
            experiment=None,
            priority: bool = False,
            **options,
    ):
        ExperimentJob.__init__(self, experiment, comparison, priority, options)

    def init_test_path(self):
        pass

    def _init_trf_jobs(self, existing: bool = False):
        if isinstance(self.model, StructuredModel):
            return self.experiment._locate_missing_trfs(self.model.model, existing=existing, **self.options)
        else:
            return self.experiment._locate_model_test_trfs(self.model, existing=existing, **self.options)

    def _execute(self):
        self.init_sub_jobs()
        for job in self.trf_jobs:
            job._execute()
        for job in self.get_followup_jobs():
            job.execute()

    def has_followup_jobs(self):
        return False

    def get_followup_jobs(self, log=None):
        return []


# Not currently used; keep it in case automatically reducing models becomes of interest again
class ModelJob(ExperimentJob):
    """Job for model-comparison

    Parameters
    ----------
    model
        StructuredModel or comparison (see :class:`TRFExperiment` module docstring).
    report
        Schedule a model-test report.
    experiment
        The experiment instance providing access to the data.
    reduce_model
        Reduce the model until it only contains predictors significant at the
        .05 level.
    parent
        Parent job (for reduced model jobs).
    priority
        Insert job at the beginning of the queue (default ``False``).
    reduction_tag
        Tag to use for reduced models (to distinguish different reduction
        algorithms, default ``'red'``).
    ...
        Model-test parameters.
    """
    def __init__(
            self,
            model: str,
            experiment: 'TRFExperiment' = None,
            report: bool = False,
            reduce_model: bool = False,
            parent: UserJob = None,
            priority: bool = False,
            reduction_tag: str = 'red',
            metric: str = 'z',
            smooth: float = False,
            cv: bool = True,
            **options,
    ):
        model = experiment._coerce_comparison(model, cv)
        if isinstance(reduce_model, float):
            assert 0. < reduce_model < 1.
        elif not isinstance(reduce_model, bool):
            raise TypeError(f"{reduce_model=}")
        if reduce_model and not isinstance(model, StructuredModel):
            raise ValueError("reduce_model requires incremental model-comparison as base")

        # spacial description for reduced models
        if parent is not None:
            i = 1
            current = parent
            while current.parent is not None:
                i += 1
                current = current.parent
            public_name = '%s-%s%i' % (experiment._x_desc(current.model, True), reduction_tag, i)
        else:
            public_name = None

        options['cv'] = cv
        ExperimentJob.__init__(self, experiment, model, priority, options, public_name, report)
        self._test_options = {'metric': metric, 'smooth': smooth}
        self._reduction_tag = reduction_tag
        self.reduce_model = reduce_model
        self.parent = parent
        self._reduction_results = []

    def is_same(self, other: 'ModelJob'):
        return (
            ExperimentJob.is_same(self, other) and
            self._test_options == other._test_options)

    def init_test_path(self):
        self.experiment.reset()
        self.test_path = self.experiment.make_model_test_report(self.model, public_name=self.public_model_name, path_only=True, **self.options, **self._test_options)

    def _init_trf_jobs(self, existing: bool = False):
        return self.experiment._locate_model_test_trfs(self.model, existing=existing, **self.options)

    def _execute(self):
        self.init_sub_jobs()
        for job in self.trf_jobs:
            job._execute()
        job = self.reduced_model_job()
        if job:
            job.execute()
            self._reduction_results.extend(job._reduction_results)
        elif self.report:
            return self.experiment.make_model_test_report(self.model, public_name=self.public_model_name, **self.options, **self._test_options)

    def has_followup_jobs(self):
        return self.reduce_model

    def get_followup_jobs(self, log):
        try:
            job = self.reduced_model_job()
        except:
            log.exception("Generating reduced model job for %s", self.test_path or self.public_model_name)
            job = None
        else:
            if job is None:
                log.info("No further reduction of %s", self.test_path or self.public_model_name)
            else:
                log.info("Reduced model of %s", self.test_path or self.public_model_name)
        if job is None:
            return []
        else:
            return [job]

    def make_test_report(self):
        assert self.test_path is not None, "Job has no test-report"
        self.experiment.reset()
        self.experiment.make_model_test_report(self.model, public_name=self.public_model_name, **self.options, **self._test_options)

    def reduced_model_job(self):
        if not self.reduce_model:
            return
        p_threshold = .05 if self.reduce_model is True else self.reduce_model
        self.experiment.reset()
        ress = self.experiment.load_model_test(self.model, **self.options, **self._test_options)
        self._reduction_results.append(ress)
        # no terms left
        if len(ress) == 0:
            return
        # find term to remove
        pmins = [(term, res.p.min()) for term, res in ress.items()]
        least_term, pmax = max(pmins, key=itemgetter(1))
        if pmax <= p_threshold:
            return  # all regressors are significant
        # check for ambiguity
        pmins = [term for term, p in pmins if p == pmax]
        if len(pmins) > 1:
            tmaxs = [(term, ress[term].t.max()) for term in pmins]
            least_term, tmin = min(tmaxs, key=itemgetter(1))
        # remove term
        model = self.model.without(least_term)
        return ModelJob(model, self.experiment, self.report, self.reduce_model, self, self.priority, self._reduction_tag, **self._test_options, **self.options)

    def reduction_table(self, labels=None, vertical=False, title=None, caption=None):
        """Table with steps of model reduction

        Parameters
        ----------
        labels : dict {str: str}
            Substitute new labels for predictors.
        vertical : bool
            Orient table vertically.
        title : text
            Title for the table.
        caption : text
            Caption for the table.
        """
        if not self._reduction_results:
            self.execute()
        if labels is None:
            labels = {}
        n_steps = len(self._reduction_results)
        # find terms
        terms = []
        for ress in self._reduction_results:
            terms.extend(term for term in ress.keys() if term not in terms)
        n_terms = len(terms)
        # cell content
        cells = {}
        for x in terms:
            for i, ress in enumerate(self._reduction_results):
                if x in ress:
                    res = ress[x]
                    pmin = res.p.min()
                    t_cell = fmtxt.stat(res.t.max(), stars=pmin)
                    p_cell = fmtxt.p(pmin)
                else:
                    t_cell = p_cell = ''
                cells[i, x] = t_cell, p_cell

        if vertical:
            t = fmtxt.Table('ll' + 'l' * n_terms, title=title, caption=caption)
            t.cells('Step', '')
            for x in terms:
                t.cell(labels.get(x, x))
            t.midrule()
            for i in range(n_steps):
                t_row = t.add_row()
                p_row = t.add_row()
                t_row.cells(i + 1, fmtxt.symbol('t', 'max'))
                p_row.cells('', fmtxt.symbol('p'))
                for x in terms:
                    t_cell, p_cell = cells[i, x]
                    t_row.cell(t_cell)
                    p_row.cell(p_cell)
        else:
            t = fmtxt.Table('l' + 'rr' * n_steps, title=title, caption=caption)
            t.cell()
            for _ in range(n_steps):
                t.cell(fmtxt.symbol('t', 'max'))
                t.cell(fmtxt.symbol('p'))
            t.midrule()
            for x in terms:
                t.cell(labels.get(x, x))
                for i in range(n_steps):
                    t.cells(*cells[i, x])
        return t


def split_lines(string):
    return (line for line in (line.strip() for line in string) if line and not line.startswith('#'))


def read_job_file(filename: PathArg) -> List[TRFsJob]:
    """Read file with job definitions

    Returns
    -------
    jobs : list of UserJob
        All jobs found in the file.
    """
    with open(filename) as fid:
        text = fid.read()
    code = compile(text, filename, 'exec')
    namespace = {}
    # execute code allowing local imports
    filename = os.path.abspath(filename)
    file_dir = dirname(filename)
    if file_dir:
        sys.path.insert(0, file_dir)
    try:
        exec(code, namespace)
    finally:
        if file_dir:
            sys.path.remove(file_dir)
    # retrieve jobs
    jobs = list(namespace.get('JOBS', ()))
    if not jobs:
        raise RuntimeError(f"No JOBS in file {filename}")
    return jobs


def make_jobs_command():
    """Command-line command

    Usage::

        $ trf-tools-make-jobs jobs.py
    """
    argparser = ArgumentParser(description="TRF-Tools make-jobs")
    argparser.add_argument('job_file')
    argparser.add_argument('--notify', type=str, default='', help="Send email notification to this address when the jobs are finshed")
    argparser.add_argument('-p', '--processes', type=int, default=None, help='Number of processes to use for jobs (0 to disable multiprocessing; negative number for "all but").')
    argparser.add_argument('-n', '--nice', type=int, default=None, help='Scheduling priority (-20 is highest priority, 19 is lowest).')
    argparser.add_argument('--open', action='store_true', help="Open new reports in browser")
    args = argparser.parse_args()

    if args.processes or args.nice:
        configure(args.processes, nice=args.nice)

    if args.notify:
        notification = Notifier(args.notify, 'TRF-Jobs', debug=False)
    else:
        notification = contextlib.nullcontext()

    handler = ScreenHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    with notification:
        make_jobs(args.job_file, args.open)
