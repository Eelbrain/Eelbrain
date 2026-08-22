import threading
from types import SimpleNamespace

import numpy as np

from eelbrain import Dataset, Var
from eelbrain._exceptions import ConfigurationError, DataError
from eelbrain._experiment.exceptions import FileMissingError
from eelbrain._wxgui import pipeline_gui
from eelbrain._wxgui.pipeline_gui import PipelineFrame, _format_user_error


def test_format_user_error():
    title, message = _format_user_error(FileMissingError("raw.fif not found"))
    assert title == "Missing input"
    assert "required input file" in message
    assert "raw.fif not found" in message

    title, message = _format_user_error(FileNotFoundError("missing", "No file", "trans.fif"))
    assert title == "Missing file"
    assert "trans.fif" in message

    title, message = _format_user_error(DataError("bad montage"))
    assert title == "Data error"
    assert message == "bad montage"

    title, message = _format_user_error(ConfigurationError("bad setup"))
    assert title == "Configuration error"
    assert message == "bad setup"

    assert _format_user_error(RuntimeError("programmer error")) is None


def test_result_columns():
    "The one piece of per-task logic left in the unified job queue"
    ica = SimpleNamespace(n_components_=12, exclude=[0, 3])
    assert PipelineFrame._result_columns('ica', ica) == ('12', '2')

    rej_ds = Dataset({'accept': Var(np.array([True, False, True]))})
    assert PipelineFrame._result_columns('epoch_rej', rej_ds) == ('3', '1')

    # every computable task has both status labels and a result mapping
    assert PipelineFrame._MISSING_STATUS.keys() == PipelineFrame._DONE_STATUS.keys()
    for kind in PipelineFrame._MISSING_STATUS:
        assert PipelineFrame._result_columns(kind, ica if kind == 'ica' else rej_ds)


class _FakeList:
    "Minimal wx.ListCtrl stand-in recording the cells that were written"

    def __init__(self, rows: list[tuple], n_columns: int = 3):
        self.rows = rows
        self.n_columns = n_columns
        self.written = {}  # {(row, col): value}
        self.cleared = False

    def GetItemCount(self):
        return len(self.rows)

    def GetColumnCount(self):
        return self.n_columns

    def GetItemText(self, row, col):
        return self.written.get((row, col), self.rows[row][col])

    def SetItem(self, row, col, value):
        self.written[row, col] = value

    def DeleteAllItems(self):
        self.cleared = True


def _frame(**attrs) -> PipelineFrame:
    "PipelineFrame with only the attributes a method under test needs (no wx window)"
    frame = PipelineFrame.__new__(PipelineFrame)
    frame.__dict__.update(attrs)
    return frame


_ICA_SCOPE = ('ica', 'ica', None, '1-40')
_OTHER_SCOPE = ('ica', 'ica', None, 'ica-2')


def test_displayed_row_requires_a_matching_scope():
    "A job never writes into a row of the table it was not minted for"
    frame = _frame(
        _table_scope=lambda: _ICA_SCOPE,
        _find_row=lambda combo: 2,
    )
    assert frame._displayed_row(_ICA_SCOPE, ('R0000',)) == 2
    # same combo, but the Raw choice has moved on since the job was queued
    assert frame._displayed_row(_OTHER_SCOPE, ('R0000',)) == -1
    assert frame._displayed_row(('epoch_rej', 'man', 'target', '1-40'), ('R0000',)) == -1


def test_queue_jobs_dedupes_within_one_scope_only():
    "The same combo in two tables is two jobs; the same combo in one table is one"
    started = []

    def start_compute():  # as the real one: a worker is now running
        started.append(True)
        frame._compute_token = object()

    frame = _frame(
        _job_queue=[],
        _job_in_progress=None,
        _job_queue_lock=threading.Lock(),
        _n_total=0,
        _compute_token=None,
        _list=_FakeList([('R0000', 'no ICA', '—')]),
        _table_scope=lambda: _ICA_SCOPE,
        _status_col=lambda: 1,
        _find_row=lambda combo: 0,
        _start_compute=start_compute,
        _update_progress=lambda: None,
    )
    spec = object()
    frame._queue_jobs(_ICA_SCOPE, [(('R0000',), spec)])
    frame._queue_jobs(_ICA_SCOPE, [(('R0000',), spec)])  # already queued
    frame._queue_jobs(_OTHER_SCOPE, [(('R0000',), spec)])  # a different table
    assert [entry[0] for entry in frame._job_queue] == [_ICA_SCOPE, _OTHER_SCOPE]
    assert frame._n_total == 2
    # only the row of the table on display is marked, and only for its own scope
    assert frame._list.GetItemText(0, 1) == 'queued'
    assert started == [True]


def test_compute_job_holds_the_pipeline_lock_except_for_the_fit():
    "Loading and saving are serialized against the refresh walk; the computation is not"
    held = {}

    class _Job:
        def __call__(self):
            held['fit'] = frame._pipeline_lock.locked()
            return 'RESULT'

    class _Spec:
        def make_job(self):
            held['make_job'] = frame._pipeline_lock.locked()
            return _Job()

        def save_result(self, job, result):
            held['save_result'] = frame._pipeline_lock.locked()
            return result

    frame = _frame(_pipeline_lock=threading.Lock())
    assert frame._compute_job('ica', _Spec(), ('R0000',)) == 'RESULT'
    # an hour-long fit must not keep the refresh thread out
    assert held == {'make_job': True, 'fit': False, 'save_result': True}
    assert not frame._pipeline_lock.locked()


def test_refresh_holds_the_pipeline_lock(monkeypatch):
    "The refresh walk never runs while the worker is loading or saving"
    monkeypatch.setattr(pipeline_gui.wx, 'CallAfter', lambda *args: posted.append(args))
    posted = []
    locked = []
    frame = _frame(
        _pipeline_lock=threading.Lock(),
        _compute_rows=lambda token, scope: locked.append(frame._pipeline_lock.locked()) or ([], {}),
    )
    frame._refresh_thread(object(), _ICA_SCOPE)
    assert locked == [True]
    assert not frame._pipeline_lock.locked()  # released before the table update is posted
    assert posted
