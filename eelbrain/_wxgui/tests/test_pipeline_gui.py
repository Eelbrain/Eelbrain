import logging
import threading
from types import SimpleNamespace

import numpy as np
import wx

from eelbrain import Dataset, Var
from eelbrain._exceptions import ConfigurationError, DataError
from eelbrain._experiment.epoch_rejection import ChannelModelRejection, ManualRejection
from eelbrain._experiment.exceptions import FileMissingError
from eelbrain._wxgui import pipeline_gui
from eelbrain._wxgui.pipeline_gui import COMMON_BRAIN_ROW, GREY, PLACEHOLDER, TASKS, TASKS_BY_NAME, Layout, PipelineFrame, _format_user_error


def pipeline(
        sessions: int = 1,
        tasks: int = 1,
        runs: int = 1,
        concatenate_runs: bool = True,
) -> SimpleNamespace:
    """Minimal stand-in exposing the pipeline attributes the tasks read."""
    return SimpleNamespace(
        _log=logging.getLogger('pipeline_gui_test'),
        _sessions=[f's{i}' for i in range(sessions)],
        _tasks=[f't{i}' for i in range(tasks)],
        _runs=[f'{i}' for i in range(runs)],
        _raw={'raw': SimpleNamespace(_concatenate_runs=concatenate_runs)},
        _epoch_rejection={'manual': ManualRejection(), 'auto': None},
    )


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


def test_task_layout():
    "Column layout, and the Status index derived from it"
    p = pipeline()
    assert TASKS_BY_NAME['bad_chs'].layout(p, 'raw').columns == (('Subject', 180), ('Status', 110), ('N bad', 90))
    assert TASKS_BY_NAME['ica'].layout(p, 'raw').columns == (('Subject', 180), ('Status', 110), ('Components', 110), ('Rejected', 90))
    assert TASKS_BY_NAME['epoch_rej'].layout(p, 'raw').columns == (('Subject', 180), ('Status', 110), ('N total', 90), ('N rejected', 90))
    assert TASKS_BY_NAME['mri'].layout(p, None).columns == (('Subject', 180), ('MRI subject', 170), ('Status', 130))
    assert TASKS_BY_NAME['coreg'].layout(p, None).columns == (('Subject', 140), ('Session', 80), ('MRI subject', 140), ('Status', 110))

    # Status sits after the key fields for the per-recording tasks, but further
    # right for MRI/Coregistration, whose leading columns are display-only
    for task, raw_name, status_col in [(TASKS_BY_NAME['bad_chs'], 'raw', 1), (TASKS_BY_NAME['mri'], None, 2), (TASKS_BY_NAME['coreg'], None, 3)]:
        layout = task.layout(p, raw_name)
        assert layout.status_col == status_col
        assert layout.columns[status_col][0] == 'Status'
        assert len(layout.columns) == status_col + 1 + len(task.detail_columns)

    # a layout is a value: it compares equal across derivations, so it can identify
    # a table in a job's scope (see PipelineFrame._table_scope)
    assert TASKS_BY_NAME['ica'].layout(p, 'raw') == TASKS_BY_NAME['ica'].layout(pipeline(), 'raw')
    assert TASKS_BY_NAME['ica'].layout(p, 'raw') != TASKS_BY_NAME['ica'].layout(pipeline(sessions=2), 'raw')


def test_layout_iter_arg():
    "Rows are iterated over exactly the key fields the columns were built from"
    p = pipeline(sessions=2)
    assert TASKS_BY_NAME['ica'].layout(p, 'raw').iter_arg == ['subject', 'session']
    assert TASKS_BY_NAME['mri'].layout(p, None).iter_arg == 'subject'
    assert TASKS_BY_NAME['coreg'].layout(p, None).iter_arg == ['subject', 'session']


def test_task_key_fields():
    "Key fields expand with the state fields that vary in the experiment"
    bad_chs, ica, coreg = TASKS_BY_NAME['bad_chs'], TASKS_BY_NAME['ica'], TASKS_BY_NAME['coreg']
    assert bad_chs.key_fields(pipeline(), 'raw') == ('subject',)
    assert bad_chs.key_fields(pipeline(sessions=2, runs=3), 'raw') == ('subject', 'session', 'run')
    assert bad_chs.key_fields(pipeline(sessions=2, tasks=2, runs=2), 'raw') == ('subject', 'session', 'task', 'run')

    # ICA is cached per recording, but not per run when runs are concatenated
    assert ica.key_fields(pipeline(runs=3), 'raw') == ('subject',)
    assert ica.key_fields(pipeline(runs=3, concatenate_runs=False), 'raw') == ('subject', 'run')
    assert ica.key_fields(pipeline(sessions=2, tasks=2), 'raw') == ('subject', 'session')  # ICA ignores task

    assert coreg.key_fields(pipeline(), None) == ('subject', 'session')

    # extra columns follow the key fields, so a row is exactly as wide as its columns
    p = pipeline(sessions=2, runs=2, concatenate_runs=False)
    layout = ica.layout(p, 'raw')
    assert layout.columns[1:3] == (('Session', 90), ('Run', 90))
    assert layout.status_col == len(layout.key_fields)


def _layout(name: str, **kwargs) -> Layout:
    "Layout of a task for a pipeline with the given fields varying"
    task = TASKS_BY_NAME[name]
    return task.layout(pipeline(**kwargs), 'raw' if task.shows_raw else None)


def test_task_status_bar():
    "Status-bar summary for each task"
    bad_chs = _layout('bad_chs')
    assert TASKS_BY_NAME['bad_chs'].status_bar([('R01', 'done', '3'), ('R02', 'error', PLACEHOLDER), ('R03', 'done', '0')], bad_chs) == "2 / 3 subjects · bad channels defined  (1 error)"
    assert TASKS_BY_NAME['bad_chs'].status_bar([('R01', 'done', '3')], bad_chs) == "1 / 1 subjects · bad channels defined"
    # bad channels are stored per recording, so with a second session the rows are
    # recordings rather than subjects
    assert TASKS_BY_NAME['bad_chs'].status_bar([('R01', 's0', 'done', '3'), ('R01', 's1', 'error', PLACEHOLDER)], _layout('bad_chs', sessions=2)) == "1 / 2 recordings · bad channels defined  (1 error)"

    # queued and computing rows have no artifact either, so they count as missing
    ica_rows = [('R01', 'selected', '30', '2'), ('R02', 'no ICA', PLACEHOLDER, PLACEHOLDER), ('R03', 'queued', PLACEHOLDER, PLACEHOLDER), ('R04', '⟳', PLACEHOLDER, PLACEHOLDER), ('R05', 'no data', PLACEHOLDER, PLACEHOLDER)]
    assert TASKS_BY_NAME['ica'].status_bar(ica_rows, _layout('ica')) == "1 / 5 subjects · ICA selected  (3 missing ICA file)"
    # one row per recording once a key field beyond subject varies
    assert TASKS_BY_NAME['ica'].status_bar([('R01', 'a', 'selected', '30', '2'), ('R02', 'b', 'no ICA', PLACEHOLDER, PLACEHOLDER)], _layout('ica', sessions=2)) == "1 / 2 recordings · ICA selected  (1 missing ICA file)"

    assert TASKS_BY_NAME['epoch_rej'].status_bar([('R01', 'done', '100', '5'), ('R02', 'missing', PLACEHOLDER, PLACEHOLDER)], _layout('epoch_rej')) == "1 / 2 subjects · epoch rejection done"
    # epoch rejection is per subject, so a second session does not change the noun
    assert TASKS_BY_NAME['epoch_rej'].status_bar([('R01', 'done', '100', '5')], _layout('epoch_rej', sessions=2)) == "1 / 1 subjects · epoch rejection done"

    # the common brain is not a subject and stays out of the count
    mri = _layout('mri')
    mri_rows = [('R01', 'R01', 'ok'), ('R02', 'fsaverage', 'template'), ('R03', 'R03', 'no MRI'), (COMMON_BRAIN_ROW, 'fsaverage', 'ok')]
    assert TASKS_BY_NAME['mri'].status_bar(mri_rows, mri) == "2 / 3 subjects · MRI available  (1 missing)"
    assert TASKS_BY_NAME['mri'].status_bar([('R01', 'R01', 'ok'), (COMMON_BRAIN_ROW, 'fsaverage', 'missing')], mri) == "1 / 1 subjects · MRI available"

    # Coregistration has two key fields but counts sessions, not recordings
    assert TASKS_BY_NAME['coreg'].status_bar([('R01', 's1', 'R01', 'ok'), ('R02', 's1', 'R02', 'missing')], _layout('coreg')) == "1 / 2 sessions · coregistration done  (1 missing)"


def test_task_row_colour():
    "Rows needing attention are red, rows that are not actionable are grey"
    assert TASKS_BY_NAME['bad_chs'].row_colour(('R01', 'error', PLACEHOLDER), _layout('bad_chs')) is wx.RED
    assert TASKS_BY_NAME['bad_chs'].row_colour(('R01', 'done', '3'), _layout('bad_chs')) is None

    # an ICA without a single rejected component is almost always an oversight
    ica = _layout('ica')
    assert TASKS_BY_NAME['ica'].row_colour(('R01', 'selected', '30', '0'), ica) is wx.RED
    assert TASKS_BY_NAME['ica'].row_colour(('R01', 'selected', '30', '2'), ica) is None
    assert TASKS_BY_NAME['ica'].row_colour(('R01', 'no ICA', PLACEHOLDER, PLACEHOLDER), ica) is None
    # the same rule at the wider layout, addressed through the Status column
    assert TASKS_BY_NAME['ica'].row_colour(('R01', 'a', 'selected', '30', '0'), _layout('ica', sessions=2)) is wx.RED

    mri = _layout('mri')
    assert TASKS_BY_NAME['mri'].row_colour(('R01', 'R01', 'no MRI'), mri) is wx.RED
    assert TASKS_BY_NAME['mri'].row_colour(('R01', 'R01', 'ok'), mri) is None
    assert TASKS_BY_NAME['mri'].row_colour((COMMON_BRAIN_ROW, 'fsaverage', 'ok'), mri) is GREY

    assert TASKS_BY_NAME['coreg'].row_colour(('R01', 's1', 'R01', 'missing'), _layout('coreg')) is wx.RED
    assert TASKS_BY_NAME['coreg'].row_colour(('R01', 's1', 'R01', 'ok'), _layout('coreg')) is None

    assert TASKS_BY_NAME['epoch_rej'].row_colour(('R01', 'missing', PLACEHOLDER, PLACEHOLDER), _layout('epoch_rej')) is None


def test_task_computable():
    "Only tasks that can make their artifact in bulk offer the compute button"
    p = pipeline()
    assert [task.name for task in TASKS if task.compute_label] == ['ica', 'epoch_rej']
    assert TASKS_BY_NAME['ica'].computable(p, None)
    assert not TASKS_BY_NAME['mri'].computable(p, None)
    # a ManualRejection is edited in its own GUI; only an automatic one is computed
    p._epoch_rejection = {'manual': ManualRejection(), 'auto': ChannelModelRejection()}
    assert TASKS_BY_NAME['epoch_rej'].computable(p, 'auto')
    assert not TASKS_BY_NAME['epoch_rej'].computable(p, 'manual')
    assert not TASKS_BY_NAME['epoch_rej'].computable(p, None)


def test_task_missing_row():
    "A row with no artifact is exactly as wide as the columns, for every task"
    for task in TASKS:
        layout = task.layout(pipeline(sessions=2), 'raw' if task.shows_raw else None)
        row = task.missing_row(('R01',) * len(layout.key_fields), layout)
        assert len(row) == len(layout.columns)
        assert row[layout.status_col] == task.missing_status
        # every column that is not a key field has no value to show
        assert set(row[len(layout.key_fields):layout.status_col]) <= {PLACEHOLDER}
        assert set(row[layout.status_col + 1:]) <= {PLACEHOLDER}
    # an alternative status keeps the placeholders
    assert TASKS_BY_NAME['ica'].missing_row(('R01',), _layout('ica'), 'no data') == ('R01', 'no data', PLACEHOLDER, PLACEHOLDER)
    # MRI/Coregistration have a display-only column before Status; it is filled too
    assert TASKS_BY_NAME['coreg'].missing_row(('R01', 's1'), _layout('coreg')) == ('R01', 's1', PLACEHOLDER, 'missing')


def test_result_columns():
    "The detail columns describing a freshly computed artifact"
    ica = SimpleNamespace(n_components_=12, exclude=[0, 3])
    rej_ds = Dataset({'accept': Var(np.array([True, False, True]))})
    assert TASKS_BY_NAME['ica'].result_columns(ica) == ('12', '2')
    assert TASKS_BY_NAME['epoch_rej'].result_columns(rej_ds) == ('3', '1')

    # every computable task has both status labels and a result mapping
    for task in TASKS:
        if task.compute_label is None:
            continue
        assert task.missing_status and task.done_status
        result = ica if task.name == 'ica' else rej_ds
        assert len(task.result_columns(result)) == len(task.detail_columns)


class _FakeList:
    "Minimal wx.ListCtrl stand-in recording the cells that were written"

    def __init__(self, rows: list[tuple] = ()):
        self.rows = [tuple(row) for row in rows]
        self.columns = []  # [(title, width), ...] as installed
        self.colours = {}  # {row: colour}
        self.cleared = False

    def InsertColumn(self, col, label, width):
        assert col == len(self.columns)
        self.columns.append((label, width))

    def ClearAll(self):
        self.columns = []
        self.rows = []
        self.cleared = True

    def GetItemCount(self):
        return len(self.rows)

    def GetColumnCount(self):
        return len(self.columns)

    def GetItemText(self, row, col):
        return self.rows[row][col]

    def InsertItem(self, row, value):
        assert row == len(self.rows)
        self.rows.append((value,) + ('',) * (len(self.columns) - 1))
        return row

    def SetItem(self, row, col, value):
        assert col < len(self.columns), f"{col=} beyond the installed columns"
        self.rows[row] = self.rows[row][:col] + (value,) + self.rows[row][col + 1:]

    def SetItemTextColour(self, row, colour):
        self.colours[row] = colour

    def DeleteAllItems(self):
        self.rows = []
        self.cleared = True


def _frame(**attrs) -> PipelineFrame:
    "PipelineFrame with only the attributes a method under test needs (no wx window)"
    frame = PipelineFrame.__new__(PipelineFrame)
    frame.__dict__.update(attrs)
    return frame


def _table_frame(name: str, rows: list[tuple] = (), **kwargs) -> PipelineFrame:
    "Frame showing one task's table, with the layout _setup_columns would have cached"
    task = TASKS_BY_NAME[name]
    layout = _layout(name, **kwargs)
    fake = _FakeList()
    for i, (label, width) in enumerate(layout.columns):
        fake.InsertColumn(i, label, width=width)
    for row in rows:
        idx = fake.InsertItem(len(fake.rows), row[0])
        for col, value in enumerate(row[1:], 1):
            fake.SetItem(idx, col, value)
    return _frame(_list=fake, _layout=layout, _current_task=lambda: task)


# see PipelineFrame._table_scope
_ICA_LAYOUT = _layout('ica')
_ICA_SCOPE = (TASKS_BY_NAME['ica'], None, None, '1-40', _ICA_LAYOUT)
_OTHER_SCOPE = (TASKS_BY_NAME['ica'], None, None, 'ica-2', _ICA_LAYOUT)


def test_setup_columns_installs_and_caches_the_layout():
    "The cached layout is the one the columns were installed from"
    p = pipeline(sessions=2, runs=2, concatenate_runs=False)
    task = TASKS_BY_NAME['ica']
    frame = _frame(
        _pipeline=p,
        _list=_FakeList([('stale', 'row')]),
        _current_task=lambda: task,
        _raw_name=lambda: 'raw',
    )
    frame._setup_columns()
    assert frame._layout == task.layout(p, 'raw')
    assert frame._list.columns == list(frame._layout.columns)
    assert frame._list.rows == []  # the previous table's rows are gone

    # switching to a Raw pipe that concatenates runs drops the Run column again
    p._raw['ica'] = SimpleNamespace(_concatenate_runs=True)
    frame._raw_name = lambda: 'ica'
    frame._setup_columns()
    assert frame._layout.key_fields == ('subject', 'session')
    assert [label for label, _ in frame._list.columns] == ['Subject', 'Session', 'Status', 'Components', 'Rejected']


def test_populate_table_fits_the_rows_to_the_columns():
    "Rows built off the main thread are exactly as wide as the installed columns"
    task = TASKS_BY_NAME['ica']
    layout = _layout('ica', sessions=2)
    rows = [
        (*combo, task.done_status, '30', n_excluded) if n_excluded else task.missing_row(combo, layout)
        for combo, n_excluded in [(('R01', 's0'), '2'), (('R02', 's0'), '0'), (('R03', 's1'), None)]
    ]
    frame = _table_frame('ica', sessions=2)
    frame.__dict__.update(_refresh_token='TOKEN', _job_specs={'stale': True}, _refresh_status_bar=lambda: None)
    frame._populate_table(rows, 'TOKEN')
    assert frame._list.rows == rows
    assert frame._job_specs == {}  # the previous table's specs are gone
    # only the ICA with no rejected component is coloured
    assert frame._list.colours == {0: wx.NullColour, 1: wx.RED, 2: wx.NullColour}

    # a table arriving for a stale token is dropped
    frame._populate_table([], 'OTHER')
    assert frame._list.rows == rows


def test_fill_row_writes_one_resolved_row():
    "The second refresh pass fills a loading row in, addressed by its position"
    task = TASKS_BY_NAME['ica']
    layout = _layout('ica')
    loading = [task.missing_row((subject,), layout, pipeline_gui.LOADING) for subject in ('R01', 'R02')]
    frame = _table_frame('ica', loading)
    frame.__dict__.update(_refresh_token='TOKEN', _job_specs={}, _refresh_status_bar=lambda: None, _table_scope=lambda: 'SCOPE')
    frame._fill_row('TOKEN', 'SCOPE', 1, ('R02',), ('R02', 'selected', '30', '2'), 'SPEC')
    assert frame._list.rows == [loading[0], ('R02', 'selected', '30', '2')]
    assert frame._job_specs == {('SCOPE', ('R02',)): 'SPEC'}

    # a row arriving for a stale token, for a row that is gone, or for a combo that no
    # longer sits at that position is dropped, spec and all
    for token, index, combo in [('OTHER', 0, ('R01',)), ('TOKEN', 2, ('R03',)), ('TOKEN', 0, ('R03',))]:
        frame._fill_row(token, 'SCOPE', index, combo, (*combo, 'selected', '30', '9'), 'DROPPED')
    assert frame._list.rows == [loading[0], ('R02', 'selected', '30', '2')]
    assert frame._job_specs == {('SCOPE', ('R02',)): 'SPEC'}


def test_status_bar_shows_progress_while_rows_load():
    "A partly filled table reports its progress rather than an undercount"
    task = TASKS_BY_NAME['ica']
    layout = _layout('ica')
    rows = [('R01', 'selected', '30', '2'), *(task.missing_row((subject,), layout, pipeline_gui.LOADING) for subject in ('R02', 'R03'))]
    frame = _table_frame('ica', rows)
    texts = []
    frame.__dict__.update(SetStatusText=texts.append)
    frame._refresh_status_bar()
    assert texts == ["Loading… 1 / 3"]

    # once no row is loading any more the task's own summary takes over
    frame._list.SetItem(1, 1, 'no ICA')
    frame._list.SetItem(2, 1, 'selected')
    frame._refresh_status_bar()
    assert texts[-1] == "2 / 3 subjects · ICA selected  (1 missing ICA file)"


def test_set_row_result_writes_status_details_and_colour():
    "One row write for every task, addressed through the cached layout"
    frame = _table_frame('ica', [('R01', 'queued', PLACEHOLDER, PLACEHOLDER)])
    ica = SimpleNamespace(n_components_=30, exclude=[1, 4])
    task = TASKS_BY_NAME['ica']
    frame._set_row_result(0, task.done_status, task.result_columns(ica))
    assert frame._list.rows == [('R01', 'selected', '30', '2')]
    assert frame._list.colours == {0: wx.NullColour}

    # the Status column sits further right for Coregistration
    frame = _table_frame('coreg', [('R01', 's1', 'R01', 'missing')])
    frame._set_row_result(0, TASKS_BY_NAME['coreg'].done_status, ())
    assert frame._list.rows == [('R01', 's1', 'R01', 'ok')]
    assert frame._list.colours == {0: wx.NullColour}


def test_displayed_row_requires_a_matching_scope():
    "A job never writes into a row of the table it was not minted for"
    frame = _frame(
        _table_scope=lambda: _ICA_SCOPE,
        _find_row=lambda combo: 2,
    )
    assert frame._displayed_row(_ICA_SCOPE, ('R0000',)) == 2
    # same combo, but the Raw choice has moved on since the job was queued
    assert frame._displayed_row(_OTHER_SCOPE, ('R0000',)) == -1
    assert frame._displayed_row((TASKS_BY_NAME['epoch_rej'], 'man', 'target', '1-40', _layout('epoch_rej')), ('R0000',)) == -1
    # the same table, but the Raw choice has changed the number of key-field columns
    assert frame._displayed_row((TASKS_BY_NAME['ica'], None, None, '1-40', _layout('ica', sessions=2)), ('R0000',)) == -1


def test_queue_jobs_dedupes_within_one_scope_only():
    "The same combo in two tables is two jobs; the same combo in one table is one"
    started = []

    def start_compute():  # as the real one: a worker is now running
        started.append(True)
        frame._compute_token = object()

    frame = _table_frame('ica', [('R0000', 'selected', '30', '0')])
    frame.__dict__.update(
        _job_queue=[],
        _job_in_progress=None,
        _job_queue_lock=threading.Lock(),
        _n_total=0,
        _compute_token=None,
        _table_scope=lambda: _ICA_SCOPE,
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
    assert frame._list.rows == [('R0000', 'queued', PLACEHOLDER, PLACEHOLDER)]
    # the row was red for having no rejected component; queueing it clears that
    assert frame._list.colours == {0: wx.NullColour}
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
    assert frame._compute_job(TASKS_BY_NAME['ica'], _Spec(), ('R0000',)) == 'RESULT'
    # an hour-long fit must not keep the refresh thread out
    assert held == {'make_job': True, 'fit': False, 'save_result': True}
    assert not frame._pipeline_lock.locked()


def test_refresh_holds_the_pipeline_lock(monkeypatch):
    "Neither refresh pass ever runs while the worker is loading or saving"
    monkeypatch.setattr(pipeline_gui.wx, 'CallAfter', lambda *args: posted.append(args))
    posted = []
    locked = []
    token = object()
    frame = _frame(
        _pipeline=pipeline(),
        _pipeline_lock=threading.Lock(),
        _refresh_token=token,
        _iter_combos=lambda scope: locked.append(frame._pipeline_lock.locked()) or iter([('R01',)]),
        _iter_rows=lambda token_, scope: locked.append(frame._pipeline_lock.locked()) or iter([(('R01',), ('R01', 'selected', '30', '2'), 'SPEC')]),
    )
    frame._refresh_thread(token, _ICA_SCOPE)
    assert locked == [True, True]
    assert not frame._pipeline_lock.locked()  # released before the table update is posted
    assert posted


def test_refresh_shows_the_rows_before_their_status(monkeypatch):
    "The table is posted from the first pass, then filled in row by row"
    monkeypatch.setattr(pipeline_gui.wx, 'CallAfter', lambda *args: posted.append(args))
    posted = []
    token = object()
    task = TASKS_BY_NAME['ica']
    rows = [(('R01',), ('R01', 'selected', '30', '2'), 'SPEC-1'), (('R02',), ('R02', 'no ICA', PLACEHOLDER, PLACEHOLDER), 'SPEC-2')]
    frame = _frame(
        _pipeline=pipeline(),
        _pipeline_lock=threading.Lock(),
        _refresh_token=token,
        _iter_combos=lambda scope: iter([('R01',), ('R02',)]),
        _iter_rows=lambda token_, scope: iter(rows),
    )
    frame._refresh_thread(token, _ICA_SCOPE)
    # the table goes up first, with every row still loading
    assert posted[0] == (frame._populate_table, [task.missing_row(combo, _ICA_LAYOUT, pipeline_gui.LOADING) for combo, _, _ in rows], token)
    # then one update per row, at the position the first pass put it
    assert posted[1] == (frame._fill_row, token, _ICA_SCOPE, 0, ('R01',), ('R01', 'selected', '30', '2'), 'SPEC-1')
    assert posted[2] == (frame._fill_row, token, _ICA_SCOPE, 1, ('R02',), ('R02', 'no ICA', PLACEHOLDER, PLACEHOLDER), 'SPEC-2')
    assert len(posted) == 3

    # a task switch between the two passes drops the table rather than paying for its rows
    posted.clear()
    frame._refresh_token = object()
    frame._refresh_thread(token, _ICA_SCOPE)
    assert posted == []
