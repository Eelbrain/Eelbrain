from types import SimpleNamespace

import numpy as np

from eelbrain import Dataset, Var
from eelbrain._exceptions import ConfigurationError, DataError
from eelbrain._experiment.exceptions import FileMissingError
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
