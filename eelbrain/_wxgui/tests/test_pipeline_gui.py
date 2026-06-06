from eelbrain._exceptions import ConfigurationError, DataError
from eelbrain._experiment.exceptions import FileMissingError
from eelbrain._wxgui.pipeline_gui import _format_user_error


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
