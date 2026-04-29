import logging
from pathlib import Path

import pytest

import eelbrain


PIPELINE = (
    "from eelbrain import Pipeline\n"
    "\n"
    "class {name}(Pipeline):\n"
    "    datatype = 'meg'\n"
    "    extension = '.fif'\n"
    "    defaults = {{'session': {session!r}}}\n"
)


def test_load_pipeline_from_file(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(PIPELINE.format(name='Experiment', session='test'))

    e = eelbrain.load_pipeline(path, root=tmp_path)

    assert isinstance(e, eelbrain.Pipeline)
    assert e.__class__.__name__ == 'Experiment'
    assert e.root == Path(tmp_path).absolute()
    assert e.get('session') == 'test'


def test_load_pipeline_overrides_screen_log_level(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(
        f"root = {str(tmp_path)!r}\n\n"
        + PIPELINE.format(name='Experiment', session='test')
    )

    e = eelbrain.load_pipeline(path, log_level='DEBUG')

    assert e._screen_log_level == logging.DEBUG
    assert e.__class__.__name__ == 'Experiment'


def test_load_pipeline_uses_root_from_module(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(f"root = {str(tmp_path)!r}\n\n" + PIPELINE.format(name='Experiment', session='test'))

    e = eelbrain.load_pipeline(path)

    assert e.root == Path(tmp_path).absolute()


def test_load_pipeline_uses_case_insensitive_root_from_module(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(f"ROOT = {str(tmp_path)!r}\n\n" + PIPELINE.format(name='Experiment', session='test'))

    e = eelbrain.load_pipeline(path)

    assert e.root == Path(tmp_path).absolute()


def test_load_pipeline_from_file_with_named_class(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(
        PIPELINE.format(name='Pilot', session='pilot')
        + "\n"
        + PIPELINE.format(name='Main', session='main')
    )

    e = eelbrain.load_pipeline(f'{path}:Main', root=tmp_path)

    assert e.__class__.__name__ == 'Main'
    assert e.get('session') == 'main'


def test_load_pipeline_errors_for_ambiguous_file(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(
        PIPELINE.format(name='Pilot', session='pilot')
        + "\n"
        + PIPELINE.format(name='Main', session='main')
    )

    with pytest.raises(ValueError, match="Found multiple Pipeline subclasses"):
        eelbrain.load_pipeline(path, root=tmp_path)


def test_load_pipeline_errors_for_file_without_pipeline(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text("VALUE = 1\n")

    with pytest.raises(ValueError, match="No Pipeline subclass found"):
        eelbrain.load_pipeline(path, root=tmp_path)


def test_load_pipeline_errors_for_non_pipeline_class(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(
        "class Experiment:\n"
        "    pass\n"
    )

    with pytest.raises(TypeError, match="is not a Pipeline subclass"):
        eelbrain.load_pipeline(f'{path}:Experiment', root=tmp_path)


def test_load_pipeline_errors_without_root(tmp_path):
    path = tmp_path / 'experiment.py'
    path.write_text(PIPELINE.format(name='Experiment', session='test'))

    with pytest.raises(TypeError, match="No experiment root specified"):
        eelbrain.load_pipeline(path)


def test_load_pipeline_from_directory(tmp_path):
    (tmp_path / 'experiment.py').write_text(PIPELINE.format(name='Experiment', session='test'))

    e = eelbrain.load_pipeline(tmp_path, root=tmp_path)

    assert e.__class__.__name__ == 'Experiment'
    assert e.get('session') == 'test'


def test_load_pipeline_prefers_pipeline_py(tmp_path):
    (tmp_path / 'pipeline.py').write_text(PIPELINE.format(name='PipelineExperiment', session='pipeline'))
    (tmp_path / 'experiment.py').write_text(PIPELINE.format(name='Experiment', session='experiment'))

    e = eelbrain.load_pipeline(tmp_path, root=tmp_path)

    assert e.__class__.__name__ == 'PipelineExperiment'
    assert e.get('session') == 'pipeline'


def test_load_pipeline_defaults_to_current_directory(tmp_path, monkeypatch):
    (tmp_path / 'pipeline.py').write_text(f"root = {str(tmp_path)!r}\n\n" + PIPELINE.format(name='Experiment', session='cwd'))
    monkeypatch.chdir(tmp_path)

    e = eelbrain.load_pipeline()

    assert e.__class__.__name__ == 'Experiment'
    assert e.get('session') == 'cwd'
