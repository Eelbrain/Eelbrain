"""Utilities for loading :class:`Pipeline` subclasses from Python files."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

from .._types import PathArg
from .pipeline import Pipeline

DEFAULT_PIPELINE_FILES = ('pipeline.py', 'experiment.py')


def _split_spec(spec: PathArg | str | None) -> tuple[Path, str | None]:
    if spec is None:
        path = Path.cwd()
        class_name = None
    elif isinstance(spec, Path):
        path = spec
        class_name = None
    else:
        spec = str(spec)
        if ':' in spec:
            path_str, class_name = spec.rsplit(':', 1)
            path = Path(path_str)
            if not class_name:
                raise ValueError(f"Invalid pipeline spec {spec!r}: missing class name after ':'.")
        else:
            path = Path(spec)
            class_name = None

    path = path.expanduser().absolute()
    if path.is_dir():
        for filename in DEFAULT_PIPELINE_FILES:
            candidate = path / filename
            if candidate.exists():
                return candidate, class_name
        filenames = ' or '.join(repr(name) for name in DEFAULT_PIPELINE_FILES)
        raise FileNotFoundError(f"No pipeline file found in {path}. Expected {filenames}.")
    if path.suffix != '.py':
        raise ValueError(
            f"Invalid pipeline spec {spec!r}: expected a Python file, a directory, or "
            f"'path/to/file.py:ClassName'."
        )
    return path, class_name


def _load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    module_name = f"_eelbrain_pipeline_{path.stem}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _find_pipeline_subclasses(module: ModuleType) -> dict[str, type[Pipeline]]:
    out = {}
    for name, value in vars(module).items():
        if not isinstance(value, type):
            continue
        if value is Pipeline or not issubclass(value, Pipeline):
            continue
        if value.__module__ != module.__name__:
            continue
        out[name] = value
    return out


def _find_root(module: ModuleType) -> PathArg | None:
    for name, value in vars(module).items():
        if name.lower() == 'root':
            return value
    return None


def load_pipeline(spec: PathArg | str | None = None, root: PathArg = None, **state) -> Pipeline:
    """Load a :class:`Pipeline` subclass from a Python file.

    Parameters
    ----------
    spec
        Path to a Python file that defines a :class:`Pipeline` subclass,
        optionally followed by ``:ClassName``.
        ``spec`` can also be a directory, in which case ``pipeline.py`` is
        tried first and then ``experiment.py``.
        If ``spec`` is omitted, the current working directory is searched.
        When only a file path is provided, the file must define exactly one
        :class:`Pipeline` subclass.
    root
        Root directory for the experiment.
    **state
        Initial state parameters passed to the :class:`Pipeline` constructor.

    Returns
    -------
    pipeline
        Instantiated :class:`Pipeline`.
    """
    path, class_name = _split_spec(spec)
    module = _load_module(path)
    pipelines = _find_pipeline_subclasses(module)
    if root is None:
        root = _find_root(module)

    if class_name is None:
        if not pipelines:
            raise ValueError(f"No Pipeline subclass found in {path}.")
        if len(pipelines) > 1:
            classes = ', '.join(sorted(pipelines))
            raise ValueError(
                f"Found multiple Pipeline subclasses in {path}: {classes}. "
                f"Use 'path/to/file.py:ClassName' to select one."
            )
        pipeline_type = next(iter(pipelines.values()))
    else:
        try:
            pipeline_type = pipelines[class_name]
        except KeyError:
            if hasattr(module, class_name):
                raise TypeError(
                    f"{path}:{class_name} exists, but {class_name} is not a Pipeline subclass."
                ) from None
            raise ValueError(
                f"No Pipeline subclass named {class_name!r} found in {path}."
            ) from None

    if root is None:
        raise TypeError(
            f"No experiment root specified for {path}. Pass root=... or define a module-level "
            f"'root' variable in the pipeline file."
        )

    return pipeline_type(root, **state)
