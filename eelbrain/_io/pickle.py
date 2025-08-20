# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from packaging.version import Version
from pathlib import Path
from pickle import dump, HIGHEST_PROTOCOL, Unpickler
from itertools import chain
import os
from typing import Any

import numpy

from .._data_obj import Dataset, NDVar, Var, SourceSpaceBase, ismodelobject
from .._types import PathArg
from .._utils import IS_WINDOWS, tqdm, ui


NUMPY_1 = Version(numpy.__version__) < Version('2')


class EelUnpickler(Unpickler):

    def find_class(self, module, name):
        "Backwards-compatibility for changes in module paths"
        if module.startswith('eelbrain.'):
            if module == 'eelbrain.vessels.data':
                module = 'eelbrain._data_obj'
                class_names = {'var': 'Var', 'factor': 'Factor', 'ndvar': 'NDVar', 'datalist': 'Datalist', 'dataset': 'Dataset'}
                name = class_names[name]
            elif module.startswith('eelbrain.data.'):
                if module.startswith('eelbrain.data.load'):
                    module = module.replace('.data.load', '.load')
                elif module.startswith('eelbrain.data.stats'):
                    module = module.replace('.data.stats', '._stats')
                elif module.startswith('eelbrain.data.data_obj'):
                    module = module.replace('.data.data_obj', '._data_obj')
                else:
                    raise NotImplementedError(f"{module=}, {name=}")
        elif module.startswith('pathlib'):
            if name == 'WindowsPath' and not IS_WINDOWS:
                name = 'Path'
        elif NUMPY_1 and module.startswith('numpy._core.numeric'):
            # This affected some pickles created with numpy 2
            module = module.replace('numpy._core.numeric', 'numpy.core.numeric')

        return Unpickler.find_class(self, module, name)


def pickle(
        obj: Any,
        dest: PathArg = None,
        protocol: int = HIGHEST_PROTOCOL,
) -> None:
    """Pickle a Python object (see :mod:`pickle`).

    Parameters
    ----------
    obj
        Python object to save.
    dest
        Path to destination where to save the file. If no destination is
        provided, a file dialog is shown. If a destination without extension is
        provided, ``.pickle`` is appended.
    protocol
        Pickle protocol (default is ``pickle.HIGHEST_PROTOCOL``).

        .. Warning::
            Later versions of Python support higher versions of the pickle
            protocol.
            For pickles that can be opened in Python 2, use ``protocol<=2``.
            For pickles that can be opened in Python ≤ 3.7, use ``protocol<=4``.

    See Also
    --------
    eelbrain.load.unpickle
    """
    if dest is None:
        filetypes = [("Pickled Python Objects (*.pickle)", '*.pickle')]
        dest = ui.ask_saveas("Pickle Destination", "", filetypes)
        if dest is False:
            raise RuntimeError("User canceled")
        else:
            print(f'dest={dest!r}')
    else:
        dest = Path(dest).expanduser()
        if not dest.suffix:
            dest = dest.with_suffix('.pickle')

    try:
        with open(dest, 'wb') as fid:
            dump(obj, fid, protocol)
    except SystemError as exception:
        if exception.args[0] == 'error return without exception set':
            if os.path.exists(dest):
                os.remove(dest)
            raise IOError("An error occurred while pickling. This could be due to an attempt to pickle an array (or NDVar) that is too big. Try saving several smaller arrays.")
        else:
            raise


def unpickle(path: PathArg = None):
    """Load pickled Python objects from a file.

    Parameters
    ----------
    path
        Path to a pickle file. If omitted, a system file dialog is shown.
        If the user cancels the file dialog, a RuntimeError is raised.

    Notes
    -----
    This function is similar to Python's builtin :mod:`pickle`
    ``pickle.load(open(path))``, but also loads object saved
    with older versions of Eelbrain, and allows using a system file dialog to
    select a file.

    If you see ``ValueError: unsupported pickle protocol``, the pickle file
    was saved with a higher version of Python; in order to make pickles
    backwards-compatible, use :func:`~eelbrain.save.pickle` with a lower
    ``protocol=2``.
    To batch-convert multiple pickle files, use :func:`~eelbrain.load.convert_pickle_protocol`

    See Also
    --------
    eelbrain.save.pickle
    eelbrain.load.convert_pickle_protocol
    """
    if path is None:
        filetypes = [("Pickles (*.pickle)", '*.pickle'), ("All files", '.*')]
        path = ui.ask_file("Select File to Unpickle", "Select a file to unpickle", filetypes)
        if path is False:
            raise RuntimeError("User canceled")
        else:
            print(f"unpickle {path}")
    else:
        path = Path(path).expanduser()
        if not path.exists():
            for ext in ('.pickle', '.pickled'):
                new_path = path.with_suffix(ext)
                if new_path.exists():
                    path = new_path
                    break

    with open(path, 'rb') as fid:
        unpickler = EelUnpickler(fid, encoding='latin1')
        try:
            return unpickler.load()
        except EOFError:
            raise EOFError(f"Corrupted file, writing may have been interrupted: {path}")


def update_subjects_dir(
        obj: object,
        subjects_dir: PathArg,
        depth: int = 0,
) -> None:
    """Update FreeSurfer :attr:`~eelbrain.SourceSpace.subjects_dir` on source-space data

    Operates in-place.

    Parameters
    ----------
    obj
        Object on which to replace :attr:`~eelbrain.SourceSpace.subjects_dir`.
    subjects_dir
        New values for subjects_dir.
    depth
        Depth for visiting ``obj`` content and attributes.
        Default (0) only applies the function to ``obj`` without recursion,
        assuming that ``obj`` is itself an :class:`~eelbrain.NDVar` or :class:`~eelbrain.SourceSpace`.
        Use ``depth`` to replace ``subjects_dir`` on, e.g.,
        :class:`NDVars <eelbrain.NDVar>` in a :class:`list`, :class:`dict` values,
        or on object attributes.
        Use a negative number for an exhaustive search.

    Notes
    -----
    Use the ``depth`` parameter to recursively update content and attributes in
    ``obj``, for example, ``obj`` may be a list or :class:`dict` containing
    :class:`NDVars <eelbrain.NDVar>`, or a :class:`~eelbrain.BoostingResult` object.

    The following elements are searched:

      - Attributes of objects that have a ``__dict__``.
      - :class:`dict` values.
      - :class:`list` and :class:`tuple` items.
    """
    if isinstance(obj, SourceSpaceBase):
        if obj.subjects_dir != subjects_dir:
            obj.subjects_dir = subjects_dir
            if obj._subjects_dir is not None:
                obj._subjects_dir = subjects_dir
    elif isinstance(obj, NDVar):
        for dim in obj.dims:
            if isinstance(dim, SourceSpaceBase):
                if dim.subjects_dir == subjects_dir:
                    break
                dim.subjects_dir = subjects_dir
                if dim._subjects_dir is not None:
                    dim._subjects_dir = subjects_dir
        else:
            for v in obj.info.values():
                update_subjects_dir(v, subjects_dir, depth)
    elif depth:
        if isinstance(obj, Var):
            values = obj.info.values()
        elif isinstance(obj, Dataset):
            values = chain(obj.info, obj.values())
        elif ismodelobject(obj):
            return
        else:
            if hasattr(obj, '__dict__'):
                values = obj.__dict__.values()
            else:
                values = ()

            if isinstance(obj, dict):
                values = chain(values, obj.values())
            elif isinstance(obj, (tuple, list)):
                values = chain(values, obj)

        for v in values:
            update_subjects_dir(v, subjects_dir, depth - 1)


def convert_pickle_protocol(
        root: PathArg = None,
        to_protocol: int = 4,
        pattern: str = '**/*.pickle',
):
    """Load and re-save pickle files with a specific protocol

    Parameters
    ----------
    root
        Root directory to look for pickle files.
    to_protocol
        Protocol to re-save with.
    pattern
        Filename pattern used to find pickle files (default is all ``*.pickle``
        files under ``root``).

    Notes
    -----
    Python 3.8 introduced a new pickle protocol 5, which older versions of
    Python can't read. Trying to unpickle such files results in the following
    error::

        ValueError: unsupported pickle protocol: 5

    A simple solution to make those files readable in Python 3.7 and below is
    to use Python 3.8 to load these files and re-save them with a lower protocol
    version (e.g., version 4).
    The ``convert_pickle_protocol`` function allows doing that for a large
    number of files with a single command (by default, to all files in the
    ``root`` directory).
    """
    if root is None:
        root = ui.ask_dir("Select folder", "Select folder to search for pickle files")
        if root is False:
            raise RuntimeError("User canceled")
        else:
            print(f"Searching {root}")
    root = Path(root)
    for path in tqdm(root.glob(pattern), "Converted", unit=' files'):
        obj = unpickle(path)
        pickle(obj, path, to_protocol)
