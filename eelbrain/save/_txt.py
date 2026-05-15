# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Write text files"""
from collections.abc import Iterable

from .._types import PathArg
from .._utils import ui


def txt(iterator: Iterable, fmt: str = '%s', delim: str = '\n', dest: PathArg = None):
    """Write any object that supports iteration to a text file

    Parameters
    ----------
    iterator
        Object that iterates over values to be saved.
    fmt
        Format-string which is used to format the iterator's values.
    delim
        The delimiter which is inserted between values.
    dest
        Path to save at; by default, a system save-as dialog is displayed.
    """
    if dest is None:
        name = repr(iterator)[:20]
        msg = f"Save {name}..."
        dest = ui.ask_saveas(msg, msg, [("Plain Text File", '*.txt')])

    if dest:
        with open(dest, 'w') as file:
            file.write(delim.join(fmt % v for v in iterator))
