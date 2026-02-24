# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import fnmatch
import re

import numpy as np

from .._data_obj import cellname
from . import stats
from .contrast import parse


# array functions:  work on array, take axis argument
np_afuncs = {'min': np.min,
             'max': np.max,
             'sum': np.sum}
# binary functions:  work on two arrays
np_bfuncs = {'subtract': np.subtract,
             'add': np.add}
# unary functions:  work on a single array
np_ufuncs = {'abs': np.abs,
             'negative': np.negative}


class TContrastSpec:
    "Parse a contrast expression and expose methods to apply it"

    def __init__(self, contrast, cells, indexes):
        """Parse a contrast expression and expose methods to apply it

        Parameters
        ----------
        contrast : str
            Contrast specification.
        cells : tuple of cells
            Cells that occur in the contrast (each cell is represented by a str
            or a tuple of str).
        indexes : dict {cell: index}
            Indexes for the data of every cell.
        """
        ast = parse(contrast)
        n_buffers, cells_in_contrast = _t_contrast_rel_properties(ast)
        pcells, mcells = _t_contrast_rel_expand_cells(cells_in_contrast, cells)

        self.contrast = contrast
        self.indexes = indexes
        self._ast = ast
        self._pcells = pcells
        self._mcells = mcells
        self._n_buffers = n_buffers

        # data buffers
        self._buffer_shape = None
        self._buffer = None
        self._y_perm = None

    def map(self, y):
        "Apply contrast without retainig data buffers"
        buff = np.empty((self._n_buffers,) + y.shape[1:])
        data = _t_contrast_rel_data(y, self.indexes, self._pcells, self._mcells)
        tmap = _t_contrast_rel(self._ast, data, buff)
        return tmap

    def __call__(self, y, out, perm):
        "Apply contrast to permutation of the data, storing and recycling data buffers"
        buffer_shape = (self._n_buffers,) + y.shape[1:]
        if self._buffer_shape != buffer_shape:
            self._buffer = np.empty(buffer_shape)
            self._y_perm = np.empty_like(y)
            self._buffer_shape = buffer_shape
        self._y_perm[perm] = y
        data = _t_contrast_rel_data(self._y_perm, self.indexes, self._pcells, self._mcells)
        tmap = _t_contrast_rel(self._ast, data, self._buffer, out)
        return tmap


def _t_contrast_rel_properties(item):
    """Find properties of a compiled t-contrast

    Parameters
    ----------
    item : tuple
        Contrast specification.

    Returns
    -------
    n_buffers : int
        Number of buffer maps needed.
    cells : set
        names of all cells that occur in the contrast.
    """
    if item[0] == 'ufunc':
        needed_buffers, cells = _t_contrast_rel_properties(item[2])
        return needed_buffers + 1, cells
    elif item[0] in ('bfunc', 'afunc'):
        _, _, items_ = item
        local_buffers = len(items_)
        cells = set()
        for i, item_ in enumerate(items_):
            available_buffers = local_buffers - i - 1
            needed_buffers, cells_ = _t_contrast_rel_properties(item_)
            additional_buffers = needed_buffers - available_buffers
            if additional_buffers > 0:
                local_buffers += additional_buffers
            cells.update(cells_)
        return local_buffers, cells
    else:
        return 0, set(item[1:])


def _t_contrast_rel_expand_cells(cells, all_cells):
    """Find cells that are an average of other cells

    Parameters
    ----------
    cells : set
        Cells occurring in the contrast.
    all_cells : tuple
        All cells in the data.

    Returns
    -------
    primary_cells : set
        All cells that occur directly in the data.
    mean_cells : dict
        ``{name: components}`` dictionary (components being a tuple with all
        cells to be averaged).
    """
    # check all cells have same number of components
    ns = {1 if isinstance(cell, str) else len(cell) for cell in all_cells}
    ns.update(1 if isinstance(cell, str) else len(cell) for cell in cells)
    if len(ns) > 1:
        msg = ("Not all cells have the same number of components: %s" %
               str(tuple(cells) + tuple(all_cells)))
        raise ValueError(msg)

    # convert cells to str for fnmatch
    all_cellnames = tuple(cellname(cell, '|') for cell in all_cells)

    primary_cells = set()
    mean_cells = {}
    for cell in cells:
        if cell in all_cells:
            primary_cells.add(cell)
        else:
            r = re.compile(fnmatch.translate(cellname(cell, '|')))
            base = tuple(c for c, cn in zip(all_cells, all_cellnames) if r.match(cn))
            if len(base) == 0:
                raise ValueError("%r does not match any cells in data %r" %
                                 (cellname(cell, '|'), ', '.join(all_cellnames)))
            mean_cells[cell] = base
            primary_cells.update(base)

    return primary_cells, mean_cells


def _t_contrast_rel_data(y, indexes, cells, mean_cells):
    "Create {cell: data} dictionary"
    data = {cell: y[indexes[cell]] for cell in cells}
    for name, cells_ in mean_cells.items():
        cell = cells_[0]
        x = data[cell].copy()
        for cell in cells_[1:]:
            x += data[cell]
        x /= len(cells_)
        data[name] = x

    return data


def _t_contrast_rel(item, data, buff, out=None):
    "Execute a t_contrast (recursive)"
    if item[0] == 'ufunc':
        _, func, item_ = item
        tmap = _t_contrast_rel(item_, data, buff[1:], buff[0])
        tmap = func(tmap, out)
    elif item[0] == 'bfunc':
        _, func, items = item
        tmap1 = _t_contrast_rel(items[0], data, buff[2:], buff[1])
        tmap2 = _t_contrast_rel(items[1], data, buff[2:], buff[0])
        tmap = func(tmap1, tmap2, out)
    elif item[0] == 'afunc':
        _, func, items_ = item
        tmaps = buff[:len(items_)]
        for i, item_ in enumerate(items_):
            _t_contrast_rel(item_, data, buff[i + 1:], tmaps[i])
        tmap = func(tmaps, axis=0, out=out)
    else:
        _, c1, c0 = item
        tmap = stats.t_1samp(data[c1] - data[c0], out)

    return tmap
