# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Data containers and basic operations

Data Representation
===================

Data is stored in three main vessels:

:class:`Factor`:
    stores categorical data
:class:`Var`:
    stores numeric data
:class:`NDVar`:
    stores numerical data where each cell contains an array if data (e.g., EEG
    or MEG data)


managed by

    * Dataset


Effect types
------------

These are elementary effects in a Model, and identified by :func:`is_effect`

 - _Effect
   - Factor
   - Interaction
   - NestedEffect
 - Var
 - NonBasicEffect


Names
-----

Data-object names seem to have conflicting roles depending on how they are used.
Potentially, data-object names could fulfill the following functions::

1) Names should be effortless to set and update

  - Always update when assigning to dataset
  - With ``name`` argument in methods
  - Any explicit naming should override inherited names

2) Keep track of data source

  - For example labeling a predictor; after normalizing I don't want the
    predictor to be called ``sgram * (srcm / sgram)``

3) Informative labels for plotting and in test results:

  - ``log(5 * x)`` should show up as this, not ``x``
  - Test results need names like ``a - b``

4) Provide term names for models

  - For example, ``m = v + w + v * w`` should not error due to duplicate name

Should operations like ``a + b`` update the name?

 - Requirement 2 is in direct conflict with 3 and 4.
 - Internal operations (``a += b``) should not, because a could be in a dataset.

Current implementation:

 - ``.name`` is always inherited
 - ``.info['longname']`` reflects history



Examples of other potentially desirable API:

```
data['std'] = data['sgram'].std('frequency', name='Standard deviation')
plot.UTSStat('std', data=data)
```

Possible implementation: when ``name`` is explicitly set, ``info['longname']``
is added. Should it survive only one dataset assignment? Would need another
variable like ``autoname``: 0 -> keep but augment; 1 -> replace

```
data['y'] = Var(y, name='Cluster value')
plot.Scatter('y', ..., data=data)  # -> 'Cluster value'
plot.Scatter('y.log()', ..., data=data)  # -> 'log(Cluster value)'

# This situation is ambiguous / could go either way
data['ydiv'] = 1 / data['y']
plot.Scatter('ydiv', ..., ds=ds)  # -> '1 / Cluster value'
# This is probably more conservative:
ds['Realname'] = 1 / ds['y']
plot.Scatter('Realname', ..., ds=ds)  # -> 'Realname'
```
"""
from __future__ import annotations

from copy import copy, deepcopy
import fnmatch
from functools import cached_property, partial
from itertools import chain, combinations, product, repeat, zip_longest
from keyword import iskeyword
from math import ceil, floor, isnan, log
from numbers import Integral, Number
from pathlib import Path
import pickle
import operator
import os
import re
import string
from typing import Any, Callable, Collection, Dict, Iterable, Iterator, List, Literal, Optional, Type, Union, Sequence, Tuple
from warnings import warn

import numpy
from matplotlib.ticker import FixedLocator, Formatter, FormatStrFormatter, FuncFormatter
import mne
from .mne_fixes.legacy_api import label_src_vertno_sel
import nibabel
from nibabel.freesurfer import read_annot, read_geometry
import numpy as np
from numpy import newaxis
from numpy.typing import ArrayLike, DTypeLike
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.signal
import scipy.stats
from scipy.linalg import inv, norm
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform

from . import fmtxt, _info
from ._exceptions import DimensionMismatchError, EvalError, IncompleteModelError
from ._data_opt import gaussian_smoother
from ._text import enumeration
from ._types import PathArg
from ._utils import deprecated, deprecate_kwarg, mne_utils, intervals, ui, n_decimals, natsorted
from ._utils.numpy_utils import (
    INT_TYPES, FULL_SLICE, FULL_AXIS_SLICE,
    aslice, apply_numpy_index, deep_array, digitize_index, digitize_slice_endpoint,
    index_length, index_to_bool_array, index_to_int_array, optimize_index, slice_to_arange)
from .mne_fixes import MNE_EPOCHS, MNE_EVOKED, MNE_RAW, MNE_LABEL
from functools import reduce


preferences = dict(fullrepr=False,  # whether to display full arrays/dicts in __repr__ methods
                   repr_len=5,  # length of repr
                   dataset_str_n_cases=500,
                   var_repr_n_cases=100,
                   factor_repr_n_cases=100,
                   bool_fmt='%s',
                   float_fmt='%.6g',
                   int_fmt='%s',
                   factor_repr_use_labels=True,
                   short_repr=True,  # "A % B" vs "Interaction(A, B)"
                   )


SRC_RE = re.compile(r'^(ico|oct|vol)-(\d+)(?:-(\w+))?$')
UNNAMED = '<?>'
LIST_INDEX_TYPES = (*INT_TYPES, slice)
EXPAND_INDEX_TYPES = (*INT_TYPES, np.ndarray)
_pickled_ds_wildcard = ("Pickled Dataset (*.pickle)", '*.pickle')
_tex_wildcard = ("TeX (*.tex)", '*.tex')
_tsv_wildcard = ("Plain Text Tab Separated Values (*.txt)", '*.txt')
_txt_wildcard = ("Plain Text (*.txt)", '*.txt')
EVAL_CONTEXT = {'numpy': numpy}  # updated at end of file

AxisArg = Union[None, str, Sequence[str], 'NDVar']
DimsArg = Union[str, Sequence[str]]


class IndexFormatter(Formatter):
    "Matplotlib tick-label formatter for categories"
    def __init__(self, labels):
        self.labels = labels
        self.n = len(labels)

    def __call__(self, x, pos=None):
        i = int(round(x))
        if i < 0 or i >= self.n:
            return ''
        else:
            return self.labels[i]


def _effect_eye(n):
    """Effect coding for n categories. E.g.::

    Examples
    --------
    >>> _effect_eye(4)
    array([[ 1,  0,  0],
           [ 0,  1,  0],
           [ 0,  0,  1],
           [-1, -1, -1]])
    """
    x = np.zeros((n, n - 1))
    np.fill_diagonal(x, 1.)
    x[-1] = -1.
    return x


def _effect_interaction(a, b):
    k = a.shape[1]
    out = [a[:, i, None] * b for i in range(k)]
    return np.hstack(out)


def combine_cells(cell_1: 'CellArg', cell_2: 'CellArg') -> Tuple[str, ...]:
    if cell_1 is None:
        return cell_2
    elif cell_2 is None:
        return cell_1
    if isinstance(cell_1, str):
        cell_1 = (cell_1,)
    if isinstance(cell_2, str):
        cell_2 = (cell_2,)
    return *cell_1, *cell_2


def cellname(cell, delim=' '):
    """Consistent ``str`` representation for cells.

    * for Factor cells: the cell (str)
    * for Interaction cell: delim.join(cell).

    """
    if isinstance(cell, str):
        return cell
    elif isinstance(cell, (list, tuple)):
        return delim.join(cell)
    elif cell is None:
        return ''
    else:
        return str(cell)


def longname(x, none=False):
    if isnumeric(x) and 'longname' in x.info:
        return x.info['longname']
    elif hasattr(x, 'name'):
        if x.name is not None:
            return x.name
    elif np.isscalar(x):
        return f'{x:g}'
    elif not none:
        raise TypeError(f"{x!r}")
    return None if none else '<unnamed>'


def get_name(x):
    if hasattr(x, 'name'):
        return x.name
    elif isinstance(x, np.ndarray):
        return None
    else:
        return f'{x:g}'


def op_name(
        obj: object,  # Var, NDVar, scalar
        operand: str = None,
        other: object = None,  # Var, NDVar, scalar; None for unary operations
        info: dict = None,
        name: str = None,  # name parameter, if specified
) -> (Union[str, None], dict):
    """name, info for Var/NDVar operation

    Notes
    -----
    See module level documentation on names.
    """
    if info is None:
        if hasattr(obj, 'info'):
            info = obj.info
        else:
            info = {}

    if name is not None:
        out_name = out_longname = name
    else:
        out_name = get_name(obj)
        if operand is None:
            out_longname = info.get('longname')
        else:
            out_longname = longname(obj, True)
            if out_longname is not None:
                if operand.endswith('('):
                    out_longname = f"{out_longname})"
                elif ' ' in out_longname:
                    out_longname = f"({out_longname})"

                if other is None:  # unary operation
                    out_longname = f"{operand}{out_longname}"
                else:
                    other_longname = longname(other, True)
                    if other_longname is None:
                        out_longname = None
                    else:
                        if ' ' in other_longname:
                            other_longname = f"({other_longname})"
                        out_longname = f"{out_longname} {operand} {other_longname}"

    if out_longname is not None:
        info = {**info, 'longname': out_longname}
    return out_name, info


def nice_label(
        x: Union[Factor, Var, NDVar],
        labels: Dict[CellArg, str] = None,
):
    if labels is not None and x.name in labels:
        return labels[x.name]
    elif 'label' in x.info:
        return x.info['label']
    else:
        return longname(x)


def dataobj_repr(obj: Any, value: bool = False):
    """Describe data-objects as parts of __repr__"""
    if isdataobject(obj):
        if obj.name is not None:
            return obj.name
    elif value and not isinstance(obj, np.ndarray):
        return obj
    return f'<{obj.__class__.__name__}>'


def rank(x, tol=1e-8):
    """Rank of a matrix

    http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html

    """
    s = np.linalg.svd(x, compute_uv=False)
    return np.sum(np.where(s > tol, 1, 0))


def isbalanced(x):
    """Determine whether x is balanced

    Parameters
    ----------
    x : categorial
        Categorial Model, Factor or Interaction.
    """
    if isinstance(x, Model):
        return all(isbalanced(e) for e in x.effects)
    else:
        return len({np.sum(x == c) for c in x.cells}) <= 1


def iscategorial(x):
    "Determine wether x is categorial"
    if isinstance(x, (Factor, NestedEffect)):
        return True
    elif isinstance(x, Interaction):
        return x.is_categorial
    elif isinstance(x, Model):
        return all(iscategorial(e) for e in x.effects)
    else:
        return False


# type checks
#############
# _Effect -> Factor, Interaction, NestedEffect
def isdatacontainer(x: Any) -> bool:
    "Determine whether x is a data-object, including Datasets"
    return isinstance(x, (Datalist, Dataset, Model, NDVar, Var, _Effect, NonbasicEffect))


def isdataobject(x: Any) -> bool:
    "Determine whether x is a data-object, excluding Datasets"
    return isinstance(x, (Datalist, Model, NDVar, Var, _Effect, NonbasicEffect))


def isdatalist(x: Any, contains: Type = None, test_all: bool = True) -> bool:
    """Test whether x is a Datalist instance

    Parameters
    ----------
    x
        Object to test.
    contains
        Test whether the content is instances of a specific class.
    test_all
        If contains is provided, test all items' class (otherwise just test the
        first item).
    """
    is_dl = isinstance(x, Datalist)
    if is_dl and contains:
        if test_all:
            is_dl = all(isinstance(item, contains) for item in x)
        else:
            is_dl = isinstance(x[0], contains)
    return is_dl


def iseffect(x: Any) -> bool:
    return isinstance(x, (Var, _Effect, NonbasicEffect))


def ismodelobject(x: Any) -> bool:
    return isinstance(x, (Model, Var, _Effect, NonbasicEffect))


def isnestedin(item, item2) -> bool:
    "Determine whether ``item`` is nested in ``item2``"
    if isinstance(item, NestedEffect):
        return item2 in find_factors(item.nestedin)
    else:
        return False


def partially_nested(item1, item2) -> bool:
    """Determine whether there is a complete or partial nesting relationship

    Used to determine whether a model should include an interaction effect
    between item1 and item2.
    """
    if isinstance(item2, NestedEffect):
        if isinstance(item1, NestedEffect):
            raise NotImplementedError(
                "Interaction between two nested effects is not implemented. "
                "Please specify model explicitly")
        return partially_nested(item2, item1)
    elif isinstance(item1, NestedEffect):
        nestedin = find_factors(item1.nestedin)
        return any(e in nestedin for e in find_factors(item2))
    else:
        return False


def isnumeric(x: Any) -> bool:
    "Determine wether x is numeric (a Var or an NDVar)"
    return isinstance(x, (NDVar, Var))


def isuv(x: Any, interaction: bool = False) -> bool:
    "Determine whether x is univariate (a Var or a Factor)"
    if interaction:
        return isinstance(x, (Factor, Var, Interaction))
    else:
        return isinstance(x, (Factor, Var))


def isboolvar(x: Any) -> bool:
    "Determine whether x is a Var whose data type is boolean"
    return isinstance(x, Var) and x.x.dtype.kind == 'b'


def isintvar(x: Any) -> bool:
    "Determine whether x is a Var whose data type is integer"
    return isinstance(x, Var) and x.x.dtype.kind in 'iu'


def is_higher_order_effect(e1, e0):
    """Determine whether e1 is a higher order term of e0

    Return True if e1 is a higher order term of e0 (i.e., if all factors in
    e0 are contained in e1).

    Parameters
    ----------
    e1, e0 : effects
        The effects to compare.
    """
    f1s = find_factors(e1)
    return all(f in f1s for f in find_factors(e0))


def assert_has_no_empty_cells(x):
    """Raise a ValueError iff a categorial has one or more empty cells"""
    if isinstance(x, Factor):
        return
    elif isinstance(x, Interaction):
        if not x.is_categorial:
            return
        if x._empty_cells:
            raise NotImplementedError(f"{dataobj_repr(x)} contains empty cells: {enumeration(x._empty_cells)}")
    elif isinstance(x, Model):
        empty = []
        for e in x.effects:
            if isinstance(e, Interaction) and e.is_categorial:
                if e._empty_cells:
                    empty.append((dataobj_repr(e), ', '.join(map(str, e._empty_cells))))
        if empty:
            items = ['%s (%s)' % pair for pair in empty]
            raise NotImplementedError(f"{dataobj_repr(x)} contains empty cells in {enumeration(items)}")
    else:
        raise TypeError(f"Need categorial, got {x!r}")


def hasrandom(x):
    """True if x is or contains a random effect, False otherwise"""
    if isinstance(x, (Factor, NestedEffect)):
        return x.random
    elif isinstance(x, Interaction):
        for e in x.base:
            if isinstance(e, Factor) and e.random:
                return True
    elif isinstance(x, Model):
        return any(hasrandom(e) for e in x.effects)
    return False


def as_case_identifier(
        x: Union[str, Var, Factor, Interaction],
        data: Dataset = None,
        allow_nonunique: bool = False,
):
    "Coerce input to a variable that can identify each of its cases"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Parameter was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if not isinstance(x, (Var, Factor, Interaction)):
        raise TypeError(f"Need a Var, Factor or Interaction to identify cases, got {x!r}")

    if not allow_nonunique:
        if isinstance(x, Var):
            n = len(x.values)
        elif isinstance(x, Factor):
            n = x.n_cells
        elif isinstance(x, Interaction):
            n = len(set(x))
        else:
            raise RuntimeError

        if n < len(x):
            raise ValueError(f"Variable can not serve as a case identifier because it has at least one non-unique value: {x!r}")

    return x


def asarray(x, kind=None, sub=None, data=None, n=None, return_n=False) -> numpy.ndarray:
    "Coerce input to array"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r} Array parameter was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isinstance(x, Var):
        x = x.x
    else:
        x = np.asarray(x)

    if kind is not None and x.dtype.kind not in kind:
        # boolean->int conversion
        if 'i' in kind and x.dtype.kind == 'b':
            x = x.astype(int)
        else:
            raise TypeError(f"Expected array of kind {kind!r}, got {x.dtype.kind!r} ({x.dtype})")

    return _apply_sub(x, sub, n, return_n)


def _apply_sub(x, sub, n, return_n):
    if n is None:
        if return_n:
            n = len(x)
    elif len(x) != n:
        raise ValueError("Arguments have different length")
    if sub is not None:
        x = x[sub]
    return (x, n) if return_n else x


def ascategorial(x, sub=None, data=None, n=None, return_n=False):
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Parameter was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if iscategorial(x):
        pass
    elif isinstance(x, Interaction):
        x = Interaction([e if isinstance(e, Factor) else e.as_factor() for
                         e in x.base])
    else:
        x = asfactor(x)

    return _apply_sub(x, sub, n, return_n)


def asdataobject(x, sub=None, data=None, n=None, return_n=False):
    "Convert to any data object or numpy array."
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Data object was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isdataobject(x):
        pass
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = Datalist(x)

    return _apply_sub(x, sub, n, return_n)


def asepochs(x, sub=None, data=None, n=None, return_n=False) -> mne.BaseEpochs:
    "Convert to mne Epochs object"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Epochs object was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isinstance(x, MNE_EPOCHS):
        pass
    else:
        raise TypeError(f"Need mne Epochs object, got {x!r}")

    return _apply_sub(x, sub, n, return_n)


def asfactor(x, sub=None, data=None, n=None, return_n=False) -> Factor:
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Factor was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isinstance(x, Factor):
        pass
    elif hasattr(x, 'as_factor'):
        x = x.as_factor(name=x.name)
    else:
        x = Factor(x)

    return _apply_sub(x, sub, n, return_n)


def asindex(x) -> numpy.ndarray:
    if isinstance(x, Factor):
        return x != ''
    elif isinstance(x, Var):
        return x.x
    else:
        return x


def asmodel(x, sub=None, data=None, n=None, return_n=False, require_names=False) -> Model:
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Model was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isinstance(x, Model):
        pass
    else:
        x = Model(x)

    if require_names and any(e.name is None for e in x.effects):
        raise ValueError(f"{x}: All relevant effects need to be named")

    return _apply_sub(x, sub, n, return_n)


def asndvar(
        x: NDVarArg,
        sub: IndexArg = None,
        data: Dataset = None,
        n: int = None,
        dtype: np.dtype = None,
        return_n: bool = False,
        ragged: bool = False,
) -> NDVar:
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Ndvar was specified as string, but no Dataset was specified")
        x = data.eval(x)

    # convert MNE objects
    is_ragged = False
    if isinstance(x, NDVar):
        pass
    elif isinstance(x, MNE_EPOCHS):
        from .load.mne import epochs_ndvar
        x = epochs_ndvar(x)
    elif isinstance(x, MNE_EVOKED):
        from .load.mne import evoked_ndvar
        x = evoked_ndvar(x)
    elif isinstance(x, MNE_RAW):
        from .load.mne import raw_ndvar
        x = raw_ndvar(x)
    elif isinstance(x, list):
        if isinstance(x[0], MNE_EVOKED):
            from .load.mne import evoked_ndvar
            x = evoked_ndvar(x)
        else:
            x_list = [asndvar(xi) for xi in x]
            if ragged:
                if dtype is not None:
                    x_list = [xi if xi.x.dtype == dtype else xi.astype(dtype) for xi in x_list]
                if isinstance(x, Datalist):
                    x = Datalist(x_list, x.name, x._fmt)
                else:
                    x = Datalist(x_list, x_list[0].name)
                is_ragged = True
            else:
                x = combine(x_list)
    elif hasattr(x, '_default_plot_obj'):
        x = x._default_plot_obj()
    else:
        raise TypeError(f"NDVar required, got {x!r}")

    x, n = _apply_sub(x, sub, n, return_n=True)
    if dtype is not None and not is_ragged and x.x.dtype != dtype:
        x = x.astype(dtype)
    return (x, n) if return_n else x


def asnumeric(x, sub=None, data=None, n=None, return_n=False, array=False):
    "Var, NDVar"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Numeric argument was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isnumeric(x):
        pass
    elif array:
        x = np.asarray(x)
    else:
        raise TypeError(f"Numeric argument required (Var or NDVar), got {x!r}")

    return _apply_sub(x, sub, n, return_n)


def assub(sub, data=None, return_n=False):
    "Interpret the sub argument."
    if sub is None:
        return (None, None) if return_n else None
    elif isinstance(sub, str):
        if data is None:
            raise TypeError(f"{sub=}: parameter was specified as string, but no Dataset was specified")
        sub = data.eval(sub)

    if isinstance(sub, Var):
        sub = sub.x
    elif not isinstance(sub, np.ndarray):
        raise TypeError(f"{sub=}: needs to be Var or array")

    if return_n:
        n = len(sub) if sub.dtype.kind == 'b' else None
        return sub, n
    else:
        return sub


def asuv(x, sub=None, data=None, n=None, return_n=False, interaction=False):
    "Coerce to Var or Factor"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Parameter was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if isuv(x, interaction):
        pass
    elif all(isinstance(v, str) for v in x):
        x = Factor(x)
    else:
        x = Var(x)

    return _apply_sub(x, sub, n, return_n)


def asvar(x, sub=None, data=None, n=None, return_n=False) -> Var:
    "Coerce to Var"
    if isinstance(x, str):
        if data is None:
            raise TypeError(f"{x!r}: Var was specified as string, but no Dataset was specified")
        x = data.eval(x)

    if not isinstance(x, Var):
        x = Var(x)

    return _apply_sub(x, sub, n, return_n)


def index_ndim(index):
    """Determine the dimensionality of an index

    Parameters
    ----------
    index : numpy_index
        Any valid numpy index.

    Returns
    -------
    ndim : int
        Number of index dimensions: 0 for an index to a single element, 1 for
        an index to a sequence.
    """
    if isinstance(index, slice):
        return 1
    elif isinstance(index, Integral):
        return 0
    elif np.iterable(index):
        return 1
    else:
        raise TypeError(f"{index!r}: unknown index type")


def _empty_like(obj, n=None, name=None):
    "Create an empty object of the same type as obj"
    n = n or len(obj)
    name = name or obj.name
    if isinstance(obj, Factor):
        return Factor([''], repeat=n, name=name)
    elif isinstance(obj, Var):
        return Var(np.empty(n) * np.nan, name)
    elif isinstance(obj, NDVar):
        shape = (n,) + obj.shape[1:]
        return NDVar(np.empty(shape) * np.nan, obj.dims, name)
    elif isdatalist(obj):
        return Datalist([None] * n, name, obj._fmt)
    else:
        raise TypeError(f"{type(obj)}: Type not supported")


def all_equal(a, b, nan_equal=False):
    """Test two data-objects for equality

    Equivalent to ``numpy.all(a == b)`` but faster and capable of treating nans
    as equal.

    Paramaters
    ----------
    a : Var | Factor | NDVar
        Variable to compare.
    a : Var | Factor | NDVar
        Variable to compare.
    nan_equal : bool
        Treat ``nan == nan`` as ``True``.

    Returns
    -------
    all_equal : bool
        True if all entries in
    """
    if a.__class__ is not b.__class__:
        raise TypeError(f"Comparing {a.__class__} with {b.__class__}")
    elif len(a) != len(b):
        raise ValueError(f"a and b have different lengths ({len(a)} vs {len(b)})")
    elif isinstance(a, Factor):
        if a._codes == b._codes:
            return np.array_equal(a.x, b.x)
        else:
            return np.all(a == b)
    elif isinstance(a, (Var, NDVar)):
        if not nan_equal or a.x.dtype.kind in 'ib':  # can't be nan
            return np.array_equal(a.x, b.x)
        else:
            mask = np.isnan(a.x)
            buf = np.isnan(b.x)
            mask &= buf
            np.equal(a.x, b.x, buf)
            buf |= mask
            return buf.all()
    else:
        raise TypeError(f"Comparison for {a.__class__} is not implemented")


# --- sorting ---

def align(d1, d2, i1='index', i2=None, out='data'):
    """Align two data-objects based on index variables

    Before aligning, two data-objects ``d1`` and ``d2`` describe the same cases,
    but their order does not correspond. :func:`align` uses the index variables
    ``i1`` and ``i2`` to match each case in ``d2`` to a case in ``d1`` (i.e.,
    ``d1`` is used as the basis for the case order in the output), and returns
    reordered versions of of ``d1`` and ``d2`` with matching cases. Cases that
    are present in only one of ``d1`` and ``d2`` are dropped.

    Parameters
    ----------
    d1, d2 : data-object
        Two data objects which are to be aligned
    i1, i2 : str | Var | Factor | Interaction
        Indexes for cases in d1 and d2. If d1 and d2 are Datasets, i1 and i2
        can be keys for variables in d1 and d2.  If i2 is identical to i1 it can
        be omitted. Indexes have to supply a unique value for each case.
    out : 'data' | 'index'
        **'data'**: returns the two aligned data objects. **'index'**: returns
        two indices index1 and index2 which can be used to align the datasets
        with ``d1[index1]; d2[index2]``.

    Returns
    -------
    d1_aligned : data-object | array
        Aligned copy of ``d1`` (or index to align ``d1`` if ``out='index'``).
    d2_aligned : data-object | array
        Aligned copy of ``d2`` (or index to align ``d2`` if ``out='index'``).

    See Also
    --------
    align1 : Align one data-object to an index variable

    Examples
    --------
    See :ref:`exa-align` example.
    """
    if i2 is None and isinstance(i1, str):
        i2 = i1
    i1 = as_case_identifier(i1, data=d1, allow_nonunique=True)
    i2 = as_case_identifier(i2, data=d2)
    if type(i1) is not type(i2):
        raise TypeError(f"i1 and i2 need to be of the same type, got:\n{i1=}\n{i2=}")

    idx1 = []
    idx2 = []
    for i, case_id in enumerate(i1):
        if case_id in i2:
            idx1.append(i)
            idx2.append(i2.index(case_id)[0])

    idx1 = optimize_index(idx1)
    idx2 = optimize_index(idx2)
    if out == 'data':
        return d1[idx1], d2[idx2]
    elif out == 'index':
        return idx1, idx2
    else:
        raise ValueError(f"{out=}")


def align1(d, to, by='index', out='data'):
    """Align a data object to an index variable

    Parameters
    ----------
    d : data-object
        Data object with cases that should be aligned to ``idx``.
    to : data-object
        Index array to which ``d`` should be aligned. If ``to`` is a
        :class:`Dataset`, use ``to[by]``.
    by : str | data-object
        Variable labeling cases in ``d`` for aligning them to ``to``. If ``d``
        is a :class:`Dataset`, ``by`` can be the name of a variable in ``d``.
    out : 'data' | 'index'
        Return a restructured copy of ``d`` (default) or an index array into
        ``d``.

    Returns
    -------
    d_aligned : data-object | array
        Aligned copy of ``d`` (or index to align ``d`` to ``idx`` if
        ``out='index'``).

    See Also
    --------
    align : Align two data-objects
    """
    if isinstance(to, Dataset):
        if not isinstance(by, str):
            raise TypeError(f"by={by}: needs to be a str if to is a Dataset")
        to = asuv(by, data=to, interaction=True)
    else:
        to = asuv(to, interaction=True)
    if not isinstance(by, str):
        # check d_idx length
        if isinstance(d, Dataset):
            if len(by) != d.n_cases:
                raise ValueError(f"by={by}: does not have the same number of cases as d (by: {len(by)}, d: {d.n_cases})")
        elif len(by) != len(d):
            raise ValueError(f"by={by}: does not have the same number of cases as d (d_idx: {len(by)}, d: {len(d)})")
    by = asuv(by, data=d, interaction=True)

    align_idx = np.empty(len(to), int)
    for i, v in enumerate(to):
        where = by.index(v)
        if len(where) == 1:
            align_idx[i] = where[0]
        elif len(where) == 0:
            raise ValueError(f"{v} does not occur in d_idx")
        else:
            raise ValueError(f"{v} occurs more than once in d_idx")

    if out == 'data':
        return d[align_idx]
    elif out == 'index':
        return align_idx
    else:
        raise ValueError(f"out={out!r}")


def choose(choice, sources, name=None):
    """Combine data-objects picking from a different object for each case

    Parameters
    ----------
    choice : array of int
        Array specifying for each case from which of the sources the data should
        be taken.
    sources : list of data-objects
        Data that should be combined.
    name : str
        Name for the new data-object (optional).

    Notes
    -----
    Analogous to :func:`numpy.choose`. Only implemented for NDVars at this time.
    """
    choice = asarray(choice, 'i')
    if choice.min() < 0:
        raise ValueError("Choice can not be < 0")
    elif choice.max() > len(sources) - 1:
        raise ValueError("choice contains values exceeding the number of sources")

    s0 = sources[0]
    s1 = sources[1:]
    if isinstance(s0, NDVar):
        if not all(isinstance(s, NDVar) for s in s1):
            raise TypeError("Sources have different types")
        elif any(s.dims != s0.dims for s in s1):
            raise DimensionMismatchError("Sources have different dimensions")
        x = np.empty_like(s0.x)
        index_flat = np.empty(len(choice), bool)
        index = index_flat.reshape(index_flat.shape + (1,) * (x.ndim - 1))
        for i, s in enumerate(sources):
            np.equal(choice, i, index_flat)
            np.copyto(x, s.x, where=index)
        return NDVar(x, s0.dims, name, {})
    else:
        raise NotImplementedError


def shuffled_index(n, cells=None):
    """Return an index to shuffle a data-object

    Parameters
    ----------
    n : int
        Number of cases in the index.
    cells : categorial
        Only shuffle cases within cells.

    Returns
    -------
    index : array of int
        Array with in indexes for shuffling a data-object.

    Notes
    -----
    If ``cells`` is not specified, this is equivalent to
    ``numpy.random.permutation(n)``.
    """
    if cells is None:
        return np.random.permutation(n)
    cells = ascategorial(cells, n=n)
    out = np.arange(n)
    for cell in cells.cells:
        index = cells == cell
        out[index] = np.random.permutation(out[index])
    return out


def combine(
        items: Iterable,
        name: str = None,
        check_dims: bool = True,
        incomplete: Literal['raise', 'drop', 'fill in'] = 'raise',
        dim_intersection: bool = False,
        to_list: bool = False,
):
    """Combine a list of items of the same type into one item.

    Parameters
    ----------
    items
        Sequence of data objects to combine (Dataset, Var, Factor,
        NDVar or Datalist). A sequence of numbers is converted to :class:`Var`,
        a sequence of strings is converted to :class:`Factor`.
    name
        Name for the resulting data-object. If None, the name of the combined
        item is the common prefix of all items.
    check_dims
        For :class:`NDVar` columns, check dimensions for consistency between
        items (e.g., channel locations in a :class:`Sensor`). Default is
        ``True``. Set to ``False`` to ignore mismatches.
    incomplete
        Only applies when combining Datasets: how to handle variables that are
        missing from some of the input Datasets. With ``"raise"`` (default),
        raise a :exc:`ValueError`. With ``"drop"``, drop partially missing
        variables. With ``"fill in"``, retain partially missing variables and
        fill in missing values with empty values (``""`` for factors, ``NaN``
        for numerical variables).
    dim_intersection
        Only applies to combining :class:`NDVar`: normally, when :class:`NDVar`
        have mismatching dimensions, a :exc:`DimensionMismatchError` is raised.
        With ``dim_intersection=True``, the intersection is used instead.
    to_list
        Only applies to combining :class:`NDVar`: normally, when :class:`NDVar`
        have mismatching dimensions, a :exc:`DimensionMismatchError` is raised.
        With ``to_list=True``, the :class:`NDVar` are added as :class:`list` of
        :class:`NDVar` instead.

    Notes
    -----
    The info dict inherits only entries that are equal (``x is y or
    np.array_equal(x, y)``) for all items.
    If ``len(items)`` is 1, ``items[0]`` is returned directly.
    """
    if not isinstance(incomplete, str):
        raise TypeError(f"{incomplete=}, need str")
    elif incomplete not in ('raise', 'drop', 'fill in'):
        raise ValueError(f"{incomplete=}")

    # check input
    if not isinstance(items, Sequence):
        items = list(items)
    if len(items) == 0:
        raise ValueError(f"{items=}: combine() called with empty sequence")

    # find type
    first_item = items[0]
    if isinstance(first_item, Number):
        return Var(items, name)
    elif isinstance(first_item, str):
        return Factor(items)
    stype = type(first_item)
    if isinstance(first_item, mne.BaseEpochs):
        return mne.concatenate_epochs(items)
    elif not isdatacontainer(first_item):
        return Datalist(items)
    elif any(type(item) is not stype for item in items[1:]):
        raise TypeError(f"All items to be combined need to have the same type, got {enumeration({type(i) for i in items})}.")

    # find name
    if name is None:
        names = [item.name for item in items if item.name is not None]
        name = os.path.commonprefix(names) or None

    # combine objects
    if stype is Dataset:
        out = Dataset(name=name, info=_info.merge_info(items))
        if incomplete == 'fill in':
            # find all keys and data types
            keys = list(first_item.keys())
            sample = dict(first_item)
            for item in items:
                for key in item:
                    if key not in keys:
                        keys.append(key)
                        sample[key] = item[key]
            # create output
            for key in keys:
                pieces = [ds[key] if key in ds else
                          _empty_like(sample[key], ds.n_cases) for ds in items]
                out[key] = combine(pieces, check_dims=check_dims, dim_intersection=dim_intersection, to_list=to_list)
        else:
            keys = set(first_item)
            if incomplete == 'raise':
                other_keys = [set(item) for item in items[1:]]
                if any(keys_i != keys for keys_i in other_keys):
                    info = '\n'.join([f"{i:<2}: {', '.join(sorted(keys_i))}" for i, keys_i in enumerate([keys, *other_keys])])
                    raise ValueError(f"Datasets have unequal keys. Use with incomplete='drop' or incomplete='fill in' to combine anyways. Keys present:\n{info}")
                out_keys = first_item
            else:
                keys.intersection_update(*items[1:])
                out_keys = (k for k in first_item if k in keys)

            for key in out_keys:
                out[key] = combine([ds[key] for ds in items], check_dims=check_dims, dim_intersection=dim_intersection, to_list=to_list)
        return out
    elif stype is Var:
        x = np.hstack([i.x for i in items])
        return Var(x, name, _info.merge_info(items))
    elif stype is Factor:
        random = {f.random for f in items}
        if len(random) > 1:
            raise ValueError("Factors have different values for random parameter")
        random = random.pop()
        labels = first_item._labels
        if all(f._labels == labels for f in items[1:]):
            x = np.hstack([f.x for f in items])
            return Factor(x, name, random, labels=labels)
        else:
            x = sum((i.as_labels() for i in items), [])
            return Factor(x, name, random)
    elif stype is NDVar:
        v_have_case = [v.has_case for v in items]
        if all(v_have_case):
            has_case = True
            all_dims = [item.dims[1:] for item in items]
        elif any(v_have_case):
            raise DimensionMismatchError(f"{name}: Some items have a Case dimension, others do not")
        else:
            has_case = False
            all_dims = [item.dims for item in items]

        dims, *other_dims = all_dims
        if all(dims_stackable(dims_i, dims, check_dims) for dims_i in other_dims):
            sub_items = items
        elif to_list:
            if has_case:
                return list(chain.from_iterable(items))
            else:
                return items
        elif dim_intersection:
            dims = reduce(lambda x, y: intersect_dims(x, y, check_dims), all_dims)
            idx = {dim.name: dim for dim in dims}
            # reduce data to common dimension range
            sub_items = []
            for item in items:
                if item.dims[has_case:] == dims:
                    sub_items.append(item)
                else:
                    sub_items.append(item.sub(**idx))
        elif check_dims and all(dims_stackable(dims_i, dims, True) for dims_i in other_dims):
            raise DimensionMismatchError.from_dims_list("Some NDVars have mismatching dimensions; set check_dims=False to ignore non-critical differences (e.g. adjacency)", all_dims, check_dims)
        else:
            raise DimensionMismatchError.from_dims_list("Some NDVars have mismatching dimensions; use to_list=True to combine them into a list, or dim_intersection=True to discard elements not present in all", all_dims, check_dims)
        # combine data
        if has_case:
            x = np.concatenate([v.x for v in sub_items], axis=0)
        else:
            x = np.stack([v.x for v in sub_items])
        return NDVar(x, (Case, *dims), name, _info.merge_info(sub_items))
    elif stype is Datalist:
        return Datalist(sum(items, []), name, items[0]._fmt)
    else:
        raise RuntimeError("combine with stype = %r" % stype)


def find_factors(obj):
    "Return the list of all factors contained in obj"
    if isinstance(obj, EffectList):
        f = {id(f): f for e in obj for f in find_factors(e)}
        return EffectList(f.values())
    elif isuv(obj):
        return EffectList([obj])
    elif isinstance(obj, Model):
        f = {id(f): f for e in obj.effects for f in find_factors(e)}
        return EffectList(f.values())
    elif isinstance(obj, NestedEffect):
        return find_factors(obj.effect)
    elif isinstance(obj, Interaction):
        return obj.base
    else:  # NonbasicEffect
        try:
            return EffectList(obj.factors)
        except BaseException:
            raise TypeError("%r has no factors" % obj)


class EffectList(list):
    def __repr__(self):
        return 'EffectList((%s))' % ', '.join(self.names())

    def __contains__(self, item):
        for f in self:
            if ((f.name == item.name) and (type(f) is type(item)) and (len(f) == len(item)) and np.all(item == f)):
                return True
        return False

    def index(self, item):
        for i, f in enumerate(self):
            if (len(f) == len(item)) and np.all(item == f):
                return i
        raise ValueError("Factor %r not in EffectList" % item.name)

    def names(self):
        names = [e.name if isuv(e) else repr(e) for e in self]
        return [UNNAMED if n is None else n for n in names]


class Named:

    def __init__(
            self,
            name: str = None,
            info: Dict = None,
    ):
        self.info = {} if info is None else dict(info)
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self.info.pop('longname', None)
        self._name = name


class Var(Named):
    """Container for scalar data.

    Parameters
    ----------
    x
        Data; is converted with ``np.asarray(x)``. Multidimensional arrays
        are flattened as long as only 1 dimension is longer than 1.
    name
        Name of the variable
    info
        Info dictionary. The "longname" entry is used for display purposes.
    repeat
        repeat each element in ``x``, either a constant or a different number
        for each element.
    tile
        Repeat ``x`` as a whole ``tile`` many times.

    Attributes
    ----------
    x : numpy.ndarray
        The data stored in the Var.
    name : None | str
        The Var's name.

    Notes
    -----
    While :py:class:`Var` objects support a few basic operations in a
    :py:mod:`numpy`-like fashion (``+``, ``-``, ``*``, ``/``, ``//``), their
    :py:attr:`Var.x` attribute provides access to the corresponding
    :py:class:`numpy.array` which can be used for anything more complicated.
    :py:attr:`Var.x` can be read and modified, but should not be replaced.
    """
    df = 1
    ndim = 1
    random = False

    def __init__(
            self,
            x: ArrayLike,
            name: str = None,
            info: dict = None,
            repeat: Union[ArrayLike, int] = 1,
            tile: int = 1,
    ):
        if isinstance(x, str):
            raise TypeError("Var can't be initialized with a string")
        if isinstance(info, int):
            raise RuntimeError("Var argument order has changed; please update your code")

        if isinstance(x, Iterator):
            x = list(x)
        x = np.asarray(x)
        if x.dtype.kind in 'OUSV':
            alt_type = 'Factor' if x.dtype.kind == 'S' else 'Datalist'
            type_name = np.typename(x.dtype.char)
            raise TypeError(f"x of type {type_name} (dtype {x.dtype!r}): Var needs numerical data type. Consider using a {alt_type} instead.")
        elif x.ndim > 1:
            if sum(i > 1 for i in x.shape) <= 1:
                x = np.ravel(x)
            else:
                raise ValueError(f"x with shape {x.shape}; x needs to be one-dimensional. Use NDVar class for data with more than one dimension.")

        if not (isinstance(repeat, Integral) and repeat == 1):
            x = np.repeat(x, repeat)

        if tile > 1:
            x = np.tile(x, tile)

        Named.__init__(self, name, info)
        self.x = x

    def __setstate__(self, state):
        if len(state) == 3:
            x, name, info = state
        else:
            x, name = state
            info = {}
        self.x = x
        self._name = name
        self.info = info

    def __getstate__(self):
        return self.x, self._name, self.info

    def __repr__(self, full=False):
        n_cases = preferences['var_repr_n_cases']

        if isintvar(self):
            fmt = preferences['int_fmt']
        elif isboolvar(self):
            fmt = preferences['bool_fmt']
        else:
            fmt = preferences['float_fmt']

        if full or len(self.x) <= n_cases:
            x = [fmt % v for v in self.x]
        else:
            x = [fmt % v for v in self.x[:n_cases]]
            x.append('... (N=%s)' % len(self.x))

        args = ['[%s]' % ', '.join(x)]
        if self.name is not None:
            args.append('name=%r' % self.name)

        if self.info:
            args.append('info=%r' % self.info)

        return "Var(%s)" % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

    __array_priority__ = 15

    @property
    def __array_interface__(self):
        return self.x.__array_interface__

    # container ---
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        index = asindex(index)
        x = self.x[index]
        if isinstance(x, np.ndarray):
            return Var(x, self.name, self.info)
        else:
            return x

    def __setitem__(self, index, value):
        self.x[index] = value

    def __contains__(self, value):
        return value in self.x

    # numeric ---
    def __bool__(self):
        raise TypeError("The truth value of a Var is ambiguous. Use v.any() or v.all()")

    def __neg__(self):
        return Var(-self.x, *op_name(self, '-'))

    def __pos__(self):
        return self

    def __abs__(self):
        return self.abs()

    def __invert__(self):
        return Var(~self.x, *op_name(self, '~'))

    def _arg_x(self, other):
        if isnumeric(other):
            return other.x
        else:
            return other

    def __add__(self, other):
        if ismodelobject(other):
            # ??? should Var + Var return sum or Model?
            return Model((self, other))
        args = op_name(self, '+', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self + x_other, dims, *args)
        x = self.x + self._arg_x(other)
        return Var(x, *args)

    def __iadd__(self, other):
        self.x += self._arg_x(other)
        return self

    def __radd__(self, other):
        x = self._arg_x(other) + self.x
        return Var(x, *op_name(other, '+', self))

    def __sub__(self, other):
        "subtract: values are assumed to be ordered. Otherwise use .sub method."
        args = op_name(self, '-', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self - x_other, dims, *args)
        x = self.x - self._arg_x(other)
        return Var(x, *args)

    def __isub__(self, other):
        self.x -= self._arg_x(other)
        return self

    def __rsub__(self, other):
        x = self._arg_x(other) - self.x
        return Var(x, *op_name(other, '-', self))

    def __mul__(self, other):
        if isinstance(other, Model):
            return Model((self,)) * other
        elif iscategorial(other):
            return Model((self, other, self % other))
        args = op_name(self, '*', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self * x_other, dims, *args)
        x = self.x * self._arg_x(other)
        return Var(x, *args)

    def __imul__(self, other):
        self.x *= self._arg_x(other)
        return self

    def __rmul__(self, other):
        x = self._arg_x(other) * self.x
        return Var(x, *op_name(other, '*', self))

    def __floordiv__(self, other):
        args = op_name(self, '//', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self // x_other, dims, *args)
        x = self.x // self._arg_x(other)
        return Var(x, *args)

    def __ifloordiv__(self, other):
        self.x //= self._arg_x(other)
        return self

    def __mod__(self, other):
        if isinstance(other, Model):
            return Model(self) % other
        elif isinstance(other, Var):
            pass
        elif ismodelobject(other):
            return Interaction((self, other))
        args = op_name(self, '%', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self % x_other, dims, *args)
        x = self.x % self._arg_x(other)
        return Var(x, *args)

    def __imod__(self, other):
        self.x %= self._arg_x(other)
        return self

    def __lt__(self, y):
        return self.x < self._arg_x(y)

    def __le__(self, y):
        return self.x <= self._arg_x(y)

    def __eq__(self, y):
        return self.x == self._arg_x(y)

    def __ne__(self, y):
        return self.x != self._arg_x(y)

    def __gt__(self, y):
        return self.x > self._arg_x(y)

    def __ge__(self, y):
        return self.x >= self._arg_x(y)

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __div__(self, other):
        if isinstance(other, Var):
            pass
        elif iscategorial(other):  # separate slope for each level (for ANCOVA)
            dummy_factor = other.as_dummy_complete
            codes = dummy_factor * self.as_effects
            # center
            means = codes.sum(0) / dummy_factor.sum(0)
            codes -= dummy_factor * means
            # create effect
            name = '%s per %s' % (self.name, other.name)
            return NonbasicEffect(codes, [self, other], name, beta_labels=other.dummy_complete_labels)
        args = op_name(self, '/', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self / x_other, dims, *args)
        x = self.x / self._arg_x(other)
        return Var(x, *args)

    def __idiv__(self, other):
        self.x /= self._arg_x(other)
        return self

    def __pow__(self, other):
        args = op_name(self, '**', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self ** x_other, dims, *args)
        x = self.x ** self._arg_x(other)
        return Var(x, *args)

    def __ipow__(self, other):
        self.x **= self._arg_x(other)
        return self

    def __and__(self, other):
        args = op_name(self, '&', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self & x_other, dims, *args)
        return Var(self.x & self._arg_x(other), *args)

    def __iand__(self, other):
        self.x &= self._arg_x(other)
        return self

    def __xor__(self, other):
        args = op_name(self, '^', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self ^ x_other, dims, *args)
        return Var(self.x ^ self._arg_x(other), *args)

    def __ixor__(self, other):
        self.x ^= self._arg_x(other)
        return self

    def __or__(self, other):
        args = op_name(self, '|', other)
        if isinstance(other, NDVar):
            dims, x_other, x_self = other._align(self)
            return NDVar(x_self | x_other, dims, *args)
        return Var(self.x | self._arg_x(other), *args)

    def __ior__(self, other):
        self.x |= self._arg_x(other)
        return self

    def __round__(self, n=0):
        return Var(np.round(self.x, n), self.name, self.info)

    def _coefficient_names(self, method):
        return longname(self),

    def _summary(self, width=80):
        is_nan = np.isnan(self.x)
        n_nan = np.sum(is_nan)
        nan_str = f'NaN:{n_nan}' if n_nan else ''
        x = self.x[~is_nan] if n_nan else self.x
        unique = np.unique(x)
        if len(unique) <= 10:
            ns = [(v, np.sum(self.x == v)) for v in unique]
            items = [f'{v:g}:{n}' if n > 1 else f'{v:g}' for v, n in ns]
            if nan_str:
                items.append(nan_str)
            # check whether all items fit
            n = -2
            for i, item in enumerate(items):
                n += 2 + len(item)
                if n >= width:
                    break
            else:
                return ', '.join(items)
        out = f'{x.min():g} - {x.max():g}'
        if nan_str:
            out += f', {nan_str}'
        return out

    def abs(self, name=None):
        "Return a Var with the absolute value."
        return Var(np.abs(self.x), *op_name(self, 'abs(', name=name))

    def argmax(self):
        """Index of the largest value

        See also
        --------
        .max
        """
        return np.argmax(self.x)

    def argmin(self):
        """Index of the smallest value

        See Also
        --------
        .min
        """
        return np.argmin(self.x)

    def argsort(self, kind='quicksort'):
        """:func:`numpy.argsort`

        Parameters
        ----------
        kind : 'quicksort' | 'mergesort' | 'heapsort'
            Sorting algorithm (default 'quicksort').

        Returns
        -------
        index_array : array of int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.
        """
        return np.argsort(self.x, kind=kind)

    def astype(self, dtype):
        """Copy of the Var with data cast to the specified type

        Parameters
        ----------
        dtype : numpy dtype
            Numpy data-type specification (see :meth:`numpy.ndarray.astype`).
        """
        return Var(self.x.astype(dtype), self.name, self.info)

    @property
    def as_dummy(self):
        "For dummy coding"
        return self.x[:, None]

    @property
    def as_effects(self):
        "For effect coding"
        return self.x[:, None] - self.x.mean()

    def as_factor(self, labels='%s', name=None, random=False):
        """Convert the Var into a Factor

        Parameters
        ----------
        labels : str | dict
            Either a format string for converting values into labels (default:
            ``'%r'``) or a dictionary mapping values to labels (see examples).
            In a dictionary, multiple values can be assigned the same label by
            providing multiple keys in a tuple. A special key 'default' can be
            used to assign values that are not otherwise specified in the
            dictionary (by default this is the empty string ``''``).
        name : str
            Name of the output Factor (default is the current name).
        random : bool
            Whether the Factor is a random Factor (default ``False``).

        Examples
        --------
        >>> v = Var([0, 1, 2, 3])
        >>> v.as_factor()
        Factor(['0', '1', '2', '3'])
        >>> v.as_factor({0: 'a', 1: 'b'})
        Factor(['a', 'b', '', ''])
        >>> v.as_factor({(0, 1): 'a', (2, 3): 'b'})
        Factor(['a', 'a', 'b', 'b'])
        >>> v.as_factor({0: 'a', 1: 'b', 'default': 'c'})
        Factor(['a', 'b', 'c', 'c'])
        """
        labels_ = {}
        if isinstance(labels, dict):
            # flatten
            for key, v in labels.items():
                if isinstance(key, Sequence) and not isinstance(key, str):
                    for k in key:
                        labels_[k] = v
                else:
                    labels_[key] = v

            default = labels_.pop('default', '')
            if default is not None:
                for key in np.unique(self.x):
                    if key not in labels_:
                        labels_[key] = default
        else:
            for value in np.unique(self.x):
                labels_[value] = labels % value

        if name is None or name is True:
            name = self.name

        return Factor(self.x, name, random, labels=labels_)

    def copy(self, name=None):
        "Return a deep copy"
        return Var(self.x.copy(), *op_name(self, name=name))

    def count(self, value: float = None, start: int = -1) -> Var:
        """Count the number of occurrence of each value

        Parameters
        ----------
        value
            Value which is to be counted.
        start
            Value at which to start counting (with the default of -1, the first
            occurrence will be 0).

        Examples
        --------
        >>> v = Var([1, 2, 3, 1, 1, 1, 3])
        >>> v.count()
        Var([0, 0, 0, 1, 2, 3, 1])
        >>> v.count(3)
        Var([0, 0, 1, 1, 1, 1, 2])
        """
        if value is None:
            x = np.empty(len(self.x), int)
            index = np.empty(len(self.x), bool)
            for v in np.unique(self.x):
                np.equal(self.x, v, index)
                x[index] = np.arange(index.sum())
        else:
            x = np.cumsum(self == value) + start
        return Var(x, self.name)

    def aggregate(
            self,
            x: CategorialArg,
            func: Callable = np.mean,
            name: str = None,
    ) -> Var:
        """Summarize cases within cells of ``x``

        Parameters
        ----------
        x
            Model defining cells in which to aggregate.
        func
            Function that converts arrays into scalars, used to summarize data
            within each cell of ``x``.
        name
            Name of the output (default is the current name).

        Returns
        -------
        aggregated_var
            A :class:`Var` instance with a single value for each cell in ``x``.
        """
        if x is None:
            x_out = [func(self.x)]
        elif len(x) != len(self):
            raise ValueError(f"Length mismatch: {len(self)} (Var) != {len(x)} (x)")
        else:
            x_out = [func(self.x[x == cell]) for cell in x.cells]
        x_out = numpy.array(x_out)
        if self.x.dtype.kind in 'bi':
            x_with_dtype = numpy.array(x_out, dtype=self.x.dtype)
            if numpy.all(x_with_dtype == x_out):
                x_out = x_with_dtype
        return Var(x_out, *op_name(self, name=name))

    @property
    def beta_labels(self):
        return [self.name]

    def diff(self, to_end=None, to_begin=None, name=None):
        """The differences between consecutive values

        Parameters
        ----------
        to_end : scalar (optional)
            Append ``to_end`` at the end.
        to_begin : scalar (optional)
            Add ``to_begin`` at the beginning.
        name : str
            Name of the output (default is the current name).

        Returns
        -------
        diff : Var
            Difference.

        Examples
        --------
        >>> v = Var([1, 2, 4])
        >>> v.diff()
        Var([1, 2])
        >>> v.diff(to_begin=-1)
        Var([-1, 1, 2])
        """
        args = op_name(self, 'diff(', name=name)
        if len(self) == 0:
            return Var(np.empty(0), *args)
        return Var(np.ediff1d(self.x, to_end, to_begin), *args)

    # def difference(self, x, v1, v2, match):
    #     """
    #     Subtract x==v2 from x==v1; sorts values according to match (ascending)
    #
    #     Parameters
    #     ----------
    #     x : categorial
    #         Model to define cells.
    #     v1, v2 : str | tuple
    #         Cells on x for subtraction.
    #     match : categorial
    #         Model that defines how to mach cells in v1 to cells in v2.
    #     """
    #     raise NotImplementedError
    #     # FIXME: use celltable
    #     assert isinstance(x, Factor)
    #     I1 = (x == v1);         I2 = (x == v2)
    #     Y1 = self[I1];          Y2 = self[I2]
    #     m1 = match[I1];         m2 = match[I2]
    #     s1 = np.argsort(m1);    s2 = np.argsort(m2)
    #     y = Y1[s1] - Y2[s2]
    #     name = "{n}({x1}-{x2})".format(n=self.name,
    #                                    x1=x.cells[v1],
    #                                    x2=x.cells[v2])
    #     return Var(y, name, info=self.info.copy())

    @classmethod
    def from_dict(cls, base, values, name=None, default=0, info=None):
        """
        Construct a Var object by mapping ``base`` to ``values``.

        Parameters
        ----------
        base : sequence
            Sequence to be mapped to the new Var.
        values : dict
            Mapping from values in base to values in the new Var.
        name : str
            Name for the new Var.
        default : scalar
            Default value to supply for entries in ``base`` that are not in
            ``values``.

        Examples
        --------
        >>> base = Factor('aabbcde')
        >>> Var.from_dict(base, {'a': 5, 'e': 8}, default=0)
        Var([5, 5, 0, 0, 0, 0, 8])
        """
        return cls([values.get(b, default) for b in base], name, info=info)

    @classmethod
    def from_apply(cls, base, func, name=None, info=None):
        """
        Construct a Var instance by applying a function to each value in a base

        Parameters
        ----------
        base : sequence, len = n
            Base for the new Var. Can be an NDVar, if ``func`` is a
            dimensionality reducing function such as :func:`numpy.mean`.
        func : callable
            A function that when applied to each element in ``base`` returns
            the desired value for the resulting Var.
        name : str
            Name for the new Var.
        info : dict
            Info for the new Var.
        """
        if isinstance(base, (Var, NDVar)):
            base = base.x

        if isinstance(func, np.ufunc):
            x = func(base)
        elif getattr(base, 'ndim', 1) > 1:
            x = func(base.reshape((len(base), -1)), axis=1)
        else:
            x = np.array([func(val) for val in base])

        return cls(x, name, info=info)

    def index(self, value):
        """``v.index(value)`` returns an array of indices where v equals value

        Returns
        -------
        index : array of int
            Array of positive :class:`int` indices.
        """
        return np.flatnonzero(self == value)

    def isany(self, *values):
        "Boolean index, True where the Var is equal to one of the values"
        return np.isin(self.x, values)

    def isin(self, values):
        "Boolean index, True where the Var value is in values"
        if isinstance(values, dict):
            values = tuple(values)
        return np.isin(self.x, values)

    def isnan(self):
        "Return boolean :class:`Var` indicating location of ``NaN`` values"
        return Var(np.isnan(self.x), *op_name(self, 'isnan('))

    def isnot(self, *values):
        "Boolean index, True where the Var is not equal to one of the values"
        return np.isin(self.x, values, invert=True)

    def isnotin(self, values):
        "Boolean index, True where the Var value is not in values"
        return np.isin(self.x, values, invert=True)

    def log(self, base=None, name=None):
        """Element-wise log

        Parameters
        ----------
        base : scalar
            Base of the log (default is the natural log).
        name : str
            Name of the output Var (default is the current name).
        """
        if base is None:
            x = np.log(self.x)
        elif base == 2:
            x = np.log2(self.x)
        elif base == 10:
            x = np.log10(self.x)
        else:
            x = np.log(self.x)
            x /= log(base)

        if base is None:
            op = 'log('
        else:
            op = f'log{base:g}('
        return Var(x, *op_name(self, op, name=name))

    def max(self):
        """The largest value

        See Also
        --------
        .argmax
        .min
        """
        return self.x.max()

    def mean(self):
        "The mean"
        return self.x.mean()

    def min(self):
        """The smallest value

        See Also
        --------
        .argmin
        .max
        """
        return self.x.min()

    def repeat(self, repeats, name=None):
        """
        Repeat each element ``repeats`` times

        Parameters
        ----------
        repeats : int | array of int
            Number of repeats, either a constant or a different number for each
            element.
        name : str
            Name of the output Var (default is current name).
        """
        return Var(self.x.repeat(repeats), *op_name(self, name=name))

    def split(
            self,
            n: int = 2,
            labels: Dict[int, str] = None,
            name: str = None,
    ):
        """
        A Factor splitting y into ``n`` categories with equal number of cases

        Parameters
        ----------
        n
            number of categories
        labels
            Dictionary mapping indices to labels; for example
            ``n=2, labels={0: 'low', 1: 'high'}``.
        name
            Name of the output Var (default is current name).

        Examples
        --------
        Use n = 2 for a median split::

            >>> y = Var([1,2,3,4])
            >>> y.split(2)
            Factor(['0', '0', '1', '1'])

            >>> z = Var([7, 6, 5, 4, 3, 2])
            >>> z.split(3)
            Factor(['2', '2', '1', '1', '0', '0'])

        """
        y = self.x

        d = 100. / n
        percentile = np.arange(d, 100., d)
        values = [scipy.stats.scoreatpercentile(y, p) for p in percentile]
        x = np.zeros(len(y), dtype=int)
        for v in values:
            x += y > v
        if name is None:
            name = self.name
        return Factor(x, name, labels=labels)

    def std(self, ddof: float = 0):
        "The standard deviation"
        return self.x.std(ddof=ddof)

    def sort_index(self, descending=False):
        """Create an index that could be used to sort the Var.

        Parameters
        ----------
        descending : bool
            Sort in descending instead of an ascending order.
        """
        idx = np.argsort(self.x, kind='mergesort')
        if descending:
            idx = idx[::-1]
        return idx

    def sum(self):
        "The sum over all values"
        return self.x.sum()

    def tile(self, repeats, name=None):
        """Construct a Var by repeating ``self`` ``repeats`` times

        Parameters
        ----------
        repeats : int
            Number of repeats.
        name : str
            Name of the output Var (default is current name).
        """
        if name is None:
            name = self.name
        return Var(self.x, name, tile=repeats)

    @property
    def values(self):
        return np.unique(self.x)

    def var(self, ddof: float = 0):
        "The variance"
        return self.x.var(ddof=ddof)


class _Effect:

    _n_cases = None

    # numeric ---
    def __bool__(self):
        raise TypeError(f"The truth value of a {self.__class__.__name__} is ambiguous")

    def __add__(self, other):
        return Model(self) + other

    def __mul__(self, other):
        if partially_nested(other, self):
            return Model((self, other))
        return Model((self, other, self % other))

    def __mod__(self, other):
        if isinstance(other, Model):
            return Model((self % e for e in other.effects))
        return Interaction((self, other))

    def __len__(self):
        return self._n_cases

    def as_var(
            self,
            labels: Dict[CellArg, float],
            default: float = None,
            name: str = None,
    ) -> Var:
        """Convert into a Var

        Parameters
        ----------
        labels
            A ``{old_value: new_value}`` mapping.
        default
            Default value for old values not mentioned in ``labels``. If not
            specified, old values missing from ``labels`` will raise a
            ``KeyError``.
        name
            Name of the output Var (default is current name).
        """
        if default is None:
            x = [labels[v] for v in self]
        else:
            x = [labels.get(v, default) for v in self]
        return Var(x, name or self.name)

    def count(self, value: Tuple[str, ...] = None, start: int = -1) -> Var:
        """Cumulative count of the occurrences of ``value``

        Parameters
        ----------
        value
            Cell which is to be counted.
        start
            Value at which to start counting (with the default of -1, the first
            occurrence will be 0).

        Returns
        -------
        count : Var of int,  len = len(self)
            Cumulative count of value in self.

        Examples
        --------
        >>> a = Factor('abcabcccc')
        >>> b = Factor('aaabbbccc')
        >>> i = a % b
        >>> i.count()
        Var([0, 0, 0, 0, 0, 0, 0, 1, 2])
        >>> i.count(('c', 'c'))
        Var([-1, -1, -1, -1, -1, -1, 0, 1, 2])
        """
        if value is None:
            x = np.empty(len(self), int)
            for cell in self.cells:
                index = self == cell
                x[index] = np.arange(index.sum())
        else:
            x = np.cumsum(self == value) + start
        return Var(x)

    def enumerate_cells(self, name=None):
        """Enumerate the occurrence of each cell value throughout the data

        Parameters
        ----------
        name : None | str
            Name for the returned Var.

        Returns
        -------
        enum : Var
            Result.

        Examples
        --------
        >>> f = Factor('aabbccabc')
        >>> f.enumerate_cells()
        Var([0, 1, 0, 1, 0, 1, 2, 2, 2])
        """
        counts = {cell: 0 for cell in self.cells}
        enum = np.empty(len(self), int)
        for i, value in enumerate(self):
            enum[i] = counts[value]
            counts[value] += 1
        return Var(enum, name)

    def index(self, cell):
        """Array with ``int`` indices equal to ``cell``

        Examples
        --------
        >>> f = Factor('abcabcabc')
        >>> f
        Factor(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
        >>> f.index('b')
        array([1, 4, 7])
        >>> f[f.index('b')] = 'new_b'
        >>> f
        Factor(['a', 'new_b', 'c', 'a', 'new_b', 'c', 'a', 'new_b', 'c'])
        """
        return np.flatnonzero(self == cell)

    def index_opt(self, cell):
        """Find an optimized index for a given cell.

        Returns
        -------
        index : slice | array
            If possible, a ``slice`` object is returned. Otherwise, an array
            of indices (as with ``e.index(cell)``).
        """
        index = np.flatnonzero(self == cell)
        d_values = np.unique(np.diff(index))
        if len(d_values) == 1:
            start = index.min() or None
            step = d_values[0]
            stop = index.max() + 1
            if stop > len(self) - step:
                stop = None
            if step == 1:
                step = None
            index = slice(start, stop, step)
        return index

    def sort_index(self, descending=False, order=None):
        """Create an index that could be used to sort this data_object.

        Parameters
        ----------
        descending : bool
            Sort in descending instead of the default ascending order.
        order : None | sequence
            Sequence of cells to define a custom order. Any cells that are not
            present in ``order`` will be omitted in the sort_index, i.e. the
            sort_index will be shorter than its source.

        Returns
        -------
        sort_index : array of int
            Array which can be used to sort a data_object in the desired order.
        """
        idx = np.empty(len(self), dtype=np.intp)
        if order is None:
            cells = self._sorted_cells()
        else:
            cells = order
            idx.fill(-1)

        for i, cell in enumerate(cells):
            idx[self == cell] = i

        sort_idx = np.argsort(idx, kind='mergesort')
        if order is not None:
            excluded = np.count_nonzero(idx == -1)
            if excluded:
                sort_idx = sort_idx[excluded:]

        if descending:
            if not isinstance(descending, bool):
                raise TypeError("descending=%s, need bool" % repr(descending))
            sort_idx = sort_idx[::-1]

        return sort_idx

    def _sorted_cells(self):
        raise NotImplementedError


class Factor(_Effect):
    """Container for categorial data.

    Parameters
    ----------
    x
        Sequence of Factor values (see also the ``labels`` kwarg).
    name
        Name of the Factor.
    random
        Treat Factor as random factor (for ANOVA; default is False).
    repeat
        repeat each element in ``x``, either a constant or a different number
        for each element.
    tile
        Repeat ``x`` as a whole ``tile`` many times.
    labels
        An optional dictionary mapping values as they occur in ``x`` to the
        Factor's cell labels.
    default
        Label to assign values not in ``label`` (by default this is
        ``str(value)``).

    Attributes
    ----------
    .name : None | str
        The Factor's name.
    .cells : tuple of str
        Ordered names of all cells. Order is determined 1) by the order of
        cells in the ``labels`` argument, and 2) for cells that do not occur in
        ``labels`` it is determined by first occurrence in ``x``.
    .random : bool
        Whether the factor represents a random or fixed effect (for ANOVA).

    Examples
    --------
    The most obvious way to initialize a Factor is a list of strings::

        >>> Factor(['in', 'in', 'in', 'out', 'out', 'out'])
        Factor(['in', 'in', 'in', 'out', 'out', 'out'])

    The same can be achieved with a list of integers plus a labels dict::

        >>> Factor([1, 1, 1, 0, 0, 0], labels={1: 'in', 0: 'out'})
        Factor(['in', 'in', 'in', 'out', 'out', 'out'])

    Or more parsimoniously:

        >>> Factor([1, 0], labels={1: 'in', 0: 'out'}, repeat=3)
        Factor(['in', 'in', 'in', 'out', 'out', 'out'])

    Since the Factor initialization simply iterates over the ``x``
    argument, a Factor with one-character codes can also be initialized
    with a single string::

        >>> Factor('iiiooo')
        Factor(['i', 'i', 'i', 'o', 'o', 'o'])
    """
    def __init__(
            self,
            x: Iterable[Any],
            name: str = None,
            random: bool = False,
            repeat: Union[int, Sequence[int]] = 1,
            tile: Union[int, Sequence[int]] = 1,
            labels: Dict[Any, str] = None,
            default: str = None,
    ):
        if isinstance(x, Iterator):
            x = list(x)
        elif isdataobject(x):
            if name is None:
                name = x.name
            if isinstance(x, Var):
                x = x.x
        n_cases = len(x)
        self.name = name
        self.random = random

        if n_cases == 0 or not (np.any(repeat) or np.any(tile)):
            self.x = np.empty(0, np.uint32)
            self._labels = {}
            self._init_secondary()
            return

        # find mapping and ordered values
        labels = {} if labels is None else dict(labels)
        if isinstance(x, Factor):
            # translate label keys to x codes
            labels = {x._codes[src]: dst for src, dst in labels.items() if src in x._codes}
            # add labels that are not in `labels`
            if default is None:  # from x
                labels.update({code: label for code, label in x._labels.items() if code not in labels})
            else:  # use default
                labels.update({code: default for code in x._labels if code not in labels})
            x = x.x

        if isinstance(x, np.ndarray) and x.dtype.kind in 'ifb':
            assert x.ndim == 1
            unique = np.unique(x)
            for v in unique:
                if v not in labels:
                    if default is None:
                        labels[v] = str(v)
                    else:
                        labels[v] = default
            # find labels corresponding to unique values
            u_labels = [labels[v] for v in unique]
            # merge identical labels
            u_label_index = np.array([u_labels.index(label) for label in u_labels])
            x_ = u_label_index[np.digitize(x, unique, True)]
            # {label: code}
            codes = dict(zip(u_labels, u_label_index))
        else:
            # convert x to codes
            highest_code = -1
            codes = {}  # {label -> code}
            x_ = np.empty(n_cases, dtype=np.uint32)
            for i, value in enumerate(x):
                if value in labels:
                    label = labels[value]
                elif default is not None:
                    label = labels[value] = default
                elif isinstance(value, str):
                    label = labels[value] = value
                else:
                    label = labels[value] = str(value)

                if label in codes:
                    x_[i] = codes[label]
                else:  # new code
                    x_[i] = codes[label] = highest_code = highest_code + 1

            if highest_code >= 2**32:
                raise RuntimeError("Too many categories in this Factor")

        # redefine labels for new codes
        labels = {codes[label]: label for label in labels.values() if label in codes}

        if not (isinstance(repeat, int) and repeat == 1):
            x_ = x_.repeat(repeat)
        if tile != 1:
            x_ = np.tile(x_, tile)
        self.x = x_
        self._labels = labels
        self._init_secondary()

    def _init_secondary(self):
        self._codes = {label: code for code, label in self._labels.items()}
        self._n_cases = len(self.x)

    def __setstate__(self, state):
        self.x = state['x']
        self.name = state['name']
        self.random = state['random']
        if 'ordered_labels' in state:
            # 0.13:  ordered_labels replaced labels
            self._labels = state['ordered_labels']
        else:
            labels = state['labels']
            codes = {label: code for code, label in labels.items()}
            self._labels = {codes[label]: label for label in natsorted(labels.values())}
        self._init_secondary()

    def __getstate__(self):
        state = {'x': self.x,
                 'name': self.name,
                 'random': self.random,
                 'ordered_labels': self._labels}
        return state

    def __repr__(self, full=False):
        use_labels = preferences['factor_repr_use_labels']
        n_cases = preferences['factor_repr_n_cases']

        if use_labels:
            values = self.as_labels()
        else:
            values = self.x.tolist()

        if full or len(self.x) <= n_cases:
            x = repr(values)
        else:
            x = [repr(v) for v in values[:n_cases]]
            x.append('<... N=%s>' % len(self.x))
            x = '[' + ', '.join(x) + ']'

        args = [x]

        if self.name is not None:
            args.append('name=%r' % self.name)

        if self.random:
            args.append('random=True')

        if not use_labels:
            args.append('labels=%s' % self._labels)

        return 'Factor(%s)' % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

    # container ---
    def __getitem__(self, index):
        index = asindex(index)
        x = self.x[index]
        if isinstance(x, np.ndarray):
            return Factor(x, self.name, self.random, labels=self._labels)
        else:
            return self._labels[x]

    def __setitem__(self, index, x):
        # convert x to code
        if isinstance(x, str):
            self.x[index] = self._get_code(x)
        else:
            self.x[index] = list(map(self._get_code, x))

        # obliterate redundant labels
        for code in set(self._labels).difference(self.x):
            del self._codes[self._labels.pop(code)]

    def _get_code(self, label):
        "Add the label if it does not exists and return its code"
        try:
            return self._codes[label]
        except KeyError:
            code = 0
            while code in self._labels:
                code += 1

            if code >= 2**32:
                raise ValueError("Too many categories in this Factor.")

            self._labels[code] = label
            self._codes[label] = code
            return code

    def __iter__(self):
        return (self._labels[i] for i in self.x)

    def __contains__(self, value):
        return value in self._codes

    # numeric ---
    def __eq__(self, other):
        return self.x == self._encode(other)

    def __ne__(self, other):
        return self.x != self._encode(other)

    def _encode(self, x):
        if isinstance(x, str):
            return self._codes.get(x, -1)
        elif len(x) == 0:
            return np.empty(0, dtype=int)
        elif isinstance(x, Factor):
            mapping = [self._codes.get(x._labels.get(xcode, -1), -1) for
                       xcode in range(x.x.max() + 1)]
            return np.array(mapping)[x.x]
        else:
            return np.array([self._codes.get(label, -1) for label in x])

    def __call__(self, other):
        """Create a nested effect.

        A factor A is nested in another factor B if
        each level of A only occurs together with one level of B.

        """
        return NestedEffect(self, other)

    @property
    def as_dummy(self):
        codes = np.empty((self._n_cases, self.df))
        for i, cell in enumerate(self.cells[:-1]):
            codes[:, i] = (self == cell)
        return codes

    @property
    def as_dummy_complete(self):
        x = self.x
        categories = np.unique(x)
        out = np.empty((len(self), len(categories)))
        for i, cat in enumerate(categories):
            np.equal(x, cat, out[:, i])
        return out

    @property
    def as_effects(self):
        shape = (self._n_cases, self.df)
        codes = np.empty(shape)
        for i, cell in enumerate(self.cells[:-1]):
            codes[:, i] = (self == cell)

        contrast = (self == self.cells[-1])
        codes -= contrast[:, None]
        return codes

    def _coefficient_names(self, method):
        if method == 'dummy':
            return ["%s:%s" % (self.name, cell) for cell in self.cells[:-1]]
        contrast_cell = self.cells[-1]
        return ["%s:%s-%s" % (self.name, cell, contrast_cell)
                for cell in self.cells[:-1]]

    def as_labels(self):
        "Convert the Factor to a list of str"
        return [self._labels[v] for v in self.x]

    def as_var(
            self,
            labels: Dict[str, float] = None,
            default: float = None,
            name: str = None,
    ) -> Var:
        """Convert into a Var

        Parameters
        ----------
        labels
            A ``{old_value: new_value}`` mapping; defaults to coercing the
            cell values to numbers.
        default
            Default value for old values not mentioned in ``labels``. If not
            specified, old values missing from ``labels`` will raise a
            ``KeyError``.
        name
            Name of the output Var (default is current name).
        """
        if labels is None:
            if all(v.isdigit() for v in self.cells):
                labels = {cell: int(cell) for cell in self.cells}
            else:
                labels = {cell: float(cell) for cell in self.cells}
        return _Effect.as_var(self, labels, default, name)

    @property
    def beta_labels(self):
        cells = self.cells
        txt = '{0}=={1}'
        return [txt.format(cells[i], cells[-1]) for i in range(len(cells) - 1)]

    @property
    def cells(self):
        return tuple(self._labels.values())

    def _sorted_cells(self):
        return natsorted(self.cells)

    def _cellsize(self):
        "int if all cell sizes are equal, otherwise a {cell: size} dict"
        buf = np.empty(self.x.shape, bool)
        ns = {value: np.equal(self.x, code, buf).sum() for code, value
              in self._labels.items()}
        n_set = set(ns.values())
        if len(n_set) == 1:
            return n_set.pop()
        else:
            return ns

    def _summary(self, width=80):
        if self.random:
            info = ['random']
            n_suffix = 9
        else:
            info = []
            n_suffix = 0
        ns = [(label, np.sum(self.x == code)) for code, label in self._labels.items()]
        items = [f'{label}:{n}' if n > 1 else label for label, n in ns]
        if sum(map(len, items)) + 2 * len(items) - 2 + n_suffix <= width:
            exhaustive = True
            suffix = ''
        else:
            exhaustive = False
            suffix = '...'
            info.insert(0, f'{len(items)} cells')

        if info:
            suffix = f"{suffix} ({', '.join(info)})"

        if not exhaustive:
            n = len(suffix)
            for i, item in enumerate(items):
                n += len(item)
                if n > width:
                    break
                n += 2
            items = items[:i]
        return f"{', '.join(items)}{suffix}"

    def count(self, value: str = None, start: int = -1) -> Var:
        """Cumulative count of the occurrences of ``value``

        Parameters
        ----------
        value
            Value which is to be counted.
        start
            Value at which to start counting (with the default of -1, the first
            occurrence will be 0).

        Returns
        -------
        count : Var of int,  len = len(self)
            Cumulative count of value in self.

        Examples
        --------
        >>> a = Factor('abcabcabccc')
        >>> a
        Factor(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'c', 'c'])
        >>> a.count()
        array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4])
        >>> a.count('a')
        array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        """
        if value is None:
            x = np.empty(len(self.x), int)
            index = np.empty(len(self.x), bool)
            for v in np.unique(self.x):
                np.equal(self.x, v, index)
                x[index] = np.arange(index.sum())
        else:
            x = np.cumsum(self == value) + start
        return Var(x)

    def aggregate(
            self,
            x: CategorialArg,
            name: str = None,
    ) -> Factor:
        """Summarize the Factor by collapsing within the cells of ``x``

        Raises an error if there are cells that contain more than one value.

        Parameters
        ----------
        x
            A categorial model defining cells to collapse.
        name
            Name of the output Factor (default is current name).

        Returns
        -------
        factor
            A copy of self with only one value for each cell in ``x``.
        """
        if x is None:
            cells = [None]
            indexes = [slice(None)]
        elif len(x) != len(self):
            raise ValueError(f"x={dataobj_repr(x)} of length {len(x)} for Factor {dataobj_repr(self)} of length {len(self)}")
        else:
            cells = x.cells
            indexes = [x == cell for cell in cells]

        x_out = []
        for cell, index in zip(cells, indexes):
            x_i = np.unique(self.x[index])
            if len(x_i) > 1:
                labels = tuple(self._labels[code] for code in x_i)
                desc = '' if cell is None else f' in cell {cell!r}'
                raise ValueError(f"Can not determine aggregated value for Factor {dataobj_repr(self)}{desc} because it contains multiple values {labels}. Set drop_bad=True in order to ignore this inconsistency and drop the Factor.")
            else:
                x_out.append(x_i[0])

        if name is None or name is True:
            name = self.name
        return Factor(x_out, name, self.random, labels=self._labels)

    def copy(self, name=None, repeat=1, tile=1):
        "A deep copy"
        if name is None or name is True:
            name = self.name
        return Factor(self.x, name, self.random, repeat, tile, self._labels)

    @property
    def df(self):
        return max(0, len(self._labels) - 1)

    def endswith(self, substr: str) -> np.ndarray:
        """An index that is true for all cases whose name ends with ``substr``

        Parameters
        ----------
        substr
            String for selecting cells that end with substr.

        Returns
        -------
        index
            Index that is ``True`` for all cases whose label ends with
            ``substr``.

        See Also
        --------
        .startswith
        .matches

        Examples
        --------
        >>> a = Factor(['a1', 'a2', 'b1', 'b2'])
        >>> a.endswith('1')
        array([True, False,  True,  False], dtype=bool)
        """
        values = [v for v in self.cells if v.endswith(substr)]
        return self.isin(values)

    def floodfill(self, regions, empty=''):
        """Fill in empty regions in a Factor from the nearest non-empty value

        Parameters
        ----------
        regions : array_like | str
            How to define regions to fill. Can be an object with same length as
            the factor that indicates regions to fill (see example). Can also
            be ``"previous"``, in which case the last value before the empty
            cell is used.
        empty : str
            Value that is to be treated as empty (default is '').

        Examples
        --------
        >>> f = Factor(['', '', 'a', '', '', '', 'b', ''])
        >>> f.floodfill([1, 1, 1, 1, 2, 2, 2, 2])
        Factor(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
        >>> f.floodfill([1, 1, 1, 2, 2, 2, 2, 2])
        Factor(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'])
        >>> f.floodfill([1, 1, 1, 1, 1, 1, 1, 1])
        Factor(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b'])
        """
        if isinstance(regions, str) and regions not in ('previous',):
            raise ValueError("demarcation=%r" % (regions,))

        out = self.copy(None)
        if empty not in self._codes:
            return out
        empty = out._codes[empty]
        x = out.x

        if isinstance(regions, str):
            if regions == 'previous':
                is_empty = np.flatnonzero(x == empty)
                if is_empty[0] == 0:
                    is_empty = is_empty[1:]
                for i in is_empty:
                    x[i] = x[i - 1]
            else:
                raise RuntimeError("demarcation=%r" % (regions,))
        else:
            assert len(regions) == self._n_cases
            i_region_start = 0
            region_v = -1 if regions[0] is None else None
            fill_with = empty
            for i in range(self._n_cases):
                if regions[i] == region_v:
                    if x[i] == empty:
                        if fill_with != empty:
                            x[i] = fill_with
                    else:
                        if fill_with == empty:
                            x[i_region_start:i] = x[i]
                        fill_with = x[i]
                else:  # region change
                    region_v = regions[i]
                    fill_with = x[i]
                    if fill_with == empty:
                        i_region_start = i

        # remove redundant label
        if empty not in x:
            del out._codes[out._labels.pop(empty)]

        return out

    def get_index_to_match(self, other):
        """Generate index to conform to another Factor's order

        Assuming that ``other`` is a reordered version of self,
        ``get_index_to_match()`` generates an index to transform from the order
        of ``self`` to the order of ``other``.
        To guarantee exact matching, each value can only occur once in ``self``.

        Examples
        --------
        >>> index = factor1.get_index_to_match(factor2)
        >>> all(factor1[index] == factor2)
        True

        """
        assert self._labels == other._labels
        index = []
        for v in other.x:
            where = np.where(self.x == v)[0]
            if len(where) == 1:
                index.append(where[0])
            else:
                msg = "%r contains several cases of %r" % (self, v)
                raise ValueError(msg)
        return np.array(index)

    def isany(self, *values):
        """Find the index of entries matching one of the ``*values``

        Returns
        -------
        index : array of bool
            For each case True if the value is in values, else False.

        Examples
        --------
        >>> a = Factor('aabbcc')
        >>> b.isany('b', 'c')
        array([False, False,  True,  True,  True,  True], dtype=bool)
        """
        return self.isin(values)

    def isin(self, values: Collection[str]) -> np.ndarray:
        """Find the index of entries matching one of the ``values``

        Returns
        -------
        index
            For each case True if the value is in values, else False.

        Examples
        --------
        >>> f = Factor('aabbcc')
        >>> f.isin(('b', 'c'))
        array([False, False,  True,  True,  True,  True], dtype=bool)
        """
        return np.isin(self.x, self._encode(values))

    def isnot(self, *values):
        """Find the index of entries not in ``values``

        Returns
        -------
        index : array of bool
            For each case False if the value is in values, else True.
        """
        return self.isnotin(values)

    def isnotin(self, values):
        """Find the index of entries not in ``values``

        Returns
        -------
        index : array of bool
            For each case False if the value is in values, else True.
        """
        return np.isin(self.x, self._encode(values), invert=True)

    def label_length(self, name=None):
        """Create Var with the length of each label string

        Parameters
        ----------
        name : str
            Name of the output Factor (default is current name).

        Examples
        --------
        >>> f = Factor(['a', 'ab', 'long_label'])
        >>> f.label_length()
        Var([1, 2, 10])
        """
        label_lengths = {code: len(label) for code, label in self._labels.items()}
        x = np.empty(len(self))
        for i, code in enumerate(self.x):
            x[i] = label_lengths[code]

        if name is None:
            name = self.name
            if self.name:
                longname = self.name + '.label_length()'
            else:
                longname = 'label_length'
        else:
            longname = name

        return Var(x, name, {"longname": longname})

    def matches(self, pattern: str) -> np.ndarray:
        """An index that is true for all cases whose name matches ``pattern``

        Parameters
        ----------
        pattern
            :mod:`fnmatch` pattern for selecting cells.

        Returns
        -------
        index
            Index that is ``True`` for all cases whose label matches
            ``pattern``.

        See Also
        --------
        .startswith
        .endswith

        Examples
        --------
        >>> a = Factor(['a1', 'a2', 'b1', 'b2'])
        >>> a.matches('b*')
        array([False, False,  True,  True], dtype=bool)
        """
        values = [v for v in self.cells if fnmatch.fnmatch(v, pattern)]
        return self.isin(values)

    @property
    def n_cells(self):
        return len(self._labels)

    def update_labels(self, labels):
        """Change one or more labels in place

        Parameters
        ----------
        labels : dict
            Mapping from old labels to new labels. Existing labels that are not
            in ``labels`` are kept.

        Examples
        --------
        >>> f = Factor('aaabbbccc')
        >>> f
        Factor(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])
        >>> f.update_labels({'a': 'v1', 'b': 'v2'})
        >>> f
        Factor(['v1', 'v1', 'v1', 'v2', 'v2', 'v2', 'c', 'c', 'c'])

        In order to create a copy of the Factor with different labels just
        use the labels argument when initializing a new Factor:

        >>> Factor(f, labels={'c': 'v3'})
        Factor(['v1', 'v1', 'v1', 'v2', 'v2', 'v2', 'v3', 'v3', 'v3'])
        """
        new_labels = {code: labels.get(label, label) for code, label in self._labels.items()}
        # check for merged labels
        codes_ = sorted(new_labels)
        labels_ = [new_labels[c] for c in codes_]
        for i, label in enumerate(labels_):
            first_i = labels_.index(label)
            if first_i < i:
                old_code = codes_[i]
                new_code = codes_[first_i]
                self.x[self.x == old_code] = new_code
                del new_labels[old_code]

        self._labels = new_labels
        self._codes = {l: c for c, l in new_labels.items()}

    def sort_cells(self, order):
        """Reorder the cells of the Factor (in-place)

        The cell order controls the order in which data are displayed in tables
        and plots.

        Parameters
        ----------
        order : sequence of str
            New cell order. Needs to contain each cell exactly once.
        """
        new_order = tuple(order)
        new = set(new_order)
        old = set(self.cells)
        if new != old:
            invalid = new.difference(old)
            if invalid:
                raise ValueError("Factor does not have cells: %s" % ', '.join(invalid))
            missing = old.difference(new)
            if missing:
                raise ValueError("Factor has cennls not in order: %s" % ', '.join(missing))
            raise RuntimeError("Factor.sort_cells comparing %s and %s" % (old, new))
        self._labels = {self._codes[cell]: cell for cell in new_order}

    def lower(self, name: str = None) -> Factor:
        "All labels to lower-case"
        labels = {code: label.lower() for code, label in self._labels.items()}
        if name is None:
            name = self.name
        return Factor(self.x, name, self.random, labels=labels)

    def upper(self, name: str = None) -> Factor:
        "All labels to upper-case"
        labels = {code: label.upper() for code, label in self._labels.items()}
        if name is None:
            name = self.name
        return Factor(self.x, name, self.random, labels=labels)

    def startswith(self, substr: str) -> np.ndarray:
        """An index that is true for all cases whose name starts with ``substr``

        Parameters
        ----------
        substr
            String for selecting cells that start with substr.

        Returns
        -------
        index
            Index that is ``True`` for all cases whose label starts with
            ``substr``.

        See Also
        --------
        .endswith
        .matches

        Examples
        --------
        >>> a = Factor(['a1', 'a2', 'b1', 'b2'])
        >>> a.startswith('b')
        array([False, False,  True,  True], dtype=bool)
        """
        values = [v for v in self.cells if v.startswith(substr)]
        return self.isin(values)

    def table_categories(self):
        "A table containing information about categories"
        table = fmtxt.Table('rll')
        table.title(self.name)
        for title in ['i', 'Label', 'n']:
            table.cell(title)
        table.midrule()
        for code, label in self._labels.items():
            table.cell(code)
            table.cell(label)
            table.cell(np.sum(self.x == code))
        return table

    def repeat(self, repeats, name=None):
        """Repeat each element ``repeats`` times

        Parameters
        ----------
        repeats : int | array of int
            Number of repeats, either a constant or a different number for each
            element.
        name : str
            Name of the output Factor (default is current name).
        """
        if name is None or name is True:
            name = self.name
        return Factor(self.x, name, self.random, repeats, labels=self._labels)

    def tile(self, repeats, name=None):
        """Construct a Factor by repeating ``self`` ``repeats`` times

        Parameters
        ----------
        repeats : int
            Number of repeats.
        name : str
            Name of the output Factor (default is current name).
        """
        if name is None or name is True:
            name = self.name
        return Factor(self.x, name, self.random, tile=repeats, labels=self._labels)


class NDVar(Named):
    """Container for n-dimensional data.

    Parameters
    ----------
    x : array_like
        The data.
    dims
        The dimensions characterizing the axes of the data. If present, ``Case``
        should always occupy the first position.
    name
        Name for the NDVar.
    info
        A dictionary with data properties (can contain arbitrary
        information that will be accessible in the info attribute).


    Notes
    -----
    An :class:`NDVar` consists of the following components:

    - A :class:`numpy.ndarray`, stored in the :attr:`.x` attribute.
    - Meta-information describing each axis of the array using a
      :class:`Dimension` object (for example, :class:`UTS` for uniform
      time series, or :class:`Sensor` for a sensor array). These
      dimensions are stored in the :attr:`.dims` attribute, with the ith
      element of :attr:`.dims` describing the ith axis of :attr:`.x`.
    - A dictionary containing other meta-information stored in the
      :attr:`.info` attribute.
    - A name stored in the :attr:`.name` attribute.

    :class:`NDVar` objects support the native :func:`abs` and :func:`round`
    functions.

    *Indexing*: For classical indexing, indexes need to be provided in the
    correct sequence. For example, assuming ``ndvar``'s first axis is time,
    ``ndvar[0.1]`` retrieves a slice at time = 0.1 s. If time is the second
    axis, the same can be achieved with ``ndvar[:, 0.1]``.
    In :meth:`NDVar.sub`, dimensions can be specified as keywords, for example,
    ``ndvar.sub(time=0.1)``, regardless of which axis represents the time
    dimension.

    *Shallow copies*: When generating a derived NDVars, :attr:`x` and
    :attr:`dims` are generated without copying data whenever possible.
    A shallow copy of :attr:`info` is stored. This means that modifying a
    derived NDVar in place can affect the NDVar it was derived from.
    When indexing an NDVar, the new NDVar will contain a view
    on the data whenever possible based on the underlying array (See `NumPy
    Indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing
    .html>`_). This only matters when explicitly modifying an NDVar in place
    (e.g., ``ndvar += 1``) because NDVar methods that return NDVars never
    implicitly modify the original NDVars in place (see `this note
    <https://mail.python.org/pipermail/python-dev/2003-October/038855.html>`_).


    Examples
    --------
    - Generating :class:`NDVar`: :ref:`exa-generate-ndvar`, :ref:`exa-mtrf`
    - Work with :class:`NDVar`: :ref:`exa-cluster-based-mu`
    - Convert :class:`NDVar` to univariate :class:`Var`: :ref:`exa-compare-topographies`
    """
    def __init__(
            self,
            x: ArrayLike,
            dims: Union[Dimension, Sequence[Dimension]],
            name: str = None,
            info: dict = None,
    ):
        if isinstance(name, dict):  # backwards compatibility
            warn("NDVar argument order has changed; please update your code", DeprecationWarning)
            name, info = info, name

        # check data shape
        if isinstance(dims, Dimension) or dims is Case or isinstance(dims, str):
            dims_ = [dims]
        else:
            dims_ = list(dims)

        x = np.asanyarray(x)
        if len(dims_) != x.ndim:
            raise DimensionMismatchError(f"Unequal number of dimensions (data: {x.ndim}, dims: {len(dims_)})")

        first_dim = dims_[0]
        if first_dim is Case or (isinstance(first_dim, str) and first_dim == 'case'):
            dims_[0] = Case(len(x))

        if not all(isinstance(dim, Dimension) for dim in dims_):
            raise TypeError(f"dims={dims}: All dimensions must be Dimension subclass objects")
        elif any(isinstance(dim, Case) for dim in dims_[1:]):
            raise TypeError(f"dim={dims}: Only the first dimension can be Case")

        # check dimensions
        for dim, n in zip(dims_, x.shape):
            if len(dim) != n:
                raise DimensionMismatchError(f"Dimension {dim.name!r} length mismatch: {n} in data, {len(dim)} in dimension")

        self.x = x
        self.dims = tuple(dims_)
        Named.__init__(self, name, info)
        self._init_secondary()

    def _init_secondary(self):
        self.has_case = isinstance(self.dims[0], Case)
        self._truedims = self.dims[self.has_case:]
        self.dimnames = tuple(dim.name for dim in self.dims)
        self.ndim = len(self.dims)
        self.shape = self.x.shape
        self._dim_2_ax = {dimname: i for i, dimname in enumerate(self.dimnames)}
        # Dimension attributes
        for dim in self._truedims:
            if hasattr(self, dim.name):
                raise ValueError(f"Invalid dimension name: {dim.name!r} (name is reserved for an NDVar attribute)")
            else:
                setattr(self, dim.name, dim)

    def __setstate__(self, state):
        # backwards compatibility
        if 'properties' in state:
            state['info'] = state.pop('properties')
        if isinstance(state['dims'][0], str):
            state['dims'] = (Case(len(state['x'])),) + state['dims'][1:]

        self.x = state['x']
        self.dims = state['dims']
        self._name = state['name']
        self.info = state['info']
        self._init_secondary()

    def __getstate__(self):
        return {'dims': self.dims, 'x': self.x, 'name': self._name, 'info': self.info}

    __array_priority__ = 15

    @property
    def __array_interface__(self):
        return self.x.__array_interface__

    # numeric ---
    def __bool__(self):
        raise TypeError("The truth value of an NDVar is ambiguous. Use v.any() or v.all()")

    def __neg__(self):
        return NDVar(-self.x, self.dims, *op_name(self, '-'))

    def __pos__(self):
        return self

    def __abs__(self):
        return self.abs()

    def __invert__(self):
        return NDVar(~self.x, self.dims, *op_name(self, '~'))

    def __lt__(self, other):
        return NDVar(self.x < self._ialign(other), self.dims, *op_name(self, '<', other, _info.for_boolean(self.info)))

    def __le__(self, other):
        return NDVar(self.x <= self._ialign(other), self.dims, *op_name(self, '<=', other, _info.for_boolean(self.info)))

    def __eq__(self, other):
        return NDVar(self.x == self._ialign(other), self.dims, *op_name(self, '==', other, _info.for_boolean(self.info)))

    def __ne__(self, other):
        return NDVar(self.x != self._ialign(other), self.dims, *op_name(self, '!=', other, _info.for_boolean(self.info)))

    def __gt__(self, other):
        return NDVar(self.x > self._ialign(other), self.dims, *op_name(self, '>', other, _info.for_boolean(self.info)))

    def __ge__(self, other):
        return NDVar(self.x >= self._ialign(other), self.dims, *op_name(self, '>=', other, _info.for_boolean(self.info)))

    def _align(self, other: NDVar):
        """Align data from 2 NDVars.

        Notes
        -----
        For unequal but overlapping dimensions, the intersection is used.
        For example, ``c = a + b`` with ``a`` [-100 300] ms and ``b`` [0 400] ms
        ``c`` will be [0 300] ms.
        """
        if isinstance(other, Var):
            if self.has_case:
                return self.dims, self.x, self._ialign(other)
            else:
                dims = (Case, *self.dims)
                x_self = self.x[newaxis]
                x_other = other.x.reshape([-1, *repeat(1, self.ndim)])
                return dims, x_self, x_other
        elif isinstance(other, NDVar):
            # union of dimensions
            dimnames = []
            if self.has_case or other.has_case:
                dimnames.append('case')
            for name in chain(self.dimnames, other.dimnames):
                if name not in dimnames:
                    dimnames.append(name)

            # find data axes
            self_axes = [name if name in self.dimnames else None for name in dimnames]
            other_axes = [name if name in other.dimnames else None for name in dimnames]

            # find dims
            dims = []
            crop = False
            crop_self = []
            crop_other = []
            for name, other_name in zip(self_axes, other_axes):
                if name is None:
                    dim = other.get_dim(other_name)
                    cs = co = FULL_SLICE
                elif other_name is None:
                    dim = self.get_dim(name)
                    cs = co = FULL_SLICE
                else:
                    self_dim = self.get_dim(name)
                    other_dim = other.get_dim(other_name)
                    if self_dim == other_dim:
                        dim = self_dim
                        cs = co = FULL_SLICE
                    else:
                        dim = self_dim.intersect(other_dim)
                        crop = True
                        cs = self_dim._array_index(dim)
                        co = other_dim._array_index(dim)
                dims.append(dim)
                crop_self.append(cs)
                crop_other.append(co)

            x_self = self.get_data(self_axes)
            x_other = other.get_data(other_axes)
            if crop:
                x_self = x_self[tuple(crop_self)]
                x_other = x_other[tuple(crop_other)]
            return dims, x_self, x_other
        elif isinstance(other, np.ndarray):
            if other.shape == self.x.shape:
                return self.dims, self.x, other
            else:
                raise ValueError(f"Array has wrong shape {other.shape} for {self}")
        elif np.isscalar(other):
            return self.dims, self.x, other
        else:
            raise TypeError(f"{other!r}; need NDVar, Var or scalar")

    def _ialign(
            self,
            other: NDVar,  # align data in this NDVar to self
            index: tuple = None,  # Array-index into self to which to align (for assignment)
    ):
        "Align for self-modifying operations (+=, ...)"
        if np.isscalar(other):
            return other
        elif isinstance(other, Var):
            assert self.has_case
            n = len(other)
            shape = (n,) + (1,) * (self.x.ndim - 1)
            return other.x.reshape(shape)
        elif isinstance(other, NDVar):
            # filter out dimensions that are skipped in assignment
            if index is None or isinstance(index, (slice, list, np.ndarray)):
                self_dims = self.dimnames
            elif isinstance(index, INT_TYPES):
                self_dims = self.dimnames[1:]
            elif isinstance(index, tuple):
                self_dims = [dim for i, dim in zip_longest(index, self.dimnames) if not isinstance(i, INT_TYPES)]
            else:
                raise NotImplementedError(f"Index {index!r} of type {type(index)}")
            # make sure other does not have dimensions not in self
            missing = set(other.dimnames).difference(self_dims)
            if missing:
                raise ValueError(f"{other!r} contains dimensions not in NDVar: {', '.join(missing)}")
            # find index into other
            i_other = []
            for dim in self_dims:
                if dim in other.dimnames:
                    i_other.append(dim)
                else:
                    i_other.append(None)
            return other.get_data(i_other)
        else:
            other = np.asarray(other)
            if other.shape == self.shape:
                return other
            else:
                raise ValueError(f"array of shape {other.shape}: For NDVar operations with arrays, shape needs to match the NDVar exactly")

    def __add__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self + x_other, dims, *op_name(self, '+', other))

    def __iadd__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__add__(other)
        self.x += self._ialign(other)
        return self

    def __radd__(self, other):
        return NDVar(other + self.x, self.dims, *op_name(other, '+', self))

    def __div__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self / x_other, dims, *op_name(self, '/', other))

    def __idiv__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__div__(other)
        self.x /= self._ialign(other)
        return self

    def __rdiv__(self, other):
        return NDVar(other / self.x, self.dims, *op_name(other, '/', self))

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __floordiv__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self // x_other, dims, *op_name(self, '//', other))

    def __ifloordiv__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__floordiv__(other)
        self.x //= self._ialign(other)
        return self

    def __rfloordiv__(self, other):
        return NDVar(other // self.x, self.dims, *op_name(other, '//', self))

    def __mod__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self % x_other, dims, *op_name(self, '%', other))

    def __imod__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__mod__(other)
        self.x %= self._ialign(other)
        return self

    def __rmod__(self, other):
        return NDVar(other % self.x, self.dims, *op_name(other, '//', self))

    def __mul__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self * x_other, dims, *op_name(self, '*', other))

    def __imul__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__mul__(other)
        self.x *= self._ialign(other)
        return self

    def __rmul__(self, other):
        return NDVar(other * self.x, self.dims, *op_name(other, '*', self))

    def __pow__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(np.power(x_self, x_other), dims, *op_name(self, '**', other))

    def __ipow__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__pow__(other)
        self.x **= self._ialign(other)
        return self

    def __rpow__(self, other):
        return NDVar(other ** self.x, self.dims, *op_name(other, '**', self))

    def __round__(self, n=0):
        return NDVar(np.round(self.x, n), self.dims, self.name, self.info)

    def __sub__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self - x_other, dims, *op_name(self, '-', other))

    def __isub__(self, other):
        if self.x.dtype.kind == 'b':
            return self.__sub__(other)
        self.x -= self._ialign(other)
        return self

    def __rsub__(self, other):
        return NDVar(other - self.x, self.dims, *op_name(other, '-', self))

    def __and__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self & x_other, dims, *op_name(self, '&', other))

    def __iand__(self, other):
        self.x &= self._ialign(other)
        return self

    def __rand__(self, other):
        return NDVar(other & self.x, self.dims, *op_name(other, '&', self))

    def __xor__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self ^ x_other, dims, *op_name(self, '^', other))

    def __ixor__(self, other):
        self.x ^= self._ialign(other)
        return self

    def __rxor__(self, other):
        return NDVar(other ^ self.x, self.dims, *op_name(other, '^', self))

    def __or__(self, other):
        dims, x_self, x_other = self._align(other)
        return NDVar(x_self | x_other, dims, *op_name(self, '|', other))

    def __ior__(self, other):
        self.x |= self._ialign(other)
        return self

    def __ror__(self, other):
        return NDVar(other | self.x, self.dims, *op_name(other, '|', self))

    # container ---
    def _dim_index_unravel(self, index):
        "Convert ravelled array index to dimension index"
        if self.ndim == 1:
            return self.dims[0]._dim_index(index)
        return self._dim_index(np.unravel_index(index, self.x.shape))

    def _dim_index(self, index):
        "Convert array index to dimension index"
        if isinstance(index, tuple):
            return tuple(dim._dim_index(i) for dim, i in zip(self.dims, index))
        return self.dims[0]._dim_index(index)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.sub(*index)
        else:
            return self.sub(index)

    def __setitem__(self, key, value):
        index = self._array_index(key)
        if isinstance(value, NDVar):
            value = self._ialign(value, index)
        self.x[index] = value

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        dim = self.dims[0]
        name = self.dimnames[0]
        for value in dim:
            yield self.sub(**{name: value})

    def __repr__(self):
        items = ['NDVar']
        if self.name is not None:
            items.append(repr(self.name))
        if self.x.dtype != np.float64:
            items.append(self.x.dtype.name)
        if np.ma.isMaskedArray(self.x):
            items.append(f"{self.x.mask.mean():.0%} masked")
        desc = ' '.join(items)
        dims = ', '.join([f'{len(dim)} {dim.name}' for dim in self.dims])
        return f"<{desc}: {dims}>"

    def _summary(self, width=80):
        items = [f'{len(dim)} {dim.name}' for dim in self.dims[1:]]
        n = 0
        for i, item in enumerate(items):
            n += len(item)
            if n > width - 3:
                return ', '.join(items[:i]) + '...'
        out = ', '.join(items)
        range_desc = f'{np.nanmin(self.x):g} - {np.nanmax(self.x):g}'
        if len(out) + 2 + len(range_desc) <= width:
            if out:
                out += '; '
            out += range_desc
        return out

    def abs(self, name=None):
        """Compute the absolute values

        Parameters
        ----------
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        abs : NDVar
            NDVar with same dimensions and absolute values.
        """
        return NDVar(np.abs(self.x), self.dims, *op_name(self, 'abs(', name=name))

    def all(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, bool]:
        """Whether all values are nonzero over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute whether there are any nonzero values at all.
            An boolean NDVar with the same dimensions as the data can be used
            to find nonzero values in specific elements (if the NDVar has cases
            on a per case basis).
        **regions
            Regions over which to aggregate as keywords.
            For example, to check whether all values between time=0.1 and
            time=0.2 are non-zero, use ``ndvar.all(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        any
            Boolean data indicating presence of nonzero value over specified
            dimensions. Return a Var if only the case dimension remains, and a
            boolean if the function collapses over all data.

        Examples
        --------
        Examples for

        >>> ndvar
        <NDVar 'utsnd': 60 case, 5 sensor, 100 time>

        Check whether all values are nonzero:

        >>> ndvar.all()
        True

        Check whether each time point contains at least one non-zero value

        >>> ndvar.all(('case', 'sensor'))
        <NDVar 'utsnd': 100 time>

        Check for nonzero values between time=0.1 and time=0.2

        >>> ndvar.all(time=(0.1, 0.2))
        <NDVar 'utsnd': 60 case, 5 sensor>

        """
        return self._aggregate_over_dims(axis, regions, np.all)

    def any(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, bool]:
        """Compute presence of any value other than zero over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute whether there are any nonzero values at all.
            An boolean NDVar with the same dimensions as the data can be used
            to find nonzero values in specific elements (if the NDVar has cases
            on a per case basis).
        **regions
            Regions over which to aggregate. For example, to check for nonzero
            values between time=0.1 and time=0.2, use
            ``ndvar.any(time=(0.1, 0.2))``.
        name : str
            Name of the output :class:`NDVar` (default is the current name).

        Returns
        -------
        any
            Boolean data indicating presence of nonzero value over specified
            dimensions. Return a Var if only the case dimension remains, and a
            boolean if the function collapses over all data.
        """
        return self._aggregate_over_dims(axis, regions, np.any)

    def argmax(
            self,
            axis: Union[str, int] = None,
            name: str = None,
    ) -> Union[float, str, tuple, NDVar, Var]:
        """Find the index of the largest value

        ``ndvar[ndvar.argmax()]`` is equivalent to ``ndvar.max()``.

        Parameters
        ----------
        axis
            Axis along which to find the maximum (by default find the maximum
            in the whole :class:`NDVar`).
        name
            Name of the output :class:`NDVar` (default is the current name).

        Returns
        -------
        argmax
            Index appropriate for the NDVar's dimensions. If NDVar has more
            than one dimensions, a tuple of indices.
        """
        if axis is not None:
            if isinstance(axis, str):
                axis = self.get_axis(axis)
            dim = self.dims[axis]
            x = self.x.argmax(axis)
            x = dim._dim_index(x)
            dims = [dim_ for i, dim_ in enumerate(self.dims) if i != axis]
            info = _info._update({}, self.info)
            if dim._axis_unit:
                info['unit'] = dim._axis_unit
            return self._package_aggregated_output(x, dims, name, info)
        return self._dim_index_unravel(self.x.argmax())

    def argmin(
            self,
            axis: Union[str, int] = None,
            name: str = None,
    ) -> Union[float, str, tuple, NDVar, Var]:
        """Find the index of the smallest value

        ``ndvar[ndvar.argmin()]`` is equivalent to ``ndvar.min()``.

        Parameters
        ----------
        axis
            Axis along which to find the minimum (by default find the minimum
            in the whole :class:`NDVar`).
        name
            Name of the output :class:`NDVar` (default is the current name).

        Returns
        -------
        argmin
            Index appropriate for the NDVar's dimensions. If NDVar has more
            than one dimensions, a tuple of indices.
        """
        if axis is not None:
            if isinstance(axis, str):
                axis = self.get_axis(axis)
            dim = self.dims[axis]
            x = self.x.argmin(axis)
            x = dim._dim_index(x)
            dims = [dim_ for i, dim_ in enumerate(self.dims) if i != axis]
            info = _info._update({}, self.info)
            if dim._axis_unit:
                info['unit'] = dim._axis_unit
            return self._package_aggregated_output(x, dims, name, info)
        return self._dim_index_unravel(self.x.argmin())

    def _array_index(self, arg):
        "Convert dimension index to array index"
        if isinstance(arg, NDVar):
            if arg.x.dtype.kind != 'b':
                raise IndexError("Only boolean NDVar can be used as index")
            elif arg.dims == self.dims:
                return arg.x
            target_dims = tuple(dim if dim in arg.dimnames else None for dim in self.dimnames)
            shared_dims = tuple(filter(None, target_dims))
            self_dims = self.get_dims(shared_dims)
            args_dims = arg.get_dims(shared_dims)
            if args_dims != self_dims:
                raise DimensionMismatchError(
                    f'The index has different dimensions than the NDVar\n'
                    f'NDVar: {self_dims}\nIndex: {args_dims}')
            x = arg.x
            if arg.dimnames != shared_dims:
                if any(dim not in shared_dims for dim in arg.dimnames):
                    missing = (dim for dim in arg.dimnames if dim not in shared_dims)
                    raise DimensionMismatchError(
                        f"Index has dimensions {', '.join(missing)} not in {self}")
                source_axes = tuple(range(arg.ndim))
                dest_axes = tuple(shared_dims.index(dim) for dim in arg.dimnames)
                x = np.moveaxis(x, source_axes, dest_axes)
            for axis, dim in enumerate(target_dims):
                if dim is None:
                    x = np.expand_dims(x, axis)
                    x = np.repeat(x, self.shape[axis], axis)
            return x
        elif isinstance(arg, tuple):
            return tuple(dim._array_index(i) for dim, i in zip(self.dims, arg))
        elif isinstance(arg, np.ndarray) and arg.ndim > 1:
            raise NotImplementedError
        else:
            return self.dims[0]._array_index(arg)

    def assert_dims(self, dims):
        if self.dimnames != dims:
            raise DimensionMismatchError(f"Dimensions of {self!r} do not match {dims}")

    def aggregate(
            self,
            x: CategorialArg = None,
            func: Callable = np.mean,
            name: str = None,
    ) -> NDVar:
        """Summarize data in each cell of ``x``

        Parameters
        ----------
        x
            Categorial whose cells define which cases to aggregate.
        func
            Function that is used to create a summary of the cases falling
            into each cell of ``x``. The function needs to accept the data as
            first argument and ``axis`` as keyword-argument. Default is
            :func:`numpy.mean`.
        name
            Name of the output :class:`NDVar` (default is the current name).

        Returns
        -------
        aggregated_ndvar
            NDVar with data aggregated over cells of ``x``.
        """
        if not self.has_case:
            raise DimensionMismatchError(f"{self!r} has no case dimension")
        elif x is None:
            x_out = func(self.x, axis=0)
            dims = self.dims[1:]
        elif len(x) != len(self):
            raise ValueError(f"x={x}: length mismatch, len(self)={len(self)}, len(x)={len(x)}")
        else:
            cell_data = [func(self.x[x == cell], axis=0) for cell in x.cells]
            if np.ma.isMaskedArray(self.x):
                x_out = np.ma.stack(cell_data)
            else:
                x_out = np.stack(cell_data)
            dims = (Case, *self.dims[1:])

        # update info for summary
        if 'summary_info' in self.info:
            info = self.info.copy()
            info.update(info.pop('summary_info'))
        else:
            info = self.info
        return NDVar(x_out, dims, name or self.name, info)

    def _aggregate_over_dims(
            self,
            axis: AxisArg,
            regions: dict,  # regions keyword arguments
            func: Callable,
            mask: Any = None,  # replace masked values with this
    ):
        name = regions.pop('name', None)
        out = regions.pop('out', None)
        if out is not None:
            raise NotImplementedError('out parameter')
        if regions:
            data = self.sub(**regions)
            additional_axis = [dim for dim in regions if data.has_dim(dim)]
            if additional_axis:
                if isinstance(axis, NDVar):
                    data = data.sub(axis)
                    axis = additional_axis
                elif not axis:
                    axis = additional_axis
                elif isinstance(axis, str):
                    axis = [axis] + additional_axis
                else:
                    axis = list(axis) + additional_axis
            return data._aggregate_over_dims(axis, {'name': name}, func, mask)
        elif isinstance(axis, NDVar):
            if mask is not None:
                raise NotImplementedError
            dims, self_x, index = self._align(axis)
            # move indexed dimensions to the back so they can be flattened
            src = [ax for ax, n in enumerate(index.shape) if n != 1]
            n_flatten = len(src)
            dst = list(range(-n_flatten, 0))
            x_t = np.moveaxis(self_x, src, dst)
            x_flat = x_t.reshape((*x_t.shape[:-n_flatten], -1))
            index_flat = index.ravel()
            x = func(x_flat[..., index_flat], -1)
            dims = [dim for i, dim in enumerate(dims) if i not in src]
        elif isinstance(axis, str):
            axis = self._dim_2_ax[axis]
            x = func(self.get_data(mask=mask), axis=axis)
            dims = [self.dims[i] for i in range(self.ndim) if i != axis]
        elif axis is None:
            return func(self.get_data(mask=mask))
        else:
            axes = tuple([self._dim_2_ax[dim_name] for dim_name in axis])
            x = func(self.get_data(mask=mask), axes)
            dims = [self.dims[i] for i in range(self.ndim) if i not in axes]

        return self._package_aggregated_output(x, dims, name, _info.for_data(x, self.info))

    def astype(self, dtype):
        """Copy of the NDVar with data cast to the specified type

        Parameters
        ----------
        dtype : numpy dtype
            Numpy data-type specification (see :meth:`numpy.ndarray.astype`).
        """
        return NDVar(self.x.astype(dtype), self.dims, self.name, self.info)

    def bin(
            self,
            step: float = None,
            start: float = None,
            stop: float = None,
            func: Union[Callable, str] = None,
            dim: str = None,
            name: str = None,
            nbins: int = None,
            label: Literal['start', 'center'] = 'center',
    ) -> NDVar:
        """Bin the data along a given dimension (default ``'time'``)

        Parameters
        ----------
        step
            Time step between bins.
        start
            Earliest time point (default is from the beginning).
        stop
            End of the data to use (the default is to use as many whole
            ``tstep`` intervals as fit in the data).
        func
            How to summarize data in each time bin. Can be the name of a numpy
            function that takes an axis parameter (e.g., 'sum', 'mean', 'max') or
            'extrema' which selects the value with the maximum absolute value.
            The default depends on ``ndvar.info['meas']``:
            'p': minimum;
            'f': maximum;
            't', 'r': extrema;
            otherwise: mean.
        dim
            Dimension over which to bin. If the NDVar has more than one
            dimension, the default is ``'time'``.
        name
            Name of the output NDVar (default is the current name).
        nbins
            Instead of specifying ``step``, ``nbins`` can be specified to divide
            ``dim`` into an even number of bins.
        label
            How to assign labels to the new bins. For example, with
            ``dim='time'``, the new time axis can assign to each bin either the
            center or the start time of the bin on the original time axis.

        Returns
        -------
        binned_ndvar
            NDVar with data binned along the time axis (i.e., each time point
            reflects one time bin).
        """
        if nbins is not None:
            if step is not None:
                raise TypeError(f"{step=}, {nbins=}: can only specify one of step and nbins")
            elif not isinstance(nbins, int):
                raise TypeError(f"{nbins=}: need int")
            elif nbins < 1:
                raise ValueError(f"{nbins=}: needs to be >= 1")
        elif step is None and nbins is None:
            raise TypeError("need to specify one of step and nbins")
        elif label not in ('start', 'center'):
            raise ValueError(f"{label=}")

        if dim is None:
            if len(self.dims) == 1 + self.has_case:
                dim = self.dims[-1].name
            elif self.has_dim('time'):
                dim = 'time'
            else:
                raise TypeError("NDVar has more then 1 dimensions, the dim argument needs to be specified")

        # summary-func
        if func is None:
            meas = self.info.get('meas', '').lower()
            if meas == 'p':
                func = np.min
            elif meas == 'f':
                func = np.max
            elif meas in ('t', 'r'):
                func = extrema
            else:
                func = np.mean
        elif isinstance(func, str):
            if func in EVAL_CONTEXT:
                func = EVAL_CONTEXT[func]
            else:
                func = getattr(numpy, func)
        elif not callable(func):
            raise TypeError(f"{func=}")

        axis = self.get_axis(dim)
        dim = self.get_dim(dim)
        edges, out_dim = dim._bin(start, stop, step, nbins, label)
        edges_index = dim._array_index(edges)

        out_shape = list(self.shape)
        out_shape[axis] = len(edges) - 1
        if isinstance(self.x, np.ma.masked_array):
            x = np.ma.empty(out_shape)
        else:
            x = np.empty(out_shape)
        idx_prefix = FULL_AXIS_SLICE * axis
        for i, bin_ in enumerate(intervals(edges_index)):
            src_index = (*idx_prefix, slice(*bin_))
            dst_index = (*idx_prefix, i)
            x[dst_index] = func(self.x[src_index], axis=axis)

        dims = list(self.dims)
        dims[axis] = out_dim
        info = {**self.info, 'bins': tuple(intervals(edges))}
        return NDVar(x, dims, name or self.name, info)

    def clip(self, min=None, max=None, name=None, out=None):
        """Clip data (see :func:`numpy.clip`)

        Parameters
        ----------
        min : scalar | Var | NDVar
            Minimum value.
        max : scalar | Var | NDVar
            Maximum value.
        name : str
            Name of the output NDVar (default is the current name).
        out : NDVar
            Container for output.
        """
        if min is not None:
            min = self._ialign(min)
        if max is not None:
            max = self._ialign(max)
        if out is not None:
            if out is not self:
                assert out.dims == self.dims
            self.x.clip(min, max, out.x)
            return out
        else:
            x = self.x.clip(min, max)
            return NDVar(x, self.dims, name or self.name, self.info)

    def copy(self, name=None):
        """A deep copy of the NDVar's data

        Parameters
        ----------
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        ndvar_copy : NDVar
            An copy of the ndvar with a deep copy of the data.

        Notes
        -----
        The info dictionary is still a shallow copy.
        """
        return NDVar(self.x.copy(), self.dims, name or self.name, self.info)

    def diff(
            self,
            dim: str = None,
            n: int = 1,
            pad: bool = True,
            name: str = None,
    ) -> NDVar:
        """Discrete difference (see :func:`numpy.diff`)

        parameters
        ----------
        dim
            Dimension along which to operate.
        n
            Number of times to difference (default 1).
        pad
            Pad the ``dim`` dimension of the result to conserve NDVar shape
            (default). For exmaple, ``diff([0, 1, 1, 0]) -> [0, 1, 0, -1]``.
        name
            Name of the output NDVar (default is the current name).

        Returns
        -------
        NDVar
            NDVar with the ``n`` th differences. If the input is masked, the
            previous mask is extended by ``n`` to mask all values in ``diff``
            that incorporate previsouly masked values.
        """
        if n == 0:
            return self
        elif n < 0:
            raise ValueError(f'{n=}')
        # find axis
        if dim is None:
            if self.ndim - self.has_case == 1:
                axis = self.ndim - 1
            else:
                raise TypeError("Need to specify dimension over which to differentiate")
        else:
            axis = self.get_axis(dim)
        # find padding
        if pad:
            idx = FULL_AXIS_SLICE * axis + (slice(0, 1),)
            prepend = self.x[idx]
            if n > 1:
                prepend = np.repeat(prepend, n, axis)
        else:
            raise NotImplementedError(f'{pad=}')
        # differentiate
        x = np.diff(self.x, n, axis, prepend)
        if isinstance(self.x, np.ma.masked_array):
            structure = deep_array([0, 1, 1], axis, x.ndim)
            mask = scipy.ndimage.binary_dilation(self.x.mask, structure, n)
            x = np.ma.masked_array(x, mask)
        return NDVar(x, self.dims, name or self.name, self.info)

    def dot(
            self,
            ndvar: NDVar,
            dims: Union[str, Sequence[str]] = None,
            name: str = None,
    ) -> Union[NDVar, Var, float]:
        """Dot product

        Parameters
        ----------
        ndvar
            Second NDVar, has to have at least the dimension ``dim``.
        dims
            Dimension(s) over which to compute the dot product (default is the
            last dimension shared with ``ndvar``).
        name
            Name of the output NDVar (default is ``ndvar.name``).

        Examples
        --------
        Compute the first 6 DSS components::

            >>> to_dss, from_dss = dss(x)
            >>> x_dss_6 = to_dss[:6].dot(x, 'sensor')
        """
        if dims is None:
            for dims in self.dimnames[::-1]:
                if ndvar.has_dim(dims):
                    break
        x1_axes = self._get_axes(dims)
        x2_axes = ndvar._get_axes(dims)

        # find whether one dimension is subset of other
        x1_index = {}
        x2_index = {}
        for i1, i2 in zip(x1_axes, x2_axes):
            dim_x1 = self.dims[i1]
            dim_x2 = ndvar.dims[i2]
            if dim_x1 != dim_x2:
                out_dim = dim_x1.intersect(dim_x2)
                if dim_x1 != out_dim:
                    x1_index[i1] = dim_x1._array_index_to(out_dim)
                if dim_x2 != out_dim:
                    x2_index[i2] = dim_x2._array_index_to(out_dim)
        # trim data
        x1 = self.x
        for i, index in x1_index.items():
            x1 = np.take(x1, index, i)
        x2 = ndvar.x
        for i, index in x2_index.items():
            x2 = np.take(x2, index, i)

        x = np.tensordot(x1, x2, (x1_axes, x2_axes))

        # output dimensions
        x1_dims = [dim for i, dim in enumerate(self.dims) if i not in x1_axes]
        x2_dims = [dim for i, dim in enumerate(ndvar.dims) if i not in x2_axes]
        dims = [*x1_dims, *x2_dims]

        if name is None:
            name = ndvar.name

        if len(dims) == 0:
            return x
        elif len(dims) == 1 and isinstance(dims[0], Case):
            return Var(x, name)
        else:
            return NDVar(x, dims, name, {})

    def envelope(self, dim='time', name=None):
        """Compute the Hilbert envelope of a signal

        Parameters
        ----------
        dim : str
            Dimension over which to compute the envelope (default 'time').
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        envelope : NDVar
            NDVar with identical dimensions containing the envelope.

        Notes
        -----
        The Hilbert envelope is computed with :func:`scipy.signal.hilbert`::

        >>> numpy.abs(scipy.signal.hilbert(x))

        This function can be very slow when the number of time samples is
        uneven.
        """
        axis = self.get_axis(dim)
        # hilbert more efficient for x^2 shape
        n = self.shape[axis]
        log_n = log(n, 2)
        if log_n % 1:
            use_n = int(round(2 ** ceil(log_n)))
        else:
            use_n = None
        x = np.abs(scipy.signal.hilbert(self.x, use_n, axis=axis))
        if use_n:
            x = x[aslice(axis, stop=n)]
        return NDVar(x, self.dims, name or self.name, self.info)

    def extrema(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, float]:
        """Extrema (value farthest away from 0) over given dimensions

        For each data point,
        ``extremum = max(x) if max(x) >= abs(min(x)) else min(x)``.

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, ``None`` to
            compute the maximum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the extrema in specific elements (if the data has a case
            dimension, the extrema are computed for each case).
        **regions
            Regions over which to aggregate. For example, to get the extrema
            between 0.1 and 0.2 s, use ``ndvar.extrema(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        extrema
            Extrema over specified dimensions. Return a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.

        See Also
        --------
        .max
        .min
        """
        return self._aggregate_over_dims(axis, regions, extrema)

    def fft(self, dim=None, name=None):
        """Fast fourier transform

        Parameters
        ----------
        dim : str
            Dimension along which to operate (the default is the ``time``
            dimension if present).
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        fft : NDVar
            NDVar containing the FFT, with the ``time`` dimension replaced by
            ``frequency``.
        """
        if dim is None:
            if self.ndim - self.has_case == 1:
                dim = self.dimnames[-1]
            elif self.has_dim('time'):
                dim = 'time'
            else:
                raise ValueError("NDVar has more than one dimension, you need "
                                 "to specify along which dimension to operate.")
        axis = self.get_axis(dim)
        x = np.abs(np.fft.rfft(self.x, axis=axis))
        if dim == 'time':
            uts = self.get_dim(dim)
            freqs = np.fft.rfftfreq(len(uts), uts.tstep)
            freq = Scalar('frequency', freqs, 'Hz')
        else:
            n = self.shape[axis]
            freqs = np.fft.rfftfreq(n, 1. / n)
            freq = Scalar('frequency', freqs, 'Hz')
        dims = self.dims[:axis] + (freq,) + self.dims[axis + 1:]
        info = _info.default_info('Amplitude', self.info)
        return NDVar(x, dims, name or self.name, info)

    def flatnonzero(self):
        """Return indices where a 1-d NDVar is non-zero

        Like :func:`numpy.flatnonzero`.
        """
        if self.ndim != 1:
            raise ValueError("flatnonzero only applies to 1-d NDVars")
        dim = self.dims[0]
        return [dim._dim_index(index) for index in np.flatnonzero(self.x)]

    def get_axis(self, name: str) -> int:
        "Return the data axis for a given dimension name"
        if self.has_dim(name):
            return self._dim_2_ax[name]
        else:
            raise DimensionMismatchError(f"{self} has no dimension named {name!r}")

    def _get_axes(self, dims: Union[str, Sequence[str]]) -> Sequence[int]:
        if isinstance(dims, str):
            return self.get_axis(dims),
        else:
            return [self.get_axis(dim) for dim in dims]

    def get_data(self, dims: DimsArg = None, mask: float = None) -> np.ndarray:
        """Retrieve the NDVar's data with a specific axes order.

        Parameters
        ----------
        dims
            Sequence of dimension names (or single dimension name). The array
            that is returned will have axes in this order. To insert a new
            axis with size 1 use ``numpy.newaxis``/``None``. The default is the
            order of dimensions in the NDVar.
        mask
            If data is a masked array, set masked values to ``mask``.

        Returns
        -------
        data
            A reference to, view on or copy of the data (conservative memory usage).
        """
        if dims is None:
            x = self.x
        else:
            if isinstance(dims, str):
                dims = (dims,)
            dims_ = [d for d in dims if d is not newaxis]
            if set(dims_) != set(self.dimnames) or len(dims_) != len(self.dimnames):
                raise DimensionMismatchError(f"Requested dimensions {dims} from {self}")
            # transpose
            axes = [self.dimnames.index(d) for d in dims_]
            x = self.x.transpose(axes)

        # apply mask
        if mask is not None and isinstance(x, np.ma.MaskedArray):
            x = x.filled(mask)

        # insert axes
        if dims is not None and len(dims) > len(dims_):
            expand_dims = np.ma.expand_dims if isinstance(x, np.ma.MaskedArray) else np.expand_dims
            for ax, dim in enumerate(dims):
                if dim is newaxis:
                    x = expand_dims(x, ax)

        return x

    def get_dim(self, name: str) -> Dimension:
        "Return the Dimension object named ``name``"
        return self.dims[self.get_axis(name)]

    def get_dimnames(
            self,
            names: Sequence[Optional[str]] = None,
            first: Union[str, Sequence[Optional[str]]] = None,
            last: Union[str, Sequence[Optional[str]]] = None,
    ) -> Tuple[str, ...]:
        """Fill in a partially specified tuple of Dimension names

        Parameters
        ----------
        names
            Dimension names. Names specified as ``None`` are inferred.
        first
            Instead of ``names``, specify a constraint on the initial
            dimension(s) only.
        last
            Instead of ``names``, specify a constraint on the last
            dimension(s) only.

        Returns
        -------
        inferred_names
            Dimension names in the same order as in ``names``.
        """
        if first is not None or last is not None:
            if names is not None:
                raise TypeError("Can only specify names or first/last, not both")
            head = () if first is None else (first,) if isinstance(first, str) else first
            tail = () if last is None else (last,) if isinstance(last, str) else last
            n_mid = len(self.dims) - len(head) - len(tail)
            if n_mid < 0:
                raise ValueError(f"first={first!r}, last={last!r}: more arguments than dimensions ({', '.join(self.dimnames)})")
            out = [*head, *repeat(None, n_mid), *tail]
        elif len(names) != len(self.dims):
            raise ValueError(f"{names!r}: wrong number of dimensions for {self}")
        else:
            out = list(names)

        dims = [dim for dim in self.dimnames if dim not in out]
        for i in range(len(out)):
            if out[i] is None:
                out[i] = dims.pop(0)
            elif out[i] not in self.dimnames:
                raise ValueError(f"NDVar has no {out[i]} dimension")
        if len(set(out)) != len(out):
            arg_repr = f'{names!r}' if names else f"first={first!r}, last={last!r}"
            raise ValueError(f"{arg_repr}: duplicate name")
        return tuple(out)

    def get_dims(
            self,
            names: Sequence[Optional[str]] = None,
            first: Union[str, Sequence[Optional[str]]] = None,
            last: Union[str, Sequence[Optional[str]]] = None,
    ) -> Tuple[Dimension]:
        """Return a tuple with the requested Dimension objects

        Parameters
        ----------
        names
            Names of the dimension objects. If ``None`` is inserted in place of
            names, these dimensions are inferred.
        first
            Instead of ``names``, specify a constraint on the initial
            dimension(s) only.
        last
            Instead of ``names``, specify a constraint on the last
            dimension(s) only.

        Returns
        -------
        dims
            Dimension objects in the same order as in ``names``.
        """
        if first or last or names is None or None in names:
            names = self.get_dimnames(names, first, last)
        return tuple([self.get_dim(name) for name in names])

    def has_dim(self, name):
        return name in self._dim_2_ax

    def label_clusters(self, threshold=0, tail=0, name=None):
        """Find and label clusters of values exceeding a threshold

        Parameters
        ----------
        threshold : scalar
            Threshold value for clusters (default 0 to find clusters of
            non-zero values).
        tail : 0 | -1 | 1
            Whether to label cluster smaller than threshold, larger than
            threshold, or both (default).
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        clusters : NDVar
            NDVar of int, each cluster labeled with a unique integer value.
            ``clusters.info['cids']`` contains an array of all cluster IDs.

        Notes
        -----
        Together, the labels in the ``clusters`` array and the identifiers in
        ``clusters.info['cids']`` can be used as input for :mod:`scipy.ndimage`
        functions. For example, count the number of elements in each label
        of a binary ``mask``::

            >>> from scipy import ndimage
            >>> labels = mask.label_clusters()
            >>> ns = ndimage.measurements.sum(mask, labels, labels.info['cids'])
            >>> dict(zip(labels.info['cids'], ns))
            {1: 35.0, 6: 1.0, 37: 1.0}
            >>> (labels == 1).sum()  # confirm count in label 1
            35
        """
        from ._stats.testnd import Adjacency, label_clusters

        custom = [dim._adjacency_type == 'custom' for dim in self.dims]
        if any(custom):
            if sum(custom) > 1:
                raise NotImplementedError("More than one non-adjacent dimension")
            nad_ax = custom.index(True)
        else:
            nad_ax = 0

        if nad_ax:
            x = self.x.swapaxes(0, nad_ax)
            adjacency = Adjacency(
                (self.dims[nad_ax],) + self.dims[:nad_ax] + self.dims[nad_ax + 1:])
        else:
            x = self.x
            adjacency = Adjacency(self.dims)

        cmap, cids = label_clusters(x, threshold, tail, adjacency, None)

        if nad_ax:
            cmap = cmap.swapaxes(0, nad_ax)

        info = {**self.info, 'cids': cids}
        return NDVar(cmap, self.dims, name or self.name, info)

    def log(self, base=None, name=None):
        """Element-wise log

        Parameters
        ----------
        base : scalar
            Base of the log (default is the natural log).
        name : str
            Name of the output NDVar (default is the current name).
        """
        mod = np.ma if isinstance(self.x, np.ma.masked_array) else np
        if base == 2:
            x = mod.log2(self.x)
        elif base == 10:
            x = mod.log10(self.x)
        else:
            x = mod.log(self.x)
            if base is not None:
                x /= log(base)

        if base is None:
            op = 'log('
        else:
            op = f'log{base:g}('
        return NDVar(x, self.dims, *op_name(self, op, name=name))

    def mask(self, mask, name=None, missing=None, fill_value=None):
        """Create a masked version of this NDVar (see :class:`numpy.ma.MaskedArray`)

        Parameters
        ----------
        mask : bool NDVar
            Mask, with equal dimensions (``True`` values will be masked).
        name : str
            Name of the output NDVar (default is the current name).
        missing : bool
            Whether to mask values that are outside of ``mask`` (i.e., when
            ``mask``'s dimensions only cover part of the NDVar).
            The default is to raise a ``ValueError`` if ``mask`` is missing values.
        fill_value
            Value to substitute for masked values if necessary
            (see :class:`numpy.ma.MaskedArray`).

        See Also
        --------
        .unmask : remove mask
        .get_mask : retrieve mask

        Examples
        --------
        In operations such as :meth:`NDVar.mean`, standard :mod:`numpy` behavior
        applies, i.e., masked values are ignored::

            >>> x = NDVar([1, 2, 3], Case)
            >>> x.mean()
            2.0
            >>> y = x.mask([True, False, False])
            >>> y.mean()
            2.5

        """
        x_mask = self._ialign(mask)
        if x_mask.dtype.kind != 'b':
            x_mask = x_mask.astype(bool)
        if x_mask.shape != self.x.shape:
            for ax, (n_mask, n_self) in enumerate(zip(x_mask.shape, self.x.shape)):
                if n_mask == n_self:
                    continue
                elif n_mask == 1:
                    x_mask = np.repeat(x_mask, n_self, ax)
                elif missing is None:
                    raise ValueError("Unable to broadcast mask to NDVar; use missing parameter to fill in missing values")
                else:
                    dim = self.dims[ax]
                    mask_dim = mask.get_dim(dim.name)
                    new_shape = list(x_mask.shape)
                    new_shape[ax] = n_self
                    x_new = np.empty(new_shape, bool)
                    x_new.fill(missing)
                    old_index = dim._array_index(mask_dim),
                    x_new[FULL_AXIS_SLICE * ax + old_index] = x_mask
                    x_mask = x_new
        x = np.ma.MaskedArray(self.x, x_mask, fill_value=fill_value)
        return NDVar(x, self.dims, name or self.name, self.info)

    def unmask(
            self,
            fill_value: Union[float, str] = None,
            name: str = None,
    ):
        """Remove mask from a masked ``NDVar``

        Parameters
        ----------
        fill_value
            Vakue to fill in for previously masked values; can be the name of any
            numpy method, derived from the previously unmasked values (e.g.,
            ``mean`` or ``max``).
        name : str
            Name of the output NDVar (default is the current name).
        """
        if isinstance(self.x, np.ma.masked_array):
            if fill_value is None:
                x = self.x.data
            else:
                if isinstance(fill_value, str):
                    fill_value = getattr(self.x.compressed(), fill_value)()
                x = self.x.filled(fill_value)
        else:
            x = self.x
        if name is None:
            name = self.name
        return NDVar(x, self.dims, name, self.info)

    def get_mask(self, name: str = None) -> NDVar:
        "Retriev the mask as :class:`NDVar`"
        assert isinstance(self.x, np.ma.masked_array), "NDVar is not masked"
        return NDVar(self.x.mask, self.dims, name or self.name, self.info)

    def max(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, float]:
        """Compute the maximum over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the maximum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the maximum in specific elements (if the data has a case
            dimension, the maximum is computed for each case).
        **regions
            Regions over which to aggregate. For example, to get the maximum
            between time=0.1 and time=0.2, use ``ndvar.max(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        max
            The maximum over specified dimensions. Return a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.

        See Also
        --------
        .argmax
        .extrema
        """
        return self._aggregate_over_dims(axis, regions, np.max)

    def mean(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, float]:
        """Compute the mean over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the mean over all dimensions.
            A boolean NDVar with the same dimensions as the data can be used
            to compute the mean in specific elements (if the data has a case
            dimension, the mean is computed for each case).
        **regions
            Regions over which to aggregate. For example, to get the mean
            between time=0.1 and time=0.2, use ``ndvar.mean(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        mean
            The mean over specified dimensions. Return a Var if only the case
            dimension remains, and a float if the function collapses over all
            data.
        """
        return self._aggregate_over_dims(axis, regions, np.mean)

    def min(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, float]:
        """Compute the minimum over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the minimum over all dimensions.
            A boolean NDVar with the same dimensions as the data can be used
            to compute the minimum in specific elements (if the data has a case
            dimension, the minimum is computed for each case).
        **regions
            Regions over which to aggregate. For example, to get the minimum
            between time=0.1 and time=0.2, use ``ndvar.min(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        min
            The minimum over specified dimensions. Return a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.

        See Also
        --------
        .argmin
        .extrema
        """
        return self._aggregate_over_dims(axis, regions, np.min)

    def norm(self, dim, ord=2, name=None):
        """Norm over ``dim``

        Parameters
        ----------
        dim : str
            Dimension over which to operate.
        ord : scalar
            See description of vector norm for :func:`scipy.linalg.norm`
            (default 2).
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        norm : NDVar
            Norm over ``dim``.

        Examples
        --------
        To normalize ``x`` along the sensor dimension:

        >>> x /= x.norm('sensor')
        """
        axis = self.get_axis(dim)
        x = norm(self.x, ord, axis)
        if self.ndim == 1:
            return x
        if isinstance(self.x, np.ma.masked_array):
            all_masked = np.all(self.x.mask, axis)
            any_masked = np.any(self.x.mask, axis)
            if np.any(all_masked != any_masked):
                raise ValueError(f"Norm along {dim!r} with inconsistent mask")
            mask = all_masked
            x = np.ma.masked_array(x, mask)
        dims = self.dims[:axis] + self.dims[axis + 1:]
        return self._package_aggregated_output(x, dims, name)

    def ols(self, x, name=None):
        """Sample-wise ordinary least squares regressions

        Parameters
        ----------
        x : Model | str
            Predictor or predictors. A Model to regress over cases, or a
            dimension name to regress against values of one of the
            ``NDVar``'s dimensions. A Model with multiple ``Var``s can be
            supplied as argument list of ``Var``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        beta : NDVar
            Per sample beta weights. The case dimension reflects the predictor
            variables in the same order as the Model's effects.

        Notes
        -----
        The intercept is generated internally, and betas for the intercept are
        not returned. If you need access to more details of the results,
        consider using :class:`testnd.LM`.

        See Also
        --------
        .ols_t : T-values for regression coefficients
        """
        from ._stats import stats

        info = _info.default_info('beta', self.info)
        if 'summary_info' in info:
            del info['summary_info']

        if isinstance(x, str):
            if x.startswith('.'):
                x = x[:]
            dimnames = self.get_dimnames(first=x)
            dim = self.get_dim(x)
            values = dim._as_scalar_array()
            y = self.get_data(dimnames)
            p = asmodel(Var(values, x))._parametrize()
            betas = stats.lm_betas(y, p)[1]
            out_dims = self.get_dims(dimnames[1:])
        elif not self.has_case:
            raise DimensionMismatchError("Can only apply regression to NDVar with case dimension")
        else:
            x = asmodel(x)
            if len(x) != len(self):
                raise DimensionMismatchError(f"Predictors do not have same number of cases ({len(x)}) as the dependent variable ({len(self)})")

            betas = stats.lm_betas(self.x, x._parametrize())[1:]  # drop intercept
            out_dims = (Case,) + self.dims[1:]
        return self._package_aggregated_output(betas, out_dims, name, info)

    def ols_t(self, x, name=None):
        """
        Compute T-values for sample-wise ordinary least squares regressions

        Parameters
        ----------
        x : Model
            Predictor or predictors. Can also be supplied as argument that can
            be converted to a Model, for example ``Var`` or list of ``Var``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        t : NDVar
            Per sample t-values. The case dimension reflects the predictor
            variables in the same order as the Model's effects.

        Notes
        -----
        Betas for the intercept are not returned.

        See Also
        --------
        .ols : Regression coefficients
        """
        from ._stats import stats

        if not self.has_case:
            raise DimensionMismatchError(
                "Can only apply regression to NDVar with case dimension")

        x = asmodel(x)
        if len(x) != len(self):
            raise DimensionMismatchError(
                "Predictors do not have same number of cases (%i) as the "
                "dependent variable (%i)" % (len(x), len(self)))

        t = stats.lm_t(self.x, x._parametrize())[2][1:]  # drop intercept
        if name is None:
            name = self.name
        return NDVar(t, (Case, *self.dims[1:]), name, self.info)

    def _package_aggregated_output(self, x, dims, name, info=None):
        if len(dims) == 0:
            return x
        if info is None:
            info = self.info
        args = op_name(self, info=info, name=name)
        if len(dims) == 1 and isinstance(dims[0], Case):
            return Var(x, *args)
        else:
            return NDVar(x, dims, *args)

    def quantile(self, q=0.5, axis: AxisArg = None, interpolation='linear', **regions) -> Union[NDVar, Var, float]:
        """The value such that q of the NDVar's values are lower

        (See func:`numpy.quantile`)

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the maximum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the maximum in specific elements (if the data has a case
            dimension, the maximum is computed for each case).
        q : float
            Quantile to compute, between 0 and 1 inclusive.
        interpolation : str
            See func:`numpy.quantile`.
        **regions
            Regions over which to aggregate. For example, to get the maximum
            between time=0.1 and time=0.2, use ``ndvar.max(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).
        """
        if isinstance(self.x, np.ma.masked_array):
            np_func = np.nanquantile
        else:
            np_func = np.quantile
        func = partial(np_func, q=q, interpolation=interpolation)
        return self._aggregate_over_dims(axis, regions, func, mask=np.nan)

    def repeat(self, repeats, name=None):
        """Repeat slices of the NDVar along the case dimension

        Parameters
        ----------
        repeats : int | array of ints
            The number of repetitions for each element. `repeats` is
            broadcasted to fit the shape of the given dimension.
        name : str
            Name of the output NDVar (default is the current name).
        """
        if self.has_case:
            x = self.x.repeat(repeats, axis=0)
            dims = self.dims
        else:
            x = self.x[newaxis].repeat(repeats, axis=0)
            dims = (Case(repeats),) + self.dims
        return NDVar(x, dims, name or self.name, self.info)

    def rename_dim(self, dim: str, to: str, name: str = None):
        """Rename one of the dimensions

        Parameters
        ----------
        dim
            Current name.
        to
            New name.
        name
            Name of the output NDVar (default is the current name).
        """
        axis = self.get_axis(dim)
        dims = list(self.dims)
        dims[axis] = dims[axis]._rename(to)
        return NDVar(self.x, dims, name or self.name, self.info)

    def residuals(self, x, name=None):
        """
        The residuals of sample-wise ordinary least squares regressions

        Parameters
        ----------
        x : Model
            Predictor or predictors. Can also be supplied as argument that can
            be converted to a Model, for example ``Var`` or list of ``Var``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        residuals : NDVar
            Residual for each case and sample (same dimensions as data).
        """
        if not self.has_case:
            raise DimensionMismatchError(
                "Can only apply regression to NDVar with case dimension")

        x = asmodel(x)
        if len(x) != len(self):
            raise DimensionMismatchError(
                "Predictors do not have same number of cases (%i) as the "
                "dependent variable (%i)" % (len(x), len(self)))

        from ._stats import stats
        res = stats.residuals(self.x, x)
        return NDVar(res, self.dims, name or self.name, self.info)

    def rms(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, float]:
        """Compute the root mean square over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the standard deviation over all values.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the RMS in specific elements (if the data has a case
            dimension, the RMS is computed for each case).
        **regions
            Regions over which to aggregate. For example, to get the RMS
            between time=0.1 and time=0.2, use ``ndvar.rms(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        rms
            The root mean square over specified dimensions. Return a Var if
            only the case dimension remains, and a float if the function
            collapses over all data.
        """
        from ._stats.stats import rms
        return self._aggregate_over_dims(axis, regions, rms)

    def sign(self, name=None):
        """Element-wise indication of the sign

        Parameters
        ----------
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        sign : NDVar
            NDVar of same shape, ``-1 if x < 0, 0 if x==0, 1 if x > 0``.

        Notes
        -----
        Like :func:`numpy.sign`.
        """
        return NDVar(np.sign(self.x), self.dims, name or self.name, self.info)

    def smooth(self, dim, window_size=None, window='hamming', mode='center', window_samples=None, fix_edges=False, name=None):
        """Smooth data by convolving it with a window

        Parameters
        ----------
        dim : str
            Dimension along which to smooth.
        window_size : scalar
            Size of the window (in dimension units, i.e., for time in
            seconds). For finite windows this is the full size of the window,
            for a gaussian window it is the standard deviation.
        window : str | tuple
            Window type, input to :func:`scipy.signal.get_window`. For example
            'boxcar', 'triang', 'hamming' (default). For dimensions with
            irregular spacing, such as :class:`SourceSpace`, only ``gaussian``
            is implemented.
        mode : 'left' | 'center' | 'right' | 'full'
            Alignment of the output to the input relative to the window:

            - ``left``: sample in the output corresponds to the left edge of
              the window.
            - ``center``: sample in the output corresponds to the center of
              the window.
            - ``right``: sample in the output corresponds to the right edge of
              the window.
            - ``full``: return the full convolution. This is only implemented
              for smoothing time axis.

        window_samples : scalar
            Size of the window in samples (this parameter is used to specify
            window size in array elements rather than in units of the dimension;
            it is mutually exclusive with ``window_size``).
        fix_edges : bool
            Standard convolution smears values around the edges resulting in
            some data loss. The ``fix_edges`` option renormalizes the smoothing
            window when it overlaps an edge to make sure that
            ``x.smooth('time').sum('time') == x.sum('time')``. Only implemented
            for ``mode='center'``).
        name : str
            Name for the smoothed NDVar.

        Returns
        -------
        smoothed_ndvar : NDVar
            NDVar with identical dimensions containing the smoothed data.

        Notes
        -----
        To perform Gaussian smoothing with a given full width at half maximum,
        the standard deviation can be calculated with the following conversion::

        >>> std = fwhm / (2 * (sqrt(2 * log(2))))
        """
        if (window_size is None) == (window_samples is None):
            desc = "" if window_size is None else f"{window_size=}, {window_samples=}: "
            raise TypeError(f"{desc}Must specify exactly one of window_size or window_samples")
        axis = self.get_axis(dim)
        dim_object = self.get_dim(dim)
        dims = self.dims
        if window == 'gaussian':
            if mode != 'center':
                raise ValueError(f"{mode=}; for gaussian smoothing, mode must be 'center'")
            elif fix_edges:
                raise NotImplementedError(f"{fix_edges=} with {window=}")
            elif window_samples is not None:
                if dim_object._adjacency_type == 'custom':
                    raise ValueError(f"{window_samples=} for dimension with adjacency not based on adjacency")
                raise NotImplementedError("Gaussian smoothing for window_samples")
            else:
                dist = dim_object._distances()
            m = gaussian_smoother(dist, window_size)
            x = np.tensordot(m, self.x, (1, axis))
            if axis:
                x = np.moveaxis(x, 0, axis)
        elif dim_object._adjacency_type == 'custom':
            raise ValueError(f"{window=} for {dim_object.__class__.__name__} dimension (must be 'gaussian')")
        else:
            if window_samples:
                n = window_samples
            elif dim == 'time':
                n = int(round(window_size / dim_object.tstep))
                if not n:
                    raise ValueError(f"{window_size=}: Window too small for sampling rate")
            else:
                raise NotImplementedError(f"{window=} for {dim_object.__class__.__name__} dimension")
            window = scipy.signal.get_window(window, n, False)
            window /= window.sum()
            window.shape = (1,) * axis + (n,) + (1,) * (self.ndim - axis - 1)
            if mode == 'center':
                x = scipy.signal.convolve(self.x, window, 'same')
                if fix_edges:
                    # Each original voxel should be used exactly 1 time
                    n0 = (n - 1) // 2  # how many input samples need to be fixed (left edge)
                    w_center = (n - 1) // 2  # window sample which is aligned to x
                    for i in range(n0):  # i = origin sample
                        window_i = window[aslice(axis, start=w_center - i)]
                        # renormalize window and subtract values of initial convolution
                        window_i = (window_i / window_i.sum()) - window_i
                        x[aslice(axis, stop=i + (n - w_center))] += window_i * self.x[aslice(axis, i, i + 1)]
                    n1 = n // 2  # samples to fix right edge
                    nx = x.shape[axis]
                    for i in range(n1):
                        window_i = window[aslice(axis, stop=w_center + i + 1)]
                        window_i = (window_i / window_i.sum()) - window_i
                        x[aslice(axis, start=nx - 1 - w_center - i)] += window_i * self.x[aslice(axis, nx - 1 - i, nx - i)]
            elif fix_edges:
                raise NotImplementedError(f"{fix_edges=} with {mode=}")
            else:
                x = scipy.signal.convolve(self.x, window, 'full')
                if mode == 'left':
                    x = x[aslice(axis, stop=self.shape[axis])]
                    # x[aslice(axis, stop=n)] *= weights
                elif mode == 'right':
                    x = x[aslice(axis, start=-self.shape[axis])]
                elif mode == 'full':
                    if not isinstance(dim_object, UTS):
                        raise NotImplementedError(f"{mode=} for {dim_object.__class__.__name__} dimension")
                    dims = list(dims)
                    tmin = dim_object.tmin - dim_object.tstep * floor((n - 1) / 2)
                    dims[axis] = UTS(tmin, dim_object.tstep, dim_object.nsamples + n - 1)
                else:
                    raise ValueError(f"{mode=}")
        return NDVar(x, dims, name or self.name, self.info)

    def std(
            self,
            axis: AxisArg = None,
            ddof: float = 0,
            **regions,
    ) -> Union[NDVar, Var, float]:
        """Compute the standard deviation over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the standard deviation over all values.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the standard deviation in specific elements (if the data
            has a case dimension, the standard deviation is computed for each
            case).
        ddof
            Means Delta Degrees of Freedom (see :func:`numpy.std`).
        **regions
            Regions over which to aggregate. For example, to get the STD
            between time=0.1 and time=0.2, use ``ndvar.std(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        std
            The standard deviation over specified dimensions. Return a Var if
            only the case dimension remains, and a float if the function
            collapses over all data.
        """
        return self._aggregate_over_dims(axis, regions, partial(np.std, ddof=ddof))

    def summary(self, *dims, **regions):
        r"""Aggregate specified dimensions.

        .. warning::
            Data is collapsed over the different dimensions in turn using the
            provided function with an axis argument. For certain functions
            this is not equivalent to collapsing over several axes concurrently
            (e.g., np.var).

        dimension:
            A whole dimension is specified as string argument. This
            dimension is collapsed over the whole range.
        range:
            A range within a dimension is specified through a keyword-argument.
            Only the data in the specified range is included. Use like the
            :py:meth:`.sub` method.


        **additional kwargs:**

        func : callable
            Function used to collapse the data. Needs to accept an "axis"
            kwarg (default: np.mean)
        name : str
            Name for the new NDVar.

        Returns
        -------
        summary : float | Var | NDVar
            Result of applying the summary function over specified dimensions.

        Examples
        --------
        Assuming ``data`` is a normal time series. Get the average in a time
        window::

            >>> y = data.summary(time=(.1, .2))

        Get the peak in a time window::

            >>> y = data.summary(time=(.1, .2), func=np.max)

        Assuming ``meg`` is an NDVar with dimensions time and sensor. Get the
        average across sensors 5, 6, and 8 in a time window::

            >>> roi = [5, 6, 8]
            >>> y = meg.summary(sensor=roi, time=(.1, .2))

        Get the peak in the same data:

            >>> roi = [5, 6, 8]
            >>> peak = meg.summary(sensor=roi, time=(.1, .2), func=np.max)

        Get the RMS over all sensors

            >>> meg_rms = meg.summary('sensor', func=rms)

        """
        if 'func' in regions:
            func = regions.pop('func')
        elif 'summary_func' in self.info:
            func = self.info['summary_func']
        else:
            func = np.mean
        name = regions.pop('name', None)
        if len(dims) + len(regions) == 0:
            dims = ('case',)

        if regions:
            dims = list(dims)
            data = self.sub(**regions)
            dims.extend(dim for dim in regions if data.has_dim(dim))
            return data.summary(*dims, func=func, name=name)
        else:
            x = self.x
            axes = [self._dim_2_ax[dim] for dim in dims]
            dims = list(self.dims)
            for axis in sorted(axes, reverse=True):
                x = func(x, axis=axis)
                dims.pop(axis)

            # update info for summary
            info = self.info
            if 'summary_info' in info:
                info = info.copy()
                info.update(info.pop('summary_info'))
            return self._package_aggregated_output(x, dims, name, info)

    def sub(self, *args, **kwargs):
        """Retrieve a slice through the NDVar.

        Return a new NDVar with a slice of the current NDVar's data.
        The slice is specified using arguments and keyword arguments.

        Indexes for dimensions can be either specified as arguments in the
        order of the data axes, or with dimension names as keywords; for::

            >>> x = datasets.get_uts(True)['utsnd']
            >>> x
            <NDVar 'utsnd': 60 case, 5 sensor, 100 time>
            >>> x.sub(time=0.1)
            <NDVar 'utsnd': 60 case, 5 sensor>

        ``x.sub(time=0.1)`` is equivalent to ``x.sub((), (), 0.1)`` and
        ``x[:, :, 0.1]``.

        Tuples are reserved for slicing and are treated like ``slice`` objects.
        Use lists for indexing arbitrary sequences of elements.

        The name of the new NDVar can be set with a ``name`` keyword
        (``x.sub(time=0.1, name="new_name")``). The default is the name of the
        current NDVar.
        """
        name = kwargs.pop('name', None)
        dims = list(self.dims)
        n_axes = len(dims)
        index = [FULL_SLICE] * n_axes
        index_args = [None] * n_axes
        add_axis = False

        # sequence args
        for i, arg in enumerate(args):
            if isinstance(arg, NDVar):
                if arg.has_case:
                    raise ValueError("NDVar with case dimension can not serve as NDVar index")
                dimax = self.get_axis(arg.dims[0].name)
                if index_args[dimax] is None:
                    index_args[dimax] = arg
                else:
                    raise IndexError(f"Index for {arg.dims[0].name} dimension specified twice.")
            elif arg is newaxis:
                if i > 0:
                    raise IndexError("numpy.newaxis (None) must be in first index position")
                elif self.has_case:
                    raise IndexError("NDVar already has case dimension")
                add_axis = True
            else:
                index_args[i] = arg

        # sequence kwargs
        for dimname, arg in kwargs.items():
            dimax = self.get_axis(dimname)
            if index_args[dimax] is None:
                index_args[dimax] = arg
            else:
                raise IndexError(f"Index for {dimname} dimension specified twice.")

        # process indexes
        for dimax, idx in enumerate(index_args):
            if idx is None:
                continue
            dim = self.dims[dimax]

            # find index
            idx = dim._array_index(idx)
            index[dimax] = idx

            # find corresponding dim
            if np.isscalar(idx):
                dims[dimax] = None
            elif dimax >= self.has_case:
                dims[dimax] = dim[idx]
            else:
                dims[dimax] = Case
        if add_axis:
            dims.insert(0, Case)

        # adjust index dimension
        if sum(isinstance(idx, EXPAND_INDEX_TYPES) for idx in index) > 1:
            ndim_increment = 0
            for i in range(n_axes - 1, -1, -1):
                idx = index[i]
                if ndim_increment and isinstance(idx, (slice, np.ndarray)):
                    if isinstance(idx, slice):
                        idx = slice_to_arange(idx, self.x.shape[i])
                    elif idx.dtype.kind == 'b':
                        idx = np.flatnonzero(idx)
                    index[i] = idx[FULL_AXIS_SLICE + (None,) * ndim_increment]

                if isinstance(idx, np.ndarray):
                    ndim_increment += 1

        # create NDVar
        x = self.x[tuple(index)]
        if add_axis:
            x = np.expand_dims(x, 0)
        dims = [dim for dim in dims if dim is not None]
        return self._package_aggregated_output(x, dims, name)

    def sum(self, axis: AxisArg = None, **regions) -> Union[NDVar, Var, float]:
        """Compute the sum over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the sum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the sum in specific elements (if the data has a case
            dimension, the sum is computed for each case).
        **regions
            Regions over which to aggregate. For example, to get the sum
            between time=0.1 and time=0.2, use ``ndvar.sum(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        sum
            The sum over specified dimensions. Return a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.
        """
        return self._aggregate_over_dims(axis, regions, np.sum)

    def threshold(self, v, tail=1, name=None):
        """Set all values below a threshold to 0.

        Parameters
        ----------
        v : scalar
            Threshold value.
        tail : -1 | 0 | 1
            Tailedness.
            1: set values below v to 0 (default);
            0: set values between -v and v to 0;
            -1: set values above v to 0.
        name : str
            Name of the output NDVar (default is the current name).
        """
        if tail == 0:
            v = abs(v)
            idx = self.x >= v
            np.logical_or(idx, self.x <= -v, idx)
        elif tail == 1:
            idx = self.x >= v
        elif tail == -1:
            idx = self.x <= v
        else:
            raise ValueError("Invalid value tail=%r; need -1, 0 or 1" % (tail,))
        return NDVar(np.where(idx, self.x, 0), self.dims, name or self.name, self.info)

    def var(
            self,
            axis: AxisArg = None,
            ddof: float = 0,
            **regions,
    ) -> Union[NDVar, Var, float]:
        """Compute the variance over given dimensions

        Parameters
        ----------
        axis
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the sum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the variance in specific elements (if the data has a case
            dimension, the variance is computed for each case).
        ddof
            Degrees of freedom (default 0; see :func:`numpy.var`).
        **regions
            Regions over which to aggregate. For example, to get the variance
            between time=0.1 and time=0.2, use ``ndvar.var(time=(0.1, 0.2))``.
        name : str
            Name of the output NDVar (default is the current name).

        Returns
        -------
        var
            The variance over specified dimensions. Return a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.
        """
        return self._aggregate_over_dims(axis, regions, partial(np.var, ddof=ddof))

    def nonzero(self):
        """Return indices where the NDVar is non-zero

        Like :func:`numpy.nonzero`.
        """
        return tuple(dim._dim_index(index) for dim, index in zip(self.dims, self.x.nonzero()))

    @classmethod
    def zeros(
            cls,
            dims: Union[NDVar, Dimension, Sequence[Dimension]],
            name: str = None,
            info: dict = None,
            dtype: DTypeLike = None,
    ):
        """A new :class:`NDVar` initialized with 0"""
        if isinstance(dims, NDVar):
            if dtype is None:
                dtype = dims.x.dtype
            dims = dims.dims
        else:
            if isinstance(dims, Dimension):
                dims = (dims,)
            if dtype is None:
                dtype = float
        shape = [len(dim) for dim in dims]
        return cls(np.zeros(shape, dtype), dims, name, info)


def extrema(x, axis=None):
    "Extract the extreme values in x"
    max = np.max(x, axis)
    min = np.min(x, axis)
    if np.isscalar(max):
        return max if abs(max) > abs(min) else min
    if isinstance(x, np.ma.masked_array):
        return np.ma.where(np.abs(max) >= np.abs(min), max, min)
    else:
        return np.where(np.abs(max) >= np.abs(min), max, min)


class Datalist(list):
    """:py:class:`list` subclass for including lists in in a Dataset.

    Parameters
    ----------
    items
        Content for the Datalist.
    name
        Name of the Datalist.
    fmt : 'repr' | 'str' | 'strlist'
        How to format items when converting Datasets to tables (default 'repr'
        uses the normal object representation).


    Notes
    -----
    Modifications:

     - adds certain methods that makes indexing behavior more similar to numpy
       and other data objects
     - blocks methods for in place modifications that would change the lists's
       length


    Examples
    --------
    Concise string representation:

    >>> l = [['a', 'b'], [], ['a']]
    >>> print(Datalist(l))
    [['a', 'b'], [], ['a']]
    >>> print(Datalist(l, fmt='strlist'))
    [[a, b], [], [a]]
    """
    _fmt = 'repr'  # for backwards compatibility with old pickles

    def __init__(
            self,
            items: Sequence[Any] = None,
            name: str = None,
            fmt: str = 'repr',
    ):
        if fmt not in ('repr', 'str', 'strlist'):
            raise ValueError(f"{fmt=}")

        self.name = name
        self._fmt = fmt
        if items:
            super(Datalist, self).__init__(items)
        else:
            super(Datalist, self).__init__()

    def __deepcopy__(self, memo):
        return Datalist((deepcopy(item, memo) for item in self), self.name, self._fmt)

    def __repr__(self):
        args = super(Datalist, self).__repr__()
        if self.name is not None:
            args += ', %s' % repr(self.name)
        if self._fmt != 'repr':
            args += ', fmt=%s' % repr(self._fmt)
        return "Datalist(%s)" % args

    def __str__(self):
        return "[%s]" % ', '.join(self._item_repr(i) for i in self)

    def _item_repr(self, item):
        if self._fmt == 'str':
            return str(item)
        elif self._fmt == 'repr':
            out = repr(item)
            if len(out) > 15:
                return out[:12] + '...'
            else:
                return out
        elif self._fmt == 'strlist':
            return "[%s]" % ', '.join(item)
        else:
            raise RuntimeError("Datalist._fmt=%s" % repr(self._fmt))

    def _summary(self, width=80):
        types = sorted({type(v) for v in self})
        if len(types) == 1:
            return f'{types[0].__name__}'
        items = [f'{t.__name__}:{sum(isinstance(v, t) for v in self)}' for t in types]
        n = 0
        for i, item in enumerate(items):
            if n + len(item) > width:
                return ', '.join(items[:i]) + '...'
            n += len(item) + 2
        return ', '.join(items)

    def __eq__(self, other):
        if len(self) != len(other):
            raise ValueError("Unequal length")
        return np.array([s == o for s, o in zip(self, other)])

    def __ne__(self, other):
        if len(self) != len(other):
            raise ValueError("Unequal length")
        return np.array([s != o for s, o in zip(self, other)])

    def __getitem__(self, index):
        if isinstance(index, Integral):
            return list.__getitem__(self, index)
        elif isinstance(index, slice):
            return Datalist(list.__getitem__(self, index), fmt=self._fmt)
        else:
            index = asindex(index)
            return Datalist(apply_numpy_index(self, index), fmt=self._fmt)

    def __setitem__(self, key, value):
        if isinstance(key, LIST_INDEX_TYPES):
            list.__setitem__(self, key, value)
        elif isinstance(key, np.ndarray):
            if key.dtype.kind == 'b':
                key = np.flatnonzero(key)
            elif key.dtype.kind != 'i':
                raise TypeError("Array index needs to be int or bool type")

            if np.iterable(value):
                if len(key) != len(value):
                    raise ValueError("Need one value per index when setting a "
                                     "range of entries in a Datalist.")
                for k, v in zip(key, value):
                    list.__setitem__(self, k, v)
            else:
                for k in key:
                    list.__setitem__(self, k, value)
        else:
            raise NotImplementedError("Datalist indexing with %s" % type(key))

    def __add__(self, other):
        return Datalist(super(Datalist, self).__add__(other), fmt=self._fmt)

    def __sub__(self, other):
        assert len(other) == len(self)
        return Datalist([s - o for s, o in zip(self, other)], self.name, self._fmt)

    def aggregate(
            self,
            x: CategorialArg,
            func: Union[Callable, str] = 'mean',
    ) -> Datalist:
        """
        Summarize cases for each cell in x

        Parameters
        ----------
        x
            Cells which to aggregate.
        func
            How to merge entries. Can be a :mod:`numpy` function such as
            :func:`numpy.mean`, or a :class:`str` special method:
            ``'mean'``: sum elements and divide by cell length (default).
        """
        if x is None:
            cell_xs = [self]
        elif len(x) != len(self):
            raise ValueError(f"x={dataobj_repr(x)}: Length mismatch, len(x)={len(x)}, len(self)={len(self)}")
        else:
            cell_xs = (self[x == cell] for cell in x.cells)

        x_out = []
        try:
            for x_cell in cell_xs:
                if len(x_cell) == 1:
                    xc = x_cell[0]
                elif isinstance(func, Callable):
                    xc = func(x_cell, axis=1)
                elif func == 'mean':
                    xc = reduce(operator.add, x_cell)
                    xc /= len(x_cell)
                else:
                    raise ValueError(f"{func=}")
                x_out.append(xc)
        except TypeError:
            raise TypeError(f"{dataobj_repr(self)}: Objects in Datalist do not support aggregating with {func=} (if aggregating a Dataset, try dropping this variable)")

        return Datalist(x_out, fmt=self._fmt)

    def __iadd__(self, other):
        return self + other

    def append(self, p_object):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def insert(self, index, p_object):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def pop(self, index=None):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def remove(self, value):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def _update_listlist(self, other):
        "Update list elements from another list of lists"
        if len(self) != len(other):
            raise ValueError("Unequal length")
        for i in range(len(self)):
            if any(item not in self[i] for item in other[i]):
                self[i] = sorted(set(self[i]).union(other[i]))


def assert_is_legal_dataset_key(key):
    if iskeyword(key):
        raise ValueError(f"{key!r} is a reserved keyword and can not be used as variable name in a Dataset")
    elif key.isidentifier():
        return
    else:
        raise ValueError(f"{key!r} is not a valid keyword and can not be used as variable name in a Dataset")


def as_legal_dataset_key(key):
    "Convert str to a legal dataset key"
    if iskeyword(key):
        return "%s_" % key
    elif key.isidentifier():
        return key
    else:
        if ' ' in key:
            key = key.replace(' ', '_')
        for c in string.punctuation:
            if c in key:
                key = key.replace(c, '_')

        if key == '':
            key = '_'
        elif key[0].isdigit():
            key = "_%s" % key

        if key.isidentifier():
            return key
        else:
            raise RuntimeError(f"Could not convert {key!r} to legal dataset key")


def cases_arg(cases, n_cases) -> Iterable:
    "Coerce cases argument to iterator"
    if isinstance(cases, Integral):
        if cases < 1:
            cases = n_cases + cases
            if cases < 0:
                raise ValueError(f"{cases=}: Can't get table for fewer than 0 cases")
        else:
            cases = min(cases, n_cases)
        return range(cases)
    elif isinstance(cases, Iterable):
        return cases
    else:
        raise TypeError(f"{cases=}")


class Dataset(dict):
    """Store multiple variables pertaining to a common set of measurement cases

    Parameters
    ----------
    items : dict | list
        Items in the Dataset (either specified as ``{key: data_object}``
        dictionary, or as ``[data_object]`` list in which data-object names will
        be used as keys).
        The Dataset stores the input items directly, without making a copy.
    name : str
        Name for the Dataset.
    caption : str
        Caption for the table.
    info : dict
        Info dictionary, can contain arbitrary entries and can be accessed
        as ``.info`` attribute after initialization. The Dataset makes a
        shallow copy.
    n_cases : int
        Specify the number of cases in the Dataset if no items are added
        upon initialization (by default the number is inferred when the
        fist item is added).


    Attributes
    ----------
    n_cases : None | int
        The number of cases in the Dataset (corresponding to the number of
        rows in the table representation). None if no variables have been
        added.
    n_items : int
        The number of items (variables) in the Dataset (corresponding to the
        number of columns in the table representation).


    Notes
    -----
    A Dataset represents a data table as a ``{variable_name: value_list}``
    dictionary. Each variable corresponds to a column, and each index in the
    value list corresponds to a row, or case.

    The Dataset class inherits basic behavior from :py:class:`dict`.
    Dictionary keys are enforced to be :py:class:`str` objects and should
    correspond to the variable names.
    As for a dictionary, The Dataset's length (``len(ds)``) reflects the number
    of variables in the Dataset (i.e., the number of rows).


    **Assigning data**

    The :class:`Dataset` assumes certain properties of the items that are
    assigned, for example they need to support :mod:`numpy` indexing.
    When assigning items that are not :mod:`eelbrain` data containers, they are
    coerced in the following manner:

    - 1-d :class:`numpy.ndarray` are coerced to :class:`Var`; other
      :class:`numpy.ndarray` are assigned as is
    - Objects conforming to the Python :class:`collections.Sequence` abstract
      base class are coerced to :class:`Datalist`
    - :class:`mne.Epochs` are assigned as is
    - For advanced use, additional classes can be assigned as is by extending the
      :attr:`Dataset._value_type_exceptions` class attribute tuple


    **Accessing Data**

    Standard indexing with :class:`str` is used to access the contained Var
    and Factor objects:

    - ``ds['var1']`` --> ``var1``.
    - ``ds['var1',]`` --> ``Dataset([var1])``.
    - ``ds['var1', 'var2']`` --> ``Dataset([var1, var2])``

    When indexing numerically, the first index defines cases (rows):

    - ``ds[1]`` --> row 1
    - ``ds[1:5]`` or ``ds[1,2,3,4]`` --> rows 1 through 4
    - ``ds[1, 5, 6, 9]`` or ``ds[[1, 5, 6, 9]]`` --> rows 1, 5, 6 and 9

    The second index accesses columns, so case indexing can be combined with
    column indexing:

     - ``ds[:4, :2]`` --> first 4 rows of first 2 columns

    Index a single case retrieves an individual case as ``{name: value}``
    dictionaries:

    - ``ds[1]`` --> ``{'var': 1, 'factor': 'value', ...}``

    The :meth:`.itercases` method can be used to iterate over cases as
    :class:`dict`.


    **Naming**

    While Var and Factor objects themselves need not be named, they need
    to be named when added to a Dataset. This can be done by a) adding a
    name when initializing the Dataset::

        >>> ds = Dataset((('v1', var1), ('v2', var2)))

    or b) by adding the Var or Factor with a key::

        >>> ds['v3'] = var3

    If a Var/Factor that is added to a Dataset does not have a name, the new
    key is automatically assigned to the Var/Factor's ``.name`` attribute.


    Examples
    --------
    - :ref:`exa-intro`: basic functionality
    - :ref:`exa-dataset`: how to construct datasets
    """
    _value_type_exceptions = (MNE_EPOCHS,)

    @staticmethod
    def _is_kv_pair(item):
        return isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)

    def __init__(self, items=None, name=None, caption=None, info=None, n_cases=None):
        dict.__init__(self)  # skips __setitem__()
        self.n_cases = None if n_cases is None else int(n_cases)
        self.name = name
        self.info = {} if info is None else dict(info)
        self._caption = caption
        # initial items
        if items is None:
            return
        elif isinstance(items, dict):
            items_ = items.items()
        else:
            if not isinstance(items, (list, tuple)):
                items = list(items)
            if all(map(self._is_kv_pair, items)):
                items_ = items
            else:
                if any(getattr(item, 'name', None) is None for item in items):
                    raise ValueError(f"{items!r}: when specified without keys, all items need to be named")
                items_ = [(item.name, item) for item in items]
        for key, value in items_:
            self.__setitem__(key, value)

    def __setstate__(self, state):
        # for backwards compatibility
        self.name = state['name']
        self.info = state['info']
        self._caption = state.get('caption', None)

    def __reduce__(self):
        return self.__class__, (tuple(self.items()), self.name, self._caption,
                                self.info, self.n_cases)

    def __delitem__(self, key):
        if isinstance(key, str):
            dict.__delitem__(self, key)
        elif isinstance(key, tuple):
            for k in key:
                dict.__delitem__(self, k)
        else:
            raise KeyError(key)

    def __getitem__(self, index):
        """
        possible::

            >>> ds[9]        (int) -> dictionary for one case
            >>> ds[9:12]     (slice) -> subset with those cases
            >>> ds[[9, 10, 11]]     (list) -> subset with those cases
            >>> ds['MEG1']  (strings) -> Var
            >>> ds['MEG1', 'MEG2']  (list of strings) -> list of vars; can be nested!

        """
        if isinstance(index, slice):
            return self.sub(index)
        elif isinstance(index, str):
            return super(Dataset, self).__getitem__(index)
        elif isinstance(index, Integral):
            return self.get_case(index)
        elif not np.iterable(index):
            raise KeyError("Invalid index for Dataset: %r" % index)
        elif len(index) == 0:
            return self.sub(index)
        elif all(isinstance(item, str) for item in index):
            return self.sub(keys=index)
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise KeyError("Invalid index for Dataset: %s" % repr(index))

            i0, i1 = index
            if isinstance(i0, str):
                return self[i0][i1]
            elif isinstance(i1, str):
                return self[i1][i0]
            elif np.iterable(i0) and isinstance(i0[0], str):
                return self[i1, i0]
            elif np.iterable(i1) and all(isinstance(item, str) for item in i1):
                keys = i1
            else:
                keys = Datalist(self.keys())[i1]
                if isinstance(keys, str):
                    return self[i1][i0]
            return Dataset(((k, self[k][i0]) for k in keys), self.name,
                           self._caption, self.info)
        else:
            return self.sub(index)

    def __repr__(self):
        if self.n_cases is None:
            items = []
            if self.name:
                items.append('name=%r' % self.name)
            if self.info:
                info = repr(self.info)
                if len(info) > 60:
                    info = '<...>'
                items.append('info=%s' % info)
            item_repr = ', '.join(items)
            return f'{self.__class__.__name__}({item_repr})'

        items = []
        for key in self:
            v = self[key]
            if isinstance(v, Var):
                lbl = 'V'
            elif isinstance(v, Factor):
                lbl = 'F'
            elif isinstance(v, NDVar):
                lbl = 'Vnd'
            else:
                lbl = type(v).__name__
            item = f'{key!r}:{lbl}'
            if isdataobject(v) and v.name != key:
                item += f'<{v.name!r}>'
            items.append(item)
        name = '' if self.name is None else f' {self.name!r}'
        item_repr = ', '.join(items)
        return f"<{self.__class__.__name__}{name} ({self.n_cases} cases) {item_repr}>"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise NotImplementedError
        p.text(self.__repr__())

    def _repr_html_(self):
        return self._display_table()._repr_html_()

    def __setitem__(self, index, item):
        if isinstance(index, str):
            assert_is_legal_dataset_key(index)

            # coerce to data-object
            if isdataobject(item) or isinstance(object, Datalist):
                item.name = index
                if isinstance(item, NDVar) and not item.has_case:
                    if self.n_cases == 1:  # coerce case-less NDVar
                        item = item[None]
                        n = 1
                    else:
                        n = 0
                else:
                    n = len(item)
            else:
                if isinstance(item, np.ndarray):
                    if item.ndim == 1:
                        item = Var(item, index)
                elif isinstance(item, self._value_type_exceptions):
                    pass
                elif isinstance(item, Sequence):
                    item = Datalist(item, index)
                else:
                    raise TypeError(f"{item!r}: Unsupported type for Dataset")
                n = len(item)

            # make sure the item has the right length
            if self.n_cases is None:
                self.n_cases = n
            elif self.n_cases != n:
                raise ValueError(f"Can not assign item to Dataset[{index!r}]. The item`s length {n} is different from the number of cases in the Dataset {self.n_cases}.")

            super(Dataset, self).__setitem__(index, item)
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError(f"{index}: Dataset indexes can have at most two components; direct access to NDVars is not implemented")
            idx, key = index
            if isinstance(idx, str):
                key, idx = idx, key
            elif not isinstance(key, str):
                raise IndexError(f"{index}; index needs one str")

            if key in self:
                self[key][idx] = item
            elif isinstance(idx, slice) and idx.start is None and idx.stop is None and idx.step is None:
                if isdataobject(item):
                    if isinstance(item, NDVar) and not item.has_case:
                        if self.n_cases is None:
                            raise IndexError("Can't assign slice of empty Dataset")
                        item = item.repeat(self.n_cases)
                    self[key] = item
                elif self.n_cases is None:
                    raise IndexError("Can't assign slice of empty Dataset")
                elif isinstance(item, str):
                    self[key] = Factor([item], repeat=self.n_cases)
                elif np.isscalar(item):
                    self[key] = Var([item], repeat=self.n_cases)
                else:
                    raise IndexError(f"{item!r} is not supported for slice-assignment of a new variable. Use a str for a new Factor or a scalar for a new Var.")
            else:
                raise IndexError("When assigning a new item in a Dataset, all values need to be set (ds[:,'name'] = ...)")
        else:
            raise IndexError(f"{index}: not a valid Dataset index")

    def _display_table(self, cases=0, title=None):
        items = []  # caption
        if cases == 0 and self.n_cases > preferences['dataset_str_n_cases']:
            cases = preferences['dataset_str_n_cases']
            items.append(f"... ({cases} of {self.n_cases} rows shown, use .as_table() to see the whole Dataset)")
        ndvars = [key for key, v in self.items() if isinstance(v, NDVar)]
        if ndvars:
            items.append(f"NDVars: {', '.join(ndvars)}")
        if items:
            caption = '; '.join(items)
        else:
            caption = None
        return self.as_table(cases, '%.5g', midrule=True, count=True, title=title, caption=caption, lfmt=True)

    def __str__(self):
        if sum(isuv(i) or isdatalist(i) for i in self.values()) == 0:
            return self.__repr__()
        return str(self._display_table())

    def _check_n_cases(self, x, empty_ok=True):
        """Check that an input argument has the appropriate length.

        Also raise an error if empty_ok is False and the Dataset is empty.
        """
        if self.n_cases is None:
            if not empty_ok:
                raise RuntimeError("Dataset is empty.")
        elif self.n_cases != len(x):
            raise ValueError(
                f"{dataobj_repr(x)} with length {len(x)}: The Dataset has a "
                f"different length ({self.n_cases})")

    @staticmethod
    def as_key(name: Optional[str], default: str = None):
        """Convert a string ``name`` to a legal dataset key

        This is a shortcut to simplify storing varaibles with non-compliant
        names, consisting mostly of replacing invalid characters with '_'.
        Note that the result is not unique.

        If ``name`` is ``None``, return ``default``, or raise a ValueError if
        ``default`` is not set.

        Examples
        --------
        >>> Dataset.as_key('var-1|2')
        'var_1_2'
        """
        if name is None:
            if default is None:
                raise ValueError("Unnamed variable")
            return default
        return as_legal_dataset_key(name)

    def add(self, item, replace=False):
        """``ds.add(item)`` -> ``ds[item.name] = item``

        unless the Dataset already contains a variable named item.name, in
        which case a KeyError is raised. In order to replace existing
        variables, set ``replace`` to True::

            >>> ds.add(item, True)

        """
        if not isdataobject(item):
            raise ValueError("Not a valid data-object: %r" % item)
        elif item.name is None:
            raise ValueError("Dataset.add(obj) can only take named objects "
                             "(obj.name can not be None)")
        elif (item.name in self) and not replace:
            raise KeyError("Dataset already contains variable named %r" % item.name)
        else:
            self[item.name] = item

    def add_empty_var(self, name, dtype=np.float64):
        """Create an empty variable in the dataset

        Parameters
        ----------
        name : str
            Name for the new variable.
        dtype : numpy dtype
            Data type of the new variable (default is float64).

        Returns
        -------
        var : Var
            The new variable.
        """
        if self.n_cases is None:
            err = "Can't add variable to a Dataset without length"
            raise RuntimeError(err)
        x = np.empty(self.n_cases, dtype=dtype)
        v = Var(x)
        self[name] = v
        return v

    def as_dataframe(self):
        """Convert to a :class:`pandas.DataFrame`

        Notes
        -----
        Only includes :class:`Var` and :class:`Factor` items.
        """
        import pandas

        data = {}
        for key, column in self.items():
            if isinstance(column, Var):
                data[key] = column.x
            elif isinstance(column, Factor):
                categories = [column._labels.get(i, '') for i in range(max(column._labels) + 1)]
                data[key] = pandas.Categorical.from_codes(column.x, categories)
        return pandas.DataFrame(data)

    def as_table(
            self,
            cases: Union[int, Iterable[int]] = 0,
            fmt: str = '%.6g',
            sfmt: Optional[str] = '%s',
            sort: bool = False,
            header: bool = True,
            midrule: bool = False,
            count: bool = False,
            title: str = None,
            caption: str = None,
            ifmt: str = '%s',
            bfmt: str = '%s',
            lfmt: bool = False,
            nan: str = None,
    ):
        r"""
        Create an fmtxt.Table containing all Vars and Factors in the Dataset.

        Can be used for exporting in different formats such as csv.

        Parameters
        ----------
        cases
            Cases to include (int includes that many cases from the beginning,
            0 includes all; negative number works like negative indexing).
        fmt
            Format string for float variables (default ``'%.6g'``).
        sfmt
            Formatting for strings (None -> code; default ``'%s'``).
        sort
            Sort the columns alphabetically.
        header
            Include the varibale names as a header row.
        midrule
            print a midrule after table header.
        count
            Add an initial column containing the case number.
        title
            Title for the table.
        caption
            Caption for the table (default is the Dataset's caption).
        ifmt
            Formatting for integers (default ``'%s'``).
        bfmt
            Formatting for booleans (default ``'%s'``).
        lfmt
            Include Datalists.
        nan
            Label to write for NaN.
        """
        cases = cases_arg(cases, self.n_cases)
        if cases is None:
            return fmtxt.Table('')
        keys = [k for k, v in self.items() if isuv(v) or (lfmt and isdatalist(v))]
        if sort:
            keys = sorted(keys)

        if caption is None:
            caption = self._caption

        columns = {}
        if count:
            columns['#'] = numpy.arange(self.n_cases)[cases]
        for key in keys:
            data_obj = self[cases, key]
            if isinstance(data_obj, Factor):
                if sfmt is None:
                    columns[key] = data_obj.x
                elif sfmt == '%s':
                    columns[key] = data_obj
                else:
                    columns[key] = [sfmt % v for v in data_obj]
            elif isdatalist(data_obj):
                columns[key] = [data_obj._item_repr(v) for v in data_obj]
            else:
                if isintvar(data_obj):
                    fmt_ = ifmt
                elif isboolvar(data_obj):
                    fmt_ = bfmt
                else:
                    if nan is not None:
                        columns[key] = [nan if isnan(v) else fmtxt.Number(v, fmt=fmt) for v in data_obj]
                        continue
                    fmt_ = fmt
                columns[key] = [fmtxt.Number(v, fmt=fmt_) for v in data_obj]

        table = fmtxt.Table('l' * len(columns), True, title, caption)
        if header:
            for name in columns:
                table.cell(name)
            if midrule:
                table.midrule()

        for row in zip(*columns.values()):
            table.cells(*row)

        return table

    def _asfmtext(self, **_):
        return self.as_table()

    def eval(self, expression):
        """
        Evaluate an expression involving items stored in the Dataset.

        Parameters
        ----------
        expression : str
            Python expression to evaluate.

        Notes
        -----
        ``ds.eval(expression)`` is equivalent to
        ``eval(expression, globals, ds)`` with ``globals=numpy`` plus some
        Eelbrain functions.

        Examples
        --------
        In a Dataset containing factors 'A' and 'B'::

            >>> ds.eval('A % B')
            A % B

        """
        if not isinstance(expression, str):
            raise TypeError(f"{expression=}: needs str")
        try:
            return eval(expression, EVAL_CONTEXT, self)
        except Exception as exception:
            ds_repr = f"Dataset {self.name!r}" if self.name else "Dataset"
            raise EvalError(expression, exception, ds_repr) from None

    @classmethod
    def from_caselist(
            cls,
            names: Sequence[str],
            cases: Sequence[Sequence[str, Number, NDVar]],
            name: str = None,
            caption: str = None,
            info: dict = None,
            random: Union[str, Collection[str]] = None,
            check_dims: bool = True,
            dim_intersection: bool = False,
            to_list: bool = False,
    ):
        """Create a Dataset from a list of cases

        Parameters
        ----------
        names
            Names for the variables.
        cases
            A sequence of cases, whereby each case is itself represented as a
            sequence of values (str or scalar). Variable type (Factor or Var)
            is inferred from whether values are str or not.
        name
            Name for the Dataset.
        caption
            Caption for the table.
        info
            Info dictionary, can contain arbitrary entries and can be accessed
            as ``.info`` attribute after initialization. The Dataset makes a
            shallow copy.
        random
            Names of the columns that should be assigned as random factor.
        check_dims
            For :class:`NDVar` columns, check dimensions for consistency between
            cases (e.g., channel locations in a :class:`Sensor`). Default is
            ``True``. Set to ``False`` to ignore mismatches.
        dim_intersection
            Only applies to combining :class:`NDVar`: normally, when :class:`NDVar`
            have mismatching dimensions, a :exc:`DimensionMismatchError` is raised.
            With ``dim_intersection=True``, the intersection is used instead.
        to_list
            Only applies to combining :class:`NDVar`: normally, when :class:`NDVar`
            have mismatching dimensions, a :exc:`DimensionMismatchError` is raised.
            With ``to_list=True``, the :class:`NDVar` are added as :class:`list` of
            :class:`NDVar` instead.

        Examples
        --------
        See :ref:`exa-dataset`
        """
        if isinstance(names, Iterator):
            names = list(names)
        if isinstance(cases, Iterator):
            cases = list(cases)
        if len(cases) == 0:
            raise ValueError(f"{cases=}: no data")
        if isinstance(random, str):
            random = [random]
        elif isinstance(random, Iterator):
            random = list(random)
        elif random is None:
            random = []
        n_cases = set(map(len, cases))
        if len(n_cases) > 1:
            raise ValueError('not all cases have same length')
        n_cases = n_cases.pop()
        if len(names) != n_cases:
            raise ValueError(f'{names=}: {len(names)} names but {n_cases} cases')
        items = {key: combine([case[i] for case in cases], check_dims=check_dims, dim_intersection=dim_intersection, to_list=to_list) for i, key in enumerate(names)}

        for key in random:
            item = items[key]
            if isinstance(item, Factor):
                item.random = True
            else:
                raise ValueError(f"{random=}: {key!r} is not a Factor but {item}")
        return cls(items, name, caption, info)

    @classmethod
    def from_dataframe(
            cls,
            df,
            random: Union[str, Collection[str]] = None,
            skip: Union[str, Collection[str]] = None,
    ) -> Dataset:
        """Create a dataset from a :class:`pandas.DataFrame`

        Parameters
        ----------
        df : pandas.DataFrame
            The input data frame.
        random
            Names of the columns that should be assigned as random factor.
        """
        import pandas

        if isinstance(random, str):
            random = [random]
        elif isinstance(random, Iterator):
            random = list(random)
        elif random is None:
            random = []

        if isinstance(skip, str):
            skip = [skip]
        elif isinstance(skip, Iterator):
            skip = list(skip)
        elif skip is None:
            skip = []

        items = {}
        for name, item in df.items():
            if name in skip:
                continue
            key = cls.as_key(name)
            if isinstance(item, pandas.Series):
                items[key] = Var(item, name=name)
            elif isinstance(item, pandas.Categorical):
                items[key] = Factor(item, name=name)

        for key in random:
            item = items[key]
            if isinstance(item, Factor):
                item.random = True
            else:
                raise ValueError(f"{random=}: {key!r} is not a Factor but {item}")
        return cls(items)

    @classmethod
    def from_r(cls, name) -> Dataset:
        """Create a Dataset from an R data frame through ``rpy2``

        Parameters
        ----------
        name : str
            Name of the dataframe in R.

        Examples
        --------
        Getting an example dataset from R:

        >>> from rpy2.robjects import r
        >>> r('data(sleep)')
        >>> ds = Dataset.from_r('sleep')
        >>> print(ds)
        extra   group   ID
        ------------------
        0.7     1       1
        -1.6    1       2
        -0.2    1       3
        -1.2    1       4
        -0.1    1       5
        3.4     1       6
        3.7     1       7
        0.8     1       8
        0       1       9
        2       1       10
        1.9     2       1
        0.8     2       2
        1.1     2       3
        0.1     2       4
        -0.1    2       5
        4.4     2       6
        5.5     2       7
        1.6     2       8
        4.6     2       9
        3.4     2       10
        """
        from rpy2 import robjects as ro
        df = ro.r[name]
        if not isinstance(df, ro.DataFrame):
            raise ValueError("R object %r is not a DataFrame")
        ds = cls(name=name)
        for item_name, item in df.items():
            if isinstance(item, ro.FactorVector):
                x = np.array(item)
                labels = {i: l for i, l in enumerate(item.levels, 1)}
                ds[item_name] = Factor(x, labels=labels)
            elif isinstance(item, (ro.FloatVector, ro.IntVector)):
                x = np.array(item)
                ds[item_name] = Var(x)
            else:
                raise NotImplementedError(str(type(item)))
        return ds

    def get_case(self, i):
        "The i'th case as a dictionary"
        return {k: v[i] for k, v in self.items()}

    def get_subsets_by(self, x, exclude=(), name='{name}[{cell}]'):
        """Split the Dataset by the cells of ``x``

        Parameters
        ----------
        x : categorial
            Model defining cells into which to split the dataset.
        exclude : sequence of str
            Cells of ``x`` which should be ignored.
        name : str
            Name for the new datasets (formatted with ``self.name`` and
            ``cell``).

        Returns
        -------
        sub_datasets : dict
            ``{cell: sub_dataset}`` dictionary.
        """
        if isinstance(x, str):
            x = self.eval(x)
        return {cell: self.sub(x == cell, name.format(name=self.name, cell=cell)) for
                cell in x.cells if cell not in exclude}

    def aggregate(
            self,
            x: CategorialArg = None,
            drop_empty: bool = True,
            name: str = None,
            count: Union[bool, str] = 'n',
            drop_bad: bool = False,
            drop: Sequence[str] = (),
            equal_count: bool = False,
            never_drop: Sequence[str] = (),
            func: Callable = np.mean,
    ) -> Dataset:
        """
        Return a Dataset with one case for each cell in x.

        Parameters
        ----------
        x
            Model defining cells to which to reduce cases. By default (``None``)
            the Dataset is reduced to a single case.
        drop_empty
            Drops empty cells in ``x`` from the Dataset. Currently has to be
            ``True``.
        name
            Name of the new Dataset.
        count
            Add a variable with this name to the new Dataset, containing the
            number of cases in each cell of ``x``.
        drop_bad
            Drop bad items: silently drop any items for which compression
            raises an error. This concerns primarily factors with non-unique
            values for cells in x (if drop_bad is False, an error is raised
            when such a Factor is encountered)
        drop
            Additional data-objects to drop.
        equal_count
            Make sure the same number of rows go into each average. First, the
            cell with the smallest number of rows is determined. Then, for each
            cell, rows beyond that number are dropped.
        never_drop
            Raise an error if the ``drop_bad=True`` setting would lead to
            dropping a variable whose name is in ``never_drop``.
        func
            Function for aggregating numerical variables (:class:`Var` and
            :class:`NDVar`); default is :func:`numpy.mean`

        Notes
        -----
        Handle mne Epoch objects by creating a list with an mne Evoked object
        for each cell.
        """
        if not drop_empty:
            raise NotImplementedError(f'{drop_empty=}')

        if isinstance(x, str):
            if equal_count:
                return self.equalize_counts(x).aggregate(x, drop_empty, name, count, drop_bad, drop, False, never_drop, func)
            elif x == '':
                x = None
        elif equal_count:
            raise NotImplementedError('For equal_count, x needs to be specified as str')

        if x is not None:
            x = ascategorial(x, data=self)

        if name and '{name}' in name:
            name = name.format(name=self.name)
        ds = Dataset(name=name, info=self.info)

        if count:
            if x is None:
                ds[count] = Var([self.n_cases])
            else:
                ds[count] = Var([np.sum(x == cell) for cell in x.cells])

        for k, v in self.items():
            if k in drop:
                continue
            try:
                if isnumeric(v):
                    ds[k] = v.aggregate(x, func=func)
                elif hasattr(v, 'aggregate'):
                    ds[k] = v.aggregate(x)
                elif isinstance(v, MNE_EPOCHS):
                    if x is None:
                        ds[k] = [v.average()]
                    else:
                        ds[k] = [v[x == cell].average() for cell in x.cells]
                else:
                    raise TypeError(f"{v}: unsupported type for Dataset.aggregate()")
            except BaseException:
                if drop_bad and k not in never_drop:
                    pass
                else:
                    raise

        return ds

    def copy(self, name=None):
        """Create a shallow copy of the dataset

        Parameters
        ----------
        name : str
            Name for the new dataset (default is ``self.name``).
        """
        return Dataset(self, name or self.name, self._caption, self.info, self.n_cases)

    def equalize_counts(self, x, n=None):
        """Create a copy of the Dataset with equal counts in each cell of x

        Parameters
        ----------
        x : categorial
            Model which defines the cells in which to equalize the counts.
        n : int
            Number of cases per cell (the default is the maximum possible, i.e.
            the number of cases in the cell with the least number of cases).
            Negative numbers to subtract from maximum possible.

        Returns
        -------
        equalized_ds : Dataset
            Dataset with equal number of cases in each cell of x.

        Notes
        -----
        First, the cell with the smallest number of rows is determined (empty
        cells are ignored). Then, for each cell, rows beyond that number are
        dropped.
        """
        x = ascategorial(x, data=self)
        self._check_n_cases(x, empty_ok=False)
        indexes = np.array([x == cell for cell in x.cells])
        n_by_cell = indexes.sum(1)
        n_max = np.setdiff1d(n_by_cell, [0]).min()
        if n is None:
            n_ = n_max
        elif n < 0:
            n_ = n_max + n
        else:
            n_ = n

        if n_ < 0 or n_ > n_max:
            raise ValueError("Invalid value n=%i; the maximum numer of cases "
                             "per cell is %i" % (n, n_max))

        for index in indexes:
            np.logical_and(index, index.cumsum() <= n_, index)
        index = indexes.any(0)
        return self[index]

    def head(self, n=10, title=None):
        "Table with the first n cases in the Dataset"
        return self._display_table(n, title)

    def index(self, name='index', start=0):
        """Add an index to the Dataset (i.e., ``range(n_cases)``)

        Parameters
        ----------
        name : str
            Name of the new index variable.
        start : int
            Number at which to start the index.
        """
        if not isinstance(name, str):
            raise TypeError("name=%r" % (name,))
        self[name] = Var(np.arange(start, self.n_cases + start))

    def itercases(self, start=None, stop=None):
        "Iterate through cases (each case represented as a dict)"
        if start is None:
            start = 0

        if stop is None:
            stop = self.n_cases
        elif stop < 0:
            stop = self.n_cases - stop

        for i in range(start, stop):
            yield self.get_case(i)

    @property
    def n_items(self):
        return super(Dataset, self).__len__()

    def rename(self, old, new):
        """Shortcut to rename a data-object in the Dataset.

        Parameters
        ----------
        old : str
            Current name of the data-object.
        new : str
            New name for the data-object.
        """
        if old not in self:
            raise KeyError(f"No item named {old!r}")
        if new in self:
            raise ValueError(f"Dataset already has variable named {new!r}")
        assert_is_legal_dataset_key(new)
        self[new] = self.pop(old)

    def repeat(
            self,
            repeats: Union[int, Sequence[[int]]],
            name: str = None,
    ):
        """
        Return a new Dataset with each row repeated ``n`` times.

        Parameters
        ----------
        repeats
            Number of repeats, either a constant or a different number for each
            element.
        name
            Name for the new Dataset.
        """
        if self.n_cases is None:
            raise RuntimeError("Can't repeat Dataset with unspecified n_cases")
        if name and '{name}' in name:
            name = name.format(name=self.name)

        if isinstance(repeats, Integral):
            n_cases = self.n_cases * repeats
        else:
            n_cases = sum(repeats)

        items = ((k, v.repeat(repeats)) for k, v in self.items())
        return Dataset(items, name, self._caption, self.info, n_cases)

    @property
    def shape(self):
        return (self.n_cases, self.n_items)

    def sort(self, order, descending=False):
        """Sort the Dataset in place.

        Parameters
        ----------
        order : str | data-object
            Data object (Var, Factor or interactions) according to whose values
            to sort the Dataset, or its name in the Dataset.
        descending : bool
            Sort in descending instead of an ascending order.

        See Also
        --------
        .sort_index : Create an index that could be used to sort the Dataset
        .sorted : Create a sorted copy of the Dataset
        """
        idx = self.sort_index(order, descending)
        for k in self:
            self[k] = self[k][idx]

    def sort_index(self, order, descending=False):
        """Create an index that could be used to sort the Dataset.

        Parameters
        ----------
        order : str | data-object
            Data object (Var, Factor or interactions) according to whose values
            to sort the Dataset, or its name in the Dataset.
        descending : bool
            Sort in descending instead of an ascending order.

        See Also
        --------
        .sort : sort the Dataset in place
        .sorted : Create a sorted copy of the Dataset
        """
        if isinstance(order, str):
            order = self.eval(order)

        if not len(order) == self.n_cases:
            raise ValueError(f"Order must be of same length as Dataset; got length {len(order)}")

        return order.sort_index(descending=descending)

    def save(self):
        """Shortcut to save the Dataset, will display a system file dialog

        Notes
        -----
        Use specific save methods for more options.

        See Also
        --------
        .save_pickled : Pickle the Dataset
        .save_txt : Save as text file
        .save_tex : Save as teX table
        .as_table : Create a table with more control over formatting
        """
        title = "Save Dataset"
        if self.name:
            title += ' %s' % self.name
        msg = ""
        filetypes = [_pickled_ds_wildcard, _tsv_wildcard, _tex_wildcard]
        path = ui.ask_saveas(title, msg, filetypes, defaultFile=self.name)
        _, ext = os.path.splitext(path)
        if ext == '.pickle':
            self.save_pickled(path)
        elif ext == '.txt':
            self.save_txt(path)
        elif ext == '.tex':
            self.save_tex(path)
        else:
            raise ValueError(f"Unrecognized extension: {ext!r}. Needs to be .pickle, .txt or .tex.")

    def save_rtf(self, path=None, fmt='%.3g'):
        """Save the Dataset as TeX table.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.tex' is appended.
        fmt : format string
            Formatting for scalar values.
        """
        table = self.as_table(fmt=fmt)
        table.save_rtf(path)

    def save_tex(self, path=None, fmt='%.3g', header=True, midrule=True):
        """Save the Dataset as TeX table.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.tex' is appended.
        fmt : format string
            Formatting for scalar values.
        header : bool
            Include the varibale names as a header row.
        midrule : bool
            print a midrule after table header.
        """
        if not isinstance(path, str):
            title = "Save Dataset"
            if self.name:
                title += ' %s' % self.name
            title += " as TeX Table"
            msg = ""
            path = ui.ask_saveas(title, msg, [_tex_wildcard],
                                 defaultFile=self.name)

        _, ext = os.path.splitext(path)
        if not ext:
            path += '.tex'

        table = self.as_table(fmt=fmt, header=header, midrule=midrule)
        table.save_tex(path)

    def save_txt(
            self,
            path: PathArg = None,
            fmt: str = '%s',
            delimiter: str = '\t',
            header: bool = True,
            nan: str = None,
    ):
        """Save the Dataset as text file.

        Parameters
        ----------
        path
            Target file name (by default, a Save As dialog is displayed). If
            ``path`` is missing an extension, ``'.txt'`` is appended.
        fmt
            Formatting for scalar values.
        delimiter
            Column delimiter (default is tab).
        header
            write the variables' names in the first line
        nan
            Label to write for NaN.
        """
        if path is None:
            path = ui.ask_saveas(f"Save {self.name or 'Dataset'} as Text", "", [_tsv_wildcard], defaultFile=self.name)
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.txt')

        table = self.as_table(fmt=fmt, header=header, nan=nan)
        table.save_tsv(path, delimiter, fmt)

    def save_pickled(self, path=None):
        """Pickle the Dataset.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.pickle' is appended.
        """
        if not isinstance(path, str):
            title = "Pickle Dataset"
            if self.name:
                title += ' %s' % self.name
            msg = ""
            path = ui.ask_saveas(title, msg, [_pickled_ds_wildcard],
                                 defaultFile=self.name)

        _, ext = os.path.splitext(path)
        if not ext:
            path += '.pickle'

        with open(path, 'wb') as fid:
            pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)

    def sorted(self, order, descending=False):
        """Create an sorted copy of the Dataset.

        Parameters
        ----------
        order : str | data-object
            Data object (Var, Factor or interactions) according to whose values
            to sort the Dataset, or its name in the Dataset.
        descending : bool
            Sort in descending instead of an ascending order.

        See Also
        --------
        .sort : sort the Dataset in place
        .sort_index : Create an index that could be used to sort the Dataset
        """
        idx = self.sort_index(order, descending)
        return self[idx]

    def sub(self, index=None, keys=None, name=None):
        """Access a subset of the data in the Dataset.

        Parameters
        ----------
        index : int | array | str
            Index for selecting a subset of cases. Can be an valid numpy index
            or a string (the name of a variable in Dataset, or an expression
            to be evaluated in the Dataset's namespace).
        keys : sequence of str | str
            Only include items with those keys (default all items). Use a
            :class:`str` to retrieve a single item directly.
        name : str
            name for the new Dataset.

        Returns
        -------
        data : Dataset | data_object
            Either the :class:`Dataset` with cases restricted to ``index``, or,
            if ``key`` is a :class:`str`, a single item restricted to ``index``.

        Notes
        -----
        Index is passed on to numpy objects, which means that advanced indexing
        always returns a copy of the data, whereas basic slicing (using slices)
        returns a view.
        """
        if index is None:
            if keys is None:
                return self.copy(name)
            elif isinstance(keys, str):
                return dict.__getitem__(self, keys)
            else:
                items = {k: dict.__getitem__(self, k) for k in keys}
        elif isinstance(index, Integral):
            if keys is None:
                return self.get_case(index)
            elif isinstance(keys, str):
                return dict.__getitem__(self, keys)[index]
            else:
                return {k: dict.__getitem__(self, k)[index] for k in keys}
        else:
            if isinstance(index, str):
                index = self.eval(index)
            index = asindex(index)
            if keys is None:
                keys = self.keys()
            elif isinstance(keys, str):
                return dict.__getitem__(self, keys)[index]
            items = {k: dict.__getitem__(self, k)[index] for k in keys}

        return Dataset(items, name or self.name, self._caption, self.info)

    def summary(self, width: int = None):
        """A summary of the Dataset's contents

        Parameters
        ----------
        width
            Width in characters (default depends on current terminal size).
        """
        if width is None:
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 100
        width -= max(map(len, self))
        width -= max(len(v.__class__.__name__) for v in self.values())
        width -= 6
        out = fmtxt.Table('lll')
        out.cells('Key', 'Type', 'Values')
        out.midrule()
        for k, v in self.items():
            out.cell(k)
            out.cell(v.__class__.__name__)
            if hasattr(v, '_summary'):
                summary = v._summary(width)
            else:
                summary = ''
            out.cell(summary)
        name = 'Dataset' if self.name is None else self.name
        out.caption(f"{name}: {self.n_cases} cases")
        return out

    def tail(self, n=10, title=None):
        "Table with the last n cases in the Dataset"
        return self._display_table(range(-n, 0), title)

    def tile(self, repeats, name=None):
        """Concatenate ``repeats`` copies of the dataset

        Parameters
        ----------
        repeats : int
            Number of repeats.
        name : str
            Name for the new dataset (default is ``self.name``).
        """
        items = {name: item.tile(repeats) for name, item in self.items()}
        if name is None:
            name = self.name
        return Dataset(items, name, self._caption, self.info, self.n_cases * repeats)

    def to_r(self, name=None):
        """Place the Dataset into R as dataframe using rpy2

        Parameters
        ----------
        name : str
            Name for the R dataframe (default is self.name).

        Examples
        --------
        >>> from rpy2.robjects import r
        >>> ds = datasets.get_uv()
        >>> print(ds[:6])
        A    B    rm     intvar   fltvar     fltvar2    index
        -----------------------------------------------------
        a1   b1   s000   13       0.25614    0.7428     True
        a1   b1   s001   8        -1.5174    -0.75498   True
        a1   b1   s002   11       -0.5071    -0.13828   True
        a1   b1   s003   11       2.1491     -2.1249    True
        a1   b1   s004   15       -0.19358   -1.03      True
        a1   b1   s005   17       2.141      -0.51745   True
        >>> ds.to_r('df')
        >>> print(r("head(df)"))
           A  B   rm intvar     fltvar    fltvar2 index
        1 a1 b1 s000     13  0.2561439  0.7427957  TRUE
        2 a1 b1 s001      8 -1.5174371 -0.7549815  TRUE
        3 a1 b1 s002     11 -0.5070960 -0.1382827  TRUE
        4 a1 b1 s003     11  2.1490761 -2.1249203  TRUE
        5 a1 b1 s004     15 -0.1935783 -1.0300188  TRUE
        6 a1 b1 s005     17  2.1410424 -0.5174519  TRUE

        """
        import rpy2.robjects as ro

        if name is None:
            name = self.name
            if name is None:
                raise TypeError('Need a valid name for the R data frame')

        items = {}
        for k, v in self.items():
            if isinstance(v, Var):
                if v.x.dtype.kind == 'b':
                    item = ro.BoolVector(v.x)
                elif v.x.dtype.kind == 'i':
                    item = ro.IntVector(v.x)
                else:
                    item = ro.FloatVector(v.x)
            elif isinstance(v, Factor):
                x = ro.IntVector(v.x)
                codes = sorted(v._labels)
                levels = ro.IntVector(codes)
                labels = ro.StrVector(tuple(v._labels[c] for c in codes))
                item = ro.FactorVector(x, levels, labels)
            else:
                continue
            items[k] = item

        df = ro.DataFrame(items)
        ro.globalenv[name] = df

    def update(self, ds, replace=False, info=True):
        """Update the Dataset with all variables in ``ds``.

        Parameters
        ----------
        ds : Dataset | dict
            A Dataset or other dictionary-like object whose keys are strings
            and whose values are data-objects.
        replace : bool
            If a variable in ds is already present, replace it. If False,
            duplicates raise a ValueError (unless they are equivalent).
        info : bool
            Also update the info dictionary.

        Notes
        -----
        By default, if a key is present in both Datasets, and the corresponding
        variables are not equal on all cases, a ValueError is raised. If all
        values are equal, the variable in ds is copied into the Dataset that is
        being updated (the expected behavior of .update()).
        """
        if not isinstance(ds, Dataset):
            ds = Dataset(ds)
            info = False

        if ds.n_cases is None or self.n_cases is None:
            pass
        elif ds.n_cases != self.n_cases:
            raise ValueError(f"Trying to update dataset with {self.n_cases} cases from dataset with {ds.n_cases} cases")

        if not replace:
            unequal = []
            for key in set(self).intersection(ds):
                own = self[key]
                other = ds[key]
                if len(own) != len(other):
                    unequal.append((key, 'unequal length'))
                elif not np.all(own == other):
                    unequal.append((key, 'unequal values'))
            if unequal:
                raise ValueError(f"Inconsistent variables present: {enumeration(f'{name} ({msg})' for name, msg in unequal)}")

        for k, v in ds.items():
            self[k] = v

        if info:
            self.info.update(ds.info)

    def zip(self, *variables):
        """Iterate through the values of multiple variables

        ``ds.zip('a', 'b')`` is equivalent to ``zip(ds['a'], ds['b'])``.
        """
        return zip(*map(self.eval, variables))


class Interaction(_Effect):
    """Represents an Interaction effect.

    Usually not initialized directly but through operations on Factors/Vars.

    Parameters
    ----------
    base : sequence
        List of data-objects that form the basis of the interaction.

    Attributes
    ----------
    base : list
        All effects.
    """
    def __init__(self, base):
        base_ = EffectList()
        n_vars = 0

        for b in base:
            if isuv(b):
                base_.append(b.copy())
                n_vars += isinstance(b, Var)
            elif isinstance(b, Interaction):
                base_.extend(b.base)
                n_vars += not b.is_categorial
            elif isinstance(b, NestedEffect):
                base_.append(b)
            else:
                raise TypeError(f"{b}: Invalid type for Interaction")

        if n_vars > 1:
            raise TypeError("No Interaction between two Var objects")

        if len(base_) < 2:
            raise ValueError(f"{base}: Interaction needs a base of at least two Factors")
        n = len(base_[0])
        if not all(len(f) == n for f in base_[1:]):
            raise ValueError(f"{base}: different number of cases")
        self.__setstate__({'base': base_, 'is_categorial': not bool(n_vars)})

    def __setstate__(self, state):
        self.base = state['base']
        self.is_categorial = state['is_categorial']
        # secondary attributes
        self._n_cases = len(self.base[0])
        self.nestedin = EffectList()
        for e in self.base:
            if isinstance(e, NestedEffect) and not any(np.all(e.nestedin == ne) for ne in self.nestedin):
                self.nestedin.append(e.nestedin)
        self.base_names = [str(f.name) for f in self.base]
        self.name = ' x '.join(self.base_names)
        self.random = False
        self.df = reduce(operator.mul, [f.df for f in self.base])
        # For cells: find raw Factors
        self._factors = EffectList(e if isinstance(e, Factor) else e.effect for e in self.base if isinstance(e, (Factor, NestedEffect)))
        self.cell_header = tuple(f.name for f in self._factors)
        # TODO: beta-labels
        self.beta_labels = ['?'] * self.df

    def __getstate__(self):
        return {'base': self.base, 'is_categorial': self.is_categorial}

    def __repr__(self):
        names = [UNNAMED if f.name is None else f.name for f in self.base]
        if preferences['short_repr']:
            return ' % '.join(names)
        else:
            return "Interaction({n})".format(n=', '.join(names))

    @cached_property
    def _value_set(self):
        return set(self)

    @cached_property
    def cells(self):
        return tuple(cell for cell in self._all_cells if cell in self._value_set)

    def _sorted_cells(self):
        all_cells = product(*(f._sorted_cells() for f in self._factors))
        return [cell for cell in all_cells if cell in self._value_set]

    @cached_property
    def _all_cells(self):
        return [*product(*(f.cells for f in self._factors))]

    @cached_property
    def _empty_cells(self):
        if len(self._all_cells) != len(self.cells):
            return [cell for cell in self._all_cells if cell not in self._value_set]

    # container ---
    def __getitem__(self, index):
        if isinstance(index, Var):
            index = index.x

        out = tuple(f[index] for f in self.base)

        if index_ndim(index) == 1:
            return Interaction(out)
        else:
            return out

    def __contains__(self, item):
        if isinstance(item, tuple):
            return item in self._value_set
        return self.base.__contains__(item)

    def __iter__(self):
        yield from zip(*self.base)

    # numeric ---
    def __eq__(self, other):
        if isinstance(other, Interaction) and len(other.base) == len(self.base):
            x = np.vstack([b == bo for b, bo in zip(self.base, other.base)])
            return np.all(x, 0)
        elif isinstance(other, tuple) and len(other) == len(self.base):
            x = np.vstack([factor == level for factor, level in zip(self.base, other)])
            return np.all(x, 0)
        else:
            return np.zeros(len(self), bool)

    def __ne__(self, other):
        if isinstance(other, Interaction) and len(other.base) == len(self.base):
            x = np.vstack([b != bo for b, bo in zip(self.base, other.base)])
            return np.any(x, 0)
        elif isinstance(other, tuple) and len(other) == len(self.base):
            x = np.vstack([factor != level for factor, level in zip(self.base, other)])
            return np.any(x, 0)
        return np.ones(len(self), bool)

    def as_factor(self, delim=' ', name=None):
        """Convert the Interaction to a factor

        Parameters
        ----------
        delim : str
            Delimiter to join factor cell values (default ``" "``).
        name : str
            Name for the Factor (default is None).

        Examples
        --------
        >>> print(ds[::20, 'A'])
        Factor(['a1', 'a1', 'a2', 'a2'], name='A')
        >>> print(ds[::20, 'B'])
        Factor(['b1', 'b2', 'b1', 'b2'], name='B')
        >>> i = ds.eval("A % B")
        >>> print(i.as_factor()[::20])
        Factor(['a1 b1', 'a1 b2', 'a2 b1', 'a2 b2'], name='AxB')
        >>> print(i.as_factor("_")[::20])
        Factor(['a1_b1', 'a1_b2', 'a2_b1', 'a2_b2'], name='AxB')
        """
        return Factor(self.as_labels(delim), name)

    def as_cells(self):
        """All values as a list of tuples."""
        return [case for case in self]

    @cached_property
    def as_dummy(self):
        codelist = [f.as_dummy for f in self.base]
        return reduce(_effect_interaction, codelist)

    @cached_property
    def as_effects(self):  # Effect coding
        codelist = [f.as_effects for f in self.base]
        return reduce(_effect_interaction, codelist)

    def _coefficient_names(self, method):
        if self.df == 1:
            return [self.name]
        return ["%s %i" % (self.name, i) for i in range(self.df)]

    def as_labels(self, delim=' '):
        """All values as a list of strings.

        Parameters
        ----------
        delim : str
            Delimiter with which to join the elements of cells.
        """
        return [delim.join(filter(None, map(str, case))) for case in self]

    def aggregate(
            self,
            x: CategorialArg,
    ) -> Interaction:
        return Interaction(f.aggregate(x) for f in self.base)

    def isin(self, cells):
        """An index that is true where the Interaction equals any of the cells.

        Parameters
        ----------
        cells : sequence of tuples
            Cells for which the index will be true. Cells described as tuples
            of strings.
        """
        is_v = [self == cell for cell in cells]
        return np.any(is_v, 0)


def box_cox_transform(x, p, name=None):
    """The Box-Cox transform of x as :class:`Var`

    With ``p=0``, this is the log of x; otherwise ``(x**p - 1) / p``

    Parameters
    ----------
    x : Var
        Source data.
    p : scalar
        Parameter for Box-Cox transform.
    name : str
        Name for the output Var.
    """
    if isinstance(x, Var):
        x = x.x

    if p == 0:
        y = np.log(x)
    else:
        y = (x ** p - 1) / p

    return Var(y, name)


class NestedEffect(_Effect):

    def __init__(self, effect, nestedin):
        if not isinstance(nestedin, (Factor, Interaction)):
            raise TypeError(f"Nested in {dataobj_repr(nestedin)}: Effects can only be nested in Factor or Interaction")
        elif not iscategorial(nestedin):
            raise ValueError(f"Nested in {dataobj_repr(nestedin)}: Effects can only be nested in categorial base")

        self.effect = effect
        self.nestedin = nestedin
        self.random = effect.random
        self.cells = effect.cells
        self._n_cases = len(effect)

        if isinstance(self.effect, Factor):
            e_name = self.effect.name
        else:
            e_name = f'({self.effect})'
        self.name = f"{e_name}({nestedin.name})"

        if len(nestedin) != self._n_cases:
            raise ValueError(f"Unequal lengths: effect {e_name!r} len={len(effect)}, nested in {nestedin.name!r} len={len(nestedin)}")

    def __repr__(self):
        return self.name

    def __iter__(self):
        return self.effect.__iter__()

    def __eq__(self, other):
        return self.effect == other

    def __getitem__(self, index):
        if isinstance(index, Integral):
            return self.effect[index]
        return NestedEffect(self.effect[index], self.nestedin[index])

    def _sorted_cells(self):
        return self.effect._sorted_cells()

    @property
    def df(self):
        return len(self.effect.cells) - len(self.nestedin.cells)

    @property
    def as_effects(self):
        "Effect codes"
        codes = np.zeros((self._n_cases, self.df))
        ix = 0
        for outer_cell in self.nestedin.cells:
            outer_idx = (self.nestedin == outer_cell)
            inner_model = self.effect[outer_idx]
            n = len(inner_model.cells)
            inner_codes = _effect_eye(n)
            for i, cell in enumerate(inner_model.cells):
                codes[self.effect == cell, ix:ix + n - 1] = inner_codes[i]
            ix += n - 1

        return codes

    def _coefficient_names(self, method):
        return ["%s %i" % (self.name, i) for i in range(self.df)]


class NonbasicEffect:

    def __init__(self, effect_codes, factors, name, nestedin=[],
                 beta_labels=None):
        if beta_labels is not None and len(beta_labels) != effect_codes.shape[1]:
            raise ValueError("beta_labels need one entry per model column "
                             "(%s); got %s"
                             % (effect_codes.shape[1], repr(beta_labels)))
        self.nestedin = nestedin
        self.name = name
        self.random = False
        self.as_effects = effect_codes
        self._n_cases, self.df = effect_codes.shape
        self.factors = factors
        self.beta_labels = beta_labels

    def __repr__(self):
        txt = "<NonbasicEffect: {n}>"
        return txt.format(n=self.name)

    # container ---
    def __len__(self):
        return self._n_cases

    def _coefficient_names(self, method):
        if self.beta_labels is None:
            return ["%s %i" % (self.name, i) for i in range(self.df)]
        else:
            return self.beta_labels


class Model:
    """A list of effects.

    Parameters
    ----------
    x : effect | iterator of effects
        Effects to be included in the model (Var, Factor, Interaction ,
        ...). Can also contain models, in which case all the model's
        effects will be added.

    Attributes
    ----------
    effects : list
        Effects included in the model (:class:`Var`, :class:`Factor`, etc.
        objects)
    """
    def __init__(self, x):
        effects = EffectList()

        # find effects in input
        if iseffect(x):
            effects.append(x)
            n_cases = len(x)
        elif isinstance(x, Model):
            effects += x.effects
            n_cases = len(x)
        else:
            n_cases = None
            for e in x:
                # check n_cases
                if n_cases is None:
                    n_cases = len(e)
                elif len(e) != n_cases:
                    e0 = effects[0]
                    raise ValueError(f"All effects contained in a Model need to describe the same number of cases. {dataobj_repr(e0)} has {len(e0)} cases, {dataobj_repr(e)} has {len(e)} cases.")

                # find effects
                if iseffect(e):
                    effects.append(e)
                elif isinstance(e, Model):
                    effects += e.effects
                else:
                    raise TypeError(f"Model needs to be initialized with effect (Var, Factor, Interaction, ...) and/or Model objects (got {type(e)})")

        # check dfs
        df = sum(e.df for e in effects) + 1  # intercept
        if df > n_cases:
            raise ValueError(
                "Model overspecified (%i cases for %i model df)" %
                (n_cases, df))

        # beta indices
        for e in effects:
            if isinstance(e, Factor) and len(e.cells) == 1:
                raise ValueError(f"The Factor {dataobj_repr(e)} has only one level ({e.cells[0]}). The intercept is implicit in each model and should not be specified explicitly.")

        self.effects = effects
        self.df = df
        self._init_secondary()

    def _init_secondary(self):
        self.df_total = len(self.effects[0])
        self.df_error = self.df_total - self.df
        self.name = ' + '.join([str(e.name) for e in self.effects])

    def __setstate__(self, state):
        self.effects = state['effects']
        self.df = sum(e.df for e in self.effects) + 1  # intercept
        self._init_secondary()

    def __getstate__(self):
        return {'effects': self.effects}

    def __repr__(self):
        names = self.effects.names()
        if preferences['short_repr']:
            return ' + '.join(names)
        else:
            x = ', '.join(names)
            return "Model((%s))" % x

    def __str__(self):
        return str(self.as_table())

    def array(self, coding: Literal['effect', 'dummy']) -> (numpy.ndarray, List[str]):
        "Model matrix as numpy array and column names"
        p = Parametrization(self, coding)
        return p.x, p.column_names

    def info(self):
        """A :class:`fmtxt.Table` with information about the model"""
        table = fmtxt.Table('rl')
        table.cells('Df', 'Term')
        table.midrule()
        for e in self.effects:
            table.cells(e.df, e.name, )
        if self.df_error:
            table.midrule()
            table.cells(self.df_error, 'Residuals')
        return table

    # container ---
    def __len__(self):
        return self.df_total

    def __getitem__(self, sub):
        if isinstance(sub, str):
            for e in self.effects:
                if e.name == sub:
                    return e
            raise KeyError(sub)
        elif isinstance(sub, INT_TYPES):
            return tuple(e[sub] for e in self.effects)
        else:
            return Model((x[sub] for x in self.effects))

    def __contains__(self, effect):
        return id(effect) in map(id, self.effects)

    def sorted(self):
        """Sorted copy of the Model, interactions last"""
        out = []
        i = 1
        while len(out) < len(self.effects):
            for e in self.effects:
                if len(e.factors) == i:
                    out.append(e)
            i += 1
        return Model(out)

    # numeric ---
    def __add__(self, other):
        return Model((self, other))

    def __mul__(self, other):
        return Model((self, other, self % other))

    def __mod__(self, other):
        out = []
        for e_self in self.effects:
            for e_other in Model(other).effects:
                if isinstance(e_self, Var) and isinstance(e_other, Var):
                    out.append(e_self * e_other)
                elif not partially_nested(e_self, e_other):
                    out.append(e_self % e_other)
        return Model(out)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        elif not len(self) == len(other):
            return False
        elif not len(self.effects) == len(other.effects):
            return False

        for e, eo in zip(self.effects, other.effects):
            if not np.all(e == eo):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # repr ---
    @property
    def model_eq(self):
        return self.name

    # coding ---
    @cached_property
    def _effect_to_beta(self):
        """An array indicating for each effect which beta weights it occupies

        Returns
        -------
        effects_to_beta : np.ndarray (n_effects, 2)
            For each effect, indicating the first index in betas and df
        """
        out = np.empty((len(self.effects), 2), np.intp)
        beta_start = 1
        for i, e in enumerate(self.effects):
            out[i, 0] = beta_start
            out[i, 1] = e.df
            beta_start += e.df
        return out

    def as_table(self, method='dummy', cases=0, group_terms=True):
        """Return a table with the model codes

        Parameters
        ----------
        method : 'effect' | 'dummy'
            Coding scheme: effect coding or dummy coding.
        cases : int | iterator of int
            Cases to include (int includes that many cases from the beginning,
            0 includes all; negative number works like negative indexing).
        group_terms : bool
            Group model columns that represent the same effect under one
            heading.

        Returns
        --------
        table : FMText Table
            The full model as a table.
        """
        itre_cases = cases_arg(cases, self.df_total)
        p = self._parametrize(method)
        table = fmtxt.Table('l' * len(p.column_names))

        # Header
        if group_terms:
            for term in p.effect_names:
                index = p.terms[term]
                ncolumns = index.stop - index.start
                table.cell(term, width=ncolumns)
        else:
            for term in p.column_names:
                table.cell(term)
        table.midrule()

        # data
        for case in itre_cases:
            for i in p.x[case]:
                table.cell('%g' % i)

        return table

    def head(self, n=10):
        "Table with the first n cases in the Model"
        return self.as_table(cases=n)

    # checking model properties
    def check(self):
        "Shortcut to check linear independence and orthogonality"
        return self.lin_indep() + self.orthogonal()

    def lin_indep(self):
        "Check the Model for linear independence of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
        for i in range(ne):
            for j in range(i + 1, ne):
                e1 = self.effects[i]
                e2 = self.effects[j]
                x = np.hstack((codes[i], codes[j]))
                if rank(x) < x.shape[1]:
                    msg.append(f"Linear Dependence Warning: {e1.name} and {e2.name}")
        return msg

    def orthogonal(self):
        "Check the Model for orthogonality of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
        for i, j in combinations(range(ne), 2):
            e1 = self.effects[i]
            e2 = self.effects[j]
            for i1, i2 in product(range(e1.df), range(e2.df)):
                if np.dot(codes[i][:, i1], codes[j][:, i2]) != 0:
                    msg.append(f"Not orthogonal: {e1.name} and {e2.name}")
                    break
        return msg

    def _parametrize(self, method='effect'):
        "Create a design matrix"
        return Parametrization(self, method)

    def _incomplete_error(self, caller):
        df_table = self.info()
        df_table[-1, 1] = 'Unexplained'
        return IncompleteModelError(f"{caller} requires a fully specified model, but {self.name} only has {self.df} degrees of freedom for {self.df_total} cases:\n{df_table}")

    def repeat(self, n):
        "Repeat each row of the Model ``n`` times"
        return Model(e.repeat(n) for e in self.effects)

    def tail(self, n=10):
        "Table with the last n cases in the Model"
        return self.as_table(cases=range(-n, 0))


class Parametrization:
    """Parametrization of a model

    Parameters
    ----------
    model
        Model to be parametrized.
    method
        Coding scheme: effect coding or dummy coding.

    Attributes
    ----------
    model : Model
        The model that is parametrized.
    x : array (n_cases, n_coeffs)
        Design matrix.
    terms : {str: slice}
        Location of each term in x.
    column_names : list of str
        Name of each column.

    Notes
    -----
    A :class:`Model` is a list of effects. A :class:`Parametrization` contains
    a realization of those effects in a model matrix with named columns.
    """
    def __init__(
            self,
            model: Model,
            method: Literal['effect', 'dummy'],
    ):
        model = asmodel(model)
        x = np.empty((model.df_total, model.df))
        x[:, 0] = 1
        column_names = ['intercept']
        effect_names = ['intercept']
        higher_level_effects = {}
        terms = {'intercept': slice(0, 1)}
        i = 1
        for e in model.effects:
            j = i + e.df
            if method == 'effect':
                x[:, i:j] = e.as_effects
            elif method == 'dummy':
                x[:, i:j] = e.as_dummy
            else:
                raise ValueError(f"{method=}")
            name = longname(e)
            if name in terms:
                raise KeyError(f"Duplicate term name: {name!r}")
            terms[name] = slice(i, j)
            effect_names.append(name)
            col_names = e._coefficient_names(method)
            column_names.extend(col_names)
            for col, col_name in enumerate(col_names, i):
                terms[col_name] = slice(col, col + 1)
            i = j

            # find comparison models
            higher_level_effects[name] = [
                e_ for e_ in model.effects if
                e_ is not e and is_higher_order_effect(e_, e)
            ]

        # model basics
        self.model = model
        self.method = method
        self.x = x
        self.terms = terms
        self.column_names = column_names
        self.effect_names = effect_names
        self._higher_level_effects = higher_level_effects

        # projector
        x_t = x.T
        self.g = inv(x_t.dot(x))
        self.projector = self.g.dot(x_t)

    def reduced_model_index(self, term):
        "Boolean index into model columns for model comparison"
        out = np.ones(self.x.shape[1], bool)
        out[self.terms[term]] = False
        for e in self._higher_level_effects[term]:
            out[self.terms[e.name]] = False
        return out


class PermutedParametrization:

    def __init__(self, parametrization: Parametrization, g: bool = False):
        self.parametrization = parametrization
        self.x = np.empty_like(parametrization.x)
        self.projector = np.empty_like(parametrization.projector)
        self.g = g

    def permute(self, perm: np.ndarray):
        self.parametrization.x.take(perm, 0, self.x)
        self.parametrization.projector.take(perm, 1, self.projector)
        if self.g is not False:
            self.g = inv(self.x.T.dot(self.x), True)


# ---NDVar dimensions---

def _subgraph_edges(adjacency, int_index):
    "Extract adjacency for a subset of a graph"
    idx = np.logical_and(np.isin(adjacency[:, 0], int_index),
                         np.isin(adjacency[:, 1], int_index))
    if np.any(idx):
        new_c = adjacency[idx]

        # remap to new vertex indices
        if np.any(np.diff(int_index) < 1):  # non-monotonic index
            argsort = np.argsort(int_index)
            flat_conn_ = np.digitize(new_c.ravel(), int_index[argsort], True)
            flat_conn = argsort[flat_conn_]
        else:
            flat_conn = np.digitize(new_c.ravel(), int_index, True)

        return flat_conn.reshape(new_c.shape).astype(np.uint32)
    else:
        return np.empty((0, 2), dtype=np.uint32)


def adjacency_from_name_pairs(neighbors, items, allow_missing=False):
    pairs = set()
    for src, dst in neighbors:
        if src not in items or dst not in items:
            if allow_missing:
                continue
            else:
                item = src if src not in items else dst
                raise ValueError(f"Adjacency contains unknown item {item!r}")
        a = items.index(src)
        b = items.index(dst)
        if a < b:
            pairs.add((a, b))
        else:
            pairs.add((b, a))
    return np.array(sorted(pairs), np.uint32)


class Dimension:
    """Base class for dimensions.

    Parameters
    ----------
    name : str
        Dimension name.
    adjacency : 'grid' | 'none' | array of int, (n_edges, 2)
        Adjacency between elements. Set to ``"none"`` for no connections or
        ``"grid"`` to use adjacency in the sequence of elements as connection.
        Set to :class:`numpy.ndarray` to specify custom adjacency. The array
        should be of shape (n_edges, 2), and each row should specify one
        connection [i, j] with i < j, with rows sorted in ascending order. If
        the array's dtype is uint32, property checks are disabled to improve
        efficiency.

    Attributes
    ----------
    x : array_like
        Numerical values (e.g. for locating categories on an axis).
    values : sequence
        Meaningful point descriptions (e.g. time points, sensor names, ...).
    """
    _ADJACENCY_TYPES = ('grid', 'none', 'custom', 'vector')
    _axis_unit = None
    _default_adjacency = 'none'  # for loading old pickles

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            name: str,
            adjacency: AdjacencyArg,
    ):
        # requires __len__ to work
        self.name = name
        self._adjacency_type, self._adjacency = self._adjacency_arg(adjacency)

    def _adjacency_arg(self, adjacency: AdjacencyArg):
        if isinstance(adjacency, str):
            if adjacency not in self._ADJACENCY_TYPES:
                raise ValueError(f"{adjacency=}")
            return adjacency, None
        elif len(adjacency) == 0:
            return 'none', None
        elif isinstance(adjacency, np.ndarray) and adjacency.dtype == np.uint32:
            # assume that adjacency is inherited, skip checks
            return 'custom', adjacency
        else:
            return 'custom', self._coerce_adjacency(adjacency)

    def _coerce_adjacency(self, adjacency: ArrayLike) -> np.ndarray:
        adjacency = np.asarray(adjacency)
        if adjacency.dtype.kind != 'i':
            raise TypeError(f"adjacency array needs to be integer type, got {adjacency.dtype}")
        elif adjacency.shape != (len(adjacency), 2):
            raise ValueError(f"adjacency requires shape (n_edges, 2), got array with shape {adjacency.shape}")
        elif adjacency.min() < 0:
            raise ValueError("adjacency can not have negative values")
        elif adjacency.max() >= len(self):
            raise ValueError("adjacency has value larger than number of elements in dimension")
        elif np.any(adjacency[:, 0] >= adjacency[:, 1]):
            raise ValueError("All edges [i, j] must have i < j")
        elif np.any(np.diff(adjacency, axis=0) > 0):
            edges = list(map(tuple, adjacency))
            edges.sort()
            adjacency = np.array(edges, np.uint32)
        else:
            adjacency = adjacency.astype(np.uint32)
        return adjacency

    def __getstate__(self):
        return {'name': self.name, 'connectivity': self._adjacency,
                'connectivity_type': self._adjacency_type}

    def __setstate__(self, state):
        self.name = state['name']
        self._adjacency = state.get('connectivity', None)
        self._adjacency_type = state.get('connectivity_type', self._default_adjacency)

    def __len__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self._eq(other, True)

    def _eq(self, other, check: bool):
        if not isinstance(other, self.__class__) or not other.name == self.name:
            return False
        elif len(other) != len(self):
            return False
        elif check:
            if other._adjacency_type == self._adjacency_type:
                if self._adjacency_type == 'custom':
                    if other._adjacency is None or self._adjacency is None:
                        return True
                    return np.array_equal(other._adjacency, self._adjacency)
                return True
            return False
        return True

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, index):
        """Array-like Indexing

        Possible indexes:

          - int -> label or value for that location
          - [int] -> Dimension object with 1 location
          - [int, ...] -> Dimension object
        """
        raise NotImplementedError

    def _bin(self, start, stop, step, nbins, label):
        "Divide Dimension into bins"
        raise NotImplementedError(f"Binning for {self.__class__.__name__} dimension")

    @classmethod
    def _concatenate(cls, dims: Sequence[Dimension]):
        "Concatenate multiple dimension instances"
        raise NotImplementedError(f"Can't concatenate along {cls.__name__} dimensions")

    @staticmethod
    def _concatenate_adjacency(dims: Sequence[Dimension]):
        c_types = {dim._adjacency_type for dim in dims}
        if len(c_types) > 1:
            raise NotImplementedError("concatenating with differing adjacency")
        c_type = c_types.pop()
        if c_type == 'custom':
            raise NotImplementedError("concatenating with custom adjacency")
        return c_type

    @staticmethod
    def _concatenate_attr(dims: Sequence[Dimension], attr: str):
        attrs = {getattr(dim, attr) for dim in dims}
        if len(attrs) > 1:
            desc = ', '.join(map(repr, attrs))
            raise DimensionMismatchError(f"Different values for {attr}: {desc}")
        return attrs.pop()

    def _as_scalar_array(self):
        raise TypeError(f"{self.__class__.__name__} dimension has no scalar representation")

    def _as_uv(self):
        return Var(self._axis_data(), self.name)

    def _axis_data(self):
        "x for plot command"
        return np.arange(len(self))

    def _axis_extent(self):
        "Extent for plots with continuous x-axis"
        return 0, len(self) - 1

    def _axis_im_extent(self):
        "Extent for im plots; needs to extend beyond end point locations"
        return -0.5, len(self) - 0.5

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        """Find axis decoration parameters for this dimension

        Parameters
        ----------
        scalar : bool
            If True, the axis is scalar and labels should correspond to the axis
            value. If False, the axis represents categorial bins (e.g.,
            im-plots).
        label : bool | str
            Label (if True, return an appropriate axis-specific label).

        Returns
        -------
        formatter : matplotlib Formatter
            Axis tick formatter.
        locator : matplotlib Locator
            Tick-locator.
        label : str | None
            Return the default axis label if label==True, otherwise the label
            argument.
        """
        raise NotImplementedError

    def _axis_label(self, label):
        if label is True:
            if self._axis_unit:
                return "%s [%s]" % (self.name.capitalize(), self._axis_unit)
            else:
                return self.name.capitalize()
        return label

    def dimindex(self, arg):
        "Convert a dimension index to an array index"
        # backwards compatibility
        return self._array_index(arg)

    def _array_index(self, arg):
        """Convert a dimension-semantic index to an array-like index

        Subclasses need to handle dimension-specific cases

        args that are handled by the Dimension baseclass:
         - None
         - boolean array
         - boolean NDVars
        args handled recursively:
         - list
         - tuple -> slice(*tuple)
         - Var -> Var.x
        """
        if arg is None:
            return None  # pass through None, for example for slice
        elif isinstance(arg, NDVar):
            return self._array_index_for_ndvar(arg)
        elif isinstance(arg, Var):
            return self._array_index(arg.x)
        elif isinstance(arg, np.ndarray):
            if arg.dtype.kind != 'b':
                raise TypeError(f"array of type {arg.dtype.kind} not supported as index for {self._dimname()}")
            elif arg.ndim != 1:
                raise IndexError(f"Boolean index for {self._dimname()} needs to be 1d, got array of shape {arg.shape}")
            elif len(arg) != len(self):
                raise IndexError(f"Got boolean index of length {len(arg)} for {self._dimname()} of length {len(self)}")
            return arg
        elif isinstance(arg, tuple):
            if len(arg) > 3:
                raise IndexError(f"{arg!r}: tuple indexes signify intervals and need to be of length 1, 2 or 3")
            return self._array_index_for_slice(*arg)
        elif isinstance(arg, list):
            if len(arg) == 0:
                return np.empty(0, np.intp)
            return np.array([self._array_index(a) for a in arg])
        elif isinstance(arg, slice):
            return self._array_index_for_slice(arg.start, arg.stop, arg.step)
        else:
            raise TypeError(f"Unknown index type for {self._dimname()}: {arg!r}")

    def _array_index_for_ndvar(self, arg):
        if arg.x.dtype.kind != 'b':
            raise IndexError(f"{arg}: only NDVars with boolean data can serve as indexes")
        elif arg.ndim != 1:
            raise IndexError(f"{arg}: only NDVars with ndim == 1 can serve as indexes")
        dim = arg.dims[0]
        if not isinstance(dim, self.__class__):
            raise IndexError(f"{arg}: must have {self.__class__} dimension")
        elif dim == self:
            return arg.x
        index_to_arg = self._array_index_to(dim)
        return index_to_arg[arg.x]

    def _array_index_for_slice(self, start, stop=None, step=None):
        if step is not None and not isinstance(step, Integral):
            raise TypeError(f"Slice index step for {self._dimname()} must be int, not {step!r}")

        if start is None:
            start_ = None
        else:
            start_ = self._array_index(start)
            if not isinstance(start_, int):
                raise TypeError(f"{start!r} is not an unambiguous slice start for {self._dimname()}")

        if stop is None:
            stop_ = None
        else:
            stop_ = self._array_index(stop)
            if not isinstance(stop_, int):
                raise TypeError(f"{stop!r} is not an unambiguous slice start for {self._dimname()}")

        return slice(start_, stop_, step)

    def _array_index_to(self, other):
        "Int index to access data from self in an order consistent with other"
        raise NotImplementedError(f"Internal alignment for {self.__class__}")

    def _is_superset_of(self, dim):
        "Test whether self is a superset of dim"
        raise NotImplementedError

    def index_into_dim(self, dim):
        "Index into a subset dimension"
        raise NotImplementedError

    def _dimname(self):
        if self.name.lower() == self.__class__.__name__.lower():
            return f'{self.__class__.__name__} dimension'
        else:
            return f'{self.__class__.__name__} dimension ({self.name!r})'

    def _dim_index(self, arg):
        "Convert an array index to a dimension index"
        if isinstance(arg, slice):
            start = None if arg.start is None else self._dim_index(arg.start)
            stop = None if arg.stop is None else self._dim_index(arg.stop)
            return slice(start, stop, arg.step)
        elif np.isscalar(arg):
            return arg
        else:
            return [self._dim_index(i) for i in index_to_int_array(arg, len(self))]

    def _distances(self):
        "Distance matrix for dimension elements (square form)"
        raise NotImplementedError(f"Distances for {self.__class__.__name__}")

    def intersect(self, dim, check_dims=True):
        """Create a Dimension that is the intersection with dim

        Parameters
        ----------
        dim : Dimension
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple).

        Returns
        -------
        intersection : Dimension
            The intersection with dim (returns itself if dim and self are
            equal).
        """
        raise NotImplementedError

    def _union(self, other):
        """Create a Dimension that is the union with dim

        Parameters
        ----------
        other : Dimension
            Dimension to form union with.

        Returns
        -------
        union : Dimension
            The union with dim (returns itself if dim and self are
            equal).
        """
        raise NotImplementedError

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : None | Dataset
            A dataset with variables describing cluster properties.
        """
        return None

    def _cluster_property_labels(self):
        return []

    def adjacency(self):
        """Retrieve the dimension's adjacency graph

        Returns
        -------
        adjacency : array of int, (n_pairs, 2)
            array of sorted ``[src, dst]`` pairs, with all ``src < dst``.
        """
        if self._adjacency is None:
            self._adjacency = self._generate_adjacency()
        return self._adjacency

    def _generate_adjacency(self):
        raise NotImplementedError("Adjacency for %s dimension." % self.name)

    def _melt_vars(self) -> dict:
        "Variables to add when melting the dimension"
        return {}

    def _subgraph(self, index=None):
        """Adjacency parameter for new Dimension instance

        Parameters
        ----------
        index : array_like
            Index if the new dimension is a subset of the current dimension.
        """
        if self._adjacency_type == 'custom':
            if self._adjacency is None:
                return 'custom'
            elif index is None:
                return self._adjacency
            return _subgraph_edges(self._adjacency,
                                   index_to_int_array(index, len(self)))
        return self._adjacency_type

    def _rename(self, to):
        out = copy(self)
        out.name = to
        return out


class Case(Dimension):
    """Case dimension

    Parameters
    ----------
    n : int
        Number of cases.
    adjacency : 'grid' | 'none' | array of int, (n_edges, 2)
        Adjacency between elements. Set to ``"none"`` for no connections or
        ``"grid"`` to use adjacency in the sequence of elements as connection.
        Set to :class:`numpy.ndarray` to specify custom adjacency. The array
        should be of shape (n_edges, 2), and each row should specify one
        connection [i, j] with i < j, with rows sorted in ascending order. If
        the array's dtype is uint32, property checks are disabled to improve
        efficiency.

    Examples
    --------
    When initializing an :class:`NDVar`, the case dimension can be speciied
    with the bare class and the number of cases will be inferred from the data:

    >>> NDVar([[1, 2], [3, 4]], (Case, Categorial('column', ['1', '2'])))
    <NDVar: 2 case, 2 column>
    """
    _DIMINDEX_RAW_TYPES = INT_TYPES + (slice, list)

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            n: int,
            adjacency: AdjacencyArg = 'none',
    ):
        Dimension.__init__(self, 'case', adjacency)
        self.n = int(n)

    def __getstate__(self):
        out = Dimension.__getstate__(self)
        out['n'] = self.n
        return out

    def __setstate__(self, state):
        Dimension.__setstate__(self, state)
        self.n = state['n']

    def __repr__(self):
        if self._adjacency_type == 'none':
            return "Case(%i)" % self.n
        elif self._adjacency_type == 'grid':
            return "Case(%i, 'grid')" % self.n
        else:
            return "Case(%i, <custom adjacency>)" % self.n

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        if isinstance(item, Integral):
            if item < 0:
                if item < -self.n:
                    raise IndexError(item)
                item += self.n
            elif item > self.n:
                raise IndexError(item)
            return item
        else:
            return Case(index_length(item, self.n), self._subgraph(item))

    def __iter__(self):
        return iter(range(self.n))

    def _as_scalar_array(self):
        return np.arange(self.n)

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        if scalar:
            locator = None
            formatter = FormatStrFormatter('%i')
        else:
            locator = FixedLocator(np.arange(len(self)), 10)
            formatter = FormatStrFormatter('%.0f')
        return formatter, locator, self._axis_label(label)

    def _array_index(self, arg):
        if isinstance(arg, self._DIMINDEX_RAW_TYPES):
            return arg
        elif isinstance(arg, Var) and arg.x.dtype.kind in 'bi':
            return arg.x
        elif isinstance(arg, np.ndarray) and arg.dtype.kind in 'bi':
            return arg
        elif isinstance(arg, tuple):
            return slice(*arg) if arg else FULL_SLICE
        else:
            raise TypeError(f"Index {arg} of type {type(arg)} for Case dimension")

    @classmethod
    def _concatenate(cls, dims: Sequence[Case]):
        return Case

    def _dim_index(self, arg):
        return arg

    def _rename(self, to):
        return Categorial(to, map(str, range(self.n)))


class Space(Dimension):
    """Represent multiple directions in space

    Parameters
    ----------
    directions : str
        A sequence of directions, each indicated by a single capitalized
        character, from the following set: [A]nterior, [P]osterior, [L]eft,
        [R]ight, [S]uperior and [I]nferior.
    name : str
        Dimension name.

    Notes
    -----
    Adjacency is set to ``'none'``, but :class:`Space` is not a valid
    dimension to treat as mass-univariate.
    """

    _DIRECTIONS = {
        'A': 'anterior',
        'P': 'posterior',
        'L': 'left',
        'R': 'right',
        'S': 'superior',
        'I': 'inferior',
    }

    def __init__(
            self,
            directions: str,
            name: str = 'space',
    ):
        if not isinstance(directions, str):
            raise TypeError(f"{directions=}")
        n = len(directions)
        all_directions = set(directions)
        if len(all_directions) != n:
            raise ValueError(f"{directions=} contains duplicate direction")
        invalid = all_directions.difference(self._DIRECTIONS)
        if invalid:
            raise ValueError(f"{directions=} contains invalid directions: {', '.join(map(repr, invalid))}")
        Dimension.__init__(self, name, 'vector')
        self._directions = directions

    def __getstate__(self):
        out = Dimension.__getstate__(self)
        out['directions'] = self._directions
        return out

    def __setstate__(self, state):
        Dimension.__setstate__(self, state)
        self._directions = state['directions']

    def __repr__(self):
        return "Space(%r)" % self._directions

    def __len__(self):
        return len(self._directions)

    def _eq(self, other, check: bool):
        return Dimension._eq(self, other, check) and other._directions == self._directions

    def __getitem__(self, item):
        if not all(i in self._directions for i in item):
            raise IndexError(item)
        return Space(item)

    def __iter__(self):
        return iter(self._directions)

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        # like Categorial
        locator = FixedLocator(np.arange(len(self._directions)))
        formatter = IndexFormatter(self._directions)
        return formatter, locator, self._axis_label(label)

    def _array_index(self, arg):
        if isinstance(arg, str):
            if len(arg) == 1:
                return self._directions.index(arg)
            else:
                return [self._directions.index(s) for s in arg]
        elif isinstance(arg, list):
            return [self._directions.index(s) for s in arg]
        elif isinstance(arg, tuple):
            return slice(*map(self._array_index, arg)) if arg else FULL_SLICE
        elif isinstance(arg, slice):
            return slice(
                None if arg.start is None else self._array_index(arg.start),
                None if arg.stop is None else self._array_index(arg.stop),
                arg.step)
        else:
            raise TypeError(f"{arg} is an invalid index for {self.__class__.__name__}")

    def _dim_index(self, arg):
        if isinstance(arg, Integral):
            return self._directions[arg]
        else:
            return ''.join(self._directions[i] for i in arg)

    def intersect(self, dim, check_dims=True):
        """Create a dimension object that is the intersection with dim

        Parameters
        ----------
        dim : Space
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency.

        Returns
        -------
        intersection : Space
            The intersection with ``dim`` (returns itself if ``dim`` and
            ``self`` are equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        if self._directions == dim._directions:
            return self
        self_dirs = set(self._directions)
        dim_dirs = set(dim._directions)
        out_dirs = self_dirs.intersection(dim_dirs)
        if self_dirs == out_dirs:
            return self
        elif dim_dirs == out_dirs:
            return dim
        else:
            directions = ''.join(c for c in self._directions if c in dim._directions)
            return Space(directions, self.name)


class Categorial(Dimension):
    """Simple categorial dimension

    Parameters
    ----------
    name
        Dimension name.
    values
        Names of the levels.
    adjacency
        Adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - ``"grid"`` to use adjacency in ``values``
        - list of connections based on items in ``values`` (e.g.,
          ``values=['v1', 'v2', 'v3'],
          adjacency=[('v1', 'v3'), ('v2', 'v3'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection ``[i, j]`` with ``i < j``. If the array's dtype is
          ``uint32``, property checks are disabled to improve efficiency.

    """
    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            name: str,
            values: Sequence[str],
            adjacency: AdjacencyArg = 'none',
    ):
        self.values = tuple(values)
        if len(set(self.values)) < len(self.values):
            raise ValueError(f"{values=}: Dimension can not have duplicate values")
        if not all(isinstance(x, str) for x in self.values):
            raise ValueError(f"{values=}: All Categorial values must be strings")
        Dimension.__init__(self, name, adjacency)

    def _coerce_adjacency(self, adjacency):
        if isinstance(adjacency[0][0], str):
            return adjacency_from_name_pairs(adjacency, self.values)
        else:
            return Dimension._coerce_adjacency(self, adjacency)

    def __getstate__(self):
        out = Dimension.__getstate__(self)
        out['values'] = self.values
        return out

    def __setstate__(self, state):
        # backwards compatibility
        if 'connectivity' not in state:
            state['connectivity'] = None
            state['connectivity_type'] = 'none'
        self.values = state['values']
        if isinstance(self.values, np.ndarray):
            self.values = tuple(str(v) for v in self.values)
        # /backwards compatibility
        Dimension.__setstate__(self, state)

    def __repr__(self):
        args = (repr(self.name), str(self.values))
        return "%s(%s)" % (self.__class__.__name__, ', '.join(args))

    def __len__(self):
        return len(self.values)

    def _eq(self, other, check: bool):
        return Dimension._eq(self, other, check) and self.values == other.values

    def __getitem__(self, index):
        if isinstance(index, Integral):
            return self.values[index]
        else:
            return self.__class__(self.name, apply_numpy_index(self.values, index), self._subgraph(index))

    def _as_uv(self):
        return Factor(self.values, name=self.name)

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        locator = FixedLocator(np.arange(len(self)))
        formatter = IndexFormatter(self.values)
        return formatter, locator, self._axis_label(label)

    def _array_index(self, arg):
        if isinstance(arg, str):
            if arg in self.values:
                return self.values.index(arg)
            else:
                raise IndexError(arg)
        elif isinstance(arg, self.__class__):
            return [self._array_index(v) for v in arg.values]
        else:
            return super(Categorial, self)._array_index(arg)

    @classmethod
    def _concatenate(cls, dims: Sequence[Categorial]):
        dims = list(dims)
        name = cls._concatenate_attr(dims, 'name')
        adjacency = cls._concatenate_adjacency(dims)
        values = sum((dim.values for dim in dims), ())
        return cls(name, values, adjacency)

    def _dim_index(self, index):
        if isinstance(index, Integral):
            return self.values[index]
        else:
            return Dimension._dim_index(self, index)

    def intersect(self, dim, check_dims=False):
        """Create a dimension object that is the intersection with dim

        Parameters
        ----------
        dim : type(self)
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : type(self)
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        if self.values == dim.values:
            return self
        index = np.array([v in dim.values for v in self.values])
        if np.all(index):
            return self
        elif index.sum() == len(dim):
            return dim
        else:
            return self[index]


class Scalar(Dimension):
    """Scalar dimension

    Parameters
    ----------
    name
        Name of the dimension.
    values
        Scalar value for each sample of the dimension.
    unit
        Unit of the values.
    tick_format
        Format string for formatting axis tick labels ('%'-format, e.g. '%.0f'
        to round to nearest integer).
    adjacency
        Adjacency between elements. Set to ``"none"`` for no connections or
        ``"grid"`` to use adjacency in the sequence of elements as connection.
        Set to :class:`numpy.ndarray` to specify custom adjacency. The array
        should be of shape (n_edges, 2), and each row should specify one
        connection [i, j] with i < j, with rows sorted in ascending order. If
        the array's dtype is uint32, property checks are disabled to improve
        efficiency.
    """
    _default_adjacency = 'grid'

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            name: str,
            values: ArrayLike,
            unit: str = None,
            tick_format: str = None,
            adjacency: AdjacencyArg = 'grid',
    ):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{values.shape=}, needs to be one-dimensional")
        elif np.any(np.diff(values) <= 0):
            raise ValueError("Values for Scalar must increase monotonically")
        elif tick_format and '%' not in tick_format:
            raise ValueError(f"{tick_format=}, needs to include '%'")
        self.values = values
        self.unit = unit
        self._axis_unit = unit
        self.tick_format = tick_format
        Dimension.__init__(self, name, adjacency)

    def __getstate__(self):
        out = Dimension.__getstate__(self)
        out.update(values=self.values, unit=self.unit,
                   tick_format=self.tick_format)
        return out

    def __setstate__(self, state):
        # backwards compatibility
        if 'connectivity' not in state:
            state['connectivity'] = None
            state['connectivity_type'] = 'grid'
        Dimension.__setstate__(self, state)
        self.values = state['values']
        self.unit = self._axis_unit = state.get('unit')
        self.tick_format = state.get('tick_format')

    def __repr__(self):
        if len(self.values) == 1:
            v_repr = f"[{self.values[0]:g}]"
        elif len(self.values) <= 4:
            values = [f'{v:g}' for v in self.values]
            v_repr = f"[{', '.join(values)}]"
        else:
            v_repr = f"[{self.values[0]:g}, ..., {self.values[-1]:g}] ({len(self)})"
        args = [repr(self.name), v_repr]
        if self.unit is not None or self.tick_format is not None:
            args.append(repr(self.unit))
        if self.tick_format is not None:
            args.append(repr(self.tick_format))
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __len__(self):
        return len(self.values)

    def _eq(self, other, check: bool):
        return Dimension._eq(self, other, check) and np.array_equal(self.values, other.values)

    def __getitem__(self, index):
        if isinstance(index, Integral):
            return self.values[index]
        return self.__class__(self.name, self.values[index], self.unit,
                              self.tick_format, self._subgraph(index))

    def _as_scalar_array(self):
        return self.values

    def _axis_data(self):
        return self.values

    def _axis_extent(self):
        return self[0], self[-1]

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        if scalar:
            if self.tick_format:
                formatter = FormatStrFormatter(self.tick_format)
            else:
                formatter = None
            locator = None
        else:
            # categories (im-plot)
            #  - for small number of values, label bins at their center
            #  - for large number of values, make sure endpoints are included
            if len(self) < 10:
                locations = np.arange(len(self))
                values = self.values
            else:
                locations = np.arange(-0.5, len(self), 0.5)
                values = [*self.values, self.values[-1]]
            locator = FixedLocator(locations, 10)

            if self.tick_format:
                fmt = self.tick_format
            else:
                step = (self.values[-1] - self.values[0]) / min(len(self), 10)
                if step > 2:
                    n_digits = 0
                else:
                    n_digits = int(log(1 / step, 10)) + 1
                fmt = f'%.{n_digits}f'
            formatter = FuncFormatter(lambda i, pos: fmt % values[int(round(i))])
        return formatter, locator, self._axis_label(label)

    def _bin(
            self,
            start: float = None,
            stop: float = None,
            step: int = None,  # -> step in dim space
            nbins: int = None,  # -> equally divide in array space
            label: str = 'center',
    ) -> (list, 'Scalar'):
        islice = self._array_index_for_slice(start, stop, step)
        istart = 0 if islice.start is None else islice.start
        istop = None if islice.stop is None else islice.stop
        start = self.values[istart]
        stop = None if istop is None else self.values[istop]
        if nbins is not None:
            if istop is None:
                istop = len(self)
            n_source_steps = istop - istart
            if n_source_steps % nbins != 0:
                raise ValueError(f"nbins={nbins!r}: length {n_source_steps} {self.name} can not be divided equally")
            istep = int(n_source_steps / nbins)
            edges = list(self.values[istart:istop:istep])
            edges.append(stop)
            # values for new Dimension
            if label == 'start':
                out_values = edges[:-1]
            elif istep % 2:
                loc = np.arange(istart + istep / 2, istop, istep)
                out_values = np.interp(loc, np.arange(len(self.values)), self.values)
            else:
                out_values = self.values[istart + istep // 2: istop: istep]
        else:
            if stop is None:
                latest_stop = self.values[-1] + step
                edges = np.arange(start, latest_stop, step)
            else:
                edges = np.arange(start, stop + step / 10, step)
                if edges[-1] != stop:
                    raise ValueError(f"start={start}, stop={stop}, step={step}: stop not at bin edge")
            interp = scipy.interpolate.interp1d(self.values, np.arange(len(self.values)), 'cubic', fill_value='extrapolate')
            edges_i = interp(edges)
            if stop is None and round(edges_i[-1]) != len(self.values):
                raise ValueError(f"start={start}, step={step}: dimension end point not at bin edge")
            edges = list(edges)

            # new dimensions
            if label == 'start':
                out_values = edges[:-1]
            else:
                values_i = [(a + b) / 2 for a, b in intervals(edges_i)]
                out_values = np.interp(values_i, edges_i, edges)
        out_dim = Scalar(self.name, out_values, self.unit, self.tick_format)
        return edges, out_dim

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : None | Dataset
            A dataset with variables describing cluster properties.
        """
        ds = Dataset()
        where = [np.flatnonzero(cluster) for cluster in x]
        ds[f'{self.name}_min'] = Var([self.values[w[0]] for w in where])
        ds[f'{self.name}_max'] = Var([self.values[w[-1]] for w in where])
        return ds

    def _cluster_property_labels(self):
        return [f'{self.name}_min', f'{self.name}_max']

    @classmethod
    def _concatenate(cls, dims: Sequence[Scalar]):
        dims = list(dims)
        name = cls._concatenate_attr(dims, 'name')
        unit = cls._concatenate_attr(dims, 'unit')
        tick_format = cls._concatenate_attr(dims, 'tick_format')
        values = np.concatenate([s.values for s in dims])
        adjacency = cls._concatenate_adjacency(dims)
        return cls(name, values, unit, tick_format, adjacency)

    def _array_index(self, arg):
        if isinstance(arg, self.__class__):
            s_idx, a_idx = np.nonzero(self.values[:, None] == arg.values)
            return s_idx[np.argsort(a_idx)]
        elif np.isscalar(arg):
            try:
                return digitize_index(arg, self.values, 0.3)
            except IndexError as error:
                raise IndexError(f"{error.args[0]}: Ambiguous index for {self._dimname()}")
        elif isinstance(arg, np.ndarray) and arg.dtype.kind == self.values.dtype.kind:
            if np.setdiff1d(arg, self.values):
                raise IndexError(f"Index {arg!r} includes values not in dimension: {np.setdiff1d(arg, self.values)}")
            return np.digitize(arg, self.values, True)
        else:
            return Dimension._array_index(self, arg)

    def _array_index_for_slice(self, start, stop=None, step=None):
        if start is not None:
            start = digitize_slice_endpoint(start, self.values)
        if stop is not None:
            stop = digitize_slice_endpoint(stop, self.values)
        return slice(start, stop, step)

    def _dim_index(self, index):
        if np.isscalar(index):
            return self.values[index]
        else:
            return Dimension._dim_index(self, index)

    def intersect(self, dim, check_dims=False):
        """Create a dimension object that is the intersection with dim

        Parameters
        ----------
        dim : type(self)
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : type(self)
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        if np.all(self.values == dim.values):
            return self
        index = np.isin(self.values, dim.values)
        if np.all(index):
            return self
        elif index.sum() == len(dim):
            return dim
        return self[index]


# for unpickling backwards compatibility
Ordered = Scalar


class Sensor(Dimension):
    """Dimension class for representing sensor information

    Parameters
    ----------
    locations : array_like  (n_sensor, 3)
        list of sensor locations in ALS coordinates, i.e., for each sensor a
        ``(anterior, left, superior)`` coordinate triplet.
    names
        Sensor names, same order as ``locs`` (default is ``['0', '1', '2',
        ...]``).
    sysname
        Name of the sensor system.
    proj2d
        Default 2d projection (default is ``'z-root'``; for options see notes
        below).
    adjacency : str | list of (str, str) | array of int, (n_edges, 2)
        Adjacency between elements. Can be specified as:

        - ``"none"`` for no connections
        - ``"grid"`` to use adjacency in the sensor names
        - list of connections (e.g., ``[('OZ', 'O1'), ('OZ', 'O2'), ...]``)
        - :class:`numpy.ndarray` of int, shape (n_edges, 2), to specify
          connections in terms of indices. Each row should specify one
          connection ``[i, j]`` with ``i < j``. If the array's dtype is
          ``uint32``, property checks are disabled to improve efficiency.

    Attributes
    ----------
    channel_idx : dict
        Dictionary mapping channel names to indexes.
    locations : array  (n_sensors, 3)
        Spatial position of all sensors.
    names : list of str
        Ordered list of sensor names.
    right : NDVar
        Sensor position along left-right axis.
    anterior : numpy.array  (n_sensors,)
        Sensor position along posterior-anterior axis.
    superior : numpy.array  (n_sensors,)
        Sensor position along inferior-superior axis.

    Notes
    -----
    The following are possible 2d-projections:

    ``'z root'``:
        the radius of each sensor is set to equal the root of the vertical
        distance from the top of the net.
    ``'cone'``:
        derive x/y coordinate from height based on a cone transformation.
    ``'lower cone'``:
        only use cone for sensors with z < 0.
    Axis and sign :
        For example, ``x+`` for anterior, ``x-`` for posterior.

    Examples
    --------
    >>> locations = [(0,  0,   0),
    ...              (0, -.25, -.45)]
    >>> sensor_dim = Sensor(locations, names=["Cz", "Pz"])
    """
    _default_adjacency = 'custom'
    _proj_aliases = {'left': 'x-', 'right': 'x+', 'back': 'y-', 'front': 'y+', 'top': 'z+', 'bottom': 'z-'}

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            locations: Sequence,
            names: Sequence[str] = None,
            sysname: str = None,
            proj2d: str = 'z root',
            adjacency: Union[str, Sequence] = 'none',
    ):
        # 'z root' transformation fails with 32-bit floats
        self.locations = locations = np.asarray(locations, dtype=np.float64)
        n = len(locations)
        if locations.shape != (n, 3):
            raise ValueError(f"locs needs to have shape (n_sensors, 3), got {locations.shape=}")
        elif np.isnan(locations).any():
            raise ValueError("locs contain NaN values")
        self.sysname = sysname
        self.default_proj2d = self._interpret_proj(proj2d)

        if names is None:
            names = [str(i) for i in range(n)]
        elif len(names) != n:
            raise ValueError(f"Length mismatch: got {n} locs but {len(names)} names")
        self.names = Datalist(names)
        Dimension.__init__(self, 'sensor', adjacency)
        self._init_secondary()

    @cached_property
    @deprecated('0.42', 'use .locations instead')
    def locs(self):
        return self.locations

    def _coerce_adjacency(self, adjacency):
        if isinstance(adjacency[0][0], str):
            return adjacency_from_name_pairs(adjacency, self.names, allow_missing=True)
        else:
            return Dimension._coerce_adjacency(self, adjacency)

    def _init_secondary(self):
        self.x = self.locations[:, 0]
        self.y = self.locations[:, 1]
        self.z = self.locations[:, 2]

        self.channel_idx = {name: i for i, name in enumerate(self.names)}
        # short names
        prefix = os.path.commonprefix(self.names)
        if prefix:
            n_pf = len(prefix)
            self.channel_idx.update({name[n_pf:]: i for i, name in
                                     enumerate(self.names)})

        # cache for transformed locations
        self._transformed = {}

    def __getstate__(self):
        out = Dimension.__getstate__(self)
        out.update(proj2d=self.default_proj2d, locs=self.locations, names=self.names, sysname=self.sysname)
        return out

    def __setstate__(self, state):
        if 'name' not in state:
            state['name'] = 'sensor'
            state['connectivity_type'] = 'custom'
        Dimension.__setstate__(self, state)
        self.locations = state['locs']
        self.names = state['names']
        self.sysname = state['sysname']
        self.default_proj2d = state['proj2d']
        self._init_secondary()

    def __repr__(self):
        return "<Sensor n=%i, name=%r>" % (len(self), self.sysname)

    def __len__(self):
        return len(self.locations)

    def _eq(self, other, check: bool):  # Based on equality of sensor names
        return Dimension._eq(self, other, check) and np.all(other.names == self.names)

    def __getitem__(self, index):
        if np.isscalar(index):
            return self.names[index]
        else:
            return Sensor(self.locations[index], self.names[index], self.sysname,
                          self.default_proj2d, self._subgraph(index))

    def _as_uv(self):
        return Factor(self.names, name=self.name)

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        locator = FixedLocator(np.arange(len(self)), 10)
        formatter = IndexFormatter(self.names)
        return formatter, locator, self._axis_label(label)

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool  (n_clusters, n_sensors)
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : None | Dataset
            A dataset with variables describing cluster properties.
        """
        return Dataset({'n_sensors': Var(x.sum(1))})

    def _cluster_property_labels(self):
        return ['n_sensors']

    def _array_index(self, arg):
        "Convert a dimension-semantic index to an array-like index"
        if isinstance(arg, str):
            return self.channel_idx[arg]
        elif isinstance(arg, Sensor):
            return np.array([self.names.index(name) for name in arg.names])
        elif isinstance(arg, Integral) or (isinstance(arg, np.ndarray) and arg.dtype.kind == 'i'):
            return arg
        else:
            return super(Sensor, self)._array_index(arg)

    def _array_index_to(self, other):
        "Int index to access data from self in an order consistent with other"
        try:
            return np.array([self.names.index(name) for name in other.names])
        except ValueError:
            missing = (name for name in other.names if name not in self.names)
            raise IndexError(f"{other}: contains different sensors {', '.join(missing)}")

    def _dim_index(self, index):
        if np.isscalar(index):
            return self.names[index]
        else:
            return Dimension._dim_index(self, index)

    def _distances(self):
        return squareform(pdist(self.locations))

    def _generate_adjacency(self):
        raise RuntimeError("Sensor adjacency is not defined. Use Sensor.set_adjacency().")

    @classmethod
    def from_xyz(cls, path=None, **kwargs):
        """Create a Sensor instance from a text file with xyz coordinates"""
        locs = []
        names = []
        with open(path) as f:
            l1 = f.readline()
            n = int(l1.split()[0])
            for line in f:
                elements = line.split()
                if len(elements) == 4:
                    x, y, z, name = elements
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    locs.append((x, y, z))
                    names.append(name)
        assert len(names) == n
        return cls(locs, names, **kwargs)

    @classmethod
    def from_sfp(cls, path=None, **kwargs):
        """Create a Sensor instance from an sfp file"""
        locs = []
        names = []
        for line in open(path):
            elements = line.split()
            if len(elements) == 4:
                name, x, y, z = elements
                x = float(x)
                y = float(y)
                z = float(z)
                locs.append((x, y, z))
                names.append(name)
        return cls(locs, names, **kwargs)

    @classmethod
    def from_lout(cls, path=None, transform_2d=None, **kwargs):
        """Create a Sensor instance from a ``*.lout`` file"""
        kwargs['transform_2d'] = transform_2d
        locs = []
        names = []
        with open(path) as fileobj:
            fileobj.readline()
            for line in fileobj:
                w, x, y, t, f, name = line.split('\t')
                x = float(x)
                y = float(y)
                locs.append((x, y, 0))
                names.append(name)
        return cls(locs, names, **kwargs)

    @classmethod
    def from_montage(cls, montage, channels=None):
        """From :class:`~mne.channels.DigMontage`

        Parameters
        ----------
        montage : str | mne.channels.DigMontage
            Montage, or name to load a standard montage (see
            :func:`mne.channels.make_standard_montage`).
        channels : list of str
            Channel names in the desired order (optional).
        """
        if isinstance(montage, str):
            sysname = montage
            obj = mne.channels.make_standard_montage(montage)
            try:
                cm, names = mne.channels.read_ch_adjacency(montage)
                adjacency = _matrix_graph(cm)
            except ValueError:
                adjacency = 'none'
        else:
            sysname = None
            obj = montage
            adjacency = 'none'

        if isinstance(obj, mne.channels.DigMontage):
            digs = [dig for dig in obj.dig if dig['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG]
            locations = np.vstack([dig['r'] for dig in digs])
            names = obj.ch_names
        elif hasattr(mne.channels, 'Montage') and isinstance(obj, mne.channels.Montage):  # Montage removed in 0.20
            locations = obj.pos
            names = obj.ch_names
        else:
            raise TypeError(f'montage={obj!r}')

        if channels is not None:
            index = np.array([names.index(ch) for ch in channels])
            locations = locations[index]
            names = [names[i] for i in index]
            if not isinstance(adjacency, str):
                adjacency = _subgraph_edges(adjacency, index)

        return cls(locations, names, sysname, adjacency=adjacency)

    def _interpret_proj(self, proj):
        if proj == 'default':
            return self.default_proj2d
        elif proj in self._proj_aliases:
            return self._proj_aliases[proj]
        elif proj is None:
            return 'z+'
        else:
            return proj

    @cached_property
    def right(self):
        return NDVar(self.x, self)

    @cached_property
    def anterior(self):
        return NDVar(self.y, self)

    @cached_property
    def superior(self):
        return NDVar(self.z, self)

    def get_adjacency(self):
        """Sensor adjacency as list of ``(name_1, name_2)``"""
        if self._adjacency_type != 'custom':
            raise ValueError("No custom adjacency")
        pairs = [(self.names[a], self.names[b]) for a, b in self._adjacency]
        sorted_pairs = [tuple(sorted(pair)) for pair in pairs]
        return sorted(sorted_pairs)

    def get_locs_2d(
            self,
            proj: str = 'default',
            extent: float = 1,
            frame: float = 0,
            invisible: bool = True,
    ):
        """Compute a 2 dimensional projection of the sensor locations

        Parameters
        ----------
        proj
            How to transform 3d coordinates into a 2d map; see class
            documentation for options.
        extent
            coordinates will be scaled with minimum value 0 and maximum value
            defined by the value of ``extent``.
        frame
            Distance of the outermost points from 0 and ``extent`` (default 0).
        invisible
            Return invisible sensors (sensors that would be hidden behind the
            head; default True).

        Returns
        -------
        locs_2d : array (n_sensor, 2)
            Sensor position 2d projection in x, y coordinates.
        """
        proj = self._interpret_proj(proj)

        index = (proj, extent, frame)
        if index in self._transformed:
            locs2d = self._transformed[index]
        else:
            locs2d = self._make_locs_2d(proj, extent, frame)
            self._transformed[index] = locs2d

        if not invisible:
            visible = self._visible_sensors(proj)
            if visible is not None:
                return locs2d[visible]
        return locs2d

    @cached_property
    def _sphere_fit(self):
        """Fit the 3d sensor locations to a sphere

        Returns
        -------
        params : tuple
            Radius and center (r, cx, cy, cz).
        """
        # initial guess of sphere parameters (radius and center)
        center_0 = np.mean(self.locations, 0)
        radius_0 = np.mean(np.sqrt(np.sum((self.locations - center_0) ** 2, axis=1)))
        # error function
        if len(self) >= 6:
            def err(params):
                # params: [cx, cy, cz, rx, ry, rz]
                centered = self.locations - params[:3]  # -> c=0
                centered /= params[3:]  # -> r=1
                centered **= 2
                length = np.sum(centered, 1)
                length -= 1
                return length
            radius_0 = [radius_0, radius_0, radius_0]
        else:
            def err(params):
                # params: [cx, cy, cz, r]
                centered = self.locations - params[:3]  # -> c=0
                centered **= 2
                length = np.sum(centered, 1)
                length -= params[3]
                return length
        start_params = np.hstack((center_0, radius_0))
        estimate, _ = scipy.optimize.leastsq(err, start_params)
        center = estimate[:3]
        radius = estimate[3:] if len(estimate) == 6 else estimate[3]
        return center, radius

    def _make_locs_2d(
            self,
            proj: str,
            extent: float,
            frame: float,
    ):
        if proj in ('cone', 'lower cone', 'z root'):
            # center the sensor locations based on the sphere and scale to radius 1
            center, radius = self._sphere_fit
            locs3d = self.locations - center
            locs3d /= radius

            # implement projection
            locs2d = np.copy(locs3d[:, :2])

            if proj == 'cone':
                locs2d[:, [0, 1]] *= (1 - locs3d[:, [2]])
            elif proj == 'lower cone':
                lower_half = locs3d[:, 2] < 0
                if any(lower_half):
                    locs2d[lower_half] *= (1 - locs3d[lower_half][:, [2]])
            elif proj == 'z root':
                z = locs3d[:, 2]
                z_dist = (z.max() + 0.01) - z  # distance form top (add a small
                # buffer so that multiple points at z-max don't get stuck
                # together)
                r = np.sqrt(z_dist)  # desired 2d radius
                r_xy = np.sqrt(np.sum(locs3d[:, :2] ** 2, 1))  # current radius in xy
                idx = (r_xy != 0)  # avoid zero division
                stretch = r[idx] / r_xy[idx]  # stretching factor accounting for current r
                locs2d[idx, :] *= stretch[:, None]
        else:
            match = re.match('([xyz])([+-])', proj)
            if match:
                ax, sign = match.groups()
                if ax == 'x':
                    locs2d = np.copy(self.locations[:, 1:])
                    if sign == '-':
                        locs2d[:, 0] *= -1
                elif ax == 'y':
                    locs2d = np.copy(self.locations[:, [0, 2]])
                    if sign == '+':
                        locs2d[:, 0] *= -1
                elif ax == 'z':
                    locs2d = np.copy(self.locations[:, :2])
                    if sign == '-':
                        locs2d[:, 1] *= -1
            else:
                raise ValueError("invalid proj kwarg: %r" % proj)

        # correct extent
        if extent:
            locs2d -= np.min(locs2d, axis=0)  # move to bottom left
            locs2d /= (np.max(locs2d) / extent)  # scale to extent
            locs2d += (extent - np.max(locs2d, axis=0)) / 2  # center
            if frame:
                locs2d *= (1 - 2 * frame)
                locs2d += frame

        return locs2d

    def _topomap_outlines(self, proj):
        "Outline argument for mne-python topomaps"
        proj = self._interpret_proj(proj)
        if proj in ('cone', 'lower cone', 'z root', 'z+'):
            return 'top'
        else:
            return None

    def _visible_sensors(self, proj):
        "Create an index for sensors that are visible under a given proj"
        proj = self._interpret_proj(proj)
        match = re.match('([xyz])([+-])', proj)
        if match:
            # logger.debug("Computing sensors visibility for %s" % proj)
            ax, sign = match.groups()

            # depth:  + = closer
            depth = self.locations[:, 'xyz'.index(ax)]
            if sign == '-':
                depth = -depth

            locs2d = self.get_locs_2d(proj)

            n_vertices = len(locs2d)
            all_vertices = np.arange(n_vertices)
            out = np.ones(n_vertices, bool)

            # find duplicate points
            # TODO OPT:  use pairwise distance
            x, y = np.where(cdist(locs2d, locs2d) == 0)
            duplicate_vertices = ((v1, v2) for v1, v2 in zip(x, y) if v1 < v2)
            for v1, v2 in duplicate_vertices:
                if depth[v1] > depth[v2]:
                    out[v2] = False
                    # logger.debug("%s is hidden behind %s" % (self.names[v2], self.names[v1]))
                else:
                    out[v1] = False
                    # logger.debug("%s is hidden behind %s" % (self.names[v1], self.names[v2]))
            use_vertices = all_vertices[out]  # use for hull check

            hull = ConvexHull(locs2d[use_vertices])
            hull_vertices = use_vertices[hull.vertices]

            # for each point:
            # find the closest point on the hull
            # determine whether it's in front or behind
            non_hull_vertices = np.setdiff1d(use_vertices, hull_vertices, True)

            hull_locs = locs2d[hull_vertices]
            non_hull_locs = locs2d[non_hull_vertices]
            dists = cdist(non_hull_locs, hull_locs)

            closest = np.argmin(dists, 1)
            hide_non_hull_vertices = depth[non_hull_vertices] < depth[hull_vertices][closest]
            hide_vertices = non_hull_vertices[hide_non_hull_vertices]
            # logger.debug("%s are hidden behind convex hull" % ' '.join(self.names[hide_vertices]))
            out[hide_vertices] = False
            return out
        else:
            return None

    def index(
            self,
            include: Union[str, Sequence[str, int]] = None,
            exclude: Union[str, Sequence[str, int]] = None,
    ) -> NDVar:
        """Construct an index for specified sensors

        Parameters
        ----------
        include
            Sensors to exclude (by name or index).
        exclude
            Sensors to exclude (by name or index).

        Returns
        -------
        index
            Boolean :class:`NDVar` indexing selected channels.
        """
        if (include is None) == (exclude is None):
            raise TypeError(f"{include=}, {exclude=}: Need to specify exactly one of include, exclude")
        elif include is not None:
            if isinstance(include, str):
                include = [include]
            index = np.zeros(len(self), dtype=bool)
            for ch in include:
                try:
                    index[self.channel_idx[ch]] = True
                except KeyError:
                    raise ValueError(f"Invalid channel name: {ch!r}")
            return NDVar(index, (self,))
        else:
            return ~self.index(exclude)

    def _normalize_sensor_names(self, names, missing='raise'):
        "Process a user-input list of sensor names"
        valid_chs = set()
        missing_chs = set()
        int_to_name = None
        for name in names:
            if name in self.names:
                valid_chs.add(name)
                continue
            elif isinstance(name, Integral) or name.isdigit():
                key = int(name) if isinstance(name, str) else name
                if int_to_name is None:
                    int_to_name = {}
                    for ch_name in self.names:
                        if match := re.search(r'\d+', ch_name):
                            int_to_name[int(match.group())] = ch_name

                if key in int_to_name:
                    valid_chs.add(int_to_name[key])
                    continue
            missing_chs.add(name)

        if missing == 'raise':
            if missing_chs:
                raise ValueError(f"The following channels are not in the raw data: {', '.join(sorted(missing_chs))}")
            return sorted(valid_chs)
        elif missing == 'return':
            return sorted(valid_chs), missing_chs
        else:
            raise ValueError(f"{missing=}")

    def intersect(self, dim, check_dims=True):
        """Create a Sensor dimension that is the intersection with dim

        Parameters
        ----------
        dim : Sensor
            Sensor dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (e.g., channel locations). Default
            is ``True``. Set to ``False`` to intersect channels based on names
            only and ignore mismatch between locations for channels with the
            same name.

        Returns
        -------
        sensor : Sensor
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        n_self = len(self)
        names = set(self.names).intersection(dim.names)
        n_intersection = len(names)
        if n_intersection == n_self:
            return self
        elif n_intersection == len(dim.names):
            return dim

        index = np.array([name in names for name in self.names])
        if check_dims:
            other_index = np.array([name in names for name in dim.names])
            if not np.all(self.locations[index] == dim.locations[other_index]):
                raise DimensionMismatchError("Sensor locations don't match between dimension objects")
        return self[index]

    def neighbors(self, connect_dist: float) -> Dict[int, np.ndarray]:
        """Find neighboring sensors.

        Parameters
        ----------
        connect_dist
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor.

        Returns
        -------
        neighbors
            Dictionaries whose keys are sensor indices, and whose values are
            lists of neighbors represented as sensor indices.
        """
        nb = {}
        pairwise_distances = squareform(pdist(self.locations))
        np.fill_diagonal(pairwise_distances, pairwise_distances.max() * 99)
        for i, distances in enumerate(pairwise_distances):
            nb[i] = np.flatnonzero(distances < (distances.min() * connect_dist))
        return nb

    def set_adjacency(self, neighbors=None, connect_dist=None):
        """Define the sensor adjacency through neighbors or distance

        Parameters
        ----------
        neighbors : sequence of (str, str)
            A list of connections, all assumed to be bidirectional.
        connect_dist : None | scalar
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor.
            e.g., 1.75 or 1.6
        """
        pairs = set()
        if neighbors is not None and connect_dist is not None:
            raise TypeError("Can only specify either neighbors or connect_dist")
        elif connect_dist is None:
            for src, dst in neighbors:
                a = self.names.index(src)
                b = self.names.index(dst)
                if a < b:
                    pairs.add((a, b))
                else:
                    pairs.add((b, a))
        else:
            nb = self.neighbors(connect_dist)
            for k, vals in nb.items():
                for v in vals:
                    if k < v:
                        pairs.add((k, v))
                    else:
                        pairs.add((v, k))

        self._adjacency = np.array(sorted(pairs), np.uint32)
        self._adjacency_type = 'custom'

    @property
    def values(self):
        return self.names


def as_sensor(obj) -> Sensor:
    "Coerce to Sensor instance"
    if isinstance(obj, Sensor):
        return obj
    elif isinstance(obj, NDVar):
        return obj.get_dim('sensor')
    elif isinstance(obj, (mne.Info, mne.channels.channels.UpdateChannelsMixin)):
        from .load.mne import sensor_dim
        return sensor_dim(obj)
    else:
        raise TypeError(f"Can't get sensors from {obj}")


def _point_graph(coords, dist_threshold):
    "Adjacency graph for points based on distance"
    n = len(coords)
    dist = pdist(coords)

    # construct vertex pairs corresponding to dist
    graph = np.empty((len(dist), 2), np.uint32)
    i0 = 0
    for vert, di in enumerate(range(n - 1, 0, -1)):
        i1 = i0 + di
        graph[i0:i1, 0] = vert
        graph[i0:i1, 1] = np.arange(vert + 1, n)
        i0 = i1

    return graph[dist < dist_threshold]


def _matrix_graph(matrix):
    "Create adjacency from matrix"
    coo = matrix.tocoo()
    assert np.all(coo.data)
    edges = {(min(a, b), max(a, b)) for a, b in zip(coo.col, coo.row) if a != b}
    return np.array(sorted(edges), np.uint32)


def _tri_graph(tris):
    """Create adjacency graph from triangles

    Parameters
    ----------
    tris : array_like, (n_tris, 3)
        Triangles.

    Returns
    -------
    edges : array (n_edges, 2)
        All edges between vertices of tris.
    """
    pairs = set()
    for tri in tris:
        a, b, c = sorted(tri)
        pairs.add((a, b))
        pairs.add((a, c))
        pairs.add((b, c))
    return np.array(sorted(pairs), np.uint32)


def _mne_tri_soure_space_graph(source_space, vertices_list):
    "Adjacency graph for a triangulated mne source space"
    i = 0
    graphs = []
    for ss, verts in zip(source_space, vertices_list):
        if len(verts) == 0:
            continue

        tris = ss['use_tris']
        if tris is None:
            raise ValueError("Adjacency unavailable. The source space does "
                             "not seem to be an ico source space.")

        # graph for the whole source space
        src_vertices = ss['vertno']
        graph = _tri_graph(tris)

        # select relevant edges
        if not np.array_equal(verts, src_vertices):
            if not np.all(np.isin(verts, src_vertices)):
                raise RuntimeError("Not all vertices are in the source space")
            edge_in_use = np.logical_and(np.isin(graph[:, 0], verts),
                                         np.isin(graph[:, 1], verts))
            graph = graph[edge_in_use]

        # reassign vertex ids based on present vertices
        if len(verts) != verts.max() + 1:
            graph = (np.digitize(graph.ravel(), verts, True)
                     .reshape(graph.shape).astype(np.uint32))

        # account for index of previous source spaces
        if i > 0:
            graph += i
        i += len(verts)

        graphs.append(graph)
    return np.vstack(graphs)


class SourceSpaceBase(Dimension):
    _kinds = ()
    _default_parc = 'aparc'
    _default_adjacency = 'custom'
    _vertex_re = re.compile(r'([RL])(\d+)')

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            vertices: List[ArrayLike],
            subject: str,
            src: str,
            subjects_dir: PathArg,
            parc: Union[Factor, str],
            adjacency: AdjacencyArg,
            name: str,
            filename: str,
    ):
        self.vertices = [np.asarray(vertices_i, int) for vertices_i in vertices]
        self.subject = subject
        self.src = src
        self._subjects_dir = str(subjects_dir)
        self._filename = filename
        # secondary attributes (also set when unpickling)
        self._init_secondary()
        Dimension.__init__(self, name, adjacency)

        # parc
        if parc is None or parc is False:
            self.parc = None
        elif isinstance(parc, Factor):
            if len(parc) != self._n_vert:
                raise ValueError(f"{parc=}: wrong length {len(parc)} for SourceSpace with {self._n_vert} vertices")
            self.parc = parc
        elif isinstance(parc, str):
            self.parc = self._read_parc(parc)
        else:
            raise TypeError(f"{parc=}: needs to be Factor or string")

    def _read_parc(self, parc: str) -> Factor:
        raise NotImplementedError(f"{parc=}: can't set parcellation from annotation files for {self.__class__.__name__}. Consider using a Factor instead.")

    def _init_secondary(self):
        # The source-space type is needed to determine adjacency
        m = SRC_RE.match(self.src)
        if not m:
            raise ValueError(f"src={self.src!r}")
        kind, grade, suffix = m.groups()
        if kind not in self._kinds:
            raise ValueError(f'src={self.src!r}: {self.__class__.__name__} is wrong class')
        self.kind = kind
        self.grade = int(grade)
        self._n_vert = sum(len(v) for v in self.vertices)

    @classmethod
    def from_file(
            cls,
            subjects_dir: PathArg,
            subject: str,
            src: str,
            parc: str = None,
            label: mne.Label = None,
            source_spaces: mne.SourceSpaces = None,  # speeds up initialization
    ):
        """SourceSpace dimension from MNE source space file"""
        if source_spaces is None:
            filename = Path(subjects_dir) / subject / 'bem' / f'{subject}-{src}-src.fif'
            source_spaces = mne.read_source_spaces(str(filename))
        return cls._from_mne(subjects_dir, subject, src, parc, label, source_spaces)

    @classmethod
    def from_mne_source_spaces(
            cls,
            source_spaces: mne.SourceSpaces,
            src: str,
            subjects_dir: PathArg,
            parc: str = None,
            label: mne.Label = None,
    ):
        """SourceSpace dimension from :class:`mne.SourceSpaces` object

        Notes
        -----
        The ``source_spaces`` are permanently stored, which can increase file
        size when pickling. To avoid this, use :meth:`from_file` instead.
        """
        subject = source_spaces[0]['subject_his_id']
        return cls._from_mne(subjects_dir, subject, src, parc, label, source_spaces, filename=source_spaces)

    @classmethod
    def _from_mne(
            cls,
            subjects_dir: PathArg,
            subject: str,
            src: str,
            parc: str = None,
            label: mne.Label = None,
            source_spaces: mne.SourceSpaces = None,  # speeds up initialization
            **kwargs,
    ):
        if parc is None:
            parc = cls._default_parc
        if label is None:
            vertices = [ss['vertno'] for ss in source_spaces]
        else:
            vertices, _ = label_src_vertno_sel(label, source_spaces)
        return cls(vertices, subject, src, subjects_dir, parc, **kwargs)

    @property
    def subjects_dir(self):
        if self._subjects_dir:
            return self._subjects_dir
        else:
            return mne.utils.get_subjects_dir(self._subjects_dir, True)

    @subjects_dir.setter
    def subjects_dir(self, value: Optional[PathArg]):
        self._subjects_dir = value

    def _sss_path(self, subjects_dir=None):
        if subjects_dir is None:
            subjects_dir = self.subjects_dir
        return Path(subjects_dir) / self.subject / 'bem' / self._filename.format(subject=self.subject, src=self.src)

    def __getstate__(self):
        state = Dimension.__getstate__(self)
        state.update(vertno=self.vertices, subject=self.subject, src=self.src, subjects_dir=self._subjects_dir, parc=self.parc, filename=self._filename)
        return state

    def __setstate__(self, state):
        if 'name' not in state:
            state['name'] = 'source'
            state['connectivity_type'] = 'custom'
        Dimension.__setstate__(self, state)
        self.vertices = state['vertno']
        self.subject = state['subject']
        self.src = state['src']
        self._subjects_dir = state['subjects_dir']
        self._filename = state.get('filename', '{subject}-{src}-src.fif')
        self.parc = state['parc']
        self._init_secondary()

    def __repr__(self):
        out = "<" + self.__class__.__name__
        if self.name != 'source':
            out += ' ' + self.name
        vert_repr = ', '.join(str(len(v)) for v in self.vertices)
        out += " [%s], %r" % (vert_repr, self.subject)
        if self.src is not None:
            out += ', %r' % self.src
        if self.parc is not None:
            out += ', parc=%s' % self.parc.name
        return out + '>'

    def __len__(self):
        return self._n_vert

    def _eq(self, other, check: bool):
        return (
            Dimension._eq(self, other, check)
            and self.subject == other.subject
            and self.src == other.src
            and all(np.array_equal(s, o) for s, o in zip(self.vertices, other.vertices)))

    def _assert_same_base(self, other):
        "Assert that ``other`` is based on the same source space"
        if self.subject != other.subject:
            raise IndexError(f"Source spaces can not be compared because they are defined on different MRI subjects ({self.subject} vs {other.subject}). Consider using eelbrain.morph_source_space().")
        elif self.src != other.src:
            raise IndexError(f"Source spaces of different types ({self.src} vs {other.src})")
        elif self.subjects_dir != other.subjects_dir:
            raise IndexError(f"Source spaces have differing subjects_dir:\n{self.subjects_dir}\n{other.subjects_dir}\nUse load.update_subjects_dir to set a common directory.")

    def __getitem__(self, index):
        raise NotImplementedError

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        formatter = FormatStrFormatter('%i')
        locator = FixedLocator(np.arange(len(self)), 10)
        return formatter, locator, self._axis_label(label)

    def _copy(self, subject=None, parc=None):
        if subject is None:
            subject = self.subject
        elif not isinstance(self._filename, str):
            raise ValueError(f"Can't change subject on {self.__class__.__name__} with embedded MNE.SourceSpace")
        if parc is None:
            parc = self.parc
        return self.__class__(self.vertices, subject, self.src, self.subjects_dir, parc, self._subgraph(), self.name, self._filename)

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : Dataset
            A dataset with variables describing cluster properties along this
            dimension: "n_sources".
        """
        if np.any(np.sum(x, 1) == 0):
            raise ValueError("Empty cluster")

        ds = Dataset()

        # no clusters
        if len(x) == 0:
            ds['n_sources'] = Var([])
            if self.parc is not None:
                ds['location'] = Factor([])
            return ds

        # n sources
        ds['n_sources'] = Var(x.sum(1))

        # location
        if self.parc is not None:
            locations = []
            for x_ in x:
                parc_entries = self.parc[x_]
                argmax = np.argmax(np.bincount(parc_entries.x))
                location = parc_entries[argmax]
                locations.append(location)
            ds['location'] = Factor(locations)

        return ds

    def _cluster_property_labels(self):
        out = ['n_sources']
        if self.parc:
            out.append('location')
        return out

    def _distances(self):
        "Surface distances between source space vertices"
        # don't cache for memory reason (4687 -> 168 MB)
        dist = -np.ones((self._n_vert, self._n_vert))
        sss = self.get_source_space()
        i0 = 0
        for vertices, ss in zip(self.vertices, sss):
            if ss['dist'] is None:
                path = self._sss_path()
                raise RuntimeError(f"Source space does not contain source distance information. To add distance information, run:\nsrc = mne.read_source_spaces({path!r})\nmne.add_source_space_distances(src)\nsrc.save({path!r}, overwrite=True)")
            i = i0 + len(vertices)
            dist[i0:i, i0:i] = ss['dist'][vertices, vertices[:, None]].toarray()
            i0 = i
        return dist

    def adjacency(self, disconnect_parc=False):
        """Create source space adjacency

        Parameters
        ----------
        disconnect_parc : bool
            Reduce adjacency to label-internal connections.

        Returns
        -------
        connetivity : array of int, (n_pairs, 2)
            array of sorted [src, dst] pairs, with all src < dts.
        """
        if self._n_vert == 0:
            return np.empty((0, 2), np.uint32)
        elif self._adjacency is None:
            if self.src is None or self.subject is None or self.subjects_dir is None:
                raise ValueError(
                    "In order for a SourceSpace dimension to provide "
                    "adjacency information it needs to be initialized with "
                    "src, subject and subjects_dir parameters")

            self._adjacency = adjacency = self._compute_adjacency()
            assert adjacency.max() < len(self)
        else:
            adjacency = self._adjacency

        if disconnect_parc:
            parc = self.parc
            if parc is None:
                raise RuntimeError("SourceSpace has no parcellation (use "
                                   ".set_parc())")
            idx = np.array([parc[s] == parc[d] for s, d in adjacency])
            adjacency = adjacency[idx]

        return adjacency

    def _compute_adjacency(self):
        raise NotImplementedError("Adjacency for %r source space" % self.kind)

    def circular_index(self, seeds, extent=0.05, name="globe"):
        """Return an index into all vertices closer than ``extent`` of a seed

        Parameters
        ----------
        seeds : array_like, (3,) | (n, 3)
            Seed location(s) around which to build index.
        extent : float
            Index vertices closer than this (in m in 3d space).
        name : str
            Name of the NDVar.

        Returns
        -------
        roi : NDVar  ('source',)
            Index into the spherical area around ``seeds``.
        """
        seeds = np.atleast_2d(seeds)
        dist = cdist(self.coordinates, seeds)
        mindist = np.min(dist, 1)
        x = mindist < extent
        dims = (self,)
        info = {'seeds': seeds, 'extent': extent}
        return NDVar(x, dims, name, info)

    @cached_property
    def coordinates(self):
        sss = self.get_source_space()
        coords = [ss['rr'][v] for ss, v in zip(sss, self.vertices)]
        return np.vstack(coords)

    @cached_property
    def normals(self):
        sss = self.get_source_space()
        normals = [ss['nn'][v] for ss, v in zip(sss, self.vertices)]
        return np.vstack(normals)

    def _array_index(self, arg, allow_vertex=True):
        if isinstance(arg, str):
            if self.parc is not None and arg in self.parc:
                return self.parc == arg
            elif allow_vertex:
                return self._array_index_for_vertex(arg)
            elif self.parc is None:
                raise IndexError(f"{arg!r}: {self.__class__.__name__} has no parcellation")
            else:
                raise IndexError(f"{arg!r}: {self.__class__.__name__} has no such label")
        elif isinstance(arg, Integral) or (isinstance(arg, np.ndarray) and arg.dtype.kind in 'ib'):
            return arg
        elif isinstance(arg, Sequence):
            if all(isinstance(v, str) for v in arg):
                # First, try to interpret all items as labels
                label_index = None
                vertices = arg
                if self.parc is not None:
                    # FIXME: Some volume labels may be too small to apply to a single voxel (e.g., transversetemporal with 10 mm grid). This could be handled better by storing all label names in self.parc.info to detect valid label names that are not present in the data
                    labels = [a for a in arg if a in self.parc]
                    if labels:
                        label_index = self.parc.isin(arg)
                        if len(labels) == len(arg):
                            return label_index
                        vertices = [a for a in arg if a not in self.parc]
                # For items that are not in the parcellation, try to interpret them as vertex IDs
                if allow_vertex:
                    vertex_index = [self._array_index_for_vertex(v) for v in vertices]
                    if label_index is None:
                        return vertex_index
                    else:
                        label_index[vertex_index] = True
                        return label_index
                else:
                    raise IndexError(f"Unknown labels: {vertices}")
            elif all(isinstance(v, INT_TYPES) for v in arg):
                return arg
            else:
                raise IndexError(f"{arg!r}")
        else:
            return Dimension._array_index(self, arg)

    def _array_index_for_vertex(self, vertex_desc):
        raise NotImplementedError(f"Index for {vertex_desc!r}")

    def _array_index_to(self, other):
        "Int index to access data from self in an order consistent with other"
        self._assert_same_base(other)
        if any(np.any(np.setdiff1d(o, s, True)) for s, o in zip(self.vertices, other.vertices)):
            raise IndexError(f"{other}: contains sources not in {self}")
        bool_index = np.hstack([np.isin(s, o) for s, o in zip(self.vertices, other.vertices)])
        return np.flatnonzero(bool_index)

    def get_source_space(self, subjects_dir=None):
        "Read the corresponding MNE source space"
        if isinstance(self._filename, mne.SourceSpaces):
            return self._filename
        if self.src is None:
            raise TypeError("Unknown source-space. Specify the src parameter when initializing SourceSpace.")
        path = self._sss_path(subjects_dir)
        if path.exists():
            return mne.read_source_spaces(str(path))
        else:
            return self._setup_source_space()

    def _setup_source_space(self):
        raise NotImplementedError

    def index_for_label(self, label):
        """Return the index for a label

        Parameters
        ----------
        label : str | sequance of str
            One or several names of regions in the current parcellation.

        Returns
        -------
        index : NDVar of bool
            Index into the source space dim that corresponds to the label.
        """
        idx = self._array_index(label, allow_vertex=False)
        if isinstance(label, str):
            name = label
        elif isinstance(label, Sequence):
            name = '+'.join(label)
        elif isinstance(label, MNE_LABEL):
            name = label.name
        else:
            raise TypeError(f"{label!r}")
        return NDVar(index_to_bool_array(idx, len(self)), (self,), name)

    def _is_superset_of(self, dim):
        self._assert_same_base(dim)
        return all(np.all(np.isin(d, s)) for s, d in zip(self.vertices, dim.vertices))

    def index_into_dim(self, dim):
        if not self._is_superset_of(dim):
            raise ValueError(f"{dim}: Index source space has unknown vertices")
        index = np.hstack([np.isin(s, d) for s, d in zip(self.vertices, dim.vertices)])
        return NDVar(index, (self,))

    def intersect(self, dim, check_dims=True):
        """Create a Source dimension that is the intersection with dim

        Parameters
        ----------
        dim : Source
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : Source
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        self._assert_same_base(dim)
        index = np.hstack([np.isin(s, o) for s, o in zip(self.vertices, dim.vertices)])
        return self[index]

    def _melt_vars(self) -> dict:
        if self.parc is None:
            return {}
        else:
            return {'parc': self.parc}

    @property
    def values(self):
        raise NotImplementedError


class SourceSpace(SourceSpaceBase):
    """MNE surface-based source space

    Parameters
    ----------
    vertices
        The vertex identities of the dipoles in the source space.
        Specified as a list of two integer arrays, identifying the vertices in
        the left and the right hemisphere separately.
        Can include an empty list in case no vertices are present in a
        hemisphere.
    subject
        The mri-subject name.
    src
        The kind of source space used (e.g., 'ico-4'; only ``ico`` is currently
        supported.
    subjects_dir
        The path to the subjects_dir (needed to locate the source space
        file).
    parc
        Add a parcellation to the source space to identify vertex location.
        Only applies to ico source spaces, default is 'aparc'.
        Use ``None`` to initialize without parcellation.
    adjacency
        Adjacency between elements. Set to ``"none"`` for no connections or
        ``"grid"`` to use adjacency in the sequence of elements as connection.
        Set to :class:`numpy.ndarray` to specify custom adjacency. The array
        should be of shape (n_edges, 2), and each row should specify one
        connection [i, j] with i < j, with rows sorted in ascending order. If
        the array's dtype is uint32, property checks are disabled to improve
        efficiency.
    name
        Dimension name (default ``"source"``).
    filename
        Filename template for the MNE source space file.

    Attributes
    ----------
    coordinates : array (n_sources, 3)
        Spatial coordinate for each source.
    normals : array (n_sources, 3)
        Orientation (direction) of each source.
    parc : Factor
        Parcellation (one label for each source).

    See Also
    --------
    VolumeSourceSpace : volume source space

    Notes
    -----
    besides numpy indexing, the following indexes are possible:

     - mne Label objects
     - 'lh' or 'rh' to select an entire hemisphere

    """
    _kinds = ('ico', 'oct')

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            vertices: List[ArrayLike],
            subject: str = None,
            src: str = None,
            subjects_dir: PathArg = None,
            parc: Union[Factor, str] = 'aparc',
            adjacency: AdjacencyArg = 'custom',
            name: str = 'source',
            filename: str = '{subject}-{src}-src.fif',
    ):
        SourceSpaceBase.__init__(self, vertices, subject, src, subjects_dir, parc, adjacency, name, filename)

    def _init_secondary(self):
        SourceSpaceBase._init_secondary(self)
        assert len(self.vertices) == 2, "ico-based SourceSpaces need exactly two vertices arrays"
        self.lh_vertices = self.vertices[0]
        self.rh_vertices = self.vertices[1]
        self.lh_n = len(self.lh_vertices)
        self.rh_n = len(self.rh_vertices)

    @cached_property
    def hemi(self):
        return Factor(['lh', 'rh'], repeat=[self.lh_n, self.rh_n])

    def _read_parc(self, parc: str) -> Factor:
        label_dir = Path(self.subjects_dir) / self.subject / 'label'
        codes = []
        labels = {}
        for hemi, vertices in zip(('lh', 'rh'), self.vertices):
            if not len(vertices):
                continue
            annot, ctab, names = read_annot(label_dir / f'{hemi}.{parc}.annot')
            annot = annot[vertices]
            first_index = annot.min()
            if first_index >= 1000:  # fsaverage_sym
                annot -= first_index
            if hemi == 'rh' and labels:
                offset = max(labels) + 2
                annot += offset
            else:
                offset = 0
            codes.append(annot)
            labels[-1 + offset] = f'unknown-{hemi}'
            labels.update({i + offset: f'{name.decode()}-{hemi}' for i, name in enumerate(names)})
        return Factor(np.concatenate(codes), parc, labels=labels)

    def __iter__(self):
        return (temp % v for temp, vertices in
                zip(('L%i', 'R%i'), self.vertices) for v in vertices)

    def __getitem__(self, index):
        if isinstance(index, Integral):
            if index < self.lh_n:
                return 'L%i' % self.lh_vertices[index]
            elif index < self._n_vert:
                return 'R%i' % self.rh_vertices[index - self.lh_n]
            else:
                raise IndexError("SourceSpace Index out of range: %i" % index)

        int_index = index_to_int_array(index, self._n_vert)
        bool_index = np.bincount(int_index, minlength=self._n_vert).astype(bool)

        # vertices
        boundaries = np.cumsum(list(chain((0,), (len(v) for v in self.vertices))))
        vertices = [v[bool_index[start: stop]] if stop > start else np.empty(0, int)
                    for v, (start, stop) in zip(self.vertices, intervals(boundaries))]

        # parc
        parc = None if self.parc is None else self.parc[index]

        return SourceSpace(vertices, self.subject, self.src, self.subjects_dir,
                           parc, self._subgraph(int_index), self.name)

    def _as_uv(self):
        return Factor((f'{hemi}{i}' for hemi, vertices in zip(('L', 'R'), self.vertices) for i in vertices), name=self.name)

    def _cluster_properties(self, x):
        ds = SourceSpaceBase._cluster_properties(self, x)
        # hemi
        hemis = []
        for x_ in x:
            where = np.flatnonzero(x_)
            src_in_lh = (where < self.lh_n)
            if np.all(src_in_lh):
                hemis.append('lh')
            elif np.any(src_in_lh):
                hemis.append('bh')
            else:
                hemis.append('rh')
        ds['hemi'] = Factor(hemis)
        return ds

    def _cluster_property_labels(self):
        return [*SourceSpaceBase._cluster_property_labels(), 'hemi']

    @classmethod
    def _concatenate(cls, dims: Sequence[SourceSpace]):
        dims = list(dims)
        subject = cls._concatenate_attr(dims, 'subject')
        src = cls._concatenate_attr(dims, 'src')
        subjects_dir = cls._concatenate_attr(dims, 'subjects_dir')
        name = cls._concatenate_attr(dims, 'name')
        filename = cls._concatenate_attr(dims, '_filename')
        # vertices
        dims_vertices = [dim.vertices for dim in dims]
        vertices = [np.concatenate(vertices) for vertices in zip(*dims_vertices)]
        if any(len(np.unique(v)) < len(v) for v in vertices):
            raise ValueError("Can't concatenate SourceSpace that overlap")
        argsorts = [np.argsort(v) for v in vertices]
        vertices = [v[argsort] for v, argsort in zip(vertices, argsorts)]
        # parc
        parcs = [dim.parc for dim in dims]
        if any(parc is None for parc in parcs):
            parc = None
        else:
            parc = combine(parcs)
            i0 = 0
            for argsort in argsorts:
                i1 = i0 + len(argsort)
                parc[i0:i1] = parc[i0 + argsort]
                i0 = i1
        return SourceSpace(vertices, subject, src, subjects_dir, parc, name=name, filename=filename)

    def _link_midline(self, maxdist=0.015):
        """Link sources in the left and right hemispheres

        Link each source to the nearest source in the opposing hemisphere if
        that source is closer than ``maxdist``.

        Parameters
        ----------
        maxdist : scalar [m]
            Add an interhemispheric connection between any two vertices whose
            distance is less than this number (in meters; default 0.015).
        """
        if self.kind != 'ico':
            raise ValueError("Can only link hemispheres in 'ico' source spaces")
        old_con = self.adjacency()

        # find vertices to connect
        coords_lh = self.coordinates[:self.lh_n]
        coords_rh = self.coordinates[self.lh_n:]
        dists = cdist(coords_lh, coords_rh)
        close_lh, close_rh = np.nonzero(dists < maxdist)
        unique_close_lh = np.unique(close_lh)
        unique_close_rh = np.unique(close_rh)
        new_con = {(lh, np.argmin(dists[lh]) + self.lh_n) for lh in
                   unique_close_lh}
        new_con.update((np.argmin(dists[:, rh]), rh + self.lh_n) for rh in
                       unique_close_rh)
        new_con = np.array(sorted(new_con), np.uint32)
        self._adjacency = np.vstack((old_con, new_con))

    def _compute_adjacency(self):
        src = self.get_source_space()
        if self.kind == 'oct':
            raise NotImplementedError("Adjacency for oct source space")
        return _mne_tri_soure_space_graph(src, self.vertices)

    def _array_index(self, arg, allow_vertex=True):
        if isinstance(arg, mne.BiHemiLabel):
            lh_idx = self._array_index_hemilabel(arg.lh)
            rh_idx = self._array_index_hemilabel(arg.rh)
            return np.hstack((lh_idx, rh_idx))
        elif isinstance(arg, mne.Label):
            idx = np.zeros(len(self), bool)
            idx_part = self._array_index_hemilabel(arg)
            if arg.hemi == 'lh':
                idx[:self.lh_n] = idx_part
            elif arg.hemi == 'rh':
                idx[self.lh_n:] = idx_part
            else:
                raise ValueError(f"{arg!r} with unknown value for label.hemi: {arg.hemi!r}")
            return idx
        elif isinstance(arg, str):
            if arg == 'lh':
                return slice(self.lh_n)
            elif arg == 'rh':
                if self.rh_n:
                    return slice(self.lh_n, None)
                else:
                    return slice(0, 0)
        elif isinstance(arg, self.__class__):
            sv = self.vertices
            ov = arg.vertices
            if all(np.array_equal(s, o) for s, o in zip(sv, ov)):
                return FULL_SLICE
            elif any(any(np.setdiff1d(o, s)) for o, s in zip(ov, sv)):
                raise IndexError("Index contains unknown sources")
            else:
                return np.hstack([np.isin(s, o, True) for s, o in zip(sv, ov)])
        return SourceSpaceBase._array_index(self, arg, allow_vertex)

    def _array_index_for_vertex(self, vertex_desc):
        m = self._vertex_re.match(vertex_desc)
        if m is None:
            raise IndexError(f"{vertex_desc!r}: neither a label nor a valid vertex description")
        hemi, vertex = m.groups()
        vertex = int(vertex)
        vertices = self.vertices[hemi == 'R']
        i = int(np.searchsorted(vertices, vertex))
        if vertices[i] == vertex:
            if hemi == 'R':
                return i + self.lh_n
            else:
                return i
        else:
            raise IndexError(f"{vertex_desc!r}: SourceSpace does not contain this vertex")

    def _array_index_hemilabel(self, label: mne.Label):
        stc_vertices = self.vertices[label.hemi == 'rh']
        idx = np.isin(stc_vertices, label.vertices, True)
        return idx

    def _dim_index(self, index):
        if np.isscalar(index):
            if index >= self.lh_n:
                return 'R%i' % (self.rh_vertices[index - self.lh_n])
            else:
                return 'L%i' % (self.lh_vertices[index])
        else:
            return SourceSpaceBase._dim_index(self, index)

    def _label(self, vertices, name, color, subjects_dir=None, sss=None):
        lh_vertices, rh_vertices = vertices
        if sss is None:
            sss = self.get_source_space(subjects_dir)

        if len(lh_vertices):
            lh = mne.Label(lh_vertices, hemi='lh', color=color).fill(sss, name + '-lh')
        else:
            lh = None

        if len(rh_vertices):
            rh = mne.Label(rh_vertices, hemi='rh', color=color).fill(sss, name + '-rh')
        else:
            rh = None

        return lh, rh

    def _mask_label(self, subjects_dir=None):
        "Create a Label that masks the areas not covered in this SourceSpace"
        sss = self.get_source_space(subjects_dir)
        if self.lh_n:
            lh_verts = np.setdiff1d(sss[0]['vertno'], self.lh_vertices)
        else:
            lh_verts = ()

        if self.rh_n:
            rh_verts = np.setdiff1d(sss[1]['vertno'], self.rh_vertices)
        else:
            rh_verts = ()

        return self._label((lh_verts, rh_verts), 'mask', (0, 0, 0), subjects_dir, sss)

    def _mask_ndvar(self, subjects_dir=None):
        if subjects_dir is None:
            subjects_dir = self.subjects_dir
        sss = self.get_source_space(subjects_dir)
        vertices = [sss[0]['vertno'], sss[1]['vertno']]
        data = [np.isin(vert, self_vert) for vert, self_vert in
                zip(vertices, self.vertices)]
        source = SourceSpace(vertices, self.subject, self.src, subjects_dir,
                             self.parc.name, name=self.name)
        return NDVar(np.concatenate(data), (source,))

    def _melt_vars(self):
        return {'hemi': self.hemi, **SourceSpaceBase._melt_vars(self)}

    def _read_surf(self, hemi, surf='orig'):
        path = Path(f'{self.subjects_dir}/{self.subject}/surf/{hemi}.{surf}')
        return read_geometry(path)

    def index_for_label(self, label: Union[str, Sequence[str], mne.Label, mne.BiHemiLabel]) -> NDVar:
        """Return the index for a label

        Parameters
        ----------
        label
            The name of a region in the current parcellation, ``'lh'``, ``'rh'``,
            or an :mod:`mne`:class:`~mne.label.Label` object. If the label does not
            match any sources in the SourceEstimate, a ValueError is raised.

        Returns
        -------
        index : boolean NDVar
            Index into the source space dim that corresponds to the label.
        """
        return SourceSpaceBase.index_for_label(self, label)

    def surface_coordinates(self, surf='white'):
        """Load surface coordinates for any FreeSurfer surface

        Parameters
        ----------
        surf : str
            Name of the FreeSurfer surface.

        Returns
        -------
        coords : array (n_sources, 3)
            Coordinates for each source contained in the source space.
        """
        out = []
        for hemi, vertices in zip(('lh', 'rh'), self.vertices):
            if len(vertices) == 0:
                continue
            coords, _ = self._read_surf(hemi, surf)
            out.append(coords[vertices])

        if len(out) == 1:
            return out[0]
        else:
            return np.vstack(out)

    def _setup_source_space(self):
        spacing = self.src.replace('-', '')
        return mne.setup_source_space(self.subject, spacing, subjects_dir=self.subjects_dir, n_jobs=1, add_dist=False)


class VolumeSourceSpace(SourceSpaceBase):
    """MNE volume source space

    Parameters
    ----------
    vertices
        Integer array with vertex identities of the source space.
    subject
        The mri-subject name.
    src
        The description of the source space (e.g. ``vol-7``).
    subjects_dir
        The path to the FreeSurfer subjects-directory (needed to locate the
        source space file).
    parc
        Add a parcellation to the source space to identify vertex location.
    adjacency : 'grid' | 'none' | array of int, (n_edges, 2)
        Adjacency between elements. Set to ``"none"`` for no connections or
        ``"grid"`` to use adjacency in the sequence of elements as connection.
        Set to :class:`numpy.ndarray` to specify custom adjacency. The array
        should be of shape (n_edges, 2), and each row should specify one
        connection [i, j] with i < j, with rows sorted in ascending order. If
        the array's dtype is uint32, property checks are disabled to improve
        efficiency.
    name
        Dimension name (default ``"source"``).
    filename
        Filename template for the MNE source space file.

    See Also
    --------
    SourceSpace : surface-based source space
    """
    _kinds = ('vol',)
    _default_parc = None  # early version of mne-Python did not scale parcellations for scaled brains

    @deprecate_kwarg('connectivity', 'adjacency', '0.41', '0.42')
    def __init__(
            self,
            vertices: ArrayLike,
            subject: str = None,
            src: str = None,
            subjects_dir: PathArg = None,
            parc: str = None,
            adjacency: AdjacencyArg = 'custom',
            name: str = 'source',
            filename: str = '{subject}-{src}-src.fif',
    ):
        if isinstance(vertices, np.ndarray):
            vertices = [vertices]
        SourceSpaceBase.__init__(self, vertices, subject, src, subjects_dir, parc, adjacency, name, filename)

    def _read_parc(self, parc: str) -> Factor:
        return self._read_volume_parc(self.subjects_dir, self.subject, parc, self.coordinates)

    @staticmethod
    def _read_volume_parc(
            subjects_dir: PathArg,
            subject: str,
            parc: str,
            coordinates: np.ndarray,
    ) -> Factor:
        path = Path(subjects_dir) / subject / 'mri' / f'{parc}.mgz'
        if not path.exists():
            raise ValueError(f"{parc=}: parcellation does not exist at {path}")
        mgz = nibabel.load(str(path))
        voxel_to_mri = mgz.affine.copy()
        voxel_to_mri[:3] /= 1000
        mri_to_voxel = inv(voxel_to_mri)
        voxel_coords = mne.transforms.apply_trans(mri_to_voxel, coordinates)
        voxel_coords = np.round(voxel_coords).astype(int)
        x, y, z = voxel_coords.T
        data = numpy.asanyarray(mgz.dataobj)
        x = data[x, y, z]
        labels = mne_utils.get_volume_source_space_labels()
        return Factor(x, labels=labels, name=parc)

    def _init_secondary(self):
        SourceSpaceBase._init_secondary(self)
        if len(self.vertices) != 1:
            raise ValueError("A VolumeSourceSpace needs exactly one vertices array")

    @cached_property
    def hemi(self):
        return Factor(np.sign(self.coordinates[:, 0]), labels={-1: 'lh', 0: 'midline', 1: 'rh'})

    @cached_property
    def lh_n(self):
        return np.sum(self.hemi == 'lh')

    @cached_property
    def rh_n(self):
        return np.sum(self.hemi == 'rh')

    def __iter__(self):
        return map(str, self.vertices[0])

    def __getitem__(self, index):
        if isinstance(index, Integral):
            try:
                return str(self.vertices[0][index])
            except IndexError:
                raise IndexError("VolumeSourceSpace Index out of range: %i" % index)
        else:
            parc = None if self.parc is None else self.parc[index]
            return VolumeSourceSpace([self.vertices[0][index]], self.subject, self.src, self.subjects_dir, parc, self._subgraph(index), self.name, self._filename)

    def _as_uv(self):
        return Factor(self.vertices[0], name=self.name)

    def _compute_adjacency(self):
        src = self.get_source_space()
        coords = src[0]['rr'][self.vertices[0]]
        dist_threshold = self.grade * 0.0011
        return _point_graph(coords, dist_threshold)

    def _distances(self):
        sss = self.get_source_space()
        coords = sss[0]['rr'][self.vertices[0]]
        return squareform(pdist(coords))

    def _array_index(self, arg, allow_vertex=True):
        if isinstance(arg, str):
            if arg in ('lh', 'rh'):
                return self.hemi == arg
        return SourceSpaceBase._array_index(self, arg, allow_vertex)

    def _array_index_for_vertex(self, vertex_desc):
        m = re.match(r'(\d+)$', vertex_desc)
        if m is None:
            raise IndexError(f"{vertex_desc!r}: neither a label nor a valid vertex description")
        return int(np.searchsorted(self.vertices[0], int(m.group(1))))

    def _dim_index(self, index):
        if np.isscalar(index):
            return str(self.vertices[0][index])
        else:
            return SourceSpaceBase._dim_index(self, index)


class UTS(Dimension):
    """Dimension object for representing uniform time series

    Parameters
    ----------
    tmin
        First time point (inclusive).
    tstep
        Time step between samples.
    nsamples
        Number of samples.
    unit
        Unit used for plotting.
    name
        Dimension name, used as :class:`NDVar` attribute name.

    Attributes
    ----------
    tmin : float
        Lowest time point in seconds.
    tmax : float
        Largest time point [s].
    tstep : float
        Time step for each sample [s].
    nsamples : int
        Number of samples.
    tstop : float
        Time sample after ``tmax`` [s] (consistent with indexing excluding end
        point).
    times : array (nsamples,)
        Array with all time points.

    Notes
    -----
    Special indexing:

    (tstart, tstop) : tuple
        Restrict the time to the indicated window (either end-point can be
        None).

    """
    _default_adjacency = 'grid'
    _tol = 0.000001  # tolerance for deciding if time values are equal
    plot_s_as_ms: float = 3  # plot s as ms when the absolute times in s are smaller than this

    def __init__(
            self,
            tmin: float,
            tstep: float,
            nsamples: int,
            unit: str = 's',
            name: str = 'time',
    ):
        Dimension.__init__(self, name, 'grid')
        self.tmin = float(tmin)  # Python float has superior precision
        self.tstep = float(tstep)
        self.nsamples = int(nsamples)
        self._axis_unit = self.unit = unit
        self._init_secondary()

    def _init_secondary(self):
        self.tmax = self.tmin + self.tstep * (self.nsamples - 1)
        self.tstop = self.tmin + self.tstep * self.nsamples
        self._n_decimals = max(n_decimals(self.tmin), n_decimals(self.tstep))

    @cached_property
    def times(self):
        return self.tmin + np.arange(self.nsamples) * self.tstep

    @classmethod
    def from_int(cls, first, last, sfreq):
        """Create a UTS dimension from sample index and sampling frequency

        Parameters
        ----------
        first : int
            Index of the first sample, relative to 0.
        last : int
            Index of the last sample, relative to 0.
        sfreq : scalar
            Sampling frequency, in Hz.
        """
        tmin = first / sfreq
        nsamples = last - first + 1
        tstep = 1. / sfreq
        return cls(tmin, tstep, nsamples)

    @classmethod
    def from_range(cls, tstart, tstop, tstep):
        """Create time axis from interval and samplingrate"""
        n_samples = int((tstop - tstart) // tstep)
        return cls(tstart, tstep, n_samples)

    def __getstate__(self):
        out = Dimension.__getstate__(self)
        out.update(tmin=self.tmin, tstep=self.tstep, nsamples=self.nsamples, unit=self.unit)
        return out

    def __setstate__(self, state):
        if 'name' not in state:
            state['name'] = 'time'
            state['connectivity'] = None
            state['connectivity_type'] = 'grid'
        Dimension.__setstate__(self, state)
        self.tmin = state['tmin']
        self.tstep = state['tstep']
        self.nsamples = state['nsamples']
        self.unit = state.get('unit', 's')
        self._init_secondary()

    def __repr__(self):
        args = [self.tmin, self.tstep, self.nsamples]
        if self.unit != 's':
            args.append(self.unit)
        args = ', '.join(map(repr, args))
        return f"UTS({args})"

    def _as_scalar_array(self):
        return self.times

    def _axis_data(self):
        return self.times

    def _axis_extent(self):
        return self.tmin, self.tmax

    def _axis_im_extent(self):
        return self.tmin - 0.5 * self.tstep, self.tmax + 0.5 * self.tstep

    def _axis_format(
            self,
            scalar: bool,
            label: Union[bool, str],
    ):
        # display s -> ms
        s_to_ms = False
        if self.unit == 's' and (v := self.plot_s_as_ms):
            s_to_ms = -v < self.tmax < v and -v < self.tmin < v

        if label is True or label == '__unit__':
            if s_to_ms:
                unit = 'ms'
            else:
                unit = self.unit
            if label == '__unit__':
                label = unit
            else:
                label = f"{self.name.capitalize()} [{unit}]"

        if s_to_ms:
            if scalar:
                fmt = FuncFormatter(lambda x, pos: f'{1e3 * x:.0f}')
            else:
                fmt = FuncFormatter(lambda x, pos: f'{1e3 * self.times[int(round(x))]:.0f}')
        else:
            if scalar:
                fmt = FuncFormatter(lambda x, pos: f'{x:.5g}')
            else:
                fmt = FuncFormatter(lambda x, pos: f'{self.times[x]:.5g}')
        return fmt, None, label

    def _bin(self, start, stop, step, nbins, label):
        if nbins is not None:
            raise NotImplementedError("nbins for UTS dimension")

        if start is None:
            start = self.tmin

        if stop is None:
            stop = self.tstop

        n_bins = int(ceil((stop - start) / step))
        edges = [start + n * step for n in range(n_bins)]
        # Drop last bin if it is empty
        if self._array_index(stop) > self._array_index(edges[-1]):
            edges.append(stop)
        else:
            n_bins -= 1
        # New dimension
        tmin = start + step / 2 if label == 'center' else start
        out_dim = UTS(tmin, step, n_bins)
        return edges, out_dim

    @classmethod
    def _concatenate(cls, dims: Sequence[UTS]):
        if len(tsteps := {uts.tstep for uts in dims}) > 1:
            raise ValueError(f'UTS dimensions have incompatible tstep: {dims}')
        tmin = dims[0].tmin
        tstep = tsteps.pop()
        n_samples = sum(uts.nsamples for uts in dims)
        return UTS(tmin, tstep, n_samples)

    def __len__(self):
        return self.nsamples

    def _eq(self, other, check: bool):
        return (Dimension._eq(self, other, check)
                and abs(self.tmin - other.tmin) < self._tol
                and abs(self.tstep - other.tstep) < self._tol)

    def __contains__(self, index):
        return self.tmin - self.tstep / 2 < index < self.tstop - self.tstep / 2

    def __getitem__(self, index):
        if isinstance(index, Integral):
            return self.times[index]
        elif not isinstance(index, slice):
            # convert index to slice
            int_index = index_to_int_array(index, self.nsamples)
            start = int_index[0]
            steps = np.unique(np.diff(int_index))
            if len(steps) > 1:
                raise NotImplementedError("non-uniform time series")
            step = steps[0]
            stop = int_index[-1] + step
            index = slice(start, stop, step)

        start = 0 if index.start is None else index.start
        if start < 0:
            start += self.nsamples
        if index.stop is None:
            stop = self.nsamples
        elif index.stop < 0:
            stop = index.stop + self.nsamples
        else:
            stop = min(index.stop, self.nsamples)

        tmin = self.times[start]
        nsamples = stop - start
        if nsamples < 0:
            raise IndexError(f"{index!r}: Time index out of range")

        if index.step is None or index.step == 1:
            tstep = self.tstep
        else:
            tstep = self.tstep * index.step
            nsamples = int(ceil(nsamples / index.step))

        return UTS(tmin, tstep, nsamples)

    def _cluster_bounds(self, x):
        """Cluster start and stop in samples

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.
        """
        # find indices of cluster extent
        row, col = np.nonzero(x)
        try:
            ts = [col[row == i][[0, -1]] for i in range(len(x))]
        except IndexError:
            raise ValueError("Empty cluster")
        ts = np.array(ts)
        return ts

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : Dataset
            A dataset with variables describing cluster properties along this
            dimension: "tstart", "tstop", "duration".
        """
        ds = Dataset()

        # no clusters
        if len(x) == 0:
            ds['tstart'] = Var([])
            ds['tstop'] = Var([])
            ds['duration'] = Var([])
            return ds

        # create time values
        bounds = self._cluster_bounds(x)
        tmin = self.times[bounds[:, 0]]
        tmax = self.times[bounds[:, 1]]
        ds['tstart'] = Var(tmin)
        ds['tstop'] = Var(tmax + self.tstep)
        ds['duration'] = ds.eval("tstop - tstart")
        return ds

    def _cluster_property_labels(self):
        return ['tstart', 'tstop', 'duration']

    def _array_index(self, arg, fraction=False):
        if np.isscalar(arg):
            i = (arg - self.tmin) / self.tstep
            if not fraction:
                i = int(round(i))
            if i < 0:
                raise ValueError(f"Time index {arg} out of range ({self.tmin}, {self.tmax})")
            return i
        elif fraction:
            raise NotImplementedError
        elif isinstance(arg, UTS):
            if self.tmin == arg.tmin:
                start = None
                stop = arg.nsamples
            elif arg.tmin < self.tmin:
                raise DimensionMismatchError("The index time dimension starts before the reference time dimension")
            else:
                start_float = (arg.tmin - self.tmin) / self.tstep
                start = int(round(start_float))
                if abs(start_float - start) > self._tol:
                    raise DimensionMismatchError("The index time dimension contains values not contained in the reference time dimension")
                stop = start + arg.nsamples

            if self.tstep == arg.tstep:
                step = None
            elif self.tstep > arg.tstep:
                raise DimensionMismatchError("The index time dimension has a higher sampling rate than the reference time dimension")
            else:
                step_float = arg.tstep / self.tstep
                step = int(round(step_float))
                if abs(step_float - step) > self._tol:
                    raise DimensionMismatchError("The index time dimension contains values not contained in the reference time dimension")

            if stop == self.nsamples:
                stop = None

            return slice(start, stop, step)
        elif isinstance(arg, np.ndarray) and arg.dtype.kind in 'fi':
            return np.array([self._array_index(i) for i in arg])
        else:
            return super(UTS, self)._array_index(arg)

    def _array_index_for_slice(self, start: float, stop: float = None, step: float = None):
        """Create a slice into the time axis

        Parameters
        ----------
        start
            Interpreted as cut-off, i.e., for an axis with times [0.001, 0.002, 0.003, ...]
            index [0.0021:] would start with sample 0.003.
        """
        if (start is not None) and (stop is not None) and (start >= stop):
            raise ValueError("tstart must be smaller than tstop")

        if start is None:
            start_ = None
        elif start < self.tmin:
            start_ = None  # numpy behavior
        else:
            start_float = (start - self.tmin) / self.tstep
            start_ = int(start_float)
            if start_float - start_ > 0.000001:
                start_ += 1

        if stop is None:
            stop_ = None
        else:
            stop_float = (stop - self.tmin) / self.tstep
            stop_ = int(stop_float)
            if stop_float - stop_ > 0.000001:
                stop_ += 1

        if step is None:
            step_ = None
        else:
            step_float = step / self.tstep
            step_ = int(round(step_float))
            if step_ != round(step_float, 4):
                raise ValueError(f"Time step {step}: needs to be a multiple of the data time step ({self.tstep})")

        return slice(start_, stop_, step_)

    def _array_index_to(self, other):
        "Int index to access data from self in an order consistent with other"
        if not isinstance(other, UTS):
            raise IndexError(f"{other}: not a valid index for UTS dimension")
        start = (other.tmin - self.tmin) / self.tstep
        if start % 1 or start < 0:
            raise IndexError(f"{other}: incompatible time axis")
        start = int(round(start))
        step = other.tstep / self.tstep
        if step % 1:
            raise IndexError(f"{other}: incompatible time axis")
        step = int(round(step))
        if other.tstop > self.tstop:
            raise IndexError(f"{other}: incompatible time axis")
        stop = start + other.nsamples
        return np.arange(start, stop, step)

    def _dim_index(self, arg):
        if isinstance(arg, slice):
            return slice(None if arg.start is None else self._dim_index(arg.start),
                         None if arg.stop is None else self._dim_index(arg.stop),
                         None if arg.step is None else arg.step * self.tstep)
        elif np.isscalar(arg):
            return round(self.tmin + arg * self.tstep, self._n_decimals)
        else:
            return Dimension._dim_index(self, arg)

    def intersect(self, dim, check_dims=True):
        """Create a UTS dimension that is the intersection with ``dim``

        Parameters
        ----------
        dim : UTS
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : UTS
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self == dim:
            return self
        elif self.tstep != dim.tstep:
            raise NotImplementedError("Intersection of UTS with unequal tstep :(")

        tstep = self.tstep
        tmin_diff = abs(self.tmin - dim.tmin) / tstep
        if abs(tmin_diff - round(tmin_diff)) > self._tol:
            raise DimensionMismatchError("UTS dimensions have different times")
        tmin = max(self.tmin, dim.tmin)

        tmax = min(self.tmax, dim.tmax)
        nsamples = int(round((tmax - tmin) / tstep)) + 1
        if nsamples <= 0:
            raise DimensionMismatchError("UTS dimensions don't overlap")

        return UTS(tmin, tstep, nsamples)

    def _union(self, other):
        # sloppy implementation
        if self == other:
            return self
        tstep = min(self.tstep, other.tstep)
        tmin = min(self.tmin, other.tmin)
        n_samples = int(round((max(self.tstop, other.tstop) - tmin) / tstep))
        return UTS(tmin, tstep, n_samples)


def dims_stackable(dims1: Sequence, dims2: Sequence, check_dims: bool):
    """Check whether two NDVars can be stacked"""
    if len(dims1) != len(dims2):
        return False
    return all(d1._eq(d2, check_dims) for d1, d2 in zip(dims1, dims2))


def intersect_dims(dims1, dims2, check_dims: bool = True):
    """Find the intersection between two multidimensional spaces

    Parameters
    ----------
    dims1, dims2 : tuple of dimension objects
        Two spaces involving the same dimensions with overlapping values.
    check_dims
        Check dimensions for consistency (e.g., channel locations in a Sensor
        dimension). Default is ``True``. Set to ``False`` to ignore non-fatal
        mismatches.

    Returns
    -------
    dims : tuple of Dimension objects
        Intersection of dims1 and dims2.
    """
    return tuple([d1.intersect(d2, check_dims) for d1, d2 in zip(dims1, dims2)])


EVAL_CONTEXT.update(Var=Var, Factor=Factor, extrema=extrema)

NDVarArg = Union[NDVar, str]
VarArg = Union[Var, str]
NumericArg = Union[Var, NDVar, str]
CategorialVariable = Union[Factor, Interaction, NestedEffect]
CategorialArg = Union[CategorialVariable, str]
AdjacencyArg = Union[Literal['grid', 'none', 'vector', 'custom'], ArrayLike]
FactorArg = Union[Factor, str]
CellArg = Union[str, Tuple[str, ...]]
IndexArg = Union[Var, np.ndarray, str]
ModelArg = Union[Model, Var, CategorialArg]
UVArg = Union[VarArg, CategorialArg]

# Types that can be coerced to NDVar with asndvar(); lists of those can be too
NDVarTypes = (NDVar, str, MNE_RAW, MNE_EPOCHS, MNE_EVOKED)
