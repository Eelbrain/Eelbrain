"""TRF-local helpers for applying numpy math to NDVars."""

import math
from numbers import Number
from typing import Sequence, Union

import numpy as np

from ..._data_obj import NDVar


MUV = Union[NDVar, np.ndarray, Sequence, Number]


def element_wise(element_func, numpy_func):
    def func(x: MUV, name: str = None, info: dict = None):
        if isinstance(x, NDVar):
            return NDVar(numpy_func(x.x), x.dims, name, info)
        if isinstance(x, np.ndarray):
            return numpy_func(x)
        if isinstance(x, Sequence):
            return [element_func(xi) for xi in x]
        return element_func(x)
    return func


arctanh = element_wise(math.atanh, np.arctanh)
