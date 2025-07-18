"""Exceptions used throughout Eelbrain"""
from typing import Collection

from ._text import enumeration, plural


class KeysMissing(KeyError):
    "A styling dictionary is missing a key (more information than KeyError)"
    def __init__(self, keys: Collection, from_name: str, from_dict: dict):
        KeyError.__init__(self, keys, from_name, from_dict)

    def __str__(self):
        keys, from_name, from_dict = self.args
        n = len(keys)
        return f"{plural('Key', n)} {enumeration(map(repr, keys))} missing from {from_name}={from_dict!r}"


class DefinitionError(Exception):
    "MneExperiment definition error"


class EvalError(Exception):
    "Error while evaluating expression"
    def __init__(self, expression, exception, context):
        Exception.__init__(self, f"Error evaluating {expression!r} in {context}: {exception}")


class DimensionMismatchError(Exception):
    "Trying to align NDVars with mismatching dimensions"

    @classmethod
    def from_dims_list(cls, message, dims_list, check: bool):
        from ._data_obj import dims_stackable

        unique_dims = []
        for dims in dims_list:
            if any(dims_stackable(dims, dims_, check) for dims_ in unique_dims):
                continue
            else:
                unique_dims.append(dims)
        desc = '\n'.join(map(str, unique_dims))
        return cls(f'{message}\n{desc}')


class WrongDimensionError(Exception):
    "Dimension that is supported"


class IncompleteModelError(Exception):
    "Function requires a fully specified model"


class OldVersionError(Exception):
    "Trying to load a file from a version that is no longer supported"


class ZeroVarianceError(Exception):
    "Data with zero variance"
