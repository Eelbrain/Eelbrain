# Author: Proloy Das <proloyd94@gmail.com>
"""
optimized statistics functions [mostly handles the dispatching]

"""
from . import vector2d, vector3d


def t2_stat(y, out, n_dims=3):
    dispacher = vector2d if n_dims == 2 else vector3d
    return dispacher.t2_stat(y, out)


def mean_norm_rotated(y, rotation, out, n_dims=3):
    assert rotation.shape == (y.shape[0], n_dims, n_dims)
    return vector3d.mean_norm_rotated(y, rotation, out)


def t2_stat_rotated(y, rotation, out, n_dims=3):
    assert rotation.shape == (y.shape[0], n_dims, n_dims)
    dispacher = vector2d if n_dims == 2 else vector3d
    return dispacher.t2_stat_rotated(y, rotation, out)
