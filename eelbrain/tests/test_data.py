# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from copy import deepcopy
from itertools import chain, product
from math import log
import operator
import os
import pickle
import shutil
from string import ascii_lowercase
import tempfile

import mne
import numpy as np
from numpy import newaxis
from numpy.testing import (
    assert_equal, assert_array_equal, assert_allclose,
    assert_array_almost_equal)
import pytest

from eelbrain import (
    datasets, load, Var, Factor, NDVar, Datalist, Dataset, Celltable,
    Case, Categorial, Scalar, Sensor, UTS, set_tmin,
    align, align1, choose, combine,
    cwt_morlet, shuffled_index)
from eelbrain._data_obj import (
    all_equal, asvar, assub, FULL_AXIS_SLICE, longname, SourceSpace,
    assert_has_no_empty_cells)
from eelbrain._exceptions import DimensionMismatchError
from eelbrain._stats.stats import rms
from eelbrain.testing import assert_dataobj_equal, assert_dataset_equal, assert_fmtxt_str_equals, assert_source_space_equal, requires_mne_sample_data, skip_on_windows


OPERATORS = {
    '+': (operator.add, operator.iadd),
    '-': (operator.sub, operator.isub),
    '*': (operator.mul, operator.imul),
    '/': (operator.truediv, operator.itruediv),
    '//': (operator.floordiv, operator.ifloordiv),
    '%': (operator.mod, operator.imod),
    '**': (pow, operator.ipow),
    '|': (operator.or_, operator.ior),
    '^': (operator.xor, operator.ixor),
    '&': (operator.and_, operator.iand),
}
FLOAT_OPERATORS = {k: ops for k, ops in OPERATORS.items() if k not in '|^&'}


def test_aggregate():
    "Test aggregation methods"
    ds = datasets.get_uts()
    drop = ('rm', 'ind', 'YBin', 'YCat')

    # don't handle inconsistencies silently
    with pytest.raises(ValueError):
        ds.aggregate('A%B')

    dsa = ds.aggregate('A%B', drop=drop)
    assert_array_equal(dsa['n'], [15, 15, 15, 15])
    idx1 = ds.eval("numpy.logical_and(A=='a0', B=='b0')")
    assert dsa['Y', 0] == ds['Y', idx1].mean()

    # unequal cell counts
    ds = ds[:-3]
    dsa = ds.aggregate('A%B', drop=drop)
    assert_array_equal(dsa['n'], [15, 15, 15, 12])
    idx1 = ds.eval("numpy.logical_and(A=='a0', B=='b0')")
    assert dsa['Y', 0] == ds['Y', idx1].mean()

    # equalize count
    dsa = ds.aggregate('A%B', drop=drop, equal_count=True)
    assert_array_equal(dsa['n'], [12, 12, 12, 12])
    idx1_12 = np.logical_and(idx1, idx1.cumsum() <= 12)
    assert dsa['Y', 0] == ds['Y', idx1_12].mean()

    # equalize count with empty cell
    sds = ds.sub("numpy.logical_or(A == 'a1', B == 'b1')")
    dsa = sds.aggregate('A%B', drop=drop, equal_count=True)
    assert_array_equal(dsa['n'], [12, 12, 12])


def test_align():
    "Testing align() and align1() functions"
    ds = datasets.get_uv()
    # index the dataset
    ds.index()
    ds['aindex'] = ds.eval("A.enumerate_cells()")
    # subset
    idx4 = np.arange(0, ds.n_cases, 4)
    idx4i = idx4[::-1]
    ds2 = ds.sub(np.arange(0, ds.n_cases, 2))
    # shuffle the whole dataset
    shuffle_index = np.arange(ds.n_cases)
    np.random.shuffle(shuffle_index)
    ds_shuffled = ds[shuffle_index]

    # align1: align Dataset to index
    dsa = align1(ds2, idx4)
    assert_array_equal(dsa['index'], idx4, "align1() failure")
    dsa = align1(ds2, idx4i)
    assert_array_equal(dsa['index'], idx4i, "align1() failure")
    # d_idx as Var
    dsa = align1(ds2[::2], idx4, idx4i)
    assert_array_equal(dsa['index'], idx4i, "align1() failure")
    with pytest.raises(ValueError):
        align1(ds2, idx4, idx4i)

    # Factor index
    with pytest.raises(ValueError):
        align1(ds, ds['rm', ::-1], 'rm')
    fds = ds[:20]
    dsa = align1(fds, fds['rm', ::-1], 'rm')
    assert_array_equal(dsa['index'], np.arange(19, -1, -1), "align1 Factor")

    # align two datasets
    dsa1, dsa2 = align(ds, ds2)
    assert_array_equal(dsa1['index'], dsa2['index'], "align() failure")
    dsa1, dsa2 = align(ds, ds2[::-1])
    assert_array_equal(dsa1['index'], dsa2['index'], "align() failure")
    dsa1, dsa2 = align(ds, ds_shuffled)
    assert_dataset_equal(dsa1, dsa2)

    # align using categorial
    dsa1, dsa2 = align(ds, ds_shuffled, 'A % aindex')
    assert_dataset_equal(dsa1, dsa2)
    dsa1, dsa2 = align(ds, ds_shuffled, 'aindex % A')
    assert_dataset_equal(dsa1, dsa2)


def test_celltable():
    "Test the Celltable class."
    ds = datasets.get_uts()
    ds['cat'] = Factor('abcd', repeat=15)

    ct = Celltable('Y', 'A', data=ds)
    assert ct.n_cases == 60
    assert ct.n_cells == 2
    assert repr(ct) == "Celltable(Y, A)"
    assert repr(Celltable(ds['Y'].x, 'A', data=ds)) == "Celltable(<ndarray>, A)"
    assert repr(Celltable(ds['Y'].x, ds['A'].x, data=ds)) == "Celltable(<ndarray>, <Factor>)"

    ct = Celltable('Y', 'A', match='rm', data=ds)
    assert ct.n_cases == 30
    assert ct.n_cells == 2

    # cat argument
    ct = Celltable('Y', 'cat', cat=('c', 'b'), data=ds)
    assert ct.n_cases == 30
    assert ct.x[0] == 'c'
    assert ct.x[-1] == 'b'
    with pytest.raises(ValueError):
        Celltable('Y', 'cat', cat=('c', 'e'), data=ds)

    ct = Celltable('Y', 'A', match='rm', data=ds)
    assert ct.n_cases == 30
    assert np.all(ct.groups['a0'] == ct.groups['a1'])

    ct = Celltable('Y', 'cat', match='rm', cat=('c', 'b'), data=ds)
    assert ct.n_cases == 30
    assert ct.x[0] == 'c'
    assert ct.x[-1] == 'b'

    # catch unequal length
    with pytest.raises(ValueError):
        Celltable(ds['Y', :-1], 'cat', data=ds)
    with pytest.raises(ValueError):
        Celltable(ds['Y', :-1], 'cat', match='rm', data=ds)

    # coercion of numerical X
    x = ds.eval("A == 'a0'")
    ct = Celltable('Y', x, cat=(None, None), data=ds)
    assert ct.cat == ('False', 'True')
    assert_array_equal(ct.data['True'], ds['Y', x])

    ct = Celltable('Y', x, cat=('True', 'False'), data=ds)
    assert ('True', 'False') == ct.cat
    assert_array_equal(ct.data['True'], ds['Y', x])

    # test coercion of Y
    ct = Celltable(ds['Y'].x, 'A', data=ds)
    assert isinstance(ct.y, np.ndarray)
    ct = Celltable(ds['Y'].x, 'A', data=ds, coercion=asvar)
    assert isinstance(ct.y, Var)

    # test sub
    ds_sub = ds.sub("A == 'a0'")
    ct_sub = Celltable('Y', 'B', data=ds_sub)
    ct = Celltable('Y', 'B', sub="A == 'a0'", data=ds)
    assert_dataobj_equal(ct_sub.y, ct.y)
    ct_sub = Celltable('Y', 'B', sub="Var(A == 'a0')", cat=('b0', 'b1'), data=ds)
    assert_dataobj_equal(ct_sub.y, ct.y)

    # test sub with rm
    ct_sub = Celltable('Y', 'B', match='rm', data=ds_sub)
    ct = Celltable('Y', 'B', match='rm', sub="A == 'a0'", data=ds)
    assert_dataobj_equal(ct_sub.y, ct.y)

    # Interaction match
    ct = Celltable('Y', 'A', match='B % rm', data=ds)
    assert ct.all_within
    assert_dataobj_equal(combine((ct.data['a0'], ct.data['a1'])), ds['Y'])

    # test rm sorting
    ds = Dataset()
    ds['rm'] = Factor('abc', repeat=4)
    ds['Y'] = Var(np.arange(3.).repeat(4))
    ds['X'] = Factor('ab', repeat=2, tile=3)
    idx = np.arange(12)
    np.random.shuffle(idx)
    ds = ds[idx]
    ct = Celltable('Y', 'X', 'rm', data=ds)
    assert_array_equal(ct.match, Factor('abc', tile=2))
    assert_array_equal(ct.y, np.tile(np.arange(3.), 2))
    assert_array_equal(ct.x, Factor('ab', repeat=3))


def test_coercion():
    "Test data class coercion"
    ds = datasets.get_uts()
    ds['avar'] = Var.from_dict(ds['A'], {'a0': 0, 'a1': 1})

    assert_array_equal(assub("A == 'a0'", ds), ds['A'] == 'a0')
    assert_array_equal(assub("avar == 0", ds), ds['avar'] == 0)


def test_choose():
    "Test choose()"
    ds = datasets.get_uts(True)[::4]
    utsnd = ds['utsnd']
    utsnd2 = utsnd + 1.
    idx = ds['B'] == 'b0'
    idxi = np.invert(idx)

    y = choose(idx, (utsnd, utsnd2))
    assert_array_equal(y.x[idx], utsnd2.x[idx])
    assert_array_equal(y.x[idxi], utsnd.x[idxi])

    with pytest.raises(DimensionMismatchError):
        choose(idx, (utsnd, utsnd.sub(sensor='1')))


def test_combine():
    "Test combine()"
    ds1 = datasets.get_uts()
    ds2 = datasets.get_uts()
    n = ds1.n_cases
    ds = combine((ds1, ds2))
    assert_array_equal(ds2['Y'].x, ds['Y'].x[n:])

    # list of numbers
    assert_dataobj_equal(combine((1., 2., 1.)), Var((1., 2., 1.)))
    assert_dataobj_equal(combine(('a', 'b', 'a')), Factor('aba'))

    # combine Datasets with unequal keys
    del ds1['Y']
    # raise
    with pytest.raises(ValueError):
        combine((ds1, ds2))
    with pytest.raises(ValueError):
        combine((ds2, ds1))
    # drop
    del ds2['YCat']
    ds = combine((ds1, ds2), incomplete='drop')
    assert 'Y' not in ds
    assert 'YCat' not in ds
    # fill in
    ds = combine((ds1, ds2), incomplete='fill in')
    assert_array_equal(ds['Y'].x[n:], ds2['Y'].x)
    assert_array_equal(np.isnan(ds['Y'].x[:n]), True)
    assert_array_equal(ds['YCat'][:n], ds1['YCat'])
    assert_array_equal(ds['YCat'][n:], '')

    # invalid input
    with pytest.raises(ValueError):
        combine(())
    with pytest.raises(TypeError):
        combine((ds2['A'], ds2['Y']))

    # combine NDVar with unequel dimensions
    ds = datasets.get_uts(utsnd=True)
    y = ds['utsnd']
    y1 = y.sub(sensor=['0', '1', '2', '3'])
    y2 = y.sub(sensor=['1', '2', '3', '4'])
    ds1 = Dataset((y1,), info={'a': np.arange(2), 'b': [np.arange(2)]})
    ds2 = Dataset((y2,), info={'a': np.arange(2), 'b': [np.arange(2)]})
    with pytest.raises(DimensionMismatchError):
        combine((ds1, ds2))
    dsc = combine((ds1, ds2), dim_intersection=True)
    y = dsc['utsnd']
    assert list(y.sensor.names) == ['1', '2', '3']
    dims = ('case', 'sensor', 'time')
    ref = np.concatenate((y1.get_data(dims)[:, 1:], y2.get_data(dims)[:, :3]))
    assert_array_equal(y.get_data(dims), ref, "combine utsnd")
    # info
    assert_array_equal(dsc.info['a'], np.arange(2))
    assert len(dsc.info['b']) == 1
    assert_array_equal(dsc.info['b'][0], np.arange(2))


def test_datalist():
    "Test Datalist class"
    dl = Datalist(range(10))

    # indexing
    assert dl[3] == 3
    x = dl[:3]
    assert isinstance(x, Datalist)
    assert_array_equal(x, np.arange(3))
    assert_array_equal(dl[8:], np.arange(8, 10))
    x = dl[np.arange(10) < 3]
    assert isinstance(x, Datalist)
    assert_array_equal(x, np.arange(3))
    assert_array_equal(dl[np.arange(3)], np.arange(3))
    assert_array_equal(dl[[True, False] * 5], np.arange(0, 10, 2))

    # __add__
    x = dl + list(range(10, 12))
    assert isinstance(x, Datalist)
    assert_array_equal(x, np.arange(12))

    # aggregate
    x = dl.aggregate(Factor('ab', repeat=5))
    assert isinstance(x, Datalist)
    assert_array_equal(x, [2.0, 7.0])

    # repr
    dl = Datalist([['a', 'b'], [], ['a']])
    assert str(dl) == "[['a', 'b'], [], ['a']]"
    dl = Datalist([['a', 'b'], [], ['a']], fmt='strlist')
    assert str(dl) == '[[a, b], [], [a]]'
    assert str(dl[:2]) == '[[a, b], []]'

    # eq
    a = Datalist([[], [1], [], [1]])
    b = Datalist([[], [], [2], [1]])
    assert_array_equal(a == b, [True, False, False, True])
    assert_array_equal(a != b, [False, True, True, False])

    # deepcopy
    ac = deepcopy(a)
    assert ac is not a
    assert_array_equal(ac == a, True)
    ac[0].append(1)
    assert_array_equal(ac == a, [False, True, True, True])

    # __setitem__
    ac[:2] = (1, 2)
    assert_array_equal(ac == [1, 2, [], [1]], True)
    ac[np.arange(2, 4)] = [3, 4]
    assert_array_equal(ac == list(range(1, 5)), True)
    with pytest.raises(ValueError):
        ac[np.arange(2)] = np.arange(3)

    # update
    a._update_listlist(b)
    assert_array_equal(a == [[], [1], [2], [1]], True)


def test_dataset():
    "Basic dataset operations"
    ds = Dataset()

    # naming
    ds['f'] = Factor('abab')
    assert ds['f'].name == 'f'

    # ds.add()
    with pytest.raises(ValueError):
        ds.add(Factor('aabb'))  # no name
    ds.add(Factor('aabb', name='g'))
    assert ds['g'].name == 'g'

    # initialization
    assert_dataobj_equal(Dataset([ds['f'], ds['g']]), ds)
    assert_dataobj_equal(Dataset({'f': Factor('abab'), 'g': Factor('aabb')}), ds)

    # aggregate
    ds = Dataset({
        'x': Factor('abab'),
        'y': Factor('aabb'),
        'z': Factor('aaaa'),
        'v': Var([1, 1, 2, 2]),
    })
    assert_dataset_equal(
        ds.aggregate('x', drop_bad=True),
        Dataset({
            'x': Factor('ab'),
            'z': Factor('aa'),
            'v': Var([1.5, 1.5]),
            'n': Var([2, 2]),
        }))
    assert_dataset_equal(
        ds.aggregate('', drop_bad=True),
        Dataset({
            'z': Factor('a'),
            'v': Var([1.5]),
            'n': Var([4]),
        }))

    # ds.update()
    ds = Dataset()
    ds.update({'f': Factor('abab')})
    assert ds['f'].name == 'f'

    # checks on assignemnt
    ds = Dataset()
    ds['a'] = Factor('abab')
    # key check
    with pytest.raises(ValueError):
        ds[:, '1'] = 'value'
    # value check
    with pytest.raises(ValueError):
        ds['b'] = Factor('abcde')  # length mismatch
    with pytest.raises(TypeError):
        ds['b'] = {i: i for i in range(4)}

    # from_caselist
    target = Dataset({
        'y': Var([1, 2]),
        'x': Factor('ab'),
        'z': Factor('uv', random=True),
    })
    names = ['y', 'x', 'z']
    cases = [[1, 'a', 'u'], [2, 'b', 'v']]
    ds = Dataset.from_caselist(names, cases, random='z')
    assert_dataobj_equal(ds, target)
    # Length 1
    ds0 = Dataset.from_caselist(names, cases[:1], random='z')
    assert_dataobj_equal(ds0, target[:1])


def test_dataset_combining():
    "Test Dataset combination methods"
    ds = datasets.get_uv()
    del ds['fltvar'], ds['intvar'], ds['A']

    ds2 = datasets.get_uv()
    del ds2['fltvar'], ds2['intvar']
    ds.update(ds2)
    assert_array_equal(ds['A'], ds2['A'])

    ds2 = datasets.get_uv()
    del ds2['fltvar'], ds2['intvar']
    ds2['B'][5] = 'something_else'
    del ds['A']
    with pytest.raises(ValueError):
        ds.update(ds2)


def test_dataset_indexing():
    """Test Dataset indexing"""
    ds = datasets.get_uv()
    ds.index('case')

    # indexing values
    assert ds['A', 1] == ds['A'][1]
    assert ds[1, 'A'] == ds['A'][1]

    # indexing variables
    assert_dataobj_equal(ds[:, 'A'], ds['A'])
    assert_dataobj_equal(ds['A', :], ds['A'])
    assert_dataobj_equal(ds[:10, 'A'], ds['A'][:10])
    assert_dataobj_equal(ds['A', :10], ds['A'][:10])
    assert_dataobj_equal(ds.sub("case < 10", 'A'), ds['A'][:10])

    # new Dataset through indexing
    ds2 = Dataset()
    ds2['A'] = ds['A']
    assert_dataset_equal(ds[('A',)], ds2)
    ds2['B'] = ds['B']
    assert_dataset_equal(ds['A', 'B'], ds2)
    assert_dataset_equal(ds[('A', 'B'), :10], ds2[:10])
    assert_dataset_equal(ds[:10, ('A', 'B')], ds2[:10])

    # empty index
    assert_dataobj_equal(ds2[[]], Dataset([Factor([], 'A'), Factor([], 'B')]))

    # assigning value
    ds[2, 'A'] = 'hello'
    assert ds[2, 'A'] == 'hello'
    ds['A', 2] = 'not_hello'
    assert ds[2, 'A'] == 'not_hello'

    # assigning new factor
    ds['C', :] = 'c'
    assert np.all(ds.eval("C == 'c'"))

    # assigning new Var
    ds['D1', :] = 5.
    ds[:, 'D2'] = 5.
    assert_array_equal(ds['D1'], 5)
    assert_array_equal(ds['D2'], 5)

    # test illegal names
    f = Factor('aaabbb')
    with pytest.raises(ValueError):
        ds['%dsa'] = f
    with pytest.raises(ValueError):
        ds['432'] = f
    with pytest.raises(ValueError):
        ds['%dsa', :] = 'value'
    with pytest.raises(ValueError):
        ds[:, '%dsa'] = 'value'
    with pytest.raises(ValueError):
        ds['432', :] = 4.
    with pytest.raises(ValueError):
        ds[:, '432'] = 4.

    # deleting items
    del ds['A']
    assert 'A' not in ds
    with pytest.raises(KeyError):
        _ = ds['A']
    del ds['B', 'rm']
    assert 'B' not in ds and 'rm' not in ds


def test_dataset_repr():
    "Test Dataset string representation methods"
    ds = datasets.get_uts()

    assert repr(ds) == "<Dataset (60 cases) 'A':F, 'B':F, 'rm':F, 'ind':F, 'Y':V, 'YBin':F, 'YCat':F, 'uts':Vnd>"
    assert_fmtxt_str_equals(ds[:2], """
    #   A    B    rm    ind   Y        YBin   YCat
    ----------------------------------------------
    0   a0   b0   R00   R00   2.0977   c1     c1
    1   a0   b0   R01   R01   1.8942   c1     c1
    ----------------------------------------------
    NDVars: uts
    """)
    assert_fmtxt_str_equals(ds.summary(50), """
    Key    Type     Values
    --------------------------------------------------
    A      Factor   a0:30, a1:30
    B      Factor   b0:30, b1:30
    rm     Factor   R00:4, R01:4... (15 cells, random)
    ind    Factor   R00, R01... (60 cells, random)
    Y      Var      -3.53027 - 3.04498
    YBin   Factor   c1:34, c2:26
    YCat   Factor   c1:17, c2:24, c3:19
    uts    NDVar    100 time; -2.67343 - 4.56283
    --------------------------------------------------
    Dataset: 60 cases
    """)
    assert_fmtxt_str_equals(ds[:5].summary(), """
    Key    Type     Values
    -----------------------------------------------------------
    A      Factor   a0:5
    B      Factor   b0:5
    rm     Factor   R00, R01, R02, R03, R04 (random)
    ind    Factor   R00, R01, R02, R03, R04 (random)
    Y      Var      0.77358, 1.01346, 1.89424, 2.09773, 2.55396
    YBin   Factor   c1:4, c2
    YCat   Factor   c1:2, c2:2, c3
    uts    NDVar    100 time; -0.634835 - 4.56283
    -----------------------------------------------------------
    Dataset: 5 cases
    """)
    # .head() and .tail() without NDVars
    del ds['uts']
    assert str(ds.head()) == str(ds[:10])
    assert str(ds.tail()) != str(ds[-10:])


def test_dataset_sorting():
    "Test Dataset sorting methods"
    test_array = np.arange(10)
    ds = Dataset()
    ds['v'] = Var(test_array)
    ds['f'] = Factor(test_array)

    # shuffle the Dataset
    rand_idx = test_array.copy()
    np.random.shuffle(rand_idx)
    ds_shuffled = ds[rand_idx]

    # ascending, Var, copy
    dsa = ds_shuffled.sorted('v')
    assert_dataset_equal(dsa, ds)

    # descending, Factor, in-place
    ds_shuffled.sort('f', descending=True)
    assert_dataset_equal(ds_shuffled, ds[::-1])


def test_dim_categorial():
    "Test Categorial Dimension"
    values = ['a', 'b', 'c', 'abc']
    name = 'cat'
    dim = Categorial(name, values)

    # basic properties
    print(dim)
    assert len(dim) == len(values)

    # persistence
    s = pickle.dumps(dim, pickle.HIGHEST_PROTOCOL)
    dim_ = pickle.loads(s)
    assert dim_ == dim

    # indexing
    sub_values = values[:2]
    idx = dim._array_index(sub_values)
    assert dim[idx] == Categorial(name, sub_values)
    assert dim._array_index('a') == values.index('a')
    assert dim._array_index('abc') == values.index('abc')
    with pytest.raises(TypeError):
        dim._array_index(('a', 'b', 'c'))

    # intersection
    dim2 = Categorial(name, ['c', 'b', 'e'])
    dim_i = dim.intersect(dim2)
    assert dim_i == Categorial(name, ['b', 'c'])

    # adjacency
    dim = Categorial(name, ['c', 'b', 'e'], [('b', 'c'), ('b', 'e')])
    assert_array_equal(dim.adjacency(), [[0, 1], [1, 2]])


def test_dim_scalar():
    "Test Scalar Dimension"
    d = Scalar('scalar', [20, 30, 40, 50, 60, 70])
    assert repr(d) == "Scalar('scalar', [20, ..., 70] (6))"

    assert d._array_index(20) == 0
    assert d._array_index(30) == 1
    assert d._array_index(21) == 0
    with pytest.raises(IndexError):
        d._array_index(25)

    # binning
    edges, dim = d._bin(step=20)
    assert edges == [20, 40, 60, 80]
    assert dim == Scalar('scalar', [30, 50, 70])
    edges, dim = d._bin(start=30, stop=70, step=20)
    assert edges == [30, 50, 70]
    assert dim == Scalar('scalar', [40, 60])
    # range not divisible by step
    with pytest.raises(ValueError):
        d._bin(start=30, step=20)
    with pytest.raises(ValueError):
        d._bin(stop=70, step=20)
    # nbins
    edges, dim = d._bin(nbins=3)
    assert edges == [20, 40, 60, None]
    assert dim == Scalar('scalar', [30, 50, 70])
    edges, dim = d._bin(nbins=2)
    assert edges == [20, 50, None]
    assert dim == Scalar('scalar', [35, 65])
    # uneven bin size
    with pytest.raises(ValueError):
        d._bin(nbins=4)
    # approximate start/stop
    edges, dim = d._bin(25, 65, nbins=2)
    assert edges == [30, 50, 70]
    edges, dim = d._bin(25, 65, 20)
    assert edges == [30, 50, 70]


def test_dim_uts():
    "Test UTS Dimension"
    uts = UTS(-0.1, 0.005, 301)

    # basic indexing
    assert uts._array_index(-0.1) == 0
    assert uts._array_index(0) == 20
    assert uts._array_index(1.4) == 300
    assert uts._array_index(1.405) == 301
    with pytest.raises(ValueError):
        uts._array_index(-.15)

    assert uts._array_index(slice(0, 1.4)) == slice(20, 300)
    assert uts._array_index(slice(0, 1.5)) == slice(20, 320)
    assert uts._array_index(slice(-1, 0)) == slice(None, 20)

    # make sure indexing rounds correctly for floats
    assert uts._array_index(-0.101) == 0
    assert uts._array_index(-0.099) == 0
    for i, s in enumerate(np.arange(0, 1.4, 0.05)):
        idx = uts._array_index((-0.1 + s, s))
        assert idx.start == 10 * i
        assert idx.stop == 20 + 10 * i

    # intersection
    uts1 = UTS(-0.1, 0.01, 50)
    uts2 = UTS(0, 0.01, 20)
    intersection = uts1.intersect(uts2)
    assert intersection == uts2
    idx = uts1._array_index((0, 0.2))
    assert uts1[idx] == uts2


def test_effect():
    "Test _Effect class"
    # .enumerate_cells()
    f1 = Factor('aabbccaabbcc')
    f2 = Factor('abababababab')
    i = f1 % f2

    n1 = np.concatenate((np.tile([0, 1], 3), np.tile([2, 3], 3)))
    assert_array_equal(f1.enumerate_cells(), n1)
    assert_array_equal(f2.enumerate_cells(), np.arange(6).repeat(2))
    assert_array_equal(i.enumerate_cells(), np.arange(2).repeat(6))


def test_equality():
    u = Var(np.arange(5.))
    v = Var(np.arange(5.))
    assert all_equal(u, v)
    u[-1] = np.nan
    assert not all_equal(u, v)
    v[-1] = np.nan
    assert not all_equal(u, v)
    assert all_equal(u, v, True)


def test_factor():
    "Test basic Factor functionality"
    # initializing
    assert_array_equal(Factor('ab'), ['a', 'b'])
    assert_array_equal(Factor('ab', repeat=2), ['a', 'a', 'b', 'b'])
    assert_array_equal(Factor('ab', repeat=[2, 1]), ['a', 'a', 'b'])
    empty_factor = Factor([])
    assert len(empty_factor) == 0
    assert_dataobj_equal(Factor(np.empty(0)), empty_factor)
    # from Factor
    f = Factor('aabbcc')
    assert_array_equal(Factor(f), f)
    assert_array_equal(Factor(f, labels={'a': 'b'}), Factor('bbbbcc'))
    f2 = Factor(f, labels={'a': '1'}, default='2')
    assert_array_equal(f2, Factor('112222'))

    # removing a cell
    f = Factor('aabbcc')
    assert f.cells == ('a', 'b', 'c')
    assert f.n_cells == 3
    f[f == 'c'] = 'a'
    assert f.cells == ('a', 'b')
    assert f.n_cells == 2

    # cell order
    assert Factor('ab').cells == ('a', 'b')
    # alphabetical if labels is unspecified
    assert Factor('ba').cells == ('b', 'a')
    # follow labels arg
    a = np.tile(np.arange(3), 3)
    f = Factor(a, labels={2: 'a', 1: 'b', 0: 'c'})
    assert f.cells == ('a', 'b', 'c')
    assert f[:2].cells == ('b', 'c')
    # not alphabetical
    f = Factor(a, labels={0: 'c', 1: 'b', 2: 'a'})
    assert f.cells == ('c', 'b', 'a')
    assert f[:2].cells == ('c', 'b')
    f[f == 'b'] = 'c'
    assert f.cells == ('c', 'a')
    # initialize from factor
    f = Factor(a, labels={0: 'c', 1: 'b', 2: 'a'})
    f2 = Factor(f, labels={'c': 'c', 'b': 'c', 'a': 'a'})
    assert f2.cells == ('c', 'a')
    # superfluous label
    f2 = Factor(f, labels={'c': 'a', 'x': 'c', 'b': 'b', 'a': 'c'})
    assert f2.cells == ('a', 'b', 'c')
    # sort
    f = Factor(a, labels={0: 'c', 1: 'b', 2: 'a'})
    f.sort_cells(('a', 'c', 'b'))
    assert f.cells == ('a', 'c', 'b')

    # label length
    lens = [2, 5, 32, 2, 32, 524]
    f = Factor(['a' * length for length in lens], 'f')
    fl = f.label_length()
    assert_array_equal(fl, lens)
    assert fl.info['longname'] == 'f.label_length()'
    lens2 = [3, 5, 32, 2, 32, 523]
    f2 = Factor(['b' * length for length in lens2], 'f2')
    assert_array_equal(fl - f2.label_length(), [a - b for a, b in zip(lens, lens2)])

    # equality
    f = Factor('aabbcc')
    assert_equal(f == Factor('aabbcc'), True)
    assert_equal(f == Factor('bbccaa'), False)
    assert_equal(f == Factor('aabxxx'), (True, True, True, False, False, False))
    assert_equal(f == Var(np.ones(6)), False)

    # unary operations
    with pytest.raises(TypeError):
        bool(f)

    # Factor.as_var()
    assert_array_equal(f.as_var(dict(zip('abc', range(3)))), [0, 0, 1, 1, 2, 2])
    assert_array_equal(f.as_var({'a': 1}, 2), [1, 1, 2, 2, 2, 2])
    with pytest.raises(KeyError):
        f.as_var({'a': 1})
    numeric = Factor(['1', '10', '6', '1'])
    assert_array_equal(numeric.as_var(), [1, 10, 6, 1])

    # .count()
    a = Factor('abcabcabccc')
    assert_array_equal(a.count().x, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4])
    assert_array_equal(a.count('a').x, [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

    # Factor.floodfill()
    f_ = Factor([' ', ' ', '1', '2', ' ', ' ', '3', ' ', ' ', '2', ' ', ' ', '1'])
    regions_0 = [ 1,   1,   1,   2,   2,   2,   3,   3,   3,   2,   2,   1,   1]
    regions_1 = [ 1,   1,   1,   2,   2,   3,   3,   2,   2,   2,   2,   1,   1]
    regions_2 = [ 1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2]
    regions_3 = ['1', '1', '1', '2', '2', '2', '3', '3', '2', '2', '2', '2', '1']
    target_p = [' ', ' ', '1', '2', '2', '2', '3', '3', '3', '2', '2', '2', '1']
    assert_array_equal(f_.floodfill(regions_0, ' '), Var(regions_0).as_factor())
    assert_array_equal(f_.floodfill(regions_1, ' '), Var(regions_1).as_factor())
    assert_array_equal(f_.floodfill(regions_2, ' '), regions_3)
    assert_array_equal(f_.floodfill('previous', ' '), target_p)
    f_ = Factor(['', '', 'a', '', 'e', 'r', ''])
    assert_array_equal(f_.floodfill([1, 1, 1, 11, 11, 11, 11]), Factor('aaaeerr'))

    # cell-based index
    f = Factor(['a1', 'a10', 'b1', 'b10'])
    assert_array_equal(f.startswith('a'), [1, 1, 0, 0])
    assert_array_equal(f.endswith('1'), [1, 0, 1, 0])
    assert_array_equal(f.matches('a?'), [1, 0, 0, 0])
    assert_array_equal(f.matches('b*'), [0, 0, 1, 1])


def test_factor_relabel():
    "Test Factor.relabel() method"
    f = Factor('aaabbbccc')
    f.update_labels({'a': 'd'})
    assert_array_equal(f, Factor('dddbbbccc'))
    f.update_labels({'d': 'c', 'c': 'd'})
    assert_array_equal(f, Factor('cccbbbddd'))
    f.update_labels({'d': 'c'})
    assert_array_equal(f, Factor('cccbbbccc'))
    # label not in f
    f.update_labels({'b': 'x', 'a': 'c'})
    assert_array_equal(f, Factor('cccxxxccc'))


def test_interaction():
    "Test Interaction"
    ds = datasets.get_uv()
    A = ds['A']  # noqa
    B = ds['B']  # noqa
    i = A % B
    # eq for sequence
    assert_array_equal(i == A % B, True)
    assert_array_equal(i == B % A, False)
    assert_array_equal(i == A, False)
    assert_array_equal(i == ds['fltvar'], False)
    assert_array_equal(ds.eval("A%B") == Factor(ds['A']) % B, True)
    # eq for element
    for a, b in product(A.cells, B.cells):
        assert_array_equal(i == (a, b), np.logical_and(A == a, B == b))

    # Interaction.as_factor()
    a = Factor('aabb')
    i = a % Factor('cdcd')
    assert_dataobj_equal(i.as_factor(), Factor(['a c', 'a d', 'b c', 'b d']))
    i = a % Factor(['c', '', 'c', ''])
    assert_dataobj_equal(i.as_factor(), Factor(['a c', 'a', 'b c', 'b']))

    # .count()
    a = Factor('abcabcccc')
    b = Factor('aaabbbccc')
    i = a % b
    assert_array_equal(i.count().x, [0, 0, 0, 0, 0, 0, 0, 1, 2])
    assert_array_equal(i.count(('c', 'c')).x, [-1, -1, -1, -1, -1, -1, 0, 1, 2])

    # pickling
    ip = pickle.loads(pickle.dumps(i))
    assert_dataobj_equal(ip, i)


def test_isin():
    "Test .isin() methods"
    values = np.array([  6, -6, 6, -2, -1, 0, -10, -5, -10, -6])
    v = values[0]
    v2 = values[:2]
    labels = {i: c for i, c in enumerate(ascii_lowercase, -10)}
    vl = labels[v]
    v2l = [labels[v_] for v_ in v2]

    target = np.logical_or(values == v2[0], values == v2[1])
    inv_target = np.invert(target)
    index_target = np.flatnonzero(values == v)
    empty = np.array([])

    var = Var(values)
    assert_array_equal(var.index(v), index_target)
    assert_array_equal(var.isin(v2), target)
    assert_array_equal(var.isany(*v2), target)
    assert_array_equal(var.isnot(*v2), inv_target)
    assert_array_equal(var.isnotin(v2), inv_target)

    var0 = Var([])
    assert_array_equal(var0.isin(v2), empty)
    assert_array_equal(var0.isany(*v2), empty)
    assert_array_equal(var0.isnot(*v2), empty)
    assert_array_equal(var0.isnotin(v2), empty)

    f = Factor(values, labels=labels)
    assert_array_equal(f.index(vl), index_target)
    assert_array_equal(f.isin(v2l), target)
    assert_array_equal(f.isany(*v2l), target)
    assert_array_equal(f.isnot(*v2l), inv_target)
    assert_array_equal(f.isnotin(v2l), inv_target)

    f0 = Factor([])
    assert_array_equal(f0.isin(v2l), empty)
    assert_array_equal(f0.isany(*v2l), empty)
    assert_array_equal(f0.isnot(*v2l), empty)
    assert_array_equal(f0.isnotin(v2l), empty)


def test_longname():
    "Test info['longname'] entry"
    ds = Dataset()
    u = Var([2], 'u')
    v = Var([1], 'v')

    # simple operations, also tested in test_var()
    assert longname(v.abs()) == 'abs(v)'
    assert longname(u * v) == "u * v"
    assert longname(u * v.abs()) == "u * abs(v)"

    # Dataset assigning
    ds['abs_v'] = v.abs()
    assert longname(ds['abs_v']) == 'abs_v'


def test_model():
    "Test Model class"
    a = Factor('ab', repeat=3, name='a')
    b = Factor('ab', tile=3, name='b')
    u = Var([1, 1, 1, -1, -1, -1], 'u')
    v = Var([1., 2., 3., 4., 5., 6.], 'v')
    w = Var([1., 0., 0., 1., 1., 0.], 'w')

    # model repr
    m = a * b + v
    assert repr(m) == "a + b + a % b + v"
    lines = ("intercept   a   b   a x b   v",
             "-----------------------------",
             "1           1   1   1       1",
             "1           1   0   0       2",
             "1           1   1   1       3",
             "1           0   0   0       4",
             "1           0   1   0       5",
             "1           0   0   0       6")
    assert str(m) == '\n'.join(lines)
    assert str(m.head(2)) == '\n'.join(lines[:4])
    assert str(m.tail(2)) == '\n'.join(lines[:2] + lines[-2:])
    str(m.info())

    # model without explicit names
    x1 = Factor('ab', repeat=2)
    x2 = Factor('ab', tile=2)
    m = x1 * x2
    assert repr(m) == "<?> + <?> + <?> % <?>"

    # catch explicit intercept
    intercept = Factor('i', repeat=4, name='intercept')
    with pytest.raises(ValueError):
        _ = a * intercept

    # different var/factor combinations
    assert a * b == a + b + a % b
    assert a * v == a + v + a % v
    assert a * (v + w) == a + v + w + a % v + a % w

    # parametrization
    m = v + w + v * w
    p = m._parametrize('dummy')
    assert p.column_names == ['intercept', 'v', 'w', 'v * w']
    assert_array_equal(p.x[:, p.terms['intercept']], 1)
    assert_array_equal(p.x[:, p.terms['v']], v.x[:, None])
    assert_array_equal(p.x[:, p.terms['w']], w.x[:, None])
    assert_array_equal(p.x[:, p.terms['v * w']], (v * w).x[:, None])

    # persistence
    mp = pickle.loads(pickle.dumps(m, pickle.HIGHEST_PROTOCOL))
    mpp = mp._parametrize('dummy')
    assert_array_equal(mpp.x, p.x)

    # nested Vars
    m = (v + w) * u
    assert_dataobj_equal(m.effects[2], u)
    assert_dataobj_equal(m.effects[3], v * u)
    assert_dataobj_equal(m.effects[4], w * u)
    m = u * (v + w)
    assert_dataobj_equal(m.effects[0], u)
    assert_dataobj_equal(m.effects[3], u * v)
    assert_dataobj_equal(m.effects[4], u * w)
    m = (v + w) % u
    assert_dataobj_equal(m.effects[0], v * u)
    assert_dataobj_equal(m.effects[1], w * u)
    m = u % (v + w)
    assert_dataobj_equal(m.effects[0], u * v)
    assert_dataobj_equal(m.effects[1], u * w)


def test_ndvar():
    "Test the NDVar class"
    ds = datasets.get_uts(utsnd=True)

    with pytest.raises(TypeError):
        bool(ds['utsnd'])

    # names
    x = ds[0, 'uts']
    assert repr(x) == "<NDVar 'uts': 100 time>"
    assert repr(x * 2) == "<NDVar 'uts': 100 time>"
    assert repr(x * 2 + 5) == "<NDVar 'uts': 100 time>"
    assert repr(-x * 2 + 5) == "<NDVar 'uts': 100 time>"
    assert repr(x > 0) == "<NDVar 'uts' bool: 100 time>"

    # meaningful slicing
    x = ds['utsnd']
    with pytest.raises(KeyError):
        x.sub(sensor='5')
    assert_equal(x.sub(sensor='4'), x.x[:, 4])
    assert_equal(x.sub(sensor=['4', '3', '2']), x.x[:, [4, 3, 2]])
    assert_equal(x.sub(sensor=['4']), x.x[:, [4]])
    assert_equal(x.sub(case=1, sensor='4'), x.x[1, 4])

    # setup indices
    s_case = slice(10, 13)
    s_sensor = slice('2', '4')
    s_time = slice(0.1, 0.2)
    b_case = np.bincount([10, 11, 12], minlength=len(x)).astype(bool)
    b_sensor = np.array([False, False, True, True, False])
    b_time = np.bincount(range(30, 40), minlength=len(x.time)).astype(bool)
    a_case = np.arange(10, 13)
    a_sensor = ['2', '3']
    a_time = np.arange(0.1, 0.2, 0.01)

    # slicing with different index kinds
    tgt = x.x[s_case, 2:4, 30:40]
    assert tgt.shape == (3, 2, 10)
    # single
    assert_equal(x.sub(case=s_case, sensor=s_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=a_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=b_sensor, time=b_time), tgt)
    # bool & slice
    assert_equal(x.sub(case=b_case, sensor=s_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=b_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=s_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=b_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=b_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=s_sensor, time=b_time), tgt)
    # bool & array
    assert_equal(x.sub(case=b_case, sensor=a_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=b_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=a_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=b_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=b_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=a_sensor, time=b_time), tgt)
    # slice & array
    assert_equal(x.sub(case=s_case, sensor=a_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=s_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=a_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=s_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=s_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=a_sensor, time=s_time), tgt)
    # all three
    assert_equal(x.sub(case=a_case, sensor=b_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=s_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=a_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=s_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=a_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=b_sensor, time=a_time), tgt)

    # norm
    y = x / x.norm('sensor')
    assert_allclose(y.norm('sensor'), 1.)
    y = ds['uts'].mean('case').norm('time')
    assert isinstance(y, float)

    # dot
    m = NDVar([1, 0, -1, 0, 0], x.sensor)
    # -> time
    y = m.dot(x[0])
    assert_array_equal(y.x, x.x[0, 0] - x.x[0, 2])
    # -> case x time
    y_all = m.dot(x)
    assert len(y_all) == len(x)
    assert_dataobj_equal(y_all[0], y)
    # -> scalar
    y = m.dot(x[0, :, 0.200])
    assert y == x.x[0, 0, 40] - x.x[0, 2, 40]
    # multiple dimensions
    m = NDVar([[1, 0, -1, 0, 0], [-1, 0, 1, 0, 0]], (Case, x.sensor))
    y_target = x.x[0, 0] - x.x[0, 2] - x.x[1, 0] + x.x[1, 2]
    y = m.dot(x[:2], ('case', 'sensor'))
    assert_allclose(y.x, y_target)
    y = m.dot(x[:2], ('sensor', 'case'))
    assert_allclose(y.x, y_target)

    # Var
    v_case = Var(b_case)
    assert_equal(x.sub(case=v_case, sensor=b_sensor, time=a_time), tgt)

    # univariate result
    assert_dataobj_equal(x.sub(sensor='2', time=0.1),
                         Var(x.x[:, 2, 30], x.name))
    assert x.sub(case=0, sensor='2', time=0.1) == x.x[0, 2, 30]

    # baseline correction
    x_bl = x - x.summary(time=(None, 0))
    # assert that the baseline is 0
    bl = x_bl.summary('case', 'sensor', time=(None, 0))
    assert abs(bl) < 1e-10

    # iteration
    for i, xi in enumerate(x):
        assert_dataobj_equal(xi, x[i])
        if i > 4:
            break


def test_ndvar_binning():
    "Test NDVar.bin()"
    x = np.arange(10)
    time = UTS(-0.1, 0.1, 10)
    x_dst = x.reshape((5, 2)).mean(1)
    time_dst = np.arange(0., 0.9, 0.2)

    # 1-d
    ndvar = NDVar(x, (time,))
    b = ndvar.bin(0.2)
    assert_array_equal(b.x, x_dst, "Binned data")
    assert_array_equal(b.time, time_dst, "Bin times")
    b = ndvar.sub(time=(0, 0.8)).bin(0.4)
    assert b.shape == (2,)

    # 2-d
    ndvar = NDVar(np.vstack((x, x, x)), ('case', time))
    b = ndvar.bin(0.2)
    assert_array_equal(b.x, np.vstack((x_dst, x_dst, x_dst)), "Binned data")
    assert_array_equal(b.time, time_dst, "Bin times")

    # masked NDVar
    x = np.vstack((x, x + 1, x + 2))
    ndvar = NDVar(x, ('case', time)).mask(x % 2)
    binned = ndvar.bin(0.2)
    assert_array_equal(binned.x.data, np.vstack([x[0, ::2], x[1, 1::2], x[2, ::2]]))
    binned_max = ndvar.bin(0.2, func='max')
    assert_array_equal(binned_max.x.data, np.vstack([x[0, ::2], x[1, 1::2], x[2, ::2]]))
    binned_extrema = ndvar.bin(0.2, func='extrema')
    assert_array_equal(binned_extrema.x.data, binned_max.x.data)

    # time:
    x = np.ones((5, 70))
    ndvar = NDVar(x, ('case', UTS(0.45000000000000007, 0.005, 70)))
    binned_ndvar = ndvar.bin(0.05)
    assert_array_equal(binned_ndvar.x, 1.)
    assert binned_ndvar.shape == (5, 7)

    # n_bins
    x = np.ones((2, 601))
    ndvar = NDVar(x, ('case', UTS(-0.1, 0.001, 601)))
    binned_ndvar = ndvar.bin(0.1, 0.1, 0.4)
    assert binned_ndvar.shape == (2, 3)

    # Small trailing bin
    time = UTS(0.0, 0.001, 239235)
    x = np.ones(time.nsamples)
    ndvar = NDVar(x, time)
    x_binned = ndvar.bin(0.015625)
    assert not np.isnan(x_binned.x).any()


def test_ndvar_adjacency():
    "Test NDVar dimensions with conectvity graph"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    # non-monotonic index
    sub_mono = x.sub(sensor=['2', '3', '4'])
    sub_nonmono = x.sub(sensor=['4', '3', '2'])
    argsort = np.array([2, 1, 0])
    conn = argsort[sub_mono.sensor.adjacency().ravel()].reshape((-1, 2))
    assert_equal(sub_nonmono.sensor.adjacency(), conn)

    # date for labeling
    x1 = ds.eval("utsnd[numpy.logical_and(A=='a0', B=='b0')].mean('case')")
    x2 = ds.eval("utsnd[A=='a1'].mean('case')")
    x = x1 + x2
    # insert point that is connected by sensors but not by grid
    x.x[0, 50:55] = 4

    # custom adjacency on first axis
    labels = x.label_clusters(3)
    assert len(labels.info['cids']) == 5
    assert_array_equal(np.unique(labels.x), np.append([0], labels.info['cids']))

    # custom adjacency second
    sensor, time = x.dims
    x = NDVar(x.x.T, (time, sensor))
    labels = x.label_clusters(3)
    assert len(labels.info['cids']) == 5

    # disconnected
    cat = Categorial('categorial', ('a', 'b', 'c', 'd', 'e'))
    x = NDVar(x.x, (time, cat))
    labels = x.label_clusters(3)
    assert len(labels.info['cids']) == 13

    # ordered
    scalar = Scalar('ordered', range(5))
    x = NDVar(x.x, (time, scalar))
    labels = x.label_clusters(3)
    assert len(labels.info['cids']) == 6


def ndvar_index(x, dimname, index, a_index, index_repr=True):
    "Helper function for test_ndvar_indexing"
    ax = x.get_axis(dimname)
    index_prefix = FULL_AXIS_SLICE * ax
    if dimname != 'case':
        dim = x.get_dim(dimname)
        assert_equal(dim._array_index(index), a_index)
        if index_repr is not False:
            if index_repr is True:
                index_repr = index
            assert dim._dim_index(a_index) == index_repr
    x_array = x.x[index_prefix + (a_index,)]
    x1 = x.sub(**{dimname: index})
    x2 = x[index_prefix + (index,)]
    assert_array_equal(x1.x, x_array)
    assert_dataobj_equal(x2, x1)


def test_ndvar_indexing():
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    # case
    ndvar_index(x, 'case', 1, 1)
    ndvar_index(x, 'case', [0, 3], [0, 3])
    ndvar_index(x, 'case', slice(0, 10, 2), slice(0, 10, 2))

    # Sensor
    ndvar_index(x, 'sensor', '0', 0)
    ndvar_index(x, 'sensor', ['0', '2'], [0, 2])
    ndvar_index(x, 'sensor', slice('0', '2'), slice(0, 2))
    ndvar_index(x, 'sensor', 0, 0, False)
    ndvar_index(x, 'sensor', [0, 2], [0, 2], False)
    ndvar_index(x, 'sensor', slice(0, 2), slice(0, 2), False)

    # UTS
    ndvar_index(x, 'time', 0, 20)
    ndvar_index(x, 'time', 0.1, 30)
    ndvar_index(x, 'time', 0.102, 30, False)
    ndvar_index(x, 'time', [0, 0.1, 0.2], [20, 30, 40])
    ndvar_index(x, 'time', slice(0.1, None), slice(30, None))
    ndvar_index(x, 'time', slice(0.2), slice(40))
    ndvar_index(x, 'time', slice(0.202), slice(41), False)
    ndvar_index(x, 'time', slice(0.1, 0.2), slice(30, 40))
    ndvar_index(x, 'time', slice(0.102, 0.2), slice(31, 40), False)
    ndvar_index(x, 'time', slice(0.1, None, 0.1), slice(30, None, 10))
    ndvar_index(x, 'time', slice(0.1, None, 1), slice(30, None, 100))

    # NDVar as index
    sens_mean = x.mean(('case', 'time'))
    idx = sens_mean > 0
    pos = sens_mean[idx]
    assert_array_equal(pos.x > 0, True)

    # NDVar as index along one dimension
    x_tc = x.sub(sensor='1')
    x_time = NDVar(x_tc.time.times >= 0.3, dims=(x_tc.time,))
    assert_dataobj_equal(x_tc[x_time], x_tc.sub(time=(0.3, None)))
    # NDVar whose dimension is smaller
    x_time_sub = x_time.sub(time=(0.2, None))
    assert_dataobj_equal(x_tc[x_time_sub], x_tc.sub(time=(0.3, None)))

    # out of range index
    assert_dataobj_equal(x.sub(time=(0.1, 0.81)), x.sub(time=(0.1, 0.91)))
    assert_dataobj_equal(x.sub(time=(-0.3, 0.1)), x.sub(time=(-0.2, 0.1)))

    # newaxis
    with pytest.raises(IndexError):
        _ = x[newaxis]
    x0 = x[0]
    assert not x0.has_case
    assert x0[newaxis].has_case

    # Scalar
    x = cwt_morlet(ds['uts'], [8, 10, 13, 17])
    with pytest.raises(IndexError):
        _ = x[:, 9]
    with pytest.raises(IndexError):
        _ = x[:, 6]
    ndvar_index(x, 'frequency', 10, 1)
    ndvar_index(x, 'frequency', 10.1, 1, False)
    ndvar_index(x, 'frequency', 9.9, 1, False)
    ndvar_index(x, 'frequency', [8.1, 10.1], [0, 1], False)
    ndvar_index(x, 'frequency', slice(8, 13), slice(0, 2))
    ndvar_index(x, 'frequency', slice(8, 13.1), slice(0, 3), False)
    ndvar_index(x, 'frequency', slice(8, 13.1, 2), slice(0, 3, 2), False)

    # Categorial
    x = NDVar(x.x, ('case', Categorial('cat', ['8', '10', '13', '17']), x.time))
    with pytest.raises(TypeError):
        _ = x[:, 9]
    with pytest.raises(IndexError):
        _ = x[:, '9']
    ndvar_index(x, 'cat', '13', 2)
    ndvar_index(x, 'cat', ['8', '13'], [0, 2])
    ndvar_index(x, 'cat', slice('8', '13'), slice(0, 2))
    ndvar_index(x, 'cat', slice('8', None, 2), slice(0, None, 2))

    # SourceSpace
    x = datasets.get_mne_stc(True, subject='fsaverage')
    with pytest.raises(TypeError):
        _ = x[:'insula-rh']
    with pytest.raises(TypeError):
        _ = x['insula-lh':'insula-rh']
    with pytest.raises(TypeError):
        _ = x['insula-lh', 'insula-rh']
    ndvar_index(x, 'source', 'L90', 90)
    ndvar_index(x, 'source', 'R90', 642 + 90)
    ndvar_index(x, 'source', ['L90', 'R90'], [90, 642 + 90])
    ndvar_index(x, 'source', slice('L90', 'R90'), slice(90, 642 + 90))
    ndvar_index(x, 'source', 90, 90, False)
    ndvar_index(x, 'source', [90, 95], [90, 95], False)
    ndvar_index(x, 'source', slice(90, 95), slice(90, 95), False)
    ndvar_index(x, 'source', 'insula-lh', x.source.parc == 'insula-lh', False)
    ndvar_index(x, 'source', ('insula-lh', 'insula-rh'),
                x.source.parc.isin(('insula-lh', 'insula-rh')), False)
    n_lh = x.source.parc.endswith('lh').sum()
    ndvar_index(x, 'source', 'lh', slice(n_lh), False)
    ndvar_index(x, 'source', 'rh', slice(n_lh, None), False)

    # index dim != dim
    source_rh = x.source[x.source.lh_n:]
    index = NDVar(np.arange(len(source_rh)) > 100, (source_rh,))
    assert_dataobj_equal(x.sub(source=index), x.sub(source='rh').sub(source=index))
    with pytest.raises(IndexError):
        x.sub(source='lh').sub(index)

    # multiple arguments
    y = ds['utsnd'].sub(sensor=[1, 2], time=[0, 0.1])
    assert y.shape == (60, 2, 2)
    assert_array_equal(y.x, ds['utsnd'].x[:, 1:3, [20, 30]])

    # argmax
    x.x[10, 10] = 20
    assert x.argmax() == ('L10', 0.1)
    assert x[('L10', 0.1)] == 20
    assert x.sub(source='L10').argmax() == 0.1
    assert x.sub(time=0.1).argmax() == 'L10'
    # across axis
    x9 = x[:9]
    x9_max = x9.argmax('time')
    assert_array_equal(x9_max, x9.x.argmax(1) * 0.01)
    assert x9_max.info['unit'] == 's'
    assert_array_equal(x9.argmin('time'), x9.x.argmin(1) * 0.01)
    assert x9[0].argmax('time') == 0.04
    assert x9[0].argmin('time') == 0.00

    # broadcasting
    u = ds[0, 'uts']
    dim = Categorial('test_dim', ['a', 'b'])
    v = NDVar([5, 1], dim)
    for desc, (op, iop) in FLOAT_OPERATORS.items():
        y = op(v, u)
        assert_array_equal(y['a'], op(5, u.x))
        assert_array_equal(y['b'], op(1, u.x))
    # with Case from Var
    case = Var([4, 1])
    for desc, (op, iop) in FLOAT_OPERATORS.items():
        y = op(case, u)
        assert_array_equal(y[0], op(4, u.x))
        assert_array_equal(y[1], op(1, u.x))
    # Case as non-first
    x = ds[:2, 'uts']
    r = u * x
    assert_array_equal(r.x, u.x[None] * x.x)
    r2 = x * u
    assert_array_equal(r2.x, r.x)

    # assign NDVar
    x = ds['uts'].copy()
    # slice
    x[:3, :.0] = 0
    assert_array_equal(x.x[:3, :20], 0.)
    assert_array_equal(x.x[3:, 20:], ds['uts'].x[3:, 20:])
    # list
    x[[4, 5]] = 1
    assert_array_equal(x.x[4:6], 1)
    # array
    x[np.array([10, 11, 12])] = 3
    assert_array_equal(x.x[10:13], 3)
    # list, slice
    x[[4, 5], :.0] = 2
    assert_array_equal(x.x[4:6, :20], 2)
    assert_array_equal(x.x[4:6, 20:], 1)
    # array, slice
    x[np.array([10, 11, 12]), .0:] = 6
    assert_array_equal(x.x[10:13, 20:], 6)
    assert_array_equal(x.x[10:13, :20], 3)
    # set with index NDVar
    x = ds['uts'].copy()
    index = x.mean('case') < 0
    x[index] = -1
    assert x.sum(index).sum() == -index.sum()
    i_index = ~index
    assert x.sum(i_index).sum() == ds['uts'].sum(i_index).sum()
    with pytest.raises(DimensionMismatchError):
        index[x != 0] = 0.

    # assign NDVar from NDVar
    x = ds['utsnd'].copy()
    # int
    x[0] = x[1]
    assert_array_equal(x[0].x, x[1].x)
    x3 = NDVar(x[3].x.swapaxes(0, 1), x.dims[:0:-1])
    x[2] = x3
    assert_array_equal(x[2].x, x[3].x)
    # full slice
    x[:, '1'] = x[0, '2']
    assert_array_equal(x.x[30, 1], x.x[0, 2])
    with pytest.raises(ValueError):
        x[:, '1'] = x[6]
    # slice
    x_sub = NDVar(np.ones(10), UTS(0, 0.01, 10))
    x[0, '0'][0: 0.1] = x_sub
    assert_array_equal(x.x[0, 0, 20:30], 1)
    x[1, '0', 0: 0.1] = -x_sub
    assert_array_equal(x.x[1, 0, 20:30], -1)
    # list
    x = ds['utsnd'].copy()
    x[[1, 2, 3]] = x[9]
    assert_array_equal(x.x[1], x.x[9])
    assert_array_equal(x.x[2], x.x[9])
    x[np.array([5, 6])] = x[9]
    assert_array_equal(x.x[5], x.x[9])
    assert_array_equal(x.x[6], x.x[9])


def test_ndvar_summary_methods():
    "Test NDVar methods for summarizing data over axes"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    x.info['test_item'] = 1
    dim = 'sensor'
    axis = x.get_axis(dim)
    dims = ('case', 'sensor')
    axes = tuple(x.get_axis(d) for d in dims)
    idx = x > 0
    x0 = x[0]
    idx0 = idx[0]
    idx1d = idx[0, :, 0]
    xsub = x.sub(time=(0, 0.5))
    idxsub = xsub > 0
    idxsub1d = idxsub[0, 0]

    # info inheritance
    assert x.mean(('sensor', 'time')).info == x.info
    # info update for booleans
    assert x.any(('sensor', 'time')).info == {'test_item': 1}

    # numpy functions
    assert x.any() == x.x.any()
    assert_array_equal(x.any(dim), x.x.any(axis))
    assert_array_equal(x.any(dims), x.x.any(axes))
    assert_array_equal(x.any(idx0), [x_[idx0.x].any() for x_ in x.x])
    assert_array_equal(x.any(idx), [x_[i].any() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.any(idx0), x0.x[idx0.x].any())
    assert_array_equal(x.any(idxsub), xsub.any(idxsub))
    assert_array_equal(x.any(idxsub1d), xsub.any(idxsub1d))
    assert_array_equal(x.any(idx1d), x.x[:, idx1d.x].any(1))

    assert x.max() == x.x.max()
    assert_array_equal(x.max(dim), x.x.max(axis))
    assert_array_equal(x.max(dims), x.x.max(axes))
    assert_array_equal(x.max(idx0), [x_[idx0.x].max() for x_ in x.x])
    assert_array_equal(x.max(idx), x.x[idx.x].max())
    assert_array_equal(x0.max(idx0), x0.x[idx0.x].max())
    assert_array_equal(x.max(idxsub), xsub.max(idxsub))
    assert_array_equal(x.max(idxsub1d), xsub.max(idxsub1d))
    assert_array_equal(x.max(idx1d), x.x[:, idx1d.x].max(1))

    assert x.mean() == x.x.mean()
    assert_array_equal(x.mean(dim), x.x.mean(axis))
    assert_array_equal(x.mean(dims), x.x.mean(axes))
    assert_array_almost_equal(x.mean(idx0), [x_[idx0.x].mean() for x_ in x.x])
    assert_array_equal(x.mean(idx), x.x[idx.x].mean())
    assert_array_equal(x0.mean(idx0), x0.x[idx0.x].mean())
    assert_array_equal(x.mean(idxsub), xsub.mean(idxsub))
    assert_array_equal(x.mean(idxsub1d), xsub.mean(idxsub1d))
    assert_array_equal(x.mean(idx1d), x.x[:, idx1d.x].mean(1))

    assert x.min() == x.x.min()
    assert_array_equal(x.min(dim), x.x.min(axis))
    assert_array_equal(x.min(dims), x.x.min(axes))
    assert_array_equal(x.min(idx0), [x_[idx0.x].min() for x_ in x.x])
    assert_array_equal(x.min(idx), x.x[idx.x].min())
    assert_array_equal(x0.min(idx0), x0.x[idx0.x].min())
    assert_array_equal(x.min(idxsub), xsub.min(idxsub))
    assert_array_equal(x.min(idxsub1d), xsub.min(idxsub1d))
    assert_array_equal(x.min(idx1d), x.x[:, idx1d.x].min(1))

    assert x.var() == x.x.var()
    assert x.var(ddof=1) == x.x.var(ddof=1)
    assert_array_equal(x.var(dim), x.x.var(axis))
    assert_array_equal(x.var(dims, ddof=1), x.x.var(axes, ddof=1))
    assert_array_almost_equal(x.var(idx0), [x_[idx0.x].var() for x_ in x.x])
    assert_array_equal(x.var(idx), x.x[idx.x].var())
    assert_array_equal(x0.var(idx0), x0.x[idx0.x].var())
    assert_array_equal(x.var(idxsub), xsub.var(idxsub))
    assert_array_equal(x.var(idxsub1d), xsub.var(idxsub1d))
    assert_array_equal(x.var(idx1d), x.x[:, idx1d.x].var(1))

    assert x.std() == x.x.std()
    assert_array_equal(x.std(dim), x.x.std(axis))
    assert_array_equal(x.std(dims), x.x.std(axes))
    assert_array_almost_equal(x.std(idx0), [x_[idx0.x].std() for x_ in x.x])
    assert_array_equal(x.std(idx), x.x[idx.x].std())
    assert_array_equal(x0.std(idx0), x0.x[idx0.x].std())
    assert_array_equal(x.std(idxsub), xsub.std(idxsub))
    assert_array_equal(x.std(idxsub1d), xsub.std(idxsub1d))
    assert_array_equal(x.std(idx1d), x.x[:, idx1d.x].std(1))

    # non-numpy
    assert x.rms() == rms(x.x)
    assert_array_equal(x.rms(dim), rms(x.x, axis))
    assert_array_equal(x.rms(dims), rms(x.x, axes))
    assert_array_almost_equal(x.rms(idx0), [rms(x_[idx0.x]) for x_ in x.x])
    assert_array_equal(x.rms(idx), rms(x.x[idx.x]))
    assert_array_equal(x0.rms(idx0), rms(x0.x[idx0.x]))
    assert_array_equal(x.rms(idxsub), xsub.rms(idxsub))
    assert_array_equal(x.rms(idxsub1d), xsub.rms(idxsub1d))
    assert_array_equal(x.rms(idx1d), rms(x.x[:, idx1d.x], 1))

    assert x.extrema() == max(abs(x.min()), abs(x.max()))


def test_ndvar_timeseries_methods():
    "Test NDVar time-series methods"
    ds = datasets.get_uts(True)
    x = ds['utsnd']
    case, sensor, time = x.dims
    xs = NDVar(x.x.swapaxes(1, 2), (case, time, sensor), x.name, x.info.copy())

    # envelope
    env = x.envelope()
    assert_array_equal(env.x >= 0, True)
    envs = xs.envelope()
    assert_array_equal(env.x, envs.x.swapaxes(1, 2))

    # indexing
    assert len(ds[0, 'uts'][0.01:0.1].time) == 9

    # FFT
    x = ds['uts'].mean('case')
    np.sin(2 * np.pi * x.time.times, x.x)
    f = x.fft()
    assert_array_almost_equal(f.x, (f.frequency.values == 1) * (len(f) - 1))
    np.sin(4 * np.pi * x.time.times, x.x)
    f = x.fft()
    assert_array_almost_equal(f.x, (f.frequency.values == 2) * (len(f) - 1))

    # update tmin
    assert x.time.times[0] == -0.2
    x = set_tmin(x, 3.2)
    assert x.time.times[0] == 3.2


def test_nested_effects():
    """Test nested effects"""
    ds = datasets.get_uv(nrm=True)

    nested = ds.eval("nrm(B)")
    assert nested.cells == ds['nrm'].cells

    # interaction
    i = ds.eval("A % nrm(B)")
    assert i.cells == tuple(product(*(ds[f].cells for f in ['A', 'nrm'])))
    i = ds.eval("nrm(B) % A")
    assert i.cells == tuple(product(*(ds[f].cells for f in ['nrm', 'A'])))

    assert_has_no_empty_cells(ds.eval('A * B + nrm(B) + A % nrm(B)'))


@skip_on_windows  # uses R
def test_ols():
    "Test NDVar.ols() method"
    from rpy2.robjects import r

    # data-type
    assert_array_equal(NDVar([1, 2, 3], Case).ols(Var([1, 2, 3])).x, [1.])

    # simulate data
    ds = datasets.get_uts(True)
    n_times = len(ds['uts'].time)
    x = np.zeros(n_times)
    x[20:40] = np.hanning(20)
    utsc = ds.eval("uts.copy()")
    utsc.x += ds['Y'].x[:, None] * x[None, :]
    ds_ = Dataset()
    ds_['x'] = Var(ds['Y'].x)
    ds_['x2'] = ds_['x'] + np.random.normal(0, 1, ds.n_cases)

    # ols regression
    m1 = ds_['x']
    b1 = utsc.ols(m1)
    res1 = utsc.residuals(m1)
    t1 = utsc.ols_t(m1)
    m2 = ds_.eval("x + x2")
    b2 = utsc.ols(m2)
    res2 = utsc.residuals(m2)
    t2 = utsc.ols_t(m2)
    # compare with R
    for i in range(n_times):
        ds_['y'] = Var(utsc.x[:, i])
        ds_.to_r('ds')
        # 1 predictor
        r('lm1 <- lm(y ~ x, ds)')
        beta = r('coef(lm1)')[1]
        assert b1.x[0, i] == pytest.approx(beta)
        res = r('residuals(lm1)')
        assert_array_almost_equal(res1.x[:, i], res)
        t = r('coef(summary(lm1))')[5]
        assert t1.x[0, i] == pytest.approx(t)
        # 2 predictors
        r('lm2 <- lm(y ~ x + x2, ds)')
        beta = r('coef(lm2)')[1:]
        assert_array_almost_equal(b2.x[:, i], beta)
        res = r('residuals(lm2)')
        assert_array_almost_equal(res2.x[:, i], res)
        lm2_coefs = r('coef(summary(lm2))')
        t = [lm2_coefs[7], lm2_coefs[8]]
        assert_array_almost_equal(t2.x[:, i], t)

    # 3d
    utsnd = ds['utsnd']
    ds_['utsnd'] = utsnd
    b1 = ds_.eval("utsnd.ols(x)")
    res1 = ds_.eval("utsnd.residuals(x)")
    t1 = ds_.eval("utsnd.ols_t(x)")
    for i in range(len(b1.time)):
        ds_['y'] = Var(utsnd.x[:, 1, i])
        ds_.to_r('ds')
        # 1 predictor
        r('lm1 <- lm(y ~ x, ds)')
        beta = r('coef(lm1)')[1]
        assert b1.x[0, 1, i] == pytest.approx(beta)
        res = r('residuals(lm1)')
        assert_array_almost_equal(res1.x[:, 1, i], res)
        t = r('coef(summary(lm1))')[5]
        assert t1.x[0, 1, i] == pytest.approx(t)


def test_parametrization():
    ds = Dataset({
        'a': Factor('aabb'),
        'b': Factor('abab'),
        'u': Var([1, 2, 3, 4]),
        'v': Var([0, 0, 1, 1]),
    })
    # categorial
    model = ds.eval("a*b")
    p = model._parametrize()
    assert p.effect_names == ['intercept', 'a', 'b', 'a x b']
    # numeric
    model = ds.eval("u + v + u*v")
    p = model._parametrize()
    assert p.effect_names == ['intercept', 'u', 'v', 'u * v']


def test_io_pickle():
    "Test io by pickling"
    ds = datasets.get_uts()
    ds.info['info'] = "Some very useful information about the Dataset"
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.pickle')
        with open(dest, 'wb') as fid:
            pickle.dump(ds, fid, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dest, 'rb') as fid:
            ds2 = pickle.load(fid)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds2)


def test_io_txt():
    "Test Dataset io as text"
    ds = datasets.get_uv()

    # Var that has integer values as float
    ds['intflt'] = ds.eval('intvar * 1.')
    ds['intflt'].name = 'intflt'

    # io test
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.txt')
        ds.save_txt(dest)
        ds2 = load.tsv(dest, random='rm')
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds2, decimal=6)


@skip_on_windows  # uses R
def test_r():
    "Test interaction with R through rpy2"
    from rpy2.robjects import r

    r("data(sleep)")
    ds = Dataset.from_r("sleep")
    assert ds.name == 'sleep'
    extra = (0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0, 1.9, 0.8,
             1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)
    assert_array_equal(ds.eval('extra'), extra)
    assert_array_equal(ds.eval('ID'), list(map(str, range(1, 11))) * 2)
    assert_array_equal(ds.eval('group'), ['1'] * 10 + ['2'] * 10)

    # test putting
    ds.to_r('sleep_copy')
    ds_copy = Dataset.from_r('sleep_copy')
    assert_dataset_equal(ds_copy, ds)


def test_sensor():
    "Test Sensor dimension"
    locs = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
    names = ['1', '2', '3']
    sensor = Sensor(locs, names, 'test')
    s1 = sensor[[0, 1]]
    s2 = sensor[[1, 2]]
    assert tuple(s1.names) == ('1', '2')
    assert tuple(s2.names) == ('2', '3')
    assert s1 == sensor[[0, 1]]
    assert s1 != s2
    assert s1.intersect(s2) == sensor[[1]]
    assert sensor._dim_index(np.array([0, 1, 1], bool)) == ['2', '3']
    # from MNE montage
    m = mne.channels.make_standard_montage('standard_1020')
    s = Sensor.from_montage(m)
    assert s.names[0] == 'Fp1'
    # equality
    s_copy = Sensor.from_montage(m)
    assert s_copy == s
    assert s_copy[:25] != s


def test_shuffle():
    x = Factor('aabbaa')
    for _ in range(3):
        i = shuffled_index(6, x)
        assert sorted(i[2:4]) == [2, 3]
        assert sorted(i) == list(range(6))


@requires_mne_sample_data
def test_source_space():
    "Test SourceSpace Dimension"
    subject = 'fsaverage'
    data_path = mne.datasets.sample.data_path()
    mri_sdir = os.path.join(data_path, 'subjects')
    mri_dir = os.path.join(mri_sdir, subject)
    label_dir = os.path.join(mri_dir, 'label')
    label_ba1 = mne.read_label(os.path.join(label_dir, 'lh.BA1.label'))
    label_v1 = mne.read_label(os.path.join(label_dir, 'lh.V1.label'))
    label_mt = mne.read_label(os.path.join(label_dir, 'lh.MT.label'))
    label_ba1_v1 = label_ba1 + label_v1
    label_v1_mt = label_v1 + label_mt

    src = datasets._mne_source_space(subject, 'ico-5', mri_sdir)
    source = SourceSpace.from_mne_source_spaces(src, 'ico-5', mri_sdir)
    source_v1 = source[source._array_index(label_v1)]
    assert source_v1 == SourceSpace.from_mne_source_spaces(src, 'ico-5', mri_sdir, label=label_v1)
    source_ba1_v1 = source[source._array_index(label_ba1_v1)]
    source_v1_mt = source[source._array_index(label_v1_mt)]
    source_v1_intersection = source_ba1_v1.intersect(source_v1_mt)
    assert_source_space_equal(source_v1, source_v1_intersection)

    # persistence
    for ss in [source, source_v1]:
        ss_pickled = pickle.loads(pickle.dumps(ss, pickle.HIGHEST_PROTOCOL))
        assert ss_pickled == ss
        # secondary attributes
        for attr in ['kind', 'grade']:
            assert getattr(ss_pickled, attr) == getattr(ss, attr)

    # index from label
    index = source.index_for_label(label_v1)
    assert_array_equal(index.source[index.x].vertices[0],
                       np.intersect1d(source.lh_vertices, label_v1.vertices, 1))

    # parcellation and cluster localization
    parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=mri_sdir)
    indexes = [source.index_for_label(label) for label in parc
               if len(label) > 10]
    x = np.vstack([index.x for index in indexes])
    ds = source._cluster_properties(x)
    for i in range(ds.n_cases):
        assert ds[i, 'location'] == parc[i].name

    # multiple labels
    lingual_index = source._array_index('lingual-lh')
    cuneus_index = source._array_index('cuneus-lh')
    assert_array_equal(source._array_index(('cuneus-lh', 'lingual-lh')),
                       np.logical_or(cuneus_index, lingual_index))
    lingual_source = source[lingual_index]
    cuneus_source = source[cuneus_index]
    with pytest.raises(IndexError):
        _ = lingual_source._array_index(cuneus_source)
    sub_source = source[source._array_index(('cuneus-lh', 'lingual-lh'))]
    assert sub_source[sub_source._array_index('lingual-lh')] == lingual_source
    assert sub_source[sub_source._array_index('cuneus-lh')] == cuneus_source
    assert len(sub_source) == len(lingual_source) + len(cuneus_source)

    # indexing
    tgt = ['L%i' % i for i in chain(*sub_source.vertices)]
    assert_array_equal([i for i in sub_source], tgt)
    assert_array_equal([sub_source[i] for i in range(len(sub_source))], tgt)
    # hemisphere indexing
    lh = source._array_index('lh')
    source_lh = source[lh]
    assert source_lh._array_index('rh') == slice(0, 0)
    assert source_lh._array_index('lh') == slice(len(source_lh))


def test_var():
    "Test Var objects"
    base = Factor('aabbcde')

    # initialization
    x = np.arange(4)
    y = Var(x)
    assert_array_equal(y, x)
    y = Var(x, repeat=2)
    assert_array_equal(y, x.repeat(2))
    y = Var(x, repeat=x)
    assert_array_equal(y, x.repeat(x))
    y = Var.from_dict(base, {'a': 5, 'e': 8}, default=0)
    assert_array_equal(y.x, [5, 5, 0, 0, 0, 0, 8])
    with pytest.raises(RuntimeError):
        Var(x, info=1)
    # invalid dtypes
    with pytest.raises(TypeError):
        Var(np.array(['a', 'b', 'c']))
    with pytest.raises(TypeError):
        Var(np.array([None, 1, 2]))

    # basic operations
    info = {'a': 1}
    v = Var([1., 2., 3., -4.], 'v', info)
    assert_dataobj_equal(-v, Var([-1, -2, -3, 4], 'v', info))
    with pytest.raises(TypeError):
        bool(v)
    # binary operations
    c = 2
    v2 = Var([2., 2., 3., 3.], 'w', info)
    vs = [v, v2]
    vsi = [v.astype(int), v2.astype(int)]
    assert v.info == info
    for desc, (op, iop) in OPERATORS.items():
        if desc in FLOAT_OPERATORS:
            v, v2 = vs
        else:
            v, v2 = vsi
        target = op(v.x, c)
        vtarget = op(v.x, v2.x)
        # op
        if desc == '+':  # reserved for model building
            w = v.copy()
            w.x = iop(w.x, c)
        else:
            w = op(v, c)
            assert w.name == 'v'
            assert w.info == {**info, 'longname': f'v {desc} {c:g}'}
            assert_array_equal(w, target)
            # with Var
            w = op(v, v2)
            assert w.name == 'v'
            assert w.info == {**info, 'longname': f'v {desc} w'}
            assert_array_equal(w, vtarget)
        # i-op
        w = v.copy()
        w = iop(w, c)
        assert_array_equal(w, target)
        # i-op with Var
        w = v.copy()
        w = iop(w, v2)
        assert_array_equal(w, vtarget)
    v, v2 = vs

    # methods
    w = abs(v)
    assert_dataobj_equal(w, Var(np.abs(w.x), 'v', {**info, 'longname': 'abs(v)'}))
    # log
    assert_dataobj_equal(w.log(), Var(np.log(w.x), 'v', {**info, 'longname': 'log(abs(v))'}))
    assert_dataobj_equal(w.log(10), Var(np.log10(w.x), 'v', {**info, 'longname': 'log10(abs(v))'}))
    assert_dataobj_equal(w.log(42), Var(np.log(w.x) / log(42), 'v', {**info, 'longname': 'log42(abs(v))'}))

    # assignment
    tgt1 = np.arange(10)
    tgt2 = np.tile(np.arange(5), 2)
    v = Var(np.arange(10))
    v[v > 4] = np.arange(5)
    assert_array_equal(v, tgt2)
    v[5:] = np.arange(5, 10)
    assert_array_equal(v, tgt1)
    v = Var(np.arange(10))
    v[v > 4] = Var(np.arange(5))
    assert_array_equal(v, tgt2)
    v[5:] = Var(np.arange(5, 10))
    assert_array_equal(v, tgt1)

    # .count()
    v = Var([1., 2., 1.11, 2., 1.11, 4.])
    assert_array_equal(v.count().x, [0, 0, 0, 1, 1, 0])
    assert_array_equal(v.count(2).x, [-1, 0, 0, 1, 1, 1])

    # .split()
    y = Var(np.arange(16))
    for i in range(1, 9):
        split = y.split(i)
        assert len(split.cells) == i

    # .as_factor()
    v = Var(np.arange(4))
    assert_dataobj_equal(v.as_factor(), Factor('0123'))
    assert_dataobj_equal(v.as_factor({0: 'a'}), Factor(['a', '', '', '']))
    assert_dataobj_equal(v.as_factor({(0, 1): 'a', (2, 3): 'b'}), Factor('aabb'))
    assert_dataobj_equal(v.as_factor({(0, 1): 'a', 2: 'b', 'default': 'c'}),
                         Factor('aabc'))
    assert_dataobj_equal(v.as_factor({(0, 1): 'a', (2, 'default'): 'b'}),
                         Factor('aabb'))
