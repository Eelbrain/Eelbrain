# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from eelbrain.testing import assert_dataobj_equal
from eelbrain import Categorial, Dataset, Factor, NDVar, Scalar, UTS, Var, datasets, table, combine


def test_cast_to_ndvar():
    "Test table.cast_to_ndvar()"
    long_ds = datasets.get_uv()
    long_ds['scalar'] = long_ds['A'] == 'a2'
    long_ds['time'] = long_ds.eval('A%B').as_var({
        ('a1', 'b1'): 0.,
        ('a1', 'b2'): 0.1,
        ('a2', 'b1'): 0.2,
        ('a2', 'b2'): 0.3,
    })

    # categorial
    ds = table.cast_to_ndvar('fltvar', 'A', 'B%rm', data=long_ds, name='new')
    assert ds.n_cases == long_ds.n_cases / 2
    assert ds['new'].A == Categorial('A', ('a1', 'a2'))
    a1_index = long_ds['A'] == 'a1'
    a2_index = long_ds['A'] == 'a2'
    assert_array_equal(ds['new'].x[:, 0], long_ds[a1_index, 'fltvar'].x)
    assert_array_equal(ds['new'].x[:, 1], long_ds[a2_index, 'fltvar'].x)

    # scalar
    ds2 = table.cast_to_ndvar('fltvar', 'scalar', 'B%rm', data=long_ds, dim='newdim', name='new')
    assert ds2.n_cases == long_ds.n_cases / 2
    assert ds2['new'].newdim == Scalar('newdim', [False, True])
    assert_array_equal(ds['new'].x, ds2['new'].x)

    # time
    ds = table.cast_to_ndvar('fltvar', 'time', 'rm', data=long_ds, dim='uts', name='y')
    assert ds.n_cases == long_ds.n_cases / 4
    assert ds['y'].time == UTS(0, 0.1, 4)


def test_difference():
    "Test table.difference"
    ds = datasets.get_uv()
    # add a variables that should stay in the dataset
    labels = {c: c[-1] for c in ds['rm'].cells}
    ds['rmf'] = Factor(ds['rm'], labels=labels)

    dds = table.difference('fltvar', 'A', 'a1', 'a2', 'rm', data=ds)
    assert repr(dds) == "<Dataset (20 cases) 'rm':F, 'fltvar':V, 'rmf':F>"
    assert_array_equal(dds['rmf'], Factor(dds['rm'], labels=labels))
    dds = table.difference('fltvar', 'A', 'a1', 'a2', 'rm % B', data=ds)
    assert repr(dds) == "<Dataset (40 cases) 'rm':F, 'B':F, 'fltvar':V, 'rmf':F>"
    # difference of the difference
    ddds = table.difference('fltvar', 'B', 'b1', 'b2', 'rm', data=dds)
    assert repr(ddds) == "<Dataset (20 cases) 'rm':F, 'fltvar':V, 'rmf':F>"
    dds = table.difference('fltvar', 'A%B', ('a1', 'b1'), ('a2', 'b2'), 'rm', data=ds)
    assert repr(dds) == "<Dataset (20 cases) 'rm':F, 'fltvar':V, 'rmf':F>"

    # sub
    dds = table.difference('fltvar', 'A', 'a1', 'a2', 'rm', sub="B == 'b1'", data=ds)
    assert repr(dds) == "<Dataset (20 cases) 'rm':F, 'fltvar':V, 'B':F, 'rmf':F>"

    # create bigger dataset
    ds2 = ds.copy()
    ds['C', :] = 'c1'
    ds2['C', :] = 'c2'
    ds = combine((ds, ds2))
    dds = table.difference('fltvar', 'A', 'a1', 'a2', 'rm % B % C', data=ds)
    assert repr(dds) == "<Dataset (80 cases) 'rm':F, 'B':F, 'C':F, 'fltvar':V, 'rmf':F>"
    dds = table.difference('fltvar', 'A%B', ('a1', 'b1'), ('a2', 'b2'), 'rm % C', data=ds)
    assert repr(dds) == "<Dataset (40 cases) 'rm':F, 'C':F, 'fltvar':V, 'rmf':F>"


def test_frequencies():
    "Test table.frequencies"
    ds = datasets.get_uts()
    freq = table.frequencies('YCat', 'A', data=ds)
    assert str(freq) == ('#   A    c1   c2   c3\n'
                         '---------------------\n'
                         '0   a0   10   10   10\n'
                         '1   a1   7    14   9 ')
    freq = table.frequencies('YCat', 'A % B', data=ds)
    assert str(freq) == ('#   A    B    c1   c2   c3\n'
                         '--------------------------\n'
                         '0   a0   b0   5    6    4 \n'
                         '1   a0   b1   5    4    6 \n'
                         '2   a1   b0   5    6    4 \n'
                         '3   a1   b1   2    8    5 ')
    freq = table.frequencies('YCat % A', 'B', data=ds)
    assert str(freq) == ('#   B    c1_a0   c1_a1   c2_a0   c2_a1   c3_a0   c3_a1\n'
                         '------------------------------------------------------\n'
                         '0   b0   5       5       6       6       4       4    \n'
                         '1   b1   5       2       4       8       6       5    ')
    ds = Dataset()
    ds['n'] = Factor('xxxyyy')
    freq = table.frequencies('n', data=ds)
    assert str(freq) == ('#   n   n_\n'
                         '----------\n'
                         '0   x   3 \n'
                         '1   y   3 ')


def test_melt_ndvar():
    "Test table.melt_ndvar()"
    ds = datasets.get_uts(True)
    ds = ds.sub("A == 'a1'")

    lds = table.melt_ndvar('uts', ds=ds)
    assert 'time' in lds
    assert isinstance(lds['time'], Var)
    assert_array_equal(np.unique(lds['time'].x), ds['uts'].time)

    # no ds
    lds2 = table.melt_ndvar(ds['uts'])
    assert_dataobj_equal(lds2['uts'], lds['uts'])
    # no ds no case
    lds3 = table.melt_ndvar(ds[0, 'utsnd'], 'sensor')
    assert_dataobj_equal(lds3[0, 'utsnd'], ds[0, 'utsnd'][0])

    # sensor
    lds = table.melt_ndvar("utsnd.summary(time=(0.1, 0.2))", ds=ds, varname='summary')
    assert set(lds['sensor'].cells) == set(ds['utsnd'].sensor.names)

    # NDVar out
    lds = table.melt_ndvar("utsnd", 'sensor', ds=ds)
    assert 'utsnd' in lds
    assert isinstance(lds['utsnd'], NDVar)
    assert_dataobj_equal(lds[:ds.n_cases, 'utsnd'], ds.eval("utsnd.sub(sensor='0')"))

    # more than one dimensions
    with pytest.raises(ValueError):
        table.melt_ndvar('utsnd', ds=ds)


def test_repmeas():
    "Test table.repmeas (repeated measures table)"
    ds = datasets.get_uv()
    print(table.repmeas('fltvar', 'A', 'rm', data=ds))
    print(table.repmeas('fltvar', 'A%B', 'rm', data=ds))
    print(table.repmeas('fltvar', 'A', 'B%rm', data=ds))

    # with int model
    ds['Bv'] = ds['B'].as_var({'b1': 1, 'b2': 2})
    print(table.repmeas('fltvar', 'A', 'Bv%rm', data=ds))

    # test naturalization of cellnames
    ds['ANum'] = Factor(ds['A'], labels={'a1': '1', 'a2': '2'})
    print(table.repmeas('fltvar', 'ANum', 'rm', data=ds))

    # with empty cell name
    ds['A'].update_labels({'a1': ''})
    print(table.repmeas('fltvar', 'A', 'rm', data=ds))
