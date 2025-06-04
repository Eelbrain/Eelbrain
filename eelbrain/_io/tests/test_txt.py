# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import Dataset, datasets, load
from eelbrain.testing import TempDir, assert_dataobj_equal, assert_dataset_equal, file_path


def test_r_tsv_io():
    "Test reading output of write.table"
    path = file_path('r-write.table.txt')
    ds = load.tsv(path, types={'row': 'f'})
    assert_array_equal(ds['row'], ['1', '2'])
    assert_array_equal(ds['participant'], [1, 1])
    assert_array_equal(ds['condition'], ['3B', '3B'])
    assert_array_equal(ds['bin'], [0, 0])


def test_tsv_io():
    """Test tsv I/O"""
    tempdir = TempDir()
    names = ['A', 'B', 'rm', 'intvar', 'fltvar', 'fltvar2', 'index']
    ds = datasets.get_uv()
    ds['fltvar'][5:10] = np.nan
    ds[:4, 'rm'] = ''

    # save and load
    dst = Path(tempdir) / 'ds.txt'
    ds.save_txt(dst)
    ds1 = load.tsv(dst, random='rm')
    assert_dataset_equal(ds1, ds, decimal=10)
    ds1 = load.tsv(dst, skiprows=1, names=names, random='rm')
    assert_dataset_equal(ds1, ds, decimal=10)
    # delimiter
    for delimiter in [' ', ',']:
        ds.save_txt(dst, delimiter=delimiter)
        ds1 = load.tsv(dst, delimiter=delimiter, random='rm')
        assert_dataset_equal(ds1, ds, decimal=10)

    # guess data types with missing
    intvar2 = ds['intvar'].as_factor()
    intvar2[10:] = ''
    ds_intvar = Dataset((intvar2,))
    ds_intvar.save_txt(dst)
    ds_intvar1 = load.tsv(dst, empty='nan')
    assert_dataobj_equal(ds_intvar1['intvar', :10], ds['intvar', :10])
    assert_array_equal(ds_intvar1['intvar', 10:], np.nan)

    # str with space
    ds[:5, 'A'] = 'a 1'
    ds.save_txt(dst)
    ds1 = load.tsv(dst, random='rm')
    assert_dataset_equal(ds1, ds, decimal=10)
    ds.save_txt(dst, delimiter=' ')
    ds1 = load.tsv(dst, delimiter=' ', random='rm')
    assert_dataset_equal(ds1, ds, decimal=10)

    # Fixed column width
    path = file_path('fox-prestige')
    ds1 = load.tsv(path, delimiter=' ', skipinitialspace=True)
    assert ds1[1] == {'id': 'GENERAL.MANAGERS', 'education': 12.26, 'income': 25879, 'women': 4.02, 'prestige': 69.1, 'census': 1130, 'type': 'prof'}

    # Empty rows
    out = ds.copy()
    out['intvar'] = out['intvar'].as_factor()
    index = out['intvar'] == '11'
    assert index.sum() == 6
    out[index, 'intvar'] = ''
    out.save_txt(dst)
    ds1 = load.tsv(dst, random='rm')
    assert_dataset_equal(ds1, out, decimal=10)
    ds1 = load.tsv(dst, random='rm', empty='11')
    assert_dataset_equal(ds1, ds, decimal=10)
    ds1 = load.tsv(dst, random='rm', empty=11)
    assert_dataset_equal(ds1, ds, decimal=10)
