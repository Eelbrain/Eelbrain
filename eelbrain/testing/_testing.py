# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Utilities for testing"
from contextlib import ContextDecorator, contextmanager
from functools import reduce
import os
from operator import mul
from pathlib import Path
import shutil
import tempfile

import mne
import numpy as np
from numpy.testing import assert_array_equal
import pytest

import eelbrain._wxgui
from .._config import CONFIG
from .._data_obj import Dataset, NDVar, Var, Factor, isdatalist, isuv


class TempDir(str):
    "After MNE-Python mne.utils"
    def __new__(cls):
        return str.__new__(cls, tempfile.mkdtemp())

    def __del__(self):
        shutil.rmtree(self, ignore_errors=True)


def assert_dataset_equal(ds1, ds2, decimal=None):
    """
    Raise an assertion if two Datasets are not equal up to desired precision.

    Parameters
    ----------
    ds1, ds2 : Dataset
        Datasets to compare.
    decimal : None | int
        Desired precision (default is exact match).
    """
    assert ds1.keys() == ds2.keys()
    for k in ds1.keys():
        assert_dataobj_equal(ds1[k], ds2[k], decimal=decimal)
    assert ds1.info.keys() == ds2.info.keys()


def assert_dataobj_equal(d1, d2, decimal=None, name=True):
    """Assert that two data-objects are equal up to desired precision.

    Parameters
    ----------
    d1, d2 : data-objects
        Data-objects to compare.
    decimal : None | int
        Desired precision (default is exact match).
    name : bool
        Assert that ``d1.name == d2.name``.
    """
    assert type(d1) == type(d2)  # noqa: E721
    if name:
        assert d1.name == d2.name
    assert len(d1) == len(d2)
    if isuv(d1, interaction=True):
        if isinstance(d1, Var):
            is_equal = np.isclose(d1.x, d2.x, equal_nan=True, rtol=0, atol=10**-decimal if decimal else 0)
        else:
            is_equal = d1 == d2
        if not np.all(is_equal):
            ds = Dataset()
            ds['value'] = d1.copy()
            ds['target'] = d2.copy()
            ds['unequal'] = Factor(is_equal, labels={True: '', False: 'x'})
            if isinstance(d1, Var):
                ds['difference'] = d1 - d2
            raise AssertionError(f'Two {d1.__class__.__name__}s named {d1.name!r} have unequal values:\n\n{ds}')
        if isinstance(d1, Factor):
            if d1.random != d2.random:
                raise AssertionError(f"{d1.name}: d1.random={d1.random}, d2.random={d2.random}")
    elif isinstance(d1, NDVar):
        assert d1.dims == d2.dims
        if decimal:
            is_different = np.max(np.abs(d1.x - d2.x)) >= 10**-decimal
        else:
            is_different = np.any(d1.x != d2.x)

        if is_different:
            n = reduce(mul, d1.x.shape)
            difference = np.abs(d1.x - d2.x)
            if decimal:
                different = difference >= 10**-decimal
            else:
                different = d1.x != d2.x
            n_different = different.sum()
            mean_diff = difference[different].sum() / n_different
            raise AssertionError(f"NDVars named {d1.name!r} have unequal values. Difference in {n_different} of {n} values, average difference={mean_diff}.")
    elif isdatalist(d1):
        assert all(item1 == item2 for item1, item2 in zip(d1, d2))
    elif isinstance(d1, Dataset):
        assert_dataset_equal(d1, d2, decimal)
    else:
        raise TypeError(f"{d1.__class__.__name__} is not a data-object type: {d1!r}")


def assert_source_space_equal(src1, src2):
    """Assert that two SourceSpace objects are identical

    Parameters
    ----------
    src1, src2 : SourceSpace objects
        SourceSpace objects to compare.
    """
    assert_array_equal(src1.vertices[0], src2.vertices[0])
    assert_array_equal(src1.vertices[1], src2.vertices[1])
    assert src1.subject == src2.subject
    assert src1.src == src2.src
    assert src1.subjects_dir == src2.subjects_dir


class GUITestContext(ContextDecorator):
    modules = (
        eelbrain._wxgui.select_epochs,
        eelbrain._wxgui.select_components,
        eelbrain._wxgui.history,
        eelbrain._wxgui.load_stcs,
    )

    def __init__(self):
        self._i = 0

    def __enter__(self):
        self._i += 1
        if self._i == 1:
            for mod in self.modules:
                mod.TEST_MODE = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._i -= 1
        if self._i == 0:
            for mod in self.modules:
                mod.TEST_MODE = False


gui_test_context = GUITestContext()


def gui_test(function):
    return gui_test_context(requires_framework_build(function))


def assert_fmtxt_str_equals(
        fmtxt,
        target: str,
        indent: int = 4,
):
    """Allow clean and PEP8 compliant target strings in testing"""
    as_str = str(fmtxt)
    as_str = '\n'.join(line.rstrip() for line in as_str.split('\n'))  # rstrip whitespace (for pep8)
    target = '\n'.join(line[indent:] for line in target.splitlines()[1:-1])
    assert as_str == target


class ConfigContext(ContextDecorator):

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.old = None

    def __enter__(self):
        self.old = CONFIG[self.key]
        CONFIG[self.key] = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        CONFIG[self.key] = self.old


hide_plots_context = ConfigContext('show', False)


def hide_plots(function):
    return hide_plots_context(requires_framework_build(function))


@contextmanager
def working_directory(wd):
    "Context for temporarily changing the working directory"
    cwd = os.getcwd()
    os.chdir(wd)
    try:
        yield
    finally:
        os.chdir(cwd)


def requires_framework_build(function):
    return pytest.mark.framework_build(function)


def requires_mne_sample_data(function):
    if mne.datasets.sample.data_path(download=False):
        return function
    else:
        return pytest.mark.skip('mne sample data unavailable')(function)


def requires_mne_testing_data(function):
    if mne.datasets.testing.data_path(download=False):
        return function
    else:
        return pytest.mark.skip('mne testing data unavailable')(function)


def requires_r_ez(function):
    from .._utils.r_bridge import r, r_warning_filter

    with r_warning_filter:
        success = r('require(ez)')[0]

    if success:
        return function
    else:
        return pytest.mark.skip('r-ez unavailable')(function)


def skip_on_windows(function):
    if os.name == 'nt':
        return pytest.mark.skip('Test disabled on Windows')(function)
    else:
        return function


def file_path(name):
    "Path to test data file in the test_data directory"
    root = Path(__file__).parents[2]
    if name == 'fox-prestige':
        path = root / 'examples' / 'statistics' / 'Fox_Prestige_data.txt'
    else:
        path = root / 'test_data' / name

    if path.exists():
        return path
    else:
        raise IOError("Testing file does not exist. Test can only be executed from source repository.")


def path(string):
    "OS-independent path check"
    return str(Path(string))
