# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

from eelbrain import datasets, load
from eelbrain.testing import assert_dataobj_equal, file_path


@pytest.mark.file_test
def test_pickle():
    ds = datasets.get_uts()

    ds_2 = load.unpickle(file_path('uts-py2.pickle'))
    assert_dataobj_equal(ds_2, ds, 15)
    ds_3 = load.unpickle(file_path('uts-py3.pickle'))
    assert_dataobj_equal(ds_3, ds, 15)
