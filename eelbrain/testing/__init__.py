# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pytest

pytest.register_assert_rewrite('eelbrain.testing._testing')

from ..datasets._simple import get_ndvar
from ._testing import (
    gui_test, hide_plots,
    requires_framework_build, requires_mne_sample_data, requires_mne_testing_data, requires_r_ez,
    skip_on_windows,
    TempDir, working_directory,
    assert_dataset_equal, assert_dataobj_equal, assert_fmtxt_str_equals, assert_source_space_equal,
    file_path, path,
)
