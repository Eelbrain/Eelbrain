# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from ..fmtxt import Section
from ._sensor import sensor_results, sensor_time_results
from ._source import source_time_results
from ._uts import time_results


def result_report(res, ds, title=None, colors=None):
    """Automatically generate section from an NDTest result object

    Parameters
    ----------
    res : NDTest
        Test-result.
    ds : Dataset
        Dataset containing the data on which the test was performed.
    """
    sec = Section(title or res._name())

    dims = {dim.name for dim in res._dims}
    sec.append(res.info_list())

    if dims == {'time'}:
        sec.append(time_results(res, ds, colors))
    elif dims == {'sensor'}:
        sec.append(sensor_results(res, ds, colors))
    elif dims == {'time', 'sensor'}:
        sec.append(sensor_time_results(res, ds, colors))
    elif dims == {'time', 'source'}:
        sec.append(source_time_results(res, ds, colors))
    else:
        raise NotImplementedError(f"{dims=}")
    return sec
