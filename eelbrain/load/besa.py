"""Tools for loading data from the BESA-MN pipeline.

.. autosummary::
   :toctree: generated

   mrat_data
   dat_file
   dat_set
   roi

"""
# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
from glob import glob
import re

import numpy as np
from scipy.io import loadmat

from .._data_obj import Dataset, Factor, NDVar, UTS, Scalar, combine
from .._utils import ui


_mat_wildcard = ('Matlab data file (*.mat)', '*.mat')


def dat_file(path):
    """Load an besa source estimate from a dat file

    Parameters
    ----------
    path : str
        Path to the dat file.

    Returns
    -------
    src : NDVar
        Source estimate read from the dat file, with additional info in the
        src.info dict.
    """
    info = {}
    pattern = re.compile(r"(.+):\s*(.+)")
    with open(path) as fid:
        in_header = True
        while in_header:
            line = fid.readline().strip()
            match = pattern.match(line)
            if match:
                key, value = match.groups()
                info[key] = value
            elif line.startswith('=='):
                in_header = False
            else:
                continue

        # time axis
        line = fid.readline()
        times = tuple(map(float, line.split()[2:]))
        tstep = (times[1] - times[0]) / 1000
        tstart = times[0] / 1000
        nsamples = len(times)
        time = UTS(tstart, tstep, nsamples)

        # data
        n_locs = int(info['Locations'])
        n_times = int(info['Time samples'])
        data = np.fromfile(fid, 'float64', sep=" ")
        data = data.reshape((n_locs, n_times + 3))
        # locs = data[:, :3]
        data = data[:, 3:]
        source = Scalar("source", np.arange(n_locs))
        src = NDVar(data, (source, time), info, 'src')

    return src


def dat_set(path, subjects=[], conditions=[]):
    """Load multiple dat files as a Dataset

    Parameters
    ----------
    path : str
        The path to the dat files, contain the placeholders '{subject}' and
        '{condition}'. Can contain ``*``.
    subjects : list
        Subject identifiers. If the list is empty, they are inferred based on
        the path and existing files.
    conditions : list
        Condition labels. If the list is empty, they are inferred based on
        the path and existing files.

    Returns
    -------
    ds : Dataset
        Dataset containing the variables 'subject', 'condition' and 'src' (the
        source estimate).

    See Also
    --------
    dat_set_paths : find just the paths to check whether all files are found
    add_dat_set_epochs : add epochs to the Dataset returned by dat_set_paths()
    """
    ds = dat_set_paths(path, subjects, conditions)
    ds = add_dat_set_epochs(ds)
    return ds


def dat_set_paths(path, subjects=[], conditions=[]):
    """Find paths for a set of dat files

    Parameters
    ----------
    path : str
        The path to the dat files, contain the placeholders '{subject}' and
        '{condition}'. Can contain ``*``.
    subjects : list
        Subject identifiers. If the list is empty, they are inferred based on
        the path and existing files.
    conditions : list
        Condition labels. If the list is empty, they are inferred based on
        the path and existing files.

    Returns
    -------
    ds : Dataset
        Dataset containing the variables 'subject', 'condition' and 'path'.
    """
    # check path
    if ('{subject}' not in path) or ('{condition}' not in path):
        err = ("The path needs to contain the placeholders '{subject}' and "
               "'{condition}'. Got %r." % path)
        raise ValueError(err)

    if not subjects:
        find_subjects = True
        subjects = set()
    else:
        find_subjects = False

    if not conditions:
        find_conditions = True
        conditions = set()
    else:
        find_conditions = False

    # infer subjects and/or conditions from file names
    if find_subjects or find_conditions:
        glob_pattern = path.format(subject='*', condition='*')
        paths = glob(glob_pattern)
        path_ = path.replace('*', '.*')
        path_ = path_.format(subject='(?P<subject>.+)',
                             condition='(?P<condition>.+)')
        pattern = re.compile(path_)
        for path_ in paths:
            m = pattern.match(path_)
            if find_subjects:
                subjects.add(m.group('subject'))
            if find_conditions:
                conditions.add(m.group('condition'))

        subjects = sorted(subjects)
        conditions = sorted(conditions)

    paths = []
    for condition in conditions:
        for subject in subjects:
            path_ = path.format(subject=subject, condition=condition)
            paths_ = glob(path_)
            if len(paths_) == 0:
                err = f"No dat file found for {subject!r}/{condition!r}"
                raise ValueError(err)
            elif len(paths_) > 1:
                err = f"Multiple dat files found for {subject!r}/{condition!r}"
                raise ValueError(err)
            else:
                paths.append(paths_[0])

    # create Dataset
    info = {'path': path}
    ds = Dataset(info=info)
    ds['subject'] = Factor(subjects, tile=len(conditions), random=True)
    ds['condition'] = Factor(conditions, repeat=len(subjects))
    ds['path'] = paths
    return ds


def add_dat_set_epochs(ds, name='src'):
    """
    Read epochs for a Dataset created with :func:`dat_set_paths`

    Parameters
    ----------
    ds : Dataset
        Dataset as returned by :func:`dat_set_paths`
    name : str
        Name for the variable containing the epochs.

    Returns
    -------
    ds : Dataset
        Reference to the input Dataset ds, which is modified in place.
    """
    # read dat files
    stcs = []
    for case in ds.itercases():
        path = case['path']
        stc = dat_file(path)
        stcs.append(stc)

    # create source estimate NDVar
    src = combine(stcs)
    src.info.update(unit="nA", meas="I")
    ds[name] = src
    return ds


def roi(path, adjust_index=True):
    """Load a BESA-MN ROI saved in a ``*.mat`` file.

    Parameters
    ----------
    path : str
        Path to the ``*.mat`` file containing the ROI.
    adjust_index : bool
        Adjust the index for Python (Matlab indexes start with 1, Python
        indexes start with 0).

    Returns
    -------
    roi : array, shape = (n_sources,)
        ROI source indexes.
    """
    mat = loadmat(path)
    roi_idx = np.ravel(mat['roi_sources'])
    if adjust_index:
        roi_idx -= 1
    return roi_idx


def roi_results(path=None, varname=None):
    """
    Load the meg data from a saved besa-mn ROI results object

    Parameters
    ----------
    path : str | None
        Path to the ``*.m`` file containing the saved results. If None, a file
        can be selected using a system file dialog.
    varname : str | None
        If the ``*.m`` file contains more than one variable, the name of the
        variable containing the results can be specified as string.

    Returns
    -------
    besa_mn_data : Dataset
        A Dataset containing the following variables: 'meg' the meg ROI
        waveforms; 'condition' the condition label; 'subject' a subject
        identifier.
    """
    if path is None:
        path = ui.ask_file("Pick a Matlab file with Besa-MN results",
                           "Pick a Matlab File", [_mat_wildcard])
        if not path:
            return

    m = loadmat(path)
    if varname is None:
        keys = [k for k in m.keys() if not k.startswith('_')]
        if len(keys) == 0:
            raise ValueError("Noe data in .mat file")
        elif len(keys) == 1:
            varname = keys[0]
        else:
            err = ("More than one variables in .mat file. Use the varname "
                   "parameter.")
            raise ValueError(err)
    res = m[varname]

    ds = Dataset()

    n_c, n_s, n_t = res['waves'][0][0].shape
    x = res['waves'][0][0].reshape(n_c * n_s, n_t)
    times = np.ravel(res['latencies'][0][0])
    time = UTS(times[0] / 1000, .001, len(times))
    ds['meg'] = NDVar(x, dims=('case', time))

    conds = [c[0] for c in res['conditions'][0][0][0]]
    ds['condition'] = Factor(conds, repeat=n_s)

    ds['subject'] = Factor(map(str, range(n_s)), tile=n_c, random=True)

    return ds


def mrat_data(path=None, tstart=-0.1, roi=None, varname=None):
    """
    Load meg data from a saved mrat dataset object

    Parameters
    ----------
    path : str | None
        Path to the ``*.m`` file containing the saved results. If None, a file
        can be selected using a system file dialog.
    tstart : scalar
        Time value of the first sample in the data.
    roi : numpy index
        Index of the sources to load (Python style indexing, i.e., the first
        source has index 0).
    varname : str | None
        If the ``*.m`` file contains more than one variable, the name of the
        variable containing the results can be specified as string.

    Returns
    -------
    mrat_data : Dataset
        A Dataset containing the following variables: 'meg' the meg ROI
        waveforms; 'condition' the condition label; 'subject' a subject
        identifier.
    """
    if path is None:
        path = ui.ask_file("Pick a Matlab file with Besa-MN results",
                           "Pick a Matlab File", [_mat_wildcard])
        if not path:
            return

    m = loadmat(path)
    if varname is None:
        keys = [k for k in m.keys() if not k.startswith('_')]
        if len(keys) == 0:
            raise ValueError("Noe data in .mat file")
        elif len(keys) == 1:
            varname = keys[0]
        else:
            err = ("More than one variables in .mat file. Use the varname "
                   "parameter.")
            raise ValueError(err)
    data = m[varname]
    conds = [c[0] for c in data['conditionNames'][0][0][0]]

    xs = []
    n_s = None
    for cond in conds:
        if n_s is None:
            n_s = len(data['data'][0][0][cond][0][0][0])
        x_ = np.array([data['data'][0][0][cond][0][0][0][i]
                       for i in range(n_s)])
        xs.append(x_)
    x = np.concatenate(xs)

    if roi is None:
        raise NotImplementedError("Need to specify roi parameter")
    else:
        x = np.mean(x[:, roi], 1)

    ds = Dataset()
    _, n_samples = x.shape
    time = UTS(tstart, .001, n_samples)
    ds['meg'] = NDVar(x, dims=('case', time))

    ds['condition'] = Factor(conds, repeat=n_s)

    ds['subject'] = Factor(map(str, range(n_s)), tile=len(conds), random=True)

    return ds
