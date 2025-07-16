# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Tools for loading data form eyelink edf files."""
from glob import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .._data_obj import Dataset, Datalist, Var
from .._utils import ui
from .._utils.subp import get_bin

__all__ = ('Edf', 'read_edf')


class Edf:
    """Eyelink .edf file reader.

    Reads an eyelink .edf file and extracts epoch acceptability
    based on contamination with ocular artifacts (saccades and blinks).

    Parameters
    ----------
    path : str(path) | None
        Path to the .edf file. The If path contains '*', the files matching
        the pattern are concatenated. If None, a file-open dialogue will be
        displayed.

    Notes
    -----
    An edf file reader is initialized with the path to the corresponding
    file::

        >>> path = '/path/to/edf.edf'
        >>> edf = load.eyelink.Edf(path)

    There are several ways of retrieving trial acceptability values (see the
    used methods' documentation for more information):


    1. **For all triggers:**
       Acceptability for all triggers can be added to a Dataset with a single
       command if the Dataset contains the same events as the edf file::

          >>> edf.mark_all(ds, ...)


    2. **For a subset of triggers:**
       Often it is more efficient to compute acceptability only for a subset of
       the triggers contained in the edf file. For those cases, the trigger time
       should first be added to the complete Dataset with::

           >>> edf.add_t_to(ds)

       Now, the Dataset can be decimated::

           >>> ds = ds.sub(...)

       and acceptability can be added for the subset::

           >>> edf.mark(ds, ...)

    3. **Customized:**
       For a more customized use, all triggers from the edf can be retrieved using::

           >>> ds_edf = edf.get_triggers()

       The ``ds_edf['t_edf']`` time variable can be used to add trigger time values to
       arbitrary events, which can then in turn be used with::

           >>> edf.mark(...)

    """

    def __init__(self, path=None, samples=False):
        if path is None:
            path = ui.ask_file("Load an eyelink .edf file", "Pick the edf file",
                               [('eyelink data format (*.edf)', '*.edf')])

        # find all paths from which to read
        self.path = path
        if '*' in path:
            self.paths = glob(path)
        else:
            self.paths = [path]

        triggers = []
        artifacts = []
        pos = []
        for path in self.paths:
            edf_asc = read_edf(path)
            triggers += find_edf_triggers(edf_asc)
            artifacts += find_edf_artifacts(edf_asc)
            if samples:
                pos += find_edf_pos(edf_asc)

        dtype = [('T', np.uint32), ('Id', np.uint8)]
        self.triggers = np.array(triggers, dtype=dtype)

        dtype = np.dtype([('event', np.str_, 6), ('start', np.uint32), ('stop', np.uint32)])
        self.artifacts = np.array(artifacts, dtype=dtype)

        self.has_samples = bool(samples)
        if samples:
            self.time = np.array([item[0] for item in pos], dtype=np.uint32)
            self.xpos = np.array([item[1] for item in pos], dtype=np.float16)
            self.ypos = np.array([item[2] for item in pos], dtype=np.float16)
            self.pdia = np.array([item[3] for item in pos], dtype=np.float16)

    def __getstate__(self):
        state = {'path': self.path, 'paths': self.paths,
                 'triggers': self.triggers, 'artifacts': self.artifacts,
                 'has_samples': self.has_samples}
        if self.has_samples:
            state.update(time=self.time, xpos=self.xpos, ypos=self.ypos,
                         pdia=self.pdia)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return "Edf(%r)" % self.path

    def assert_trigger_match(self, ds=None, trigger='trigger'):
        """
        Make sure the Edf and another event list describe the same events.

        Raises an error if the triggers in ``trigger`` do not match the
        triggers in the Edf file(s).

        Parameters
        ----------
        ds : Dataset
            Dataset with events.
        trigger : str | array
            If ``ds`` is a Dataset, ``trigger`` should be a string naming the
            variable in `ds` containing the trigger values. If ``ds`` is
            ``None``, ``trigger`` should be a sequence of event triggers.
        """
        if isinstance(trigger, str):
            trigger = ds[trigger]

        edf_trigger = self.triggers['Id']
        if len(trigger) != len(edf_trigger):
            mm = min(len(trigger), len(edf_trigger))
            for i in range(mm):
                if trigger[i] != edf_trigger[i]:
                    break
            name = getattr(ds, 'name', 'None')
            raise ValueError(f"Dataset {name:r} containes different number of events from edf file {self.path:r} ({len(trigger)} vs {len(edf_trigger)}); first mismatch at {i}.")

        check = (trigger == edf_trigger)
        if not all(check):
            raise ValueError(f"Event ID mismatch: {np.where(check == False)[0]}")

    def add_t_to(self, ds, trigger='trigger', t_edf='t_edf'):
        """Add EDF trigger times as a variable to Dataset ds.

        These trigger times can then be used for Edf.add_by_T(ds) after ds hads
        been decimated.

        Parameters
        ----------
        ds : Dataset
            The Dataset to which the variable is added
        trigger : str | Var | None
            variable (or its name in the Dataset) containing event trigger
            values. Values in this variable are checked against the events in
            the EDF file, and an error is raised if there is a mismatch. This
            test can be skipped by setting trigger=None.
        t_edf : str
            Name for the target variable holding the edf trigger times.

        """
        if trigger:
            self.assert_trigger_match(ds=ds, trigger=trigger)
            if isinstance(trigger, str):
                trigger = ds[trigger]

        ds[t_edf] = Var(self.triggers['T'])

    def filter(self, ds, tstart=-0.1, tstop=0.6, use=['ESACC', 'EBLINK'],
               t='t_edf'):
        """Remove bad events from ``ds``

        Return a copy of the Dataset ``ds`` with all bad events removed. A
        Dataset containing all the bad events is stored in
        ``ds.info['rejected']``.

        Parameters
        ----------
        ds : Dataset
            The Dataset that is to be filtered
        tstart : scalar
            Start of the time window in which to look for artifacts
        tstop : scalar
            End of the time window in which to look for artifacts
        use : list of str
            List of events types which are to be treated as artifacts (possible
            are 'ESACC' and 'EBLINK')
        t : str | Var
            Variable describing edf-relative timing for the events in ``ds``.
            Usually this is a string key for a variable in ``ds``.

        """
        if isinstance(t, str):
            t = ds[t]
        accept = self.get_accept(t, tstart=tstart, tstop=tstop, use=use)
        accepted = ds.sub(accept)
        rejected = ds.sub(accept == False)
        accepted.info['rejected'] = rejected
        return accepted

    def get_accept(self, t=None, tstart=-0.1, tstop=0.6, use=['ESACC', 'EBLINK']):
        """Find good epochs

        Return a boolean Var indicating for each epoch whether it should be
        accepted or not based on ocular artifacts in the edf file.

        Parameters
        ----------
        t : array-like | None
            List of time points (in the edf file's time coordinates). If None,
            the edf's trigger events are used.
        tstart : scalar
            start of the epoch relative to the event (in seconds)
        tstop : scalar
            end of the epoch relative to the even (in seconds)

        """
        if t is None:
            t = self.triggers['T']

        # conert to ms
        start = int(tstart * 1000)
        stop = int(tstop * 1000)

        self._debug = []

        # get data for triggers
        accept = np.empty(len(t), bool)

        x = tuple(self.artifacts['event'] == name for name in use)
        idx = np.any(x, axis=0)
        artifacts = self.artifacts[idx]

        for i, ti in enumerate(t):
            starts_before_tstop = artifacts['start'] < ti + stop
            stops_after_tstart = artifacts['stop'] > ti + start
            overlap = np.all((starts_before_tstop, stops_after_tstart), axis=0)
            accept[i] = not np.any(overlap)

            self._debug.append(overlap)

        return accept

    def get_t(self, name='t_edf'):
        "Retrieve all trigger times in the Dataset"
        return Var(self.triggers['T'], name=name)

    def get_triggers(self, trigger='trigger', t_edf='t_edf'):
        """
        Return a Dataset with triggers and corresponding edf time values

        Parameters
        ----------
        trigger : str
            Name for the trigger value variable.
        t_edf : str
            Name for the variable containing edf event times.

        Returns
        -------
        ds : Dataset
            Dataset with triggers and corresponding edf times.
        """
        ds = Dataset(info={'edf': self})
        ds[trigger] = Var(self.triggers['Id'])
        ds[t_edf] = self.get_t(name=t_edf)
        return ds

    def mark(self, ds, tstart=-0.1, tstop=0.6, good=None, bad=False,
             use=['ESACC', 'EBLINK'], t='t_edf', target='accept'):
        """Mark events in ``ds`` as acceptable or not.

        ``ds`` needs to contain edf trigger times in a variable whose name is
        specified by the ``T`` argument.

        Parameters
        ----------
        ds : Dataset
            Dataset that contains the data to work with.
        tstart : scalar
            start of the time window relevant for rejection.
        tstop : scalar
            stop of the time window relevant for rejection.
        good : bool | None
            vale assigned to epochs that should be retained based on
            the eye-tracker data.
        bad : bool | None
            value that is assigned to epochs that should be rejected
            based on the eye-tracker data.
        use : list of str
            Artifact categories to include
        t : Var
            variable providing the trigger time values
        target : Var
            variable to which the good/bad values are assigned (if it does not
            exist, a new variable will be created with all values True
            initially)

        """
        if isinstance(target, str):
            if target in ds:
                target = ds[target]
            else:
                ds[target] = target = Var(np.ones(ds.n_cases, dtype=bool))

        if isinstance(t, str):
            t = ds[t]

        accept = self.get_accept(t, tstart=tstart, tstop=tstop, use=use)
        if good is not None:
            target[accept] = good
        if bad is not None:
            target[np.invert(accept)] = bad

    def mark_all(self, ds, tstart=-0.1, tstop=0.6, good=None, bad=False,
                 use=['ESACC', 'EBLINK'],
                 trigger='trigger', target='accept'):
        """
        Mark epochs in ``ds`` based on blinks and saccades

        Mark each epoch in ``ds`` for acceptability based on overlap with
        blinks and saccades. ds needs to contain the same number of triggers
        as the edf file. For adding acceptability to a decimated ds, use
        Edf.add_t_to(ds, ...) and then Edf.mark(ds, ...).

        Parameters
        ----------
        ds : Dataset
            Dataset that contains the data to work with.
        tstart : scalar
            start of the time window relevant for rejection.
        tstop : scalar
            stop of the time window relevant for rejection.
        good :
            vale assigned to epochs that should be retained based on
            the eye-tracker data.
        bad :
            value that is assigned to epochs that should be rejected
            based on the eye-tracker data.
        use : list of str
            Artifact categories to include
        trigger : Var | None
            variable containing trigger values for asserting that the Dataset
            contains the same triggers as the edf file(s). The test can be
            skipped by setting ``trigger=None``.
        target : Var
            variable to which the good/bad values are assigned (if it does not
            exist, a new variable will be created with all values True
            initially)
        """
        if trigger:
            self.assert_trigger_match(ds=ds, trigger=trigger)
        t = self.get_t()
        self.mark(ds, tstart, tstop, good, bad, use, t, target)


def events(path, samples=False, ds=None, trigger='trigger', t_edf='t_edf'):
    """Read events from an edf file

    Parameters
    ----------
    path : str
        Filename.
    samples : bool
        Read continuous eye position data as well as events. This is needed to
        extract eye position data later.
    ds : Dataset
        Existing Dataset to which the edf-triggers should be added. If the
        Dataset contains a variable called 'trigger' whose content does not
        match the edf triggers, a ValueError is raised. ds is always modified
        in place, but returned for consistency.
    trigger : str
        Name of the trigger variable.
    t_edf : str
        Name of the edf time variable.

    Returns
    -------
    ds : Dataset
        Dataset with events form the edf file (if the ``ds`` input argument is
        provided, the ds that is returned is the same object as the input ds).
    """
    edf = Edf(path, samples=samples)
    if ds is None:
        ds = edf.get_triggers(trigger, t_edf)
    else:
        if 'edf' in ds.info:
            raise ValueError("ds.info already contains 'edf' entry.")
        if trigger in ds:
            edf.assert_trigger_match(ds)
        else:
            ds[trigger] = Var(edf.triggers['Id'])
        ds.info['edf'] = edf
        edf.add_t_to(ds, trigger, t_edf)

    return ds


def artifact_epochs(ds, tmin=-0.2, tmax=0.6, crop=True, esacc='sacc',
                    eblink='blink', t_edf='t_edf'):
    """Find blink and saccade artifact information by event

    Parameters
    ----------
    ds : Dataset
        Dataset with 'edf' entry in its info dictionary (usually a Dataset
        returned by ``load.eyelink.events()``)
    tmin, tmax : scalar
        Relative start and end points of the epoch (in seconds).
    crop : bool
        Crop events to epoch beginning and end (i.e., if an artifact starts
        before the epoch, set its start to the first sample in the epoch).
    esacc, eblink : None | str
        Name for the variable containing the corresponding information. If
        None, the corresponding variable is not added.
    t_edf : str
        Name of the ds variable containing edf times.

    Returns
    -------
    sacc : Datalist
        Saccade periods.
    blink : Datalist
        Blink periods.
    """
    edf = ds.info['edf']
    start = edf.artifacts['start']
    stop = edf.artifacts['stop']
    is_blink = (edf.artifacts['event'] == 'EBLINK')
    is_sacc = (edf.artifacts['event'] == 'ESACC')

    # edf times are in ms; convert them to s:
    dtype = np.dtype([('event', np.str_, 6), ('start', np.float64), ('stop', np.float64)])
    artifacts_s = edf.artifacts.astype(dtype)
    artifacts_s['start'] /= 1000.
    artifacts_s['stop'] /= 1000.

    sacc = Datalist(name=esacc) if esacc else None
    blink = Datalist(name=eblink) if eblink else None
    for t in ds[t_edf]:
        t_s = t / 1000.
        epoch_idx = np.logical_and(stop > t + tmin, start < t + tmax)
        if esacc:
            idx = np.logical_and(epoch_idx, is_sacc)
            epoch = artifacts_s[idx]
            epoch['start'] -= t_s
            epoch['stop'] -= t_s
            if crop and len(epoch):
                if epoch['start'][0] < tmin:
                    epoch['start'][0] = tmin
                if epoch['stop'][-1] > tmax:
                    epoch['stop'][-1] = tmax
            sacc.append(epoch)
        if eblink:
            idx = np.logical_and(epoch_idx, is_blink)
            epoch = artifacts_s[idx]
            epoch['start'] -= t_s
            epoch['stop'] -= t_s
            if crop and len(epoch):
                if epoch['start'][0] < tmin:
                    epoch['start'][0] = tmin
                if epoch['stop'][-1] > tmax:
                    epoch['stop'][-1] = tmax
            blink.append(epoch)

    return sacc, blink


def add_artifact_epochs(ds, tmin=-0.2, tmax=0.6, crop=True, esacc='sacc',
                        eblink='blink', t_edf='t_edf'):
    """Add a Datalist containing artifact information by event

    Parameters
    ----------
    ds : Dataset
        Dataset with 'edf' entry in its info dictionary (usually a Dataset
        returned by ``load.eyelink.events()``)
    tmin, tmax : scalar
        Relative start and end points of the epoch (in seconds).
    crop : bool
        Crop events to epoch beginning and end (i.e., if an artifact starts
        before the epoch, set its start to the first sample in the epoch).
    esacc, eblink : None | str
        Name for the variable containing the corresponding information. If
        None, the corresponding variable is not added.
    t_edf : str
        Name of the ds variable containing edf times.

    Returns
    -------
    ds : Dataset
        Return the input Dataset for consistency with similar functions; the
        Dataset is modified in place.
    """
    sacc, blink = artifact_epochs(ds, tmin, tmax, crop, esacc, eblink, t_edf)

    if esacc:
        ds.add(sacc)
    if eblink:
        ds.add(blink)
    return ds


def read_edf(fname, what='events'):
    """
    Read the content of an edf file as text using edf2asc

    Convert an "eyelink data format" (.edf) file to a temporary directory
    and parse its content.

    Parameters
    ----------
    fname : str
        Filename.
    what : 'all' | 'events' | 'samples'
        What type of information to read
    """
    if not os.path.isfile(fname):
        err = "%r is not a file." % fname
        raise ValueError(err)

    temp_dir = tempfile.mkdtemp()

    # edf2asc does not seem to handle spaces in filenames?
    if ' ' in fname:
        dst = os.path.join(temp_dir, os.path.basename(fname).replace(' ', '_'))
#         shutil.copy(fname, dst)
        os.symlink(fname, dst)
        fname = dst

    # construct the conversion command
    cmd = [get_bin('edfapi', 'edf2asc'),  # options in Manual p. 106
           '-t', ]  # use only tabs as delimiters
    if what == 'events':
        cmd.append('-e')  # outputs event data only
    elif what == 'samples':
        cmd.append('-s')  # outputs sample data only
    elif what == 'all':
        raise NotImplementedError()
    else:
        raise ValueError("what must be 'events' or 'samples', not %r" % what)

    cmd.extend(('-nst',  # blocks output of start events
                '-p', temp_dir,  # writes output with same name to <path> directory
                fname))

    # run the subprocess
    p = subprocess.Popen(cmd)
    stdout, stderr = p.communicate()
    # Don't check return code because it always return 255

    # find asc file
    name, _ = os.path.splitext(os.path.basename(fname))
    ascname = os.path.extsep.join((name, 'asc'))
    asc_path = os.path.join(temp_dir, ascname)
    if not os.path.exists(asc_path):
        print("======\nstdout\n======\n%s" % stdout)
        print("======\nstderr\n======\n%s" % stderr)
        raise subprocess.CalledProcessError(p.returncode, cmd, (stdout, stderr))
    with open(asc_path) as asc_file:
        asc_str = asc_file.read()

    # clean
    shutil.rmtree(temp_dir)
    return asc_str


def find_edf_triggers(asc_str):
    """Find artifacts in an edf asci representation

    Parameters
    ----------
    asc_str : str
        String with edf asci represenation as returned by read_edf.
    """
    re_trigger = re.compile(r'\bMSG\t(\d+)\tMEG Trigger: (\d+)')
    triggers = re_trigger.findall(asc_str)
    return triggers


def find_edf_artifacts(asc_str, kind='EBLINK|ESACC'):
    """Find artifacts in an edf asci representation

    Parameters
    ----------
    asc_str : str
        String with edf asci represenation as returned by read_edf.
    kind : 'EBLINK|ESACC' | 'EBLINK' | 'ESACC'
        Kind of artifact to search.
    """
    for kind_part in kind.split('|'):
        if kind_part not in ['EBLINK', 'ESACC']:
            raise ValueError("invalid kind parameter: %r" % kind)
    re_artifact = re.compile(r'\b(%s)\t[LR]\t(\d+)\t(\d+)' % kind)
    artifacts = re_artifact.findall(asc_str)
    return artifacts


def find_edf_pos(asc_str):
    """Find position values in an edf asci representation

    Parameters
    ----------
    asc_str : str
        String with edf asci represenation as returned by read_edf.
    """
    re_pos = re.compile(r'\b(\d+)\t(\d+\.\d*)\t(\d+\.\d*)\t(\d+\.\d*)')
    pos = re_pos.findall(asc_str)
    return pos
