"""Some basic example datasets for testing."""
from itertools import product
from pathlib import Path
import shutil
import string
from typing import Literal, Optional, Tuple, Union

import mne
from mne import minimum_norm as mn
import numpy as np

from .. import _info, load
from .._data_obj import Dataset, Factor, Var, NDVar, Case, Categorial, Scalar, Sensor, Space, UTS
from .._ndvar import concatenate, convolve
from .._types import PathArg


def permute(variables):
    """Create a Dataset from permuting variables.

    Parameters
    ----------
    variables : sequence
        Sequence of (name, values) tuples. For examples:
        ``(('A', ('a1', 'a2')), ('B', ('b1', 'b2')))``

    Examples
    --------
    >>> print(permute((('A', ('a1', 'a2')),('B', ('b1', 'b2')))))
    A    B
    -------
    a1   b1
    a1   b2
    a2   b1
    a2   b2
    """
    names = (v[0] for v in variables)
    cases = tuple(product(*(v[1] for v in variables)))
    return Dataset.from_caselist(names, cases)


def _get_continuous(
        n_samples: int = 100,
        ynd: bool = False,
        seed: int = 0,
        xn: int = 0,
) -> Dataset:
    """Generate continuous data for temporal response function tests

    Parameters
    ----------
    n_samples
        Number of samples to simulate.
    ynd
        Include 3d ``y``.
    seed
        Seed for :func:`numpy.random.seed` (``None`` to skip seeding random
        state; default us 0).
    xn
        Number of rows in ``xn``.

    Returns
    -------
    data
        Dataset with simulated data:

         - ``x1``: random time series
         - ``x2``: two random time series
         - ``h1`` and ``h2``: Kernels corresponding to ``x1`` and ``x2``
         - ``y``: convolution of ``(x1 * h1) + (x2 * h2)``
    """
    random = np.random if seed is None else np.random.RandomState(seed)
    time = UTS(0, 0.1, n_samples)
    h_time = UTS(0, 0.1, 10)
    xdim = Scalar('xdim', [0, 1])

    x1 = NDVar(random.normal(0, 1, (n_samples,)), (time,), 'x1')
    h1 = NDVar(np.array([0, 0, 1, 3, 0, 0, 0, 0, 2, 3]), (h_time,), 'h1')

    x2 = NDVar(random.normal(0, 1, (2, n_samples,)), (xdim, time), 'x2')
    h2 = NDVar(np.array([[0, 0, 0, 0, 0, 0, -1, -3, 0, 0],
                         [0, 0, 2, 2, 0, 0, 0, 0, 0, 0]]),
               (xdim, h_time), 'h2')

    y = convolve(h1, x1)
    y += convolve(h2, x2)
    y.name = 'y'
    y.info = _info.for_eeg()
    out = {'y': y, 'x1': x1, 'h1': h1, 'x2': x2, 'h2': h2}
    if ynd:
        dim = Sensor([[-1, 0, 0], [1, 0, 0]], ['SL', 'SR'], 'TEST-2-CH')
        out['h1nd'] = h1 = concatenate([h1, -h1], dim, 'h1')
        out['h2nd'] = h2 = concatenate([h2, h2 * 0], dim, 'h2')
        out['ynd'] = convolve(h1, x1) + convolve(h2, x2)
    if xn:
        xn_dim = Scalar('xdim', range(xn))
        out['xn'] = NDVar(random.normal(0, 1, (xn, n_samples)), (xn_dim, time))
        out['hn'] = NDVar(random.normal(0, 1, (xn, 10)), (xn_dim, h_time))
        out['yn'] = convolve(out['hn'], out['xn'])
        out['yn'] += random.normal(0, 1, (n_samples,))
    return Dataset(out)


def get_loftus_masson_1994():
    "Dataset used for illustration purposes by Loftus and Masson (1994)"
    ds = Dataset()
    ds['subject'] = Factor(range(1, 11), tile=3, random=True)
    ds['exposure'] = Var([1, 2, 5], repeat=10)
    ds['n_recalled'] = Var([10, 6, 11, 22, 16, 15, 1, 12, 9, 8,
                            13, 8, 14, 23, 18, 17, 1, 15, 12, 9,
                            13, 8, 14, 25, 20, 17, 4, 17, 12, 12])
    return ds


def get_mne_epochs():
    """MNE-Python Epochs"""
    data_path = Path(mne.datasets.sample.data_path())
    raw_path = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    events_path = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-eve.fif'
    raw = mne.io.Raw(raw_path)
    events = mne.read_events(events_path)
    epochs = mne.Epochs(raw, events, 32, -0.1, 0.4, preload=True)
    return epochs


def get_mne_evoked(ndvar=False):
    """MNE-Python Evoked

    Parameters
    ----------
    ndvar : bool
        Convert to NDVar (default False).
    """
    data_path = Path(mne.datasets.sample.data_path())
    evoked_path = data_path / 'MEG' / 'sample' / 'sample_audvis-ave.fif'
    evoked = mne.Evoked(evoked_path, "Left Auditory")
    if ndvar:
        return load.mne.evoked_ndvar(evoked)
    else:
        return evoked


def get_mne_stc(ndvar=False, src='ico-5', subject='sample'):
    """MNE-Python SourceEstimate

    Parameters
    ----------
    ndvar : bool
        Convert to NDVar (default False; src="ico-4" is false, but it works as
        long as the source space is not accessed).
    src : 'ico-5' | 'vol-7' | 'oct-4
        Source space to use.

    Notes
    -----
    Source space only available for ``oct-4``, ``sample``.
    """
    data_path = Path(mne.datasets.testing.data_path())
    meg_sdir = data_path / 'MEG' / 'sample'
    subjects_dir = data_path / 'subjects'
    # scaled subject
    if subject == 'fsaverage_scaled':
        subject_dir = subjects_dir / subject
        if not subject_dir.exists():
            mne.scale_mri('fsaverage', subject, .9, subjects_dir=subjects_dir, skip_fiducials=True, labels=False, annot=True)
        data_subject = 'fsaverage'
    else:
        data_subject = subject

    if src == 'vol-7':
        inv = mn.read_inverse_operator(str(meg_sdir / 'sample_audvis_trunc-meg-vol-7-meg-inv.fif'))
        evoked = mne.read_evokeds(str(meg_sdir / 'sample_audvis_trunc-ave.fif'), 'Left Auditory')
        stc = mn.apply_inverse(evoked, inv, method='MNE', pick_ori='vector')
        if data_subject == 'fsaverage':
            m = mne.compute_source_morph(stc, 'sample', data_subject, subjects_dir)
            stc = m.apply(stc)
            stc.subject = subject
        elif subject != 'sample':
            raise ValueError(f"subject={subject!r}")
        if ndvar:
            return load.mne.stc_ndvar(stc, subject, 'vol-7', subjects_dir, 'MNE', sss_filename='{subject}-volume-7mm-src.fif')
        else:
            return stc
    elif src == 'oct-4':
        if subject != 'sample':
            raise ValueError(f"subject={subject!r}: source space only available for 'sample'")
        inv = mne.minimum_norm.read_inverse_operator(meg_sdir / 'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
        evokeds = mne.read_evokeds(meg_sdir / 'sample_audvis_trunc-ave.fif')
        evoked = mne.combine_evoked([evokeds[i].apply_baseline() for i in [0, 1]], [1, 1])
        stc = mne.minimum_norm.apply_inverse(evoked, inv)
    elif src == 'ico-5':
        stc_path = meg_sdir / f'{data_subject}_audvis_trunc-meg'
        stc = mne.read_source_estimate(str(stc_path), subject)
    else:
        raise ValueError(f"src={src!r}")

    if ndvar:
        return load.mne.stc_ndvar(stc, subject, src, subjects_dir)
    else:
        return stc


def _mne_source_space(subject, src_tag, subjects_dir):
    """Load mne source space

    Parameters
    ----------
    subject : str
        Subejct
    src_tag : str
        Spacing (e.g., 'ico-4').
    """
    src_file = Path(subjects_dir) / subject / 'bem' / f'{subject}-{src_tag}-src.fif'
    src, spacing = src_tag.split('-')
    if src_file.exists():
        return mne.read_source_spaces(src_file, False)
    elif src == 'ico':
        ss = mne.setup_source_space(subject, spacing=src + spacing, subjects_dir=subjects_dir, add_dist=True)
    elif src == 'vol':
        mri_file = subjects_dir / subject / 'mri' / 'orig.mgz'
        bem_file = subjects_dir / subject / 'bem' / 'sample-5120-5120-5120-bem-sol.fif'
        ss = mne.setup_volume_source_space(subject, pos=float(spacing), mri=mri_file, bem=bem_file, mindist=0., exclude=0., subjects_dir=subjects_dir)
    else:
        raise ValueError(f"{src_tag=}")
    mne.write_source_spaces(src_file, ss)
    return ss


def get_mne_sample(
        tmin: float = -0.1,
        tmax: float = 0.4,
        baseline: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0),
        sns: Union[bool, Literal['eeg', 'mag', 'grad']] = False,
        src: Literal[False, 'ico', 'vol'] = None,
        sub: str = "modality=='A'",
        ori: Literal['free', 'fixed', 'vector'] = 'free',
        snr: float = 2,
        method: str = 'dSPM',
        rm: bool = False,
        stc: bool = False,
        hpf: float = 0,
        proj: bool = True,
        fsaverage: bool = False,
):
    """Load events and epochs from the MNE sample data

    Parameters
    ----------
    tmin
        Relative time of the first sample of the epoch.
    tmax
        Relative time of the last sample of the epoch.
    baseline
        Period for baseline correction.
    sns
        Add sensor space data as NDVar as ``ds['meg']`` (default ``False``).
        Set to ``'grad'`` to load gradiometer data.
    src
        Add source space data as NDVar as ``ds['src']`` (default ``False``).
    sub
        Expression for subset of events to load. For a very small dataset use e.g.
        ``[0,1]``.
    ori
        Orientation of sources.
    snr
        MNE inverse parameter.
    method
        MNE inverse parameter.
    rm
        Pretend to be a repeated measures dataset (adds 'subject' variable).
    stc
        Add mne SourceEstimate for source space data as ``ds['stc']`` (default
        ``False``).
    hpf
        High pass filter cutoff.
    proj
        Add projectors.
    fsaverage
        Morph data to FSAverage template brain.

    Returns
    -------
    ds : Dataset
        Dataset with epochs from the MNE sample dataset in ``ds['epochs']``.
    """
    if ori == 'free':
        loose = 1
        fixed = False
        pick_ori = None
    elif ori == 'fixed':
        loose = 0
        fixed = True
        pick_ori = None
    elif ori == 'vector':
        loose = 1
        fixed = False
        pick_ori = 'vector'
    else:
        raise ValueError(f"{ori=}")

    data_dir = Path(mne.datasets.sample.data_path())
    meg_dir = data_dir / 'MEG' / 'sample'
    raw_file = meg_dir / 'sample_audvis_filt-0-40_raw.fif'
    event_file = meg_dir / 'sample_audvis_filt-0-40-eve.fif'
    subjects_dir = data_dir / 'subjects'
    subject = 'sample'
    label_path = subjects_dir / subject / 'label' / '%s.label'

    if not event_file.exists():
        raw = mne.io.Raw(raw_file)
        events = mne.find_events(raw, stim_channel='STI 014')
        mne.write_events(event_file, events)
    ds = load.mne.events(raw_file, events=event_file)
    if hpf:
        ds.info['raw'].load_data()
        ds.info['raw'].filter(hpf, None, n_jobs=1)
    ds.index()
    ds.info['subjects_dir'] = subjects_dir
    ds.info['subject'] = subject
    ds.info['label'] = label_path

    # get the trigger variable form the dataset for eaier access
    trigger = ds['trigger']

    # use trigger to add various labels to the dataset
    ds['condition'] = Factor(trigger, labels={1: 'LA', 2: 'RA', 3: 'LV', 4: 'RV', 5: 'smiley', 32: 'button'})
    ds['side'] = Factor(trigger, labels={1: 'L', 2: 'R', 3: 'L', 4: 'R', 5: 'None', 32: 'None'})
    ds['modality'] = Factor(trigger, labels={1: 'A', 2: 'A', 3: 'V', 4: 'V', 5: 'None', 32: 'None'})

    if rm:
        ds = ds.sub('trigger < 5')
        ds = ds.equalize_counts('side % modality')
        subject_f = ds.eval('side % modality').enumerate_cells()
        ds['subject'] = subject_f.as_factor('s%r', random=True)

    if sub:
        ds = ds.sub(sub)

    load.mne.add_mne_epochs(ds, tmin, tmax, baseline)
    if sns:
        data_arg = 'mag' if sns is True else sns
        ds['meg'] = load.mne.epochs_ndvar(ds['epochs'], data=data_arg, sysname='neuromag306', proj=proj)

    if not src:
        return ds
    elif src == 'ico':
        src_tag = inv_tag = 'ico-4'
        if fixed:
            inv_tag += '-fixed'
    elif src == 'vol':
        src_tag = inv_tag = 'vol-10'
    else:
        raise ValueError(f"{src=}")
    epochs = ds['epochs']

    # get inverse operator
    inv_file = meg_dir / f'sample_eelbrain_{inv_tag}-inv.fif'
    if inv_file.exists():
        inv = mne.minimum_norm.read_inverse_operator(inv_file)
    else:
        fwd_file = meg_dir / f'sample-{src_tag}-fwd.fif'
        bem_dir = subjects_dir / subject / 'bem'
        bem_file = bem_dir / 'sample-5120-5120-5120-bem-sol.fif'
        trans_file = meg_dir / 'sample_audvis_raw-trans.fif'

        if fwd_file.exists():
            fwd = mne.read_forward_solution(fwd_file)
        else:
            src_ = _mne_source_space(subject, src_tag, subjects_dir)
            fwd = mne.make_forward_solution(epochs.info, trans_file, src_, bem_file)
            mne.write_forward_solution(fwd_file, fwd)

        cov_file = meg_dir / 'sample_audvis-cov.fif'
        cov = mne.read_cov(cov_file)
        inv = mn.make_inverse_operator(epochs.info, fwd, cov, loose=loose, depth=None, fixed=fixed)
        mne.minimum_norm.write_inverse_operator(inv_file, inv)
    ds.info['inv'] = inv

    stcs = mn.apply_inverse_epochs(epochs, inv, 1. / (snr ** 2), method, pick_ori=pick_ori)
    ds['src'] = load.mne.stc_ndvar(stcs, subject, src_tag, subjects_dir, method, fixed)
    if stc:
        ds['stc'] = stcs

    if fsaverage:
        src_to = _mne_source_space('fsaverage', src_tag, subjects_dir)
        morph = mne.compute_source_morph(stcs[0], subject, 'fsaverage', subjects_dir, src_to=src_to)
        stcs_m = [morph.apply(stc) for stc in stcs]
        ds['srcm'] = load.mne.stc_ndvar(stcs_m, 'fsaverage', src_tag, subjects_dir, method, fixed)

    return ds


def get_ndvar(case=0, time=100, frequency=8, cat=0, sensor=0, name='ndvar'):
    dims = []
    if case:
        dims.append(Case(case))
    if time:
        dims.append(UTS(-0.1, 0.01, time))
    if frequency:
        dims.append(Scalar('frequency', np.logspace(2, 3.5, frequency)))
    if cat:
        dims.append(Categorial('cat', string.ascii_lowercase[:cat]))
    if sensor:
        dims.append(get_sensor(sensor))
    shape = [len(dim) for dim in dims]
    x = np.random.normal(0, 1, shape)
    return NDVar(x, dims, name)


def get_uts(utsnd=False, seed=0, nrm=False, vector3d=False):
    """Create a sample Dataset with 60 cases and random data.

    Parameters
    ----------
    utsnd : bool
        Add a sensor by time NDVar (called 'utsnd').
    seed : None | int
        If not None, call ``numpy.random.seed(seed)`` to ensure replicability.
    nrm : bool
        Add a nested random effect Factor "nrm" (nested in ``A``).
    vector3d : bool
        Add a space x time vector time series as ``v3d``.

    Returns
    -------
    ds : Dataset
        Datasets with data from random distributions.
    """
    random = np.random if seed is None else np.random.RandomState(seed)

    ds = Dataset()

    # add a model
    ds['A'] = Factor(['a0', 'a1'], repeat=30)
    ds['B'] = Factor(['b0', 'b1'], repeat=15, tile=2)
    ds['rm'] = Factor(('R%.2i' % i for i in range(15)), tile=4, random=True)
    ds['ind'] = Factor(('R%.2i' % i for i in range(60)), random=True)

    # add dependent variables
    rm_var = np.tile(random.normal(size=15), 4)
    y = np.hstack((random.normal(size=45), random.normal(1, size=15)))
    y += rm_var
    ds['Y'] = Var(y)
    ybin = random.randint(0, 2, size=60)
    ds['YBin'] = Factor(ybin, labels={0: 'c1', 1: 'c2'})
    ycat = random.randint(0, 3, size=60)
    ds['YCat'] = Factor(ycat, labels={0: 'c1', 1: 'c2', 2: 'c3'})

    # add a uts NDVar
    time = UTS(-.2, .01, 100)
    y = random.normal(0, .5, (60, len(time)))
    y += rm_var[:, None]
    y[:15, 20:60] += np.hanning(40) * 1  # interaction
    y[:30, 50:80] += np.hanning(30) * 1  # main effect
    ds['uts'] = NDVar(y, ('case', time))

    # add sensor NDVar
    if utsnd:
        y = random.normal(0, 1, (60, 5, len(time)))
        y += rm_var[:, None, None]
        # add interaction
        win = np.hanning(50)
        y[:15, 0, 50:] += win * 3
        y[:15, 1, 50:] += win * 2
        y[:15, 4, 50:] += win
        # add main effect
        y[30:, 2, 25:75] += win * 2.5
        y[30:, 3, 25:75] += win * 1.5
        y[30:, 4, 25:75] += win
        # add spectral effect
        freq = 15.0  # >= 2
        x = np.sin(time.times * freq * 2 * np.pi)
        for i in range(30):
            shift = random.randint(0, 100 / freq)
            y[i, 2, 25:75] += 1.1 * win * x[shift: 50 + shift]
            y[i, 3, 25:75] += 1.5 * win * x[shift: 50 + shift]
            y[i, 4, 25:75] += 0.5 * win * x[shift: 50 + shift]

        sensor = get_sensor(5)
        dims = ('case', sensor, time)
        ds['utsnd'] = NDVar(y, dims, info=_info.for_eeg())

    # nested random effect
    if nrm:
        ds['nrm'] = Factor([f'{a}{i:02}' for a in 'AB' for _ in range(2) for
                            i in range(15)], random=True)

    if vector3d:
        x = random.normal(0, 1, (60, 3, 100))
        # main effect
        x[:30, 0, 50:80] += np.hanning(30) * 0.7
        x[:30, 1, 50:80] += np.hanning(30) * -0.5
        x[:30, 2, 50:80] += np.hanning(30) * 0.3
        ds['v3d'] = NDVar(x, (Case, Space('RAS'), time))

    return ds


def get_uv(seed=0, nrm=False, vector=False):
    """Dataset with random univariate data

    Parameters
    ----------
    seed : None | int
        Seed the numpy random state before generating random data.
    nrm : bool
        Add a nested random-effects variable (default False).
    vector : bool
        Add a 3d vector variable as ``ds['v']`` (default ``False``).
    """
    random = np.random if seed is None else np.random.RandomState(seed)

    ds = permute([('A', ('a1', 'a2')),
                  ('B', ('b1', 'b2')),
                  ('rm', ['s%03i' % i for i in range(20)])])
    ds['rm'].random = True
    ds['intvar'] = Var(random.randint(5, 15, 80))
    ds['intvar'][:20] += 3
    ds['fltvar'] = Var(random.normal(0, 1, 80))
    ds['fltvar'][:40] += 1.
    ds['fltvar2'] = Var(random.normal(0, 1, 80))
    ds['fltvar2'][40:] += ds['fltvar'][40:].x
    ds['index'] = Var(np.repeat([True, False], 40))
    if nrm:
        ds['nrm'] = Factor(['s%03i' % i for i in range(40)], tile=2, random=True)
    if vector:
        x = random.normal(0, 1, (80, 3))
        x[:40] += [.3, .3, .3]
        ds['v'] = NDVar(x, (Case, Space('RAS')))
    return ds


def get_sensor(n):
    assert n == 5
    locs = np.array([[-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0]])
    sensor = Sensor(locs, sysname='test_sens')
    sensor.set_adjacency(connect_dist=1.75)
    return sensor


def setup_samples_experiment(
        dst: PathArg,
        n_subjects: int = 3,
        n_segments: int = 4,
        n_sessions: int = 1,
        n_visits: int = 1,
        name: str = 'SampleExperiment',
        mris: bool = False,
        mris_only: bool = False,
        pick: str = 'mag',
):
    """Setup up file structure for the ``SampleExperiment`` class

    Corresponding to the pipeline in ``examples/experiment``.

    Parameters
    ----------
    dst
        Path. ``dst`` should exist, a new folder called ``name`` will be
        created within ``dst``.
    n_subjects
        Number of subjects.
    n_segments
        Number of data segments to include in each file.
    n_sessions
        Number of sessions.
    n_visits
        Number of visits.
    name
        Name for the directory for the new experiment (default
        ``'SampleExperiment'``).
    mris
        Set up MRIs.
    mris_only
        Only create MRIs, skip MEG data (add MRIs to existing experiment data).
    pick
        Pick a certain channel type (``''`` to copy all channels).
    """
    # find data source
    data_path = Path(mne.datasets.sample.data_path())
    fsaverage_path = Path(mne.datasets.fetch_fsaverage())

    # setup destination
    dst = Path(dst).expanduser().resolve()
    root = dst / name
    root.mkdir(exist_ok=mris_only)

    if n_sessions > 1 and n_visits > 1:
        raise NotImplementedError
    n_recordings = n_subjects * max(n_sessions, n_visits)
    subjects = [f'R{s_id:04}' for s_id in range(n_subjects)]

    meg_sdir = root / 'meg'
    meg_sdir.mkdir(exist_ok=mris_only)

    if mris:
        mri_sdir = root / 'mri'
        if mris_only and mri_sdir.exists():
            shutil.rmtree(mri_sdir)
        mri_sdir.mkdir()
        # copy rudimentary fsaverage
        surf_names = ['inflated', 'white', 'orig', 'orig_avg', 'curv', 'sphere']
        files = {
            'bem': ['fsaverage-head.fif', 'fsaverage-inner_skull-bem.fif'],
            'label': ['lh.aparc.annot', 'rh.aparc.annot'],
            'surf': [f'{hemi}.{name}' for hemi, name in product(['lh', 'rh'], surf_names)],
            'mri': [],
        }
        dst_s_dir = mri_sdir / 'fsaverage'
        dst_s_dir.mkdir()
        # from fsaverage
        for dir_name, file_names in files.items():
            src_dir = fsaverage_path / dir_name
            dst_dir = dst_s_dir / dir_name
            dst_dir.mkdir()
            for file_name in file_names:
                shutil.copy(src_dir / file_name, dst_dir / file_name)
        # source space
        src_src = fsaverage_path / 'bem' / 'fsaverage-ico-3-src.fif'
        src_dst = dst_s_dir / 'bem' / 'fsaverage-ico-3-src.fif'
        if not src_src.exists():
            src = mne.setup_source_space('fsaverage', 'ico1', subjects_dir=fsaverage_path.parent)
            src.save(src_src)
        shutil.copy(src_src, src_dst)
        # create scaled brains
        trans = mne.Transform(4, 5, [[ 0.9998371,  -0.00766024,  0.01634169,  0.00289569],
                                     [ 0.00933457,  0.99443108, -0.10497498, -0.0205526 ],
                                     [-0.01544655,  0.10511042,  0.9943406,  -0.04443745],
                                     [ 0.,          0.,          0.,          1.        ]])
        for subject in subjects:
            mne.scale_mri('fsaverage', subject, 1., subjects_dir=mri_sdir, skip_fiducials=True, labels=False)
            meg_dir = meg_sdir / subject
            meg_dir.mkdir(exist_ok=mris_only)
            trans.save(str(meg_dir / f'{subject}-trans.fif'))
    if mris_only:
        return

    # MEG
    raw_path = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = mne.io.read_raw_fif(str(raw_path))
    raw.info['bads'] = []
    sfreq = raw.info['sfreq']

    # find segmentation points
    events = mne.find_events(raw)
    events[:, 0] -= raw.first_samp
    segs = []
    n = 0
    t_start = 0
    for sample, _, trigger in events:
        if trigger == 5:  # smiley
            n += 1
        if n == n_segments:
            t = sample / sfreq
            segs.append((t_start, t))
            if len(segs) == n_recordings:
                break
            t_start = t
            n = 0
    else:
        raise ValueError("Not enough data in sample raw. Try smaller ns.")

    if n_visits > 1:
        sessions = ['sample', *(f'sample {i}' for i in range(1, n_visits))]
    elif n_sessions > 1:
        sessions = ['sample%i' % (i + 1) for i in range(n_sessions)]
    else:
        sessions = ['sample']

    for subject in subjects:
        meg_dir = meg_sdir / subject
        meg_dir.mkdir(exist_ok=mris)
        for session in sessions:
            start, stop = segs.pop()
            raw_ = raw.copy().crop(start, stop)
            raw_.load_data()
            if pick:
                raw_.pick_types(pick, stim=True, exclude=[])
            raw_.save(str(meg_dir / f'{subject}_{session}-raw.fif'))
