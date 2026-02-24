# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Wrapper for binaries

For permission errors: try ``os.chmod`` or ``os.chown``


subprocess documentation
------------------------

http://docs.python.org/library/subprocess.html
http://www.doughellmann.com/PyMOTW/subprocess/
http://explanatorygap.net/2010/05/10/python-subprocess-over-shell/
http://codeghar.wordpress.com/2011/12/09/introduction-to-python-subprocess-module/
"""
import fnmatch
import logging
import os
import subprocess

import mne
from mne.utils import get_subjects_dir, run_subprocess

from . import ui


# environment variables that can hold the path to this tool
_env_vars = {'mne': 'MNE_ROOT',
             'freesurfer': 'FREESURFER_HOME',
             'edfapi': 'EYELINK_HOME'}

# bin directory relative to tool location
_bin_dirs = {'mne': 'bin',
             'freesurfer': 'bin',
             'edfapi': ''}

# binaries to look for to test whether the tool directory is set correctly
_test_bins = {'mne': ('mne_add_patch_info', 'mne_analyze'),
              'freesurfer': ('mri_annotation2label',),
              'edfapi': ('edf2asc',)}


def get_root(package):
    "Get the root path for a package"
    if package not in _env_vars:
        raise KeyError(f"Unknown binary package: {package!r}")
    root = mne.get_config(_env_vars[package])
    if root:
        test_root(package, root)
    else:
        root = _ask_user_for_bin_dir(package)
    return root


def get_bin(package, name):
    "Get path for a binary"
    root = get_root(package)
    bin_path = os.path.join(root, _bin_dirs[package], name)
    if os.path.exists(bin_path):
        return bin_path
    else:
        raise OSError("Binary does not exist: " + bin_path)


def test_root(package, root):
    bin_dir = os.path.join(root, _bin_dirs[package])
    return all(os.path.exists(os.path.join(bin_dir, f)) for f in _test_bins[package])


def _ask_user_for_bin_dir(package):
    """
    Change the location from which binaries are used.

    Parameters
    ----------
    package : str
        Binary package for which to set the directory. One from:
        ``['mne', 'freesurfer', 'edfapi']``
    """
    title = f"Select {package} Directory"
    message = f"Please select the directory of the {package} package."
    while True:
        answer = ui.ask_dir(title, message, must_exist=True)
        if answer:
            if test_root(package, answer):
                mne.set_config(_env_vars[package], answer)
                return answer
            else:
                ui.message("Wrong Directory", "This is not the right directory.")
        else:
            raise OSError(f"{package} directory not set")


_verbose = 1


def command_exists(cmd):
    """Return True if the corresponding command exists

    based http://stackoverflow.com/a/11069822/166700
    """
    return subprocess.call(["type", cmd], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) == 0


def open_in_finder(path):
    """Open ``path`` in the finder"""
    os.system(f'open {path}')


def get_fs_home():
    fs_home = mne.get_config('FREESURFER_HOME')
    save_fs_home = False
    while True:
        problem = fs_home_problem(fs_home)
        if problem:
            save_fs_home = True
            print(problem)
            if fs_home == os.environ.get('FREESURFER_HOME', 0):
                print("WARNING: This directory is set as FREESURFER_HOME "
                      "environment variable. As long as you don't remove this "
                      "environment variable, you will be asked for the proper "
                      "FreeSurfer location every time a FreeSurfer command is "
                      "run.")
            message = "Please select the directory where FreeSurfer is installed"
            print(message)
            fs_home = ui.ask_dir("Select FreeSurfer Directory", message)
            if fs_home is False:
                raise RuntimeError("Could not find FreeSurfer")
        else:
            break
    if save_fs_home:
        mne.set_config('FREESURFER_HOME', fs_home)
    return fs_home


def run_freesurfer_command(command, subjects_dir):
    "Run a FreeSurfer command"
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = subjects_dir
    env['FREESURFER_HOME'] = fs_home = get_fs_home()
    bin_path = os.path.join(fs_home, 'bin')
    if bin_path not in env['PATH']:
        env['PATH'] = ':'.join((bin_path, env['PATH']))

    # run command
    run_subprocess(command, env=env)


def fs_home_problem(fs_home):
    """Check FREESURFER_HOME path

    Return str describing problem or None if the path is okay.
    """
    if fs_home is None:
        return "The FreeSurfer path is not set"
    elif not os.path.exists(fs_home):
        return f"The FreeSurfer path does not exist: {fs_home}"
    else:
        test_path = os.path.join(fs_home, 'bin', 'mri_surf2surf')
        if not os.path.exists(test_path):
            return ("The FreeSurfer path is invalid (%s does not contain bin/"
                    "mri_surf2surf)" % fs_home)


def _run(cmd, v=None, cwd=None, block=True):
    """Run a command

    Parameters
    ----------
    cmd: list of strings
        command that is submitted to subprocess.Popen.
    v : 0 | 1 | 2 | None
        verbosity level (0: nothing;  1: stderr;  2: stdout;  None: use
        _verbose module attribute)

    """
    if v is None:
        v = _verbose

    if v > 1:
        print("> COMMAND:")
        for line in cmd:
            print(repr(line))

    sp = subprocess.Popen(list(map(str, cmd)), cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if block:
        stdout, stderr = sp.communicate()

        if v > 1:
            print("\n> stdout:")
            print(stdout)

        if v > 0 and stderr:
            print('\n> stderr:')
            print(stderr)

        return stdout, stderr


def setup_mri(subject, subjects_dir=None, ico=4, block=False, redo=False):
    """Prepare an MRI for use in the mne-pipeline

    Performes the following steps:

     - runs mne_setup_mri
     - runs mne_setup_source_space
     - runs mne_watershed_bem
     - creates symlinks for watershed files in the bem directory
     - runs mne_setup_forward_model (see MNE manual section 3.7, p. 25)


    Parameters
    ----------
    subject : str
        Subject whose MRI should be processed.
    subjects_dir : None | str(path)
        Overrides `sys.environ['SUBJECTS_DIR']` if not `None`.
    ico : int
        `--ico` argument for `mne_setup_forward_model`.
    block : bool
        Block the Python interpreter until mne_setup_forward_model is
        finished.
    redo : bool
        Run the commands even if the target files already exist.


    .. note::
        The utility needs permission to access the MRI files. In case of a
        permission error, the following shell command can be used on the mri
        folder to set permissions appropriately::

            $ sudo chmod -R 7700 mri-folder

    """
    if subjects_dir is None:
        subjects_dir = os.environ['SUBJECTS_DIR']
        _sub_dir = None
    else:
        _sub_dir = os.environ.get('SUBJECTS_DIR', None)
        os.environ['SUBJECTS_DIR'] = subjects_dir

    bemdir = os.path.join(subjects_dir, subject, 'bem')

    # mne_setup_mri
    tgt = os.path.join(subjects_dir, subject, 'mri', 'T1', 'neuromag', 'sets',
                       'COR.fif')
    if redo or not os.path.exists(tgt):
        cmd = [get_bin('mne', 'mne_setup_mri'),
               '--subject', subject,
               '--mri', 'T1']
        _run(cmd)

    # mne_setup_source_space
    tgt = os.path.join(bemdir, '%s-ico-%i-src.fif' % (subject, ico))
    if redo or not os.path.exists(tgt):
        cmd = [get_bin('mne', 'mne_setup_source_space'),
               '--subject', subject,
               '--ico', ico]
        _run(cmd)

    # mne_watershed_bem
    cmd = [get_bin('mne', 'mne_watershed_bem'), '--subject', subject]
    if redo:
        cmd.append('--overwrite')
    _run(cmd)

    # symlinks (MNE-manual 3.6, p. 24)
    for name in ['inner_skull', 'outer_skull', 'outer_skin']:
        src = os.path.join('watershed', f'{subject}_{name}_surface')
        dest = os.path.join(bemdir, f'{name}.surf')
        if os.path.exists(dest):
            if os.path.islink(dest):
                abs_src = os.path.join(bemdir, src)
                if os.path.realpath(dest) == abs_src:
                    pass
                else:
                    logging.debug(f"replacing symlink: {dest!r}")
                    os.remove(dest)
                    os.symlink(src, dest)
                    # can raise an OSError: [Errno 13] Permission denied
            else:
                raise OSError(f"{dest!r} exists and is no symlink")
        else:
            logging.debug(f"creating symlink: {dest!r}")
            os.symlink(src, dest)

    # mne_setup_forward_model
    cmd = [get_bin('mne', "mne_setup_forward_model"),
           '--subject', subject,
           '--surf',
           '--ico', ico,
           '--homog']

    _run(cmd, block=block)  # -> creates a number of files in <mri_sdir>/bem
    if _sub_dir:
        os.environ['SUBJECTS_DIR'] = _sub_dir


def _run_mne_gui(name, cwd, modal, subject, subjects_dir):
    root = get_root('mne')
    env = os.environ.copy()
    env['MNE_ROOT'] = root
    if subjects_dir is not None:
        env['SUBJECTS_DIR'] = subjects_dir
    if subject:
        env['SUBJECT'] = subject

    setup_path = os.path.join(root, 'bin', 'mne_setup_sh').replace(' ', r'\ ')
    setup = f'. {setup_path}'

    cmd = os.path.join(root, 'bin', name).replace(' ', r'\ ')
    p = subprocess.Popen(setup + ';' + cmd, shell=True, cwd=cwd, env=env)

    if modal:
        print(f"Waiting for {name} to be closed...")
        p.wait()


def run_mne_analyze(fif_dir, subject=None, subjects_dir=None, modal=False):
    """Invoke mne_analyze (e.g., for manual coregistration)

    Parameters
    ----------
    fif_dir : str
        the directory containing the fiff files.
    subject : None | str
        The name of the MRI subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    modal : bool
        Causes the shell to block until mne_analyze is closed.


    Notes
    -----
    **Coregistration Procedure** (For more information see MNE-manual 3.10 &
    12.11):

    #. File > Load Surface: select the subject`s directory and "Inflated"
    #. File > Load digitizer data: select the fiff file
    #. View > Show viewer

       a. ``Options``: Switch cortical surface display off, make
          scalp transparent. ``Done``

    #. Adjust > Coordinate Alignment:

       a. set LAP, Nasion and RAP.
       b. ``Align using fiducials``.
       c. (``Omit``)
       d. Run ``ICP alignment`` with 20 steps
       e. ``Save default``

    this creates a file next to the raw file with the '-trans.fif' extension.

    """
    _run_mne_gui('mne_analyze', fif_dir, modal, subject, subjects_dir)


def run_mne_browse_raw(fif_dir, subject=None, subjects_dir=None, modal=False):
    """Invoke mne_browse_raw

    Parameters
    ----------
    fif_dir : str
        the directory containing the fiff files.
    subject : None | str
        The name of the MRI subject.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    modal : bool
        Causes the shell to block until mne_browse_raw is closed.
    """
    _run_mne_gui('mne_browse_raw', fif_dir, modal, subject, subjects_dir)


# freesurfer---


def _fs_hemis(arg):
    if arg == '*':
        return ['lh', 'rh']
    elif arg in ['lh', 'rh']:
        return [arg]
    else:
        raise ValueError(f"hemi has to be 'lh', 'rh', or '*' (no {arg!r})")


def _fs_subjects(arg, exclude=[], subjects_dir=None):
    if '*' in arg:
        subjects_dir = get_subjects_dir(subjects_dir)
        subjects = fnmatch.filter(os.listdir(subjects_dir), arg)
        subjects = list(filter(os.path.isdir, subjects))
        for subject in exclude:
            if subject in subjects:
                subjects.remove(subject)
    else:
        subjects = [arg]
    return subjects
