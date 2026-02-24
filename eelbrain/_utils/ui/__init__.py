"""Deploy UI elements depending on the current environment."""
import os

USE_WX = True

_progress_monitors = []


def get_ui():
    global USE_WX

    if USE_WX:
        try:
            from . import wx_ui
            from ..._wxgui import get_app
            get_app()
            return wx_ui
        except ImportError:
            USE_WX = False

    from . import tk_ui
    return tk_ui


def ask_saveas(title="Save File", message="Please Pick a File Name",
               filetypes=(("Pickled Python Objects (*.pickle)", '*.pickle'),),
               defaultDir=None, defaultFile=False):
    """Display a save-as dialog

    Parameters
    ----------
    title : str
        Title of the dialog.
    message : str
        Message in the dialog.
    filetypes : sequence of tuples
        Sequence of (label, pattern) tuples.
    defaultDir : None | str
        Default directory to save at.
    defaultFile : None | str
        Default file name.

    Returns
    -------
    result : False | str
        The path as str if the user selects a dialog, otherwise ``False``.
    """
    result = get_ui().ask_saveas(title, message, filetypes, defaultDir,
                                 defaultFile)
    return result


def ask_dir(title="Select Folder",
            message="Please Pick a Folder",
            must_exist=True):
    return get_ui().ask_dir(title, message, must_exist)


def ask_file(title="Pick File",
             message="Please Pick a File",
             filetypes=[("All files", '*')],
             directory='',
             mult=False):
    """
    Ask for an existing file.

    Parameters
    ----------
    title, message : str
        Title and message for the dialog.
    filetypes : sequence of tuples
        Sequence of (label, pattern) tuples.
    directory : None | str
        Path to initial directory.
    mult : bool
        Allow selecting multiple files.

    Returns
    -------
    paths : False | str | list
        If the user cancels: False. Otherwise, if mult=False a single path, and
        if mult=True a list of paths.
    """
    return get_ui().ask_file(title, message, filetypes, directory, mult)


def ask(title="Overwrite File?",
        message="Duplicate filename. Do you want to overwrite?",
        cancel=False,
        default=True,  # True=YES, False=NO, None=Nothing
        ):
    return get_ui().ask(title, message, cancel, default)


def ask_color(default=(0, 0, 0)):
    return get_ui().ask_color(default)


def ask_str(msg, title, default=''):
    return get_ui().ask_str(msg, title, default)


def kill_progress_monitors():
    while len(_progress_monitors) > 0:
        p = _progress_monitors.pop()
        p.terminate()


def message(title, message="", icon='i'):
    return get_ui().message(title, message, icon)


def copy_file(path):
    return get_ui().copy_file(path)


def copy_text(text):
    return get_ui().copy_text(text)


def test_targetpath(path, cancel=True):
    """Test whether ``path`` is okay to write to

    If the directory does not exist, the user is asked whether it should be
    created.

    Parameters
    ----------
    path : str
        Path to test.
    cancel : bool
        Add a cancel button. If clicked, a KeyboardInterrupt Exception is
        raised.

    Returns
    -------
    success : bool
        True if path is a valid path to write to, False otherwise.
    """
    if not path:
        return False

    dirname = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(dirname):
        msg = f"The directory {dirname!r} does not exist. Should it be created?"
        answer = ask("Create Directory?", msg, cancel=cancel)
        if answer:
            os.makedirs(dirname)
        elif answer is None:  # cancel
            err = f"User canceled because the directory {dirname!r} does not exist"
            raise KeyboardInterrupt(err)

    return os.path.exists(dirname)
