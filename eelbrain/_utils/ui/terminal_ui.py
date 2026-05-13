"""Terminal-based implementation of the Eelbrain ui functions."""
import os


def ask_saveas(title, messagefiletypes, defaultDir, defaultFile):
    msg = f"{title} ({message}): "
    path = input(msg)
    path = os.path.expanduser(path)

    dirname = os.path.split(path)[0]
    if os.path.exists(path):
        if ask(title="File Exists. Overwrite?",
               message=repr(path)):
            return path
        else:
            return False
    elif os.path.exists(dirname):
        return path
    else:
        if ask(title="Directory does not exist. Create?",
               message=repr(dirname)):
            os.makedirs(dirname)
            return path
        else:
            return False


def ask_dir(title="Select Folder", message="Please Pick a Folder",
            must_exist=True):
    msg = f"{title} ({message}): "
    path = input(msg)
    path = os.path.expanduser(path)
    if os.path.exists(path) and os.path.isdir(path):
        return path
    else:
        return False


def ask_file(title, message, filetypes, directory, mult):
    msg = f"{title} ({message}): "
    path = input(msg)
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return path
    else:
        return False


def ask(title="Overwrite File?",
        message="Duplicate filename. Do you want to overwrite?",
        cancel=False, default=True):
    print(title)
    print(message)
    c = ''
    while c not in ['y', 'n', 'c']:
        c = input("([y]es / [n]o / [c]ancel)")
    if c == 'y':
        return True
    elif c == 'n':
        return False
    else:
        return None


def ask_color(parent=None, default=None):
    c = input('Color = ')
    return eval(c)


def message(title, message=None, icon='i'):
    if icon:
        title = f"{icon}: {title}"
    print(title)
    if message:
        print(message)


def copy_file(path):
    raise NotImplementedError


def copy_text(text):
    raise NotImplementedError
