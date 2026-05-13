"""Basic GUI methods using Tkinter

http://tkinter.unpythonic.net/wiki/tkFileDialog
"""
from tkinter import Tk
import tkinter.filedialog
from tkinter.filedialog import Open, SaveAs
import tkinter.messagebox
import tkinter.simpledialog


root = Tk()  # initialize application
root.withdraw()  # hide root window


def ask_saveas(title, message, filetypes, defaultDir=None, defaultFile=None):
    dlg = SaveAs(title=title, message=message, filetypes=filetypes)
    filename = dlg.show()
    return filename


def ask_dir(title="Select Folder", message="Please Pick a Folder",
            must_exist=True):
    return tkinter.filedialog.askdirectory(title=title, mustexist=must_exist)


def ask_file(title, message, filetypes, directory, mult):
    dlg = Open(title=title, filetypes=filetypes, initialdir=directory,
               multiple=mult)
    out = dlg.show()
    return out


def ask(title="Overwrite File?",
        message="Duplicate filename. Do you want to overwrite?",
        cancel=False, default=True,  # True=YES, False=NO, None=Nothing
        ):
    return tkinter.messagebox.askyesno(title, message)


def ask_str(message, title, default=''):
    return tkinter.simpledialog.askstring(title, message, initialvalue=default)


def copy_file(path):
    raise NotImplementedError


def copy_text(text):
    # http://stackoverflow.com/a/4203897/166700
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.destroy()


def message(title, message="", icon='i'):
    if icon in 'i?':
        tkinter.messagebox.showinfo(title, message)
    elif icon == '!':
        tkinter.messagebox.showwarning(title, message)
    elif icon == 'error':
        tkinter.messagebox.showerror(title, message)
    else:
        raise ValueError(f"Invalid icon argument: {icon!r}")


def show_help(obj):
    print(getattr(obj, '__doc__', 'no docstring'))
