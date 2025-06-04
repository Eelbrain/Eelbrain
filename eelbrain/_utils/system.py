# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
import sys

IS_OSX = sys.platform == 'darwin'
IS_WINDOWS = os.name == 'nt'

if IS_OSX:
    from .macos import user_activity  # noqa: F401
else:
    from .null_os import user_activity  # noqa: F401


def restore_main_spec():
    """On windows, running a multiprocessing job seems to sometimes remove this attribute"""
    if IS_WINDOWS:
        main_module = sys.modules['__main__']
        if not hasattr(main_module, '__spec__'):
            main_module.__spec__ = None
