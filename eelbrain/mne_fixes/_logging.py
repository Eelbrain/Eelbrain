"""Helper functions for dealing with logging

By default, mne's logger has a highly verbose stream handler logging to the
screen. Here this screen handler is set to a higher level (WARNING), but only
if Eelbrain is the primary import (i.e., it is imported before mne) and if the
user has not stored another configuration.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import functools
from logging import WARNING, FileHandler, Formatter, StreamHandler, getLogger
import sys
import warnings
first_mne_import = 'mne' not in sys.modules

import mne

from .._utils import log_level


def reset_logger(logger):
    # lower mne's screen logging level, but defer to user-defined setting
    level = log_level(mne.get_config('MNE_LOGGING_LEVEL', WARNING))
    formatter = Formatter("%(levelname)-8s %(name)s:%(message)s")
    for h in logger.handlers:
        if isinstance(h, StreamHandler):
            h.setFormatter(formatter)
            if h.level < level:
                h.setLevel(level)


if first_mne_import:
    reset_logger(mne.utils.logger)


class CaptureLog:
    """Context to capture log from a specific logger and write it to a file

    Parameters
    ----------
    filename : str
        Where to write the log file.
    mode : str
        Mode for opening the log file (default 'w').
    name : str
        Name of the logger from which to capture (default 'mne').
    """

    def __init__(self, filename, mode='w', logger='mne', level='debug'):
        self.logger = logger
        self.level = log_level(level)
        self.handler = FileHandler(filename, mode)
        self.handler.setLevel(self.level)
        self.handler.setFormatter(Formatter("%(levelname)-8s :%(message)s"))
        self._old_level = None

    def __enter__(self):
        logger = getLogger(self.logger)
        logger.addHandler(self.handler)
        if logger.level == 0 or logger.level > self.level:
            self._old_level = logger.level
            logger.setLevel(self.level)
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.close()
        logger = getLogger(self.logger)
        logger.removeHandler(self.handler)
        if self._old_level is not None:
            logger.setLevel(self._old_level)


def suppress_mne_warning(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'The measurement information indicates a low-pass frequency', RuntimeWarning, 'mne')
            return func(*args, **kwargs)
    return new_func
