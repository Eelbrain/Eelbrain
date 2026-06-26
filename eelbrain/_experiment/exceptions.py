"""Exceptions for Pipeline"""


class FileMissingError(Exception):
    "An input file is missing"


class FileDeficientError(Exception):
    "An input file is deficient"


class ICAChannelsChangedError(Exception):
    """Bad channels changed since the ICA was estimated.

    Raised when launching the ICA component-selection GUI but the sensors in
    the current data no longer match those the ICA was estimated on.
    """

    def __init__(self, path: str):
        self.path = path
        super().__init__("Bad channels have changed since creating the ICA")
