"""Exceptions for Pipeline"""
from collections.abc import Sequence


class FileMissingError(Exception):
    """An input file is missing"""


class ICAMissingError(FileMissingError):
    """An ICA input file is missing"""


class FileDeficientError(Exception):
    """An input file is deficient"""


class ICAChannelsChangedError(Exception):
    """Bad channels changed since the ICA was estimated.

    Raised when launching the ICA component-selection GUI but the sensors in
    the current data no longer match those the ICA was estimated on.

    Parameters
    ----------
    path
        Path to the ICA file.
    bads_before
        Bad channels when the ICA was estimated (data channels that are absent
        from the ICA, since they were excluded from the decomposition).
    bads_after
        Bad channels in the current data.
    """

    def __init__(
            self,
            path: str,
            bads_before: Sequence[str],
            bads_after: Sequence[str],
    ):
        self.path = path
        self.bads_before = tuple(bads_before)
        self.bads_after = tuple(bads_after)
        super().__init__(f"Bad channels have changed since creating the ICA (before: {', '.join(self.bads_before) or 'none'}; now: {', '.join(self.bads_after) or 'none'})")
