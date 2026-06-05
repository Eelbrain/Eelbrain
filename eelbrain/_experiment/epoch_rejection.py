# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Epoch (trial) rejection settings and the rejection-file input node.

``Pipeline.epoch_rejection`` is a ``{name: EpochRejection}`` dictionary selected
through the ``epoch_rejection`` state. This is trial-level rejection
(accept/reject individual epochs and per-epoch channel interpolation), distinct
from ICA-based artifact removal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import load
from .._data_obj import Dataset
from .configuration import Configuration
from .derivative_cache import Input, Request, file_fingerprint
from .epochs import PrimaryEpoch
from .pathing import rej_file_path


class EpochRejection(Configuration):
    """Base class for :attr:`Pipeline.epoch_rejection` settings.

    Parameters
    ----------
    interpolation
        Enable by-epoch channel interpolation from the rejection file.
    """
    DICT_ATTRS = ('interpolation',)

    def __init__(self, interpolation: bool = True):
        self.interpolation = interpolation


class ManualRejection(EpochRejection):
    """Rejection from a manually created selection file.

    The selection file is created through the epoch-rejection GUI or
    :meth:`Pipeline.make_epoch_rejection`.

    See Also
    --------
    Pipeline.epoch_rejection
    """


class RejectionInput(Input):
    name = 'epoch-rejection-input'

    def __init__(
            self,
            root: str | Path,
            epoch_rejection: dict[str, EpochRejection | None],
            epochs: dict[str, Any],
    ):
        self.root = Path(root)
        self.epoch_rejection = epoch_rejection
        self.epochs = epochs

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        rejection = self.epoch_rejection[ctx.state['epoch_rejection']]
        if rejection is None:
            return {'kind': 'none'}
        return {
            'rej': rejection,
            'file': file_fingerprint(ctx.root, self.path(ctx), 'rej-file'),
        }

    def path(self, ctx: Request) -> Path:
        epoch = self.epochs[ctx.state['epoch']]
        if not isinstance(epoch, PrimaryEpoch):
            raise RuntimeError(f"{epoch=}")
        return ctx.root / rej_file_path(ctx.state, epoch=epoch.name)

    def load(self, ctx: Request) -> Dataset:
        return load.unpickle(self.path(ctx))
