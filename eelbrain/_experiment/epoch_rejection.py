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

import numpy as np

from .. import load, save
from .._data_obj import Dataset
from .._exceptions import ConfigurationError
from .._data_obj import Datalist
from .._info import INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS, INTERPOLATE_WINDOWS_MAX
from .._meeg._channel_model import ChannelModel
from .._meeg.base import new_rejection_ds
from .configuration import Configuration
from .derivative_cache import CachePolicy, Dependency, Derivative, Input, Request, file_fingerprint
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


class ChannelModelRejection(EpochRejection):
    """Automatically generated rejection using a :class:`ChannelModel` (EEG only).

    A :class:`~eelbrain._meeg.ChannelModel` is fit to predict each EEG sensor
    from the others; in each epoch, channels that are poorly predicted (error
    above ``score_threshold``) are considered bad. An epoch with more than
    ``max_interpolate`` bad channels is rejected; otherwise its bad channels are
    marked for interpolation. The rejection file is generated and cached
    automatically (no manual selection).

    Time-resolved (windowed) detection with
    :meth:`ChannelModel.find_bad_windows` is used for long epochs: always for
    variable-length epochs (loaded as a list of epochs), and for equal-length
    epochs longer than ``continuous`` seconds. Each channel is then interpolated
    only over the time window in which it is bad, and such epochs are never
    rejected wholesale. Shorter equal-length epochs use whole-epoch detection
    with :meth:`ChannelModel.score`.

    Parameters
    ----------
    max_interpolate
        Reject an epoch when it has more than this many bad channels; with this
        many or fewer, mark the bad channels for interpolation instead. For long
        epochs this caps the number of channels interpolated simultaneously.
    fit_threshold
        Amplitude threshold for excluding epochs from fitting the model (see
        :meth:`ChannelModel.fit`).
    score_threshold
        Error threshold above which a channel is considered bad in an epoch (see
        :meth:`ChannelModel.score`).
    raw
        ``raw`` pipeline setting providing the data to fit the model. The
        default (``None``) uses the same ``raw`` as for scoring.
    interpolation
        Apply the by-epoch channel interpolation when loading epochs (default
        ``True``).
    continuous
        Duration threshold in seconds: equal-length epochs longer than this use
        time-resolved (windowed) detection instead of whole-epoch detection
        (default 5). Variable-length epochs always use windowed detection.
    window, hop, min_duration, merge_gap
        Time-resolved detection parameters for long epochs (see
        :meth:`ChannelModel.find_bad_windows`).
    model, alpha, epsilon
        :class:`ChannelModel` parameters.

    See Also
    --------
    Pipeline.epoch_rejection
    """
    DICT_ATTRS = ('interpolation', 'fit_threshold', 'score_threshold', 'max_interpolate', 'raw', 'continuous', 'window', 'hop', 'min_duration', 'merge_gap', 'model', 'alpha', 'epsilon')

    def __init__(
            self,
            max_interpolate: int = 5,
            fit_threshold: float = 50e-6,
            score_threshold: float = 50e-6,
            raw: str | None = None,
            interpolation: bool = True,
            continuous: float = 5.,
            window: float = 1.0,
            hop: float = 0.5,
            min_duration: float = 0.1,
            merge_gap: float | None = None,
            model: str = 'huber',
            alpha: float = 1e-4,
            epsilon: float = 1.35,
    ):
        super().__init__(interpolation)
        self.max_interpolate = max_interpolate
        self.fit_threshold = fit_threshold
        self.score_threshold = score_threshold
        self.raw = raw
        self.continuous = continuous
        self.window = window
        self.hop = hop
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.model = model
        self.alpha = alpha
        self.epsilon = epsilon


class RejectionInput(Input):
    name = 'epoch-rejection-input'
    key_fields = ('subject', 'session', 'acquisition', 'run', 'raw', 'epoch', 'epoch_rejection')

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
            'file': file_fingerprint(ctx.root, self.path(ctx)),
        }

    def path(self, ctx: Request) -> Path:
        epoch = self.epochs[ctx.state['epoch']]
        if not isinstance(epoch, PrimaryEpoch):
            raise RuntimeError(f"{epoch=}")
        return ctx.root / rej_file_path(ctx.state, epoch=epoch.name, datatype=ctx.datatype)

    def load(self, ctx: Request) -> Dataset:
        return load.unpickle(self.path(ctx))


class ChannelModelRejectionDerivative(Derivative[Dataset]):
    """Cached rejection file generated by a :class:`ChannelModelRejection`."""
    name = 'epoch-rejection-channel-model'
    key_fields = ('subject', 'session', 'acquisition', 'run', 'raw', 'epoch', 'epoch_rejection')
    # Always detect artifacts on the original reference because re-referencing transfers noise
    fixed_state = {'reference': ''}
    cache_policy = CachePolicy.REQUIRED
    cache_suffix = '.pickle'
    # Options for loading epochs to fit/score the model.
    _EPOCH_OPTIONS = {'reject': False, 'ndvar': True, 'data': 'sensor'}

    def __init__(self, epochs: dict[str, Any], epoch_rejection: dict[str, EpochRejection | None]):
        self.epochs = epochs
        self.epoch_rejection = epoch_rejection

    def _separate_fit_raw(self, ctx: Request) -> str | None:
        rej = self.epoch_rejection[ctx.state['epoch_rejection']]
        if rej.raw and rej.raw != ctx.state['raw']:
            return rej.raw
        return None

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        if not isinstance(epoch, PrimaryEpoch):
            raise RuntimeError(f"{epoch=}")
        deps = [Dependency('epochs', label='score-epochs', options=self._EPOCH_OPTIONS)]
        fit_raw = self._separate_fit_raw(ctx)
        if fit_raw is not None:
            deps.append(Dependency('epochs', label='fit-epochs', state={'raw': fit_raw}, options=self._EPOCH_OPTIONS))
        return tuple(deps)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'epoch_rejection': self.epoch_rejection[ctx.state['epoch_rejection']]}

    def build(self, ctx: Request) -> Dataset:
        rej = self.epoch_rejection[ctx.state['epoch_rejection']]
        score_ds = ctx.load('score-epochs')
        if 'eeg' not in score_ds:
            raise ConfigurationError(f"epoch_rejection={ctx.state['epoch_rejection']!r}: ChannelModelRejection requires EEG data, but {ctx.state['subject']}/{ctx.state['epoch']} has none")
        eeg = score_ds['eeg']
        if self._separate_fit_raw(ctx) is not None:
            fit_eeg = ctx.load('fit-epochs')['eeg']
        else:
            fit_eeg = eeg
        model = ChannelModel(rej.model, alpha=rej.alpha, epsilon=rej.epsilon)
        model.fit(fit_eeg, threshold=rej.fit_threshold)

        # use time-resolved detection for variable-length epochs and for
        # equal-length epochs longer than ``continuous`` seconds
        if isinstance(eeg, Datalist):
            continuous = True
        else:
            continuous = (eeg.time.tstop - eeg.time.tmin) > rej.continuous
        if continuous:
            rej_ds = new_rejection_ds(score_ds, windows=True)
            rej_ds.info[INTERPOLATE_WINDOWS_MAX] = rej.max_interpolate
            rej_ds[INTERPOLATE_WINDOWS] = model.find_bad_windows(eeg, threshold=rej.score_threshold, max_exclude=rej.max_interpolate, window=rej.window, hop=rej.hop, min_duration=rej.min_duration, merge_gap=rej.merge_gap)
            return rej_ds

        scores = model.score(eeg, threshold=rej.score_threshold, max_exclude=rej.max_interpolate + 1)
        rej_ds = new_rejection_ds(score_ds, interpolation=True)
        names = scores.get_dim('sensor').names
        score_data = scores.get_data(('case', 'sensor'))
        accept = rej_ds['accept']
        tag = rej_ds['rej_tag']
        interpolate = rej_ds[INTERPOLATE_CHANNELS]
        for i in range(score_ds.n_cases):
            bad = [names[j] for j in np.flatnonzero(score_data[i] > rej.score_threshold)]
            if len(bad) > rej.max_interpolate:
                accept[i] = False
                tag[i] = 'channel-model'
            else:
                interpolate[i] = bad
        return rej_ds

    def load(self, ctx: Request, path: Path) -> Dataset:
        return load.unpickle(path)

    def save(self, ctx: Request, path: Path, value: Dataset) -> None:
        save.pickle(value, path)
