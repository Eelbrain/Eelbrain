# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Covariance derivatives.

These nodes depend on lower-level epoch/raw derivatives through
``ctx.load(...)``. They must not receive injected ``Pipeline.load_*`` methods.
"""
from pathlib import Path
from typing import Any

import mne
import numpy

from .configuration import Configuration
from .derivative_cache import Dependency, Derivative, Request
from .preprocessing import raw_node_name


class RawCovariance(Configuration):
    DICT_ATTRS = ('method',)

    def __init__(self, method: str = 'empirical'):
        self.method = method

    def make(self, raw: mne.io.BaseRaw) -> mne.Covariance:
        if self.method == 'ad_hoc':
            return mne.cov.make_ad_hoc_cov(raw.info)
        return mne.compute_raw_covariance(raw, method=self.method)


class EpochCovariance(Configuration):
    DICT_ATTRS = ('epoch', 'method', 'keep_sample_mean')

    def __init__(self, epoch: str, method: str = 'empirical', keep_sample_mean: bool = True):
        self.epoch = epoch
        self.method = method
        self.keep_sample_mean = keep_sample_mean

    def make(self, epochs: mne.Epochs, log_path: Path) -> mne.Covariance:
        method = 'empirical' if self.method == 'best' else self.method
        cov = mne.compute_covariance(epochs, self.keep_sample_mean, method=method)

        if self.method == 'best':
            if mne.pick_types(epochs.info, meg='grad', eeg=True, ref_meg=False).size:
                raise NotImplementedError(f"cov={self.name!r}: 'best' regularization is not implemented for EEG or gradiometer sensors; use a different setting for cov.")
            elif epochs is None:
                raise NotImplementedError(f"cov={self.name!r}: 'best' regularization is not implemented for covariance based on raw data; use a different setting for cov.")
            reg_vs = numpy.arange(0, 0.21, 0.01)
            covs = [mne.cov.regularize(cov, epochs.info, mag=v, rank=None) for v in reg_vs]

            # compute whitened global field power
            evoked = epochs.average()
            picks = mne.pick_types(evoked.info, meg='mag', ref_meg=False)
            gfps = [mne.whiten_evoked(evoked, cov, picks).data.std(0) for cov in covs]
            vs = [gfp.mean() for gfp in gfps]
            i = numpy.argmin(numpy.abs(1 - numpy.array(vs)))
            cov = covs[i]
            values = '\n'.join([f"{reg:.2f}: {gfp}" for reg, gfp in zip(reg_vs, gfps)])
            Path(log_path).write_text(f'Picked mag={reg_vs[i]}\nGFP:\n{values}')

        return cov


class CovDerivative(Derivative[mne.Covariance]):
    name = 'cov'
    key_fields = ('subject', 'session', 'raw', 'cov')
    cache_suffix = '-cov.fif'

    # Fixed options used when loading epochs for covariance estimation.
    # Declared on both the Dependency edge and the build() load call so that
    # cache validation and the actual load request stay in sync.
    _EPOCH_COV_OPTIONS = {
        'baseline': True,
        'ndvar': False,
        'data': 'sensor',
        'reject': False,
        'samplingrate': None,
        'decim': 1,
        'pad': 0,
        'trigger_shift': True,
        'interpolate_bads': False,
    }

    def __init__(self, covs: dict[str, RawCovariance | EpochCovariance]):
        self._covs = covs

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        cov = self._covs[ctx.state['cov']]
        if isinstance(cov, EpochCovariance):
            return (Dependency('epochs', state={'epoch': cov.epoch}, options=self._EPOCH_COV_OPTIONS),)
        elif isinstance(cov, RawCovariance):
            return (Dependency(raw_node_name(ctx.state['raw']), options={'noise': True}, label='raw'),)
        raise NotImplementedError(f"{cov=}")

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        cov = self._covs[ctx.state['cov']]
        return cov._as_dict()

    def build(self, ctx: Request) -> mne.Covariance:
        cov = self._covs[ctx.state['cov']]
        if isinstance(cov, EpochCovariance):
            cov_path = self.path(ctx)
            cov_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = cov_path.with_suffix('.info.txt')
            ds = ctx.load('epochs')
            return cov.make(ds['epochs'], log_path)
        elif isinstance(cov, RawCovariance):
            raw = ctx.load('raw')
            return cov.make(raw)
        raise NotImplementedError(f"{cov=}")

    def load(self, ctx: Request, path: Path) -> mne.Covariance:
        cov = mne.read_cov(path)
        if cov.data.dtype != 'float64':
            cov['data'] = cov['data'].astype(float)
        return cov

    def save(self, ctx: Request, path: Path, value: mne.Covariance) -> None:
        value.save(path, overwrite=True)
