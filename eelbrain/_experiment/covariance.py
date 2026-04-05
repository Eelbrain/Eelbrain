# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Covariance derivatives.

These nodes depend on lower-level epoch/raw derivatives through
``ctx.load(...)``. They must not receive injected ``Pipeline.load_*`` methods.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy

from .derivative_cache import Dependency, Derivative, Request
from .preprocessing import load_raw_dependency, raw_data_dependency


def cov_node_name(cov: str) -> str:
    return f'cov:{cov}'


@dataclass
class RawCovariance:
    method: str = 'empirical'
    key: str = field(init=False, default=None)

    def make(
            self,
            raw: mne.io.BaseRaw,
    ) -> mne.Covariance:
        if self.method == 'ad_hoc':
            return mne.cov.make_ad_hoc_cov(raw.info)
        return mne.compute_raw_covariance(raw, method=self.method)


@dataclass
class EpochCovariance:
    epoch: str
    method: str = 'empirical'
    keep_sample_mean: bool = True
    key: str = field(init=False, default=None)

    def make(
            self,
            epochs: mne.Epochs,
            log_path: Path,
    ) -> mne.Covariance:
        method = 'empirical' if self.method == 'best' else self.method
        cov = mne.compute_covariance(epochs, self.keep_sample_mean, method=method)

        if self.method == 'best':
            if mne.pick_types(epochs.info, meg='grad', eeg=True, ref_meg=False).size:
                raise NotImplementedError(f"cov={self.key!r}: 'best' regularization is not implemented for EEG or gradiometer sensors; use a different setting for cov.")
            elif epochs is None:
                raise NotImplementedError(f"cov={self.key!r}: 'best' regularization is not implemented for covariance based on raw data; use a different setting for cov.")
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
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw', 'epoch', 'cov', 'rej')
    cache_suffix = '-cov.fif'

    def __init__(
            self,
            cov_name: str,
            cov: RawCovariance | EpochCovariance,
    ):
        self.cov_name = cov_name
        self.cov = cov
        self.name = cov_node_name(cov_name)
        self.cov.key = cov_name

    def _events_state(self, ctx: Request) -> dict[str, Any]:
        if isinstance(self.cov, EpochCovariance):
            return {}
        return {'raw': ctx.state['raw']}

    def _rej_state(self, ctx: Request) -> dict[str, Any]:
        if isinstance(self.cov, EpochCovariance):
            return {'epoch': self.cov.epoch}
        return {}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        if isinstance(self.cov, EpochCovariance):
            return (Dependency('epochs', state={'epoch': self.cov.epoch}, options={
                'baseline': True,
                'add_bads': True,
                'ndvar': False,
                'data': 'sensor',
                'data_raw': False,
                'reject': False,
                'samplingrate': None,
                'decim': 1,
                'pad': 0,
                'trigger_shift': True,
                'interpolate_bads': False,
            }),)
        return (raw_data_dependency(ctx, noise=True),)

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'raw': ctx.state['raw'],
            'cov': vars(self.cov).copy(),
            'rej': ctx.state['rej'],
            'epoch': ctx.state['epoch'],
        }

    def build(self, ctx: Request) -> mne.Covariance:
        if isinstance(self.cov, EpochCovariance):
            cov_path = self.path(ctx)
            cov_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = cov_path.with_suffix('.info.txt')
            ds = ctx.load('epochs', state={'epoch': self.cov.epoch}, options={
                'baseline': True,
                'add_bads': True,
                'ndvar': False,
                'data': 'sensor',
                'data_raw': False,
                'reject': False,
                'samplingrate': None,
                'decim': 1,
                'pad': 0,
                'trigger_shift': True,
                'interpolate_bads': False,
            })
            epochs = ds['epochs']
            return self.cov.make(epochs, log_path)
        raw = load_raw_dependency(ctx, ctx.state['raw'], noise=True)
        return self.cov.make(raw)

    def load(
            self,
            ctx: Request,
            path: Path) -> mne.Covariance:
        cov = mne.read_cov(path)
        if cov.data.dtype != 'float64':
            cov['data'] = cov['data'].astype(float)
        return cov

    def save(
            self,
            ctx: Request,
            path: Path,
            value: mne.Covariance,
    ) -> None:
        value.save(path, overwrite=True)
