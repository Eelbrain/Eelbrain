# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Covariance derivatives.

These nodes depend on lower-level epoch/raw derivatives through
``ctx.load(...)``. They must not receive injected ``Pipeline.load_*`` methods.
"""
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import mne
import numpy

from .derivative_cache import Dependency, Derivative, DerivativeContext
from .events import EPOCHS_DATA
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
            log: logging.Logger,
    ):
        self.cov_name = cov_name
        self.cov = cov
        self.log = log
        self.name = cov_node_name(cov_name)
        self.cov.key = cov_name

    def _events_state(self, ctx: DerivativeContext) -> dict[str, Any]:
        if isinstance(self.cov, EpochCovariance):
            return {}
        return {'raw': ctx.get('raw')}

    def _rej_state(self, ctx: DerivativeContext) -> dict[str, Any]:
        if isinstance(self.cov, EpochCovariance):
            return {'epoch': self.cov.epoch}
        return {}

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        if isinstance(self.cov, EpochCovariance):
            return (Dependency(EPOCHS_DATA, options={
                'baseline': True,
                'ndvar': False,
                'add_bads': True,
                'reject': False,
                'samplingrate': None,
                'decim': 1,
                'pad': 0,
                'data_raw': False,
                'vardef': None,
                'data': 'sensor',
                'trigger_shift': True,
                'interpolate_bads': False,
                'epoch': self.cov.epoch,
            }),)
        return (raw_data_dependency(ctx, noise=True),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        return {
            'raw': ctx.get('raw'),
            'cov': vars(self.cov).copy(),
            'rej': ctx.get('rej'),
            'epoch': ctx.get('epoch'),
        }

    def build(self, ctx: DerivativeContext) -> mne.Covariance:
        if isinstance(self.cov, EpochCovariance):
            cov_path = self.path(ctx, mkdir=True)
            self.log.debug("Make cov-file %s", cov_path)
            log_path = cov_path.with_suffix('.info.txt')
            ds = ctx.load(EPOCHS_DATA, state={'epoch': self.cov.epoch}, options={
                'baseline': True,
                'ndvar': False,
                'add_bads': True,
                'reject': False,
                'samplingrate': None,
                'decim': 1,
                'pad': 0,
                'data_raw': False,
                'vardef': None,
                'data': 'sensor',
                'trigger_shift': True,
                'interpolate_bads': False,
            })
            return self.cov.make(ds['epochs'], log_path)
        raw = load_raw_dependency(ctx, ctx.get('raw'), noise=True)
        return self.cov.make(raw)

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> mne.Covariance:
        cov = mne.read_cov(path)
        if cov.data.dtype != 'float64':
            cov['data'] = cov['data'].astype(float)
        return cov

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: mne.Covariance,
    ) -> None:
        value.save(path, overwrite=True)
