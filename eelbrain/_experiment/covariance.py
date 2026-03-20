# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Covariance matrix computation and cache nodes."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy

from .derivative_cache import Dependency, Derivative, DerivativeContext


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
            log_path: str,
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
    name = 'cov'
    path_template = 'cov-file'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw', 'epoch', 'cov', 'rej')

    def _events_state(self, ctx: DerivativeContext) -> dict[str, Any]:
        cov = ctx.pipeline._covs[ctx.get('cov')]
        if isinstance(cov, EpochCovariance):
            return {}
        return {'raw': ctx.get('raw')}

    def _rej_state(self, ctx: DerivativeContext) -> dict[str, Any]:
        cov = ctx.pipeline._covs[ctx.get('cov')]
        if isinstance(cov, EpochCovariance):
            return {'epoch': cov.epoch}
        return {}

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        cov = ctx.pipeline._covs[ctx.get('cov')]
        if isinstance(cov, EpochCovariance):
            return (
                Dependency('events', state=self._events_state),
                Dependency('raw-input-bads'),
                Dependency('rej-input', state=self._rej_state),
            )
        return (Dependency('noise-raw-input'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            cov = p._covs[p.get('cov')]
            return {
                'raw': p.get('raw'),
                'cov': vars(cov).copy(),
                'rej': p.get('rej'),
                'epoch': p.get('epoch'),
            }

    def build(self, ctx: DerivativeContext) -> mne.Covariance:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            p._log.debug("Make cov-file %s", p.get('cov-file'))
            cov = p._covs[p.get('cov')]
            if isinstance(cov, EpochCovariance):
                log_path = p.get('cov-info-file', mkdir=True)
                ds = p.load_epochs(None, True, False, decim=1, epoch=cov.epoch)
                return cov.make(ds['epochs'], log_path)
            raw = p.load_raw(noise=True)
            return cov.make(raw)

    def load(self, ctx: DerivativeContext, path: str) -> mne.Covariance:
        cov = mne.read_cov(path)
        if cov.data.dtype != 'float64':
            cov['data'] = cov['data'].astype(float)
        return cov

    def save(self, ctx: DerivativeContext, path: str, value: mne.Covariance) -> None:
        value.save(path, overwrite=True)
