# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Covariance derivatives.

These nodes depend on lower-level epoch/raw derivatives through
``ctx.load(...)``. They must not receive injected ``Pipeline.load_*`` methods.
"""
from pathlib import Path
from typing import Any

import mne
import numpy

from .._data_obj import Datalist
from .configuration import Configuration
from .derivative_cache import Dependency, Derivative, Request
from .preprocessing import Reference, canonical_recording, raw_node_name


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

    def __init__(
            self,
            epoch: str,
            method: str = 'empirical',
            keep_sample_mean: bool = True,
    ):
        self.epoch = epoch
        self.method = method
        self.keep_sample_mean = keep_sample_mean

    def make(self, epochs_list: list[mne.BaseEpochs], log_path: Path) -> mne.Covariance:
        """Compute the covariance from one or more :class:`mne.Epochs` objects (variable-length epochs arrive as one object per epoch)."""
        if len(epochs_list) > 1:
            if not self.keep_sample_mean:
                raise NotImplementedError(f"cov={self.name!r}: keep_sample_mean=False is not implemented for variable-length epochs (MNE would subtract a separate mean for each epoch)")
            if self.method == 'best':
                raise NotImplementedError(f"cov={self.name!r}: method={self.method!r} for variable-length epochs (requires averaging epochs)")
        # MNE expects zero mean data
        for epochs in epochs_list:
            epochs.apply_baseline((None, None))
        info = epochs_list[0].info
        # We need a single Epochs object
        if len(epochs_list) == 1:
            epochs = epochs_list[0]
        else:
            for epochs in epochs_list[1:]:
                if epochs.ch_names != info['ch_names'] or epochs.info['bads'] != info['bads']:
                    raise ValueError(f"cov={self.name!r}: variable-length epochs must have the same channels and bad channels")
                if (epochs.info['dev_head_t'] is None) != (info['dev_head_t'] is None) or (info['dev_head_t'] is not None and not numpy.allclose(epochs.info['dev_head_t']['trans'], info['dev_head_t']['trans'])):
                    raise ValueError(f"cov={self.name!r}: variable-length epochs must have the same head position (dev_head_t)")
            data = numpy.concatenate([epochs.get_data() for epochs in epochs_list], axis=-1)
            epochs = mne.EpochsArray(data, info, baseline=None, proj=False, verbose=False)

        method = 'empirical' if self.method == 'best' else self.method
        cov = mne.compute_covariance(epochs, self.keep_sample_mean, method=method)

        if self.method == 'best':
            if mne.pick_types(epochs.info, meg='grad', eeg=True, ref_meg=False).size:
                raise NotImplementedError(f"cov={self.name!r}: 'best' regularization is not implemented for EEG or gradiometer sensors; use a different setting for cov.")
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
    cache_suffix = '-cov.fif'
    # source localization handles EEG referencing internally
    fixed_state = {'reference': ''}

    def override_key_fields(self, ctx: Request) -> tuple[str, ...]:
        # ``epoch_rejection`` only affects an epoch-based covariance (which loads
        # rejected epochs); a noise (raw) covariance does not depend on it.
        fields = ['subject', 'session', 'acquisition', 'raw', 'cov']
        if isinstance(self._covs[ctx.state['cov']], EpochCovariance):
            fields.append('epoch_rejection')
        return tuple(fields)

    # Fixed options used when loading epochs for covariance estimation.
    # Declared on both the Dependency edge and the build() load call so that
    # cache validation and the actual load request stay in sync.

    def __init__(self, covs: dict[str, RawCovariance | EpochCovariance], raw, references: dict[str, Reference | None], recordings: frozenset[tuple[str, str, str, str, str]]):
        self._covs = covs
        self.raw = raw
        self._references = references
        self._recordings = recordings

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        cov = self._covs[ctx.state['cov']]
        if isinstance(cov, EpochCovariance):
            return (Dependency('epochs', state={'epoch': cov.epoch}, options={'ndvar': False, 'decim': 1}),)
        elif isinstance(cov, RawCovariance):
            # Only the noise recording's sensor data is used; pin a canonical
            # recording so identity does not depend on the ambient task/run.
            recording = canonical_recording(self._recordings, ctx.state['subject'], ctx.state.get('session'), ctx.state.get('acquisition'))
            raw_state = {'task': recording[0], 'run': recording[1]} if recording else None
            return (Dependency(raw_node_name(ctx.state['raw']), options={'noise': True}, label='raw', state=raw_state),)
        raise NotImplementedError(f"{cov=}")

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'cov': self._covs[ctx.state['cov']],
            'source_reference_add': self._references['average'].add,
        }

    def build(self, ctx: Request) -> mne.Covariance:
        cov = self._covs[ctx.state['cov']]
        reference = self._references['average']
        montage = self.raw.root_source_pipe(ctx.state['raw']).montage
        if isinstance(cov, EpochCovariance):
            cov_path = self.path(ctx)
            cov_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = cov_path.with_suffix('.info.txt')
            epochs_value = ctx.load('epochs')['epochs']
            epochs_list = list(epochs_value) if isinstance(epochs_value, Datalist) else [epochs_value]
            for epochs in epochs_list:
                reference._prepare_source_data(epochs, montage)
            return cov.make(epochs_list, log_path)
        elif isinstance(cov, RawCovariance):
            raw = ctx.load('raw')
            if reference.add:
                raw.load_data()
            reference._prepare_source_data(raw, montage)
            return cov.make(raw)
        raise NotImplementedError(f"{cov=}")

    def load(self, ctx: Request, path: Path) -> mne.Covariance:
        cov = mne.read_cov(path)
        if cov.data.dtype != 'float64':
            cov['data'] = cov['data'].astype(float)
        return cov

    def save(self, ctx: Request, path: Path, value: mne.Covariance) -> None:
        value.save(path, overwrite=True)
