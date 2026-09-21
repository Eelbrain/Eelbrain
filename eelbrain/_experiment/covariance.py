# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Covariance derivatives.

These nodes depend on lower-level epoch/raw derivatives through
``ctx.load(...)``. They must not receive injected ``Pipeline.load_*`` methods.
"""
from pathlib import Path
from typing import Any

import mne
from mne._fiff.pick import _picks_by_type
import numpy

from .._data_obj import Datalist
from .._text import enumeration
from .configuration import Configuration
from .derivative_cache import Dependency, Derivative, Request
from .preprocessing import Reference, canonical_recording, raw_node_name


# Key under which :meth:`Covariance.make` records the conditioning of the covariance it
# built. ``mne.Covariance`` is a :class:`dict`, and the key is dropped when the
# covariance is written to FIFF, so it never reaches consumers of the cached artifact:
# it only carries the values from :meth:`CovDerivative.build` to
# :meth:`CovDerivative.artifact_metadata`, which stores them in the manifest.
CONDITION_INFO = 'eelbrain_condition_info'


def _block_spectra(
        cov: mne.Covariance,
        info: mne.Info,
) -> dict[str, numpy.ndarray]:
    """Retained eigenvalues per regularization block, in descending order.

    The blocks are the ones :func:`mne.cov.regularize` regularizes with ``rank=None``:
    magnetometers and gradiometers jointly (``'meg'``) after Maxwell filtering, which
    makes them linearly dependent, and separately otherwise; EEG on its own. Each block
    is decomposed on its own through :func:`mne.cov.prepare_noise_cov`, which zeroes
    the eigenvalues outside the estimated rank, the same way :func:`mne.cov.regularize`
    finds the subspace it regularizes.

    Parameters
    ----------
    cov
        Covariance to decompose.
    info
        Measurement info, restricted to the channels of ``cov`` and in their order.
    """
    out = {}
    for block, picks in _picks_by_type(info, meg_combined='auto', ref_meg=False, exclude=[]):
        block_info = mne.pick_info(info, picks, verbose='error')
        block_cov = mne.pick_channels_cov(cov, block_info['ch_names'], exclude=[], verbose='error')
        eig = mne.cov.prepare_noise_cov(block_cov, block_info, block_info['ch_names'], rank=None, verbose='error')['eig']
        retained = numpy.sort(eig[eig > 0])[::-1]
        if len(retained) > 1:
            out[block] = retained
    return out


def bound_condition_number(
        cov: mne.Covariance,
        info: mne.Info,
        max_condition: float,
) -> tuple[mne.Covariance, dict[str, Any]]:
    """Regularize ``cov`` just far enough to bound the condition number of its whitener.

    MNE whitens each channel-type block by scaling its eigen-directions with
    :math:`1/\\sqrt\\lambda`, so the smallest retained eigenvalue is amplified
    :math:`\\sqrt{\\lambda_{max}/\\lambda_{min}}` times more than the largest. For
    Maxwell-filtered data that ratio routinely reaches ten orders of magnitude, because
    the SSS basis is ill-conditioned; the inverse solution then treats a near-null
    direction of that basis as by far the most reliable measurement available, even
    though the noise recording says little about it.

    :func:`mne.cov.regularize` adds ``reg`` times the mean retained eigenvalue to the
    diagonal within the retained subspace of each of its blocks, leaving the null space at exactly zero so the rank is
    preserved. Solving :math:`(\\lambda_{max} + r\\bar\\lambda) / (\\lambda_{min}
    + r\\bar\\lambda) = K` for :math:`r` gives the smallest regularization that brings a
    block to the target condition number ``K``.

    Targeting a condition number rather than fixing ``reg`` makes this self-limiting: a
    block that already satisfies ``K`` yields ``r = 0`` and is returned untouched, while
    the blocks that do not all end up with the same bound on whitener amplification,
    which keeps recordings comparable within a group analysis.

    Parameters
    ----------
    cov
        Covariance to regularize.
    info
        Measurement info covering the channels of ``cov``.
    max_condition
        Target condition number; ``0`` to return ``cov`` unchanged.

    Returns
    -------
    cov
        The regularized covariance, or ``cov`` itself when nothing was applied.
    condition_info
        ``condition`` (per block, before regularization) and ``regularization`` (per
        block, omitted when none was applied).
    """
    # A diagonal covariance (e.g. ad-hoc) is well conditioned by construction, and
    # mne.cov.regularize does not handle the 1-dimensional data of a diagonal covariance
    if max_condition == 0 or cov['diag']:
        return cov, {}
    info = mne.pick_info(info, [info['ch_names'].index(name) for name in cov.ch_names], verbose='error')
    # Any channel type left out below would silently receive mne.cov.regularize's own
    # default of 0.1, so refuse rather than regularize a type this does not measure
    unhandled = sorted(set(info.get_channel_types(unique=True, only_data_chs=True)) - {'mag', 'grad', 'eeg'})
    if unhandled:
        raise NotImplementedError(f"Covariance with max_condition is not implemented for {enumeration(unhandled)} channels; use max_condition=0")
    condition = {}
    regularization = {}
    for block, retained in _block_spectra(cov, info).items():
        condition[block] = retained[0] / retained[-1]
        if condition[block] > max_condition:
            regularization[block] = (retained[0] - max_condition * retained[-1]) / (retained.mean() * (max_condition - 1))
    condition_info = {'condition': condition}
    if regularization:
        condition_info['regularization'] = regularization
        meg = regularization.get('meg', 0.)  # one block for magnetometers and gradiometers after Maxwell filtering
        cov = mne.cov.regularize(cov, info, mag=regularization.get('mag', meg), grad=regularization.get('grad', meg), eeg=regularization.get('eeg', 0.), exclude=[], rank=None, verbose='error')
    return cov, condition_info


class Covariance(Configuration):
    """Base class for covariance estimation.

    Parameters
    ----------
    method
        Covariance estimation method.
    max_condition
        Limit how unevenly the covariance weights different directions in the data, by
        adding just enough regularization to keep its condition number at or below this
        value (see Notes); ``0`` to use the covariance as estimated, without
        regularization.

    Notes
    -----
    Source localization uses the noise covariance to *whiten* the data. Whitening
    decomposes sensor space into the eigen-directions of the covariance and scales each
    direction by :math:`1/\\sqrt\\lambda`, the inverse square root of its noise variance
    (its eigenvalue). The inverse solution fits the whitened data, which means that the
    residual in each direction is weighted by :math:`1/\\lambda`: the lower the estimated
    noise in a direction, the more exactly the solution is required to reproduce the
    measured signal in that direction. The condition number of the covariance, the ratio
    between its largest and its smallest retained eigenvalue, is thus the ratio between the
    weight given to the quietest and to the noisiest direction.

    These weights are appropriate only if the small eigenvalues are reliable estimates of
    genuinely low noise, and for very small eigenvalues they are not. Maxwell filtering
    reconstructs the data from an ill-conditioned basis, which leaves directions whose
    variance is tiny but still above the tolerance that MNE-Python uses to estimate the
    rank of the covariance, so they are retained in the whitener. Their eigenvalues reflect
    the numerical properties of the filter rather than measured noise, and the smallest
    eigenvalues of a covariance estimated from a short recording are in any case the least
    accurate. When the condition number reaches ``1e10``, as it routinely does after
    Maxwell filtering, the residual in such a direction is weighted ``1e10`` times more
    than the residual in the noisiest direction, and any signal that leaks into it, for
    example through small differences in head position between the noise recording and
    the data, is fitted at the expense of the actual measurements.

    ``max_condition`` bounds this ratio. It adds the same constant to every retained
    eigenvalue (the ``reg`` parameter of :func:`mne.cov.regularize`, which adds ``reg``
    times the mean retained eigenvalue to the diagonal), with ``reg`` chosen as the smallest
    value that brings the condition number down to ``max_condition``. A covariance that
    already meets the target is left unchanged, and covariances that exceed it all end up
    with the same condition number, which keeps whitening comparable across subjects.
    Magnetometers, gradiometers and EEG channels form separate blocks, except that
    magnetometers and gradiometers form a single block after Maxwell filtering, which makes
    them linearly dependent.

    The default of ``1e6`` is meant to intervene only in the pathological range. Since
    whitening scales each direction by :math:`1/\\sqrt\\lambda`, at a condition number of
    ``1e6`` the quietest direction is amplified 1000 times more than the noisiest; at
    ``1e10``, where MNE-Python starts warning that the whitener is likely to be unstable,
    the factor is 100,000. At the other end, the default of :func:`mne.cov.regularize`
    (``reg=0.1``) bounds the condition number by roughly ten times the rank, i.e. a few
    hundred for a Maxwell-filtered recording, but it regularizes every covariance
    regardless of whether it needs it. The default sits between these two values:
    covariances that MNE-Python would regularize by default but that whiten reliably are
    left untouched, and only covariances that are far worse conditioned than MNE-Python's
    default ever produces are regularized.
    """

    DICT_ATTRS = ('method', 'max_condition')

    def __init__(
            self,
            method: str = 'empirical',
            max_condition: float = 1e6,
    ):
        # A condition number is at least 1, and reaching exactly 1 (a perfectly white
        # covariance) would take infinite regularization
        if max_condition != 0 and max_condition <= 1:
            raise ValueError(f"{max_condition=}: must be > 1 (or 0 to disable regularization)")
        self.method = method
        self.max_condition = max_condition

    def make(
            self,
            data: mne.io.BaseRaw | mne.BaseEpochs | list[mne.BaseEpochs],
            log_path: Path,
    ) -> mne.Covariance:
        """Estimate the covariance from ``data``, bounding its condition number.

        Parameters
        ----------
        data
            Data to estimate the covariance from (variable-length epochs arrive as a
            list with one :class:`mne.Epochs` object per epoch).
        log_path
            Path for the method's log file, when it writes one.
        """
        cov = self._make(data, log_path)
        info = (data[0] if isinstance(data, list) else data).info
        cov, condition_info = bound_condition_number(cov, info, self.max_condition)
        if condition_info:
            cov[CONDITION_INFO] = condition_info
        return cov


class RawCovariance(Covariance):
    """Noise covariance estimated from a continuous (empty room) recording.

    Parameters
    ----------
    method
        Covariance estimation method for :func:`mne.compute_raw_covariance`, or
        ``'ad_hoc'`` for :func:`mne.cov.make_ad_hoc_cov`.
    max_condition
        Limit the condition number of the covariance through regularization; see
        :class:`Covariance` for details.
    """

    def _make(self, data: mne.io.BaseRaw, log_path: Path) -> mne.Covariance:
        if self.method == 'ad_hoc':
            return mne.cov.make_ad_hoc_cov(data.info)
        return mne.compute_raw_covariance(data, method=self.method)


class EpochCovariance(Covariance):
    """Covariance estimated from epochs.

    Parameters
    ----------
    epoch
        Name of the epoch to estimate the covariance from.
    method
        Covariance estimation method for :func:`mne.compute_covariance`, or ``'best'``
        to pick the magnetometer regularization whose whitened global field power is
        closest to 1.
    keep_sample_mean
        Keep the sample mean (see :func:`mne.compute_covariance`).
    max_condition
        Limit the condition number of the covariance through regularization; see
        :class:`Covariance` for details.
    """

    DICT_ATTRS = (*Covariance.DICT_ATTRS, 'epoch', 'keep_sample_mean')

    def __init__(
            self,
            epoch: str,
            method: str = 'empirical',
            keep_sample_mean: bool = True,
            max_condition: float = 1e6,
    ):
        Covariance.__init__(self, method, max_condition)
        self.epoch = epoch
        self.keep_sample_mean = keep_sample_mean

    def _make(self, epochs_list: list[mne.BaseEpochs], log_path: Path) -> mne.Covariance:
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
            data = epochs_list[0]
        else:
            for epochs in epochs_list[1:]:
                if epochs.ch_names != info['ch_names'] or epochs.info['bads'] != info['bads']:
                    raise ValueError(f"cov={self.name!r}: variable-length epochs must have the same channels and bad channels")
                if (epochs.info['dev_head_t'] is None) != (info['dev_head_t'] is None) or (info['dev_head_t'] is not None and not numpy.allclose(epochs.info['dev_head_t']['trans'], info['dev_head_t']['trans'])):
                    raise ValueError(f"cov={self.name!r}: variable-length epochs must have the same head position (dev_head_t)")
            # Each epoch is already zero-mean, so the full-window baseline is a numerical
            # no-op; it marks the data as baseline-corrected for mne.compute_covariance
            concatenated = numpy.concatenate([epochs.get_data() for epochs in epochs_list], axis=-1)
            data = mne.EpochsArray(concatenated, info, baseline=(None, None), proj=False, verbose=False)

        method = 'empirical' if self.method == 'best' else self.method
        cov = mne.compute_covariance(data, self.keep_sample_mean, method=method)

        if self.method == 'best':
            if mne.pick_types(data.info, meg='grad', eeg=True, ref_meg=False).size:
                raise NotImplementedError(f"cov={self.name!r}: 'best' regularization is not implemented for EEG or gradiometer sensors; use a different setting for cov.")
            reg_vs = numpy.arange(0, 0.21, 0.01)
            covs = [mne.cov.regularize(cov, data.info, mag=v, rank=None) for v in reg_vs]

            # compute whitened global field power
            evoked = data.average()
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

    def __init__(self, covs: dict[str, Covariance], raw, references: dict[str, Reference | None], recordings: frozenset[tuple[str, str, str, str, str]]):
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

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        # Dependents are affected by the covariance itself, not by the setting that
        # produced it: a recording whose covariance already satisfies ``max_condition``
        # is regularized by 0 and comes out identical either way. Reporting the
        # regularization that was actually applied — rather than ``max_condition`` —
        # therefore limits invalidation to the recordings whose covariance did change.
        cov = {key: value for key, value in self._covs[ctx.state['cov']]._as_dict().items() if key != 'max_condition'}
        fingerprint = {'cov': cov, 'source_reference_add': self._references['average'].add}
        ctx.ensure()  # the stored metadata describes the current covariance only once it is up to date
        regularization = ctx.artifact_metadata.get('regularization')
        if regularization:
            fingerprint['regularization'] = regularization
        return fingerprint

    def build(self, ctx: Request) -> mne.Covariance:
        cov = self._covs[ctx.state['cov']]
        reference = self._references['average']
        montage = self.raw.root_source_pipe(ctx.state['raw']).montage
        if isinstance(cov, EpochCovariance):
            data = ctx.load('epochs')['epochs']
            # Variable-length epochs arrive as one Epochs object per epoch
            data = list(data) if isinstance(data, Datalist) else [data]
            for epochs in data:
                reference._prepare_source_data(epochs, montage)
        elif isinstance(cov, RawCovariance):
            data = ctx.load('raw')
            if reference.add:
                data.load_data()
            reference._prepare_source_data(data, montage)
        else:
            raise NotImplementedError(f"{cov=}")
        cov_path = self.path(ctx)
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        return cov.make(data, cov_path.with_suffix('.info.txt'))

    def artifact_metadata(self, ctx: Request, value: mne.Covariance) -> dict[str, Any]:
        # Not compared by compare_manifests, so recording the conditioning of every
        # covariance is free; dependency_fingerprint() reads back ``regularization``
        return value.get(CONDITION_INFO, {})

    def load(self, ctx: Request, path: Path) -> mne.Covariance:
        cov = mne.read_cov(path)
        if cov.data.dtype != 'float64':
            cov['data'] = cov['data'].astype(float)
        return cov

    def save(self, ctx: Request, path: Path, value: mne.Covariance) -> None:
        value.save(path, overwrite=True)
