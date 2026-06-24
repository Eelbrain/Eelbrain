# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Epoch and evoked sensor derivatives.

Dependency structure:

    epochs
    ├── PrimaryEpoch (single run) / ContinuousEpoch / SecondaryEpoch
    │     ├── epoch-events               (trial metadata, see events tree)
    │     └── recording-epochs
    │           ├── selected-events      (same epoch, provides trial timings)
    │           └── raw
    │
    ├── PrimaryEpoch (combine runs)
    │     ├── epoch-events               (aggregated across all runs)
    │     └── recording-epochs  ×N      (one per run)
    │           ├── selected-events      (for that run)
    │           └── raw
    │
    └── SuperEpoch
          └── epochs  ×N                 (one per sub-epoch, each recursing into this tree)

Explanations:

- The ``epoch`` node combines events (``epoch-events``) with data epochs.
  The reason for keeping those separate is to make data caches less dependent on event changes.
- :class:`SuperEpochs` combine already epoched data (at the ``epochs`` node).
- Other epoch types work by selecting events before loading data (at the ``recording-epochs`` node).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Sequence
import shutil
import warnings

import mne
import numpy as np

from ... import load
from ..._data_obj import Datalist, Dataset, combine
from ..._exceptions import ConfigurationError, DimensionMismatchError
from ..._info import BAD_CHANNELS, INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS, INTERPOLATE_WINDOWS_MAX
from ..._mne import shift_mne_epoch_trigger
from ..._text import n_of
from ..._meeg.interpolation import _interpolate_bads_eeg, _interpolate_bads_meg, _interpolate_bad_windows_eeg, _interpolate_bad_windows_meg
from ..derivative_cache import CachePolicy, Dependency, Derivative, Request, UncachedDerivative
from ..preprocessing import RawPipeGraph, Reference, raw_node_name
from ..test_def import TestDims
from .config import EPOCH_EXTRACT_OPTIONS, ContinuousEpoch, EpochBase, EpochCollection, PrimaryEpoch, SecondaryEpoch, SuperEpoch


def _drop_bad_eeg_channels_with_missing_locs(
        epochs_list: Sequence[mne.Epochs],
) -> None:
    """Drop EEG bad channels that can not be interpolated due to missing locations."""
    for epochs in epochs_list:
        bad_channels = epochs.info['bads']
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
        missing = []
        for pick in picks:
            ch = epochs.info['chs'][pick]
            if ch['ch_name'] not in bad_channels:
                continue
            loc = ch['loc'][:3]
            if not np.isfinite(loc).all() or np.allclose(loc, 0):
                missing.append(ch['ch_name'])
        if missing:
            warnings.warn(
                f"Dropping EEG bad {n_of(len(missing), 'channel')} with missing sensor location before interpolation: {', '.join(missing)}",
                RuntimeWarning,
            )
            epochs.drop_channels(missing)


def _evoked_comments(evoked: list[mne.Evoked]) -> list[str]:
    return [e.comment or 'No comment' for e in evoked]


# save/load one or multiple epochs objects
def _flatten_epochs(value) -> list[mne.BaseEpochs]:
    """Flatten a (possibly nested) epochs artifact into a list of MNE Epochs."""
    if isinstance(value, mne.BaseEpochs):
        return [value]
    out = []
    for item in value:
        out.extend(_flatten_epochs(item))
    return out


def _save_epochs(path: Path, value) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir()
    if isinstance(value, Datalist):
        for i, epochs in enumerate(value):
            epochs.save(path / f'epochs-{i:04d}-epo.fif', overwrite=True)
    else:
        value.save(path / 'epochs-0000-epo.fif', overwrite=True)


def _load_epochs(path: Path, metadata: dict[str, Any]):
    if metadata['kind'] == 'datalist':
        return Datalist(
            [mne.read_epochs(path / relpath, proj=False) for relpath in metadata['files']],
            metadata['name'],
            metadata['fmt'],
        )
    return mne.read_epochs(path / metadata['file'], proj=False)


def _epochs_artifact_metadata(value) -> dict[str, Any]:
    if isinstance(value, Datalist):
        return {
            'kind': 'datalist',
            'files': [f'epochs-{i:04d}-epo.fif' for i in range(len(value))],
            'name': value.name,
            'fmt': value._fmt,
        }
    return {'kind': 'single', 'file': 'epochs-0000-epo.fif'}


class RecordingEpochsDerivative(Derivative[Any]):
    """MNE epochs extracted from a single raw recording.

    Loads the event shell from :class:`~events.SelectedEventsDerivative`,
    reads the corresponding raw data, and extracts fixed-length or
    variable-length MNE :class:`~mne.Epochs`.  Always restricted to one
    task/run combination; multi-run aggregation is handled by
    :class:`EpochsDerivative`.

    Options
    -------
    baseline
        Baseline correction to apply during epoch extraction.
    samplingrate / decim
        Sampling rate or decimation override.
    pad
        Extra time padding before epoch extraction.
    trigger_shift
        Whether to apply trigger shifting from the epoch definition.
    tmin, tmax, tstop
        Time window overrides.
    interpolate_bads
        Whether to interpolate bad channels while building epochs.
    reject
        Whether to apply per-epoch rejection state.
    """
    name = 'recording-epochs'
    key_fields = ('subject', 'session', 'run', 'raw', 'epoch', 'epoch_rejection', 'reference')
    cache_suffix = '.epochs'
    OPTION_DEFAULTS = {
        'baseline': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'trigger_shift': True,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'reject': True,
    }

    def __init__(self, raw: RawPipeGraph, epochs: dict[str, EpochBase], references: dict[str, Reference | None], cache: bool = False):
        self.raw = raw
        self.epochs = epochs
        self.references = references
        if not cache:
            self.cache_policy = CachePolicy.NEVER

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        if not isinstance(epoch, (PrimaryEpoch, SecondaryEpoch, ContinuousEpoch)):
            raise TypeError(f"{epoch=}")
        raw_name = ctx.state['raw']
        state = {'task': epoch.task}
        event_options = ctx.options_for('selected-events', 'reject', *EPOCH_EXTRACT_OPTIONS)
        return (
            Dependency('selected-events', state=state, options=event_options),
            Dependency(raw_node_name(raw_name), label='raw', state=state),
        )

    def dependency_fingerprint_override(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> dict[str, Any] | None:
        if dep.name != 'selected-events':
            return None
        epoch = self.epochs[ctx.state['epoch']]
        ds = ctx.load('selected-events')
        out = {'sample': ds['sample'], 'bad_channels': ds.info[BAD_CHANNELS]}
        for attr in epoch._eval_attrs():
            out[attr] = getattr(epoch, attr)
        if ds.info.get(INTERPOLATE_CHANNELS, False) and INTERPOLATE_CHANNELS in ds:
            out[INTERPOLATE_CHANNELS] = ds[INTERPOLATE_CHANNELS]
        if ds.info.get(INTERPOLATE_WINDOWS, False) and INTERPOLATE_WINDOWS in ds:
            out[INTERPOLATE_WINDOWS] = ds[INTERPOLATE_WINDOWS]
        return out

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {
            'epoch': self.epochs[ctx.state['epoch']],
            'reference': self.references[ctx.state['reference']],
        }

    def build(self, ctx: Request):
        epoch = self.epochs[ctx.state['epoch']]
        ds = ctx.load('selected-events')
        raw = ctx.load('raw')
        if ds.info[BAD_CHANNELS]:
            raw.info['bads'] = sorted(set(raw.info['bads'] + ds.info[BAD_CHANNELS]))
        ds.info['raw'] = raw
        tmin, tmax, tstop, baseline, decim, variable_tmax = epoch._extraction_parameters(ds, ctx.options)
        if variable_tmax:
            epochs_list = load.mne.variable_length_mne_epochs(ds, tmin, tmax, baseline, allow_truncation=True, decim=decim, reject_by_annotation=False, i_start='sample', trigger='value')
            epoch_value = Datalist(epochs_list, 'epochs')
        else:
            epochs = load.mne.mne_epochs(ds, tmin, tmax, baseline, i_start='sample', decim=decim, drop_bad_chs=False, tstop=tstop, reject_by_annotation=False, trigger='value')
            if ctx.options['trigger_shift'] and epoch.post_baseline_trigger_shift:
                shift = ds.eval(epoch.post_baseline_trigger_shift)
                epochs = shift_mne_epoch_trigger(epochs, shift, epoch.post_baseline_trigger_shift_min, epoch.post_baseline_trigger_shift_max)
            else:
                # Baseline is already applied to the data; clear the attribute so that the
                # downstream combine() -> mne.concatenate_epochs() does not apply it a second
                # time (double baseline correction). The post_baseline_trigger_shift branch
                # above already yields baseline=None via shift_mne_epoch_trigger.
                epochs.baseline = None
            assert len(epochs) == ds.n_cases
            epoch_value = epochs
            epochs_list = [epoch_value]

        if ctx.options['interpolate_bads']:
            _drop_bad_eeg_channels_with_missing_locs(epochs_list)
            data_types = TestDims.coerce('sensor').data_to_ndvar(epochs_list[0].info)
            if ds.info.get(INTERPOLATE_WINDOWS, False) and any(ds[INTERPOLATE_WINDOWS]):
                # time-resolved interpolation for long, variable-length epochs
                windows_all = list(ds[INTERPOLATE_WINDOWS])
                max_interpolate = ds.info[INTERPOLATE_WINDOWS_MAX]
                interp_cache = {}
                offset = 0
                for epochs in epochs_list:
                    windows = windows_all[offset:offset + len(epochs)]
                    offset += len(epochs)
                    if 'mag' in data_types:
                        _interpolate_bad_windows_meg(epochs, windows, interp_cache)
                    if 'eeg' in data_types:
                        _interpolate_bad_windows_eeg(epochs, windows, max_interpolate)
            elif ds.info[INTERPOLATE_CHANNELS] and any(ds[INTERPOLATE_CHANNELS]):
                bads_all = epochs_list[0].info['bads']
                bads_individual = [sorted(set(bads_all + bads_i)) for bads_i in ds[INTERPOLATE_CHANNELS]]
                if 'mag' in data_types:
                    interp_cache = {}
                    _interpolate_bads_meg(epoch_value, bads_individual, interp_cache)
                if 'eeg' in data_types:
                    _interpolate_bads_eeg(epoch_value, bads_individual)
            else:
                for epochs in epochs_list:
                    epochs.interpolate_bads(reset_bads=False)

        # EEG re-referencing, after channel interpolation
        reference = self.references[ctx.state['reference']]
        if reference is not None:
            if 'eeg' not in TestDims.coerce('sensor').data_to_ndvar(epochs_list[0].info):
                raise ConfigurationError(f"reference={ctx.state['reference']!r}: {ctx.state['subject']}/{epoch.name} has no EEG channels to re-reference; set reference='' for data without EEG.")
            montage = self.raw.root_source_pipe(ctx.state['raw']).montage
            epochs_list = [reference._apply_reference(epochs, montage=montage) for epochs in epochs_list]
            epoch_value = Datalist(epochs_list, 'epochs') if variable_tmax else epochs_list[0]

        return epoch_value

    def load(self, ctx: Request, path: Path):
        return _load_epochs(path, ctx.artifact_metadata)

    def save(self, ctx: Request, path: Path, value) -> None:
        _save_epochs(path, value)

    def artifact_metadata(self, ctx: Request, value) -> dict[str, Any]:
        return _epochs_artifact_metadata(value)


class EpochsDerivative(Derivative[Any]):
    """Epoch dataset aggregating across runs and sub-epochs.

    For single-run :class:`PrimaryEpoch` and :class:`ContinuousEpoch`, wraps
    :class:`RecordingEpochsDerivative` directly.  For combine-all
    :class:`PrimaryEpoch` epochs, concatenates per-run
    :class:`RecordingEpochsDerivative` results.  For :class:`SuperEpoch`,
    concatenates sub-epoch :class:`EpochsDerivative` results.

    Options
    -------
    ndvar
        Whether to convert epoch data to NDVars (``True | False | 'both'``).
    data
        Sensor representation to return.
    (remaining options forwarded to :class:`RecordingEpochsDerivative`)
    """
    name = 'epochs'
    key_fields = ('subject', 'session', 'raw', 'epoch', 'epoch_rejection', 'reference')
    cache_suffix = '.epochs'
    OPTION_DEFAULTS = {
        'baseline': False,
        'samplingrate': None,
        'decim': None,
        'pad': 0,
        'trigger_shift': True,
        'tmin': None,
        'tmax': None,
        'tstop': None,
        'interpolate_bads': False,
        'reject': True,
    }
    VIEW_OPTION_DEFAULTS = {
        'ndvar': True,
        'data': 'sensor',
    }

    def __init__(self, raw, epochs: dict[str, Any], runs_for: dict[tuple[str, str, str], tuple[str, ...]], cache: bool = False):
        self.raw = raw
        self.epochs = epochs
        self._runs_for = runs_for
        if not cache:
            self.cache_policy = CachePolicy.NEVER

    def _find_runs(self, ctx: Request, epoch) -> tuple[str, ...]:
        """Runs to aggregate over"""
        if isinstance(epoch, PrimaryEpoch):
            if epoch.run is None:
                key = (ctx.state['subject'], ctx.state['session'], epoch.task)
                if key in self._runs_for:
                    return self._runs_for[key]
            return ()
        if isinstance(epoch, SecondaryEpoch):
            return self._find_runs(ctx, self.epochs[epoch.sel_epoch])
        return ()

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            raise TypeError(f"{epoch=}: load_epochs not supported for EpochCollection")
        if isinstance(epoch, SuperEpoch):
            # Inject explicitly-overridden INHERITED_PARAMS as direct options so sub-epochs
            # are loaded with the SuperEpoch's window/baseline/decim rather than their own.
            epoch_overrides = {k: getattr(epoch, k) for k in epoch._explicit_params if k in epoch.INHERITED_PARAMS}
            forward_keys = [k for k in self.OPTION_DEFAULTS if k not in epoch_overrides]
            sub_options = ctx.options_for('epochs', *forward_keys, ndvar=False, data='sensor', **epoch_overrides)
            return tuple(
                Dependency('epochs', label=sub_epoch, state={'epoch': sub_epoch}, options=sub_options)
                for sub_epoch in epoch.sub_epochs
            )
        runs = self._find_runs(ctx, epoch)
        rec_options = ctx.options_for('recording-epochs', *self.OPTION_DEFAULTS)
        sel_options = ctx.options_for('epoch-events', 'reject', *EPOCH_EXTRACT_OPTIONS)
        state = {'task': epoch.task}
        if runs:
            return (
                Dependency('epoch-events', options=sel_options, state=state),
                *[Dependency('recording-epochs', label=f'epochs-{run}', state={**state, 'run': run}, options=rec_options) for run in runs],
            )
        return (
            Dependency('epoch-events', options=sel_options, state=state),
            Dependency('recording-epochs', state=state, options=rec_options),
        )

    def dependency_fingerprint_override(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> dict[str, Any] | None:
        """Depend on the subset of events that is actually relevant for the epoch"""
        if dep.name != 'epoch-events':
            return None
        epoch = self.epochs[ctx.state['epoch']]
        ds = ctx.load(dep.label or dep.name)
        out = {'sample': ds['sample'], 'bad_channels': ds.info[BAD_CHANNELS]}
        for attr in epoch._eval_attrs():
            out[attr] = getattr(epoch, attr)
        if ds.info.get(INTERPOLATE_CHANNELS, False) and INTERPOLATE_CHANNELS in ds:
            out[INTERPOLATE_CHANNELS] = ds[INTERPOLATE_CHANNELS]
        if ds.info.get(INTERPOLATE_WINDOWS, False) and INTERPOLATE_WINDOWS in ds:
            out[INTERPOLATE_WINDOWS] = ds[INTERPOLATE_WINDOWS]
        return out

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'epoch': self.epochs[ctx.state['epoch']]}

    def build(self, ctx: Request):
        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, SuperEpoch):
            epochs_list = []
            for sub_epoch in epoch.sub_epochs:
                ds = ctx.load(sub_epoch)
                epoch_value = ds['epochs']
                if ctx.options['trigger_shift'] and epoch.post_baseline_trigger_shift:
                    if isinstance(epoch_value, Datalist):
                        raise NotImplementedError("post_baseline_trigger_shift for variable-length SuperEpoch")
                    shift = ds.eval(epoch.post_baseline_trigger_shift)
                    epoch_value = shift_mne_epoch_trigger(epoch_value, shift, epoch.post_baseline_trigger_shift_min, epoch.post_baseline_trigger_shift_max)
                if isinstance(epoch_value, Datalist):
                    epochs_list.extend(epoch_value)
                else:
                    epochs_list.append(epoch_value)
            return Datalist(epochs_list, 'epochs')
        runs = self._find_runs(ctx, epoch)
        if runs:
            return Datalist([ctx.load(f'epochs-{run}') for run in runs], 'epochs')
        return ctx.load('recording-epochs')

    def load(self, ctx: Request, path: Path):
        return _load_epochs(path, ctx.artifact_metadata)

    def save(self, ctx: Request, path: Path, value) -> None:
        _save_epochs(path, value)

    def artifact_metadata(self, ctx: Request, value) -> dict[str, Any]:
        return _epochs_artifact_metadata(value)

    def apply_view_options(self, ctx: Request, epoch_value):
        epoch = self.epochs[ctx.state['epoch']]
        data = TestDims.coerce(ctx.view_options['data'])
        if not data.sensor:
            raise ValueError(f"data={data.string!r}; load_evoked is for loading sensor data")
        if data.sensor is not True and not ctx.view_options['ndvar']:
            raise ValueError(f"data={data.string!r} with ndvar=False")

        if isinstance(epoch, SuperEpoch):
            dss = []
            for sub_epoch in epoch.sub_epochs:
                ds = ctx.load(sub_epoch)
                ds[:, 'epoch'] = sub_epoch
                dss.append(ds)
            ds = combine(dss)
        else:
            ds = ctx.load('epoch-events')

        # Flatten to a list of MNE Epochs (variable-length epochs are stored as
        # single-trial Epochs and can be nested when aggregating across runs).
        epochs_list = _flatten_epochs(epoch_value)
        # Variable-length epochs have differing numbers of samples and cannot be
        # concatenated into a single Epochs object.
        variable_tmax = len({epochs.times.size for epochs in epochs_list}) > 1
        if variable_tmax:
            ds['epochs'] = Datalist(epochs_list, 'epochs')
        else:
            ds['epochs'] = combine(epochs_list)

        ndvar = ctx.view_options['ndvar']
        if ndvar:
            info = epochs_list[0].info
            sensor_types = data.data_to_ndvar(info)
            ds.info['sensor_types'] = sensor_types
            source_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            for data_kind in sensor_types:
                sysname = source_pipe._get_sysname(info, ds.info['subject'], data_kind)
                adjacency = source_pipe._get_adjacency(data_kind)
                name = 'meg' if data_kind == 'mag' and 'grad' not in sensor_types else data_kind
                if variable_tmax:
                    ys = Datalist([load.mne.epochs_ndvar(epochs, data=data_kind, sysname=sysname, adjacency=adjacency, name=data_kind)[0] for epochs in epochs_list])
                    if isinstance(data.sensor, str):
                        ys = Datalist([getattr(y, data.sensor)('sensor') for y in ys])
                else:
                    ys = load.mne.epochs_ndvar(ds['epochs'], data=data_kind, sysname=sysname, adjacency=adjacency)
                    if isinstance(data.sensor, str):
                        ys = getattr(ys, data.sensor)('sensor')
                ds[name] = ys
            if ndvar != 'both':
                del ds['epochs']

        return ds


class EvokedDerivative(Derivative[list[mne.Evoked]]):
    """Evoked dataset with cached MNE evoked objects as internal artifact.

    Options
    -------
    baseline
        Baseline correction to apply at load time.
    ndvar
        Whether to convert the returned data to NDVars.
    cat
        Optional subset of model cells to keep.
    data
        Sensor representation to return.
    samplingrate
        Sampling rate override for the underlying epochs artifact.
    decim
        Decimation override for the underlying epochs artifact.
    """
    name = 'evoked'
    key_fields = (
        'subject', 'session', 'raw',
        'epoch', 'epoch_rejection', 'reference', 'model', 'equalize_evoked_count',
    )
    cache_suffix = '-ave.fif'
    OPTION_DEFAULTS = {
        'samplingrate': None,
        'decim': None,
    }
    VIEW_OPTION_DEFAULTS = {
        'baseline': False,
        'ndvar': False,
        'cat': None,
        'data': 'sensor',  # TODO
    }

    def __init__(self, raw, epochs: dict[str, Any]):
        self.raw = raw
        self.epochs = epochs

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        epoch = self.epochs[ctx.state['epoch']]
        options = {
            'baseline': True if epoch.post_baseline_trigger_shift else False,
            'samplingrate': ctx.options['samplingrate'],
            'decim': ctx.options['decim'],
            'interpolate_bads': 'keep',
            'reject': True,
            'trigger_shift': True,
            'ndvar': False,
            'data': 'sensor',
        }
        return (
            Dependency('epochs', options=options),
            Dependency('epoch-events', options=ctx.options_for('epoch-events', 'samplingrate', 'decim', reject=True)),
        )

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {}

    def dependency_fingerprint_override(self, ctx: Request, dep: Dependency, dep_ctx: Request) -> dict[str, Any] | None:
        if dep.name != 'epoch-events':
            return None
        model = ctx.state['model']
        if model:
            ds = ctx.load(dep.label or dep.name)
            ds = self._aggregate(ds, ctx)
            return {'model': ds.eval(model)}
        return {}

    def build(self, ctx: Request) -> list[mne.Evoked]:
        model = ctx.state['model']
        data = ctx.load('epochs')
        data = self._aggregate(data, ctx)
        data.rename('epochs', 'evoked')
        model_vars = model.split('%') if model else ()
        for evoked, *cell in data.zip('evoked', *model_vars):
            evoked.info['description'] = "Eelbrain"
            evoked.comment = ' | '.join(cell)
        return data['evoked']

    @staticmethod
    def _aggregate(data: Dataset, ctx: Request) -> Dataset:
        return data.aggregate(
            ctx.state['model'],
            never_drop=('epochs',),
            drop_bad=True,
            equal_count=ctx.state['equalize_evoked_count'] == 'eq',
            drop=('sample', 't_edf', 'onset', 'index', 'value'),
        )

    def load(self, ctx: Request, path: Path) -> list[mne.Evoked]:
        return mne.read_evokeds(path, proj=False)

    def save(self, ctx: Request, path: Path, value: list[mne.Evoked]) -> None:
        mne.write_evokeds(path, value, overwrite=True)

    def dependency_fingerprint(self, ctx: Request, view: str | None = None) -> dict[str, Any]:
        if view is None:
            return self.fingerprint(ctx)
        if view != 'shell':
            raise ValueError(f"{self.name!r} does not define dependency view {view!r}")
        return self.fingerprint(ctx)

    def load_view(self, ctx: Request, view: str):
        if view != 'shell':
            return super().load_view(ctx, view)

        epoch = self.epochs[ctx.state['epoch']]
        if isinstance(epoch, EpochCollection):
            dss = []
            for sub_epoch in epoch.collect:
                ds = ctx.load(
                    'evoked',
                    state={'epoch': sub_epoch},
                    options=ctx.options_for('evoked', *self.OPTION_DEFAULTS),
                    view='shell',
                )
                ds[:, 'epoch'] = sub_epoch
                dss.append(ds)
            return combine(dss)

        data = ctx.load('epoch-events')
        return self._aggregate(data, ctx)

    def apply_view_options(self, ctx: Request, evoked: list[mne.Evoked]) -> Dataset:
        ds = ctx.load(view='shell')
        cat = ctx.view_options['cat']
        if cat:
            ds = ds.sub(ds.eval(ctx.state['model']).isin(cat))

        # Unpack evoked objects and map them to the ds rows
        model = ctx.state['model']
        model_vars = model.split('%') if model else ()
        cells = [' | '.join(cell) or 'No comment' for cell in ds.zip(*model_vars)] if model_vars else ['No comment']
        evoked_by_cell = dict(zip(_evoked_comments(evoked), evoked))
        if len(evoked_by_cell) != len(evoked):
            raise RuntimeError(f"Cached evoked data contains duplicate comments: {_evoked_comments(evoked)!r}")
        try:
            evoked = [evoked_by_cell[cell] for cell in cells]
        except KeyError:
            raise RuntimeError(f"Error reading cached evoked: available={tuple(evoked_by_cell)}, requested={tuple(cells)}") from None

        # Baseline
        epoch = self.epochs[ctx.state['epoch']]
        baseline = ctx.view_options['baseline']
        if baseline is True:
            baseline = epoch.baseline
        if baseline and not epoch.post_baseline_trigger_shift:
            for evoked_i in evoked:
                mne.baseline.rescale(evoked_i.data, evoked_i.times, baseline, 'mean', copy=False)

        # NDVar
        data = TestDims.coerce(ctx.view_options['data'])
        to_ndvar = isinstance(data.sensor, str) or ctx.view_options['ndvar']
        if to_ndvar:
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
            source_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            for sensor_type in sensor_types:
                sysname = source_pipe._get_sysname(info, ctx.state['subject'], sensor_type)
                adjacency = source_pipe._get_adjacency(sensor_type)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
                if sensor_type != 'eog' and isinstance(data.sensor, str):
                    ds[name] = getattr(ds[name], data.sensor)('sensor')
        else:
            ds['evoked'] = evoked
        return ds


class EvokedGroupDatasetDerivative(UncachedDerivative[Dataset]):
    """Group-level sensor evoked dataset assembled from subject datasets.

    Options
    -------
    baseline
        Baseline correction to apply at load time.
    ndvar
        Whether to convert the returned data to NDVars.
    cat
        Optional subset of model cells to keep.
    samplingrate
        Sampling rate override for the underlying evoked artifact.
    decim
        Decimation override for the underlying evoked artifact.
    data
        Sensor representation to return.
    """
    name = 'evoked-group-dataset'
    OPTION_DEFAULTS = {
        'baseline': False,
        'ndvar': True,
        'samplingrate': None,
        'decim': None,
        'data': 'sensor',
    }
    VIEW_OPTION_DEFAULTS = {
        'cat': None,
    }

    def __init__(self, raw, groups):
        self.raw = raw
        self.groups = groups

    def key(self, ctx: Request) -> dict[str, Any]:
        return {'subjects': tuple(self.groups[ctx.state['group']]), 'options': ctx.options}

    def fingerprint(self, ctx: Request) -> dict[str, Any]:
        return {'subjects': tuple(self.groups[ctx.state['group']])}

    def dependencies(self, ctx: Request) -> tuple[Dependency, ...]:
        options = ctx.options_for('evoked', 'baseline', 'samplingrate', 'decim', 'data')
        return tuple(
            Dependency('evoked', label=subject, state={'subject': subject}, options=options)
            for subject in self.groups[ctx.state['group']]
        )

    def build(self, ctx: Request) -> Dataset:
        dss = [ctx.load(subject) for subject in self.groups[ctx.state['group']]]
        data = TestDims.coerce(ctx.options['data'])
        ndvar = ctx.options['ndvar']
        individual_ndvar = isinstance(data.sensor, str)
        if individual_ndvar:
            ndvar = False
        elif ndvar:
            for ds in dss:
                for evoked in ds['evoked']:
                    evoked.info['bads'] = []
        ds = combine(dss, incomplete='drop')
        if not ndvar and not individual_ndvar:
            lens = [len(evoked.times) for evoked in ds['evoked']]
            ulens = set(lens)
            if len(ulens) > 1:
                err = ["Unequal time axis sampling (len):"]
                alens = np.array(lens)
                for length in ulens:
                    subjects = ', '.join(ds[alens == length, 'subject'].cells)
                    err.append(f"{length}: {subjects}")
                raise DimensionMismatchError('\n'.join(err))
            return ds
        if ndvar and not individual_ndvar:
            evoked = ds['evoked']
            del ds['evoked']
            info = evoked[0].info
            sensor_types = ds.info['sensor_types'] = data.data_to_ndvar(info)
            source_pipe = self.raw.root_source_pipe(ctx.state['raw'])
            subject = ds[0, 'subject']
            for sensor_type in sensor_types:
                sysname = source_pipe._get_sysname(info, subject, sensor_type)
                adjacency = source_pipe._get_adjacency(sensor_type)
                name = 'meg' if sensor_type == 'mag' else sensor_type
                ds[name] = load.mne.evoked_ndvar(evoked, data=sensor_type, sysname=sysname, adjacency=adjacency)
                if sensor_type != 'eog' and isinstance(data.sensor, str):
                    ds[name] = getattr(ds[name], data.sensor)('sensor')
        return ds
