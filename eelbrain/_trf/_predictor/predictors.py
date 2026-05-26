# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""High-level predictor abstractions for the TRF pipeline."""

from itertools import chain
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np

from ... import load
from ..._data_obj import Categorial, Dataset, Factor, NDVar, UTS, Var, combine
from ..._experiment.definitions import typed_arg
from ..._ndvar.ndvar import resample, set_tmin
from ..._ndvar.uts import pad
from .._code import Code, NDVAR_SHUFFLE_METHODS
from .._ndvar import shuffle
from .base import epoch_impulse_predictor, event_impulse_predictor


def t_stop_ds(ds: Dataset, t: float):
    "Dummy-event for the end of the last step"
    t_stop = ds.info['tstop'] + t
    out = {}
    for k, v in ds.items():
        if k == 'time':
            out['time'] = Var([t_stop])
        elif isinstance(v, Var):
            out[k] = Var(np.asarray([0], v.x.dtype))
        elif isinstance(v, Factor):
            out[k] = Factor([''])
        else:
            raise ValueError(f"{k!r} in predictor: {v!r}")
    return Dataset(out)


class EventPredictor:
    """Generate an impulse for each event."""

    def __init__(
            self,
            value: Union[float, str] = 1.,
            latency: Union[float, str] = 0.,
            sel: str = None,
    ):
        self.value = typed_arg(value, float, str)
        self.latency = typed_arg(latency, float, str)
        self.sel = typed_arg(sel, str)

    def _generate(self, uts: UTS, ds: Dataset, code: Code):
        assert code.stim is None
        if self.sel:
            raise NotImplementedError
        return epoch_impulse_predictor((ds.n_cases, uts), self.value, self.latency, code.code, ds)

    def _generate_continuous(self, uts: UTS, ds: Dataset, code: Code):
        assert code.stim is None
        if self.sel:
            ds = ds.sub(self.sel)
        return event_impulse_predictor(uts, 'T_relative', self.value, self.latency, code.code, ds)


class FilePredictorBase:

    def __init__(
            self,
            resample: Literal['bin', 'resample'] = None,
            sampling: Literal['continuous', 'discrete'] = None,
    ):
        assert resample in (None, 'bin', 'resample')
        self.resample = resample
        self.sampling = sampling

    def _resample(self, x: NDVar, tstep: float = None):
        if tstep is None or x.time.tstep == tstep:
            pass
        elif x.time.tstep > tstep:
            raise ValueError(f"Requested samplingrate rate is higher than in file ({1/tstep:g} > {1/x.time.tstep:g})")
        elif self.resample == 'bin':
            x = x.bin(tstep, label='start')
        elif self.resample == 'resample':
            srate = 1 / tstep
            int_srate = int(round(srate))
            srate = int_srate if abs(int_srate - srate) < .001 else srate
            x = resample(x, srate)
        elif self.resample is None:
            raise RuntimeError(f"{x.name} has tstep={x.time.tstep}, not {tstep}. Set the {self.__class__.__name__} resample parameter to enable automatic resampling.")
        else:
            raise RuntimeError(f"{self.resample=}")
        return x

    def _sampling(
            self,
            data_type: Literal['nuts', 'uts'] = None,
            nuts_method: str = None,
    ):
        if data_type == 'uts':
            return self.sampling or 'continuous'
        elif data_type == 'nuts' or nuts_method:
            if nuts_method == 'step':
                return 'continuous'
            elif nuts_method == 'is':
                return None
            elif nuts_method is None:
                return 'discrete'
            else:
                raise RuntimeError(f'{nuts_method=}')
        else:
            return self.sampling


class FilePredictor(FilePredictorBase):
    """Predictor stored in files corresponding to specific stimuli."""

    def __init__(
            self,
            resample: Literal['bin', 'resample'] = None,
            columns: bool = False,
            sampling: Literal['continuous', 'discrete'] = None,
    ):
        self.columns = columns
        super().__init__(resample, sampling)

    def _load(self, tstep: float, filename: str, directory: Path) -> NDVar:
        path = directory / f'{filename}.pickle'
        x = load.unpickle(path)
        if isinstance(x, list):
            for x in x:
                if x.time.tstep == tstep:
                    break
            else:
                raise IOError(f"Predictor file {path.name} is a list but does not contain a predictor with {tstep=}")
        elif isinstance(x, NDVar):
            x = self._resample(x, tstep)
        elif not isinstance(x, Dataset):
            raise TypeError(f'Predictor file {path.name} has invalid type {type(x)}:\n{x!r}')
        return x

    def _generate(self, tmin: float, tstep: float, n_samples: int, code: Code, directory: Path):
        file_name = code.nuts_file_name(self.columns)
        x = self._load(tstep, file_name, directory)
        if isinstance(x, Dataset):
            if tmin is None:
                tmin = 0
            if tstep is None:
                tstep = 0.001
            if n_samples is None:
                if 'tstop' in x.info:
                    tstop = x.info['tstop']
                else:
                    tstop = x[-1, 'time'] + 0.5
                n_samples = int((tstop - tmin) // tstep)
            uts = UTS(tmin, tstep, n_samples)
            x = self._ds_to_ndvar(x, uts, code)
            x.info['sampling'] = self._sampling('nuts', code.nuts_method)
        elif isinstance(x, NDVar):
            if code.nuts_method:
                raise code.error(f"Suffix {code.nuts_method} reserved for non-uniform time series predictors")
            x = pad(x, tmin, nsamples=n_samples, set_tmin=True)
            x.info['sampling'] = self._sampling('uts')
        else:
            raise RuntimeError(x)

        if code.shuffle in NDVAR_SHUFFLE_METHODS:
            x = shuffle(x, code.shuffle, code.shuffle_index, code.shuffle_angle)
            code.register_shuffle(index=True)
        return x

    def _generate_continuous(
            self,
            uts: UTS,
            ds: Dataset,
            stim_var: str,
            code: Code,
            directory: Path,
    ):
        cache = {stim: self._load(uts.tstep, code.with_stim(stim).nuts_file_name(self.columns), directory) for stim in ds[stim_var].cells}
        stim_type = {type(s) for s in cache.values()}
        assert len(stim_type) == 1
        stim_type = stim_type.pop()
        if stim_type is Dataset:
            dss = []
            for t, stim in ds.zip('T_relative', stim_var):
                x = cache[stim].copy()
                x['time'] += t
                dss.append(x)
                if code.nuts_method:
                    dss.append(t_stop_ds(x, t))
            x = self._ds_to_ndvar(combine(dss), uts, code)
        elif stim_type is NDVar:
            v = cache[ds[0, stim_var]]
            dimnames = v.get_dimnames(first='time')
            dims = (uts, *v.get_dims(dimnames[1:]))
            x = NDVar.zeros(dims, code.key)
            for t, stim in ds.zip('T_relative', stim_var):
                x_stim = cache[stim]
                i_start = uts._array_index(t + x_stim.time.tmin)
                i_stop = i_start + len(x_stim.time)
                if i_stop > len(uts):
                    raise ValueError(f"{code.string_without_rand} for {stim} is longer than the data")
                x.x[i_start:i_stop] = x_stim.get_data(dimnames)
        else:
            raise RuntimeError(f"{stim_type=}")
        return x

    def _ds_to_ndvar(self, ds: Dataset, uts: UTS, code: Code):
        if self.columns:
            column_key, mask_key = code.nuts_columns
            if column_key is None:
                column_key = 'value'
                ds[:, column_key] = 1
        else:
            column_key = 'value'
            mask_key = 'mask' if 'mask' in ds else None

        if 'time' in ds:
            time_col = 'time'
        elif 'onset' in ds:
            time_col = 'onset'
        elif 'i_start' in ds:
            sfreq = ds.info.get('sfreq') or ds.info.get('sampling_rate')
            if sfreq is None:
                raise KeyError(
                    "Predictor Dataset has 'i_start' (sample index) but no 'time'/'onset'. "
                    "Add ds.info['sfreq'] or ds.info['sampling_rate'] to convert to seconds, "
                    f"or provide a 'time' column. Columns: {list(ds)}"
                )
            ds['time'] = ds['i_start'].x.astype(float) / float(sfreq)
            time_col = 'time'
        else:
            raise KeyError(f"Predictor Dataset must have 'time', 'onset', or 'i_start' column; got {list(ds)}")

        if column_key not in ds:
            if 'trigger' in ds:
                ds[column_key] = Var(np.ones(ds.n_cases))
            else:
                raise KeyError(f"Predictor Dataset must have '{column_key}' or 'trigger' column; got {list(ds)}")

        if mask_key:
            mask = ds[mask_key].x
            assert mask.dtype.kind == 'b', "'mask' must be boolean"
        else:
            mask = None

        if code.shuffle_index:
            shuffle_mask = ds[code.shuffle_index].x
            if shuffle_mask.dtype.kind != 'b':
                raise code.error("shuffle index must be boolean", -1)
            elif code.shuffle == 'permute' and mask is not None:
                assert not np.any(shuffle_mask[~mask])
        elif code.shuffle == 'permute':
            shuffle_mask = mask
        else:
            shuffle_mask = None

        if code.shuffle == 'remask':
            if mask is None:
                raise code.error("$remask for predictor without mask", -1)
            rng = code._get_rng()
            if shuffle_mask is None:
                rng.shuffle(mask)
            else:
                remask = mask[shuffle_mask]
                rng.shuffle(remask)
                mask[shuffle_mask] = remask
            code.register_shuffle(index=True)

        if mask is not None:
            ds[column_key] *= mask

        if code.shuffle == 'permute':
            rng = code._get_rng()
            if shuffle_mask is None:
                rng.shuffle(ds[column_key].x)
            else:
                values = ds[column_key].x[shuffle_mask]
                rng.shuffle(values)
                ds[column_key].x[shuffle_mask] = values
            code.register_shuffle(index=True)

        if code.nuts_method == 'is':
            dim = Categorial('representation', ('step', 'impulse'))
            x = NDVar.zeros((dim, uts), name=code.key)
            x_step, x_impulse = x
        else:
            x = NDVar.zeros(uts, name=code.key)
            if code.nuts_method == 'step':
                x_step, x_impulse = x, None
            elif not code.nuts_method:
                x_step, x_impulse = None, x
            else:
                raise code.error(f"NUTS-method={code.nuts_method!r}")

        dt = uts.tstep / 2
        ds = ds[(ds['time'] > uts.tmin - dt) & (ds['time'] < uts.tmax + dt)]
        if x_impulse is not None:
            for t, v in ds.zip(time_col, column_key):
                x_impulse[t] += v
        if x_step is not None:
            t_stops = ds[1:, time_col]
            if ds[-1, column_key] != 0:
                if 'tstop' not in ds.info:
                    raise code.error("For step representation, the predictor datasets needs to contain ds.info['tstop'] to determine the end of the last step", -1)
                t_stops = chain(t_stops, [ds.info['tstop']])
            for t0, t1, v in zip(ds[time_col], t_stops, ds[column_key]):
                x_step[t0:t1] = v
        return x


class SessionPredictor(FilePredictorBase):
    """Predictor with time axis corresponding to experiment time."""

    def _load(self, tstep: Optional[float], filename: str, directory: Path) -> NDVar:
        path = directory / f'{filename}.pickle'
        x = load.unpickle(path)
        x = self._resample(x, tstep)
        return x

    def _generate(
            self,
            tmin: float,
            tstep: float,
            n_samples: int,
            code: Code,
            directory: Path,
            subject: str,
            recording: str,
    ):
        if code.stim is not None:
            raise code.error(f"{self.__class__.__name__} cannot have stimulus", -1)
        if code.nuts_method:
            raise code.error(f"Suffix {code.nuts_method} reserved for non-uniform time series predictors")
        if code.shuffle:
            raise code.error(f"Shuffling not available for {self.__class__.__name__}")
        file_name = f"{subject} {recording}~{code.string}"
        x = self._load(tstep, file_name, directory)
        x = pad(x, tmin, nsamples=n_samples, set_tmin=True)
        x.info['sampling'] = self._sampling('uts')
        return x

    def _epoch_for_data(
            self,
            x: NDVar,
            utss: List[UTS],
            onset_times: List[float],
    ) -> List[NDVar]:
        out = []
        for uts, t0 in zip(utss, onset_times):
            if t0:
                t0 = x.time.tstep * round(t0 / x.time.tstep)
                x_shifted = set_tmin(x, x.time.tmin - t0)
            else:
                x_shifted = x
            if x_shifted.time.tstep == uts.tstep:
                x_resampled = x_shifted
            else:
                x_cropped = pad(x_shifted, uts.tmin - 2, uts.tstop + 2)
                x_resampled = self._resample(x_cropped, uts.tstep)
            x_matching = pad(x_resampled, uts.tmin, uts.tstop, set_tmin=True)
            assert x_matching.time == uts
            out.append(x_matching)
        return out


class MakePredictor:
    """Predictor calls ``experiment.make_predictor()``."""

    pass
