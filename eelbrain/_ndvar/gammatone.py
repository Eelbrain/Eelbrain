# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import repeat
from math import floor
from typing import Literal

from eelbrain import NDVar, Scalar, UTS
from eelbrain._utils import tqdm
import numpy

from ._gammatone import aggregate_left, aggregate_right


def gammatone_bank(
        wav: NDVar,
        f_min: float,
        f_max: float,
        n: int,
        tstep: float = None,
        integration_cycles: float = None,
        integration_window: float = None,
        location: Literal['left', 'right'] = 'right',
        name: str = None,
) -> NDVar:
    """Gammatone filterbank response

    Parameters
    ----------
    wav
        Sound input.
    f_min
        Lower frequency cutoff.
    f_max
        Upper frequency cutoff.
    n
        Number of filter channels.
    tstep
        Time step size in the output (default is same as ``wav``).
    integration_cycles
        Number of cycles over which to integrate filter output (default 2).
    integration_window
        Integration time window in seconds (can be specified as alternative to
        ``integration_cycles``; e.g. ``0.010`` for 10 ms).
    location
        Location of the output relative to the input time axis:

        - ``right``: gammatone sample at end of integration window (default)
        - ``left``: gammatone sample at beginning of integration window

        Since gammatone filter response depends on ``integration_window``, the
        filter response will be delayed relative to the analytic envelope. To
        prevent this delay, use `location='left'`
    name
        NDVar name (default is ``wav.name``).

    Notes
    -----
    This function uses the `Gammatone <https://github.com/Lightning-Sandbox/gammatone>`_
    library, which can be installed with::

        $ pip install gammatone
    """
    try:
        from gammatone.filters import centre_freqs, erb_filterbank
        from gammatone.gtgram import make_erb_filters
    except ModuleNotFoundError:
        raise ModuleNotFoundError("gammatone module not installed. Install using:\n\n  $ pip install gammatone")

    assert location in ('left', 'right')
    fs = 1 / wav.time.tstep
    cfs = centre_freqs(fs, n, f_min, f_max)
    if integration_window is None:
        if integration_cycles is None:
            integration_cycles = 2
        integration_window_sec = integration_cycles / cfs
        integration_window_lens = numpy.ceil(integration_window_sec / wav.time.tstep).astype(int)
    elif integration_cycles is not None:
        raise ValueError(f"{integration_cycles=}, {integration_window=} (parameters are mutually exclusive")
    else:
        integration_window_len = int(round(integration_window * fs))
        integration_window_lens = repeat(integration_window_len)

    wav_ = wav
    tmin = wav.time.tmin
    if tstep is None:
        tstep = wav.time.tstep
    wave = wav_.get_data('time')
    # based on gammatone library gtgram, rewritten to reduce memory footprint
    output_n_samples = floor(len(wave) * wav.time.tstep / tstep)
    output_step = tstep / wav.time.tstep
    output_data = numpy.zeros((n, output_n_samples))
    disable = n * output_n_samples < 200_000  # 100 bands * 2 s * 1000 Hz
    for i, cf, window in tqdm(zip(range(n - 1, -1, -1), cfs, integration_window_lens), "Gammatone filterbank", total=len(cfs), unit='band', disable=disable):
        fcoefs = numpy.flipud(make_erb_filters(fs, cf))
        xf = erb_filterbank(wave, fcoefs)
        xf **= 2
        if location == 'left':
            aggregate_left(xf[0], output_n_samples, output_step, window, output_data[i])
        else:
            aggregate_right(xf[0], output_n_samples, output_step, window, output_data[i])
        output_data[i] /= window
    output_data = numpy.sqrt(output_data, out=output_data)
    # package output
    freq_dim = Scalar('frequency', cfs[::-1], 'Hz')
    time_dim = UTS(tmin, tstep, output_n_samples)
    if name is None:
        name = wav.name
    return NDVar(output_data, (freq_dim, time_dim), name)
