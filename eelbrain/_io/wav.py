# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""I/O for wave files"""
from pathlib import Path
from typing import Literal

import numpy as np

from .._data_obj import NDVar, Scalar, UTS
from .._types import PathArg
from .._utils import ui


FILETYPES = [("WAV files", "*.wav")]


def load_wav(
        filename: PathArg | None = None,
        name: str | None = None,
        backend: Literal['wave', 'scipy', 'librosa'] = 'wave',
) -> NDVar:
    """Load a wav file as NDVar

    Parameters
    ----------
    filename
        Filename of the wav file. If not filename is specified, a file dialog is
        shown to select one.
    name
        NDVar name (default is the file name).
    backend
        Whether to read the file using the builtin :mod:`wave` module,
        :mod:`scipy.io.wavfile`, or :func:`librosa.load`.

    Returns
    -------
    wav : NDVar
        NDVar with the wav file's data. If the file contains a single channel,
        the NDVar dimensions are ``(time,)``; if it contains several channels,
        they are ``(time, channel)``. ``wav.info`` contains entries for
        ``filename`` and ``samplingrate``.

    Notes
    -----
    Uses :mod:`scipy.io.wavfile`.
    """
    if filename is None:
        path = ui.ask_file("Load WAV File", "Select WAV file to load as NDVar", FILETYPES)
        if not path:
            return
    else:
        path = Path(filename)
        if not path.suffix and not path.exists():
            path = path.with_suffix('.wav')

    if backend == 'wave':
        import wave
        with wave.open(str(path), 'rb') as fp:
            n_channels = fp.getnchannels()
            n_frames = fp.getnframes()
            n_bytes = fp.getsampwidth()
            srate = fp.getframerate()
            data = fp.readframes(n_frames)
        data = np.frombuffer(data, f'<i{n_bytes}')
        if n_channels > 1:
            data = data.reshape((-1, n_channels))
    elif backend == 'scipy':
        from scipy.io import wavfile
        srate, data = wavfile.read(path)
    elif backend == 'librosa':
        import librosa
        data, srate = librosa.load(path)
        data = data.T
    else:
        raise ValueError(f"{backend=}")

    time = UTS(0, 1. / srate, data.shape[0])
    if name is None:
        name = path.name
    info = {'filename': str(path), 'samplingrate': srate}
    if data.ndim == 1:
        return NDVar(data, (time,), name, info)
    elif data.ndim == 2:
        chan = Scalar('channel', np.arange(data.shape[1]))
        return NDVar(data, (time, chan), name, info)
    else:
        raise NotImplementedError(f"Data with {data.ndim} dimensions")


def save_wav(
        ndvar: NDVar,
        filename: PathArg = None,
        toint: bool = False,
):
    """Save an NDVar as wav file

    Parameters
    ----------
    ndvar
        Sound data. Values should either be floating point numbers between -1
        and +1, or 16 bit integers.
    filename
        Where to save. If unspecified a file dialog will ask for the location.
    toint
        Convert floating point data to 16 bit integer (default False).

    Notes
    -----
    Uses :mod:`scipy.io.wavfile`.
    """
    from scipy.io import wavfile

    if ndvar.has_dim('channel'):
        data = ndvar.get_data(('time', 'channel'))
    else:
        data = ndvar.get_data('time')

    if toint and data.dtype != np.int16:
        above = data >= 2**15
        below = data < -2**15
        if np.any(above) or np.any(below):
            n = np.sum(above) + np.sum(below)
            print(f"WARNING: clipping {n} samples")
            data[above] = 2**15 - 1
            data[below] = -2**15

        data = data.astype(np.int16)
    elif data.dtype.kind != 'i' and (data.max() > 1. or data.min() < -1.):
        raise ValueError("Floating point data should be in range [-1, 1]. Set toint=True to save as 16 bit integer data.")

    if filename is None:
        msg = f"Save {ndvar.name}..."
        filename = ui.ask_saveas(msg, msg, FILETYPES)
    srate = int(round(1. / ndvar.time.tstep))
    wavfile.write(filename, srate, data)
