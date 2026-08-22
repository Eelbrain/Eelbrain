# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from eelbrain import gammatone_bank
from eelbrain.testing.data import generate_sound

pytest.importorskip('gammatone')


@pytest.mark.parametrize('location', ['left', 'right'])
def test_gammatone_bank(location):
    "Fixed integration window: compare with gtgram from the gammatone library"
    from gammatone.filters import centre_freqs
    from gammatone.gtgram import gtgram

    sound = generate_sound()
    fs = 1 / sound.time.tstep
    tstep, window = 1 / 100, 0.020
    n = 4  # each band is filtered separately, so more bands only make the test slower
    gt = gammatone_bank(sound, 20, 2000, n, tstep, integration_window=window, location=location)
    # gtgram integrates over the window starting at the output sample; for location='right' the window ends at the output sample, which is equivalent to delaying the input (leading zeros only delay the filterbank response)
    pad = round(window * fs) - 1 if location == 'right' else 0
    wave = numpy.concatenate([numpy.zeros(pad), sound.get_data('time')])
    target = gtgram(wave, fs, window, tstep, n, 20, 2000)
    # gtgram stops at the last full window, gammatone_bank continues to the end of the sound
    assert_allclose(gt.get_data(('frequency', 'time'))[:, :target.shape[1]], target)
    assert_array_equal(gt.frequency.values, centre_freqs(fs, n, 20, 2000)[::-1])
    assert gt.time.tmin == sound.time.tmin
    assert gt.time.tstep == tstep
    assert gt.time.nsamples == 200

    # Frequency-dependent integration window: compare with single-band gtgram
    gt = gammatone_bank(sound, 20, 2000, n, tstep, location=location)
    data = gt.get_data(('frequency', 'time'))
    for i, cf in enumerate(gt.frequency.values):
        n_window = int(numpy.ceil(2 / cf * fs))  # default: 2 cycles of the centre frequency
        pad = n_window - 1 if location == 'right' else 0
        wave = numpy.concatenate([numpy.zeros(pad), sound.get_data('time')])
        target = gtgram(wave, fs, n_window / fs, tstep, 1, cf, cf)[0]  # with a single channel, gtgram uses f_min as the centre frequency
        assert_allclose(data[i, :len(target)], target, err_msg=f"{cf=}")
