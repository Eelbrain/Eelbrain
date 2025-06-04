from pathlib import Path

import numpy

from .. import NDVar, UTS


TEST_DATA_DIRECTORY = Path(__file__).parents[2] / 'test_data'


def generate_sound():
    sample_rate = 44100
    duration = 2

    carrier_frequencies = [440, 880, 1320, 1760]
    modulator_frequencies = [2, 3, 5, 7]
    modulation_indices = [50, 100, 150, 200]

    t = numpy.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    modulators = [numpy.sin(2 * numpy.pi * mod_freq * t) for mod_freq in modulator_frequencies]
    signal = sum(numpy.sin(2 * numpy.pi * (carrier_freq + modulation_index * modulator) * t)
                 for carrier_freq, modulation_index, modulator in
                 zip(carrier_frequencies, modulation_indices, modulators))

    signal /= numpy.max(signal)
    return NDVar(signal, UTS(0, 1 / sample_rate, duration * sample_rate))
