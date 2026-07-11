# skip test: data unavailable
from eelbrain import *

from mouse import e


res = e.load_test('surprise', 0.3, 0.5, 0.05, epoch='target', raw='fastica1-40', inv='fixed-3-dSPM', mask='frontotemporal-lh')
plot.brain.butterfly(res)
