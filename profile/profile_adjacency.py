import cProfile
import os
import pstats

import mne
import numpy as np
import eelbrain
from eelbrain import *

fname = 'profile_of_adjacency.profile'

mne.set_log_level('warning')
configure(n_workers=False)

ds = datasets.get_mne_sample(-0.1, 0.2, src='ico', sub="modality == 'A'")

code = '''
res = testnd.TTestIndependent('src', 'side', 'L', 'R', ds=ds, samples=5, pmin=0.5,
                       tstart=0.05, mintime=0.02, minsource=10)
'''

cProfile.run(code, fname)

p = pstats.Stats(fname)
p.sort_stats('cumulative')
p.print_stats(20)
