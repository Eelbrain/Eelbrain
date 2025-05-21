import numpy as np
from scipy import ndimage
import mne
from eelbrain._stats import testnd as tnd
from eelbrain import *

mne.set_log_level('warning')


ds = datasets.get_mne_sample(-0.1, 1.0, src='ico', sub="[0]")
src = ds['src']
bin_map = (src.abs()[0] > 1).x
out = np.empty(bin_map.shape, np.uint32)
struct = ndimage.generate_binary_structure(2, 1)
struct[::2] = False
conn = src.source.adjacency()
criteria = None

print("tnd._label_clusters_binary(bin_map, out, struct, False, bin_map.shape, conn, criteria)")
