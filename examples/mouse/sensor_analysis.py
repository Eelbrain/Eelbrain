# skip test: data unavailable
from eelbrain import *

from mouse import e


res = e.load_test('surprise', 0.3, 0.5, 0.05, data='sensor', baseline=False, epoch='target', raw='fastica1-40')
plot.TopoButterfly(res)
# plot.TopoArray(res)
# plot.TopomapBins(res)  # fixme: layout
