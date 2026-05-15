import numpy as np
from eelbrain import *
import eelbrain

ds = datasets.get_uts(True)
y = ds['utsnd'].x
n_cases = len(y)
n_tests = np.product(y.shape[1:])
y = y.reshape((n_cases, n_tests))
m = ds.eval("A*B*rm")
p = m._parametrize()
fmap = np.empty((3, n_tests))
e_ms = eelbrain._stats.glm._hopkins_ems_array(m)

print("timeit -n1000 opt.anova_full_fmaps(y, p.x, p.projector, fmap, m._effect_to_beta, e_ms)")
