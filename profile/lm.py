import numpy as np

np.random.seed(0)

n_cases = 100
n_tests = 1000
n_betas = 3
n_fs = 2

# lm
y = np.random.normal(0, 1, (n_cases, n_tests))
x = np.random.normal(0, 1, (n_cases, n_betas))
xsinv = np.random.normal(0, 1, (n_betas, n_cases))
betas = np.empty((n_tests, n_betas))
ss = np.empty(n_tests)
perm = np.arange(n_cases)
np.random.shuffle(perm)

# bigger arrays
n_testsl = 10**6
yl = np.random.normal(0, 1, (n_cases, n_testsl))
betasl = np.empty((n_testsl, n_betas))
ssl = np.empty(n_testsl)


# ANOVA
f_map = np.empty((n_fs, n_tests))
effects = np.array([[0, 2], [2, 1]], dtype=np.int16)
df_res = n_cases - n_betas - 1

print("n_cases=%i; n_tests=%i; n_betas=%i" % (n_cases, n_tests, n_betas))
print("timeit opt.lm_betas(y, x, xsinv, betas)")
print("timeit -n1000 opt.lm_res_ss(y, x, xsinv, ss)")
print("timeit -n1000 opt._anova_fmaps(y, x, xsinv, f_map, effects, df_res)")
print("timeit opt._ss(y, ss)")
