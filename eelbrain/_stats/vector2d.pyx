# Author: Proloy Das <proloyd94@gmail.com>
# cython: boundscheck=False, wraparound=False, language_level=3
# distutils: language = c++
"""
optimized statistics functions

"""
cimport cython
from libc.math cimport sin, cos
cimport numpy as np


cdef double r_TOL = 2.220446049250313e-16

@cython.cdivision(True)
def t2_stat(
        const np.npy_float64[:,:,:] y,
        np.npy_float64[:] out,
):
    cdef unsigned long i, v, u, case
    cdef double norm, temp, max_eig, TOL

    cdef double mean[2]
    cdef double sigma[2][2]

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        # Initialization
        norm = 0
        for v in range(n_dims):
            mean[v] = 0.0
            for u in range(n_dims):
                sigma[u][v] = 0.0
        # Computation
        for case in range(n_cases):
            for v in range(n_dims):
                temp = y[case, v, i]
                mean[v] += temp
                for u in range(v + 1):
                    sigma[u][v] += temp * y[case, u, i]
        for v in range(n_dims):
            for u in range(v + 1):
                sigma[u][v] -= mean[u] * mean[v] / n_cases
                sigma[u][v] /= (n_cases - 1)
        # check non-zero variance
        for v in range(n_dims):
            if sigma[v][v] != 0:
                break
        else:
            out[i] = 0
            continue

        det_sigma = sigma[0][0] * sigma[1][1] - sigma[0][1]**2
        norm = mean[0] * mean[0] * sigma[1][1]
        norm += mean[1] * mean[1] * sigma[0][0]
        norm -= 2 * mean[0] * mean[1] * sigma[0][1]
        norm /= (det_sigma + r_TOL)
        out[i] = norm / n_cases
    return out


@cython.cdivision(True)
def t2_stat_rotated(
        const np.npy_float64[:,:,:] y,
        const np.npy_float64[:,:,:] rotation,
        np.npy_float64[:] out,
):
    cdef unsigned long i, v, u, case, vi
    cdef double norm, temp, TOL, max_eig

    cdef double mean[2]
    cdef double tempv[2]
    cdef double sigma[2][2]

    cdef unsigned long n_cases = y.shape[0]
    cdef unsigned long n_dims = y.shape[1]
    cdef unsigned long n_tests = y.shape[2]

    for i in range(n_tests):
        # Initialization
        norm = 0
        for v in range(n_dims):
            mean[v] = 0.0
            for u in range(n_dims):
                sigma[u][v] = 0.0
        # Computation
        for case in range(n_cases):
            # rotation
            for u in range(n_dims):
                tempv[u] = 0
                for vi in range(n_dims):
                    tempv[u] += rotation[case, u, vi] * y[case, vi, i]
                mean[u] += tempv[u]
                for v in range(u + 1):      # Only upper triangular part is meaningful
                    sigma[v][u] += tempv[u] * tempv[v]
        for u in range(n_dims):
            for v in range(u + 1):      # Only upper triangular part is meaningful
                sigma[v][u] -= mean[u] * mean[v] / n_cases
                sigma[v][u] /=  (n_cases - 1)
        # check non-zero variance
        for v in range(n_dims):
            if sigma[v][v] != 0:
                break
        else:
            out[i] = 0
            continue

        det_sigma = sigma[0][0] * sigma[1][1] - sigma[0][1]**2
        norm = mean[0] * mean[0] * sigma[1][1]
        norm += mean[1] * mean[1] * sigma[0][0]
        norm -= 2 * mean[0] * mean[1] * sigma[0][1]
        norm /= (det_sigma + r_TOL)
        out[i] = norm / n_cases
    return out
