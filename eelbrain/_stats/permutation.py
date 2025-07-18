# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import chain, repeat
from math import ceil, pi, sin
import random

import numpy as np

from .._data_obj import NDVar, Var, NestedEffect
from .._utils import intervals
from . import vector


_YIELD_ORIGINAL = 0
# for testing purposes, yield original order instead of permutations


def _resample_params(n: int, samples: int) -> (int, int):
    """Decide whether to do permutations or random resampling

    Parameters
    ----------
    n
        Number of observations.
    samples
        ``samples`` parameter (number of resampling iterations, or < 0 to
        sample all permutations).

    Returns
    -------
    actual_n_samples : int
        Adapted number of resamplings that will be done.
    samples_param : int
        Samples parameter for the resample function (-1 to do all permutations,
        otherwise same as n_samples).
    """
    n_perm = 2 ** n
    if n_perm - 1 <= samples:
        samples = -1

    if samples < 0:
        n_samples = n_perm - 1
    else:
        n_samples = samples

    return n_samples, samples


def permute_order(n, samples=10000, replacement=False, unit=None, rng=None):
    """Generator function to create indices to shuffle n items

    Parameters
    ----------
    n : int
        Number of cases.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    replacement : bool
        whether random samples should be drawn with replacement or without.
    unit : categorial
        Factor specifying unit of measurement (e.g. subject). If unit is
        specified, resampling proceeds by first resampling the categories of
        unit (with or without replacement) and then shuffling the values
        within units (no replacement).
    rng : numpy.random.RandomState
        Random number generator. By default, a random state with seed 0 is used.

    Returns
    -------
    Iterator over index.
    """
    n = int(n)
    samples = int(samples)
    if samples < 0:
        raise NotImplementedError("Complete permutation for resampling through reordering")

    if _YIELD_ORIGINAL:
        original = np.arange(n)
        for _ in range(samples):
            yield original
        return

    if rng is None:
        rng = np.random.RandomState(0)

    if unit is None or unit is False:
        if replacement:
            for _ in range(samples):
                yield rng.randint(n, n)
        else:
            index = np.arange(n)
            for _ in range(samples):
                rng.shuffle(index)
                yield index
    else:
        if replacement:
            raise NotImplementedError("Replacement and units")
        idx_orig = np.arange(n)
        idx_perm = np.empty_like(idx_orig)
        unit_idxs = [np.flatnonzero(unit == cell) for cell in unit.cells]
        if isinstance(unit, NestedEffect):
            dst_idxs_iter = ((unit_idxs[i] for i in order)
                             for order in permute_order(len(unit_idxs), samples, rng=rng))
        else:
            dst_idxs_iter = repeat(unit_idxs, samples)

        for dst_idxs in dst_idxs_iter:
            for src, dst in zip(unit_idxs, dst_idxs):
                v = idx_orig[src]
                rng.shuffle(v)
                idx_perm[dst] = v
            yield idx_perm


def permute_sign_flip(n, samples=10000, rng=None, out=None):
    """Iterate over indices for ``samples`` permutations of the data

    Parameters
    ----------
    n : int
        Number of cases.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    rng : random.Random
        Random number generator.
    out : array of int8  (n,)
        Buffer for the ``sign`` variable that is yielded in each iteration.

    Yields
    ------
    sign : array of int8  (n,)
        Sign for each case (``1`` or ``-1``; ``sign`` is the same array object
        but its content modified in every iteration).
    """
    n = int(n)
    if rng is None and samples >= 0:
        rng = random.Random(0)

    if out is None:
        out = np.empty(n, np.int8)
    else:
        assert out.shape == (n,)

    if n > 62:  # Python 2 limit for xrange
        if samples < 0:
            raise NotImplementedError("All possibilities for more than 62 cases")
        n_groups = ceil(n / 62.)
        group_size = int(ceil(n / n_groups))
        out_parts = chain(range(0, n, group_size), [n])
        for _ in zip(*(permute_sign_flip(stop - start, samples, rng, out[start: stop])
                       for start, stop in intervals(out_parts))):
            yield out
        return

    # determine possible number of permutations
    n_perm_possible = 2 ** n
    if samples < 0:
        # do all permutations
        sample_sequences = range(1, n_perm_possible)
    else:
        # random resampling
        sample_sequences = rng.sample(range(1, n_perm_possible), samples)

    for seq in sample_sequences:
        out.fill(1)
        for i in (i for i, s in enumerate(bin(seq)[-1:1:-1]) if s == '1'):
            out[i] = -1
        yield out


def resample(y, samples=10000, replacement=False, unit=None):
    """
    Generator function to resample a dependent variable (y) multiple times

    Parameters
    ----------
    y : Var | NDVar
        Variable which is to be resampled.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    replacement : bool
        whether random samples should be drawn with replacement or without.
    unit : categorial
        Factor specifying unit of measurement (e.g. subject). If unit is
        specified, resampling proceeds by first resampling the categories of
        unit (with or without replacement) and then shuffling the values
        within units (no replacement).

    Returns
    -------
    Iterator over Y_resampled. One copy of ``y`` is made, and this copy is
    yielded in each iteration with shuffled data.
    """
    if isinstance(y, Var):
        pass
    elif isinstance(y, NDVar):
        if not y.has_case:
            raise ValueError("Need NDVar with cases")
    else:
        raise TypeError("Need Var or NDVar")

    out = y.copy(f'{y.name}_resampled')

    for index in permute_order(len(out), samples, replacement, unit):
        out.x[index] = y.x
        yield out


def random_seeds(samples: int):
    """Sequence of seeds for permutation based on random numbers

    Parameters
    ----------
    samples
        Number of samples to yield.

    Returns
    -------
    samples : array of uint32  (samples,)
        A random unsigned integer to serve as seed.
    """
    rng = np.random.RandomState(0)
    return rng.randint(2**32, size=samples, dtype=np.uint32)


def _sample_xi_by_rejection(n: int, rng: np.random.RandomState):
    """Return a sample (or samples) from the distribution p(x) = 2 * np.sin(x/2) ** 2 / pi

    See [1]_ for why samples from this distribution is required to sample
    random rotation matrices.

    ..[1] Miles, R. E. (1965). On random rotations in R^3. Biometrika, 52(3/4), 636-639.

    Parameters
    ----------
    n
        Number of the samples.
    rng
        Random generator.

    Returns
    -------
    ndarray
        samples drawn from the distribution
    """
    samples = np.empty(n)
    i = 0
    while i < n:
        z = rng.random() * pi
        u = rng.random() * 2 / pi

        if u <= 2 * sin(z / 2) ** 2 / pi:
            samples[i] = z
            i += 1

    return samples


def rand_rotation_matrices(n: int, seed: int):
    """Function to create random rotation matrices in 3D

    Parameters
    ----------
    n : int
        Number of rotation matrices to return.
    seed : int
        Seed the random state.

    Returns
    -------
    rotation : array (n, 3, 3)
        Sampled rotation matrices.
    """
    rng = np.random.RandomState(seed)
    phi = np.arccos(rng.uniform(-1, 1, n))
    theta = rng.uniform(0, 2 * pi, n)
    xi = _sample_xi_by_rejection(n, rng)
    return vector.rotation_matrices(phi, theta, xi, np.empty((n, 3, 3)))
