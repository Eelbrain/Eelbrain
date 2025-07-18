'''Statistical tests for NDVars

Common Attributes
-----------------

The following attributes are always present. For ANOVA, they are lists with the
corresponding items for different effects.

t/f/... : NDVar
    Map of the statistical parameter.
p_uncorrected : NDVar
    Map of uncorrected p values.
p : NDVar | None
    Map of corrected p values (None if no correct was applied).
clusters : Dataset | None
    Table of all the clusters found (None if no clusters were found, or if no
    clustering was performed).
n_samples : None | int
    The actual number of permutations. If ``samples = -1``, i.e. a complete set
    or permutations is performed, then ``n_samples`` indicates the actual
    number of permutations that constitute the complete set.
'''
from datetime import datetime, timedelta
from functools import cached_property, reduce, partial
from itertools import chain, repeat
from math import ceil
from multiprocessing.shared_memory import SharedMemory
import logging
import operator
import os
import re
import socket
from threading import Thread
from time import time as current_time
from typing import Iterable, List, Union

import numpy as np
import scipy.stats
from scipy import ndimage

from .. import fmtxt, _info, _text
from ..fmtxt import FMText
from .._celltable import Celltable
from .._config import CONFIG, mpc
from .._data_obj import (
    CategorialArg, CellArg, IndexArg, ModelArg, NDVarArg, VarArg,
    Dataset, Var, Factor, Interaction, NestedEffect,
    NDVar, Categorial, UTS,
    ascategorial, asmodel, asndvar, asvar, assub,
    cellname, combine, dataobj_repr, longname)
from .._exceptions import OldVersionError, WrongDimensionError, ZeroVarianceError
from .._utils import deprecate_ds_arg, user_activity, restore_main_spec
from .._utils.numpy_utils import FULL_AXIS_SLICE
from .._utils.notebooks import trange
from . import opt, stats, vector
from .adjacency import Adjacency, find_peaks
from .adjacency_opt import merge_labels, tfce_increment
from .glm import MPTestMapper, _nd_anova
from .permutation import (
    _resample_params, permute_order, permute_sign_flip, random_seeds,
    rand_rotation_matrices)
from .t_contrast import TContrastSpec
from .test import star, star_factor, _independent_measures_args, _related_measures_args


__test__ = False


def check_for_vector_dim(y: NDVar) -> None:
    for dim in y.dims:
        if dim._adjacency_type == 'vector':
            raise WrongDimensionError(f"{dim}: mass-univariate methods are not suitable for vectors. Consider using vector norm as test statistic, or using a testnd.Vector test function.")


def check_variance(x):
    if x.ndim != 2:
        x = x.reshape((len(x), -1))
    if opt.has_zero_variance(x):
        raise ZeroVarianceError("y contains data column with zero variance")


class NDTest:
    """Baseclass for testnd test results

    Attributes
    ----------
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    """
    _state_common = ('y', 'match', 'sub', 'samples', 'tfce', 'pmin', '_cdist', 'tstart', 'tstop', '_dims')
    _state_specific = ()
    _statistic = None
    _statistic_tail = 0

    @property
    def _attributes(self):
        return self._state_common + self._state_specific

    def __init__(self, y, match, sub, samples, tfce, pmin, cdist, tstart, tstop):
        self.y = dataobj_repr(y)
        self.match = dataobj_repr(match, True)
        self.sub = sub
        self.samples = samples
        self.tfce = tfce
        self.pmin = pmin
        self._cdist = cdist
        self.tstart = tstart
        self.tstop = tstop
        self._dims = y.dims[1:]

    def __getstate__(self):
        return {name: getattr(self, name) for name in self._attributes}

    def __setstate__(self, state):
        # backwards compatibility:
        if 'Y' in state:
            state['y'] = state.pop('Y')
        if 'X' in state:
            state['x'] = state.pop('X')

        for name in self._attributes:
            setattr(self, name, state.get(name))

        # backwards compatibility:
        if 'tstart' not in state:
            cdist = self._first_cdist
            self.tstart = cdist.tstart
            self.tstop = cdist.tstop
        if '_dims' not in state:  # 0.17
            if 't' in state:
                self._dims = state['t'].dims
            elif 'r' in state:
                self._dims = state['r'].dims
            elif 'f' in state:
                self._dims = state['f'][0].dims
            else:
                raise RuntimeError("Error recovering old test results dims")

        self._expand_state()

    def __repr__(self):
        args = self._repr_test_args()
        if self.sub is not None:
            if isinstance(self.sub, np.ndarray):
                sub_repr = '<array>'
            else:
                sub_repr = repr(self.sub)
            args.append(f'sub={sub_repr}')
        if self._cdist:
            args += self._repr_cdist()
        else:
            args.append('samples=0')

        return f"<{self.__class__.__name__} {', '.join(args)}>"

    def _repr_test_args(self):
        """List of strings describing parameters unique to the test

        Will be joined with ``", ".join(repr_args)``
        """
        raise NotImplementedError()

    def _repr_cdist(self):
        """List of results (override for MultiEffectResult)"""
        return self._cdist._repr_test_args(self.pmin) + self._cdist._repr_clusters()

    def _expand_state(self):
        "Override to create secondary results"
        cdist = self._cdist
        if cdist is None:
            self.tfce_map = None
            self.p = None
            self._kind = None
        else:
            self.tfce_map = cdist.tfce_map
            self.p = cdist.probability_map
            self._kind = cdist.kind

    def _desc_samples(self):
        if self.samples == -1:
            return f"a complete set of {self.n_samples} permutations"
        elif self.samples is None:
            return "no permutations"
        else:
            return f"{self.n_samples} random permutations"

    def _desc_timewindow(self):
        tstart = self._time_dim.tmin if self.tstart is None else self.tstart
        tstop = self._time_dim.tstop if self.tstop is None else self.tstop
        return f"{_text.ms(tstart)} - {_text.ms(tstop)} ms"

    def _asfmtext(self, **_):
        max_stat, p = self._max_statistic(return_p=True)
        return FMText([fmtxt.eq(self._statistic, max_stat, 'max', stars=p), ', ', fmtxt.peq(p)])

    def _default_plot_obj(self):
        raise NotImplementedError

    def _iter_cdists(self):
        yield None, self._cdist

    @property
    def _first_cdist(self):
        return self._cdist

    @property
    def permutation_distribution(self):
        if self._cdist is not None:
            return self._cdist.dist

    def _plot_model(self):
        "Determine x for plotting categories"
        return None

    def _plot_sub(self):
        if isinstance(self.sub, str) and self.sub == "<unsaved array>":
            raise RuntimeError("The sub parameter was not saved for previous versions of Eelbrain. Please recompute this result with the current version.")
        return self.sub

    def _assert_has_cdist(self):
        if self._cdist is None:
            raise RuntimeError("This method only applies to results of tests with threshold-based clustering and tests with a permutation distribution (samples > 0)")

    def masked_parameter_map(self, pmin: float = 0.05, **sub) -> NDVar:
        """Statistical parameter map masked by significance

        Parameters
        ----------
        pmin
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, ``pmin=1`` includes all clusters regardless of their
            p-value.

        Returns
        -------
        NDVar
            Parameter map with a mask that is ``True`` (masked) where p > pmin.
        """
        self._assert_has_cdist()
        return self._cdist.masked_parameter_map(pmin, **sub)

    def cluster(self, cluster_id: int) -> NDVar:
        """Retrieve a specific cluster as NDVar

        Parameters
        ----------
        cluster_id
            Cluster id.

        Returns
        -------
        cluster
            NDVar of the cluster, 0 outside the cluster.

        Notes
        -----
        Clusters only have stable ids for thresholded cluster distributions.
        """
        self._assert_has_cdist()
        return self._cdist.cluster(cluster_id)

    @cached_property
    def clusters(self):
        if self._cdist is None:
            return None
        else:
            return self.find_clusters(None, True)

    def find_clusters(
            self,
            pmin: float = None,
            maps: bool = False,
            **sub,
    ) -> Dataset:
        """Find significant regions or clusters

        Parameters
        ----------
        pmin
            Threshold p-value. For threshold-based tests, all clusters with a
            p-value smaller than ``pmin`` are included (default 1);
            for other tests, find contiguous regions with ``p ≤ pmin`` (default
            0.05).
        maps
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default ``False``).

        Returns
        -------
        Dataset
            Dataset with information about the clusters.
        """
        self._assert_has_cdist()
        return self._cdist.clusters(pmin, maps, **sub)

    def find_peaks(self) -> Dataset:
        """Find peaks in a threshold-free cluster distribution

        Returns
        -------
        Dataset
            Dataset with information about the peaks.
        """
        self._assert_has_cdist()
        return self._cdist.find_peaks()

    def compute_probability_map(self, **sub) -> NDVar:
        """Compute a probability map

        Returns
        -------
        NDVar
            Map of p-values.
        """
        self._assert_has_cdist()
        return self._cdist.compute_probability_map(**sub)

    def info_list(self, computation: bool = True) -> fmtxt.List:
        "List with information about the test"
        out = fmtxt.List("Mass-univariate statistics:")
        out.add_item(self._name())
        dimnames = [dim.name for dim in self._dims]
        dimlist = out.add_sublist(f"Over {_text.enumeration(dimnames)}")
        if 'time' in dimnames:
            dimlist.add_item(f"Time interval: {self._desc_timewindow()}.")

        cdist = self._first_cdist
        if cdist is None:
            out.add_item("No inferential statistics")
            return out

        # inference
        l = out.add_sublist("Inference:")
        if cdist.kind == 'raw':
            l.add_item("Based on maximum statistic")
        elif cdist.kind == 'tfce':
            l.add_item("Based on maximum statistic with threshold-"
                       "free cluster enhancement (Smith & Nichols, 2009)")
        elif cdist.kind == 'cluster':
            l.add_item("Based on maximum cluster mass statistic")
            sl = l.add_sublist("Cluster criteria:")
            for dim in dimnames:
                if dim == 'time':
                    sl.add_item(f"Minimum cluster duration {_text.ms(cdist.criteria.get('mintime', 0))} ms")
                elif dim == 'source':
                    sl.add_item(f"At least {cdist.criteria.get('minsource', 0)} contiguous sources.")
                elif dim == 'sensor':
                    sl.add_item(f"At least {cdist.criteria.get('minsensor', 0)} contiguous sensors.")
                else:
                    value = cdist.criteria.get(f'min{dim}', 0)
                    sl.add_item(f"Minimum number of contiguous elements in {dim}: {value}")
        # n samples
        l.add_item(f"In {self._desc_samples()}")

        # computation
        if computation:
            out.add_item(cdist.info_list())

        return out

    @property
    def _statistic_map(self):
        return getattr(self, self._statistic)

    def _max_statistic(
            self,
            mask: NDVar = None,
            return_time: bool = False,
            return_p: bool = False,
            sub: dict = None,
    ):
        tail = getattr(self, 'tail', self._statistic_tail)
        return self._max_statistic_from_map(self._statistic_map, self.p, tail, mask, return_time, return_p, sub)

    @staticmethod
    def _max_statistic_from_map(
            stat_map: NDVar,
            p_map: NDVar,
            tail: int,
            mask: NDVar = None,
            return_time: bool = False,
            return_p: bool = False,
            sub: dict = None,
    ):
        if sub:
            stat_map = stat_map.sub(**sub)
            if stat_map.x.size == 0:
                return (None,) * (1 + return_p + return_time)
            p_map = p_map.sub(**sub)
            if mask is not None:
                mask = mask.sub(**sub)

        if mask is None and p_map.min() <= .05:
            mask = p_map <= .05

        if tail == 0:
            max_stat = stat_map.extrema(mask)
        elif tail == 1:
            max_stat = stat_map.max(mask)
        else:
            max_stat = stat_map.min(mask)

        out = [max_stat]
        if return_p:
            out.append(p_map.min())
        if return_time:
            if mask is not None:
                stat_map = stat_map.mask(~mask, missing=True)
            if dims := [dim for dim in stat_map.dimnames if dim != 'time']:
                if max_stat > 0:
                    stat_map = stat_map.max(dims)
                else:
                    stat_map = stat_map.min(dims)
            if max_stat > 0:
                time = stat_map.argmax('time')
            else:
                time = stat_map.argmin('time')
            out.append(time)
        return out if len(out) > 1 else out[0]

    @property
    def n_samples(self):
        if self.samples == -1:
            return self._first_cdist.samples
        else:
            return self.samples

    @property
    def _time_dim(self):
        for dim in self._first_cdist.dims:
            if isinstance(dim, UTS):
                return dim
        return None


class TContrastRelated(NDTest):
    """Mass-univariate contrast based on t-values

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : categorial
        Model containing the cells which are compared with the contrast.
    contrast : str
        Contrast specification: see Notes.
    match : Factor
        Match cases for a repeated measures test.
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    samples : int
        Number of samples for permutation test (default 10,000).
    pmin : None | scalar (0 < pmin < 1)
        Threshold for forming clusters:  use a t-value equivalent to an
        uncorrected p-value for a related samples t-test (with df =
        len(match.cells) - 1).
    tmin : scalar
        Threshold for forming clusters as t-value.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    min...
        Minimum cluster size criteria: ``min`` followed by the simension name,
        for example:
        ``mintime=0.050`` for minimum duration of 50 ms;
        ``minsource=10`` to require at least 10 sources;
        ``minsensor=10`` to requre at least 10 sensors).

    See Also
    --------
    testnd : Information on the different permutation methods

    Notes
    -----
    A contrast specifies the steps to calculate a map based on *t*-values.
    Contrast definitions can contain:

    - Comparisons using ``>`` or ``<`` and data cells to compute *t*-maps.
      For example, ``"cell1 > cell0"`` will compute a *t*-map of the comparison
      if ``cell1`` and ``cell0``, being positive where ``cell1`` is greater than
      ``cell0`` and negative where ``cell0`` is greater than ``cell1``.
      If the data is defined based on an interaction, cells are specified with
      ``|``, e.g. ``"a1 | b1 > a0 | b0"``. Cells can contain ``*`` to average
      multiple cells. Thus, if the second factor in the model has cells ``b1``
      and ``b0``, ``"a1 | * > a0 | *"`` would compare ``a1`` to ``a0``
      while averaging ``b1`` and ``b0`` within ``a1`` and ``a0``.
    - Unary numpy functions ``abs`` and ``negative``, e.g.
      ``"abs(cell1 > cell0)"``.
    - Binary numpy functions ``subtract`` and ``add``, e.g.
      ``"add(a>b, a>c)"``.
    - Numpy functions for multiple arrays ``min``, ``max`` and ``sum``,
      e.g. ``min(a>d, b>d, c>d)``.

    Cases with zero variance are set to t=0.

    Examples
    --------
    To find cluster where both of two pairwise comparisons are reliable,
    i.e. an intersection of two effects, one could use
    ``"min(a > c, b > c)"``.

    To find a specific kind of interaction, where a is greater than b, and
    this difference is greater than the difference between c and d, one
    could use ``"(a > b) - abs(c > d)"``.
    """
    _state_specific = ('x', 'contrast', 't', 'tail')
    _statistic = 't'

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: CategorialArg,
            contrast: str,
            match: CategorialArg = None,
            sub: CategorialArg = None,
            data: Dataset = None,
            tail: int = 0,
            samples: int = 10000,
            pmin: float = None,
            tmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            **criteria):
        if match is None:
            raise TypeError("The `match` parameter needs to be specified for TContrastRelated")
        ct = Celltable(y, x, match, sub, data=data, coercion=asndvar, dtype=np.float64)
        check_for_vector_dim(ct.y)
        check_variance(ct.y.x)

        # setup contrast
        t_contrast = TContrastSpec(contrast, ct.cells, ct.data_indexes)

        # original data
        tmap = t_contrast.map(ct.y.x)

        n_threshold_params = sum((pmin is not None, tmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            threshold = cdist = None
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, tmin and tfce can be specified")
        else:
            if pmin is not None:
                df = len(ct.match.cells) - 1
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            else:
                threshold = None

            cdist = NDPermutationDistribution(
                ct.y, samples, threshold, tfce, tail, 't', "t-contrast",
                tstart, tstop, criteria, parc, force_permutation)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_order(len(ct.y), samples, unit=ct.match)
                run_permutation(t_contrast, cdist, iterator)

        # NDVar map of t-values
        info = _info.for_stat_map('t', threshold, tail=tail, old=ct.y.info)
        t = NDVar(tmap, ct.y.dims[1:], 't', info)

        # store attributes
        NDTest.__init__(self, ct.y, ct.match, sub, samples, tfce, pmin, cdist,
                        tstart, tstop)
        self.x = ('%'.join(ct.x.base_names) if isinstance(ct.x, Interaction) else
                  ct.x.name)
        self.contrast = contrast
        self.tail = tail
        self.tmin = tmin
        self.t = t

        self._expand_state()

    def _name(self):
        if self.y:
            return "T-Contrast:  %s ~ %s" % (self.y, self.contrast)
        else:
            return "T-Contrast:  %s" % self.contrast

    def _plot_model(self):
        return self.x

    def _repr_test_args(self):
        args = [repr(self.y), repr(self.x), repr(self.contrast)]
        if self.tail:
            args.append("tail=%r" % self.tail)
        if self.match:
            args.append('match=%r' % self.match)
        return args


class Correlation(NDTest):
    """Mass-univariate Pearson correlation significance test

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : continuous
        The continuous predictor variable.
    norm : None | categorial
        Categories in which to normalize (z-score) x.
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    samples : int
        Number of samples for permutation test (default 10,000).
    pmin : None | scalar (0 < pmin < 1)
        Threshold for forming clusters:  use an r-value equivalent to an
        uncorrected p-value.
    rmin : None | scalar
        Threshold for forming clusters.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    match : None | categorial
        When permuting data, only shuffle the cases within the categories
        of match.
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    p_uncorrected : NDVar
        Map of p-values uncorrected for multiple comparison.
    r : NDVar
        Map of correlation values (with threshold contours).
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).

    See Also
    --------
    testnd : Information on the different permutation methods
    """
    _state_specific = ('x', 'norm', 'n', 'df', 'r')
    _statistic = 'r'

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: VarArg,
            norm: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            samples: int = 10000,
            pmin: float = None,
            rmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            match: CategorialArg = None,
            parc: str = None,
            **criteria):
        sub, n = assub(sub, data, True)
        y, n = asndvar(y, sub, data, n, np.float64, True)
        check_for_vector_dim(y)
        if not y.has_case:
            raise ValueError("Dependent variable needs case dimension")
        x = asvar(x, sub, data, n)
        if norm is not None:
            norm = ascategorial(norm, sub, data, n)
        if match is not None:
            match = ascategorial(match, sub, data, n)

        self.x = x.name
        name = f"{longname(y)} ~ {longname(x)}"

        # Normalize by z-scoring the data for each subject
        # normalization is done before the permutation b/c we are interested in
        # the variance associated with each subject for the z-scoring.
        y = y.copy()
        if norm is not None:
            for cell in norm.cells:
                idx = (norm == cell)
                y.x[idx] = scipy.stats.zscore(y.x[idx], None)

        # subtract the mean from y and x so that this can be omitted during
        # permutation
        y -= y.summary('case')
        x = x - x.mean()

        n = len(y)
        df = n - 2

        rmap = stats.corr(y.x, x.x)

        n_threshold_params = sum((pmin is not None, rmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            threshold = cdist = None
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, rmin and tfce can be specified")
        else:
            if pmin is not None:
                threshold = stats.rtest_r(pmin, df)
            elif rmin is not None:
                threshold = abs(rmin)
            else:
                threshold = None

            cdist = NDPermutationDistribution(
                y, samples, threshold, tfce, 0, 'r', name,
                tstart, tstop, criteria, parc)
            cdist.add_original(rmap)
            if cdist.do_permutation:
                iterator = permute_order(n, samples, unit=match)
                run_permutation(stats.corr, cdist, iterator, x.x)

        # compile results
        info = _info.for_stat_map('r', threshold)
        r = NDVar(rmap, y.dims[1:], name, info)

        # store attributes
        NDTest.__init__(self, y, match, sub, samples, tfce, pmin, cdist, tstart, tstop)
        self.norm = None if norm is None else norm.name
        self.rmin = rmin
        self.n = n
        self.df = df
        self.r = r

        self._expand_state()

    def _expand_state(self):
        NDTest._expand_state(self)

        r = self.r

        # uncorrected probability
        pmap = stats.rtest_p(r.x, self.df)
        info = _info.for_p_map()
        p_uncorrected = NDVar(pmap, r.dims, 'p_uncorrected', info)
        self.p_uncorrected = p_uncorrected
        self.r_p = [[r, self.p]] if self.samples else None

    def _name(self):
        if self.y and self.x:
            return "Correlation:  %s ~ %s" % (self.y, self.x)
        else:
            return "Correlation"

    def _repr_test_args(self):
        args = [repr(self.y), repr(self.x)]
        if self.norm:
            args.append('norm=%r' % self.norm)
        return args

    def _default_plot_obj(self):
        if self.samples:
            return self.masked_parameter_map()
        else:
            return self.r


class NDDifferenceTest(NDTest):

    difference = None

    @staticmethod
    def _difference_name(diff):
        long_name = longname(diff, True)
        if longname is None or len(long_name) > 80:
            return 'difference'
        else:
            return long_name

    def _get_mask(self, p=0.05):
        self._assert_has_cdist()
        if not 1 >= p > 0:
            raise ValueError(f"{p=}: needs to be between 1 and 0")
        if p == 1:
            if self._cdist.kind != 'cluster':
                raise ValueError(f"{p=} is only a valid mask for threshold-based cluster tests")
            mask = self._cdist.cluster_map == 0
        else:
            mask = self.p > p
        return self._cdist.uncrop(mask, self.difference, True)

    def masked_difference(self, p=0.05, name=None):
        """Difference map masked by significance

        Parameters
        ----------
        p : scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, ``pmin=1`` includes all clusters regardless of their
            p-value.
        name : str
            Name of the output NDVar.
        """
        mask = self._get_mask(p)
        return self.difference.mask(mask, name=name)


class NDMaskedC1Mixin:

    def masked_c1(self, p=0.05):
        """``c1`` map masked by significance of the ``c1``-``c0`` difference

        Parameters
        ----------
        p : scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, ``pmin=1`` includes all clusters regardless of their
            p-value.
        """
        mask = self._get_mask(p)
        return self.c1_mean.mask(mask)


class TTestOneSample(NDDifferenceTest):
    """Mass-univariate one sample t-test

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    popmean : scalar
        Value to compare y against (default is 0).
    match : None | categorial
        Combine data for these categories before testing.
    sub : index
        Perform test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    samples : int
        Number of samples for permutation test (default 10,000).
    pmin : None | scalar (0 < pmin < 1)
        Threshold for forming clusters:  use a t-value equivalent to an
        uncorrected p-value.
    tmin : scalar
        Threshold for forming clusters as t-value.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.
    difference : NDVar
        The difference value entering the test (``y`` if popmean is 0).
    n : int
        Number of cases.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    p_uncorrected : NDVar
        Map of p-values uncorrected for multiple comparison.
    t : NDVar
        Map of t-values.
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).

    See Also
    --------
    testnd : Information on the different permutation methods

    Notes
    -----
    Data points with zero variance are set to t=0.
    """
    _state_specific = ('popmean', 'tail', 'n', 'df', 't', 'difference')
    _statistic = 't'

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            popmean: float = 0,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
            samples: int = 10000,
            pmin: float = None,
            tmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            **criteria):
        ct = Celltable(y, match=match, sub=sub, data=data, coercion=asndvar, dtype=np.float64)
        check_for_vector_dim(ct.y)

        n = len(ct.y)
        if n < 3:
            raise ValueError(f"{y=}: not enough cases for t-test")
        df = n - 1
        y = ct.y.summary()
        tmap = stats.t_1samp(ct.y.x)
        if popmean:
            raise NotImplementedError("popmean != 0")
            diff = y - popmean
            if np.any(diff < 0):
                diff.info['cmap'] = 'xpolar'
            diff.name = self._difference_name(diff)
        else:
            diff = y

        n_threshold_params = sum((pmin is not None, tmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            threshold = cdist = None
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, tmin and tfce can be specified")
        else:
            if pmin is not None:
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            else:
                threshold = None

            if popmean:
                y_perm = ct.y - popmean
            else:
                y_perm = ct.y
            n_samples, samples = _resample_params(len(y_perm), samples)
            cdist = NDPermutationDistribution(
                y_perm, n_samples, threshold, tfce, tail, 't', '1-Sample t-Test',
                tstart, tstop, criteria, parc, force_permutation)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_sign_flip(n, samples)
                run_permutation(opt.t_1samp_perm, cdist, iterator)

        # NDVar map of t-values
        info = _info.for_stat_map('t', threshold, tail=tail, old=ct.y.info)
        t = NDVar(tmap, ct.y.dims[1:], 't', info)

        # store attributes
        NDDifferenceTest.__init__(self, ct.y, ct.match, sub, samples, tfce, pmin, cdist, tstart, tstop)
        self.popmean = popmean
        self.n = n
        self.df = df
        self.tail = tail
        self.t = t
        self.tmin = tmin
        self.difference = diff
        self._expand_state()

    def __setstate__(self, state):
        if 'diff' in state:
            state['difference'] = state.pop('diff')
        NDTest.__setstate__(self, state)

    def _expand_state(self):
        NDTest._expand_state(self)

        t = self.t
        pmap = stats.ttest_p(t.x, self.df, self.tail)
        info = _info.for_p_map(t.info)
        p_uncorr = NDVar(pmap, t.dims, 'p', info)
        self.p_uncorrected = p_uncorr

    def _name(self):
        if self.y:
            return "One-Sample T-Test:  %s" % self.y
        else:
            return "One-Sample T-Test"

    def _repr_test_args(self):
        args = [repr(self.y)]
        if self.popmean:
            args.append(repr(self.popmean))
        if self.match:
            args.append('match=%r' % self.match)
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args

    def _default_plot_obj(self):
        if self.samples:
            return self.masked_difference()
        else:
            return self.difference


class TTestIndependent(NDDifferenceTest):
    """Mass-univariate independent samples t-test

    The test data can be specified in two forms:

     - In long form, with ``y`` supplying the data, ``x`` specifying condition
       for each case.
     - With ``y`` and ``x`` supplying data for the two conditions.

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : categorial | NDVar
        Model containing the cells which should be compared, or NDVar to which
        ``y`` should be compared. In the latter case, the next three parameters
        are ignored.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Combine cases with the same cell on ``x % match``.
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    samples : int
        Number of samples for permutation test (default 10,000).
    pmin : None | scalar (0 < pmin < 1)
        Threshold p value for forming clusters. None for threshold-free
        cluster enhancement.
    tmin : scalar
        Threshold for forming clusters as t-value.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    c1_mean : NDVar
        Mean in the c1 condition.
    c0_mean : NDVar
        Mean in the c0 condition.
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.
    difference : NDVar
        Difference between the mean in condition c1 and condition c0.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    p_uncorrected : NDVar
        Map of p-values uncorrected for multiple comparison.
    t : NDVar
        Map of t-values.
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).

    See Also
    --------
    testnd : Information on the different permutation methods

    Notes
    -----
    Cases with zero variance are set to t=0.
    """
    _state_specific = ('x', 'c1', 'c0', 'tail', 't', 'n1', 'n0', 'df', 'c1_mean',
                       'c0_mean')
    _statistic = 't'

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: Union[CategorialArg, NDVarArg],
            c1: CellArg = None,
            c0: CellArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
            samples: int = 10000,
            pmin: float = None,
            tmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            **criteria):
        y, y1, y0, c1, c0, match, x_name, c1_name, c0_name = _independent_measures_args(y, x, c1, c0, match, data, sub, True)
        check_for_vector_dim(y)

        n1 = len(y1)
        n0 = len(y0)
        n = n1 + n0
        df = n - 2
        if df < 1:
            raise ValueError(f"Not enough cases for t-test: {n1=}, {n0=}, {df=}")

        groups = np.arange(n) < n1
        groups.dtype = np.int8
        tmap = stats.t_ind(y.x, groups)

        n_threshold_params = sum((pmin is not None, tmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            threshold = cdist = None
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, tmin and tfce can be specified")
        else:
            if pmin is not None:
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            else:
                threshold = None

            cdist = NDPermutationDistribution(y, samples, threshold, tfce, tail, 't', 'Independent Samples t-Test', tstart, tstop, criteria, parc, force_permutation)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_order(n, samples)
                run_permutation(stats.t_ind, cdist, iterator, groups)

        # store attributes
        NDDifferenceTest.__init__(self, y, match, sub, samples, tfce, pmin, cdist, tstart, tstop)
        self.x = x_name
        self.c0 = c0
        self.c1 = c1
        self.n1 = n1
        self.n0 = n0
        self.df = df
        self.tail = tail
        info = _info.for_stat_map('t', threshold, tail=tail, old=y.info)
        self.t = NDVar(tmap, y.dims[1:], 't', info)
        self.tmin = tmin
        self.c1_mean = y1.mean('case', name=cellname(c1_name))
        self.c0_mean = y0.mean('case', name=cellname(c0_name))
        self._expand_state()

    def _expand_state(self):
        NDTest._expand_state(self)

        # difference
        diff = self.c1_mean - self.c0_mean
        if np.any(diff.x < 0):
            diff.info['cmap'] = 'xpolar'
        diff.name = self._difference_name(diff)
        self.difference = diff

        # uncorrected p
        pmap = stats.ttest_p(self.t.x, self.df, self.tail)
        info = _info.for_p_map(self.t.info)
        p_uncorr = NDVar(pmap, self.t.dims, 'p', info)
        self.p_uncorrected = p_uncorr

        # composites
        if self.samples:
            diff_p = self.masked_difference()
        else:
            diff_p = self.difference
        self.all = [self.c1_mean, self.c0_mean, diff_p]

    def _name(self):
        cmp = '=><'[self.tail]
        desc = f"{self.c1} {cmp} {self.c0}"
        if self.y:
            desc = f"{self.y} ~ {desc}"
        return f"Independent-Samples T-Test:  {desc}"

    def _plot_model(self):
        return self.x

    def _plot_sub(self):
        return "(%s).isin(%s)" % (self.x, (self.c1, self.c0))

    def _repr_test_args(self):
        if self.c1 is None:
            args = [f'{self.y!r} (n={self.n1})', f'{self.x!r} (n={self.n0})']
        else:
            args = [f'{self.y!r}', f'{self.x!r}', f'{self.c1!r} (n={self.n1})', f'{self.c0!r} (n={self.n0})']
        if self.match:
            args.append(f'match{self.match!r}')
        if self.tail:
            args.append(f'tail={self.tail}')
        return args

    def _default_plot_obj(self):
        if self.samples:
            diff = self.masked_difference()
        else:
            diff = self.difference
        return [self.c1_mean, self.c0_mean, diff]


class TTestRelated(NDMaskedC1Mixin, NDDifferenceTest):
    """Mass-univariate related samples t-test

    The test data can be specified in two forms:

     - In long form, with ``y`` supplying the data, ``x`` specifying condition
       for each case and ``match`` determining which cases are related.
     - In wide/repeated measures form, with ``y`` and ``x`` both supplying data
       with matching case order.

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : categorial | NDVar
        Model containing the cells which should be compared, or NDVar to which
        ``y`` should be compared. In the latter case, the next three parameters
        are ignored.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison).
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed, default);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    samples
        Number of samples for permutation test (default 10,000).
    pmin
        Threshold for forming clusters:  use a t-value equivalent to an
        uncorrected p-value.
    tmin
        Threshold for forming clusters as t-value.
    tfce
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation
        Conduct permutations regardless of whether there are any clusters.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    c1_mean : NDVar
        Mean in the c1 condition.
    c0_mean : NDVar
        Mean in the c0 condition.
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.
    difference : NDVar
        Difference between the mean in condition c1 and condition c0.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    p_uncorrected : NDVar
        Map of p-values uncorrected for multiple comparison.
    t : NDVar
        Map of t-values.
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    n : int
        Number of cases.

    See Also
    --------
    testnd : Information on the different permutation methods

    Notes
    -----
    Also known as dependent t-test, paired t-test or repeated measures t-test.

    Permutation distributions are generated by randomly switching condition
    labels within categories of ``match``.
    For each permutation, a random subset of ``match`` is selected and the c1/c0
    labels are switched for those cases.

    T-values for cases with zero variance are set to t=0.
    """
    _state_specific = ('x', 'c1', 'c0', 'tail', 't', 'n', 'df', 'c1_mean',
                       'c0_mean')
    _statistic = 't'

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: Union[CategorialArg, NDVarArg],
            c1: CellArg = None,
            c0: CellArg = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            tail: int = 0,
            samples: int = 10000,
            pmin: float = None,
            tmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            **criteria):
        y1, y0, c1, c0, match, n, x_name, c1_name, c0_name = _related_measures_args(y, x, c1, c0, match, data, sub, True)
        check_for_vector_dim(y1)

        if n <= 2:
            raise ValueError("Not enough observations for t-test (n=%i)" % n)
        df = n - 1
        diff = y1 - y0
        tmap = stats.t_1samp(diff.x)

        n_threshold_params = sum((pmin is not None, tmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            threshold = cdist = None
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, tmin and tfce can be specified")
        else:
            if pmin is not None:
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            else:
                threshold = None

            n_samples, samples = _resample_params(len(diff), samples)
            cdist = NDPermutationDistribution(
                diff, n_samples, threshold, tfce, tail, 't', 'Related Samples t-Test',
                tstart, tstop, criteria, parc, force_permutation)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_sign_flip(n, samples)
                run_permutation(opt.t_1samp_perm, cdist, iterator)

        # NDVar map of t-values
        info = _info.for_stat_map('t', threshold, tail=tail, old=y1.info)
        t = NDVar(tmap, y1.dims[1:], 't', info)

        # store attributes
        NDDifferenceTest.__init__(self, y1, match, sub, samples, tfce, pmin, cdist, tstart, tstop)
        self.x = x_name
        self.c0 = c0
        self.c1 = c1

        self.n = n
        self.df = df
        self.tail = tail
        self.t = t
        self.tmin = tmin

        self.c1_mean = y1.mean('case', name=cellname(c1_name))
        self.c0_mean = y0.mean('case', name=cellname(c0_name))

        self._expand_state()

    def _expand_state(self):
        NDTest._expand_state(self)

        # difference
        diff = self.c1_mean - self.c0_mean
        if np.any(diff.x < 0):
            diff.info['cmap'] = 'xpolar'
        diff.name = self._difference_name(diff)
        self.difference = diff

        # uncorrected p
        pmap = stats.ttest_p(self.t.x, self.df, self.tail)
        info = _info.for_p_map()
        self.p_uncorrected = NDVar(pmap, self.t.dims, 'p', info)

        # composites
        if self.samples:
            diff_p = self.masked_difference()
        else:
            diff_p = self.difference
        self.all = [self.c1_mean, self.c0_mean, diff_p]

    def _name(self):
        if self.tail == 0:
            comp = "%s == %s" % (self.c1, self.c0)
        elif self.tail > 0:
            comp = "%s > %s" % (self.c1, self.c0)
        else:
            comp = "%s < %s" % (self.c1, self.c0)

        if self.y:
            return "Related-Samples T-Test:  %s ~ %s" % (self.y, comp)
        else:
            return "Related-Samples T-Test:  %s" % comp

    def _plot_model(self):
        return self.x

    def _plot_sub(self):
        return "(%s).isin(%s)" % (self.x, (self.c1, self.c0))

    def _repr_test_args(self):
        args = [repr(self.y), repr(self.x)]
        if self.c1 is not None:
            args.extend((repr(self.c1), repr(self.c0), repr(self.match)))
        args[-1] += " (n=%i)" % self.n
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args

    def _default_plot_obj(self):
        if self.samples:
            diff = self.masked_difference()
        else:
            diff = self.difference
        return [self.c1_mean, self.c0_mean, diff]


class MultiEffectNDTest(NDTest):

    _state_common = ('x', 'effects', *NDTest._state_common)

    def __init__(
            self,
            x: str,
            effects: Iterable[str],
            *args,
    ):
        NDTest.__init__(self, *args)
        self.x = x
        self.effects = tuple(effects)

    def _repr_test_args(self):
        args = [repr(self.y), repr(self.x)]
        if self.match is not None:
            args.append(f'match={self.match!r}')
        return args

    def _repr_cdist(self):
        args = self._cdist[0]._repr_test_args(self.pmin)
        for cdist in self._cdist:
            effect_args = cdist._repr_clusters()
            args.append(f"{cdist.name!r}: {', '.join(effect_args)}")
        return args

    def _asfmtext(self, **_):
        if len(self.effects) == 1:
            max_stat, p = self._max_statistic(0, return_p=True)
            return FMText([fmtxt.eq(self._statistic, max_stat, 'max', stars=p), ', ', fmtxt.peq(p)])
        table = fmtxt.Table('llll')
        table.cells('Effect', fmtxt.symbol(self._statistic, 'max'), fmtxt.symbol('p'), 'sig')
        table.midrule()
        for i, effect in enumerate(self.effects):
            max_stat, p = self._max_statistic(i, return_p=True)
            table.cell(effect)
            table.cell(fmtxt.stat(max_stat))
            table.cell(fmtxt.p(p))
            table.cell(star(p))
        return table

    def _expand_state(self):
        # clusters
        cdists = self._cdist
        if cdists is None:
            self._kind = None
        else:
            self.tfce_maps = [cdist.tfce_map for cdist in cdists]
            self.p = [cdist.probability_map for cdist in cdists]
            self._kind = cdists[0].kind

    def _effect_index(self, effect: Union[int, str]):
        if isinstance(effect, str):
            return self.effects.index(effect)
        else:
            return effect

    def _iter_cdists(self):
        for cdist in self._cdist:
            yield cdist.name.capitalize(), cdist

    @property
    def _first_cdist(self):
        if self._cdist is None:
            return None
        else:
            return self._cdist[0]

    @property
    def permutation_distribution(self):
        if self._first_cdist is not None:
            return [cdist.dist for cdist in self._cdist]

    def _max_statistic(
            self,
            effect: Union[str, int],
            mask: NDVar = None,
            return_time: bool = False,
            return_p: bool = False,
            sub: dict = None,
    ):
        i = self._effect_index(effect)
        stat_map = self._statistic_map[i]
        p_map = self.p[i]
        tail = getattr(self, 'tail', self._statistic_tail)
        return self._max_statistic_from_map(stat_map, p_map, tail, mask, return_time, return_p, sub)

    def cluster(self, cluster_id, effect=0):
        """Retrieve a specific cluster as NDVar

        Parameters
        ----------
        cluster_id : int
            Cluster id.
        effect : int | str
            Index or name of the effect from which to retrieve a cluster
            (default is the first effect).

        Returns
        -------
        cluster : NDVar
            NDVar of the cluster, 0 outside the cluster.

        Notes
        -----
        Clusters only have stable ids for thresholded cluster distributions.
        """
        self._assert_has_cdist()
        i = self._effect_index(effect)
        return self._cdist[i].cluster(cluster_id)

    def compute_probability_map(self, effect=0, **sub):
        """Compute a probability map

        Parameters
        ----------
        effect : int | str
            Index or name of the effect from which to use the parameter map
            (default is the first effect).

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        self._assert_has_cdist()
        i = self._effect_index(effect)
        return self._cdist[i].compute_probability_map(**sub)

    def masked_parameter_map(
            self,
            effect: Union[str, int] = 0,
            pmin: float = 0.05,
            **sub,
    ) -> NDVar:
        """Statistical parameter map masked by significance

        Parameters
        ----------
        effect
            Index or name of the effect for which to retrieve the parameter map.
        pmin
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, ``pmin=1`` includes all clusters regardless of their
            p-value.

        Returns
        -------
        NDVar
            Parameter map with a mask that is ``True`` (masked) where p > pmin.
        """
        self._assert_has_cdist()
        i = self._effect_index(effect)
        return self._cdist[i].masked_parameter_map(pmin, **sub)

    def find_clusters(
            self,
            pmin: float = None,
            maps: bool = False,
            effect: Union[int, str] = None,
            **sub,
    ) -> Dataset:
        """Find significant regions or clusters

        Parameters
        ----------
        pmin
            Threshold p-value. For threshold-based tests, all clusters with a
            p-value smaller than ``pmin`` are included (default 1);
            for other tests, find contiguous regions with ``p ≤ pmin`` (default
            0.05).
        maps
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default ``False``).
        effect
            Index or name of the effect from which to find clusters (default is
            all effects).

        Returns
        -------
        Dataset
            Dataset with information about the clusters.
        """
        self._assert_has_cdist()
        if effect is not None:
            i = self._effect_index(effect)
            cdist = self._cdist[i]
            ds = cdist.clusters(pmin, maps, **sub)
            ds[:, 'effect'] = cdist.name
            return ds
        dss = [self.find_clusters(pmin, maps, i, **sub) for i in range(len(self.effects))]
        info = {}
        for ds, cdist in zip(dss, self._cdist):
            if 'clusters' in ds.info:
                info[f'{cdist.name} clusters'] = ds.info.pop('clusters')
        out = combine(dss)
        out.info.update(info)
        return out

    def find_peaks(self) -> Dataset:
        """Find peaks in a TFCE distribution

        Returns
        -------
        Dataset
            Dataset with information about the peaks.
        """
        self._assert_has_cdist()
        dss = []
        for cdist in self._cdist:
            ds = cdist.find_peaks()
            ds[:, 'effect'] = cdist.name
            dss.append(ds)
        return combine(dss)


class ANOVA(MultiEffectNDTest):
    """Mass-univariate ANOVA

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : Model
        Independent variables.
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    samples : int
        Number of samples for permutation test (default 10,000).
    pmin : None | scalar (0 < pmin < 1)
        Threshold for forming clusters:  use an f-value equivalent to an
        uncorrected p-value.
    fmin : scalar
        Threshold for forming clusters as f-value.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    match : categorial | False
        When permuting data, only shuffle the cases within the categories
        of match. By default, ``match`` is determined automatically based on
        the random efects structure of ``x``.
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    effects : tuple of str
        Names of the tested effects, in the same order as in other attributes.
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.
    f : list of NDVar
        Maps of F values.
    p : list of NDVar | None
        Maps of p-values corrected for multiple comparison (or None if no
        correction was performed).
    p_uncorrected : list of NDVar
        Maps of p-values uncorrected for multiple comparison.
    tfce_maps : list of NDVar | None
        Maps of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).

    See Also
    --------
    testnd : Information on the different permutation methods

    Examples
    --------
    Mass-univariate :ref:`exa-mu-anova` example.
    For information on model specification see the univariate
    :class:`~eelbrain.test.ANOVA` examples.
    """
    _state_specific = ('pmin', '_effects', '_dfs_denom', 'f')
    _statistic = 'f'
    _statistic_tail = 1

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: ModelArg,
            sub: IndexArg = None,
            data: Dataset = None,
            samples: int = 10000,
            pmin: float = None,
            fmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            match: Union[CategorialArg, bool] = None,
            parc: str = None,
            force_permutation: bool = False,
            **criteria):
        x_arg = x
        sub_arg = sub
        sub, n = assub(sub, data, True)
        y, n = asndvar(y, sub, data, n, np.float64, True)
        check_for_vector_dim(y)
        x = asmodel(x, sub, data, n, require_names=True)
        if match is None:
            random_effects = [e for e in x.effects if e.random]
            if not random_effects:
                match = None
            elif len(random_effects) > 1:
                raise NotImplementedError("Automatic match parameter for model with more than one random effect. Set match manually.")
            else:
                match = random_effects[0]
        elif match is not False:
            match = ascategorial(match, sub, data, n)

        lm = _nd_anova(x)
        effects = lm.effects
        dfs_denom = lm.dfs_denom
        fmaps = lm.map(y.x)

        # Cluster-based tests
        n_threshold_params = sum((pmin is not None, fmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            cdists = None
            thresholds = repeat(None, len(effects))
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, fmin and tfce can be specified")
        else:
            if pmin is not None:
                thresholds = [scipy.stats.f.isf(pmin, e.df, df_den) for e, df_den in zip(effects, dfs_denom)]
            elif fmin is not None:
                thresholds = tuple(repeat(abs(fmin), len(effects)))
            else:
                thresholds = tuple(repeat(None, len(effects)))

            cdists = [NDPermutationDistribution(y, samples, thresh, tfce, 1, 'f', e.name, tstart, tstop, criteria, parc, force_permutation) for e, thresh in zip(effects, thresholds)]

            # Find clusters in the actual data
            do_permutation = 0
            for cdist, fmap in zip(cdists, fmaps):
                cdist.add_original(fmap)
                do_permutation += cdist.do_permutation
            # Generate null distribution
            if do_permutation:
                iterator = permute_order(len(y), samples, unit=match)
                run_permutation_me(lm, cdists, iterator)

        # create ndvars
        dims = y.dims[1:]
        f = []
        for e, fmap, df_den, f_threshold in zip(effects, fmaps, dfs_denom, thresholds):
            info = _info.for_stat_map('f', f_threshold, tail=1, old=y.info)
            f.append(NDVar(fmap, dims, e.name, info))

        # store attributes
        x_desc = x_arg if isinstance(x_arg, str) else x.name  # TODO: x.name should use * when appropriate
        effect_names = (e.name for e in effects)
        MultiEffectNDTest.__init__(self, x_desc, effect_names, y, match, sub_arg, samples, tfce, pmin, cdists, tstart, tstop)
        self._effects = effects
        self._dfs_denom = dfs_denom
        self.f = f

        self._expand_state()

    def _expand_state(self):
        # backwards compatibility
        if hasattr(self, 'effects') and not isinstance(self.effects[0], str):
            self._effects = self.effects
            del self.effects
        if not hasattr(self, 'effects'):
            self.effects = tuple([e.name for e in self._effects])

        MultiEffectNDTest._expand_state(self)

        # backwards compatibility
        if hasattr(self, 'df_den'):
            df_den_temp = {e.name: df for e, df in self.df_den.items()}
            del self.df_den
            self._dfs_denom = tuple(df_den_temp[e] for e in self.effects)

        # f-maps with clusters
        pmin = self.pmin or 0.05
        if self.samples:
            f_and_clusters = []
            for e, fmap, df_den, cdist in zip(self._effects, self.f, self._dfs_denom, self._cdist):
                # create f-map with cluster threshold
                f0 = scipy.stats.f.isf(pmin, e.df, df_den)
                info = _info.for_stat_map('f', f0)
                f_ = NDVar(fmap.x, fmap.dims, e.name, info)
                # add overlay with cluster
                if cdist.probability_map is not None:
                    f_and_clusters.append([f_, cdist.probability_map])
                else:
                    f_and_clusters.append([f_])
            self.f_probability = f_and_clusters

        # uncorrected probability
        p_uncorr = []
        for e, f, df_den in zip(self._effects, self.f, self._dfs_denom):
            info = _info.for_p_map()
            pmap = scipy.stats.f.sf(f.x, e.df, df_den)
            p_ = NDVar(pmap, f.dims, e.name, info)
            p_uncorr.append(p_)
        self.p_uncorrected = p_uncorr

    def _name(self):
        if self.y:
            return "ANOVA:  %s ~ %s" % (self.y, self.x)
        else:
            return "ANOVA:  %s" % self.x

    def _plot_model(self):
        return '%'.join(e.name for e in self._effects if isinstance(e, Factor) or (isinstance(e, NestedEffect) and isinstance(e.effect, Factor)))

    def _plot_sub(self):
        return super(ANOVA, self)._plot_sub()

    def _default_plot_obj(self):
        if self.samples:
            return [self.masked_parameter_map(e) for e in self.effects]
        else:
            return self._statistic_map

    def table(self, title=None, caption=None, clusters=False):
        """Table listing all effects and corresponding smallest p-values

        Parameters
        ----------
        title : text
            Title for the table.
        caption : text
            Caption for the table.
        clusters : bool | float
            Include properties of all significant clusters (default ``False``;
            use float to include clusters with p ≤ ``clusters``).

        Returns
        -------
        table : eelbrain.fmtxt.Table
            ANOVA table.
        """
        # table columns
        columns = ['#', 'Effect']
        alignment = ['r', 'l']
        if clusters:
            cluster_properties = self._first_cdist._cluster_property_labels()
            columns.extend(cluster_properties)
            alignment.extend('l' * len(cluster_properties))
        else:
            cluster_properties = []
        columns.append('f_max')
        alignment.append('r')
        if self.p is not None:
            columns.extend(['p', 'sig'])
            alignment.extend(['r', 'l'])
        table = fmtxt.Table(alignment, title=title, caption=caption)
        table.cells(*columns)
        table.midrule()

        cluster_pmin = None if clusters is True else clusters
        show_map_stats = True
        for i, effect in enumerate(self.effects):
            table.cell(i)
            table.cell(effect)
            if self.p is not None:
                if clusters:
                    cluster_table = self.find_clusters(cluster_pmin, True, effect=effect)
                    show_map_stats = cluster_table.n_cases == 0
                    for ci in range(cluster_table.n_cases):
                        if ci:
                            table.cells('', '')
                        # properties
                        for key in cluster_properties:
                            table.cell(cluster_table[ci, key])
                        # f_max, p, sig
                        table.cell(fmtxt.stat(cluster_table[ci, 'cluster'].max()))
                        table.cell(fmtxt.p(cluster_table[ci, 'p']))
                        table.cell(star(cluster_table[ci, 'p']))
                    if show_map_stats:
                        for _ in cluster_properties:
                            table.cell('')
            if show_map_stats:
                table.cell(fmtxt.stat(self.f[i].max()))
                if self.p is not None:
                    pmin = self.p[i].min()
                    table.cell(fmtxt.p(pmin))
                    table.cell(star(pmin))
        return table


class Vector(NDDifferenceTest):
    """Test a vector field for vectors with non-random direction

    Parameters
    ----------
    y : NDVar
        Dependent variable (needs to include one vector dimension).
    match : None | categorial
        Combine data for these categories before testing.
    sub : index
        Perform test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    samples : int
        Number of samples for permutation test (default 10000).
    tmin : scalar
        Threshold value for forming clusters.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    norm : bool
        Use the vector norm as univariate test statistic (instead of Hotelling’s
        T-Square statistic).
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    n : int
        Number of cases.
    difference : NDVar
        The vector field averaged across cases.
    t2 : NDVar | None
        Hotelling T-Square map; ``None`` if the test used ``norm=True``.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or ``None`` if no
        correction was performed).
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.

    See Also
    --------
    testnd : Information on the different permutation methods

    Notes
    -----
    The permutation test for vector data is described in [1]_.
    Computation of the T-Square statistic relies on [2]_.

    References
    ----------
    .. [1] Das, Proloy, Christian Brodbeck, Jonathan Z. Simon, and Behtash Babadi.
        Neuro-Current Response Functions: A Unified Approach to MEG Source Analysis
        under the Continuous Stimuli Paradigm. NeuroImage 211 (May 2020): 116528.
        `10.1016/j.neuroimage.2020.116528 <https://doi.org/10.1016/j.neuroimage.2020.116528>`_
    .. [2] Kopp, J. (2008). Efficient numerical diagonalization of hermitian 3 x
        3 matrices. International Journal of Modern Physics C, 19(3), 523-548.
        `10.1142/S0129183108012303 <https://doi.org/10.1142/S0129183108012303>`_
    """
    _state_specific = ('difference', 'n', '_v_dim', 't2')

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            samples: int = 10000,
            tmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            norm: bool = False,
            **criteria):
        use_norm = bool(norm)
        ct = Celltable(y, match=match, sub=sub, data=data, coercion=asndvar, dtype=np.float64)

        n = len(ct.y)
        cdist = NDPermutationDistribution(ct.y, samples, tmin, tfce, 1, 'norm', 'Vector test', tstart, tstop, criteria, parc, force_permutation)

        v_dim = ct.y.dimnames[cdist._vector_ax + 1]
        v_mean = ct.y.mean('case')
        v_mean_norm = v_mean.norm(v_dim)
        if not use_norm:
            t2_map = self._vector_t2_map(ct.y)
            cdist.add_original(t2_map.x if v_mean.ndim > 1 else t2_map)
            if v_mean.ndim == 1:
                self.t2 = t2_map
            else:
                self.t2 = NDVar(t2_map, v_mean_norm.dims, 't2', _info.for_stat_map('t2'))
        else:
            cdist.add_original(v_mean_norm.x if v_mean.ndim > 1 else v_mean_norm)
            self.t2 = None

        if cdist.do_permutation:
            iterator = random_seeds(samples)
            vector_perm = partial(self._vector_perm, use_norm=use_norm)
            run_permutation(vector_perm, cdist, iterator)

        # store attributes
        NDTest.__init__(self, ct.y, ct.match, sub, samples, tfce, None, cdist, tstart, tstop)
        self.difference = v_mean
        self._v_dim = v_dim
        self.n = n

        self._expand_state()

    def __setstate__(self, state):
        if 'diff' in state:
            state['difference'] = state.pop('diff')
        NDTest.__setstate__(self, state)

    @property
    def _statistic(self):
        return 'norm' if self.t2 is None else 't2'

    def _name(self):
        if self.y:
            return f"Vector test:  {self.y}"
        else:
            return "Vector test"

    def _repr_test_args(self):
        args = []
        if self.y:
            args.append(repr(self.y))
        if self.match:
            args.append(f'match={self.match!r}')
        return args

    @staticmethod
    def _vector_perm(y, out, seed, use_norm):
        n_cases, n_dims, n_tests = y.shape
        assert n_dims == 3
        rotation = rand_rotation_matrices(n_cases, seed)
        if use_norm:
            return vector.mean_norm_rotated(y, rotation, out)
        else:
            return vector.t2_stat_rotated(y, rotation, out)

    @staticmethod
    def _vector_t2_map(y):
        dimnames = y.get_dimnames(first=('case', 'space'))
        x = y.get_data(dimnames)
        t2_map = stats.t2_1samp(x)
        if y.ndim == 2:
            return np.float64(t2_map)
        else:
            dims = y.get_dims(dimnames[2:])
            return NDVar(t2_map, dims)


class VectorDifferenceIndependent(Vector):
    """Test difference between two vector fields for non-random direction

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : categorial | NDVar
        Model containing the cells which should be compared, or NDVar to which
        ``y`` should be compared. In the latter case, the next three parameters
        are ignored.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Combine cases with the same cell on ``x % match``.
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    samples : int
        Number of samples for permutation test (default 10000).
    tmin : scalar
        Threshold value for forming clusters.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    norm : bool
        Use the vector norm as univariate test statistic (instead of Hotelling’s
        T-Square statistic).
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    n : int
        Total number of cases.
    n1 : int
        Number of cases in ``c1``.
    n0 : int
        Number of cases in ``c0``.
    c1_mean : NDVar
        Mean in the c1 condition.
    c0_mean : NDVar
        Mean in the c0 condition.
    difference : NDVar
        Difference between the mean in condition c1 and condition c0.
    t2 : NDVar | None
        Hotelling T-Square map; ``None`` if the test used ``norm=True``.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.

    See Also
    --------
    testnd : Information on the different permutation methods
    """
    _state_specific = ('difference', 'c1_mean', 'c0_mean', 'n', '_v_dim', 't2')
    _statistic = 'norm'

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: Union[CategorialArg, NDVarArg],
            c1: str = None,
            c0: str = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            samples: int = 10000,
            tmin: float = None,
            tfce: bool = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            norm: bool = False,
            **criteria):
        use_norm = bool(norm)
        y, y1, y0, c1, c0, match, x_name, c1_name, c0_name = _independent_measures_args(y, x, c1, c0, match, data, sub, True)
        self.n1 = len(y1)
        self.n0 = len(y0)
        self.n = len(y)

        cdist = NDPermutationDistribution(y, samples, tmin, tfce, 1, 'norm', 'Vector test (independent)', tstart, tstop, criteria, parc, force_permutation)

        self._v_dim = v_dim = y.dimnames[cdist._vector_ax + 1]
        self.c1_mean = y1.mean('case', name=cellname(c1_name))
        self.c0_mean = y0.mean('case', name=cellname(c0_name))
        self.difference = self.c1_mean - self.c0_mean
        self.difference.name = self._difference_name(self.difference)
        v_mean_norm = self.difference.norm(v_dim)
        if not use_norm:
            raise NotImplementedError("t2 statistic not implemented for VectorDifferenceIndependent")
        else:
            cdist.add_original(v_mean_norm.x if self.difference.ndim > 1 else v_mean_norm)
            self.t2 = None

        if cdist.do_permutation:
            iterator = random_seeds(samples)
            vector_perm = partial(self._vector_perm, use_norm=use_norm)
            run_permutation(vector_perm, cdist, iterator, self.n1)

        NDTest.__init__(self, y, match, sub, samples, tfce, None, cdist, tstart, tstop)
        self._expand_state()

    def _name(self):
        if self.y:
            return f"Vector test (independent):  {self.y}"
        else:
            return "Vector test (independent)"

    @staticmethod
    def _vector_perm(y, n1, out, seed, use_norm):
        assert use_norm
        n_cases, n_dims, n_tests = y.shape
        assert n_dims == 3
        # randomize directions
        rotation = rand_rotation_matrices(n_cases, seed)
        # randomize groups
        cases = np.arange(n_cases)
        np.random.shuffle(cases)
        # group 1
        mean_1 = np.zeros((n_dims, n_tests))
        for case in cases[:n1]:
            mean_1 += np.tensordot(rotation[case], y[case], ((1,), (0,)))
        mean_1 /= n1
        # group 0
        mean_0 = np.zeros((n_dims, n_tests))
        for case in cases[n1:]:
            mean_0 += np.tensordot(rotation[case], y[case], ((1,), (0,)))
        mean_0 /= (n_cases - n1)
        # difference
        mean_1 -= mean_0
        norm = scipy.linalg.norm(mean_1, 2, axis=0)
        if out is not None:
            out[:] = norm
        return norm


class VectorDifferenceRelated(NDMaskedC1Mixin, Vector):
    """Test difference between two vector fields for non-random direction

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    x : categorial | NDVar
        Model containing the cells which should be compared, or NDVar to which
        ``y`` should be compared. In the latter case, the next three parameters
        are ignored.
    c1 : str | tuple | None
        Test condition (cell of ``x``). ``c1`` and ``c0`` can be omitted if
        ``x`` only contains two cells, in which case cells will be used in
        alphabetical order.
    c0 : str | tuple | None
        Control condition (cell of ``x``).
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison).
    sub : index
        Perform the test with a subset of the data.
    data : Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    samples : int
        Number of samples for permutation test (default 10000).
    tmin : scalar
        Threshold value for forming clusters.
    tfce : bool | scalar
        Use threshold-free cluster enhancement. Use a scalar to specify the
        step of TFCE levels (for ``tfce is True``, 0.1 is used).
    tstart : scalar
        Start of the time window for the permutation test (default is the
        beginning of ``y``).
    tstop : scalar
        Stop of the time window for the permutation test (default is the
        end of ``y``).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation: bool
        Conduct permutations regardless of whether there are any clusters.
    norm : bool
        Use the vector norm as univariate test statistic (instead of Hotelling’s
        T-Square statistic).
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    n : int
        Number of cases.
    c1_mean : NDVar
        Mean in the ``c1`` condition.
    c0_mean : NDVar
        Mean in the ``c0`` condition.
    difference : NDVar
        Difference between the mean in condition ``c1`` and condition ``c0``.
    t2 : NDVar | None
        Hotelling T-Square map; ``None`` if the test used ``norm=True``.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or ``None`` if no
        correction was performed).
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    clusters : None | Dataset
        For cluster-based tests, a table of all clusters. Otherwise a table of
        all significant regions (or ``None`` if permutations were omitted).
        See also the :meth:`.find_clusters` method.

    See Also
    --------
    Vector : One-sample vector test, notes on vector test implementation
    testnd : Information on the different permutation methods
    """
    _state_specific = ('x', 'c1', 'c0', 'difference', 'c1_mean', 'c0_mean', 'n', '_v_dim', 't2')

    @user_activity
    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: Union[CategorialArg, NDVarArg],
            c1: str = None,
            c0: str = None,
            match: CategorialArg = None,
            sub: IndexArg = None,
            data: Dataset = None,
            samples: int = 10000,
            tmin: float = None,
            tfce: bool = False,
            tstart: float = None,
            tstop: float = None,
            parc: str = None,
            force_permutation: bool = False,
            norm: bool = False,
            **criteria):
        use_norm = bool(norm)
        y1, y0, c1, c0, match, n, x_name, c1_name, c0_name = _related_measures_args(y, x, c1, c0, match, data, sub, True)
        difference = y1 - y0
        difference.name = 'difference'

        n_samples, samples = _resample_params(n, samples)
        cdist = NDPermutationDistribution(difference, n_samples, tmin, tfce, 1, 'norm', 'Vector test (related)', tstart, tstop, criteria, parc, force_permutation)

        v_dim = difference.dimnames[cdist._vector_ax + 1]
        v_mean = difference.mean('case')
        v_mean_norm = v_mean.norm(v_dim)
        if not use_norm:
            t2_map = self._vector_t2_map(difference)
            cdist.add_original(t2_map.x if v_mean.ndim > 1 else t2_map)
            if v_mean.ndim == 1:
                self.t2 = t2_map
            else:
                self.t2 = NDVar(t2_map, v_mean_norm.dims, 't2', _info.for_stat_map('t2'))
        else:
            cdist.add_original(v_mean_norm.x if v_mean.ndim > 1 else v_mean_norm)
            self.t2 = None

        if cdist.do_permutation:
            iterator = random_seeds(n_samples)
            vector_perm = partial(self._vector_perm, use_norm=use_norm)
            run_permutation(vector_perm, cdist, iterator)

        # store attributes
        NDTest.__init__(self, difference, match, sub, samples, tfce, None, cdist, tstart, tstop)
        self.difference = v_mean
        self.c1_mean = y1.mean('case', name=cellname(c1_name))
        self.c0_mean = y0.mean('case', name=cellname(c0_name))
        self._v_dim = v_dim
        self.n = n
        self.x = x_name
        self.c0 = c0
        self.c1 = c1
        self._expand_state()

    def _name(self):
        if self.y:
            return f"Vector test (related):  {self.y}"
        else:
            return "Vector test (related)"

    def _repr_test_args(self):
        args = [repr(self.y), repr(self.x)]
        if self.c1 is not None:
            args.extend((repr(self.c1), repr(self.c0), repr(self.match)))
        args[-1] += " (n=%i)" % self.n
        return args

    def __setstate__(self, state):
        if 'x' not in state:
            state['x'] = None
            state['c1'] = state['c1_mean'].name
            state['c0'] = state['c0_mean'].name
        Vector.__setstate__(self, state)


def flatten(array, adjacency):
    """Reshape SPM buffer array to 2-dimensional map for adjacency processing

    Parameters
    ----------
    array : ndarray
        N-dimensional array (with non-adjacent dimension at first position).
    adjacency : Adjacency
        N-dimensional adjacency.

    Returns
    -------
    flat_array : ndarray
        The input array reshaped if necessary, making sure that input and output
        arrays share the same underlying data buffer.
    """
    if array.ndim == 2 or not adjacency.custom:
        return array
    else:
        out = array.reshape((array.shape[0], -1))
        assert out.base is array
        return out


def flatten_1d(array):
    if array.ndim == 1:
        return array
    else:
        out = array.ravel()
        assert out.base is array
        return out


def label_clusters(stat_map, threshold, tail, adjacency, criteria):
    """Label clusters

    Parameters
    ----------
    stat_map : array
        Statistical parameter map (non-adjacent dimension on the first
        axis).

    Returns
    -------
    cmap : np.ndarray of uint32
        Array with clusters labelled as integers.
    cluster_ids : np.ndarray of uint32
        Identifiers of the clusters that survive the minimum duration
        criterion.
    """
    cmap = np.empty(stat_map.shape, np.uint32)
    bin_buff = np.empty(stat_map.shape, bool)
    cmap_flat = flatten(cmap, adjacency)

    if tail == 0:
        int_buff = np.empty(stat_map.shape, np.uint32)
        int_buff_flat = flatten(int_buff, adjacency)
    else:
        int_buff = int_buff_flat = None

    cids = _label_clusters(stat_map, threshold, tail, adjacency, criteria,
                           cmap, cmap_flat, bin_buff, int_buff, int_buff_flat)
    return cmap, cids


def _label_clusters(stat_map, threshold, tail, conn, criteria, cmap, cmap_flat,
                    bin_buff, int_buff, int_buff_flat):
    """Find clusters on a statistical parameter map

    Parameters
    ----------
    stat_map : array
        Statistical parameter map (non-adjacent dimension on the first
        axis).
    cmap : array of int
        Buffer for the cluster id map (will be modified).

    Returns
    -------
    cluster_ids : np.ndarray of uint32
        Identifiers of the clusters that survive the minimum duration
        criterion.
    """
    # compute clusters
    if tail >= 0:
        bin_map_above = np.greater(stat_map, threshold, bin_buff)
        cids = _label_clusters_binary(bin_map_above, cmap, cmap_flat, conn,
                                      criteria)

    if tail <= 0:
        bin_map_below = np.less(stat_map, -threshold, bin_buff)
        if tail < 0:
            cids = _label_clusters_binary(bin_map_below, cmap, cmap_flat, conn,
                                          criteria)
        else:
            cids_l = _label_clusters_binary(bin_map_below, int_buff,
                                            int_buff_flat, conn, criteria)
            x = cmap.max()
            int_buff[bin_map_below] += x
            cids_l += x
            cmap += int_buff
            cids = np.concatenate((cids, cids_l))

    return cids


def label_clusters_binary(bin_map, adjacency, criteria=None):
    """Label clusters in a boolean map

    Parameters
    ----------
    bin_map : numpy.ndarray
        Binary map.
    adjacency : Adjacency
        Adjacency corresponding to ``bin_map``.
    criteria : dict
        Cluster criteria.

    Returns
    -------
    cmap : numpy.ndarray of uint32
        Array with clusters labelled as integers.
    cluster_ids : numpy.ndarray of uint32
        Sorted identifiers of the clusters that survive the selection criteria.
    """
    cmap = np.empty(bin_map.shape, np.uint32)
    cmap_flat = flatten(cmap, adjacency)
    cids = _label_clusters_binary(bin_map, cmap, cmap_flat, adjacency, criteria)
    return cmap, cids


def _label_clusters_binary(bin_map, cmap, cmap_flat, adjacency, criteria):
    """Label clusters in a binary array

    Parameters
    ----------
    bin_map : np.ndarray
        Binary map of where the parameter map exceeds the threshold for a
        cluster (non-adjacent dimension on the first axis).
    cmap : np.ndarray
        Array in which to label the clusters.
    cmap_flat : np.ndarray
        Flat copy of cmap (ndim=2, only used when all_adjacent==False)
    adjacency : Adjacency
        Adjacency.
    criteria : None | list
        Cluster size criteria, list of (axes, v) tuples. Collapse over axes
        and apply v minimum length).

    Returns
    -------
    cluster_ids : np.ndarray of uint32
        Sorted identifiers of the clusters that survive the selection criteria.
    """
    # find clusters
    n = ndimage.label(bin_map, adjacency.struct, cmap)
    if n <= 1:
        # in older versions, n is 1 even when no cluster is found
        if n == 0 or cmap.max() == 0:
            return np.array((), np.uint32)
        else:
            cids = np.array((1,), np.uint32)
    elif adjacency.custom:
        cids = merge_labels(cmap_flat, n, *adjacency.custom[0])
    else:
        cids = np.arange(1, n + 1, 1, np.uint32)

    # apply minimum cluster size criteria
    if criteria and cids.size:
        for axes, v in criteria:
            cids = np.setdiff1d(cids,
                                [i for i in cids if np.count_nonzero(np.equal(cmap, i).any(axes)) < v],
                                True)
            if cids.size == 0:
                break

    return cids


def tfce(stat_map, tail, adjacency, dh=0.1):
    tfce_im = np.empty(stat_map.shape, np.float64)
    tfce_im_1d = flatten_1d(tfce_im)
    bin_buff = np.empty(stat_map.shape, bool)
    int_buff = np.empty(stat_map.shape, np.uint32)
    int_buff_flat = flatten(int_buff, adjacency)
    int_buff_1d = flatten_1d(int_buff)
    return _tfce(stat_map, tail, adjacency, tfce_im, tfce_im_1d, bin_buff, int_buff,
                 int_buff_flat, int_buff_1d, dh)


def _tfce(stat_map, tail, conn, out, out_1d, bin_buff, int_buff,
          int_buff_flat, int_buff_1d, dh=0.1, e=0.5, h=2.0):
    "Threshold-free cluster enhancement"
    out.fill(0)

    # determine slices
    if tail == 0:
        hs = chain(np.arange(-dh, stat_map.min(), -dh),
                   np.arange(dh, stat_map.max(), dh))
    elif tail < 0:
        hs = np.arange(-dh, stat_map.min(), -dh)
    else:
        hs = np.arange(dh, stat_map.max(), dh)

    # label clusters in slices at different heights
    # fill each cluster with total section value
    # each point's value is the vertical sum
    for h_ in hs:
        if h_ > 0:
            np.greater_equal(stat_map, h_, bin_buff)
            h_factor = h_ ** h
        else:
            np.less_equal(stat_map, h_, bin_buff)
            h_factor = (-h_) ** h

        c_ids = _label_clusters_binary(bin_buff, int_buff, int_buff_flat, conn, None)
        tfce_increment(c_ids, int_buff_1d, out_1d, e, h_factor)

    return out


class StatMapProcessor:

    def __init__(self, tail, max_axes, parc):
        """Reduce a statistical map to the relevant maximum statistic"""
        self.tail = tail
        self.max_axes = max_axes
        self.parc = parc

    def max_stat(self, stat_map):
        if self.tail == 0:
            v = np.abs(stat_map, stat_map).max(self.max_axes)
        elif self.tail > 0:
            v = stat_map.max(self.max_axes)
        else:
            v = -stat_map.min(self.max_axes)

        if self.parc is None:
            return v
        else:
            return [v[idx].max() for idx in self.parc]


class TFCEProcessor(StatMapProcessor):

    def __init__(self, tail, max_axes, parc, shape, adjacency, dh):
        StatMapProcessor.__init__(self, tail, max_axes, parc)
        self.shape = shape
        self.adjacency = adjacency
        self.dh = dh

        # Pre-allocate memory buffers used for cluster processing
        self._bin_buff = np.empty(shape, bool)
        self._int_buff = np.empty(shape, np.uint32)
        self._tfce_im = np.empty(shape, np.float64)
        self._tfce_im_1d = flatten_1d(self._tfce_im)
        self._int_buff_flat = flatten(self._int_buff, adjacency)
        self._int_buff_1d = flatten_1d(self._int_buff)

    def max_stat(self, stat_map):
        v = _tfce(
            stat_map, self.tail, self.adjacency, self._tfce_im, self._tfce_im_1d,
            self._bin_buff, self._int_buff, self._int_buff_flat, self._int_buff_1d,
            self.dh,
        ).max(self.max_axes)
        if self.parc is None:
            return v
        else:
            return [v[idx].max() for idx in self.parc]


class ClusterProcessor(StatMapProcessor):

    def __init__(self, tail, max_axes, parc, shape, adjacency, threshold,
                 criteria):
        StatMapProcessor.__init__(self, tail, max_axes, parc)
        self.shape = shape
        self.adjacency = adjacency
        self.threshold = threshold
        self.criteria = criteria

        # Pre-allocate memory buffers used for cluster processing
        self._bin_buff = np.empty(shape, bool)

        self._cmap = np.empty(shape, np.uint32)
        self._cmap_flat = flatten(self._cmap, adjacency)

        if tail == 0:
            self._int_buff = np.empty(shape, np.uint32)
            self._int_buff_flat = flatten(self._int_buff, adjacency)
        else:
            self._int_buff = self._int_buff_flat = None

    def max_stat(self, stat_map, threshold=None):
        if threshold is None:
            threshold = self.threshold
        cmap = self._cmap
        cids = _label_clusters(stat_map, threshold, self.tail, self.adjacency, self.criteria, cmap, self._cmap_flat, self._bin_buff, self._int_buff, self._int_buff_flat)
        if self.parc is not None:
            v = []
            for idx in self.parc:
                clusters_v = ndimage.sum(stat_map[idx], cmap[idx], cids)
                if len(clusters_v):
                    if self.tail <= 0:
                        np.abs(clusters_v, clusters_v)
                    v.append(clusters_v.max())
                else:
                    v.append(0)
            return v
        elif len(cids):
            clusters_v = ndimage.sum(stat_map, cmap, cids)
            if self.tail <= 0:
                np.abs(clusters_v, clusters_v)
            return clusters_v.max()
        else:
            return 0


def get_map_processor(kind, *args):
    if kind == 'tfce':
        return TFCEProcessor(*args)
    elif kind == 'cluster':
        return ClusterProcessor(*args)
    elif kind == 'raw':
        return StatMapProcessor(*args)
    else:
        raise ValueError("kind=%s" % repr(kind))


class NDPermutationDistribution:
    """Accumulate information on a cluster statistic.

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    samples : int
        Number of permutations.
    threshold : scalar > 0
        Threshold-based clustering.
    tfce : bool | scalar
        Threshold-free cluster enhancement.
    tail : 1 | 0 | -1
        Which tail(s) of the distribution to consider. 0 is two-tailed,
        whereas 1 only considers positive values and -1 only considers
        negative values.
    meas : str
        Label for the parameter measurement (e.g., 't' for t-values).
    name : None | str
        Name for the comparison.
    tstart, tstop : None | scalar
        Restrict the time window for finding clusters (None: use the whole
        epoch).
    criteria : dict
        Dictionary with threshold criteria for cluster size: 'mintime'
        (seconds) and 'minsource' (n_sources).
    parc : str
        Collect permutation statistics for all regions of the parcellation of
        this dimension. For threshold-based test, the regions are disconnected.
    force_permutation : bool
        Conduct permutations regardless of whether there are any clusters.


    Notes
    -----
    Use of the NDPermutationDistribution proceeds in 3 steps:

    - initialize the NDPermutationDistribution object: ``cdist = NDPermutationDistribution(...)``
    - use a copy of y cropped to the time window of interest:
      ``y = cdist.Y_perm``
    - add the actual statistical map with ``cdist.add_original(pmap)``
    - if any clusters are found (``if cdist.n_clusters``):

      - proceed to add statistical maps from permuted data with
        ``cdist.add_perm(pmap)``.


    Permutation data shape: case, [vector, ][non-adjacent, ] ...
    internal shape: [non-adjacent, ] ...
    """
    dist = None
    tfce_warning = None

    def __init__(self, y, samples, threshold, tfce=False, tail=0, meas='?', name=None, tstart=None, tstop=None, criteria={}, parc=None, force_permutation=False):
        assert y.has_case
        assert parc is None or isinstance(parc, str)
        if tfce and threshold:
            raise RuntimeError(f"{threshold=}, {tfce=}: mutually exclusive parameters")
        elif tfce:
            if tfce is not True:
                tfce = abs(tfce)
            kind = 'tfce'
        elif threshold:
            threshold = float(threshold)
            kind = 'cluster'
            assert threshold > 0
        else:
            kind = 'raw'

        # vector: will be removed for stat_map
        vector = [d._adjacency_type == 'vector' for d in y.dims[1:]]
        has_vector_ax = any(vector)
        if has_vector_ax:
            vector_ax = vector.index(True)
        else:
            vector_ax = None

        # prepare temporal cropping
        if (tstart is None) and (tstop is None):
            y_perm = y
            self._crop_for_permutation = False
            self._crop_idx = None
        else:
            t_ax = y.get_axis('time') - 1
            y_perm = y.sub(time=(tstart, tstop))
            # for stat-maps
            if vector_ax is not None and vector_ax < t_ax:
                t_ax -= 1
            t_slice = y.time._array_index(slice(tstart, tstop))
            self._crop_for_permutation = True
            self._crop_idx = FULL_AXIS_SLICE * t_ax + (t_slice,)

        dims = list(y_perm.dims[1:])
        if has_vector_ax:
            del dims[vector_ax]

        # custom adjacency: move non-adjacent adjacency to first axis
        custom = [d._adjacency_type == 'custom' for d in dims]
        n_custom = sum(custom)
        if n_custom > 1:
            raise NotImplementedError("More than one axis with custom adjacency")
        nad_ax = None if n_custom == 0 else custom.index(True)
        if nad_ax:
            swapped_dims = list(dims)
            swapped_dims[0], swapped_dims[nad_ax] = dims[nad_ax], dims[0]
        else:
            swapped_dims = dims
        adjacency = Adjacency(swapped_dims, parc)
        assert adjacency.vector is None

        # cluster map properties
        ndim = len(dims)

        # prepare cluster minimum size criteria
        if criteria:
            if kind != 'cluster':
                raise ValueError("Can not use cluster size criteria when doing threshold free cluster evaluation")
            criteria_ = []
            for k, v in criteria.items():
                if m := re.match(r'min(\w+)', k):
                    dimname = m.group(1)
                    if not y.has_dim(dimname):
                        raise TypeError(f"{k}={v!r}: invalid keyword argument (no dimension named {dimname})")
                    ax = y.get_axis(dimname) - 1
                    if dimname == 'time':
                        v = int(ceil(v / y.time.tstep))
                else:
                    raise TypeError(f"{k}={v!r}: invalid keyword argument")

                if nad_ax:
                    if ax == 0:
                        ax = nad_ax
                    elif ax == nad_ax:
                        ax = 0

                axes = tuple(i for i in range(ndim) if i != ax)
                criteria_.append((axes, v))
        else:
            criteria_ = None

        # prepare distribution
        samples = int(samples)
        if parc:
            for parc_ax, parc_dim in enumerate(swapped_dims):
                if parc_dim.name == parc:
                    break
            else:
                raise ValueError("parc=%r (no dimension named %r)" % (parc, parc))

            if parc_dim._adjacency_type == 'none':
                parc_indexes = np.arange(len(parc_dim))
            elif kind == 'tfce':
                raise NotImplementedError(
                    f"TFCE for parc={parc!r} ({parc_dim.__class__.__name__} dimension)")
            elif parc_dim._adjacency_type == 'custom':
                if not hasattr(parc_dim, 'parc'):
                    raise NotImplementedError(f"parc={parc!r}: dimension has no parcellation")
                parc_indexes = tuple(np.flatnonzero(parc_dim.parc == cell) for
                                     cell in parc_dim.parc.cells)
                parc_dim = Categorial(parc, parc_dim.parc.cells)
            else:
                raise NotImplementedError(f"parc={parc!r}")
            dist_shape = (samples, len(parc_dim))
            dist_dims = ('case', parc_dim)
            max_axes = tuple(chain(range(parc_ax), range(parc_ax + 1, ndim)))
        else:
            dist_shape = (samples,)
            dist_dims = None
            max_axes = None
            parc_indexes = None

        # arguments for the map processor
        shape = tuple(map(len, swapped_dims))
        if kind == 'raw':
            map_args = (kind, tail, max_axes, parc_indexes)
        elif kind == 'tfce':
            dh = 0.1 if tfce is True else tfce
            map_args = (kind, tail, max_axes, parc_indexes, shape, adjacency, dh)
        else:
            map_args = (kind, tail, max_axes, parc_indexes, shape, adjacency, threshold, criteria_)

        self.kind = kind
        self.y_perm = y_perm
        self.dims = tuple(dims)  # external stat map dims (cropped time)
        self.shape = shape  # internal stat map shape
        self._adjacency = adjacency
        self.samples = samples
        self.dist_shape = dist_shape
        self._dist_dims = dist_dims
        self._max_axes = max_axes
        self.threshold = threshold
        self.tfce = tfce
        self.tail = tail
        self._nad_ax = nad_ax
        self._vector_ax = vector_ax
        self.tstart = tstart
        self.tstop = tstop
        self.parc = parc
        self.meas = meas
        self.name = name
        self._criteria = criteria_
        self.criteria = criteria
        self.map_args = map_args
        self.has_original = False
        self.do_permutation = False
        self.dt_perm = None
        self._finalized = False
        self._init_time = current_time()
        self._host = socket.gethostname()
        self.force_permutation = force_permutation

        from .. import __version__
        self._version = __version__

    def _crop(self, im):
        "Crop an original stat_map"
        if self._crop_for_permutation:
            return im[self._crop_idx]
        else:
            return im

    def uncrop(
            self,
            ndvar: NDVar,  # NDVar to uncrop
            to: NDVar,  # NDVar that has the target time dimensions
            default: float = 0,  # value to fill in uncropped area
    ):
        if self.tstart is None and self.tstop is None:
            return ndvar
        target_time = to.get_dim('time')
        t_ax = ndvar.get_axis('time')
        dims = list(ndvar.dims)
        dims[t_ax] = target_time
        shape = list(ndvar.shape)
        shape[t_ax] = len(target_time)
        t_slice = target_time._array_index(slice(self.tstart, self.tstop))
        x = np.empty(shape, ndvar.x.dtype)
        x.fill(default)
        x[FULL_AXIS_SLICE * t_ax + (t_slice,)] = ndvar.x
        return NDVar(x, dims, ndvar.name, ndvar.info)

    def add_original(self, stat_map):
        """Add the original statistical parameter map.

        Parameters
        ----------
        stat_map : array
            Parameter map of the statistic of interest (uncropped).
        """
        if self.has_original:
            raise RuntimeError("Original pmap already added")
        logger = logging.getLogger(__name__)
        logger.debug("Adding original parameter map...")

        # crop/reshape stat_map
        stat_map = self._crop(stat_map)
        if self._nad_ax:
            stat_map = stat_map.swapaxes(0, self._nad_ax)

        # process map
        if self.kind == 'tfce':
            dh = 0.1 if self.tfce is True else self.tfce
            n_steps = max(0, stat_map.max()) // dh + max(0, -stat_map.min()) // dh
            if n_steps > 10000:
                raise RuntimeError(f"TFCE requested with {n_steps:.0f} steps; currently 10000 is set as limit to avoid excessive computation times. Consider setting the tfce parameter to a larger step size.")
            self.tfce_warning = n_steps < 1
            cmap = tfce(stat_map, self.tail, self._adjacency, dh)
            cids = None
            n_clusters = True
        elif self.kind == 'cluster':
            cmap, cids = label_clusters(stat_map, self.threshold, self.tail,
                                        self._adjacency, self._criteria)
            n_clusters = len(cids)
            # clean original cluster map
            idx = np.isin(cmap, cids, invert=True).reshape(self.shape)
            cmap[idx] = 0
        else:
            cmap = stat_map
            cids = None
            n_clusters = True

        self._t0 = current_time()
        self._original_cluster_map = cmap
        self._cids = cids
        self.n_clusters = n_clusters
        self.has_original = True
        self.dt_original = self._t0 - self._init_time
        self._original_param_map = stat_map

        # create the distribution container
        if self.force_permutation or (self.samples and n_clusters):
            self.do_permutation = True
            self.dist = np.zeros(self.dist_shape)
        else:
            self.finalize()

    def _aggregate_dist(self, **sub):
        """Aggregate permutation distribution to one value per permutation

        Parameters
        ----------
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        dist : array, shape = (samples,)
            Maximum value for each permutation in the given region.
        """
        dist = self.dist

        if sub:
            if self._dist_dims is None:
                raise TypeError("NDPermutationDistribution does not have parcellation")
            dist_ = NDVar(dist, self._dist_dims)
            dist_sub = dist_.sub(**sub)
            dist = dist_sub.x

        if dist.ndim > 1:
            axes = tuple(range(1, dist.ndim))
            dist = dist.max(axes)

        return dist

    def __repr__(self):
        items = [self.kind]
        if self.has_original:
            if self.kind == 'cluster':
                items.append(f"{self.n_clusters} clusters")
        else:
            items.append("no data")
        return f"<NDPermutationDistribution: {', '.join(items)}>"

    def __getstate__(self):
        if not self._finalized:
            raise RuntimeError("Cannot pickle cluster distribution before all permutations have been added.")
        state = {
            name: getattr(self, name) for name in (
                'name', 'meas', '_version', '_host', '_init_time',
                # settings ...
                'kind', 'threshold', 'tfce', 'tail', 'criteria', 'samples', 'tstart', 'tstop', 'parc',
                # data properties ...
                'dims', 'shape', '_nad_ax', '_vector_ax', '_criteria',
                # results ...
                'dt_original', 'dt_perm', 'n_clusters', '_dist_dims', 'dist', '_original_param_map', '_original_cluster_map', '_cids',
            )}
        state['_connectivity'] = self._adjacency
        state['version'] = 3
        return state

    def __setstate__(self, state):
        # backwards compatibility
        version = state.pop('version', 0)
        if version == 0:
            if '_connectivity_src' in state:
                del state['_connectivity_src']
                del state['_connectivity_dst']
            if '_connectivity' in state:
                del state['_connectivity']
            if 'N' in state:
                state['samples'] = state.pop('N')
            if '_version' not in state:
                state['_version'] = '< 0.11'
            if '_host' not in state:
                state['_host'] = 'unknown'
            if '_init_time' not in state:
                state['_init_time'] = None
            if 'parc' not in state:
                if state['_dist_dims'] is None:
                    state['parc'] = None
                else:
                    raise OldVersionError("This pickled test is from a previous version of Eelbrain and is not compatible anymore. Please recompute this test.")
            elif isinstance(state['parc'], tuple):
                if len(state['parc']) == 0:
                    state['parc'] = None
                elif len(state['parc']) == 1:
                    state['parc'] = state['parc'][0]
                else:
                    raise OldVersionError("This pickled test is from a previous version of Eelbrain and is not compatible anymore. Please recompute this test.")

            nad_ax = state['_nad_ax']
            state['dims'] = dims = state['dims'][1:]
            state['_connectivity'] = Adjacency(
                (dims[nad_ax],) + dims[:nad_ax] + dims[nad_ax + 1:],
                state['parc'])
        if version < 2:
            state['_vector_ax'] = None
        if version < 3:
            state['tfce'] = ['kind'] == 'tfce'

        self._adjacency = state.pop('_connectivity')
        for k, v in state.items():
            setattr(self, k, v)
        self.has_original = True
        self.finalize()

    def _repr_test_args(self, pmin):
        "Argument representation for TestResult repr"
        args = [f'samples={self.samples}']
        if pmin is not None:
            args.append(f"pmin={pmin!r}")
        elif self.kind == 'tfce':
            arg = f"tfce={self.tfce!r}"
            if self.tfce_warning:
                arg = f"{arg} [WARNING: The TFCE step is larger than the largest value in the data]"
            args.append(arg)
        if self.tstart is not None:
            args.append(f"tstart={self.tstart!r}")
        if self.tstop is not None:
            args.append(f"tstop={self.tstop!r}")
        for k, v in self.criteria.items():
            args.append(f"{k}={v!r}")
        return args

    def _repr_clusters(self):
        info = []
        if self.kind == 'cluster':
            if self.n_clusters == 0:
                info.append("no clusters")
            else:
                info.append(f"{self.n_clusters} clusters")

        if self.n_clusters and self.samples:
            info.append(f"{fmtxt.peq(self.probability_map.min())}")

        return info

    def _package_ndvar(self, x, info=None, external_shape=False):
        "Generate NDVar from map with internal shape"
        if not self.dims:
            if isinstance(x, np.ndarray):
                return x.item()
            return x
        if not external_shape and self._nad_ax:
            x = x.swapaxes(0, self._nad_ax)
        return NDVar(x, self.dims, self.name, info)

    def finalize(self):
        "Package results and delete temporary data"
        if self.dt_perm is None:
            self.dt_perm = current_time() - self._t0

        # original parameter map
        param_contours = {}
        if self.kind == 'cluster':
            if self.tail >= 0:
                param_contours[self.threshold] = (0.7, 0.7, 0)
            if self.tail <= 0:
                param_contours[-self.threshold] = (0.7, 0, 0.7)
        info = _info.for_stat_map(self.meas, contours=param_contours)
        self.parameter_map = self._package_ndvar(self._original_param_map, info)

        # TFCE map
        if self.kind == 'tfce':
            self.tfce_map = self._package_ndvar(self._original_cluster_map)
        else:
            self.tfce_map = None

        # cluster map
        if self.kind == 'cluster':
            self.cluster_map = self._package_ndvar(self._original_cluster_map)
        else:
            self.cluster_map = None

        self._finalized = True

    def data_for_permutation(self, raw=True):
        """Retrieve data flattened for permutation

        Parameters
        ----------
        raw : bool
            Return SharedMemory and array shape instead of a numpy array
            (for multiprocessing).
        """
        # get data in the right shape
        x = self.y_perm.x
        if self._vector_ax:
            x = np.moveaxis(x, self._vector_ax + 1, 1)
        if self._nad_ax is not None:
            dst = 1
            src = 1 + self._nad_ax
            if self._vector_ax is not None:
                dst += 1
                if self._vector_ax > self._nad_ax:
                    src += 1
            if dst != src:
                x = x.swapaxes(dst, src)
        # flat y shape
        ndims = 1 + (self._vector_ax is not None)
        n_flat = 1 if x.ndim == ndims else reduce(operator.mul, x.shape[ndims:])
        y_flat_shape = x.shape[:ndims] + (n_flat,)

        if not raw:
            return x.reshape(y_flat_shape)

        shared_memory = SharedMemory(create=True, size=x.nbytes)
        array = np.ndarray(x.shape, x.dtype, buffer=shared_memory.buf)
        array[:] = x
        return shared_memory, y_flat_shape, x.shape[ndims:]

    @staticmethod
    def _cluster_properties(cluster_map, cids):
        """Create a Dataset with cluster properties

        Parameters
        ----------
        cluster_map : NDVar
            NDVar in which clusters are marked by bearing the same number.
        cids : array_like of int
            Numbers specifying the clusters (must occur in cluster_map) which
            should be analyzed.

        Returns
        -------
        cluster_properties : Dataset
            Cluster properties. Which properties are included depends on the
            dimensions.
        """
        ndim = cluster_map.ndim
        n_clusters = len(cids)

        # setup compression
        compression = []
        for ax, dim in enumerate(cluster_map.dims):
            extents = np.empty((n_clusters, len(dim)), dtype=np.bool_)
            axes = tuple(i for i in range(ndim) if i != ax)
            compression.append((ax, dim, axes, extents))

        # find extents for all clusters
        c_mask = np.empty(cluster_map.shape, np.bool_)
        for i, cid in enumerate(cids):
            np.equal(cluster_map, cid, c_mask)
            for ax, dim, axes, extents in compression:
                np.any(c_mask, axes, extents[i])

        # prepare Dataset
        ds = Dataset()
        ds['id'] = Var(cids)

        for ax, dim, axes, extents in compression:
            properties = dim._cluster_properties(extents)
            if properties is not None:
                ds.update(properties)

        return ds

    def _cluster_property_labels(self):
        return [l for dim in self.dims for l in dim._cluster_property_labels()]

    def cluster(self, cluster_id: int) -> NDVar:
        """Retrieve a specific cluster as NDVar

        Parameters
        ----------
        cluster_id
            Cluster id.

        Returns
        -------
        NDVar
            NDVar of the cluster, 0 outside the cluster.

        Notes
        -----
        Clusters only have stable ids for thresholded cluster distributions.
        """
        if self.kind != 'cluster':
            raise RuntimeError(f'Only cluster-based tests have clusters with stable ids, this is a {self.kind} distribution. Instead, use .find_clusters() with maps=True, which will add the cluster map to the table returned by that function.')
        elif cluster_id not in self._cids:
            raise ValueError(f'No cluster with id {cluster_id!r}')

        out = self.parameter_map * (self.cluster_map == cluster_id)
        properties = self._cluster_properties(self.cluster_map, (cluster_id,))
        for k in properties:
            out.info[k] = properties[0, k]
        return out

    def clusters(
            self,
            pmin: float = None,
            maps: bool = True,
            **sub,
    ) -> Dataset:
        """Find significant clusters

        Parameters
        ----------
        pmin
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default True).
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if pmin is None:
            if self.samples > 0 and self.kind != 'cluster':
                pmin = 0.05
        elif self.samples == 0:
            raise ValueError(f"{pmin=}: Can not determine p values in distribution without permutations")

        if sub:
            param_map = self.parameter_map.sub(**sub)
        else:
            param_map = self.parameter_map

        if self.kind == 'cluster':
            if sub:
                cluster_map = self.cluster_map.sub(**sub)
                cids = np.setdiff1d(cluster_map.x, [0])
            else:
                cluster_map = self.cluster_map
                cids = np.array(self._cids)

            if len(cids):
                # measure original clusters
                cluster_v = ndimage.sum(param_map.x, cluster_map.x, cids)

                # p-values
                if self.samples:
                    # p-values: "the proportion of random partitions that
                    # resulted in a larger test statistic than the observed
                    # one" (179)
                    dist = self._aggregate_dist(**sub)
                    n_larger = np.sum(dist >= np.abs(cluster_v[:, None]), 1)
                    cluster_p = n_larger / self.samples

                    # select clusters
                    if pmin is not None:
                        idx = cluster_p <= pmin
                        cids = cids[idx]
                        cluster_p = cluster_p[idx]
                        cluster_v = cluster_v[idx]

                    # p-value corrected across parc
                    if sub:
                        dist = self._aggregate_dist()
                        n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                        cluster_p_corr = n_larger / self.samples
            else:
                cluster_v = cluster_p = cluster_p_corr = []

            ds = self._cluster_properties(cluster_map, cids)
            ds['v'] = Var(cluster_v)
            if self.samples:
                ds['p'] = Var(cluster_p)
                if sub:
                    ds['p_parc'] = Var(cluster_p_corr)

            threshold = self.threshold
        else:
            p_map = self.compute_probability_map(**sub)
            bin_map = np.less_equal(p_map.x, pmin)
            threshold = None

            # find clusters (reshape to internal shape for labelling)
            if self._nad_ax:
                bin_map = bin_map.swapaxes(0, self._nad_ax)
            if sub:
                raise NotImplementedError("sub")
                # need to subset adjacency!
            c_map, cids = label_clusters_binary(bin_map, self._adjacency)
            if self._nad_ax:
                c_map = c_map.swapaxes(0, self._nad_ax)

            # Dataset with cluster info
            cluster_map = NDVar(c_map, p_map.dims, "clusters")
            ds = self._cluster_properties(cluster_map, cids)
            ds.info['clusters'] = cluster_map
            min_pos = ndimage.minimum_position(p_map.x, c_map, cids)
            ds['p'] = Var([p_map.x[pos] for pos in min_pos])

        if 'p' in ds:
            ds['sig'] = star_factor(ds['p'])

        # expand clusters
        if maps:
            shape = (ds.n_cases,) + param_map.shape
            c_maps = np.empty(shape, dtype=param_map.x.dtype)
            c_mask = np.empty(param_map.shape, dtype=np.bool_)
            for i, cid in enumerate(cids):
                np.equal(cluster_map.x, cid, c_mask)
                np.multiply(param_map.x, c_mask, c_maps[i])

            # package ndvar
            dims = ('case',) + param_map.dims
            param_contours = {}
            if threshold:
                if self.tail >= 0:
                    param_contours[threshold] = (0.7, 0.7, 0)
                if self.tail <= 0:
                    param_contours[-threshold] = (0.7, 0, 0.7)
            info = _info.for_stat_map(self.meas, contours=param_contours)
            info['summary_func'] = np.sum
            ds['cluster'] = NDVar(c_maps, dims, info=info)
        else:
            ds.info['clusters'] = self.cluster_map

        return ds

    def find_peaks(self):
        """Find peaks in a TFCE distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self.kind == 'cluster':
            raise RuntimeError("Not a threshold-free distribution")

        param_map = self._original_param_map
        probability_map = self.probability_map.x
        if self._nad_ax:
            probability_map = probability_map.swapaxes(0, self._nad_ax)

        peaks = find_peaks(self._original_cluster_map, self._adjacency)
        peak_map, peak_ids = label_clusters_binary(peaks, self._adjacency)

        ds = Dataset()
        ds['id'] = Var(peak_ids)
        v = ds.add_empty_var('v')
        if self.samples:
            p = ds.add_empty_var('p')

        bin_buff = np.empty(peak_map.shape, bool)
        for i, id_ in enumerate(peak_ids):
            idx = np.equal(peak_map, id_, bin_buff)
            v[i] = param_map[idx][0]
            if self.samples:
                p[i] = probability_map[idx][0]

        return ds

    def compute_probability_map(self, **sub):
        """Compute a probability map

        Parameters
        ----------
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if not self.samples:
            raise RuntimeError("Can't compute probability without permutations")

        if self.kind == 'cluster':
            cpmap = np.ones(self.shape)
            if self.n_clusters:
                cids = self._cids
                dist = self._aggregate_dist(**sub)
                cluster_map = self._original_cluster_map
                param_map = self._original_param_map

                # measure clusters
                cluster_v = ndimage.sum(param_map, cluster_map, cids)

                # p-values: "the proportion of random partitions that resulted
                # in a larger test statistic than the observed one" (179)
                n_larger = np.sum(dist >= np.abs(cluster_v[:, None]), 1)
                cluster_p = n_larger / self.samples

                c_mask = np.empty(self.shape, bool)
                for i, cid in enumerate(cids):
                    np.equal(cluster_map, cid, c_mask)
                    cpmap[c_mask] = cluster_p[i]
            # revert to original shape
            if self._nad_ax:
                cpmap = cpmap.swapaxes(0, self._nad_ax)

            dims = self.dims
        else:
            if self.kind == 'tfce':
                stat_map = self.tfce_map
            else:
                if self.tail == 0:
                    stat_map = self.parameter_map.abs()
                elif self.tail < 0:
                    stat_map = -self.parameter_map
                else:
                    stat_map = self.parameter_map

            if sub:
                stat_map = stat_map.sub(**sub)
            dims = stat_map.dims if isinstance(stat_map, NDVar) else None

            cpmap = np.zeros(stat_map.shape) if dims else 0.
            if self.dist is None:  # flat stat-map
                cpmap += 1
            else:
                dist = self._aggregate_dist(**sub)
                idx = np.empty(stat_map.shape, bool)
                actual = stat_map.x if self.dims else stat_map
                for v in dist:
                    cpmap += np.greater_equal(v, actual, idx)
                cpmap /= self.samples

        if dims:
            return NDVar(cpmap, dims, self.name, _info.for_cluster_pmap())
        else:
            return cpmap

    def masked_parameter_map(self, pmin=0.05, name=None, **sub):
        """Parameter map masked by significance

        Parameters
        ----------
        pmin : scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, ``pmin=1`` includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map, masked with
            p <= pmin.
        """
        if not 1 >= pmin > 0:
            raise ValueError(f"pmin={pmin}: needs to be between 1 and 0")

        if name is None:
            name = self.parameter_map.name

        if sub:
            param_map = self.parameter_map.sub(**sub)
        else:
            param_map = self.parameter_map

        if pmin == 1:
            if self.kind != 'cluster':
                raise ValueError(f"{pmin=} is only a valid mask for threshold-based cluster tests")
            mask = self.cluster_map == 0
        else:
            probability_map = self.compute_probability_map(**sub)
            mask = probability_map > pmin
        return param_map.mask(mask, name)

    @cached_property
    def probability_map(self):
        if self.samples:
            return self.compute_probability_map()
        else:
            return None

    @cached_property
    def _default_plot_obj(self):
        if self.samples:
            return [[self.parameter_map, self.probability_map]]
        else:
            return [[self.parameter_map]]

    def info_list(self, title="Computation Info"):
        "List with information on computation"
        l = fmtxt.List(title)
        l.add_item("Eelbrain version:  %s" % self._version)
        l.add_item("Host Computer:  %s" % self._host)
        if self._init_time is not None:
            l.add_item("Created:  %s" % datetime.fromtimestamp(self._init_time)
                       .strftime('%y-%m-%d %H:%M'))
        l.add_item("Original time:  %s" % timedelta(seconds=round(self.dt_original)))
        l.add_item("Permutation time:  %s" % timedelta(seconds=round(self.dt_perm)))
        return l


class _MergedTemporalClusterDist:
    """Merge permutation distributions from multiple tests"""

    def __init__(self, cdists):
        if isinstance(cdists[0], list):
            self.effects = [d.name for d in cdists[0]]
            self.samples = cdists[0][0].samples
            dist = {}
            for i, effect in enumerate(self.effects):
                if any(d[i].n_clusters for d in cdists):
                    dist[effect] = np.column_stack([d[i].dist for d in cdists if d[i].dist is not None])
            if len(dist):
                dist = {c: d.max(1) for c, d in dist.items()}
        else:
            self.samples = cdists[0].samples
            if any(d.n_clusters for d in cdists):
                dist = np.column_stack([d.dist for d in cdists if d.dist is not None])
                dist = dist.max(1)
            else:
                dist = None

        self.dist = dist

    def correct_cluster_p(self, res):
        clusters = res.find_clusters()
        keys = list(clusters.keys())

        if not clusters.n_cases:
            return clusters
        if isinstance(res, MultiEffectNDTest):
            keys.insert(-1, 'p_parc')
            cluster_p_corr = []
            for cl in clusters.itercases():
                n_larger = np.sum(self.dist[cl['effect']] > np.abs(cl['v']))
                cluster_p_corr.append(float(n_larger) / self.samples)
        else:
            keys.append('p_parc')
            vs = np.array(clusters['v'])
            n_larger = np.sum(self.dist > np.abs(vs[:, None]), 1)
            cluster_p_corr = n_larger / self.samples
        clusters['p_parc'] = Var(cluster_p_corr)
        clusters = clusters[keys]

        return clusters


def distribution_worker(dist, in_queue, kill_beacon):
    "Worker that accumulates values and places them into the distribution"
    for i in trange(dist.shape[0], desc="Permutation test", unit=' permutations', disable=CONFIG['tqdm']):
        dist[i] = in_queue.get()
        if kill_beacon.is_set():
            break


def permutation_worker(in_queue, out_queue, memory_name, y_flat_shape, stat_map_shape, test_func, args, map_args, kill_beacon):
    "Worker function for permutation test"
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])

    shared_memory = SharedMemory(memory_name)
    y = np.ndarray(y_flat_shape, np.float64, buffer=shared_memory.buf)
    stat_map = np.empty(stat_map_shape)
    stat_map_flat = stat_map.ravel()
    map_processor = get_map_processor(*map_args)
    while not kill_beacon.is_set():
        perm = in_queue.get()
        if perm is None:
            break
        test_func(y, *args, stat_map_flat, perm)
        max_v = map_processor.max_stat(stat_map)
        out_queue.put(max_v)
    shared_memory.close()


def run_permutation(test_func, dist, iterator, *args):
    "Perform permutation test"
    if CONFIG['n_workers']:
        workers, out_queue, kill_beacon, shared_memory = setup_workers(test_func, dist, args)

        try:
            for perm in iterator:
                out_queue.put(perm)

            for _ in range(len(workers) - 1):
                out_queue.put(None)

            logger = logging.getLogger(__name__)
            for w in workers:
                w.join()
                logger.debug("worker joined")
        except KeyboardInterrupt:
            kill_beacon.set()
            raise
        finally:
            shared_memory.close()
            shared_memory.unlink()
    else:
        y = dist.data_for_permutation(False)
        map_processor = get_map_processor(*dist.map_args)
        stat_map = np.empty(dist.shape)
        stat_map_flat = stat_map.ravel()
        for i, perm in enumerate(iterator):
            test_func(y, *args, stat_map_flat, perm)
            dist.dist[i] = map_processor.max_stat(stat_map)
    dist.finalize()


def setup_workers(test_func, dist, func_args):
    "Initialize workers for permutation test"
    logger = logging.getLogger(__name__)
    logger.debug("Setting up %i worker processes...", CONFIG['n_workers'])
    permutation_queue = mpc.SimpleQueue()
    dist_queue = mpc.SimpleQueue()
    kill_beacon = mpc.Event()

    restore_main_spec()

    # permutation workers
    shared_memory, y_flat_shape, stat_map_shape = dist.data_for_permutation()
    args = (permutation_queue, dist_queue, shared_memory.name, y_flat_shape, stat_map_shape, test_func, func_args, dist.map_args, kill_beacon)
    workers = []
    for _ in range(CONFIG['n_workers']):
        w = mpc.Process(target=permutation_worker, args=args)
        w.start()
        workers.append(w)

    # distribution worker
    args = (dist.dist, dist_queue, kill_beacon)
    w = Thread(target=distribution_worker, args=args)
    w.start()
    workers.append(w)

    return workers, permutation_queue, kill_beacon, shared_memory  # return shared_memory so that it does not get garbage collected


def run_permutation_me(
        test: MPTestMapper,
        dists: List[NDPermutationDistribution],
        iterator: Iterable[np.ndarray],
):
    dist = dists[0]
    if dist.kind == 'cluster':
        thresholds = tuple(d.threshold for d in dists)
    else:
        thresholds = None

    if CONFIG['n_workers']:
        workers, out_queue, kill_beacon, shared_memory = setup_workers_me(test, dists, thresholds)

        try:
            for perm in iterator:
                out_queue.put(perm)

            for _ in range(len(workers) - 1):
                out_queue.put(None)

            logger = logging.getLogger(__name__)
            for w in workers:
                w.join()
                logger.debug("worker joined")
        except KeyboardInterrupt:
            kill_beacon.set()
            raise
        finally:
            shared_memory.close()
            shared_memory.unlink()
    else:
        y = dist.data_for_permutation(False)
        map_processor = get_map_processor(*dist.map_args)

        stat_maps = test.preallocate(dist.shape)
        if thresholds:
            stat_maps_iter = tuple(zip(stat_maps, thresholds, dists))
        else:
            stat_maps_iter = tuple(zip(stat_maps, dists))

        for i, perm in enumerate(iterator):
            test.map(y, perm)
            if thresholds:
                for m, t, d in stat_maps_iter:
                    if d.do_permutation:
                        d.dist[i] = map_processor.max_stat(m, t)
            else:
                for m, d in stat_maps_iter:
                    if d.do_permutation:
                        d.dist[i] = map_processor.max_stat(m)

    for d in dists:
        if d.do_permutation:
            d.finalize()


def setup_workers_me(test_func, dists, thresholds):
    "Initialize workers for multi-effect permutation test"
    logger = logging.getLogger(__name__)
    logger.debug("Setting up %i worker processes...", CONFIG['n_workers'])
    permutation_queue = mpc.SimpleQueue()
    dist_queue = mpc.SimpleQueue()
    kill_beacon = mpc.Event()

    restore_main_spec()

    # permutation workers
    dist = dists[0]
    shared_memory, y_flat_shape, stat_map_shape = dist.data_for_permutation()
    args = (permutation_queue, dist_queue, shared_memory.name, y_flat_shape, stat_map_shape, test_func, dist.map_args, thresholds, kill_beacon)
    workers = []
    for _ in range(CONFIG['n_workers']):
        w = mpc.Process(target=permutation_worker_me, args=args)
        w.start()
        workers.append(w)

    # distribution worker
    args = ([dist.dist for dist in dists], dist_queue, kill_beacon)
    w = Thread(target=distribution_worker_me, args=args)
    w.start()
    workers.append(w)

    return workers, permutation_queue, kill_beacon, shared_memory


def permutation_worker_me(in_queue, out_queue, memory_name, y_flat_shape, stat_map_shape, test, map_args, thresholds, kill_beacon):
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])

    shared_memory = SharedMemory(memory_name)
    y = np.ndarray(y_flat_shape, np.float64, buffer=shared_memory.buf)
    iterator = test.preallocate(stat_map_shape)
    if thresholds:
        iterator = tuple(zip(iterator, thresholds))
    else:
        iterator = tuple(iterator)
    map_processor = get_map_processor(*map_args)
    while not kill_beacon.is_set():
        perm = in_queue.get()
        if perm is None:
            break
        test.map(y, perm)

        if thresholds:
            max_v = [map_processor.max_stat(m, t) for m, t in iterator]
        else:
            max_v = [map_processor.max_stat(m) for m in iterator]
        out_queue.put(max_v)
    shared_memory.close()


def distribution_worker_me(dists, in_queue, kill_beacon):
    "Worker that accumulates values and places them into the distribution"
    for dist in dists:
        if dist is not None:
            n = dist.shape[0]
            break
    else:
        raise RuntimeError("No distirbutions")
    for i in trange(n, desc="Permutation test", unit=' permutations', disable=CONFIG['tqdm']):
        for dist, v in zip(dists, in_queue.get()):
            if dist is not None:
                dist[i] = v
        if kill_beacon.is_set():
            break


# Backwards compatibility for pickling
_ClusterDist = NDPermutationDistribution
corr = Correlation
ttest_1samp = TTestOneSample
ttest_ind = TTestIndependent
ttest_rel = TTestRelated
t_contrast_rel = TContrastRelated
anova = ANOVA
