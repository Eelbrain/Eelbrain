# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""General linear model

References
----------
Fox, J. (2008). Applied regression analysis and generalized linear models.
    2nd ed. Sage, Los Angeles.
Hopkins, K. D. (1976). A Simplified Method for Determining Expected Mean
    Squares and Error Terms in the Analysis of Variance. Journal of
    Experimental Education, 45(2), 13--18.
"""
from typing import Sequence

import numpy as np
from scipy.linalg import lstsq
import scipy.stats

from .. import fmtxt
from functools import cached_property
from .._data_obj import (
    ModelArg, IndexArg, VarArg,
    Dataset, Model, asmodel, assub, asvar, assert_has_no_empty_cells, find_factors,
    hasrandom, is_higher_order_effect, isbalanced, iscategorial, isnestedin)
from .._utils import deprecate_ds_arg
from .opt import anova_fmaps, anova_full_fmaps, lm_res_ss, ss
from . import test


# Method to use for least squares estimation:
# (0) Use scipy.linalg.lstsq
# (1) Use lstsq after Fox (2008) with caching of the model transformation
_lm_lsq = 0  # for the LM class


def hopkins_ems(x):
    """Find components of the F-test denominator according to Hopkins (1976)

    Return a dictionary supplying for each of a model's effects the components
    of the F-test denominator (error term).

    "The E(MS) for any source of variation for any ANOVA model, in addition
    to the specified effect (main effect or interaction), includes the
    interaction of this effect with any random factor or combinations of
    random factors, plus any random effect nested within the specified effect,
    plus any random effect that either crosses or is nested within each
    ingredient and the specified effect." (Hopkins, 1976: 18)

    Parameters
    ----------
    x : Model
        ANOVA model. Needs to balanced and completely specified.
    """
    if x.df_error > 0:
        raise x._incomplete_error("Hopkins E(MS) estimate")
    elif not any(f.random for f in find_factors(x)):
        raise ValueError(
            f"model={x.name}; need at least one random effect in a fully "
            f"specified model")

    return tuple(_find_hopkins_ems(e, x) for e in x.effects)


def _hopkins_ems_array(x):
    """Construct E(MS) array"""
    n = len(x.effects)
    out = np.zeros((n, n), dtype=np.int8)
    for row, erow in enumerate(x.effects):
        for col, ecol in enumerate(x.effects):
            out[row, col] = _hopkins_test(erow, ecol)
    return out


def _hopkins_test(e, e2):
    """Test whether e2 is in the E(MS) of e.

    Parameters
    ----------
    e : effect
        effect whose E(MS) is being constructed
    e2 : effect
        Model effect which is tested for inclusion in E(MS) of e
    """
    if e is e2:
        return False
    e_factors = find_factors(e)
    e2_factors = find_factors(e2)
    return (
        all((f in e_factors or f.random) for f in e2_factors)
        and all((f in e2_factors or isnestedin(e2, f)
                 or any(isnestedin(f2, f) for f2 in e2_factors))
                for f in e_factors)
    )


def _find_hopkins_ems(e, x):
    "Tuple with all effects included in the Hopkins E(MS)"
    return tuple(e2 for e2 in x.effects if _hopkins_test(e, e2))


class LM:
    """Fit a linear model to a dependent variable

    Parameters
    ----------
    y
        Dependent variable.
    x
        Model.
    sub
        Only use part of the data
    data
        Dataset to use data from.

    Attributes
    ----------
    f, p : scalar
        Test of the null-hypothesis that the model does not explain a
        significant amount of the variance in the dependent variable.
    """
    def __init__(
            self,
            y: VarArg,
            x: ModelArg,
            sub: IndexArg = None,
            data: Dataset = None,
    ):
        # prepare input
        sub = assub(sub, data)
        y = asvar(y, sub, data)
        x = asmodel(x, sub, data)

        assert len(y) == len(x)
        assert x.df_error > 0

        # fit
        p = x._parametrize()
        if _lm_lsq == 0:  # use scipy (faster)
            beta, ss_res, _, _ = lstsq(p.x, y.x)
        elif _lm_lsq == 1:  # Fox
            # estimate least squares approximation
            beta = np.dot(p.projector, y.x)
            # estimate
            y_est = np.dot(p.x, beta)
            self._residuals = residuals = y.x - y_est
            ss_res = np.sum(residuals ** 2)
        else:
            raise ValueError

        # SS total
        ss_total = self.ss_total = np.sum((y.x - y.mean()) ** 2)
        df_total = self.df_total = x.df_total
        self.ms_total = ss_total / df_total

        # SS residuals
        self.ss_res = ss_res
        df_res = self.df_res = x.df_error
        self.ms_res = ss_res / df_res

        # SS explained
        ss_model = self.ss = self.ss_model = ss_total - ss_res
        df_model = self.df = self.df_model = x.df - 1  # don't count intercept
        self.ms_model = self.ms = ss_model / df_model

        # store stuff
        self.y = y
        self.x = x
        self._p = p
        self.sub = sub
        self.beta = beta

    def __repr__(self):
        # repr kwargs
        args = [self.y.name, self.x.name]
        if self.sub:
            args.append('sub=%r' % getattr(self.sub, 'name', '<...>'))
        return "LM(%s)" % ', '.join(args)

    def anova(self, title='ANOVA', empty=True, ems=False):
        """ANOVA table for the linear model"""
        x = self.x
        values = np.dot(self._p.x, self.beta)

        if x.df_error == 0:
            e_ms = hopkins_ems(x)
        elif hasrandom(x):
            raise x._incomplete_error("Mixed effects ANOVA")
        else:
            e_ms = False

        # table head
        table = fmtxt.Table('l' + 'r' * (5 + ems))
        if title:
            table.title(title)

        if not isbalanced(x):
            table.caption("Warning: Model is unbalanced, use test.ANOVA")

        table.cell()
        headers = ["SS", "df", "MS"]
        if ems:
            headers += ["E(MS)"]
        headers += ["F", "p"]
        for hd in headers:
            table.cell(hd, r"\textbf", just='c')
        table.midrule()

        # MS for factors (Needed for models involving random effects)
        ms_s = {}
        ss_s = {}
        for e in x.effects:
            idx = x.full_index[e]
            ss_s[e] = ss = np.sum(values[:, idx].sum(1) ** 2)
            ms_s[e] = (ss / e.df)

        # table body
        results = {}
        for i, e in enumerate(x.effects):
            ms = ms_s[e]
            if e_ms:
                e_ems = e_ms[i]
                df_d = sum(c.df for c in e_ems)
                ms_d = sum(ms_s[c] for c in e_ems)
                e_ms_name = ' + '.join(repr(c) for c in e_ems)
            else:
                df_d = self.df_res
                ms_d = self.ms_res
                e_ms_name = "Res"

            # F-test
            if ms_d:
                f = ms / ms_d
                p = 1 - scipy.stats.distributions.f.cdf(f, e.df, df_d)
                f_tex = fmtxt.stat(f, stars=test.star(p))
            else:
                f_tex = None
                p = None
            # add to table
            if e_ms_name or empty:
                table.cell(e.name)
                table.cell(ss_s[e])
                table.cell(e.df)
                table.cell(ms)
                if ems:
                    table.cell(e_ms_name)
                table.cell(f_tex)
                table.cell(fmtxt.P(p))
            # store results
            results[e.name] = {'SS': ss, 'df': e.df, 'MS': ms, 'E(MS)': e_ms_name, 'F': f, 'p': p}

        # Residuals
        if self.df_res > 0:
            table.cell("Residuals")
            table.cell(self.ss_res)
            table.cell(self.df_res)
            table.cell(self.ms_res)

        return table

    @cached_property
    def f(self):
        return self.ms_model / self.ms_res

    @cached_property
    def p(self):
        return scipy.stats.distributions.f.sf(self.f, self.df_model, self.df_res)

    @cached_property
    def regression_table(self):
        """
        Not fully implemented!

        A table containing slope coefficients for all effects.

        """
        # header
        table = fmtxt.Table('l' * 4)
        df = self.x.df_error
        table.cell()
        table.cell(fmtxt.symbol('\\beta'))
        table.cell(fmtxt.symbol('T', df))
        table.cell(fmtxt.symbol('p'))
        table.midrule()
        # body
        q = 1
        ne = len(self.x.effects)
        for ie, e in enumerate(self.x.effects):
            table.cell(e.name + ':')
            table.endline()
            for i, name in enumerate(e.beta_labels):  # Fox pp. 106 ff.
                beta = self.beta[q + i]
                t = 0
                p = 0
                # todo: T/p
                table.cell(name)
                table.cell(beta)
                table.cell(t)
                table.cell(fmtxt.p(p))
            q += e.df
            if ie < ne - 1:
                table.empty_row()
        return table

    @cached_property
    def residuals(self):
        return self.y.x - np.dot(self._p.x, self.beta)


def _nd_anova(x):
    "Create an appropriate ANOVA mapper"
    x = asmodel(x)
    assert_has_no_empty_cells(x)
    if hasrandom(x):
        if not iscategorial(x):
            raise NotImplementedError("Random effects ANOVA with continuous predictors")
        elif x.df_error != 0:
            raise x._incomplete_error("Mixed effects ANOVA")
        elif isbalanced(x):
            return _BalancedMixedNDANOVA(x)
    elif iscategorial(x) and isbalanced(x):
        return _BalancedFixedNDANOVA(x)
    return _IncrementalNDANOVA(x)


class MPTestMapper:
    "Baseclass for a test to be run with multiprocessing"

    def preallocate(self, y_shape: Sequence[int]) -> np.ndarray:  #
        "Pre-allocate an output array container"
        raise NotImplementedError()

    def map(
            self,
            y: np.ndarray,  # (n_cases, ...)
            perm: np.ndarray = None,  # (n_cases,) permutation index
    ) -> None:
        "Process y and pu result into output array container"
        raise NotImplementedError()


class _NDANOVA(MPTestMapper):
    """Efficiently fit a model to multiple dependent variables."""
    def __init__(self, x, effects, dfs_denom):
        self.x = x
        self.p = x._parametrize()
        self._n_obs = len(x)
        self.effects = effects
        self.n_effects = len(effects)
        self.dfs_nom = [e.df for e in effects]
        self.dfs_denom = dfs_denom
        self._flat_f_map = None

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.x.name)

    def map(self, y, perm=None):
        """Fit the model to multiple dependent variables

        Parameters
        ----------
        y : np.array (n_cases, ...)
            Assumes that the first dimension of y provides cases.
            Other than that, shape is free to vary and output shape will match
            input shape.
        perm : None | array (n_cases, )
            Permutation.

        Returns
        -------
        f_maps : list
            A list with maps of F values (order corresponding to self.effects).
        """
        if y.shape[0] != self._n_obs:
            raise ValueError(f"y has wrong number of observations ({y.shape[0]}, model has {self._n_obs})")

        # find result container
        if self._flat_f_map is None:
            shape = (self.n_effects,) + y.shape[1:]
            f_map = np.empty(shape)
            flat_f_map = f_map.reshape((self.n_effects, -1))
        else:
            f_map = None
            flat_f_map = self._flat_f_map

        if y.ndim > 2:
            y = y.reshape((self._n_obs, -1))

        self._map(y, flat_f_map, perm)
        return f_map

    def _map(self, y, flat_f_map, perm):
        raise NotImplementedError

    def p_maps(self, f_maps):
        """Convert F-maps for uncorrected p-maps

        Parameters
        ----------
        f_maps : numpy array (n_effects, ...)
            Maps of f-values as returned by self.map().

        Returns
        -------
        p_maps : numpy array (n_effects, ...)
            Maps of uncorrected p values (corresponding to f_maps).
        """
        p_maps = np.empty_like(f_maps)
        for i in range(len(f_maps)):
            p_maps[i] = scipy.stats.f.sf(f_maps[i], self.dfs_nom[i], self.dfs_denom[i])
        return p_maps

    def preallocate(self, y_shape):
        """Pre-allocate an output array container.

        Parameters
        ----------
        y_shape : tuple
            Data shape (excluding case), will allow preallocation of containers
            for results.

        Returns
        -------
        f_maps : array
            Properly shaped output array. Every time .map() is called, the
            content of this array will change (and map() will not return
            anything)
        """
        shape = (self.n_effects, *y_shape)
        f_map = np.empty(shape)
        self._flat_f_map = f_map.reshape((self.n_effects, -1))
        return f_map


class _BalancedNDANOVA(_NDANOVA):
    "For balanced but not fully specified models"
    def __init__(self, x, effects, dfs_denom):
        _NDANOVA.__init__(self, x, effects, dfs_denom)

        self._effect_to_beta = x._effect_to_beta
        self._x_full_perm = None
        self._x_proj_perm = None

    def _map(self, y, flat_f_map, perm):
        if perm is None:
            x_full = self.p.x
            x_proj = self.p.projector
        else:
            if self._x_full_perm is None:
                self._x_full_perm = np.empty_like(self.p.x)
                self._x_proj_perm = np.empty_like(self.p.projector)
            x_full = self.p.x.take(perm, 0, self._x_full_perm)
            x_proj = self.p.projector.take(perm, 1, self._x_proj_perm)

        self._map_balanced(y, flat_f_map, x_full, x_proj)

    def _map_balanced(self, y, flat_f_map, x_full, xsinv):
        raise NotImplementedError


class _BalancedFixedNDANOVA(_BalancedNDANOVA):
    "For balanced but not fully specified models"
    def __init__(self, x):
        if x.df_error <= 0:
            raise ValueError(f"Model {x.name} is overspecified with df_error={x.df_error}; all effects are fixed, should one be random?")
        effects = x.effects
        dfs_denom = (x.df_error,) * len(effects)
        _BalancedNDANOVA.__init__(self, x, effects, dfs_denom)
        self.df_error = x.df_error

    def _map_balanced(self, y, flat_f_map, x_full, xsinv):
        anova_fmaps(y, x_full, xsinv, flat_f_map, self._effect_to_beta, self.df_error)


class _BalancedMixedNDANOVA(_BalancedNDANOVA):
    """For balanced, fully specified models.

    Object for efficiently fitting a model to multiple dependent variables.

    Parameters
    ----------
    x : Model
        Model which will be fitted to the data.

    Notes
    -----
    E(MS) for F statistic is determined after Hopkins (1976)
    """
    def __init__(self, x):
        e_ms = hopkins_ems(x)
        df_den = tuple(sum(e.df for e in ms_effects) for ms_effects in e_ms)
        keep = tuple(i for i, df in enumerate(df_den) if df)
        effects = tuple(x.effects[i] for i in keep)
        dfs_denom = tuple(df_den[i] for i in keep)
        _BalancedNDANOVA.__init__(self, x, effects, dfs_denom)
        self._e_ms_array = _hopkins_ems_array(x)

    def _map_balanced(self, y, flat_f_map, x_full, xsinv):
        anova_full_fmaps(y, x_full, xsinv, flat_f_map, self._effect_to_beta, self._e_ms_array)


class _IncrementalNDANOVA(_NDANOVA):
    def __init__(self, x):
        comparisons = IncrementalComparisons(x)
        _NDANOVA.__init__(self, x, comparisons.effects, comparisons.dfs_denom)

        self._comparisons = comparisons
        self._ss_diff = None
        self._ms_e = None
        self._ss_res = None

        self._x_orig = {}
        self._full_ss_i = -1
        for m, i in comparisons.relevant_models:
            if m is None:  # intercept only
                self._x_orig[i] = None
                self._full_ss_i = i
            else:
                p = m._parametrize()
                self._x_orig[i] = (p.x, p.projector)
        if comparisons.mixed and self._full_ss_i == -1:
            # need full SS
            self._x_orig[-1] = None
        self._x_perm = None

    def preallocate(self, y_shape):
        f_map = _NDANOVA.preallocate(self, y_shape)

        shape = self._flat_f_map.shape[1]
        self._ss_diff = np.empty(shape)
        self._ms_e = np.empty(shape)
        self._ss_res = {i: np.empty(shape) for i in self._x_orig.keys()}
        return f_map

    def _map(self, y, flat_f_map, perm):
        if self._ss_diff is None:
            shape = y.shape[1]
            ss_diff = ms_diff = np.empty(shape)
            ms_e = np.empty(shape)
            ss_res = {i: np.empty(shape) for i in self._x_orig.keys()}
        else:
            ss_diff = ms_diff = self._ss_diff
            ms_e = self._ms_e
            ss_res = self._ss_res

        if perm is None:
            x_dict = self._x_orig
        else:
            x_orig = self._x_orig
            x_dict = self._x_perm
            if x_dict is None:
                self._x_perm = x_dict = {}
                for i, x in x_orig.items():
                    if x is None:
                        x_dict[i] = None
                    else:
                        x_dict[i] = (np.empty_like(x[0]), np.empty_like(x[1]))

            for i in x_dict:
                if x_dict[i]:
                    x_orig[i][0].take(perm, 0, x_dict[i][0])
                    x_orig[i][1].take(perm, 1, x_dict[i][1])

        # calculate ss_res for all models
        for i, x in x_dict.items():
            if x is None:  # TODO:  use the same across permutations?
                ss(y, ss_res[i])
            else:
                x_full, xsinv = x
                lm_res_ss(y, x_full, xsinv, ss_res[i])

        # incremental comparisons
        if not self._comparisons.mixed:
            np.divide(ss_res[0], self.x.df_error, ms_e)
        for i, (i_test, (i1, i0)) in enumerate(self._comparisons.comparisons.items()):
            if self._comparisons.mixed:
                i_ems = self._comparisons.ems_idx[i_test]
                np.subtract(ss_res[self._full_ss_i], ss_res[i_ems], ms_e)
                np.divide(ms_e, self.dfs_denom[i], ms_e)
            df_diff = self._comparisons.x.effects[i_test].df
            np.subtract(ss_res[i0], ss_res[i1], ss_diff)
            np.divide(ss_diff, df_diff, ms_diff)
            np.divide(ms_diff, ms_e, flat_f_map[i])


def effect_id(effects):
    return tuple(map(id, effects))


class IncrementalComparisons:
    """Determine models for incremental comparisons

    Parameters
    ----------
    x : Model
        Model for which to derive incremental comparisons.

    Attributes
    ----------
    mixed : bool
        Model contains any random effect.
    effects : tuple[Effect]
        All testable effects.
    dfs_denom : tuple[int]
        Denominator df for each effect.
    models : dict {int -> Model}
        Models as indexed from comparisons.
    comparisons : {int: (int, int)}
        Comparisons for testable effects in x; ordered dictionary mapping
        ``{effect_id: (model_1_id, int model_0_id)}``.
    ems_idx : list[Union[int, None]]
        For each effect, the model-id of the E(MS) model.
    """
    def __init__(self, x):
        if x.df_error == 0:
            self.mixed = is_mixed = True
        elif x.df_error > 0:
            if hasrandom(x):
                raise x._incomplete_error("Mixed effects ANOVA")
            self.mixed = is_mixed = False
        else:
            raise ValueError("Model Overdetermined")

        self.x = x
        self.comparisons = {}  # {effect_i: (model1_id, model0_id)}
        self.models = {0: x}  # int -> Model
        relevant_models = set()
        model_idxs = {effect_id(x.effects): 0}  # effect tuple -> ind
        next_idx = 1
        self.ems_idx = []

        if is_mixed:
            # find relevant models for E(MS) computation
            for e_test, e_ms_effects in zip(x.effects, hopkins_ems(x)):
                if not e_ms_effects:
                    idx = None
                else:
                    id_ = effect_id(e_ms_effects)
                    if id_ in model_idxs:
                        idx = model_idxs[id_]
                    else:
                        idx = model_idxs[id_] = next_idx
                        next_idx += 1
                        self.models[idx] = Model(e_ms_effects)
                        relevant_models.add(idx)
                self.ems_idx.append(idx)

        # Find comparisons for each effect
        for i_test, e_test in enumerate(x.effects):
            model0_effects = tuple(e for e in x.effects if e is not e_test and not is_higher_order_effect(e, e_test))
            model0_id = effect_id(model0_effects)

            # get model 0
            if model0_id in model_idxs:
                idx0 = model_idxs[model0_id]
                model0 = self.models[idx0]
            else:
                idx0 = model_idxs[model0_id] = next_idx
                next_idx += 1
                if len(model0_effects):
                    model0 = Model(model0_effects)
                else:
                    model0 = None

            # test whether comparison is feasible
            if model0 is None:
                df_res_0 = x.df_total
            else:
                df_res_0 = model0.df_error

            if e_test.df > df_res_0:
                # print(f"Skipping {e_test}: overspecified")
                continue
            elif is_mixed and self.ems_idx[i_test] is None:
                # print("Skipping {e_test}: no E(MS)")
                continue
            elif idx0 not in self.models:
                self.models[idx0] = model0

            # get model 1
            model1_effects = model0_effects + (e_test,)
            model1_id = effect_id(model1_effects)
            if model1_id in model_idxs:
                idx1 = model_idxs[model1_id]
            else:
                idx1 = model_idxs[model1_id] = next_idx
                next_idx += 1
                self.models[idx1] = Model(model1_effects)

            # store comparison
            self.comparisons[i_test] = (idx1, idx0)
            relevant_models.add(idx1)
            relevant_models.add(idx0)

        # for i_ss, model in self.models.iteritems()

        self.relevant_models = tuple((self.models[idx], idx) for idx in relevant_models)
        self.effects = tuple(x.effects[i] for i in self.comparisons.keys())
        if self.mixed:
            self.dfs_denom = tuple(self.models[self.ems_idx[i_test]].df - 1 for i_test in self.comparisons)
        else:
            self.dfs_denom = (self.x.df_error,) * len(self.effects)

    def __repr__(self):
        return f"IncrementalComparisons({self.x.name})"

    def __str__(self):
        out = ["Incremental comparisons:"]
        for i_test, (_, m0) in self.comparisons.items():
            out.append(f"  {self.x.effects[i_test].name} > {self.models[m0].name}")
        if self.mixed:
            out.append("E(MS):")
            for i_test in self.comparisons:
                ems = self.ems_idx[i_test]
                desc = "N/A" if ems is None else self.models[ems].name
                out.append(f"  {self.x.effects[i_test].name}: {desc}")
        return '\n'.join(out)


class IncrementalFTest:
    """Incremental F-Test between two linear models

    Test the null hypothesis that the model of lm1 does not explain more
    variance than that of lm0 (with model_1 == model_0 + q factors,  q > 0).
    If lm1 is None it is assumed that lm1 is the full model with 0 residuals
    (Fox 2008, p. 109 f.).

    Parameters
    ----------
    lm1, lm0 : Model
        The two models to compare.
    ms_e, df_e : scalar | None
        Parameters for random effects models: the Expected value of MS;
        if None, the error MS of lm1 is used (valid for fixed effects
        models).

    Attributes
    ----------
    lm1 : Model
        The extended model.
    lm0 : Model
        The control model.
    ss : scalar
        the difference in the SS explained by the two models.
    df : int
        The difference in df between the two models.
    ms : scalar
        The MS of the difference.
    f, p : scalar
        F and p valuer of the comparison.
    """
    def __init__(self, lm1, lm0, ms_e=None, df_e=None, name=None):
        if lm1 is None:
            lm1_ss_res = 0
            lm1_ms_res = 0
            lm1_df_res = 0
        else:
            lm1_ss_res = lm1.ss_res
            lm1_ms_res = lm1.ms_res
            lm1_df_res = lm1.df_res

        if ms_e is None:
            assert df_e is None
            ms_e = lm1_ms_res
            df_e = lm1_df_res
        else:
            assert df_e is not None

        ss_diff = lm0.ss_res - lm1_ss_res
        df_diff = lm0.df_res - lm1_df_res
        ms_diff = ss_diff / df_diff

        if df_e > 0:
            f = ms_diff / ms_e
            p = scipy.stats.f.sf(f, df_diff, df_e)
        else:
            f = None
            p = None

        self.lm0 = lm0
        self.lm1 = lm1
        self.ss = ss_diff
        self.ms = ms_diff
        self.df = df_diff
        self.ms_e = ms_e
        self.dfe = df_e
        self.f = f
        self.p = p
        self.name = name

    def _asfmtext(self, **_):
        return fmtxt.FMText([fmtxt.eq('F', self.f, (self.df, self.dfe)), ', ', fmtxt.peq(self.p)])

    def __repr__(self):
        name = f' {self.name!r}' if self.name else ''
        return f"<{self.__class__.__name__}{name}: {self._asfmtext()}"


@deprecate_ds_arg
class ANOVA:
    """Univariate ANOVA.

    Parameters
    ----------
    y
        dependent variable
    x
        Model to fit to y
    sub
        Only use part of the data.
    data
        Dataset to use data from.
    title
        Title for the results table.
    caption
        Caption for the results table.

    Attributes
    ----------
    effects : tuple of str
        Names of the tested effects, in the same order as in other attributes.
    f_tests : tuple
        F-tests for all effects.
    residuals : None | tuple
        For fixed effects models, a ``(ss, df, ms)`` tuple; for mixed effects
        models ``None``.

    Notes
    -----
    Mixed effects models require balanced models and full model specification
    so that E(MS) can be estimated according to Hopkins (1976).

    Examples
    --------
    Simple n-way between subjects ANOVA::

        >>> ds = datasets.get_uv(nrm=True)
        >>> print(test.ANOVA('fltvar', 'A*B', data=ds))
                        SS   df      MS          F        p
        ---------------------------------------------------
        A            28.69    1   28.69   25.69***   < .001
        B             0.04    1    0.04    0.03        .855
        A x B         1.16    1    1.16    1.04        .310
        Residuals    84.85   76    1.12
        ---------------------------------------------------
        Total       114.74   79

    For repeated measures designs, whether a factors is fixed or random is
    determined based on the :attr:`Factor.random` attribute, which is usually
    specified at creation::

        >>> ds['rm'].random
        True

    Thus, with ``rm`` providing the measurement unit (subject for a
    within-subject design), the ``A*B`` model can be fitted as repeated measures
    design::

        >>> print(test.ANOVA('fltvar', 'A*B*rm', data=ds))
                    SS   df      MS   MS(denom)   df(denom)          F        p
        -----------------------------------------------------------------------
        A        28.69    1   28.69        1.21          19   23.67***   < .001
        B         0.04    1    0.04        1.15          19    0.03        .859
        A x B     1.16    1    1.16        1.01          19    1.15        .297
        -----------------------------------------------------------------------
        Total   114.74   79

    Nested effects are specified with parentheses. For example, if each
    condition of ``B`` was run with separate subjects (in other words, ``B`` is
    a between-subjects factor), ``subject`` is nested in ``B``, which is specified
    as ``subject(B)``::

        >>> print(test.ANOVA('fltvar', 'A * B * nrm(B)', data=ds))
                    SS   df      MS   MS(denom)   df(denom)          F        p
        -----------------------------------------------------------------------
        A        28.69    1   28.69        1.11          38   25.80***   < .001
        B         0.04    1    0.04        1.12          38    0.03        .856
        A x B     1.16    1    1.16        1.11          38    1.05        .313
        -----------------------------------------------------------------------
        Total   114.74   79

    Numerical variables can be coerced to categorial factors in the model::

        >>> ds = datasets.get_loftus_masson_1994()
        >>> print=(test.ANOVA('n_recalled', 'exposure.as_factor()*subject', data=ds))
                    SS       df   MS         F         p
        ---------------------------------------------------
        exposure     52.27    2   26.13   42.51***   < .001
        ---------------------------------------------------
        Total      1005.87   29
    """
    def __init__(
            self,
            y: VarArg,
            x: ModelArg,
            sub: IndexArg = None,
            data: Dataset = None,
            title: fmtxt.FMTextLike = None,
            caption: fmtxt.FMTextLike = None,
    ):
        # prepare kwargs
        sub, n = assub(sub, data, return_n=True)
        y, n = asvar(y, sub, data, n, return_n=True)
        x = asmodel(x, sub, data, n, require_names=True)
        assert_has_no_empty_cells(x)

        # save args
        self.y = y
        self.x = x
        self.title = title
        self.caption = caption
        self._log = []

        # decide which E(MS) model to use
        if x.df_error > 0:
            if hasrandom(x):
                raise x._incomplete_error("Mixed effects ANOVA")
        elif x.df_error < 0:
            raise ValueError("Model Overdetermined")

        # list of (name, SS, df, MS, F, p)
        names = []
        f_tests = []
        if len(x.effects) == 1:
            self._log.append("single factor model")
            lm1 = LM(y, x)
            f_tests.append(lm1)
            names.append(x.name)
            self.residuals = lm1.ss_res, lm1.df_res, lm1.ms_res
            self._is_mixed = False
        else:
            comparisons = IncrementalComparisons(x)
            is_mixed = self._is_mixed = comparisons.mixed
            self._log.append(f"{'FM'[is_mixed]}ixed effects model")

            # fit the models
            lms = {idx: LM(y, model) if model.df_error > 0 else None for model, idx in comparisons.relevant_models}

            # incremental F-tests
            for i_test, (i1, i0) in comparisons.comparisons.items():
                e_test = comparisons.x.effects[i_test]
                lm0 = lms[i0]
                lm1 = lms[i1]

                if is_mixed:
                    lm_ems = lms[comparisons.ems_idx[i_test]]
                    ms_e = lm_ems.ms_model
                    df_e = lm_ems.df_model
                else:
                    full_lm = lms[0]
                    ms_e = full_lm.ms_res
                    df_e = full_lm.df_res

                res = IncrementalFTest(lm1, lm0, ms_e, df_e, e_test.name)
                f_tests.append(res)
                names.append(e_test.name)

            if is_mixed:
                self.residuals = None
            else:
                full_lm = lms[0]
                self.residuals = full_lm.ss_res, full_lm.df_res, full_lm.ms_res

        self.effects = tuple(names)
        self.f_tests = tuple(f_tests)

    def __repr__(self):
        table = '\n'.join(F'  {line}' for line in str(self).splitlines())
        return f"<ANOVA: {self.y.name} ~ {self.x.name}\n{table}\n>"

    def __str__(self):
        return str(self.table())

    def _repr_html_(self):
        return fmtxt.html(self.table())

    def _asfmtext(self, **_):
        return self.table()

    def print_log(self):
        out = self._log[:]
        print('\n'.join(out))

    def table(self, title: fmtxt.FMTextLike = None, caption: fmtxt.FMTextLike = None):
        """ANOVA table

        Parameters
        ----------
        title : text
            Title for the table.
        caption : text
            Caption for the table.

        Returns
        -------
        table : eelbrain.fmtxt.Table
            ANOVA table.
        """
        if title is None:
            title = self.title
        if caption is None:
            caption = self.caption
        # table head
        table = fmtxt.Table('l' + 'r' * (5 + 2 * self._is_mixed), title=title, caption=caption)
        table.cells('', 'SS', 'df', 'MS')
        if self._is_mixed:
            table.cells(fmtxt.symbol('MS', 'denom'), fmtxt.symbol('df', 'denom'))
        table.cells('F', 'p')
        table.midrule()

        # table body
        for name, f_test in zip(self.effects, self.f_tests):
            table.cell(name)
            table.cell(fmtxt.stat(f_test.ss))
            table.cell(f_test.df)
            table.cell(fmtxt.stat(f_test.ms))
            if self._is_mixed:
                table.cell(fmtxt.stat(f_test.ms_e))
                table.cell(f_test.dfe)

            if f_test.f:
                stars = test.star(f_test.p)
                table.cell(fmtxt.stat(f_test.f, stars=stars))
                table.cell(fmtxt.p(f_test.p))
            else:
                table.endline()

        # residuals
        if self.x.df_error > 0:
            table.empty_row()
            table.cell("Residuals")
            ss, df, ms = self.residuals
            table.cell(fmtxt.stat(ss))
            table.cell(df)
            table.cell(fmtxt.stat(ms))
            table.endline()

        # total
        table.midrule()
        table.cell("Total")
        ss = np.sum((self.y.x - self.y.mean()) ** 2)
        table.cell(fmtxt.stat(ss))
        table.cell(len(self.y) - 1)
        return table
