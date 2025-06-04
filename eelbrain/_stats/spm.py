# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistical Parametric Mapping"""
from itertools import repeat
from typing import Dict, Literal, Sequence, Union

import numpy as np

from .. import _info, fmtxt
from .._data_obj import Dataset, Factor, Var, NDVar, Case, IndexArg, ModelArg, NDVarArg, Parametrization, PermutedParametrization, asmodel, asndvar, assub, combine
from .._exceptions import DimensionMismatchError
from .._utils import deprecate_ds_arg
from . import stats
from .glm import MPTestMapper
from .test import star
from .testnd import TTestOneSample, MultiEffectNDTest, NDPermutationDistribution, permute_order, run_permutation_me


class LMMapper(MPTestMapper):

    def __init__(
            self,
            parametrization: Parametrization,
    ):
        self.parametrization = parametrization
        self.permuted_parametrization = PermutedParametrization(parametrization, g=True)
        self._flat_t_buffer = None

    def preallocate(self, y_shape: Sequence[int]) -> np.ndarray:
        n_columns = len(self.parametrization.column_names)
        shape = (n_columns, *y_shape)
        buffer = np.empty(shape)
        self._flat_t_buffer = buffer.reshape((n_columns, -1))
        return buffer

    def map(
            self,
            y: np.ndarray,  # (n_cases, ...)
            perm: np.ndarray = None,  # (n_cases,) permutation index
    ) -> None:
        if perm is None:
            parametrization = self.parametrization
        else:
            parametrization = self.permuted_parametrization
            parametrization.permute(perm)
        stats.lm_t(y, parametrization, out_t=self._flat_t_buffer)


class LM(MultiEffectNDTest):
    """Fixed effects linear model

    Parameters
    ----------
    y
        Dependent variable.
    x
        Model to fit.
    sub
        Only use part of the data.
    data
        Optional Dataset providing data for y/model.
    coding
        Model parametrization (default is dummy coding). Vars are centered for
        effect coding (but not for dummy coding).
    subject
        Optional information used by :class:`LMGroup`; if subject is a column in
        ``ds`` it will be extracted automatically.
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
    force_permutation
        Conduct permutations regardless of whether there are any clusters.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    See Also
    --------
    LMGroup

    Examples
    --------
    See :ref:`exa-two-stage` example.

    Notes
    -----
    By default, this model generates a permutation distribution to correct for
    multiple comparisons. This is not needed for a two-stage model, where
    correction occurs at the group level. When fitting two-stage models, set
    ``samples=0`` to skip this and save time.

    This class stores a shallow copy of ``y.info`` (for predicting).
    """
    _state_specific = ('model', 'coding', '_coeffs_flat', '_se_flat', 'subject_variables', '_parametrization', '_y_info')
    # _statistic = 't'  # would be consistent with testnd but require turning LM.t() into an attr

    @deprecate_ds_arg
    def __init__(
            self,
            y: NDVarArg,
            x: ModelArg,
            sub: IndexArg = None,
            data: Dataset = None,
            coding: Literal['dummy', 'effect'] = 'dummy',
            subject: str = None,
            samples: int = 10000,
            pmin: float = None,
            tmin: float = None,
            tfce: Union[float, bool] = False,
            tstart: float = None,
            tstop: float = None,
            force_permutation: bool = False,
            **criteria,
    ):
        sub_arg = sub
        sub, n_cases = assub(sub, data, return_n=True)
        y, n_cases = asndvar(y, sub, data, n_cases, return_n=True)
        model = asmodel(x, sub, data, n_cases)
        parametrization = model._parametrize(coding)
        ß_maps, se_maps, t_maps = stats.lm_t(y.x, parametrization)
        # find variables to keep
        variables = {}
        if data is not None:
            for key, item in data.items():
                if isinstance(item, Factor):
                    if sub is not None:
                        item = item[sub]
                    if len(item.cells) == 1:
                        variables[key] = item.cells[0]
        # subject
        if subject is not None:
            if not isinstance(subject, str):
                raise TypeError(f"{subject=}: needs to be string or None")
            variables['subject'] = subject

        # Cluster-based tests
        n_effects = len(parametrization.column_names)
        n_threshold_params = sum((pmin is not None, tmin is not None, bool(tfce)))
        if n_threshold_params == 0 and not samples:
            cdists = None
        elif n_threshold_params > 1:
            raise ValueError("Only one of pmin, tmin and tfce can be specified")
        else:
            if pmin is not None:
                df = len(y) - model.df
                thresholds = tuple(repeat(stats.ttest_t(pmin, df), n_effects))
            elif tmin is not None:
                thresholds = tuple(repeat(abs(tmin), n_effects))
            else:
                thresholds = tuple(repeat(None, n_effects))

            cdists = [NDPermutationDistribution(y, samples, thresh, tfce, 0, 't', name, tstart, tstop, criteria, None, force_permutation) for name, thresh in zip(parametrization.column_names, thresholds)]

            # Find clusters in the actual data
            do_permutation = 0
            for cdist, t_map in zip(cdists, t_maps):
                cdist.add_original(t_map)
                do_permutation += cdist.do_permutation
            # Generate null distribution
            if do_permutation:
                iterator = permute_order(len(y), samples)
                run_permutation_me(LMMapper(parametrization), cdists, iterator)

        x_desc = x if isinstance(x, str) else model.name  # TODO: x.name should use * when appropriate
        MultiEffectNDTest.__init__(self, x_desc, parametrization.column_names, y, None, sub_arg, samples, tfce, pmin, cdists, tstart, tstop)
        self.coding = coding
        self._coeffs_flat = ß_maps.reshape((len(ß_maps), -1))
        self._se_flat = se_maps.reshape((len(se_maps), -1))
        self.model = model
        self._parametrization = parametrization
        self._y_info = y.info.copy()
        self.subject_variables = variables
        self._expand_state()

    def _expand_state(self):
        self.subject = self.subject_variables.get('subject', None)
        self._shape = tuple(map(len, self._dims))
        self.column_names = self._parametrization.column_names
        self.n_cases = self.model.df_total
        self.dims = self._dims
        MultiEffectNDTest._expand_state(self)

    def _name(self):
        y = f'{self.y} ' if self.y else ''
        return f"LM: {y}~ {self.model.name}, {self.subject}>"

    def __setstate__(self, state):
        # backwards compatibility
        if 'dims' in state:
            state['_dims'] = state.pop('dims')
            state['_coeffs_flat'] = state.pop('coeffs')
            state['_se_flat'] = state.pop('se')
            if 'subject' in state:
                state['subject_variables'] = {'subject': state.pop('subject')}
            if 'p' in state:
                state['_parametrization'] = state.pop('p')
            else:
                state['_parametrization'] = state['model']._parametrize(state['coding'])
        MultiEffectNDTest.__setstate__(self, state)

    def _coefficient(self, term):
        """Regression coefficient for a given term"""
        index = self._index(term)
        return self._coeffs_flat[index].reshape((1,) + self._shape)

    def _default_plot_obj(self):
        if self.samples:
            return [self.masked_parameter_map(e) for e in self.effects]
        else:
            return [self.t(term) for term in self.column_names]

    def _index(self, term: str) -> int:
        if term in self.column_names:
            return self.column_names.index(term)
        elif term in self._parametrization.terms:
            index = self._parametrization.terms[term]
            if index.stop - index.start > 1:
                raise NotImplementedError("Term has more than one column")
            return index.start
        else:
            raise KeyError(f"{term=}")

    def coefficient(self, term):
        ":class:`NDVar` with regression coefficient for a given term (or ``'intercept'``)"
        return NDVar(self._coefficient(term)[0], self.dims, term, self._y_info)

    def predict(
            self,
            values: Union[Sequence[float], Dict[str, float]],
            name: str = None,
    ) -> NDVar:
        """Predict ``y`` based on given values of ``x``

        Parameters
        ----------
        values
            Give as list of values, in the same order as ``x``, or as dictionary
            mapping predictor names to values (missing predictors are
            substituted with the mean from the original data).
        name
            Name the resulting :class:`NDVar`.
        """
        names = [e.name for e in self._parametrization.model.effects]
        if isinstance(values, dict):
            if unknown := set(values).difference(names):
                raise ValueError(f"{values}: Unknown terms ({', '.join(unknown)})")
            x_in = []
            for e in self._parametrization.model.effects:
                if isinstance(e, Var):
                    if e.name in values:
                        v_i = values[e.name]
                        if self._parametrization.method == 'effect':
                            v_i -= e.mean()
                    elif self._parametrization.method == 'effect':
                        v_i = 0
                    else:
                        v_i = e.mean()
                    x_in.append(v_i)
                else:
                    raise NotImplementedError("Predict for categorial models")
        else:
            x_in = list(values)
            if (l1 := len(x_in)) != (l2 := len(self._parametrization.x.shape(1) - 1)):
                raise ValueError(f"{values}: wrong length (got {l1}, need {l2})")
            if self._parametrization.method == 'effect':
                x_in = [v / e.mean() for e, v in zip(self._parametrization.model.effects, x_in)]
        # Add intercept
        x = np.append(1, x_in)
        y = x.dot(self._coeffs_flat)
        return NDVar(y.reshape(self._shape), self.dims, name, self._y_info)

    def t(self, term: str) -> NDVar:
        ":class:`NDVar` with t-values for a given term (or ``'intercept'``)."
        index = self._index(term)
        se = self._se_flat[index]
        flat_index = se == 0.
        any_flat = np.any(flat_index)
        if any_flat:
            se = se.copy()
            se[flat_index] = 1.
        t = self._coeffs_flat[index] / se
        if any_flat:
            t[np.logical_and(flat_index, t == 0)] = 0.
            t[np.logical_and(flat_index, t != 0)] *= np.inf
        info = _info.for_stat_map('t')
        info['term'] = term
        return NDVar(t.reshape(self._shape), self.dims, term, info)

    def _n_columns(self):
        return {term: s.stop - s.start for term, s in self._parametrization.terms.items()}


class LMGroup:
    """Group level analysis for linear model :class:`LM` objects

    Parameters
    ----------
    lms : sequence of LM
        A separate :class:`LM` object for each subject.

    Attributes
    ----------
    column_names : [str]
        Names of the linear model columns.
    column_keys : [str]
        Corresponding dataset keys (with invalid characters replaced).
    tests : None | {str: TTestRelated}
        Tests computed with :meth:`compute_column_ttests`.
    samples : None | int
        Number of samples used to compute tests in :attr:`tests`.

    See Also
    --------
    LM

    Examples
    --------
    See :ref:`exa-two-stage` example.
    """
    def __init__(self, lms):
        # check lms
        lm0 = lms[0]
        n_columns_by_term = lm0._n_columns()
        for lm in lms[1:]:
            if lm.dims != lm0.dims:
                raise DimensionMismatchError("LMs have incompatible dimensions")
            elif lm._n_columns() != n_columns_by_term:
                raise ValueError(f"Model for {lm0.subject} and {lm.subject} don't match")
            elif lm.coding != lm0.coding:
                raise ValueError("Models have incompatible coding")

        # make sure to have a unique subject label for each lm
        name_i = 0
        subjects = [lm.subject for lm in lms]
        str_names = tuple(filter(None, subjects))
        if len(set(str_names)) < len(str_names):
            raise ValueError(f"Duplicate subject names: {', '.join(map(repr, str_names))}")
        new_name = 'S000'
        for i in range(len(subjects)):
            if not subjects[i]:
                while new_name in subjects:
                    name_i += 1
                    new_name = 'S%03i' % name_i
                subjects[i] = new_name

        self._lms = lms
        self._subjects = tuple(subjects)
        self.tests = None
        self._init_secondary()

    def __setstate__(self, state):
        self._lms = state['lms']
        self._subjects = state['subjects']
        self.tests = state.get('tests')
        self._init_secondary()

    def _init_secondary(self):
        lm = self._lms[0]
        self.dims = lm.dims
        self.coding = lm.coding
        self.column_names = lm.column_names
        if self.tests is None:
            self.samples = None
        else:
            self.samples = self.tests[self.column_names[0]].samples
        # subject variables
        self.subject_variables = Dataset()
        self.subject_variables['subject'] = Factor(self._subjects, random=True)
        keys = set().union(*(lm.subject_variables for lm in self._lms))
        for key in keys:
            values = [lm.subject_variables.get(key, '') for lm in self._lms]
            self.subject_variables[key] = Factor(values)
        self.column_keys = [Dataset.as_key(name) for name in self.column_names]

    def __getstate__(self):
        return {'lms': self._lms, 'tests': self.tests, 'subjects': self._subjects}

    def __repr__(self):
        lm = self._lms[0]
        return f"<LMGroup {lm.y!r}, {lm.x!r}, n={len(self._lms)}>"

    def coefficients(self, term):
        "Coefficients for one term as :class:`NDVar`"
        x = np.concatenate([lm._coefficient(term) for lm in self._lms])
        return NDVar(x, (Case,) + self.dims, name=term)

    def coefficients_dataset(self, terms=None, long=False):
        """Regression coefficients in a :class:`Dataset`

        By default, each regression coefficient is assigned as separate column.
        With ``long=True``, a long form table is produced, in which the
        coefficients for different terms are stacked vertically and the ``term``
        :class:`Factor` specifies which term the coefficients correspond to.

        Parameters
        ----------
        terms : str | sequence of str
            Terms for which to retrieve coefficients (default is all terms).
        long : bool
            Produce a table in long form.

        Returns
        -------
        ds : Dataset
            If the dataset is in the wide form (default), each model coefficient
            is assigned as :class:`NDVar` with a key corresponding to the
            ``.column_keys`` attribute.
            If the dataset is in the long form (with ``long=True``), it has the
            following entries:

             - ``coeff``: the coefficients in an :class:`NDVar`
             - ``term``: a :class:`Factor` with the name of the term

            In addition, all available subject-variables are assigned.
        """
        if isinstance(terms, str):
            terms = (terms,)
        elif terms is None:
            terms = self.column_names

        if long:
            ds = Dataset({
                'coeff': combine([self.coefficients(term) for term in terms]),
                'term': Factor(terms, repeat=len(self._lms)),
                **self.subject_variables.tile(len(terms)),
            })
        else:
            ds = Dataset({Dataset.as_key(name): self.coefficients(name) for name in terms})
            ds.update(self.subject_variables)
        return ds

    def column_ttest(self, term, return_data=False, popmean=0, *args, **kwargs):
        """One-sample t-test on a single model column

        Parameters
        ----------
        term : str
            Name of the term to test.
        return_data : bool
            Return the individual subjects' coefficients along with test
            results.
        popmean : scalar
            Value to compare y against (default is 0).
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use a t-value equivalent to an
            uncorrected p-value.
        tmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.

        Returns
        -------
        result : TTestOneSample
            T-test result.
        data : Dataset (only with ``return_data=True``)
            Dataset with subjects' coefficients.

        Notes
        -----
        Performs a one-sample t-test on coefficient estimates from all subjects
        to test the hypothesis that the coefficient is different from popmean
        in the population.
        """
        coeff = self.coefficients(term)
        res = TTestOneSample(coeff, popmean, None, None, None, *args, **kwargs)
        if return_data:
            ds = Dataset({'coeff': coeff})
            ds.update(self.subject_variables)
            ds['n'] = Var([lm.n_cases for lm in self._lms])
            return res, ds
        else:
            return res

    def design(self, subject=None):
        "Table with the design matrix"
        if subject is None:
            lm = self._lms[0]
            subject = self._subjects[0]
        else:
            for lm, lm_subject in zip(self._lms, self._subjects):
                if lm_subject == subject:
                    break
            else:
                raise ValueError(f"subject={subject!r}")

        table = lm.model.as_table(lm.coding)
        table.caption(f"Design matrix for {subject}")
        return table

    def compute_column_ttests(self, *args, **kwargs):
        """Compute all tests and store them in :attr:`self.tests`

        Parameters like :meth:`.column_ttest`, starting with ``popmean``.
        """
        self.tests = {}
        for term in self.column_names:
            self.tests[term] = self.column_ttest(term, False, *args, **kwargs)
        self.samples = self.tests[self.column_names[0]].samples

    def info_list(self):
        info = fmtxt.List("LMGroup info")
        for effect in self.column_names:
            res = self.tests[effect]
            info.add_sublist(effect, [res.info_list()])
        return info

    def table(self, title=None, caption=None):
        """Table listing all terms and corresponding smallest p-values

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
        if not self.tests:
            raise RuntimeError("Need to precompute tests with .compute_column_ttests()")
        table = fmtxt.Table('lrrl', title=title, caption=caption)
        table.cells('Term', 't_max', 'p', 'sig')
        table.midrule()
        for term, res in self.tests.items():
            table.cell(term)
            table.cell(fmtxt.stat(res.t.extrema()))
            pmin = res.p.min()
            table.cell(fmtxt.p(pmin))
            table.cell(star(pmin))
        return table


# for backwards compatibility
RandomLM = LMGroup
