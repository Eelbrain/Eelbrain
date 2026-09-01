# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistical test definitions (:class:`Configuration` classes)."""
from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from ... import testnd
from ... import test
from ..._data_obj import CellArg, Dataset, NDVar, Var, combine
from ..._exceptions import ConfigurationError
from ..._utils.parse import find_variables
from ..configuration import Configuration
from ..data import DataSpec
from ..variable_def import Variables, VarDef, GroupVar

if TYPE_CHECKING:
    from ..derivative_cache import Request


__test__ = False
TAIL_REPR = {0: '=', 1: '>', -1: '<'}


def validate_tests(test_dict):
    "Interpret dict with test definitions"
    for key, config in test_dict.items():
        if not isinstance(config, Test):
            raise TypeError(f"Invalid object for test definition {key}: {config!r}")


def tail_arg(tail):
    try:
        if tail == 0:
            return 0
        elif tail > 0:
            return 1
        else:
            return -1
    except Exception:
        raise TypeError(f"{tail=}; needs to be 0, -1 or 1")


class Test(Configuration):
    """Base class for test definitions."""
    kind = None
    DICT_ATTRS = ('kind', 'model', 'vars')

    def __init__(
            self,
            desc: str,
            model: str | None = None,  # within-subject model; None for single-trial analysis
            vars: dict[str, VarDef] | None = None,  # dynamic variables
            cat: tuple[CellArg, ...] | None = None,  # cells in model to load
            depend_on: Collection[str] = (),  # non-model variables
    ):
        self.desc: str = desc
        self._test_vars: set[str] = set()
        if model is None:
            self.model = None
        else:
            model_vars = [v for v in map(str.strip, model.split('%')) if v]
            self.model = '%'.join(model_vars)
            self._test_vars.update(model_vars)
        self.cat = cat
        try:
            self.vars = Variables(vars)
        except Exception as error:
            raise ConfigurationError(f"{vars=} ({error})")
        self._test_vars.update(depend_on)

    def _resolve_vars(
            self,
            data: Dataset,
            groups: dict[str, tuple[str, ...]] = None,
    ) -> dict[str, Any]:
        """Add the variables this test reads to ``data``, and return their values

        The same call serves both consumers of a test's variables: the node that
        builds the data adds them, and the node that validates a cached result
        resolves them against the event *shell* and records the returned values.
        Recording values rather than the definitions behind them keeps a label
        change that does not reach the retained events from invalidating the
        result; passing :attr:`_test_vars` as ``names`` makes the coverage checked
        rather than assumed (see
        :meth:`~eelbrain._experiment.variable_def.Variables.resolve`).

        Parameters
        ----------
        data
            The events, or an event shell describing them.
        groups
            Members of each group, from :attr:`Pipeline.groups`. ``None`` for data
            from a single subject, where across-subject variables are absent.
        """
        return self.vars.resolve(data, groups, names=self._test_vars)

    def _as_dict_without_vars(self) -> dict[str, Any]:
        """The definition without :attr:`vars`

        A node that fingerprints the event shell for a test does not need to
        record the variable definitions producing the event labels (see
        :meth:`~eelbrain._experiment.variable_def.Variables.resolve`).
        """
        return {key: value for key, value in self._as_dict().items() if key != 'vars'}

    def _make(self, y, ds, force_permutation, kwargs):
        raise NotImplementedError(f"For {self.__class__.__name__}")

    def _make_vec(self, y, ds, force_permutation, kwargs):
        raise NotImplementedError(f"Vector test for {self.__class__.__name__}")

    def _make_uv(self, y, ds):
        raise NotImplementedError(f"UV sets for {self.__class__.__name__}")


class TTestOneSample(Test):
    """One-sample t-test

    Parameters
    ----------
    tail
        Tailedness of the test. ``0`` for two-tailed (default), ``1`` for upper tail
        and ``-1`` for lower tail.

    See Also
    --------
    Pipeline.tests
    """
    kind = 'ttest_1samp'
    DICT_ATTRS = Test.DICT_ATTRS + ('tail',)

    def __init__(self, tail: int = 0):
        tail = tail_arg(tail)
        desc = f"{TAIL_REPR[tail]} 0"
        Test.__init__(self, desc, '')
        self.tail = tail

    def _make(self, y, ds, force_permutation, kwargs):
        return testnd.TTestOneSample(y, match='subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)

    def _make_vec(self, y, ds, force_permutation, kwargs):
        if self.tail:
            raise ValueError("Vector-tests cannot be tailed")
        return testnd.Vector(y, match='subject', data=ds, **kwargs)

    def _make_uv(self, y, ds):
        return test.TTestOneSample(y, match='subject', data=ds, tail=self.tail)


class TTestIndependent(Test):
    """Independent measures t-test (comparing groups of subjects)

    Parameters
    ----------
    model
        The model which defines the cells that are used in the test.
        Can be ``"group"`` to compare two groups defined on the pipeline.
    c1
        The experimental group. Should be a group name.
    c0
        The control group, defined like ``c1``.
    tail
        Tailedness of the test. ``0`` for two-tailed (default), ``1`` for upper tail
        and ``-1`` for lower tail.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    Sample test definitions, assuming that the experiment has two groups called
    ``'younger'`` and ``'older'``::

        groups = {
            'younger': Group(['S001', 'S002']),
            'older': Group(['S011', 'S012']),
        }
        tests = {
            'older=younger': TTestIndependent('group', 'older', 'younger'),
            'older>younger': TTestIndependent('group', 'older', 'younger', tail=1),
        }
    """
    kind = 'ttest_ind'
    DICT_ATTRS = Test.DICT_ATTRS + ('c1', 'c0', 'tail')

    def __init__(self, model: str, c1: CellArg, c0: CellArg, tail: int = 0):
        if model == 'group':
            vars_ = {'group': GroupVar((c1, c0))}
        elif '%' in model:
            # assume 'group' is between, others are within
            raise NotImplementedError(f"{model=}: model with % for {self.__class__.__name__}")
        else:
            vars_ = None
        tail = tail_arg(tail)
        desc = f'{c1} {TAIL_REPR[tail]} {c0}'
        Test.__init__(self, desc, '', vars=vars_, depend_on=[model])
        self.between_model = model
        self.c1 = c1
        self.c0 = c0
        self.tail = tail

    def _as_dict(self):
        return {**Test._as_dict(self), 'model': self.between_model}

    def _make(self, y, ds, force_permutation, kwargs):
        return testnd.TTestIndependent(y, self.between_model, self.c1, self.c0, 'subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)

    def _make_uv(self, y, ds):
        return test.TTestIndependent(y, self.between_model, self.c1, self.c0, 'subject', data=ds, tail=self.tail)


class TTestRelated(Test):
    """Related measures t-test

    Parameters
    ----------
    model
        The model which defines the cells that are used in the test. It is
        specified in the ``"x % y"`` format (like interaction definitions) where
        ``x`` and ``y`` are variables in the experiment's events.
    c1
        The experimental condition. If the ``model`` is a single factor the
        condition is a :class:`str` specifying a value on that factor. If
        ``model`` is composed of several factors the cell is defined as a
        :class:`tuple` of :class:`str`, one value on each of the factors.
    c0
        The control condition, defined like ``c1``.
    tail
        Tailedness of the test. ``0`` for two-tailed (default), ``1`` for upper tail
        and ``-1`` for lower tail.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    Sample test definitions::

        tests = {
            'surprising=expected': TTestRelated('surprise', 'surprising', 'expected'),
        }

    Notes
    -----
    For a t-test between two epochs, use an
    :class:`~eelbrain.pipeline.EpochCollection` epoch and ``model='epoch'``.
    """
    kind = 'ttest_rel'
    DICT_ATTRS = Test.DICT_ATTRS + ('c1', 'c0', 'tail')

    def __init__(self, model: str, c1: CellArg, c0: CellArg, tail: int = 0):
        tail = tail_arg(tail)
        desc = f'{c1} {TAIL_REPR[tail]} {c0}'
        Test.__init__(self, desc, model, cat=(c1, c0))
        self.c1 = c1
        self.c0 = c0
        self.tail = tail

    def _make(self, y, ds, force_permutation, kwargs):
        return testnd.TTestRelated(y, self.model, self.c1, self.c0, 'subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)

    def _make_vec(self, y, ds, force_permutation, kwargs):
        if self.tail:
            raise ValueError("Vector-tests cannot be tailed")
        return testnd.VectorDifferenceRelated(y, self.model, self.c1, self.c0, 'subject', data=ds, force_permutation=force_permutation, **kwargs)

    def _make_uv(self, y, ds):
        return test.TTestRelated(y, self.model, self.c1, self.c0, 'subject', data=ds, tail=self.tail)


class TContrastRelated(Test):
    """Contrasts of T-maps (see :class:`eelbrain.testnd.TContrastRelated`)

    Parameters
    ----------
    model
        The model which defines the cells that are used in the test. It is
        specified in the ``"x % y"`` format (like interaction definitions) where
        ``x`` and ``y`` are variables in the experiment's events.
    contrast
        Contrast specification using cells form the specified model (see
        :class:`eelbrain.testnd.TContrastRelated`)).
    tail
        Tailedness of the test. ``0`` for two-tailed (default), ``1`` for upper tail
        and ``-1`` for lower tail.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    Sample test definitions::

        tests = {
            'a_b_intersection': TContrastRelated{'abc', 'min(a > c, b > c)', tail=1),
        }

    """
    kind = 't_contrast_rel'
    DICT_ATTRS = Test.DICT_ATTRS + ('contrast', 'tail')

    def __init__(self, model: str, contrast: str, tail: int = 0):
        tail = tail_arg(tail)
        Test.__init__(self, contrast, model)
        self.contrast = contrast
        self.tail = tail

    def _make(self, y, ds, force_permutation, kwargs):
        return testnd.TContrastRelated(y, self.model, self.contrast, 'subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)


class ANOVA(Test):
    """ANOVA test

    Parameters
    ----------
    x
        ANOVA model specification, including ``subject`` for participant random
        effect (e.g., ``"x * y * subject"``; see :class:`eelbrain.test.ANOVA`).
    model
        Model for grouping trials before averaging (by default all fixed effects
        in ``x``). Should be specified in the ``"x % y"`` format (like
        interaction definitions) where ``x`` and ``y`` are variables in the
        experiment's events.
    vars
        Variables to add dynamically.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    Sample test definitions::

        tests = {
            'one_way': ANOVA('word_type * subject'),
            'two_way': ANOVA('word_type * meaning * subject'),
        }

    """
    kind = 'anova'
    DICT_ATTRS = Test.DICT_ATTRS + ('x',)

    def __init__(self, x: str, model: str = None, vars: dict = None):
        x_items = [item.strip() for item in x.split('*')]
        items = sorted(x_items)
        nested_in = (re.match(r'^subject\((\w+)\)?$', item) for item in items)
        between_items = []
        for match in filter(None, nested_in):
            between_item = match.group(1)
            items.remove(match.string)
            items.remove(between_item)
            between_items.append(between_item)
        if model is None:
            if 'subject' in items:
                items.remove('subject')
            elif not between_items:
                raise ConfigurationError(f"{x=} without model: for mixed ANOVA, 'subject' needs to be in x; for between-subject ANOVA, model needs to be set explicitly")
            model = '%'.join(items)
        else:
            model_items = list(filter(None, (item.strip() for item in model.split('%'))))
            between_items.extend(set(items).difference(model_items))
        desc = ' * '.join(x_items)
        Test.__init__(self, desc, model, vars=vars, depend_on=between_items)
        self.x = '*'.join(x_items)

    def _make(self, y, ds, force_permutation, kwargs):
        return testnd.ANOVA(y, self.x, data=ds, force_permutation=force_permutation, **kwargs)

    def _make_uv(self, y, ds):
        return test.ANOVA(y, self.x, data=ds)


class TwoStageTest(Test):
    """Two-stage test: T-test of regression coefficients

    Stage 1: fit a regression model to the data for each subject.
    Stage 2: test coefficients from stage 1 against 0 across subjects.

    Parameters
    ----------
    stage_1
        Stage 1 model specification. Coding for categorial predictors uses 0/1 dummy
        coding.
    vars
        Add new variables for the stage 1 model. This is useful for specifying
        coding schemes based on categorial variables.
        Each entry specifies a variable with the following schema:
        ``{name: definition}``. ``definition`` can be either a string that is
        evaluated in the events-:class:`Dataset`, or a
        ``(source_name, {value: code})``-tuple (see example below).
        ``source_name`` can also be an interaction, in which case cells are joined
        with spaces (``"f1_cell f2_cell"``).
    model
        This parameter can be supplied to perform stage 1 tests on condition
        averages. If ``model`` is not specified, the stage1 model is fit on single
        trial data.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    The first example assumes 2 categorical variables present in events,
    'a' with values 'a1' and 'a2', and 'b' with values 'b1' and 'b2'. These are
    recoded into 0/1 codes::

        TwoStageTest(
            "a_num + b_num + a_num * b_num + index + a_num * index",
            vars={
                'a_num': ('a', {'a1': 0, 'a2': 1}),
                'b_num': ('b', {'b1': 0, 'b2': 1}),
            }),

    The second test definition uses the "index" variable which is always present
    and specifies the chronological index of the events as an integer count.
    This variable can thus be used to test for a linear change over time. Due
    to the numeric nature of these variables interactions can be computed by
    multiplication::

        TwoStageTest("a_num + index + a_num * index",
                     vars={'a_num': ('a', {'a1': 0, 'a2': 1})

    Numerical variables can also defined using data-object methods (e.g.
    :meth:`Factor.label_length`) or from interactions::

        TwoStageTest('wordlength', vars={'wordlength': 'word.label_length()'})
        TwoStageTest("ab", vars={'ab': ('a%b', {'a1 b1': 0, 'a1 b2': 1, 'a2 b1': 1, 'a2 b2': 2})})
    """
    kind = 'two-stage'
    DICT_ATTRS = Test.DICT_ATTRS + ('stage_1',)

    def __init__(self, stage_1: str, vars: dict = None, model: str = None):
        Test.__init__(self, stage_1, model, vars=vars, depend_on=find_variables(stage_1))
        self.stage_1 = stage_1

    def make_stage_1(self, y, data, subject, sub=None):
        """Assumes that model has already been applied"""
        return testnd.LM(y, self.stage_1, sub=sub, data=data, samples=0, subject=subject)

    @staticmethod
    def make_stage_2(lms, kwargs):
        lm = testnd.LMGroup(lms)
        lm.compute_column_ttests(**kwargs)
        return lm

    def make(self, y, ds, force_permutation, kwargs):
        lms = [self.make_stage_1(y, ds, subject, f"subject=={subject!r}") for subject in ds['subject'].cells]
        return self.make_stage_2(lms, kwargs)


@dataclass(frozen=True)
class ResolvedTestNDSpec:
    """Resolved request-local plan for `testnd` execution.

    This combines a :class:`DataSpec` semantic data description with the current
    request-local ``testnd`` kwargs.
    """

    data: DataSpec
    kwargs: dict[str, Any]

    @classmethod
    def from_request(
            cls,
            ctx: Request,
            time: bool = True,
    ) -> ResolvedTestNDSpec:
        data = ctx.options['data']
        pmin = ctx.options['pmin']
        kwargs = {
            'samples': ctx.options['samples'],
            'parc': data._testnd_parc(ctx.options.get('disconnect_labels', False)),
        }
        if time:
            kwargs['tstart'] = ctx.options['tstart']
            kwargs['tstop'] = ctx.options['tstop']
        if pmin == 'tfce':
            kwargs['tfce'] = True
        elif pmin is not None:
            kwargs['pmin'] = pmin
        return cls(data, kwargs)

    def make_result(
            self,
            node: Any,
            y: str | Var | NDVar | list[NDVar],
            ds: Dataset,
            test: Test,
            force_permutation: bool = False,
    ) -> Any:
        test_obj = test if isinstance(test, Test) else node.tests[test]
        if isinstance(y, str):
            y = ds.eval(y)
        if isinstance(y, Var):
            return test_obj._make_uv(y, ds)
        if isinstance(y, list):
            dim = 'sensor' if y[0].has_dim('sensor') else 'source'
            return test_obj._make_uv(combine([getattr(yi, 'mean')(dim) for yi in y]), ds)
        if isinstance(y, NDVar) and y.has_dim('space'):
            return test_obj._make_vec(y, ds, force_permutation, self.kwargs)
        return test_obj._make(y, ds, force_permutation, self.kwargs)
