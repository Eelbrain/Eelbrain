# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import annotations

from dataclasses import dataclass
from inspect import getfullargspec
import re
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

import mne

from .. import testnd
from .. import test
from .._data_obj import CellArg, Dataset, NDVar, Var, combine
from .._exceptions import ConfigurationError
from .configuration import Configuration
from .variable_def import Variables, VarDef, GroupVar

if TYPE_CHECKING:
    from .derivative_cache import Request


__test__ = False
TAIL_REPR = {0: '=', 1: '>', -1: '<'}


def validate_tests(test_dict):
    "Interpret dict with test definitions"
    for key, config in test_dict.items():
        if not isinstance(config, Test):
            raise TypeError(f"Invalid object for test definition {key}: {config!r}")


def guess_y(ds: Dataset, default: str = None) -> str:
    "Given a dataset, guess the dependent variable"
    for y in ('srcm', 'src', 'meg', 'eeg'):
        if y in ds:
            return y
    if default is not None:
        return default
    raise RuntimeError(f"Could not find data in {ds}")


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
            model: str = None,  # within-subject model; None for single-trial analysis
            vars: dict[str, VarDef] | None = None,  # dynamic variables
            cat: tuple[CellArg, ...] = None,  # cells in model to load
            depend_on: Collection[str] = (),  # non-model variables
    ):
        self.desc = desc
        if model is None:
            self._test_vars = []
            self.model = None
        else:
            self._test_vars = [v for v in map(str.strip, model.split('%')) if v]
            self.model = '%'.join(self._test_vars)
        self.cat = cat
        try:
            self.vars = Variables(vars)
        except Exception as error:
            raise ConfigurationError(f"vars={vars} ({error})")
        self._test_vars.extend(depend_on)

    def _find_test_vars(self):
        "Find variables and groups used in a test definition"
        vs = set(self._test_vars)
        groups = set()
        for name, variable in self.vars.vars.items():
            if name in vs:
                vs.remove(name)
                vs.update(variable._input_vars())
                if isinstance(variable, GroupVar):
                    groups.update(variable.groups)
        return vs, groups

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
    tail : int
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
    model : str
        The model which defines the cells that are used in the test. Usually
        ``"group"``.
    c1 : str | tuple
        The experimental group. Should be a group name.
    c0 : str | tuple
        The control group, defined like ``c1``.
    tail : int
        Tailedness of the test. ``0`` for two-tailed (default), ``1`` for upper tail
        and ``-1`` for lower tail.

    See Also
    --------
    Pipeline.tests

    Examples
    --------
    Sample test definitions, assuming that the experiment has two groups called
    ``'younger'`` and ``'older'``::

        variables = {
            'age': GroupVar(['younger', 'older']),
        }
        tests = {
            'old=young': TTestIndependent('group', 'older', 'younger'),
            'old>young': TTestIndependent('group', 'older', 'younger', tail=1),
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
    model : str
        The model which defines the cells that are used in the test. It is
        specified in the ``"x % y"`` format (like interaction definitions) where
        ``x`` and ``y`` are variables in the experiment's events.
    c1 : str | tuple
        The experimental condition. If the ``model`` is a single factor the
        condition is a :class:`str` specifying a value on that factor. If
        ``model`` is composed of several factors the cell is defined as a
        :class:`tuple` of :class:`str`, one value on each of the factors.
    c0 : str | tuple
        The control condition, defined like ``c1``.
    tail : int
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
    model : str
        The model which defines the cells that are used in the test. It is
        specified in the ``"x % y"`` format (like interaction definitions) where
        ``x`` and ``y`` are variables in the experiment's events.
    contrast : str
        Contrast specification using cells form the specified model (see
        :class:`eelbrain.testnd.TContrastRelated`)).
    tail : int
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
    x : str
        ANOVA model specification, including ``subject`` for participant random
        effect (e.g., ``"x * y * subject"``; see :class:`eelbrain.test.ANOVA`).
    model : str
        Model for grouping trials before averaging (by default all fixed effects
        in ``x``). Should be specified in the ``"x % y"`` format (like
        interaction definitions) where ``x`` and ``y`` are variables in the
        experiment's events.
    vars : tuple | dict
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


class TestDims:
    """Data shape for test

    Parameters
    ----------
    string : str
        String describing data.
    time : bool
        Whether the base data contains a time axis.
    morph : bool
        If loading source space data, whether the data is morphed to the common
        brain.
    """
    RE = re.compile(r"^(source|sensor|meg|eeg)(?:\.(mean|rms))?$")
    source = None
    sensor = None

    def __init__(self, string, time=True, morph=False):
        self.time = bool(time)
        self.morph = bool(morph)
        m = self.RE.match(string)
        if m is None:
            raise ValueError(f"data={string!r}: invalid test dimension description")
        dim, aggregate = m.groups()
        if dim in ('meg', 'mag'):
            self._to_ndvar = ('mag',)
            self.y_name = 'meg'  # see .load_epochs()
            dim = 'sensor'
        elif dim in ('eeg', 'planar1', 'planar2'):
            self._to_ndvar = (dim,)
            self.y_name = dim
            dim = 'sensor'
        elif dim == 'sensor':
            self._to_ndvar = None
            self.y_name = 'meg'
        elif dim == 'source':
            self._to_ndvar = None
            self.y_name = 'srcm' if self.morph else 'src'
        else:
            raise RuntimeError(f"{string=} ({dim=})")
        setattr(self, dim, aggregate or True)
        if sum(map(bool, (self.source, self.sensor))) != 1:
            raise ValueError(f"data={string!r}: invalid test dimension description")
        self.string = string

        dims = []
        if self.source is True:
            dims.append('source')
        elif self.sensor is True:
            dims.append('sensor')
        if self.time is True:
            dims.append('time')
        self.dims = tuple(dims)

        # whether parc is used from subjects or from common-brain
        if self.source is True:
            self.parc_level = 'common'
        elif self.source:
            self.parc_level = 'individual'
        else:
            self.parc_level = None

    @classmethod
    def coerce(cls, obj, time=True, morph=False):
        if isinstance(obj, cls):
            if obj.time == time and obj.morph == morph:
                return obj
            else:
                return cls(obj.string, time, morph)
        else:
            return cls(obj, time, morph)

    def __repr__(self):
        return f"TestDims({self.string!r})"

    def __eq__(self, other):
        if not isinstance(other, TestDims):
            return False
        return self.string == other.string and self.time == other.time

    def _testnd_parc(self, disconnect_labels: bool) -> str | None:
        if self.source is True:
            return 'source' if disconnect_labels else None
        if disconnect_labels:
            raise TypeError(f"{disconnect_labels=}: invalid for data={self.string!r}")
        return None

    def data_to_ndvar(self, info: mne.Info) -> list[str]:
        assert self.sensor
        if self._to_ndvar is None:
            return info.get_channel_types(unique=True, only_data_chs=True)
        else:
            return self._to_ndvar


@dataclass(frozen=True)
class ResolvedTestNDSpec:
    """Resolved request-local plan for `testnd` execution.

    This combines a :class:`TestDims` semantic data description with the current
    request-local ``testnd`` kwargs.
    """

    data: TestDims
    kwargs: dict[str, Any]

    @classmethod
    def from_request(
            cls,
            ctx: Request,
            data: TestDims,
    ) -> ResolvedTestNDSpec:
        pmin = ctx.options['pmin']
        kwargs = {
            'samples': ctx.options['samples'],
            'tstart': ctx.options['tstart'],
            'tstop': ctx.options['tstop'],
            'parc': data._testnd_parc(ctx.options.get('disconnect_labels', False)),
        }
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


class ROITestResult:
    """Test results for temporal tests in one or more ROIs

    Attributes
    ----------
    subjects : tuple of str
        Subjects included in the test.
    samples : int
        ``samples`` parameter used for permutation tests.
    res : {str: NDTest} dict
        Test result for each ROI.
    n_trials_ds : Dataset
        Dataset describing how many trials were used in each condition per
        subject.
    """

    def __init__(self, subjects, samples, n_trials_ds, merged_dist, res):
        self.subjects = subjects
        self.samples = samples
        self.n_trials_ds = n_trials_ds
        self.merged_dist = merged_dist
        self.res = res

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in getfullargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)
