# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from inspect import getfullargspec
import re
from typing import Collection, List, Tuple, Union

import mne

from .. import testnd
from .. import test
from .._data_obj import CellArg
from .._exceptions import DefinitionError
from .._utils.parse import find_variables
from .definitions import Definition
from .variable_def import Variables, GroupVar


__test__ = False
TAIL_REPR = {0: '=', 1: '>', -1: '<'}


def assemble_tests(test_dict):
    "Interpret dict with test definitions"
    out = {}
    for key, params in test_dict.items():
        if isinstance(params, Test):
            out[key] = params
            continue
        elif not isinstance(params, dict):
            raise TypeError(f"Invalid object for test definition {key}: {params!r}")
        params = params.copy()
        if 'stage 1' in params:
            params['stage_1'] = params.pop('stage 1')
        kind = params.pop('kind')
        if kind in TEST_CLASSES:
            out[key] = TEST_CLASSES[kind](**params)
        else:
            raise DefinitionError(f"Unknown test kind in test definition {key}: {kind}")
    return out


def tail_arg(tail):
    try:
        if tail == 0:
            return 0
        elif tail > 0:
            return 1
        else:
            return -1
    except Exception:
        raise TypeError("tail=%r; needs to be 0, -1 or 1" % (tail,))


class Test(Definition):
    "Baseclass for any test"
    kind = None
    DICT_ATTRS = ('kind', 'model', 'vars')

    def __init__(
            self,
            desc: str,
            model: str = None,  # within-subject model; None for single-trial analysis
            vars: Union[str, tuple, list, dict] = None,  # dynamic variables
            cat: Tuple[CellArg, ...] = None,  # cells in model to load
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
            raise DefinitionError(f"vars={vars} ({error})")
        self._test_vars.extend(depend_on)

    def _find_test_vars(self):
        "Find variables and groups used in a test definition"
        vs = set(self._test_vars)
        groups = set()
        for name, variable in self.vars.vars.items():
            if name in vs:
                vs.remove(name)
                vs.update(variable.input_vars())
                if isinstance(variable, GroupVar):
                    groups.update(variable.groups)
        return vs, groups

    def make(self, y, ds, force_permutation, kwargs):
        raise NotImplementedError(f"For {self.__class__.__name__}")

    def make_vec(self, y, ds, force_permutation, kwargs):
        raise NotImplementedError(f"Vector test for {self.__class__.__name__}")

    def make_uv(self, y, ds):
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
    MneExperiment.tests
    """
    kind = 'ttest_1samp'
    DICT_ATTRS = Test.DICT_ATTRS + ('tail',)

    def __init__(self, tail: int = 0):
        tail = tail_arg(tail)
        desc = f"{TAIL_REPR[tail]} 0"
        Test.__init__(self, desc, '')
        self.tail = tail

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.TTestOneSample(y, match='subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)

    def make_vec(self, y, ds, force_permutation, kwargs):
        if self.tail:
            raise ValueError("Vector-tests cannot be tailed")
        return testnd.Vector(y, match='subject', data=ds, **kwargs)

    def make_uv(self, y, ds):
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
    MneExperiment.tests

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
            raise NotImplementedError(f"model={model!r}: model with % for {self.__class__.__name__}")
        else:
            vars_ = None
        tail = tail_arg(tail)
        desc = '%s %s %s' % (c1, TAIL_REPR[tail], c0)
        Test.__init__(self, desc, '', vars=vars_, depend_on=[model])
        self.between_model = model
        self.c1 = c1
        self.c0 = c0
        self.tail = tail

    def _as_dict(self):
        return {**Test._as_dict(self), 'model': self.between_model}

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.TTestIndependent(y, self.between_model, self.c1, self.c0, 'subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)

    def make_uv(self, y, ds):
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
    MneExperiment.tests

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

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.TTestRelated(y, self.model, self.c1, self.c0, 'subject', data=ds, tail=self.tail, force_permutation=force_permutation, **kwargs)

    def make_vec(self, y, ds, force_permutation, kwargs):
        if self.tail:
            raise ValueError("Vector-tests cannot be tailed")
        return testnd.VectorDifferenceRelated(y, self.model, self.c1, self.c0, 'subject', data=ds, force_permutation=force_permutation, **kwargs)

    def make_uv(self, y, ds):
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
    MneExperiment.tests

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

    def make(self, y, ds, force_permutation, kwargs):
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
    MneExperiment.tests

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
                raise DefinitionError(f"{x=} without model: for mixed ANOVA, 'subject' needs to be in x; for between-subject ANOVA, model needs to be set explicitly")
            model = '%'.join(items)
        else:
            model_items = list(filter(None, (item.strip() for item in model.split('%'))))
            between_items.extend(set(items).difference(model_items))
        desc = ' * '.join(x_items)
        Test.__init__(self, desc, model, vars=vars, depend_on=between_items)
        self.x = '*'.join(x_items)

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.ANOVA(y, self.x, data=ds, force_permutation=force_permutation, **kwargs)

    def make_uv(self, y, ds):
        return test.ANOVA(y, self.x, data=ds)


class TwoStageTest(Test):
    """Two-stage test: T-test of regression coefficients

    Stage 1: fit a regression model to the data for each subject.
    Stage 2: test coefficients from stage 1 against 0 across subjects.

    Parameters
    ----------
    stage_1 : str
        Stage 1 model specification. Coding for categorial predictors uses 0/1 dummy
        coding.
    vars : dict
        Add new variables for the stage 1 model. This is useful for specifying
        coding schemes based on categorial variables.
        Each entry specifies a variable with the following schema:
        ``{name: definition}``. ``definition`` can be either a string that is
        evaluated in the events-:class:`Dataset`, or a
        ``(source_name, {value: code})``-tuple (see example below).
        ``source_name`` can also be an interaction, in which case cells are joined
        with spaces (``"f1_cell f2_cell"``).
    model : str
        This parameter can be supplied to perform stage 1 tests on condition
        averages. If ``model`` is not specified, the stage1 model is fit on single
        trial data.

    See Also
    --------
    MneExperiment.tests

    Examples
    --------
    The first example assumes 2 categorical variables present in events,
    'a' with values 'a1' and 'a2', and 'b' with values 'b1' and 'b2'. These are
    recoded into 0/1 codes::

        TwoStageTest("a_num + b_num + a_num * b_num + index + a_num * index"},
                     vars={'a_num': ('a', {'a1': 0, 'a2': 1}),
                           'b_num': ('b', {'b1': 0, 'b2': 1})})

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


TEST_CLASSES = {
    'anova': ANOVA,
    'ttest_1samp': TTestOneSample,
    'ttest_rel': TTestRelated,
    'ttest_ind': TTestIndependent,
    't_contrast_rel': TContrastRelated,
    'two-stage': TwoStageTest,
}


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

    def data_to_ndvar(self, info: mne.Info) -> List[str]:
        assert self.sensor
        if self._to_ndvar is None:
            return info.get_channel_types(unique=True, only_data_chs=True)
        else:
            return self._to_ndvar


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


class ROI2StageResult(ROITestResult):
    """Test results for 2-stage tests in one or more ROIs

    Attributes
    ----------
    subjects : tuple of str
        Subjects included in the test.
    samples : int
        ``samples`` parameter used for permutation tests.
    res : {str: LMGroup} dict
        Test result for each ROI.
    n_trials_ds : Dataset
        Dataset describing how many trials were used in each condition per
        subject.
    """
