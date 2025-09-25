"""Variables

With multiple sessions, the same variable name might have multiple definitions::

variables = {
    'name': [Var(session='1'), Var(session='2')],
    'name2': Var(session='2'),
}

"""
from fnmatch import fnmatch as fnmatch_func
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from .._data_obj import Factor, Var, assert_is_legal_dataset_key
from .._utils.numpy_utils import INT_TYPES
from .._utils.parse import find_variables
from .definitions import DefinitionError


# Some event columns are reserved for Eelbrain
RESERVED_VAR_KEYS = ('subject', 'task', 'visit')


def as_vardef_var(v):
    "Coerce ds.eval() output for use as variable"
    if isinstance(v, np.ndarray):
        if v.dtype.kind == 'b':
            return Var(v.astype(int))
        return Var(v)
    return v


class VarDef:
    _pickle_args = ('task',)

    def __init__(self, task):
        self.task = task

    def __getstate__(self):
        return {k: getattr(self, k) for k in self._pickle_args}

    def __setstate__(self, state):
        for k in self._pickle_args:
            setattr(self, k, state[k])

    @property
    def _eq_args(self):
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other._eq_args == self._eq_args

    def apply(self, ds, e):
        raise NotImplementedError

    def input_vars(self):
        raise NotImplementedError


class EvalVar(VarDef):
    """Variable based on evaluating a statement

    Parameters
    ----------
    code
        Statement to evaluate.
    task
        Only apply the variable to events from this task.

    See Also
    --------
    Pipeline.variables
    """
    _pickle_args = ('task', 'code')

    def __init__(self, code: str, task: str = None):
        super(EvalVar, self).__init__(task)
        assert isinstance(code, str)
        self.code = code

    def __repr__(self):
        return "EvalVar(%r)" % self.code

    @property
    def _eq_args(self):
        return self.code,

    def apply(self, ds, e):
        return as_vardef_var(ds.eval(self.code))

    def input_vars(self):
        return find_variables(self.code)


class LabelVar(VarDef):
    """Variable assigning labels to values

    Parameters
    ----------
    source
        Variable supplying the values (e.g., ``"trigger"``).
    codes
        Mapping values in ``source`` to values in the new variable. The type
        of the values determines whether the output is a :class:`Factor`
        (:class:`str` values) or a :class:`Var` (numerical values).
    default
        Label for values not in ``codes``. By default, this is ``''`` for
        categorial and 0 for numerical output. Set to ``False`` to pass through
        unlabeled input values.
    task
        Only apply the variable to events from this task.
    fnmatch
        Treat keys in ``codes`` as :mod:`fnmatch` patterns.

    See Also
    --------
    Pipeline.variables
    """
    _pickle_args = ('task', 'source', 'codes', 'labels', 'is_factor', 'default', 'fnmatch')

    def __init__(
            self,
            source: str,
            codes: Dict[Union[str, float, Tuple[str, ...], Tuple[float, ...]], Union[str, float]],
            default: Union[bool, str, float] = True,
            task: str = None,
            fnmatch: bool = False,
    ):
        super(LabelVar, self).__init__(task)
        self.source = source
        self.codes = codes
        self.labels = {}
        is_factor = None
        for key, v in codes.items():
            if is_factor is None:
                is_factor = isinstance(v, str)
            elif isinstance(v, str) != is_factor:
                raise DefinitionError(f"LabelVar with {codes=}: value type inconsistent, need all or none to be str")

            if isinstance(key, tuple):
                for k in key:
                    self.labels[k] = v
            else:
                self.labels[key] = v
        self.is_factor = is_factor
        if default is True:
            default = '' if is_factor else 0
        elif default is not None:
            if isinstance(default, str) != is_factor:
                raise TypeError(f"{default=}")
        self.default = default
        self.fnmatch = fnmatch

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source!r}, {self.codes})"

    @property
    def _eq_args(self):
        return self.source, self.labels, self.default, self.fnmatch

    def apply(self, ds, e):
        source = ds.eval(self.source)
        if self.fnmatch:
            labels = {}
            for value in source.cells:
                for pattern, target in self.labels.items():
                    if fnmatch_func(value, pattern):
                        labels[value] = target
        else:
            labels = self.labels
        if self.is_factor:
            return Factor(source, labels=labels, default=self.default)
        else:
            return Var.from_dict(source, labels, default=self.default)

    def input_vars(self):
        return find_variables(self.source)


class GroupVar(VarDef):
    """Group membership for each subject

    Parameters
    ----------
    groups
        Groups to label. A sequence of group names to label each subject with
        the group it belongs to (subjects can't be members of more than one
        group). Alternatively, a ``{group: label}`` dictionary can be used to
        assign a different label based on group membership.
    task
        Only apply the variable to events from this task.

    See Also
    --------
    Pipeline.variables

    Examples
    --------
    Assuming an experiment that defines two groups, ``'patient'`` and
    ``'control'``, these groups could be labeled with::

        GroupVar(['patient', 'control'])

    """
    _pickle_args = ('task', 'groups')

    def __init__(
            self,
            groups: Union[Sequence[str], Dict[str, str]],
            task: str = None,
    ):
        super(GroupVar, self).__init__(task)
        self.groups = groups

    def __repr__(self):
        return f"GroupVar({self.groups!r})"

    @property
    def _eq_args(self):
        return self.groups,

    def apply(self, ds, e):
        return e.label_groups(ds['subject'], self.groups)

    @classmethod
    def from_string(cls, string):
        groups = {}
        for item in string.split(','):
            if ':' in item:
                src, dst = map(str.strip, item.split(':'))
            else:
                src = dst = item.strip()
            groups[src] = dst
        if all(k == v for k, v in groups.items()):
            groups = tuple(sorted(groups))
        return cls(groups)

    def input_vars(self):
        return ()


def parse_named_vardef(string):
    if '=' not in string:
        raise DefinitionError(f"variable {string!r}: needs '='")
    name, vdef = string.split('=', 1)
    return name.strip(), parse_vardef(vdef)


def parse_vardef(string):
    string = string.strip()
    if string.startswith('group:'):
        return GroupVar.from_string(string[6:])
    else:
        return EvalVar(string)


class Variables:
    """Set of variable definitions

    Parameters
    ----------
    arg : str | tuple | dict
        The ``vars`` argument.
    """
    def __init__(self, arg: dict):
        if arg is None:
            arg = ()
        elif isinstance(arg, str):
            arg = (arg,)
        elif isinstance(arg, dict):
            arg = arg.items()
        elif not isinstance(arg, (tuple, list)):
            raise TypeError(f"vars={arg!r}")

        self.vars = {}
        for item in arg:
            if isinstance(item, str):
                name, vdef = parse_named_vardef(item)
            else:
                name, vdef = item
                if isinstance(vdef, str):
                    vdef = parse_vardef(vdef)
                elif isinstance(vdef, VarDef):
                    pass
                elif isinstance(vdef, dict):
                    if 'default' in vdef:
                        vdef = vdef.copy()
                        default = vdef.pop('default')
                    else:
                        default = True
                    vdef = LabelVar('trigger', vdef, default)
                elif isinstance(vdef, tuple):
                    vdef = LabelVar(*vdef)
                else:
                    raise DefinitionError(f"Variable {name!r}: {vdef!r}")

            assert_is_legal_dataset_key(name)
            if name in RESERVED_VAR_KEYS:
                raise DefinitionError(f"Variable {name!r}: reserved name")
            self.vars[name] = vdef

    def __getstate__(self):
        return {'vars': self.vars}

    def __setstate__(self, state):
        self.vars = state['vars']

    def _check_trigger_vars(self):
        for key, var in self.vars.items():
            if isinstance(var, LabelVar) and var.source == 'trigger':
                if not all(isinstance(v, INT_TYPES) for v in var.labels):
                    raise DefinitionError(f"Variable {key!r}: {var} codes must be integers")

    def __repr__(self):
        return '\n'.join(["Variables(", *(f'    {k!r}: {v},' for k, v in self.vars.items()), ')'])

    def __eq__(self, other):
        return isinstance(other, Variables) and other.vars == self.vars

    def apply(self, ds, e, group_only=False):
        task = ds.info.get('task', None)
        for name, vdef in self.vars.items():
            if group_only and not isinstance(vdef, GroupVar):
                continue
            elif vdef.task is None or vdef.task == task:
                ds[name] = vdef.apply(ds, e)
