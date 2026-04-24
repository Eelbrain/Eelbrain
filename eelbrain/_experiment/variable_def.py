"""Variables"""
from typing import Any
from fnmatch import fnmatch as fnmatch_func
from collections.abc import Sequence

from .._data_obj import Dataset, Factor, Var, asuv, assert_is_legal_dataset_key
from .._utils.numpy_utils import INT_TYPES
from .._utils.parse import find_variables
from .configuration import Configuration, ConfigurationError


# Some event columns are reserved for Eelbrain
RESERVED_VAR_KEYS = ('subject', 'task', 'visit')


class VarDef(Configuration):
    """Base class for adding variables to events"""

    def __init__(self, task):
        self.task = task

    def _apply(self, ds, groups):
        raise NotImplementedError

    def _input_vars(self):
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
    DICT_ATTRS = ('task', 'code')

    def __init__(self, code: str, task: str = None):
        super().__init__(task)
        assert isinstance(code, str)
        self.code = code

    def __repr__(self):
        return f"EvalVar({self.code!r})"

    def _apply(self, ds, groups):
        return asuv(self.code, data=ds)

    def _input_vars(self):
        return find_variables(self.code)


class LabelVar(VarDef):
    """Variable assigning labels to values

    Parameters
    ----------
    source
        Variable supplying the values (e.g., ``"value"``).
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
    DICT_ATTRS = ('task', 'source', 'labels', 'is_factor', 'default', 'fnmatch')

    def __init__(
            self,
            source: str,
            codes: dict[str | float | tuple[str, ...] | tuple[float, ...], str | float],
            default: bool | str | float = True,
            task: str = None,
            fnmatch: bool = False,
    ):
        super().__init__(task)
        self.source = source
        self.codes = codes
        self.labels = {}
        is_factor = None
        for key, v in codes.items():
            if is_factor is None:
                is_factor = isinstance(v, str)
            elif isinstance(v, str) != is_factor:
                raise ConfigurationError(f"LabelVar with {codes=}: value type inconsistent, need all or none to be str")

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

    def _apply(self, ds, groups):
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

    def _input_vars(self):
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
    DICT_ATTRS = ('task', 'groups')

    def __init__(
            self,
            groups: Sequence[str] | dict[str, str],
            task: str = None,
    ):
        super().__init__(task)
        self.groups = groups

    def __repr__(self):
        return f"GroupVar({self.groups!r})"

    def _apply(self, ds, groups):
        return label_groups(ds['subject'], self.groups, groups)

    @classmethod
    def _from_string(cls, string):
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

    def _input_vars(self):
        return ()


class Variables(Configuration):
    """Set of variable definitions

    Parameters
    ----------
    arg
        Dictionary mapping variable names to :class:`VarDef` instances.
    """

    def __init__(self, arg: dict[str, VarDef] | None = None):
        self.vars = {}
        if not arg:
            return
        for name, vdef in arg.items():
            if not isinstance(vdef, VarDef):
                raise TypeError(f"Variable {name!r}: expected VarDef, got {vdef!r}")
            assert_is_legal_dataset_key(name)
            if name in RESERVED_VAR_KEYS:
                raise ConfigurationError(f"Variable {name!r}: reserved name")
            self.vars[name] = vdef

    def _as_dict(self):
        return self.vars

    def _check_trigger_vars(self):
        for key, var in self.vars.items():
            if isinstance(var, LabelVar) and var.source == 'value':
                if not all(isinstance(v, INT_TYPES) for v in var.labels):
                    raise ConfigurationError(f"Variable {key!r}: {var} codes must be integers")

    def __repr__(self):
        return '\n'.join(["Variables(", *(f'    {k!r}: {v},' for k, v in self.vars.items()), ')'])

    def __bool__(self):
        return bool(self.vars)

    def _apply(self, ds, groups, group_only=False):
        task = ds.info.get('task', None)
        for name, vdef in self.vars.items():
            if group_only and not isinstance(vdef, GroupVar):
                continue
            elif vdef.task is None or vdef.task == task:
                ds[name] = vdef._apply(ds, groups)


def apply_vardef(
        ds: Dataset,
        vardef: Variables | None | str,
        tests: dict[str, Any],
        groups: dict[str, Any],
) -> None:
    if isinstance(vardef, str):
        vardef = tests[vardef].vars
    if vardef:
        vardef._apply(ds, groups)


def label_groups(subject, groups, subject_groups):
    """Generate Factor for group membership."""
    if not isinstance(groups, dict):
        groups = {g: g for g in groups}
    labels = {s: [label for group, label in groups.items() if s in subject_groups[group]] for s in subject.cells}
    problems = [s for s, g in labels.items() if len(g) != 1]
    if problems:
        desc = (', '.join(labels[s]) if labels[s] else 'no group' for s in problems)
        msg = ', '.join('%s (%s)' % pair for pair in zip(problems, desc))
        raise ValueError(f"Groups {groups} are not unique for subjects: {msg}")
    labels = {s: g[0] for s, g in labels.items()}
    return Factor(subject, labels=labels)
