"""Variables

A :class:`VarDef` adds one column to the events. Both halves of what the
pipeline does with them -- adding them to data, and tracking them in the cache
-- are the same operation, :meth:`Variables.resolve`: visit the definitions in
order and add each one whose input columns the data provides.

Adding
------
:meth:`Variables.resolve` is called wherever variables need to be present:
per subject in ``labeled-events``, and again on the nodes that combine subjects
(``evoked-group-dataset``, ``evoked-stc-group-dataset``, ``trf-group-dataset``),
where it is applied as a *view* -- the variables are part of the returned data,
but not of those nodes' fingerprints. What the second call can add depends on
the data rather than on the definitions: an evoked dataset has only the event
columns that survive averaging, and a TRF dataset carries no event columns at
all, so there a variable keyed on ``subject`` is the only one that applies.

Tracking
--------
The same operation produces cache fingerprints. A consumer builds a *shell* --
the non-data columns of the dataset it analyzes -- resolves the variables it
reads against it, and records the resulting values. Where the shell follows from
configuration alone it is synthesized (``trf-model-test`` builds one from the
group and the epoch); otherwise it comes from the node that owns those columns:
the ``shell`` view of ``evoked`` for data averaged within model cells, the
events themselves for single-trial data. Either way it is loaded in
:meth:`Derivative.dependency_fingerprint_override`, which is the only place a
node may load during a cache-validity check.

Recording *values* rather than definitions is what keeps cache churn down:
editing a label that no retained event maps to leaves the result valid. It also
covers a value that comes from outside the definition, such as the
:attr:`Pipeline.groups` behind a :class:`GroupVar`, which no definition-based
fingerprint can observe. Passing ``names`` to :meth:`Variables.resolve` makes
the coverage checked rather than assumed: a variable the consumer reads but that
does not resolve raises instead of being silently dropped.

The one node that fingerprints *definitions* is ``labeled-events``, and
necessarily so: it produces the event columns, so it can not be validated
against its own output. That is cheap and correct, but it means a definition
that spans subjects would invalidate every subject's events whenever any one
subject's entry changed. Such variables -- :class:`GroupVar`,
:class:`LabelVar` on ``'subject'``, and anything derived from either -- are
therefore marked :attr:`VarDef.across_subjects` and deferred to the nodes that
combine subjects. The cost is that they are absent from single-subject data, and
so can not be used where data is processed one subject at a time, such as in an
epoch ``sel`` expression or an evoked ``model``.
"""
from __future__ import annotations

from typing import Any
from fnmatch import fnmatch as fnmatch_func
from collections.abc import Collection, Sequence

from .._data_obj import EVAL_CONTEXT_NAMES, Dataset, Factor, Var, asuv, assert_is_legal_dataset_key
from .._info import INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS
from .._text import enumeration
from .._utils.numpy_utils import INT_TYPES
from .._utils.parse import find_variables
from .configuration import Configuration, ConfigurationError
from .pathing import BIDS_ENTITY_KEYS


# Event columns that are written by the pipeline itself, and would be silently overwritten by a user
# variable. Only names that are present regardless of configuration belong here; the general guarantee
# that a variable never replaces a column is enforced in :meth:`Variables.resolve`, which is what covers
# the names that follow from a call rather than from the definitions (TRF metrics and kernels, the
# evoked response). This list is the subset that can be reported when the pipeline is constructed.
RESERVED_VAR_KEYS = (
    *BIDS_ENTITY_KEYS,  # promoted from ds.info to a column where recordings are combined
    'sample', 'value', 'onset', 'index', 'epoch', 'accept',  # event columns the pipeline reads by name
    INTERPOLATE_CHANNELS, INTERPOLATE_WINDOWS,  # copied from the rejection file
    'epochs', 'evoked', 'src', 'stc', 'label_tc', 'model',  # data columns added downstream
    'epoch_time', 'events', 'tmax',  # ContinuousEpoch
)


def _find_unresolvable_columns(names: set[str], data: Dataset) -> set[str]:
    """Which of ``names`` neither ``data`` nor :meth:`Dataset.eval` can supply"""
    return names.difference(data).difference(EVAL_CONTEXT_NAMES)


class VarDef(Configuration):
    """Base class for adding variables to events"""
    # Whether the definition spans subjects, which defers it to the nodes that combine subjects (see module docstring)
    across_subjects = False

    def __init__(self, task):
        self.task = task

    def _apply(self, ds, groups):
        raise NotImplementedError

    def _input_vars(self) -> set[str]:
        raise NotImplementedError

    def _applies_to_task(self, data: Dataset) -> bool:
        "Whether this definition's task restriction admits ``data``"
        if self.task is None:
            return True
        task = data.info.get('task', None)
        if task is not None:
            return self.task == task
        elif 'task' not in data:
            return False
        # where recordings are combined, the task is a column instead
        tasks = set(data['task'])
        if tasks == {self.task}:
            return True
        elif self.task not in tasks:
            return False
        raise NotImplementedError(f"{self} is restricted to task={self.task!r}, but the data combines tasks {enumeration([repr(t) for t in sorted(tasks)])}; a task-restricted variable can not be applied to data that combines several tasks")


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

    def _input_vars(self) -> set[str]:
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

    Notes
    -----
    With ``source='subject'`` the definition spans subjects (it lists a value
    for each of them), which defers it to the point where different subjects'
    data are combined; see :class:`GroupVar` for the implications. Only that
    exact source is deferred: a ``source`` that merely involves the subject,
    such as ``'subject%value'``, is applied per subject like any other event
    variable, and so any subject's entry invalidates every subject's events.
    """
    DICT_ATTRS = ('task', 'source', 'labels', 'is_factor', 'default', 'fnmatch')

    def __init__(
            self,
            source: str,
            codes: dict[str | float | tuple[str, ...] | tuple[float, ...], str | float],
            default: str | float | bool | None = True,
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
        elif default is False:
            default = None
        elif default is not None:
            if isinstance(default, str) != is_factor:
                raise TypeError(f"{default=}")
        self.default = default
        self.fnmatch = fnmatch
        self.across_subjects = self.source == 'subject'

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

    def _input_vars(self) -> set[str]:
        return find_variables(self.source)


class GroupVar(VarDef):
    """Group membership for each subject

    Parameters
    ----------
    groups
        Groups to label. A sequence of group names to label each subject with
        the group it belongs to (subjects must not be members of more than one
        group). Alternatively, a ``{group: label}`` dictionary can be used to
        assign a label different from the group name.
    task
        Only apply the variable to events from this task.

    See Also
    --------
    Pipeline.variables

    Notes
    -----
    A ``GroupVar`` is an *across-subject* variable: its definition spans
    subjects, so it is added where different subjects' data are combined. It is
    thus only present in data that spans subjects (e.g.
    ``e.load_selected_events(-1)``, but not ``e.load_selected_events()``).
    Consequently it can not be used in definitions that are evaluated per
    subject, such as an epoch ``sel`` expression or an evoked ``model``. The
    same applies to any variable derived from a ``GroupVar``.

    Examples
    --------
    Assuming an experiment that defines two groups, ``'patient'`` and
    ``'control'``, these groups could be labeled with::

        GroupVar(['patient', 'control'])

    """
    DICT_ATTRS = ('task', 'groups')
    across_subjects = True

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

    def _input_vars(self) -> set[str]:
        return {'subject'}


class Variables(Configuration):
    """Set of variable definitions

    Parameters
    ----------
    arg
        Dictionary mapping variable names to variable definitions (:class:`EvalVar`, :class:`GroupVar` or :class:`LabelVar`).

    Attributes
    ----------
    event_vars
        Variables that are applied to a single subject's data.
    across_subject_vars
        Variables that are deferred to where subjects are combined.
    """

    def __init__(self, arg: dict[str, VarDef] | None = None):
        self.vars = {}
        if arg:
            for name, vdef in arg.items():
                if not isinstance(vdef, VarDef):
                    raise TypeError(f"Variable {name!r}: expected VarDef, got {vdef!r}")
                assert_is_legal_dataset_key(name)
                if name in RESERVED_VAR_KEYS:
                    raise ConfigurationError(f"Variable {name!r}: reserved name; this column is written by the pipeline itself and a variable of the same name would be overwritten")
                # variables are applied in definition order, so an input defined here has to come first
                if late := [v for v in vdef._input_vars() if v not in self.vars and v in arg]:
                    raise ConfigurationError(f"Variable {name!r}: {vdef} uses {enumeration([repr(v) for v in late])}, which {'are' if len(late) > 1 else 'is'} defined later; variables are applied in the order they are defined, so an input has to be defined first")
                self.vars[name] = vdef
        self.across_subject_vars = self._find_across_subject_vars()
        self.event_vars = {name: vdef for name, vdef in self.vars.items() if name not in self.across_subject_vars}

    def _as_dict(self):
        return self.vars

    def _find_across_subject_vars(
            self,
            deferred: Collection[str] = (),
    ) -> dict[str, VarDef]:
        """The across-subject variables among these, in definition order

        Parameters
        ----------
        deferred
            Names of across-subject variables from an enclosing scope.
            A test's ``vars`` are applied on top of :attr:`Pipeline.variables`,
            so a test variable can read a global one.

        Notes
        -----
        A variable is across-subject when it is one itself, or when *any* of its
        inputs is: a deferred input is only added where subjects are combined,
        and evaluating a variable requires all of its inputs, so it can not be
        evaluated any earlier. A single forward pass suffices because an input
        defined here is always defined first (see :meth:`__init__`).
        """
        across = set(deferred)
        out = {}
        for name, vdef in self.vars.items():
            if vdef.across_subjects or not across.isdisjoint(vdef._input_vars()):
                across.add(name)
                out[name] = vdef
        return out

    def _check_trigger_vars(self):
        for key, var in self.vars.items():
            if isinstance(var, LabelVar) and var.source == 'value':
                if not all(isinstance(v, INT_TYPES) for v in var.labels):
                    raise ConfigurationError(f"Variable {key!r}: {var} codes must be integers")

    def _check_group_vars(self, groups: Collection[str], desc: str) -> None:
        """Make sure every :class:`GroupVar` refers to defined groups

        Checked by the :class:`Pipeline`, which is what knows the groups: these are
        also a test's ``vars``, user-defined objects shared between pipelines that may
        define different groups. Resolving a ``GroupVar`` happens in ``fingerprint()``,
        where a configuration error would surface during a cache-validity check.
        """
        for name, vdef in self.vars.items():
            if isinstance(vdef, GroupVar) and (undefined := [group for group in vdef.groups if group not in groups]):
                raise ConfigurationError(f"{desc}[{name!r}]: {vdef} refers to undefined group(s) {enumeration([repr(group) for group in undefined])}")

    def __repr__(self):
        return '\n'.join(["Variables(", *(f'    {k!r}: {v},' for k, v in self.vars.items()), ')'])

    def __bool__(self):
        return bool(self.vars)

    def resolve(
            self,
            data: Dataset,
            groups: dict[str, tuple[str, ...]] = None,
            names: set[str] | None = None,
            across_subject_only: bool = False,
            require_inputs: bool = False,
    ) -> dict[str, Any]:
        """Add variables to ``data``, in place

        Parameters
        ----------
        data
            Dataset to add the variables to. A variable whose input columns are
            missing is skipped, so this can also be a *shell*: a dataset with
            only the columns that are known without loading data, used to
            resolve variable values for a cache fingerprint.
        groups
            Members of each group, from :attr:`Pipeline.groups`. ``None`` for
            data from a single subject, where across-subject variables are not
            applied (see the module docstring).
        names
            Variables the caller needs. If any of these can not be resolved
            in the evaluation context, :exc:`ValueError` is raised.
            Those ``names`` that ``data`` provides are returned.
        across_subject_only
            Only add the across-subject variables. For :attr:`Pipeline.variables`
            on data that combines subjects, where the event variables are already
            present, applied per subject: re-deriving them from columns that have
            been averaged would not reproduce them. Not for a test's ``vars``,
            which are applied to the combined data in full.
        require_inputs
            ``data`` holds every column that will ever be available, so a
            variable whose inputs are missing is a definition error rather than
            one that belongs to a later stage. For the events, where the
            variables are first applied.

        Returns
        -------
        values
            The column for each of ``names`` that ``data`` provides; empty without
            ``names``.

        Notes
        -----
        Which names in a definition are input columns follows from ``data`` rather than
        from the expression, since ``data`` could override a ``builtin`` like ``max``.
        A definition that fails to evaluate is therefore reported as a configuration
        error naming the columns it had to work with.

        A variable never replaces a column that ``data`` already provides, since that
        column could be the analysis data itself (a TRF metric or kernel, the evoked
        response) or another variable's. Which names are at stake depends on the data
        rather than on the definitions, so this is checked here rather than against
        ``RESERVED_VAR_KEYS``, which only covers the names that are known up front.
        """
        use_vars = self.across_subject_vars if across_subject_only else self.vars
        for name, vdef in use_vars.items():
            if groups is None and name in self.across_subject_vars:
                continue
            elif not vdef._applies_to_task(data):
                continue
            elif name in data:
                raise ConfigurationError(f"Variable {name!r}: {vdef} would overwrite the {name!r} column that the data already provides; rename the variable")
            elif missing := _find_unresolvable_columns(vdef._input_vars(), data):
                if require_inputs:
                    raise ConfigurationError(f"Variable {name!r}: {vdef} is computed from {enumeration([repr(key) for key in missing])}, which {'are' if len(missing) > 1 else 'is'} not among the event columns {enumeration([repr(key) for key in data])}")
                continue
            try:
                data[name] = vdef._apply(data, groups)
            except Exception as error:
                # An input the data does not provide was resolved from the evaluation context instead, which is the likely cause
                shadowed = [key for key in vdef._input_vars() if key not in data and key in EVAL_CONTEXT_NAMES]
                detail = f"; {enumeration([repr(key) for key in shadowed])} {'are' if len(shadowed) > 1 else 'is'} not among them and {'were' if len(shadowed) > 1 else 'was'} resolved from the evaluation context instead" if shadowed else ''
                raise ConfigurationError(f"Variable {name!r}: {vdef} could not be computed from the columns {enumeration([repr(key) for key in data]) or 'of an empty dataset'} ({error}){detail}") from error

        if names is None:
            return {}
        if missing := _find_unresolvable_columns(names, data):
            raise ValueError(f"{enumeration([repr(name) for name in missing])} can not be resolved from {enumeration([repr(key) for key in data]) or 'an empty dataset'}; a variable can only be used where the data provides what it is computed from")
        return {name: data[name] for name in names if name in data}


def label_groups(
        subject: Factor,
        groups: Sequence[str] | dict[str, str],
        subject_groups: dict[str, tuple[str, ...]],
) -> Factor:
    """Generate Factor for group membership.

    Parameters
    ----------
    subject
        Subject of each case.
    groups
        Groups to label, as sequence of group names or ``{group: label}``.
    subject_groups
        Members of each group, from :attr:`Pipeline.groups`.
    """
    if not isinstance(groups, dict):
        groups = {g: g for g in groups}
    labels = {s: [label for group, label in groups.items() if s in subject_groups[group]] for s in subject.cells}
    problems = [s for s, g in labels.items() if len(g) != 1]
    if problems:
        desc = (', '.join(labels[s]) if labels[s] else 'no group' for s in problems)
        msg = ', '.join(f'{p} ({d})' for p, d in zip(problems, desc))
        raise ValueError(f"Groups {groups} are not unique for subjects: {msg}")
    return Factor(subject, labels={s: g[0] for s, g in labels.items()})
