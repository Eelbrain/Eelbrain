from collections.abc import Collection
from pathlib import Path

import numpy as np

from .._data_obj import Dataset, Factor, Var
from .._io.txt import tsv
from .._text import enumeration
from .configuration import Configuration, ConfigurationError


class GroupBase(Configuration):

    pass


class Group(GroupBase):
    """Group defined as collection of subjects

    Parameters
    ----------
    subjects
        Group members. Either a collection of subject names, or an expression
        based on the columns of the BIDS dataset's ``participants.tsv`` file
        which is true for the group members (e.g., ``"hand == 'R'"``).

    See Also
    --------
    Pipeline.groups

    Notes
    -----
    Expressions are evaluated like :meth:`Dataset.eval` on the rows of
    ``participants.tsv`` that correspond to the experiment's subjects. Columns
    in which all values are numbers are available as :class:`Var`, other columns
    as :class:`Factor`. Missing values (``n/a``) are ``NaN`` in a :class:`Var`
    and ``'n/a'`` in a :class:`Factor`. A subject with a missing value is thus
    never included by a numeric comparison (``"age > 30"``) or by ``==``, but it
    is included by ``!=`` (use ``"hand.isin(('L', 'A'))"`` rather than
    ``"hand != 'R'"`` to leave out subjects with unknown handedness).

    Examples
    --------
    Define groups based on ``participants.tsv`` columns::

        groups = {
            'right-handed': Group("hand == 'R'"),
            'young-female': Group("(age < 30) & (sex == 'F')"),
        }

    """

    def __init__(self, subjects: Collection[str] | str):
        if isinstance(subjects, str):
            self.subjects = subjects  # expression, evaluated in assemble_groups()
        else:
            self.subjects = set(subjects)
            if len(self.subjects) != len(subjects):
                raise ConfigurationError(f"At least one duplicate subject in {subjects}")

    @staticmethod
    def _coerce(obj):
        if isinstance(obj, GroupBase):
            return obj
        elif isinstance(obj, dict):
            if 'base' in obj:
                return SubGroup(**obj)
            else:
                return SubGroup('all', **obj)
        else:
            return Group(obj)


class SubGroup(GroupBase):
    """Group defined by removing subjects from a base group

    Parameters
    ----------
    base
        The name of the group to base the new group on (e.g., ``'all'``).
    exclude
        Subjects to exclude (e.g., ``("R0026", "R0042", "R0066")``).

    See Also
    --------
    Pipeline.groups
    """

    def __init__(self, base: str, exclude: Collection[str]):
        self.base = base
        self.exclude = {exclude} if isinstance(exclude, str) else set(exclude)


def load_participants(root: Path, subjects: set[str]) -> Dataset:
    "Subject variables from the BIDS ``participants.tsv`` file, with one row for each of ``subjects``"
    path = root / 'participants.tsv'
    if not path.exists():
        raise ConfigurationError(f"A group is defined through an expression, which requires subject variables, but {path} does not exist")
    participants = tsv(path, encoding='utf-8-sig')
    if 'participant_id' not in participants:
        raise ConfigurationError(f"{path} has no participant_id column")
    # BIDS marks missing values with n/a, which makes load.tsv read numeric columns as Factor
    for key, column in list(participants.items()):
        if isinstance(column, Factor) and key != 'participant_id':
            try:
                participants[key] = Var([float('nan' if value == 'n/a' else value) for value in column])
            except ValueError:
                pass
    participants['subject'] = Factor([participant.removeprefix('sub-') for participant in participants['participant_id']])
    if missing := subjects.difference(participants['subject'].cells):
        raise ConfigurationError(f"A group is defined through an expression, which requires subject variables for all subjects, but {path} has no entry for {enumeration(sorted(missing))}")
    return participants.sub(participants['subject'].isin(subjects))


def assemble_groups(
    groups: dict, 
    subjects: set[str], 
    root: Path,
) -> dict:
    if 'all' in groups:  # Pipeline needs access to all subjects
        raise ConfigurationError("The group name 'all' is reserved and can't be used for a user-defined group")
    all_groups = {k: Group._coerce(v) for k, v in groups.items()}
    all_groups['all'] = Group(subjects)
    base_groups = {k: g for k, g in all_groups.items() if isinstance(g, Group)}
    sub_groups = {k: g for k, g in all_groups.items() if isinstance(g, SubGroup)}
    assert len(base_groups) + len(sub_groups) == len(all_groups)
    # check base-groups
    groups = {}
    participants = None
    for key, group in base_groups.items():
        if isinstance(group.subjects, str):
            if participants is None:
                participants = load_participants(root, subjects)
            if group.subjects in subjects:
                hint = f" To define a group with a single subject, use Group([{group.subjects!r}])."
            else:
                hint = ''
            try:
                index = participants.eval(group.subjects)
            except Exception as error:
                raise ConfigurationError(f"Group {key}: can't evaluate {group.subjects!r} with the columns of participants.tsv ({enumeration(participants.keys())}): {error}.{hint}") from error
            if isinstance(index, Var):
                index = index.x
            if not (isinstance(index, np.ndarray) and index.dtype.kind == 'b' and index.shape == (participants.n_cases,)):
                raise ConfigurationError(f"Group {key}: {group.subjects!r} does not evaluate to a boolean value for each subject, but to {index!r}.{hint}")
            groups[key] = tuple(sorted(participants[index, 'subject']))
        else:
            missing = group.subjects - subjects
            if missing:
                raise ConfigurationError(f"Group {key} contains non-existing subjects: {missing}")
            groups[key] = tuple(sorted(group.subjects))
    # assign subgroups
    while sub_groups:
        for key, group in sub_groups.items():
            if group.base in groups:
                break
        else:
            raise ValueError("Groups contain unresolvable definition")
        group = sub_groups.pop(key)
        base_subjects = set(groups[group.base])
        invalid = group.exclude - base_subjects
        if invalid:
            raise ConfigurationError(f"Group {key} trying to exclude subjects not contained in its base {group.base}: {invalid}")
        groups[key] = tuple(sorted(base_subjects - group.exclude))
    return groups
