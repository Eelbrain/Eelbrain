from collections.abc import Collection

from .configuration import Configuration, ConfigurationError


class GroupBase(Configuration):

    pass


class Group(GroupBase):
    """Group defined as collection of subjects

    Parameters
    ----------
    subjects : collection of str
        Group members.

    See Also
    --------
    Pipeline.groups
    """

    def __init__(self, subjects: Collection[str]):
        if isinstance(subjects, str):
            self.subjects = {subjects}
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
    base : str
        The name of the group to base the new group on (e.g., ``'all'``).
    exclude : collection of str
        Subjects to exclude (e.g., ``("R0026", "R0042", "R0066")``).

    See Also
    --------
    Pipeline.groups
    """

    def __init__(self, base: str, exclude: Collection[str]):
        self.base = base
        self.exclude = {exclude} if isinstance(exclude, str) else set(exclude)


def assemble_groups(groups: dict, subjects: set[str]) -> dict:
    if 'all' in groups:  # Pipeline needs access to all subjects
        raise ConfigurationError("The group name 'all' is reserved and can't be used for a user-defined group")
    all_groups = {k: Group._coerce(v) for k, v in groups.items()}
    all_groups['all'] = Group(subjects)
    base_groups = {k: g for k, g in all_groups.items() if isinstance(g, Group)}
    sub_groups = {k: g for k, g in all_groups.items() if isinstance(g, SubGroup)}
    assert len(base_groups) + len(sub_groups) == len(all_groups)
    # check base-groups
    groups = {}
    for key, group in base_groups.items():
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
