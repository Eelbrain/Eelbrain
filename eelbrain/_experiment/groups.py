from typing import Collection, Set

from .definitions import Definition, DefinitionError


class GroupBase(Definition):

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
                raise DefinitionError(f"At least one duplicate subject in {subjects}")

    def _link(self, key: str, all_subjects: Set[str]):
        missing = self.subjects - all_subjects
        if missing:
            raise DefinitionError(f"Group {key} contains non-existing subjects: {missing}")
        return tuple(sorted(self.subjects))

    @staticmethod
    def coerce(obj):
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

    def _link(self, key: str, all_subjects: Set[str]):
        invalid = self.exclude - all_subjects
        if invalid:
            raise DefinitionError(f"Group {key} trying to exclude subjects not contained in its base {self.base}: {invalid}")
        return tuple(sorted(all_subjects - self.exclude))


def assemble_groups(groups: dict, subjects: Set[str]) -> dict:
    if 'all' in groups:  # Pipeline needs access to all subjects
        raise DefinitionError("The group name 'all' is reserved and can't be used for a user-defined group")
    all_groups = {k: Group.coerce(v) for k, v in groups.items()}
    all_groups['all'] = Group(subjects)
    base_groups = {k: g for k, g in all_groups.items() if isinstance(g, Group)}
    sub_groups = {k: g for k, g in all_groups.items() if isinstance(g, SubGroup)}
    assert len(base_groups) + len(sub_groups) == len(all_groups)
    # check base-groups
    groups = {key: group._link(key, subjects) for key, group in base_groups.items()}
    # assign subgroups
    while sub_groups:
        for key, group in sub_groups.items():
            if group.base in groups:
                break
        else:
            raise ValueError("Groups contain unresolvable definition")
        group = sub_groups.pop(key)
        groups[key] = group._link(key, set(groups[group.base]))
    return groups
