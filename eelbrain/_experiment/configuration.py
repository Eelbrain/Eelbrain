# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import inspect
import logging
from typing import Any
from collections.abc import Sequence

from .._exceptions import ConfigurationError, ConfigurationKeyError
from .._text import enumeration, plural
from .._utils.parse import find_variables


class Configuration:
    DICT_ATTRS = None
    name = None

    def _as_dict(self) -> dict[str, Any]:
        """Return the serialized semantic definition of this configuration.

        This method is the base ``Configuration`` contract for turning a
        configuration object into plain Python data. In the experiment graph,
        the returned mapping is used as the stable definition for:

        - configuration equality (:meth:`Configuration.__eq__`)
        - cache fingerprints and manifests via
          :meth:`eelbrain._experiment.derivative_cache.DerivativeRegistry.canonicalize`
        - explicit derivative ``definitions`` payloads, i.e. serialized
          configuration definitions for epoch, test, parcellation, raw-pipe,
          and inverse-solution fingerprints

        New subclasses should usually implement this by declaring
        :attr:`DICT_ATTRS`, a tuple of attribute names that fully describes the
        configuration's semantic definition. The base implementation then
        returns ``{name: getattr(self, name) for name in self.DICT_ATTRS}``.

        Include in :attr:`DICT_ATTRS` only deterministic definition fields that
        should affect equality and cache identity. Do not include runtime or
        graph-dependent fields that are cached later, such as bound names or
        other dependent parameters populated by configuration-family-specific
        resolution hooks.

        Returns
        -------
        dict
            Plain Python mapping describing the configuration's semantic
            definition.
        """
        if self.DICT_ATTRS is None:
            raise NotImplementedError(f"{self.__class__.__name__}.DICT_ATTRS")
        out = {'type': self.__class__.__name__}
        out.update({k: getattr(self, k) for k in self.DICT_ATTRS})
        return out

    def _store_name(self, name: str) -> None:
        """Store the bound name for diagnostics and runtime convenience."""
        self.name = name

    def __eq__(self, other):
        if isinstance(other, dict):
            return self._as_dict() == other
        elif self.__class__ is other.__class__:
            return self._as_dict() == other._as_dict()
        else:
            return False

    def _repr_args(self):
        args = []
        for name, param in inspect.signature(self.__class__).parameters.items():
            value = getattr(self, name)
            if param.default is param.empty:
                args.append(repr(value))
            elif value != param.default:
                args.append(f'{name}={value!r}')
        return args

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(self._repr_args())})"


class ConfigurationDict(dict):
    """Mapping of names to configurations with informative missing-key errors.

    Parameters
    ----------
    kind
        Singular description of the configuration kind, used in the error
        message when a key is missing (e.g. ``'predictor'``).
    ...
        Items, as for :class:`dict`.
    """

    def __init__(self, kind: str, *args, **kwargs):
        self._kind = kind
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        raise ConfigurationKeyError(key, self._kind, self)

    def __reduce__(self):  # keep pickle / copy.deepcopy working with the custom __init__
        return (self.__class__, (self._kind, dict(self)))


# Names become components of cache/derivative file paths and of dependency-node
# identifiers (e.g. 'raw@{name}'). Forbid the characters that would corrupt that
# structure: whitespace, the '@' and ':' node-name separators, and path separators.
_FORBIDDEN_NAME_CHARS = frozenset(' \t@:/\\')


def name_ok(key: str, allow_empty: bool) -> bool:
    if not isinstance(key, str):
        return False
    elif not key:
        return allow_empty
    elif key in ('.', '..'):  # unsafe as path components
        return False
    return not _FORBIDDEN_NAME_CHARS.intersection(key)


def check_names(keys, attribute, allow_empty: bool):
    invalid = [key for key in keys if not name_ok(key, allow_empty)]
    if invalid:
        raise ConfigurationError(f"Invalid {plural('name', len(invalid))} for {attribute}: {enumeration(invalid)}. Names can not contain whitespace or any of the characters '@', ':', '/' and '\\'.")


def compound(items):
    out = ''
    for item in items:
        if item == '*':
            if not out.endswith('*'):
                out += '*'
        elif item:
            if out and not out.endswith('*'):
                out += '_'
            out += item
    return out


def dict_change(
        old: dict[str, Any],
        new: dict[str, Any],
):
    "Readable representation of dict change"
    lines = []
    keys = set(new)
    keys.update(old)
    for key in sorted(keys):
        if key not in new:
            lines.append(f"{key}: {old[key]!r} -> key removed")
        elif key not in old:
            lines.append(f"{key}: new key -> {new[key]!r}")
        elif new[key] != old[key]:
            lines.append(f"{key}: {old[key]!r} -> {new[key]!r}")
    return lines


def log_dict_change(
        log: logging.Logger,
        kind: str,
        name: str,
        old: dict[str, Any] | None,
        new: dict[str, Any] | None,
):
    if new is None:
        log.warning("  %s %s removed", kind, name)
    elif old is None:
        log.info("  %s %s added", kind, name)
    else:
        log.warning("  %s %s changed:", kind, name)
        for line in dict_change(old, new):
            log.warning("    %s", line)


def log_list_change(log, kind, name, old, new):
    log.warning("  %s %s changed:", kind, name)
    removed = tuple(v for v in old if v not in new)
    if removed:
        log.warning("    Members removed: %s", ', '.join(map(str, removed)))
    added = tuple(v for v in new if v not in old)
    if added:
        log.warning("    Members added: %s", ', '.join(map(str, added)))


def find_epoch_vars(params):
    "Find variables used in a primary epoch definition"
    out = set()
    if params.get('sel'):
        out.update(find_variables(params['sel']))
    if 'trigger_shift' in params and isinstance(params['trigger_shift'], str):
        out.add(params['trigger_shift'])
    if 'post_baseline_trigger_shift' in params:
        out.add(params['post_baseline_trigger_shift'])
    return out


def find_epochs_vars(epochs):
    "Find variables used in all epochs"
    todo = list(epochs)
    out = {}
    while todo:
        for e in tuple(todo):
            p = epochs[e]
            if 'sel_epoch' in p:
                if p['sel_epoch'] in out:
                    out[e] = find_epoch_vars(p)
                    out[e].update(out[p['sel_epoch']])
                    todo.remove(e)
            elif 'sub_epochs' in p:
                if all(se in out for se in p['sub_epochs']):
                    out[e] = find_epoch_vars(p)
                    for se in p['sub_epochs']:
                        out[e].update(out[se])
                    todo.remove(e)
            elif 'collect' in p:
                if all(se in out for se in p['collect']):
                    out[e] = find_epoch_vars(p)
                    for se in p['collect']:
                        out[e].update(out[se])
                    todo.remove(e)
            else:
                out[e] = find_epoch_vars(p)
                todo.remove(e)
    return out


def find_dependent_epochs(epoch, epochs):
    "Find all epochs whose definition depends on epoch"
    todo = set(epochs).difference(epoch)
    out = [epoch]
    while todo:
        last_len = len(todo)
        for e in tuple(todo):
            p = epochs[e]
            if 'sel_epoch' in p:
                if p['sel_epoch'] in out:
                    out.append(e)
                    todo.remove(e)
            elif 'sub_epochs' in p:
                if any(se in out for se in p['sub_epochs']):
                    out.append(e)
                    todo.remove(e)
            elif 'collect' in p:
                if any(se in out for se in p['collect']):
                    out.append(e)
                    todo.remove(e)
            else:
                todo.remove(e)
        if len(todo) == last_len:
            break
    return out[1:]


def typed_arg(arg, type_, secondary_type=None):
    if secondary_type is not None and isinstance(arg, secondary_type):
        return arg
    elif arg is None:
        return None
    else:
        return type_(arg)


def sequence_arg(
        name: str,  # for error message
        arg: Sequence | None,
        item_type: type = str,
        allow_none: bool = True,
        sequence_type: type = tuple,
):
    if arg is None:
        if allow_none:
            return None
    elif isinstance(arg, item_type):
        return sequence_type((arg,))
    elif isinstance(arg, Sequence):
        out = sequence_type(arg)
        if all(isinstance(item, item_type) for item in out):
            return out
    raise TypeError(f"{name}={arg!r}: expected sequence of {item_type.__name__}")
