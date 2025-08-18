# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"A few basic operations needed throughout Eelbrain"
from collections import defaultdict
from dataclasses import dataclass, fields
import functools
import inspect
import logging
import re
from typing import Dict
from warnings import warn

from .notebooks import tqdm


LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def as_list(item):
    if isinstance(item, list):
        return item
    else:
        return [item]


def as_sequence(items, item_type=str):
    if isinstance(items, item_type):
        return items,
    return items


def ask(
        message: str,
        options: Dict[str, str],
        allow_empty: bool = False,
        help: str = None,
        default: str = '',
) -> str:
    """Ask user for input

    Parameters
    ----------
    message
        Message.
    options
        ``{command: description}`` mapping.
    allow_empty
        Allow empty string as command.
    help
        If provided, add a "help" option that prints ``help``.
    default
        Default answer; implies ``allow_empty``, but will substitute ``default``
        when user input is the empty string.

    Returns
    -------
    command : str
        The user command.
    """
    options = dict(options)
    if help is not None:
        assert 'help' not in options
        options['help'] = 'display help'
    if default:
        options[default] += ' (default)'
        allow_empty = True
    print(message)
    print('---')
    print('\n'.join(f'{k}:  {v}' for k, v in options.items()))
    while True:
        command = input(" > ") or default
        if command in options or (allow_empty and not command):
            if help is not None and command == 'help':
                print(help)
            else:
                return command
        else:
            print(f"Invalid entry - type one of ({', '.join(options)})")


def deprecated(version, replacement):
    """Decorator to deprecate functions and methods

    Also handles docstring of the deprecated function.

    Parameters
    ----------
    version : str
        Version in which the feature will be removed.
    replacement : callable | str
        Either a verbal description, or a pointer to the direct replacement
        function which takes the same arguments. In the latter case, the
        replacement is automatically called and the deprecated function is not
        used anymore.
    """
    def dec(func):
        msg = f"{func.__name__} is deprecated and will be removed in version {version}"
        if isinstance(replacement, str):
            msg += '; ' + replacement
            call_func = func
        elif replacement is not None:
            msg += f"; use {replacement.__name__} instead"
            call_func = replacement
        else:
            raise TypeError(f"{replacement=}")
        func.__doc__ = msg

        @functools.wraps(func)
        def new(*args, **kwargs):
            warn(msg, DeprecationWarning)
            return call_func(*args, **kwargs)

        return new
    return dec


def _deprecated_alias(alias, for_, version):
    """Create a deprecated alias

    Parameters
    ----------
    alias : str
        Name of the alias.
    for_ : callable
        New function (replacement).
    version : str
        Version in which the feature will be removed.
    """
    @functools.wraps(for_)
    def new(*args, **kwargs):
        warn(f"{alias} has been renamed to {for_.__name__}. {alias} will be removed in version {version}", DeprecationWarning)
        return for_(*args, **kwargs)
    return new


def deprecated_attribute(version, class_name, replacement):
    if not isinstance(replacement, str):
        raise TypeError("replacement=%r" % (replacement,))

    class Dec:

        def __init__(self, meth):
            self._meth = meth
            self.__name__ = meth.__name__
            self._msg = (
                'The %s.%s attribute is deprecated and will be removed in '
                'version %s, use %s.%s instead.' %
                (class_name, meth.__name__, version, class_name, replacement))

        def __get__(self, obj, klass=None):
            if obj is None:
                return None
            warn(self._msg, DeprecationWarning)
            return getattr(obj, replacement)

    return Dec


def deprecate_kwarg(
        old: str,
        new: str,
        since: str = None,
        remove_in: str = None,
):
    """
    Decorator factory to deprecate a renamed keyword argument.

    Parameters
    ----------
    old
        The deprecated keyword name that used to be accepted.
    new
        The new keyword name the function now uses.
    since
        Version/date when the deprecation started (for the warning message).
    remove_in
        Version/date when support will be removed (for the warning message).

    Notes
    -----
    - If both `old` and `new` are provided in a call, a TypeError is raised.
    - If `old` is used, it is moved to `new` and a DeprecationWarning is emitted.
    - Positional arguments are unaffected (this handles keyword renames).
    """
    def _decorator(func):
        sig = inspect.signature(func)
        if new not in sig.parameters:
            raise ValueError(f"{new=}: {func.__name__!r} has no parameter named {new!r}")

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{func.__name__} received both {old!r} (deprecated) and {new!r}. Use only {new!r}.")
                # Build a helpful warning message.
                parts = [f"Argument {old!r} is deprecated; use {new!r} instead."]
                if since:
                    parts.append(f"Deprecated since eelbrain {since}.")
                if remove_in:
                    parts.append(f"Support will be removed in eelbrain {remove_in}.")
                warn(" ".join(parts), DeprecationWarning, stacklevel=2)

                # Move value from old -> new
                kwargs[new] = kwargs.pop(old)

            # Let Python handle defaults/type errors as usual
            bound = sig.bind_partial(*args, **kwargs)
            return func(*bound.args, **bound.kwargs)

        return _wrapper
    return _decorator


def deprecate_ds_arg(func):
    """Backwards compatibility and deprecation for the `ds` parameter (renamed to `data`)"""
    @functools.wraps(func)
    def new(*args, ds=None, **kwargs):
        if ds is not None:
            warn("The `ds` argument has been renamed to `data`. `ds` will stop working in Eelbrain 0.41.", DeprecationWarning)
            kwargs['data'] = ds
        return func(*args, **kwargs)
    return new


def log_level(arg):
    """Convert string to logging module constant"""
    if isinstance(arg, int):
        return arg
    elif isinstance(arg, str):
        try:
            return LOG_LEVELS[arg.upper()]
        except KeyError:
            raise ValueError("Invalid log level: %s. mus be one of %s" %
                             (arg, ', '.join(LOG_LEVELS)))
    else:
        raise TypeError("Invalid log level: %s. need int or str." % repr(arg))


def set_log_level(level, logger_name='eelbrain'):
    """Set the minimum level of messages to be logged

    Parameters
    ----------
    level : str | int
        Level as string (debug, info, warning, error, critical) or
        corresponding constant from the logging module.
    logger_name : str
        Name of the logger for which to set the logging level. The default is
        the Eelbrain logger.
    """
    logging.getLogger(logger_name).setLevel(log_level(level))


class ScreenHandler(logging.StreamHandler):
    "Log handler compatible with TQDM"

    def __init__(self, formatter=None):
        logging.StreamHandler.__init__(self)
        if formatter is None:
            formatter = logging.Formatter("%(levelname)-8s:  %(message)s")
        self.setFormatter(formatter)

    def emit(self, record):
        tqdm.write(self.format(record))


def intervals(seq, first=None):
    """Iterate over each successive pair in a sequence.

    Examples
    --------
    >>> for i in intervals([1, 2, 3, 45]):
    ...     print(i)
    ...
    (1, 2)
    (2, 3)
    (3, 45)
    """
    iterator = iter(seq)
    if first is None:
        try:
            last = next(iterator)
        except StopIteration:
            return
    else:
        last = first

    for item in iterator:
        yield last, item
        last = item


@dataclass
class PickleableDataClass:

    def __getstate__(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __setstate__(self, state: dict):
        self.__init__(**state)


def _natural_keys(text):
    "Sorting key for natural sorting"
    # after http://stackoverflow.com/a/5967539/166700
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def natsorted(seq):
    return sorted(seq, key=_natural_keys)


def n_decimals(number):
    "Number of meaningful decimals in ``number``, at least 1"
    s = str(number)
    if '.' in s:
        return s[::-1].index('.')
    else:
        return 1


class keydefaultdict(defaultdict):  # noqa
    "http://stackoverflow.com/a/2912455/166700"
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret
