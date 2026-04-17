# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
A state model with registered fields, constants, and dependent values.
"""
from __future__ import annotations

from collections import defaultdict
import difflib
from functools import cached_property, reduce
from itertools import chain, product
import operator
from collections.abc import Callable, Sequence
import traceback

import numpy as np

from .. import fmtxt
from .._config import tqdm_disable
from .._utils import as_sequence
from .._utils.com import Notifier, NotNotifier
from .._utils.notebooks import tqdm
from .configuration import check_names


class LayeredDict(dict):
    """Dictionary which can store and restore states"""

    def __init__(self):
        self._states = []
        dict.__init__(self)

    def __repr__(self):
        return ("<LayeredDict with %i stored states:\n"
                "%r>" % (len(self._states), dict.__repr__(self)))

    def get_stored(self, key, level, default=None):
        """Retrieve a field value from any level

        Parameters
        ----------
        key : str
            the field name (dictionary key).
        level : int
            The level from which to retrieve the value. -1 = the current level.
        """
        return self._states[level].get(key, default)

    def restore_state(self, state=-1, discard_tip=True):
        """Restore a previously stored state

        Parameters
        ----------
        state : int | dict
            Index of the state which to restore (specified as index into a
            list of stored states, i.e., negative values access recently
            stored states).
        discard_tip : bool
            Discard the relevant state after restoring it. All states stored
            later are discarded either way.

        See Also
        --------
        .get_stored(): Retrieve a stored value without losing stored states
        """
        if isinstance(state, int):
            index = state
            state = self._states[index]
            if discard_tip:
                del self._states[index:]
            elif index != -1:  # -1 + 1 = 0
                del self._states[index + 1:]
        elif not isinstance(state, dict):
            raise TypeError(f"state needs to be either int or dict, got {state!r}")

        self.clear()
        self.update(state)

    def store_state(self):
        "Store the current state"
        self._states.append(self.copy())


class _TempStateController:
    def __init__(self, state: StateModel):
        self.state = state

    def __enter__(self):
        self.state._store_state()

    def __exit__(self, exc_type, exc_value, traceback):
        self.state._restore_state()


class StateModel:
    """
    A state model with registered fields, constants, and dependent values.

    Notes
    -----
    Any subclass should make sure to call the ``._store_state()`` method at the
    end of initialization.
    """
    owner = None  # email address as string (for notification)
    _auto_debug = False  # in notification block
    defaults = {}

    def __init__(self, **state):
        # scaffold for state
        self._fields = LayeredDict()
        self._field_values = LayeredDict()
        self._terminal_fields = []
        self._repr_kwargs = []
        self._repr_kwargs_optional = []

        # scaffold for hooks
        self._eval_handlers = defaultdict(list)
        self._post_set_handlers = defaultdict(list)
        self._set_handlers = {}
        self._slave_fields = defaultdict(list)
        self._slave_handlers = {}

        # construct initial state: make all defaults available, then set as
        # many values as we can
        self._defaults = dict(self.defaults)
        self._defaults.update(state)

        if self.owner:
            task = self.__class__.__name__
            self.notification = Notifier(self.owner, task, self._crash_report, self._auto_debug)
        else:
            self.notification = NotNotifier()

    @property
    def state(self):
        return dict(self._fields)

    def __repr__(self):
        args = [repr(arg) for arg in self._repr_args()]
        kwargs = [(arg, self._fields[arg]) for arg in self._repr_kwargs]
        no_initial_state = len(self._fields._states) == 0
        for k in self._repr_kwargs_optional:
            v = self._fields[k]
            if no_initial_state or v != self._fields.get_stored(k, level=0):
                kwargs.append((k, v))
        args.extend(f'{k}={v!r}' for k, v in kwargs)
        return f"{self.__class__.__name__}({', '.join(args)})"

    def _repr_args(self) -> tuple[str, ...]:
        """Positional arguments to show in repr"""
        return ()

    def _bind_eval(self, key, handler):
        self._eval_handlers[key].append(handler)

    def _bind_post_set(self, key, handler):
        handlers = self._post_set_handlers[key]
        if handler not in handlers:
            handlers.append(handler)

    def _bind_set(self, key, handler):
        if key in self._set_handlers:
            raise KeyError(f"set-handler for {key!r} already set")
        self._set_handlers[key] = handler

    def _crash_report(self):
        out = []

        # try:
        #     source = inspect.getsource(self.__class__)
        # except Exception as e:
        #     source = "Failed to retrieve source:\n" + traceback.format_exc(e)
        # out.append(source)

        try:
            tree = str(self.show_state())
        except Exception as e:
            tree = "Failed to retrieve state:\n" + traceback.format_exc(e)
        out.append(tree)

        # package versions
        from .. import __version__
        import mne
        import scipy
        out.append(f"Eelbrain {__version__}\nmne-python {mne.__version__}\nSciPy {scipy.__version__}\nNumPy {np.__version__}")
        return out

    def _register_constant(self, key, value):
        value = self._defaults.get(key, value)
        if value is None:
            raise ValueError(f"The {key!r} field needs to be set as default")
        self._fields[key] = value

    def _register_field(
            self,
            key: str,
            values: Sequence[str] = None,
            default: str = None,
            set_handler: Callable = None,
            eval_handler: Callable = None,
            post_set_handler=None,
            depends_on: Sequence[str] = None,
            slave_handler: Callable = None,
            allow_empty: bool = False,
            repr: bool = None,
    ):
        """Register an iterable field

        Parameters
        ----------
        key : str
            Name of the field.
        values : None | sequence of str
            Possible values for this field, if known.
        default : None | str
            Set the default value (if None, the first element in values).
        set_handler : None | callable
            Function to call instead of updating the state value. The return
            value of the set_handler is sent to the post_set_handler.
        eval_handler : None | callable
            Function to use for evaluating a value before setting. Can be
            called without actually setting the value; any parameter changes
            need to be evaluated in post_set_handlers.
        post_set_handler : None | callable
            Function to call after the value is changed. Needs to be able to
            handle non-existing values for ``e.set(..., vmatch=False)`` calls.
        depends_on : str | sequence of str
            Slave fields: Fields in depends_on trigger change in ``key``.
        slave_handler : func
            Slave fields: Function that determines the new value of ``key``.
        allow_empty : bool
            Allow empty string in ``values``.
        repr : bool
            By default, fields are shown in ``repr`` if they are different from
            the value at initialization. Set to ``True`` to always show them
            (as long as there are at least 2 ``values``).
        """
        if key in self._fields:
            raise KeyError(f"Field already exists: {key!r}")

        if depends_on is not None:
            if set_handler is not None or eval_handler is not None or post_set_handler is not None:
                raise RuntimeError("Slave values can't have other handlers")
            elif slave_handler is None:
                raise RuntimeError("Slave value requires slave_handler")
            self._register_slave_field(key, depends_on, slave_handler)
            if default is None:
                default = slave_handler(self._fields)
        if set_handler is not None:
            self._bind_set(key, set_handler)
        if eval_handler is not None:
            self._bind_eval(key, eval_handler)
        if post_set_handler is not None:
            self._bind_post_set(key, post_set_handler)

        default = self._defaults.get(key, default)

        if values:
            values = tuple(values)
            check_names(values, key, allow_empty)
            if default is None:
                default = values[0]
            elif default not in values:
                raise ValueError(f"Default {default!r} for {key!r} not in values {values}")
            self._field_values[key] = values

        # repr
        if repr is True:
            if values and len(values) > 1:
                self._repr_kwargs.append(key)
        elif repr is None:
            if values and len(values) > 1:
                self._repr_kwargs_optional.append(key)
        elif repr is not False:
            raise TypeError(f"{repr=}")

        self._terminal_fields.append(key)
        self._fields[key] = ''
        if default is not None:
            self.set(**{key: default})

    def _register_slave_field(self, key, depends_on, handler):
        """Register a field that strictly depends on one or more other fields

        Parameters
        ----------
        key : str
            Field name.
        depends_on : str | sequence of str
            Fields that trigger change.
        handler : func
            Function that determines the new value.

        Notes
        -----
        Restrictions:

        - Slave fields can not have any other handlers
        - Slave fields can not depend on other slave fields
        """
        if isinstance(depends_on, str):
            depends_on = (depends_on,)
        for dep in depends_on:
            self._slave_fields[dep].append(key)
        self._slave_handlers[key] = handler
        self._fields[key] = handler(self._fields)

    def format(self, string: str, vmatch: bool = True, **kwargs) -> str:
        """Format a string with the current state values.

        Parameters
        ----------
        string
            Template string.
        vmatch
            For fields with known names, only allow existing field names.
        ...
            State parameters.

        Returns
        -------
        str
            ``string`` formatted with current state values.
        """
        self.set(match=vmatch, **kwargs)
        return string.format(**self._fields)

    def get(self, key, vmatch: bool = True, **state):
        if state:
            self.set(match=vmatch, **state)
        value = self._fields[key]
        if isinstance(value, str) and '{' in value:
            return value.format(**self._fields)
        return value

    def get_field_values(self, field, exclude=()):
        """Find values for a field taking into account exclusion

        Parameters
        ----------
        field : str
            Field for which to find values.
        exclude : list of str
            Exclude these values.
        """
        values = self._field_values[field]
        if isinstance(exclude, str):
            exclude = (exclude,)

        if exclude:
            values = [v for v in values if v not in exclude]
        else:
            values = list(values)

        return values

    def iter(self, fields, exclude=None, values=None, progress_bar=None, **constants):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : sequence | str
            Field(s) over which should be iterated.
        exclude : dict  {str: iterator over str}
            Exclude values from iteration (``{field: values_to_exclude}``).
        values : dict  {str: iterator over str}
            Fields with custom values to iterate over (instead of the
            corresponding field values) with {name: (sequence of values)}
            entries.
        progress_bar : str
            Message to show in the progress bar.
        ...
            Fields with constant values throughout the iteration.
        """
        if isinstance(fields, str):
            fields = (fields,)
            yield_str = True
        else:
            yield_str = False

        iter_fields = []
        for field in fields:
            if field in constants or field in iter_fields:
                continue
            if field not in self._fields:
                raise ValueError(f"{field!r}: not an iterable field")
            if field not in self._field_values:
                continue
            iter_fields.append(field)

        # check values and exclude
        if values:
            bad = set(values).difference(iter_fields)
            if bad:
                raise ValueError(f"{values=}: keys that are not iterated over ({', '.join(bad)})")
        else:
            values = {}
        if exclude:
            bad = set(exclude).difference(iter_fields)
            if bad:
                raise ValueError(f"{exclude=}: keys that are not iterated over ({', '.join(bad)})")
        else:
            exclude = {}

        # set constants (before .get_field_values() call)
        self.set(**constants)

        # gather values to iterate over
        v_lists = []
        for field in iter_fields:
            if field in values:
                v_lists.append(as_sequence(values[field]))
            else:
                exclude_ = exclude.get(field, None)
                v_lists.append(self.get_field_values(field, exclude_))

        if len(v_lists):
            # setup progress bar
            n = reduce(operator.mul, map(len, v_lists))
            if tqdm_disable() and progress_bar:
                disable = False
                if progress_bar is True:
                    progress_bar = ' '.join(iter_fields)
                elif not isinstance(progress_bar, str):
                    raise TypeError(f"{progress_bar=}")
            else:
                disable = True
            # iteration
            with self._temporary_state:
                for v_list in tqdm(product(*v_lists), progress_bar, n, disable=disable):
                    self._restore_state(discard_tip=False)
                    self.set(**dict(zip(iter_fields, v_list)))
                    if yield_str:
                        yield self.get(fields[0])
                    else:
                        yield tuple([self.get(f) for f in fields])
        else:
            yield ()

    def _restore_state(self, state=-1, discard_tip=True):
        """Restore a previously stored state

        Parameters
        ----------
        state : int
            Index of the state which to restore (specified as index into a
            list of stored states, i.e., negative values access recently
            stored states).
        discard_tip : bool
            Discard the relevant state after restoring it. All states stored
            later are discarded either way.
        """
        if isinstance(state, int):
            s1 = s2 = state
        else:
            s1, s2 = state
        self._fields.restore_state(s1, discard_tip)
        self._field_values.restore_state(s2, discard_tip)

    def reset(self):
        """Reset all field values to the state at initialization

        This function can be used in cases where the same Pipeline instance
        is used to perform multiple independent operations, where parameters set
        during one operation should not affect the next operation.
        """
        self._restore_state(0, False)

    def set(self, match=True, **state):
        """Set the value of one or more fields.

        Parameters
        ----------
        match : bool
            For fields with pre-defined values, only allow valid values (default
            ``True``).
        ... :
            Fields and values to set. Invalid fields raise a KeyError. Unless
            match == False, Invalid values raise a ValueError.
        """
        if not state:
            return

        handled_state = {}  # fields with special set handlers
        for k in list(state):
            v = state[k]
            if k not in self._fields:
                raise TypeError(f"{k}={v!r}: No field named {k!r}")
            elif v is None:
                state.pop(k)
                continue
            elif k in self._set_handlers:
                handled_state[k] = self._set_handlers[k](state.pop(k))
                continue
            elif not isinstance(v, str):
                raise TypeError(f"{k}={v!r}: Values have to be strings")
            # eval values
            eval_handlers = self._eval_handlers[k]
            if eval_handlers:
                for handler in eval_handlers:
                    try:
                        v = handler(v)
                    except ValueError:
                        if match:
                            raise
                    if not isinstance(v, str):
                        raise RuntimeError(f"Invalid conversion from handler {handler}: {k}={v!r}")
                    state[k] = v
            elif match and k in self._field_values and v not in self._field_values[k]:
                matches = difflib.get_close_matches(v, self._field_values[k], 1)
                if matches:
                    alt = f"Did you mean {matches[0]!r}? "
                else:
                    alt = ''
                raise ValueError(f"{k}={v!r}. {alt}To see all valid values use e.show_fields(); To set a non-existent value, use e.set({k}={v!r}, match=False).")

        self._fields.update(state)

        # fields depending on changes in other fields
        slave_state = {}
        for state_key in set(state).union(handled_state).intersection(self._slave_fields):
            for slave_key in self._slave_fields[state_key]:
                if slave_key not in slave_state:
                    v = self._slave_handlers[slave_key](self._fields)
                    if v is not None:
                        slave_state[slave_key] = v
        self._fields.update(slave_state)

        # call post_set handlers
        for k, v in chain(state.items(), handled_state.items(), slave_state.items()):
            for handler in self._post_set_handlers[k]:
                handler(k, v)

    def show_fields(self, str_out: bool = False) -> str | None:
        """A table for all iterable fields and their values.

        Parameters
        ----------
        str_out
            Return the table as a string (instead of printing it).
        """
        lines = []
        for key in self._field_values:
            values = list(self._field_values[key])
            line = f'{key}:'
            head_len = len(line) + 1
            while values:
                v = repr(values.pop(0))
                if values:
                    v += ','
                if len(v) < 80 - head_len:
                    line += ' ' + v
                else:
                    lines.append(line)
                    line = ' ' * head_len + v

                if not values:
                    lines.append(line)

        table = '\n'.join(lines)
        if str_out:
            return table
        else:
            print(table)

    def show_state(self, temp=None, empty=False, hide=()):
        """List field values.

        Parameters
        ----------
        temp : None | str | sequence of str
            Only show the specified field or fields.
        empty : bool
            Show empty variables (items whose value is the empty string '').
        hide : collection of str
            State variables to hide.

        Returns
        -------
        state : Table
            Table of field values.
        """
        table = fmtxt.Table('lll')
        table.cells('Key', '*', 'Value')
        table.caption('*: Value is modified from initialization state.')
        table.midrule()

        if temp is None:
            keys = chain(self._repr_kwargs, self._repr_kwargs_optional)
        elif isinstance(temp, str):
            keys = (temp,)
        else:
            keys = temp

        for k in sorted(keys):
            if k in hide:
                continue

            v = self._fields[k]
            if v != self._fields.get_stored(k, level=0):
                mod = '*'
            else:
                mod = ''

            if empty or mod or v:
                table.cells(k, mod, repr(v))

        return table

    def _store_state(self):
        """Store the current state

        See also
        --------
        ._restore_state() : restore a previously stored state
        """
        self._fields.store_state()
        self._field_values.store_state()

    @cached_property
    def _temporary_state(self):
        return _TempStateController(self)
