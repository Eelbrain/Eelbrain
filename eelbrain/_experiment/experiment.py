# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import defaultdict
import difflib
from functools import cached_property, reduce
from glob import glob
from itertools import chain, product
import operator
import os
import re
import shutil
import subprocess
from time import localtime, strftime
from typing import Callable, List, Sequence
import traceback

import numpy as np

from .. import fmtxt
from .._config import CONFIG
from .._text import enumeration, n_of, plural
from .._utils import as_sequence, ask
from .._utils.com import Notifier, NotNotifier
from .._utils.notebooks import tqdm
from .definitions import check_names, compound


def _etree_expand(node, state):
    for tk, tv in node.items():
        if tk == '.':
            continue

        for k, v in state.items():
            name = '{%s}' % tk
            if str(v).startswith(name):
                tv[k] = {'.': v.replace(name, '')}
        if len(tv) > 1:
            _etree_expand(tv, state)


def _etree_node_repr(node, name, indent=0):
    head = ' ' * indent
    out = [(name, head + node['.'])]
    for k, v in node.items():
        if k == '.':
            continue

        out.extend(_etree_node_repr(v, k, indent=indent + 3))
    return out


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
            raise TypeError("state needs to be either int or dict, got %r" %
                            (state,))

        self.clear()
        self.update(state)

    def store_state(self):
        "Store the current state"
        self._states.append(self.copy())


class _TempStateController:
    def __init__(self, experiment):
        self.experiment = experiment

    def __enter__(self):
        self.experiment._store_state()

    def __exit__(self, exc_type, exc_value, traceback):
        self.experiment._restore_state()


class TreeModel:
    """
    A hierarchical collection of format strings and field values

    Notes
    -----
    Any subclass should make sure to call the ``._store_state()`` method at the
    end of initialization.
    """
    owner = None  # email address as string (for notification)
    _auto_debug = False  # in notification block
    _fmt_pattern = re.compile(r'\{([\w-]+)}')
    # a dictionary of static templates (i.e., templates that do not have any hooks)
    _templates = {}
    defaults = {}
    _repr_args = ()

    def __init__(self, **state):
        # scaffold for state
        self._fields = LayeredDict()
        self._field_values = LayeredDict()
        self._terminal_fields = []
        self._secondary_cache = defaultdict(tuple)  # secondary cache-files
        self._repr_kwargs = []
        self._repr_kwargs_optional = []

        # scaffold for hooks
        self._compound_members = {}
        self._compounds = defaultdict(list)
        self._eval_handlers = defaultdict(list)
        self._post_set_handlers = defaultdict(list)
        self._set_handlers = {}
        self._slave_fields = defaultdict(list)
        self._slave_handlers = {}

        # construct initial state: make all defaults available, then set as
        # many values as we can
        self._defaults = dict(self.defaults)
        self._defaults.update(state)
        for k, v in self._templates.items():
            if v is None or isinstance(v, str):
                self._register_constant(k, v)
            elif isinstance(v, tuple):
                self._register_field(k, v, v[0], allow_empty=True)
            else:
                raise TypeError(f"Invalid templates field value: {v!r}. Need None, tuple or string")

        if self.owner:
            task = self.__class__.__name__
            self.notification = Notifier(self.owner, task, self._crash_report, self._auto_debug)
        else:
            self.notification = NotNotifier()

    def __repr__(self):
        args = [f'{self._fields[arg]!r}' for arg in self._repr_args]
        kwargs = [(arg, self._fields[arg]) for arg in self._repr_kwargs]
        no_initial_state = len(self._fields._states) == 0
        for k in self._repr_kwargs_optional:
            v = self._fields[k]
            if no_initial_state or v != self._fields.get_stored(k, level=0):
                kwargs.append((k, v))
        args.extend(f'{k}={v!r}' for k, v in kwargs)
        return f"{self.__class__.__name__}({', '.join(args)})"

    def _bind_eval(self, key, handler):
        self._eval_handlers[key].append(handler)

    def _bind_post_set(self, key, handler):
        handlers = self._post_set_handlers[key]
        if handler not in handlers:
            handlers.append(handler)

    def _bind_set(self, key, handler):
        if key in self._set_handlers:
            raise KeyError("set-handler for %r already set" % key)
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

    def _find_missing_fields(self):
        """Check that all field names occurring in templates are valid entries

        Raises
        ------
        KeyError
            If any field names occurring in templates are not registered fields.
        """
        # find field names occurring in field values but not as fields
        missing = set()
        for temp in self._fields.values():
            for field in self._fmt_pattern.findall(temp):
                if field not in self._fields:
                    missing.add(field)
        if missing:
            raise KeyError(f"The following fields occur in templates but are undefined: {', '.join(sorted(missing))}")

    def _register_compound(self, key, elements):
        """Register a field that is composed out of other fields

        The compound always reflects ``' '.join(elements)`` including only
        elements that are not empty.

        Parameters
        ----------
        key : str
            The name of the compound field.
        elements : tuple of str
            The field names of the elements.
        """
        self._compound_members[key] = elements
        for e in elements:
            self._compounds[e].append(key)
            self._bind_post_set(e, self._update_compounds)
        self._fields[key] = None
        self._update_compound(key)

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
            raise KeyError("Field already exists: %r" % key)

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
        if key in self._repr_args:
            pass
        elif repr is True:
            if values and len(values) > 1:
                self._repr_kwargs.append(key)
        elif repr is None:
            if values and len(values) > 1:
                self._repr_kwargs_optional.append(key)
        elif repr is not False:
            raise TypeError(f"repr={repr!r}")

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

    def expand_template(self, temp, keep=()):
        """Expand all constant variables in a template

        Parameters
        ----------
        temp : str
            Template or name of the template which should be expanded.
        keep : container (implements __contains__)
            Names of the variables which should not be expanded.

        Returns
        -------
        formatted_temp : str
            Template with all variables replaced by their values, except
            variables which have entries in field_values or in ``keep``.
        """
        temp = self._fields.get(temp, temp)

        while True:
            stop = True
            for name in self._fmt_pattern.findall(temp):
                if (name in keep) or (self._field_values.get(name, False)):
                    pass
                else:
                    temp = temp.replace('{%s}' % name, self._fields[name])
                    stop = False

            if stop:
                break

        return temp

    def find_keys(self, temp, root=True):
        """Find all terminal field names that are relevant for a template.

        Parameters
        ----------
        temp : str
            Template (or field name) for which to find terminal field names.
        root : bool
            Include "root" if present (default True).

        Returns
        -------
        keys : list
            All terminal field names that are relevant for formatting ``temp``.
        """
        if temp in self._terminal_fields:
            return [temp]

        if temp in self._compound_members:
            temporary_keys = list(self._compound_members[temp])
        else:
            temp = self._fields.get(temp, temp)
            temporary_keys = self._fmt_pattern.findall(temp)

        keys = []
        while temporary_keys:
            key = temporary_keys.pop(0)
            if key == 'root':
                if root:
                    keys.append('root')
            elif key in self._terminal_fields:
                keys.append(key)
            else:
                keys.extend(self.find_keys(key, root))

        # remove duplicates
        return list(dict.fromkeys(keys))

    def format(self, string: str, vmatch: bool = True, **kwargs) -> str:
        """Format a string (i.e., replace any '{xxx}' fields with their values)

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
            The template temp formatted with current state values.
        """
        self.set(match=vmatch, **kwargs)

        while self._fmt_pattern.search(string):
            string = string.format(**self._fields)

        return string

    def get(self, temp, **state):
        return self.format('{%s}' % temp, **state)

    def _get_rel(self, temp, start):
        "Get the path of ``temp`` relative to ``start`` (both field names)"
        abs_ = self.get(temp)
        start_ = self.get(start)
        return os.path.relpath(abs_, start_)

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

        # find actual fields to iterate over:
        iter_fields = []
        for field in fields:
            for terminal_field in self.find_keys(field):
                if terminal_field in constants or terminal_field in iter_fields:
                    continue
                iter_fields.append(terminal_field)

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
            if CONFIG['tqdm'] and progress_bar:
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

    def iter_temp(self, temp, exclude=None, values={}, **constants):
        """
        Iterate through all paths conforming to a template given in ``temp``.

        Parameters
        ----------
        temp : str
            Name of a template in the MneExperiment.templates dictionary, or
            a path template with variables indicated as in ``'{var_name}'``
        """
        # if the name is an existing template, retrieve it
        temp = self.expand_template(temp, values.keys())

        # find variables for iteration
        variables = set(self._fmt_pattern.findall(temp))
        variables.difference_update(constants)

        for _ in self.iter(variables, exclude, values, **constants):
            path = temp.format(**self._fields)
            yield path

    def _partial(self, temp, skip=()):
        "Format a template while leaving some slots unfilled"
        skip = set(skip)
        fields = self._fields.copy()
        fields.update({k: '{%s}' % k for k in skip})
        string = '{%s}' % temp

        while set(self._fmt_pattern.findall(string)).difference(skip):
            string = string.format(**fields)

        return string

    def _copy_state(self) -> (dict, dict):
        """Copy of the state that can be used with ``._restore_state()``"""
        return self._fields.copy(), self._field_values.copy()

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

        This function can be used in cases where the same MneExperiment instance
        is used to perform multiple independent operations, where parameters set
        during one operation should not affect the next operation.
        """
        self._restore_state(0, False)

    def set(self, match=True, allow_asterisk=False, **state):
        """Set the value of one or more fields.

        Parameters
        ----------
        match : bool
            For fields with pre-defined values, only allow valid values (default
            ``True``).
        allow_asterisk : bool
            If a value contains ``'*'``, set the value without the normal value
            evaluation and checking mechanisms (default ``False``).
        ... :
            Fields and values to set. Invalid fields raise a KeyError. Unless
            match == False, Invalid values raise a ValueError.
        """
        if not state:
            return

        # expand compounds
        if state.pop('expand_compounds', True):
            for k in list(state):
                if k in self._compound_members:
                    fields = self._compound_members[k]
                    v = state.pop(k)
                    values = v.split(' ')
                    for i, field in enumerate(fields):
                        field_values = self._field_values[field]
                        vi = values[i] if len(values) > i else None
                        if vi in field_values:
                            continue
                        elif '' in field_values:
                            values.insert(i, '')
                        else:
                            raise ValueError(f"{k}={v!r}")
                    if len(values) != len(fields):
                        raise ValueError(f"{k}={v!r}")
                    state.update(zip(fields, values))

        handled_state = {}  # fields with special set handlers
        for k in list(state):
            v = state[k]
            if k not in self._fields:
                raise TypeError(f"{k}={v!r}: No template named {k!r}")
            elif v is None:
                state.pop(k)
                continue
            elif k in self._set_handlers:
                handled_state[k] = self._set_handlers[k](state.pop(k))
                continue
            elif not isinstance(v, str):
                raise TypeError(f"{k}={v!r}: Values have to be strings")
            elif '*' in v and allow_asterisk:
                continue
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

    def show_fields(self, str_out=False):
        """
        Generate a table for all iterable fields and ther values.

        Parameters
        ----------
        str_out : bool
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
        """List all top-level fields and their values

        (Top-level fields are fields whose values do not contain templates)

        Parameters
        ----------
        temp : None | str
            Only show variables relevant to this template.
        empty : bool
            Show empty variables (items whose value is the empty string '').
        hide : collection of str
            State variables to hide.

        Returns
        -------
        state : Table
            Table of (relevant) variables and their values.
        """
        table = fmtxt.Table('lll')
        table.cells('Key', '*', 'Value')
        table.caption('*: Value is modified from initialization state.')
        table.midrule()

        if temp is None:
            keys = chain(self._repr_kwargs, self._repr_kwargs_optional)
        else:
            keys = self.find_keys(temp)

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

    def show_tree(self, root='root', fields=None):
        """
        Print a tree of the filehierarchy implicit in the templates

        Parameters
        ----------
        root : str
            Name of the root template (e.g., 'besa-root').
        fields : list of str
            Which fields to include in the tree (default is all).
        """
        if fields is None:
            fields = self._fields
        else:
            # find all implied fields
            new_fields = set(fields)
            fields = {}
            while new_fields:
                k = new_fields.pop()
                fields[k] = v = self._fields[k]
                new_fields.update([f for f in self._fmt_pattern.findall(v) if f not in fields])

        tree = {'.': self.get(root)}
        root_temp = '{%s}' % root
        for k, v in fields.items():
            if str(v).startswith(root_temp):
                tree[k] = {'.': v.replace(root_temp, '')}
        _etree_expand(tree, fields)
        nodes = _etree_node_repr(tree, root)
        name_len = max(len(n) for n, _ in nodes)
        path_len = max(len(p) for _, p in nodes)
        pad = ' ' * (80 - name_len - path_len)
        print('\n'.join(n.ljust(name_len) + pad + p.ljust(path_len) for n, p in nodes))

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

    def _update_compound(self, key):
        items = [self.get(k) for k in self._compound_members[key]]
        self.set(**{key: compound(items)}, expand_compounds=False)

    def _update_compounds(self, key, _):
        for compound_ in self._compounds[key]:
            self._update_compound(compound_)


class FileTree(TreeModel):
    """:class:`TreeModel` subclass for a file system hierarchy"""
    _repr_args = ('root',)
    _safe_delete = 'root'  # directory from which to rm without warning

    def __init__(self, **state):
        TreeModel.__init__(self, **state)
        self._make_handlers = {}
        self._cache_handlers = {}
        self._register_field('root', eval_handler=self._eval_root)

    def _bind_cache(self, key, handler):
        """Bind a cache function to a ``*-file`` key

        The cache function is called every time the file name is retrieved and
        should recreate the file if it is outdated.

        The cache function can return the filename of the created file since
        it is called every time the specific file is requested. Note that this
        causes problems for ``glob()``.
        """
        if key in self._cache_handlers:
            raise RuntimeError(f"Cache handler for {key!r} already defined")
        elif key in self._make_handlers:
            raise RuntimeError(f"Already defined make handler for {key!r}")
        self._cache_handlers[key] = handler

    def _bind_make(self, key, handler):
        """Bind a make function to a ``*-file`` key

        The make function is called only when the file name is retrieved and
        the file does not exist.
        """
        if key in self._cache_handlers:
            raise RuntimeError(f"Already defined cache handler for {key!r}")
        elif key in self._make_handlers:
            raise RuntimeError(f"Make handler for {key!r} already defined")
        self._make_handlers[key] = handler

    @staticmethod
    def _eval_root(root):
        root = os.path.abspath(os.path.expanduser(root))
        if root != '':
            root = os.path.normpath(root)
        return root

    def get(
            self,
            temp: str,
            fmatch: bool = False,
            vmatch: bool = True,
            match: bool = True,
            mkdir: bool = False,
            make: bool = False,
            **state,
    ):
        """
        Retrieve a formatted template

        Parameters
        ----------
        temp
            Name of the requested template.
        fmatch
            "File-match": If the template contains asterisk (``*``), use ``glob`` to
            expand it. An IOError is raised if the pattern does not match
            exactly one file.
        vmatch
            "Value match": Require existence of the assigned value (only
            applies for fields with stored values).
        match
            Do any matching (i.e., ``match=False`` sets ``fmatch`` as well as
            ``vmatch`` to ``False``).
        mkdir
            If the directory containing the file does not exist, create it.
        make
            If a requested file does not exists, make it if possible.
        """
        if not match:
            fmatch = vmatch = False

        path = TreeModel.get(self, temp, vmatch=vmatch, **state)
        path = os.path.expanduser(path)

        # assert the presence of the file
        if fmatch and ('*' in path):
            paths = glob(path)
            if len(paths) == 0 and make and temp in self._make_handlers:
                self._make_handlers[temp]()
                paths = glob(path)

            if len(paths) == 1:
                path = paths[0]
            elif len(paths) > 1:
                raise IOError(f"More than one files match {path!r}: {paths}")
            else:
                raise IOError(f"No file found for {path!r}")

        # create the directory
        if mkdir:
            if temp.endswith('dir'):
                dirname = path
            else:
                dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                root = self.get('root')
                if root == '':
                    raise IOError("Prevented from creating directories because root is not set")
                elif os.path.exists(root):
                    os.makedirs(dirname)
                else:
                    raise IOError(f"Prevented from creating directories because root does not exist: {root!r}")

        # make the file
        if make:
            if temp in self._cache_handlers:
                path = self._cache_handlers[temp]() or path
            elif not os.path.exists(path):
                if temp in self._make_handlers:
                    with self._temporary_state:
                        self._make_handlers[temp]()
                elif temp.endswith('-dir'):
                    os.makedirs(path)
                else:
                    raise RuntimeError(f"No make handler for {temp!r}")

        return path

    def glob(self, temp, inclusive=False, **state) -> List[str]:
        """Find all files matching a certain pattern

        Parameters
        ----------
        temp : str
            Name of the path template for which to find files.
        inclusive : bool
            Treat all unspecified fields as ``*`` (default False).

        See Also
        --------
        copy : Copy files.
        move : Move files.
        rm : Delete files.

        Notes
        -----
        State parameters can include an asterisk ('*') to match multiple files.
        Uses :func:`glob.glob`.
        """
        pattern = self._glob_pattern(temp, inclusive, **state)
        return glob(pattern)

    def _glob_pattern(self, temp, inclusive=False, **state) -> str:
        if inclusive:
            for key in self._terminal_fields:
                if key in state or key == 'root':
                    continue
                elif key in self._field_values and len(self._field_values[key]) == 1:
                    continue
                state[key] = '*'
        with self._temporary_state:
            pattern = self.get(temp, allow_asterisk=True, **state)
        return pattern

    def _find_files_with_target(self, action, temp, dst_root, inclusive, overwrite, confirm, state):
        if dst_root is None:
            if 'root' not in state:
                raise TypeError("Need to specify at least one of root and dst_root")
            dst_root = self.get('root')
        if not os.path.exists(dst_root):
            raise IOError(f'Destination does not exist: {dst_root}')
        src_filenames = self.glob(temp, inclusive, **state)
        n = len(src_filenames)
        if n == 0:
            print("No files matching pattern.")
            return None, None

        root = self.get('root')
        errors = [filename for filename in src_filenames if not filename.startswith(root)]
        if errors:
            raise ValueError(f"{len(errors)} files are not located in the root directory ({errors[0]}, ...)")
        rel_filenames = {src: os.path.relpath(src, root) for src in src_filenames}
        dst_filenames = {src: os.path.join(dst_root, filename) for src, filename in rel_filenames.items()}
        if overwrite is not True:
            exist = [src for src, dst in dst_filenames.items() if os.path.exists(dst)]
            if exist:
                if overwrite is None:
                    raise ValueError(f"{len(exist)} of {n} files already exist; use overwrite parameter")
                elif overwrite is False:
                    if len(exist) == n:
                        print(f"All {n} files already exist; use overwrite=True to replace them")
                        return None, None
                    n -= len(exist)
                    for src in exist:
                        src_filenames.remove(src)
                else:
                    raise TypeError(f"overwrite={overwrite!r}")

        if not confirm:
            print(f"{action} {self.get('root')} -> {dst_root}:")
            for src in src_filenames:
                print("  " + rel_filenames[src])
            if input(f"{action} {n} files? (confirm with 'yes'): ") != 'yes':
                return None, None
        return src_filenames, [dst_filenames[src] for src in src_filenames]

    def copy(self, temp, dst_root=None, inclusive=False, confirm=False, overwrite=None, **state):
        """Copy files to a different root folder

        Parameters
        ----------
        temp : str
            Name of the path template for which to find files.
        dst_root : str
            Path to the root to which the files should be moved. If the target
            is the experiment's root directory, specify ``root`` as the source
            root and leave ``dst_root`` unspecified.
        inclusive : bool
            Treat all unspecified fields as ``*`` (default False).
        confirm : bool
            Skip asking for confirmation before copying the files.
        overwrite : bool
            ``True`` to overwrite target files if they already exist. ``False``
            to quietly keep exising files.

        See Also
        --------
        glob : Find all files matching a template.
        move : Move files.
        rm : Delete files.
        make_copy : Copy a file by substituting a field

        Notes
        -----
        State parameters can include an asterisk ('*') to match multiple files.
        """
        src_filenames, dst_filenames = self._find_files_with_target('Copy', temp, dst_root, inclusive, overwrite, confirm, state)
        if not src_filenames:
            return
        for src, dst in tqdm(zip(src_filenames, dst_filenames), "Copying", len(src_filenames)):
            dirname = os.path.dirname(dst)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

    def move(self, temp, dst_root=None, inclusive=False, confirm=False, overwrite=None, **state):
        """Move files to a different root folder

        Parameters
        ----------
        temp : str
            Name of the path template for which to find files.
        dst_root : str
            Path to the root to which the files should be moved. If the target
            is the experiment's root directory, specify ``root`` as the source
            root and leave ``dst_root`` unspecified.
        inclusive : bool
            Treat all unspecified fields as ``*`` (default False).
        confirm : bool
            Skip asking for confirmation before moving the files.
        overwrite : bool
            Overwrite target files if they already exist.

        See Also
        --------
        copy : Copy files.
        glob : Find all files matching a template.
        rm : Delete files.

        Notes
        -----
        State parameters can include an asterisk ('*') to match multiple files.
        """
        if overwrite is False:
            raise ValueError(f"{overwrite=}")
        src_filenames, dst_filenames = self._find_files_with_target('Move', temp, dst_root, inclusive, overwrite, confirm, state)
        if not src_filenames:
            return
        for src, dst in tqdm(zip(src_filenames, dst_filenames), "Moving", len(src_filenames)):
            dirname = os.path.dirname(dst)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            os.rename(src, dst)

    def show_file_status(
            self,
            temp,
            row,
            col: str = None,
            count: bool = True,
            present: str = 'time',
            absent: str = '-',
            **kwargs,
    ):
        """Compile a table about the existence of files

        Parameters
        ----------
        temp
            The name of the path template for the files to examine.
        row
            Field over which to alternate rows.
        col
            Field over which to alternate columns (default is a single column).
        count
            Add a column with a number for each line (default True).
        present
            String to display when a given file is present. ``'time'`` to use
            last modification date and time (default); ``'date'`` for date only.
        absent
            String to display when a given file is absent (default ``'-'``).
        ...
            ``self.iter()`` kwargs.
        """
        if col is None:
            col_v = (None,)
            ncol = 1
        else:
            col_v = self.get_field_values(col)
            ncol = len(col_v)

        # table header
        table = fmtxt.Table('r' * bool(count) + 'l' * (ncol + 1))
        if count:
            table.cell()
        table.cell(row)
        if col is None:
            table.cell(temp)
        else:
            for name in col_v:
                table.cell(name)
        table.midrule()

        # body
        for i, row_v in enumerate(self.iter(row, **kwargs)):
            if count:
                table.cell(i)
            table.cell(row_v)
            for v in col_v:
                if v is None:
                    path = self.get(temp)
                else:
                    path = self.get(temp, **{col: v})

                if os.path.exists(path):
                    if present == 'time':
                        r = strftime('%x %X', localtime(os.path.getmtime(path)))
                    elif present == 'date':
                        r = strftime('%x', localtime(os.path.getmtime(path)))
                    else:
                        r = present
                else:
                    r = absent
                table.cell(r)

        return table

    def show_file_status_mult(self, files, fields, count=True, present='X',
                              absent='-', **kwargs):
        """
        Compile a table about the existence of multiple files

        Parameters
        ----------
        files : str | list of str
            The names of the path templates whose existence to list.
        fields : str | list of str
            The names of the variables for which to list files (i.e., for each
            unique combination of ``fields``, list ``files``).
        count : bool
            Add a column with a number for each subject.
        present : str
            String to display when a given file is present.
        absent : str
            String to display when a given file is absent.

        Examples
        --------
        >>> e.show_file_status_mult(['raw-file', 'trans-file', 'fwd-file'],
        ... 'subject')
             Subject   Raw-file   Trans-file   Fwd-file
        -----------------------------------------------
         0   AD001     X          X            X
         1   AD002     X          X            X
         2   AD003     X          X            X
        ...
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        if not isinstance(fields, (list, tuple)):
            fields = [fields]

        ncol = (len(fields) + len(files))
        table = fmtxt.Table('r' * bool(count) + 'l' * ncol)
        if count:
            table.cell()
        for name in fields + files:
            table.cell(name.capitalize())
        table.midrule()

        for i, _ in enumerate(self.iter(fields, **kwargs)):
            if count:
                table.cell(i)

            for field in fields:
                table.cell(self.get(field))

            for temp in files:
                path = self.get(temp)
                if os.path.exists(path):
                    table.cell(present)
                else:
                    table.cell(absent)

        return table

    def show_in_finder(self, temp, **kwargs):
        "Reveal the file corresponding to the ``temp`` template in the Finder."
        fname = self.get(temp, **kwargs)
        subprocess.call(["open", "-R", fname])

    def rename(self, old, new, exclude=False):
        """Rename all files corresponding to a pattern (or template)

        Parameters
        ----------
        old : str
            Template for the files to be renamed. Can interpret '*', but will
            raise an error in cases where more than one file fit the pattern.
        new : str
            Template for the new names.

        Examples
        --------
        The following command will collect a specific file for each subject and
        place it in a common folder:

        >>> e.rename('info-file', '/some_other_place/{subject}_info.txt')
        """
        new = self.expand_template(new)
        files = []
        for old_name in self.iter_temp(old, exclude):
            if '*' in old_name:
                matches = glob(old_name)
                if len(matches) == 1:
                    old_name = matches[0]
                elif len(matches) > 1:
                    err = ("Several files fit the pattern %r" % old_name)
                    raise ValueError(err)

            if os.path.exists(old_name):
                new_name = self.format(new)
                files.append((old_name, new_name))

        if not files:
            print("No files found for %r" % old)
            return

        old_pf = os.path.commonprefix([pair[0] for pair in files])
        new_pf = os.path.commonprefix([pair[1] for pair in files])
        n_pf_old = len(old_pf)
        n_pf_new = len(new_pf)

        table = fmtxt.Table('lll')
        table.cells('Old', '', 'New')
        table.midrule()
        table.caption("%s -> %s" % (old_pf, new_pf))
        for old, new in files:
            table.cells(old[n_pf_old:], '->', new[n_pf_new:])

        print(table)

        msg = "Rename %s files (confirm with 'yes')? " % len(files)
        if input(msg) == 'yes':
            for old, new in files:
                dirname = os.path.dirname(new)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                os.rename(old, new)

    def rename_field(self, temp, field, old, new, exclude=False, **kwargs):
        """Change the value of one field in paths corresponding to a template

        Parameters
        ----------
        temp : str
            Template name.
        field : str
            Field to change.
        old : str
            Old value.
        new : str
            New value.
        kwargs :
            ``self.iter_temp`` arguments.
        """
        items = []  # (tag, src, dst)
        kwargs[field] = old
        dst_kwa = {field: new}
        for src in self.iter_temp(temp, exclude, ** kwargs):
            dst = self.get(temp, **dst_kwa)
            if os.path.exists(src):
                if os.path.exists(dst):
                    tag = 'o'
                else:
                    tag = ' '
            else:
                tag = 'm'
            items.append((tag, src, dst))

        src_prefix = os.path.commonprefix(tuple(item[1] for item in items))
        dst_prefix = os.path.commonprefix(tuple(item[2] for item in items))
        src_crop = len(src_prefix)
        dst_crop = len(dst_prefix)

        # print info
        if src_prefix == dst_prefix:
            lines = ['in ' + src_prefix, '']
        else:
            lines = [src_prefix, '->' + dst_prefix, '']

        for tag, src, dst in items:
            lines.append('%s %s -> %s' % (tag, src[src_crop:], dst[dst_crop:]))
        lines.append('')
        msg = 'Legend  m: source is missing;  o: will overwite a file'
        lines.append(msg)
        print('\n'.join(lines))
        rename = tuple(item for item in items if item[0] == ' ')
        if not rename:
            return

        msg = "Rename %i files (confirm with 'yes')? " % len(rename)
        if input(msg) != 'yes':
            return

        for _, src, dst in rename:
            os.rename(src, dst)
        print("Done")

    def rm(self, temp, inclusive=False, confirm=False, **constants):
        """Remove all files corresponding to a template

        Asks for confirmation before deleting anything. Uses glob, so
        individual templates can be set to '*'.

        Parameters
        ----------
        temp : str
            Name of the path template for which to find and delete files.
        inclusive : bool
            Treat all unspecified fields as ``*`` (default False).
        confirm : bool
            Confirm removal of the selected files. If False (default) the user
            is prompted for confirmation with a list of files; if True, the
            files are removed immediately.
        **others** :
            Set field values (values can be '*' to match all).

        See Also
        --------
        glob : Find all files matching a template.
        copy : Copy files
        move : Move files.
        """
        files = self.glob(temp, inclusive, **constants)
        secondary_files = []
        for stemp in self._secondary_cache[temp]:
            secondary_files.extend(self.glob(stemp, inclusive, **constants))

        if files or secondary_files:
            is_dir = [os.path.isdir(path) for path in files]
            # Confirm deletion
            if not confirm:
                n_dirs = sum(is_dir)
                n_files = len(files) - n_dirs
                desc = []
                if n_dirs:
                    desc.append(n_of(n_dirs, 'directory'))
                if n_files:
                    desc.append(n_of(n_files, 'file'))
                if secondary_files:
                    desc.append(n_of(len(secondary_files), 'secondary file'))
                info = f"Delete {enumeration(desc)}?"

                # Confirm if deleting files not in managed space
                safe_root = self.get(self._safe_delete)
                n_unsafe = len(files) - sum(path.startswith(safe_root) for path in files)
                if n_unsafe:
                    info += f"\n!\n! {plural('item', n_unsafe)} outside of {self._safe_delete}\n!"

                # prompt
                while not confirm:
                    command = ask(info, {'yes': 'delete files', 'no': "don't delete files (default)", 'show': 'list the files'}, allow_empty=True)
                    if command == 'yes':
                        confirm = True
                    elif command == 'show':
                        print("root: %s\n" % self.get('root'))
                        print('\n'.join(self._remove_root(files)))
                    elif command in ('no', ''):
                        return
                    else:
                        raise RuntimeError(f"command={command!r}")

            print('deleting...')
            dirs = (p for p, isdir in zip(files, is_dir) if isdir)
            files = (p for p, isdir in zip(files, is_dir) if not isdir)
            for path in dirs:
                shutil.rmtree(path)
            for path in chain(files, secondary_files):
                os.remove(path)
        else:
            print("No files found for %r" % temp)

    def _remove_root(self, paths):
        root = self.get('root')
        root_len = len(root)
        return (path[root_len:] if path.startswith(root) else path
                for path in paths)
