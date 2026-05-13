# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Parse contrast expressions

AST
===

Nested tuple composed of:
- comparison:  ``('comp', cell_1, cell_0)``
- Unary functions:  ``('ufunc', func, arg)``
- Binary functions:  ``('bfunc', func, [arg1, arg2])``
- Array functions:  ``('afunc', func, [arg1, arg2, ...])``
where ``arg1`` etc. are in turn comparisons and functions.


Grammar
=======

Terminals
---------

- name
- comparator:  <, >
- operator:  +, -
- , ( ) |


Rules
-----

- SPM -> cell comparator cell
- SPM -> ufunc ( SPM )
- SPM -> bfunc ( SPM , SPM )
- SPM -> nfunc ( args )
- SPM -> SPM operator SPM

- args -> SPM
- args -> SPM , args

- cell -> name
- cell -> name | cell


"""
import re
# from token import LPAR, RPAR, NAME, COMMA, GREATER, LESS, VBAR

import numpy as np

SYNTAX_CHARS = '()|<>,+-'

# array functions:  work on array, take axis argument
np_afuncs = {'min': np.min,
             'max': np.max,
             'sum': np.sum}
# binary functions:  work on two arrays
np_bfuncs = {'subtract': np.subtract,
             'add': np.add}
# unary functions:  work on a single array
np_ufuncs = {'abs': np.abs,
             'negative': np.negative}

FUNCS = {name: (kind, func) for funcs, kind in
         ((np_afuncs, 'afunc'), (np_bfuncs, 'bfunc'), (np_ufuncs, 'ufunc'))
         for name, func in funcs.items()}
OP_FUNCS = {'-': np.subtract, '+': np.add}


RE_NAME = re.compile(r"[*\w\d ]*[*\w\d]")


class Streamer:

    def __init__(self, expression):
        tokens, indexes = tokenize(expression)
        assert len(indexes) == len(tokens)
        self._expression = expression
        self._tokens = tokens
        self._indexes = indexes
        self._i = -1
        self._imax = len(tokens) - 1
        self.eof = False

    @property
    def current(self):
        return self._tokens[self._i]

    def __iter__(self):
        return self

    def __next__(self, increment=1):
        self._i += increment
        if self._i >= self._imax:
            if self._i == self._imax:
                self.eof = True
            else:
                raise StopIteration
        return self._tokens[self._i]

    def lookahead(self, increment=1):
        i = self._i + increment
        if i > self._imax:
            return None
        else:
            return self._tokens[i]

    def error(self, desc):
        raise SyntaxError(desc, (None, 0, self._indexes[self._i] + 1,
                                 self._expression))


def tokenize(expression):
    i = 0
    imax = len(expression)
    tokens = []
    indexes = []
    while i < imax:
        if expression[i].isspace():
            i += 1
            continue
        elif expression[i] in SYNTAX_CHARS:
            tokens.append(expression[i])
            indexes.append(i)
            i += 1
            continue
        m = RE_NAME.match(expression, i)
        if m:
            tokens.append(m.group())
            indexes.append(i)
            i = m.end()
            continue
        raise SyntaxError("Invalid contrast expression syntax",
                          (None, 0, i + 1, expression))
    return tokens, indexes


def parse_spm(streamer, process_op=True):
    # parse function
    lookahead = streamer.lookahead()
    if lookahead in FUNCS and streamer.lookahead(2) == '(':
        kind, func = FUNCS[next(streamer)]
        next(streamer)
        args = parse_args(streamer)
        if kind == 'ufunc':
            if len(args) != 1:
                streamer.error(f"{func.__name__} function needs exactly 1 argument")
            args = args[0]
        elif kind == 'bfunc':
            if len(args) != 2:
                streamer.error(f"{func.__name__} function needs exactly 2 arguments")
        elif len(args) < 2:
            streamer.error(f"{func.__name__} function needs at least 2 arguments")
        return kind, func, args
    elif lookahead == '(':
        next(streamer)
        out = parse_spm(streamer)
        if next(streamer) != ')':
            streamer.error("Expected )")
    else:
        # beginning of cell comparator cell
        cell_1 = parse_cell(streamer)
        comp = next(streamer)
        if comp not in '<>':
            streamer.error("Expected '<' or '>'")
        cell_2 = parse_cell(streamer)
        if comp == '<':
            cell_1, cell_2 = cell_2, cell_1
        out = ('comp', cell_1, cell_2)

    if not process_op:
        return out
    # look for operator
    while (not streamer.eof) and streamer.lookahead() not in ',)':
        op = next(streamer)
        if op in OP_FUNCS:
            arg = parse_spm(streamer, False)
            out = ('bfunc', OP_FUNCS[op], (out, arg))
        else:
            streamer.error("Expected ), + or -")
    return out


def parse_cell(streamer):
    cell = [next(streamer)]
    while streamer.lookahead() == '|':
        next(streamer)
        cell.append(next(streamer))
    if len(cell) == 1:
        return cell[0]
    else:
        return tuple(cell)


def parse_args(streamer):
    args = [parse_spm(streamer)]
    delim = next(streamer)
    while delim == ',':
        args.append(parse_spm(streamer))
        delim = next(streamer)
    if delim != ')':
        streamer.error("Expected ')'")
    return tuple(args)


def parse(expression):
    streamer = Streamer(expression)
    return parse_spm(streamer)
