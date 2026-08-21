# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import ast
import builtins

from .._data_obj import EVAL_CONTEXT


FLOAT_PATTERN = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
POS_FLOAT_PATTERN = r"^[+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
INT_PATTERN = r"^-?\d+$"
# matches float as well as NaN:
FLOAT_NAN_PATTERN = r"^\s*(([Nn][Aa][Nn])|([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?))\s*$"
PERCENT_PATTERN = r"^\s*(([Nn][Aa][Nn])|([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?))\s*%\S*$"

EVAL_CONTEXT_NAMES = set(vars(builtins)).union(EVAL_CONTEXT)


def find_variables(expr: str) -> set[str]:
    """Find the variables participating in an expressions

    Returns
    -------
    variables
        Variables occurring in expr.
    """
    try:
        st = ast.parse(expr)
    except SyntaxError as error:
        raise ValueError(f"Invalid expression: {expr!r} ({error})")
    names = {n.id for n in ast.walk(st) if isinstance(n, ast.Name)}
    return names.difference(EVAL_CONTEXT_NAMES)
