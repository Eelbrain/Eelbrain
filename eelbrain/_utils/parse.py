# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import ast


FLOAT_PATTERN = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
POS_FLOAT_PATTERN = r"^[+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
INT_PATTERN = r"^-?\d+$"
POS_INT_PATTERN = r"^0*[1-9]\d*$"
# matches float as well as NaN:
FLOAT_NAN_PATTERN = r"^\s*(([Nn][Aa][Nn])|([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?))\s*$"
PERCENT_PATTERN = r"^\s*(([Nn][Aa][Nn])|([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?))\s*%\S*$"


def find_variables(expr: str) -> set[str]:
    """Find the variables participating in an expressions

    Returns
    -------
    variables
        Every name occurring in ``expr``. Whether a name is read from the data or
        resolved by :meth:`Dataset.eval` itself (``abs``, ``numpy``, ...) can not be
        decided from the expression alone, and is settled against the data by
        :meth:`~eelbrain._experiment.variable_def.Variables.resolve`.
    """
    try:
        st = ast.parse(expr)
    except SyntaxError as error:
        raise ValueError(f"Invalid expression: {expr!r} ({error})")
    return {n.id for n in ast.walk(st) if isinstance(n, ast.Name)}
