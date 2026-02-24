from pathlib import Path
from typing import Union
from collections.abc import Sequence


# https://matplotlib.org/stable/users/explain/colors/colors.html
ColorArg = Union[str, Sequence[float], tuple[str, float]]
PathArg = Union[Path, str]
