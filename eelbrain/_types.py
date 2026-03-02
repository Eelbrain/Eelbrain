from pathlib import Path
from collections.abc import Sequence


# https://matplotlib.org/stable/users/explain/colors/colors.html
ColorArg = str | Sequence[float] | tuple[str, float]
PathArg = Path | str
