# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import sys

import matplotlib

from tqdm import tqdm, trange


def use_inline_backend() -> bool:
    "Check whether matplotlib is using a backend that renders figures into a notebook (Jupyter inline, ipympl, marimo, ...) rather than a GUI window"
    # matplotlib.get_backend() also initializes backend
    backend = dict.__getitem__(matplotlib.rcParams, 'backend')
    if not isinstance(backend, str):
        return False
    # Only known notebook module:// backends; other module:// backends (e.g., mplcairo, PyCharm's backend_interagg) are GUI backends
    return backend.endswith('inline') or backend == 'nbAgg' or backend.startswith(('module://matplotlib_inline', 'module://ipympl', 'module://marimo'))


def in_ipython() -> bool:
    "Check whether code is running in an IPython shell (which displays open pyplot figures at the end of each cell)"
    ipython = sys.modules.get('IPython')
    return ipython is not None and ipython.get_ipython() is not None


# import inline tqdm
if use_inline_backend():
    try:
        import ipywidgets as _  # noqa
    except ImportError:
        pass
    else:
        from tqdm.auto import tqdm, trange  # noqa
