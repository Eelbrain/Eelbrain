# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import matplotlib

from tqdm import tqdm, trange


def use_inline_backend():
    "Check whether matplotlib is using an inline backend, e.g. for notebooks"
    # matplotlib.get_backend() also initializes backend
    backend = dict.__getitem__(matplotlib.rcParams, 'backend')
    return isinstance(backend, str) and (backend.endswith('inline') or backend == 'nbAgg')


# import inline tqdm
if use_inline_backend():
    try:
        import ipywidgets as _  # noqa
    except ImportError:
        pass
    else:
        from tqdm.auto import tqdm, trange  # noqa
