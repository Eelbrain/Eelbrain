# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Configure Eelbrain"""
import logging
import multiprocessing
import os
import sys
from typing import Any, Dict, Literal, Union

from matplotlib.colors import to_rgb

from ._utils import IS_OSX, IS_WINDOWS, ScreenHandler


SUPPRESS_WARNINGS = True
CONFIG: Dict[str, Any] = {
    'n_workers': multiprocessing.cpu_count(),
    'eelbrain': True,
    'autorun': None,
    'show': True,
    'format': 'svg',
    'figure_background': 'white',
    'prompt_toolkit': 'wx' if IS_WINDOWS else 'eelbrain',
    'animate': True,
    'nice': 0,
    'tqdm': False,  # disable=CONFIG['tqdm']
    'log': False,
}

# Python 3.8 switched default to spawn, which makes pytest hang  (https://docs.python.org/3/whatsnew/3.8.html#multiprocessing)
if sys.version_info.minor >= 8 and IS_OSX:
    method = 'fork'
else:
    method = None
mpc = multiprocessing.get_context(method)


def configure(
        n_workers: Union[bool, int] = None,
        frame: bool = None,
        autorun: bool = None,
        show: bool = None,
        format: str = None,
        figure_background: Any = None,
        prompt_toolkit: Literal[False, 'eelbrain', 'wx'] = None,
        animate: bool = None,
        nice: int = None,
        tqdm: bool = None,
        log: bool = None,
):
    """Set basic configuration parameters for the current session

    Parameters
    ----------
    n_workers
        Number of worker processes to use in multiprocessing enabled
        computations. ``False`` to disable multiprocessing. ``True`` (default)
        to use as many processes as cores are available. Negative numbers to use
        all but n available CPUs.
    frame
        Open figures in the Eelbrain application. This provides additional
        functionality such as copying a figure to the clipboard. If False, open
        figures as normal matplotlib figures.
    autorun
        When a figure is created, automatically enter the GUI mainloop. By
        default, this is True when the figure is created in interactive mode
        but False when the figure is created in a script (in order to run the
        GUI at a specific point in a script, call :func:`eelbrain.gui.run`).
    show
        Show plots on the screen when they're created (disable this to create
        plots and save them without showing them on the screen).
    format
        Default format for plots (for example "png", "svg", ...).
    figure_background : bool | matplotlib color
        While :mod:`matplotlib` uses a gray figure background by default,
        Eelbrain uses white. Set this parameter to ``False`` to use the default
        from :attr:`matplotlib.rcParams`, or set it to a valid matplotblib
        color value to use an arbitrary color. ``True`` to revert to the default
        white.
    prompt_toolkit
        In IPython 5, prompt_toolkit allows running the GUI main loop in
        parallel to the Terminal, meaning that the IPython terminal and GUI
        windows can be used without explicitly switching between Terminal and
        GUI. Set to
        ``'eelbrain'`` to use the Eelbrain-specific GUI loop (default);
        ``'wx'`` to use iPython's GUI loop;
        ``False`` to block the terminal instead (can be more stable when using
        GUIs).
    animate
        Animate plot navigation (default True).
    nice : int [-20, 19]
        Scheduling priority for muliprocessing (larger number yields more to
        other processes; negative numbers require root privileges).
    tqdm
        Enable or disable :mod:`tqdm` progress bars.
    log
        Enable logging (for debugging Eelbrain).
    """
    # don't change values before raising an error
    logger = logging.getLogger('Eelbrain')
    new: Dict[str, Any] = {}
    if n_workers is not None:
        if n_workers is True:
            new['n_workers'] = multiprocessing.cpu_count()
        elif n_workers is False:
            new['n_workers'] = 0
        elif isinstance(n_workers, int):
            cpu_count = multiprocessing.cpu_count()
            if n_workers < 0:
                if cpu_count + n_workers < 1:
                    raise ValueError(f"{n_workers=}, but only {cpu_count} CPUs are available")
                new['n_workers'] = cpu_count + n_workers
            else:
                if n_workers > cpu_count:
                    logger.warning(f"Configure {n_workers=} with {cpu_count=}")
                new['n_workers'] = n_workers
        else:
            raise TypeError(f"{n_workers=}")
    if frame is not None:
        new['eelbrain'] = bool(frame)
    if autorun is not None:
        new['autorun'] = bool(autorun)
    if show is not None:
        new['show'] = bool(show)
    if format is not None:
        new['format'] = format.lower()
    if figure_background is not None:
        if figure_background is True:
            figure_background = 'white'
        elif figure_background is not False:
            to_rgb(figure_background)
        new['figure_background'] = figure_background
    if prompt_toolkit is not None:
        if prompt_toolkit is True:
            new['prompt_toolkit'] = 'eelbrain'
        elif prompt_toolkit in ('eelbrain', 'wx', False):
            new['prompt_toolkit'] = prompt_toolkit
        else:
            raise ValueError(f"{prompt_toolkit=}; needs to be 'wx', 'eelbrain' or False")
    if animate is not None:
        new['animate'] = bool(animate)
    if nice is not None:
        nice = int(nice)
        if not -20 <= nice < 20:
            raise ValueError(f"{nice=}; needs to be in range [-20, 19]")
        elif nice < 0 and not os.getuid() == 0:
            raise ValueError(f"{nice=}; values < 0 require root privileges")
        new['nice'] = nice
    if tqdm is not None:
        new['tqdm'] = not tqdm

    # logging
    if log is True and not CONFIG['log']:
        logger.setLevel(logging.DEBUG)
        handler = ScreenHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.debug("Enabling logger")
        new['log'] = handler
    elif log is False and CONFIG['log']:
        handler = CONFIG['log']
        logger = logging.getLogger('Eelbrain')
        logger.removeHandler(handler)
        new['log'] = False

    CONFIG.update(new)


def tqdm_disable():
    """disable parameter for TQDM"""
    return CONFIG['tqdm'] or bool(os.environ.get('TQDM_DISABLE'))
