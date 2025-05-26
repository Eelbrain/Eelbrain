#  Utilities working with (and needing to import) plots
from itertools import chain
import os
from typing import Dict, List, Tuple, Union

import numpy
from tqdm import tqdm

from .._types import PathArg
from .._utils import ui
from ._base import TimeSlicer


def save_movie(
        figures: Union[List[List[TimeSlicer]], Dict[Tuple[int, int], TimeSlicer]],
        filename: PathArg = None,
        time_dilation: float = 4,
        tstart: float = None,
        tstop: float = None,
        **kwargs,
):
    """Save a movie combining multiple figures with moving time axes

    Parameters
    ----------
    figures
        Either a nested list of figures (each inner list is a row of plots)
        or a dictionary specifying placement as ``{(xpos, ypos): figure}``.
    filename
        Filename for the movie (default will open a GUI for selecting the
        filename).
    time_dilation
        Factor by which to stretch time (default 4). Time dilation is
        controlled through the frame-rate; if the ``fps`` keyword argument
        is specified, ``time_dilation`` is ignored.
    tstart
        Time axis start (default is the earliest time in ``figures``).
    tstop
        Time axis stop (default includes the latest time in ``figures``).
    ...
        :func:`imageio.mimwrite` parmeters.

    Notes
    -----
    Might work best with::

        configure(prompt_toolkit=False)
    """
    import imageio
    from PIL import Image

    if filename is None:
        filename = ui.ask_saveas("Save movie...", "Save movie as...", [('Movie (*.mov)', '*.mov')])
        if not filename:
            return
    else:
        filename = os.path.expanduser(filename)

    # Interpret figures parameter
    if isinstance(figures, dict):
        figure_list = list(figures.values())
        coords = list(figures)
    elif isinstance(figures, (list, tuple)):
        if not isinstance(figures[0], (list, tuple)):
            figures = [figures]
        figure_list = list(chain.from_iterable(figures))
        coords = []
    else:
        raise TypeError(f"{figures=}")

    # Check time dimensions
    time_dims = list(filter(None, (getattr(f, '_time_dim', None) for f in figure_list)))
    if tstart is None:
        tstart = min(dim.tmin for dim in time_dims)
    if tstop is None:
        tstop = max(dim.tstop for dim in time_dims)
    if 'fps' in kwargs:
        tstep = time_dilation / kwargs['fps']
    else:
        tstep = min(dim.tstep for dim in time_dims)
        kwargs['fps'] = 1. / tstep / time_dilation

    # Generate movie
    times = numpy.arange(tstart, tstop, tstep)
    ims = []
    x_max = y_max = None
    for t in tqdm(times, "Frames", len(times)):
        t_ims = []
        for fig in figure_list:
            fig.set_time(t)
            f_im = Image.fromarray(fig._im_array(), 'RGBA')
            t_ims.append(f_im)

        # Infer positions
        if not coords:
            i = y = 0
            for row in figures:
                dy = 0
                x = 0
                for fig in row:
                    im = t_ims[i]
                    coords.append((x, y))
                    x += im.size[0]
                    dy = max(dy, im.size[1])
                    i += 1
                y += dy

        # Infer movie image dimensions
        if x_max is None:
            x_max = max(x + im.size[0] for (x, y), im in zip(coords, t_ims))
            y_max = max(y + im.size[1] for (x, y), im in zip(coords, t_ims))

        # Generate image
        im_buf = Image.new('RGBA', (x_max, y_max), (1, 1, 1, 0))
        for (x, y), im in zip(coords, t_ims):
            im_buf.paste(im, (x, y))

        # Store image
        ims.append(numpy.array(im_buf))

    # Save movie
    imageio.mimwrite(filename, ims, **kwargs)
