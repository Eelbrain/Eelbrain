"""Eelbrain GUIs"""
from typing import Sequence, Tuple, Union

import mne

from ._data_obj import Dataset
from ._types import ColorArg, PathArg


def run(block=False):
    """Hand over command to the GUI (quit the GUI to return to the terminal)

    Parameters
    ----------
    block : bool
        Block the Terminal even if the GUI is capable of being run in parallel.
        Control returns to the Terminal when the user quits the GUI application.
        This is also useful to prevent plots from closing at the end of a
        script.
    """
    from ._wxgui import run

    run(block)


def select_components(
        path: PathArg,
        data: Union[Dataset, mne.io.BaseRaw],
        sysname: str = None,
        adjacency: Union[str, Sequence] = None,
        decim: int = None,
        debug: bool = False,
):
    """GUI for selecting ICA-components

    Parameters
    ----------
    path
        Path to the ICA file.
    data
        Data to use for displying component time course during source selection.
        Can be specified as :class:`mne.io.Raw` object with continuous data, or
        as :class:`Dataset` with epoched data (``data['epochs']`` should contain
        an :class:`mne.Epochs` object).
        Optionally, ``data['index']`` can provide labels to display for epochs
        (the default is ``range(n_epochs)``).
        Further :class:`Factor` can be used to plot condition averages.
    sysname
        Optional, to define sensor adjacency.
    adjacency
        Optional, to define sensor adjacency (see
        :func:`eelbrain.load.mne.sensor_dim`).
    decim
        Decimate the data for display (only applies when data is a ``Raw``
        object; default is to approximate 100 Hz samplingrate).

    Notes
    -----
    The ICA object does not need to be computed on the same data that is in
    ``ds``. For example, the ICA can be computed on a raw file but component
    selection done using the epochs that will be analyzed.

    .. note::
        If the terminal becomes unresponsive after closing the GUI, try
        disabling ``prompt_toolkit`` with :func:`configure`:
        ``eelbrain.configure(prompt_toolkit=False)``.
    """
    from ._wxgui.app import get_app
    from ._wxgui.select_components import TEST_MODE, Document, Frame, Model

    get_app()  # make sure app is created
    doc = Document(path, data, sysname, adjacency, decim=decim)
    model = Model(doc)
    frame = Frame(model)
    frame.Show()
    frame.Raise()
    if TEST_MODE:
        return frame
    run()
    if debug:
        return frame


def select_epochs(
        ds: Union[Dataset, mne.BaseEpochs],
        data: str = 'meg',
        accept: str = 'accept',
        blink: str = 'blink',
        tag: str = 'rej_tag',
        trigger: str = 'trigger',
        path: PathArg = None,
        nplots: Union[int, Tuple[int, int]] = None,
        topo: bool = None,
        mean: bool = None,
        vlim: float = None,
        color: ColorArg = 'k',
        lw: float = 0.5,
        mark: Sequence = None,
        mcolor: ColorArg = 'r',
        mlw: float = 0.8,
        antialiased: bool = True,
        pos: Tuple[int, int] = None,
        size: Tuple[int, int] = None,
        allow_interpolation: bool = True,
):
    """GUI for rejecting trials of MEG/EEG data

    Parameters
    ----------
    ds
        The data for which to select trials. If ``ds`` is a :class:`Dataset`,
        the subsequent parameters up to 'trigger' specify names of variables in
        ``ds``; if ``ds`` is an :class:`mne.Epochs` object, these parameters are
        ignored.
    data
        Name of the epochs data in ``ds`` (default ``'meg'``).
    accept
        Name of the boolean :class:`Var` in ``ds`` to accept or reject epochs
        (default ``'accept'``).
    blink
        Name of the eye tracker data in ``ds`` if present (default ``'blink'``).
    tag
        Name of the rejection tag (storing the reason for rejecting a
        specific epoch; default ``'rej_tag'``).
    trigger
        Name of the ``int`` :class:`Var` containing the event trigger for each
        epoch (used when loading a rejection file to assert that it comes from
        the same data; default ``'trigger'``).
    path
        Path to the desired rejection file. If the file already exists it
        is loaded as starting values. The extension determines the format
        (``*.pickle`` or ``*.txt``).
    nplots
        Number of epoch plots per page. Can be an ``int`` to produce a
        square layout with that many epochs, or an ``(n_rows, n_columns)``
        tuple. The default is to use the settings from the last session.
    topo
        Show a topomap plot of the time point under the mouse cursor.
        Default (None): use settings form last session.
    mean
        Show a plot of the page mean at the bottom right of the page.
        Default (None): use settings form last session.
    vlim
        Limit of the epoch plots on the y-axis. If None, a value is
        determined automatically to show all data.
    color : matplotlib color
        Color for primary data (default is black).
    lw
        Linewidth for normal sensor plots.
    mark : None | index for sensor dim
        Sensors to plot as individual traces with a separate color.
    mcolor : matplotlib color
        Color for marked traces.
    mlw
        Line width for marked sensor plots.
    antialiased
        Perform Antialiasing on epoch plots (associated with a minor speed
        cost).
    pos
        Window position on screen. Default (None): use settings form last
        session.
    size
        Window size on screen. Default (None): use settings form last
        session.
    allow_interpolation
        Whether to allow interpolating individual channels by epoch (default
        True).

    Notes
    -----
    Exclude bad epochs and interpolate or remove bad channels.

    * Use the `Bad Channels` button in the toolbar to exclude channels from
      analysis (use the `GA` button to plot the grand average and look for
      channels that are consistently bad).
    * Click the `Threshold` button to automatically reject epochs in which the
      signal exceeds a certain threshold.
    * Click on an epoch plot to toggle rejection of that epoch.
    * Press ``i`` on the keyboard to toggle channel interpolation for the
      channel that is closest to the cursor along the y-axis.
    * Press ``shift-i`` on the keyboard to edit a list of interpolated channels
      for the epoch under the cursor.


    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    right-arrow go to the next page
    left-arrow  go to the previous page
    b           butterfly plot of the epoch under the pointer
    c           pairwise sensor correlation plot or the current epoch
    t           topomap plot of the epoch/time point under the pointer
    i           interpolate the channel nearest to the pointer on the y-axis
    shift-i     open dialog to enter channels for interpolation
    =========== ============================================================
    """
    from ._wxgui.app import get_app
    from ._wxgui.select_epochs import TEST_MODE, Document, Frame, Model

    get_app()  # make sure app is created
    bad_chs = None
    doc = Document(ds, data, accept, blink, tag, trigger, path, bad_chs, allow_interpolation)
    model = Model(doc)
    frame = Frame(None, model, nplots, topo, mean, vlim, color, lw, mark, mcolor, mlw, antialiased, pos, size, allow_interpolation)
    frame.Show()
    frame.Raise()
    if TEST_MODE:
        return frame
    else:
        run()


def load_stcs():
    """GUI for detecting and loading source estimates

    Notes
    -----
    This GUI prompts the user for the directory containing the
    source estimates, then automatically detects the experimental
    design, and loads the data into a :class:`Dataset`. The dataset
    can be added to the IPython session's namespace, or passed on
    to the statistics GUI.

    See Also
    --------
    eelbrain.load.mne.DatasetSTCLoader : programmatic access
        to this functionality
    """
    from ._wxgui.app import get_app
    from ._wxgui.load_stcs import STCLoaderFrame, TEST_MODE

    get_app()
    frame = STCLoaderFrame(None)
    if TEST_MODE:
        return frame
    else:
        run()
