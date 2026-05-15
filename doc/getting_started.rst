***************
Getting Started
***************

.. contents:: Contents
   :local:

.. currentmodule:: eelbrain


Documentation
-------------

Documentation can be found here:

* :ref:`exa-intro` to the fundamental Eelbrain data types.
* :ref:`examples` demonstrating different applications.
* `Eelbrain, a Python toolkit for time-continuous analysis with temporal response functions <https://doi.org/10.7554/eLife.85012>`_: A tutorial on using Eelbrain for analyzing an EEG experiment with temporal response functions. Including a `GitHub repoitory <https://github.com/Eelbrain/Alice>`_ with code to reproduce all figures in the paper.
* :ref:`experiment-class-guide` for step-by-step instructions on setting up a M/EEG group level analysis pipeline.
* :ref:`reference` on API and details of all the functionality.


Getting help
------------

To get help, you may want to check (and contribute to):

* The Eelbrian `wiki <https://github.com/Eelbrain/Eelbrain/wiki>`_.
* GitHub `Dicussions <https://github.com/Eelbrain/Eelbrain/discussions>`_.


Plots and GUIs
--------------

Eelbrain plots have two different "modes".
When created in a notebook, plots are static images, but they can be modified through initial parameters, the plots' methods, and matplotlib.
When created from an iPython terminal, plots should open in separate windows with different GUI elements that allow interaction with the plots.
This allows, for example, scrolling through time.

When using a terminal, the terminal usually stays active while allowing interaction with plots.
This functionality relies on `Prompt Toolkit <https://python-prompt-toolkit.readthedocs.io>`_, and can become unreliable with highly interactive GUIs (primarily the ICA and trial selection GUIs).
These GUIs may be more reliable after disabling Prompt Toolkit through calling::

    >>> configure(prompt_toolkit=False)

After disabling Prompt Toolkit, the GUI has to be explicitly invoked. This can be done by running ``gui.run()``::

    >>> configure(prompt_toolkit=False)
    >>> plot.Array(array_1)
    >>> plot.Array(array_2)
    >>> gui.run()

Or by using ``run=True`` in a plot command::

    >>> configure(prompt_toolkit=False)
    >>> plot.Array(array, run=True)

To return control to the Terminal, quit the ``python`` application that opens the plot windows.

.. note::
    On macOS, the GUI backend that Eelbrain uses when run from the command-line interpreter requires a special build of Python called a "Framework build". Eelbrain installs a shortcut to start `IPython <ipython.readthedocs.io>`_ with a Framework build::

        $ eelbrain

    This automatically launches IPython with the "eelbrain" profile. A default startup script that executes ``from eelbrain import *`` is created, and can be changed in the corresponding `IPython profile <http://ipython.readthedocs.io/en/stable/interactive/tutorial.html?highlight=startup#startup-files>`_.


Interacting with other Python libraries
---------------------------------------

`Pandas <https://pandas.pydata.org>`_
    Convert an Eelbrain :class:`Dataset` to a :class:`pandas.DataFrame` using :meth:`Dataset.as_dataframe`. Useful libraries: `Pingouin <https://pingouin-stats.org>`_ (statistics); `Seaborn <http://seaborn.pydata.org>`_ (plotting).
`R <http://r-project.org>`_
    When using R from Python through the :mod:`rpy2` bridge, transfer data between R ``data.frame`` and Eelbrain :class:`Dataset` using :meth:`Dataset.from_r` and :meth:`Dataset.to_r`.


Windows: Scrolling
------------------

Scrolling inside a plot axes normally uses arrow keys, but this is currently
not possible on Windows (due to an issue in Matplotlib). Instead, the following
keys can be used:

+--------+--------+--------+
|        | ↑ ``i``|        |
+--------+--------+--------+
| ← ``j``|        | → ``l``|
+--------+--------+--------+
|        | ↓ ``k``|        |
+--------+--------+--------+
