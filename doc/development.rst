.. _development:

***********
Development
***********

Eelbrain is hosted on `GitHub <https://github.com/Eelbrain/Eelbrain>`_.


The Development Version
-----------------------

Development takes place on the ``main`` branch, while release versions are maintained on
branches called ``r/0.26`` etc. For further information on working with
GitHub see `GitHub's instructions <https://help.github.com/articles/fork-a-repo/>`_.

The repository contains a conda environment that contains everything needed to use Eelbrain except Eelbrain itself.
First, clone the repository (or your `fork <https://help.github.com/articles/fork-a-repo>`_), and change into the repository directory::

    $ git clone https://github.com/Eelbrain/Eelbrain.git
    $ cd Eelbrain

To generate the ``eeldev`` environment, use::

    $ mamba env create --file=env-dev.yml

The development version of Eelbrain can then be installed with ``pip``::

    $ mamba activate eeldev
    $ pip install -e .

On macOS, the ``$ eelbrain`` shell script to run ``iPython`` with the framework
build is not installed properly by ``pip``; in order to fix this, run::

    $ ./fix-bin

In Python, you can make sure that you are working with the development version::

    >>> import eelbrain
    >>> eelbrain.__version__
    'dev'


Contributing
------------

Contributions to code and documentation are welcome as pull requests into the ``main`` branch.

Style guides:

- Python code style follows `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ (mostly)
- The documentation is written in `ReStructured Text <https://www.sphinx-doc.org/en/master/usage/restructuredtext>`_
- Docstrings follow the `numpydoc style  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_


Useful tools:

- Graphical frontend for git: `SourceTree <https://www.sourcetreeapp.com>`_
- Python IDE: `PyCharm <https://www.jetbrains.com/pycharm>`_


Testing
-------

Tests for individual modules are included in folders called ``tests``, usually
on the same level as the module.
To run all tests, run ``$ make test`` from the Eelbrain project directory.
On macOS, tests needs to run with the framework build of Python;
if you get a corresponding error, run ``$ ./fix-bin pytest`` from the
``Eelbrain`` repository root.
