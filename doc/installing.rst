**********
Installing
**********

For the simplest experience, follow the :ref:`install-full`.
For alternative ways of installing, see :ref:`install-custom`.

.. contents:: Contents
   :local:


.. _install-full:

Recommended Setup
-----------------

The recommended tool for deploying Eelbrain is the `Mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ package manager:

1. `Install Mamba <https://conda-forge.org/download/>`_.

2. Create an environment containing Eelbrain along with other libraries required for a project. An example environment is provided in the `Alice <https://github.com/Eelbrain/Alice>`_ repository's `environment.yml <https://github.com/Eelbrain/Alice/blob/main/environment.yml>`_ file:

.. code-block:: bash

    $ mamba env create --file=https://github.com/Eelbrain/Alice/raw/main/environment.yml


By default, this new environment will be called ``eelbrain`` (as specified in the `environment.yml <https://github.com/Eelbrain/Alice/blob/main/environment.yml>`_ file), and can be activated with the following command (note the change in the command line prefix):

.. code-block:: bash

    (base) $ mamba activate eelbrain
    (eelbrain) $


You will have to activate the environment every time you start a new shell session.

Eelbrain can then be used from this environment, for example through `Jupyter Lab <https://jupyterlab.readthedocs.io/en/latest/>`_:

.. code-block:: bash

    (eelbrain) $ jupyter lab


.. SEEALSO::
    Mamba is an extension of `Conda <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_. The Conda documentation provides more information on `environments <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.


Updating
^^^^^^^^

In an existing environment, Eelbrain can generally be updated with the following command (assuming the target environment is currently active):

.. code-block:: bash

    (eelbrain) $ mamba update eelbrain


However, in complex environments this can lead to package conflicts (mamba will display an error message).
In such cases it may be easier to just cerate a new environment.

The currently installed version can be displayed with the ``mamba list`` command:

.. code-block:: bash

    (eelbrain) $ mamba list eelbrain
    # packages in environment at ~/miniforge3/envs/test:
    #
    # Name                    Version                   Build  Channel
    eelbrain                  0.39.11         py311h86e7398_0    conda-forge

Sometimes Mamba may run into difficulties while updating and it may be easier to create a new environment instead.


.. _install-custom:

Alternative Installations
-------------------------

Eelbrain can be installed as pre-compiled library from `conda-forge <https://conda-forge.org>`_:

.. code-block:: bash

    $ mamba install eelbrain

or with `conda`:

.. code-block:: bash

    $ conda install -c conda-forge eelbrain

Alternatively, Eelbrain is also hosted on the Python Package Index (`PyPI <https://pypi.org/project/eelbrain/>`_), but installing from PyPI requires local compilation:

.. code-block:: bash

    $ pip install eelbrain

The default PYPI installation omits optional dependencies required for using the GUIs and for creating `PySurfer <https://pysurfer.github.io>`_/`Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_ based anatomical plots. In order to install these dependencies as well, use one of:

.. code-block:: bash

    $ pip install eelbrain[brain]
    $ pip install eelbrain[gui]
    $ pip install eelbrain[full]


.. SEEALSO::
    For more installing options, including pre-releases, see the `wiki <https://github.com/christianbrodbeck/Eelbrain/wiki/Installing>`_.


Making your analysis future-proof
---------------------------------

Newer version of Eelbrain support files generated with previous versions.
However, running the same code with different versions can lead to slightly different results.
This does not just apply to Eelbrain, but equally to the libraries it relies on like NumPy and MNE-Python, and happens for example when underlying implementations change, which can lead to different rounding errors.
These changes should be very small, but they can sometimes change a p-value slightly, so it might be undesirable when revisiting at a previously finished analysis.
In order to be able to replicate results exactly in the future, it might be useful to keep a
`record <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments>`_
of the environment with which the analysis was done.
