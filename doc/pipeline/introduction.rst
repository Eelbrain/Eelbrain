.. currentmodule:: eelbrain.pipeline

***********************************
Introduction
***********************************

This page introduces basic concepts used by the pipeline:
the pipeline script, the pipeline GUI, state parameters, and caching.


.. contents:: Contents
   :local:

Workflow
========

Working with a :class:`Pipeline` typically involves 3 different components or workflows:

1. Setting up the :class:`Pipeline` script
2. Data preparation
3. Analysis

These can be achieved in different ways, but the following are the recommended steps.

The :class:`Pipeline` script
----------------------------

:class:`Pipeline` is a template for the pipeline.
This template is adapted to a specific experiment by specifying properties of the experiment as attributes (technically, by creating a `subclass <https://docs.python.org/3/tutorial/classes.html>`_).
The recommended workflow for this is to write a separate script containing this subclass (e.g., ``pipeline.py``).
This provides a stable record of global analysis settings.

An instance of this pipeline then provides access to different analysis stages through its methods:

 - ``.load_...`` methods are for loading data and results.
   Most of these return Eelbrain data types by default, but they can be used to load :mod:`mne` objects by setting ``ndvar=False`` (e.g., :meth:`Pipeline.load_epochs`).
 - ``.show_...`` methods are for retrieving and displaying information at different stages.
 - ``.plot_...`` methods are for generating plots of the data.
 - ``.make_...`` methods are for programmatically accessing processing steps that require user input, like ICA component selection, and caching some intermediate results.

For example, :meth:`Pipeline.load_test` can be used to directly load a mass-univariate test result, without a need to explicitly load data at any intermediate stage.
On the other hand, :meth:`Pipeline.load_epochs` can be used to load the corresponding data epochs, for example to perform a different analysis that may not be implemented in the pipeline.

It is recommended to organize analysis scripts in a dedicated folder, separate from the dataset.
For example, code in ``~/Code/MyProject`` for a dataset at ``~/Data/MyProject``.
Version-controlling a separate code folder (e.g., with `Git <https://git-scm.com>`_) makes it easy to track the history of your analysis.

The project folder typically contains:

1. A :class:`Pipeline` subclass that describes the experiment structure — by convention in ``pipeline.py``.
2. Analysis scripts and/or notebooks that import the pipeline.

A minimal ``MyProject/pipeline.py`` looks like this::

    from eelbrain.pipeline import *

    ROOT = "~/Data/MyProject"  # Where the data is stored

    class MyExperiment(Pipeline):

        # Define experiment attributes here

.. note::
    If your project contains Jupyter Notebooks, consider `Jupytext <https://jupytext.readthedocs.io/>`_ to efficiently track those notebooks in Git.

Data preparation
----------------

Data preparation involves steps that require visual inspection and human decisions, like bad-channel marking, ICA component selection, trial rejection, and MRI coregistration.
The preferred tool for all of these is the pipeline GUI, launched from the command line::

    $ cd  ~/Code/MyProject
    $ eelbrain-gui

The GUI shows the status for every subject in a single table and opens the relevant sub-GUI (ICA component browser, epoch rejection viewer, MNE coregistration tool) on double-click.
It also lets you compute ICA decompositions for all missing subjects in one click.

The same steps can alternatively be performed programmatically from an interactive Python session (iPython, a Jupyter notebook, or a terminal), which is useful for scripting or automation::

    >>> e = eelbrain.load_pipeline("~/Code/MyProject")
    >>> e.make_ica_selection()   # opens ICA GUI for current subject
    >>> e.next()                 # advance to next subject
    >>> e.make_epoch_rejection() # opens epoch rejection GUI

Analysis
--------

Once data preparation is complete, statistical analysis and visualization are best done in notebooks or analysis scripts that can be re-run as needed::

    import eelbrain

    e = eelbrain.load_pipeline()
    result = e.load_test('my_test', tstart=0.1, tstop=0.3)
    eelbrain.plot.brain.cluster(result.clusters[0], ...)

Notebooks and scripts typically live in the project code directory alongside ``pipeline.py`` and can be version-controlled together with the pipeline definition.


.. _pipeline-load-pipeline:

Loading the pipeline: :func:`eelbrain.load_pipeline`
====================================================

:func:`eelbrain.load_pipeline` is the recommended way to instantiate a pipeline from any location — the command line, a Jupyter notebook, or an interactive Python session.
It searches for ``pipeline.py`` (and then ``experiment.py``) when given a directory, and reads the ``root`` variable and the :class:`Pipeline` subclass automatically::

    >>> import eelbrain
    >>> e = eelbrain.load_pipeline("~/Code/MyProject")

If you are already working inside the project directory, omit the path entirely -
this allows relative imports for analysis scripts in the project directory::

    >>> # from MyProject/my_analysis.py:
    >>> e = eelbrain.load_pipeline()
    >>> # from MyProject/analysis/my_analysis.py:
    >>> e = eelbrain.load_pipeline('..')

For advanced Python workflows, you can also import the class directly::

    >>> from my_experiment import MyExperiment
    >>> e = MyExperiment("~/Data/Experiment")


.. _pipeline-gui:

The pipeline GUI
================

The pipeline GUI is the recommended tool for all data-preparation steps.
Launch it from the command line by pointing it at the project directory (or any path accepted by :func:`eelbrain.load_pipeline`)::

    $ eelbrain-gui ~/Code/MyProject

With no argument it uses the current working directory::

    $ cd ~/Code/MyProject
    $ eelbrain-gui

The GUI opens a window with a **Task** dropdown that gives access to:

Bad Channels
    Shows and allows modifying bad channels.
    Double-click on a row to open a visualization of the raw data.
    Right-click to get bad channels as text.

ICA
    Shows the ICA status (missing / selected / number of components rejected) for every subject.
    Double-clicking a row opens the ICA component selection browser for that subject.
    If the ICA decomposition is missing, it is computed first, which can take some time.
    The **Make ICA** button computes ICA decompositions for all subjects that are still missing one.

Epoch rejection
    Shows the trial-rejection status (done / missing) for the selected epoch, rejection method, and raw pipeline combination.
    Double-clicking opens the epoch rejection GUI for that subject.
    For automatic rejection methods, the GUI is read-only and the **Compute rejection** button generates missing rejection files.

MRI
    Shows whether each subject has a FreeSurfer reconstruction (full recon, scaled template, or missing) and whether the common brain (fsaverage) is present.
    Double-clicking the common-brain row when it is missing offers to download fsaverage automatically.

Coregistration
    Shows the coregistration status (OK / missing) for each subject–session combination.
    Double-clicking opens the MNE coregistration GUI pre-loaded with the subject's raw file and, if one already exists, the current transformation.
    For subjects without a FreeSurfer reconstruction the GUI opens against the template brain so the user can use MNE's "Scale MRI" feature to create a scaled copy.


.. _state-parameters:

State parameters
================

A :class:`Pipeline` instance has a state, which determines what data and settings it is currently using.
Not all settings are always relevant.
For example, :ref:`state-subject` is relevant for steps applied separately to each subject, like :meth:`~Pipeline.load_trf`, whereas :ref:`state-group` defines the group of subjects in group level analysis, such as in :meth:`~Pipeline.load_trfs` or :meth:`~Pipeline.load_model_test`.

State parameters can be set after a :class:`Pipeline` has been initialized to affect the analysis, for example::

    >>> my_experiment = eelbrain.load_pipeline()
    >>> my_experiment.set(raw='ica', epoch='story')

sets up ``my_experiment`` to use the ``"ica"`` node of the :ref:`Pipeline-preprocessing`, and the ``"story"`` epoch (defined in :attr:`Pipeline.epochs`). Most methods also accept state parameters, so :meth:`Pipeline.set` does not have to be used separately::

    >>> trf = my_experiment.load_trf(..., raw='ica', epoch='story')

Each state parameter is further documented in the section of this guide it belongs to:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Parameter
     - Selects
     - Section
   * - :ref:`state-subject`
     - The current subject
     - :doc:`data`
   * - :ref:`state-session`
     - The recording session to work with
     - :doc:`data`
   * - :ref:`state-task`
     - The task to work with (usually set through ``epoch``)
     - :doc:`data`
   * - :ref:`state-acquisition`
     - The BIDS acquisition parameter set to analyze
     - :doc:`data`
   * - :ref:`state-run`
     - The run to work with
     - :doc:`data`
   * - :ref:`state-raw`
     - The preprocessing pipeline for continuous data
     - :doc:`preprocessing`
   * - :ref:`state-epoch`
     - The data epoch for analysis
     - :doc:`preprocessing`
   * - :ref:`state-epoch_rejection`
     - The epoch-level artifact rejection method
     - :doc:`preprocessing`
   * - :ref:`state-reference`
     - The EEG re-reference applied to epochs
     - :doc:`preprocessing`
   * - :ref:`state-mri`
     - The MEG/EEG-subject to MRI-subject mapping
     - :doc:`source`
   * - :ref:`state-cov`
     - The noise covariance estimation method
     - :doc:`source`
   * - :ref:`state-src`
     - The source space
     - :doc:`source`
   * - :ref:`state-inv`
     - The inverse solution
     - :doc:`source`
   * - :ref:`state-parc`
     - The brain parcellation
     - :doc:`source`
   * - :ref:`state-adjacency`
     - Channel adjacency for cluster-based tests
     - :doc:`source`
   * - :ref:`state-group`
     - The subject group for group-level analysis
     - :doc:`evoked`
   * - :ref:`state-equalize_evoked_count`
     - Equalizing epoch counts across conditions
     - :doc:`evoked`


Basic configuration
===================

..
    .. py:attribute:: Pipeline.owner
       :type: str

    Set :attr:`Pipeline.owner` to your email address if you want to be able to
    receive notifications. Whenever you run a sequence of commands ``with
    Pipeline.notification:`` you will get an email once the respective code
    has finished executing or run into an error, for example::

        >>> e = MyExperiment()
        >>> with e.notification:
        ...     result = e.load_test('mytest', samples=10000)
        ...

    will send you an email as soon as the test is finished (or the program
    encountered an error)

.. py:attribute:: Pipeline.screen_log_level
   :type: str

Determines the amount of information displayed on the screen while using
a :class:`Pipeline` (see :mod:`logging`).
This class attribute is used as the default for the ``screen_log_level``
initialization parameter.

.. py:attribute:: Pipeline.defaults
   :type: Dict[str, str]

The defaults dictionary can contain default settings for
experiment analysis parameters (see :ref:`state-parameters`), e.g.::

    defaults = {
        'epoch': 'my_epoch',
        'cov': 'noreg',
        'raw': '1-40',
    }


Caching
=======

:class:`Pipeline` caches intermediate results and validates them when they are
loaded. Stale intermediate cache entries are recomputed on demand. Files
stored outside ``cache-dir`` are treated as user-managed outputs and are not
overwritten automatically when they become stale; the corresponding error or GUI
dialog explains whether to recompute, delete, or explicitly accept the existing
file.

Cache files become stale when the relevant pipeline definitions or data sources change.
Such files are not automatically deleted (and can be accessed again by restoring the relevant definitions).
Use :meth:`Pipeline.clean_cache` to scan the cache and delete files that are stale.

Continuous files take up a lot of hard drive space.
By default, files for many pre-processing steps are cached.
This can be controlled with the ``cache`` parameter of the corresponding node definition:
set ``cache=False`` to avoid caching.
To remove files that have already been cached, set ``cache=False`` and then use :meth:`Pipeline.clean_cache`.


.. _Pipeline-example:

Example
=======

The following is a complete example for an experiment class definition file
(the source file can be found in the Eelbrain examples folder at
``examples/imagenet/pipeline.py``):

.. literalinclude:: ../../examples/imagenet/pipeline.py

The event structure is illustrated by looking at the first few events::

    >>> e = load_pipeline()
    >>> data = e.load_events()
    >>> data.head()
    #     sample    value     event     onset    SOA       subject   position
    -------------------------------------------------------------------------
    0     2814      1         unused    2.345    5.0392    01        begin
    1     8861      4         stim_on   7.3842   1.0242    01        middle
    2     10090     3         resp      8.4083   0.2925    01        middle
    3     10441     4         stim_on   8.7008   0.915     01        middle
    4     11539     3         resp      9.6158   0.63417   01        middle
    5     12300     4         stim_on   10.25    0.90167   01        middle
    6     13382     3         resp      11.152   0.64833   01        middle
