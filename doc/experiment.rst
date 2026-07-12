.. currentmodule:: eelbrain.pipeline

.. _experiment-class-guide:

***********************************
The :class:`Pipeline`
***********************************

.. seealso::
     - :class:`Pipeline` class reference for details on all available methods

.. contents:: Contents
   :local:


Introduction
============

The :class:`Pipeline` currently implements the following analysis steps:

#. Preprocessing
#. Epoching and evoked responses
#. Source localization
#. Mass univariate group-level statistics
#. Temporal response function analysis

The input to the pipeline is a BIDS dataset containing raw M/EEG data files and, optionally, MRI files for source localization.
The pipeline automatizes the complete analysis, and provides an interface for preprocessing steps that require user intervention like ICA.
It allows access to the data at any intermediate stage, to allow for customizing the analysis.
It caches intermediate results to make access to these data fast and efficient.

Working with a :class:`Pipeline` typically involves 3 different components or workflows:

1. Setting up the :class:`Pipeline` script
2. Data preparation
3. Analysis

These can be achieved in different ways, but the following are the recommended steps.

The :class:`Pipeline` script
----------------------------

:class:`Pipeline` is a template for the pipeline.
This template is adapted to a specific experiment by specifying properties of the experiment as attributes (technically, by creating a `subclass <https://docs.python.org/3/tutorial/classes.html>`_).
This is described in detail below in :ref:`step-by-step`.
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


Data preparation
----------------

Steps that require visual inspection and human decisions, like bad-channel marking, ICA component selection, trial rejection, and MRI coregistration.
The preferred tool for all of these is the pipeline GUI, launched from the command line::

    $ cd  ~/Code/MyProject
    $ eelbrain-gui

The GUI shows the preparation status for every subject in a single table and opens the relevant sub-GUI (ICA component browser, epoch rejection viewer, MNE coregistration tool) on double-click.
It also lets you compute ICA decompositions for all missing subjects in one click.

The same steps can alternatively be performed programmatically from an interactive Python session (iPython, a Jupyter notebook, or a terminal), which is useful for scripting or automation::

    >>> e = eelbrain.load_pipeline("~/Code/MyProject")
    >>> e.make_ica_selection()   # opens ICA GUI for current subject
    >>> e.next()                 # advance to next subject
    >>> e.make_epoch_rejection() # opens epoch rejection GUI

Analysis
--------

Once data preparation is complete, statistical analysis and visualization are best done in Jupyter notebooks or analysis scripts that can be re-run as needed::

    import eelbrain

    e = eelbrain.load_pipeline()
    result = e.load_test('my_test', tstart=0.1, tstop=0.3)
    eelbrain.plot.brain.cluster(result.clusters[0], ...)

Notebooks and scripts typically live in the project code directory alongside ``pipeline.py`` and can be version-controlled together with the pipeline definition.


.. _step-by-step:

Step by Step
============

.. contents:: Contents
   :local:


.. _Pipeline-filestructure:

Setting up the file structure
-----------------------------

The pipeline expects input dataset in `BIDS (Brain Imaging Data Structure) <https://bids.neuroimaging.io/>`_ format. (To convert your data into BIDS format, use the `MNE-BIDS <https://mne.tools/mne-bids/stable/use.html>`_ library.) In the schema below, curly brackets indicate slots that the pipeline will replace with specific names::


    root                              {root}
    subject folder                       /sub-{subject}
    session folder                          /ses-{session}
    datatype folder                            /{datatype}
    raw data file                                 /sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_{datatype}.fif
    derivatives root                     /derivatives
    MNE derivatives                         /mne
    subject folder                             /sub-{subject}
    session folder                                /ses-{session}
    datatype folder                                  /{datatype}
    trans file                                          /sub-{subject}_ses-{session}_trans.fif
    ICA decomposition                                   /sub-{subject}_ses-{session}_acq-{acquisition}_run-{run}_desc-{raw}_ica.fif
    FreeSurfer SUBJECTS_DIR                 /freesurfer
    mri for each subject                       /sub-{subject}
    mri for template brain                     /fsaverage
    Eelbrain generated files                /eelbrain


.. note::
    In BIDS specification, ``{root}/derivatives`` is for files that do not fit into the BIDS structure, such as FreeSurfer MRIs and Eelbrain-generated files.


``{subject}``, ``{session}``, ``{task}``, ``{acquisition}``, and ``{run}`` are `BIDS entities <https://bids-specification.readthedocs.io/en/stable/appendices/entities.html>`_. ``{session}``, ``{acquisition}``, and ``{run}`` are optional. ``{datatype}`` is inferred by the pipeline from the data files, and can be ``'meg'`` or ``'eeg'``. There can be other entities depending on the dataset, such as `split <https://bids-specification.readthedocs.io/en/stable/appendices/entities.html#split>`_.


``MRI`` files (including ``trans-file``) are optional and only needed for source localization. The ``{root}/derivatives/freesurfer`` directory is `FreeSurfer <https://surfer.nmr.mgh.harvard.edu>`_ subject directory. They either contain the files created by FreeSurfer's `recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_ command, or are created by the MNE-Python coregistration utility for scaled template brains. An ``fsaverage`` folder can be used to store the template brain. Note that the pipeline doesn't use the NIfTI format that BIDS specifies. A corresponding ``trans-file`` is created with the MNE-Python coregistration utility in either case (see more information on using `structural MRIs <https://github.com/Eelbrain/Eelbrain/wiki/Coregistration%3A-Structural-MRI>`_ or the `fsaverage template brain <https://github.com/Eelbrain/Eelbrain/wiki/Coregistration%3A-Template-Brain>`_).


A BIDS dataset can be scanned by initializing a :class:`Pipeline` with the data ``{root}`` location, for example::

    e = Pipeline("~/Data/Experiment")


Assuming a subject without explicit ``{session}`` is named "S001", the pipeline will look for data at the following locations:

- The raw data file at ``~/Data/Experiment/sub-S001/meg/sub-S001_task-words_meg.fif``
- The trans-file from the coregistration at ``~/Data/Experiment/derivatives/mne/sub-S001/meg/sub-S001_trans.fif``
- The FreeSurfer MRI-directory at ``~/Data/Experiment/derivatives/freesurfer/sub-S001``
- The template brain MRI-directory at ``~/Data/Experiment/derivatives/freesurfer/fsaverage``

The subjects and corresponding MRIs that were discovered can be shown
in the ``eelbrain-gui``, or using :meth:`Pipeline.show_subjects`::

    >>> e.show_subjects()
    #    subject   mri
    -----------------------------------------
    0    R0026     R0026
    1    R0040     fsaverage * 0.92
    2    R0176     fsaverage * 0.954746600461
    ...


Setting up the analysis code
----------------------------

It is recommended to organize analysis scripts in a dedicated folder, for example ``~/Code/MyProject``.
Version-controlling this folder (e.g., with `Git <https://git-scm.com>`_) makes it easy to track the history of your analysis.

The project folder typically contains:

1. A :class:`Pipeline` subclass that describes the experiment structure — by convention in ``pipeline.py``.
2. Analysis scripts or Jupyter notebooks that import the pipeline.

A minimal ``MyProject/pipeline.py`` looks like this::

    from eelbrain.pipeline import *

    ROOT = "~/Data/MyExperiment"

    class MyExperiment(Pipeline):

        # Define experiment attributes here

.. note::
    If your project contains Jupyter Notebooks, consider `Jupytext <https://jupytext.readthedocs.io/>`_ to efficiently track those notebooks in Git.

.. _pipeline-load-pipeline:

Loading the pipeline: :func:`eelbrain.load_pipeline`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`eelbrain.load_pipeline` is the recommended way to instantiate a pipeline from any location — the command line, a Jupyter notebook, or an interactive Python session.
It searches for ``pipeline.py`` (and then ``experiment.py``) when given a directory, and reads the ``root`` variable and the :class:`Pipeline` subclass automatically::

    >>> import eelbrain
    >>> e = eelbrain.load_pipeline("~/Code/MyProject")

If you are already working inside the project directory, omit the path entirely::

    >>> e = eelbrain.load_pipeline()

For advanced Python workflows, you can also import the class directly::

    >>> from my_experiment import MyExperiment
    >>> e = MyExperiment("~/Data/Experiment")


.. _pipeline-gui:

The pipeline GUI: ``eelbrain-gui``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


.. _Pipeline-preprocessing:

Pre-processing
--------------

Make sure an appropriate pre-processing pipeline is defined as
:attr:`Pipeline.raw`.

To inspect raw data for a given pre-processing step use::

    >>> e.set(raw='1-40')
    >>> y = e.load_raw(ndvar=True)
    >>> p = plot.TopoButterfly(y, xlim=10, w=0)

Which will plot a 10 s excerpt and allow scrolling through the rest of the data.


.. _Pipeline-events:

Events
------

By default, events are read from BIDS side-car files.
Triggers in raw data files provide a fallback.
If needed, set :attr:`Pipeline.merge_triggers` to handle spurious events.
Use the :attr:`Pipeline.variables` settings to add event labels.
Events are represented as :class:`~eelbrain.Dataset` objects and can be inspected with
corresponding methods and functions, for example::

    >>> e = MyExperiment("~/Data/Experiment")
    >>> data = e.load_events()
    >>> data.head()
    >>> print(table.frequencies('value', data=data))


For more complex designs and variables, you can override methods that provide
complete control over the events. These are the transformations applied to
events from BIDS side-cars or from raw-file triggers (in this order):

 - :meth:`Pipeline.fix_events`: Change event order, timing and remove/add
   events
 - :attr:`Pipeline.variables`: Add labels based on triggers
 - :meth:`Pipeline.label_events`: Add any more complex labels


Defining data epochs
--------------------

Once events are properly labeled, define :attr:`Pipeline.epochs`.

There is one special epoch to define, which is called ``'cov'``. This is the
data epoch that will be used to estimate the sensor noise covariance matrix for
source estimation.

In order to find the right ``sel`` epoch parameter, it can be useful to actually
load the events with :meth:`Pipeline.load_events` and test different
selection strings. The epoch selection is determined by evaluating the
epoch's ``sel`` expression in the events Dataset. Thus, a specific setting could be
tested with::

    >>> data = e.load_events()
    >>> print(data.sub("event == 'value'"))

For datasets with a ``run`` entity, :class:`PrimaryEpoch` combines all runs for
the selected subject/session/task/acquisition by default. To analyze a single run, set the
epoch's ``run`` parameter, for example ``PrimaryEpoch('task', run='1')``.


Bad channels
------------

Flat channels are automatically excluded from the analysis.

An initial check for noisy channels can be done by looking at the raw data (see
:ref:`Pipeline-preprocessing` above).
If this inspection reveals bad channels, they can be excluded using
:meth:`Pipeline.make_bad_channels`.

Another good check for bad channels is plotting the average evoked response,
and looking for channels which are uncorrelated with neighboring
channels. To plot the average before trial rejection, use::

    >>> data = e.load_epochs(epoch='epoch', reject=False)
    >>> plot.TopoButterfly('meg', data=data)

The neighbor correlation can also be quantified, using::

    >>> nc = neighbor_correlation(concatenate(data['meg']))
    # Plot topographical map of the neighbor correlation
    >>> plot.Topomap(nc)
    # Check for channels whose average correlation with its neighbors is < 0.3
    >>> nc.sensor.names[nc < 0.3]
    Datalist(['MEG 099'])
    # Remove that channel
    >>> e.make_bad_channels(['MEG 099'])


A simple way to cycle through subjects when performing a manual pre-processing
step is :meth:`Pipeline.next`.

If a general threshold is adequate, the selection of bad channels based on
neighbor-correlation can be automated using the
:meth:`Pipeline.make_bad_channels_neighbor_correlation` method::

    >>> for subject in e:
    ...     e.make_bad_channels_neighbor_correlation(0.3)


ICA
---

If preprocessing includes ICA, each subject's ICA decomposition must be computed and unwanted components must be selected for removal.

The preferred workflow is the :ref:`pipeline-gui`.
Open it, select the ICA task from the **Task** dropdown, then:

* Click **Make ICA** to compute decompositions for all subjects that are still missing one (runs in the background).
* Double-click a subject row to open the ICA component browser and mark components for removal.

Alternatively, the same steps can be performed programmatically.
The :ref:`state-raw` state must be set to the ICA stage before calling :meth:`Pipeline.make_ica_selection`::

    >>> e.set(raw='ica')
    >>> e.make_ica_selection()

To cycle through subjects::

    >>> e.make_ica_selection(epoch='epoch', decim=10)
    >>> e.next()
    subject: 'R1801' -> 'R2079'
    >>> e.make_ica_selection(epoch='epoch', decim=10)
    ...

See :meth:`Pipeline.make_ica_selection` for display options.


Trial and channel rejection
---------------------------

Different methods for artifact rejection in epoched data
can be defined in :attr:`Pipeline.epoch_rejection`.

Bad trials can be manually rejected with :class:`ManualRejection`, or detected
automatically with :class:`ChannelModelRejection`.
Automatic rejection can also mark bad EEG channels for interpolation within an
epoch, or within shorter windows for long and variable-length epochs.
Rejections are always specific to a given ``raw`` state, primary epoch, and
``epoch_rejection`` setting.

For example::

    class Experiment(Pipeline):

        epoch_rejection = {
            'manual': ManualRejection(),
            'auto': ChannelModelRejection(max_interpolate=5),
        }

In the :ref:`pipeline-gui`, select the **Epoch rejection** task, choose the epoch and raw pipeline from the dropdowns, and double-click a subject row to open the rejection GUI for that subject.
For automatic rejection, click **Compute rejection** to generate missing files and double-click rows to inspect them.

Alternatively, cycle through subjects programmatically::

    >>> e.set(raw='ica1-40', epoch='word', epoch_rejection='manual')
    >>> e.make_epoch_rejection()
    >>> e.next()
    subject: 'R1801' -> 'R2079'
    >>> e.make_epoch_rejection()
    ...

To reject trials based on a pre-determined amplitude threshold::

    >>> for subject in e:
    ...     e.make_epoch_rejection(auto=1e-12)
    ...


.. _Pipeline-intro-cov:

Empty room noise covariance
---------------------------

To use empty room data for estimating the noise covariance, follow these steps:

- Set up empty room data according to the `instruction in BIDS specification <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetoencephalography.html#empty-room-meg-recordings>`_.
- Use the empty room covariance through :ref:`state-cov` with ``e.set(cov='emptyroom')``.


.. _Pipeline-intro-analysis:

Analysis
--------

With preprocessing completed, there are different options for analyzing the
data.

The most flexible option is loading data from the desired processing stage using
one of the many ``.load_...`` methods of the :class:`Pipeline`. For
example, load a :class:`eelbrain.Dataset` with source-localized condition averages using
:meth:`Pipeline.load_evoked` (with ``inv`` set for source space), then test a hypothesis using one of the
mass-univariate test from the :mod:`testnd` module. To make this kind of
analysis replicable, it is probably useful to write the complete analysis as a
separate script that imports the experiment (see the `example experiment folder
<https://github.com/Eelbrain/Eelbrain/tree/master/examples/mouse>`_).

Many statistical comparisons can also be specified in the
:attr:`Pipeline.tests` attribute, and then loaded directly using the
:meth:`Pipeline.load_test` method. This has the advantage that the tests
will be cached automatically and, once computed, can be loaded very quickly.
However, these definitions are not quite as flexible as writing a custom script.

.. _Pipeline-example:

Example
=======

The following is a complete example for an experiment class definition file
(the source file can be found in the Eelbrain examples folder at
``examples/imagenet/pipeline.py``):

.. literalinclude:: ../examples/imagenet/pipeline.py

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


Experiment Definition
=====================

.. contents:: Contents
   :local:


Basic setup
-----------

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

:class:`Pipeline` caches intermediate results and validates them when they are
loaded. Most stale intermediate cache entries are recomputed on demand. Files
stored outside ``cache-dir`` are treated as user-managed outputs and are not
overwritten automatically when they become stale; the corresponding error or GUI
dialog explains whether to recompute, delete, or explicitly accept the existing
file. Cached tests are likewise not overwritten silently; use the corresponding
``make`` or ``redo`` option to regenerate them.

.. py:attribute:: Pipeline.screen_log_level
   :type: str

Determines the amount of information displayed on the screen while using
an :class:`Pipeline` (see :mod:`logging`).
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


Finding files
-------------

.. py:attribute:: Pipeline.ignore_entities
   :type: Dict[str, list[str]]

Exclude certain entities from the experiment, e.g.::

    ignore_entities = {
        'subject': ['S666', 'S999'],
        'session': ['02'],
    }

.. py:attribute:: Pipeline.mri_subjects
   :type: Dict[str, Dict[str, str]]

Map MEG/EEG subjects to FreeSurfer MRI subjects. Keys in ``mri_subjects`` are names for different mappings and correspond to values of the :ref:`state parameter <state-parameters>` ``mri``; the inner dictionaries map :ref:`state-subject` values to MRI subject names (i.e., directory names under ``{root}/derivatives/freesurfer``). By default, an identity mapping is used (each subject uses their own MRI directory), but custom mappings can be defined, for example to let several subjects share a template brain or to point to individually scaled MRI subjects, e.g.::

    mri_subjects = {
        '': {  # default identity mapping
            'S001': 'S001',
            'S002': 'S002',
        },
        'fsaverage': {  # all subjects use the template brain
            'S001': 'fsaverage',
            'S002': 'fsaverage',
        },
    }

.. .. py:attribute:: Pipeline.datatype
..    :type: str

.. Data type for the raw data directory. By default, this is ``meg``, i.e., the experiment will look for raw files at ``{root}/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_run-{run}_meg.fif``. After setting ``datatype = 'eeg'``, the experiment will look at ``{root}/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg.fif``.


.. py:attribute:: Pipeline.preload
   :type: bool

Whether to preload raw data into memory before creating epochs. Default is ``False``. It is observed that in some datasets reading raw data when creating epochs is time consuming, and in these cases setting ``preload=True`` can speed up epoch creation.


Reading files
-------------

.. note::
    Gain more control over reading files by adding a ``'raw'`` entry with a :class:`RawSource` to :attr:`Pipeline.raw`.

.. py:attribute:: Pipeline.stim_channel
   :type: str | Sequence[str]

By default, events are loaded from all stim channels; use this parameter to restrict events to one or several stim channels.

.. py:attribute:: Pipeline.merge_triggers
   :type: int

Use a non-default ``merge`` parameter for :func:`.load.mne.events`.

.. py:attribute:: Pipeline.trigger_shift
   :type: float | Dict[str, float]

Set this attribute to shift all trigger times by a constant (in seconds). For example, with ``trigger_shift = 0.03`` a trigger that originally occurred 35.10 seconds into the recording will be shifted to 35.13. If the trigger delay differs between subjects, this attribute can also be a dictionary mapping subject names to shift values, e.g. ``trigger_shift = {'S001': 0.02, 'S002': 0.05, ...}``.

The MEG system used to acquire the data determines the sensor neighborhood graph
(adjacency). This is usually detected automatically; when it needs to be set
explicitly, define a ``'raw'`` entry with a :class:`RawSource` in
:attr:`Pipeline.raw` and set its ``sysname`` (and/or ``adjacency``) parameter.
For example, for data from NYU New York::

    raw = {
        'raw': RawSource(sysname='KIT-157'),
        '1-40': RawFilter('raw', 1, 40),
    }


Pre-processing (raw)
--------------------

.. py:attribute:: Pipeline.raw

Define a pre-processing pipeline as a series of linked processing steps
(:mod:`mne` refers to continuous data that is not time-locked to a specific event as :class:`~mne.io.Raw`, with filenames matching ``*_raw.fif``):

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   RawFilter
   RawICA
   RawApplyICA
   RawMaxwell
   RawOversampledTemporalProjection
   RawSource
   RawReReference


Each preprocessing step is defined as a named entry with its input as first argument (``source``).
The raw data that constitutes the input to the pipeline can be accessed as ``"raw"``
For example, the following definition sets up a pipeline for MEG, using TSSS, a band-pass filter and ICA::

    class Experiment(Pipeline):

        raw = {
            'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
            '1-40': RawFilter('tsss', 1, 40),
            'ica': RawICA('1-40', 'task', 'extended-infomax', n_components=0.99),
        }

To use the ``raw --> TSSS --> 1-40 Hz band-pass`` pipeline, use ``e.set(raw="1-40")``.
To use ``raw --> TSSS --> 1-40 Hz band-pass --> ICA``, select ``e.set(raw="ica")``.

The following is an example for EEG using band-pass filter and ICA::

    class Experiment(Pipeline):

        raw = {
            '1-20': RawFilter('raw', 1, 20, cache=False),
            'ica': RawICA('1-20', 'stories'),
            # Use the same ICA, but with a high pass filter with a lower cutoff frequency:
            '0.2-20': RawFilter('raw', 0.2, 20, cache=False),
            '0.2-20ica': RawApplyICA('0.2-20', 'ica'),
        }


.. note::
    Continuous files take up a lot of hard drive space.
    By default, files for many pre-processing steps are cached.
    This can be controlled with the ``cache`` parameter: set ``cache=False`` to avoid caching.
    To remove files that have already been cached, set ``cache=False`` and then use :meth:`Pipeline.clean_cache`.


Events
------

.. note::
    Gain more control over events through overriding :meth:`Pipeline.fix_events` and :meth:`Pipeline.label_events`.

.. py:attribute:: Pipeline.variables

Event variables add labels and variables to the events:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   LabelVar
   EvalVar
   GroupVar


Most of the time, the main purpose of this attribute is to turn trigger values
(the ``value`` column in the events Dataset) into meaningful labels::


    class Mouse(Pipeline):

        variables = {
            'stimulus': LabelVar('value', {(162, 163): 'target', (166, 167): 'prime'}),
            'prediction': LabelVar('value', {162: 'expected', 163: 'unexpected'}),
        }

This defines a variable called "stimulus", and on this variable all events
that have triggers 162 and 163 have the value ``"target"``, and events with
trigger 166 and 167 have the value ``"prime"``.
The "prediction" variable only labels triggers 162 and 163.
Unmentioned trigger values are assigned the empty string (``''``).


Epochs
------

.. py:attribute:: Pipeline.epochs

Epochs are specified as a ``{name: epoch_definition}`` dictionary. Names are
:class:`str`, and ``epoch_definition`` are instances of the classes
described below:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   PrimaryEpoch
   SecondaryEpoch
   SuperEpoch
   ContinuousEpoch
   EpochCollection


Examples::

    epochs = {
        # some primary epochs:
        'picture': PrimaryEpoch('words', "stimulus == 'picture'"),
        'word': PrimaryEpoch('words', "stimulus == 'word'"),
        # use the picture baseline for the sensor covariance estimate
        'cov': SecondaryEpoch('picture', tmax=0),
        # another secondary epoch:
        'animal_words': SecondaryEpoch('noun', sel="word_type == 'animal'"),
        # a superset-epoch:
        'all_stimuli': SuperEpoch(('picture', 'word')),
        # estimate one TRF for each member epoch:
        'stimuli_separate': EpochCollection(('picture', 'word')),
    }

.. py:attribute:: Pipeline.epoch_rejection

Epoch-level artifact rejection is controlled through the
:ref:`state-epoch_rejection` state.
Define :attr:`Pipeline.epoch_rejection` as a ``{name: EpochRejection}``
dictionary of trial-rejection settings.

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   ManualRejection
   ChannelModelRejection

The empty rejection name (``epoch_rejection=''``) is always available and means
that no epoch-level rejection is applied.
Add a :class:`ManualRejection` entry for rejection files edited in the GUI, and
use :class:`ChannelModelRejection` for automatically generated EEG rejection and
channel-interpolation files.


References (re-referencing)
---------------------------

.. py:attribute:: Pipeline.references

EEG re-referencing applied to epochs *after* channel interpolation (so that bad
channels do not contaminate the reference). References are defined as a
``{name: reference_definition}`` dictionary and selected through the
:ref:`state-reference` state:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   Reference

An ``'average'`` reference (``Reference('average')``) is always available. It can
be overridden, for example to reconstruct an implicit recording reference channel
(a channel such as ``Cz`` that was the recording reference is absent from the data
but can be reconstructed as zeros before averaging)::

    references = {
        # override the built-in 'average' to reconstruct the implicit Cz reference:
        'average': Reference('average', add='Cz'),
        # mastoid reference:
        'mastoid': Reference(['M1', 'M2']),
    }

This differs from :class:`RawReReference`, which re-references the continuous raw
data *before* epoching and interpolation. ``references`` is orthogonal to
``raw``, ``epoch`` and ``epoch_rejection``, so different references can be compared with
``e.set(reference=...)`` without duplicating epoch definitions.

.. note::
    The reference is only applied to EEG channels. Loading data that contains no
    EEG channels with a non-empty ``reference`` raises an error; use
    ``reference=''`` for such data. Source localization handles EEG referencing
    internally (via MNE's average-reference projector) and always uses
    ``reference=''`` regardless of the current state.


Temporal Response Functions
---------------------------

Pipeline-managed TRF analyses are configured through predictors, estimators,
and optional named models.
Use :meth:`Pipeline.load_trf` to compute or load a single subject's TRF and
:meth:`Pipeline.load_trfs` to assemble TRFs and fit metrics for a subject group.

.. py:attribute:: Pipeline.predictors

Predictors are defined as a ``{name: predictor_definition}`` dictionary:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   EventPredictor
   UTSPredictor
   NUTSPredictor

:class:`EventPredictor` creates impulses from the events Dataset.
:class:`UTSPredictor` and :class:`NUTSPredictor` load per-stimulus predictor
files from ``{root}/derivatives/predictors``.

.. py:attribute:: Pipeline.estimators

Estimators are defined as a ``{name: estimator_definition}`` dictionary.
The built-in ``'boosting'`` estimator is always available and can be overridden
to change its parameters.

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   Boosting
   NCRF

.. py:attribute:: Pipeline.models

Named model strings can be defined as abbreviations and reused in
:meth:`Pipeline.load_trf` and :meth:`Pipeline.load_trfs`.

.. py:attribute:: Pipeline.stim_var

Column in the events Dataset that identifies the stimulus for file predictors
(default ``'stimulus'``).

Example::

    class Experiment(Pipeline):

        predictors = {
            'onset': EventPredictor(),
            'env': UTSPredictor(resample='resample'),
            'word': NUTSPredictor(),
        }
        stim_var = 'stimulus'
        estimators = {
            'boosting': Boosting(partitions=5),
        }
        models = {
            'acoustic': 'onset + env',
        }

    e = Experiment("~/Data/Experiment")
    e.set(epoch='story', raw='1-40', inv='')
    trf = e.load_trf('acoustic + word-frequency', -0.1, 0.5)
    trfs = e.load_trfs('all', 'acoustic', -0.1, 0.5)


Tests
-----

.. py:attribute:: Pipeline.tests

Statistical tests are defined as ``{name: test_definition}`` dictionary.
This allows automatic caching of permutation test results when using :meth:`Pipeline.load_test`.
Tests are defined using the following classes:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   TTestOneSample
   TTestRelated
   TTestIndependent
   ANOVA
   TContrastRelated
   TwoStageTest


Example::

    tests = {
        'my_anova': ANOVA('noise * word_type * subject'),
        'my_ttest': TTestRelated('noise', 'a_lot_of_noise', 'no_noise'),
    }


Subject groups
--------------

.. py:attribute:: Pipeline.groups

A subject group called ``'all'`` containing all subjects is always implicitly
defined. Additional subject groups can be defined in
:attr:`Pipeline.groups` with ``{name: group_definition}``
entries:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   Group
   SubGroup

Example::

    groups = {
        'good': SubGroup('all', ['R0013', 'R0666']),
        'bad': Group(['R0013', 'R0666']),
    }


Parcellations (:attr:`parcs`)
-----------------------------

.. py:attribute:: Pipeline.parcs

A parcellation determines how the brain surface is divided into regions.
A number of standard parcellations are automatically defined (see
:ref:`state-parc` below). Additional parcellations can be defined in
the :attr:`Pipeline.parcs` dictionary with ``{name: parc_definition}``
entries.


.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   SubParc
   CombinationParc
   SeededParc
   IndividualSeededParc
   FreeSurferParc
   FSAverageParc


Visualization defaults
----------------------

.. py:attribute:: Pipeline.brain_plot_defaults

The :attr:`Pipeline.brain_plot_defaults` dictionary can contain options
that change defaults for brain plots. The following options are available:

surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
    Freesurfer surface to use as brain geometry.
views : :class:`str` | iterator of :class:`str`
    View or views to show in the figure. Can also be set for each parcellation,
    see :attr:`Pipeline.parc`.
foreground : mayavi color
    Figure foreground color (i.e., the text color).
background : mayavi color
    Figure background color.
smoothing_steps : ``None`` | :class:`int`
    Number of smoothing steps to display data.


.. _state-parameters:

State Parameters
================

An :class:`Pipeline` instance has a state, which determines what data and settings it is currently using.
Not all settings are always relevant.
For example, :ref:`state-subject` is relevant for steps applied separately to each subject, like :meth:`~Pipeline.make_ica_selection`, whereas :ref:`state-group` defines the group of subjects in group level analysis, such as in :meth:`~Pipeline.load_test`.

State Parameters can be set after an :class:`Pipeline` has been initialized to affect the analysis, for example::

    >>> my_experiment = Pipeline()
    >>> my_experiment.set(raw='1-40', cov='noreg')

sets up ``my_experiment`` to use a 1-40 Hz band-pass filter as preprocessing, and to use sensor covariance matrices without regularization. Most methods also accept state parameters, so :meth:`Pipeline.set` does not have to be used separately.

.. contents:: Contents
   :local:


.. _state-session:

``session``
-----------

Which session to work with.


.. _state-task:

``task``
-----------

Which task to work with (usually set automatically when :ref:`state-epoch` is set).


.. _state-acquisition:

``acquisition``
---------------

Which BIDS acquisition parameter set to analyze. Acquisitions are independent
analysis branches and are never combined by the pipeline. Run aggregation is
restricted to runs belonging to the selected acquisition. For datasets without
an ``acq-`` entity, this state is the empty string.


.. _state-run:

``run``
-------

Which run to work with. For :class:`PrimaryEpoch` definitions without an
explicit ``run`` parameter, events and epochs are combined across all available
runs for the current subject/session/task/acquisition.


.. _state-raw:

``raw``
-------

Select the preprocessing pipeline applied to the continuous data. Options are
all the processing steps defined in :attr:`Pipeline.raw`, as well as
``"raw"`` for using unprocessed raw data.


.. _state-subject:

``subject``
-----------

Any subject in the experiment.


.. _state-group:

``group``
---------

Any group defined in :attr:`Pipeline.groups`. Will restrict the analysis
to that group of subjects.


.. _state-epoch:

``epoch``
---------

Any epoch defined in :attr:`Pipeline.epochs`. Specify the epoch on which
the analysis should be conducted.


.. _state-epoch_rejection:

``epoch_rejection``
-------------------

Selects an entry from :attr:`Pipeline.epoch_rejection`.
``e.set(epoch_rejection='')`` is always available and disables epoch-level
rejection. Other values correspond to user-defined entries such as
``ManualRejection`` or ``ChannelModelRejection`` settings.


.. _state-reference:

``reference`` (EEG re-referencing)
----------------------------------

Selects an EEG re-reference defined in :attr:`Pipeline.references`, applied to
epochs after channel interpolation. ``e.set(reference='')`` (the default) applies
no epoch-stage re-referencing; ``e.set(reference='average')`` applies the
corresponding :class:`Reference`. Loading sensor-space data that contains no EEG
channels with a non-empty ``reference`` raises an error. Source localization
handles EEG referencing internally.


.. _state-equalize_evoked_count:

``equalize_evoked_count``
-------------------------

By default, the analysis uses all epochs marked as good during rejection.
Set ``equalize_evoked_count='eq'`` to discard trials to make sure the same number of epochs goes into each cell of the model (see ``equal_count`` parameter to :meth:`.Dataset.aggregate`).

'' (default)
    Use all epochs.
'eq'
    Make sure the same number of epochs ``n`` is used in each cell by discarding epochs.
    The first ``n`` epochs are used for each condition (assuming that habituation increases by condition).


.. _state-cov:

``cov``
-------

The method for correcting the sensor covariance.

'noreg'
    Use raw covariance as estimated from the data (do not regularize).
'bestreg' (default)
    Find the regularization parameter that leads to optimal whitening of the
    baseline.
'reg'
    Use the default regularization parameter (0.1).
'auto'
    Use automatic selection of the optimal regularization method, as described in :func:`mne.compute_covariance`.
'emptyroom'
    Empty room covariance; for required setup, see :ref:`Pipeline-intro-cov`.
'ad_hoc'
    Use diagonal covariance based on :func:`mne.cov.make_ad_hoc_cov`.


.. _state-src:

``src``
-------

The source space to use.

 - ``ico-x``: Surface source space based on icosahedral subdivision of the
   white matter surface ``x`` steps (e.g., ``ico-4``, the default).
 - ``vol-x``: Volume source space based on a volume grid with ``x`` mm
   resolution (``x`` is the distance between sources, e.g. ``vol-10`` for a
   10 mm grid).


.. _state-inv:

``inv``
-------

What inverse solution to use for source localization.
``inv`` can be set with :meth:`Pipeline.set_inv`,
which has a detailed description of the options.
``inv`` can also be set directly using the appropriate string,
e.g., ``e.set(inv='fixed-6-MNE')``.
To determine the string corresponding to a given set of parameters,
use :meth:`Pipeline.inv_str`. For example::

    >>> Pipeline.inv_str('fixed', snr=6, method='MNE')
    'fixed-6-MNE'

Consequently, the following two are equivalent for setting ``inv``::

    >>> Pipeline.set_inv('fixed', snr=6, method='MNE')
    >>> Pipeline.set(inv='fixed-6-MNE')


.. _state-parc:

``parc`` (parcellations)
---------------------------------

The parcellation determines how the brain surface is divided into regions.
Parcellations included with FreeSurfer can directly be used:

- FreeSurfer Parcellations: ``aparc.a2005s``, ``aparc.a2009s``, ``aparc``, ``aparc.DKTatlas``, ``PALS_B12_Brodmann``, ``PALS_B12_Lobes``, ``PALS_B12_OrbitoFrontal``, ``PALS_B12_Visuotopic``.

Additional parcellation can be defined in the :attr:`Pipeline.parcs`
attribute. Parcellations are used in different contexts:

- When loading source space data, the current ``parc`` state determines the parcellation of the source space (change the state parameter with ``e.set(parc='aparc')``).
- When loading tests, setting the ``parc`` parameter treats each label as a
  separate ROI. For spatial cluster-based tests that means that no clusters can
  cross the boundary between two labels. On the other hand, using the ``mask``
  parameter treats all named labels as connected surface, but discards any
  sources labeled as ``"unknown"``. For example, loading a test with
  ``mask='PALS_B12_Lobes'`` will perform a whole-brain test on the cortex, while
  discarding subcortical sources.

Parcellations are set with their name, with the exception of
:class:`SeededParc`: for those, the name is followed by the radius in mm, for
example, to use seeds defined in a parcellation named ``'myparc'`` with a radius
of 25 mm around the seed, use ``e.set(parc='myparc-25')``.

A few additional parcellations that provide homogeneous masks are included
for backwards compatibility. For future work, it is recommended to build
such masks from ``aparc`` or another parcellation with more fine-grained
subdivision into labels.

- ``cortex``: All sources in cortex, based on the FreeSurfer "cortex" label.
- ``lobes``: Modified version of ``PALS_B12_Lobes`` in which the limbic lobe is merged into the other 4 lobes.
- ``lobes-op``: One large region encompassing occipital and parietal lobe in each hemisphere.
- ``lobes-ot``: One large region encompassing occipital and temporal lobe in each hemisphere.



.. _state-adjacency:

``adjacency``
----------------

Possible values: ``''``, ``'link-midline'``

Adjacency refers to the edges connecting data channels (sensors for sensor
space data and sources for source space data). These edges are used to find
clusters in cluster-based permutation tests. For source spaces, the default is
to use FreeSurfer surfaces in which the two hemispheres are unconnected. By
setting ``adjacency='link-midline'``, this default adjacency can be
modified so that the midline gyri of the two hemispheres get linked at sources
that are at most 15 mm apart. This parameter currently does not affect sensor
space adjacency.
