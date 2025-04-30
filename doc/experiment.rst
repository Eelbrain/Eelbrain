.. currentmodule:: eelbrain.pipeline

.. _experiment-class-guide:

***********************************
The :class:`MneExperiment` Pipeline
***********************************

.. seealso::
     - :class:`MneExperiment` class reference for details on all available methods
     - `Pipeline wiki page <https://github.com/christianbrodbeck/Eelbrain/wiki/MNE-Pipeline>`_
       for additional information
     - `TRFExperiment <https://trf-tools.readthedocs.io/latest/pipeline.html>`_: an experimental extension of the pipeline to Temporal Response Function analysis

.. contents:: Contents
   :local:


Introduction
============

The :class:`MneExperiment` pipeline manages the following analysis steps:

#. Preprocessing
#. Epoching
#. Optional source localization
#. Mass univariate group-level statistics

The input to the pipeline are the raw M/EEG data files and, optionally, MRI files for source localization.
The first three steps are based on :mod:`mne` functions; statistics are based on Eelbrain functions.
The pipeline automatizes the complete analysis, and provides an interface for preprocessing steps that require user intervention like ICA.
It allows access to the data at any intermediate stage, to allow for customizing the analysis.
It caches intermediate results to make access to these data fast and efficient.

:class:`MneExperiment` is a template for the pipeline.
This template is adapted to a specific experiment by specifying properties of the experiment as attributes (technically, by creating a `subclass <https://docs.python.org/3/tutorial/classes.html>`_).
An instance of this pipeline then provides access to different analysis stages through its methods:

 - ``.load_...`` methods are for loading data and results.
   Most of these return Eelbrain data types by default, but they can be used to load :mod:`mne` objects by setting ``ndvar=False`` (e.g., :meth:`MneExperiment.load_epochs`).
 - ``.show_...`` methods are for retrieving and displaying information at different stages.
 - ``.plot_...`` methods are for generating plots of the data.
 - ``.make_...`` methods are for generating various intermediate results.
   Most of these methods do not have to be called by the user, as they are invoked automatically when needed.
   An exception are those that require user input, like ICA component selection, which are mentioned below.

For example, :meth:`MneExperiment.load_test` can be used to directly load a mass-univariate test result, without a need to explicitly load data at any intermediate stage.
On the other hand, :meth:`MneExperiment.load_epochs` can be used to load the corresponding data epochs, for example to perform a different analysis that may not be implemented in the pipeline.


Step by Step
============

.. contents:: Contents
   :local:


.. _MneExperiment-filestructure:

Setting up the file structure
-----------------------------

The first steps for setting up the pipeline are:

- Arranging the input data files in the expected file structure
- Defining an :class:`MneExperiment` subclass with the parameters required to find those files

The pipeline expects input files in a strictly determined folder/file structure.
In the schema below, curly brackets indicate slots that the pipeline will replace with specific
names. For example, ``{subject}`` will be replaced with each specific subject's name::

    Root                            {root}
    M/EEG directory                    /{data_dir}
    M/EEG subject                         /{subject}
    trans-file                               /{subject}-trans.fif
    raw-file                                 /{subject}_{session}-raw.fif
    MRI directory                      /mri
    MRI subject                           /{subject}


``{data_dir}``, the directory in which the pipeline looks for the raw data, is determined by the :attr:`MneExperiment.data_dir` attribute. By default it is ``'meg'``, but it can be changed, for example, to ``'eeg'``. This is merely to make the filenames less confusing when e.g. working with EEG data, it does not influence the analysis in any other way.

``MRI`` files (including ``trans-file``) are optional and only needed for source localization. The ``{root}/mri/{subject}`` directories are `FreeSurfer <https://surfer.nmr.mgh.harvard.edu>`_ subject directories. They either contain the files created by FreeSurfer's `recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_ command, or are created by the MNE-Python coregistration utility for scaled template brains. A corresponding ``trans-file`` is created with the MNE-Python coregistration utility in either case (see more information on using `structural MRIs <https://github.com/christianbrodbeck/Eelbrain/wiki/Coregistration%3A-Structural-MRI>`_ or the `fsaverage template brain <https://github.com/christianbrodbeck/Eelbrain/wiki/Coregistration%3A-Template-Brain>`_).

``{session}`` refers to the name of the recording session. The name of one or several recording session(s) has to be specified on an :class:`MneExperiment` subclass, using the  :attr:`MneExperiment.sessions` attribute. Those names will be used to find the raw data files, by filling in the ``raw-file`` template::

    from eelbrain.pipeline import *

    class MyExperiment(MneExperiment):

        data_dir = 'eeg'
        sessions = 'words'



The final step to locating the files is providing the ``{root}`` location when initializing that subclass, for example::


    e = MyExperiment("~/Data/Experiment")


The pipeline will then determine the subject names based on the names of the folders inside the M/EEG directory. Only names matching a specific expression will be considered, for example "S" followed by 3 or more digits. This expression can be customized in :attr:`MneExperiment.subject_re`.

Assuming a subject is named "S001", the pipeline will look for data at the following locations:

- The raw data file at ``~/Data/Experiment/meg/S001/S001_words-raw.fif`` (the session is called "words" which is specified in ``MyExperiment.sessions``)
- The trans-file from the coregistration at ``~/Data/Experiment/meg/S001/S001-trans.fif``
- The FreeSurfer MRI-directory at ``~/Data/Experiment/mri/S001``

The setup can be tested using :meth:`MneExperiment.show_subjects`, which shows
a list of the subjects and corresponding MRIs that were discovered::

    >>> e.show_subjects()
    #    subject   mri
    -----------------------------------------
    0    R0026     R0026
    1    R0040     fsaverage * 0.92
    2    R0176     fsaverage * 0.954746600461
    ...


.. note::
    The default input format for M/EEG data is the FIFF format (``*-raw.fif`` files). To specify an alternative input data format, see :attr:`MneExperiment.raw`.


.. py:attribute:: MneExperiment.visits

.. note::
    If participants come back for the experiment on multiple occasions, a
    :attr:`visits` attribute might also be needed. For details see the
    corresponding `wiki page <https://github.com/christianbrodbeck/Eelbrain/
    wiki/MneExperiment-analysis-options#multiple-visits>`_.

Setting up the analysis code
----------------------------

It is recommended to organize analysis scripts in a dedicated folder.
For example, we will assume that all analysis scripts will be saved in a directory called ``~/Code/MyProject``.
This makes it easy to keep track of the history of this folder, for example using `Git <https://git-scm.com>`_.

You will want to access the :class:`MneExperiment` subclass (``MyExperiment`` above) from different locations (for instance, from a terminal to do artifact rejection, and from different Jupyter Notebooks to pursue different analyses).
Thus, it makes sense to define the experiment subclass in a separate Python file, and ``run`` or ``import`` that file as needed.
IN the example above, the following would be saved in ``~/Code/MyProject/my_experiment.py``::

    from eelbrain.pipeline import *

    class MyExperiment(MneExperiment):

        data_dir = 'eeg'
        sessions = 'words'

    e = MyExperiment("~/Data/Experiment")


From a terminal, this could then be used as follows::

    ~/Code/MyProject $ eelbrain  # eelbrain on macOS; iPython on Linux
    In [1]: run my_experiment.py
    In [2]: e.show_subjects()
    #    subject   mri
    -----------------------------------------
    0    R0026     R0026
    1    R0040     fsaverage * 0.92
    2    R0176     fsaverage * 0.954746600461
    ...


Similarly, you can ``run my_experiment.py`` in the first cell of a Jupyter Notebook that is saved in the same folder.

.. note::
    If your project contains Jupyter Notebooks, consider `Jupytext <https://jupytext.readthedocs.io/>`_ to efficiently track those notebooks in Git.


.. _MneExperiment-preprocessing:

Pre-processing
--------------

Make sure an appropriate pre-processing pipeline is defined as
:attr:`MneExperiment.raw`.

To inspect raw data for a given pre-processing step use::

    >>> e.set(raw='1-40')
    >>> y = e.load_raw(ndvar=True)
    >>> p = plot.TopoButterfly(y, xlim=10, w=0)

Which will plot a 10 s excerpt and allow scrolling through the rest of the data.


.. _MneExperiment-events:

Events
------

If needed, set :attr:`MneExperiment.merge_triggers` to handle spurious events.
Then, add event labels.
Initially, events are only labeled with the trigger ID. Use the
:attr:`MneExperiment.variables` settings to add labels.
Events are represented as :class:`Dataset` objects and can be inspected with
corresponding methods and functions, for example::

    >>> e = MyExperiment("~/Data/Experiment")
    >>> data = e.load_events()
    >>> data.head()
    >>> print(table.frequencies('trigger', data=data))


For more complex designs and variables, you can override methods that provide
complete control over the events. These are the transformations applied to
the triggers extracted from raw files (in this order):

 - :meth:`MneExperiment.fix_events`: Change event order, timing and remove/add
   events
 - :attr:`MneExperiment.variables`: Add labels based on triggers
 - :meth:`MneExperiment.label_events`: Add any more complex labels


Defining data epochs
--------------------

Once events are properly labeled, define :attr:`MneExperiment.epochs`.

There is one special epoch to define, which is called ``'cov'``. This is the
data epoch that will be used to estimate the sensor noise covariance matrix for
source estimation.

In order to find the right ``sel`` epoch parameter, it can be useful to actually
load the events with :meth:`MneExperiment.load_events` and test different
selection strings. The epoch selection is determined by
``selection = event_ds.eval(epoch['sel'])``. Thus, a specific setting could be
tested with::

    >>> data = e.load_events()
    >>> print(data.sub("event == 'value'"))


Bad channels
------------

Flat channels are automatically excluded from the analysis.

An initial check for noisy channels can be done by looking at the raw data (see
:ref:`MneExperiment-preprocessing` above).
If this inspection reveals bad channels, they can be excluded using
:meth:`MneExperiment.make_bad_channels`.

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
step is :meth:`MneExperiment.next`.

If a general threshold is adequate, the selection of bad channels based on
neighbor-correlation can be automated using the
:meth:`MneExperiment.make_bad_channels_neighbor_correlation` method::

    >>> for subject in e:
    ...     e.make_bad_channels_neighbor_correlation(0.3)


ICA
---

If preprocessing includes ICA, select which ICA components should be removed.
To open the ICA selection GUI, The experiment :ref:`state-raw` state needs to be
set to the ICA stage of the pipeline::

    >>> e.set(raw='ica')
    >>> e.make_ica_selection()

See :meth:`MneExperiment.make_ica_selection` for more information on display
options and on how to precompute ICA decomposition for all subjects.

When selecting ICA components for multiple subject, a simple way to cycle
through subjects is :meth:`MneExperiment.next`, like::

    >>> e.make_ica_selection(epoch='epoch', decim=10)
    >>> e.next()
    subject: 'R1801' -> 'R2079'
    >>> e.make_ica_selection(epoch='epoch', decim=10)
    >>> e.next()
    subject: 'R2079' -> 'R2085'
    ...


Trial selection
---------------

For each primary epoch that is defined, bad trials can be rejected using
:meth:`MneExperiment.make_epoch_selection`. Rejections are specific to a given ``raw``
state::

    >>> e.set(raw='ica1-40', epoch='word')
    >>> e.make_epoch_selection()
    >>> e.next()
    subject: 'R1801' -> 'R2079'
    >>> e.make_epoch_selection()
    ...

To reject trials based on a pre-determined threshold, a loop can be used::

    >>> for subject in e:
    ...     e.make_epoch_selection(auto=1e-12)
    ...


.. _MneExperiment-intro-analysis:

Analysis
--------

With preprocessing completed, there are different options for analyzing the
data.

The most flexible option is loading data from the desired processing stage using
one of the many ``.load_...`` methods of the :class:`MneExperiment`. For
example, load a :class:`Dataset` with source-localized condition averages using
:meth:`MneExperiment.load_evoked_stc`, then test a hypothesis using one of the
mass-univariate test from the :mod:`testnd` module. To make this kind of
analysis replicable, it is probably useful to write the complete analysis as a
separate script that imports the experiment (see the `example experiment folder
<https://github.com/christianbrodbeck/Eelbrain/tree/master/examples/mouse>`_).

Many statistical comparisons can also be specified in the
:attr:`MneExperiment.tests` attribute, and then loaded directly using the
:meth:`MneExperiment.load_test` method. This has the advantage that the tests
will be cached automatically and, once computed, can be loaded very quickly.
However, these definitions are not quite as flexible as writing a custom script.

Finally, for tests defined in :attr:`MneExperiment.tests`, the
:class:`MneExperiment` can generate HTML report files. These are generated with
the :meth:`MneExperiment.make_report` and :meth:`MneExperiment.make_report_rois`
methods.

.. Warning::
    If source files are changed (raw files, epoch rejection or bad channel
    files, ...) reports are not updated automatically unless the corresponding
    :meth:`MneExperiment.make_report` function is called again. For this reason
    it is useful to have a script to generate all desired reports. Running the
    script ensures that all reports are up-to-date, and will only take seconds
    if nothing has to be recomputed (for an example see ``make-reports.py`` in
    the `example experiment folder
    <https://github.com/christianbrodbeck/Eelbrain/tree/master/examples/mouse>`_).


.. _MneExperiment-example:

Example
=======

The following is a complete example for an experiment class definition file
(the source file can be found in the Eelbrain examples folder at
``examples/mouse/mouse.py``):

.. literalinclude:: ../examples/mouse/mouse.py

The event structure is illustrated by looking at the first few events::

    >>> from mouse import *
    >>> data = e.load_events()
    >>> data.head()
    trigger   i_start   T        SOA     subject   stimulus   prediction
    --------------------------------------------------------------------
    182       104273    104.27   12.04   S0001
    182       116313    116.31   1.313   S0001
    166       117626    117.63   0.598   S0001     prime      expected
    162       118224    118.22   2.197   S0001     target     expected
    166       120421    120.42   0.595   S0001     prime      expected
    162       121016    121.02   2.195   S0001     target     expected
    167       123211    123.21   0.596   S0001     prime      unexpected
    163       123807    123.81   2.194   S0001     target     unexpected
    167       126001    126      0.598   S0001     prime      unexpected
    163       126599    126.6    2.195   S0001     target     unexpected


Experiment Definition
=====================

.. contents:: Contents
   :local:


Basic setup
-----------

.. py:attribute:: MneExperiment.owner
   :type: str

Set :attr:`MneExperiment.owner` to your email address if you want to be able to
receive notifications. Whenever you run a sequence of commands ``with
mne_experiment.notification:`` you will get an email once the respective code
has finished executing or run into an error, for example::

    >>> e = MyExperiment()
    >>> with e.notification:
    ...     e.make_report('mytest', tstart=0.1, tstop=0.3)
    ...

will send you an email as soon as the report is finished (or the program
encountered an error)

.. py:attribute:: MneExperiment.auto_delete_results
   :type: bool

Whenever a :class:`MneExperiment` instance is initialized with a valid
``root`` path, it checks whether changes in the class definition invalidate
previously computed results. By default, the user is prompted to confirm
the deletion of invalidated results. Set :attr:`auto_delete_results` to ``True``
to delete them automatically without interrupting initialization.

.. py:attribute:: MneExperiment.auto_delete_cache
   :type: bool

:class:`MneExperiment` caches various intermediate results. By default, if a
change in the experiment definition would make cache files invalid, the outdated
files are automatically deleted. Set :attr:`.auto_delete_cache` to ``'ask'`` to
ask for confirmation before deleting files. This can be useful to prevent
accidentally deleting files that take long to compute when editing the pipeline
definition.
When using this option, set :attr:`MneExperiment.screen_log_level` to
``'debug'`` to learn about what change caused the cache to be invalid.

.. py:attribute:: MneExperiment.screen_log_level
   :type: str

Determines the amount of information displayed on the screen while using
an :class:`MneExperiment` (see :mod:`logging`).

.. py:attribute:: MneExperiment.defaults
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

.. py:attribute:: MneExperiment.sessions
   :type: str | Sequence[str]

The name, or a list of names of the raw data files (see :ref:`MneExperiment-filestructure`).

.. py:attribute:: MneExperiment.data_dir
   :type: str

Folder name for the raw data directory. By default, this is ``meg``, i.e., the experiment will look for raw files at ``root/meg/{subject}/{subject}_{session}-raw.fif``. After setting ``data_dir = 'eeg'``, the experiment will look at ``root/eeg/{subject}/{subject}_{session}-raw.fif``.

.. py:attribute:: MneExperiment.subject_re
   :type: str

Subjects are identified on initialization by looking for folders in the :attr:`MneExperiment.data_dir` directory (``meg`` by default) whose name matches the :attr:`.MneExperiment.subject_re` regular expression. By default, this is one or more characters or underline, followed by one or more digits, for example: ``S001``, ``subject_1``, ``R0001`` (for information about how to define a different pattern, see :mod:`re`).


Reading files
-------------

.. note::
    Gain more control over reading files through adding a :class:`RawPipe` to :attr:`MneExperiment.raw`.

.. py:attribute:: MneExperiment.stim_channel
   :type: str | Sequence[str]

By default, events are loaded from all stim channels; use this parameter to restrict events to one or several stim channels.

.. py:attribute:: MneExperiment.merge_triggers
   :type: int

Use a non-default ``merge`` parameter for :func:`.load.mne.events`.

.. py:attribute:: MneExperiment.trigger_shift
   :type: float | Dict[str, float]

Set this attribute to shift all trigger times by a constant (in seconds). For example, with ``trigger_shift = 0.03`` a trigger that originally occurred 35.10 seconds into the recording will be shifted to 35.13. If the trigger delay differs between subjects, this attribute can also be a dictionary mapping subject names to shift values, e.g. ``trigger_shift = {'S001': 0.02, 'S002': 0.05, ...}``.

.. py:attribute:: MneExperiment.meg_system
   :type: str

Specify the MEG system used to acquire the data so that the right sensor neighborhood graph can be loaded. This is usually automatic, but is needed for KIT files convert with with :mod:`mne` < 0.13. Equivalent to the ``sysname`` parameter in :func:`.load.mne.epochs_ndvar` etc. For example, for data from NYU New York, the correct value is ``meg_system="KIT-157"``.


Pre-processing (raw)
--------------------

.. py:attribute:: MneExperiment.raw

Define a pre-processing pipeline as a series of linked processing steps
(:mod:`mne` refers to continuous data that is not time-locked to a specific event as :class:`~mne.io.Raw`, with filenames matching ``*-raw.fif``):

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

    class Experiment(MneExperiment):

        sessions = 'session'
        raw = {
            'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
            '1-40': RawFilter('tsss', 1, 40),
            'ica': RawICA('1-40', 'session', 'extended-infomax', n_components=0.99),
        }
        
To use the ``raw --> TSSS --> 1-40 Hz band-pass`` pipeline, use ``e.set(raw="1-40")``. 
To use ``raw --> TSSS --> 1-40 Hz band-pass --> ICA``, select ``e.set(raw="ica")``.

The following is an example for EEG using band-pass filter, ICA and re-referencing::

    class Experiment(MneExperiment):

        data_dir = 'eeg'
        sessions = ['stories', 'tones']
        raw = {
            '1-20': RawFilter('raw', 1, 20, cache=False),
            'ica': RawICA('1-20', 'stories'),
            'reref': RawReReference('ica', ['A1', 'A2'], 'A2')
            # Use the same ICA, but with a high pass filter with a lower cutoff frequency:
            '0.2-20': RawFilter('raw', 0.2, 20, cache=False),
            '0.2-20ica': RawApplyICA('0.2-20', 'ica'),
            '0.2reref': RawReReference('0.2-20ica', ['A1', 'A2'], 'A2'),
        }


.. note::
    Continuous files take up a lot of hard drive space. By default, files for most pre-processing steps are cached. This can be controlled with the ``cache`` parameter: set ``cache=False`` to avoid caching. To delete files corresponding to a specific step (e.g., ``raw='1-40'``), use the :meth:`MneExperiment.rm` method::

        >>> e.rm('cached-raw-file', True, raw='1-40')


Events
------

.. note::
    Gain more control over events through overriding :meth:`MneExperiment.fix_events` and :meth:`MneExperiment.label_events`.

.. py:attribute:: MneExperiment.variables

Event variables add labels and variables to the events:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   LabelVar
   EvalVar
   GroupVar


Most of the time, the main purpose of this attribute is to turn trigger
values into meaningful labels::


    class Mouse(MneExperiment):

        variables = {
            'stimulus': LabelVar('trigger', {(162, 163): 'target', (166, 167): 'prime'}),
            'prediction': LabelVar('trigger', {162: 'expected', 163: 'unexpected'}),
        }

This defines a variable called "stimulus", and on this variable all events
that have triggers 162 and 163 have the value ``"target"``, and events with
trigger 166 and 167 have the value ``"prime"``.
The "prediction" variable only labels triggers 162 and 163.
Unmentioned trigger values are assigned the empty string (``''``).


Epochs
------

.. py:attribute:: MneExperiment.epochs

Epochs are specified as a ``{name: epoch_definition}`` dictionary. Names are
:class:`str`, and ``epoch_definition`` are instances of the classes
described below:

.. autosummary::
   :toctree: generated
   :template: class_nomethods.rst

   PrimaryEpoch
   SecondaryEpoch
   SuperEpoch


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
    }


Tests
-----

.. py:attribute:: MneExperiment.tests

Statistical tests are defined as ``{name: test_definition}`` dictionary.
This allows automatic caching of permutation test results when using :meth:`MneExperiment.load_test`.
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

.. py:attribute:: MneExperiment.groups

A subject group called ``'all'`` containing all subjects is always implicitly
defined. Additional subject groups can be defined in
:attr:`MneExperiment.groups` with ``{name: group_definition}``
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

.. py:attribute:: MneExperiment.parcs

A parcellation determines how the brain surface is divided into regions.
A number of standard parcellations are automatically defined (see
:ref:`state-parc` below). Additional parcellations can be defined in
the :attr:`MneExperiment.parcs` dictionary with ``{name: parc_definition}``
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

.. py:attribute:: MneExperiment.brain_plot_defaults

The :attr:`MneExperiment.brain_plot_defaults` dictionary can contain options
that changes defaults for brain plots (for reports and movies). The following
options are available:

surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
    Freesurfer surface to use as brain geometry.
views : :class:`str` | iterator of :class:`str`
    View or views to show in the figure. Can also be set for each parcellation,
    see :attr:`MneExperiment.parc`.
foreground : mayavi color
    Figure foreground color (i.e., the text color).
background : mayavi color
    Figure background color.
smoothing_steps : ``None`` | :class:`int`
    Number of smoothing steps to display data.


.. _state-parameters:

State Parameters
================

An :class:`MneExperiment` instance has a state, which determines what data and settings it is currently using.
Not all settings are always relevant.
For example, :ref:`state-subject` is relevant for steps applied separately to each subject, like :meth:`~MneExperiment.make_ica_selection`, whereas :ref:`state-group` defines the group of subjects in group level analysis, such as in :meth:`~MneExperiment.load_test`.

State Parameters can be set after an :class:`MneExperiment` has been initialized to affect the analysis, for example::

    >>> my_experiment = MneExperiment()
    >>> my_experiment.set(raw='1-40', cov='noreg')

sets up ``my_experiment`` to use a 1-40 Hz band-pass filter as preprocessing, and to use sensor covariance matrices without regularization. Most methods also accept state parameters, so :meth:`MneExperiment.set` does not have to be used separately.

.. contents:: Contents
   :local:


.. _state-session:

``session``
-----------

Which raw session to work with (one of :attr:`MneExperiment.sessions`; usually
set automatically when :ref:`state-epoch` is set)


.. _state-visit:

``visit``
---------

Which visit to work with (one of :attr:`MneExperiment.visits`)


.. _state-raw:

``raw``
-------

Select the preprocessing pipeline applied to the continuous data. Options are
all the processing steps defined in :attr:`MneExperiment.raw`, as well as
``"raw"`` for using unprocessed raw data.


.. _state-subject:

``subject``
-----------

Any subject in the experiment (subjects are identified based on :attr:`MneExperiment.subject_re`).


.. _state-group:

``group``
---------

Any group defined in :attr:`MneExperiment.groups`. Will restrict the analysis
to that group of subjects.


.. _state-epoch:

``epoch``
---------

Any epoch defined in :attr:`MneExperiment.epochs`. Specify the epoch on which
the analysis should be conducted.


.. _state-rej:

``rej`` (trial rejection)
-------------------------

Trial rejection can be turned off ``e.set(rej='')``, meaning that no trials are
rejected, and back on, meaning that the corresponding rejection files are used
``e.set(rej='man')``.


.. _state-model:

``model``
---------

While the :ref:`state-epoch` state parameter determines which events are
included when loading data, the ``model`` parameter determines how these events
are split into different condition cells. The parameter should be set to the
name of a categorial event variable which defines the desired cells.
In the :ref:`MneExperiment-example`,
``e.load_evoked(epoch='target', model='prediction')``
would load responses to the target, averaged for expected and unexpected trials.

Cells can also be defined based on crossing two variables using the ``%`` sign.
In the :ref:`MneExperiment-example`, to load corresponding primes together with
the targets, you would use
``e.load_evoked(epoch='word', model='stimulus % prediction')``.


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
`empty_room`
    Empty room covariance; for required setup, see `Empty room covariance <https://github.com/christianbrodbeck/Eelbrain/wiki/MneExperiment-analysis-options#empty-room-covariance>`_.
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
``inv`` can be set with :meth:`MneExperiment.set_inv`,
which has a detailed description of the options.
``inv`` can also be set directly using the appropriate string,
e.g., ``e.set(inv='fixed-6-MNE-0')``.
To determine the string corresponding to a given set of parameters,
use :meth:`MneExperiment.inv_str`. For example::

    >>> MneExperiment.inv_str('fixed', snr=6, method='MNE', depth=0)
    'fixed-6-MNE-0'

Consequently, the following two are equivalent for setting ``inv``::

    >>> MneExperiment.set_inv('fixed', snr=6, method='MNE', depth=0)
    >>> MneExperiment.set(inv='fixed-6-MNE-0')


.. _state-parc:

``parc``/``mask`` (parcellations)
---------------------------------

The parcellation determines how the brain surface is divided into regions.
Parcellations included with FreeSurfer can directly be used:

- FreeSurfer Parcellations: ``aparc.a2005s``, ``aparc.a2009s``, ``aparc``, ``aparc.DKTatlas``, ``PALS_B12_Brodmann``, ``PALS_B12_Lobes``, ``PALS_B12_OrbitoFrontal``, ``PALS_B12_Visuotopic``.

Additional parcellation can be defined in the :attr:`MneExperiment.parcs`
attribute. Parcellations are used in different contexts:

- When loading source space data, the current ``parc`` state determines the parcellation of the source space (change the state parameter with ``e.set(parc='aparc')``).
- When loading tests, setting the ``parc`` parameter treats each label as a
  separate ROI. For spatial cluster-based tests that means that no clusters can
  cross the boundary between two labels. On the other hand, using the ``mask``
  parameter treats all named labels as connected surface, but discards any
  sources labeled as ``"unknown"``. For example, loading a test with
  ``mask='PALS_B12_Lobes'`` will perform a whole-brain test on the cortex, while
  discarding subcortical sources.

Parcellations are set with their name, with the expception of
:class:`SeededParc`: for those, the name is followed by the radious in mm, for
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



.. _state-connectivity:

``connectivity``
----------------

Possible values: ``''``, ``'link-midline'``

Connectivity refers to the edges connecting data channels (sensors for sensor
space data and sources for source space data). These edges are used to find
clusters in cluster-based permutation tests. For source spaces, the default is
to use FreeSurfer surfaces in which the two hemispheres are unconnected. By
setting ``connectivity='link-midline'``, this default connectivity can be
modified so that the midline gyri of the two hemispheres get linked at sources
that are at most 15 mm apart. This parameter currently does not affect sensor
space connectivity.


.. _state-select_clusters:

``select_clusters`` (cluster selection criteria)
------------------------------------------------

In thresholded cluster test, clusters are initially filtered with a minimum
size criterion. This can be changed with the ``select_clusters`` analysis
parameter with the following options:

================ ======== =========== ===========
Name             Min time Min sources Min sensors
================ ======== =========== ===========
``"all"``        -        -           -
``"10ms"``       10 ms    10          4
``""`` (default) 25 ms    10          4
``"large"``      25 ms    20          8
================ ======== =========== ===========

To change the cluster selection criterion use for example::

    >>> e.set(select_clusters='all')
