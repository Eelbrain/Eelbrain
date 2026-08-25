.. currentmodule:: eelbrain.pipeline

************************
Preprocessing and epochs
************************

This section covers the steps that turn continuous raw recordings into clean data epochs:
the raw preprocessing pipeline, event handling, epoch definitions, artifact rejection, and EEG re-referencing.
These steps are shared between :doc:`evoked` and :doc:`trf`.

.. contents:: Contents
   :local:


.. _Pipeline-preprocessing:

Raw pipeline
============

.. py:attribute:: Pipeline.raw

Define a pre-processing pipeline as a series of linked processing steps
(:mod:`mne` refers to continuous data that is not time-locked to a specific event as :class:`~mne.io.Raw`, with filenames matching ``*_raw.fif``):

.. autosummary::
   :toctree: ../generated
   :template: class_nomethods.rst

   RawSource
   RawFilter
   RawICA
   RawApplyICA
   RawMaxwell
   RawOversampledTemporalProjection
   RawReReference


Each preprocessing step is defined as a named entry with its input as first argument (``source``).
The raw data that constitutes the input to the pipeline can be accessed as ``"raw"``.
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

To inspect raw data for a given pre-processing step use::

    >>> e.set(raw='1-40')
    >>> y = e.load_raw(ndvar=True)
    >>> p = plot.TopoButterfly(y, xlim=10, w=0)

Which will plot a 10 s excerpt and allow scrolling through the rest of the data.

For EEG, make sure the ``montage`` and ``adjacency`` are defined correctly.
They can customized by adding :class:`RawSource` to :attr:`Pipeline.raw`.
These can be tested with :class:`plot.SensorMap`::

    >>> raw = e.load_raw(raw='raw')
    >>> plot.SensorMap(raw, adjacency=True)

:class:`plot.SensorMap` is also useful for determining sensor names for
:attr:`Pipeline.references`.


Bad channels
============

Flat channels are automatically excluded from the analysis.

An initial check for noisy channels can be done by looking at the raw data (see
:ref:`Pipeline-preprocessing` above), or through the **Bad Channels** task of the :ref:`pipeline-gui`.
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
===

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


.. _Pipeline-events:

Events
======

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


Event variables
---------------

.. py:attribute:: Pipeline.variables

Event variables add labels and variables to the events:

.. autosummary::
   :toctree: ../generated
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

Some column names are reserved, because the pipeline writes them itself and a
variable of the same name would be overwritten: the BIDS entities (``subject``,
``session``, ``task``, ``acquisition``, ``run``), the event columns ``sample``,
``value``, ``onset``, ``index``, ``epoch``, ``accept``,
``interpolate_channels`` and ``interpolate_windows``, the data columns
``epochs``, ``evoked``, ``src``, ``stc``, ``label_tc`` and ``model``, and
``epoch_time``, ``events`` and ``tmax`` (used by
:class:`ContinuousEpoch`). Using one of these as a variable name raises an
error.

Variables come in two kinds, which differ in where they are added:

- *Event* variables are computed from the events themselves, and are present in
  all data. This covers :class:`EvalVar` and most :class:`LabelVar` definitions.
- *Across-subject* variables have definitions that span subjects, and are only
  added where different subjects' data are combined. This covers
  :class:`GroupVar`, a :class:`LabelVar` on ``'subject'``, and any variable
  derived from either, such as ``EvalVar("diagnosis == 'patient'")``.

An across-subject variable is thus only present in data that spans subjects
(e.g. ``e.load_selected_events('all')``, but not ``e.load_selected_events('01')``), and
can not be used where data is processed one subject at a time, such as in an
epoch ``sel`` expression or as an evoked ``model``. Use a :class:`GroupVar` to
compare groups (see :ref:`pipeline-groups`).

Where subjects are combined, each variable is added if the combined data still
provides what it is computed from. A variable keyed on the subject, such as a
behavioral score, therefore reaches group analyses even where the individual
events are no longer present::

    variables = {
        'score': LabelVar('subject', {'S001': 3.2, 'S002': 4.5}),
    }

Variables are applied in the order they are defined, so a variable that builds
on another has to come after it.

.. py:attribute:: Pipeline.cache_event_labels
   :type: bool

Whether to cache the output of :meth:`Pipeline.label_events` (default ``True``).
Set to ``False`` if :meth:`Pipeline.label_events` reads from external files whose changes should trigger cache invalidation.


Epochs
======

.. py:attribute:: Pipeline.epochs

Once events are properly labeled, define data epochs in :attr:`Pipeline.epochs`.
Epochs are specified as a ``{name: epoch_definition}`` dictionary. Names are
:class:`str`, and ``epoch_definition`` are instances of the classes
described below:

.. autosummary::
   :toctree: ../generated
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

The epoch selection is determined by evaluating the
epoch's ``sel`` expression in the events Dataset.
In order to find the right ``sel`` parameter, it can be useful to actually
load the events with :meth:`Pipeline.load_events` and test different
selection strings::

    >>> data = e.load_events()
    >>> print(data.sub("event == 'value'"))

For datasets with a ``run`` entity, :class:`PrimaryEpoch` combines all runs for
the selected subject/session/task/acquisition by default. To analyze a single run, set the
epoch's ``run`` parameter, for example ``PrimaryEpoch('task', run='1')``.

There is one special epoch name, ``'cov'``: the
data epoch that will be used to estimate the sensor noise covariance matrix for
source estimation (see :doc:`source`).

:class:`ContinuousEpoch` extracts continuous data segments spanning multiple
events, which is mainly useful for TRF analysis of continuous stimuli
(see :ref:`trf-epochs` for how predictors are generated for the different epoch types).


Epoch rejection
===============

.. py:attribute:: Pipeline.epoch_rejection

Different methods for artifact rejection in epoched data
can be defined in :attr:`Pipeline.epoch_rejection` as a ``{name: EpochRejection}``
dictionary of trial-rejection settings, selected through the
:ref:`state-epoch_rejection` state.

.. autosummary::
   :toctree: ../generated
   :template: class_nomethods.rst

   ManualRejection
   ChannelModelRejection

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

The empty rejection name (``epoch_rejection=''``) is always available and means
that no epoch-level rejection is applied.

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


References (re-referencing)
===========================

.. py:attribute:: Pipeline.references

EEG re-referencing applied to epochs *after* channel interpolation (so that bad
channels do not contaminate the reference). References are defined as a
``{name: reference_definition}`` dictionary and selected through the
:ref:`state-reference` state:

.. autosummary::
   :toctree: ../generated
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


State parameters
================

.. _state-raw:

``raw``
-------

Select the preprocessing pipeline applied to the continuous data. Options are
all the processing steps defined in :attr:`Pipeline.raw`, as well as
``"raw"`` for using unprocessed raw data.


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
