.. currentmodule:: eelbrain.pipeline

**************
Data structure
**************

The pipeline reads its input from a `BIDS (Brain Imaging Data Structure) <https://bids.neuroimaging.io/>`_ dataset.
This section describes the expected file structure, how the pipeline discovers the dataset's contents, and the settings that control reading raw files and events.

.. contents:: Contents
   :local:


.. _Pipeline-filestructure:

The BIDS file structure
=======================

The pipeline expects the input dataset in BIDS format. (To convert your data into BIDS format, use the `MNE-BIDS <https://mne.tools/mne-bids/stable/use.html>`_ library.) In the schema below, curly brackets indicate slots that the pipeline will replace with specific names::


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
    TRF predictors                          /predictors
    subject-specific TRF predictors         /subject-predictors
    Eelbrain generated files                /eelbrain


.. note::
    In BIDS specification, ``{root}/derivatives`` is for files that do not fit into the BIDS structure, such as FreeSurfer MRIs and Eelbrain-generated files.


``{subject}``, ``{session}``, ``{task}``, ``{acquisition}``, and ``{run}`` are `BIDS entities <https://bids-specification.readthedocs.io/en/stable/appendices/entities.html>`_. ``{session}``, ``{acquisition}``, and ``{run}`` are optional. ``{datatype}`` is inferred by the pipeline from the data files, and can be ``'meg'`` or ``'eeg'``. There can be other entities depending on the dataset, such as `split <https://bids-specification.readthedocs.io/en/stable/appendices/entities.html#split>`_.


``MRI`` files (including ``trans-file``) are optional and only needed for source localization. The ``{root}/derivatives/freesurfer`` directory is a `FreeSurfer <https://surfer.nmr.mgh.harvard.edu>`_ subject directory. See :doc:`source` for how these files are created and used.

The ``{root}/derivatives/predictors`` and ``{root}/derivatives/subject-predictors`` directories hold predictor files for TRF analysis; see :doc:`trf` for their naming conventions.


Scanning the dataset
====================

A BIDS dataset is scanned by initializing a :class:`Pipeline` with the data ``{root}`` location, for example::

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


Excluding files
===============

.. py:attribute:: Pipeline.ignore_entities
   :type: Dict[str, list[str]]

Exclude certain entities from the experiment.
Keys correspond to the ``ignore_...`` parameters of `mne_bids.get_entity_vals <https://mne.tools/mne-bids/stable/generated/mne_bids.get_entity_vals.html>`_, e.g.::

    ignore_entities = {
        'ignore_subjects': ['S666', 'S999'],
        'ignore_sessions': ['02'],
    }


Reading files
=============

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

.. py:attribute:: Pipeline.preload
   :type: bool

Whether to preload raw data into memory before creating epochs. Default is ``False``. It is observed that in some datasets reading raw data when creating epochs is time consuming, and in these cases setting ``preload=True`` can speed up epoch creation.

The MEG system used to acquire the data determines the sensor neighborhood graph
(adjacency). This is usually detected automatically; when it needs to be set
explicitly, define a ``'raw'`` entry with a :class:`RawSource` in
:attr:`Pipeline.raw` and set its ``sysname`` (and/or ``adjacency``) parameter.
For example, for data from NYU New York::

    raw = {
        'raw': RawSource(sysname='KIT-157'),
        '1-40': RawFilter('raw', 1, 40),
    }


State parameters
================

.. _state-subject:

``subject``
-----------

Any subject in the experiment.
A simple way to cycle through subjects when performing a manual step for every subject is :meth:`Pipeline.next`; to loop through subjects in a script, iterate over the pipeline itself (``for subject in e: ...``).


.. _state-session:

``session``
-----------

Which session to work with.


.. _state-task:

``task``
--------

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
