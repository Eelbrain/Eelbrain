.. currentmodule:: eelbrain.pipeline

************
TRF analysis
************

Temporal response function (TRF) analysis models the M/EEG response as a continuous response to one or more predictor time series, such as the acoustic envelope of continuous speech.
The pipeline manages TRF estimation on top of the shared :doc:`preprocessing` and :doc:`source` stages:
it assembles predictors and responses, fits and caches per-subject TRFs, and provides group-level datasets and statistical model comparisons.

The main entry points are:

- :meth:`Pipeline.load_trf`: compute or load a single subject's TRF
- :meth:`Pipeline.load_trfs`: assemble TRFs and fit metrics for a group of subjects
- :meth:`Pipeline.load_model_test`: statistically compare the predictive power of two models
- :meth:`Pipeline.load_predictor`: load a predictor by itself, e.g. for inspection
- :meth:`Pipeline.show_model_terms`: display the terms in a model or comparison

.. contents:: Contents
   :local:


Overview
========

A TRF analysis is configured through three :class:`Pipeline` attributes:
:attr:`Pipeline.predictors` defines where predictor variables come from,
:attr:`Pipeline.estimators` defines how TRFs are fit,
and :attr:`Pipeline.models` defines named abbreviations for sets of predictors.
For example::

    class Experiment(Pipeline):

        epochs = {
            'story': ContinuousEpoch('listening'),
        }
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

With the :class:`Pipeline` specification, TRFs can be loaded and analyzed::

    e = Experiment("~/Data/Experiment")
    # Set general parameters
    e.set(epoch='story', raw='1-40', inv='', estimator='boosting')
    # Load a single TRF
    trf = e.load_trf('acoustic + word-frequency', -0.1, 0.5)
    # Load a group dataset with TRFs for all subjects
    trfs = e.load_trfs('all', 'acoustic', -0.1, 0.5)

Like all pipeline results, TRFs are cached: the first call to :meth:`Pipeline.load_trf` fits the model, subsequent calls load the cached result, and results are recomputed automatically when an input (preprocessing, rejection, a predictor file, …) changes.


.. _trf-predictors:

Predictors
==========

.. py:attribute:: Pipeline.predictors

Predictors are defined as a ``{name: predictor_definition}`` dictionary:

.. autosummary::
   :toctree: ../generated
   :template: class_nomethods.rst

   EventPredictor
   UTSPredictor
   NUTSPredictor
   SubjectUTSPredictor

:class:`EventPredictor` generates impulses from the events :class:`~eelbrain.Dataset` itself.
The other predictors load per-stimulus (or per-subject) files, described next.

Predictor files
---------------

File predictors are added to the experiment as files in the ``{root}/derivatives/predictors`` directory. Filenames follow this pattern: ``{stimulus}~{key}[-{variant}].pickle``.

- ``stimulus`` is an arbitrary name for the stimulus represented by this file (see `Stimuli`_ below).
- ``key`` is the key used for defining this predictor in :attr:`Pipeline.predictors`.
- ``variant`` is an optional description that allows several predictor files to use the same entry in :attr:`Pipeline.predictors`. That allows, for example, only defining a single ``gammatone`` predictor in :attr:`Pipeline.predictors` for different variations of the spectrogram (``gammatone-1``, ``gammatone-8``, ``gammatone-on-1``, etc.).

For example::

    predictors = {
        'gammatone': UTSPredictor(resample='bin'),
        'word': NUTSPredictor(),
    }

Assuming a stimulus called ``story``, this would match the following predictor files:

- ``predictors/story~gammatone-1.pickle``: an :class:`~eelbrain.NDVar` uniform time series (UTS) predictor, which can be invoked with the model term ``gammatone-1`` (see `Models`_)
- ``predictors/story~gammatone-8.pickle``: as above, but invoked with the model term ``gammatone-8``
- ``predictors/story~word.pickle``: a :class:`~eelbrain.Dataset` representing one or multiple non-uniform time series (NUTS) predictors (through different columns in the dataset). The specific model term would include a column name, for example, a model term ``word-surprisal`` would use the values of the ``"surprisal"`` column in the dataset (see :class:`NUTSPredictor`).

UTS predictors (:class:`UTSPredictor`) are stored as :class:`~eelbrain.NDVar` objects with a time axis matching the stimulus.
When an analysis is done at a lower sampling rate than the stored file, the predictor is resampled according to the definition's ``resample`` parameter.

NUTS predictors (:class:`NUTSPredictor`) are stored as :class:`~eelbrain.Dataset` objects with a ``time`` column (time stamp of each event in seconds) and further columns with event values.
When loading the predictor, they are converted to a uniform time series by placing impulses at the time stamps.
The columns to use are specified in the model term, as ``{key}-{value-column}`` or ``{key}-{value-column}-{mask-column}`` (the boolean mask column sets the value to zero wherever it is ``False``); the bare ``{key}`` invokes an intercept, i.e. a unit impulse at each time stamp.
Appending ``-step`` invokes a step function instead of impulses: the predictor holds each event's value until the next event (this requires ``ds.info['tstop']`` in the predictor file to determine the end of the last step).

Subject-specific predictors (:class:`SubjectUTSPredictor`) provide a separate predictor file for each subject, stored in ``{root}/derivatives/subject-predictors``, for example for predictors derived from a subject's own behavior. See the class documentation for the file-naming conventions.

Changed predictor files
-----------------------

Changes to predictor files are detected automatically:
when a predictor file is replaced, cached TRFs that used it become stale and are recomputed on demand.
Only the data that actually feeds a given TRF counts — for a :class:`NUTSPredictor`, only the ``time`` column and the columns invoked by the model term.
Re-saving a file with identical relevant data accordingly does not invalidate anything.

Stimuli
-------

.. py:attribute:: Pipeline.stim_var
   :type: str

In order to load the correct predictor file for a model term, the pipeline needs to know which stimulus was presented in each trial.
The :attr:`Pipeline.stim_var` attribute names the events column that identifies the stimulus (default ``'stimulus'``).
Thus, given the following events::

    #    sample    value    onset    subject   stimulus
    ---------------------------------------------------
    0    1863      1        3.726    S01       s1
    1    30672     5        61.344   S01       s2
    ...

Using the term ``gammatone`` in a model would find predictor files based on the ``stimulus`` column: ``s1~gammatone.pickle``, ``s2~gammatone.pickle``, ...

Multiple stimuli per trial
--------------------------

To look up the stimulus in an event column other than :attr:`Pipeline.stim_var`, specify the relevant column in the model term with ``~``.
For example, assume a selective attention task in which two speakers talk simultaneously.
The attended speaker (``fg``) and the unattended speaker (``bg``) can each be considered one stimulus.
In addition, the acoustic mixture of the two speakers may be considered a third stimulus (``mix``).
Events may look like this::

    #    sample    value    onset    subject   fg   bg   mix
    --------------------------------------------------------
    0    1863      1        3.726    S01       s1   s3   s13
    1    30672     5        61.344   S01       s2   s4   s24
    ...


The default stimulus could be specified as ``stim_var = 'fg'``. Other stimuli (or "streams") can then be selected in model terms with ``~``:

- ``gammatone`` would find predictors based on the default ``fg`` column: ``s1~gammatone.pickle``, ``s2~gammatone.pickle``, ...
- ``bg~gammatone`` would find predictors based on the ``bg`` column: ``s3~gammatone.pickle``, ``s4~gammatone.pickle``, ...
- ``mix~gammatone`` would find predictors based on the ``mix`` column: ``s13~gammatone.pickle``, ``s24~gammatone.pickle``, ...

Loading predictors
------------------

To inspect a predictor as it would enter the analysis, load it with :meth:`Pipeline.load_predictor`::

    >>> x = e.load_predictor('story~gammatone-8', tstep=0.01)

The ``filter_x`` parameter of the TRF methods controls whether predictors are filtered with the same filters as the M/EEG data (the :class:`RawFilter` steps of the current ``raw`` pipeline): ``True`` filters all predictors, ``'continuous'`` filters only time-continuous predictors (see the ``sampling`` parameter of :class:`UTSPredictor`).


.. _pipeline-models:

Models
======

.. py:attribute:: Pipeline.models
   :type: Dict[str, str]

A TRF model is a set of *terms*, each term specifying one predictor variable (see `Predictors`_ for how terms map to predictor definitions and files).
Models are constructed by combining terms with ``+``:

- ``x="gammatone-1"`` is a model with a single predictor
- ``x="gammatone-1 + gammatone-on-1"`` is a model with two predictors

To shorten long model specifications, named sub-models can be defined in :attr:`Pipeline.models`. For example, with::

    models = {
        'auditory': 'gammatone-8 + gammatone-on-8',
    }

the combined auditory model can then be invoked as ``auditory``, and used as part of larger models, such as ``x="auditory + word-surprisal"``.
Named models can build on other named models defined before them.

Use :meth:`Pipeline.show_model_terms` to list all terms in a model, e.g.::

    >>> e.show_model_terms('auditory')
    #   term
    ------------------
    0   gammatone-8
    1   gammatone-on-8

Lag windows
-----------

The TRF lag window is specified with the ``tstart`` and ``tstop`` arguments of the TRF methods.
Individual terms can override the model-wide window with slice syntax, e.g. ``'gammatone[0.2:] + word[-0.1:0.8]'`` — an omitted boundary uses ``tstart``/``tstop``.
A lag window applied to a named model is distributed to its member terms (explicit member lags take precedence).


.. _trf-comparisons:

Comparisons
===========

Statistical model tests compare the predictive power of two TRF models.
Basic comparisons are constructed with ``>``, ``<`` (one-tailed) and ``=`` (two-tailed):

- ``x="gammatone-1 + gammatone-on-1 > gammatone-1"`` tests whether predictive power improves when adding the ``gammatone-on-1`` predictor to a model already containing the ``gammatone-1`` predictor.
- ``x="gammatone-1 = gammatone-on-1"`` tests whether the predictive power of ``gammatone-1`` or that of ``gammatone-on-1`` is higher.
- ``x="gammatone-1 > 0"`` tests whether the predictive power of ``gammatone-1`` is higher than zero.

To simplify common tests with large models, the following shortcuts exist:

.. list-table::
   :header-rows: 1
   :widths: 10 20 25 45

   * - Shortcut
     - Example
     - Expansion
     - Description
   * - ``@``
     - ``a + b + c @ a``
     - ``a + b + c > b + c``
     - Unique contribution of ``a`` to the left-hand-side model
   * - ``+@``
     - ``b + c +@ a``
     - ``a + b + c > b + c``
     - Effect of adding ``a`` to the left-hand-side model
   * - ``@ … > …``
     - ``a + b + c @ a > b``
     - ``a + c > b + c``
     - Compare the unique contributions of ``a`` and ``b``
   * - ``+@ … > …``
     - ``c +@ a > b``
     - ``a + c > b + c``
     - Compare the effect of adding ``a`` vs. adding ``b``

Use :meth:`Pipeline.show_model_terms` to list the terms in the two models involved in a comparison::

    >>> e.show_model_terms('auditory @ gammatone-8')
    x1               x0
    -------------------------------
    gammatone-8
    gammatone-on-8   gammatone-on-8

Lag windows in comparisons
--------------------------

Terms in a comparison can carry lag windows (see `Lag windows`_), which makes it possible to test the contribution of a predictor within a specific range of lags.
When a term with a lag window is omitted with ``@``, only that window is removed from the matching term, and the reduced model keeps the complement:

- ``a + b @ b[0.2:]`` tests the contribution of ``b`` at late lags (from 0.2 s to ``tstop``); the reduced model is ``a + b[:0.2]``.
- ``a + b @ b[:0.2]`` tests the contribution of ``b`` at early lags (from ``tstart`` up to 0.2); the reduced model is ``a + b[0.2:]``.
- ``a + b @ b[:0.2] = b[0.2:]`` compares the contribution of early and late lags of ``b``.
- ``a +@ b[:0.2]`` tests the effect of adding ``b`` restricted to lags up to 0.2 s.

The omitted window has to lie within the window of a single term of the full model (``a + b[:0.3] @ b[0.2:0.5]`` is an error).
Open bounds in the omitted window take the bound of the term they are removed from, so ``a + b[0:0.5] @ b[:0.3]`` is equivalent to ``a + b[0:0.5] @ b[0:0.3]``.
Terms in the reduced model are displayed with open bounds that stand for ``tstart``/``tstop``, e.g.::

    >>> e.show_model_terms('gammatone-8 + word-surprisal @ word-surprisal[0.3:]')
    x1               x0
    -------------------------------------
    gammatone-8      gammatone-8
    word-surprisal
                     word-surprisal[:0.3]

Common comparisons
------------------

Common questions and corresponding comparisons:

.. list-table::
   :header-rows: 1
   :widths: 80 20

   * - Question
     - Comparison
   * - Is there a brain response to ``a`` when controlling for ``b``?
     - ``a + b @ a``
   * - Is there a brain response to ``a`` within 0.2 s (when controlling for ``b``)?
     - ``a + b @ a[:0.2]``
   * - Is there a brain region that represents ``a`` more than ``b``?
     - ``a > b``


.. _trf-epochs:

Epochs and predictor generation
===============================

TRFs are estimated on the data epoch selected through the :ref:`state-epoch` state, like any other analysis.
How predictors are generated depends on the type of the epoch definition (see :doc:`preprocessing` for the epoch types):

Trial-based epochs (:class:`PrimaryEpoch`, :class:`SecondaryEpoch`, :class:`SuperEpoch`)
    Each epoch corresponds to a single stimulus, identified by the epoch's entry in the stimulus column (see `Stimuli`_).
    The stimulus' predictor is resampled to the analysis sampling rate and aligned to the epoch's time axis (padded or cropped to the epoch length).
    An :class:`EventPredictor` places one impulse per epoch, at a fixed or event-dependent latency.

:class:`ContinuousEpoch`
    The epoch consists of continuous data segments that each span multiple events, with an ``epoch_time`` axis whose zero is the first selected event.
    Predictors are assembled per segment: each event's stimulus predictor is placed at the event's position on the segment's time axis (``epoch_time``).
    For a :class:`NUTSPredictor`, the per-stimulus event tables are shifted to their position and merged, so impulses (or steps, with ``-step``) form a single continuous time series per segment.
    An :class:`EventPredictor` places one impulse per event in the segment.
    This is the natural representation for continuous paradigms such as listening to stories, where trials would be arbitrary subdivisions.

:class:`EpochCollection`
    A separate TRF is estimated for each member epoch, and :meth:`Pipeline.load_trfs` returns them as separate rows (cases) in the resulting :class:`~eelbrain.Dataset`.

Epoch-level artifact rejection (see :doc:`preprocessing`) applies to TRF analysis as well; for long continuous epochs, :class:`ChannelModelRejection` can mark bad channel-windows for interpolation.


Estimation
==========

.. py:attribute:: Pipeline.estimators

Estimators are defined as a ``{name: estimator_definition}`` dictionary, and selected with the ``estimator`` parameter of the TRF methods.
The built-in ``'boosting'`` estimator is always available and can be overridden
to change its parameters.

.. autosummary::
   :toctree: ../generated
   :template: class_nomethods.rst

   Estimator
   Boosting
   NCRF

Example definitions::

    class TRFExperiment(Pipeline):

        estimators = {
            'forward': Boosting(basis=0.050, selective_stopping=1, partitions=5),
            'backward': Boosting(basis=0, backward=True, partitions=5),
            'ncrf': NCRF(),
        }

These definitions can then be invoked like::

    trfs = e.load_trfs(..., estimator='forward')

For :class:`Boosting`, the analysis *space* is determined by the :ref:`state-inv` state, as for evoked analysis:
with ``inv=''`` the TRF is fit to sensor data, with a non-empty inverse solution it is fit to source-localized data (in source space, the :ref:`state-parc` state masks the source space: sources labeled ``"unknown"`` are excluded).
The :class:`NCRF` estimator fits source currents directly from sensor data and requires ``inv=''``.

.. py:attribute:: Pipeline.default_data
   :type: str

In sensor space, the ``data`` parameter of the TRF methods selects the sensor type to fit (e.g. ``'meg'``, ``'eeg'``, or an aggregate like ``'eeg.rms'``).
:attr:`Pipeline.default_data` sets the default (if unset, ``'eeg'`` for EEG datasets and ``'meg'`` otherwise).


Group analysis and model tests
==============================

:meth:`Pipeline.load_trfs` assembles per-subject TRFs into a group-level :class:`~eelbrain.Dataset` with one row per subject (× member epoch for an :class:`EpochCollection`), containing the estimator's fit metrics and the TRF components; source-space TRFs are morphed to the common brain so that subjects are comparable::

    trfs = e.load_trfs('all', 'auditory + word-surprisal', -0.1, 0.8)

:meth:`Pipeline.load_model_test` computes a cache-managed statistical test of a model comparison (see `Comparisons`_), based on the cross-validated predictive power of the two models::

    result = e.load_model_test('auditory +@ word-surprisal', -0.1, 0.8)

The ``metric`` parameter selects the fit metric to test (by default ``'ev'``, the proportion of explained variance), and by default the comparison determines the test: a one-sample test against zero for ``x > 0``, and a related-measures test with the comparison's tail otherwise; the ``test`` parameter can name a test defined in :attr:`Pipeline.tests` instead (for example, to compare groups).
Subjects are selected through the :ref:`state-group` state.


Batch estimation and distributed fitting
========================================

The pipeline computes and caches TRFs whenever they are requested through one of the methods for accessing results; a script that requests all needed TRFs and tests (e.g., by looping over models and calling :meth:`Pipeline.load_trf`) thus serves to pre-compute them, skipping whatever is already cached.

For fitting TRFs on a machine without access to the raw data (for example, a compute cluster), :meth:`Pipeline.load_trf_job` loads the data and returns a picklable job object that carries the response and predictors; executing the job (calling it) fits the TRF and returns the result.
