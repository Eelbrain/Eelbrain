.. currentmodule:: eelbrain.pipeline

***************
Evoked analysis
***************

Once preprocessing is complete, event-related analysis proceeds from condition averages (evoked responses) to group-level statistics.
This section covers subject groups, loading evoked data, and mass-univariate tests.

.. contents:: Contents
   :local:


.. _Pipeline-intro-analysis:

Analysis workflow
=================

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


.. _pipeline-groups:

Subject groups
==============

.. py:attribute:: Pipeline.groups

A subject group called ``'all'`` containing all subjects is always implicitly
defined. Additional subject groups can be defined in
:attr:`Pipeline.groups` with ``{name: group_definition}``
entries:

.. autosummary::
   :toctree: ../generated
   :template: class_nomethods.rst

   Group
   SubGroup

Example::

    groups = {
        'good': SubGroup('all', ['R0013', 'R0666']),
        'bad': Group(['R0013', 'R0666']),
    }

The current group is selected through the :ref:`state-group` state, and restricts group-level analysis to the group's subjects.
Groups are used the same way in :doc:`trf` (e.g., the ``subjects`` argument of :meth:`Pipeline.load_trfs` and the ``group`` state in :meth:`Pipeline.load_model_test`).

To compare groups statistically, add a :class:`GroupVar` to :attr:`Pipeline.variables`, and use it in a :class:`TTestIndependent` or in an
:class:`ANOVA` with subject nested in the group variable::

    class MyExperiment(Pipeline):

        groups = {
            'patient': Group(['S001', 'S002']),
            'control': Group(['S011', 'S012']),
        }
        variables = {
            'diagnosis': GroupVar(['patient', 'control']),
        }
        tests = {
            'patient=control': TTestIndependent('diagnosis', 'patient', 'control'),
        }


Tests
=====

.. py:attribute:: Pipeline.tests

Statistical tests are defined as ``{name: test_definition}`` dictionary.
This allows automatic caching of permutation test results when using :meth:`Pipeline.load_test`.
Tests are defined using the following classes:

.. autosummary::
   :toctree: ../generated
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


State parameters
================

.. _state-group:

``group``
---------

Any group defined in :attr:`Pipeline.groups`. Will restrict the analysis
to that group of subjects.


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
