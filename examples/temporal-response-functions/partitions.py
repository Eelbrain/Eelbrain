"""
.. _exa-data_split:
.. currentmodule:: eelbrain

Data partitions for boosting
============================

The boosting algorithm can use two different forms of cross-validation: cross-validation as stopping criterion (always on), and cross-validation for model evaluation (on by default, ``test=1``). Cross-validation for model evaluation ensures that fit metrics, such as the explained variance, reflect how well the model predicts data that were not used to estimate it; fit metrics computed from the same data that were used to estimate the model would overestimate its predictive power. Both forms of cross-validation require partitioning the data into different segments. The :func:`eelbrain.plot.preview_partitions` function is for exploring the effect of different parameters on the way the data is split.

.. contents:: Sections
    :local:
    :backlinks: top

Validation set
--------------
During boosting, every training step consists in modifying one element of the kernel/TRF. After every such step, the new TRF is evaluated against the validation data. To illustrate the validation set by itself, the plots in this section disable the test set (``test=0``). For continuous data (without :class:`Case` dimension), the default is then to split the data into 5 equal-length segments, and perform 5 model fits, each using one of the segments as validation set. In the plots below, each "Split" shown on the y-axis corresponds to a separate run of the boosting algorithm. The TRF returned by the :func:`boosting` function is based on the average TRF of those 5 runs.
"""
# sphinx_gallery_thumbnail_number = 4
from eelbrain import *


p = plot.preview_partitions(test=0)

###############################################################################
# The number of partitions can be controlled with the `partitions` parameter:

p = plot.preview_partitions(partitions=7, test=0)
p = plot.preview_partitions(partitions=10, test=0)

###############################################################################
# Validation and testing sets
# ---------------------------
# Testing the result of a model fit with cross-validation requires data that was never used during training. This is the default in the :func:`boosting` function (``test=1``): each partition is used in turn as testing set. For each testing set, the remaining partitions are used in different runs as training and validation data. The results of those runs are then averaged to predict responses in the testing data. In the plots, the runs are grouped by testing set (separated by black lines), and the y-axis labels the test partition. This nested loop means that the number of boosting runs can get large quickly when using many partitions (*k* partitions result in ``k * (k - 1)`` runs), so the default for continuous data is to use five partitions:

p = plot.preview_partitions()

###############################################################################
# Continuous experiments with several trials
# ------------------------------------------
# Experiments with continuous stimuli, such as stories, often consist of a few long trials, which can differ in length.
# For data with trials, the ``split`` parameter determines how partitions are formed.
# To preview partitions for a specific experiment, ``cases`` can be a list specifying the duration of each trial in seconds:

trials = [180, 240, 204, 192, 228]

###############################################################################
# By default (``split='cases'``), each partition consists of whole trials, so that trial boundaries are always partition boundaries. The number of partitions defaults to the number of trials, i.e., each trial is held out in turn. With trials of different lengths, the partitions then also differ in size:

p = plot.preview_partitions(trials, test=0)

###############################################################################
# With fewer partitions than trials, trials are assigned to partitions in turn (partition *i* receives trials ``[i::partitions]``). When the number of trials is not a multiple of the number of partitions, the partitions then differ in the number of trials, and thus substantially in size. This is a problem when the same setting has to apply to all participants of an experiment, as in a pipeline: when one participant is missing one of five trials, ``partitions=5`` is not possible with four whole trials, and ``partitions=3`` holds out twice as much data in one split as in the others. Even with the number of trials as partitions (the default), a participant with a missing trial is analyzed with a different cross-validation scheme (fewer runs, less training data per run) than the other participants:

p = plot.preview_partitions(trials[:4], partitions=3, test=0, legend=False)

###############################################################################
# With ``split='time'``, the partitions are instead contiguous spans of (approximately) equal length along the concatenated time axis. This allows using fewer partitions than trials, which is useful when the experiment has many trials, and more partitions than trials, which is necessary when the experiment has only 1 or 2 trials. A trial that straddles a partition boundary is cut into pieces. To avoid very short pieces, a partition boundary that falls within 15 s of a trial boundary is moved to that trial boundary. In :func:`boosting`, this limit is fixed at 15 s; in :func:`plot.preview_partitions` it can be changed with the ``min_section`` parameter (in seconds). In the example, two of the four partition boundaries fall within 15 s of a trial boundary and are moved there, so that the third partition consists of exactly one trial:

p = plot.preview_partitions(trials, partitions=5, test=0, split='time')

###############################################################################
# With ``split='within_cases'``, each trial is divided into ``partitions`` pieces of equal length, and each partition is formed from the corresponding piece of every trial. Every partition thus contains data from every trial, in proportion to the trial's length:

p = plot.preview_partitions(trials, partitions=5, test=0, split='within_cases')

###############################################################################
# With a test set, the validation sets are nested within each test split in the same way as for continuous data:

p = plot.preview_partitions(trials, partitions=5, split='time')

###############################################################################
# Because the scheme is specified explicitly, it does not change with the number of trials. For example, with ``split='time'`` and ``partitions=5``, a participant who is missing one of the five trials is analyzed with the same kind of cross-validation as the other participants (whereas the default, ``split='cases'``, would hold out each of the remaining four trials in turn):

p = plot.preview_partitions(trials[:4], partitions=5, test=0, split='time')

###############################################################################
# Trial-based experiments
# -----------------------
# Experiments with many short trials, each containing one event (e.g., one word), are best analyzed without ever cutting a trial. This is what the default, ``split='cases'``, does: trials are assigned to partitions in an interleaved fashion (partition *i* receives trials ``[i::partitions]``), so that each partition samples the whole experiment evenly across time:

p = plot.preview_partitions(20, partitions=5, test=0)
p = plot.preview_partitions(20, partitions=3, test=0)

###############################################################################
# When trials belong to different conditions, the ``model`` parameter makes sure that each partition samples each condition evenly, by assigning trials to partitions separately within each condition. Here, trials alternate between two conditions in blocks of three:

p = plot.preview_partitions(12, partitions=3, test=0, model=Factor('aaabbb', tile=2))

###############################################################################
# With ``split='time'``, trials that are shorter than 30 s are never cut (see above), so that each partition consists of a contiguous block of trials. In this example, trials are 10 s long:

p = plot.preview_partitions([10] * 20, partitions=5, test=0, split='time')

###############################################################################
# With a test set, the validation sets are nested in the same way as for continuous data:

p = plot.preview_partitions(20, partitions=5)

###############################################################################
# Choosing a scheme
# -----------------
# Partitions that consist of whole trials or contiguous periods of time test whether the model generalizes to a different period of the experiment. However, when the trials differ systematically (e.g., different conditions, learning or fatigue effects), a test partition that stems from a single trial may not represent the data as a whole. Partitions within trials (``split='within_cases'``) avoid this because every partition samples every trial, at the cost of more segment boundaries. For trial-based experiments with different conditions, the ``model`` parameter achieves the same without cutting trials.
