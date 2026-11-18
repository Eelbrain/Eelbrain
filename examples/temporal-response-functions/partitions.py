"""
.. _exa-data_split:
.. currentmodule:: eelbrain

Data partitions for cross-validation
====================================

TRF estimation algorithms can use two different forms of cross-validation: cross-validation for optimizing regularization, and cross-validation for model evaluation. Cross-validation for model evaluation ensures that fit metrics, such as the explained variance, reflect how well the model predicts data that were not used to estimate it; fit metrics computed from the same data that were used to estimate the model would overestimate its predictive power. Both forms of cross-validation require partitioning the data into different segments. The :func:`eelbrain.plot.preview_partitions` function is for exploring the effect of different parameters on the way the data is split. When working with a :class:`~pipeline.Pipeline`, use :meth:`Pipeline.show_partitions <pipeline.Pipeline.show_partitions>` to inspect or preview partitioning for a specific dataset.

.. contents:: Sections
    :local:
    :backlinks: top

Validation set
--------------
During boosting, every training step consists in modifying one element of the kernel/TRF. After every such step, the new TRF is evaluated against the validation data. To illustrate the validation set by itself, the plots in this section disable the test set (``test=0``). The default is then to split the data into 5 contiguous segments of equal length, and perform 5 model fits, each using one of the segments as validation set. In the plots below, each "Split" shown on the y-axis corresponds to a separate run of the boosting algorithm. The TRF returned by the :func:`boosting` function is based on the average TRF of those 5 runs.
"""
from eelbrain import *


p = plot.preview_partitions(test=0, legend='upper center')

###############################################################################
# The number of partitions can be controlled with the `partitions` parameter:

p = plot.preview_partitions(partitions=7, test=0, legend=False)
p = plot.preview_partitions(partitions=10, test=0, legend=False)

###############################################################################
# Validation and testing sets
# ---------------------------
# Testing the result of a model fit with cross-validation requires data that was never used during training. This is the default in the :func:`boosting` function (``test=1``): each partition is used in turn as testing set. For each testing set, the remaining partitions are used in different runs as training and validation data. The results of those runs are then averaged to predict responses in the testing data. In the plots, the runs are grouped by testing set (separated by black lines), and the y-axis labels the test partition. This nested loop means that the number of boosting runs can get large quickly when using many partitions (*k* partitions result in ``k * (k - 1)`` runs), so the default is to use five partitions:

p = plot.preview_partitions(legend='upper center')

###############################################################################
# Continuous experiments with several trials
# ------------------------------------------
# Experiments with continuous stimuli, such as audiobooks, often consist of a few long trials, which can differ in length.
# For data with trials, the ``split`` parameter determines how partitions are formed.
# To preview partitions for a specific experiment, ``cases`` can be a list specifying the duration of each trial in seconds:

trials = [180, 240, 204, 192, 228]

###############################################################################
# By default (``split='time'``), the partitions are contiguous spans of (approximately) equal length along the concatenated time axis. A trial that straddles a partition boundary is cut into pieces. To avoid very short pieces, a partition boundary that falls within 15 s of a trial boundary is moved to that trial boundary. This limit can be changed with the ``min_section`` parameter (in seconds). In the example, two of the four partition boundaries fall within 15 s of a trial boundary and are moved there, so that the third partition consists of exactly one trial:

p = plot.preview_partitions(trials, test=0, legend='upper center')

###############################################################################
# With ``split='cases'``, each partition instead consists of whole trials, so that trial boundaries are always partition boundaries. With as many partitions as trials, each trial is held out in turn. With trials of different lengths, the partitions then also differ in size:

p = plot.preview_partitions(trials, partitions=5, test=0, split='cases', legend=False)

###############################################################################
# With fewer partitions than trials, trials are assigned to partitions in turn (partition *i* receives trials ``[i::partitions]``). When the number of trials is not a multiple of the number of partitions, the partitions then differ in the number of trials:

p = plot.preview_partitions(trials, partitions=3, test=0, split='cases', legend=False)

###############################################################################
# With ``split='within-cases'``, each trial is divided into ``partitions`` pieces of equal length, and each partition is formed from the corresponding piece of every trial. Every partition thus contains data from every trial, in proportion to the trial's length:

p = plot.preview_partitions(trials, partitions=5, test=0, split='within-cases', legend=False)

###############################################################################
# With a test set, the validation sets are nested within each test split in the same way as for continuous data:

p = plot.preview_partitions(trials, legend=False)

###############################################################################
# None of these schemes depend on the number of trials. For example, with the default, a participant who is missing one of the five trials is analyzed with the same kind of cross-validation as the other participants:

p = plot.preview_partitions(trials[:4], test=0, legend=False)

###############################################################################
# Trial-based experiments
# -----------------------
# Experiments with many short trials, each containing one event (e.g., one word), are best analyzed without ever cutting a trial. This is what ``split='cases'`` does: trials are assigned to partitions in an interleaved fashion (partition *i* receives trials ``[i::partitions]``), so that each partition samples the whole experiment evenly across time:

p = plot.preview_partitions(20, partitions=5, test=0, split='cases', legend=False)
p = plot.preview_partitions(20, partitions=3, test=0, split='cases', legend=False)

###############################################################################
# When trials belong to different conditions, the ``model`` parameter makes sure that each partition samples each condition evenly, by assigning trials to partitions separately within each condition:

p = plot.preview_partitions(12, partitions=3, test=0, split='cases', model=Factor('AABB', tile=3), legend='fig')

###############################################################################
# With the default, ``split='time'``, trials that are shorter than 30 s are never cut (see above), so that each partition consists of a contiguous block of trials. In this example, trials are 10 s long:

p = plot.preview_partitions([10] * 20, test=0, legend=False)

###############################################################################
# With a test set, the validation sets are nested in the same way as for continuous data:

p = plot.preview_partitions(20, partitions=5, split='cases', legend=False)

###############################################################################
# Choosing a scheme
# -----------------
# Partitions that consist of whole trials or contiguous periods of time test whether the model generalizes to a different period of the experiment. However, when the trials differ systematically (e.g., different conditions, learning or fatigue effects), a test partition that stems from a single trial may not represent the data as a whole. Partitions within trials (``split='within-cases'``) avoid this because every partition samples every trial, at the cost of more segment boundaries. For trial-based experiments with different conditions, the ``model`` parameter achieves the same without cutting trials.

###############################################################################
# Walk-forward validation
# -----------------------
# In all the splits above, the validation data is surrounded by training data. With ``scheme='walk-forward'``, each split instead trains only on the data preceding its validation set. With ``split='within-cases'``, each trial is divided into ``partitions`` partitions, of which all but the first are short windows at the end of the trial (by default each 5% of the trial). These windows are held out one at a time, resulting in ``partitions - 1`` splits.

p = plot.preview_partitions(1, partitions=4, test=0, split='within-cases', scheme='walk-forward', legend='upper center')
p = plot.preview_partitions(2, partitions=4, test=0, split='within-cases', scheme='walk-forward', legend=False)

###############################################################################
# With this scheme, a ``validate`` value between 0 and 1 sets the length of each validation window as proportion of the trial:

p = plot.preview_partitions(1, partitions=4, validate=0.1, test=0, split='within-cases', scheme='walk-forward', legend='fig')

###############################################################################
# With ``split='time'`` (the default), the windows are instead at the end of the data as a whole, so that each split trains on everything that was recorded before its validation set:

p = plot.preview_partitions(trials, partitions=4, validate=0.1, test=0, split='time', scheme='walk-forward', legend=False)
