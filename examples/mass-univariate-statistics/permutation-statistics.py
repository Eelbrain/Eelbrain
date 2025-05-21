# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Permutation statistics
======================

.. currentmodule:: eelbrain

Eelbrain implents three methods for estimating null-distributions in permutation tests:
max-statistic, cluster-mass and TFCE.
This example illustrates these using a simple *t*-test.
For simulating more complex datasets see other examples in this section.

For the sake of speed, the tests here are based on 1000 permutations of the data (``samples=1000``).
For precise *p*-values, 10000 permutations (the default) are preferable.

.. contents:: Contents
   :local:

"""
# sphinx_gallery_thumbnail_number = 2
from eelbrain import *
import scipy.stats


data = datasets.simulate_erp()
eeg = data['eeg']
p = plot.TopoButterfly(eeg, t=0.13, head_radius=0.35)

###############################################################################
# Max statistic
# -------------
# The max statistic is simply the maximum across the entire *t*-map. 
# The permutation distribution consist of ``max(t)`` for each permutation. 
# This is the default, and is also the fastest test.

result = testnd.TTestOneSample(eeg, samples=1000)

###############################################################################
# Visualize the permutation distribution.
# We consider values significant if they are larger than 95% of the permutation
# distribution, so we look for the 95th percentile:

significance_threshold = scipy.stats.scoreatpercentile(result.permutation_distribution, 95)
p = plot.Histogram(
    result.permutation_distribution, title='Distribution of $max(t)$',
    xlabel='max(t)', ylabel='Number of permutations')
p.add_vline(significance_threshold, color='red', linestyle='--')
_ = p.axes[0].text(significance_threshold, 40, ' 95%', color='red')

###############################################################################
# In the actual comparison, any data point with a *t*-value that is more extreme
# than 95% of the permutation distribution is considered signifiacant:

p = plot.TopoButterfly(result, t=0.13, head_radius=0.35)

###############################################################################
# Significant regions can be retrieved as :class:`Dataset` of :class:`NDVar` for further analysis (e.g., for defining ROIs).
# This function will find all contiguous regions in which *p* ≤ .05:

significan_regions = result.find_clusters(0.05, maps=True)
significan_regions

###############################################################################
p = plot.TopoButterfly(significan_regions[0, 'cluster'], t=0.13, head_radius=0.35)

###############################################################################
# Cluster-based tests
# -------------------
# In cluster-based tests, a first steps consists in finding contiguous regions of "meaningful" effect, so-called clusters.
# In order to find contiguous regions, the algorithm needs to know which channels are
# neighbors. This information is refered to as the sensor adjacency (i.e., which sensors
# are connected). The adjacency graph can be visualized to confirm that it is set correctly.

p = plot.SensorMap(eeg, adjacency=True)

###############################################################################
# We also need to define what constitutes a "meaningful" effect, i.e., an effect that should be included in the cluster.
# This can be defined based on the magnitude of the statistic (e.g., a *t*-value).
# It is commonly set to a *t* value equivalent to an uncorrected *p*-vale of .05.
# This is specified with the ``pmin=.05`` argument.
#
# A summary statistic is then computed for each cluster. In Eelbrain this is the *cluster-mass* statistic:
# the sum of all values in the cluster
# (i.e., the sum of the *t* values at all datapoints that are part of the cluster).
# The clustering procedure is repeated in each permutation, and the largest cluster-mass value is retained, 
# to derive a permutation distribution for cluster-mass values expected under the null hypothesis.
# Each cluster in the actual data can then be evaluated against this distribution.

result = testnd.TTestOneSample(eeg, samples=1000, pmin=.05)
p = plot.TopoButterfly(result, t=0.13, head_radius=0.35)

###############################################################################
# Compared to the ``max(t)`` approach above:
#
# 1) Clusters tend to be larger, because an uncorrected *p* = .05 (the cluster forming threshold) usually corresponds to a lower *t*-value than a corrected *p* = .05.
# 2) A second, later cluster emerged, showing that this test can be more powerful when effects are temporally and spatially extended.
#
# Because clusters are formed before computing the permutation distribution, there are usually many non-significant clusters (we don not want to list them all here):

result.find_clusters().head()

###############################################################################
# List only significant clusters
result.find_clusters(pmin=0.05)

###############################################################################
# Threshold-free cluster enhancement (TFCE)
# -----------------------------------------
# TFCE is an image processing algorithm that enhances cluster-like features, but without setting an arbitrary threshold.
# It thus combines the advantage the max statistic approach (not having to set an arbitrary threshold) with the advantage of the cluster-based test of increased sensitivity for effects that are extended in space and time. 

result = testnd.TTestOneSample(eeg, samples=1000, tfce=True)
p = plot.TopoButterfly(result, t=0.13, head_radius=0.35)

###############################################################################
# Result representation is analogous to the max statistic approach, with the addition that the TFCE map can be visualized:

p = plot.TopoButterfly(result.tfce_map, t=0.13, head_radius=0.35)
