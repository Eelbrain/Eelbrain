"""
.. _exa-lm:

Volume source space
===================

Basic analysis of volume source space vector data.

.. contents:: Contents
   :local:

Dataset
^^^^^^^
Use the :mod:`mne` sample data:
"""
from eelbrain import *


# Load the dataset
data = datasets.get_mne_sample(src='vol', ori='vector')
# Auditory stimuli to left or right ear:
data.head()

###############################################################################
# Set the parcellation.
# For reference, show the labels that are in the parcellation
data['src'] = set_parc(data['src'], 'aparc+aseg')
data['src'].source.parc.cells

###############################################################################
# One-sample test
# ^^^^^^^^^^^^^^^
# A one-sample test can be used to detect significant activations

result = testnd.Vector('src', sub="side == 'R'", data=data, samples=1000, tfce=True, tstart=0.050)

###############################################################################
# A butterfly plot can give a quick overview of amplitudes over time.
# With a window-based :mod:`matplotlib` backend, the time can be adjusted
# interactively. In a notebook, use different :class:`plot.GlassBrain` plots.
# (see next section):

from eelbrain_plotly_viz import EelbrainPlotly2DViz

# Create visualization with sample data
viz = EelbrainPlotly2DViz(result.difference)

# Run interactive dashboard
viz.run()

""
# butterfly, brain = plot.GlassBrain.butterfly(result)
# brain.set_time(0.090)

###############################################################################
# Amplitude in ROI
# ^^^^^^^^^^^^^^^^
# Extract the amplitude time course in the left auditory cortex:
# Subset of data in transverse temporal gyrus
# Vector length (norm)
# Mean in the ROI
data['a1l'] = data['src'].sub(source='ctx-lh-transversetemporal').norm('space').mean('source')

###############################################################################
# Plot source time course by side of auditory stimulus
p = plot.UTSStat('a1l', 'side', data=data, title='STC in left A1')

###############################################################################
# Directional ROI
# ^^^^^^^^^^^^^^^
# An alternative is to project the signal onto a vector to extract signed
# time course data. First, define an ROI with vector data in a desired
# anatomical region:
roi_data = result.difference.sub(source='ctx-lh-transversetemporal', time=0.090)
plot.GlassBrain(roi_data)

###############################################################################
# Then, project all data onto this vector:
data['a1l_vec'] = roi_data.dot(data['src'], ('space', 'source'))
p = plot.UTSStat('a1l_vec', 'side', data=data, title='STC in left A1')
