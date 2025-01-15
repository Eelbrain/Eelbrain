"""
.. _exa-lm:

Volume source space
===================

Basic operations for volume source space vector data.
"""
from eelbrain import *


# Load the dataset
data = datasets.get_mne_sample(src='vol', ori='vector')
# Auditory stimuli to left or right ear:
data.head()

###############################################################################
# Set the parcellation
data['src'] = set_parc(data['src'], 'aparc+aseg')

###############################################################################
# Show the labels that are in the parcellation
data['src'].source.parc.cells

###############################################################################
# Extract the amplitude time course in the left auditory cortex:
# Subset of data in transverse temporal gyrus
# Vector length (norm)
# Mean in the ROI
data['a1l'] = data['src'].sub(source='ctx-lh-transversetemporal').norm('space').mean('source')

###############################################################################
# Plot source time course by side of auditory stimulus
p = plot.UTSStat('a1l', 'side', data=data, title='STC in left A1')
