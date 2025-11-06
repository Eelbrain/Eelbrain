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
from pathlib import Path

from eelbrain import *
import mne


# Load the dataset
data = datasets.get_mne_sample(src='vol', ori='vector')
# Auditory stimuli to left or right ear:
data.head()

###############################################################################
# Set the parcellation.
# For reference, show the labels that are in the parcellation
data['src'] = set_parc(data['src'], 'aparc+aseg')
data['src'].source.parc.cells[:3]

###############################################################################
# One-sample test
# ^^^^^^^^^^^^^^^
# A one-sample test can be used to detect significant activations

result = testnd.Vector('src', sub="side == 'R'", data=data, samples=250, tfce=True, tstart=0.05, tstop=0.200)

###############################################################################
# In a notebook, start an interactive vizualization using `LiveNeuron <https://github.com/liang-bo96/LiveNeuron>`_ (uncomment the code below):

# from eelbrain_plotly_viz import EelbrainPlotly2DViz

# viz = EelbrainPlotly2DViz(result.difference, layout_mode='horizontal', realtime=True, arrow_scale=0.2)
# viz.show_in_jupyter()

###############################################################################
# A butterfly plot can give a quick overview of amplitudes over time.
# In an interactive iPython session, a combination of butterfly and 
# anatomical plot can be used with a window-based :mod:`matplotlib` 
# backend, where the time can be adjusted interactively: 

# butterfly, brain = plot.GlassBrain.butterfly(result)
# brain.set_time(0.090)

###############################################################################
# For static visualization, we can use a combination of :class:`plot.Butterfly` and :class:`plot.GlassBrain` plots.

y = result.masked_difference()

# mne.datasets.fetch_fsaverage(subjects_dir)
fname_src_fsaverage = Path(y.source.subjects_dir) / "fsaverage" / "bem" / "fsaverage-vol-5-src.fif"
src_fs = mne.read_source_spaces(fname_src_fsaverage)
morph = mne.compute_source_morph(
    y.source.get_source_space(),
    subject_from=y.source.subject,
    subjects_dir=y.source.subjects_dir,
    niter_affine=[10, 10, 5],
    niter_sdr=[10, 10, 5],  # just for speed
    src_to=src_fs,
    verbose=True,
)

""
# Morph to average brain for visualization
y = morph_source_space(y, 'fsaverage', morph=morph)

""
# Extract vector norm (amplitude)
y_norm = y.norm('space')
# Split data by hemisphere
butterfly_data = [y_norm.sub(source=hemi, name=hemi.capitalize()) for hemi in ['lh', 'rh']]
# Buterfly plot
p = plot.Butterfly(butterfly_data, axh=2, axw=3)
# Mark time points for anatomical visualization
times = [0.090, 0.160]
for t in times:
    p.add_vline(t)

""
# Glassbrain plots at the relevant time points
for t in times:
    p = plot.GlassBrain(y.sub(time=t), title=f"{t*1000:.0f} ms", vmax=4)

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
peak_time = 0.085
p = plot.UTSStat('a1l', 'side', data=data, title='STC in left A1')
p.add_vline(peak_time)

###############################################################################
# Directional ROI
# ^^^^^^^^^^^^^^^
# An alternative is to project the signal onto a vector to extract signed
# time course data. First, define an ROI with vector data in a desired
# anatomical region:
roi_data = result.difference.sub(source='ctx-lh-transversetemporal', time=0.090)
p = plot.GlassBrain(roi_data)

###############################################################################
# Then, project all data onto this vector:
data['a1l_vec'] = roi_data.dot(data['src'], ('space', 'source'))
p = plot.UTSStat('a1l_vec', 'side', data=data, title='STC in left A1')
