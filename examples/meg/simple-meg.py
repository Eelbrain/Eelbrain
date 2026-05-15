# dataset: mne_sample
"""
This example demonstrates how to load and plot MEG data

It uses the mne sample dataset which included presentaton of simple (A)uditory
and (V)isual stimuli to the (L)eft and (R)ight ear/visual hemifield, as well as
presentation of smiley faces and button presses by the subject.

All the lines starting with "print" are there just to display information about
the data and could be left out for a pure analysis script.

"""
import os
import mne
from eelbrain import *

# find the path for the mne sample data file
datapath = mne.datasets.sample.data_path()
raw_path = os.path.join(datapath, 'MEG', 'sample',
                        'sample_audvis_filt-0-40_raw.fif')

# Load the events from the samepl data file
ds = load.mne.events(raw_path)

# print the first 10 events to check what we loaded
print(ds[:10])
# check how many events of which trigger we have
print(table.frequencies('trigger', data=ds))

# retrieve the trigger variable form the dataset for easier access
trigger = ds['trigger']
# use the trigger variable to add more meaningful labels to the dataset
condition_labels = {1: 'LA', 2: 'RA', 3: 'LV', 4: 'RV', 5: 'smiley', 32: 'button'}
ds['condition'] = Factor(trigger, labels=condition_labels)
side_labels = {1: 'L', 2: 'R', 3: 'L', 4: 'R', 5: 'None', 32: 'None'}
ds['side'] = Factor(trigger, labels=side_labels)
modality_labels = {1: 'A', 2: 'A', 3: 'V', 4: 'V', 5: 'None', 32: 'None'}
ds['modality'] = Factor(trigger, labels=modality_labels)

# print the first 10 events with the new labels
print(ds[:10])
# print a nicer table with event frequencies
print(table.frequencies('side', 'modality', data=ds))
# don't print None cells
print(table.frequencies('side', 'modality', data=ds, sub="modality != 'None'"))

# Extract a subset of events for which to load data
ds_sub = ds.sub("modality != 'None'")
# this would be doing the same more explicitly:
##side = ds['side']
##index = side == 'left'
##ds_left = ds.sub(index)

# Load epochs for our selection. Baseline correct from the beginning of the
# epoch to the trigger i.e., t=0). Reject trials with peak to peak values larger
# than 3 pico tesla.
ds_sub = load.mne.add_epochs(ds_sub, -0.1, 0.6, baseline=(None, 0),
                             reject=3e-12, sysname='neuromag306mag')
# check how many events are left
print(table.frequencies('modality', data=ds_sub))

# Plot a butterfly plot with flexible topography of the grand average
plot.TopoButterfly('meg', data=ds_sub)
# plot all conditions separately ('%' is to specify an interaction)
plot.TopoButterfly('meg', 'side % modality', data=ds_sub)

# compare left and right visual stimulation in a cluster t-test (this might
# take a minute -- increase samples to 10000 for a better test)
res = testnd.TTestIndependent('meg', 'side', 'L', 'R', sub="modality == 'V'",
                              data=ds_sub, samples=100, pmin=0.05, tstart=0,
                              tstop=0.4, mintime=0.01)
# show parameters for all clusters
print(res.clusters)
# show parameters for cluster with p <= 0.1 (res.clusters is a Dataset)
print(res.clusters.sub("p <= 0.1"))
# plot the significant clusters
p = plot.TopoArray(res)
p.set_topo_ts(0.1, 0.17, 0.26)
