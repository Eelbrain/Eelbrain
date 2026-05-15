"""
Model coding
============

Illustrates how to inspect coding of regression models.
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *
from matplotlib import pyplot


ds = Dataset()
ds['A'] = Factor(['a1', 'a0'], repeat=4)
ds['B'] = Factor(['b1', 'b0'], repeat=2, tile=2)
ds.head()

###############################################################################
# Create a fixed effects model:
m = ds.eval('A * B')
m

###############################################################################
# Show the model using dummy coding:
m.as_table()

###############################################################################
# Create random effects model:
ds['subject'] = Factor(['s1', 's2'], tile=4, name='subject', random=True)
m = ds.eval('A * B * subject')
m
###############################################################################
# Show the model using dummy coding:
m.as_table()

###############################################################################
# Or with effect coding:

m.as_table('effect')

###############################################################################
# Plot model matrix:
figure, axes = pyplot.subplots(1, 2, figsize=(6, 3))
for ax, coding in zip(axes, ['dummy', 'effect']):
    array, names = m.array(coding)
    ax.imshow(array, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(coding)
    ax.set_xticks([i - 0.5 for i in range(len(names))], names, rotation=-60, ha='left')
