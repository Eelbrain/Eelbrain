"""
.. _exa-intro:
.. currentmodule:: eelbrain

Introduction
============

Data are represented with three primary data-objects:

* :class:`Factor` for categorial variables
* :class:`Var` for scalar variables
* :class:`NDVar` for multidimensional data (e.g. a variable measured at
  different time points)

Multiple variables belonging to the same dataset can be grouped in a
:class:`Dataset` object.

.. contents:: Contents
   :local:


Factor
------

A :class:`Factor` is a container for one-dimensional, categorial data: Each
case is described by a string label. The most obvious way to initialize a
:class:`Factor` is a list of strings:

"""
# sphinx_gallery_thumbnail_number = 3
from eelbrain import *

a = Factor(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], name='A')
a

###############################################################################
# Since Factor initialization simply iterates over the given data, the
# same Factor could be initialized with:

a = Factor('aaaabbbb', name='A')
a

###############################################################################
# There are other shortcuts to initialize factors  (see also
# the :class:`Factor` class documentation):

a = Factor(['a', 'b', 'c'], repeat=4, name='A')
a

###############################################################################
# Indexing works like for arrays:

a[0]

###############################################################################
a[0:6]

###############################################################################
# All values present in a :class:`Factor` are accessible in its
# :attr:`Factor.cells` attribute:

a.cells

###############################################################################
# Based on the Factor's cell values, boolean indexes can be generated:

a == 'a'

###############################################################################
a.isany('a', 'b')

###############################################################################
a.isnot('a', 'b')

###############################################################################
# Interaction effects can be constructed from multiple factors with the ``%``
# operator:

b = Factor(['d', 'e'], repeat=2, tile=3, name='B')
b

###############################################################################
i = a % b
i

###############################################################################
# Interaction effects are in many ways interchangeable with factors in places
# where a categorial model is required:

i.cells

###############################################################################
i == ('a', 'd')

###############################################################################
# Var
# ---
#
# The :class:`Var` class is a container for one-dimensional
# :py:class:`numpy.ndarray`:

y = Var([1, 2, 3, 4, 5, 6])
y

###############################################################################
# Indexing works as for factors

y[5]

###############################################################################
y[2:]

###############################################################################
# Many array operations can be performed on the object directly

y + 1

###############################################################################
# For any more complex operations the corresponding :py:class:`numpy.ndarray`
# can be retrieved in the :attr:`Var.x` attribute:

y.x

###############################################################################
# .. Note::
#     The :attr:`Var.x` attribute is not intended to be replaced; rather, a new
#     :class:`Var` object should be created for a new array.
#
#
# NDVar
# -----
#
# :class:`NDVar` objects are containers for multidimensional data, and manage the
# description of the dimensions along with the data. :class:`NDVar` objects are
# usually constructed automatically by an importer function (see
# :ref:`reference-io`), for example by importing data from MNE-Python through
# :mod:`load.mne`.
#
# Here we use data from a simulated EEG experiment as example:

data = datasets.simulate_erp(snr=0.5)
eeg = data['eeg']
eeg

###############################################################################
# This representation shows that ``eeg`` contains 80 trials of data (cases),
# with 140 time points and 35 EEG sensors.
#
# The object provides access to the underlying array...

eeg.x

###############################################################################
# ... and dimension descriptions:

eeg.sensor

###############################################################################
eeg.time

###############################################################################
# Eelbrain functions take advantage of the dimensions descriptions (such as
# sensor locations), for example for plotting:

p = plot.TopoButterfly(eeg, t=0.130)

###############################################################################
# :class:`NDVar` offer functionality similar to :class:`numpy.ndarray`, but
# take into account the properties of the dimensions. For example, through the
# :meth:`NDVar.sub` method, indexing can be done using meaningful descriptions,
# such as indexing a time slice in seconds ...

eeg_130 = eeg.sub(time=0.130)
p = plot.Topomap(eeg_130)
eeg_130

###############################################################################
# ... or extracting data from a specific sensor:

eeg_fz = eeg.sub(sensor='Fz')
p = plot.UTSStat(eeg_fz)
eeg_fz

###############################################################################
# Other methods allow aggregating data, for example an RMS over sensor ...

eeg_rms = eeg.rms('sensor')
plot.UTSStat(eeg_rms)
eeg_rms

###############################################################################
# ... or a mean in a time window:

eeg_average = eeg.mean(time=(0.100, 0.150))
p = plot.Topomap(eeg_average)

###############################################################################
# Dataset
# -------
#
# A :class:`Dataset` is a container for multiple variables
# (:class:`Factor`, :class:`Var` and :class:`NDVar`) that describe the same
# cases. It can be thought of as a data table with columns corresponding to
# different variables and rows to different cases.
# Consider the dataset containing the simulated EEG data used above:

data

###############################################################################
# Because this can be more output than needed, the
# :meth:`Dataset.head` method only shows the first couple of rows:

data.head()

###############################################################################
# This dataset containes severeal univariate columns: ``cloze``, ``predictability``, and ``n_chars``.
# The last line also indicates that the dataset contains an :class:`NDVar` called ``eeg``.
# The :class:`NDVar` is not displayed as column because it contains many values per row.
# In the :class:`NDVar`, the :class:`Case` dimension corresponds to the row in the dataset
# (which here corresponds to simulated trial number):

data['eeg']

###############################################################################
# The type and value range of each entry in the :class:`Dataset` can be shown using the :meth:`Dataset.summary` method:

data.summary()

###############################################################################
# An even shorter summary can be generated by the string representation:

repr(data)

###############################################################################
# Here, ``80 cases`` indicates that the Dataset contains 80 rows. The
# subsequent dictionary-like representation shows the keys and the types of the
# corresponding values (``F``: :class:`Factor`, ``V``: :class:`Var`, ``Vnd``: :class:`NDVar`).
#
# Datasets can be indexed with columnn names, ...

data['cloze']

###############################################################################
# ... row numbers, ...

data[2:5]

###############################################################################
# ... or both, in wich case row comes before column:

data[2:5, 'n_chars']

###############################################################################
# Array-based indexing also allows indexing based on the Dataset's variables:

data['n_chars'] == 3

###############################################################################
data[data['n_chars'] == 3]

###############################################################################
# :meth:`Dataset.eval` allows evaluatuing code strings in the namespace
# defined by the dataset, which means that dataset variables can be invoked
# with just their name:

data.eval("predictability == 'high'")

###############################################################################
# Many dataset methods allow using code strings as shortcuts for expressions
# involving dataset variables, for example indexing:

data.sub("predictability == 'high'").head()

###############################################################################
# Columns in the :class:`Dataset` can be used to define models, for statistics,
# aggregating and plotting.
# Any string specified as argument in those functions will be evaluated in the
# dataset, thuse, because we can use:

data.eval("eeg.sub(sensor='Cz')")

###############################################################################
# ... we can quickly plot the time course of a sensor by condition:

p = plot.UTSStat("eeg.sub(sensor='Cz')", "predictability", data=data)

###############################################################################
p = plot.UTSStat("eeg.sub(sensor='Fz')", "n_chars", data=data, colors='viridis')

###############################################################################
# Or calculate a difference wave:

data_average = data.aggregate('predictability')
data_average

###############################################################################
difference = data_average[1, 'eeg'] - data_average[0, 'eeg']
p = plot.TopoArray(difference, t=[None, None, 0.400])

###############################################################################
# For examples of how to construct datasets from scratch see :ref:`exa-dataset`.
