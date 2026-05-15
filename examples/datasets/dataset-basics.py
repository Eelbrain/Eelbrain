"""
.. _exa-dataset:

Dataset basics
==============

.. contents:: Contents
   :local:

Load and prepare an example dataset:
"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from eelbrain import *
import numpy
import pandas


df = pandas.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/psych/Tal_Or.csv')
data = Dataset.from_dataframe(df)
data['cond'] = data['cond'].as_factor({0: 'low', 1: 'high'})
data['gender'] = data['gender'].as_factor({1: 'male', 2: 'female'})

###############################################################################
# Inspecting datasets
# -------------------
#
# The whole dataset can be displayed like any variable in iPython (in a plain text environment, use ``print(data)``).
# For larger datasets it can be more convenient to print only the first few cases...

data.head()

###############################################################################
# ... or a summary of variables:
data.summary()

###############################################################################
# Individual rows and columns can be retrieved with common indexing:

data[10:15]

###############################################################################

data[2]

###############################################################################

data['age']

###############################################################################
# Using datasets in functions
# ---------------------------
#
# Datasets collect information describing the same cases (rows) on different variables (columns).
# This can simplify calling functions that combine information from multiple columns.
# Columns can be supplied as strings, and the dataset in the ``data`` parameter:

table.frequencies('cond', 'gender', data=data)

###############################################################################

p = plot.Scatter('pmi', 'age', 'gender', data=data, w=3, legend=(.65, .2), alpha=.4)

###############################################################################
# These strings cannot only be keys, but they can be Python code that can be evaluated in the dataset.
# For example, if this is possible:

data.eval('age < 40')  # equivalent to `data['age'] < 40`

###############################################################################
# Then, this can be used directly for plotting:

p = plot.Scatter('pmi', 'age', 'gender', sub="age < 40", data=data, w=3, legend=(.65, .4), alpha=.4)

###############################################################################
# As in other cases, ``%`` is used to specify interaction between categorial variables:

p = plot.Barplot('age', 'cond % gender', data=data, w=3)

###############################################################################
# And ``*`` expands to main effects plus interaction:

test.ANOVA('age', 'cond * gender', data=data)

###############################################################################
# Constructing datasets
# ---------------------
#
# While datasets can be imported from external data sources, it is also often convenient to store new data in a table on the fly.
#
# A dataset can be constructed column by column, by adding one variable after
# another:

# initialize an empty Dataset:
ds = Dataset()
# numeric values are added as Var object:
ds['y'] = Var(numpy.random.normal(0, 1, 6))
# categorical data as represented in Factors:
ds['a'] = Factor(['a', 'b', 'c'], repeat=2)
# A variable that's equal in all cases can be assigned quickly:
ds[:, 'z'] = 0.
# check the result:
ds

###############################################################################
# An alternative way of constructing a dataset is case by case (i.e., row by
# row):

rows = []
for i in range(6):
    subject = f'S{i}'
    y = numpy.random.normal(0, 1)
    a = 'abc'[i % 3]
    rows.append([subject, y, a])
ds = Dataset.from_caselist(['subject', 'y', 'a'], rows, random='subject')
ds
