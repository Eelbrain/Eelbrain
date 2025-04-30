.. image:: https://img.shields.io/conda/vn/conda-forge/eelbrain.svg
   :target: https://github.com/christianbrodbeck/Eelbrain/wiki/Installing

.. image:: https://img.shields.io/conda/pn/conda-forge/eelbrain.svg
   :target: https://anaconda.org/conda-forge/eelbrain

.. image:: https://zenodo.org/badge/3651023.svg
   :target: https://zenodo.org/badge/latestdoi/3651023

|

Eelbrain |version|
==================
`Open-source <https://github.com/christianbrodbeck/Eelbrain>`_ `Python <https://www.python.org>`_ toolkit for MEG and EEG data analysis, including:

* Mass-univariate :ref:`permutation tests<examples>`
* `Temporal Response Function <https://doi.org/10.7554/eLife.85012>`_ estimator for fitting single-trial time series models
* :ref:`Pipeline<experiment-class-guide>` for M/EEG group level analysis
* Interactive plots and :ref:`GUIs<ref-guis>` for ICA and trial selection


Manual
------

.. toctree::
   :maxdepth: 1

   installing
   getting_started
   changes
   publications
   development

.. toctree::
   :maxdepth: 2

   reference
   auto_examples/index
   recipes
   experiment


.. seealso::
   - `GitHub <http://github.com/christianbrodbeck/Eelbrain>`_: `Wiki <https://github.com/christianbrodbeck/Eelbrain/wiki>`_, `Discussions <https://github.com/christianbrodbeck/Eelbrain/discussions>`_ and `Issues <https://github.com/christianbrodbeck/Eelbrain/issues>`_
   - `Mailing list <https://groups.google.com/forum/#!forum/eelbrain>`_ for announcements
   - Eelbrain on the `Python Package Index <https://pypi.python.org/pypi/eelbrain>`_

To cite Eelbrain, find the appropriate `DOI <https://doi.org/10.5281/zenodo.598150>`_.
Eelbrain relies on
`NumPy <http://www.numpy.org>`_,
`Numba <http://numba.pydata.org>`_,
`Cython <http://cython.org>`_,
`SciPy <http://scipy.org>`_,
`Matplotlib <http://matplotlib.org>`_,
`MNE-Python <http://martinos.org/mne/stable/index.html>`_,
`PySurfer <http://pysurfer.github.io>`_,
`WxPython <http://wxpython.org>`_,
and incorporates icons from the `Tango Desktop Project <http://tango.freedesktop.org>`_.

----

Eelbrain is maintained by `Christian Brodbeck <http://christianbrodbeck.net>`_
at `McMaster University <https://www.mcmaster.ca>`_.
Current funding:
`NIH <https://www.nih.gov>`_ `1R01DC020961-01A1 <https://reporter.nih.gov/search/Y2Ps2Pe_KkGmlGTE_W2dtw/project-details/10804052>`_ (2024-);
`NIH <https://www.nih.gov>`_ `1R01MH132660-01A1 <https://reporter.nih.gov/search/Y2Ps2Pe_KkGmlGTE_W2dtw/project-details/10864540>`_ (2024-);
`NSF <http://nsf.gov>`_ `2207770 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=2207770>`_ (2023-);
Past funding:
`NSF <http://nsf.gov>`_ `2043903 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=2043903>`_ (2021-2024);
`NSF <http://nsf.gov>`_ `1754284 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1754284>`_ (2020-2021);
`NIH <https://www.nih.gov>`_ R01-DC-014085 (2016-2020);
`NYU Abu Dhabi Institute <http://nyuad.nyu.edu/en/research/nyuad-institute.html>`_ G1001 (2011-2016).
