.. currentmodule:: eelbrain.pipeline

*******************
Source localization
*******************

Estimating brain-level currents from sensor data requires a structural MRI (or a scaled template brain), a coregistration between the MRI and the sensor positions, a noise covariance estimate, and an inverse solution.
This section covers all of these; source localization applies equally to :doc:`evoked` and :doc:`trf` (both are performed in source space by setting the :ref:`state-inv` state).

.. contents:: Contents
   :local:


MRIs
====

The ``{root}/derivatives/freesurfer`` directory is a `FreeSurfer <https://surfer.nmr.mgh.harvard.edu>`_ subject directory (see :ref:`Pipeline-filestructure`).
Each subject's MRI directory either contains the files created by FreeSurfer's `recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_ command, or is created by the MNE-Python coregistration utility as a scaled copy of the template brain. An ``fsaverage`` folder is used to store the template brain; if it is missing, the **MRI** task of the :ref:`pipeline-gui` offers to download it automatically.

See the wiki for more information on using `structural MRIs <https://github.com/Eelbrain/Eelbrain/wiki/Coregistration%3A-Structural-MRI>`_ or the `fsaverage template brain <https://github.com/Eelbrain/Eelbrain/wiki/Coregistration%3A-Template-Brain>`_.

.. py:attribute:: Pipeline.mri_subjects
   :type: Dict[str, Dict[str, str]]

Map MEG/EEG subjects to FreeSurfer MRI subjects. Keys in ``mri_subjects`` are names for different mappings and correspond to values of the :ref:`state-mri` state parameter; the inner dictionaries map :ref:`state-subject` values to MRI subject names (i.e., directory names under ``{root}/derivatives/freesurfer``). By default, an identity mapping is used (each subject uses their own MRI directory), but custom mappings can be defined, for example to let several subjects share a template brain or to point to individually scaled MRI subjects, e.g.::

    mri_subjects = {
        '': {  # default identity mapping
            'S001': 'S001',
            'S002': 'S002',
        },
        'fsaverage': {  # all subjects use the template brain
            'S001': 'fsaverage',
            'S002': 'fsaverage',
        },
    }


Coregistration
==============

A ``trans-file`` records the coregistration of the MRI with the head shape in the raw data file.
It is created with the MNE-Python coregistration utility, most conveniently through the **Coregistration** task of the :ref:`pipeline-gui`:
double-clicking a subject row opens the MNE coregistration GUI pre-loaded with the subject's raw file and, if one already exists, the current transformation.
For subjects without a FreeSurfer reconstruction the GUI opens against the template brain, so that MNE's "Scale MRI" feature can be used to create a scaled copy.


Noise covariance
================

Source estimation with MNE methods requires an estimate of the sensor noise covariance matrix.
By default, this is estimated from the data epoch called ``'cov'``, which is defined like any other epoch in :attr:`Pipeline.epochs` (see :doc:`preprocessing`); a common choice is the pre-stimulus baseline::

    epochs = {
        'picture': PrimaryEpoch('words', "stimulus == 'picture'"),
        'cov': SecondaryEpoch('picture', tmax=0),
    }

The regularization applied to the covariance matrix is controlled through the :ref:`state-cov` state.

.. _Pipeline-intro-cov:

Empty room noise covariance
---------------------------

To use empty room data for estimating the noise covariance, follow these steps:

- Set up empty room data according to the `instruction in BIDS specification <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetoencephalography.html#empty-room-meg-recordings>`_.
- Use the empty room covariance through :ref:`state-cov` with ``e.set(cov='emptyroom')``.


Inverse solution
================

The inverse solution is selected through the :ref:`state-inv` state, most conveniently with :meth:`Pipeline.set_inv`, and the source space through :ref:`state-src`.
Once these are set, source-space data is loaded by the same methods that load sensor-space data — for example, :meth:`Pipeline.load_evoked` and :meth:`Pipeline.load_epochs` return source estimates when ``inv`` is set (``src_baseline`` and ``morph`` parameters), and TRFs are fit in source space (see :doc:`trf`).


Parcellations (:attr:`Pipeline.parcs`)
======================================

.. py:attribute:: Pipeline.parcs

A parcellation determines how the brain surface is divided into regions.
A number of standard parcellations are automatically defined (see
:ref:`state-parc` below). Additional parcellations can be defined in
the :attr:`Pipeline.parcs` dictionary with ``{name: parc_definition}``
entries.


.. autosummary::
   :toctree: ../generated
   :template: class_nomethods.rst

   SubParc
   CombinationParc
   SeededParc
   IndividualSeededParc
   FreeSurferParc
   FSAverageParc


Visualization defaults
======================

.. py:attribute:: Pipeline.brain_plot_defaults

The :attr:`Pipeline.brain_plot_defaults` dictionary can contain options
that change defaults for brain plots. The following options are available:

surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
    Freesurfer surface to use as brain geometry.
views : :class:`str` | iterator of :class:`str`
    View or views to show in the figure. Can also be set for each parcellation,
    see :attr:`Pipeline.parcs`.
foreground : mayavi color
    Figure foreground color (i.e., the text color).
background : mayavi color
    Figure background color.
smoothing_steps : ``None`` | :class:`int`
    Number of smoothing steps to display data.


State parameters
================

.. _state-mri:

``mri``
-------

Selects a mapping from MEG/EEG subjects to MRI subjects defined in
:attr:`Pipeline.mri_subjects`. The default (``''``) is the identity mapping, in
which each subject uses their own MRI directory.


.. _state-cov:

``cov``
-------

The method for correcting the sensor covariance.

'noreg'
    Use raw covariance as estimated from the data (do not regularize).
'bestreg' (default)
    Find the regularization parameter that leads to optimal whitening of the
    baseline.
'reg'
    Use the default regularization parameter (0.1).
'auto'
    Use automatic selection of the optimal regularization method, as described in :func:`mne.compute_covariance`.
'emptyroom'
    Empty room covariance; for required setup, see :ref:`Pipeline-intro-cov`.
'ad_hoc'
    Use diagonal covariance based on :func:`mne.make_ad_hoc_cov`.


.. _state-src:

``src``
-------

The source space to use.

 - ``ico-x``: Surface source space based on icosahedral subdivision of the
   white matter surface ``x`` steps (e.g., ``ico-4``, the default).
 - ``vol-x``: Volume source space based on a volume grid with ``x`` mm
   resolution (``x`` is the distance between sources, e.g. ``vol-10`` for a
   10 mm grid).


.. _state-inv:

``inv``
-------

What inverse solution to use for source localization.
``inv`` can be set with :meth:`Pipeline.set_inv`,
which has a detailed description of the options.
``inv`` can also be set directly using the appropriate string,
e.g., ``e.set(inv='fixed-6-MNE')``.
To determine the string corresponding to a given set of parameters,
use :meth:`Pipeline.inv_str`. For example::

    >>> Pipeline.inv_str('fixed', snr=6, method='MNE')
    'fixed-6-MNE'

Consequently, the following two are equivalent for setting ``inv``::

    >>> Pipeline.set_inv('fixed', snr=6, method='MNE')
    >>> Pipeline.set(inv='fixed-6-MNE')


.. _state-parc:

``parc`` (parcellations)
------------------------

The parcellation determines how the brain surface is divided into regions.
Parcellations included with FreeSurfer can directly be used:

- FreeSurfer Parcellations: ``aparc.a2005s``, ``aparc.a2009s``, ``aparc``, ``aparc.DKTatlas``, ``PALS_B12_Brodmann``, ``PALS_B12_Lobes``, ``PALS_B12_OrbitoFrontal``, ``PALS_B12_Visuotopic``.

Additional parcellation can be defined in the :attr:`Pipeline.parcs`
attribute. Parcellations are used in different contexts:

- When loading source space data, the current ``parc`` state determines the parcellation of the source space (change the state parameter with ``e.set(parc='aparc')``).
- When loading tests, the ``parc`` state masks the source space: all named
  labels are treated as one connected surface, and any sources labeled as
  ``"unknown"`` are discarded. For example, loading a test with
  ``parc='PALS_B12_Lobes'`` will perform a whole-brain test on the cortex, while
  discarding subcortical sources. Setting ``disconnect_labels=True`` instead
  treats each label as a separate ROI, so that for spatial cluster-based tests
  no clusters can cross the boundary between two labels.

Parcellations are set with their name, with the exception of
:class:`SeededParc`: for those, the name is followed by the radius in mm, for
example, to use seeds defined in a parcellation named ``'myparc'`` with a radius
of 25 mm around the seed, use ``e.set(parc='myparc-25')``.

A few additional parcellations that provide homogeneous masks are included
for backwards compatibility. For future work, it is recommended to build
such masks from ``aparc`` or another parcellation with more fine-grained
subdivision into labels.

- ``cortex``: All sources in cortex, based on the FreeSurfer "cortex" label.
- ``lobes``: Modified version of ``PALS_B12_Lobes`` in which the limbic lobe is merged into the other 4 lobes.
- ``lobes-op``: One large region encompassing occipital and parietal lobe in each hemisphere.
- ``lobes-ot``: One large region encompassing occipital and temporal lobe in each hemisphere.


.. _state-adjacency:

``adjacency``
-------------

Possible values: ``''``, ``'link-midline'``

Adjacency refers to the edges connecting data channels (sensors for sensor
space data and sources for source space data). These edges are used to find
clusters in cluster-based permutation tests. For source spaces, the default is
to use FreeSurfer surfaces in which the two hemispheres are unconnected. By
setting ``adjacency='link-midline'``, this default adjacency can be
modified so that the midline gyri of the two hemispheres get linked at sources
that are at most 15 mm apart. This parameter currently does not affect sensor
space adjacency.
