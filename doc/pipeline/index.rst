.. currentmodule:: eelbrain.pipeline

.. _experiment-class-guide:

***********************************
The :class:`Pipeline`
***********************************

The :class:`Pipeline` implements group-level M/EEG analysis workflows:

#. Preprocessing
#. Epoching
#. Source localization
#. Evoked responses and mass univariate group-level statistics
#. Temporal response function (TRF) analysis

The input to the pipeline is a BIDS dataset containing raw M/EEG data files and, optionally, MRI files for source localization.
The pipeline automatizes the analysis, and provides an interface for preprocessing steps that require user intervention like ICA.
It allows access to the data at intermediate stages, to allow for customizing the analysis.
It caches intermediate results to make access to these data fast and efficient.

This guide is organized in the order of a typical analysis.

.. seealso::
     - :class:`Pipeline` class reference for details on all available methods

.. toctree::
   :caption: Detailed Contents
   :maxdepth: 2

   introduction
   data
   preprocessing
   source
   evoked
   trf
