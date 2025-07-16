"""Denoising source separation"""

# Authors: Daniel McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)
# flake8: noqa

import numpy as np
from mne import Epochs, EpochsArray, compute_covariance


def dss(data, data_max_components=None, data_thresh=0,
        bias_max_components=None, bias_thresh=0, return_data=True):
    """Process physiological data with denoising source separation (DSS)

    Implementation follows the procedure described in Särelä & Valpola [1]_
    and de Cheveigné & Simon [2]_.

    Parameters
    ----------
    data : instance of Epochs | array of shape (n_trials, n_channels, n_times)
        Data to be denoised.
    data_max_components : int | None
        Maximum number of components to keep during PCA decomposition of the
        data. ``None`` (the default) keeps all suprathreshold components.
    data_thresh : float | None
        Threshold (relative to the largest component) above which components
        will be kept during decomposition of the data. The default keeps all
        non-zero values; to keep all values, specify ``thresh=None``.
    bias_max_components : int | None
        Maximum number of components to keep during PCA decomposition of the
        bias function. ``None`` (the default) keeps all suprathreshold
        components.
    bias_thresh : float | None
        Threshold (relative to the largest component) below which components
        will be discarded during decomposition of the bias function. ``None``
        (the default) keeps all non-zero values; to keep all values, pass
        ``thresh=None`` and ``max_components=None``.
    return_data : bool
        Whether to return the denoised data along with the denoising matrix.

    Returns
    -------
    dss_mat : array of shape (n_dss_components, n_channels)
        The denoising matrix. Apply to data via ``np.dot(dss_mat, ep)``, where
        ``ep`` is an epoch of shape (n_channels, n_samples).
    dss_data : array of shape (n_trials, n_dss_components, n_samples)
        The denoised data. Note that the DSS components are orthogonal virtual
        channels and may be fewer in number than the number of channels in the
        input Epochs object. Returned only if ``return_data`` is ``True``.

    References
    ----------
    .. [1] Särelä, Jaakko, and Valpola, Harri (2005). Denoising source
    separation. Journal of Machine Learning Research 6: 233–72.

    .. [2] de Cheveigné, Alain, and Simon, Jonathan Z. (2008). Denoising based
    on spatial filtering. Journal of Neuroscience Methods, 171(2): 331-339.
    """
    if isinstance(data, (Epochs, EpochsArray)):
        data_cov = compute_covariance(data).data
        bias_cov = np.cov(data.average().pick_types(meg=True, eeg=True, ref_meg=False).data)
        if return_data:
            data = data.get_data()
    elif isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError('Data to denoise must have shape '
                             '(n_trials, n_channels, n_times).')
        data_cov = np.sum([np.dot(trial, trial.T) for trial in data], axis=0)
        bias_cov = np.cov(data.mean(axis=0))
    else:
        raise TypeError('Data to denoise must be an instance of mne.Epochs or '
                        'a numpy array.')
    dss_mat = _dss(data_cov, bias_cov, data_max_components, data_thresh,
                   bias_max_components, bias_thresh)
    if return_data:
        # next line equiv. to: np.array([np.dot(dss_mat, ep) for ep in data])
        dss_data = np.einsum('ij,hjk->hik', dss_mat, data)
        return dss_mat, dss_data
    else:
        return dss_mat


def _dss(data_cov, bias_cov, data_max_components=None, data_thresh=None,
         bias_max_components=None, bias_thresh=None):
    """Process physiological data with denoising source separation (DSS)

    Acts on covariance matrices; allows specification of arbitrary bias
    functions (as compared to the public ``dss`` function, which forces the
    bias to be the evoked response).
    """
    data_eigval, data_eigvec = _pca(data_cov, data_max_components, data_thresh)
    W = np.sqrt(1 / data_eigval)  # diagonal of whitening matrix
    # bias covariance projected into whitened PCA space of data channels
    bias_cov_white = (W * data_eigvec).T.dot(bias_cov).dot(data_eigvec) * W
    # proj. matrix from whitened data space to a space maximizing bias fxn
    bias_eigval, bias_eigvec = _pca(bias_cov_white, bias_max_components,
                                    bias_thresh)
    # proj. matrix from data to bias-maximizing space (DSS space)
    dss_mat = (W[np.newaxis, :] * data_eigvec).dot(bias_eigvec)
    # normalize DSS dimensions
    N = np.sqrt(1 / np.diag(dss_mat.T.dot(data_cov).dot(dss_mat)))
    return (N * dss_mat).T


def _pca(cov, max_components=None, thresh=0):
    """Perform PCA decomposition

    Parameters
    ----------
    cov : array-like
        Covariance matrix
    max_components : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).
    thresh : float | None
        Threshold (relative to the largest component) above which components
        will be kept. The default keeps all non-zero values; to keep all
        values, specify ``thresh=None`` and ``max_components=None``.

    Returns
    -------
    eigval : array
        1-dimensional array of eigenvalues.
    eigvec : array
        2-dimensional array of eigenvectors.
    """

    if thresh is not None and (thresh > 1 or thresh < 0):
        raise ValueError('Threshold must be between 0 and 1 (or None).')
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.abs(eigval)
    sort_ix = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, sort_ix]
    eigval = eigval[sort_ix]
    if max_components is not None:
        eigval = eigval[:max_components]
        eigvec = eigvec[:, :max_components]
    if thresh is not None:
        suprathresh = np.where(eigval / eigval.max() > thresh)[0]
        eigval = eigval[suprathresh]
        eigvec = eigvec[:, suprathresh]
    return eigval, eigvec
