# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import mne
import numpy
from numpy.testing import assert_array_equal
import pytest
import scipy.linalg

from eelbrain._experiment.covariance import RawCovariance, _block_spectra, bound_condition_number


def _covariance(
        info: mne.Info,
        rank: int,
        condition: float,
        scale: float = 1e-10,
) -> mne.Covariance:
    """Rank-deficient covariance whose retained eigenvalues span ``condition`` down from ``scale``."""
    n = len(info['ch_names'])
    rotation = numpy.linalg.qr(numpy.random.RandomState(0).normal(size=(n, n)))[0]
    eig = numpy.zeros(n)
    eig[:rank] = scale * numpy.logspace(0, -numpy.log10(condition), rank)
    return mne.Covariance(rotation @ numpy.diag(eig) @ rotation.T, info['ch_names'], [], [], nfree=100000)


def _meg_info(maxwell: bool) -> mne.Info:
    info = mne.create_info([f'MEG{i:03d}' for i in range(30)], 1000., ['mag'] * 10 + ['grad'] * 20)
    if maxwell:
        # the SSS record that makes mne treat magnetometers and gradiometers as one block
        with info._unlock():
            info['proc_history'] = [{'max_info': {'sss_info': {'in_order': 8, 'components': numpy.ones(80, int)}}}]
    return info


def _meg_covariance(info: mne.Info) -> mne.Covariance:
    "Block-diagonal covariance with ill-conditioned magnetometers and well-conditioned gradiometers"
    mag = _covariance(mne.pick_info(info, mne.pick_types(info, meg='mag')), 8, 1e8, 1e-26)
    grad = _covariance(mne.pick_info(info, mne.pick_types(info, meg='grad')), 16, 1e2, 1e-24)
    return mne.Covariance(scipy.linalg.block_diag(mag.data, grad.data), info['ch_names'], [], [], nfree=100000)


@pytest.fixture
def eeg_info():
    return mne.create_info([f'EEG{i:03d}' for i in range(60)], 1000., 'eeg')


def test_bound_condition_number(eeg_info):
    "Regularization is applied only where needed, and hits the requested condition number"
    cov = _covariance(eeg_info, 40, 1e10)
    out, meta = bound_condition_number(cov, eeg_info, 1e6)
    assert meta['condition']['eeg'] == pytest.approx(1e10, rel=1e-3)
    assert meta['regularization']['eeg'] > 0
    retained = _block_spectra(out, eeg_info)['eeg']
    assert retained[0] / retained[-1] == pytest.approx(1e6, rel=1e-6)
    # the rank is preserved: regularization stays inside the retained subspace
    assert mne.compute_rank(out, info=eeg_info) == mne.compute_rank(cov, info=eeg_info) == {'eeg': 40}


def test_bound_condition_number_not_needed(eeg_info):
    "A covariance that already satisfies max_condition is returned untouched"
    cov = _covariance(eeg_info, 40, 1e2)
    out, meta = bound_condition_number(cov, eeg_info, 1e6)
    assert out is cov
    assert meta['condition']['eeg'] == pytest.approx(1e2, rel=1e-3)
    assert 'regularization' not in meta


def test_bound_condition_number_meg_blocks():
    "Without Maxwell filtering, magnetometers and gradiometers are bounded separately, as mne.cov.regularize treats them"
    info = _meg_info(maxwell=False)
    cov = _meg_covariance(info)
    out, meta = bound_condition_number(cov, info, 1e6)
    assert meta['condition']['mag'] == pytest.approx(1e8, rel=1e-3)
    assert meta['condition']['grad'] == pytest.approx(1e2, rel=1e-3)
    assert list(meta['regularization']) == ['mag']
    retained = _block_spectra(out, info)
    assert retained['mag'][0] / retained['mag'][-1] == pytest.approx(1e6, rel=1e-6)
    assert_array_equal(retained['grad'], _block_spectra(cov, info)['grad'])


def test_bound_condition_number_maxwell():
    "After Maxwell filtering, magnetometers and gradiometers form one block"
    info = _meg_info(maxwell=True)
    cov = _meg_covariance(info)
    out, meta = bound_condition_number(cov, info, 1e6)
    assert list(meta['condition']) == ['meg']
    assert meta['condition']['meg'] == pytest.approx(1e10, rel=1e-3)
    retained = _block_spectra(out, info)['meg']
    assert retained[0] / retained[-1] == pytest.approx(1e6, rel=1e-6)


def test_bound_condition_number_disabled(eeg_info):
    "max_condition=0 disables the check entirely"
    cov = _covariance(eeg_info, 40, 1e10)
    out, meta = bound_condition_number(cov, eeg_info, 0)
    assert out is cov
    assert meta == {}


def test_bound_condition_number_diagonal():
    "A diagonal (ad-hoc) covariance is well conditioned by construction and passed through"
    info = mne.create_info([f'MEG{i:03d}' for i in range(6)], 1000., 'mag')
    cov = mne.cov.make_ad_hoc_cov(info)
    out, meta = bound_condition_number(cov, info, 1e6)
    assert out is cov
    assert meta == {}


def test_bound_condition_number_unhandled_channel_type():
    "A channel type that is not measured here must not silently get mne.cov.regularize's default"
    info = mne.create_info([f'SEEG{i:03d}' for i in range(10)], 1000., 'seeg')
    cov = _covariance(info, 8, 1e10)
    with pytest.raises(NotImplementedError):
        bound_condition_number(cov, info, 1e6)


@pytest.mark.parametrize('max_condition', [1., 0.5, -1])
def test_max_condition_must_exceed_one(max_condition):
    "A condition number below 1 is unsatisfiable and would solve to a negative regularization"
    with pytest.raises(ValueError):
        RawCovariance(max_condition=max_condition)
