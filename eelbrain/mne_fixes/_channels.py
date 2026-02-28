# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Private functionality from mne.channels.channels"
import mne
from mne.io.constants import FIFF


def _adjacency_id(info: mne.Info, ch_type: str):
    "Guess system ID for channel-adjacency; based on :func:`mne.channels.find_ch_adjacency`"
    # Excerpt from mne.channels.find_ch_adjacency
    (has_vv_mag, has_vv_grad, is_old_vv, has_4D_mag, ctf_other_types, has_CTF_grad, n_kit_grads, has_any_meg, has_eeg_coils, has_eeg_coils_and_meg, has_eeg_coils_only, has_neuromag_122_grad, has_csd_coils) = _get_ch_info(info)
    conn_name = None
    if has_vv_mag and ch_type == 'mag':
        conn_name = 'neuromag306mag'
    elif has_vv_grad and ch_type == 'grad':
        conn_name = 'neuromag306planar'
    elif has_4D_mag:
        if any((key := f'MEG {i}') in info['ch_names'] for i in range(149, 249)):
            idx = info['ch_names'].index(key)
            grad = info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_MAGNES_GRAD
            mag = info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_MAGNES_MAG
            if ch_type == 'grad' and grad:
                conn_name = 'bti248grad'
            elif ch_type == 'mag' and mag:
                conn_name = 'bti248'
        elif 'MEG 148' in info['ch_names'] and ch_type == 'mag':
            idx = info['ch_names'].index('MEG 148')
            if info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_MAGNES_MAG:
                conn_name = 'bti148'
    elif has_CTF_grad and ch_type == 'mag':
        if info['nchan'] < 100:
            conn_name = 'ctf64'
        elif info['nchan'] > 200:
            conn_name = 'ctf275'
        else:
            conn_name = 'ctf151'
    # End excerpt
    return conn_name


# copied from mne.channels.channels
def _get_ch_info(info):
    """Get channel info for inferring acquisition device."""
    chs = info['chs']
    # Only take first 16 bits, as higher bits store CTF comp order
    coil_types = {ch['coil_type'] & 0xFFFF for ch in chs}
    channel_types = {ch['kind'] for ch in chs}

    has_vv_mag = any(k in coil_types for k in
                     [FIFF.FIFFV_COIL_VV_MAG_T1, FIFF.FIFFV_COIL_VV_MAG_T2,
                      FIFF.FIFFV_COIL_VV_MAG_T3])
    has_vv_grad = any(k in coil_types for k in [FIFF.FIFFV_COIL_VV_PLANAR_T1,
                                                FIFF.FIFFV_COIL_VV_PLANAR_T2,
                                                FIFF.FIFFV_COIL_VV_PLANAR_T3])
    has_neuromag_122_grad = any(k in coil_types
                                for k in [FIFF.FIFFV_COIL_NM_122])

    is_old_vv = ' ' in chs[0]['ch_name']

    has_4D_mag = FIFF.FIFFV_COIL_MAGNES_MAG in coil_types
    ctf_other_types = (FIFF.FIFFV_COIL_CTF_REF_MAG,
                       FIFF.FIFFV_COIL_CTF_REF_GRAD,
                       FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD)
    has_CTF_grad = (FIFF.FIFFV_COIL_CTF_GRAD in coil_types
                    or (FIFF.FIFFV_MEG_CH in channel_types
                        and any(k in ctf_other_types for k in coil_types)))
    # hack due to MNE-C bug in IO of CTF
    # only take first 16 bits, as higher bits store CTF comp order
    n_kit_grads = sum(ch['coil_type'] & 0xFFFF == FIFF.FIFFV_COIL_KIT_GRAD
                      for ch in chs)

    has_any_meg = any([has_vv_mag, has_vv_grad, has_4D_mag, has_CTF_grad,
                       n_kit_grads])
    has_eeg_coils = (FIFF.FIFFV_COIL_EEG in coil_types
                     and FIFF.FIFFV_EEG_CH in channel_types)
    has_eeg_coils_and_meg = has_eeg_coils and has_any_meg
    has_eeg_coils_only = has_eeg_coils and not has_any_meg
    has_csd_coils = (FIFF.FIFFV_COIL_EEG_CSD in coil_types
                     and FIFF.FIFFV_EEG_CH in channel_types)

    return (has_vv_mag, has_vv_grad, is_old_vv, has_4D_mag, ctf_other_types,
            has_CTF_grad, n_kit_grads, has_any_meg, has_eeg_coils,
            has_eeg_coils_and_meg, has_eeg_coils_only, has_neuromag_122_grad,
            has_csd_coils)
