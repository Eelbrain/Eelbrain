"""Pure state-to-path helpers for graph-managed experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mne_bids import BIDSPath


BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')
BIDS_ENTITY_PREFIX_MAP = {
    'subject': 'sub',
    'session': 'ses',
    'task': 'task',
    'acquisition': 'acq',
    'run': 'run',
    'split': 'split',
}
BIDS_PATH_KEYS = ('datatype', 'suffix', 'extension', *BIDS_ENTITY_KEYS)


def _state_value(state: dict[str, str | None], key: str) -> str | None:
    value = state.get(key)
    return None if value in (None, '') else value


def _bids_name(
        state: dict[str, str | None],
        entity_keys: tuple[str, ...],
        *,
        suffix: str | None,
) -> str:
    parts = []
    for key in BIDS_ENTITY_KEYS:
        if key not in entity_keys:
            continue
        value = _state_value(state, key)
        if value is not None:
            parts.append(f"{BIDS_ENTITY_PREFIX_MAP[key]}-{value}")
    if suffix:
        parts.append(suffix)
    return '_'.join(parts)


def bids_path(
        state: dict[str, str | None],
        *,
        noise: bool = False,
) -> BIDSPath:
    kwargs = {key: _state_value(state, key) for key in BIDS_PATH_KEYS}
    path = BIDSPath(root=state['root'], **kwargs)
    if noise:
        return path.find_empty_room()
    else:
        return path


def subject_session_basename(state: dict[str, str | None]) -> str:
    return _bids_name(state, ('subject', 'session'), suffix=state['suffix'])


def raw_basename(state: dict[str, str | None]) -> str:
    return _bids_name(state, ('subject', 'session', 'acquisition', 'task', 'run', 'split'), suffix=state['suffix'])


def epoch_basename(state: dict[str, str | None]) -> str:
    return _bids_name(state, ('subject', 'session', 'acquisition', 'run', 'split'), suffix=state['suffix'])


def test_basename(state: dict[str, str | None]) -> str:
    return _bids_name(state, ('session', 'run', 'split'), suffix=state['suffix'])


def deriv_dir(state: dict[str, str | None]) -> Path:
    return Path(state['root']) / 'derivatives'


def raw_dir(state: dict[str, str | None]) -> Path:
    path = Path(state['root']) / f"sub-{state['subject']}"
    if state.get('session'):
        path /= f"ses-{state['session']}"
    return path / state['datatype']


def cache_dir(state: dict[str, str | None]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'cache'


def log_dir(state: dict[str, str | None]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'logs'


def results_dir(state: dict[str, str | None]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'results'


def methods_dir(state: dict[str, str | None]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'methods'


def raw_cache_dir(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'raw' / subject_session_basename(state)


def ica_file_path(
        state: dict[str, str | None],
        *,
        raw: str = 'ica',
) -> Path:
    return deriv_dir(state) / 'ica' / f"{epoch_basename(state)}_raw-{raw}_ica.fif"


def cached_raw_file_path(state: dict[str, str | None]) -> Path:
    return raw_cache_dir(state) / f"{raw_basename(state)}_raw-{state['raw']}.fif"


def event_file_path(state: dict[str, str | None]) -> Path:
    return raw_cache_dir(state) / f"{raw_basename(state)}_raw-{state['raw']}_evts.pickle"


def selected_events_file_path(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'selected-events' / f"{epoch_basename(state)}_raw-{state['raw']}_epoch-{state['epoch']}_rej-{state['rej']}_sel.pickle"


def evoked_file_path(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'evoked' / f"{epoch_basename(state)}_raw-{state['raw']}_epoch-{state['epoch']}_rej-{state['rej']}_model-{state['model']}_count-{state['equalize_evoked_count']}_ave.fif"


def epochs_file_path(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'epochs' / f"{epoch_basename(state)}_raw-{state['raw']}_epoch-{state['epoch']}_rej-{state['rej']}_epo.pickle"


def trans_file_path(state: dict[str, str | None]) -> Path:
    return deriv_dir(state) / 'trans' / f"{subject_session_basename(state)}_trans.fif"


def rej_file_path(
        state: dict[str, str | None],
        *,
        epoch: str | None = None,
        rej: str | None = None,
) -> Path:
    epoch_name = state['epoch'] if epoch is None else epoch
    rej_name = state['rej'] if rej is None else rej
    return deriv_dir(state) / 'eelbrain' / 'epoch selection' / f"{epoch_basename(state)}_raw-{state['raw']}_epoch-{epoch_name}_rej-{rej_name}_epoch.pickle"


def cov_file_path(state: dict[str, str | None]) -> Path:
    return raw_cache_dir(state) / f"{epoch_basename(state)}_raw-{state['raw']}_cov-{state['cov']}_rej-{state['rej']}_cov.fif"


def cov_info_file_path(state: dict[str, str | None]) -> Path:
    return raw_cache_dir(state) / f"{epoch_basename(state)}_raw-{state['raw']}_cov-{state['cov']}_rej-{state['rej']}_info.txt"


def mri_sdir(state: dict[str, str | None]) -> Path:
    return deriv_dir(state) / 'freesurfer'


def mri_dir(state: dict[str, str | None]) -> Path:
    return mri_sdir(state) / state['mrisubject']


def bem_dir(state: dict[str, str | None]) -> Path:
    return mri_dir(state) / 'bem'


def bem_file_path(state: dict[str, str | None]) -> Path:
    return bem_dir(state) / f"{state['mrisubject']}-inner_skull-bem.fif"


def src_file_path(state: dict[str, str | None]) -> Path:
    return bem_dir(state) / f"{state['mrisubject']}-{state['src']}-src.fif"


def source_morph_file_path(state: dict[str, str | None]) -> Path:
    return bem_dir(state) / f"{state['mrisubject']}-{state['common_brain']}-{state['src']}-morph.h5"


def epochs_stc_file_path(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'epochs-stc' / f"{epoch_basename(state)}_mrisubject-{state['mrisubject']}_src-{state['src']}_raw-{state['raw']}_cov-{state['cov']}_rej-{state['rej']}_inv-{state['inv']}_stc.pickle"


def evoked_stc_file_path(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'evoked-stc' / f"{epoch_basename(state)}_mrisubject-{state['mrisubject']}_src-{state['src']}_raw-{state['raw']}_cov-{state['cov']}_rej-{state['rej']}_model-{state['model']}_count-{state['equalize_evoked_count']}_inv-{state['inv']}_stc.pickle"


def label_dir(state: dict[str, str | None]) -> Path:
    return mri_dir(state) / 'label'


def annot_file_path(state: dict[str, str | None], hemi: str) -> Path:
    return label_dir(state) / f'{hemi}.{state["parc"]}.annot'


def annot_stamp_path(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'annot' / state['mrisubject'] / f'{state["parc"]}.stamp'


def fwd_file_path(state: dict[str, str | None]) -> Path:
    return raw_cache_dir(state) / f"{epoch_basename(state)}_mrisubject-{state['mrisubject']}_src-{state['src']}_fwd.fif"


def inv_file_path(state: dict[str, str | None]) -> Path:
    return raw_cache_dir(state) / f"{epoch_basename(state)}_mrisubject-{state['mrisubject']}_src-{state['src']}_raw-{state['raw']}_cov-{state['cov']}_rej-{state['rej']}_inv-{state['inv']}_inv.fif"


def test_dir(state: dict[str, str | None]) -> Path:
    return cache_dir(state) / 'test'


def time_str(t) -> str:
    if t is None:
        return ''
    return f'{round(t * 1000):d}'


def time_window_str(window, delim='-') -> str:
    return delim.join(map(time_str, window))


def _slug(value: Any) -> str:
    return str(value).replace('/', '-').replace(' ', '_')


def join_stem_parts(*parts: Any) -> str:
    out = []
    for part in parts:
        if part in (None, '', False):
            continue
        if isinstance(part, (list, tuple)):
            out.extend(_slug(item) for item in part if item not in (None, '', False))
        else:
            out.append(_slug(part))
    return '_'.join(out)


def _short_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def test_result_cache_path(
        state: dict[str, str | None],
        stem: str,
        key: dict[str, Any],
) -> Path:
    group = _slug(state.get('group') or 'all')
    return test_dir(state) / group / f'{stem}_key-{_short_hash(key)}.pickle'


def report_export_path(
        state: dict[str, str | None],
        report_kind: str,
        stem: str,
        single_subject: bool = False,
) -> Path:
    if single_subject:
        return results_dir(state) / report_kind / 'subjects' / _slug(state['subject']) / f'{stem}.html'
    return results_dir(state) / report_kind / 'groups' / _slug(state['group']) / f'{stem}.html'


def movie_export_path(
        state: dict[str, str | None],
        stem: str,
        single_subject: bool = False,
) -> Path:
    if single_subject:
        return results_dir(state) / 'movies' / 'subjects' / _slug(state['subject']) / f'{stem}.mov'
    return results_dir(state) / 'movies' / 'groups' / _slug(state['group']) / f'{stem}.mov'


def coreg_report_path(state: dict[str, str | None]) -> Path:
    title = 'Coregistration'
    if state.get('group') != 'all':
        title += f" {state['group']}"
    if state.get('mri'):
        title += f" {state['mri']}"
    return methods_dir(state) / f'{title}.html'
