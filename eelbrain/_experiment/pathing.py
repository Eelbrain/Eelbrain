"""Pure state-to-path helpers for graph-managed experiment artifacts."""

from __future__ import annotations

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


def _state_value(state: dict[str, Any], key: str) -> str | None:
    value = state.get(key)
    return None if value in (None, '') else value


def _bids_name(
        state: dict[str, Any],
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
        state: dict[str, Any],
        *,
        noise: bool = False,
) -> BIDSPath:
    kwargs = {key: _state_value(state, key) for key in BIDS_PATH_KEYS}
    path = BIDSPath(root=state['root'], **kwargs)
    if noise:
        return path.find_empty_room()
    else:
        return path


def subject_session_basename(state: dict[str, Any]) -> str:
    return _bids_name(state, ('subject', 'session'), suffix=state['suffix'])


def raw_basename(state: dict[str, Any]) -> str:
    return _bids_name(state, ('subject', 'session', 'acquisition', 'task', 'run', 'split'), suffix=state['suffix'])


def epoch_basename(state: dict[str, Any]) -> str:
    return _bids_name(state, ('subject', 'session', 'acquisition', 'run', 'split'), suffix=state['suffix'])


def test_basename(state: dict[str, Any]) -> str:
    return _bids_name(state, ('session', 'run', 'split'), suffix=state['suffix'])


def deriv_dir(state: dict[str, Any]) -> Path:
    return Path(state['root']) / 'derivatives'


def raw_dir(state: dict[str, Any]) -> Path:
    path = Path(state['root']) / f"sub-{state['subject']}"
    if state.get('session'):
        path /= f"ses-{state['session']}"
    return path / state['datatype']


def cache_dir(state: dict[str, Any]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'cache'


def log_dir(state: dict[str, Any]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'logs'


def results_dir(state: dict[str, Any]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'results'


def methods_dir(state: dict[str, Any]) -> Path:
    return deriv_dir(state) / 'eelbrain' / 'methods'


def ica_file_path(
        state: dict[str, Any],
        *,
        raw: str = 'ica',
) -> Path:
    return deriv_dir(state) / 'ica' / f"{epoch_basename(state)}_raw-{raw}_ica.fif"


def trans_file_path(state: dict[str, Any]) -> Path:
    return deriv_dir(state) / 'trans' / f"{subject_session_basename(state)}_trans.fif"


def rej_file_path(
        state: dict[str, Any],
        *,
        epoch: str | None = None,
        rej: str | None = None,
) -> Path:
    epoch_name = state['epoch'] if epoch is None else epoch
    rej_name = state['rej'] if rej is None else rej
    return deriv_dir(state) / 'eelbrain' / 'epoch selection' / f"{epoch_basename(state)}_raw-{state['raw']}_epoch-{epoch_name}_rej-{rej_name}_epoch.pickle"


def mri_sdir(state: dict[str, Any]) -> Path:
    return deriv_dir(state) / 'freesurfer'


def mri_dir(state: dict[str, Any]) -> Path:
    return mri_sdir(state) / state['mrisubject']


def bem_dir(state: dict[str, Any]) -> Path:
    return mri_dir(state) / 'bem'


def bem_file_path(state: dict[str, Any]) -> Path:
    return bem_dir(state) / f"{state['mrisubject']}-inner_skull-bem.fif"


def src_file_path(state: dict[str, Any]) -> Path:
    return bem_dir(state) / f"{state['mrisubject']}-{state['src']}-src.fif"


def label_dir(state: dict[str, Any]) -> Path:
    return mri_dir(state) / 'label'


def annot_file_path(state: dict[str, Any], hemi: str) -> Path:
    return label_dir(state) / f'{hemi}.{state["parc"]}.annot'


def annot_stamp_path(state: dict[str, Any]) -> Path:
    return cache_dir(state) / 'annot' / state['mrisubject'] / f'{state["parc"]}.stamp'


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


def report_export_path(
        state: dict[str, Any],
        report_kind: str,
        stem: str,
        single_subject: bool = False,
) -> Path:
    if single_subject:
        return results_dir(state) / report_kind / 'subjects' / _slug(state['subject']) / f'{stem}.html'
    return results_dir(state) / report_kind / 'groups' / _slug(state['group']) / f'{stem}.html'


def movie_export_path(
        state: dict[str, Any],
        stem: str,
        single_subject: bool = False,
) -> Path:
    if single_subject:
        return results_dir(state) / 'movies' / 'subjects' / _slug(state['subject']) / f'{stem}.mov'
    return results_dir(state) / 'movies' / 'groups' / _slug(state['group']) / f'{stem}.mov'


def coreg_report_path(state: dict[str, Any]) -> Path:
    title = 'Coregistration'
    if state.get('group') != 'all':
        title += f" {state['group']}"
    if state.get('mri'):
        title += f" {state['mri']}"
    return methods_dir(state) / f'{title}.html'
