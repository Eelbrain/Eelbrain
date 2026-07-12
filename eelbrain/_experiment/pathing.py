"""Pure state-to-path helpers for graph-managed experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mne_bids import BIDSPath


BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run')
BIDS_ENTITY_PREFIX_MAP = {
    'subject': 'sub',
    'session': 'ses',
    'task': 'task',
    'acquisition': 'acq',
    'run': 'run',
}
DERIV_DIR = Path('derivatives')
CACHE_DIR = DERIV_DIR / 'eelbrain' / 'cache'
LOG_DIR = DERIV_DIR / 'eelbrain' / 'logs'
RESULTS_DIR = DERIV_DIR / 'eelbrain' / 'results'
METHODS_DIR = DERIV_DIR / 'eelbrain' / 'methods'
MRI_SDIR = DERIV_DIR / 'freesurfer'


def _state_value(state: dict[str, Any], key: str) -> str | None:
    value = state.get(key)
    return None if value in (None, '') else value


def _bids_name(
        state: dict[str, Any],
        entity_keys: tuple[str, ...],
        suffix: str | None = None,
        *,
        datatype: str,
) -> str:
    parts = []
    for key in BIDS_ENTITY_KEYS:
        if key not in entity_keys:
            continue
        value = _state_value(state, key)
        if value is not None:
            parts.append(f"{BIDS_ENTITY_PREFIX_MAP[key]}-{value}")
    if suffix is None:
        suffix = datatype
    if suffix:
        parts.append(suffix)
    return '_'.join(parts)


def bids_path(root: Path, state: dict[str, Any], extension: str, *, datatype: str, suffix: str = None, noise: bool = False) -> BIDSPath:
    kwargs = {key: _state_value(state, key) for key in BIDS_ENTITY_KEYS}
    path = BIDSPath(
        root=root,
        suffix=suffix or datatype,
        extension=extension,
        datatype=datatype,
        **kwargs,
    )
    if noise:
        return path.find_empty_room()
    else:
        return path


def raw_basename(state: dict[str, Any], *, datatype: str) -> str:
    return _bids_name(state, ('subject', 'session', 'task', 'acquisition', 'run'), datatype=datatype)


def epoch_basename(state: dict[str, Any], *, datatype: str) -> str:
    return _bids_name(state, ('subject', 'session', 'acquisition', 'run'), datatype=datatype)


def test_basename(state: dict[str, Any], *, datatype: str) -> str:
    return _bids_name(state, ('session', 'acquisition', 'run'), datatype=datatype)


def raw_dir(state: dict[str, Any], *, datatype: str) -> Path:
    path = Path(f"sub-{state['subject']}")
    if state.get('session'):
        path /= f"ses-{state['session']}"
    return path / datatype


def ica_file_path(state: dict[str, Any], raw: str, concatenate_runs: bool = False, *, datatype: str) -> Path:
    if concatenate_runs:
        entity_keys = ('subject', 'session', 'acquisition')
    else:
        entity_keys = ('subject', 'session', 'acquisition', 'run')
    basename = _bids_name(state, entity_keys, suffix='', datatype=datatype)
    return DERIV_DIR / 'mne' / raw_dir(state, datatype=datatype) / f"{basename}_desc-{raw}_ica.fif"


def trans_file_path(state: dict[str, Any], *, datatype: str) -> Path:
    basename = _bids_name(state, ('subject', 'session'), suffix='', datatype=datatype)
    return DERIV_DIR / 'mne' / raw_dir(state, datatype=datatype) / f"{basename}_trans.fif"


def rej_file_path(state: dict[str, Any], epoch: str | None = None, epoch_rejection: str | None = None, *, datatype: str) -> Path:
    epoch_name = state['epoch'] if epoch is None else epoch
    rej_name = state['epoch_rejection'] if epoch_rejection is None else epoch_rejection
    basename = _bids_name(state, ('subject', 'session', 'acquisition', 'run'), suffix='', datatype=datatype)
    return DERIV_DIR / 'mne' / raw_dir(state, datatype=datatype) / f"{basename}_raw-{state['raw']}_epoch-{epoch_name}_rej-{rej_name}_epoch.pickle"


def mri_dir(state: dict[str, Any]) -> Path:
    return MRI_SDIR / state['mrisubject']


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
    return CACHE_DIR / 'annot' / state['mrisubject'] / f'{state["parc"]}.stamp'


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
        return RESULTS_DIR / report_kind / 'subjects' / _slug(state['subject']) / f'{stem}.html'
    return RESULTS_DIR / report_kind / 'groups' / _slug(state['group']) / f'{stem}.html'


def movie_export_path(
        state: dict[str, Any],
        stem: str,
        single_subject: bool = False,
) -> Path:
    if single_subject:
        return RESULTS_DIR / 'movies' / 'subjects' / _slug(state['subject']) / f'{stem}.mov'
    return RESULTS_DIR / 'movies' / 'groups' / _slug(state['group']) / f'{stem}.mov'


def coreg_report_path(state: dict[str, Any]) -> Path:
    title = 'Coregistration'
    if state.get('group') != 'all':
        title += f" {state['group']}"
    if state.get('mri'):
        title += f" {state['mri']}"
    return METHODS_DIR / f'{title}.html'
