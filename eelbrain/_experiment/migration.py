"""Migrate graph-managed derivative files from legacy to current path layout.

Legacy layout stored ICA and coregistration files in flat, pipeline-named
directories with the datatype repeated in the filename, and stored the
Pipeline-specific bad-channels and epoch-rejection files under an ``eelbrain``
pipeline directory::

    derivatives/ica/sub-<label>[_ses-<label>][_run-<label>]_<datatype>_raw-<raw>_ica.fif
    derivatives/trans/sub-<label>[_ses-<label>]_<datatype>_trans.fif
    derivatives/eelbrain/bad_channels/sub-<label>[_ses-<label>][_task-<label>][_run-<label>]_channels.tsv
    derivatives/eelbrain/epoch selection/sub-<label>[_ses-<label>][_run-<label>]_<datatype>_raw-<raw>_epoch-<epoch>_rej-<rej>_epoch.pickle

The current layout follows the BIDS derivatives recommendation, grouping
MNE-format outputs under a ``mne`` pipeline directory with a
``sub-/ses-/<datatype>/`` subtree and dropping the redundant datatype token
from the filename (the ``raw`` pipeline stage becomes a ``desc-`` entity)::

    derivatives/mne/sub-<label>/[ses-<label>/]<datatype>/sub-<label>[_ses-<label>][_run-<label>]_desc-<raw>_ica.fif
    derivatives/mne/sub-<label>/[ses-<label>/]<datatype>/sub-<label>[_ses-<label>]_trans.fif
    derivatives/mne/sub-<label>/[ses-<label>/]<datatype>/sub-<label>[_ses-<label>][_task-<label>][_run-<label>]_channels.tsv
    derivatives/mne/sub-<label>/[ses-<label>/]<datatype>/sub-<label>[_ses-<label>][_run-<label>]_raw-<raw>_epoch-<epoch>_rej-<rej>_epoch.pickle
"""

from __future__ import annotations

from pathlib import Path

from .pathing import DERIV_DIR


def _parse_legacy_stem(stem: str) -> tuple[dict[str, str], str, list[str]]:
    """Split a legacy filename stem into entities, datatype and trailing tokens.

    Parameters
    ----------
    stem
        Filename without extension, e.g. ``'sub-R0000_meg_raw-ica_ica'``.

    Returns
    -------
    entities
        BIDS ``key-value`` entities preceding the datatype (e.g. ``sub``, ``ses``, ``run``).
    datatype
        The bare datatype token (e.g. ``'meg'``).
    trailing
        Tokens following the datatype (e.g. ``['raw-ica', 'ica']``).
    """
    entities: dict[str, str] = {}
    datatype: str | None = None
    trailing: list[str] = []
    for token in stem.split('_'):
        if datatype is None and '-' in token:
            key, _, value = token.partition('-')
            entities[key] = value
        elif datatype is None:
            datatype = token
        else:
            trailing.append(token)
    if datatype is None:
        raise ValueError(f"Not a legacy derivative filename: {stem=}")
    return entities, datatype, trailing


def _parse_bids_entities(stem: str) -> dict[str, str]:
    """Collect all ``key-value`` BIDS entities from a filename stem (no datatype token)."""
    entities: dict[str, str] = {}
    for token in stem.split('_'):
        if '-' in token:
            key, _, value = token.partition('-')
            entities[key] = value
    return entities


def _new_dir(root: Path, entities: dict[str, str], datatype: str) -> Path:
    path = root / DERIV_DIR / 'mne' / f"sub-{entities['sub']}"
    if 'ses' in entities:
        path /= f"ses-{entities['ses']}"
    return path / datatype


def _new_basename(entities: dict[str, str], *entity_keys: str) -> str:
    parts = []
    for key in entity_keys:
        if key in entities:
            parts.append(f"{key}-{entities[key]}")
    return '_'.join(parts)


def _find_datatype(root: Path, entities: dict[str, str]) -> str | None:
    """Find the datatype directory in the BIDS source containing this recording."""
    sub_dir = root / f"sub-{entities['sub']}"
    if 'ses' in entities:
        sub_dir /= f"ses-{entities['ses']}"
    if not sub_dir.is_dir():
        return None
    prefix = _new_basename(entities, 'sub', 'ses', 'task', 'acq', 'run') + '_'
    for datatype_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
        if any(f.name.startswith(prefix) for f in datatype_dir.iterdir()):
            return datatype_dir.name
    return None


def _new_ica_path(root: Path, old_path: Path) -> Path:
    entities, datatype, trailing = _parse_legacy_stem(old_path.stem)
    raw = trailing[0].partition('-')[2]  # 'raw-<raw>' -> '<raw>'
    basename = _new_basename(entities, 'sub', 'ses', 'acq', 'run')
    return _new_dir(root, entities, datatype) / f"{basename}_desc-{raw}_ica.fif"


def _new_trans_path(root: Path, old_path: Path) -> Path:
    entities, datatype, _ = _parse_legacy_stem(old_path.stem)
    basename = _new_basename(entities, 'sub', 'ses')
    return _new_dir(root, entities, datatype) / f"{basename}_trans.fif"


def _new_bad_channels_path(root: Path, old_path: Path) -> Path | None:
    entities = _parse_bids_entities(old_path.stem)
    datatype = _find_datatype(root, entities)
    if datatype is None:
        return None
    return _new_dir(root, entities, datatype) / old_path.name


def _new_rej_path(root: Path, old_path: Path) -> Path:
    entities, datatype, trailing = _parse_legacy_stem(old_path.stem)
    basename = _new_basename(entities, 'sub', 'ses', 'acq', 'run')
    stem = '_'.join([basename, *trailing])
    return _new_dir(root, entities, datatype) / f"{stem}{old_path.suffix}"


# (legacy subdirectory relative to root, glob pattern, new-path function)
_MIGRATIONS = (
    (DERIV_DIR / 'ica', '*.fif', _new_ica_path),
    (DERIV_DIR / 'trans', '*.fif', _new_trans_path),
    (DERIV_DIR / 'eelbrain' / 'bad_channels', '*.tsv', _new_bad_channels_path),
    (DERIV_DIR / 'eelbrain' / 'epoch selection', '*.pickle', _new_rej_path),
)


def migrate_derivatives(
        root: Path | str,
        dry_run: bool = False,
        overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """Move legacy derivative files to the current BIDS-style ``mne`` layout.

    Migrates ICA, coregistration, bad-channels and epoch-rejection files.

    Parameters
    ----------
    root
        Experiment root directory.
    dry_run
        Only report the moves that would be made, without touching any files.
    overwrite
        Replace files that already exist in the current layout. By default,
        an existing destination raises :exc:`FileExistsError`.

    Returns
    -------
    moved
        List of ``(old_path, new_path)`` pairs for the files that were (or, with
        ``dry_run``, would be) moved.
    """
    root = Path(root)
    moved = []
    for legacy_subdir, pattern, new_path_func in _MIGRATIONS:
        old_dir = root / legacy_subdir
        if not old_dir.exists():
            continue
        for old_path in sorted(old_dir.glob(pattern)):
            new_path = new_path_func(root, old_path)
            if new_path is None:
                continue
            if new_path.exists() and not overwrite:
                raise FileExistsError(f"Migration target already exists: {new_path}; pass overwrite=True to replace it")
            moved.append((old_path, new_path))

    if dry_run:
        return moved

    for old_path, new_path in moved:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            old_path.replace(new_path)
        else:
            if new_path.exists():
                raise FileExistsError(f"Migration target already exists: {new_path}; pass overwrite=True to replace it")
            old_path.rename(new_path)

    for legacy_subdir, _, _ in _MIGRATIONS:
        old_dir = root / legacy_subdir
        if old_dir.is_dir() and not any(old_dir.iterdir()):
            old_dir.rmdir()
    return moved
