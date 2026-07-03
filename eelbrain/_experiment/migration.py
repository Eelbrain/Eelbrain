"""Migrate graph-managed derivative files from legacy to current path layout.

Legacy layout stored ICA and coregistration files in flat, pipeline-named
directories with the datatype repeated in the filename::

    derivatives/ica/sub-<label>[_ses-<label>][_run-<label>]_<datatype>_raw-<raw>_ica.fif
    derivatives/trans/sub-<label>[_ses-<label>]_<datatype>_trans.fif

The current layout follows the BIDS derivatives recommendation, grouping
MNE-format outputs under a ``mne`` pipeline directory with a
``sub-/ses-/<datatype>/`` subtree and dropping the redundant datatype token
from the filename (the ``raw`` pipeline stage becomes a ``desc-`` entity)::

    derivatives/mne/sub-<label>/[ses-<label>/]<datatype>/sub-<label>[_ses-<label>][_run-<label>]_desc-<raw>_ica.fif
    derivatives/mne/sub-<label>/[ses-<label>/]<datatype>/sub-<label>[_ses-<label>]_trans.fif
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


def _new_ica_path(root: Path, old_path: Path) -> Path:
    entities, datatype, trailing = _parse_legacy_stem(old_path.stem)
    raw = trailing[0].partition('-')[2]  # 'raw-<raw>' -> '<raw>'
    basename = _new_basename(entities, 'sub', 'ses', 'run')
    return _new_dir(root, entities, datatype) / f"{basename}_desc-{raw}_ica.fif"


def _new_trans_path(root: Path, old_path: Path) -> Path:
    entities, datatype, _ = _parse_legacy_stem(old_path.stem)
    basename = _new_basename(entities, 'sub', 'ses')
    return _new_dir(root, entities, datatype) / f"{basename}_trans.fif"


def migrate_derivatives(root: Path | str, dry_run: bool = False) -> list[tuple[Path, Path]]:
    """Move legacy ICA and coregistration files to the current BIDS-style layout.

    Parameters
    ----------
    root
        Experiment root directory.
    dry_run
        Only report the moves that would be made, without touching any files.

    Returns
    -------
    moved
        List of ``(old_path, new_path)`` pairs for the files that were (or, with
        ``dry_run``, would be) moved.
    """
    root = Path(root)
    moved = []
    for legacy_subdir, new_path_func in (('ica', _new_ica_path), ('trans', _new_trans_path)):
        old_dir = root / DERIV_DIR / legacy_subdir
        if not old_dir.exists():
            continue
        for old_path in sorted(old_dir.glob('*.fif')):
            new_path = new_path_func(root, old_path)
            moved.append((old_path, new_path))
            if not dry_run:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
        if not dry_run and not any(old_dir.iterdir()):
            old_dir.rmdir()
    return moved
