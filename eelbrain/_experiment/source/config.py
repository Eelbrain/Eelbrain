# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Inverse-solution and source-space configurations.

These classes normalize the user-facing ``inv`` and ``src`` specifications. The
graph nodes that build the corresponding source-space products live in
:mod:`._experiment.source.nodes`.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import mne
from mne.minimum_norm import apply_inverse, apply_inverse_epochs, make_inverse_operator

from ... import load
from ..._data_obj import NDVar
from ..configuration import Configuration


INV_METHODS = ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
SRC_RE = re.compile(r'^(ico|vol)-(\d+)(?:-(cortex|brainstem))?$')
INV_RE = re.compile(
    r"^"
    r"(free|fixed|loose\.\d+|vec)"
    r"(?:-(\d*\.?\d+))?"
    rf"-({'|'.join(INV_METHODS)})"
    r"(?:-((?:0\.)?\d+))?"
    r"(?:-(pick_normal))?"
    r"$"
)


class InverseSolution(Configuration):
    """Internal normalized inverse-operator configuration."""

    @classmethod
    def _coerce(cls, inv: str | InverseSolution) -> InverseSolution:
        if isinstance(inv, InverseSolution):
            return inv
        if isinstance(inv, str):
            return MinimumNormInverseSolution._from_string(inv)
        raise TypeError(f"{inv=}: invalid inverse solution specification")

    def _string(self) -> str:
        raise NotImplementedError(f"{self.__class__.__name__}._string()")

    def _validate_for_source_space(self, src: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}._validate_for_source_space()")

    def _build_operator(
            self,
            info: mne.Info,
            fwd: mne.Forward,
            cov: mne.Covariance,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._build_operator()")

    def _load_operator(self, path: Path):
        raise NotImplementedError(f"{self.__class__.__name__}._load_operator()")

    def _save_operator(self, path: Path, value) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}._save_operator()")

    def _apply_epochs(self, epochs_obj, operator, label=None):
        raise NotImplementedError(f"{self.__class__.__name__}._apply_epochs()")

    def _apply_evoked(self, evoked, operator):
        raise NotImplementedError(f"{self.__class__.__name__}._apply_evoked()")

    def _to_ndvar(
            self,
            stc,
            subject: str,
            src: str,
            subjects_dir: Path,
            *,
            parc: str | None,
            adjacency: str,
    ) -> NDVar:
        return load.mne.stc_ndvar(stc, subject, src, subjects_dir, self.method, self._fixed, parc=parc, adjacency=adjacency)


class MinimumNormInverseSolution(InverseSolution):
    """Normalized minimum-norm inverse configuration."""

    DICT_ATTRS = ('kind', 'ori', 'snr', 'method', 'depth', 'pick_normal')

    def __init__(
            self,
            ori: str | float = 'free',
            snr: float = 3,
            method: str = 'dSPM',
            depth: float = 0,
            pick_normal: bool = False,
    ):
        if isinstance(ori, str):
            if ori not in ('free', 'fixed', 'vec'):
                raise ValueError(f"{ori=}; needs to be 'free', 'fixed', 'vec', or float")
        elif not 0 < ori < 1:
            raise ValueError(f"{ori=}; must be in range (0, 1)")
        if snr < 0:
            raise ValueError(f"{snr=}")
        if method not in INV_METHODS:
            raise ValueError(f"{method=}")
        if not 0 <= depth <= 1:
            raise ValueError(f"{depth=}; must be in range [0, 1]")
        if pick_normal and ori in ('vec', 'fixed'):
            raise ValueError(f"{ori=} and pick_normal=True are incompatible")

        self.kind = 'minimum_norm'
        self.ori = ori
        self.snr = snr
        self.method = method
        self.depth = depth
        self.pick_normal = pick_normal

    @classmethod
    def _from_string(cls, inv: str) -> MinimumNormInverseSolution:
        m = INV_RE.match(inv)
        if m is None:
            raise ValueError(f"{inv=}: invalid inverse specification")

        ori, snr, method, depth, pick_normal = m.groups()
        if ori.startswith('loose'):
            ori = float(ori[5:])
            if not 0 < ori < 1:
                raise ValueError(f"{inv=}: loose parameter needs to be in range (0, 1)")

        if snr is None:
            snr = 0
        else:
            snr = float(snr)

        if depth is None:
            depth = 0
        else:
            depth = float(depth)

        return cls(ori, snr, method, depth, bool(pick_normal))

    def _string(self) -> str:
        if isinstance(self.ori, str):
            ori = self.ori
        else:
            ori = f'loose{str(self.ori)[1:]}'
        items = [ori]
        if self.snr > 0:
            items.append(f'{self.snr:g}')
        items.append(self.method)
        if self.depth != 0.8:
            items.append(f'{self.depth:g}')
        if self.pick_normal:
            items.append('pick_normal')
        return '-'.join(items)

    def _validate_for_source_space(self, src: str) -> None:
        if src[:3] == 'vol':
            if self.ori not in ('free', 'vec'):
                raise ValueError(f"inv={self._string()!r} with {src=}: volume source space requires free or vector inverse")
            if self.pick_normal:
                raise ValueError(f"inv={self._string()!r} with {src=}: volume source space does not support pick_normal")

    @property
    def _make_kw(self) -> dict[str, Any]:
        if self.ori == 'fixed':
            out = {'fixed': True}
        elif self.ori in ('free', 'vec'):
            out = {'loose': 1}
        else:
            out = {'loose': self.ori}

        if self.depth == 0:
            out['depth'] = None
        else:
            out['depth'] = self.depth
        return out

    @property
    def _apply_kw(self) -> dict[str, Any]:
        out = {'method': self.method, 'lambda2': 1. / self.snr ** 2 if self.snr else 0}
        if self.ori == 'vec':
            out['pick_ori'] = 'vector'
        elif self.pick_normal:
            out['pick_ori'] = 'normal'
        return out

    @property
    def _fixed(self) -> bool:
        return self._make_kw.get('fixed', False)

    def _build_operator(
            self,
            info: mne.Info,
            fwd: mne.Forward,
            cov: mne.Covariance,
    ):
        return make_inverse_operator(info, fwd, cov, use_cps=True, **self._make_kw)

    def _load_operator(self, path: Path):
        return mne.minimum_norm.read_inverse_operator(path)

    def _save_operator(self, path: Path, value) -> None:
        mne.minimum_norm.write_inverse_operator(path, value, overwrite=True)

    def _apply_epochs(self, epochs_obj, operator, label=None):
        return apply_inverse_epochs(epochs_obj, operator, label=label, **self._apply_kw)

    def _apply_evoked(self, evoked, operator):
        return apply_inverse(evoked, operator, **self._apply_kw)


def parse_src(src: str) -> tuple[str, str, str | None]:
    m = SRC_RE.match(src)
    if not m:
        raise ValueError(f'{src=}')
    kind, param, special = m.groups()
    if special and kind != 'vol':
        raise ValueError(f'{src=}')
    return kind, param, special


def eval_src(src: str) -> str:
    parse_src(src)
    return src
