# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Event-related cache nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

import mne

from .. import load, save
from .._data_obj import Dataset
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext
from .pathing import event_file_path, evoked_file_path
from .preprocessing import raw_data_dependency


BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')


def _evoked_comments(evoked: list[mne.Evoked]) -> list[str]:
    return [e.comment or 'No comment' for e in evoked]


@dataclass
class EventsSupport:
    trigger_shift: int | dict[str, int] | dict[tuple[str, str], int]
    stim_channel: str | list[str]
    merge_triggers: Any
    variables_repr: str
    has_edf: dict[str, Any]
    epochs: dict[str, Any]
    extract_events_dataset: Callable[[dict[str, Any]], Dataset]
    load_edf: Callable[[dict[str, Any]], Any]
    make_evoked: Callable[[dict[str, Any], Any, Any, bool, str | None], Dataset]
    evoked_dataset_from_cache: Callable[[dict[str, Any], list[mne.Evoked], bool, str | None], Dataset]


class EventsDerivative(Derivative[Dataset]):
    name = 'events'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw')

    def __init__(self, support: EventsSupport):
        self.support = support

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = event_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (raw_data_dependency(ctx, add_bads=False),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        subject = ctx.get('subject')
        session = ctx.get('session')
        trigger_shift = self.support.trigger_shift
        if isinstance(trigger_shift, dict):
            trigger_shift = trigger_shift.get((subject, session), trigger_shift.get(subject, 0))
        return {
            'raw': ctx.get('raw'),
            'stim_channel': self.support.stim_channel,
            'merge_triggers': self.support.merge_triggers,
            'trigger_shift': trigger_shift,
            'variables': self.support.variables_repr,
            'has_edf': bool(self.support.has_edf[subject]),
        }

    def build(self, ctx: DerivativeContext) -> Dataset:
        entities = {k: ctx.get(k) for k in BIDS_ENTITY_KEYS}
        subject = entities['subject']
        ds = self.support.extract_events_dataset(ctx.state)
        if self.support.has_edf[subject]:
            edf = self.support.load_edf(ctx.state)
            edf.add_t_to(ds)
            ds.info['edf'] = edf
        ds.info.update(entities)
        return ds

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> Dataset:
        ds = load.unpickle(path)
        ds.info.update({k: ctx.get(k) for k in BIDS_ENTITY_KEYS})
        return ds

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: Dataset,
    ) -> None:
        save.pickle(value, path)


class EvokedDerivative(Derivative[Dataset]):
    name = 'evoked'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count',
    )
    cache_policy = CachePolicy.OPTIONAL

    def __init__(self, support: EventsSupport):
        self.support = support

    def path(self, ctx: DerivativeContext, mkdir: bool = False) -> Path:
        path = evoked_file_path(ctx.state)
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency('events'),
            raw_data_dependency(ctx),
            Dependency('rej-input'),
        )

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        epoch = self.support.epochs[ctx.get('epoch')]
        return {
            'raw': ctx.get('raw'),
            'epoch': epoch.__dict__,
            'rej': ctx.get('rej'),
            'model': ctx.get('model'),
            'equalize_evoked_count': ctx.get('equalize_evoked_count'),
            'vardef': ctx.option('vardef'),
            'samplingrate': ctx.option('samplingrate'),
            'decim': ctx.option('decim'),
        }

    def build(self, ctx: DerivativeContext) -> Dataset:
        return self.support.make_evoked(
            ctx.state,
            ctx.option('samplingrate'),
            ctx.option('decim'),
            ctx.option('data_raw', False),
            ctx.option('vardef'),
        )

    def load(
            self,
            ctx: DerivativeContext,
            path: Path) -> Dataset:
        evoked = mne.read_evokeds(path, proj=False)
        return self.support.evoked_dataset_from_cache(
            ctx.state,
            evoked,
            ctx.option('data_raw', False),
            ctx.option('vardef'),
        )

    def save(
            self,
            ctx: DerivativeContext,
            path: Path,
            value: Dataset,
    ) -> None:
        mne.write_evokeds(path, value['evoked'], overwrite=True)

    def validate(
            self,
            ctx: DerivativeContext,
            path: Path,
            manifest,
    ) -> bool:
        evoked = mne.read_evokeds(path, proj=False)
        return _evoked_comments(evoked) == manifest.provenance.get('comments', [])

    def provenance(self, ctx: DerivativeContext, value: Dataset) -> dict[str, Any]:
        return {'comments': _evoked_comments(value['evoked'])}
