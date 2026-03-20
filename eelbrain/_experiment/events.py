# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Event-related cache nodes."""

from __future__ import annotations

from typing import Any

import mne

from .. import load, save
from .._data_obj import Dataset
from .derivative_cache import CachePolicy, Dependency, Derivative, DerivativeContext


BIDS_ENTITY_KEYS = ('subject', 'session', 'task', 'acquisition', 'run', 'split')


class EventsDerivative(Derivative[Dataset]):
    name = 'events'
    path_template = 'event-file'
    key_fields = ('subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw')

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (Dependency('raw-input-events'),)

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        subject = ctx.get('subject')
        session = ctx.get('session')
        trigger_shift = ctx.pipeline.trigger_shift
        if isinstance(trigger_shift, dict):
            trigger_shift = trigger_shift.get((subject, session), trigger_shift.get(subject, 0))
        return {
            'raw': ctx.get('raw'),
            'stim_channel': ctx.pipeline._stim_channel,
            'merge_triggers': ctx.pipeline.merge_triggers,
            'trigger_shift': trigger_shift,
            'variables': repr(ctx.pipeline._variables),
            'has_edf': bool(ctx.pipeline.has_edf[subject]),
        }

    def build(self, ctx: DerivativeContext) -> Dataset:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            entities = {k: p.get(k) for k in BIDS_ENTITY_KEYS}
            subject = entities['subject']
            ds = p._extract_events_dataset()
            if p.has_edf[subject]:
                edf = p.load_edf()
                edf.add_t_to(ds)
                ds.info['edf'] = edf
            ds.info.update(entities)
            return ds

    def load(self, ctx: DerivativeContext, path: str) -> Dataset:
        ds = load.unpickle(path)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            ds.info.update({k: p.get(k) for k in BIDS_ENTITY_KEYS})
        return ds

    def save(self, ctx: DerivativeContext, path: str, value: Dataset) -> None:
        save.pickle(value, path)


class EvokedDerivative(Derivative[Dataset]):
    name = 'evoked'
    path_template = 'evoked-file'
    key_fields = (
        'subject', 'session', 'task', 'acquisition', 'run', 'split', 'raw',
        'epoch', 'rej', 'model', 'equalize_evoked_count',
    )
    cache_policy = CachePolicy.OPTIONAL

    def dependencies(self, ctx: DerivativeContext) -> tuple[Dependency, ...]:
        return (
            Dependency('events'),
            Dependency('raw-input-bads'),
            Dependency('rej-input'),
        )

    def fingerprint(self, ctx: DerivativeContext) -> dict[str, Any]:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            epoch = p._epochs[p.get('epoch')]
            return {
                'raw': p.get('raw'),
                'epoch': epoch.__dict__,
                'rej': p.get('rej'),
                'model': p.get('model'),
                'equalize_evoked_count': p.get('equalize_evoked_count'),
                'vardef': ctx.option('vardef'),
                'samplingrate': ctx.option('samplingrate'),
                'decim': ctx.option('decim'),
            }

    def build(self, ctx: DerivativeContext) -> Dataset:
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            return p._make_evoked(
                ctx.option('samplingrate'),
                ctx.option('decim'),
                ctx.option('data_raw', False),
                ctx.option('vardef'),
            )

    def load(self, ctx: DerivativeContext, path: str) -> Dataset:
        evoked = mne.read_evokeds(path, proj=False)
        p = ctx.pipeline
        with p._temporary_state:
            if ctx.state:
                p.set(**ctx.state)
            return p._evoked_dataset_from_cache(
                evoked,
                ctx.option('data_raw', False),
                ctx.option('vardef'),
            )

    def save(self, ctx: DerivativeContext, path: str, value: Dataset) -> None:
        mne.write_evokeds(path, value['evoked'], overwrite=True)

    def validate(self, ctx: DerivativeContext, path: str, manifest) -> bool:
        evoked = mne.read_evokeds(path, proj=False)
        return [e.comment for e in evoked] == manifest.provenance.get('comments', [])

    def provenance(self, ctx: DerivativeContext, value: Dataset) -> dict[str, Any]:
        return {'comments': [e.comment for e in value['evoked']]}
