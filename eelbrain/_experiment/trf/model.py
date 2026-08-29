# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Model specification for TRFs

The specification layer is symbolic: :class:`Term`, :class:`Model` and
:class:`Comparison` are built by parsing and model algebra alone, without
knowledge of the model-wide ``tstart``/``tstop``. An open lag bound stands for
the model-wide default and is treated as unbounded by the algebra (overlap,
subtraction, complement).

Resolving lag windows against concrete ``tstart``/``tstop`` values is a
separate semantic step: ``resolve_lags`` fills every open bound in from the
model-wide defaults (re-deriving an omit comparison's reduced model, so that
omitted windows are verified against the concrete windows they are removed
from). ``normalize_lags`` puts ``(x, tstart, tstop)`` into the canonical form
used for cache identity: a model without lag overrides keeps the model-wide
bounds; with overrides, a window shared by all terms moves to the model-wide
bounds, and otherwise every term carries its explicit window and the
model-wide bounds are ``None``. The TRF derivative nodes apply
``normalize_lags`` when a request is resolved, so equivalent spellings share
one cached artifact.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import chain
from math import inf
from operator import attrgetter
from pathlib import Path
import pickle

from pyparsing import DelimitedList, Group, Keyword, ParseException, Literal, Optional, Regex, Word, alphanums, one_of

from ..._data_obj import Dataset
from ... import fmtxt


COMP = {1: '>', 0: '=', -1: '<'}
TAIL = {'>': 1, '=': 0, '<': -1}
NUTS_METHODS = ('step',)


class TRFModelError(Exception):
    """Error in TRF model specification"""


@dataclass(frozen=True)
class Term:
    stimulus: str | None
    code: str
    tstart: float | None = None  # lag-window override; None: use the model-wide option
    tstop: float | None = None

    def __post_init__(self):
        for attr in ('tstart', 'tstop'):
            value = getattr(self, attr)
            if value is None:
                continue
            ms = round(value * 1000)
            if abs(value * 1000 - ms) > 1e-6:
                raise TRFModelError(f"{self.string}: lag windows with sub-millisecond precision are not supported")
            object.__setattr__(self, attr, ms / 1000)  # snap to the ms grid so that the string form is lossless
        if self.tstart is not None and self.tstop is not None and self.tstart >= self.tstop:
            raise TRFModelError(f"{self.string}: tstart must be smaller than tstop")

    @cached_property
    def string(self) -> str:
        string = self.code
        if self.stimulus:
            string = f"{self.stimulus}~{string}"
        if self.tstart is None and self.tstop is None:
            return string
        tstart = '' if self.tstart is None else f'{self.tstart:.10g}'
        tstop = '' if self.tstop is None else f'{self.tstop:.10g}'
        return f"{string}[{tstart}:{tstop}]"

    @cached_property
    def key(self) -> str:
        """Dataset-compatible key for the term"""
        return Dataset.as_key(self.string)

    @cached_property
    def _items(self) -> list[str]:
        return self.code.split('-')

    @cached_property
    def predictor_key(self) -> str:
        return self._items[0]

    @cached_property
    def nuts_method(self) -> str | None:
        """NUTS representation method (the trailing ``-step``/``-is`` item, if any)"""
        if len(self._items) > 2 and self._items[-1] in NUTS_METHODS:
            return self._items[-1]
        return None

    @cached_property
    def string_without_nuts_method(self) -> str:
        if self.nuts_method:
            code = '-'.join(self._items[:-1])
            if self.stimulus:
                return f"{self.stimulus}~{code}"
            return code
        return self.string

    @cached_property
    def nuts_columns(self) -> tuple[str | None, str | None]:
        """``(value-column, mask-column)`` for a ``columns`` NUTS predictor"""
        # bare key = intercept: unit impulse at each time stamp
        column = mask = None
        n = len(self._items)
        if n > 1:
            column = self._items[1]
            n -= bool(self.nuts_method)
            if n == 3:
                mask = self._items[2]
            elif n != 2:
                raise TRFModelError(f"{self.string}: too many '-' separated elements")
        return column, mask

    @cached_property
    def uts_file_name(self) -> str:
        """File name (without extension) of the predictor file backing this term"""
        return self.without_lags().string

    @cached_property
    def nuts_file_name(self) -> str:
        """File name (without extension) of the predictor file backing this term"""
        code = self._items[0]
        return f"{self.stimulus}~{code}" if self.stimulus else code

    def with_stimulus(self, stimulus: str) -> Term:
        """Copy of the term with a different stimulus"""
        return replace(self, stimulus=stimulus)

    def without_lags(self) -> Term:
        """Copy of the term without lag-window overrides"""
        if self.tstart is None and self.tstop is None:
            return self
        return replace(self, tstart=None, tstop=None)

    def with_default_lags(self, tstart: float | None, tstop: float | None) -> Term:
        """Copy of the term with open lag bounds filled in from the defaults"""
        return replace(self, tstart=self.tstart if self.tstart is not None else tstart, tstop=self.tstop if self.tstop is not None else tstop)

    def file_label(self, stimulus: str) -> str:
        """Dependency label of the predictor file backing this term"""
        return self.without_lags().with_stimulus(stimulus).string

    @classmethod
    def _coerce(cls, x: Term | str):
        if isinstance(x, Term):
            return x
        elif isinstance(x, str):
            return parse_term(x)
        raise TypeError(x)

    def _cache_form_(self) -> str:
        """Canonical form for cache keys/fingerprints/manifests"""
        return self.string

    def __repr__(self):
        return f"<Term: {self.string}>"


def _window(term: Term, tstart: float | None = None, tstop: float | None = None) -> tuple[float, float]:
    """Resolved ``(tstart, tstop)`` lag window of ``term``, filling open bounds from the model-wide defaults (unbounded where those are unknown)"""
    tstart = term.tstart if term.tstart is not None else (-inf if tstart is None else tstart)
    tstop = term.tstop if term.tstop is not None else (inf if tstop is None else tstop)
    return tstart, tstop


def _windows_overlap(a: Term, b: Term) -> bool:
    """Whether the lag windows of two terms overlap (open bounds treated as unbounded)"""
    (a0, a1), (b0, b1) = _window(a), _window(b)
    return max(a0, b0) < min(a1, b1)


def _window_contains(term: Term, omit: Term) -> bool:
    """Whether the lag window of ``omit`` lies within the window of ``term``

    Open bounds in ``omit`` inherit the corresponding bound of ``term``; open bounds in ``term`` compare as unbounded.
    """
    t0, t1 = _window(term)
    o0, o1 = _window(omit, t0, t1)
    return t0 <= o0 < t1 and t0 < o1 <= t1


def _window_complement(term: Term, omit: Term) -> list[Term]:
    """Terms covering the part of ``term``'s lag window that ``omit`` does not cover (0, 1 or 2 terms)"""
    t0, t1 = _window(term)
    o0, o1 = _window(omit, t0, t1)
    out = []
    if t0 < o0 < t1:
        out.append(replace(term, tstop=omit.tstart))
    if t0 < o1 < t1:
        out.append(replace(term, tstart=omit.tstop))
    return out


def _shared_window(models: Sequence[Model]) -> tuple[float, float] | None:
    """The single explicit lag window shared by every term in ``models`` (else ``None``)"""
    windows = {(term.tstart, term.tstop) for model in models for term in model.terms}
    if len(windows) != 1:
        return None
    return windows.pop()


def _require_bounds(name: str, tstart: float | None, tstop: float | None) -> None:
    """Validate the model-wide bounds for a model/comparison in which no term has a lag window"""
    if tstart is None or tstop is None:
        raise TRFModelError(f"{name}: tstart and tstop are required when no term has a lag window ({tstart=}, {tstop=})")
    if tstart >= tstop:
        raise TRFModelError(f"{name}: empty lag window ({tstart=}, {tstop=})")


def _expand_term(
        term: Term,
        named_models: dict[str, Model],
) -> tuple[Term, ...]:
    """ModelTerms can represent multiple effective terms"""
    # if term.code.endswith('-is'):
    #     base_code = term.code[:-4]
    #     terms = _expand_term(replace(term, code=base_code), named_models)
    #     return (*terms, *[replace(term, code=f'{term.code}-step') for term in terms])
    if term.code.endswith('-step') and term.code[:-5] in named_models:
        terms = _expand_term(replace(term, code=term.code[:-5]), named_models)
        return tuple([replace(term, code=f'{term.code}-step') for term in terms])
    elif term.code in named_models:
        terms = named_models[term.code].terms
        if term.tstart is not None or term.tstop is not None:
            # distribute lag overrides to member terms; explicit member lags take precedence
            terms = tuple([term_i.with_default_lags(term.tstart, term.tstop) for term_i in terms])
        return terms
    else:
        return term,


@dataclass(frozen=True)
class Model:
    """Model that can be fit to data"""
    terms: tuple[Term, ...]

    def __post_init__(self):
        # Check for identical duplicates
        counts = Counter([term.string for term in self.terms])
        duplicates = [term for term, count in counts.items() if count > 1]
        if duplicates:
            raise TRFModelError(f"{self.name}: duplicate terms {', '.join(duplicates)}")
        # Check for duplicate predictors with overlapping lag windows
        by_base = defaultdict(list)
        for term in self.terms:
            by_base[term.stimulus, term.code].append(term)
        for terms in by_base.values():
            for i, term in enumerate(terms):
                for other in terms[i + 1:]:
                    if _windows_overlap(term, other):
                        raise TRFModelError(f"{self.name}: overlapping lag windows {term.string} and {other.string}")

    @cached_property
    def name(self) -> str:
        if not self.terms:
            return '0'
        return ' + '.join(term.string for term in self.terms)

    def sorted(self) -> Model:
        return Model(tuple(sorted(self.terms, key=attrgetter('string'))))

    @cached_property
    def term_names(self):
        return tuple([term.string for term in self.terms])

    @classmethod
    def from_string(cls, string: str):
        try:
            return model.parse_string(string, True)[0]
        except ParseException:
            raise TRFModelError(f"{string!r}: invalid Model")

    def _cache_form_(self) -> str:
        """Canonical form for cache keys/fingerprints/manifests"""
        return self.name

    def __repr__(self):
        return f"<Model: {self.name}>"

    def __len__(self):
        return len(self.terms)

    def __add__(self, other: Model) -> Model:
        shared = self.intersection(other)
        if shared:
            raise TRFModelError(f"{self.name} + {other.name}: shared terms {shared.name}")
        return Model(self.terms + other.terms)

    def __sub__(self, other: Model) -> Model:
        """Remove terms; a term with a lag window removes that window from the matching term, keeping the complement"""
        terms = list(self.terms)
        for omit in other.terms:
            candidates = [term for term in terms if term.stimulus == omit.stimulus and term.code == omit.code]
            if not candidates:
                raise TRFModelError(f"{self.name} - {other.name}: no term matching {omit.string}")
            if omit.tstart is None and omit.tstop is None:
                for term in candidates:
                    terms.remove(term)
                continue
            # candidate windows are disjoint, so at most one can contain the omitted window
            containing = next((term for term in candidates if _window_contains(term, omit)), None)
            if containing is None:
                raise TRFModelError(f"{self.name} - {other.name}: lag window of {omit.string} is not contained in any single term ({', '.join(term.string for term in candidates)})")
            # an open omit bound inherits the containing term's bound, which must not silently exclude another piece of a split predictor
            straddled = [term for term in candidates if term is not containing and _windows_overlap(term, omit)]
            if straddled:
                raise TRFModelError(f"{self.name} - {other.name}: lag window of {omit.string} overlaps {', '.join(term.string for term in straddled)} in addition to {containing.string}; make the omitted window explicit")
            index = terms.index(containing)
            terms[index:index + 1] = _window_complement(containing, omit)
        return Model(tuple(terms))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    @classmethod
    def coerce(
            cls,
            x: Model | str | Sequence,
            named_models: dict[str, Model] = {},
    ) -> Model:
        if isinstance(x, cls):
            model = x
        elif isinstance(x, str):
            model = cls.from_string(x)
        elif isinstance(x, Sequence):
            model = cls(tuple(Term._coerce(term) for term in x))
        else:
            raise TypeError(x)
        return model.initialize(named_models)

    def difference(self, other: Model) -> Model:
        """Terms, and parts of lag windows, in ``self`` but not in ``other``"""
        terms = []
        for term in self.terms:
            pieces = [term]
            for omit in other.terms:
                if (omit.stimulus, omit.code) != (term.stimulus, term.code):
                    continue
                pieces = [piece_i for piece in pieces for piece_i in (_window_complement(piece, omit) if _windows_overlap(piece, omit) else [piece])]
            terms.extend(pieces)
        return Model(tuple(terms))

    def intersection(self, other: Model) -> Model:
        """Terms, and parts of lag windows, in both ``self`` and ``other``"""
        terms = []
        for term in self.terms:
            for shared in other.terms:
                if (shared.stimulus, shared.code) != (term.stimulus, term.code) or not _windows_overlap(term, shared):
                    continue
                (t0, t1), (s0, s1) = _window(term), _window(shared)
                tstart = None if max(t0, s0) == -inf else max(t0, s0)
                tstop = None if min(t1, s1) == inf else min(t1, s1)
                terms.append(replace(term, tstart=tstart, tstop=tstop))
        return Model(tuple(terms))

    def initialize(self, named_models: dict[str, Model]) -> Model:
        terms = tuple(chain.from_iterable(_expand_term(term, named_models) for term in self.terms))
        if terms == self.terms:
            return self
        return Model(terms)

    def without_lags(self) -> Model:
        """Copy of the model with all lag-window overrides stripped"""
        return Model(tuple(term.without_lags() for term in self.terms))

    def term_table(self) -> fmtxt.Table:
        show_stimulus = any(term.stimulus for term in self.terms)
        show_lags = any(term.tstart is not None or term.tstop is not None for term in self.terms)
        t = fmtxt.Table('r' + 'l' * show_stimulus + 'l' + 'rr' * show_lags)
        t.cell('#')
        if show_stimulus:
            t.cell('Stimulus')
        t.cell('Code')
        if show_lags:
            t.cells('tstart', 'tstop')
        t.midrule()
        for i, term in enumerate(self.terms):
            t.cell(i)
            if show_stimulus:
                t.cell(term.stimulus)
            t.cell(term.code)
            if show_lags:
                t.cell('' if term.tstart is None else f'{term.tstart:g}')
                t.cell('' if term.tstop is None else f'{term.tstop:g}')
        return t

    def resolve_lags(self, tstart: float | None, tstop: float | None) -> Model:
        """Copy with every lag window resolved to explicit bounds from the model-wide defaults

        Raises :class:`TRFModelError` when a bound is neither set on the term
        nor available as a model-wide default, and when a resolved window is
        empty.
        """
        terms = []
        for term in self.terms:
            if term.tstart is None and tstart is None:
                raise TRFModelError(f"{self.name}: no tstart for {term.string} (set tstart, or specify it in the term's lag window)")
            if term.tstop is None and tstop is None:
                raise TRFModelError(f"{self.name}: no tstop for {term.string} (set tstop, or specify it in the term's lag window)")
            t0, t1 = _window(term, tstart, tstop)
            if t0 >= t1:
                raise TRFModelError(f"{self.name}: lag window of {term.string} is empty given {tstart=} and {tstop=}")
            terms.append(term.with_default_lags(tstart, tstop))
        if terms == list(self.terms):
            return self
        return Model(tuple(terms))

    def normalize_lags(
            self,
            tstart: float | None,
            tstop: float | None,
    ) -> tuple[Model, float | None, float | None]:
        """Canonical ``(model, tstart, tstop)`` for cache identity"""
        # A model without lag overrides keeps the model-wide bounds
        if not any(term.tstart is not None or term.tstop is not None for term in self.terms):
            _require_bounds(self.name, tstart, tstop)
            return self, tstart, tstop
        # With any override present, all windows are resolved
        resolved = self.resolve_lags(tstart, tstop)
        # A window shared by all terms moves to the model-wide bounds
        if window := _shared_window([resolved]):
            return resolved.without_lags(), *window
        return resolved, None, None


def model_comparison_table(x1: Model, x0: Model, x1_name: str = 'x1', x0_name: str = 'x0'):
    """Generate a table comparing the terms in two models"""
    # find corresponding terms
    term_map = []
    x0_terms = list(x0.term_names)
    for x1_term in x1.term_names:
        if x1_term in x0_terms:
            target = x1_term
        else:
            rand = f'{x1_term}$'
            for x0_term in x0_terms:
                if x0_term.startswith(rand):
                    target = x0_term
                    break
            else:
                target = ''
        term_map.append((x1_term, target))
        if target:
            x0_terms.remove(target)
    for x0_term in x0_terms:
        term_map.append(('', x0_term))
    # format table
    table = fmtxt.Table('ll')
    table.cells(x1_name, x0_name)
    table.midrule()
    for x1_term, x0_term in term_map:
        table.cells(x1_term, x0_term)
    return table


@dataclass
class ComparisonSpec:
    x: Model

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Comparison:
        raise NotImplementedError


@dataclass
class DirectComparison(ComparisonSpec):
    operator: str
    x0: Model

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Comparison:
        public_name = f"{self.x.name} {self.operator} {self.x0.name}"
        x = self.x.initialize(named_models)
        x0 = self.x0.initialize(named_models)
        tail = TAIL[self.operator]
        return Comparison(x, x0, tail, public_name)


@dataclass
class OmitComparison(ComparisonSpec):
    x_omit: Model

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Comparison:
        public_name = f"{self.x.name} @ {self.x_omit.name}"
        x = self.x.initialize(named_models)
        x_omit = self.x_omit.initialize(named_models)
        x0 = x - x_omit
        return Comparison(x, x0, 1, public_name, omit_base=x, omits=(None, x_omit))


@dataclass
class Omit2Comparison(ComparisonSpec):
    x1_omit: Model
    operator: str
    x0_omit: Model

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Comparison:
        public_name = f"{self.x.name} @ {self.x1_omit.name} {self.operator} {self.x0_omit.name}"
        x = self.x.initialize(named_models)
        x1_omit = self.x1_omit.initialize(named_models)
        x0_omit = self.x0_omit.initialize(named_models)
        # each side is tested by omitting the *other* side's term: x1 keeps x1_omit by omitting x0_omit, and vice versa
        x1 = x - x0_omit
        x0 = x - x1_omit
        return Comparison(x1, x0, TAIL[self.operator], public_name, omit_base=x, omits=(x0_omit, x1_omit))


@dataclass
class AddComparison(ComparisonSpec):
    x_add: Model

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Comparison:
        public_name = f"{self.x.name} +@ {self.x_add.name}"
        x = self.x.initialize(named_models)
        x_add = self.x_add.initialize(named_models)
        x1 = x + x_add
        x0 = x
        return Comparison(x1, x0, 1, public_name)


@dataclass
class Add2Comparison(ComparisonSpec):
    x1_add: Model
    operator: str
    x0_add: Model

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Comparison:
        public_name = f"{self.x.name} +@ {self.x1_add.name} {self.operator} {self.x0_add.name}"
        x = self.x.initialize(named_models)
        x1_add = self.x1_add.initialize(named_models)
        x0_add = self.x0_add.initialize(named_models)
        x1 = x + x1_add
        x0 = x + x0_add
        return Comparison(x1, x0, TAIL[self.operator], public_name)


@dataclass(frozen=True)
class Comparison:
    """Model comparison for test or report"""
    x1: Model
    x0: Model
    tail: int = 1
    public_name: str = None
    # Construction record of an omit comparison, needed because omitted terms treat bounds differently; Example: 'a + b @ b[0.6:]' with tstart=0, tstop=0.5 -> x1 = 'a + b[:0.6]'. Cleared by resolve_lags: once every bound is explicit the record has no further job. omits[i] is subtracted from omit_base to produce models[i] (x1, x0).
    omit_base: Model = field(default=None, compare=False)
    omits: tuple[Model | None, Model | None] = field(default=(None, None), compare=False)

    @cached_property
    def operator(self) -> str:
        return COMP[self.tail]

    @cached_property
    def models(self) -> tuple[Model, Model]:
        return self.x1, self.x0

    @cached_property
    def common_base(self) -> Model:
        """Terms, and parts of lag windows, shared by both models

        Like :attr:`x1_only` and :attr:`x0_only`, exact only when the lag
        windows are explicit (see :meth:`resolve_lags`): an open bound compares
        as unbounded, so a term with an open bound can misattribute lags that
        the model-wide window would exclude.
        """
        return self.x1.intersection(self.x0)

    @cached_property
    def x1_only(self) -> Model:
        """Terms, and parts of lag windows, only in ``x1`` (see :attr:`common_base`)"""
        return self.x1.difference(self.x0)

    @cached_property
    def x0_only(self) -> Model:
        """Terms, and parts of lag windows, only in ``x0`` (see :attr:`common_base`)"""
        return self.x0.difference(self.x1)

    @cached_property
    def test_term_name(self):
        if not self.x0_only:
            return self.x1_only.name

    @cached_property
    def baseline_term_name(self):
        if len(self.x0_only) == 1:
            return self.x0_only.name

    @cached_property
    def name(self) -> str:
        if self.public_name:
            return self.public_name
        return self.compose_name()

    def compose_name(
            self,
            name: Callable[[Model], str] = lambda m: m.name,
            path: bool = False,  # return valid path component (avoiding problematic characters)
    ) -> str:
        # implement only parsable comparisons
        if path:
            op = {'>': '=g', '=': '=', '<': '=l'}[self.operator]
        else:
            op = self.operator
        return f"{name(self.x1)} {op} {name(self.x0)}"

    def sorted(self) -> Comparison:
        """Copy with terms in both models sorted for stable cache identity"""
        return replace(self, x1=self.x1.sorted(), x0=self.x0.sorted())

    def resolve_lags(self, tstart: float | None, tstop: float | None) -> Comparison:
        """Copy with both models' lag windows resolved to explicit bounds (see :meth:`Model.resolve_lags`)

        For an omit comparison, the reduced model is re-derived by subtracting
        the omitted terms from the resolved full model, so that every omitted
        window is verified against, and its complement computed within, the
        concrete window it is removed from.
        """
        if self.omit_base is None:
            x1 = self.x1.resolve_lags(tstart, tstop)
            x0 = self.x0.resolve_lags(tstart, tstop)
        else:
            base = self.omit_base.resolve_lags(tstart, tstop)
            x1, x0 = (base if omit is None else base - omit for omit in self.omits)
        if x1 == self.x1 and x0 == self.x0:
            return self
        # With every bound explicit the record has no further job; clearing it keeps re-resolution independent of the original model-wide bounds
        return replace(self, x1=x1, x0=x0, omit_base=None, omits=(None, None))

    def normalize_lags(
            self,
            tstart: float | None,
            tstop: float | None,
    ) -> tuple[Comparison, float | None, float | None]:
        """Canonical ``(comparison, tstart, tstop)`` for cache identity"""
        if not any(term.tstart is not None or term.tstop is not None for model in self.models for term in model.terms):
            _require_bounds(self.name, tstart, tstop)
            return self, tstart, tstop
        resolved = self.resolve_lags(tstart, tstop)
        # A window shared by all terms in both models moves to the model-wide bounds
        if window := _shared_window(resolved.models):
            return replace(resolved, x1=resolved.x1.without_lags(), x0=resolved.x0.without_lags()), *window
        return resolved, None, None

    def _cache_form_(self) -> str:
        """Canonical expanded form for cache keys and manifests"""
        return self.compose_name()

    @classmethod
    def coerce(
            cls,
            x,
            named_models: dict[str, Model] = {},
    ) -> Model | Comparison:
        if isinstance(x, (cls, Model)):
            return x
        comp = parse_comparison(x)
        return comp.initialize(named_models)

    @classmethod
    def _coerce(
            cls,
            x,
    ) -> Comparison:
        """Coerce ``x`` and require a comparison rather than a bare model"""
        comparison = cls.coerce(x)
        if not isinstance(comparison, cls):
            raise TypeError(f"{x=}: need a model comparison")
        return comparison

    def __repr__(self):
        return f"<Comparison: {self.name}>"

    def term_table(self):
        """Generate a table comparing the terms in the two models"""
        return model_comparison_table(self.x1, self.x0)


# term
name = Word(alphanums + '_')
stimulus = Word(alphanums + '_', alphanums + '_-')
stimulus_prefix = stimulus + Literal('~').suppress().leave_whitespace()
lag_value = Regex(r'-?(\d+\.?\d*|\.\d+)').add_parse_action(lambda s, l, t: float(t[0]))
lags = Literal('[').suppress() + Optional(lag_value, None) + Literal(':').suppress() + Optional(lag_value, None) + Literal(']').suppress()
term = Optional(stimulus_prefix, '') + DelimitedList(name, '-', combine=True, min=1) + Optional(Group(lags), None)
term.add_parse_action(lambda s, l, t: Term(t[0] or None, t[1], *(t[2] if t[2] is not None else (None, None))))

# model
model = DelimitedList(term, '+').add_parse_action(lambda s, l, t: Model(tuple(t)))
null_model = Keyword('0', ident_chars=alphanums + '_-~').add_parse_action(lambda s, l, t: Model(()))

# comparison
direct_comparison = model + one_of('= < >') + (null_model | model)
direct_comparison.add_parse_action(lambda s, l, t: DirectComparison(*t))
omit_comparison = model + Literal('@').suppress() + model
omit_comparison.add_parse_action(lambda s, l, t: OmitComparison(*t))
omit2_comparison = model + Literal('@').suppress() + direct_comparison
omit2_comparison.add_parse_action(lambda s, l, t: Omit2Comparison(t[0], t[1].x, t[1].operator, t[1].x0))
add_comparison = model + Literal('+@').suppress() + model
add_comparison.add_parse_action(lambda s, l, t: AddComparison(*t))
add2_comparison = model + Literal('+@').suppress() + direct_comparison
add2_comparison.add_parse_action(lambda s, l, t: Add2Comparison(t[0], t[1].x, t[1].operator, t[1].x0))
comparison = direct_comparison ^ omit_comparison ^ omit2_comparison ^ add_comparison ^ add2_comparison


def parse_term(string: str) -> Term:
    try:
        parse = term.parse_string(string, True)
    except ParseException:
        raise TRFModelError(f"{string!r}: invalid term")
    return parse[0]


def parse_model(string: str) -> Model:
    try:
        parse = model.parse_string(string, True)
    except ParseException:
        raise TRFModelError(f"{string!r}: invalid model")
    return parse[0]


def parse_comparison(string: str) -> ComparisonSpec:
    try:
        parse = comparison.parse_string(string, True)
    except ParseException:
        raise TRFModelError(f"{string!r}: invalid comparison")
    return parse[0]


def save_models(models, path):
    path = Path(path)
    out = [(k, v.name) for k, v in models.items()]
    if path.exists():
        backup_path = path.with_suffix('.backup')
        if backup_path.exists():
            backup_path.unlink()
        path.rename(backup_path)
    with open(path, 'wb') as fid:
        pickle.dump(out, fid, pickle.HIGHEST_PROTOCOL)


def load_models(path):
    with open(path, 'rb') as fid:
        out = pickle.load(fid)
    return {k: parse_model(v) for k, v in out}


ModelArg = Model | str
