# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Model specification for TRFs
"""
from __future__ import annotations

from collections import abc, Counter
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import chain
from operator import attrgetter
from pathlib import Path
import pickle
from collections.abc import Callable, Sequence

from pyparsing import DelimitedList, ParseException, Literal, Optional, Word, alphanums, one_of

from ..._data_obj import Dataset
from ... import fmtxt


COMP = {1: '>', 0: '=', -1: '<'}
TAIL = {'>': 1, '=': 0, '<': -1}
NUTS_METHODS = ('step', 'is')


class TRFModelError(Exception):
    """Error in TRF model specification"""


@dataclass(frozen=True)
class Term:
    stimulus: str | None
    code: str

    @cached_property
    def string(self) -> str:
        if self.stimulus:
            return f"{self.stimulus}~{self.code}"
        return self.code

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
        return self.string

    @cached_property
    def nuts_file_name(self) -> str:
        """File name (without extension) of the predictor file backing this term"""
        code = self._items[0]
        return f"{self.stimulus}~{code}" if self.stimulus else code

    def with_stimulus(self, stimulus: str) -> Term:
        """Copy of the term with a different stimulus"""
        return replace(self, stimulus=stimulus)

    @classmethod
    def _coerce(cls, x: Term | str):
        if isinstance(x, Term):
            return x
        elif isinstance(x, str):
            return parse_term(x)
        # elif isinstance(x, tuple):
        #     return cls._coerce(*x)
        raise TypeError(x)

    def __repr__(self):
        return f"<Term: {self.string}>"


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
        return named_models[term.code].terms
    else:
        return term,


@dataclass(frozen=True)
class Model:
    """Model that can be fit to data"""
    terms: tuple[Term, ...]
    public_name: str = None

    def __post_init__(self):
        counts = Counter([term.string for term in self.terms])
        duplicates = [term for term, count in counts.items() if count > 1]
        if duplicates:
            raise TRFModelError(f"{self.name}: duplicate terms {', '.join(duplicates)}")

    @cached_property
    def name(self) -> str:
        if not self.terms:
            return '0'
        return ' + '.join(term.string for term in self.terms)

    @cached_property
    def sorted_key(self) -> str:
        return '+'.join(sorted([term.string for term in self.terms]))

    def sorted(self) -> Model:
        return Model(tuple(sorted(self.terms, key=attrgetter('string'))))

    @cached_property
    def dataset_based_key(self):
        term_keys = [Dataset.as_key(term.string) for term in self.terms]
        return '+'.join(sorted(term_keys))

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
        if not all(term in self.terms for term in other.terms):
            missing = [term.string for term in other.terms if term not in self.terms]
            raise ValueError(f"{self.name} - {other.name}:\nMissing terms: {', '.join(missing)}")
        return Model(tuple([term for term in self.terms if term not in other.terms]))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    @classmethod
    def coerce(cls, x: Model | str | Sequence) -> Model:
        if isinstance(x, cls):
            return x
        elif isinstance(x, str):
            return cls.from_string(x)
        elif isinstance(x, abc.Sequence):
            return cls(tuple(Term._coerce(term) for term in x))
        raise TypeError(x)

    def difference(self, other: Model) -> Model:
        terms = [term for term in self.terms if term not in other.terms]
        return Model(tuple(terms))

    def intersection(self, other: Model) -> Model:
        terms = [term for term in self.terms if term in other.terms]
        return Model(tuple(terms))

    def initialize(self, named_models: dict[str, Model]) -> Model:
        terms = list(chain.from_iterable(_expand_term(term, named_models) for term in self.terms))
        return Model(tuple(terms))

    def term_table(self) -> fmtxt.Table:
        show_stimulus = any(term.stimulus for term in self.terms)
        t = fmtxt.Table('rl' * (1 + show_stimulus))
        t.cell('#')
        if show_stimulus:
            t.cell('Stimulus')
        t.cell('Code')
        t.midrule()
        for i, term in enumerate(self.terms):
            t.cell(i)
            if show_stimulus:
                t.cell(term.stimulus)
            t.cell(term.code)
        return t

    def without(self, term: str) -> Model:
        terms = list(self.terms)
        names = [term.string for term in terms]
        if term not in names:
            raise ValueError(f"{term}: not in {self.name}")
        del terms[names.index(term)]
        return Model(tuple(terms))


@dataclass
class ModelExpression:
    """Model specification using abbreviations"""
    base: Model
    subtract: Term = None

    @classmethod
    def from_string(
            cls,
            string: str,
    ) -> ModelExpression:
        try:
            return model_expr.parse_string(string, True)[0]
        except ParseException:
            raise TRFModelError(f"{string!r}: invalid Model")

    def initialize(
            self,
            named_models: dict[str, Model],
    ) -> Model:
        """Expand into full model"""
        base = self.base.initialize(named_models)
        if not self.subtract:
            return base
        # remove subtraction
        terms = list(base.terms)
        subtract = _expand_term(self.subtract, named_models)
        for term_i in subtract:
            terms.remove(term_i)
        return Model(tuple(terms))


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
    ) -> Comparison | Model:
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
        return Comparison(x, x0, 1, public_name)


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
        # x - x1_reduced > x - x0_reduced
        #     x0_reduced > x1_reduced
        x1 = x - x0_omit
        x0 = x - x1_omit
        return Comparison(x1, x0, TAIL[self.operator], public_name)


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

    @cached_property
    def operator(self) -> str:
        return COMP[self.tail]

    @cached_property
    def models(self) -> tuple[Model, Model]:
        return self.x1, self.x0

    @cached_property
    def common_base(self) -> Model:
        return self.x1.intersection(self.x0)

    @cached_property
    def x1_only(self) -> Model:
        return self.x1.difference(self.x0)

    @cached_property
    def x0_only(self) -> Model:
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

    def __repr__(self):
        return f"<Comparison: {self.name}>"

    def term_table(self):
        """Generate a table comparing the terms in the two models"""
        return model_comparison_table(self.x1, self.x0)


# term
name = Word(alphanums + '_')
stimulus = Word(alphanums + '_', alphanums + '_-')
stimulus_prefix = stimulus + Literal('~').suppress().leave_whitespace()
term = Optional(stimulus_prefix, '') + DelimitedList(name, '-', combine=True, min=1)
term.add_parse_action(lambda s, l, t: Term(t[0] or None, t[1]))

# model
model = DelimitedList(term, '+').add_parse_action(lambda s, l, t: Model(tuple(t)))
subtract_term = Literal('-').suppress() + term
model_expr = model + Optional(subtract_term)
model_expr.add_parse_action(lambda s, l, t: ModelExpression(*t))
null_model = Literal('0').add_parse_action(lambda s, l, t: Model(()))

# comparison
direct_comparison = model + one_of('= < >') + (model ^ null_model)
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
