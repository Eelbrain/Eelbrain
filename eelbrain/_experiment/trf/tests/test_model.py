import pytest

from eelbrain._experiment.trf.model import TRFModelError, Model, Comparison, Term, parse_term


def test_term():
    # plain term
    term = parse_term('gammatone')
    assert term.stimulus is None
    assert term.code == 'gammatone'
    assert term.key == 'gammatone'
    assert term.nuts_method is None
    assert term.uts_file_name == 'gammatone'

    term = parse_term('1~gammatone')
    assert term.stimulus == '1'
    assert term.code == 'gammatone'

    term = parse_term('1~gammatone-1')
    assert term.code == 'gammatone-1'

    # NUTS: stimulus + columns
    term = parse_term('stim~word')
    assert term.nuts_columns == (None, None)
    term = parse_term('stim~word-frequency')
    assert term.nuts_columns == ('frequency', None)
    term = parse_term('stim~word-frequency-noun')
    assert term.nuts_columns == ('frequency', 'noun')
    assert term.stimulus == 'stim'
    assert term.code == 'word-frequency-noun'
    assert term.key == 'stim_word_frequency_noun'
    assert term.nuts_method is None
    assert term.uts_file_name == 'stim~word-frequency-noun'
    assert term.nuts_file_name == 'stim~word'
    assert term.with_stimulus('other').string == 'other~word-frequency-noun'

    # NUTS method suffix
    term = parse_term('stim~word-surprisal-step')
    assert term.nuts_method == 'step'
    assert term.nuts_file_name == 'stim~word'
    assert term.string_without_nuts_method == 'stim~word-surprisal'

    # too many '-' separated elements (with columns)
    with pytest.raises(TRFModelError):
        parse_term('stim~word-a-b-c').nuts_columns
    # double '--'
    with pytest.raises(TRFModelError):
        parse_term('stim~word--b')


def test_term_lags():
    # lag-window overrides
    term = parse_term('gammatone[0.2:]')
    assert term.tstart == 0.2
    assert term.tstop is None
    assert term.string == 'gammatone[0.2:]'
    assert term.without_lags() == parse_term('gammatone')
    assert parse_term(term.string) == term

    term = parse_term('word[-0.1:0.8]')
    assert term.tstart == -0.1
    assert term.tstop == 0.8
    assert parse_term(term.string) == term

    term = parse_term('stim~word-frequency[:0.9]')
    assert term.stimulus == 'stim'
    assert term.code == 'word-frequency'
    assert term.nuts_columns == ('frequency', None)
    assert term.tstart is None
    assert term.tstop == 0.9
    assert term.string == 'stim~word-frequency[:0.9]'
    assert term.with_stimulus('other').string == 'other~word-frequency[:0.9]'

    # empty slice is a no-op
    assert parse_term('gammatone[:]') == parse_term('gammatone')

    # tstart must be smaller than tstop
    with pytest.raises(TRFModelError):
        parse_term('gammatone[0.5:0.2]')

    # sub-millisecond precision is not supported; bounds snap to the ms grid
    with pytest.raises(TRFModelError):
        parse_term('gammatone[0.0005:0.1]')
    assert Term(None, 'gammatone', 0.1 + 0.2, 0.5).tstart == 0.3

    # open lag bounds fill in from model-wide defaults
    assert parse_term('gammatone[0.2:]').with_default_lags(0, 0.5) == parse_term('gammatone[0.2:0.5]')
    assert parse_term('gammatone[-0.2:]').with_default_lags(0, 0.5) == parse_term('gammatone[-0.2:0.5]')
    assert parse_term('gammatone').with_default_lags(0, 0.5) == parse_term('gammatone[0:0.5]')
    with pytest.raises(TRFModelError):  # resolved window is empty
        parse_term('gammatone[0.6:]').with_default_lags(0, 0.5)

    # same predictor with different lag windows
    model = Model.coerce('gammatone[0:0.5] + gammatone[0.5:1]')
    assert model.name == 'gammatone[0:0.5] + gammatone[0.5:1]'
    with pytest.raises(TRFModelError):
        Model.coerce('gammatone[0:0.5] + gammatone[0:0.5]')
    # overlapping lag windows for the same predictor
    with pytest.raises(TRFModelError):
        Model.coerce('gammatone + gammatone[0.2:]')
    with pytest.raises(TRFModelError):
        Model.coerce('gammatone[0:0.6] + gammatone[0.5:1]')
    with pytest.raises(TRFModelError):
        Model.coerce('gammatone[:0.5] + gammatone[0.2:]')

    # comparison with lags
    comparison = Comparison.coerce('x + gammatone[0.2:] > x')
    assert comparison.x1_only.name == 'gammatone[0.2:]'

    # a Term instance passed as a predictor-node option is stripped like a string spelling (the predictor file is lag-independent)
    from eelbrain._experiment.trf.nodes import PredictorInput
    spec = PredictorInput.key_options['term']
    assert spec.validated(None, 'term', parse_term('gammatone[0.2:0.4]')) == parse_term('gammatone')
    assert spec.validated(None, 'term', 'gammatone[0.2:0.4]') == parse_term('gammatone')


def test_named_model_lags():
    # lag overrides distribute to member terms; explicit member lags take precedence
    named = {'ab': Model.coerce('a + b[0.5:1]')}
    model = Model.coerce('ab[0.2:0.8]', named)
    assert model.name == 'a[0.2:0.8] + b[0.5:1]'
    model = Model.coerce('ab', named)
    assert model.name == 'a + b[0.5:1]'


def test_comparison_lags():
    # omitting a lag window keeps the complement in the reduced model
    comparison = Comparison.coerce('a + b @ b[:1]')
    assert comparison.x1.name == 'a + b'
    assert comparison.x0.name == 'a + b[1:]'
    comparison = Comparison.coerce('a + b @ b[1:]')
    assert comparison.x0.name == 'a + b[:1]'
    # interior window: two-piece complement
    comparison = Comparison.coerce('a + b @ b[0.5:1]')
    assert comparison.x0.name == 'a + b[:0.5] + b[1:]'
    # complement within the term's own window; open bounds inherit the term's bound
    comparison = Comparison.coerce('a + b[0:2] @ b[0:1]')
    assert comparison.x1.name == 'a + b[0:2]'
    assert comparison.x0.name == 'a + b[1:2]'
    comparison = Comparison.coerce('a + b[0:2] @ b[:1]')
    assert comparison.x1.name == 'a + b[0:2]'
    assert comparison.x0.name == 'a + b[1:2]'
    # exact match: full removal
    comparison = Comparison.coerce('a + b[:1] @ b[:1]')
    assert comparison.x0.name == 'a'
    # bare term removes all lag windows of that predictor
    comparison = Comparison.coerce('a + b[:1] @ b')
    assert comparison.x0.name == 'a'
    comparison = Comparison.coerce('a + b[:0.5] + b[0.5:] @ b')
    assert comparison.x0.name == 'a'
    # two-sided omit: early vs late window
    comparison = Comparison.coerce('a + b @ b[:1] > b[1:]')
    assert comparison.x1.name == 'a + b[:1]'
    assert comparison.x0.name == 'a + b[1:]'
    assert comparison.name == 'a + b @ b[:1] > b[1:]'
    # add with lag windows
    comparison = Comparison.coerce('a +@ b[:1]')
    assert comparison.x1.name == 'a + b[:1]'
    assert comparison.x0.name == 'a'
    comparison = Comparison.coerce('a +@ b[:1] > b[1:]')
    assert comparison.x1.name == 'a + b[:1]'
    assert comparison.x0.name == 'a + b[1:]'
    # named model: window distributes to member terms, then complements per term
    named = {'ab': Model.coerce('a + b')}
    comparison = Comparison.coerce('a + b @ ab[:1]', named)
    assert comparison.x0.name == 'a[1:] + b[1:]'
    # subtracting a lag window from an expanded model keeps the complement
    model = Model.coerce('ab', named) - Model.coerce('b[:1]')
    assert model.name == 'a + b[1:]'

    # difference/intersection decompose lag windows
    comparison = Comparison.coerce('a + b @ b[:1]')
    assert comparison.x1_only.name == 'b[:1]'
    assert not comparison.x0_only
    assert comparison.common_base.name == 'a + b[1:]'
    assert comparison.test_term_name == 'b[:1]'

    # errors
    with pytest.raises(TRFModelError):  # window not contained in the term's window
        Comparison.coerce('a + b[:1] @ b[1:2]')
    with pytest.raises(TRFModelError):  # window straddles a split
        Comparison.coerce('a + b[:1] + b[1:] @ b[0.5:1.5]')
    with pytest.raises(TRFModelError):  # open omit bound extends into another piece
        Comparison.coerce('a + b[:1] + b[1:2] @ b[0.5:]')
    with pytest.raises(TRFModelError):
        Model.coerce('a + b[:1] + b[1:2]') - Model.coerce('b[0.5:]')
    with pytest.raises(TRFModelError):  # predictor not in model
        Comparison.coerce('a + b @ c[:1]')
    with pytest.raises(TRFModelError):  # add overlapping window
        Comparison.coerce('a + b +@ b[:1]')


def test_comparison_lags_resolved():
    "Comparison.resolve_lags re-derives the reduced model with explicit lag windows"
    # omit window beyond the model-wide window: the reduced model would exceed the full model
    with pytest.raises(TRFModelError):
        Comparison.coerce('a + b @ b[0.6:]').resolve_lags(0, 0.5)
    with pytest.raises(TRFModelError):
        Comparison.coerce('a + b @ b[:0.6]').resolve_lags(0, 0.5)
    # within the model-wide window: complements computed from the concrete windows
    comparison = Comparison.coerce('a + b @ b[0.2:]')
    assert comparison.x0.name == 'a + b[:0.2]'
    resolved = comparison.resolve_lags(0, 0.5)
    assert resolved.x1.name == 'a[0:0.5] + b[0:0.5]'
    assert resolved.x0.name == 'a[0:0.5] + b[0:0.2]'
    assert resolved.name == 'a + b @ b[0.2:]'  # display name preserved
    assert resolved.resolve_lags(0, 0.5) is resolved  # idempotent
    # two-sided omit: each omitted window is checked against the full model
    Comparison.coerce('a + b @ b[:0.2] > b[0.2:]').resolve_lags(0, 0.5)
    with pytest.raises(TRFModelError):
        Comparison.coerce('a + b @ b[:0.2] > b[0.6:]').resolve_lags(0, 0.5)
    # direct comparisons re-derive nothing; term windows may exceed the model-wide window
    Comparison.coerce('a + b[:0.6] > a').resolve_lags(0, 0.5)
    # an omit bound at the model-wide bound: no empty complement piece is created
    comparison = Comparison.coerce('a + b @ b[0:0.2]')
    assert comparison.x0.name == 'a + b[:0] + b[0.2:]'  # symbolic: piece for lags before 0
    resolved = comparison.resolve_lags(0, 0.5)
    assert resolved.x0.name == 'a[0:0.5] + b[0.2:0.5]'
    # the decomposition properties are exact on a resolved comparison
    assert resolved.x1_only.name == 'b[0:0.2]'
    assert not resolved.x0_only
    assert resolved.common_base.name == 'a[0:0.5] + b[0.2:0.5]'
    # ... including for mixed open/explicit bounds (which compare as unbounded unresolved)
    resolved = Comparison.coerce('a + b[-0.1:] > a + b').resolve_lags(0, 0.5)
    assert resolved.x1_only.name == 'b[-0.1:0]'
    assert not resolved.x0_only
    assert resolved.common_base.name == 'a[0:0.5] + b[0:0.5]'


def test_model_lags_resolution():
    "Model.resolve_lags: explicit windows from the model-wide defaults"
    model = Model.coerce('a + b[0.1:0.4]')
    resolved = model.resolve_lags(0, 0.5)
    assert resolved.name == 'a[0:0.5] + b[0.1:0.4]'
    assert resolved.resolve_lags(0, 0.5) is resolved  # idempotent
    # term windows beyond the model-wide window are allowed
    assert Model.coerce('a + b[0.6:0.8]').resolve_lags(0, 0.5).name == 'a[0:0.5] + b[0.6:0.8]'
    # if all terms specify complete windows, model-wide defaults are not needed
    assert Model.coerce('a[0:0.3] + b[0.1:0.4]').resolve_lags(None, None).name == 'a[0:0.3] + b[0.1:0.4]'
    with pytest.raises(TRFModelError):  # no tstart for a
        Model.coerce('a + b[0.1:0.4]').resolve_lags(None, 0.5)
    with pytest.raises(TRFModelError):  # no tstop for a
        Model.coerce('a + b[0.1:0.4]').resolve_lags(0, None)
    with pytest.raises(TRFModelError):  # resolved window of b is empty
        Model.coerce('a + b[0.6:]').resolve_lags(0, 0.5)
    with pytest.raises(TRFModelError):  # resolved window of b is zero-width
        Model.coerce('a + b[0.5:]').resolve_lags(0, 0.5)


def test_model_lags_normalization():
    "normalize_lags: canonical (x, tstart, tstop) for cache identity"
    # no overrides: the model-wide bounds are kept
    model = Model.coerce('a + b')
    assert model.normalize_lags(0, 0.5) == (model, 0, 0.5)
    with pytest.raises(TRFModelError):  # bounds required
        model.normalize_lags(0, None)
    with pytest.raises(TRFModelError):  # empty model-wide window
        model.normalize_lags(0.5, 0)
    # a window shared by all terms moves to the model-wide bounds
    assert Model.coerce('a[0:0.4] + b[0:0.4]').normalize_lags(0, 0.5) == (Model.coerce('a + b'), 0, 0.4)
    assert Model.coerce('a[0.1:] + b[0.1:]').normalize_lags(0, 0.5) == (Model.coerce('a + b'), 0.1, 0.5)
    # distinct windows: every term explicit, model-wide bounds None
    model, tstart, tstop = Model.coerce('a + b[0.1:0.4]').normalize_lags(0, 0.5)
    assert model.name == 'a[0:0.5] + b[0.1:0.4]'
    assert tstart is None and tstop is None
    assert model.normalize_lags(None, None) == (model, None, None)  # idempotent
    # comparison: a window shared across both models moves to the model-wide bounds
    comparison, tstart, tstop = Comparison.coerce('a[0:0.4] + b[0:0.4] > a[0:0.4]').normalize_lags(0, 0.5)
    assert comparison.x1.name == 'a + b'
    assert comparison.x0.name == 'a'
    assert (tstart, tstop) == (0, 0.4)
    # mixed comparison: explicit windows, model-wide bounds None
    comparison, tstart, tstop = Comparison.coerce('a + b @ b[0.2:]').normalize_lags(0, 0.5)
    assert comparison.x1.name == 'a[0:0.5] + b[0:0.5]'
    assert comparison.x0.name == 'a[0:0.5] + b[0:0.2]'
    assert tstart is None and tstop is None
    assert comparison.normalize_lags(tstart, tstop) == (comparison, None, None)  # idempotent (omit record cleared on resolution)


def test_term_table_lags():
    table = Model.coerce('a + b[0.1:0.4]').term_table()
    assert str(table).splitlines()[0].split() == ['#', 'Code', 'tstart', 'tstop']
    table = Model.coerce('a + b').term_table()
    assert str(table).splitlines()[0].split() == ['#', 'Code']


models = {
    'x-abcd': 'x-a + x-b + x-c + x-d',
    'x-ab': 'x-a + x-b',
    'x-cd': 'x-c + x-d',
    'xyz': 'x + y + z',
}
named_models = {k: Model.coerce(v) for k, v in models.items()}


def test_model():
    xyz = Model.coerce('x + y + z')
    xy = Model.coerce('x + y')
    yz = Model.coerce('y + z')
    y = Model.coerce('y')
    z = Model.coerce('z')
    assert xy + z == xyz
    assert xyz - z == xy
    assert xy.intersection(yz) == y
    # coercing a Model instance expands named models too
    model = Model.coerce('x-ab + z')
    assert Model.coerce(model, named_models) == Model.coerce('x-a + x-b + z')
    # duplicate term
    with pytest.raises(TRFModelError):
        Model.coerce("term-1 + term-2 + term-2")


# comparison, cv, x1, x0, name
test_data = [
    # direct
    ('x + a > x + b', 'x + a', 'x + b'),
    ('x = x + y', 'x', 'x + y'),
    ('x > 0', 'x', '0'),
    ('x + a > 0', 'x + a', '0'),
    # omit
    ('x + y @ y', 'x + y', 'x'),
    ('x + y @ x', 'x + y', 'y'),
    # add
    ('x +@ y', 'x + y', 'x'),
    ('x +@ y = z', 'x + y', 'x + z'),
    ('x + y +@ z', 'x + y + z', 'x + y'),
    # named direct
    ('x-ab < x-cd', 'x-a + x-b', 'x-c + x-d'),
    # named omit
    ('x-ab @ x-b', 'x-a + x-b', 'x-a'),
    ('x-abcd @ x-ab', 'x-a + x-b + x-c + x-d', 'x-c + x-d'),
    # named add
    ('x-ab +@ x-c', 'x-a + x-b + x-c', 'x-a + x-b'),
    # named add2
    ('x-ab +@ x-c > x-d', 'x-a + x-b + x-c', 'x-a + x-b + x-d'),
    #
    ('x-abcd @ x-ab', 'x-a + x-b + x-c + x-d', 'x-c + x-d'),
    ('x-abcd @ x-ab = x-cd', 'x-a + x-b', 'x-c + x-d'),
]
# allow name being different from args[0]
test_data = [(*t, None) if len(t) == 3 else t for t in test_data]


@pytest.mark.parametrize('string,x1,x0,name', test_data, ids=[items[0] for items in test_data])
def test_comparison(string: str, x1: str, x0: str, name: str | None):
    """Assert that comparison is parsed correctly"""
    if name is None:
        name = string

    # Make sure it is not mis-recognized as model
    with pytest.raises(TRFModelError):
        Model.coerce(string)

    comparison = Comparison.coerce(string, named_models)

    assert isinstance(comparison, Comparison)
    assert comparison.x1.name == x1
    assert comparison.x0.name == x0
    assert comparison.name == name
    if x0 == '0':
        assert not comparison.x0


def test_comparison_parser():
    with pytest.raises(TRFModelError):
        Comparison.coerce('model @ whot$shift', named_models)

    for x0 in ('0foo', '01foo', '0-foo', '0~foo'):
        comparison = Comparison.coerce(f'x > {x0}')
        assert comparison.x0.name == x0


def test_comparison_cache_form():
    named = Comparison.coerce('x-ab < x-cd', named_models).sorted()
    expanded = Comparison._coerce(named._cache_form_())
    assert expanded.x1 == named.x1
    assert expanded.x0 == named.x0
    assert expanded.tail == named.tail
    assert expanded._cache_form_() == 'x-a + x-b < x-c + x-d'
    with pytest.raises(TRFModelError, match='invalid comparison'):
        Comparison._coerce('x + y')
