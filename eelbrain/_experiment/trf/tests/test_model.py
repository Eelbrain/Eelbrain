import pytest

from eelbrain._experiment.trf.model import TRFModelError, Model, ModelExpression, Comparison, parse_term


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
    # subtraction
    xy2 = ModelExpression.from_string("xyz - z").initialize(named_models)
    assert xy2 == xy
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


def test_comparison_parser():
    with pytest.raises(TRFModelError):
        Comparison.coerce('model @ whot$shift', named_models)
