# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import product

from eelbrain._experiment import TreeModel


class Tree(TreeModel):
    _templates = dict(apath="/{afield}/",
                      afield=('a1', 'a2', 'a3'),
                      field2=('', 'value'))


def test_tree():
    "Test simple formatting in the tree"
    tree = Tree()
    assert tree.get('apath') == '/a1/'
    vs = []
    for v in tree.iter('afield'):
        vs.append(v)
        assert tree.get('apath') == f'/{v}/'
        tree.set(afield='a3')
        assert tree.get('afield') == 'a3'
        assert tree.get('apath') == '/a3/'

    assert vs == ['a1', 'a2', 'a3']
    assert tree.get('afield') == 'a1'

    # test temporary state
    with tree._temporary_state:
        tree.set(afield='a2')
        assert tree.get('afield') == 'a2'
    assert tree.get('afield') == 'a1'

    # iterate
    assert list(tree.iter('afield')) == ['a1', 'a2', 'a3']
    assert list(tree.iter(('afield', 'field2'))) == [('a1', ''), ('a1', 'value'), ('a2', ''), ('a2', 'value'), ('a3', ''), ('a3', 'value')]
    assert tree.find_keys('apath') == ['afield']


class SlaveTree(TreeModel):
    _templates = {'path': '{a}_{b}_{s}_{s_a}_{s_b}'}

    def __init__(self, a_seq, b_seq, c_seq):
        TreeModel.__init__(self)
        self._register_field('a', a_seq)
        self._register_field('b', b_seq, allow_empty=True)
        self._register_field('c', c_seq)
        self._register_slave_field('s', 'a', lambda f: f['a'].upper())
        self._register_field('s_a', a_seq, depends_on='c', slave_handler=self._update_sa)
        self._register_field('s_b', b_seq, depends_on='c', slave_handler=self._update_sb, allow_empty=True)
        self._store_state()

    @staticmethod
    def _update_sa(fields):
        if fields['c'] == 'c1':
            return 'a1'
        else:
            return 'a2'

    @staticmethod
    def _update_sb(fields):
        if fields['c'] == 'c1':
            return 'b1'
        else:
            return 'b2'


def test_slave_tree():
    a_seq = ['a1', 'a2', 'a3']
    b_seq = ['b1', 'b2', '']
    c_seq = ['c1', 'c2']
    tree = SlaveTree(a_seq, b_seq, c_seq)

    # set
    assert tree.get('a') == 'a1'
    tree.set(a='a2')
    assert tree.get('a') == 'a2'
    assert tree.get('s') == 'A2'
    tree.set(b='b2')
    assert tree.get('b') == 'b2'

    tree.reset()
    assert tree.get('path') == 'a1_b1_A1_a1_b1'
    tree.set(a='a2')
    assert tree.get('path') == 'a2_b1_A2_a1_b1'

    tree.set(c='c2')
    assert tree.get('s_a') == 'a2'
    assert tree.get('s_b') == 'b2'
    tree.set(c='c1')
    assert tree.get('s_a') == 'a1'
    assert tree.get('s_b') == 'b1'

    # finde terminal keys
    assert tree.find_keys('c') == ['c']
    assert tree.find_keys('path') == ['a', 'b', 's_a', 's_b']

    # .iter()
    assert list(tree.iter('a')) == a_seq
    assert list(tree.iter(('a', 'b'))) == list(product(a_seq, b_seq))
    assert list(tree.iter(('b', 'a'))) == list(product(b_seq, a_seq))
    assert list(tree.iter(('a', 'b'), values={'b': ''})) == [(a, '') for a in a_seq]
    assert list(tree.iter(('a', 'b'), b='')) == [(a, '') for a in a_seq]
