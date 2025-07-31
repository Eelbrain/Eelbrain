# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import product
import os

from eelbrain.testing import TempDir
from eelbrain._experiment import TreeModel, FileTree


class Tree(TreeModel):
    _templates = dict(apath="/{afield}/",
                      afield=('a1', 'a2', 'a3'),
                      field2=('', 'value'))

    def __init__(self, **kwargs):
        TreeModel.__init__(self, **kwargs)
        self._register_compound('cmp', ('afield', 'field2'))


def test_tree():
    "Test simple formatting in the tree"
    tree = Tree()
    assert tree.get('apath') == '/a1/'
    vs = []
    for v in tree.iter('afield'):
        vs.append(v)
        assert tree.get('apath') == '/%s/' % v
        tree.set(afield='a3')
        assert tree.get('afield') == 'a3'
        assert tree.get('apath') == '/a3/'

    assert vs == ['a1', 'a2', 'a3']
    assert tree.get('afield') == 'a1'

    # test compound
    assert tree.get('cmp') == 'a1'
    tree.set(field2='value')
    assert tree.get('cmp') == 'a1_value'
    tree.set(field2='')
    assert tree.get('cmp') == 'a1'

    # test temporary state
    with tree._temporary_state:
        tree.set(afield='a2')
        assert tree.get('afield') == 'a2'
    assert tree.get('afield') == 'a1'

    # iterate
    assert list(tree.iter('afield')) == ['a1', 'a2', 'a3']
    assert list(tree.iter(('afield', 'field2'))) == [('a1', ''), ('a1', 'value'), ('a2', ''), ('a2', 'value'), ('a3', ''), ('a3', 'value')]
    assert list(tree.iter('cmp')) == ['a1', 'a1_value', 'a2', 'a2_value', 'a3', 'a3_value']
    assert list(tree.iter(('afield', 'cmp'))) == [('a1', 'a1'), ('a1', 'a1_value'), ('a2', 'a2'), ('a2', 'a2_value'), ('a3', 'a3'), ('a3', 'a3_value')]


class SlaveTree(TreeModel):
    _templates = {'path': '{a}_{b}_{sb_comp}_{slave}'}

    def __init__(self, a_seq, b_seq, c_seq):
        TreeModel.__init__(self)
        self._register_field('a', a_seq)
        self._register_field('b', b_seq, allow_empty=True)
        self._register_field('c', c_seq)
        self._register_compound('ab', ('a', 'b'))
        self._register_slave_field('s', 'a', lambda f: f['a'].upper())
        self._register_compound('sb', ('s', 'b'))
        self._register_slave_field('comp_slave', 'sb', lambda f: f['sb'].upper())
        # compound involving slave field
        self._register_field('s_a', a_seq, depends_on='c', slave_handler=self._update_sa)
        self._register_field('s_b', b_seq, depends_on='c', slave_handler=self._update_sb, allow_empty=True)
        self._register_compound('s_ab', ('s_a', 's_b'))
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
    ab_seq = [f'{a}_{b}' if b else a for a, b in product(a_seq, b_seq)]
    tree = SlaveTree(a_seq, b_seq, c_seq)

    # set
    assert tree.get('a') == 'a1'
    tree.set(a='a2')
    assert tree.get('a') == 'a2'
    tree.set(ab='a1 b2')
    assert tree.get('a') == 'a1'
    assert tree.get('b') == 'b2'
    tree.set(ab='a3')
    assert tree.get('a') == 'a3'
    assert tree.get('b') == ''

    tree.reset()
    assert tree.get('ab') == 'a1_b1'
    assert tree.get('sb') == 'A1_b1'
    assert tree.get('comp_slave') == 'A1_B1'
    tree.set(a='a2')
    assert tree.get('ab') == 'a2_b1'
    assert tree.get('sb') == 'A2_b1'
    assert tree.get('comp_slave') == 'A2_B1'

    # compound involving slave field
    tree.set(c='c2')
    assert tree.get('s_ab') == 'a2_b2'
    tree.set(c='c1')
    assert tree.get('s_ab') == 'a1_b1'

    # finde terminal keys
    assert tree.find_keys('c') == ['c']
    assert tree.find_keys('ab') == ['a', 'b']

    # .iter()
    assert list(tree.iter('a')) == a_seq
    assert list(tree.iter(('a', 'b'))) == list(product(a_seq, b_seq))
    assert list(tree.iter(('b', 'a'))) == list(product(b_seq, a_seq))
    # iter compound
    assert list(tree.iter('ab')) == ab_seq
    assert list(tree.iter(('c', 'ab'))) == list(product(c_seq, ab_seq))
    assert list(tree.iter('ab', values={'b': ''})) == a_seq
    assert list(tree.iter('ab', b='')) == a_seq


def test_file_tree():
    "Test file management tree"
    class Tree(FileTree):
        _templates = {'a-folder': '{root}/{folder}',
                      'a-file': '{a-folder}/{name}.txt',
                      'folder': ('f1', 'f2'),
                      'name': ('a1', 'a2', 'a3')}

        def __init__(self, *args, **kwargs):
            FileTree.__init__(self, *args, **kwargs)
            self._bind_make('a-file', self._make_a)

        def load_a(self):
            with open(self.get('a-file', make=True)) as fid:
                return fid.read()

        def _make_a(self):
            with open(self.get('a-file', mkdir=True), 'w') as fid:
                fid.write(self.format("{folder} {name}"))

    root = TempDir()
    tree = Tree(root=root)
    for _ in tree.iter_temp('a-file'):
        assert tree.load_a() == tree.format("{folder} {name}")

    for i, fname in enumerate(tree.iter_temp('a-file')):
        assert os.path.exists(fname)
    assert i == 5

    assert tree._glob_pattern('a-file', True, folder='f1') == f'{root}/f1/*.txt'

    tree.rm('a-file', name='*', folder='f1', confirm=True)
    for fname in tree.iter_temp('a-file', folder='f1'):
        assert fname[-6:-4] == tree.get('name')
        assert not os.path.exists(fname)
    for fname in tree.iter_temp('a-file', folder='f2'):
        assert fname[-6:-4] == tree.get('name')
        assert os.path.exists(fname)
