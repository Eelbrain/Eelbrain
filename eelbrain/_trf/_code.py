from functools import cached_property
import re
from typing import List

from eelbrain import Dataset
from eelbrain._experiment.definitions import CodeBase, CodeError
import numpy as np

from .._ndvar import SHUFFLE_METHODS as NDVAR_SHUFFLE_METHODS


VALUE_SHUFFLE_METHODS = ('permute', 'remask', 'relocate')
SHUFFLE_METHODS = NDVAR_SHUFFLE_METHODS + VALUE_SHUFFLE_METHODS
NUTS_METHODS = ('step', 'is')


class Code(CodeBase):
    """Predictor specification with '-' delimited instructions

    Attributes
    ----------
    string : str
        The complete code string.
    stim : str
        The stimulus; the part preceding ``|``.
    code : str
        The main body of the code; the part between ``|`` and ``$``.
    shuffle : str
        The shuffling method.
    has_permutation : bool
        The regressor is distorted in some way.
    has_randomization : bool
        The regressor is distorted in a way that includes randomness.
    """
    _sep = '-'
    _seed = None

    def __init__(self, string):
        m = re.match(
            r'(?:([\w+-]+)\~)?'  # stimulus
            r'([\w:-]+)'  # pedictor code
            r'(?:\$'  # begin shuffling
            r'(?:\[(-?\d+-?|\w*)\])?'  # band/index
            r'([a-zA-Z]+)(\d*))?$', string)
        if not m:
            raise CodeError(string, "not a valid code")
        stim, code_string, shuffle_index, shuffle, angle = m.groups()
        if shuffle:
            index_str = '' if shuffle_index is None else f'[{shuffle_index}]'
            self.shuffle_string = f"${index_str}{shuffle}{angle}"
            self.code_with_rand = f'{code_string}{self.shuffle_string}'
            if angle:
                angle = int(angle)
                if angle == 180:
                    raise CodeError(string, "shuffle angle '180' should be omitted")
                elif not 360 > angle > 0:
                    raise CodeError(string, f"shuffle angle {angle}")
            else:
                angle = 180

            if shuffle_index:
                m = re.match(r'^(-?)(\d+)(-?)$', shuffle_index)
                if m:
                    pre, index, post = m.groups()
                    if pre:
                        if post:
                            raise ValueError(f'{string!r} (shuffle index)')
                        shuffle_index = slice(int(index))
                    elif post:
                        shuffle_index = slice(int(index), None)
                    else:
                        shuffle_index = int(index)
            else:
                shuffle_index = None
        else:
            self.code_with_rand = code_string
            self.shuffle_string = ''
            shuffle_index = shuffle = angle = None
        self.stim = stim or None
        self.code = code_string
        self.shuffle = shuffle
        self.shuffle_index = shuffle_index
        self.shuffle_angle = angle
        CodeBase.__init__(self, string, code_string)
        self.has_randomization = shuffle in VALUE_SHUFFLE_METHODS or '>' in string
        self.has_permutation = shuffle in SHUFFLE_METHODS or '>' in string
        self._shuffle_done = False
        self.key = Dataset.as_key(self.string)

    @classmethod
    def from_strings(cls, stim: str, items: List[str], shuffle_string: str = ''):
        stim = f'{stim}~' if stim else ''
        code = cls._sep.join(items)
        return cls(f"{stim}{code}{shuffle_string}")

    def register_string_done(self):
        self._i = len(self._items) - 1

    def register_shuffle(self, index=False):
        "Register that shuffling has been done"
        if not self.shuffle:
            raise self.error(f"Shuffling not requested", -1)
        elif self._shuffle_done:
            raise self.error(f"Shuffling twice", -1)
        elif self.shuffle_index and not index:
            raise self.error("Shuffle index not used", -1)
        self._shuffle_done = True

    def assert_done(self):
        CodeBase.assert_done(self)
        if self.shuffle and not self._shuffle_done:
            raise self.error("Shuffling not performed", -1)

    def _get_rng(self):
        if self._seed is None:
            raise RuntimeError(f"{self} not seeded")
        return np.random.RandomState(self._seed)

    def seed(self, subject=None):
        "Seed random state"
        if not self.shuffle:
            return
        elif self._seed is not None:
            raise RuntimeError(f"{self} seeded twice")
        if subject is None:
            seed = 0
            angle_magnitude = 10
        else:
            digits = ''.join(re.findall(r'\d', subject))
            seed = int(digits)
            angle_magnitude = 10**len(digits)
        seed += self.shuffle_angle * angle_magnitude
        self._seed = seed

    @property
    def string_without_rand(self):
        if self.stim:
            return f'{self.stim}~{self.code}'
        else:
            return self.code

    def with_stim(self, stim=None):
        "Copy of the code with different stimulus"
        code_string = self.string
        if self.stim:
            i = len(self.stim) + 1
            code_string = code_string[i:]
        if stim is None:
            return code_string
        return Code(f'{stim}~{code_string}')

    @cached_property
    def nuts_method(self):
        if self._items[-1] in NUTS_METHODS:
            return self._items[-1]

    @cached_property
    def nuts_columns(self):
        n = len(self._items) - 1 - bool(self.nuts_method)
        # column
        if n > 0:
            column = self._items[1]
        else:
            column = None
        # mask
        if n <= 1:
            mask = None
        elif n == 2:
            mask = self._items[2]
        else:
            raise self.error("Wrong number of elements")
        return column, mask

    def nuts_file_name(self, columns: bool):
        if columns:
            code = self.from_strings(self.stim, self._items[:1])
        elif self._items[-1] in NUTS_METHODS:
            code = self.from_strings(self.stim, self._items[:-1])
        else:
            code = self
        return code.string_without_rand


def parse_index(code):
    """Parse current item in ``code`` for an index expression

    Returns
    -------
    index : None | int | slice
        The index that was found.
    """
    item = code.lookahead()
    if not item:
        return
    m = re.match(r"^(\d?)(:?)(\d?)$", item)
    if not m:
        return
    code.next()
    start, colon, stop = m.groups()
    if start or stop:
        start = int(start) if start else None
        stop = int(stop) if stop else None
        if stop is not None and stop < (start or 0) + 2:
            raise code.error("Redundant slice definition (length 1)")
    else:
        raise code.error("Index does nothing")
    if colon:
        if start == 0:
            raise code.error("Redundant definition (omit '0' in '0:')")
        return slice(start, stop)
    else:
        assert stop is None
        return start
