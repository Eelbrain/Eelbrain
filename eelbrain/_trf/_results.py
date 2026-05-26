import enum

from eelbrain import fmtxt
from eelbrain.fmtxt import FMTextArg
from eelbrain._text import ms
from eelbrain import test as test_, testnd
from eelbrain._stats.test import star


class TestType(enum.Enum):
    DIFFERENCE = enum.auto()
    CORRELATION = enum.auto()
    MULTI_EFFECT = enum.auto()
    TWO_STAGE = enum.auto()

    @classmethod
    def for_test(cls, test):
        if isinstance(test, testnd.LMGroup):
            return cls.TWO_STAGE
        elif isinstance(test, testnd.MultiEffectNDTest):
            return cls.MULTI_EFFECT
        elif isinstance(test, testnd.Correlation):
            return cls.CORRELATION
        else:
            return cls.DIFFERENCE


class DependentType(enum.Enum):
    UNIVARIATE = enum.auto()
    MASS_UNIVARIATE = enum.auto()

    @classmethod
    def for_test(cls, test):
        if isinstance(test, (testnd.LMGroup, testnd.NDTest)):
            return cls.MASS_UNIVARIATE
        else:
            return cls.UNIVARIATE


class ResultCollection(dict):
    dependent_type = None
    test_type = None
    _statistic = None

    def __init__(self, tests: dict = ()):
        dict.__init__(self, tests)
        for key, test in self.items():
            self._validate_test(test)

    def _validate_test(self, test):
        test_type = TestType.for_test(test)
        dependent_type = DependentType.for_test(test)
        if self.test_type is None:
            self.test_type = test_type
            self.dependent_type = dependent_type
            if test_type is not TestType.TWO_STAGE:
                self._statistic = test._statistic
        elif test_type is not self.test_type or dependent_type is not self.dependent_type:
            raise TypeError(f"{test}: all tests need to be of the same type ({self.dependent_type} {self.test_type})")

    def __reduce__(self):
        return self.__class__, (dict(self),)

    def __setitem__(self, key, test):
        self._validate_test(test)
        dict.__setitem__(self, key, test)

    def __repr__(self):
        return f"<ResultCollection: {', '.join(self.keys())}>"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise NotImplementedError
        table = str(self.table()).splitlines()
        lines = ("<ResultCollection:", *(f'  {line}' for line in table), '>')
        p.text('\n'.join(lines))

    def _default_plot_obj(self):
        out = [test._default_plot_obj() for test in self.values()]
        if isinstance(out[0], list):
            return [y for ys in out for y in ys]
        return out

    def clusters(self, p=0.05):
        """Table with significant clusters"""
        if self.dependent_type is not DependentType.MASS_UNIVARIATE:
            raise RuntimeError('Clusters only available for mass-univariate tests')
        elif self.test_type is TestType.TWO_STAGE:
            raise NotImplementedError
        else:
            table = fmtxt.Table('lrrrrll')
            table.cells('Effect', 't-start', 't-stop', fmtxt.symbol(self._statistic, 'max'), fmtxt.symbol('t', 'peak'), fmtxt.symbol('p'), 'sig', just='l')
            table.midrule()
            for key, res in self.items():
                table.cell(key)
                table.endline()
                clusters = res.find_clusters(p, maps=True)
                clusters.sort('tstart')
                if self.test_type is not TestType.MULTI_EFFECT:
                    clusters[:, 'effect'] = ''
                for effect, tstart, tstop, p_, sig, cmap in clusters.zip('effect', 'tstart', 'tstop', 'p', 'sig', 'cluster'):
                    max_stat, max_time = res._max_statistic(mask=cmap != 0, return_time=True)
                    table.cells(f'  {effect}', ms(tstart), ms(tstop), fmtxt.stat(max_stat), ms(max_time), fmtxt.p(p_), sig)
        return table

    def table(
            self,
            title: FMTextArg = None,
            caption: FMTextArg = None,
            wide: bool = False,
    ):
        """Table with effects and smallest p-value"""
        is_mass_univariate = self.dependent_type is DependentType.MASS_UNIVARIATE
        sub = 'max' if is_mass_univariate else None
        if self.test_type is TestType.TWO_STAGE:
            cols = sorted({col for res in self.values() for col in res.column_names})
            table = fmtxt.Table('l' * (1 + len(cols)), title=title, caption=caption)
            table.cell('')
            table.cells(*cols)
            table.midrule()
            for key, lmg in self.items():
                table.cell(key)
                for res in (lmg.tests[c] for c in cols):
                    pmin = res.p.min()
                    table.cell(fmtxt.FMText([fmtxt.p(pmin), star(pmin)]))
        elif self.test_type is TestType.MULTI_EFFECT:
            if wide:
                ress = list(self.values())
                effects = ress[0].effects
                assert all(res.effects == effects for res in ress[1:])
                table = fmtxt.Table('l' * (1 + len(effects)), title=title, caption=caption)
                table.cells('Test', *effects)
                table.midrule()
                for key, res in self.items():
                    table.cell(key)
                    for i, effect in enumerate(res.effects):
                        if is_mass_univariate:
                            p = res.p[i].min()
                        else:
                            p = res.f_tests[i].p
                        table.cell(fmtxt.peq(p, stars=True))
                return table
            table = fmtxt.Table('lllll', title=title, caption=caption)
            table.cells('Test', 'Effect', fmtxt.symbol(self._statistic, sub), fmtxt.symbol('p'), 'sig')
            table.midrule()
            for key, res in self.items():
                for i, effect in enumerate(res.effects):
                    table.cells(key, effect)
                    if is_mass_univariate:
                        stat = res._max_statistic(i)
                        p = res.p[i].min()
                    else:
                        stat = res.f_tests[i].F
                        p = res.f_tests[i].p
                    table.cell(fmtxt.stat(stat))
                    table.cell(fmtxt.p(p))
                    table.cell(star(p))
                    key = ''
        else:
            table = fmtxt.Table('lllll', title=title, caption=caption)
            table.cells('Effect', 'df', fmtxt.symbol(self._statistic, sub), fmtxt.symbol('p'), 'sig')
            table.midrule()
            for key, res in self.items():
                table.cell(key)
                table.cell(getattr(res, 'df', ''))
                if is_mass_univariate:
                    stat = res._max_statistic()
                    p = res.p.min()
                else:
                    stat = getattr(res, res._statistic.lower())
                    p = res.p
                table.cell(fmtxt.stat(stat))
                table.cell(fmtxt.p(p))
                table.cell(star(p))
        return table
