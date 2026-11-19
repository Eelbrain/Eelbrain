# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Summaries of several test results"""
from typing import Any

from .. import fmtxt
from .._stats.test import star
from .._stats.testnd import MultiEffectNDTest, NDTest


def result_table(
        ress: dict[str, Any],
        title: fmtxt.FMTextArg = None,
        caption: fmtxt.FMTextArg = None,
) -> fmtxt.Table:
    """Table listing several test results with their (smallest) p-values

    Parameters
    ----------
    ress
        ``{label: result}`` mapping of test results of the same kind: univariate
        (:mod:`eelbrain.test`) or mass-univariate (:mod:`eelbrain.testnd`).
        Multi-effect tests (ANOVA) get one row per effect.
    title
        Table title.
    caption
        Table caption.
    """
    if not ress:
        raise ValueError(f"{ress=}: no results")
    first = next(iter(ress.values()))
    is_mass_univariate = isinstance(first, NDTest)
    if is_mass_univariate and any(res.p is None for res in ress.values()):
        raise ValueError("Mass-univariate test results without p-values (samples=0): the result table needs p-values (use samples > 0)")
    sub = 'max' if is_mass_univariate else None
    if isinstance(first, MultiEffectNDTest) or hasattr(first, 'f_tests'):
        statistic = first._statistic if is_mass_univariate else 'F'
        table = fmtxt.Table('lllll', title=title, caption=caption)
        table.cells('Test', 'Effect', fmtxt.symbol(statistic, sub), fmtxt.symbol('p'), 'sig')
        table.midrule()
        for key, res in ress.items():
            if is_mass_univariate:
                rows = [(effect, res._max_statistic(i), res.p[i].min()) for i, effect in enumerate(res.effects)]
            else:
                rows = [(effect, f_test.f, f_test.p) for effect, f_test in zip(res.effects, res.f_tests)]
            for effect, stat, p in rows:
                table.cells(key, effect, fmtxt.stat(stat), fmtxt.p(p), star(p))
                key = ''
        return table
    table = fmtxt.Table('lllll', title=title, caption=caption)
    table.cells('Test', 'df', fmtxt.symbol(first._statistic, sub), fmtxt.symbol('p'), 'sig')
    table.midrule()
    for key, res in ress.items():
        if is_mass_univariate:
            stat, p = res._max_statistic(), res.p.min()
        else:
            stat, p = getattr(res, res._statistic.lower()), res.p
        table.cells(key, getattr(res, 'df', ''), fmtxt.stat(stat), fmtxt.p(p), star(p))
    return table
