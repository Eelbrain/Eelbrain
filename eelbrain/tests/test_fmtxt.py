import os
import shutil
import tempfile

import numpy as np

from eelbrain import fmtxt
from eelbrain.testing import TempDir, requires_framework_build
from eelbrain.fmtxt import html, tex, read_meta
from eelbrain import datasets, plot


def test_code():
    "test fmtxt.Code"
    s = '&nbsp;'
    br = '<br style="clear:left">\n'
    assert html(fmtxt.Code("a = 5\nb = a + 2")) == f"<code>a{s}={s}5{br}b{s}={s}a{s}+{s}2</code>"


def test_eq():
    "Test equation factory"
    s = fmtxt.eq('t', 0.1234)
    assert str(s) == "t = 0.12"
    assert html(s) == "t = 0.12"
    assert tex(s) == "$t = 0.12$"

    s = fmtxt.eq('t', 0.1234, 18)
    assert str(s) == "t(18) = 0.12"
    assert html(s) == "t<sub>18</sub> = 0.12"
    assert tex(s) == "$t_{18} = 0.12$"

    s = fmtxt.peq(0.1299)
    assert str(s) == "p = .130"
    assert html(s) == "p = .130"
    assert tex(s) == "$p = .130$"

    s = fmtxt.peq(0.0009)
    assert str(s) == "p < .001"
    assert html(s) == "p &lt; .001"
    assert tex(s) == "$p < .001$"


def test_fmtext():
    "Tet FMText base class"
    t = fmtxt.FMText('test')
    print(t)

    tuc = fmtxt.FMText('FMText with unicode: \x80abc')
    print(str(tuc))

    ts = fmtxt.FMText((t, tuc, 'unicode: \x80abc'))

    print(str(ts))
    print(html(ts))
    print(tex(ts))


def test_image():
    "Test FMText Image"
    tempdir = TempDir()
    filename = os.path.join(tempdir, 'rgb.png')
    rgba = np.random.uniform(0, 1, (100, 100, 4))
    rgb = rgba[:, :, :3]

    for array in (rgb, rgba):
        im = fmtxt.Image.from_array(array, alt='array')
        im.save_image(filename)
        assert im.get_html().startswith('<img src=')

        im2 = fmtxt.Image.from_file(filename, alt='array')
        assert im.get_html() == im2.get_html()


def test_list():
    list_ = fmtxt.List("Head")
    assert str(list_) == 'Head'
    assert html(list_) == 'Head\n<ul>\n</ul>'
    list_.add_item("child")
    assert str(list_) == 'Head\n- child'
    assert html(list_) == 'Head\n<ul>\n<li>child</li>\n</ul>'
    sublist = list_.add_sublist("unicode:")
    assert str(list_) == 'Head\n- child\n- unicode:'
    assert html(list_) == 'Head\n<ul>\n<li>child</li>\n<li>unicode:\n<ul>\n</ul></li>\n</ul>'
    sublist.add_item('delta: ∂')
    assert str(list_) == 'Head\n- child\n- unicode:\n  - delta: ∂'
    assert html(list_) == 'Head\n<ul>\n<li>child</li>\n<li>unicode:\n<ul>\n<li>delta: ∂</li>\n</ul></li>\n</ul>'


def test_p():
    assert str(fmtxt.p(.02)) == '.020'
    assert str(fmtxt.p(.2, stars=True)) == '.200   '
    assert str(fmtxt.p(.0119, stars=True)) == '.012*  '
    assert str(fmtxt.p(.0001, stars=True)) == '< .001***'


@requires_framework_build
def test_report():
    "Test fmtxt.Report class"
    tempdir = tempfile.mkdtemp()
    report = fmtxt.Report("Test Report")

    section = report.add_section('unicode: \xe2 abc')
    ds = datasets.get_uv()
    p = plot.Barplot('fltvar', 'A', sub="B=='b1'", data=ds, show=False)
    image = p.image()
    section.add_figure("test", image)

    report.sign()

    # report output
    print(report)
    dst = os.path.join(tempdir, 'report.html')
    report.save_html(dst)

    # test meta attribute reading
    assert read_meta(dst) == {}
    report.save_html(dst, meta={'samples': 100, 'text': 'blah'})
    assert read_meta(dst) == {'samples': '100', 'text': 'blah'}

    # clean up
    shutil.rmtree(tempdir)


def test_table():
    table = fmtxt.Table('ll')
    table.cells('A', 'B')
    table.midrule()
    table.cells('a1', 'b1', 'a2', 'b2')
    assert str(table) == 'A    B \n-------\na1   b1\na2   b2'
    assert html(table) == '<figure><table border="1" frame="hsides" rules="none" cellpadding="2"><tr>\n<td style="text-align:left">A</td>\n<td style="text-align:left">B</td>\n</tr>\n<tr>\n<td style="text-align:left">a1</td>\n<td style="text-align:left">b1</td>\n</tr>\n<tr>\n<td style="text-align:left">a2</td>\n<td style="text-align:left">b2</td>\n</tr></table></figure>'
    assert table.get_rtf() == (
        '\\trowd\n\\cellx0000\n\\cellx1000\n\\row\n'
        'A\\intbl\\cell\nB\\intbl\\cell\n\\row\n'
        'a1\\intbl\\cell\nb1\\intbl\\cell\n\\row\n'
        'a2\\intbl\\cell\nb2\\intbl\\cell\n\\row')
    assert table.get_tex() == (
        '\\begin{center}\n\\begin{tabular}{ll}\n\\toprule\n'
        'A & B \\\\\n\\midrule\n'
        'a1 & b1 \\\\\na2 & b2 \\\\\n'
        '\\bottomrule\n\\end{tabular}\n\\end{center}')

    # right-align
    table.columns = 'lr'
    assert html(table) == '<figure><table border="1" frame="hsides" rules="none" cellpadding="2"><tr>\n<td style="text-align:left">A</td>\n<td style="text-align:right">B</td>\n</tr>\n<tr>\n<td style="text-align:left">a1</td>\n<td style="text-align:right">b1</td>\n</tr>\n<tr>\n<td style="text-align:left">a2</td>\n<td style="text-align:right">b2</td>\n</tr></table></figure>'
    table.columns = 'll'

    # empty table
    assert str(fmtxt.Table('')) == ''

    # saving
    tempdir = TempDir()
    # HTML
    path = os.path.join(tempdir, 'test.html')
    table.save_html(path)
    assert open(path).read() == '<!DOCTYPE html>\n<html>\n<head>\n    <title>Untitled</title>\n<style>\n\n.float {\n    float:left\n}\n\n</style>\n</head>\n\n<body>\n\n%s\n\n</body>\n</html>\n' % html(table)
    # rtf
    path = os.path.join(tempdir, 'test.rtf')
    table.save_rtf(path)
    assert open(path).read() == (
        '{\\rtf1\\ansi\\deff0\n\n'
        '\\trowd\n\\cellx0000\n\\cellx1000\n\\row\n'
        'A\\intbl\\cell\nB\\intbl\\cell\n\\row\n'
        'a1\\intbl\\cell\nb1\\intbl\\cell\n\\row\n'
        'a2\\intbl\\cell\nb2\\intbl\\cell\n\\row\n}')
    # TeX
    path = os.path.join(tempdir, 'test.tex')
    table.save_tex(path)
    assert open(path).read() == (
        '\\begin{center}\n\\begin{tabular}{ll}\n\\toprule\n'
        'A & B \\\\\n\\midrule\n'
        'a1 & b1 \\\\\na2 & b2 \\\\\n'
        '\\bottomrule\n\\end{tabular}\n\\end{center}')
    # txt
    path = os.path.join(tempdir, 'test.txt')
    table.save_txt(path)
    assert open(path).read() == 'A    B \n-------\na1   b1\na2   b2'

    # editing
    table[0, 0] = 'X'
    assert str(table) == 'X    B \n-------\na1   b1\na2   b2'
    table[0] = ['C', 'D']
    assert str(table) == 'C    D \n-------\na1   b1\na2   b2'
    table[2, 0] = 'cd'
    assert str(table) == 'C    D \n-------\ncd   b1\na2   b2'
    table[2:4, 1] = ['x', 'y']
    assert str(table) == 'C    D\n------\ncd   x\na2   y'


def test_symbol():
    "Test fmtxt.symbol()"
    s = fmtxt.symbol('t', 21)
    assert str(s) == 't(21)'
    assert html(s) == 't<sub>21</sub>'
    assert tex(s) == '$t_{21}$'
