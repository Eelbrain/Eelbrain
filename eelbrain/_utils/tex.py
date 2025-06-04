"""Interface for TeX command line tools

Adapted from the obsolete ``tex`` module by Volker Grabsch


Original Notes
--------------
Temporary files are always cleaned up.
The TeX interpreter is automatically re-run as often as necessary,
and an exception is thrown
in case the output fails to stabilize soon enough.
The TeX interpreter is always run in batch mode,
so it won't ever get in your way by stopping your application
when there are issues with your TeX source.
Instead, an exception is thrown
that contains all information of the TeX log.


License
-------
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import subprocess
import tempfile


def convert(tex_source, input_format, output_format, max_runs=5):
    '''Convert LaTeX or TeX source to PDF or DVI.'''
    # check arguments
    assert isinstance(tex_source, str)
    try:
        (tex_cmd, output_suffix) = {
            ('tex',   'dvi'): ('tex',      '.dvi'),
            ('latex', 'dvi'): ('latex',    '.dvi'),
            ('tex',   'pdf'): ('pdftex',   '.pdf'),
            ('latex', 'pdf'): ('pdflatex', '.pdf'),
        }[(input_format, output_format)]
    except KeyError:
        raise ValueError('Unable to handle conversion: %s -> %s'
                         % (input_format, output_format))
    if max_runs < 2:
        raise ValueError('max_runs must be at least 2.')
    # create temporary directory
    with tempfile.TemporaryDirectory(prefix='tex-temp-') as tex_dir:
        # create LaTeX source file
        tex_filename = os.path.join(tex_dir, 'texput.tex')
        with open(tex_filename, 'w') as fid:
            fid.write(tex_source)

        # run LaTeX processor as often as necessary
        aux_old = None
        for i in range(max_runs):
            tex_process = subprocess.Popen(
                [tex_cmd,
                    '-interaction=batchmode',
                    '-halt-on-error',
                    '-no-shell-escape',
                    tex_filename,
                 ],
                stdin=open(os.devnull, 'r'),
                stdout=open(os.devnull, 'w'),
                stderr=subprocess.STDOUT,
                close_fds=True,
                shell=False,
                cwd=tex_dir,
                env={'PATH': os.getenv('PATH')},
            )
            tex_process.wait()
            if tex_process.returncode != 0:
                with open(os.path.join(tex_dir, 'texput.log'), 'rb') as fid:
                    log = fid.read()
                raise ValueError(log.decode())

            with open(os.path.join(tex_dir, 'texput.aux'), 'rb') as fid:
                aux = fid.read()

            if aux == aux_old:  # aux file stabilized
                output_filename = os.path.join(tex_dir, 'texput' + output_suffix)
                if not os.path.exists(output_filename):
                    raise RuntimeError('No output file was produced.')
                else:
                    with open(output_filename, 'rb') as fid:
                        return fid.read()
            aux_old = aux
            # TODO:
            # Also handle makeindex and bibtex,
            # possibly in a similar manner as described in:
            # http://vim-latex.sourceforge.net/documentation/latex-suite/compiling-multiple.html
        raise RuntimeError("%s didn't stabilize after %i runs" % ('texput.aux', max_runs))


def tex2dvi(tex_source, **kwargs):
    '''Convert TeX source to DVI.'''
    return convert(tex_source, 'tex', 'dvi', **kwargs)


def latex2dvi(tex_source, **kwargs):
    '''Convert LaTeX source to DVI.'''
    return convert(tex_source, 'latex', 'dvi', **kwargs)


def tex2pdf(tex_source, **kwargs):
    '''Convert TeX source to PDF.'''
    return convert(tex_source, 'tex', 'pdf', **kwargs)


def latex2pdf(tex_source, **kwargs):
    '''Convert LaTeX source to PDF.'''
    return convert(tex_source, 'latex', 'pdf', **kwargs)
