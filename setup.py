# https://packaging.python.org/en/latest/
# https://packaging.python.org/en/latest/guides/modernize-setup-py-project/
from packaging.version import Version
import os
from pathlib import Path
import platform
import re
from setuptools import setup, find_packages, Extension

import numpy as np

# Source distribution includes C code to allow installing without Cython
# https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False


IS_WINDOWS = os.name == 'nt'
IS_ARM = platform.machine().lower().startswith('arm')

# Cython extensions
base_args = {'define_macros': [("NPY_NO_DEPRECATED_API", "NPY_1_11_API_VERSION")]}
if IS_WINDOWS:
    open_mp_args = {
        **base_args,
        'extra_compile_args': '/openmp',
    }
elif IS_ARM:
    open_mp_args = {
        **base_args,
        'extra_compile_args': ['-Wno-unreachable-code', '-Xpreprocessor', '-fopenmp', '-O3'],
        'extra_link_args': ['-Xpreprocessor', '-fopenmp'],
    }
    base_args['extra_compile_args'] = ['-Wno-unreachable-code', '-O3']
else:
    open_mp_args = {
        **base_args,
        'extra_compile_args': ['-Wno-unreachable-code', '-fopenmp', '-O3', '-mavx'],
        'extra_link_args': ['-fopenmp'],
    }
    base_args['extra_compile_args'] = ['-Wno-unreachable-code', '-O3', '-mavx']
ext = '.pyx' if cythonize else '.c'
ext_cpp = '.pyx' if cythonize else '.cpp'
extensions = [
    Extension('eelbrain._data_opt', [f'eelbrain/_data_opt{ext}'], **base_args),
    Extension('eelbrain._trf._boosting_opt', [f'eelbrain/_trf/_boosting_opt{ext}'], **open_mp_args),
    Extension('eelbrain._ndvar._convolve', [f'eelbrain/_ndvar/_convolve{ext}'], **open_mp_args),
    Extension('eelbrain._ndvar._gammatone', [f'eelbrain/_ndvar/_gammatone{ext}'], **base_args),
    Extension('eelbrain._stats.connectivity_opt', [f'eelbrain/_stats/connectivity_opt{ext}'], **base_args),
    Extension('eelbrain._stats.opt', [f'eelbrain/_stats/opt{ext}'], **base_args),
    Extension('eelbrain._stats.vector', [f'eelbrain/_stats/vector{ext_cpp}'], include_dirs=['dsyevh3C'], **base_args),
]
if cythonize:
    extensions = cythonize(extensions)

setup(
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=extensions,
    scripts=['bin/eelbrain'],
)
