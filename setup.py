# https://packaging.python.org/en/latest/
# https://packaging.python.org/en/latest/guides/modernize-setup-py-project/
import os
import platform
import sys
from setuptools import setup, find_packages, Extension

import numpy as np

# Source distribution includes C code to allow installing without Cython
# https://cython.readthedocs.io/en/stable/src/userguide/source_files_and_compilation.html#distributing-cython-modules
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False


IS_WINDOWS = os.name == 'nt'
IS_MACOS = sys.platform == 'darwin'
IS_AMD64 = platform.machine().lower() in ('x86_64', 'amd64')

# Cython extensions
base_args = {'define_macros': [("NPY_NO_DEPRECATED_API", "NPY_1_11_API_VERSION")]}
ext_kwargs = {}
setup_kwargs = {}
# Adapt ab3 from (Apache)
# https://github.com/joerick/python-abi3-package-sample
if sys.version_info.minor >= 11 and platform.python_implementation() == "CPython":
    # Can create an abi3 wheel (typed memoryviews first available in 3.11)!
    base_args["define_macros"].append(("Py_LIMITED_API", "0x030B0000"))
    ext_kwargs["py_limited_api"] = True
    setup_kwargs["options"] = {"bdist_wheel": {"py_limited_api": "cp311"}}
UNIX_COMPILE_ARGS = ['-Wno-unreachable-code', '-O3']
if IS_WINDOWS:
    open_mp_args = {
        **base_args,
        'extra_compile_args': '/openmp',
    }
elif IS_MACOS:
    # Not everyone will have OpenMP installed, so give a code path that allows building
    # without it
    if os.getenv("OPENMP_DISABLED") == "1":
        OPENMP_ARGS = []
        OPENMP_LINK_ARGS = []
    else:
        OPENMP_ARGS = ['-Xpreprocessor', '-fopenmp']
        OPENMP_LINK_ARGS = ['-lomp']
    open_mp_args = {
        **base_args,
        'extra_compile_args': UNIX_COMPILE_ARGS + OPENMP_ARGS,
        'extra_link_args': OPENMP_LINK_ARGS,
    }
    base_args['extra_compile_args'] = UNIX_COMPILE_ARGS
else:  # Some flavor of Linux
    open_mp_args = {
        **base_args,
        'extra_compile_args': UNIX_COMPILE_ARGS + ['-fopenmp'],
        'extra_link_args': ['-fopenmp'],
    }
    base_args['extra_compile_args'] = UNIX_COMPILE_ARGS
if IS_AMD64 and not IS_WINDOWS:
    open_mp_args["extra_compile_args"].append("-mavx")
    base_args["extra_compile_args"].append("-mavx")
ext = '.pyx' if cythonize else '.c'
ext_cpp = '.pyx' if cythonize else '.cpp'
extensions = [
    Extension('eelbrain._data_opt', [f'eelbrain/_data_opt{ext}'], **base_args),
    Extension('eelbrain._trf._boosting_opt', [f'eelbrain/_trf/_boosting_opt{ext}'], **open_mp_args),
    Extension('eelbrain._ndvar._convolve', [f'eelbrain/_ndvar/_convolve{ext}'], **open_mp_args),
    Extension('eelbrain._ndvar._gammatone', [f'eelbrain/_ndvar/_gammatone{ext}'], **base_args),
    Extension('eelbrain._stats.adjacency_opt', [f'eelbrain/_stats/adjacency_opt{ext}'], **base_args),
    Extension('eelbrain._stats.opt', [f'eelbrain/_stats/opt{ext}'], **base_args),
    Extension('eelbrain._stats.vector2d', [f'eelbrain/_stats/vector2d{ext_cpp}'], **base_args),
    Extension('eelbrain._stats.vector3d', [f'eelbrain/_stats/vector3d{ext_cpp}'], include_dirs=['dsyevh3C'], **base_args),
]
if cythonize:
    extensions = cythonize(extensions)

setup(
    include_dirs=[np.get_include()],
    packages=find_packages(),
    ext_modules=extensions,
    scripts=['bin/eelbrain'],
    **setup_kwargs,
)
