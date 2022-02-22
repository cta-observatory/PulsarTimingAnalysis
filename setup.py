#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup, find_packages
import os

entry_points = {}
entry_points["console_scripts"] = [
    "add_phase_interp = cphase.add_phase_interp:main",
    "add_pulsarphase = cphase.add_pulsarphase:main",
]

docs_require = [
    "sphinx~=4.2",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx",
]

setup(
    packages=find_packages(),
    install_requires=[
        'astropy>=4.0.5,<5',
        'lstchain~=0.9.0',
        'h5py',
        'matplotlib>=3.5',
        'numba',
        'numpy',
        'pandas',
        'pyirf~=0.6.0',
        'scipy',
        'tables',
        'toml',
        'traitlets~=5.0.5',
        'setuptools_scm',
    ],
    entry_points=entry_points,
    extras_require={
        "all":  docs_require,
        "docs": docs_require,
    },
)
