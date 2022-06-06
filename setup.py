#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup, find_packages
import os

entry_points = {}
entry_points["console_scripts"] = [
    "add_phase_interp = ptiming_ana.cphase.add_phase_interp:main",
    "add_pulsarphase = ptiming_ana.cphase.add_pulsarphase:main",
    "merge_pulsar_files= ptiming_ana.cphase.merge_pulsar_files:main",
    
]

setup(
    use_scm_version={"write_to":os.path.join("ptiming_ana","_version.py")},
    packages=find_packages(),
    install_requires=[
        'astropy>=4.0.5,<5',
        'lstchain~=0.9.0',
        'gammapy~=0.19.0',
        'h5py',
        'matplotlib>=3.5',
        'numba',
        'numpy',
        'pandas',
        'scipy',
        'tables',
        'protobuf~=3.20.0',
        'toml',
        'pint-pulsar',
        'setuptools_scm',
    ],
    entry_points=entry_points,
)
