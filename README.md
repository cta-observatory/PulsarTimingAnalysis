# PulsarTimingAnalysis
[![CI tests](https://github.com/cta-observatory/PulsarTimingAnalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cta-observatory/PulsarTimingAnalysis/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cta-observatory/PulsarTimingAnalysis/graph/badge.svg?token=5xVpLUeWFZ)](https://codecov.io/gh/cta-observatory/PulsarTimingAnalysis)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13385378.svg)](https://doi.org/10.5281/zenodo.13385378)
[![Docs](https://img.shields.io/badge/View-documentation-blue)](https://cta-observatory.github.io/PulsarTimingAnalysis/)

Tools to perform the periodicity analysis of gamma-ray emission from pulsars. It focuses on the data from the LST-1 telescope of the future CTAO gamma-ray observatory. It is built on top of Gammapy and PINT libraries.
  1. Modelling and calculation of phases with LST-1 and Fermi-LAT data
  2. Statistics, building phaseograms and search for pulsations with LST-1 and Fermi-LAT data


# Installation
The easiest way to install the package is by cloning the repository and creating a new environment using `environment.yml`

```
git clone https://github.com/cta-observatory/PulsarTimingAnalysis.git
cd PulsarTimingAnalysis
mamba env create -n pulsar-lst1 -f environment.yml
conda activate pulsar-lst1
pip install .
```
**Note:** For macOS users with M chips, you may need to install the package `c-blosc2` first (since lstchain requires `python-blosc2`, which is not available for M chips architectures):

```
mamba create -c conda-forge -n pulsar-lst1 python=3.11 c-blosc2 protozfits=2 protobuf=3.20
```
Then install PulsarTimingAnalysis with the rest of the dependencies with `pip install .` after activating the environment.

# Cite
If you use the package in a publication, please cite the version used from Zenodo: https://doi.org/10.5281/zenodo.13385378

This analysis library was used in the following publications:

* *A detailed study of the very-high-energy Crab pulsar emission with the LST-1*, CTA-LST Project, A&A, 690, A167 (2024) https://doi.org/10.1051/0004-6361/202450059
  
