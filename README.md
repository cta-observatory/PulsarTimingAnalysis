# PulsarTimingAnalysis
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13385378.svg)](https://doi.org/10.5281/zenodo.13385378)

Tools to perform the periodicity analysis of gamma-ray emission from pulsars. It focuses on the data from the LST-1 telescope of the future CTAO gamma-ray observatory. It is built on top of Gammapy and PINT libraries.
  1. Modelling and calculation of phases with LST-1 and Fermi-LAT data
  2. Statistics, building phaseograms and search for pulsations with LST-1 and Fermi-LAT data


# Installation
The easiest way to install the package is by cloning the repository and creating a new environment using `environment.yml`

```
git clone https://github.com/cta-observatory/PulsarTimingAnalysis.git
cd PulsarTimingAnalysis
conda env create -n pulsar-lst1 -f environment.yml
conda activate pulsar-lst1
pip install .
```

# Cite
If you use the package in a publication, please cite the version used from Zenodo: https://doi.org/10.5281/zenodo.13385378

This analysis library was used in the following publications:

* A detailed study of the very-high-energy Crab pulsar emission with the LST-1, CTA-LST Project, A&A (2024) [arXiv:2407.02343](https://doi.org/10.48550/arXiv.2407.02343)
  
