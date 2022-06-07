# PulsarTimingAnalysis

The intention of this repository is to present a package with all the tools needed to perform the periodicity analysis of pulsars at Very High Energy using Python and focusing on the LST-1 telescope analysis chain:
  1. Modelling and calculation of phases with LST-1 and Fermi-LAT data
  2. Statistics, building phaseograms and search for pulsations with LST-1 and Fermi-LAT data


# Installation
The easiest way to install the package is by cloning the repository and creating a new environment with the environment.yml

```
git clone https://github.com/alvmas/PulsarTimingAnalysis.git
cd PulsarTimingAnalysis
conda env create -n pulsar-lst1 -f environment.yml
conda activate pulsar-lst1
pip install .
```

