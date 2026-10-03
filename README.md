# PulsarTimingAnalysis

[![CI tests](https://github.com/cta-observatory/PulsarTimingAnalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/cta-observatory/PulsarTimingAnalysis/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cta-observatory/PulsarTimingAnalysis/graph/badge.svg?token=5xVpLUeWFZ)](https://codecov.io/gh/cta-observatory/PulsarTimingAnalysis)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13385378.svg)](https://doi.org/10.5281/zenodo.13385378)
[![Docs](https://img.shields.io/badge/View-documentation-blue)](https://cta-observatory.github.io/PulsarTimingAnalysis/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE.md)

`ptiming_ana` is a Python package to perform the periodicity analysis of the
gamma-ray emission from pulsars. It is focused on the data of the LST-1, the
first Large-Sized Telescope of the Cherenkov Telescope Array Observatory
(CTAO), and also supports *Fermi*-LAT data and generic lists of events. It is
built on top of [PINT](https://github.com/nanograv/PINT),
[Gammapy](https://gammapy.org) and
[lstchain](https://github.com/cta-observatory/cta-lstchain).

## Features

The package is organised in three subpackages that cover the steps of a pulsar
analysis:

| Subpackage | Description |
|---|---|
| `ptiming_ana.cphase` | Barycentric correction and computation of the pulsar phases of LST-1 (DL2 and DL3) and *Fermi*-LAT events from the ephemeris of the pulsar (`.par` files or Jodrell Bank `.gro` files), including a fast interpolation method. |
| `ptiming_ana.phaseogram` | Phaseograms, statistics of the peaks (Li&Ma significance, excess, P1/P2 ratio), periodicity tests (χ², Z²ₙ and H tests), fits of the pulse profile and evolution of the signal with time and energy. |
| `ptiming_ana.spectral` | Phase-resolved spectral analysis (spectral energy distribution of each peak) from DL3 files using Gammapy. |

## Installation

The dependencies are easily installed with `mamba`/`conda` using the
environment file of the repository:

```bash
git clone https://github.com/cta-observatory/PulsarTimingAnalysis.git
cd PulsarTimingAnalysis
mamba env create -f environment.yml
mamba activate pulsar-lst1
pip install .
```

On Linux the package and its dependencies can also be installed only with `pip`
(`pip install .`). On macOS the conda environment is needed, since some of the
dependencies of `lstchain` are not available in PyPI for that platform.

> **Note:** PINT runs with reduced numerical precision on platforms without
> extended precision floating-point numbers (e.g. Apple Silicon). The phases
> are still accurate to ~1e-5 for the Crab pulsar, but Linux x86-64 is
> recommended to compute the phases of the final results.

See the [documentation](https://cta-observatory.github.io/PulsarTimingAnalysis/)
for more details.

## Quick start

Phaseogram, statistics of the peaks and periodicity tests of the *Fermi*-LAT
observations of the Crab pulsar included in the repository:

```python
import ptiming_ana
from ptiming_ana.phaseogram import PulsarAnalysis

ptiming_ana.setup_logging()  # show the progress and the results

analysis = PulsarAnalysis()
analysis.setFermiInputFile("tutorial/example_data/merged2_pulsar.fits")
analysis.setBackgroundLimits([0.52, 0.87])
analysis.setPeaklimits(P1_limits=[0, 0.026, 0.983, 1], P2_limits=[0.377, 0.422])
analysis.setTimeInterval(tint=3600 * 24 * 30)
analysis.setFittingParams(model="dgaussian", binned=True)
analysis.run()

peak_statistics, periodicity_tests = analysis.show_Presults()
analysis.show_fit_results()
analysis.draw_phaseogram(phase_limits=[0, 2], fit=True)
```

All the settings can also be given in a [configuration file](config.yaml)
(`analysis.set_config("config.yaml")`). A complete example is available in the
[tutorial](https://cta-observatory.github.io/PulsarTimingAnalysis/auto_examples/index.html).

The pulsar phases of LST-1 data are computed with the command-line tools:

```bash
add_DL3_phase_table --in-file dl3_LST-1.Run07253.fits --ephem crab.par --output DL3_pulsar/
add_DL2_phase_table --in-file dl2_LST-1.Run07253.h5 --ephem crab.par --interpolation
```

## Tests

```bash
pip install ".[test]"
pytest                 # add --run-network to run the tests that download data
```

## Contributing and support

Bug reports, questions and contributions are welcome through the
[issue tracker](https://github.com/cta-observatory/PulsarTimingAnalysis/issues).
See the [contribution guidelines](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## Citing

If you use the package in a publication, please cite the version used from
Zenodo: https://doi.org/10.5281/zenodo.13385378 (see also
[CITATION.cff](CITATION.cff)).

This analysis library was used in the following publications:

* *A detailed study of the very-high-energy Crab pulsar emission with the LST-1*, CTA-LST Project, A&A, 690, A167 (2024) https://doi.org/10.1051/0004-6361/202450059

## License

`ptiming_ana` is distributed under the [BSD 3-Clause License](LICENSE.md).
