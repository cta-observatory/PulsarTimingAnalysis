# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The notes of
the releases are also available in the
[GitHub releases](https://github.com/cta-observatory/PulsarTimingAnalysis/releases).

## Unreleased

### Changes in the results

- **Z²ₙ test**: the statistic was computed with `n + 1` harmonics but compared
  with a χ² distribution with `2n` degrees of freedom, overestimating the
  significance. It is now computed with the first `n` harmonics.
- **H test**: the maximum is now searched in the first 20 harmonics (instead of
  24), as defined in de Jager et al. (1989) and assumed by the parametrisation
  of the p-value from de Jager & Büsching (2010).
- The significance of the periodicity tests is computed from the logarithm of
  the p-value, so it stays finite for very strong signals (it was `inf` before).
- The significance per unit of time (`sign_t_ratio`) of the final results is
  computed with the total observation time. It was computed with the time of
  the last step of the time evolution, which for Fermi-LAT data was the sum of
  the time differences shorter than 1 s.
- The time evolution of Fermi-LAT, text file and list inputs uses the elapsed
  time without the gaps between observations (it was the sum of the time
  differences shorter than 1 s).
- The last point of the time evolution always includes all the events. Events
  after the last time interval were not included in the final results if the
  last event was separated by a long gap from the previous one.
- The observation time of a directory of DL2 files is the sum of the effective
  times of the files before the cuts (it was computed after the cuts).
- Integer cuts (e.g. `intensity: 50`, `gammaness: 0`) were silently ignored.
  They are now applied.
- `model_fromephem` raises an error if the `.gro` ephemeris file does not cover
  the observation, instead of silently using the last line of the file.
- The unbinned fit is now an extended maximum likelihood fit (the models were
  not normalised) and returns the parameters in the same units as the binned
  fit. It does not modify the phases of the analysis anymore.

### Added

- Support for the current versions of the dependencies (tested with
  `lstchain` 0.12, `pint-pulsar` 1.1, `gammapy` 1.3, NumPy 2.2, pandas 3.0 and
  `matplotlib` 3.10 on Python 3.11). Python 3.12 is added to the CI.
- `ptiming_ana.setup_logging` to show the log messages of the package.
- `add_fermi_phase_table` command-line tool.
- `tobs` argument in `PulsarAnalysis.setListsInput` and `ReadList` to provide
  the observation time of a list of events.
- `--source-name` and `--observatory` options in `add_DL2_phase_table`.
- Registry of the pulse profile models (`ptiming_ana.phaseogram.models.MODELS`).
- Test suite with more than 250 tests of the three subpackages.
- `CITATION.cff`, contribution guidelines, code of conduct and changelog.
- User guide in the documentation.

### Changed

- The minimum supported version of Python is 3.10.
- Dependencies are not pinned to exact versions anymore; `iminuit`, `pyyaml`
  and `regions` are declared as dependencies and `more-itertools` is not needed.
- `ptiming_ana` does not configure the logging system or silence other loggers
  when it is imported.
- The plotting functions do not call `plt.show()`.
- The output files of `DL3_calphase` and `fermi_calphase` are named
  `{name}_pulsar.fits` also for `.fits.gz` inputs (it was `{name}.gz_pulsar.fits`),
  and the output directory does not need to end with `/`.
- `PulsarAnalysis` is not a dataclass anymore: two analysis objects are equal
  only if they are the same object.
- Boolean options of the command-line tools can be used as flags (`--pickle`).
- The results can be saved giving a directory or a PDF file name.
- The energies are shown in GeV with a logarithmic axis without hard-coded ticks.
- Continuous integration: tests are run in pull requests and weekly, with the
  oldest and newest supported dependencies; the documentation is only deployed
  from the main branch; lint and package build jobs added.

### Fixed

- `from ptiming_ana import *` failed because `spectral` was not imported.
- `PulsarAnalysis(filename=...)` did not raise an error for unknown file formats.
- `set_config` failed for source-independent DL2 analyses with extra cuts
  (`theta_cut` argument) and when saving the results (output file).
- `energy: 0` in the configuration file raised a `TypeError`.
- Energy-dependent cuts failed if the gammaness cut was not energy-dependent.
- `ReadList` repeated a list of times 86400 times instead of converting them.
- `ReadtxtFile` failed if any optional column was missing and could not read
  the `t2` (theta2) and `at` (telescope altitude) columns.
- Single-peak models (`gaussian`, `lorentzian`) could not be fitted to P1.
- Binned fits failed if the phaseogram had empty bins.
- Uncertainties of the FWHM of the asymmetric model and mean phases of P2.
- `PhaseRegions.remove_peak` left the object in an inconsistent state.
- The observatory code was ignored when no `.tim` file was created.
- `add_fermipulsarphase` imported a module that does not exist.
- `--pickle False` was interpreted as `True` in the command-line tools.
- Invalid escape sequences (`SyntaxWarning` in Python 3.12).
- Dependabot configuration was not in the right location.

### Removed

- `ptiming_ana.cphase.add_pulsarphase` (obsolete duplicate of `add_DL2_phase_table`).
