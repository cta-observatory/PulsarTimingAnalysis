# Contributing to PulsarTimingAnalysis

Contributions are very welcome: bug reports, feature requests, improvements of
the documentation and code. This document explains how to contribute.

## Reporting problems and asking for help

- **Bugs and feature requests**: open an issue in the
  [issue tracker](https://github.com/cta-observatory/PulsarTimingAnalysis/issues).
  For bugs, please include a minimal example that reproduces the problem, the
  full error message and the versions of `ptiming_ana`, Python, `lstchain`,
  `gammapy` and `pint-pulsar` (`pip list`).
- **Questions and support**: open an issue with the `question` label or contact
  the maintainers listed in `pyproject.toml`.

Please check first whether the problem has already been reported.

## Development environment

```bash
git clone https://github.com/cta-observatory/PulsarTimingAnalysis.git
cd PulsarTimingAnalysis
mamba env create -f environment.yml
mamba activate pulsar-lst1
pip install -e ".[all]"
pre-commit install
```

## Workflow

1. Create a branch from `main` (or fork the repository if you do not have
   write access).
2. Make your changes, including tests and documentation.
3. Run the tests and the linter (see below).
4. Open a pull request describing what was changed and why, and link the
   related issues. Add the change to the *Unreleased* section of `CHANGELOG.md`.
5. The pull request is merged after the review of one of the maintainers, once
   the continuous integration passes.

## Tests

The tests are in the `tests` directory and are run with
[pytest](https://docs.pytest.org):

```bash
pytest                      # tests that do not need network access
pytest --run-network        # also the tests that download clock files and ephemerides
pytest --cov                # with the coverage report
```

New features and bug fixes must be accompanied by tests. Tests must be
deterministic (use seeded random number generators) and should not depend on
the working directory.

## Code style

- The code is formatted and linted with [ruff](https://docs.astral.sh/ruff/).
  It is run automatically before each commit if `pre-commit` is installed, or
  manually with `pre-commit run --all-files`.
- Public functions and classes are documented with docstrings following the
  [numpydoc](https://numpydoc.readthedocs.io) style.
- Use the `logging` module instead of `print`.
- Physical quantities in configuration files and docstrings must state their units.

## Documentation

The documentation is built with Sphinx:

```bash
cd docs
make html
```

and can be opened in `docs/build/html/index.html`.

## Releases

Releases are created by the maintainers from the GitHub interface (the version
number is taken from the git tag by `setuptools_scm`). Each release is archived
in [Zenodo](https://doi.org/10.5281/zenodo.13385378).

## Code of conduct

All the participants are expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
