.. _ptiming_ana:

########################################
Pulsar Timing Analysis (``ptiming_ana``)
########################################

.. currentmodule:: ptiming_ana

**Version**: |version| **Date**: |today|

**Useful links**:
`Source Repository <https://github.com/cta-observatory/PulsarTimingAnalysis>`__ |
`Issue Tracker <https://github.com/cta-observatory/PulsarTimingAnalysis/issues>`__ |
`Changelog <https://github.com/cta-observatory/PulsarTimingAnalysis/blob/main/CHANGELOG.md>`__

**License**: BSD-3

**Python**: |python_requires|

``ptiming_ana`` is a Python package to perform the periodicity analysis of the
gamma-ray emission from pulsars. It is focused on the data of the LST-1, the
first Large-Sized Telescope of the Cherenkov Telescope Array Observatory
(CTAO), and also supports *Fermi*-LAT data and generic lists of events. It is
built on top of `PINT <https://nanograv-pint.readthedocs.io>`__,
`Gammapy <https://gammapy.org>`__ and
`lstchain <https://cta-observatory.github.io/cta-lstchain/>`__.

The package covers the steps of a pulsar analysis:

1. **Phase computation** (`ptiming_ana.cphase`): barycentric correction and
   computation of the rotational phase of each event from the ephemeris of the
   pulsar.
2. **Phaseogram** (`ptiming_ana.phaseogram`): pulse profile, statistics of the
   peaks, periodicity tests, fits of the peaks and evolution of the signal with
   time and energy.
3. **Spectral analysis** (`ptiming_ana.spectral`): phase-resolved spectral
   energy distribution of the peaks.

.. _ptiming_ana_docs:

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   userguide
   auto_examples/index
   api-reference/index
   contributing


Citing
======

If you use the package in a publication, please cite the version used from
Zenodo: https://doi.org/10.5281/zenodo.13385378

The package was used in *A detailed study of the very-high-energy Crab pulsar
emission with the LST-1*, CTA-LST Project, A&A, 690, A167 (2024),
https://doi.org/10.1051/0004-6361/202450059


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
