"""
ptiming_ana: periodicity analysis of gamma-ray emission from pulsars.

Subpackages
-----------
cphase
    Computation of the pulsar phases (LST-1 DL2/DL3 and Fermi-LAT data).
phaseogram
    Phaseograms, peak statistics, periodicity tests and pulse profile fitting.
spectral
    Phase-resolved spectral analysis based on Gammapy.
"""

from . import cphase, phaseogram, spectral
from ._logging import setup_logging
from .version import __version__

__all__ = [
    "__version__",
    "cphase",
    "phaseogram",
    "setup_logging",
    "spectral",
]
