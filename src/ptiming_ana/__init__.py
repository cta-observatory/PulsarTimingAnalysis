"""
ptiming_ana: periodicity analysis of gamma-ray emission from pulsars.

Subpackages
-----------
cphase
    Computation of the pulsar phases (LST-1 DL2/DL3 and Fermi-LAT data).
io
    Reading of LST-1 DL2 files produced by lstchain or CTLearn.
phaseogram
    Phaseograms, peak statistics, periodicity tests and pulse profile fitting.
spectral
    Phase-resolved spectral analysis based on Gammapy.
"""

from . import cphase, io, phaseogram, spectral
from ._logging import setup_logging
from .version import __version__

__all__ = [
    "__version__",
    "cphase",
    "io",
    "phaseogram",
    "setup_logging",
    "spectral",
]
