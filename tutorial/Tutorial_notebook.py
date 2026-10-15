"""
Basic pulsar analysis using Fermi-LAT data
==========================================

This example shows how to produce the phaseogram of the Crab pulsar, compute
the statistics of its peaks, search for pulsations, fit the pulse profile and
study the evolution of the signal with time and energy.

It uses about 10 years of public *Fermi*-LAT observations of the Crab pulsar
(100 MeV - 1 GeV), for which the pulsar phases were already computed.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import ptiming_ana
from ptiming_ana.phaseogram import PulsarAnalysis

# sphinx_gallery_multi_image = "single"

# %%
# ``ptiming_ana`` reports the progress and the results of the analysis through
# the Python logging system. This shows the messages in the terminal/notebook:

ptiming_ana.setup_logging()


# %%
# Create the PulsarAnalysis object and settings
# ---------------------------------------------
#
# All the settings of the analysis can be read from a configuration file:

h = PulsarAnalysis()
h.set_config("./example_data/config_tutorial.yaml")

# %%
# Alternatively we can set the parameters directly
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The phase regions used to compute the statistics of the peaks (P1 and P2)
# and to estimate the background, the binning of the phaseogram, the time
# interval used to follow the evolution of the signal, the model used to fit
# the pulse profile and the energy binning:

h = PulsarAnalysis()
h.setBackgroundLimits([0.52, 0.87])
h.setPeaklimits(
    P1_limits=[0, 0.026, 0.983, 1], P2_limits=[0.377, 0.422], P3_limits=None
)
h.setBinning(50, xmin=0, xmax=1)
h.setTimeInterval(tint=3600 * 24)
h.setFittingParams(model="dgaussian", binned=True)
h.setEnergybinning(
    np.geomspace(0.1 / 1e3, 1 / 1e3, 3), do_diff=True, do_integral=False
)  # in TeV


# %%
# Extracting phases, times and energies from file and give them to the object
# ---------------------------------------------------------------------------
#
# For Fermi data there is a class to read these lists and use them in the main object.

h.setFermiInputFile("./example_data/merged2_pulsar.fits")


# %%
# But in general we can read our file (FITS, DL2, DL3…) and extract
# phases, times and energies as lists and read them as follows:

with fits.open("./example_data/merged2_pulsar.fits") as f:
    fits_table = f[1].data
    times = np.array(fits_table["BARYCENTRIC_TIME"], dtype=float)
    phases = np.array(fits_table["PULSE_PHASE"], dtype=float)
    energies = np.array(fits_table["ENERGY"], dtype=float)  # in MeV

h.setListsInput(phases, times, energies / 1e6, tel="fermi", energy_units="TeV")


# %%
# Run the code
# ------------

h.run()


# %%
# Show the results
# ----------------
#
# Overall results
# ^^^^^^^^^^^^^^^

phaseogram = h.draw_phaseogram(phase_limits=[0, 2], colorhist="xkcd:baby blue")
plt.tight_layout()

# %%
# Statistics of the peaks (Li&Ma significance, number of excess events...) and
# results of the periodicity tests (:math:`\chi^2`, :math:`Z^2_{10}` and H tests):

peak_statistics, periodicity_statistics = h.show_Presults()

# %%
peak_statistics

# %%
periodicity_statistics


# %%
# Result of the fitting
# ^^^^^^^^^^^^^^^^^^^^^

# %%
print(h.fit_model)

# %%
fit_result = h.show_fit_results()
fit_result

# %%
phaseogram = h.draw_phaseogram(
    phase_limits=[0, 2], colorhist="xkcd:baby blue", fit=True
)
plt.tight_layout()


# %%
# Results vs Time
# ^^^^^^^^^^^^^^^

TimeEv = h.show_timeEvolution()


# %%
# Results vs Energy
# ^^^^^^^^^^^^^^^^^

energy_lightcurves = h.show_lcVsEnergy()

# %%
energy_lc = h.show_all_lc(ylimits=None)

# %%
energy_results = h.show_EnergyPresults()

# %%
energy_plots = h.show_EnergyAna()


# %%
# Fit vs Energy
# ^^^^^^^^^^^^^

mean_energy_plot = h.show_meanVsEnergy()

# %%
energy_fit_results = h.show_EnergyFitresults()
