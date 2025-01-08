"""
Basic pulsar analysis using Fermi-LAT data
==========================================

"""

from ptiming_ana.phaseogram import PulsarAnalysis
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_multi_image = "single"

# %%
# Create the PulsarAnalysis object and settings
# ---------------------------------------------

h = PulsarAnalysis()
h.set_config('./example_data/config_tutorial.yaml')

# %%
# Alternatively we can set the parameters directly
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

h = PulsarAnalysis()
h.setBackgroundLimits([0.52,0.87])
h.setPeaklimits(
    P1_limits=[0,0.026, 0.983, 1],
    P2_limits=[0.377, 0.422],
    P3_limits=None
)
h.setBinning(50, xmin=0, xmax=1)
h.setTimeInterval(tint=3600*24)
h.setFittingParams(model='dgaussian', binned=True)
h.setEnergybinning(
    np.geomspace(0.1/1e3, 1/1e3, 3),
    do_diff=True,
    do_integral=False
)  # in TeV


# %%
# Extracting phases, times and energies from file and give them to the object
# ---------------------------------------------------------------------------
# 
# For Fermi data there is a class to read these lists and use them in the main object.

h.setFermiInputFile('./example_data/merged2_pulsar.fits')


# %%
# But in general we can read our file (FITS, DL2, DL3…) and extract
# phases, times and energies as lists and read them as follows:

f=fits.open('./example_data/merged2_pulsar.fits')
fits_table=f[1].data

times=np.sort(fits_table['BARYCENTRIC_TIME'].byteswap().newbyteorder())
phases=fits_table['PULSE_PHASE'].byteswap().newbyteorder()
energies=fits_table['ENERGY'].byteswap().newbyteorder()

h.setListsInput(phases, times, energies/1e6, tel='fermi', energy_units='TeV')

h.get_results = False


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

phaseogram=h.draw_phaseogram(
    phase_limits=[0, 2],
    colorhist='xkcd:baby blue'
)
plt.tight_layout()

# %%

results=h.show_Presults()


# %%
# Result of the fitting
# ^^^^^^^^^^^^^^^^^^^^^

# %%
h.fit_model

# %%
fit_result = h.show_fit_results()

# %%
phaseogram = h.draw_phaseogram(
    phase_limits=[0, 2],
    colorhist='xkcd:baby blue',
    fit=True
)
plt.tight_layout()


# %%
# Results vs Time
# ^^^^^^^^^^^^^^^

TimeEv = h.show_timeEvolution()


# %%
# The periodicity tests are not available since the signal is too strong
# (p_value too low to extrapolate a significance).


# %%
# Results vs Energy
# ^^^^^^^^^^^^^^^^^

h.show_lcVsEnergy()

# %%
energy_lc=h.show_all_lc(ylimits=None)

# %%
energy_results=h.show_EnergyPresults()

# %%
energy_plots=h.show_EnergyAna()


# %%
# Fit vs Energy
# ^^^^^^^^^^^^^

mean_energy_plot=h.show_meanVsEnergy()

h.show_EnergyFitresults()
