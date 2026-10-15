.. _userguide:

**********
User guide
**********

.. currentmodule:: ptiming_ana

A pulsar analysis with ``ptiming_ana`` has three steps: the computation of the
pulsar phases, the analysis of the phaseogram and, optionally, the spectral
analysis of the peaks. The second and third steps are configured with a
`configuration file <https://github.com/cta-observatory/PulsarTimingAnalysis/blob/main/config.yaml>`__
in YAML format, or directly with the methods of the analysis classes.

``ptiming_ana`` reports the progress and the results of the analysis with the
Python logging system. To see the messages in a script or notebook:

.. code-block:: python

   import ptiming_ana

   ptiming_ana.setup_logging()


Computing the pulsar phases
===========================

The rotational phase of each event is computed with PINT from the arrival time
of the event and the ephemeris of the pulsar. The ephemeris can be given as:

* a timing model in ``.par`` format, or
* a file with the monthly ephemerides of the Crab pulsar from the Jodrell Bank
  Observatory in ``.gro`` format, from which the model valid at the time of
  the observation is built.

The command-line tools add the phases and the barycentric times to the data
files:

.. code-block:: console

   $ add_DL3_phase_table --in-file dl3_LST-1.Run07253.fits --ephem model.par --output DL3_pulsar/
   $ add_DL3_phase_table --dir DL3/ --ephem model.par --output DL3_pulsar/
   $ add_DL2_phase_table --in-file dl2_LST-1.Run07253.h5 --ephem model.par
   $ add_fermi_phase_table --in-file ft1.fits --ft2 ft2.fits --ephem model.par --output fermi_pulsar/

Use ``--help`` to see all the options of each tool.

* DL3 files: a new file ``{name}_pulsar.fits`` is written, with two new columns
  in the events table, ``PHASE`` and ``BARYCENT_TIME`` (MJD). To read the new
  files with Gammapy, the index files have to be created
  (``lstchain_create_dl3_index_files``).
* DL2 files: a new table ``phase_info`` is added to the file, with the columns
  ``pulsar_phase`` and ``mjd_barycenter_time``.
* *Fermi*-LAT files: a new file ``{name}_pulsar.fits`` is written with the
  columns ``PULSE_PHASE`` and ``BARYCENTRIC_TIME``.

The exact calculation is slow for large files. With the ``--interpolation``
option, the phases are only computed for one of every ``--number-interpolation``
events (1000 by default) and interpolated for the rest, which is much faster
at the cost of a small loss of precision.

The same functionality is available from Python, see `ptiming_ana.cphase`.


Phaseogram analysis
===================

The main class is `ptiming_ana.phaseogram.PulsarAnalysis`:

.. code-block:: python

   from ptiming_ana.phaseogram import PulsarAnalysis

   analysis = PulsarAnalysis()
   analysis.set_config("config.yaml")
   analysis.run()

   peak_statistics, periodicity_tests = analysis.show_Presults()
   analysis.draw_phaseogram(phase_limits=[0, 2])

Input data
----------

==================================  ==========================================
Input                               Configuration / method
==================================  ==========================================
LST-1 DL3 files (with phases)       ``flags: DL2_format: False, fits_format: False`` / ``setDL3InputFile``
LST-1 DL2 files (with phases)       ``flags: DL2_format: True`` / ``setLSTInputFile``
*Fermi*-LAT file (with phases)      ``flags: fits_format: True`` / ``setFermiInputFile``
Lists of phases, times, energies    ``setListsInput``
==================================  ==========================================

The energies are expected in TeV. For lists of events selected with cuts, the
observation time (in hours) should be given with the ``tobs`` argument of
``setListsInput``.

Phase regions
-------------

The signal regions (``P1``, ``P2`` and ``P3``) and the background region
(``Bkg``) are defined by their phase edges. A region crossing phase 0 is
defined with two intervals, for instance ``P1: [0, 0.026, 0.983, 1]``. The
number of excess events and the significance of each peak
(`Li & Ma 1983 <https://doi.org/10.1086/161295>`__, Eq. 17) are computed using
the background region as the OFF region.

Periodicity tests
-----------------

Three tests of the uniformity of the phases are applied:

* :math:`\chi^2` test of the phaseogram.
* :math:`Z^2_{n}` test (`Buccheri et al. 1983
  <https://ui.adsabs.harvard.edu/abs/1983A&A...128..245B>`__) with
  :math:`n=10` harmonics.
* H test (`de Jager et al. 1989
  <https://ui.adsabs.harvard.edu/abs/1989A&A...221..180D>`__), with the
  p-value from `de Jager & Büsching 2010
  <https://doi.org/10.1051/0004-6361/201014362>`__.

For each test the statistic, the p-value and the equivalent one-sided Gaussian
significance are reported.

Fit of the pulse profile
------------------------

The peaks can be fitted with the models listed by
`ptiming_ana.phaseogram.get_model_list`: one (``gaussian``), two
(``dgaussian``) or three (``tgaussian``) Gaussians, two asymmetric Gaussians
(``asym_dgaussian``) and one (``lorentzian``) or two (``double_lorentz``)
Lorentzians. The fit is either a least-squares fit to the phaseogram
(``binned: True``) or an unbinned extended maximum likelihood fit to the phases.

Evolution with time and energy
------------------------------

The statistics are computed in cumulative time intervals (``time_binning``) to
follow the growth of the signal, and in bins of energy or above energy
thresholds (``energy_binning``). The results and plots are obtained with the
``show_*`` methods of the analysis object, and saved in a PDF file with
``results: save_results: True``.


Spectral analysis
=================

The spectral energy distribution of a peak is obtained from the DL3 files with
`ptiming_ana.spectral.SpectralPulsarAnalysis`, using the events in the
background phase region to estimate the background:

.. code-block:: python

   from ptiming_ana.spectral import SpectralPulsarAnalysis

   spectrum = SpectralPulsarAnalysis("config.yaml")
   spectrum.run(peak="P1")

   flux_points = spectrum.get_flux_points()

The energy binning, the energy range of the fit and of the spectral points,
and the spectral model are set in the configuration file. A Gammapy spectral
model can be given with the ``ref_model`` argument to be used as the model to
fit. The plots are saved in the ``results_output_directory`` and, optionally,
the reduced datasets (DL4) in the ``DL4_directory``.
