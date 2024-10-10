import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy import units as u
from dataclasses import dataclass
from .ptime_analysis import PulsarTimeAnalysis
from .phase_regions import PhaseRegions, PulsarPeak
from .lightcurve import Lightcurve
from .periodicity_test import PeriodicityTest
from .pfitting import PeakFitting
from .models import get_model_list
from .phasebinning import PhaseBinning
from .penergy_analysis import PEnergyAnalysis
from .filter_object import FilterPulsarAna
from .read_events import ReadDL3File, ReadFermiFile, ReadLSTFile, ReadList
import pickle
import yaml
import logging
import os

pd.options.mode.chained_assignment = None

LOG_FORMAT = "%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("gammapy").disabled = True


@dataclass
class PulsarAnalysis:
    """
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.

    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set daraframe to False if it is not available and want to set the attributes manually.
    pdata : List of float
        List of phases (in case no dataframe is available).
    ptimes: List of float
        List of mjd times (in case no dataframe is available).
    tobservation: float
        Total effective time of observation
    peak_limits_1: tuple
        Edges of the first peak. Set to None if no P1 is present.
    peak_limits_2: tuple
        Edges of the second peak. Set to None if no P2 is present.
    off_limits: tuple
        Edges of the OFF region
    binned: boolean
        True for a binned fitting, False for an unbinned fitting.


    Attributes
    ----------
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    tobs : float
        Effective time of observation in hours
    regions: PhaseRegions object
        Information of the OFF/signal regions
    histogram: Lightcurve object
        Information of the phaseogram
    stats: PeriodicityTest object
        Information of Statistical Tests for searching Periodicity
    fitting: PeakFitting object
        Information abot the fitting used for the peaks
    """

    def __init__(
        self, filename=None, nbins=50, tint=3600, binned=True, model="dgaussian"
    ):
        if filename is not None:
            if "fits" in filename:
                logger.info("Assuming Fermi-LAT data as an input")
                self.setFermiInputFile(filename)
                self.setTimeInterval(tint=3600 * 24 * 10)  # Every 10 days

            elif "h5" in filename:
                logger.info("Assuming LST1 data as an input")
                self.setLSTInputFile(filename)
                self.setTimeInterval(tint=3600)  # Every hour
            else:
                ValueError("FIle has no valid format")
        else:
            self.setTimeInterval(tint=tint)

        # Define default parameters for the binning
        self.setBinning(nbins=nbins)

        # Define default parameters for the fitting
        self.setFittingParams(model, binned, do_fit=False)

        # Define default parameters for the cuts
        self.setParamCuts()

    ##############################################
    # SETTINGS
    #############################################

    def setFermiInputFile(self, filename):
        if "fits" in filename:
            self.r = ReadFermiFile(filename)
            self.telescope = "fermi"
            self.energy_units = "GeV"

        else:
            raise ValueError("No FITS file given for Fermi-LAT data")

    def setListsInput(
        self, plist, tlist=None, elist=None, tel="MAGIC", energy_units="GeV"
    ):
        self.r = ReadList(plist, tlist, elist, tel)
        self.telescope = tel
        self.energy_units = energy_units

    def setDL3InputFile(
        self,
        dirname=None,
        target_radec=None,
        max_rad=0.2,
        zd_cuts=[0, 60],
        energy_dependent_theta=True,
    ):
        self.r = ReadDL3File(
            directory=dirname,
            target_radec=target_radec,
            max_rad=max_rad,
            zd_cuts=zd_cuts,
            energy_dependent_theta=energy_dependent_theta,
        )
        self.telescope = "lst"
        self.energy_units = "TeV"

    def setLSTInputFile(self, filename=None, dirname=None, src_dep=False):
        self.r = ReadLSTFile(file=filename, directory=dirname, src_dependent=src_dep)
        self.telescope = "lst"
        self.energy_units = "TeV"

    def setBinning(self, nbins, xmin=None, xmax=None):
        self.nbins = nbins
        self.binning = PhaseBinning(nbins, xmin, xmax)

    def setParamCuts(
        self,
        gammaness_cut=None,
        alpha_cut=None,
        theta2_cut=None,
        zd_cut=None,
        int_cut=None,
        energy_cut=None,
        energy_binning_cut=None,
    ):
        self.cuts = FilterPulsarAna(
            gammaness_cut,
            alpha_cut,
            theta2_cut,
            zd_cut,
            int_cut,
            energy_cut,
            energy_binning_cut,
        )

    def setEnergybinning(self, energy_edges, do_diff, do_integral):
        self.energy_edges = energy_edges
        self.EnergyAna = PEnergyAnalysis(self.energy_edges, do_diff, do_integral)

    def setFittingParams(self, model, binned=False, peak="both", do_fit=True):
        model_list = get_model_list()

        if model in model_list:
            self.fit_model = model
            self.peak = peak
            self.binned = binned
            self.do_fit = do_fit
        else:
            raise ValueError("Model given is not defined")

        self.fitting = PeakFitting(self.binned, self.fit_model, peak=self.peak)
        self.fitting.check_model()

    def setTimeInterval(self, tint):
        self.tint = tint
        self.TimeEv = PulsarTimeAnalysis(tint=self.tint)

    def setBackgroundLimits(self, OFF_limits):
        self.OFF_limits = OFF_limits

    def setPeaklimits(self, P1_limits=None, P2_limits=None, P3_limits=None):
        if P1_limits == "None":
            P1_limits = None
        if P2_limits == "None":
            P2_limits = None
        if P3_limits == "None":
            P3_limits = None

        P1P2_limits = []

        if P1_limits is not None:
            P1P2_limits += P1_limits

        if P2_limits is not None:
            P1P2_limits += P2_limits

        self.P1_limits = P1_limits
        self.P2_limits = P2_limits
        self.P1P2_limits = P1P2_limits
        self.P3_limits = P3_limits

    def set_config(self, configuration_file):
        # Read the configuration file
        with open(configuration_file, "rb") as cfile:
            conf = yaml.safe_load(cfile)

        # Read files
        self.filter_data = conf["cuts"]["filter_data"]
        if conf["flags"]["DL2_format"]:
            if os.path.isdir(conf["pulsar_file_dir"]):
                self.setLSTInputFile(
                    dirname=conf["pulsar_file_dir"],
                    src_dep=conf["flags"]["src_dependent"],
                )
            else:
                self.setLSTInputFile(
                    filename=conf["pulsar_file_dir"],
                    src_dep=conf["flags"]["src_dependent"],
                )

            if conf["cuts"]["include_DL2_extra_cuts"]:
                if conf["cuts"]["extra_cuts"]["energy_dependent"]:
                    if conf["flags"]["src_dependent"]:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            alpha_cut=conf["cuts"]["extra_cuts"]["alpha"],
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_binning_cut=conf["cuts"]["extra_cuts"][
                                "energy_binning"
                            ],
                        )
                    else:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            theta_cut=np.power(conf["cuts"]["extra_cuts"]["theta"], 2),
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_binning_cut=conf["cuts"]["extra_cuts"][
                                "energy_binning"
                            ],
                        )
                else:
                    if conf["flags"]["src_dependent"]:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            alpha_cut=conf["cuts"]["extra_cuts"]["alpha"],
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_cut=conf["cuts"]["extra_cuts"]["energy"],
                        )
                    else:
                        self.setParamCuts(
                            gammaness_cut=conf["cuts"]["extra_cuts"]["gammaness"],
                            theta_cut=np.power(conf["cuts"]["extra_cuts"]["theta"], 2),
                            zd_cut=conf["cuts"]["zd_range"],
                            int_cut=conf["cuts"]["extra_cuts"]["intensity"],
                            energy_cut=conf["cuts"]["extra_cuts"]["energy"],
                        )

            else:
                self.setParamCuts(zd_cut=conf["cuts"]["zd_range"])

        elif conf["flags"]["fits_format"]:
            self.setFermiInputFile(filename=conf["pulsar_file_dir"])

        else:
            self.is_DL3_input = True
            self.setDL3InputFile(
                dirname=conf["pulsar_file_dir"],
                target_radec=[conf["target"]["ra"], conf["target"]["dec"]],
                max_rad=conf["cuts"]["max_rad"],
                zd_cuts=conf["cuts"]["zd_range"],
                energy_dependent_theta=conf["cuts"]["energy_dependent_theta"],
            )
            self.setParamCuts(energy_cut=conf["cuts"]["extra_cuts"]["energy"])

        # Set regions
        self.setBackgroundLimits(conf["phase_regions"]["Bkg"])
        self.setPeaklimits(
            P1_limits=conf["phase_regions"]["P1"],
            P2_limits=conf["phase_regions"]["P2"],
            P3_limits=conf["phase_regions"]["P3"],
        )

        if conf["phase_binning"]["custom_binning"]:
            self.setBinning(
                conf["phase_binning"]["nbins"],
                xmin=conf["phase_binning"]["xmin"],
                xmax=conf["phase_binning"]["xmax"],
            )
        else:
            self.setBinning(conf["phase_binning"]["binning"])

        if conf["time_binning"]["run_time_analysis"]:
            units_time = conf["time_binning"]["units"]
            tint = conf["time_binning"]["tint"] * u.Unit(units_time)

            self.setTimeInterval(tint.to(u.s).value)
        else:
            self.setTimeInterval(3600 * 300)

        if conf["energy_binning"]["run_energy_analysis"]:
            units_energy = conf["energy_binning"]["units"]
            nbins_energy = conf["energy_binning"]["nbins"]
            emin = conf["energy_binning"]["emin"] * u.Unit(units_energy)
            emax = conf["energy_binning"]["emax"] * u.Unit(units_energy)
            do_integral = conf["energy_binning"]["do_integral"]
            do_diff = conf["energy_binning"]["do_diff"]

            self.setEnergybinning(
                np.geomspace(emin.to(u.TeV).value, emax.to(u.TeV).value, nbins_energy),
                do_diff,
                do_integral,
            )

        if conf["fitting"]["run_fitting"]:
            if self.P1_limits is None:
                self.setFittingParams(
                    model=conf["fitting"]["model"],
                    binned=conf["fitting"]["binned"],
                    peak="P2",
                )
            elif self.P2_limits is None:
                self.setFittingParams(
                    model=conf["fitting"]["model"],
                    binned=conf["fitting"]["binned"],
                    peak="P1",
                )
            else:
                self.setFittingParams(
                    model=conf["fitting"]["model"], binned=conf["fitting"]["binned"]
                )

        # Set output file for results
        self.get_results = conf["results"]["save_results"]
        if self.get_results:
            try:
                self.output_file = conf["results"]["output_directory"]
            except AttributeError:
                self.output_file = conf["results"]["output_file"]

            self.output_dir = os.path.dirname(self.output_file)
            if not os.path.exists(self.output_dir):
                logger.info("Creating directory: " + self.output_dir)
                os.makedirs(self.output_dir)
        else:
            self.output_file = None

    ##############################################
    # EXECUTION
    #############################################

    def shift_phases(self, xmin):
        for i in range(0, len(self.phases)):
            if self.phases[i] < xmin:
                self.phases[i] += 1

        self.info["pulsar_phase"] = self.phases

    def init_regions(self):
        OFFob = PulsarPeak(peak_limits=self.OFF_limits, peaktype="background")

        if self.P1_limits is not None:
            p1ob = PulsarPeak(peak_limits=self.P1_limits, peaktype="signal")
        else:
            p1ob = None

        if self.P2_limits is not None:
            p2ob = PulsarPeak(peak_limits=self.P2_limits, peaktype="signal")
        else:
            p2ob = None

        if self.P3_limits is not None:
            p3ob = PulsarPeak(peak_limits=self.P3_limits, peaktype="signal")
        else:
            p3ob = None

        if len(self.P1P2_limits) > 0:
            p1p2ob = PulsarPeak(peak_limits=self.P1P2_limits, peaktype="signal")

        self.regions = PhaseRegions(
            OFF_object=OFFob,
            P1_object=p1ob,
            P2_object=p2ob,
            P1P2_object=p1p2ob,
            P3_object=p3ob,
        )

    def update_info(self):
        # Fill the background regions
        self.regions.OFF.fillPeak(self.phases)

        # Fill and calculate statistics of Peaks
        for i in self.regions.dic:
            if self.regions.dic[i] is not None:
                self.regions.dic[i].fillPeak(self.phases)
                self.regions.dic[i].make_stats(self.regions, self.tobs)

        # Create the phaseogram using the Lightcurve class
        self.histogram = Lightcurve(self, self.binning)

        # Apply Periodicity stats and store them using the PeriodicityTest Class
        self.stats = PeriodicityTest(self)

    def initialize(self):
        # Read the data and filter
        try:
            self.r.run(self)
        except TypeError:
            self.r.run()

        # Extract each attribute
        self.phases = np.array(self.r.info["pulsar_phase"].to_list())
        self.info = self.r.info

        # Shift phases if necessary
        self.shift_phases(xmin=self.binning.xmin)

        # Initialize the regions object
        self.init_regions()

    def execute_stats(self, tobs):
        # Update the information at a certain interval of time and store final values
        self.TimeEv.run(self)

        # COmpute P1/P2 ratio
        self.regions.calculate_P1P2()

        # Set the final effective time of observation
        self.tobs = tobs

        # Fit the histogram using PeakFitting class. If binned is False, an Unbinned Likelihood method is used for the fitting
        if self.do_fit:
            logger.info("Fitting the data to the given model...")
            logger.info("Fit model: " + self.fit_model)
            logger.info("Binned fitting: " + str(self.binned))
            self.fitting.run(self)

        else:
            logger.info("No fit has been done since no fit parameters has been set")

    def run(self):
        # Initializa
        logger.info("Initializing...")
        self.initialize()

        # Excute stats
        logger.info(
            "Calculating statistics every " + str(self.tint / 60) + " minutes..."
        )
        self.execute_stats(self.r.tobs)

        # Execute stats in energy bins
        try:
            logger.info("Performing energy-dependent analysis...")
            self.EnergyAna.run(self)
        except AttributeError:
            logger.warning(
                "No Energy Analysis was performed. Check that you set the right energy params"
            )

        logger.info("FINISHED. Producing general results...")

        if self.get_results:
            self.save_results()

    ##############################################
    # RESULTS
    #############################################

    def check_energyana(self):
        try:
            self.EnergyAna
        except AttributeError:
            logger.info("No energy-dependent analysis has been done")
            return False
        return True

    def draw_phaseogram(
        self,
        phase_limits=[0, 2],
        stats="short",
        background=True,
        signal=["P1", "P2", "P3"],
        colorhist="xkcd:sky blue",
        colorb="black",
        colorP=["orange", "green", "purple"],
        colorfit="red",
        fit=False,
        hline=True,
    ):
        # Plot histogram from 0 to 1 and from 1 to 2 (2 periods)
        fig = plt.figure(figsize=(15, 5))
        self.histogram.show_phaseogram(
            self,
            phase_limits,
            stats,
            background,
            signal,
            colorhist,
            colorb,
            colorP,
            colorfit,
            fit,
            hline,
        )
        return fig

    def show_Presults(self):
        rpeaks = self.regions.show_peak_results()
        rstats = self.stats.show_Pstats()

        print("RESULTS FOR THE PEAK STATISTICS:" + "\n")
        print(rpeaks)
        if self.regions.P1P2_ratio is not None:
            print(
                "\n"
                + f"P1/P2 ratio={self.regions.P1P2_ratio:.2f}"
                + f"+/-{self.regions.P1P2_ratio_error:.2f}"
                + "\n"
            )
        print("\n \n" + "RESULTS FOR THE PERIODICITY SEARCH:" + "\n")
        print(rstats)

        return rpeaks, rstats

    def show_fit_results(self):
        fresult = self.fitting.show_result()
        print(fresult)
        return fresult

    def show_timeEvolution(self):
        fig1 = self.TimeEv.show_results()
        fig2 = self.TimeEv.compare_Peaksig()
        return fig1, fig2

    def show_EnergyAna(self, integral=None):
        if self.check_energyana():
            fig = plt.figure(figsize=(20, 4))
            if self.regions.P1 is None or self.regions.P2 is None:
                nplots = 2
            else:
                nplots = 3
                plt.subplot(1, nplots, 3)
                self.EnergyAna.P1P2VsEnergy(integral)

            plt.subplot(1, nplots, 1)
            self.EnergyAna.PSigVsEnergy(integral)

            plt.subplot(1, nplots, 2)
            self.EnergyAna.FWHMVsEnergy(integral)

            plt.tight_layout()
            return fig

    def show_EnergyPresults(self, integral=None):
        if self.check_energyana():
            peak_stat, p_stat = self.EnergyAna.show_EnergyPresults(integral)
            return peak_stat, p_stat

    def show_EnergyFitresults(self, integral=None):
        if self.check_energyana():
            fit_results = self.EnergyAna.show_Energy_fitresults(integral)
            return fit_results

    def show_meanVsEnergy(self, integral=None):
        if self.check_energyana():
            fig = self.EnergyAna.MeanVsEnergy(integral)
            return fig

    def show_FWHMVsEnergy(self, integral=None):
        if self.check_energyana():
            fig = plt.figure()
            self.EnergyAna.FWHMVsEnergy()
            return fig

    def show_SigVsEnergy(self, integral=None):
        if self.check_energyana():
            fig = plt.figure()
            self.EnergyAna.PSigVsEnergy(integral)
            return fig

    def show_P1P2VsEnergy(self, integral=None):
        if self.check_energyana():
            fig = plt.figure()
            self.EnergyAna.P1P2VsEnergy(integral)
            return fig

    def show_lcVsEnergy(self, integral=None):
        if self.check_energyana():
            fig_array = self.EnergyAna.show_Energy_lightcurve(integral)
            return fig_array

    def show_all_lc(self, ylimits=None):
        if self.check_energyana():
            fig = self.EnergyAna.show_joined_Energy_lightcurve(ylimits=ylimits)
            return fig

    def show_all_fits(self, integral=None):
        if self.check_energyana():
            fig = self.EnergyAna.show_joined_Energy_fits(integral)
            return fig

    def save_df(self, output_file, file_format="h5"):
        if file_format == "h5":
            self.info.to_hdf(
                output_file, key="dl2/event/telescope/parameters/LST_LSTCam"
            )

        elif file_format == "csv":
            self.info.to_csv(index=False)

    def save_results(self, output_file=None):
        if output_file is not None:
            self.output_file = output_file
            self.output_dir = os.path.dirname(self.output_file)

        with PdfPages(self.output_file) as pdf:
            pdf.savefig(
                self.draw_phaseogram(
                    phase_limits=[0, 2],
                    stats="long",
                    background=True,
                    signal=["P1", "P2", "P3"],
                    colorhist="blue",
                    colorb="black",
                    colorP=["orange", "green", "purple"],
                    colorfit="red",
                    fit=False,
                    hline=True,
                ),
                bbox_inches="tight",
                pad_inches=1,
            )
            pdf.savefig(self.TimeEv.PsigVsTime())
            pdf.savefig(self.TimeEv.PexVsTime())
            pdf.savefig(self.TimeEv.StatsVsTime())

            if self.do_fit:
                pdf.savefig(
                    self.draw_phaseogram(
                        phase_limits=[0, 2],
                        stats="long",
                        background=True,
                        signal=["P1", "P2", "P3"],
                        colorhist="blue",
                        colorb="black",
                        colorP=["orange", "green", "purple"],
                        colorfit="red",
                        fit=True,
                        hline=True,
                    ),
                    bbox_inches="tight",
                    pad_inches=1,
                )

                fitting = self.fitting.show_result()
                fitting.to_hdf(self.output_dir + "/overall_fitting.h5", key="results")

            try:
                pdf.savefig(self.show_EnergyAna(), bbox_inches="tight", pad_inches=1)
                for i in range(0, len(self.EnergyAna.show_Energy_lightcurve())):
                    pdf.savefig(
                        self.EnergyAna.show_Energy_lightcurve()[i],
                        bbox_inches="tight",
                        pad_inches=1,
                    )
                if self.do_fit:
                    for i in range(0, len(self.show_EnergyFitresults())):
                        df = self.show_EnergyFitresults()[i]
                        df.to_hdf(
                            self.output_dir + "/fitting_energy_bin" + str(i) + ".h5",
                            key="results",
                        )
            except AttributeError:
                pass

    def save_object(self, output_file):
        with open(output_file, "wb") as file:
            pickle.dump(self, file)
            file.close()
