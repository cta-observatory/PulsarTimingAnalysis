import numpy as np
import matplotlib.pyplot as plt
import copy
import logging

__all__ = ["PEnergyAnalysis"]

logger = logging.getLogger(__name__)


class PEnergyAnalysis:
    """
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.

    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set daraframe to False if it is not available and want to set the attributes manually.
    energy_edges: List of float
        Edges of the energy binning (in TeV)
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
    energy_edges: list of float
        List of edges for the energy binning
    energy_centres: list of float
        List of center of the energy bins
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    energies; list of float
        List of energies
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

    def __init__(self, energy_edges, do_diff=True, do_integral=False):
        self.energy_edges = np.array(energy_edges)
        self.energy_centres = (self.energy_edges[1:] + self.energy_edges[:-1]) / 2
        self.do_integral = do_integral
        self.do_diff = do_diff

        if self.do_diff:
            self.integral = False
        else:
            if self.do_integral:
                self.integral = True
            else:
                logger.info(
                    "No energy analysis will be performed. Please set do_diff or do_integral to True"
                )

    ##############################################
    # EXECUTION
    #############################################

    def run(self, pulsarana):
        self.energy_units = pulsarana.energy_units
        self.tobs = pulsarana.tobs

        if self.do_diff:
            # Create array of PulsarPhases objects binning in energy
            self.Parray = []
            for i in range(0, len(self.energy_edges) - 1):
                dataframe = pulsarana.info
                di = dataframe[
                    (dataframe["energy"] > self.energy_edges[i])
                    & (dataframe["energy"] < self.energy_edges[i + 1])
                ]

                logger.info(
                    "Creating object in "
                    + "energy range ("
                    + self.energy_units
                    + f"):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}"
                )
                self.Parray.append(copy.copy(pulsarana))
                self.Parray[i].setTimeInterval(self.Parray[i].tint)
                self.Parray[i].phases = np.array(di["pulsar_phase"].to_list())
                self.Parray[i].info = di

                self.Parray[i].init_regions()

                if self.Parray[i].do_fit:
                    self.Parray[i].setFittingParams(
                        self.Parray[i].fit_model,
                        self.Parray[i].binned,
                        peak=self.Parray[i].peak,
                    )

                # Update the information every 1 hour and store final values
                logger.info("Calculating statistics...")
                self.Parray[i].execute_stats(self.tobs)

        if self.do_integral:
            self.Parray_integral = []
            for i in range(0, len(self.energy_edges) - 1):
                dataframe = pulsarana.info
                di = dataframe[(dataframe["energy"] > self.energy_edges[i])]

                logger.info(
                    "Creating object in "
                    + "energy range ("
                    + self.energy_units
                    + f"):E > {self.energy_edges[i]:.2f}"
                )
                self.Parray_integral.append(copy.copy(pulsarana))
                self.Parray_integral[i].setTimeInterval(self.Parray_integral[i].tint)
                self.Parray_integral[i].phases = np.array(di["pulsar_phase"].to_list())
                self.Parray_integral[i].info = di

                self.Parray_integral[i].init_regions()

                if self.Parray_integral[i].do_fit:
                    self.Parray_integral[i].setFittingParams(
                        self.Parray_integral[i].fit_model,
                        self.Parray_integral[i].binned,
                        peak=self.Parray_integral[i].peak,
                    )

                # Update the information every 1 hour and store final values
                logger.info("Calculating statistics...")
                self.Parray_integral[i].execute_stats(self.tobs)

    ##############################################
    # RESULTS
    #############################################

    def show_Energy_lightcurve(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        fig_array = []
        for i in range(0, len(histogram_array)):
            # Plot histogram from 0 to 1 and from 1 to 2 (2 periods)
            fig = plt.figure(figsize=(12, 5))
            histogram_array[i].histogram.show_phaseogram(
                histogram_array[i],
                [0, 2],
                colorhist="C" + str(i),
                fit=True,
                time_label=False,
                stats_label=False,
                add_legend=False,
            )

            if integral:
                energy_label = (
                    "ENERGY RANGE (GeV):" + f" E > {self.energy_edges[i]*1000:.0f}"
                )
            else:
                energy_label = (
                    "ENERGY RANGE (GeV):"
                    + f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f}"
                )

            plt.annotate(
                energy_label,
                xy=(0.1, 0.9),
                xytext=(0.31, 0.9),
                fontsize=15,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", alpha=0.8),
                horizontalalignment="left",
                verticalalignment="top",
            )

            text_towrite = ""
            count = 0
            for key, value in histogram_array[i].regions.dic.items():
                if value is not None:
                    if count == 0:
                        text_towrite = (
                            text_towrite
                            + key
                            + f": Sig(Li&Ma):{value.sign:.2f}$\sigma$"
                        )
                        count += 1
                    else:
                        text_towrite = (
                            text_towrite
                            + "\n"
                            + key
                            + f": Sig(Li&Ma):{value.sign:.2f}$\sigma$"
                        )
            plt.annotate(
                text_towrite + "\n" + f"Entries={len(histogram_array[i].phases)}",
                xy=(1.05, 1.0),
                xytext=(1.05, 1.0),
                fontsize=15,
                xycoords="axes fraction",
                textcoords="offset points",
                color="black",
                bbox=dict(facecolor="white", edgecolor="black"),
                horizontalalignment="left",
                verticalalignment="top",
            )

            plt.legend(loc=4, bbox_to_anchor=(1.2, 0), fontsize=15)
            fig_array.append(fig)
            plt.show()

        return fig_array

    def show_joined_Energy_fits(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        for i in range(0, len(histogram_array)):
            if histogram_array[i].fitting.check_fit_result():
                fig = plt.figure(figsize=(17, 8))
                break
            elif (
                i == len(histogram_array) - 1
                and not histogram_array[i].fitting.check_fit_result()
            ):
                print("No fit available for any energy bin")
                return
        for i in range(0, len(histogram_array)):
            if histogram_array[i].fitting.check_fit_result():
                histogram_array[i].histogram.draw_fitting(
                    histogram_array[i],
                    color="C" + str(i),
                    density=True,
                    label="Energies(GeV):"
                    + f"{self.energy_edges[i]*1000:.2f}-{self.energy_edges[i+1]*1000:.2f}",
                )
        plt.xlim(
            2 * histogram_array[0].fitting.shift,
            1 + 2 * histogram_array[i].fitting.shift,
        )
        plt.legend(fontsize=20)
        return fig

    def show_joined_Energy_lightcurve(
        self,
        colorh=[
            "tab:red",
            "tab:purple",
            "tab:blue",
            "tab:brown",
            "tab:cyan",
            "tab:olive",
            "tab:pink",
        ],
        colorP=["orange", "green", "purple"],
        ylimits=None,
    ):
        fig = plt.figure(figsize=(17, 8))

        for i in range(0, len(self.Parray)):
            self.Parray[i].histogram.draw_density_hist(
                [0.7, 1.7],
                colorhist=colorh[i],
                label="Energies(GeV):"
                + f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f}",
                fill=False,
            )

        self.Parray[i].histogram.draw_background(self.Parray[i], "grey", hline=False)

        signal = ["P1", "P2", "P3"]
        for j in range(0, len(signal)):
            self.Parray[0].histogram.draw_peakregion(
                self.Parray[0], signal[j], color=colorP[j]
            )

        plt.legend(fontsize=15)

        if ylimits is not None:
            plt.ylim(ylimits)

        return fig

    def show_EnergyPresults(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        peak_stat = [0] * (len(self.energy_edges) - 1)
        p_stat = [0] * (len(self.energy_edges) - 1)

        for i in range(0, len(self.energy_edges) - 1):
            print(
                "Energies(GeV):"
                + f"{self.energy_edges[i]*1000:.0f}-{self.energy_edges[i+1]*1000:.0f}"
                + "\n"
            )
            peak_stat[i], p_stat[i] = histogram_array[i].show_Presults()
            print("\n \n")
            print("-------------------------------------------------------------------")

        return peak_stat, p_stat

    def show_Energy_fitresults(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        fit_results = [0] * (len(self.energy_edges) - 1)

        for i in range(0, len(self.energy_edges) - 1):
            print(
                "Energies(GeV):"
                + f"{self.energy_edges[i]*1000:.2f}-{self.energy_edges[i+1]*1000:.2f}"
                + "\n"
            )
            if histogram_array[i].fitting.check_fit_result():
                fit_results[i] = histogram_array[i].show_fit_results()
            else:
                print("No fit available for this energy range")
            print("\n \n")
            print("-------------------------------------------------------------------")

        return fit_results

    def PSigVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        P1_s = []
        P2_s = []
        P1P2_s = []

        for i in range(0, len(self.energy_centres)):
            if histogram_array[i].regions.dic["P1"] is not None:
                P1_s.append(histogram_array[i].regions.P1.sign)
            if histogram_array[i].regions.dic["P2"] is not None:
                P2_s.append(histogram_array[i].regions.P2.sign)
            if histogram_array[i].regions.dic["P1+P2"] is not None:
                P1P2_s.append(histogram_array[i].regions.P1P2.sign)

        if len(P1P2_s) > 0:
            plt.plot(self.energy_centres, P1P2_s, "o-", color="tab:red", label="P1+P2")

        if len(P1_s) > 0:
            plt.plot(self.energy_centres, P1_s, "o-", color="tab:orange", label="P1")

        if len(P2_s) > 0:
            plt.plot(self.energy_centres, P2_s, "o-", color="tab:green", label="P2")

        plt.ylabel("Significance($\sigma$)")
        plt.xticks(
            [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
            labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
        )
        plt.xlabel("E (GeV)")
        plt.legend()
        plt.tight_layout()
        plt.grid(which="both")
        plt.xscale("log")

    def P1P2_ratioVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        P1P2E = []
        P1P2E_error = []

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        if histogram_array[0].regions.P1P2_ratio is not None:
            for i in range(0, len(self.energy_centres)):
                P1P2E.append(histogram_array[i].regions.P1P2_ratio)
                P1P2E_error.append(histogram_array[i].regions.P1P2_ratio_error)
        else:
            print("Cannot calculate P1/P2 since one of the peaks is not defined")

        return (P1P2E, P1P2E_error)

    def P1P2VsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        # P1P2E = []
        # P1P2E_error = []

        ratio, ratio_error = self.P1P2_ratioVsEnergy(integral=integral)
        plt.fill_between(
            self.energy_centres,
            np.array(ratio) + np.array(ratio_error),
            np.array(ratio) - np.array(ratio_error),
            alpha=0.3,
        )
        plt.plot(self.energy_centres, ratio, "o-", label="LST-1")

        plt.ylabel("P1/P2")
        plt.xticks(
            [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
            labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
        )
        plt.xlabel("E (GeV)")
        plt.tight_layout()
        plt.xscale("log")
        plt.title("P1/P2 vs Energy")
        plt.grid(which="both")

    def FWHMVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        FP1 = []
        FP2 = []
        FP1_err = []
        FP2_err = []
        energies_F1 = []
        energies_F2 = []

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        if histogram_array[0].fitting.model == "asym_dgaussian":
            prefactor = 2.35482
            for i in range(0, len(self.energy_centres)):
                try:
                    FP1.append(
                        prefactor * histogram_array[i].fitting.params[1] / 2
                        + prefactor * histogram_array[i].fitting.params[2] / 2
                    )
                    energies_F1.append(self.energy_centres[i])
                    try:
                        FP1_err.append(
                            FP1
                            * np.sqrt(
                                (
                                    histogram_array[i].errors.params[1]
                                    / histogram_array[i].fitting.params[1]
                                )
                                ** 2
                                + (
                                    histogram_array[i].fitting.errors[2]
                                    / histogram_array[i].fitting.params[2]
                                )
                                ** 2
                            )
                        )
                    except AttributeError:
                        FP1_err.append(0)
                except AttributeError:
                    pass

                try:
                    FP2.append(
                        prefactor * histogram_array[i].fitting.params[4] / 2
                        + prefactor * histogram_array[i].fitting.params[5] / 2
                    )
                    energies_F2.append(self.energy_centres[i])
                    try:
                        FP2_err.append(
                            FP2
                            * np.sqrt(
                                (
                                    histogram_array[i].errors.params[4]
                                    / histogram_array[i].fitting.params[4]
                                )
                                ** 2
                                + (
                                    histogram_array[i].fitting.errors[5]
                                    / histogram_array[i].fitting.params[5]
                                )
                                ** 2
                            )
                        )
                    except AttributeError:
                        FP2_err.append(0)
                except AttributeError:
                    pass

        else:
            if histogram_array[0].fitting.model == "dgaussian":
                prefactor = 2.35482

            elif histogram_array[0].fitting.model == "lorentzian":
                prefactor = 2

            else:
                prefactor = 0

            for i in range(0, len(self.energy_centres)):
                try:
                    FP1.append(prefactor * histogram_array[i].fitting.params[1])
                    energies_F1.append(self.energy_centres[i])
                    try:
                        FP1_err.append(prefactor * histogram_array[i].fitting.errors[1])
                    except AttributeError:
                        FP1_err.append(0)
                except AttributeError:
                    pass

                try:
                    FP2.append(prefactor * histogram_array[i].fitting.params[3])
                    energies_F2.append(self.energy_centres[i])
                    try:
                        FP2_err.append(prefactor * histogram_array[i].fitting.errors[3])
                    except AttributeError:
                        FP2_err.append(0)
                except AttributeError:
                    pass

        energies_F1 = np.array(energies_F1)
        energies_F2 = np.array(energies_F2)
        FP1 = np.array(FP1)
        FP2 = np.array(FP2)
        FP1_err = np.array(FP1_err)
        FP2_err = np.array(FP2_err)

        if len(FP1) > 0:
            plt.errorbar(
                energies_F1 * 1000,
                FP1,
                yerr=FP1_err,
                fmt="o-",
                color="tab:orange",
                label="P1",
            )
        if len(FP2) > 0:
            plt.errorbar(
                energies_F2 * 1000,
                FP2,
                yerr=FP2_err,
                fmt="o-",
                color="tab:green",
                label="P2",
            )
        if len(FP1) <= 0 and len(FP2) <= 0:
            plt.annotate(
                "Plot not available",
                xy=(0.6, 0.6),
                xytext=(0.6, 0.6),
                fontsize=15,
                xycoords="axes fraction",
                textcoords="offset points",
                color="k",
                bbox=dict(facecolor="white", alpha=0.8),
                horizontalalignment="right",
                verticalalignment="top",
            )
        else:
            plt.ylabel("FWHM")
            plt.xlabel("E (GeV)")
            plt.legend()
            plt.tight_layout()
            plt.grid(which="both")
            plt.xscale("log")

    def MeanVsEnergy(self, integral=None):
        if integral is None:
            integral = self.integral

        if integral:
            if not self.do_integral:
                raise ValueError(
                    "Energy Integral results not produced. Check if do_integral parameter is set to True"
                )

            histogram_array = self.Parray_integral
        else:
            if not self.do_diff:
                raise ValueError(
                    "Energy Differential results not produced. Check if do_diff parameter is set to True"
                )

            histogram_array = self.Parray

        M1 = []
        M2 = []
        M1_err = []
        M2_err = []
        energies_M1 = []
        energies_M2 = []

        if histogram_array[0].fitting.model == "asym_dgaussian":
            for i in range(0, len(self.energy_centres)):
                try:
                    M1.append(histogram_array[i].fitting.params[0])
                    energies_M1.append(self.energy_centres[i])
                    try:
                        M1_err.append(histogram_array[i].fitting.errors[0])
                    except AttributeError:
                        M1_err.append(0)
                except AttributeError:
                    pass

                try:
                    M2.append(histogram_array[i].fitting.params[3])
                    energies_M2.append(self.energy_centres[i])
                    try:
                        M1_err.append(histogram_array[i].fitting.errors[3])
                    except AttributeError:
                        M1_err.append(0)
                except AttributeError:
                    pass

        elif (
            histogram_array[0].fitting.model == "dgaussian"
            or histogram_array[0].fitting.model == "lorentzian"
        ):
            for i in range(0, len(self.energy_centres)):
                try:
                    M1.append(histogram_array[i].fitting.params[0])
                    energies_M1.append(self.energy_centres[i])
                    try:
                        M1_err.append(histogram_array[i].fitting.errors[0])
                    except AttributeError:
                        M1_err.append(0)
                except AttributeError:
                    pass

                try:
                    M2.append(histogram_array[i].fitting.params[2])
                    energies_M2.append(self.energy_centres[i])
                    try:
                        M2_err.append(histogram_array[i].fitting.errors[2])
                    except AttributeError:
                        M2_err.append(0)
                except AttributeError:
                    pass

        if len(M1) == 0 and len(M2) == 0:
            print("No fit available for plotting")
            return
        elif len(M1) > 0 and len(M2) > 0:
            nplots = 2
        else:
            nplots = 1

        fig = plt.figure(figsize=(10, 5))
        if len(M1) > 0:
            plt.subplot(nplots, 1, 1)
            plt.errorbar(energies_M1, M1, yerr=M1_err, fmt="o-", color="tab:orange")
            plt.ylabel("Mean phase")
            plt.xticks(
                [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
                labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
            )
            plt.xlabel("E (GeV)")
            plt.title("P1 mean phase")
            plt.tight_layout()
            plt.grid(which="both")
            plt.xscale("log")

        if len(M2) > 0:
            plt.subplot(nplots, 1, nplots)
            plt.errorbar(energies_M2, M2, yerr=M2_err, fmt="o-", color="tab:green")
            plt.title("P2 mean phase")
            plt.ylabel("Mean phase")
            plt.xticks(
                [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1],
                labels=[20, 50, 80, 100, 200, 300, 400, 500, 700, 1000],
            )
            plt.xlabel("E (GeV)")
            plt.tight_layout()
            plt.grid(which="both")
            plt.xscale("log")

        return fig

    def PeaksVsEnergy(self, integral=None):
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        self.PSigVsEnergy(integral)

        plt.subplot(1, 3, 2)
        self.P1P2VsEnergy(integral)

        plt.subplot(1, 3, 3)
        self.FWHMVsEnergy(integral)

        plt.tight_layout()
        plt.show()
