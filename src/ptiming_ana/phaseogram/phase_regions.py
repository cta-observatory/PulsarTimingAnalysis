import logging

import numpy as np
import pandas as pd
from gammapy.stats import WStatCountsStatistic

logger = logging.getLogger(__name__)

__all__ = ["PhaseRegions", "PulsarPeak", "calculate_CountStats"]

_PEAK_ATTRIBUTES = {"P1": "P1", "P2": "P2", "P1+P2": "P1P2", "P3": "P3"}


def calculate_CountStats(on_file, off_file=None, factor=None):
    """
    Li&Ma statistics of a signal region given the background (OFF) region.

    Parameters
    ----------
    on_file : array-like
        Phases of the events in the signal region.
    off_file : array-like
        Phases of the events in the background region.
    factor : float
        Ratio between the phase extension of the signal and background regions.

    Returns
    -------
    stat : `~gammapy.stats.WStatCountsStatistic`
    yerr : float
        Uncertainty in the number of excess events.
    noff : float
        Number of background events expected in the signal region.
    """
    if off_file is None:
        raise ValueError("No off data given for the pulsar analysis")

    Non = len(on_file)
    Noff = len(off_file)
    stat = WStatCountsStatistic(n_on=Non, n_off=Noff, alpha=factor)
    yerr = np.sqrt(Non + ((factor**2) * Noff))

    return (stat, yerr, Noff * factor)


class PhaseRegions:
    """
    Container of the background (OFF) and signal (P1, P2, P1+P2, P3) regions.
    """

    def __init__(
        self,
        OFF_object=None,
        P1_object=None,
        P2_object=None,
        P1P2_object=None,
        P3_object=None,
    ):
        # Define background and null regions for both peaks and
        self.OFF = OFF_object
        if OFF_object is None:
            logger.debug("No peak statistics available since no background is provided")

        self.P1P2_ratio = None
        self.P1P2_ratio_error = None

        # Create the peaks in the previosly defined null regions
        self.create_dic(P1_object, P2_object, P1P2_object, P3_object)

    ##############################################
    # EXECUTION
    #############################################

    def create_dic(
        self, P1_object=None, P2_object=None, P1P2_object=None, P3_object=None
    ):
        self.P1 = P1_object
        self.P2 = P2_object
        self.P1P2 = P1P2_object
        self.P3 = P3_object

        if P1_object is None:
            logger.debug("No P1 limits. Cant create P1 object")
        if P2_object is None:
            logger.debug("No P2 limits. Cant create P2 object")

        self.npeaks = sum(peak is not None for peak in (self.P1, self.P2, self.P3))

        self.dic = {
            "P1": P1_object,
            "P2": P2_object,
            "P1+P2": P1P2_object,
            "P3": P3_object,
        }

    def _check_name(self, name):
        if name not in _PEAK_ATTRIBUTES:
            raise ValueError(
                f"Unknown region '{name}'. Valid regions: {list(_PEAK_ATTRIBUTES)}"
            )

    def remove_peak(self, name):
        self._check_name(name)
        if self.dic[name] is not None and name != "P1+P2":
            self.npeaks = self.npeaks - 1

        self.dic[name] = None
        setattr(self, _PEAK_ATTRIBUTES[name], None)

    def create_peak(self, peak_object, name):
        self._check_name(name)
        if self.dic[name] is None and name != "P1+P2":
            self.npeaks = self.npeaks + 1

        self.dic[name] = peak_object
        setattr(self, _PEAK_ATTRIBUTES[name], peak_object)

    def calculate_P1P2(self):
        if self.dic["P1"] is not None and self.dic["P2"] is not None:
            self.P1P2_ratio = self.P1.Nex / self.P2.Nex
            self.P1P2_ratio_error = self.P1P2_ratio * np.sqrt(
                (self.P1.yerr / self.P1.Nex) ** 2 + (self.P2.yerr / self.P2.Nex) ** 2
            )
        else:
            self.P1P2_ratio = None
            self.P1P2_ratio_error = None

    ##############################################
    # RESULTS
    #############################################

    def show_peak_results(self):
        peak_results = pd.DataFrame(
            data={"P1": [0, 0, 0, 0, 0, 0, 0]},
            index=[
                "Significance",
                "Nex",
                "Nex_error",
                "Number",
                "noff",
                "sign_t_ratio",
                "s/n ratio",
            ],
        )
        for key, value in self.dic.items():
            if value is not None:
                peak_results[key] = [
                    value.sign,
                    value.Nex,
                    value.yerr,
                    value.number,
                    value.noff,
                    value.sign_ratio,
                    value.s_n_ratio,
                ]

        return peak_results


class PulsarPeak:
    """
    A class to manipulate and store information about the statistics of certain regions in a phase list object. 2 types of regions:
        1. Signal region: peaks present in the lightcurve
        2. Background region: background statistics


    Parameters
    ----------
    peak_limits : list
        list of phase edges used to define the region.
        To define a non continuos region (for instance, the sum of two independent peaks), a list of four edges can be provided.
    peaktype : str, optional
        'background' or 'signal' type to calculate especific statistics.

    Note: To obtain peak statistics an OFF region must be defined in the phase list object first.


    Attributes
    ----------
    limits: str
        region phase edges
    type: str
        the type of region defined (signal or background)
    deltaP: float
        the total phase range of the region
    phases: numpy array
        list of phases that fall into the region
    number: int
        number of events in the region
    nregions: int
        number of phase intervals that define the region


    For 'signal' type only:

    sign: float
        Li&Ma Significance of the excess of events from the peak with respect to the background
    Nex: int
        Number of excess events in the peaks
    yerr: float
        Estimated error of the number of excess events
    sign_ratio: float
        Ratio of the significance and the square root of the time of observation.
    s_n_ratio: float
        Signal to noise ratio
    """

    def __init__(self, peak_limits, peaktype="signal"):
        if peak_limits is None or len(peak_limits) == 0 or len(peak_limits) % 2 != 0:
            raise ValueError(
                "Wrong defined regions: an even number of phase edges is needed, "
                f"got {peak_limits}"
            )
        if peaktype not in ("signal", "background"):
            raise ValueError("The type of region must be 'signal' or 'background'")

        # Define the features of the region
        self.limits = peak_limits
        self.type = peaktype
        self.nregions = len(self.limits) // 2
        self.deltaP = 0
        for i in range(1, len(self.limits), 2):
            self.deltaP = self.deltaP + (self.limits[i] - self.limits[i - 1])

        if self.deltaP <= 0:
            raise ValueError(
                f"Wrong defined regions: the phase range of {peak_limits} is not positive"
            )

    ##############################################
    # EXECUTION
    #############################################

    def fillPeak(self, phases):
        phases = np.asarray(phases)

        # Keep the events ordered by interval, as they are defined in limits
        self.phases = np.concatenate(
            [
                phases[(phases > self.limits[i - 1]) & (phases < self.limits[i])]
                for i in range(1, len(self.limits), 2)
            ]
        ).astype(float)
        self.number = len(self.phases)

    # Make statistics if the region is signal type only
    def make_stats(self, regions, tobs):
        if self.type == "signal":
            stats, yerror, noff = calculate_CountStats(
                self.phases,
                off_file=regions.OFF.phases,
                factor=(self.deltaP) / regions.OFF.deltaP,
            )
            self.sign = stats.sqrt_ts.item()
            self.Nex = stats.n_sig
            self.yerr = yerror
            self.sign_ratio = self.sign / np.sqrt(tobs)
            self.s_n_ratio = self.Nex / np.sqrt(noff)
            self.noff = noff

        else:
            logger.warning("Cannot calculate statistics for a background region")
