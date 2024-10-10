import numpy as np

# from decimal import *
from pylab import sum
from scipy.stats import chi2, norm
import pandas as pd


class PeriodicityTest:
    """
    A class to apply and store the information of the periodicty tests.

    Parameters
    ----------
    pulsar_phases : PulsarPhases object

    Attributes
    ----------
    chisqr_test :
        Results of the chi square tests. Format: [Statistic, p_value, nsigmas]
    number : int
        Number of phases used in the analysis
    cos : list of float
        Cosine moments of the list of phases
    sin: list of float
        Sine moments of the list of phases
    Zntest_res:
        Results of the Zn tests. Format: [Statistic, p_value, nsigmas]. Default is n=10
    Htest_res:
        Results of the H test. Format: [Statistic, p_value, nsigmas]
    """

    def __init__(self, pulsar_phases):
        self.apply_all_tests(pulsar_phases)

    ##############################################
    # EXECUTION
    #############################################

    def apply_all_tests(self, pulsar_phases):
        # Apply chi square test using Lightcurve Class
        self.chisqr_res = pulsar_phases.histogram.chi_sqr_pulsar_test()

        # Apply unbinned statistical tests
        self.moments(pulsar_phases)
        self.apply_moment_tests()

    def apply_moment_tests(self):
        # Default n=10 for zn test
        self.zn_test(n=10)
        self.Htest_res = self.H_test()

    def moments(self, pulsar_phases, n=25):
        # Transform list to be between 0 and 2pi
        plist = (pulsar_phases.phases) * 2 * np.pi

        # Calculate moments
        k = np.arange(1, n + 1)
        cos_moment = sum(np.cos(np.outer(plist, k)), axis=0)
        sin_moment = sum(np.sin(np.outer(plist, k)), axis=0)

        # Store the information
        self.number = len(plist)
        self.cos = cos_moment
        self.sin = sin_moment

        return (self.number, self.cos, self.sin)

    def zn_test(self, n=10):
        cos_moment = self.cos[0 : n + 1]
        sin_moment = self.sin[0 : n + 1]
        self.Zn_n = n

        # Calculate statistic and pvalue
        Zn = 2 / self.number * sum(np.power(cos_moment, 2) + np.power(sin_moment, 2))
        pvalue_zn = chi2.sf(float(Zn), 2 * n)
        sigmas_zn = norm.isf(pvalue_zn, loc=0, scale=1)

        # Store information
        self.Zntest_res = Zn, pvalue_zn, sigmas_zn

        return self.Zntest_res

    def H_test(self):
        bn = 0.398
        h = []

        # Calculate statistic and pvalue
        for m in np.arange(1, len(self.cos)):
            h.append(
                2
                / self.number
                * sum(np.power(self.cos[0:m], 2) + np.power(self.sin[0:m], 2))
                - 4 * m
                + 4
            )
        H = max(h)
        pvalue_H = np.exp(-bn * H)
        sigmas_H = norm.isf(float(pvalue_H), loc=0, scale=1)

        # Store information
        self.Htest_res = H, pvalue_H, sigmas_H

        return self.Htest_res

    ##############################################
    # RESULTS
    #############################################

    def show_Pstats(self):
        return pd.DataFrame(
            data={
                "Chi_square_test": self.chisqr_res,
                "Zn_test": self.Zntest_res,
                "H_test": self.Htest_res,
            },
            index=["Statistic", "p-value", "Number of $\sigma$"],
        )
