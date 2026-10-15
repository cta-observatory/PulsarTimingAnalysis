import numpy as np
import pandas as pd
from scipy.special import gammaln, ndtri_exp
from scipy.stats import chi2

__all__ = ["PeriodicityTest", "chi2_log_pvalue", "sigma_from_log_pvalue"]

#: Maximum number of harmonics scanned by the H-test (de Jager et al. 1989).
H_TEST_MAX_HARMONICS = 20


def chi2_log_pvalue(statistic, dof):
    """
    Natural logarithm of the p-value of a chi-square distributed statistic.

    For very large values of the statistic, where the survival function
    underflows, the asymptotic expansion of the incomplete gamma function is
    used so the result is always finite.

    Parameters
    ----------
    statistic : float
        Value of the statistic.
    dof : float
        Number of degrees of freedom.

    Returns
    -------
    float
        Natural logarithm of the p-value.
    """
    statistic = float(statistic)
    log_pvalue = float(chi2.logsf(statistic, dof))
    if np.isfinite(log_pvalue) or not np.isfinite(statistic):
        return log_pvalue

    # Q(a, x) ~ x^(a-1) exp(-x) / Gamma(a) * sum_n (a-1)...(a-n) / x^n
    a, x = dof / 2, statistic / 2
    term, series = 1.0, 1.0
    for n in range(1, 100):
        new_term = term * (a - n) / x
        if abs(new_term) >= abs(term) or new_term == 0:
            break
        term = new_term
        series += term

    return float((a - 1) * np.log(x) - x - gammaln(a) + np.log(series))


def sigma_from_log_pvalue(log_pvalue):
    """
    One-sided Gaussian significance equivalent to a p-value.

    Working with the logarithm of the p-value keeps the significance finite
    for very strong signals, where the p-value itself underflows to zero.

    Parameters
    ----------
    log_pvalue : float
        Natural logarithm of the p-value.

    Returns
    -------
    float
        Number of Gaussian standard deviations (one-sided).
    """
    return float(-ndtri_exp(min(float(log_pvalue), 0.0)))


class PeriodicityTest:
    """
    A class to apply and store the information of the periodicty tests.

    Parameters
    ----------
    pulsar_phases : PulsarPhases object

    Attributes
    ----------
    chisqr_res :
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

    def moments(self, pulsar_phases, n=H_TEST_MAX_HARMONICS):
        # Transform list to be between 0 and 2pi
        plist = np.asarray(pulsar_phases.phases, dtype=float) * 2 * np.pi

        # Calculate the trigonometric moments of the first n harmonics.
        # Looping over the harmonics avoids allocating (n_events x n) arrays.
        cos_moment = np.empty(n)
        sin_moment = np.empty(n)
        for i in range(n):
            cos_moment[i] = np.sum(np.cos((i + 1) * plist))
            sin_moment[i] = np.sum(np.sin((i + 1) * plist))

        # Store the information
        self.number = len(plist)
        self.cos = cos_moment
        self.sin = sin_moment

        return (self.number, self.cos, self.sin)

    def zn_test(self, n=10):
        if n > len(self.cos):
            raise ValueError(
                f"Only {len(self.cos)} harmonics are available, cannot compute Z^2_{n}"
            )

        # Z^2_n is the sum of the powers of the first n harmonics
        cos_moment = self.cos[0:n]
        sin_moment = self.sin[0:n]
        self.Zn_n = n

        # Calculate statistic and pvalue (chi2 distribution with 2n dof)
        Zn = 2 / self.number * np.sum(np.power(cos_moment, 2) + np.power(sin_moment, 2))
        pvalue_zn = chi2.sf(float(Zn), 2 * n)
        sigmas_zn = sigma_from_log_pvalue(chi2_log_pvalue(Zn, 2 * n))

        # Store information
        self.Zntest_res = Zn, pvalue_zn, sigmas_zn

        return self.Zntest_res

    def H_test(self):
        # p-value parametrisation from de Jager & Buesching (2010)
        bn = 0.398
        nharmonics = min(len(self.cos), H_TEST_MAX_HARMONICS)

        # H = max_m (Z^2_m - 4m + 4) for 1 <= m <= 20
        power = 2 / self.number * (np.power(self.cos, 2) + np.power(self.sin, 2))
        m = np.arange(1, nharmonics + 1)
        h = np.cumsum(power[:nharmonics]) - 4 * m + 4

        H = float(np.max(h))
        log_pvalue_H = min(-bn * H, 0.0)
        pvalue_H = np.exp(log_pvalue_H)
        sigmas_H = sigma_from_log_pvalue(log_pvalue_H)

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
            index=["Statistic", "p-value", r"Number of $\sigma$"],
        )
