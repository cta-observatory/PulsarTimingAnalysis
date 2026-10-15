from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi2, norm

from ptiming_ana.phaseogram import Lightcurve, PeriodicityTest, PhaseBinning
from ptiming_ana.phaseogram.periodicity_test import (
    H_TEST_MAX_HARMONICS,
    chi2_log_pvalue,
    sigma_from_log_pvalue,
)


def z2_statistic(phases, n):
    """Independent implementation of the Z^2_n statistic (Buccheri et al. 1983)."""
    k = np.arange(1, n + 1)
    angles = 2 * np.pi * np.outer(phases, k)
    return (
        2
        / len(phases)
        * np.sum(np.cos(angles).sum(axis=0) ** 2 + np.sin(angles).sum(axis=0) ** 2)
    )


def periodicity_test(phases, nbins=50):
    data = SimpleNamespace(phases=np.asarray(phases))
    data.histogram = Lightcurve(data, PhaseBinning(nbins))
    return PeriodicityTest(data)


@pytest.fixture(scope="module")
def uniform_phases():
    return np.random.default_rng(0).uniform(0, 1, 20000)


@pytest.fixture(scope="module")
def pulsed_phases(uniform_phases):
    rng = np.random.default_rng(1)
    return np.concatenate([uniform_phases, rng.normal(0.3, 0.02, 800) % 1])


@pytest.mark.parametrize("statistic, dof", [(30.0, 20), (5.0, 49), (200.0, 2)])
def test_chi2_log_pvalue_matches_scipy(statistic, dof):
    assert chi2_log_pvalue(statistic, dof) == pytest.approx(
        np.log(chi2.sf(statistic, dof)), rel=1e-10
    )


def test_chi2_log_pvalue_is_finite_for_strong_signals():
    """The p-value underflows to zero, but its logarithm must stay finite."""
    assert chi2.sf(27000, 49) == 0
    log_pvalue = chi2_log_pvalue(27000, 49)
    assert np.isfinite(log_pvalue)
    assert log_pvalue < -13000

    # Continuity with the exact calculation just before the underflow
    assert chi2_log_pvalue(1500.0, 20) < chi2_log_pvalue(1400.0, 20)
    assert chi2_log_pvalue(1500.0, 20) == pytest.approx(-703.2, abs=0.1)


@pytest.mark.parametrize("sigma", [0.0, 1.0, 3.0, 5.0, 8.0])
def test_sigma_from_log_pvalue(sigma):
    assert sigma_from_log_pvalue(norm.logsf(sigma)) == pytest.approx(sigma, abs=1e-9)


def test_sigma_from_log_pvalue_extreme():
    assert sigma_from_log_pvalue(-13329.7) == pytest.approx(163.24, abs=0.01)
    assert sigma_from_log_pvalue(0.0) == -np.inf


def test_zn_test_uses_n_harmonics(pulsed_phases):
    """Regression test: Z^2_10 was computed with 11 harmonics."""
    test = periodicity_test(pulsed_phases)
    statistic, pvalue, sigma = test.Zntest_res

    assert test.Zn_n == 10
    assert statistic == pytest.approx(z2_statistic(pulsed_phases, 10), rel=1e-10)
    assert statistic != pytest.approx(z2_statistic(pulsed_phases, 11), rel=1e-6)
    assert pvalue == pytest.approx(chi2.sf(statistic, 20))
    assert sigma == pytest.approx(norm.isf(pvalue), rel=1e-6)


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_zn_test_other_harmonics(pulsed_phases, n):
    test = periodicity_test(pulsed_phases)
    statistic, pvalue, _ = test.zn_test(n=n)
    assert statistic == pytest.approx(z2_statistic(pulsed_phases, n), rel=1e-10)
    assert pvalue == pytest.approx(chi2.sf(statistic, 2 * n))


def test_zn_test_too_many_harmonics(uniform_phases):
    test = periodicity_test(uniform_phases)
    with pytest.raises(ValueError, match="harmonics"):
        test.zn_test(n=H_TEST_MAX_HARMONICS + 1)


def test_h_test(pulsed_phases):
    """H = max(Z^2_m - 4m + 4) for m = 1...20 (de Jager et al. 1989)."""
    test = periodicity_test(pulsed_phases)
    statistic, pvalue, sigma = test.Htest_res

    expected = max(z2_statistic(pulsed_phases, m) - 4 * m + 4 for m in range(1, 21))
    assert statistic == pytest.approx(expected, rel=1e-10)
    # de Jager & Buesching (2010)
    assert pvalue == pytest.approx(np.exp(-0.398 * expected))
    assert sigma == pytest.approx(norm.isf(pvalue), rel=1e-6)
    assert len(test.cos) == len(test.sin) == H_TEST_MAX_HARMONICS


def test_chi_square_test(pulsed_phases):
    test = periodicity_test(pulsed_phases, nbins=25)
    statistic, pvalue, sigma = test.chisqr_res

    counts, _ = np.histogram(pulsed_phases, bins=np.linspace(0, 1, 26))
    expected = np.sum((counts - counts.mean()) ** 2 / counts.mean())
    assert statistic == pytest.approx(expected)
    assert pvalue == pytest.approx(chi2.sf(expected, 24))
    assert sigma == pytest.approx(norm.isf(pvalue), rel=1e-6)


def test_no_detection_for_uniform_phases(uniform_phases):
    test = periodicity_test(uniform_phases)
    for _, pvalue, sigma in (test.chisqr_res, test.Zntest_res, test.Htest_res):
        assert pvalue > 1e-3
        assert sigma < 3


def test_detection_of_pulsed_signal(pulsed_phases):
    test = periodicity_test(pulsed_phases)
    for _, pvalue, sigma in (test.chisqr_res, test.Zntest_res, test.Htest_res):
        assert pvalue < 1e-10
        assert sigma > 6


def test_significance_is_finite_for_very_strong_signal():
    rng = np.random.default_rng(3)
    phases = np.concatenate(
        [rng.uniform(0, 1, 20000), rng.normal(0.5, 0.01, 20000) % 1]
    )
    test = periodicity_test(phases)
    for _, pvalue, sigma in (test.chisqr_res, test.Zntest_res, test.Htest_res):
        assert pvalue == 0
        assert np.isfinite(sigma)
        assert sigma > 38


def test_results_table(pulsed_phases):
    table = periodicity_test(pulsed_phases).show_Pstats()
    assert list(table.columns) == ["Chi_square_test", "Zn_test", "H_test"]
    assert table.shape == (3, 3)
    assert list(table.index[:2]) == ["Statistic", "p-value"]
