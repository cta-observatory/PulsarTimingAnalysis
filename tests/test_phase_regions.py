import numpy as np
import pytest

from ptiming_ana.phaseogram import PhaseBinning, PhaseRegions, PulsarPeak
from ptiming_ana.phaseogram.phase_regions import calculate_CountStats


def li_ma_significance(n_on, n_off, alpha):
    """Eq. 17 of Li & Ma (1983)."""
    term_on = n_on * np.log((1 + alpha) / alpha * n_on / (n_on + n_off))
    term_off = n_off * np.log((1 + alpha) * n_off / (n_on + n_off))
    return np.sqrt(2 * (term_on + term_off))


@pytest.fixture
def phases():
    rng = np.random.default_rng(5)
    return np.concatenate([rng.uniform(0, 1, 10000), rng.normal(0.3, 0.01, 500)])


@pytest.fixture
def regions(phases):
    off = PulsarPeak([0.5, 0.9], peaktype="background")
    p1 = PulsarPeak([0.27, 0.33])
    p2 = PulsarPeak([0.0, 0.05, 0.95, 1.0])
    p1p2 = PulsarPeak([0.27, 0.33, 0.0, 0.05, 0.95, 1.0])
    regions = PhaseRegions(OFF_object=off, P1_object=p1, P2_object=p2, P1P2_object=p1p2)

    off.fillPeak(phases)
    for peak in (p1, p2, p1p2):
        peak.fillPeak(phases)
        peak.make_stats(regions, tobs=4.0)
    return regions


def test_peak_definition():
    peak = PulsarPeak([0.0, 0.026, 0.983, 1.0])
    assert peak.type == "signal"
    assert peak.nregions == 2
    assert peak.deltaP == pytest.approx(0.043)


@pytest.mark.parametrize("limits", [[0.1], [0.1, 0.2, 0.3], [], None, [0.9, 0.1]])
def test_wrong_peak_definition(limits):
    with pytest.raises(ValueError, match="Wrong defined regions"):
        PulsarPeak(limits)


def test_wrong_peak_type():
    with pytest.raises(ValueError, match="signal"):
        PulsarPeak([0.1, 0.2], peaktype="other")


def test_fill_peak():
    phases = np.array([0.01, 0.1, 0.2, 0.25, 0.5, 0.97, 0.99])
    peak = PulsarPeak([0.0, 0.05, 0.95, 1.0])
    peak.fillPeak(phases)

    assert peak.number == 3
    np.testing.assert_allclose(peak.phases, [0.01, 0.97, 0.99])

    # A list is also accepted
    peak.fillPeak(list(phases))
    assert peak.number == 3


def test_count_statistics():
    stat, error, n_off = calculate_CountStats(np.zeros(150), np.zeros(400), 0.25)
    assert n_off == pytest.approx(100)
    assert stat.n_sig == pytest.approx(50)
    assert error == pytest.approx(np.sqrt(150 + 0.25**2 * 400))
    assert stat.sqrt_ts == pytest.approx(li_ma_significance(150, 400, 0.25))

    with pytest.raises(ValueError, match="No off data"):
        calculate_CountStats(np.zeros(10), None, 0.25)


def test_peak_statistics(regions, phases):
    p1 = regions.P1
    n_on = np.sum((phases > 0.27) & (phases < 0.33))
    n_off = np.sum((phases > 0.5) & (phases < 0.9))
    alpha = 0.06 / 0.4

    assert p1.number == n_on
    assert regions.OFF.number == n_off
    assert p1.noff == pytest.approx(alpha * n_off)
    assert p1.Nex == pytest.approx(n_on - alpha * n_off)
    assert p1.yerr == pytest.approx(np.sqrt(n_on + alpha**2 * n_off))
    assert p1.sign == pytest.approx(li_ma_significance(n_on, n_off, alpha))
    assert p1.sign_ratio == pytest.approx(p1.sign / 2.0)
    assert p1.s_n_ratio == pytest.approx(p1.Nex / np.sqrt(alpha * n_off))

    # The injected signal is recovered and there is none in P2
    assert p1.Nex == pytest.approx(500, abs=4 * p1.yerr)
    assert p1.sign > 10
    assert abs(regions.P2.sign) < 4


def test_background_has_no_statistics(regions):
    regions.OFF.make_stats(regions, tobs=4.0)
    assert not hasattr(regions.OFF, "sign")


def test_p1_p2_ratio(regions):
    regions.calculate_P1P2()
    p1, p2 = regions.P1, regions.P2
    assert regions.P1P2_ratio == pytest.approx(p1.Nex / p2.Nex)
    assert regions.P1P2_ratio_error == pytest.approx(
        abs(regions.P1P2_ratio)
        * np.sqrt((p1.yerr / p1.Nex) ** 2 + (p2.yerr / p2.Nex) ** 2)
    )


def test_p1_p2_ratio_needs_both_peaks():
    regions = PhaseRegions(
        OFF_object=PulsarPeak([0.5, 0.9], peaktype="background"),
        P1_object=PulsarPeak([0.2, 0.3]),
    )
    regions.calculate_P1P2()
    assert regions.npeaks == 1
    assert regions.P2 is None
    assert regions.P1P2_ratio is None
    assert regions.P1P2_ratio_error is None


def test_results_table(regions):
    table = regions.show_peak_results()
    assert list(table.columns) == ["P1", "P2", "P1+P2"]
    assert table.loc["Significance", "P1"] == pytest.approx(regions.P1.sign)
    assert table.loc["Nex", "P1+P2"] == pytest.approx(regions.P1P2.Nex)
    assert table.loc["Number", "P2"] == regions.P2.number


def test_remove_and_create_peak(regions):
    assert regions.npeaks == 2

    regions.remove_peak("P2")
    assert regions.npeaks == 1
    assert regions.P2 is None
    assert regions.dic["P2"] is None
    assert "P2" not in regions.show_peak_results().columns

    new_peak = PulsarPeak([0.6, 0.7])
    regions.create_peak(new_peak, "P3")
    assert regions.npeaks == 2
    assert regions.P3 is new_peak
    assert regions.dic["P3"] is new_peak

    with pytest.raises(ValueError, match="Unknown region"):
        regions.remove_peak("P5")


def test_no_background():
    regions = PhaseRegions()
    assert regions.OFF is None
    assert regions.npeaks == 0


def test_phase_binning():
    binning = PhaseBinning(50)
    assert binning.nbins == 50
    assert (binning.xmin, binning.xmax) == (0, 1)
    assert binning.getNumEdges() == 51
    np.testing.assert_allclose(binning.edges, np.linspace(0, 1, 51))

    binning = PhaseBinning(10, xmin=0.5, xmax=1.5)
    np.testing.assert_allclose(binning.edges, np.linspace(0.5, 1.5, 11))

    binning.set_edges(nbins=4, xmin=0, xmax=1)
    np.testing.assert_allclose(binning.edges, [0, 0.25, 0.5, 0.75, 1])


def test_custom_phase_binning():
    binning = PhaseBinning([0, 0.1, 0.5, 1.0])
    assert binning.nbins == 3
    assert (binning.xmin, binning.xmax) == (0, 1.0)

    binning = PhaseBinning(np.array([0.2, 0.4, 1.2]))
    assert binning.nbins == 2
    assert binning.xmin == pytest.approx(0.2)


def test_find_edges():
    binning = PhaseBinning(10)
    assert binning.Find_LowHiEdge(0.25) == (2, 3)
    assert binning.Find_LowHiEdge(2.0) is None
    assert binning.Find_CloseEdge(0.24) == 2
    assert binning.Find_CloseEdge(0.27) == 3
    assert binning.Find_CloseEdge(-1) == 0
    assert binning.Find_CloseEdge(3) == 11
