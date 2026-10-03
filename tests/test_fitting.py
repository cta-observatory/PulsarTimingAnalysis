import numpy as np
import pytest

from ptiming_ana.phaseogram import PeakFitting, get_model_list

# Parameters of the synthetic pulse profile (see conftest.py). The peaks are
# fitted in the phase range starting in the centre of the OFF region (0.695),
# so the phases of the peaks are shifted by one period.
P1_MEAN, P1_SIGMA = 1.20, 0.02
P2_MEAN, P2_SIGMA = 1.40, 0.03


def fit_result(analysis):
    table = analysis.show_fit_results()
    return dict(zip(table["Name"], table["Value"])), dict(
        zip(table["Name"], table["Error"])
    )


@pytest.mark.parametrize("model", get_model_list())
def test_model_definition(model):
    peak = "P1" if model in ("gaussian", "lorentzian") else "both"
    fitting = PeakFitting(binned=True, model=model, peak=peak)
    assert fitting.model == model
    assert not fitting.check_fit_result()
    assert fitting.show_result() is None


@pytest.mark.parametrize(
    "model, peak, message",
    [
        ("unknown", "both", "not in the available model list"),
        ("gaussian", "both", "can only fit one peak"),
        ("lorentzian", "both", "can only fit one peak"),
        ("dgaussian", "P1", "needs 2 peaks"),
        ("dgaussian", "P2", "needs 2 peaks"),
        ("double_lorentz", "P1", "needs 2 peaks"),
        ("tgaussian", "P2", "needs 3 peaks"),
        ("dgaussian", "P4", "must be one of"),
    ],
)
def test_wrong_model_definition(model, peak, message):
    with pytest.raises(ValueError, match=message):
        PeakFitting(binned=True, model=model, peak=peak)


def test_binned_double_gaussian(synthetic_analysis, synthetic_events):
    values, errors = fit_result(synthetic_analysis)

    assert list(values) == ["mu", "sigma", "mu_2", "sigma_2", "A", "B", "C"]
    assert values["mu"] == pytest.approx(P1_MEAN, abs=5 * errors["mu"])
    # The model is evaluated at the centre of the bins, so the widths of peaks
    # that are narrow compared to the bin width (0.02) are slightly overestimated
    assert values["sigma"] == pytest.approx(P1_SIGMA, abs=0.004)
    assert values["mu_2"] == pytest.approx(P2_MEAN, abs=5 * errors["mu_2"])
    assert values["sigma_2"] == pytest.approx(P2_SIGMA, abs=0.004)
    assert errors["mu"] < 2e-3

    # The background is fixed to the level of the OFF region (counts per bin)
    nbins = 50
    assert errors["A"] == 0
    assert values["A"] == pytest.approx(synthetic_events.n_bkg / nbins, rel=0.05)

    # The amplitudes are the number of events in the peak per phase bin
    assert values["B"] * nbins == pytest.approx(
        synthetic_events.n_p1, abs=5 * errors["B"] * nbins
    )
    assert values["C"] * nbins == pytest.approx(
        synthetic_events.n_p2, abs=5 * errors["C"] * nbins
    )


def test_unbinned_double_gaussian(make_synthetic_analysis, synthetic_events):
    analysis = make_synthetic_analysis(model="dgaussian", binned=False)
    analysis.run()
    values, errors = fit_result(analysis)

    assert analysis.fitting.minuit.valid
    assert values["mu"] == pytest.approx(P1_MEAN, abs=5 * errors["mu"])
    assert values["sigma"] == pytest.approx(P1_SIGMA, abs=5 * errors["sigma"])
    assert values["mu_2"] == pytest.approx(P2_MEAN, abs=5 * errors["mu_2"])
    assert values["sigma_2"] == pytest.approx(P2_SIGMA, abs=5 * errors["sigma_2"])

    # Same units as the binned fit: counts per phase bin
    nbins = 50
    assert values["A"] * nbins == pytest.approx(
        synthetic_events.n_bkg, abs=5 * errors["A"] * nbins
    )
    assert values["B"] * nbins == pytest.approx(
        synthetic_events.n_p1, abs=5 * errors["B"] * nbins
    )
    assert values["C"] * nbins == pytest.approx(
        synthetic_events.n_p2, abs=5 * errors["C"] * nbins
    )


def test_binned_and_unbinned_fits_are_compatible(
    synthetic_analysis, make_synthetic_analysis
):
    analysis = make_synthetic_analysis(model="dgaussian", binned=False)
    analysis.run()

    binned, binned_errors = fit_result(synthetic_analysis)
    unbinned, _ = fit_result(analysis)
    for name in ("mu", "sigma", "mu_2", "sigma_2"):
        assert unbinned[name] == pytest.approx(
            binned[name], abs=5 * binned_errors[name]
        )


def test_unbinned_fit_does_not_modify_the_phases(make_synthetic_analysis):
    """Regression test: the phases of the analysis were shifted in place."""
    analysis = make_synthetic_analysis(model="dgaussian", binned=False)
    analysis.run()

    assert analysis.fitting.shift == pytest.approx(0.695)
    assert analysis.phases.min() >= 0
    assert analysis.phases.max() < 1


@pytest.mark.parametrize("binned", [True, False])
@pytest.mark.parametrize("model", ["double_lorentz", "asym_dgaussian"])
def test_two_peak_models(make_synthetic_analysis, model, binned):
    analysis = make_synthetic_analysis(model=model, binned=binned)
    analysis.run()
    values, _ = fit_result(analysis)

    mean_names = [name for name in values if name.startswith("mu")]
    assert values[mean_names[0]] == pytest.approx(P1_MEAN, abs=0.01)
    assert values[mean_names[1]] == pytest.approx(P2_MEAN, abs=0.01)
    assert all(np.isfinite(list(values.values())))


def test_asymmetric_gaussian_of_symmetric_peaks(make_synthetic_analysis):
    analysis = make_synthetic_analysis(model="asym_dgaussian", binned=True)
    analysis.run()
    values, errors = fit_result(analysis)

    for name, expected in [("sigma1", P1_SIGMA), ("sigma2", P1_SIGMA)]:
        assert abs(values[name]) == pytest.approx(expected, abs=5 * errors[name])


@pytest.mark.parametrize("binned", [True, False])
@pytest.mark.parametrize(
    "model, peak, mean, width, width_name",
    [
        ("gaussian", "P1", P1_MEAN, P1_SIGMA, "sigma"),
        ("gaussian", "P2", P2_MEAN, P2_SIGMA, "sigma"),
        ("lorentzian", "P2", P2_MEAN, None, "gamma_1"),
    ],
)
def test_single_peak_models(model, peak, mean, width, width_name, binned):
    """Regression test: single-peak models could only fit P2 (binned)."""
    from ptiming_ana.phaseogram import PulsarAnalysis

    rng = np.random.default_rng(21)
    if model == "gaussian":
        peak_phases = rng.normal(mean, width, 4000)
    else:
        peak_phases = mean + 0.02 * rng.standard_cauchy(4000)
        peak_phases = peak_phases[np.abs(peak_phases - mean) < 0.3]
    phases = np.concatenate([rng.uniform(0, 1, 14000), peak_phases % 1])
    times = 60000 + np.sort(rng.uniform(0, 0.04, len(phases)))

    limits = {"P1_limits": None, "P2_limits": None}
    limits[f"{peak}_limits"] = [0.14, 0.26] if peak == "P1" else [0.31, 0.49]

    analysis = PulsarAnalysis()
    analysis.setBackgroundLimits([0.52, 0.87])
    analysis.setPeaklimits(**limits)
    analysis.setBinning(50, xmin=0, xmax=1)
    analysis.setTimeInterval(tint=600)
    analysis.setFittingParams(model=model, binned=binned, peak=peak)
    analysis.setListsInput(phases, times, tobs=1.0)
    analysis.run()
    values, _ = fit_result(analysis)

    mean_name = "mu" if model == "gaussian" else "mu_1"
    assert values[mean_name] == pytest.approx(mean, abs=0.01)
    if width is not None:
        assert values[width_name] == pytest.approx(width, abs=0.005)
    else:
        assert values[width_name] == pytest.approx(0.02, abs=0.005)


def test_triple_gaussian(synthetic_events):
    from ptiming_ana.phaseogram import PulsarAnalysis

    rng = np.random.default_rng(8)
    n_p3 = 3000
    phases = np.concatenate([synthetic_events.phases, rng.normal(0.95, 0.02, n_p3) % 1])
    times = np.concatenate(
        [synthetic_events.times, rng.choice(synthetic_events.times, n_p3)]
    )

    analysis = PulsarAnalysis()
    analysis.setBackgroundLimits([0.52, 0.87])
    analysis.setPeaklimits(
        P1_limits=[0.14, 0.26], P2_limits=[0.31, 0.49], P3_limits=[0.89, 1.0]
    )
    analysis.setBinning(50, xmin=0, xmax=1)
    analysis.setTimeInterval(tint=600)
    analysis.setFittingParams(model="tgaussian", binned=True)
    analysis.setListsInput(phases, times, tel="LST", tobs=1.0)
    analysis.run()
    values, _ = fit_result(analysis)

    assert list(values) == [
        "A",
        "mu",
        "sigma",
        "mu_2",
        "sigma_2",
        "mu_3",
        "sigma_3",
        "B",
        "C",
        "D",
    ]
    assert values["mu"] == pytest.approx(P1_MEAN, abs=0.01)
    assert values["mu_2"] == pytest.approx(P2_MEAN, abs=0.01)
    assert values["mu_3"] == pytest.approx(0.95, abs=0.01)
    assert abs(values["sigma_3"]) == pytest.approx(0.02, abs=0.005)


def test_binned_fit_with_empty_bins():
    """Regression test: bins without events made the residuals infinite."""
    from ptiming_ana.phaseogram import PulsarAnalysis

    rng = np.random.default_rng(4)
    phases = np.concatenate(
        [
            rng.uniform(0, 1, 60),
            rng.normal(0.2, 0.02, 300) % 1,
            rng.normal(0.4, 0.03, 300) % 1,
        ]
    )
    times = 60000 + np.sort(rng.uniform(0, 0.02, len(phases)))

    analysis = PulsarAnalysis()
    analysis.setBackgroundLimits([0.52, 0.87])
    analysis.setPeaklimits(P1_limits=[0.14, 0.26], P2_limits=[0.31, 0.49])
    analysis.setBinning(100, xmin=0, xmax=1)
    analysis.setTimeInterval(tint=600)
    analysis.setFittingParams(model="dgaussian", binned=True)
    analysis.setListsInput(phases, times, tel="LST", tobs=0.5)
    analysis.run()

    assert np.any(analysis.histogram.lc[0] == 0)
    values, _ = fit_result(analysis)
    assert values["mu"] == pytest.approx(P1_MEAN, abs=0.01)
    assert values["mu_2"] == pytest.approx(P2_MEAN, abs=0.01)


def test_missing_peak_region(make_synthetic_analysis):
    analysis = make_synthetic_analysis(model="tgaussian", binned=True)
    with pytest.raises(ValueError, match="needs the P3 region"):
        analysis.run()
