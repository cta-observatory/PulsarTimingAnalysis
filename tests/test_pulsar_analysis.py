"""
Tests of the main analysis class with the Fermi-LAT observations of the Crab
pulsar used in the tutorial and with synthetic data.
"""

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from lstchain.io.io import dl2_params_lstcam_key
from matplotlib.figure import Figure

from ptiming_ana.phaseogram import PulsarAnalysis

# Expected results of the analysis of the Fermi-LAT example data set
# Columns: P1, P2, P1+P2, P3
FERMI_PEAK_STATISTICS = np.array(
    [
        [108.5913594, 58.00656799, 110.16571886, 14.3384333],  # Significance
        [5959.23714286, 2677.27142857, 8636.50857143, 1126.74285714],  # Nex
        [84.22028603, 62.20823982, 105.93665554, 81.25025086],  # Nex_error
        [6969.0, 3734.0, 10703.0, 4884.0],  # Number
        [1009.76285714, 1056.72857143, 2066.49142857, 3757.25714286],  # noff
        [1.18889854, 0.63507746, 1.20613522, 0.15698249],  # sign_t_ratio
        [187.53441252, 82.35892672, 189.98591475, 18.38185577],  # s/n ratio
    ]
)

# Columns: chi-square test, Z^2_10 test, H test
FERMI_PERIODICITY_STATISTICS = np.array(
    [
        [26977.43375174, 25612.34781037, 27915.97396159],  # Statistic
        [0.0, 0.0, 0.0],  # p-value
        [163.17117419, 159.54851651, 149.02774588],  # Number of sigma
    ]
)

# Columns: value, error
FERMI_FIT_RESULT = np.array(
    [
        [0.9924233191795764, 0.0015660661377770266],  # mu
        [0.026304542387059753, 0.0014698372893777872],  # sigma
        [1.3763409922947634, 0.004402295450743174],  # mu_2
        [0.0509859742803803, 0.004192067589437329],  # sigma_2
        [466.4375, 0.0],  # A
        [200.23728941446313, 10.903033719301062],  # B
        [145.13336608138428, 11.04167277916415],  # C
    ]
)


def test_create_analysis_object():
    analysis = PulsarAnalysis()
    assert isinstance(analysis, PulsarAnalysis)
    assert analysis.nbins == 50
    assert analysis.fit_model == "dgaussian"
    assert analysis.do_fit is False
    assert analysis.get_results is False


def test_analysis_objects_are_compared_by_identity():
    """Regression test: all the analysis objects were considered equal."""
    first, second = PulsarAnalysis(), PulsarAnalysis(nbins=20)
    assert first != second
    assert first == first
    assert len({first, second}) == 2


def test_input_file_format():
    with pytest.raises(ValueError, match="no valid format"):
        PulsarAnalysis(filename="events.txt")
    with pytest.raises(ValueError, match="No FITS file"):
        PulsarAnalysis().setFermiInputFile("events.h5")

    analysis = PulsarAnalysis(filename="events.fits")
    assert analysis.telescope == "fermi"
    assert analysis.tint == 3600 * 24 * 10

    analysis = PulsarAnalysis(filename="dl2_events.h5")
    assert analysis.telescope == "lst"
    assert analysis.tint == 3600


def test_wrong_fit_model():
    with pytest.raises(ValueError, match="not defined"):
        PulsarAnalysis(model="unknown")


def test_run_without_input():
    analysis = PulsarAnalysis()
    with pytest.raises(ValueError, match="No input data"):
        analysis.run()


def test_run_without_background_region(synthetic_events):
    analysis = PulsarAnalysis()
    analysis.setListsInput(synthetic_events.phases, synthetic_events.times, tobs=1)
    with pytest.raises(ValueError, match="No background region"):
        analysis.run()


def test_peak_limits():
    analysis = PulsarAnalysis()
    analysis.setPeaklimits(P1_limits=[0, 0.026, 0.983, 1], P2_limits=[0.377, 0.422])
    assert analysis.P1P2_limits == [0, 0.026, 0.983, 1, 0.377, 0.422]
    assert analysis.P3_limits is None

    # "None" strings come from the configuration files
    analysis.setPeaklimits(P1_limits="None", P2_limits=[0.377, 0.422], P3_limits="None")
    assert analysis.P1_limits is None
    assert analysis.P1P2_limits == [0.377, 0.422]


def test_load_config_file(fermi_config, fermi_file):
    """The configuration file is equivalent to setting the parameters by hand."""
    from_file = PulsarAnalysis()
    from_file.set_config(fermi_config)

    by_hand = PulsarAnalysis()
    by_hand.setBackgroundLimits([0.52, 0.87])
    by_hand.setPeaklimits(
        P1_limits=[0, 0.026, 0.983, 1], P2_limits=[0.377, 0.422], P3_limits=[0.08, 0.24]
    )
    by_hand.setBinning(50, xmin=0, xmax=1)
    by_hand.setTimeInterval(tint=3600 * 300)
    by_hand.setFittingParams(model="dgaussian", binned=True)
    by_hand.setEnergybinning(
        np.geomspace(0.1 / 1e3, 1 / 1e3, 3), do_diff=True, do_integral=False
    )  # In TeV
    by_hand.setFermiInputFile(str(fermi_file))

    for name in (
        "OFF_limits",
        "P1_limits",
        "P2_limits",
        "P3_limits",
        "P1P2_limits",
        "nbins",
        "tint",
        "fit_model",
        "binned",
        "do_fit",
        "peak",
        "telescope",
        "energy_units",
        "get_results",
        "filter_data",
    ):
        assert getattr(from_file, name) == getattr(by_hand, name), name

    np.testing.assert_allclose(from_file.binning.edges, by_hand.binning.edges)
    np.testing.assert_allclose(from_file.energy_edges, by_hand.energy_edges)
    assert from_file.r.fname == by_hand.r.fname
    assert from_file.EnergyAna.do_diff and not from_file.EnergyAna.do_integral


def test_fermi_peak_statistics(fermi_analysis):
    peaks, _ = fermi_analysis.show_Presults()

    assert list(peaks.columns) == ["P1", "P2", "P1+P2", "P3"]
    np.testing.assert_allclose(peaks.to_numpy(), FERMI_PEAK_STATISTICS, rtol=1e-7)

    regions = fermi_analysis.regions
    assert regions.P1P2_ratio == pytest.approx(2.22586215)
    assert regions.P1P2_ratio_error == pytest.approx(0.06053491)


def test_fermi_periodicity_statistics(fermi_analysis):
    _, periodicity = fermi_analysis.show_Presults()

    assert list(periodicity.columns) == ["Chi_square_test", "Zn_test", "H_test"]
    np.testing.assert_allclose(
        periodicity.to_numpy(), FERMI_PERIODICITY_STATISTICS, rtol=1e-7
    )
    # The significances are finite even if the p-values underflow
    assert np.all(np.isfinite(periodicity.to_numpy()))


def test_fermi_fit(fermi_analysis):
    assert fermi_analysis.fit_model == "dgaussian"
    result = fermi_analysis.show_fit_results()

    assert list(result["Name"]) == ["mu", "sigma", "mu_2", "sigma_2", "A", "B", "C"]
    np.testing.assert_allclose(
        result[["Value", "Error"]].to_numpy(), FERMI_FIT_RESULT, rtol=1e-5
    )


def test_fermi_observation_time(fermi_analysis):
    assert len(fermi_analysis.phases) == 43020
    assert fermi_analysis.tobs == pytest.approx(8342.59, abs=0.01)

    # The significance per unit time uses the total observation time
    p1 = fermi_analysis.regions.P1
    assert p1.sign_ratio == pytest.approx(p1.sign / np.sqrt(fermi_analysis.tobs))

    # ... which is also the last point of the time evolution
    assert fermi_analysis.TimeEv.t[-1] / 3600 == pytest.approx(fermi_analysis.tobs)


def test_fermi_phases_are_in_the_binning_range(fermi_analysis):
    assert fermi_analysis.phases.min() >= 0
    assert fermi_analysis.phases.max() < 1
    assert fermi_analysis.histogram.lc[0].sum() == len(fermi_analysis.phases)
    np.testing.assert_allclose(
        fermi_analysis.info["pulsar_phase"], fermi_analysis.phases
    )


def test_fermi_energy_analysis(fermi_analysis):
    energy_analysis = fermi_analysis.EnergyAna
    np.testing.assert_allclose(
        energy_analysis.energy_edges * 1e6, [100, 316.227766, 1000]
    )

    assert len(energy_analysis.Parray) == 2
    assert [len(p.phases) for p in energy_analysis.Parray] == [18913, 15995]
    assert energy_analysis.Parray[0].regions.P1.sign == pytest.approx(64.28311038)
    assert energy_analysis.Parray[1].regions.P2.sign == pytest.approx(39.60353929)

    # The analysis in energy bins does not change the overall results
    assert fermi_analysis.regions.P1.number == 6969


def test_analysis_from_lists_matches_file(fermi_analysis, fermi_file):
    """Giving the events as lists is equivalent to reading the Fermi-LAT file."""
    with fits.open(fermi_file) as hdul:
        table = hdul[1].data
        order = np.argsort(table["BARYCENTRIC_TIME"])
        times = np.array(table["BARYCENTRIC_TIME"], dtype=float)[order]
        phases = np.array(table["PULSE_PHASE"], dtype=float)[order]
        energies = np.array(table["ENERGY"], dtype=float)[order]

    analysis = PulsarAnalysis()
    analysis.setBackgroundLimits([0.52, 0.87])
    analysis.setPeaklimits(
        P1_limits=[0, 0.026, 0.983, 1], P2_limits=[0.377, 0.422], P3_limits=[0.08, 0.24]
    )
    analysis.setBinning(50, xmin=0, xmax=1)
    analysis.setTimeInterval(tint=3600 * 300)
    analysis.setFittingParams(model="dgaussian", binned=True)
    analysis.setListsInput(
        phases, times, energies / 1e6, tel="fermi", energy_units="TeV"
    )
    analysis.run()

    peaks, periodicity = analysis.show_Presults()
    np.testing.assert_allclose(peaks.to_numpy(), FERMI_PEAK_STATISTICS, rtol=1e-7)
    np.testing.assert_allclose(
        periodicity.to_numpy(), FERMI_PERIODICITY_STATISTICS, rtol=1e-7
    )
    assert analysis.tobs == pytest.approx(fermi_analysis.tobs)
    assert not analysis.check_energyana()


def test_synthetic_peak_statistics(synthetic_analysis, synthetic_events):
    regions = synthetic_analysis.regions

    # The regions contain +-3 sigma of the peaks (99.7% of the events)
    assert regions.P1.Nex == pytest.approx(
        0.997 * synthetic_events.n_p1, abs=4 * regions.P1.yerr
    )
    assert regions.P2.Nex == pytest.approx(
        0.997 * synthetic_events.n_p2, abs=4 * regions.P2.yerr
    )
    assert regions.P1P2.Nex == pytest.approx(regions.P1.Nex + regions.P2.Nex)
    assert regions.P1.sign > regions.P2.sign > 20
    assert regions.P3 is None
    assert synthetic_analysis.tobs == pytest.approx(synthetic_events.tobs)


def test_phase_shift(make_synthetic_analysis):
    """The phases are shifted to the range of the phase binning."""
    analysis = make_synthetic_analysis(do_fit=False)
    analysis.setBinning(50, xmin=0.5, xmax=1.5)
    analysis.setBackgroundLimits([0.52, 0.87])
    analysis.setPeaklimits(P1_limits=[1.14, 1.26], P2_limits=[1.31, 1.49])
    analysis.run()

    assert analysis.phases.min() >= 0.5
    assert analysis.phases.max() < 1.5
    assert analysis.regions.P1.sign > 20


def test_time_evolution(synthetic_analysis, synthetic_events):
    time_evolution = synthetic_analysis.TimeEv
    n_points = len(time_evolution.t)

    # One point every 10 minutes (plus the initial and the final points)
    assert n_points == int(synthetic_events.tobs * 6) + 2
    assert np.all(np.diff(time_evolution.t) > 0)
    assert time_evolution.t[-1] / 3600 == pytest.approx(synthetic_events.tobs)

    for values in (
        time_evolution.P1sTime,
        time_evolution.P2sTime,
        time_evolution.P1P2sTime,
        time_evolution.HTime,
        time_evolution.ZTime,
        time_evolution.ChiTime,
        time_evolution.P1exTime,
        time_evolution.P1exerror,
    ):
        assert len(values) == n_points

    # The last point corresponds to the whole data set
    regions = synthetic_analysis.regions
    assert time_evolution.P1sTime[-1] == pytest.approx(regions.P1.sign)
    assert time_evolution.P1P2exTime[-1] == pytest.approx(regions.P1P2.Nex)
    assert time_evolution.HTime[-1] == pytest.approx(
        synthetic_analysis.stats.Htest_res[2]
    )

    # The significance grows with time
    assert time_evolution.P1P2sTime[-1] > time_evolution.P1P2sTime[1] > 0


def test_time_evolution_includes_isolated_last_events(make_synthetic_analysis):
    """
    Regression test: if the last event was separated from the rest by a long
    gap, the events after the last time interval were not included in the results.
    """
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 1, 1000)
    times = 60000 + np.arange(1000) / 86400
    times[-1] += 1  # one day later

    analysis = PulsarAnalysis()
    analysis.setBackgroundLimits([0.52, 0.87])
    analysis.setPeaklimits(P1_limits=[0.14, 0.26], P2_limits=[0.31, 0.49])
    analysis.setTimeInterval(tint=600)
    analysis.setListsInput(phases, times, tel="MAGIC", tobs=1.0)
    analysis.run()

    assert len(analysis.phases) == 1000
    n_regions = analysis.regions.OFF.number + analysis.regions.P1P2.number
    expected = np.sum(
        ((phases > 0.52) & (phases < 0.87))
        | ((phases > 0.14) & (phases < 0.26))
        | ((phases > 0.31) & (phases < 0.49))
    )
    assert n_regions == expected


def test_energy_analysis(synthetic_analysis, synthetic_events):
    energy_analysis = synthetic_analysis.EnergyAna
    energies = synthetic_events.energies

    assert [len(p.phases) for p in energy_analysis.Parray] == [
        np.sum((energies > 0.02) & (energies < 0.1)),
        np.sum((energies > 0.1) & (energies < 1.0)),
    ]
    assert [len(p.phases) for p in energy_analysis.Parray_integral] == [
        np.sum(energies > 0.02),
        np.sum(energies > 0.1),
    ]

    peak_statistics, periodicity = synthetic_analysis.show_EnergyPresults()
    assert len(peak_statistics) == len(periodicity) == 2
    for table, analysis in zip(peak_statistics, energy_analysis.Parray):
        assert table.loc["Significance", "P1"] == pytest.approx(
            analysis.regions.P1.sign
        )
        # The total observation time is used in each energy bin
        assert analysis.tobs == pytest.approx(synthetic_events.tobs)

    fit_results = synthetic_analysis.show_EnergyFitresults()
    assert len(fit_results) == 2
    for result in fit_results:
        values = dict(zip(result["Name"], result["Value"]))
        assert values["mu"] == pytest.approx(1.2, abs=0.01)
        assert values["mu_2"] == pytest.approx(1.4, abs=0.01)

    ratio, ratio_error = energy_analysis.P1P2_ratioVsEnergy()
    assert len(ratio) == len(ratio_error) == 2
    assert ratio[0] == pytest.approx(3500 / 2500, abs=4 * ratio_error[0])


def test_peak_shape_vs_energy(synthetic_analysis):
    shape = synthetic_analysis.EnergyAna._fit_shape_vs_energy()

    assert list(shape) == ["P1", "P2"]
    for peak, mean, sigma in [("P1", 1.2, 0.02), ("P2", 1.4, 0.03)]:
        np.testing.assert_allclose(shape[peak]["energy"], [0.06, 0.55])
        np.testing.assert_allclose(shape[peak]["mean"], mean, atol=0.01)
        # FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian
        np.testing.assert_allclose(shape[peak]["fwhm"], 2.3548 * sigma, atol=0.015)
        assert np.all(shape[peak]["fwhm_err"] > 0)
        assert np.all(shape[peak]["mean_err"] > 0)


def test_energy_analysis_not_available(make_synthetic_analysis):
    analysis = make_synthetic_analysis()
    analysis.run()

    assert not analysis.check_energyana()
    assert analysis.show_EnergyAna() is None
    assert analysis.show_lcVsEnergy() is None

    analysis.setEnergybinning([0.02, 0.1, 1.0], do_diff=True, do_integral=False)
    analysis.run()
    with pytest.raises(ValueError, match="Integral results not produced"):
        analysis.show_EnergyPresults(integral=True)


@pytest.mark.parametrize("stats", ["short", "long"])
@pytest.mark.parametrize("fit", [True, False])
def test_draw_phaseogram(synthetic_analysis, stats, fit):
    figure = synthetic_analysis.draw_phaseogram(stats=stats, fit=fit)
    assert isinstance(figure, Figure)

    axes = figure.gca()
    assert axes.get_xlim() == (0, 2)
    assert axes.get_xlabel() == "Pulsar phase"
    labels = axes.get_legend_handles_labels()[1]
    assert {"OFF", "P1", "P2"} <= set(labels)
    assert ("Fit" in labels) == fit


def test_plots(synthetic_analysis):
    figures = synthetic_analysis.show_timeEvolution()
    assert len(figures[0]) == 3
    assert isinstance(figures[1], Figure)

    assert isinstance(synthetic_analysis.show_EnergyAna(), Figure)
    assert isinstance(synthetic_analysis.show_EnergyAna(integral=True), Figure)
    assert isinstance(synthetic_analysis.show_meanVsEnergy(), Figure)
    assert isinstance(synthetic_analysis.show_FWHMVsEnergy(), Figure)
    assert isinstance(synthetic_analysis.show_SigVsEnergy(), Figure)
    assert isinstance(synthetic_analysis.show_P1P2VsEnergy(), Figure)
    assert isinstance(synthetic_analysis.show_all_lc(), Figure)
    assert isinstance(synthetic_analysis.show_all_fits(), Figure)
    assert len(synthetic_analysis.show_lcVsEnergy()) == 2


def test_plots_do_not_block(synthetic_analysis, monkeypatch):
    """A library must not call plt.show(): it blocks the execution of scripts."""

    def fail(*args, **kwargs):
        raise AssertionError("plt.show() was called")

    monkeypatch.setattr(plt, "show", fail)
    synthetic_analysis.show_timeEvolution()
    synthetic_analysis.show_lcVsEnergy()
    synthetic_analysis.EnergyAna.PeaksVsEnergy()


@pytest.mark.parametrize("output", ["results.pdf", "output_directory"])
def test_save_results(synthetic_analysis, tmp_path, output):
    analysis = copy.copy(synthetic_analysis)
    n_figures = len(plt.get_fignums())
    analysis.save_results(str(tmp_path / output))

    if output.endswith(".pdf"):
        pdf_file = tmp_path / output
    else:
        pdf_file = tmp_path / output / "pulsar_analysis_results.pdf"

    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 10000
    assert len(plt.get_fignums()) == n_figures

    fit = pd.read_hdf(pdf_file.parent / "overall_fitting.h5", key="results")
    pd.testing.assert_frame_equal(fit, synthetic_analysis.show_fit_results())
    for i in range(2):
        assert (pdf_file.parent / f"fitting_energy_bin{i}.h5").exists()


def test_save_results_from_config(fermi_config_dict, write_config, tmp_path):
    config = copy.deepcopy(fermi_config_dict)
    config["results"] = {
        "save_results": True,
        "output_directory": str(tmp_path / "phaseogram"),
    }
    config["energy_binning"]["run_energy_analysis"] = False

    analysis = PulsarAnalysis()
    analysis.set_config(write_config(config))
    analysis.run()

    assert (tmp_path / "phaseogram" / "pulsar_analysis_results.pdf").exists()
    assert (tmp_path / "phaseogram" / "overall_fitting.h5").exists()


def test_save_results_without_output(synthetic_analysis):
    analysis = copy.copy(synthetic_analysis)
    analysis.output_file = None
    with pytest.raises(ValueError, match="No output file"):
        analysis.save_results()


@pytest.mark.parametrize("file_format", ["h5", "csv"])
def test_save_events(synthetic_analysis, tmp_path, file_format):
    output = tmp_path / f"events.{file_format}"
    synthetic_analysis.save_df(str(output), file_format=file_format)

    if file_format == "h5":
        events = pd.read_hdf(output, key=dl2_params_lstcam_key)
    else:
        events = pd.read_csv(output)

    assert len(events) == len(synthetic_analysis.phases)
    np.testing.assert_allclose(events["pulsar_phase"], synthetic_analysis.phases)

    with pytest.raises(ValueError, match="File format not valid"):
        synthetic_analysis.save_df(str(output), file_format="fits")


def test_save_object(make_synthetic_analysis, tmp_path):
    analysis = make_synthetic_analysis()
    analysis.run()
    output = tmp_path / "analysis.pkl"
    analysis.save_object(str(output))

    with open(output, "rb") as file:
        restored = pickle.load(file)

    assert restored.regions.P1.sign == pytest.approx(analysis.regions.P1.sign)
    np.testing.assert_allclose(restored.fitting.params, analysis.fitting.params)


@pytest.fixture
def dl2_pulsar_file(tmp_path, dl2_dataframe):
    """DL2 file with phases (one peak at phase 0.2) and theta2 values."""
    rng = np.random.default_rng(13)
    n_events = len(dl2_dataframe)
    pulsed = rng.uniform(size=n_events) < 0.1
    phases = np.where(
        pulsed, rng.normal(0.2, 0.02, n_events) % 1, rng.uniform(0, 1, n_events)
    )

    events = dl2_dataframe.copy()
    events["pulsar_phase"] = phases
    events["mjd_time"] = events["dragon_time"] / 86400 + 40587
    events["theta2"] = rng.uniform(0, 0.2, n_events)

    directory = tmp_path / "dl2"
    directory.mkdir()
    events.to_hdf(directory / "dl2_LST-1.Run00001_pulsar.h5", key=dl2_params_lstcam_key)
    return directory, events


@pytest.mark.parametrize("input_type", ["file", "directory"])
def test_dl2_analysis_from_config(
    fermi_config_dict, write_config, dl2_pulsar_file, input_type
):
    """
    Regression test: the configuration of the extra DL2 cuts in a
    source-independent analysis raised a TypeError.
    """
    directory, events = dl2_pulsar_file
    config = copy.deepcopy(fermi_config_dict)
    if input_type == "file":
        config["pulsar_file_dir"] = str(directory / "dl2_LST-1.Run00001_pulsar.h5")
    else:
        config["pulsar_file_dir"] = str(directory)
    config["flags"] = {"src_dependent": False, "DL2_format": True, "fits_format": False}
    config["cuts"]["include_DL2_extra_cuts"] = True
    config["cuts"]["extra_cuts"].update(
        {"energy": 0, "theta": 0.3, "intensity": 50, "gammaness": 0.5}
    )
    config["phase_regions"] = {
        "Bkg": [0.52, 0.87],
        "P1": [0.14, 0.26],
        "P2": "None",
        "P3": "None",
    }
    config["time_binning"] = {"run_time_analysis": True, "units": "s", "tint": 1}
    config["energy_binning"]["run_energy_analysis"] = False
    config["fitting"] = {"run_fitting": True, "model": "gaussian", "binned": True}

    analysis = PulsarAnalysis()
    analysis.set_config(write_config(config))
    analysis.run()

    selected = (
        (events["event_type"] == 32)
        & (events["gammaness"] > 0.5)
        & (events["theta2"] < 0.3**2)
        & (events["intensity"] > 50)
        & (events["alt_tel"] > np.deg2rad(30))
    )
    assert analysis.telescope == "lst"
    assert analysis.peak == "P1"
    assert len(analysis.phases) == selected.sum()
    assert analysis.tobs == pytest.approx(6 / 3600, rel=0.05)
    assert analysis.regions.P1.sign > 10
    assert analysis.regions.P2 is None
    assert analysis.regions.P1P2_ratio is None

    values = dict(zip(analysis.fitting.parnames, analysis.fitting.params))
    assert values["mu"] == pytest.approx(1.2, abs=0.01)
