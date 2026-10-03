import copy

import astropy.units as u
import numpy as np
import pytest
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.irf import EDispKernelMap
from gammapy.maps import MapAxis, RegionGeom, RegionNDMap
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)

from ptiming_ana.spectral import (
    SpectralConfigSetting,
    SpectralPulsarAnalysis,
    compute_spectral_points,
    do_fitting,
    get_kwargs_line,
    get_kwargs_points,
    get_kwargs_region,
    read_DL4_files,
    set_makers,
    set_model_to_fit,
)

TRUE_INDEX = 3.0
TRUE_AMPLITUDE = 3e-11 * u.Unit("cm-2 s-1 TeV-1")


@pytest.fixture
def spectral_config(fermi_config_dict, write_config, tmp_path):
    config = copy.deepcopy(fermi_config_dict)
    config["results_output_directory"] = str(tmp_path / "results")
    config["DL4_directory"] = str(tmp_path / "DL4")
    return write_config(config)


@pytest.fixture(scope="module")
def simulated_dataset():
    """ON/OFF spectrum of a source with a power-law spectrum (50 h)."""
    energy_axis = MapAxis.from_energy_bounds(0.01, 10, 24, unit="TeV")
    energy_axis_true = MapAxis.from_energy_bounds(
        0.005, 20, 40, unit="TeV", name="energy_true"
    )
    geom = RegionGeom.create("icrs;circle(83.633, 22.01, 0.2)", axes=[energy_axis])

    dataset = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name="simulation"
    )
    dataset.exposure.quantity = (
        np.full(dataset.exposure.data.shape, 2e5) * u.m**2 * 50 * u.h
    )
    dataset.exposure.meta["livetime"] = 50 * u.h
    dataset.mask_safe.data[...] = True
    dataset.edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis, energy_axis_true=energy_axis_true, geom=geom
    )
    dataset.models = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=TRUE_INDEX, amplitude=TRUE_AMPLITUDE, reference=0.15 * u.TeV
        ),
        name="Crab",
    )

    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, acceptance=1, acceptance_off=5
    )
    background = RegionNDMap.from_geom(geom)
    background.data += 2000 * (energy_axis.center.value[:, None, None] / 0.05) ** -2
    dataset_on_off.fake(npred_background=background, random_state=0)
    dataset_on_off.models = None

    return dataset_on_off, geom


def test_read_configuration(spectral_config, tmp_path):
    settings = SpectralConfigSetting(spectral_config)
    settings.set_all()

    assert settings.target_info == {"name": "Crab", "ra": 83.633, "dec": 22.01}
    assert settings.phase_region_dic["Bkg"] == [0.52, 0.87]
    assert settings.e_min_fitting == 20 * u.GeV
    assert settings.e_max_fitting == 700 * u.GeV
    assert settings.e_min_points == 20 * u.GeV
    assert settings.model == "PowerLaw"
    assert settings.npoints == 7
    assert settings.bins_per_decade is False
    assert settings.min_ts == 1.5

    # The output directories are created
    assert (tmp_path / "results").is_dir()
    assert (tmp_path / "DL4").is_dir()


def test_configuration_file_format():
    with pytest.raises(ValueError, match="not a YAML file"):
        SpectralConfigSetting("config.txt")
    with pytest.raises(ValueError, match="not a YAML file"):
        SpectralPulsarAnalysis("config.json")


def test_energy_geometry(spectral_config):
    settings = SpectralConfigSetting(spectral_config)
    settings.set_all()
    true_axis, reco_axis = settings.extract_energy_geometry()

    assert true_axis.name == "energy_true"
    assert true_axis.nbin == 100
    assert true_axis.edges[0].to_value("TeV") == pytest.approx(0.003)
    assert reco_axis.name == "energy"
    assert reco_axis.nbin == 40
    assert reco_axis.edges[-1].to_value("TeV") == pytest.approx(10)


def test_reading_information(spectral_config, fermi_config_dict, write_config):
    settings = SpectralConfigSetting(spectral_config)
    settings.set_all()
    # Fixed theta cut
    assert settings.extract_detailed_reading_info() == (False, 0.2, [0, 50])

    config = copy.deepcopy(settings.conf)
    config["reader"]["energy_dependent_theta"] = True
    settings = SpectralConfigSetting(write_config(config, "edep.yaml"))
    settings.set_all()
    assert settings.extract_detailed_reading_info() == (True, None, [0, 50])


def test_plot_arguments():
    assert get_kwargs_points(color="red") == {
        "sed_type": "e2dnde",
        "label": "Spectral points",
        "color": "red",
    }

    line = get_kwargs_line(energy_bounds=[0.02 * u.TeV, 1 * u.TeV], label="Model")
    assert line["label"] == "Model"
    assert line["yunits"] == u.Unit("erg cm-2 s-1")
    assert line["energy_bounds"][1] == 1 * u.TeV

    region = get_kwargs_region(alpha=0.3, facecolor="green")
    assert region["alpha"] == 0.3
    assert region["facecolor"] == "green"


def test_default_model():
    spectral_model, model = set_model_to_fit(model_name="Crab")
    assert isinstance(spectral_model, PowerLawSpectralModel)
    assert model.name == "Crab"
    assert model.spectral_model.reference.quantity == 0.15 * u.TeV


def test_user_model():
    reference = ExpCutoffPowerLawSpectralModel(index=2.0)
    spectral_model, model = set_model_to_fit(spectral_model=reference, model_name="PSR")
    assert spectral_model is reference
    assert model.spectral_model is reference


def test_unknown_model():
    with pytest.raises(ValueError, match="not available"):
        set_model_to_fit(predefined_model="LogParabola")


def test_makers():
    dataset_maker, background_maker, safe_mask_maker = set_makers(
        (0.983, 0.026), (0.52, 0.87)
    )
    assert set(dataset_maker.selection) == {"counts", "exposure", "edisp"}
    assert background_maker.off_phase == [(0.52, 0.87)]
    assert safe_mask_maker.methods == {"aeff-max"}


@pytest.mark.parametrize("stacked", [True, False])
def test_fit_and_flux_points(simulated_dataset, stacked):
    dataset, geom = simulated_dataset
    dataset = dataset.copy(name="Crab-dataset")
    _, model = set_model_to_fit(model_name="Crab")

    datasets = dataset if stacked else Datasets([dataset])
    datasets, model_best, fit, result = do_fitting(
        datasets,
        model,
        geom,
        emin_fit=20 * u.GeV,
        emax_fit=2 * u.TeV,
        stacked=stacked,
    )

    assert result.success
    parameters = model_best.spectral_model.parameters
    assert parameters["index"].value == pytest.approx(
        TRUE_INDEX, abs=4 * parameters["index"].error
    )
    assert parameters["amplitude"].quantity.to_value(
        TRUE_AMPLITUDE.unit
    ) == pytest.approx(TRUE_AMPLITUDE.value, abs=4 * parameters["amplitude"].error)
    assert parameters["index"].error < 0.1

    flux_points, flux_points_dataset = compute_spectral_points(
        datasets, model_best, 20 * u.GeV, 700 * u.GeV, npoints=5, min_ts=2, name="Crab"
    )
    table = flux_points.to_table(sed_type="e2dnde")
    assert len(table) == 5
    assert np.all(table["sqrt_ts"] > 2)
    assert not np.any(table["is_ul"])
    np.testing.assert_allclose(
        table["e_min"].quantity.to_value("GeV")[0], 20, rtol=0.15
    )

    # The flux points are compatible with the true spectrum: E^2 dN/dE ~ E^-1
    true_model = PowerLawSpectralModel(
        index=TRUE_INDEX, amplitude=TRUE_AMPLITUDE, reference=0.15 * u.TeV
    )
    energy = table["e_ref"].quantity
    expected = (true_model(energy) * energy**2).to_value(table["e2dnde"].unit)
    pull = (table["e2dnde"] - expected) / table["e2dnde_err"]
    assert np.all(np.abs(pull) < 4)

    assert len(flux_points_dataset.data.to_table()) == 5


def test_flux_points_per_decade(simulated_dataset):
    dataset, geom = simulated_dataset
    dataset = dataset.copy(name="Crab-dataset")
    _, model = set_model_to_fit(model_name="Crab")
    dataset, model_best, _, _ = do_fitting(
        dataset, model, geom, emin_fit=20 * u.GeV, emax_fit=2 * u.TeV, stacked=True
    )

    flux_points, _ = compute_spectral_points(
        dataset,
        model_best,
        10 * u.GeV,
        1 * u.TeV,
        npoints=4,
        npoints_in_decade=True,
        name="Crab",
    )
    # 4 points per decade in two decades of energy
    assert len(flux_points.to_table()) == 8


def test_fit_without_energy_range(simulated_dataset):
    """Regression test: the fit failed if no energy range was given."""
    dataset, geom = simulated_dataset
    dataset = dataset.copy(name="Crab-dataset")
    _, model = set_model_to_fit(model_name="Crab")

    dataset, model_best, _, result = do_fitting(dataset, model, geom, stacked=True)
    assert result.success
    assert dataset.mask_fit is None
    assert model_best.spectral_model.index.value == pytest.approx(TRUE_INDEX, abs=0.2)


def test_upper_limits(simulated_dataset):
    """Points below the significance threshold are flagged as upper limits."""
    dataset, geom = simulated_dataset
    dataset = dataset.copy(name="Crab-dataset")
    _, model = set_model_to_fit(model_name="Crab")
    dataset, model_best, _, _ = do_fitting(
        dataset, model, geom, emin_fit=20 * u.GeV, emax_fit=2 * u.TeV, stacked=True
    )

    flux_points, _ = compute_spectral_points(
        dataset, model_best, 20 * u.GeV, 700 * u.GeV, npoints=5, min_ts=1e4, name="Crab"
    )
    assert np.all(flux_points.to_table()["is_ul"])


def test_read_dl4_files(simulated_dataset, tmp_path):
    dataset, _ = simulated_dataset
    for obs_id in (1, 2):
        dataset.copy(name=str(obs_id)).write(
            tmp_path / f"obs_{obs_id}.fits.gz", overwrite=True
        )

    datasets = read_DL4_files(str(tmp_path), [1, 2])
    assert len(datasets) == 2
    np.testing.assert_allclose(datasets[0].counts.data, dataset.counts.data)

    np.testing.assert_allclose(datasets[1].counts_off.data, dataset.counts_off.data)
    assert datasets.names == ["1", "2"]

    # The directory can be given with or without the final slash
    assert len(read_DL4_files(str(tmp_path) + "/", [2])) == 1


@pytest.fixture
def spectral_analysis(spectral_config, simulated_dataset):
    """Analysis object with the results of the fit of the simulated dataset."""
    dataset, geom = simulated_dataset
    dataset = dataset.copy(name="Crab-dataset")

    analysis = SpectralPulsarAnalysis(spectral_config)
    analysis.spectral_model, analysis.model = analysis.set_reference_model()
    analysis.geom = geom
    (
        analysis.datasets,
        analysis.model_best,
        analysis.fit_object,
        analysis.fitting_result,
    ) = do_fitting(
        dataset,
        analysis.model,
        geom,
        emin_fit=analysis.config_params.e_min_fitting,
        emax_fit=analysis.config_params.e_max_fitting,
        stacked=True,
    )
    analysis.flux_points, analysis.flux_points_dataset = compute_spectral_points(
        analysis.datasets,
        analysis.model_best,
        analysis.config_params.e_min_points,
        analysis.config_params.e_max_points,
        analysis.config_params.npoints,
        name="Crab",
    )
    return analysis


def test_spectral_analysis_settings(spectral_analysis):
    assert spectral_analysis.config_params.target_info["name"] == "Crab"
    assert isinstance(spectral_analysis.spectral_model, PowerLawSpectralModel)
    assert spectral_analysis.model.name == "Crab"
    assert SpectralPulsarAnalysis().config_params is None

    with pytest.raises(ValueError, match="No config parameters"):
        SpectralPulsarAnalysis().run()


def test_spectral_analysis_reference_model(spectral_config):
    reference = ExpCutoffPowerLawSpectralModel(index=2.0)
    analysis = SpectralPulsarAnalysis(spectral_config, ref_model=reference)
    spectral_model, model = analysis.set_reference_model()

    assert spectral_model is reference
    assert model.spectral_model is reference


def test_spectral_analysis_results(spectral_analysis, tmp_path):
    table = spectral_analysis.get_flux_points()
    assert len(table) == 7
    assert "e2dnde" in table.colnames

    parameters = spectral_analysis.show_fitting_results()
    assert list(parameters["name"]) == ["index", "amplitude", "reference"]

    names, _ = spectral_analysis.extract_parameters()
    assert names == ["index", "amplitude"]


def test_spectral_analysis_plots(spectral_analysis, tmp_path):
    """The plots are produced and saved in the output directory."""
    spectral_analysis.plot_SED_residuals()
    spectral_analysis.plot_fp_likelihood()
    spectral_analysis.get_covariance_matrix()
    spectral_analysis.create_contour_lines_params()

    for name in [
        "SED.png",
        "likelihood_profile.png",
        "correlation.png",
        "contour_line_index_amplitude.png",
    ]:
        output = tmp_path / "results" / name
        assert output.exists(), name
        assert output.stat().st_size > 5000
