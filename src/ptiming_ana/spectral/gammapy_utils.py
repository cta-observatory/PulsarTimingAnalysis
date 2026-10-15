import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import PhaseBackgroundMaker, SafeMaskMaker, SpectrumDatasetMaker
from gammapy.maps import RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
)

from ptiming_ana.phaseogram.read_events import ReadDL3File

logger = logging.getLogger(__name__)

__all__ = [
    "compute_spectral_points",
    "do_fitting",
    "execute_makers",
    "read_DL3_files",
    "read_DL4_files",
    "set_geometry",
    "set_makers",
    "set_model_to_fit",
]


def read_DL3_files(
    directory=None,
    target_radec=None,
    max_rad=0.2,
    zd_cuts=(0, 60),
    energy_dependent_theta=True,
):
    # Read the DL3 files:
    reader = ReadDL3File(
        directory=directory,
        target_radec=target_radec,
        max_rad=max_rad,
        zd_cuts=zd_cuts,
        energy_dependent_theta=energy_dependent_theta,
    )
    obs = reader.read_all_DL3file()
    ids = reader.ids

    return (obs, ids, reader)


def set_geometry(reader, true_energy_axis, reco_energy_axis):
    on_region = reader.on_region

    # Define geomtry
    geom = RegionGeom.create(region=on_region, axes=[reco_energy_axis])

    # Define empty Dataset
    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=true_energy_axis)

    return (on_region, geom, dataset_empty)


def set_makers(on_phase_range, off_phase_range):
    # Define the SpectrumDatasetMaker
    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False,
        selection=["counts", "exposure", "edisp"],
        use_region_center=True,
    )

    # Define background maker with the phase regions
    bkg_maker = PhaseBackgroundMaker(on_phase=on_phase_range, off_phase=off_phase_range)

    # Define safe mask_maker
    safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    return (dataset_maker, bkg_maker, safe_mask_maker)


def execute_makers(
    observations,
    ids,
    dataset_empty,
    dataset_maker,
    bkg_maker,
    OGIP_dir=None,
    save_DL4=False,
    name="Crab",
    safe_mask_maker=None,
    stacked=False,
):
    datasets = Datasets()

    for observation in observations:
        obs_id = observation.obs_id

        logger.info("Executing analysis for run number " + str(obs_id))
        dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)

        dataset_on_off = bkg_maker.run(dataset=dataset, observation=observation)

        if safe_mask_maker is not None:
            dataset_on_off = safe_mask_maker.run(dataset_on_off, observation)

        # Add the name of the observed source
        dataset_on_off.meta_table["SOURCE"] = name

        datasets.append(dataset_on_off)

    if save_DL4:
        logger.info("Writing DL4 files in " + str(OGIP_dir))
        ogip_path = Path(OGIP_dir)
        for d in range(len(datasets)):
            datasets[d].write(
                filename=ogip_path / f"obs_{ids[d]}.fits.gz", overwrite=True
            )

    if stacked:
        datasets = datasets.stack_reduce()
        logger.info(
            "The total significance of the dataset is "
            + str(datasets.info_dict()["sqrt_ts"])
            + " sigma"
        )

    return datasets


def read_DL4_files(DL4_directory, obs_ids, stacked=False):
    logger.info("Reading DL4 files from " + str(DL4_directory))
    datasets = Datasets()
    for obs in obs_ids:
        file = Path(DL4_directory) / f"obs_{obs}.fits.gz"
        datasets.append(SpectrumDatasetOnOff.read(file))

    if stacked:
        datasets = Datasets(datasets).stack_reduce()
        logger.info(
            "The total significance of the dataset is "
            + str(datasets.info_dict()["sqrt_ts"])
            + " sigma"
        )

    return datasets


def set_model_to_fit(spectral_model=None, predefined_model="PowerLaw", model_name=None):
    if spectral_model is not None:
        model = SkyModel(spectral_model=spectral_model, name=model_name)

    elif predefined_model == "PowerLaw":
        spectral_model = PowerLawSpectralModel(
            index=2.9,
            amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"),
            reference=0.15 * u.TeV,
        )

        model = SkyModel(spectral_model=spectral_model, name=model_name)

    else:
        raise ValueError(
            f"Predefined model '{predefined_model}' not available. Use 'PowerLaw' "
            "or provide a gammapy spectral model"
        )

    return (spectral_model, model)


def do_fitting(dataset, model, geom, emin_fit=None, emax_fit=None, stacked=False):
    # No energy range given: all the energy bins are used in the fit
    mask_fit = None

    if (emin_fit is None) and (emax_fit is not None):
        mask_fit = geom.energy_mask(0.02 * u.GeV, emax_fit)

    elif (emax_fit is None) and (emin_fit is not None):
        mask_fit = geom.energy_mask(emin_fit, 20 * u.TeV)

    elif (emax_fit is not None) and (emin_fit is not None):
        mask_fit = geom.energy_mask(emin_fit, emax_fit)

    if stacked:
        logger.info("Number of runs is one or the dataset is stacked")
        dataset.models = model
        dataset.mask_fit = mask_fit

        fit = Fit(
            optimize_opts={
                "options": {"ftol": 1e-2, "gtol": 1e-05},
            }
        )
        result = fit.run([dataset])

        # make a copy to compare later
        model_best = model.copy()

    else:
        logger.info("Dataset not stacked")
        for dataset_one in dataset:
            dataset_one.models = model
            dataset_one.mask_fit = mask_fit

        fit = Fit()
        result = fit.run(datasets=dataset)

        # dataset = Datasets(dataset).stack_reduce()
        # dataset.models = model
        model_best = model.copy()

        # logger.info('The total significance of the dataset is ' + str(dataset.info_dict()['sqrt_ts']) + ' sigma')

    return (dataset, model_best, fit, result)


def compute_spectral_points(
    dataset,
    model,
    e_min_points,
    e_max_points,
    npoints,
    npoints_in_decade=False,
    min_ts=2,
    name="Crab",
):
    if npoints_in_decade:
        e_fit_bin = int(
            round(
                (
                    np.log10(e_max_points.to("TeV").value)
                    - np.log10(e_min_points.to("TeV").value)
                )
                * npoints
                + 1,
                0,
            )
        )

        energy_edges = (
            np.logspace(
                np.log10(e_min_points.to("TeV").value),
                np.log10(e_max_points.to("TeV").value),
                e_fit_bin,
            )
            * u.TeV
        )

    else:
        energy_edges = np.geomspace(e_min_points, e_max_points, npoints + 1)

    fpe = FluxPointsEstimator(
        energy_edges=energy_edges, source=name, selection_optional="all"
    )
    flux_points = fpe.run(datasets=dataset)

    flux_points.is_ul = flux_points.sqrt_ts < min_ts
    flux_points_dataset = FluxPointsDataset(data=flux_points, models=model)

    return (flux_points, flux_points_dataset)
