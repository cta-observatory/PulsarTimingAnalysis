"""
Computation of the pulsar phases of LST-1 (DL2 and DL3) and Fermi-LAT events
using PINT and the ephemeris of the pulsar.

Original author: Alvaro Mas Aguilar (alvmas). Modified by Paula Molina Sanjurjo
(Pau-mol).
"""

import inspect
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from gammapy.data import DataStore, EventList
from pint import models, toa
from pint.fermi_toas import load_Fermi_TOAs
from pint.models import get_model
from pint.observatory.satellite_obs import get_satellite_observatory
from scipy.interpolate import interp1d

from ptiming_ana.cphase.utils import (
    check_ephemeris_validity,
    dl2time_totim,
    model_fromephem,
)
from ptiming_ana.io import read_dl2_times

__all__ = ["DL2_calphase", "DL3_calphase", "DL3_calphase_gammapy", "fermi_calphase"]

logger = logging.getLogger(__name__)


def _pulsar_file_name(file, output_dir, extension=".fits"):
    """Path of the output file: ``{output_dir}/{input name}_pulsar{extension}``."""
    if output_dir is None:
        raise ValueError("No output directory given to save the file with the phases")

    name = os.path.basename(str(file))
    for suffix in (".gz", ".fits", ".h5"):
        name = name.removesuffix(suffix)

    return os.path.join(str(output_dir), name + "_pulsar" + extension)


def update_fermi(timelist, ephem, t):
    model = create_files(timelist, ephem, None, "model.par")

    # Upload TOAs and model
    m = models.get_model(model)

    # Calculate the phases
    logger.info("Calculating barycentric time and absolute phase")
    bt = m.get_barycentric_toas(t)
    p = m.phase(t, abs_phase=True)

    return (bt, p)


def fermi_calphase(file, ephem, output_dir, pickle=False, ft2_file=None):
    """
    Calculates barycentered times and pulsar phases from the DL2 dile using ephemeris.

    Parameters:
    -----------------
    dl2file: string
    DL2 input file with the arrival times

    ephem: string
    Ephemeris to be used (.txt file or similar)


    output_dir:string
    Directory of the the output file

    pickle: boolean
    True if want to save a pickle file with the loaded TOAs


    """
    logger.info("Input file:" + str(file))
    # Load observatory and TOAs
    get_satellite_observatory("fermi", ft2_file, overwrite=True)
    tl = load_Fermi_TOAs(file, fermiobs="fermi")

    # Extract the timelist in MJD
    timelist = []
    for i in range(len(tl)):
        timelist.append(tl[i].mjd.value)

    # Create TOAs object
    t = toa.get_TOAs_list(tl, planets=True)

    # Calculate the phases in intervals of 1000 events so that ephemeris are updated
    barycent_toas = []
    phase = []
    k = 0
    for i in range(10000, len(t), 1000):
        b, p = update_fermi(timelist[k:i], ephem, t[k:i])
        barycent_toas = barycent_toas + list(b.value)
        phase = phase + list(p.frac.value)
        k = i

    b, p = update_fermi(timelist[k:], ephem, t[k:])
    barycent_toas = barycent_toas + list(b.value)
    phase = phase + list(p.frac.value)

    # Write the output file
    dir_output = _pulsar_file_name(file, output_dir)
    os.makedirs(os.path.dirname(dir_output) or ".", exist_ok=True)

    with fits.open(file) as hdul:
        orig_cols = hdul[1].columns
        new_col1 = fits.Column(name="BARYCENTRIC_TIME", format="D", array=barycent_toas)
        new_col2 = fits.Column(name="PULSE_PHASE", format="D", array=phase)
        hdu = fits.BinTableHDU.from_columns(
            orig_cols + new_col1 + new_col2, header=hdul[1].header
        )

        logger.info("Writing outputfile in " + str(dir_output))
        hdu.writeto(dir_output, overwrite=True)

    logger.info("Finished")


def DL3_calphase_gammapy(
    DL3_direc,
    output_dir,
    ephem,
    obs_ids=None,
    obs="lst",
    create_tim_file=False,
    pickle=False,
    overwrite=True,
):
    """
    Function that reads the DL3 files and creates new EventLists with info about the PHASE of the pulsar.

    Parameters:
    ------------------
    DL3_dir: string
    path to the DL3 directory to the DL3 observation files

    output_dir: string
    path to the directory where to store the new EventList files

    ephem: string
    path to the ephemeris file. It can be a .par file or a .gro file (for the case of Crab)

    obs_ids: list
    list of obs_ids to analyze

    obs: string
    Observatory code to give to PINT

    create_tim_file: boolean
    Whether to create a temporal tim file to read and reduce TOAs with PINT or to do it step by step with PINT functions.

    pickle: boolean
    True if want to save a pickle file with the loaded TOAs


    """

    # Read the files and create datastore
    total_datastore = DataStore.from_dir(DL3_direc)

    # Extract observations
    if obs_ids is None:
        total_obs_list = total_datastore.obs_table["OBS_ID"].data
    else:
        total_obs_list = obs_ids

    observations = total_datastore.get_observations(
        total_obs_list, required_irf="point-like"
    )

    # Extract times from EventList
    for obser in observations:
        times = obser.events.time
        timelist = list(times.to_value("mjd", "long"))

        # Get the name of the files
        if create_tim_file:
            timname = "times.tim"
        else:
            timname = None

        parname = "model.par"

        # Calculate phases
        model = create_files(timelist, ephem, timname, parname, obs=obs)
        if timname is None:
            phases = compute_phases_from_times_model(timelist, model, obs=obs)[1]
        else:
            barycent_toas, phases = get_phase_list_from_tim(timname, model, pickle)

        phases = np.asarray(phases.frac)

        # Shift phases
        phases = np.where(phases < 0.0, phases + 1.0, phases)

        # Create new EventList with the phases
        table = obser.events.table
        table["PHASE"] = phases.astype("float64")
        table.sort("TIME")

        new_event_list = EventList(table)
        obser._events = new_event_list

        # Write them in a dictionary
        filename = f"dl3_pulsar_{obser.obs_id:04d}.fits.gz"
        os.makedirs(str(output_dir), exist_ok=True)
        file_path = os.path.join(str(output_dir), filename)

        logger.info("Writing outputfile in " + str(file_path))
        obser.events.write(filename=file_path, gti=obser.gti, overwrite=overwrite)

        # Removing tim file
        if timname is not None:
            Path(timname).unlink(missing_ok=True)

        # Removing .par file if it was created during execution
        if ephem.endswith(".gro"):
            Path(parname).unlink(missing_ok=True)


def DL3_calphase(
    file,
    ephem,
    output_dir,
    create_tim_file=False,
    obs="lst",
    use_interpolation=False,
    n_interp=1000,
    pickle=False,
):
    """
    Function that reads the DL3 files, calculates the phases and create a new DL3 file. The new DL3 file will have two new columns: 'PHASE' and 'BAYCENT_TIME'.

    Parameters:
    ------------------
    file: string
    path to the DL3 file in a .fits format

    ephem: string
    path to the ephemeris file. It can be a .par file or a .gro file (for the case of Crab)

    output_dir: string
    path to the output directory where to store the modified DL3 file

    pickle: boolean
    True if want to save a pickle file with the loaded TOAs

    create_tim_file: boolean
    Set to True if want to create and save a .tim file

    use_interpolation: boolean
    Set to True if want to use the interpolation method (faster but loses some precision)

    n_interp: int
    Number of events between two interpolation points.

    Returns:
    -------------------------
    A new DL3 file with two new columns: 'PHASE' and 'BAYCENT_TIME'. The name of the file will be {filename}_pulsar.fits

    """

    # Check the output before the (long) computation of the phases
    _pulsar_file_name(file, output_dir)

    with fits.open(file) as data:
        orig_cols = data[1].columns
        events_header = data[1].header.copy()

        # Extract the times from the dataframe and transform scales
        df = Table(data[1].data).to_pandas()

    df = df.sort_values("TIME")

    lst_epoch = Time(
        events_header["MJDREFI"],
        events_header["MJDREFF"],
        format="mjd",
        scale=events_header["TIMESYS"].lower(),
    )
    time_orig = df["TIME"]

    time = time_orig + lst_epoch.to_value(format="unix")
    times = Time(time, format="unix").to_value("mjd", "long")
    timelist = list(times)

    if create_tim_file:
        timname = str(os.path.basename(file).replace(".fits", "")) + ".tim"
    else:
        timname = None

    # Get the name of the files
    parname = str(os.path.basename(file).replace(".fits", "")) + ".par"

    # Calculate phases
    if not use_interpolation:
        model = create_files(timelist, ephem, timname, parname, obs=obs)
        if timname is None:
            barycent_toas, phases = compute_phases_from_times_model(
                times, model, obs=obs
            )
        else:
            logger.info("Computing phases from tim and par files")
            barycent_toas, phases = get_phase_list_from_tim(timname, model, pickle)

        phase = np.asarray(phases.frac)
        barycent_toas = np.asarray(barycent_toas)
    else:
        logger.info("Using interpolation...")
        phase, barycent_toas = compute_phase_interpolation(
            times, ephem, timname, parname, n_interp, obs, pickle
        )

    # Shift phases
    phase = np.where(phase < 0.0, phase + 1.0, phase)

    # Generate needed columns
    cols_list = []
    for j in range(len(orig_cols)):
        try:
            cols_list.append(
                fits.Column(
                    name=orig_cols[j].name,
                    format=orig_cols[j].format,
                    unit=orig_cols[j].unit,
                    array=df[orig_cols[j].name],
                )
            )

        except AttributeError:
            cols_list.append(
                fits.Column(
                    name=orig_cols[j].name,
                    format=orig_cols[j].format,
                    array=df[orig_cols[j].name],
                )
            )

    orig_cols_sorted = fits.ColDefs(cols_list)

    # New columns to add to the DL3 files
    new_cols = fits.ColDefs(
        [
            fits.Column(name="PHASE", format="D", array=phase),
            fits.Column(name="BARYCENT_TIME", format="D", array=barycent_toas),
        ]
    )

    # Create the files
    hdu = fits.BinTableHDU.from_columns(
        orig_cols_sorted + new_cols, header=events_header
    )

    # Save new file
    save_new_DL3_file(file, hdu, output_dir)

    # Removing tim file
    if create_tim_file:
        Path(timname).unlink(missing_ok=True)

    # Removing .par file if it was created during execution
    if ephem.endswith(".gro"):
        Path(parname).unlink(missing_ok=True)


def save_new_DL3_file(orig_file, new_table, output_dir):
    """Write a copy of the DL3 file replacing the EVENTS HDU with ``new_table``."""
    output_file = _pulsar_file_name(orig_file, output_dir)

    if not os.path.exists(output_dir):
        logger.info("Creating directory: " + str(output_dir))
        os.makedirs(output_dir, exist_ok=True)

    with fits.open(orig_file) as data:
        hdu_list = fits.HDUList([data[0], new_table] + list(data[2:]))

        logger.info("Writing outputfile in " + str(output_file))
        hdu_list.writeto(output_file, overwrite=True)

    return output_file


def compute_phases_from_times_model(
    times, ephem, obs="lst", include_planets=True, include_bipm=True, include_gps=True
):
    # Read model
    model = get_model(ephem)

    # Get TOA list
    toa_list = list(
        toa.TOA(
            t,
            0,
            obs=obs,
            ephem="DE421",
            include_bipm=str(include_bipm),
            include_gps=str(include_gps),
        )
        for t in times
    )
    # Load TOAs. The GPS clock correction option was removed in recent
    # versions of PINT, where it is always applied.
    kwargs = {"ephem": "DE421", "include_bipm": include_bipm, "planets": True}
    if "include_gps" in inspect.signature(toa.get_TOAs_list).parameters:
        kwargs["include_gps"] = include_gps
    t = toa.get_TOAs_list(toa_list, **kwargs)

    # Compute phases and barycentric toas
    phases = model.phase(t, abs_phase=True)
    barycent_toas = model.get_barycentric_toas(t)

    return (barycent_toas, phases)


def create_files(timelist, ephem, timname, parname, obs="lst", check_validity=True):
    """
    Creates the .tim and .par file needed for the use of PINT.

    Parameters:
    -----------------
    timelist: list
    Times of obervation in MJD

    ephem: string
    Ephemeris to be used (.par or .gro). If given .par it will not create a new model.

    timname:string
    Name of the output timfile

    parname: string
    Name of the output parfile

    check_validity: boolean
    Raise an error if the ephemeris is not valid for the times (see
    `ptiming_ana.cphase.utils.check_ephemeris_validity`)

    """
    if check_validity:
        check_ephemeris_validity(timelist, ephem)

    if timname is not None:
        logger.info("Creating .tim file")
        dl2time_totim(timelist, name=timname, obs=obs)

    logger.info("Setting the .par file")
    ephem = str(ephem)
    if ephem.endswith(".par"):
        model = ephem

    elif ephem.endswith(".gro"):
        logger.info("No .par file given. Creating .par file from .gro file...")
        # Create model from ephemeris
        model = model_fromephem(timelist, ephem, parname)

    else:
        raise ValueError(
            f"Ephemeris file {ephem} not valid: a .par or a .gro file is needed"
        )

    return model


def DL2_calphase(
    dl2file, ephem, obs="lst", use_interpolation=False, n_interp=1000, pickle=False
):
    """
    Calculates barycentered times and pulsar phases from the DL2 dile using ephemeris.

    Parameters:
    -----------------
    dl2file: string
    DL2 input file with the arrival times (produced by lstchain or CTLearn)

    ephem: string
    Ephemeris to be used (.par or .gro file)

    use_interpolation: boolean
    True if want to use the interpolation method. False otherwise

    pickle: boolean
    True if want to save a pickle file with the loaded TOAs

    use_interpolation: boolean
    Set to True if want to use the interpolation method (faster but loses some precision)

    n_interp: int
    Number of events between two interpolation points.

    Returns:
    --------
    Returns same DL2 with a new table (key='phase_info')  with the phase information.

    """

    # Read the file (lstchain or CTLearn DL2)
    logger.info("Input file:" + str(dl2file))
    df_i = read_dl2_times(dl2file)

    # Create the .tim file
    timelist = df_i.mjd_time.tolist()

    # Name of the files
    timname = str(os.path.basename(dl2file).replace(".h5", "")) + ".tim"
    parname = str(os.path.basename(dl2file).replace(".h5", "")) + ".par"

    if not use_interpolation:
        model = create_files(timelist, ephem, timname, parname, obs=obs)
        barycent_toas, phase = get_phase_list_from_tim(timname, model, pickle)
        phase = np.asarray(phase.frac)
        barycent_toas = np.asarray(barycent_toas)
    else:
        logger.info("Interpolating...")
        phase, barycent_toas = compute_phase_interpolation(
            timelist, ephem, timname, parname, n_interp, obs, pickle
        )

    # Removing tim file
    Path(timname).unlink(missing_ok=True)

    # Removing .par file if it was created during execution
    if ephem.endswith(".gro"):
        Path(parname).unlink(missing_ok=True)

    # Create new dataframe:
    df_phase = pd.DataFrame(
        {
            "obs_id": df_i["obs_id"],
            "event_id": df_i["event_id"],
            "mjd_barycenter_time": barycent_toas,
            "pulsar_phase": phase,
        }
    )
    df_phase.to_hdf(dl2file, key="phase_info")
    logger.info("Finished")


def compute_phase_interpolation(
    timelist, ephem, timname, parname, n_interp=1000, obs="lst", pickle=False
):
    """
    Calculates barycentered times and pulsar phases using an interpolation method for LST-1

    Parameters:
    -----------------
    timelist: list
    List of times of arrivals

    ephem: string
    Ephemeris to be used (.par or .gro file)

    timname: string
    Name of the .tim file

    parname: string
    Name of the .par file if needed to create

    obs: string
    Observatory code to give to PINT

    pickle: boolean
    True if want to save a pickle file with the loaded TOAs

    """

    timelist = list(timelist)
    # Extraxting reference values of times for interpolation
    timelist_n = timelist[0::n_interp]
    if timelist_n[-1] != timelist[-1]:
        timelist_n.append(timelist[-1])

    # Time in seconds:
    # timelist_s = np.array(timelist) * 86400
    # timelist_ns = np.array(timelist_n) * 86400

    # Create the files
    model = create_files(timelist_n, ephem, timname, parname, obs=obs)

    # Calculate the barycent times and phases for reference
    if timname is None:
        barycent_toas_sample, phase_sample = compute_phases_from_times_model(
            timelist_n, model, obs=obs
        )
    else:
        barycent_toas_sample, phase_sample = get_phase_list_from_tim(
            timname, model, pickle
        )

    # Time of barycent_toas in seconds:
    btime_sample_sec = np.array(barycent_toas_sample) * 86400

    # Getting the period:
    m = models.get_model(model)
    P = 1 / m["F0"].value
    logger.info("The period used for the interpolation is:" + str(P))

    # Number of complete cicles(N):
    phase_sam = np.array(phase_sample.frac) + 0.5
    N = (1 / P) * (np.diff(btime_sample_sec) - P * (1 + np.diff(phase_sam)))
    N = np.round(N)

    # For having the same dimensions:
    N = np.append([0], N)

    # The cumulative sum of N:
    sN = np.cumsum(N)

    # Sum of phases:
    sp = np.cumsum(phase_sam)
    # Sum of complementary phases shifted by 1:
    spc = np.append([0], np.cumsum(1 - phase_sam)[:-1])

    # Adding sN + sp+ spc:
    phase_s = sp + sN + spc

    # Interpolate to all values of times:
    barycent_toas = interpolate_btoas(timelist, timelist_n, barycent_toas_sample)
    barycent_toas_sec = np.array(barycent_toas) * 86400
    phase = interpolate_phase(barycent_toas_sec, btime_sample_sec, phase_s)
    phase = phase % 1
    phase = phase - 0.5

    return (phase, barycent_toas)


def interpolate_phase(timelist, time_sample, phase_s):
    logger.info("Interpolating phase...")
    interp_function_phase = interp1d(time_sample, phase_s, fill_value="extrapolate")
    phase = interp_function_phase(timelist)

    return phase


def interpolate_btoas(timelist, time_sample, barycent_toas_sample):
    logger.info("Interpolating btoas...")
    interp_function_mjd = interp1d(
        time_sample, barycent_toas_sample, fill_value="extrapolate"
    )
    barycent_toas = interp_function_mjd(timelist)

    return barycent_toas


def get_phase_list(timname, timelist, ephem, parname, obs="lst", pickle=False):
    logger.info("Creating tim file")
    dl2time_totim(timelist, name=timname, obs=obs)
    logger.info("creating TOA list")

    # Create model from ephemeris
    model = model_fromephem(timelist, ephem, parname)

    # Upload TOAs and model
    m, t = models.get_model_and_toas(model, timname, planets=True, usepickle=pickle)

    # t.compute_posvels(model,planets=True)
    # Calculate the phases
    logger.info("Calculating barycentric time and absolute phase")
    barycent_toas = m.get_barycentric_toas(t)
    phase = m.phase(t, abs_phase=True)

    Path(timname).unlink(missing_ok=True)

    return (barycent_toas, phase)


def get_phase_list_from_tim(timname, model, pickle=False):
    logger.info("creating TOA list")
    # Upload TOAs and model
    m, t = models.get_model_and_toas(model, timname, planets=True, usepickle=pickle)

    logger.debug("%s", m)
    # Calculate the phases
    logger.info("Calculating barycentric time and absolute phase")
    barycent_toas = m.get_barycentric_toas(t)
    phase = m.phase(t, abs_phase=True)

    return (barycent_toas, phase)
