#####################
###Author: Alvaro Mas Aguilar (alvmas)
# mail: alvmas@ucm.es
# Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files.
###################3


import inspect
import logging
import os

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, SkyCoord
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time
from ctapipe.coordinates import CameraFrame
from lstchain.io import global_metadata, write_metadata
from lstchain.io.io import (
    dl2_params_lstcam_key,
    dl2_params_src_dep_lstcam_key,
    write_dataframe,
    write_dl2_dataframe,
)
from pint.models import (
    parameter as p,
)
from pint.models.timing_model import (
    Component,
    TimingModel,
)

try:
    from lstchain.reco.utils import location
except ImportError:  # removed in recent versions of lstchain
    from ctapipe_io_lst.constants import LST1_LOCATION as location

logger = logging.getLogger(__name__)


__all__ = [
    "add_mjd",
    "dl2time_totim",
    "merge_dl2_pulsar",
    "model_fromephem",
    "read_ephemfile",
]


def merge_dl2_pulsar(directory, run_number, output_dir, src_dep=False):
    # Read the DL2 subrun files for the given run number
    filelist = []
    for x in os.listdir(directory):
        if str(run_number) in x:
            filelist.append(os.path.join(directory, x))
    filelist.sort()

    if len(filelist) == 0:
        raise FileNotFoundError(
            f"No files found in {directory} for run number {run_number}"
        )

    # Read the Dl2 events and concatenate the dataframes
    df_list = []
    df_src_list = []
    for file in filelist:
        df_i = pd.read_hdf(file, key=dl2_params_lstcam_key)
        df_list.append(df_i)

        # Include source dependent information if it is the case
        if src_dep:
            df_i_src = pd.read_hdf(file, key=dl2_params_src_dep_lstcam_key)
            df_src_list.append(df_i_src)

    df = pd.concat(df_list)
    if src_dep:
        df_src = pd.concat(df_src_list)

    # Write the new merged dataframe into a file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        str(os.path.basename(filelist[0]).replace(".0000_pulsar.h5", ""))
        + "_pulsar.h5",
    )
    metadata = global_metadata()
    write_metadata(metadata, output_file)

    if not src_dep:
        write_dl2_dataframe(df, output_file, meta=metadata)

    else:
        write_dl2_dataframe(df, output_file, meta=metadata)
        write_dataframe(
            df_src, output_file, dl2_params_src_dep_lstcam_key, meta=metadata
        )


def add_source_info_dl2(file, source_name):
    logger.info("Adding source information in %s", file)

    coma_factor = 1.0466
    focal = coma_factor * 28 * u.m

    df1 = pd.read_hdf(file, key=dl2_params_lstcam_key)

    source_pos = SkyCoord.from_name(source_name)

    times = Time(df1["dragon_time"], format="unix", scale="utc")
    tel_pos = SkyCoord(
        alt=np.array(df1["alt_tel"]) * u.rad,
        az=np.array(df1["az_tel"]) * u.rad,
        frame=AltAz(obstime=times, location=location),
    )

    with erfa_astrom.set(ErfaAstromInterpolator(5 * u.min)):
        camera_frame = CameraFrame(
            focal_length=focal,
            telescope_pointing=tel_pos,
            location=location,
            obstime=times,
        )
        source_campos = source_pos.transform_to(camera_frame)

    src_x = source_campos.data.x.value
    src_y = source_campos.data.y.value

    reco_src_x = df1["reco_src_x"]
    reco_src_y = df1["reco_src_y"]

    logger.info("Computing theta2...")
    theta_meters = np.sqrt(
        np.power(reco_src_x - src_x, 2) + np.power(reco_src_y - src_y, 2)
    )
    theta_on = np.rad2deg(np.arctan2(theta_meters, focal.value))
    theta2 = np.power(theta_on, 2)

    theta_meters = np.sqrt(
        np.power(reco_src_x + src_x, 2) + np.power(reco_src_y + src_y, 2)
    )
    theta_off = np.rad2deg(np.arctan2(theta_meters, focal.value))
    theta2_off = np.power(theta_off, 2)

    logger.info("Writing in new table...")
    table_source = pd.DataFrame(
        {
            "src_x": list(src_x),
            "src_y": list(src_y),
            "theta2": list(theta2),
            "theta2_off": list(theta2_off),
        }
    )
    table_source.to_hdf(file, key="source_position")


def read_ephemfile(ephem):
    """
    Read the ephem file and store the values as a dataframe

    Parameters:
    -----------------
    ephem: string
    Name of the ephemeris file

    Returns:
    -----------------
    Ephem: Dataframe with the pulsar information

    """

    colnames = [
        "PSR",
        "RAJ1",
        "RAJ2",
        "RAJ3",
        "DECJ1",
        "DECJ2",
        "DECJ3",
        "START",
        "FINISH",
        "t0geo",
        "F0",
        "F1",
        "F2",
        "RMS",
        "Observatory",
        "EPHEM",
        "PSR2",
    ]
    Ephem = pd.read_csv(ephem, sep=r"\s+", names=colnames, header=None)

    return Ephem


def dl2time_totim(times, name="times.tim", obs="lst"):
    """
    Creates a .tim file from the arrival times and the observatory (by default the lst)

    Parameters:
    -----------------
    times: string
    List of arrival times

    name: string
    Name of the output .tim file

    obs: string
    Code of the observatory
    """

    with open(name, "w") as timFile:
        timFile.write("FORMAT 1 \n")
        timFile.writelines(f"{obs} 0.0 {time} 0.0 {obs} \n" for time in times)


def model_fromephem(times, ephem, model_name):
    """
    Creates a .par file using the parameters from the ephemeris associated with the arrival time range from TOAs

    Parameters:
    -----------------
    time: string
    List of arrival times

    ephem: string
    Ephemeris to be used (.txt file or similar)

    Returns:
    --------
    Name of the saved .par file

    """

    # Read the ephemeris txt file
    df_ephem = read_ephemfile(ephem)

    # Search the line of the ephemeris at which the interval time of arrivals given belongs
    first_time = float(times[0])
    valid = np.flatnonzero(
        (df_ephem["START"].to_numpy() <= first_time)
        & (first_time <= df_ephem["FINISH"].to_numpy())
    )
    if len(valid) == 0:
        raise ValueError(
            f"No ephemeris available in {ephem} for MJD {first_time}. The file covers "
            f"MJD {df_ephem['START'].min()} - {df_ephem['FINISH'].max()}"
        )
    i = int(valid[0])

    # Select componentes of the model
    all_components = Component.component_types
    selected_components = [
        "AbsPhase",
        "AstrometryEquatorial",
        "Spindown",
        "SolarSystemShapiro",
    ]
    component_instances = []

    # Initiate the component instances (see PINT pulsar documentation)
    for cp_name in selected_components:
        component_class = all_components[cp_name]  # Get the component class
        component_instance = component_class()  # Instantiate a component object
        component_instances.append(component_instance)

    tm = TimingModel("pulsar_test", component_instances)

    # Recent versions of PINT need to know how the parameters scale in the
    # TCB-TDB conversion
    scaling = {}
    no_scaling = {}
    if "tcb2tdb_scale_factor" in inspect.signature(p.floatParameter).parameters:
        scaling = {"tcb2tdb_scale_factor": u.Quantity(1)}
        no_scaling = {"convert_tcb2tdb": False}

    # Add first and second derivative of the frequency (if not in the model)
    spindown = tm.components["Spindown"]
    for name, units in [("F1", u.Hz / u.s), ("F2", u.Hz / u.s**2)]:
        if name not in spindown.params:
            spindown.add_param(
                p.prefixParameter(
                    parameter_type="float",
                    name=name,
                    value=0.0,
                    units=units,
                    longdouble=True,
                    **scaling,
                ),
                setup=True,
            )

    # Add the RMS of the timing residuals (if not in the model)
    if "TRES" not in tm.params:
        spindown.add_param(
            p.floatParameter(
                parameter_type="float",
                name="TRES",
                value=0.0 * u.ms,
                units="ms",
                longdouble=True,
                **no_scaling,
            ),
            setup=True,
        )

    # The exponents are written in Fortran notation (e.g. -3.7D-10)
    f1 = float(str(df_ephem["F1"][i]).replace("D", "E"))
    f2 = float(str(df_ephem["F2"][i]).replace("D", "E"))

    # Give values to the parameters
    params = {
        "PSR": (df_ephem["PSR"][i],),
        "RAJ": (
            str(df_ephem["RAJ1"][i])
            + ":"
            + str(df_ephem["RAJ2"][i])
            + ":"
            + str(df_ephem["RAJ3"][i]),
        ),
        "DECJ": (
            str(df_ephem["DECJ1"][i])
            + ":"
            + str(df_ephem["DECJ2"][i])
            + ":"
            + str(df_ephem["DECJ3"][i]),
        ),
        "START": (Time(df_ephem["START"][i], format="mjd", scale="tdb"),),
        "FINISH": (Time(df_ephem["FINISH"][i], format="mjd", scale="tdb"),),
        "EPHEM": (df_ephem["EPHEM"][i],),
        "PEPOCH": (Time(int(df_ephem["t0geo"][i]), format="mjd", scale="tdb"),),
        "F0": (df_ephem["F0"][i] * u.Hz,),
        "F1": (f1 * u.Hz / u.s,),
        "F2": (f2 * u.Hz / (u.s**2),),
        "TZRMJD": (Time(df_ephem["t0geo"][i], format="mjd", scale="tdb"),),
        "TZRFRQ": (0.0 * u.Hz,),
        "TZRSITE": ("coe",),
        "T2CMETHOD": ("IAU2000B",),
        "TIMEEPH": ("FB90",),
        "PLANET_SHAPIRO": (True,),
        "TRES": ((df_ephem["RMS"][i] * 0.001 / df_ephem["F0"][i] * 1000000),),
    }

    # Create the model using PINT
    for name, info in params.items():
        par = getattr(tm, name)  # Get parameter object from name
        par.quantity = info[0]  # set parameter value
        if len(info) > 1:
            if info[1] == 1:
                par.frozen = False  # Frozen means not fit.
                par.uncertainty = info[2]

    tm.validate()
    logger.info("New model generated")

    # Create the .par file
    name = model_name
    with open(name, "w") as f:
        f.write(tm.as_parfile())

    return name


def add_mjd(file_dataframe):
    times = file_dataframe.dragon_time.values
    t = Time(times, format="unix", scale="utc")
    mjd_time = t.to_value("mjd", "long")

    # Add time in MJD
    logger.info("Adding MJD time")
    file_dataframe["mjd_time"] = mjd_time.tolist()

    return mjd_time.tolist()
