"""
Reading of LST-1 DL2 files produced by lstchain or by CTLearn.

lstchain DL2 files store one table per telescope event (``dl2_params_lstcam_key``)
with all the parameters needed for the analysis. CTLearn DL2 files follow the
ctapipe data model: the event times, telescope pointing and DL1 parameters are
stored in separate DL1 tables and the predictions of the networks in
``/dl2/event/subarray/{classification,energy,geometry}/<prefix>``.

The functions of this module give a common (lstchain-like) view of both formats,
so that the rest of the package does not need to know which one is used.
"""

import logging

import astropy.units as u
import numpy as np
import pandas as pd
import tables
from astropy.time import Time
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.reco.utils import add_delta_t_key

logger = logging.getLogger(__name__)

__all__ = [
    "CTLEARN",
    "LSTCHAIN",
    "get_dl2_format",
    "read_ctlearn_dl2",
    "read_dl2_directions",
    "read_dl2_times",
]

#: Name of the supported DL2 formats
LSTCHAIN = "lstchain"
CTLEARN = "ctlearn"

#: Keys identifying an event in a run
EVENT_KEYS = ["obs_id", "event_id"]

CTLEARN_TRIGGER_TABLE = "/dl1/event/subarray/trigger"
CTLEARN_POINTING_TABLE = "/dl1/monitoring/telescope/pointing/tel_{tel_id:03d}"
CTLEARN_PARAMETERS_TABLE = "/dl1/event/telescope/parameters/tel_{tel_id:03d}"
CTLEARN_TASKS = ("classification", "energy", "geometry")
#: Groups with the predictions of each task. The classification group was
#: renamed to particle_type in ctapipe 0.27
CTLEARN_DL2_GROUPS = {
    "classification": (
        "/dl2/event/subarray/classification",
        "/dl2/event/subarray/particle_type",
    ),
    "energy": ("/dl2/event/subarray/energy",),
    "geometry": ("/dl2/event/subarray/geometry",),
}


def get_dl2_format(path):
    """
    Format of a DL2 file: :data:`LSTCHAIN` or :data:`CTLEARN`.

    Raises
    ------
    ValueError
        If the file does not look like an lstchain or a CTLearn DL2 file.
    """
    with tables.open_file(str(path), mode="r") as h5file:
        if f"/{dl2_params_lstcam_key}" in h5file:
            return LSTCHAIN
        if any(
            group in h5file
            for groups in CTLEARN_DL2_GROUPS.values()
            for group in groups
        ):
            return CTLEARN

    raise ValueError(
        f"{path} is neither an lstchain DL2 file (no '{dl2_params_lstcam_key}' table) "
        "nor a CTLearn DL2 file (no '/dl2/event/subarray' tables)"
    )


def _find_ctlearn_table(h5file, task, prefix=None):
    """
    Table with the CTLearn predictions of ``task`` (classification, energy or
    geometry), as ``(path in the file, name)``. The name is also the prefix of
    the columns and depends on the version and configuration of CTLearn (e.g.
    ``CTLearn`` or ``CTLearnClassifier``), so it is found from the file if not
    given.

    Returns None if the file has no predictions for that task.
    """
    available = {
        node._v_name: node._v_pathname
        for group in CTLEARN_DL2_GROUPS[task]
        if group in h5file
        for node in h5file.list_nodes(group)
    }

    if prefix is not None:
        if prefix not in available:
            raise ValueError(
                f"No CTLearn {task} table called '{prefix}' in {h5file.filename}. "
                f"Available: {sorted(available)}"
            )
        return available[prefix], prefix

    if len(available) == 0:
        return None
    if len(available) > 1:
        raise ValueError(
            f"Several {task} tables found in {h5file.filename}: "
            f"{sorted(available.values())}. Choose one with the 'prefixes' argument."
        )
    name = next(iter(available))
    return available[name], name


def _read_table(path, key):
    """Read a table written by ctapipe as an astropy Table (times as Time objects)."""
    # Imported here: ctapipe is only needed to read CTLearn files
    from ctapipe.io import read_table

    return read_table(str(path), key)


def _check_same_events(table, reference, name):
    for key in EVENT_KEYS:
        if not np.array_equal(np.asarray(table[key]), np.asarray(reference[key])):
            raise ValueError(
                f"The events of the {name} table are not in the same order as in "
                "the trigger table"
            )


def read_ctlearn_dl2(path, prefixes=None, tel_id=1):
    """
    Read a CTLearn DL2 file of LST-1 into a dataframe with lstchain-like columns.

    All the events of the file are returned (also the non-cosmic ones and those
    without a valid CTLearn prediction), in the order of the trigger table, so
    that ``delta_t`` is the time elapsed since the previous *triggered* event, as
    needed to compute the effective time.

    Parameters
    ----------
    path : str or pathlib.Path
        CTLearn DL2 file.
    prefixes : dict, optional
        Name of the CTLearn tables for each task, e.g.
        ``{"classification": "CTLearn", "energy": "CTLearn", "geometry": "CTLearn"}``.
        The tasks not given are found from the file (there must be only one
        table per task).
    tel_id : int
        Telescope id of LST-1 in the file.

    Returns
    -------
    pandas.DataFrame
        Columns ``obs_id``, ``event_id``, ``event_type``, ``dragon_time`` (unix,
        UTC), ``mjd_time`` (MJD, UTC), ``delta_t``, ``alt_tel``, ``az_tel`` (rad),
        ``intensity``, ``gammaness``, ``reco_energy`` (TeV), ``reco_alt``,
        ``reco_az`` (rad) and ``ctlearn_is_valid``. If the file has no
        classification or energy predictions their columns are filled with
        NaN; without direction predictions there are no ``reco_alt`` and
        ``reco_az`` columns.
    """
    prefixes = dict(prefixes or {})
    unknown = set(prefixes) - set(CTLEARN_TASKS)
    if unknown:
        raise ValueError(f"Unknown CTLearn tasks {unknown}. Valid: {CTLEARN_TASKS}")

    with tables.open_file(str(path), mode="r") as h5file:
        found = {
            task: _find_ctlearn_table(h5file, task, prefixes.get(task))
            for task in CTLEARN_TASKS
        }
        pointing_key = CTLEARN_POINTING_TABLE.format(tel_id=tel_id)
        parameters_key = CTLEARN_PARAMETERS_TABLE.format(tel_id=tel_id)
        has_pointing = pointing_key in h5file
        has_parameters = parameters_key in h5file

    logger.info(
        "Reading CTLearn DL2 file %s (tables: %s)",
        path,
        {task: table and table[0] for task, table in found.items()},
    )

    # Event times
    trigger = _read_table(path, CTLEARN_TRIGGER_TABLE)
    time = trigger["time"]
    events = pd.DataFrame(
        {
            "obs_id": np.asarray(trigger["obs_id"]),
            "event_id": np.asarray(trigger["event_id"]),
            "event_type": np.asarray(trigger["event_type"]),
            # astropy takes care of the TAI -> UTC conversion. float64 as in
            # lstchain (enough for unix times, ~0.2 us); MJD with long precision
            "dragon_time": time.to_value("unix").astype(np.float64),
            "mjd_time": time.utc.to_value("mjd", "long"),
        }
    )
    if np.any(np.diff(events["dragon_time"].to_numpy()) < 0):
        logger.warning("The events of %s are not sorted in time", path)
    events = add_delta_t_key(events)

    # Telescope pointing: one row per event, in the order of the trigger table
    if has_pointing:
        pointing = _read_table(path, pointing_key)
        if len(pointing) != len(events):
            raise ValueError(
                f"The pointing table of {path} has {len(pointing)} rows but there "
                f"are {len(events)} events: the pointing is expected per event"
            )
        events["alt_tel"] = u.Quantity(pointing["altitude"], u.rad).value
        events["az_tel"] = u.Quantity(pointing["azimuth"], u.rad).value
    else:
        events["alt_tel"] = np.nan
        events["az_tel"] = np.nan

    # Image intensity
    if has_parameters:
        parameters = _read_table(path, parameters_key)
        _check_same_events(parameters, trigger, "DL1 parameters")
        events["intensity"] = np.asarray(parameters["hillas_intensity"])
    else:
        events["intensity"] = np.nan

    # Predictions of the networks (sorted by event, so joined by event keys)
    columns = {
        "classification": {"prediction": "gammaness"},
        "energy": {"energy": "reco_energy"},
        "geometry": {"alt": "reco_alt", "az": "reco_az"},
    }
    is_valid = np.ones(len(events), dtype=bool)
    for task, table_info in found.items():
        if table_info is None:
            logger.warning("No CTLearn %s predictions in %s", task, path)
            # Without direction there are no reco_alt/reco_az columns, so that
            # the computation of theta2 fails with a clear message
            if task != "geometry":
                for new_name in columns[task].values():
                    events[new_name] = np.nan
            continue

        table_key, name = table_info
        table = _read_table(path, table_key)
        predictions = pd.DataFrame(
            {key: np.asarray(table[key]) for key in EVENT_KEYS}
            | {
                new_name: np.asarray(table[f"{name}_{column}"], dtype=float)
                for column, new_name in columns[task].items()
            }
            | {f"is_valid_{task}": np.asarray(table[f"{name}_is_valid"], dtype=bool)}
        )
        events = events.merge(
            predictions, on=EVENT_KEYS, how="left", validate="one_to_one"
        )
        valid_task = events.pop(f"is_valid_{task}")
        # Events without prediction (not in the table) are not valid
        is_valid &= valid_task.eq(True).to_numpy()

    # CTLearn gives the direction in degrees
    for column in ("reco_alt", "reco_az"):
        if column in events:
            events[column] = np.deg2rad(events[column])

    events["ctlearn_is_valid"] = is_valid
    return events


def read_dl2_times(path):
    """
    Event identifiers and arrival times of a DL2 file (lstchain or CTLearn).

    Returns
    -------
    pandas.DataFrame
        Columns ``obs_id``, ``event_id``, ``dragon_time`` (unix) and
        ``mjd_time`` (MJD, UTC), in the order of the events in the file.
    """
    if get_dl2_format(path) == CTLEARN:
        events = read_ctlearn_dl2(path)
    else:
        events = pd.read_hdf(path, key=dl2_params_lstcam_key)
        times = Time(events["dragon_time"].to_numpy(), format="unix", scale="utc")
        events["mjd_time"] = times.to_value("mjd", "long")

    return events[EVENT_KEYS + ["dragon_time", "mjd_time"]].reset_index(drop=True)


def read_dl2_directions(path):
    """
    Event identifiers, times and reconstructed directions of a DL2 file.

    Returns
    -------
    pandas.DataFrame
        Columns ``obs_id``, ``event_id``, ``dragon_time`` (unix), ``reco_alt``
        and ``reco_az`` (rad).
    """
    if get_dl2_format(path) == CTLEARN:
        events = read_ctlearn_dl2(path)
    else:
        events = pd.read_hdf(path, key=dl2_params_lstcam_key)

    missing = [col for col in ("reco_alt", "reco_az") if col not in events]
    if missing:
        raise ValueError(f"No reconstructed direction ({missing}) in {path}")

    columns = EVENT_KEYS + ["dragon_time", "reco_alt", "reco_az"]
    return events[columns].reset_index(drop=True)
