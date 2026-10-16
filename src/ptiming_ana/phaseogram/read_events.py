import logging
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from gammapy.data import DataStore
from gammapy.utils.regions import CircleSkyRegion
from lstchain.io.io import (
    dl2_params_lstcam_key,
    get_srcdep_params,
)
from lstchain.reco.utils import add_delta_t_key, get_effective_time
from regions import PointSkyRegion

from ptiming_ana.io.dl2 import CTLEARN, EVENT_KEYS, get_dl2_format, read_ctlearn_dl2

logger = logging.getLogger(__name__)

__all__ = [
    "ReadDL3File",
    "ReadFermiFile",
    "ReadLSTFile",
    "ReadList",
    "ReadtxtFile",
    "compute_theta2",
]

#: Reference epoch of the LST-1 event times (used if not found in the DL3 header)
LST_EPOCH = Time("2018-10-01", scale="utc")


def _native_byteorder(array):
    """Return the array in native byte order (FITS columns are big-endian)."""
    array = np.asarray(array)
    return array.astype(array.dtype.newbyteorder("="), copy=False)


#: Value of event_type of the cosmic (physics trigger) events in lstchain
COSMIC_EVENT_TYPE = 32


def _effective_time_hours(dataframe):
    """
    Effective observation time (hours) from the distribution of time differences
    between consecutive triggered events (see ``lstchain.reco.utils.get_effective_time``).

    This estimate is only valid for unfiltered event lists (i.e. before the
    gamma/hadron separation cuts), where the events are consecutive triggers.
    """
    if "delta_t" not in dataframe:
        dataframe = add_delta_t_key(dataframe)

    events = pd.DataFrame(
        {
            "dragon_time": np.asarray(dataframe["dragon_time"], dtype=float),
            "delta_t": np.asarray(dataframe["delta_t"], dtype=float),
        }
    )
    # Recent versions of lstchain only count the events tagged as physics triggers
    if "event_type" in dataframe:
        events["event_type"] = np.asarray(dataframe["event_type"])
    else:
        events["event_type"] = COSMIC_EVENT_TYPE

    return get_effective_time(events)[1].value / 3600


def _tobs_from_gaps(mjd_times, max_gap_hours=None):
    """Observation time in hours as the sum of the time differences between events."""
    diff = np.diff(np.asarray(mjd_times, dtype=float))
    if max_gap_hours is not None:
        diff[diff > max_gap_hours / 24] = 0
    return float(np.sum(diff) * 24)


def _add_columns(events, table, columns, table_name):
    """
    Add to ``events`` the ``columns`` ({new name: name in table}) of ``table``.

    The rows are matched by event (obs_id, event_id) if both tables have them,
    otherwise both tables must have the same events in the same order.
    """
    if set(EVENT_KEYS) <= set(table.columns) and set(EVENT_KEYS) <= set(events.columns):
        matched = events[EVENT_KEYS].merge(
            table[EVENT_KEYS + list(columns.values())],
            on=EVENT_KEYS,
            how="left",
            validate="one_to_one",
        )
    elif len(table) == len(events):
        matched = table
    else:
        raise ValueError(
            f"The {table_name} table has {len(table)} rows but there are "
            f"{len(events)} events"
        )

    for new_name, name in columns.items():
        events[new_name] = matched[name].to_numpy()
    return events


def compute_theta2(reco_src_x, reco_src_y, src_x, src_y):
    coma_correction = 1.0466
    nominal_focal_length = 28 * coma_correction
    # src_x *= coma_correction
    # src_y *= coma_correction
    theta_meters = np.sqrt(
        np.power(reco_src_x - src_x, 2) + np.power(reco_src_y - src_y, 2)
    )
    theta = np.rad2deg(np.arctan2(theta_meters, nominal_focal_length))
    return np.power(theta, 2)


class ReadFermiFile:
    def __init__(self, file):
        if "fits" not in file:
            raise ValueError("No FITS file provided for Fermi-LAT data")
        else:
            self.fname = file

    def read_file(self):
        with fits.open(self.fname) as f:
            fits_table = f[1].data.copy()
        return fits_table

    def create_df_from_info(self, fits_table):
        time = _native_byteorder(fits_table["BARYCENTRIC_TIME"])
        phases = _native_byteorder(fits_table["PULSE_PHASE"])
        energies = _native_byteorder(fits_table["ENERGY"])
        dataframe = pd.DataFrame(
            {
                "mjd_time": time,
                "pulsar_phase": phases,
                "dragon_time": time * 3600 * 24,
                "energy": energies / 1e6,
            }
        )
        dataframe = dataframe.sort_values(by=["mjd_time"])
        self.info = dataframe
        return self.info

    def calculate_tobs(self):
        return _tobs_from_gaps(self.info["mjd_time"], max_gap_hours=5)

    def run(self, pulsarana=None):
        logger.info("Reading Fermi-LAT data file")
        ftable = self.read_file()
        self.create_df_from_info(ftable)
        self.tobs = self.calculate_tobs()
        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h" + "\n")


class ReadDL3File:
    def __init__(
        self,
        directory=None,
        target_radec=None,
        max_rad=0.1,
        zd_cuts=(0, 60),
        energy_dependent_theta=True,
    ):
        if directory is None:
            raise ValueError("No directory with the DL3 files provided")
        if target_radec is None:
            raise ValueError("No target coordinates (RA, Dec in deg) provided")

        self.direc = directory
        self.datastore = DataStore.from_dir(self.direc)

        self.target_radec = target_radec
        self.energydep_radmax = energy_dependent_theta
        self.info = None

        d_zen_max = [self.datastore.obs_table["ZEN_PNT"] < zd_cuts[1]]
        d_zen_min = [self.datastore.obs_table["ZEN_PNT"] > zd_cuts[0]]

        self.ids = self.datastore.obs_table[d_zen_max[0] * d_zen_min[0]]["OBS_ID"]
        self.zd_mask = d_zen_max[0] * d_zen_min[0]

        if not self.energydep_radmax:
            self.max_rad = max_rad

    def read_all_DL3file(self):
        obs = self.datastore.get_observations(self.ids, required_irf="point-like")
        # obs = self.datastore.get_observations(self.ids, required_irf=None)
        pos_target = SkyCoord(
            ra=self.target_radec[0] * u.deg,
            dec=self.target_radec[1] * u.deg,
            frame="icrs",
        )

        if not self.energydep_radmax:
            on_radius = self.max_rad * u.deg
            self.on_region = CircleSkyRegion(pos_target, on_radius)
        else:
            self.on_region = PointSkyRegion(pos_target)

        return obs

    def read_DL3file(self, obs_id):
        obs = self.datastore.get_observations([obs_id], required_irf="point-like")
        pos_target = SkyCoord(
            ra=self.target_radec[0] * u.deg,
            dec=self.target_radec[1] * u.deg,
            frame="icrs",
        )
        self.obs_object = obs

        if not self.energydep_radmax:
            on_radius = self.max_rad * u.deg
            self.on_region = CircleSkyRegion(pos_target, on_radius)
            self.events = obs[0].events.select_region(self.on_region).table
        else:
            self.on_region = PointSkyRegion(pos_target)
            self.events = (
                obs[0].events.select_rad_max(obs[0].rad_max, position=pos_target).table
            )

        info = self.create_dataframe()
        return info

    def calculate_tobs(self, mask=None):
        if mask is None:
            mask = self.zd_mask
        return self.datastore.obs_table[mask]["LIVETIME"].data.sum() / 3600

    def create_dataframe(self):
        missing = [
            col for col in ("PHASE", "BARYCENT_TIME") if col not in self.events.colnames
        ]
        if missing:
            raise ValueError(
                f"Columns {missing} not found in the DL3 file. The pulsar phases have "
                "to be computed first (see add_DL3_phase_table)."
            )

        df = self.events.to_pandas()
        df = df.sort_values("TIME")

        # Reference epoch of the event times, taken from the header if available
        meta = self.events.meta
        if "MJDREFI" in meta and "MJDREFF" in meta:
            epoch = Time(
                meta["MJDREFI"],
                meta["MJDREFF"],
                format="mjd",
                scale=str(meta.get("TIMESYS", "utc")).lower(),
            )
        else:
            epoch = LST_EPOCH
        time_orig = df["TIME"]

        time = time_orig + epoch.to_value(format="unix")
        # timelist = list(Time(time, format="unix").to_value("mjd"))

        info = pd.DataFrame(
            {
                "gammaness": df["GAMMANESS"].to_list(),
                "mjd_time": df["BARYCENT_TIME"].to_list(),
                "orig_time": time_orig.to_list(),
                "dragon_time": list(time),
                "energy": df["ENERGY"].to_list(),
                "pulsar_phase": df["PHASE"].to_list(),
            }
        )

        info = add_delta_t_key(info)

        return info

    def run(self, pulsarana=None):
        logger.info("Reading DL3 data files")
        if len(self.ids) == 0:
            raise ValueError(
                f"No observations found in {self.direc} within the zenith range"
            )

        info_list = []
        for obs_id in self.ids:
            logger.info("Reading run number " + str(obs_id))
            try:
                info_file = self.read_DL3file(obs_id)
                info_list.append(info_file)
            except AttributeError as err:
                raise ValueError("Failing when reading:" + str(obs_id)) from err

        self.info = pd.concat(info_list)
        self.tobs = self.calculate_tobs()


class ReadLSTFile:
    """
    Reader of LST-1 DL2 files with the pulsar phases (see add_DL2_phase_table).

    Both lstchain and CTLearn DL2 files can be read: the format is found from
    each file.

    Parameters
    ----------
    file : str, optional
        DL2 file.
    directory : str, optional
        Directory with DL2 files (all the .h5 files in it are read).
    src_dependent : bool
        Read the source-dependent parameters (only lstchain files).
    ctlearn_prefixes : dict, optional
        Name of the CTLearn tables for each task (see
        `ptiming_ana.io.read_ctlearn_dl2`). By default they are found from
        the files.
    """

    def __init__(
        self, file=None, directory=None, src_dependent=False, ctlearn_prefixes=None
    ):
        if file is None and directory is None:
            raise ValueError("No file provided")
        elif file is not None and directory is not None:
            raise ValueError("Can only provide file or directory, but not both")
        elif file is not None:
            if "h5" not in file:
                raise ValueError("No hdf5 file provided for LST data")
            else:
                self.fname = file

        elif directory is not None:
            self.direc = directory
            self.fname = []
            for x in os.listdir(self.direc):
                rel_dir = os.path.relpath(self.direc)
                rel_file = os.path.join(rel_dir, x)
                if x.endswith(".h5"):
                    self.fname.append(rel_file)
            self.fname.sort()

        self.info = None
        self.src_dependent = src_dependent
        self.ctlearn_prefixes = ctlearn_prefixes

    def _read_srcindep_events(self, fname):
        """All the events of a DL2 file (lstchain or CTLearn) with lstchain-like columns."""
        if get_dl2_format(fname) == CTLEARN:
            events = read_ctlearn_dl2(fname, prefixes=self.ctlearn_prefixes)
        else:
            events = pd.read_hdf(fname, key=dl2_params_lstcam_key)

        # Before any selection, so that delta_t is the time since the previous
        # triggered event (needed to compute the effective time)
        if "delta_t" not in events:
            events = add_delta_t_key(events)

        return events.reset_index(drop=True)

    @staticmethod
    def _add_phases(events, fname):
        if "pulsar_phase" not in events:
            phase_info = pd.read_hdf(fname, key="phase_info")
            events = _add_columns(
                events,
                phase_info,
                {"pulsar_phase": "pulsar_phase", "mjd_time": "mjd_barycenter_time"},
                "phase_info",
            )
        return events

    @staticmethod
    def _add_theta2(events, fname):
        if "theta2" in events:
            return events

        try:
            source_position = pd.read_hdf(fname, "source_position")
        except KeyError:
            logger.info("No theta2 computed")
            return events

        logger.info("Including theta2 column from source position table")
        if "theta2_on" in source_position:
            return _add_columns(
                events, source_position, {"theta2": "theta2_on"}, "source_position"
            )
        if "theta2" in source_position:
            return _add_columns(
                events, source_position, {"theta2": "theta2"}, "source_position"
            )

        # Old files: position of the source in the camera
        events = _add_columns(
            events,
            source_position,
            {"src_x": "src_x", "src_y": "src_y"},
            "source_position",
        )
        events["theta2"] = compute_theta2(
            np.array(events["reco_src_x"]),
            np.array(events["reco_src_y"]),
            np.array(events["src_x"]),
            np.array(events["src_y"]),
        )
        return events

    def read_LSTfile(self, fname, df_type="short", filter_data=True):
        if filter_data:
            if not self.src_dependent:
                df = self._read_srcindep_events(fname)
                df = self._add_phases(df, fname)
                df = self._add_theta2(df, fname)

            elif self.src_dependent:
                if get_dl2_format(fname) == CTLEARN:
                    raise ValueError(
                        "The source-dependent analysis is not available for CTLearn files"
                    )
                srcindep_df = self._read_srcindep_events(fname)
                on_df_srcdep = get_srcdep_params(fname, "on").reset_index(drop=True)
                srcindep_df = self._add_phases(srcindep_df, fname)

                # The source-dependent reconstruction replaces these columns
                srcindep_df = srcindep_df.drop(
                    columns=[
                        col
                        for col in on_df_srcdep.columns
                        if col in srcindep_df.columns
                    ]
                )

                df = pd.concat([srcindep_df, on_df_srcdep], axis=1)

            if "event_type" in df:
                df = df[df["event_type"] == COSMIC_EVENT_TYPE].copy()

            if df_type == "short":
                columns = [
                    "event_id",
                    "intensity",
                    "mjd_time",
                    "pulsar_phase",
                    "dragon_time",
                    "delta_t",
                    "gammaness",
                ]
                columns += [col for col in ("alpha", "theta2") if col in df]
                columns.append("alt_tel")
                columns += [
                    col for col in ("event_type", "ctlearn_is_valid") if col in df
                ]
                df_filtered = df[columns].copy()
            else:
                df_filtered = df

            if "reco_energy" in df:
                df_filtered["energy"] = df["reco_energy"]
            else:
                df_filtered["energy"] = df["energy"]

        else:
            df_filtered = pd.read_hdf(fname, key=dl2_params_lstcam_key)
        return df_filtered

    def _remove_invalid_events(self):
        """
        Remove the events without a valid CTLearn prediction. It is done after
        the computation of the effective time, which needs all the events.
        """
        if "ctlearn_is_valid" in self.info:
            self.info = self.info[self.info["ctlearn_is_valid"]].drop(
                columns="ctlearn_is_valid"
            )

    def calculate_tobs(self):
        return _effective_time_hours(self.info)

    def save_memory(self):
        for column in self.info.columns:
            if (
                column == "dragon_time"
                or column == "delta_t"
                or column == "mjd_barycenter_time"
            ):
                continue
            if (self.info[column].dtype != "float64") & (
                self.info[column].dtype != "float128"
            ):
                continue
            self.info[column] = self.info[column].astype("float32")

    def run(self, pulsarana, df_type="short"):
        filter_data = getattr(pulsarana, "filter_data", True)
        logger.info("Reading LST-1 data file")
        if isinstance(self.fname, list):
            if len(self.fname) == 0:
                raise ValueError(f"No HDF5 files found in {self.direc}")

            info_list = []
            tobs = 0
            for name in self.fname:
                logger.info("Reading file: " + name)
                try:
                    info_file = self.read_LSTfile(name, df_type, filter_data)
                    self.info = info_file
                    # The effective time has to be computed before the cuts
                    tobs += self.calculate_tobs()
                    self._remove_invalid_events()

                    if filter_data:
                        pulsarana.cuts.apply_fixed_cut(self)
                        if pulsarana.cuts.energy_binning_cut is not None:
                            pulsarana.cuts.apply_energydep_cuts(self)

                    self.save_memory()
                    info_list.append(self.info)
                except AttributeError as err:
                    raise ValueError("Failing when reading:" + str(name)) from err

            self.info = pd.DataFrame()
            while len(info_list) > 0:
                if len(info_list) >= 10:
                    chunk_df = pd.concat(info_list[0:10])
                    info_list = info_list[10:]
                else:
                    chunk_df = pd.concat(info_list)
                    info_list = []

                self.info = pd.concat([self.info, chunk_df])
            self.tobs = tobs

        else:
            self.info = self.read_LSTfile(self.fname, df_type, filter_data)
            self.tobs = self.calculate_tobs()
            self._remove_invalid_events()

            if filter_data:
                pulsarana.cuts.apply_fixed_cut(self)

                if pulsarana.cuts.energy_binning_cut is not None:
                    pulsarana.cuts.apply_energydep_cuts(self)

        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h")


class ReadtxtFile:
    """
    Reader of events stored in a text file with space-separated columns.

    Parameters
    ----------
    file : str
        Path to the text file (no header).
    format_txt : str or list of str
        Content of each column: ``t`` (MJD time), ``p`` (phase), ``e`` (energy),
        ``g`` (gammaness), ``a`` (alpha), ``t2`` (theta2) and ``at`` (telescope
        altitude). It can be given as a list (``["t", "p", "e"]``), as a
        space- or comma-separated string (``"t p e t2"``) or, if only
        single-character codes are used, as a plain string (``"tpe"``).
        Times and phases are mandatory.
    """

    _columns = {
        "t": "mjd_time",
        "p": "pulsar_phase",
        "e": "energy",
        "g": "gammaness",
        "a": "alpha",
        "t2": "theta2",
        "at": "alt_tel",
    }

    def __init__(self, file, format_txt):
        self.fname = file
        if isinstance(format_txt, str):
            if "," in format_txt or " " in format_txt:
                format_txt = format_txt.replace(",", " ").split()
            else:
                format_txt = list(format_txt)
        self.format = list(format_txt)

    def read_file(self):
        data = pd.read_csv(self.fname, sep=r"\s+", header=None)
        return data

    def check_format(self):
        for name in ["t", "p"]:
            if name not in self.format:
                raise ValueError("No valid format: times (t) and phases (p) are needed")

        unknown = [name for name in self.format if name not in self._columns]
        if unknown:
            raise ValueError(
                f"No valid format: unknown columns {unknown}. "
                f"Valid codes are {list(self._columns)}"
            )

    def create_df_from_info(self, df):
        if df.shape[1] < len(self.format):
            raise ValueError(
                f"The file has {df.shape[1]} columns but the format "
                f"describes {len(self.format)}"
            )

        dataframe = pd.DataFrame(
            {
                self._columns[code]: df.iloc[:, i].to_numpy()
                for i, code in enumerate(self.format)
            }
        )
        dataframe["dragon_time"] = dataframe["mjd_time"] * 3600 * 24
        if "energy" not in dataframe:
            dataframe["energy"] = np.nan

        dataframe = dataframe.sort_values(by=["mjd_time"])
        self.info = dataframe

    def calculate_tobs(self):
        return _tobs_from_gaps(self.info["mjd_time"])

    def run(self, pulsarana=None):
        data = self.read_file()
        self.check_format()
        self.create_df_from_info(data)
        self.tobs = self.calculate_tobs()

        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h")


class ReadList:
    """
    Reader of events given as lists of phases, times (MJD) and energies.

    Parameters
    ----------
    phases_list : array-like
        Pulsar phases.
    time_list : array-like
        Arrival times in MJD.
    energy_list : array-like, optional
        Energies of the events.
    tel : str
        Name of the telescope.
    tobs : float, optional
        Observation time in hours. If not given, it is estimated from the
        arrival times: for "fermi" as the elapsed time without the gaps longer
        than 5 hours; for any other telescope with
        ``lstchain.reco.utils.get_effective_time``, which is only valid if the
        list contains all the triggered events (i.e. no cuts were applied).
    """

    def __init__(
        self, phases_list, time_list=None, energy_list=None, tel="LST", tobs=None
    ):
        self.tobs_input = tobs
        if time_list is None:
            raise ValueError("The list of arrival times (MJD) is needed")

        self.plist = np.asarray(phases_list, dtype=float)
        self.tlist = np.asarray(time_list, dtype=float)
        if energy_list is None:
            self.elist = np.full(len(self.plist), np.nan)
        else:
            self.elist = np.asarray(energy_list, dtype=float)
        self.tel = tel

        if not (len(self.plist) == len(self.tlist) == len(self.elist)):
            raise ValueError("Phases, times and energies must have the same length")

    def create_df_from_info(self):
        dataframe = pd.DataFrame(
            {
                "mjd_time": self.tlist,
                "pulsar_phase": self.plist,
                "dragon_time": self.tlist * 3600 * 24,
                "energy": self.elist,
            }
        )
        dataframe = dataframe.sort_values(by=["mjd_time"])
        self.info = dataframe

    def calculate_tobs(self):
        if self.tobs_input is not None:
            return float(self.tobs_input)

        if str(self.tel).lower() == "fermi":
            return _tobs_from_gaps(self.info["mjd_time"], max_gap_hours=5)

        self.info = add_delta_t_key(self.info)
        try:
            return _effective_time_hours(self.info)
        except ValueError:
            logger.warning(
                "The effective time cannot be estimated from the time differences "
                "between events. Using the elapsed time (gaps longer than 1 hour "
                "excluded): provide tobs to set the observation time."
            )
            return _tobs_from_gaps(self.info["mjd_time"], max_gap_hours=1)

    def run(self, pulsarana=None):
        self.create_df_from_info()
        self.tobs = self.calculate_tobs()
        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h")
