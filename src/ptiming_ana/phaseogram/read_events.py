import pandas as pd
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from lstchain.reco.utils import get_effective_time, add_delta_t_key
from lstchain.io.io import (
    dl2_params_lstcam_key,
    get_srcdep_params,
)
import os
from ctapipe.io import TableLoader
from ctapipe.io import read_table
from ctapipe.time import ctao_high_res_to_time
from gammapy.data import DataStore
from gammapy.utils.regions import CircleSkyRegion
from astropy.coordinates import SkyCoord
import logging
import glob
from regions import PointSkyRegion
from gammapy.irf import RadMax2D
import h5py
logger = logging.getLogger(__name__)

logging.getLogger("gammapy").disabled = True

__all__ = [
    "compute_theta2",
    "ReadFermiFile",
    "ReadDL3File",
    "ReadLSTFile",
    "ReadtxtFile",
    "ReadList"
]


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
        f = fits.open(self.fname)
        fits_table = f[1].data
        return fits_table

    def create_df_from_info(self, fits_table):
        time = fits_table["BARYCENTRIC_TIME"].byteswap().newbyteorder()
        phases = fits_table["PULSE_PHASE"].byteswap().newbyteorder()
        energies = fits_table["ENERGY"].byteswap().newbyteorder()
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
        diff = np.array(self.info["mjd_time"].to_list()[1:]) - np.array(
            self.info["mjd_time"].to_list()[0:-1]
        )
        diff[diff > 5 / 24] = 0
        return sum(diff) * 24

    def run(self):
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
        zd_cuts=[0, 60],
        energy_dependent_theta=True,
    ):
        if directory is not None:
            self.direc = directory
            self.datastore = DataStore.from_dir(self.direc)

        self.target_radec = target_radec
        self.energydep_radmax = energy_dependent_theta
        self.info = None

        self._default_filter_h5 = "/top1/jbuces/pulsars/performance_crab/ctlearn_filter.h5"

        d_zen_max = [self.datastore.obs_table["ZEN_PNT"] < zd_cuts[1]]
        d_zen_min = [self.datastore.obs_table["ZEN_PNT"] > zd_cuts[0]]

        self.ids = self.datastore.obs_table[d_zen_max[0] * d_zen_min[0]]["OBS_ID"]
        self.zd_mask = d_zen_max[0] * d_zen_min[0]

        if not self.energydep_radmax:
            self.max_rad = max_rad

    def _load_valid_ids(self, obs_id):
        """
        Devuelve un set con los event_id que tienen in_both == 1
        para este obs_id. Si no hay HDF5 configurado, devuelve None
        (sin filtro).
        """
        h5_path = self._default_filter_h5
        key = f"run_{obs_id:05d}"
        try:
            df = pd.read_hdf(h5_path, key=key)
            valid = set(df.loc[df["in_both"] == 1, "event_id"].values)
            logger.info(f"  Filtro cruzado run {obs_id}: {len(valid)} eventos válidos")
            return valid
        except KeyError:
            logger.warning(f"  Run {obs_id} no encontrado en {h5_path}, se omite filtro cruzado")
            return None
        
    def _apply_cross_filter(self, obs_id):
        valid_ids = self._load_valid_ids(obs_id)
        if valid_ids is None:
            return
        n_before = len(self.events)
        mask = np.isin(self.events["EVENT_ID"], list(valid_ids))
        self.events = self.events[mask]
        n_after = len(self.events)
        logger.info(f"  Filtro cruzado: {n_before} → {n_after} eventos ({n_before - n_after} eliminados)")

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
            fits_files = glob.glob(os.path.join(self.direc, f"*Run{obs_id:05d}*.fits"))
            if len(fits_files) == 0:
                raise FileNotFoundError(f"No FITS file found for obs_id {obs_id} in {self.direc}")
            rad_max_irf = RadMax2D.read(fits_files[0], hdu="RAD_MAX")
            self.events = obs[0].events.select_rad_max(rad_max_irf, position=pos_target).table

        # self._apply_cross_filter(obs_id)
        info = self.create_dataframe()
        return info

    def calculate_tobs(self, mask=None):
        if mask is None:
            mask = self.zd_mask
        tobs = self.datastore.obs_table[mask]["LIVETIME"].data.sum() / 3600
        logger.info(f"Total livetime: {tobs:.3f} h")
        return tobs

    def create_dataframe(self):
        df = self.events.to_pandas()
        df = df.sort_values("TIME")

        lst = Time("2018-10-01", scale="utc")
        time_orig = df["TIME"]

        time = time_orig + lst.to_value(format="unix")
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

    def run(self, pulsarana):
        logger.info("Reading DL3 data files")
        info_list = []
        logger.info(f"Runs DL3 to process: {len(self.ids)}")
        for obs_id in self.ids:
            logger.info("Reading run number " + str(obs_id))
            try:
                info_file = self.read_DL3file(obs_id)
                info_list.append(info_file)
            except AttributeError:
                raise ValueError("Failing when reading:" + str(obs_id))

        self.info = pd.concat(info_list)
        self.tobs = self.calculate_tobs()


class ReadLSTFile:
    def __init__(self, file=None, directory=None, src_dependent=False, ctlearn=False):
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
                if x.startswith("dl2") and x.endswith(".h5"):
                    self.fname.append(rel_file)
                    self.fname.sort()

        self.info = None
        self.src_dependent = src_dependent
        self.ctlearn = ctlearn

    def add_phases(self, pname):
        dphase = pd.read_hdf(pname, key=dl2_params_lstcam_key)
        self.info["pulsar_phase"] = dphase["pulsar_phase"]

    def read_LSTfile(self, fname, df_type="short", filter_data=True):
        if filter_data:
            if not self.src_dependent:
                if self.ctlearn:
                    with h5py.File(fname, "r") as f:

                        # --- CTLearn info ---
                        ct = f["dl2/event/subarray/classification/CTLearn"][:]
                        ce = f["dl2/event/subarray/energy/CTLearn"][:]

                        # Solo eventos válidos según CTLearn
                        ctlearn_mask = ct["CTLearn_is_valid"] == 1

                        # --- Dragon time (dl1 trigger) ---
                        trigger = f["dl1/event/subarray/trigger"][:]
                        event_id = trigger["event_id"][ctlearn_mask]
                        event_type = trigger["event_type"][ctlearn_mask]
                        t_sec_all = trigger["time"][:, 0]
                        t_ns_all  = trigger["time"][:, 1]
                        dragon_time = ctao_high_res_to_time(t_sec_all, t_ns_all)
                        dragon_time = dragon_time.utc.unix
                        logger.info("Adding delta key")
                        delta_t = np.insert(np.diff(dragon_time), 0, 0)
                        dragon_time = dragon_time[ctlearn_mask]
                        delta_t = delta_t[ctlearn_mask]
                        # --- Hillas intensity (tel_001) ---
                        tel = f["dl1/event/telescope/parameters/tel_001"][:]
                        intensity = tel["hillas_intensity"][ctlearn_mask]

                        # --- Pointing altitud/azimuth ---
                        pointing = f["dl1/monitoring/telescope/pointing/tel_001"][:]
                        alt_tel = pointing["altitude"][ctlearn_mask]

                        # --- CTLearn outputs ---
                        gammaness = ct["CTLearn_prediction"][ctlearn_mask]
                        reco_energy = ce["CTLearn_energy"][ctlearn_mask]


                        # Construir DataFrame final
                        df_or = pd.DataFrame({
                            "event_id": event_id,
                            "intensity": intensity,
                            "event_type": event_type,
                            "dragon_time": dragon_time,
                            "gammaness": gammaness,
                            "alt_tel": alt_tel,
                            "reco_energy": reco_energy,
                            "delta_t": delta_t
                        })
                        
                else:
                    df_or = pd.read_hdf(fname, key=dl2_params_lstcam_key)
                    logger.info("Adding delta key")
                    df_or = add_delta_t_key(df_or)
                if "pulsar_phase" not in df_or.columns:
                    df_pulsar = pd.read_hdf(fname, key="phase_info")
                    if self.ctlearn:
                        df_pulsar = df_pulsar[ctlearn_mask].reset_index(drop=True) 
                    df_or["pulsar_phase"] = df_pulsar["pulsar_phase"].values
                    df_or["mjd_time"] = df_pulsar["mjd_barycenter_time"].values
                    logger.info("Including phase column from phase position table")
                    del df_pulsar
                if "event_type" in df_or.columns:
                    df = df_or[df_or["event_type"] == 32]
                    if "theta2" not in df.columns:
                        try:
                            df_pos = pd.read_hdf(fname, "source_position")
                            if self.ctlearn:
                                # Apply ctlearn_mask FIRST to get same length as df_or
                                df_pos = df_pos[ctlearn_mask].reset_index(drop=True)
                            # Now both df_pos and df_or have same length, apply event_type filter
                            event_type_mask = (df_or["event_type"] == 32).values
                            df_pos = df_pos[event_type_mask].reset_index(drop=True)
                            df = df.reset_index(drop=True)

                            # SAFE ASSIGNMENT: Verify lengths match before assigning
                            if len(df) != len(df_pos):
                                logger.error(f"[theta2 ERROR] Length mismatch! df={len(df)}, df_pos={len(df_pos)} after filtering")
                                logger.error(f"  This indicates index alignment failure between df and df_pos")
                                raise ValueError("theta2 assignment failed: dataframe length mismatch")
                            
                            # Extract theta2 values to avoid pandas index alignment issues
                            theta2_values = df_pos["theta2"].values if "theta2" in df_pos.columns else df_pos["theta2_on"].values
                            df["theta2"] = theta2_values
                        except AttributeError:
                            logger.info("No theta2 computed")
                            del df_pos

                    else:
                        df = df_or

            elif self.src_dependent:
                srcindep_df = pd.read_hdf(
                    fname, key=dl2_params_lstcam_key, float_precision=20
                )
                logger.info("Adding delta key")
                srcindep_df = add_delta_t_key(srcindep_df)
                on_df_srcdep = get_srcdep_params(fname, "on")

                if "pulsar_phase" not in srcindep_df:
                    df_pulsar = pd.read_hdf(fname, key="phase_info")
                    srcindep_df["pulsar_phase"] = df_pulsar["pulsar_phase"]
                    srcindep_df["mjd_time"] = df_pulsar["mjd_barycenter_time"]

                if "reco_energy" in srcindep_df.keys():
                    srcindep_df.drop(["reco_energy"])

                if "gammaness" in srcindep_df.keys():
                    srcindep_df.drop(["gammaness"])

                df = pd.concat([srcindep_df, on_df_srcdep], axis=1)
                df = df[df.event_type == 32]

            if df_type == "short":
                if self.ctlearn:
                    logger.info("Filtering columns")
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "event_type",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "theta2",
                            "alt_tel",
                            "delta_t",
                        ]
                    ]
                elif "alpha" in df and "theta2" in df:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "event_type",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "alpha",
                            "theta2",
                            "alt_tel",
                            "delta_t",
                        ]
                    ]
                elif "alpha" in df and "theta2" not in df:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "event_type",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "alpha",
                            "alt_tel",
                            "delta_t",
                        ]
                    ]
                elif "theta2" in df and "alpha" not in df:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "r",
                            "mjd_time",
                            "event_type",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "theta2",
                            "alt_tel",
                            "delta_t",
                        ]
                    ]
                else:
                    df_filtered = df[
                        [
                            "event_id",
                            "intensity",
                            "mjd_time",
                            "event_type",
                            "pulsar_phase",
                            "dragon_time",
                            "gammaness",
                            "alt_tel",
                            "delta_t",
                        ]
                    ]

                try:
                    df_filtered["energy"] = df["reco_energy"]
                except AttributeError:
                    df_filtered["energy"] = df["energy"]
            else:
                df_filtered = df
                df_filtered["energy"] = df["reco_energy"]
            self.tobs = self.calculate_tobs(df_filtered)
            logger.info("Finishing reading and filtering LST-1 data file")
        else:
            df_filtered = pd.read_hdf(fname, key=dl2_params_lstcam_key)
        return df_filtered

    def calculate_tobs(self, dataframe):
        # dataframe = add_delta_t_key(dataframe)
        return get_effective_time(dataframe)[1].value / 3600

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
        filter_data = pulsarana.filter_data
        even_odd_cuts = pulsarana.filter_even_odd
        logger.info("Reading LST-1 data file")
        if isinstance(self.fname, list):
            info_list = []
            total_obs = 0.0
            logger.info(f"Runs DL2 a procesar: {len(self.fname)}")
            for name in self.fname:
                logger.info("Reading file: " + name)
                try:
                    info_file = self.read_LSTfile(name, df_type, filter_data)
                    self.info = info_file
                    total_obs += self.tobs
                    run_number = int(name.split(".h5")[0][-5:])
                    if filter_data:
                        pulsarana.cuts.apply_fixed_cut(self, run_number)
                        if pulsarana.cuts.energy_binning_cut is not None:
                            if even_odd_cuts is True:
                                pulsarana.cuts.apply_energydep_cuts_on_even_odd(self)
                            else:
                                pulsarana.cuts.apply_energydep_cuts(self)

                    self.save_memory()
                    info_list.append(self.info)
                except AttributeError:
                    raise ValueError("Failing when reading:" + str(name))

            self.info = pd.DataFrame()
            while len(info_list) > 0:
                if len(info_list) >= 10:
                    chunk_df = pd.concat(info_list[0:10])
                    info_list = info_list[10:]
                else:
                    chunk_df = pd.concat(info_list)
                    info_list = []

                self.info = pd.concat([self.info, chunk_df])

            self.tobs = total_obs

        else:
            self.info = self.read_LSTfile(self.fname, df_type, filter_data)
            self.tobs = self.calculate_tobs(self.info)

            if filter_data:
                pulsarana.cuts.apply_fixed_cut(self)

                if pulsarana.cuts.energy_binning_cut is not None:
                    pulsarana.cuts.apply_energydep_cuts(self)

        logger.info("Finishing reading. Total time is " + str(self.tobs) + " h")


class ReadtxtFile:
    def __init__(self, file, format_txt):
        self.fname = file
        self.format = format_txt

    def read_file(self):
        data = pd.read_csv(self.fname, sep=" ", header=None)
        return data

    def check_format(self):
        for name in ["t", "p"]:
            if name not in self.format:
                raise ValueError("   No valid format")

    def create_df_from_info(self, df):
        for i in range(0, len(self.format)):
            if self.format[i] == "t":
                times = df.iloc[:, i]
            elif self.format[i] == "e":
                energies = df.iloc[:, i]
            elif self.format[i] == "p":
                phases = df.iloc[:, i]
            elif self.format[i] == "g":
                gammaness = df.iloc[:, i]
            elif self.format[i] == "a":
                alpha = df.iloc[:, i]
            elif self.format[i] == "t2":
                theta2 = df.iloc[:, i]
            elif self.format[i] == "at":
                alt_tel = df.iloc[:, i]

        dataframe = pd.DataFrame(
            {
                "mjd_time": times,
                "pulsar_phase": phases,
                "dragon_time": times * 3600 * 24,
                "energy": energies,
            }
        )

        try:
            dataframe["gammaness"] = gammaness
        except AttributeError:
            pass

        try:
            dataframe["alpha"] = alpha
        except AttributeError:
            pass

        try:
            dataframe["theta2"] = theta2
        except AttributeError:
            pass

        try:
            dataframe["alt_tel"] = alt_tel
        except AttributeError:
            pass

        dataframe = dataframe.sort_values(by=["mjd_time"])
        self.info = dataframe

    def calculate_tobs(self):
        diff = np.array(self.info["mjd_time"].to_list()[1:]) - np.array(
            self.info["mjd_time"].to_list()[0:-1]
        )
        return sum(diff) * 24

    def run(self):
        data = self.read_file()
        self.check_format()
        self.create_df_from_info(data)
        self.tobs = self.calculate_tobs()

        logger.info(
            "    Finishing reading. Total time is " + str(self.tobs) + " s" + "\n"
        )


class ReadList:
    def __init__(self, phases_list, time_list=None, energy_list=None, tel="LST"):
        self.plist = phases_list
        self.tlist = time_list
        self.elist = energy_list
        self.tel = tel

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
        if self.tel == "LST" or self.tel == "MAGIC":
            dataframe = add_delta_t_key(self.info)
            return get_effective_time(dataframe)[1].value / 3600

        elif self.tel == "fermi":
            diff = np.array(self.info["mjd_time"].to_list()[1:]) - np.array(
                self.info["mjd_time"].to_list()[0:-1]
            )
            diff[diff > 5 / 24] = 0
            return sum(diff) * 24

    def run(self):
        self.create_df_from_info()
        self.tobs = self.calculate_tobs()
        logger.info(
            "    Finishing reading. Total time is " + str(self.tobs) + " s" + "\n"
        )
