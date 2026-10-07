"""
build_histograms.py
-------------------
Script to build 4D histograms (evtid, energy, gammaness, theta2) per run
for pulsar cut optimization using the Li&Ma method.

For each DL2 run produces a pickle with:
    - on_p1, on_p2, on_p3 (optional), off: 4D histograms
    - livetime: effective observation time in hours

Usage:
    nohup python build_histograms.py --config /top1/jbuces/pulsars/performance_crab/config_phaseograms/config_DL2_rf.yaml --output_dir /top1/jbuces/pulsars/performance_crab/histograms > /top1/jbuces/pulsars/performance_crab/histograms/log.out 2>&1 &
    nohup python build_histograms.py --config /top1/jbuces/pulsars/ctlearn_crab//config_phaseograms/config_DL2.yaml --output_dir /top1/jbuces/pulsars/ctlearn_crab/histograms > /top1/jbuces/pulsars/ctlearn_crab/histograms/log.out 2>&1 &
"""

import os
import glob
import pickle
import argparse
import logging
import yaml
import numpy as np
import h5py
import pandas as pd
from astropy.time import Time
from astropy import units as u
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.reco.utils import get_effective_time, add_delta_t_key
from ctapipe.io import read_table

LOG_FORMAT = "%(asctime)s %(levelname)-6s [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# GAMMANESS_BINS = np.linspace(0, 1, 601)          # 2000 bins, resolution 0.001
GAMMANESS_BINS = np.arange(0.0, 1.0 + 0.05, 0.05) 
# THETA2_BINS    = np.linspace(0, 0.6, 601)        # 6000 bins, resolution 0.001 deg2
THETA2_BINS = np.arange(0.00, 0.4 + 0.01, 0.01) 
LOGENERGY_BINS = np.linspace(-2.0, 2.2, 22)       # 21 bins, 5 per decade
EVT_ID_BINS    = [-0.5, 0.5, 1.5]                 # 2 bins: even (0), odd (1)
INTENSITY_BINS = np.array([31.623, 100, 316.23, 1000, 3162.3, 10000, 1e9])


def read_run(fname, zd_range, run_number, CTLearn):
    """
    Read a DL2 h5 file and return a filtered dataframe with:
    ctlearn_valid + intensity cut + zd cut.
    No gammaness or theta2 cuts applied.
    """
    if CTLearn:
        with h5py.File(fname, "r") as f:
            # CTLearn tables
            ct = f["dl2/event/subarray/classification/CTLearn"][:]
            ce = f["dl2/event/subarray/energy/CTLearn"][:]

            ctlearn_mask = ct["CTLearn_is_valid"] == 1

            # Trigger
            trigger = f["dl1/event/subarray/trigger"][:]
            event_id   = trigger["event_id"][ctlearn_mask]
            event_type = trigger["event_type"][ctlearn_mask]
            t_sec = trigger["time"][:, 0][ctlearn_mask]
            t_ns  = trigger["time"][:, 1][ctlearn_mask]
            dragon_time = t_sec + t_ns * 0.25e-9
            dragon_time = Time(dragon_time, scale="tai", format="unix_tai").utc.unix

            # Hillas intensity
            tel       = f["dl1/event/telescope/parameters/tel_001"][:]
            intensity = tel["hillas_intensity"][ctlearn_mask]
            hillas_len = tel["hillas_length"][ctlearn_mask]
            hillas_wid = tel["hillas_width"][ctlearn_mask]

            # Pointing
            pointing = f["dl1/monitoring/telescope/pointing/tel_001"][:]
            alt_tel  = pointing["altitude"][ctlearn_mask]

            # CTLearn outputs
            gammaness   = ct["CTLearn_prediction"][ctlearn_mask]
            reco_energy = ce["CTLearn_energy"][ctlearn_mask]

        df = pd.DataFrame({
            "event_id":    event_id,
            "event_type":  event_type,
            "dragon_time": dragon_time,
            "intensity":   intensity,
            "alt_tel":     alt_tel,
            "gammaness":   gammaness,
            "energy":      reco_energy,
            "length":  hillas_len,
            "width":  hillas_wid,
        })
    else:
        df = pd.read_hdf(fname, key=dl2_params_lstcam_key)
        df = df.reindex(columns=[
            "event_id",
            "event_type",
            "dragon_time",
            "intensity",
            "alt_tel",
            "gammaness",
            "reco_energy",
            "length",
            "width",
        ])
        df.rename(columns={"reco_energy": "energy"}, inplace=True)
        event_type = df["event_type"].values

    # event_type == 32 (cosmics only)
    df = df[df["event_type"] == 32].reset_index(drop=True)

    # event_type mask (before filtering df): which of the ctlearn_valid events have event_type==32
    et_mask = (event_type == 32)

    # Read theta2 from source_position table, aligned with ctlearn_mask + event_type mask
    source_pos = pd.read_hdf(fname, "source_position")
    if CTLearn:
        source_pos = source_pos[ctlearn_mask].reset_index(drop=True)
    source_pos = source_pos[et_mask].reset_index(drop=True)
    df = df.reset_index(drop=True)

    theta2_col = "theta2" if "theta2" in source_pos.columns else "theta2_on"
    df["theta2"] = source_pos[theta2_col].values

    # Read pulsar phase, same alignment
    phase_info = pd.read_hdf(fname, "phase_info")
    if CTLearn:
        phase_info = phase_info[ctlearn_mask].reset_index(drop=True)
    phase_info = phase_info[et_mask].reset_index(drop=True)
    df["pulsar_phase"] = (phase_info["pulsar_phase"].values % 1.0)

    # Intensity cut: runs < 6045 use 80, others use 50
    int_cut = 80.0 if run_number < 6045 else 50.0
    logger.info(f"  Run {run_number}: intensity cut > {int_cut}")
    df = df[df["intensity"] > int_cut].reset_index(drop=True)

    # Zenith distance cut
    zd = 90.0 - np.rad2deg(df["alt_tel"])
    df = df[(zd > zd_range[0]) & (zd < zd_range[1])].reset_index(drop=True)

    logger.info(f"  Run {run_number}: {len(df)} events after ctlearn_valid + intensity + zd cuts")

    # Preselection limits taken from binning
    g_min  = GAMMANESS_BINS[0]
    t2_max = THETA2_BINS[-1]

    # Apply preselection
    df = df[(df["gammaness"] >= g_min) & (df["theta2"] <= t2_max)].reset_index(drop=True)

    logger.info(
        f"  Run {run_number}: {len(df)} events after preselection "
        f"(g >= {g_min}, theta2 <= {t2_max})"
    )
    return df


def phase_mask(phases, limits):
    """Return boolean mask for phases within the given limits list [lo1,hi1, lo2,hi2, ...]"""
    mask = np.zeros(len(phases), dtype=bool)
    for i in range(0, len(limits), 2):
        mask |= (phases >= limits[i]) & (phases < limits[i + 1])
    return mask


def build_histograms(df, P1_limits, P2_limits, P3_limits, OFF_limits):
    """
    Build 4D histograms for on_p1, on_p2, on_p3 (optional), off.

    Returns dict with keys: on_p1, on_p2, on_p3 (or None), off
    Each histogram shape: (2, 21, 2000, 6000) = (evtid, energy, gammaness, theta2)
    """
    phases    = df["pulsar_phase"].values
    log_e     = np.log10(df["energy"].values).astype("float32")
    intensity = df["intensity"].values.astype("float32")
    gammaness = df["gammaness"].values.astype("float32")
    theta2    = df["theta2"].values.astype("float32")
    odd_even  = (df["event_id"].values % 2).astype("float32")

    def make_hist_energy(mask):
        if mask.sum() == 0:
            return np.zeros(
                (len(EVT_ID_BINS) - 1,
                 len(LOGENERGY_BINS) - 1,
                 len(GAMMANESS_BINS) - 1,
                 len(THETA2_BINS) - 1),
                dtype="float32"
            )
        h, _ = np.histogramdd(
            np.column_stack([
                odd_even[mask],
                log_e[mask],
                gammaness[mask],
                theta2[mask],
            ]),
            bins=[EVT_ID_BINS, LOGENERGY_BINS, GAMMANESS_BINS, THETA2_BINS],
        )
        return h.astype("float32")
    
    def make_hist_intensity(mask):
        if mask.sum() == 0:
            return np.zeros(
                (len(EVT_ID_BINS) - 1, len(INTENSITY_BINS) - 1,
                 len(GAMMANESS_BINS) - 1, len(THETA2_BINS) - 1), dtype="float32")
        h, _ = np.histogramdd(
            np.column_stack([odd_even[mask], intensity[mask],
                             gammaness[mask], theta2[mask]]),
            bins=[EVT_ID_BINS, INTENSITY_BINS, GAMMANESS_BINS, THETA2_BINS],
        )
        return h.astype("float32")

    result_energy = {}
    result_intens = {}
    for key, limits in [("p1", P1_limits), ("p2", P2_limits), ("off", OFF_limits)]:
        m = phase_mask(phases, limits)
        result_energy[key] = make_hist_energy(m)
        result_intens[key] = make_hist_intensity(m)
    
    p3_e = make_hist_energy(phase_mask(phases, P3_limits))   if P3_limits else None
    p3_i = make_hist_intensity(phase_mask(phases, P3_limits)) if P3_limits else None

    return {
        "energy":    {**result_energy, "p3": p3_e},
        "intensity": {**result_intens, "p3": p3_i},
    }
    # return {"on_p1": on_p1, "on_p2": on_p2, "on_p3": on_p3, "off": off}


def calculate_livetime_from_trigger(fname, zd_range, ctlearn):
    """
    Calculate effective livetime from ALL triggered events (event_type==32),
    before any CTLearn or intensity cuts, using lstchain's get_effective_time.
    """
    with h5py.File(fname, "r") as f:
        if ctlearn:
            trigger  = f["dl1/event/subarray/trigger"][:]
            tel_params = f["dl1/event/telescope/parameters/tel_001"][:]
            pointing = f["dl1/monitoring/telescope/pointing/tel_001"][:]

            event_type = trigger["event_type"]
            t_sec      = trigger["time"][:, 0]
            t_ns       = trigger["time"][:, 1]
            dragon_time = t_sec + t_ns * 0.25e-9
            dragon_time = Time(dragon_time, scale="tai", format="unix_tai").utc.unix
            alt_tel    = pointing["altitude"]
        else:
            df = pd.read_hdf(fname, key=dl2_params_lstcam_key)
            event_type = df["event_type"].values
            alt_tel = df["alt_tel"].values
            dragon_time = df["dragon_time"].values
            del df
    # Keep only event_type==32 and zd cut — NO intensity, NO ctlearn
    et_mask = (event_type == 32)
    zd      = 90.0 - np.rad2deg(alt_tel)
    zd_mask = (zd > zd_range[0]) & (zd < zd_range[1])
    mask    = et_mask & zd_mask

    df = pd.DataFrame({
        "dragon_time": dragon_time[mask],
        "event_type":  event_type[mask],
    })
    df = add_delta_t_key(df)
    return get_effective_time(df)[1].to_value(u.h)


def main():
    parser = argparse.ArgumentParser(description="Build 4D histograms per run for pulsar cut optimization")
    parser.add_argument("--config",     required=True,  help="Path to config yaml")
    parser.add_argument("--output_dir", required=True,  help="Directory to save pickle files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "rb") as f:
        conf = yaml.safe_load(f)

    dl2_dir   = conf["pulsar_file_dir"]
    zd_range  = conf["cuts"]["zd_range"]
    P1_limits = conf["phase_regions"]["P1"]
    P2_limits = conf["phase_regions"]["P2"]
    P3_limits = conf["phase_regions"].get("P3", None)
    OFF_limits = conf["phase_regions"]["Bkg"]
    CTLearn = conf["flags"]["ctlearn"]

    # Collect h5 files
    h5_files = sorted(glob.glob(os.path.join(dl2_dir, "*.h5")))
    logger.info(f"Found {len(h5_files)} DL2 files in {dl2_dir}")

    for fname in h5_files:
        basename   = os.path.basename(fname)
        import re
        match      = re.search(r"Run(\d+)", basename)
        if match is None:
            logger.warning(f"Cannot extract run number from {basename}, skipping")
            continue
        run_number = int(match.group(1))

        output_file = os.path.join(args.output_dir, f"histograms_Run{run_number:05d}.pkl")
        if os.path.exists(output_file):
            logger.info(f"Run {run_number}: pickle already exists, skipping")
            continue

        logger.info(f"Processing Run {run_number}: {fname}")
        try:
            df = read_run(fname, zd_range, run_number, CTLearn)
        except Exception as e:
            logger.error(f"Run {run_number}: failed to read — {e}")
            continue

        if len(df) == 0:
            logger.warning(f"Run {run_number}: no events after cuts, skipping")
            continue

        livetime = calculate_livetime_from_trigger(fname, zd_range, CTLearn)
        logger.info(f"  Run {run_number}: livetime = {livetime:.3f} h")

        hists = build_histograms(df, P1_limits, P2_limits, P3_limits, OFF_limits)

        result = {
            "run":      run_number,
            "livetime": livetime,
            "on_p1":    hists["energy"]["p1"],
            "on_p2":    hists["energy"]["p2"],
            "on_p3":    hists["energy"]["p3"],
            "off":      hists["energy"]["off"],
            "on_p1_intensity": hists["intensity"]["p1"],
            "on_p2_intensity": hists["intensity"]["p2"],
            "on_p3_intensity": hists["intensity"]["p3"],
            "off_intensity":   hists["intensity"]["off"],
            "bins": {
                "gammaness": GAMMANESS_BINS,
                "theta2":    THETA2_BINS,
                "logenergy": LOGENERGY_BINS,
                "evt_id":    EVT_ID_BINS,
                "intensity": INTENSITY_BINS,
            },
            "event_id": df["event_id"].values,
            "energy": df["energy"].values,
            "intensity": df["intensity"].values,
            "length": df["length"].values,
            "width": df["width"].values,
            "gammaness": df["gammaness"].values,
            "theta2": df["theta2"].values, 
            "pulsar_phase": df["pulsar_phase"].values,
        }

        with open(output_file, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  Run {run_number}: saved to {output_file}")

        # Free memory
        del df, hists, result

    logger.info("Done.")


if __name__ == "__main__":
    main()
