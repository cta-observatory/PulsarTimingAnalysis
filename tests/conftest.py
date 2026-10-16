"""Shared fixtures of the ptiming_ana test suite."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

REPO_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = Path(__file__).resolve().parent / "files"
FERMI_FILE = REPO_DIR / "tutorial" / "example_data" / "merged2_pulsar.fits"

# Phase regions and shape of the synthetic pulse profile
OFF_LIMITS = [0.52, 0.87]
P1_LIMITS = [0.14, 0.26]
P2_LIMITS = [0.31, 0.49]
P1_MEAN, P1_SIGMA = 0.20, 0.02
P2_MEAN, P2_SIGMA = 0.40, 0.03

# Two lines of a Jodrell Bank-like (.gro) ephemeris of the Crab pulsar
GRO_EPHEMERIS = (
    "0531+21 05 34 31.973 22 00 52.06 59975.0 60005.0 59990.000000352 "
    "29.5867015400000 -3.67381D-10 1.0D-20 0.6 J DE200 0531+21\n"
    "0531+21 05 34 31.973 22 00 52.06 60005.0 60036.0 60020.000000118 "
    "29.5857493100000 -3.67320D-10 1.0D-20 0.5 J DE200 0531+21\n"
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run the tests that need to download data (clock files, ephemerides)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return

    skip_network = pytest.mark.skip(reason="needs --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(autouse=True)
def close_figures():
    """Close the matplotlib figures created by each test."""
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def fermi_file():
    return FERMI_FILE


@pytest.fixture(scope="session")
def fermi_config_dict():
    """Configuration of the Fermi-LAT Crab analysis, with absolute paths."""
    with open(TEST_DATA_DIR / "config_test.yaml") as config_file:
        config = yaml.safe_load(config_file)

    config["pulsar_file_dir"] = str(FERMI_FILE)
    return config


@pytest.fixture
def write_config(tmp_path):
    """Function that writes a configuration dictionary to a YAML file."""

    def _write_config(config, name="config.yaml"):
        path = tmp_path / name
        with open(path, "w") as config_file:
            yaml.safe_dump(config, config_file)
        return str(path)

    return _write_config


@pytest.fixture
def fermi_config(fermi_config_dict, write_config):
    return write_config(fermi_config_dict)


@pytest.fixture(scope="session")
def fermi_analysis(fermi_config_dict, tmp_path_factory):
    """Full analysis of the Fermi-LAT Crab example data set (run only once)."""
    from ptiming_ana.phaseogram import PulsarAnalysis

    path = tmp_path_factory.mktemp("fermi") / "config.yaml"
    with open(path, "w") as config_file:
        yaml.safe_dump(fermi_config_dict, config_file)

    analysis = PulsarAnalysis()
    analysis.set_config(str(path))
    analysis.run()
    return analysis


@pytest.fixture(scope="session")
def synthetic_events():
    """
    Events of a synthetic pulsar: two Gaussian peaks on top of a flat background.

    The arrival times follow a Poisson process (5 Hz) and the energies are
    log-uniformly distributed between 20 GeV and 1 TeV.
    """
    rng = np.random.default_rng(42)
    n_bkg, n_p1, n_p2 = 14000, 3500, 2500

    phases = np.concatenate(
        [
            rng.uniform(0, 1, n_bkg),
            rng.normal(P1_MEAN, P1_SIGMA, n_p1),
            rng.normal(P2_MEAN, P2_SIGMA, n_p2),
        ]
    )
    phases = rng.permutation(phases % 1)
    n_events = len(phases)

    times = 60000.0 + np.cumsum(rng.exponential(1 / 5.0, n_events)) / 86400
    energies = 10 ** rng.uniform(np.log10(0.02), np.log10(1), n_events)

    return SimpleNamespace(
        phases=phases,
        times=times,
        energies=energies,
        tobs=(times[-1] - times[0]) * 24,
        n_bkg=n_bkg,
        n_p1=n_p1,
        n_p2=n_p2,
    )


@pytest.fixture
def make_synthetic_analysis(synthetic_events):
    """Function creating a configured (not yet run) analysis of the synthetic events."""
    from ptiming_ana.phaseogram import PulsarAnalysis

    def _make_analysis(model="dgaussian", binned=True, peak="both", do_fit=True):
        analysis = PulsarAnalysis()
        analysis.setBackgroundLimits(OFF_LIMITS)
        analysis.setPeaklimits(P1_limits=P1_LIMITS, P2_limits=P2_LIMITS)
        analysis.setBinning(50, xmin=0, xmax=1)
        analysis.setTimeInterval(tint=600)
        analysis.setFittingParams(model=model, binned=binned, peak=peak, do_fit=do_fit)
        analysis.setListsInput(
            synthetic_events.phases,
            synthetic_events.times,
            synthetic_events.energies,
            tel="LST",
            energy_units="TeV",
            tobs=synthetic_events.tobs,
        )
        return analysis

    return _make_analysis


@pytest.fixture(scope="session")
def synthetic_analysis(synthetic_events):
    """Analysis of the synthetic events with a binned double Gaussian fit."""
    from ptiming_ana.phaseogram import PulsarAnalysis

    analysis = PulsarAnalysis()
    analysis.setBackgroundLimits(OFF_LIMITS)
    analysis.setPeaklimits(P1_limits=P1_LIMITS, P2_limits=P2_LIMITS)
    analysis.setBinning(50, xmin=0, xmax=1)
    analysis.setTimeInterval(tint=600)
    analysis.setFittingParams(model="dgaussian", binned=True)
    analysis.setEnergybinning([0.02, 0.1, 1.0], do_diff=True, do_integral=True)
    analysis.setListsInput(
        synthetic_events.phases,
        synthetic_events.times,
        synthetic_events.energies,
        tel="LST",
        energy_units="TeV",
        tobs=synthetic_events.tobs,
    )
    analysis.run()
    return analysis


@pytest.fixture
def gro_file(tmp_path):
    """Ephemeris file in .gro format covering MJD 59975 - 60036."""
    path = tmp_path / "crab.gro"
    path.write_text(GRO_EPHEMERIS)
    return str(path)


@pytest.fixture
def dl2_dataframe():
    """Minimal LST-1 DL2 table (6 seconds of data at 5 kHz, MJD 60012)."""
    rng = np.random.default_rng(7)
    n_events = 30000
    # 2023-03-09 (MJD 60012), inside the validity of the test ephemeris
    dragon_time = 1678400000.0 + np.cumsum(rng.exponential(1 / 5000, n_events))

    return pd.DataFrame(
        {
            "obs_id": 1,
            "event_id": np.arange(n_events),
            "dragon_time": dragon_time,
            "intensity": 10 ** rng.uniform(1.5, 4, n_events),
            "reco_energy": 10 ** rng.uniform(-2, 0, n_events),
            "gammaness": rng.uniform(0, 1, n_events),
            "alt_tel": np.deg2rad(rng.uniform(60, 80, n_events)),
            "az_tel": np.deg2rad(180.0),
            "reco_src_x": rng.normal(0, 0.1, n_events),
            "reco_src_y": rng.normal(0, 0.1, n_events),
            # 32 are the cosmic events; some interleaved events are included
            "event_type": np.where(np.arange(n_events) % 100 == 0, 2, 32),
        }
    )


#: Position of the Crab pulsar used in the synthetic DL2 files
CRAB_RA_DEC = (83.633, 22.0145)


@pytest.fixture(scope="session")
def dl2_directions():
    """
    Reconstructed directions (Alt/Az, rad) of the events of `dl2_dataframe`,
    at known angular distances (theta, deg) from the Crab pulsar.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
    from astropy.time import Time
    from ctapipe_io_lst.constants import LST1_LOCATION

    rng = np.random.default_rng(11)
    n_events = 30000
    # Same times as in dl2_dataframe
    dragon_time = 1678400000.0 + np.cumsum(
        np.random.default_rng(7).exponential(1 / 5000, n_events)
    )
    theta = rng.uniform(0, 0.5, n_events)

    frame = AltAz(
        obstime=Time(dragon_time, format="unix", scale="utc"), location=LST1_LOCATION
    )
    source = SkyCoord(*CRAB_RA_DEC, unit="deg", frame="icrs")
    with erfa_astrom.set(ErfaAstromInterpolator(5 * u.min)):
        source_altaz = source.transform_to(frame)
    reco = source_altaz.directional_offset_by(
        rng.uniform(0, 360, n_events) * u.deg, theta * u.deg
    )

    return SimpleNamespace(reco_alt=reco.alt.rad, reco_az=reco.az.rad, theta2=theta**2)


@pytest.fixture
def write_ctlearn_dl2():
    """
    Function writing a CTLearn-like DL2 file of LST-1 (ctapipe data model) with
    the events of a lstchain-like dataframe.

    As in the files produced by CTLearn, the DL1 tables follow the order of the
    original events, while the DL2 tables are sorted by event. Every 7th event
    has no valid prediction.
    """
    import astropy.units as u
    from astropy.table import Table
    from astropy.time import Time
    from ctapipe.io import write_table

    def _write(path, events, prefixes=None, classification_group="classification"):
        prefixes = prefixes or dict.fromkeys(
            ("classification", "energy", "geometry"), "CTLearn"
        )
        events = events.reset_index(drop=True)
        n_events = len(events)
        time = Time(events["dragon_time"].to_numpy(), format="unix", scale="utc")
        valid = np.arange(n_events) % 7 != 0
        keys = {
            "obs_id": events["obs_id"].to_numpy(),
            "event_id": events["event_id"].to_numpy(),
        }

        trigger = Table(
            keys
            | {
                "time": time,
                "tel_with_trigger": np.ones((n_events, 1), dtype=bool),
                "event_type": events["event_type"].to_numpy(),
            }
        )
        write_table(trigger, path, "/dl1/event/subarray/trigger", overwrite=True)

        pointing = Table(
            {
                "time": time,
                "azimuth": events["az_tel"].to_numpy() * u.rad,
                "altitude": events["alt_tel"].to_numpy() * u.rad,
            }
        )
        write_table(
            pointing, path, "/dl1/monitoring/telescope/pointing/tel_001", overwrite=True
        )

        parameters = Table(
            keys
            | {
                "tel_id": np.ones(n_events, dtype=int),
                "hillas_intensity": events["intensity"].to_numpy(),
            }
        )
        write_table(
            parameters, path, "/dl1/event/telescope/parameters/tel_001", overwrite=True
        )

        def _with_nan(values):
            return np.where(valid, values, np.nan)

        columns = {
            "classification": {"prediction": _with_nan(events["gammaness"])},
            "energy": {"energy": _with_nan(events["reco_energy"]) * u.TeV},
            "geometry": {
                "alt": _with_nan(np.rad2deg(events["reco_alt"])) * u.deg,
                "az": _with_nan(np.rad2deg(events["reco_az"])) * u.deg,
            },
        }
        for task, prefix in prefixes.items():
            table = Table(
                keys
                | {f"{prefix}_{name}": value for name, value in columns[task].items()}
                | {f"{prefix}_is_valid": valid}
            )
            table.sort(["obs_id", "event_id"])
            group = classification_group if task == "classification" else task
            write_table(
                table, path, f"/dl2/event/subarray/{group}/{prefix}", overwrite=True
            )
        return str(path)

    return _write


@pytest.fixture
def dl2_events(dl2_dataframe, dl2_directions):
    """
    `dl2_dataframe` with reconstructed directions and the event ids shuffled
    within each block of 10 events, so that the events are not sorted by id
    (the DL2 tables of CTLearn are).
    """
    events = dl2_dataframe.copy()
    events["reco_alt"] = dl2_directions.reco_alt
    events["reco_az"] = dl2_directions.reco_az
    rng = np.random.default_rng(3)
    event_id = events["event_id"].to_numpy().reshape(-1, 10)
    events["event_id"] = rng.permuted(event_id, axis=1).ravel()
    return events
