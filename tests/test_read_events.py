import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from lstchain.io.io import dl2_params_lstcam_key

from ptiming_ana.phaseogram import (
    FilterPulsarAna,
    PulsarAnalysis,
    ReadDL3File,
    ReadFermiFile,
    ReadList,
    ReadLSTFile,
    ReadtxtFile,
)
from ptiming_ana.phaseogram.read_events import compute_theta2


def test_compute_theta2():
    assert compute_theta2(0.1, -0.2, 0.1, -0.2) == pytest.approx(0)

    # 1 deg offset in the camera for the LST effective focal length
    focal_length = 28 * 1.0466
    offset = focal_length * np.tan(np.deg2rad(1.0))
    assert compute_theta2(offset, 0.0, 0.0, 0.0) == pytest.approx(1.0)

    theta2 = compute_theta2(np.array([0.0, offset]), np.zeros(2), 0.0, 0.0)
    np.testing.assert_allclose(theta2, [0, 1])


def test_read_fermi_file(fermi_file):
    reader = ReadFermiFile(str(fermi_file))
    reader.run()

    with fits.open(fermi_file) as hdul:
        table = hdul[1].data
        n_events = len(table)
        energies = np.array(table["ENERGY"], dtype=float)
        phases = np.array(table["PULSE_PHASE"], dtype=float)

    info = reader.info
    assert len(info) == n_events
    assert set(info.columns) == {"mjd_time", "pulsar_phase", "dragon_time", "energy"}
    assert info["mjd_time"].is_monotonic_increasing
    # Native byte order, so pandas can operate on the columns
    assert all(dtype.isnative for dtype in info.dtypes)
    # Energies converted from MeV to TeV
    assert info["energy"].sum() == pytest.approx(energies.sum() / 1e6)
    assert np.sort(info["pulsar_phase"]) == pytest.approx(np.sort(phases))
    np.testing.assert_allclose(info["dragon_time"], info["mjd_time"] * 86400)
    assert reader.tobs == pytest.approx(8342.59, abs=0.01)


def test_read_fermi_file_wrong_format():
    with pytest.raises(ValueError, match="FITS"):
        ReadFermiFile("events.h5")


def test_read_list(synthetic_events):
    reader = ReadList(
        synthetic_events.phases,
        synthetic_events.times,
        synthetic_events.energies,
        tel="LST",
        tobs=1.5,
    )
    reader.run()

    assert len(reader.info) == len(synthetic_events.phases)
    assert reader.info["mjd_time"].is_monotonic_increasing
    # The observation time given by the user is used
    assert reader.tobs == 1.5


def test_read_list_effective_time():
    """For unfiltered event lists the effective time is estimated with lstchain."""
    rng = np.random.default_rng(2)
    n_events = 50000
    times = 60000 + np.cumsum(rng.exponential(1 / 5000, n_events)) / 86400

    reader = ReadList(rng.uniform(0, 1, n_events), times, tel="LST")
    reader.run()

    elapsed = (times[-1] - times[0]) * 24
    assert reader.tobs == pytest.approx(elapsed, rel=0.02)


def test_read_list_sparse_events():
    """Lists of events after cuts: the elapsed time without gaps is used."""
    times = 60000 + np.concatenate([np.arange(100), 7200 + np.arange(100)]) / 86400

    reader = ReadList(np.linspace(0, 0.99, 200), times, tel="MAGIC")
    reader.run()

    assert reader.tobs == pytest.approx(2 * 99 / 3600)


def test_read_list_accepts_python_lists():
    """Regression test: a list of times was repeated 86400 times (MemoryError)."""
    reader = ReadList([0.1, 0.5, 0.9], [60000.0, 60000.1, 60000.4], tel="fermi")
    reader.run()

    assert len(reader.info) == 3
    assert reader.tobs == pytest.approx(0.1 * 24)  # the 0.3 d gap is not counted
    assert reader.info["energy"].isna().all()
    np.testing.assert_allclose(
        reader.info["dragon_time"], reader.info["mjd_time"] * 86400
    )


def test_read_list_sorts_events_in_time():
    reader = ReadList([0.9, 0.1, 0.5], [60000.3, 60000.0, 60000.1], [3, 1, 2], "fermi")
    reader.run()
    np.testing.assert_allclose(reader.info["pulsar_phase"], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(reader.info["energy"], [1, 2, 3])


def test_read_list_errors():
    with pytest.raises(ValueError, match="arrival times"):
        ReadList([0.1, 0.2])
    with pytest.raises(ValueError, match="same length"):
        ReadList([0.1, 0.2], [60000.0])


@pytest.mark.parametrize("format_txt", ["tpe", "t p e", "t,p,e", ["t", "p", "e"]])
def test_read_txt_file(tmp_path, format_txt):
    path = tmp_path / "events.txt"
    path.write_text("60000.2 0.7 2.0\n60000.0 0.1 1.0\n60000.1 0.4 1.5\n")

    reader = ReadtxtFile(str(path), format_txt)
    reader.run()

    np.testing.assert_allclose(reader.info["mjd_time"], [60000.0, 60000.1, 60000.2])
    np.testing.assert_allclose(reader.info["pulsar_phase"], [0.1, 0.4, 0.7])
    np.testing.assert_allclose(reader.info["energy"], [1.0, 1.5, 2.0])
    assert reader.tobs == pytest.approx(0.2 * 24)


def test_read_txt_file_optional_columns(tmp_path):
    """Regression test: multi-character codes and missing columns failed."""
    path = tmp_path / "events.txt"
    path.write_text("60000.0 0.1 0.9 0.02 1.2\n60000.1 0.4 0.8 0.03 1.3\n")

    reader = ReadtxtFile(str(path), "t p g t2 at")
    reader.run()

    np.testing.assert_allclose(reader.info["gammaness"], [0.9, 0.8])
    np.testing.assert_allclose(reader.info["theta2"], [0.02, 0.03])
    np.testing.assert_allclose(reader.info["alt_tel"], [1.2, 1.3])
    assert reader.info["energy"].isna().all()


def test_read_txt_file_wrong_format(tmp_path):
    path = tmp_path / "events.txt"
    path.write_text("60000.0 0.1\n60000.1 0.4\n")

    with pytest.raises(ValueError, match="No valid format"):
        ReadtxtFile(str(path), "te").run()
    with pytest.raises(ValueError, match="unknown columns"):
        ReadtxtFile(str(path), "tpx").run()
    with pytest.raises(ValueError, match="columns"):
        ReadtxtFile(str(path), "tpe").run()


def test_read_lst_file_arguments(tmp_path):
    with pytest.raises(ValueError, match="No file provided"):
        ReadLSTFile()
    with pytest.raises(ValueError, match="not both"):
        ReadLSTFile(file="dl2.h5", directory=str(tmp_path))
    with pytest.raises(ValueError, match="hdf5"):
        ReadLSTFile(file="dl2.fits")


def test_read_dl3_file_arguments():
    with pytest.raises(ValueError, match="No directory"):
        ReadDL3File()
    with pytest.raises(ValueError, match="target coordinates"):
        ReadDL3File(directory=".")


@pytest.fixture
def dl2_file(tmp_path, dl2_dataframe):
    """DL2 file with the phases stored in the phase_info table."""
    rng = np.random.default_rng(3)
    path = tmp_path / "dl2_LST-1.Run00001_pulsar.h5"
    dl2_dataframe.to_hdf(path, key=dl2_params_lstcam_key, mode="w")

    phase_info = pd.DataFrame(
        {
            "obs_id": dl2_dataframe["obs_id"],
            "event_id": dl2_dataframe["event_id"],
            "mjd_barycenter_time": dl2_dataframe["dragon_time"] / 86400 + 40587,
            "pulsar_phase": rng.uniform(-0.5, 0.5, len(dl2_dataframe)),
        }
    )
    phase_info.to_hdf(path, key="phase_info")
    return str(path), phase_info


def test_read_lst_dl2_file(dl2_file, dl2_dataframe):
    path, phase_info = dl2_file
    analysis = PulsarAnalysis()

    reader = ReadLSTFile(file=path)
    reader.run(analysis)

    cosmics = (dl2_dataframe["event_type"] == 32).to_numpy()
    info = reader.info
    assert len(info) == cosmics.sum()
    assert {"pulsar_phase", "mjd_time", "energy", "gammaness", "delta_t"} <= set(info)
    np.testing.assert_allclose(
        info["pulsar_phase"], phase_info["pulsar_phase"][cosmics]
    )
    np.testing.assert_allclose(info["energy"], dl2_dataframe["reco_energy"][cosmics])
    # 30000 events at 5 kHz
    assert reader.tobs == pytest.approx(6 / 3600, rel=0.05)


def test_read_lst_dl2_file_with_cuts(dl2_file):
    path, _ = dl2_file
    analysis = PulsarAnalysis()
    analysis.cuts = FilterPulsarAna(gammaness_cut=0.7, int_cut=100, energy_cut=0.05)

    reader = ReadLSTFile(file=path)
    reader.run(analysis)

    assert 0 < len(reader.info) < 10000
    assert reader.info["gammaness"].min() > 0.7
    assert reader.info["intensity"].min() > 100
    assert reader.info["energy"].min() > 0.05


def test_read_lst_dl2_file_theta2_from_source_position(dl2_file, dl2_dataframe):
    path, _ = dl2_file
    position = pd.DataFrame(
        {
            "src_x": np.zeros(len(dl2_dataframe)),
            "src_y": np.zeros(len(dl2_dataframe)),
            "theta2": np.linspace(0, 1, len(dl2_dataframe)),
        }
    )
    position.to_hdf(path, key="source_position")

    reader = ReadLSTFile(file=path)
    reader.run(PulsarAnalysis())

    cosmics = (dl2_dataframe["event_type"] == 32).to_numpy()
    np.testing.assert_allclose(reader.info["theta2"], position["theta2"][cosmics])


def test_read_lst_dl2_directory(dl2_file, dl2_dataframe, tmp_path):
    path, _ = dl2_file
    reader = ReadLSTFile(directory=str(tmp_path))
    assert len(reader.fname) == 1

    reader.run(PulsarAnalysis())
    assert len(reader.info) == (dl2_dataframe["event_type"] == 32).sum()


def test_read_lst_dl2_directory_observation_time(dl2_file, tmp_path):
    """
    Regression test: with several files the observation time was computed
    after the cuts, where the time differences between events are meaningless.
    """
    path, _ = dl2_file
    analysis = PulsarAnalysis()
    analysis.cuts = FilterPulsarAna(gammaness_cut=0.9)

    single = ReadLSTFile(file=path)
    single.run(analysis)

    directory = ReadLSTFile(directory=str(tmp_path))
    directory.run(analysis)

    assert len(directory.info) == len(single.info)
    assert directory.tobs == pytest.approx(single.tobs)
    assert directory.tobs == pytest.approx(6 / 3600, rel=0.05)
