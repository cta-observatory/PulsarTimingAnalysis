"""Tests of the reading of lstchain and CTLearn DL2 files."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tables
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from ctapipe_io_lst.constants import LST1_LOCATION
from astropy.time import Time
from ctapipe.io import read_table, write_table
from lstchain.io.io import dl2_params_lstcam_key

from ptiming_ana.cphase import DL2_calphase, add_DL2_phase_table, add_source_info_dl2
from ptiming_ana.cphase import pulsarphase_cal
from ptiming_ana.io import (
    CTLEARN,
    LSTCHAIN,
    get_dl2_format,
    read_ctlearn_dl2,
    read_dl2_directions,
    read_dl2_times,
)
from ptiming_ana.phaseogram import FilterPulsarAna, PulsarAnalysis, ReadLSTFile
from ptiming_ana.phaseogram.read_events import compute_theta2

# Same position as in the dl2_directions fixture
CRAB = SkyCoord(83.633, 22.0145, unit="deg", frame="icrs")


@pytest.fixture
def ctlearn_file(tmp_path, dl2_events, write_ctlearn_dl2):
    return write_ctlearn_dl2(tmp_path / "dl2_LST-1.Run00001.h5", dl2_events)


@pytest.fixture
def lstchain_file(tmp_path, dl2_events):
    path = tmp_path / "lstchain" / "dl2_LST-1.Run00001.h5"
    path.parent.mkdir()
    dl2_events.to_hdf(path, key=dl2_params_lstcam_key)
    return str(path)


def _valid(events):
    return np.arange(len(events)) % 7 != 0


def _exact_theta2(events):
    """
    theta2 (deg2) of the events, transforming their reconstructed direction to
    ICRS without interpolation. Reference for the computation in
    add_source_info_dl2 (which interpolates the transformation, as lstchain).
    """
    frame = AltAz(
        obstime=Time(events["dragon_time"].to_numpy(), format="unix", scale="utc"),
        location=LST1_LOCATION,
    )
    reco = SkyCoord(
        alt=events["reco_alt"].to_numpy() * u.rad,
        az=events["reco_az"].to_numpy() * u.rad,
        frame=frame,
    )
    return reco.transform_to("icrs").separation(CRAB).to_value(u.deg) ** 2


@pytest.fixture
def exact_theta2(dl2_dataframe, dl2_directions):
    events = dl2_dataframe.assign(
        reco_alt=dl2_directions.reco_alt, reco_az=dl2_directions.reco_az
    )
    return _exact_theta2(events)


def test_dl2_format(ctlearn_file, lstchain_file, tmp_path):
    assert get_dl2_format(ctlearn_file) == CTLEARN
    assert get_dl2_format(lstchain_file) == LSTCHAIN

    other = tmp_path / "other.h5"
    pd.DataFrame({"a": [1]}).to_hdf(other, key="data")
    with pytest.raises(ValueError, match="neither an lstchain"):
        get_dl2_format(other)


def test_read_ctlearn_dl2(ctlearn_file, dl2_events):
    events = read_ctlearn_dl2(ctlearn_file)
    valid = _valid(dl2_events)

    # All the events, in the order of the trigger table
    assert len(events) == len(dl2_events)
    np.testing.assert_array_equal(events["event_id"], dl2_events["event_id"])
    np.testing.assert_array_equal(events["event_type"], dl2_events["event_type"])
    np.testing.assert_allclose(events["dragon_time"], dl2_events["dragon_time"])
    np.testing.assert_allclose(events["intensity"], dl2_events["intensity"])
    np.testing.assert_allclose(events["alt_tel"], dl2_events["alt_tel"])

    # Time difference with the previous triggered event (non-cosmic included)
    np.testing.assert_allclose(
        events["delta_t"][1:], np.diff(dl2_events["dragon_time"]), atol=1e-6
    )
    mjd = Time(dl2_events["dragon_time"].to_numpy(), format="unix").mjd
    np.testing.assert_allclose(events["mjd_time"], mjd, rtol=0, atol=1e-10)

    # The predictions are matched by event id (not by position)
    np.testing.assert_array_equal(events["ctlearn_is_valid"], valid)
    np.testing.assert_allclose(
        events["gammaness"][valid], dl2_events["gammaness"][valid]
    )
    np.testing.assert_allclose(
        events["reco_energy"][valid], dl2_events["reco_energy"][valid]
    )
    np.testing.assert_allclose(events["reco_alt"][valid], dl2_events["reco_alt"][valid])
    np.testing.assert_allclose(events["reco_az"][valid], dl2_events["reco_az"][valid])
    assert events["gammaness"][~valid].isna().all()


def test_read_ctlearn_dl2_prefixes(tmp_path, dl2_events, write_ctlearn_dl2):
    # Names used by recent versions of CTLearn
    prefixes = {
        "classification": "CTLearnClassifier",
        "energy": "CTLearnRegressor",
        "geometry": "CTLearnCameraReconstructor",
    }
    path = write_ctlearn_dl2(tmp_path / "dl2.h5", dl2_events[:1000], prefixes)
    events = read_ctlearn_dl2(path)
    assert events["gammaness"].notna().any()

    # Two classification tables: the one to use has to be given
    write_ctlearn_dl2(path, dl2_events[:1000], {"classification": "Other"})
    with pytest.raises(ValueError, match="Several classification tables"):
        read_ctlearn_dl2(path)

    events = read_ctlearn_dl2(path, prefixes={"classification": "Other"})
    assert events["gammaness"].notna().any()

    with pytest.raises(ValueError, match="No CTLearn energy table called"):
        read_ctlearn_dl2(path, prefixes={"classification": "Other", "energy": "X"})


def test_read_ctlearn_dl2_particle_type(tmp_path, dl2_events, write_ctlearn_dl2):
    """Since ctapipe 0.27 the classification group is called particle_type."""
    prefixes = {
        "classification": "CTLearnClassifier",
        "energy": "CTLearnRegressor",
        "geometry": "CTLearnCameraReconstructor",
    }
    path = write_ctlearn_dl2(
        tmp_path / "dl2.h5",
        dl2_events[:1000],
        prefixes,
        classification_group="particle_type",
    )
    assert get_dl2_format(path) == CTLEARN

    events = read_ctlearn_dl2(path)
    valid = _valid(dl2_events[:1000])
    np.testing.assert_allclose(
        events["gammaness"][valid], dl2_events["gammaness"][:1000][valid]
    )

    events = read_ctlearn_dl2(path, prefixes={"classification": "CTLearnClassifier"})
    assert events["gammaness"].notna().any()


def test_read_ctlearn_dl2_without_classification(
    tmp_path, dl2_events, write_ctlearn_dl2, caplog
):
    prefixes = {"energy": "CTLearn", "geometry": "CTLearn"}
    path = write_ctlearn_dl2(tmp_path / "dl2.h5", dl2_events[:1000], prefixes)
    events = read_ctlearn_dl2(path)
    assert events["gammaness"].isna().all()
    assert "No CTLearn classification predictions" in caplog.text


def test_read_ctlearn_dl2_without_direction(tmp_path, dl2_events, write_ctlearn_dl2):
    prefixes = {"classification": "CTLearn", "energy": "CTLearn"}
    path = write_ctlearn_dl2(tmp_path / "dl2.h5", dl2_events[:1000], prefixes)

    events = read_ctlearn_dl2(path)
    assert "reco_alt" not in events
    assert events["gammaness"].notna().any()

    with pytest.raises(ValueError, match="No reconstructed direction"):
        add_source_info_dl2(path, source_position=CRAB)


@pytest.mark.parametrize("file", ["ctlearn_file", "lstchain_file"])
def test_read_dl2_times(file, dl2_events, request):
    times = read_dl2_times(request.getfixturevalue(file))

    assert list(times.columns) == ["obs_id", "event_id", "dragon_time", "mjd_time"]
    np.testing.assert_array_equal(times["event_id"], dl2_events["event_id"])
    mjd = Time(dl2_events["dragon_time"].to_numpy(), format="unix").mjd
    np.testing.assert_allclose(times["mjd_time"], mjd, rtol=0, atol=1e-10)


@pytest.mark.parametrize("file", ["ctlearn_file", "lstchain_file"])
def test_add_source_info(file, dl2_events, dl2_directions, exact_theta2, request):
    """theta2 is computed in the same way for lstchain and CTLearn files."""
    path = request.getfixturevalue(file)
    add_source_info_dl2(path, source_position=CRAB)

    table = pd.read_hdf(path, key="source_position")
    assert list(table.columns) == ["obs_id", "event_id", "theta2"]
    np.testing.assert_array_equal(table["event_id"], dl2_events["event_id"])

    valid = _valid(dl2_events) if file == "ctlearn_file" else slice(None)
    theta = np.sqrt(table["theta2"][valid].to_numpy())

    # Same as the exact transformation, up to its interpolation (< 0.01 arcsec)
    np.testing.assert_allclose(theta, np.sqrt(exact_theta2[valid]), rtol=0, atol=3e-6)

    # And close to the angular distance at which the directions were simulated.
    # Not exact: the aberration does not conserve angular distances between
    # Alt/Az and ICRS (~1 arcsec at 0.5 deg)
    np.testing.assert_allclose(
        theta, np.sqrt(dl2_directions.theta2[valid]), rtol=0, atol=1e-3
    )
    if file == "ctlearn_file":
        assert table["theta2"][~valid].isna().all()

    directions = read_dl2_directions(path)
    assert len(directions) == len(dl2_events)


def _write_phases(path, dl2_events):
    """phase_info table, sorted by event id (not in the order of the file)."""
    phase_info = pd.DataFrame(
        {
            "obs_id": dl2_events["obs_id"].to_numpy(),
            "event_id": dl2_events["event_id"].to_numpy(),
            "mjd_barycenter_time": dl2_events["dragon_time"].to_numpy() / 86400,
            "pulsar_phase": dl2_events["event_id"].to_numpy() / len(dl2_events),
        }
    ).sort_values("event_id")
    phase_info.to_hdf(path, key="phase_info")


def test_read_ctlearn_file(ctlearn_file, dl2_events, exact_theta2):
    _write_phases(ctlearn_file, dl2_events)
    add_source_info_dl2(ctlearn_file, source_position=CRAB)

    reader = ReadLSTFile(file=ctlearn_file)
    reader.run(PulsarAnalysis())

    selected = ((dl2_events["event_type"] == 32) & _valid(dl2_events)).to_numpy()
    info = reader.info
    assert len(info) == selected.sum()
    assert "ctlearn_is_valid" not in info
    np.testing.assert_array_equal(info["event_id"], dl2_events["event_id"][selected])
    np.testing.assert_allclose(
        info["pulsar_phase"], dl2_events["event_id"][selected] / len(dl2_events)
    )
    np.testing.assert_allclose(info["energy"], dl2_events["reco_energy"][selected])
    np.testing.assert_allclose(
        np.sqrt(info["theta2"]), np.sqrt(exact_theta2[selected]), rtol=0, atol=3e-6
    )

    # The effective time is computed with all the events: 30000 events at 5 kHz
    assert reader.tobs == pytest.approx(6 / 3600, rel=0.05)


def test_read_ctlearn_and_lstchain_files(ctlearn_file, lstchain_file, dl2_events):
    """With the same events both formats give the same result."""
    for path in (ctlearn_file, lstchain_file):
        _write_phases(path, dl2_events)
        add_source_info_dl2(path, source_position=CRAB)

    analysis = PulsarAnalysis()
    analysis.cuts = FilterPulsarAna(gammaness_cut=0.5, theta2_cut=0.04, int_cut=50)

    readers = [ReadLSTFile(file=path) for path in (ctlearn_file, lstchain_file)]
    for reader in readers:
        reader.run(analysis)
    ctlearn, lstchain = (reader.info for reader in readers)

    lstchain = lstchain[np.isin(lstchain["event_id"], ctlearn["event_id"])]
    assert 0 < len(ctlearn) == len(lstchain)
    np.testing.assert_allclose(ctlearn["pulsar_phase"], lstchain["pulsar_phase"])
    # CTLearn stores the directions in deg: tiny differences from the conversion
    np.testing.assert_allclose(
        np.sqrt(ctlearn["theta2"]), np.sqrt(lstchain["theta2"]), rtol=0, atol=1e-8
    )
    assert readers[0].tobs == pytest.approx(readers[1].tobs)


def test_read_ctlearn_directory(ctlearn_file, dl2_events, tmp_path):
    _write_phases(ctlearn_file, dl2_events)
    # Files that are not DL2 files are ignored
    (tmp_path / "notes.txt").write_text("not a DL2 file")

    reader = ReadLSTFile(directory=str(tmp_path))
    assert len(reader.fname) == 1

    reader.run(PulsarAnalysis())
    selected = (dl2_events["event_type"] == 32) & _valid(dl2_events)
    assert len(reader.info) == selected.sum()


def test_ctlearn_source_dependent(ctlearn_file, dl2_events):
    _write_phases(ctlearn_file, dl2_events)
    reader = ReadLSTFile(file=ctlearn_file, src_dependent=True)
    with pytest.raises(ValueError, match="not available for CTLearn"):
        reader.run(PulsarAnalysis())


@pytest.fixture
def fake_pint(monkeypatch):
    """
    Replace the computation of the phases with PINT (which needs to download
    the clock corrections and the solar system ephemeris) by a fake one: the
    "barycentric" times are the arrival times.
    """

    def _phases(timname, model, pickle=False):
        times = np.loadtxt(timname, skiprows=1, usecols=2)
        return times, SimpleNamespace(frac=(times * 86400 * 29.6) % 1 - 0.5)

    monkeypatch.setattr(pulsarphase_cal, "get_phase_list_from_tim", _phases)


@pytest.mark.parametrize("file", ["ctlearn_file", "lstchain_file"])
def test_dl2_phases(
    file, dl2_events, gro_file, tmp_path, monkeypatch, fake_pint, request
):
    path = request.getfixturevalue(file)
    monkeypatch.chdir(tmp_path)

    DL2_calphase(path, gro_file)

    phase_info = pd.read_hdf(path, key="phase_info")
    np.testing.assert_array_equal(phase_info["event_id"], dl2_events["event_id"])
    mjd = Time(dl2_events["dragon_time"].to_numpy(), format="unix").mjd
    np.testing.assert_allclose(phase_info["mjd_barycenter_time"], mjd, atol=1e-9)


@pytest.mark.parametrize("file", ["ctlearn_file", "lstchain_file"])
def test_dl2_phases_ephemeris_not_valid(file, tmp_path, monkeypatch, request):
    path = request.getfixturevalue(file)
    monkeypatch.chdir(tmp_path)

    # Ephemeris that does not cover the observation (MJD 60012)
    par_file = tmp_path / "crab.par"
    par_file.write_text(
        "PSR J0534+2200\nRAJ 05:34:31.973\nDECJ 22:00:52.06\nF0 29.6\n"
        "PEPOCH 60100\nSTART 60100\nFINISH 60130\nUNITS TDB\n"
    )
    with pytest.raises(ValueError, match="outside the validity range"):
        DL2_calphase(path, str(par_file))


def test_dl2_command_line(
    ctlearn_file, dl2_events, gro_file, tmp_path, monkeypatch, fake_pint
):
    monkeypatch.chdir(tmp_path)
    add_DL2_phase_table.main(
        [
            "--dir",
            str(tmp_path),
            "--ephem",
            gro_file,
            "--include-theta",
            "--source-ra",
            str(CRAB.ra.deg),
            "--source-dec",
            str(CRAB.dec.deg),
        ]
    )
    assert len(pd.read_hdf(ctlearn_file, key="phase_info")) == len(dl2_events)
    assert len(pd.read_hdf(ctlearn_file, key="source_position")) == len(dl2_events)

    with pytest.raises(SystemExit):
        add_DL2_phase_table.main(
            ["-f", ctlearn_file, "--ephem", gro_file, "-t", "--source-ra", "83.6"]
        )


@pytest.mark.parametrize("old_format", ["theta2_on", "theta2", "camera"])
def test_read_old_source_position(old_format, lstchain_file, dl2_events):
    """source_position tables written by previous versions (without event keys)."""
    _write_phases(lstchain_file, dl2_events)

    rng = np.random.default_rng(5)
    theta2 = rng.uniform(0, 1, len(dl2_events))
    if old_format == "camera":
        src_x = rng.normal(0, 0.1, len(dl2_events))
        src_y = rng.normal(0, 0.1, len(dl2_events))
        table = pd.DataFrame({"src_x": src_x, "src_y": src_y})
        theta2 = compute_theta2(
            dl2_events["reco_src_x"].to_numpy(),
            dl2_events["reco_src_y"].to_numpy(),
            src_x,
            src_y,
        )
    else:
        table = pd.DataFrame(
            {"dragon_time": dl2_events["dragon_time"].to_numpy(), old_format: theta2}
        )
    table.to_hdf(lstchain_file, key="source_position")

    info = ReadLSTFile(file=lstchain_file).read_LSTfile(lstchain_file)
    cosmic = (dl2_events["event_type"] == 32).to_numpy()
    np.testing.assert_allclose(info["theta2"], theta2[cosmic])


def test_source_position_with_other_events(lstchain_file, dl2_events):
    _write_phases(lstchain_file, dl2_events)
    pd.DataFrame({"theta2": np.zeros(10)}).to_hdf(lstchain_file, key="source_position")

    with pytest.raises(ValueError, match="source_position table has 10 rows"):
        ReadLSTFile(file=lstchain_file).read_LSTfile(lstchain_file)


def test_read_file_all_columns(lstchain_file, dl2_events):
    _write_phases(lstchain_file, dl2_events)
    info = ReadLSTFile(file=lstchain_file).read_LSTfile(lstchain_file, df_type="long")

    assert {"reco_src_x", "az_tel", "pulsar_phase", "energy"} <= set(info.columns)
    np.testing.assert_allclose(
        info["energy"], dl2_events["reco_energy"][dl2_events["event_type"] == 32]
    )


def test_read_ctlearn_dl2_unknown_task(ctlearn_file):
    with pytest.raises(ValueError, match="Unknown CTLearn tasks"):
        read_ctlearn_dl2(ctlearn_file, prefixes={"direction": "CTLearn"})


def test_read_ctlearn_dl2_without_dl1_tables(ctlearn_file):
    """Without pointing and DL1 parameters the events can still be read."""
    with tables.open_file(ctlearn_file, mode="a") as h5file:
        h5file.remove_node("/dl1/monitoring/telescope/pointing/tel_001")
        h5file.remove_node("/dl1/event/telescope/parameters/tel_001")

    events = read_ctlearn_dl2(ctlearn_file)
    assert events["alt_tel"].isna().all()
    assert events["intensity"].isna().all()
    assert events["gammaness"].notna().any()


def test_read_ctlearn_dl2_pointing_not_per_event(
    tmp_path, dl2_events, write_ctlearn_dl2
):
    path = write_ctlearn_dl2(tmp_path / "dl2.h5", dl2_events[:1000])
    pointing = Table(
        {
            "time": Time([1678400000.0, 1678400010.0], format="unix"),
            "azimuth": [3.1, 3.1] * u.rad,
            "altitude": [1.2, 1.2] * u.rad,
        }
    )
    write_table(
        pointing, path, "/dl1/monitoring/telescope/pointing/tel_001", overwrite=True
    )
    with pytest.raises(ValueError, match="the pointing is expected per event"):
        read_ctlearn_dl2(path)


def test_read_ctlearn_dl2_not_sorted(tmp_path, dl2_events, write_ctlearn_dl2, caplog):
    events = dl2_events[:1000].copy()
    events["dragon_time"] = events["dragon_time"].to_numpy()[::-1]
    path = write_ctlearn_dl2(tmp_path / "dl2.h5", events)

    read_ctlearn_dl2(path)
    assert "not sorted in time" in caplog.text


def test_read_source_dependent_file(lstchain_file, dl2_events):
    """The source-dependent parameters replace the source-independent ones."""
    from lstchain.io.io import dl2_params_src_dep_lstcam_key, write_dataframe

    _write_phases(lstchain_file, dl2_events)
    rng = np.random.default_rng(9)
    srcdep = pd.DataFrame(
        {
            ("on", "alpha"): rng.uniform(0, 90, len(dl2_events)),
            ("on", "gammaness"): rng.uniform(0, 1, len(dl2_events)),
            ("on", "reco_energy"): 10 ** rng.uniform(-2, 0, len(dl2_events)),
        }
    )
    # As written by lstchain: one column per (source position, parameter)
    write_dataframe(
        pd.DataFrame({str(col): values for col, values in srcdep.items()}),
        lstchain_file,
        dl2_params_src_dep_lstcam_key,
    )

    info = ReadLSTFile(file=lstchain_file, src_dependent=True).read_LSTfile(
        lstchain_file
    )
    cosmic = (dl2_events["event_type"] == 32).to_numpy()
    np.testing.assert_allclose(info["alpha"], srcdep[("on", "alpha")][cosmic])
    np.testing.assert_allclose(info["gammaness"], srcdep[("on", "gammaness")][cosmic])
    np.testing.assert_allclose(info["energy"], srcdep[("on", "reco_energy")][cosmic])
    np.testing.assert_allclose(
        info["pulsar_phase"], dl2_events["event_id"][cosmic] / len(dl2_events)
    )


def test_read_ctlearn_dl2_parameters_not_aligned(
    tmp_path, dl2_events, write_ctlearn_dl2
):
    path = write_ctlearn_dl2(tmp_path / "dl2.h5", dl2_events[:1000])
    parameters = read_table(path, "/dl1/event/telescope/parameters/tel_001")
    parameters["event_id"] = parameters["event_id"][::-1]
    write_table(
        parameters, path, "/dl1/event/telescope/parameters/tel_001", overwrite=True
    )
    with pytest.raises(ValueError, match="not in the same order"):
        read_ctlearn_dl2(path)
