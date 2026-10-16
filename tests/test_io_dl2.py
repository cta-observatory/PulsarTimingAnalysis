"""Tests of the reading of lstchain and CTLearn DL2 files."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
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
def test_add_source_info(file, dl2_events, dl2_directions, request):
    """theta2 is computed in the same way for lstchain and CTLearn files."""
    path = request.getfixturevalue(file)
    add_source_info_dl2(path, source_position=CRAB)

    table = pd.read_hdf(path, key="source_position")
    assert list(table.columns) == ["obs_id", "event_id", "theta2"]
    np.testing.assert_array_equal(table["event_id"], dl2_events["event_id"])

    valid = _valid(dl2_events) if file == "ctlearn_file" else slice(None)
    # Precision limited by the interpolation of the coordinate transformation
    np.testing.assert_allclose(
        np.sqrt(table["theta2"][valid]),
        np.sqrt(dl2_directions.theta2[valid]),
        atol=2e-4,
    )

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


def test_read_ctlearn_file(ctlearn_file, dl2_events, dl2_directions):
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
        info["theta2"], dl2_directions.theta2[selected], atol=1e-4
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
    np.testing.assert_allclose(ctlearn["theta2"], lstchain["theta2"], rtol=1e-5)
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
