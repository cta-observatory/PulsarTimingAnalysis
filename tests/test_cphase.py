import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.time import Time
from lstchain.io.io import dl2_params_lstcam_key

from ptiming_ana.cphase import (
    DL2_calphase,
    DL3_calphase,
    add_DL2_phase_table,
    add_DL3_phase_table,
    add_fermipulsarphase,
    add_mjd,
    check_ephemeris_validity,
    dl2time_totim,
    merge_dl2_pulsar,
    merge_pulsar_files,
    model_fromephem,
    read_ephemfile,
)
from ptiming_ana.cphase._cli import find_files, str_to_bool
from ptiming_ana.cphase.pulsarphase_cal import (
    _pulsar_file_name,
    create_files,
    interpolate_btoas,
    interpolate_phase,
)

#: Minimal .par ephemeris of the Crab pulsar valid for MJD 60005 - 60036
PAR_EPHEMERIS = """PSR J0534+2200
RAJ 05:34:31.973
DECJ 22:00:52.06
F0 29.58574931
F1 -3.6732e-10
PEPOCH 60020
START 60005
FINISH 60036
UNITS TDB
"""


@pytest.fixture
def par_file(tmp_path):
    path = tmp_path / "crab.par"
    path.write_text(PAR_EPHEMERIS)
    return str(path)


@pytest.mark.parametrize("times", [[60010.5, 60010.6], [60005.0, 60036.0]])
def test_ephemeris_validity_par(par_file, times):
    check_ephemeris_validity(times, par_file)


@pytest.mark.parametrize(
    "times", [[60004.9, 60010.0], [60030.0, 60036.1], [60100.0, 60101.0]]
)
def test_ephemeris_validity_par_out_of_range(par_file, times):
    with pytest.raises(ValueError, match="outside the validity range"):
        check_ephemeris_validity(times, par_file)


def test_ephemeris_validity_no_events(par_file):
    check_ephemeris_validity([], par_file)


def test_ephemeris_validity_par_without_range(tmp_path, caplog):
    """Without START/FINISH the validity cannot be checked: only a warning."""
    path = tmp_path / "no_range.par"
    path.write_text(
        "\n".join(
            line
            for line in PAR_EPHEMERIS.splitlines()
            if not line.startswith(("START", "FINISH"))
        )
    )
    with caplog.at_level(logging.WARNING):
        check_ephemeris_validity([70000.0], str(path))
    assert "cannot be checked" in caplog.text


@pytest.mark.parametrize("times", [[59980.0, 59981.0], [60000.0, 60010.0]])
def test_ephemeris_validity_gro(gro_file, times):
    # The second case covers two lines of the ephemeris file
    check_ephemeris_validity(times, gro_file)


@pytest.mark.parametrize("times", [[59970.0, 59980.0], [60030.0, 60040.0]])
def test_ephemeris_validity_gro_out_of_range(gro_file, times):
    with pytest.raises(ValueError, match="outside the validity range"):
        check_ephemeris_validity(times, gro_file)


def test_create_files_checks_the_ephemeris(par_file, tmp_path):
    times = [60040.0, 60041.0]
    with pytest.raises(ValueError, match="outside the validity range"):
        create_files(times, par_file, None, str(tmp_path / "model.par"))

    # The check can be skipped
    assert create_files(times, par_file, None, "unused", check_validity=False)


def test_read_ephemeris_file(gro_file):
    ephemeris = read_ephemfile(gro_file)

    assert len(ephemeris) == 2
    assert list(ephemeris["START"]) == [59975.0, 60005.0]
    assert list(ephemeris["FINISH"]) == [60005.0, 60036.0]
    assert ephemeris["F0"][1] == pytest.approx(29.58574931)
    assert ephemeris["F1"][1] == "-3.67320D-10"
    assert ephemeris["EPHEM"][0] == "DE200"


def test_model_from_ephemeris(gro_file, tmp_path):
    from pint.models import get_model

    par_file = str(tmp_path / "model.par")
    assert model_fromephem([60010.5, 60010.6], gro_file, par_file) == par_file

    model = get_model(par_file)
    assert model.PSR.value == "0531+21"
    assert float(model.F0.value) == pytest.approx(29.58574931, rel=1e-12)
    assert float(model.F1.value) == pytest.approx(-3.67320e-10, rel=1e-9)
    assert float(model.F2.value) == pytest.approx(1.0e-20, rel=1e-9)
    # The frequency is referred to the integer part of the arrival time
    assert float(model.PEPOCH.value) == pytest.approx(60020.0)
    assert float(model.TZRMJD.value) == pytest.approx(60020.000000118, abs=1e-10)
    assert model.TZRSITE.value == "coe"
    assert model.EPHEM.value == "DE200"
    assert model.RAJ.quantity.deg == pytest.approx(83.63322, abs=1e-5)
    assert model.DECJ.quantity.deg == pytest.approx(22.01446, abs=1e-5)
    assert {
        "AbsPhase",
        "AstrometryEquatorial",
        "Spindown",
        "SolarSystemShapiro",
    } <= set(model.components)

    with open(par_file) as file:
        par_content = file.read()
    assert "IAU2000B" in par_content
    assert "FB90" in par_content


def test_model_from_ephemeris_selects_the_valid_interval(gro_file, tmp_path):
    from pint.models import get_model

    par_file = str(tmp_path / "model.par")
    model_fromephem([59980.0], gro_file, par_file)
    assert float(get_model(par_file).F0.value) == pytest.approx(29.58670154, rel=1e-12)


@pytest.mark.parametrize("mjd", [50000.0, 59974.9, 60036.1])
def test_model_from_ephemeris_out_of_range(gro_file, tmp_path, mjd):
    """
    Regression test: the last line of the ephemeris file was silently used if
    the observation was not covered by the ephemeris.
    """
    with pytest.raises(ValueError, match="No ephemeris available"):
        model_fromephem([mjd], gro_file, str(tmp_path / "model.par"))


def test_create_files(gro_file, tmp_path):
    par_file = str(tmp_path / "model.par")
    tim_file = str(tmp_path / "times.tim")
    times = [60010.5, 60010.6]

    # A .par file is used as it is
    model_test = str(tmp_path / "model_test.par")
    Path(model_test).write_text(PAR_EPHEMERIS)
    assert create_files(times, model_test, None, par_file) == model_test
    assert not os.path.exists(par_file)

    # A .gro file is converted into a .par file
    assert create_files(times, gro_file, tim_file, par_file) == par_file
    assert os.path.exists(par_file)
    assert os.path.exists(tim_file)

    with pytest.raises(ValueError, match="not valid"):
        create_files(times, "ephemeris.txt", None, par_file)


def test_tim_file(tmp_path):
    tim_file = tmp_path / "times.tim"
    dl2time_totim([60010.123456789, 60010.2], name=str(tim_file), obs="lst")

    lines = tim_file.read_text().splitlines()
    assert lines[0].strip() == "FORMAT 1"
    assert lines[1].split() == ["lst", "0.0", "60010.123456789", "0.0", "lst"]
    assert len(lines) == 3


def test_add_mjd():
    events = pd.DataFrame({"dragon_time": [1678400000.0, 1678400001.0]})
    mjd = add_mjd(events)

    expected = Time(events["dragon_time"], format="unix", scale="utc").mjd
    np.testing.assert_allclose(mjd, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(events["mjd_time"], expected, rtol=0, atol=1e-9)
    assert (mjd[1] - mjd[0]) * 86400 == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    "file, expected",
    [
        ("dl3_LST-1.Run07253.fits", "dl3_LST-1.Run07253_pulsar.fits"),
        ("data/dl3_LST-1.Run07253.fits.gz", "dl3_LST-1.Run07253_pulsar.fits"),
        ("fermi_events.fits", "fermi_events_pulsar.fits"),
    ],
)
@pytest.mark.parametrize("output_dir", ["output", "output/"])
def test_output_file_name(file, expected, output_dir):
    """Regression test: the directory and the file name were not joined."""
    assert _pulsar_file_name(file, output_dir) == os.path.join("output", expected)


def test_output_directory_is_needed():
    with pytest.raises(ValueError, match="No output directory"):
        _pulsar_file_name("dl3.fits", None)
    with pytest.raises(ValueError, match="No output directory"):
        DL3_calphase("dl3.fits", "model.par", None)


def test_interpolation():
    sample = np.array([0.0, 10.0, 20.0])
    np.testing.assert_allclose(
        interpolate_phase([5.0, 15.0, 25.0], sample, np.array([0.0, 100.0, 200.0])),
        [50, 150, 250],
    )
    np.testing.assert_allclose(
        interpolate_btoas([5.0, 15.0], sample, sample + 0.001), [5.001, 15.001]
    )


@pytest.mark.parametrize(
    "value, expected",
    [("True", True), ("yes", True), ("1", True), ("false", False), ("N", False)],
)
def test_str_to_bool(value, expected):
    assert str_to_bool(value) is expected


def test_str_to_bool_wrong_value():
    assert str_to_bool(True) is True
    with pytest.raises(argparse.ArgumentTypeError):
        str_to_bool("maybe")


def test_find_files(tmp_path):
    for name in ["dl3_LST-1.Run07253.fits", "dl3_LST-1.Run07254.fits", "notes.txt"]:
        (tmp_path / name).write_text("")
    (tmp_path / "Run07253_subdirectory").mkdir()

    assert len(find_files(str(tmp_path))) == 3
    assert find_files(str(tmp_path), "07254") == [
        str(tmp_path / "dl3_LST-1.Run07254.fits")
    ]
    assert find_files(str(tmp_path), "07253") == [
        str(tmp_path / "dl3_LST-1.Run07253.fits")
    ]
    with pytest.raises(FileNotFoundError, match="run number 9999"):
        find_files(str(tmp_path), "9999")

    # Selection by extension
    assert find_files(str(tmp_path), extensions=".txt") == [str(tmp_path / "notes.txt")]
    assert len(find_files(str(tmp_path), extensions=(".fits", ".txt"))) == 3
    with pytest.raises(FileNotFoundError, match="with extension .h5"):
        find_files(str(tmp_path), extensions=".h5")


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ([], False),
        (["--pickle"], True),
        (["--pickle", "True"], True),
        (["--pickle", "False"], False),
    ],
)
def test_boolean_options(arguments, expected):
    """Regression test: with type=bool, '--pickle False' was parsed as True."""
    parser = add_DL3_phase_table.get_parser()
    args = parser.parse_args(
        ["-f", "dl3.fits", "--ephem", "model.par", "--output", "out"] + arguments
    )
    assert args.pickle is expected


def test_dl3_command_line_options():
    args = add_DL3_phase_table.get_parser().parse_args(
        [
            "--dir",
            "DL3",
            "--run-number",
            "07253",
            "--ephem",
            "model.par",
            "--output",
            "out",
            "--interpolation",
            "--number-interpolation",
            "500",
        ]
    )
    assert args.directory == "DL3"
    assert args.run == "07253"
    assert args.interpolation is True
    assert args.ninterp == 500
    assert args.observatory == "lst"
    assert args.create_tim is False

    # The run number is optional
    args = add_DL3_phase_table.get_parser().parse_args(
        ["--dir", "DL3", "--ephem", "model.par", "--output", "out"]
    )
    assert args.run is None


@pytest.mark.parametrize(
    "module, arguments",
    [
        (add_DL3_phase_table, ["--output", "out", "--ephem", "model.par"]),
        (add_DL3_phase_table, ["-f", "dl3.fits", "--output", "out"]),
        (add_DL3_phase_table, ["-f", "dl3.fits", "--ephem", "model.par"]),
        (
            add_DL3_phase_table,
            ["-f", "a.fits", "-d", "DL3", "--ephem", "m.par", "--output", "out"],
        ),
        (add_DL2_phase_table, ["--ephem", "model.par"]),
        (add_DL2_phase_table, ["-f", "dl2.h5"]),
        (add_fermipulsarphase, ["-f", "ft1.fits", "--ephem", "m.par", "-out", "out"]),
        (merge_pulsar_files, ["--dir", "DL2"]),
    ],
)
def test_command_line_missing_arguments(module, arguments):
    with pytest.raises(SystemExit) as error:
        module.main(arguments)
    assert error.value.code == 2


def test_dl2_command_line_options():
    args = add_DL2_phase_table.get_parser().parse_args(
        ["-f", "dl2.h5", "--ephem", "model.par", "--include-theta", "-s", "Geminga"]
    )
    assert args.include_theta is True
    assert args.source_name == "Geminga"
    assert args.interpolation is False

    args = add_DL2_phase_table.get_parser().parse_args(["-f", "a.h5", "--ephem", "m"])
    assert args.source_name == "Crab"


def test_merge_dl2_files(tmp_path, dl2_dataframe):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    half = len(dl2_dataframe) // 2
    for subrun, events in enumerate([dl2_dataframe[:half], dl2_dataframe[half:]]):
        events.to_hdf(
            input_dir / f"dl2_LST-1.Run00001.{subrun:04d}_pulsar.h5",
            key=dl2_params_lstcam_key,
        )
    # File of another run that must not be merged
    dl2_dataframe[:10].to_hdf(
        input_dir / "dl2_LST-1.Run00002.0000_pulsar.h5", key=dl2_params_lstcam_key
    )

    merge_dl2_pulsar(str(input_dir), "Run00001", str(output_dir))

    merged = pd.read_hdf(
        output_dir / "dl2_LST-1.Run00001_pulsar.h5", key=dl2_params_lstcam_key
    )
    assert len(merged) == len(dl2_dataframe)
    np.testing.assert_array_equal(merged["event_id"], dl2_dataframe["event_id"])
    np.testing.assert_allclose(merged["dragon_time"], dl2_dataframe["dragon_time"])

    with pytest.raises(FileNotFoundError, match="No files found"):
        merge_dl2_pulsar(str(input_dir), "Run00003", str(output_dir))


@pytest.fixture
def small_dl2_file(tmp_path, dl2_dataframe):
    path = tmp_path / "dl2_LST-1.Run00001.h5"
    # A subsample of events spanning the whole file is enough
    dl2_dataframe[::15].to_hdf(path, key=dl2_params_lstcam_key)
    return str(path)


@pytest.mark.network
def test_dl2_phases(small_dl2_file, gro_file, tmp_path, monkeypatch):
    """Phases of DL2 events: exact calculation and interpolation method."""
    monkeypatch.chdir(tmp_path)
    events = pd.read_hdf(small_dl2_file, key=dl2_params_lstcam_key)

    DL2_calphase(small_dl2_file, gro_file, use_interpolation=False)
    exact = pd.read_hdf(small_dl2_file, key="phase_info")

    assert list(exact.columns) == [
        "obs_id",
        "event_id",
        "mjd_barycenter_time",
        "pulsar_phase",
    ]
    np.testing.assert_array_equal(exact["event_id"], events["event_id"])
    assert exact["pulsar_phase"].between(-0.5, 0.5).all()

    # The barycentric correction is smaller than the light travel time
    # across the orbit of the Earth (~500 s)
    mjd = Time(events["dragon_time"].to_numpy(), format="unix", scale="utc").mjd
    delay = (exact["mjd_barycenter_time"].to_numpy() - mjd) * 86400
    assert np.all(np.abs(delay) < 600)
    assert np.all(np.diff(exact["mjd_barycenter_time"]) > 0)

    # The phase advances at the rotation frequency of the pulsar
    turns = np.diff(exact["mjd_barycenter_time"]) * 86400 * 29.58574931
    phase_difference = np.diff(exact["pulsar_phase"])
    residual = (turns - phase_difference + 0.5) % 1 - 0.5
    assert np.all(np.abs(residual) < 1e-3)

    # The interpolation method reproduces the exact phases
    DL2_calphase(small_dl2_file, gro_file, use_interpolation=True, n_interp=200)
    interpolated = pd.read_hdf(small_dl2_file, key="phase_info")
    difference = interpolated["pulsar_phase"] - exact["pulsar_phase"]
    difference = (difference + 0.5) % 1 - 0.5
    assert np.all(np.abs(difference) < 1e-3)
    np.testing.assert_allclose(
        interpolated["mjd_barycenter_time"],
        exact["mjd_barycenter_time"],
        rtol=0,
        atol=1e-4 / 86400,
    )

    # The temporary .tim and .par files are removed
    assert [
        name for name in os.listdir(tmp_path) if name.endswith((".tim", ".par"))
    ] == []


@pytest.fixture
def dl3_file(tmp_path):
    """Minimal DL3 file: EVENTS and GTI extensions."""
    rng = np.random.default_rng(17)
    n_events = 1500
    reference = Time("2018-10-01", scale="utc")
    start = (Time(60012.0, format="mjd", scale="utc") - reference).sec
    # The events are not sorted in time on purpose
    time = rng.permutation(start + np.cumsum(rng.exponential(1.0, n_events)))

    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="EVENT_ID", format="K", array=np.arange(n_events)),
            fits.Column(name="TIME", format="D", unit="s", array=time),
            fits.Column(
                name="RA", format="E", unit="deg", array=np.full(n_events, 83.6)
            ),
            fits.Column(
                name="DEC", format="E", unit="deg", array=np.full(n_events, 22.0)
            ),
            fits.Column(
                name="ENERGY",
                format="E",
                unit="TeV",
                array=10 ** rng.uniform(-2, 0, n_events),
            ),
        ],
        name="EVENTS",
    )
    events.header["MJDREFI"] = int(reference.mjd)
    events.header["MJDREFF"] = 0.0
    events.header["TIMESYS"] = "UTC"
    events.header["OBS_ID"] = 1

    gti = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", unit="s", array=[time.min()]),
            fits.Column(name="STOP", format="D", unit="s", array=[time.max()]),
        ],
        name="GTI",
    )

    path = tmp_path / "dl3_LST-1.Run00001.fits"
    fits.HDUList([fits.PrimaryHDU(), events, gti]).writeto(path)
    return str(path)


@pytest.mark.network
@pytest.mark.parametrize("use_interpolation", [False, True])
def test_dl3_phases(dl3_file, gro_file, tmp_path, monkeypatch, use_interpolation):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "DL3_pulsar"

    DL3_calphase(
        dl3_file,
        gro_file,
        str(output_dir),
        use_interpolation=use_interpolation,
        n_interp=200,
    )

    output_file = output_dir / "dl3_LST-1.Run00001_pulsar.fits"
    assert output_file.exists()

    with fits.open(dl3_file) as original, fits.open(output_file) as result:
        assert [hdu.name for hdu in result] == ["PRIMARY", "EVENTS", "GTI"]
        events = result["EVENTS"].data
        header = result["EVENTS"].header
        n_events = len(original["EVENTS"].data)
        original_ids = np.array(original["EVENTS"].data["EVENT_ID"])
        original_time = np.array(original["EVENTS"].data["TIME"])
        original_columns = original["EVENTS"].columns.names

        assert result["EVENTS"].columns.names == original_columns + [
            "PHASE",
            "BARYCENT_TIME",
        ]
        assert header["MJDREFI"] == 58392
        assert header["TIMESYS"] == "UTC"
        assert header["OBS_ID"] == 1
        assert result["EVENTS"].columns["TIME"].unit == "s"

        assert len(events) == n_events
        # The events are sorted in time, keeping the rows consistent
        assert np.all(np.diff(events["TIME"]) > 0)
        order = np.argsort(original_time)
        np.testing.assert_array_equal(events["EVENT_ID"], original_ids[order])
        np.testing.assert_allclose(events["TIME"], original_time[order])

        assert np.all((events["PHASE"] >= 0) & (events["PHASE"] < 1))
        mjd = 58392 + events["TIME"] / 86400
        assert np.all(np.abs(events["BARYCENT_TIME"] - mjd) * 86400 < 600)

        # The phase advances at the rotation frequency of the pulsar
        turns = np.diff(events["BARYCENT_TIME"]) * 86400 * 29.58574931
        residual = (turns - np.diff(events["PHASE"]) + 0.5) % 1 - 0.5
        assert np.all(np.abs(residual) < 2e-3)

    assert [
        name for name in os.listdir(tmp_path) if name.endswith((".tim", ".par"))
    ] == []
