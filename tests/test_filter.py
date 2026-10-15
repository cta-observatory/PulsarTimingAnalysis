from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ptiming_ana.phaseogram import FilterPulsarAna


@pytest.fixture
def events():
    rng = np.random.default_rng(11)
    n_events = 5000
    info = pd.DataFrame(
        {
            "gammaness": rng.uniform(0, 1, n_events),
            "alpha": rng.uniform(0, 90, n_events),
            "theta2": rng.uniform(0, 1, n_events),
            "intensity": 10 ** rng.uniform(1, 4, n_events),
            "energy": 10 ** rng.uniform(-2, 1, n_events),
            "alt_tel": np.deg2rad(rng.uniform(20, 90, n_events)),
            "dragon_time": np.sort(rng.uniform(0, 1000, n_events)),
        }
    )
    return SimpleNamespace(info=info)


def test_no_cuts(events):
    n_events = len(events.info)
    FilterPulsarAna().apply_fixed_cut(events)
    assert len(events.info) == n_events


def test_fixed_cuts(events):
    cuts = FilterPulsarAna(
        gammaness_cut=0.6,
        alpha_cut=12.0,
        theta2_cut=0.5,
        zd_cut=[10, 50],
        int_cut=50.0,
        energy_cut=[0.05, 2.0],
    )
    assert cuts.use_fixed_cuts()
    cuts.apply_fixed_cut(events)

    info = events.info
    zenith = 90 - np.rad2deg(info["alt_tel"])
    assert 0 < len(info) < 5000
    assert info["gammaness"].min() > 0.6
    assert info["alpha"].max() < 12.0
    assert info["theta2"].max() < 0.5
    assert info["intensity"].min() > 50
    assert info["energy"].min() > 0.05
    assert info["energy"].max() < 2.0
    assert zenith.min() > 10
    assert zenith.max() < 50


@pytest.mark.parametrize(
    "name, column, value, expected",
    [
        ("int_cut", "intensity", 50, "min"),
        ("gammaness_cut", "gammaness", 0, "min"),
        ("alpha_cut", "alpha", 15, "max"),
        ("theta2_cut", "theta2", np.float32(0.25), "max"),
    ],
)
def test_integer_cuts_are_applied(events, name, column, value, expected):
    """Regression test: integer cuts (e.g. intensity: 50) were silently ignored."""
    reference = events.info.copy()
    FilterPulsarAna(**{name: value}).apply_fixed_cut(events)

    if expected == "min":
        mask = reference[column] > value
        assert events.info[column].min() > value
    else:
        mask = reference[column] < value
        assert events.info[column].max() < value
    assert len(events.info) == mask.sum()


def test_scalar_energy_cut_is_the_minimum_energy(events):
    """Regression test: energy: 0 in the configuration file raised a TypeError."""
    n_events = len(events.info)
    FilterPulsarAna(energy_cut=0).apply_fixed_cut(events)
    assert len(events.info) == n_events

    FilterPulsarAna(energy_cut=0.1).apply_fixed_cut(events)
    assert events.info["energy"].min() > 0.1
    assert events.info["energy"].max() > 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gammaness_cut": 1.0},
        {"gammaness_cut": -0.1},
        {"gammaness_cut": [0.5, 1.2]},
        {"alpha_cut": -1},
        {"alpha_cut": [10, -5]},
        {"theta2_cut": -0.01},
        {"theta2_cut": [0.04, -0.04]},
        {"zd_cut": [-5, 50]},
        {"zd_cut": [10, 95]},
    ],
)
def test_invalid_cuts(kwargs):
    with pytest.raises(ValueError, match="no valid"):
        FilterPulsarAna(**kwargs)


def test_energy_dependent_cuts(events):
    reference = events.info.copy()
    cuts = FilterPulsarAna(
        gammaness_cut=[0.8, 0.5],
        theta2_cut=[0.4, 0.2],
        energy_binning_cut=[0.01, 0.1, 10],
    )
    assert not cuts.use_fixed_cuts()

    # Lists are not applied as fixed cuts
    cuts.apply_fixed_cut(events)
    assert len(events.info) == len(reference)

    cuts.apply_energydep_cuts(events)
    low = (
        (reference["energy"] > 0.01)
        & (reference["energy"] < 0.1)
        & (reference["gammaness"] > 0.8)
        & (reference["theta2"] < 0.4)
    )
    high = (
        (reference["energy"] > 0.1)
        & (reference["energy"] < 10)
        & (reference["gammaness"] > 0.5)
        & (reference["theta2"] < 0.2)
    )
    assert len(events.info) == low.sum() + high.sum()
    assert events.info["dragon_time"].is_monotonic_increasing


def test_energy_dependent_cut_only_alpha(events):
    """Regression test: failed if the gammaness cut was not energy-dependent."""
    reference = events.info.copy()
    cuts = FilterPulsarAna(alpha_cut=[10, 5], energy_binning_cut=[0.01, 0.1, 10])
    cuts.apply_energydep_cuts(events)

    expected = (
        (reference["energy"] < 0.1) & (reference["alpha"] < 10)
        | (reference["energy"] > 0.1) & (reference["alpha"] < 5)
    ) & (reference["energy"] > 0.01)
    assert len(events.info) == expected.sum()


def test_energy_dependent_cuts_wrong_length(events):
    cuts = FilterPulsarAna(
        gammaness_cut=[0.8, 0.5, 0.3], energy_binning_cut=[0.01, 0.1, 10]
    )
    with pytest.raises(ValueError, match="one value per energy bin"):
        cuts.apply_energydep_cuts(events)

    with pytest.raises(ValueError, match="No energy binning"):
        FilterPulsarAna(gammaness_cut=0.5).apply_energydep_cuts(events)
