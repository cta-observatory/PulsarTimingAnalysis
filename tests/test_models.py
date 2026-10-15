import inspect

import numpy as np
import pytest
from scipy.integrate import trapezoid

from ptiming_ana.phaseogram import models
from ptiming_ana.phaseogram.models import MODELS, get_model_list

X = np.linspace(-2, 3, 200001)


def test_model_list():
    assert get_model_list() == list(MODELS)
    assert set(get_model_list()) == {
        "gaussian",
        "dgaussian",
        "double_lorentz",
        "asym_dgaussian",
        "tgaussian",
        "lorentzian",
    }


@pytest.mark.parametrize("name", get_model_list())
def test_registry_matches_function_signature(name):
    spec = MODELS[name]
    function = getattr(spec["function"], "py_func", spec["function"])
    n_function_params = len(inspect.signature(function).parameters) - 1

    assert len(spec["parnames"]) == n_function_params
    assert len(set(spec["parnames"])) == len(spec["parnames"])
    assert spec["background"] in spec["parnames"]
    assert sum(p.startswith("mu") for p in spec["parnames"]) == spec["npeaks"]


def test_gaussian_normalisation():
    """The amplitude is the integral of the peak over the phase."""
    y = models.gaussian(X, 0.4, 0.03, 0.0, 7.0)
    assert trapezoid(y, X) == pytest.approx(7.0, rel=1e-6)
    assert X[np.argmax(y)] == pytest.approx(0.4, abs=1e-4)


def test_gaussian_background():
    y = models.gaussian(X, 0.4, 0.03, 5.0, 0.0)
    np.testing.assert_allclose(y, 5.0)


def test_double_gaussian_is_sum_of_gaussians():
    expected = (
        2.0
        + models.gaussian(X, 0.2, 0.02, 0.0, 3.0)
        + models.gaussian(X, 0.6, 0.05, 0.0, 4.0)
    )
    result = models.double_gaussian(X, 0.2, 0.02, 0.6, 0.05, 2.0, 3.0, 4.0)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_triple_gaussian_normalisation():
    y = models.triple_gaussian(X, 0.0, 0.1, 0.02, 0.4, 0.03, 0.7, 0.04, 1.0, 2.0, 3.0)
    assert trapezoid(y, X) == pytest.approx(6.0, rel=1e-6)


def test_asymmetric_gaussian_pdf():
    # Normalised to one, with the maximum at mu
    y = models.assymetric_gaussian_pdf(X, 0.5, 0.02, 0.06)
    assert trapezoid(y, X) == pytest.approx(1.0, rel=1e-5)
    assert X[np.argmax(y)] == pytest.approx(0.5, abs=1e-4)

    # The falling edge is wider than the rising edge
    half_maximum = X[y >= y.max() / 2]
    assert (half_maximum.max() - 0.5) == pytest.approx(
        3 * (0.5 - half_maximum.min()), rel=1e-2
    )


def test_asymmetric_gaussian_reduces_to_gaussian():
    symmetric = models.assymetric_double_gaussian(
        X, 0.2, 0.03, 0.03, 0.6, 0.05, 0.05, 1.0, 2.0, 3.0
    )
    expected = models.double_gaussian(X, 0.2, 0.03, 0.6, 0.05, 1.0, 2.0, 3.0)
    np.testing.assert_allclose(symmetric, expected, rtol=1e-10)


def test_asymmetric_gaussian_accepts_scalars():
    value = models.assymetric_double_gaussian(
        0.2, 0.2, 0.03, 0.03, 0.6, 0.05, 0.05, 1.0, 2.0, 3.0
    )
    expected = models.double_gaussian(0.2, 0.2, 0.03, 0.6, 0.05, 1.0, 2.0, 3.0)
    assert float(value) == pytest.approx(float(expected))


def test_lorentzian():
    x = np.linspace(-500, 500, 2000001)
    assert trapezoid(models.lorentz_pdf(x, 0.3, 0.02), x) == pytest.approx(
        1.0, rel=1e-4
    )

    # gamma is the half width at half maximum
    y = models.lorentzian(X, 0.3, 0.02, 0.0, 1.0)
    half_maximum = X[y >= y.max() / 2]
    assert half_maximum.max() - half_maximum.min() == pytest.approx(0.04, rel=1e-2)

    np.testing.assert_allclose(
        models.double_lorentz(X, 0.3, 0.02, 0.7, 0.05, 1.5, 2.0, 0.0),
        models.lorentzian(X, 0.3, 0.02, 1.5, 2.0),
        rtol=1e-10,
    )
