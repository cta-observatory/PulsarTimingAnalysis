"""Analytical models used to describe the pulse profile (phaseogram).

All the models return the expected number of counts per phase bin as a
function of the pulsar phase, i.e. a constant background level plus one or
more peaks whose integral (in phase) is given by the amplitude parameters.
"""

import numba as nb
import numpy as np

__all__ = [
    "MODELS",
    "assymetric_double_gaussian",
    "assymetric_gaussian_pdf",
    "double_gaussian",
    "double_lorentz",
    "gaussian",
    "get_model_list",
    "lorentz_pdf",
    "lorentzian",
    "triple_gaussian",
]


kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def gaussian(x, mu, sigma, A, B):
    return A + B / np.sqrt(2 * np.pi) / sigma * np.exp(
        -((x - mu) ** 2) / 2.0 / sigma**2
    )


@nb.njit(**kwd)
def double_gaussian(x, mu, sigma, mu_2, sigma_2, A, B, C):
    return (
        A
        + B / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2)
        + C
        / (2 * np.pi) ** (1 / 2)
        / sigma_2
        * np.exp(-((x - mu_2) ** 2) / 2.0 / sigma_2**2)
    )


@nb.njit(**kwd)
def triple_gaussian(x, Bkg, mu, sigma, mu_2, sigma_2, mu_3, sigma_3, A, B, C):
    return (
        Bkg
        + A / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2)
        + B
        / np.sqrt(2 * np.pi)
        / sigma_2
        * np.exp(-((x - mu_2) ** 2) / 2.0 / sigma_2**2)
        + C
        / np.sqrt(2 * np.pi)
        / sigma_3
        * np.exp(-((x - mu_3) ** 2) / 2.0 / sigma_3**2)
    )


@nb.njit(**kwd)
def assymetric_gaussian_pdf(x, mu, sigma1, sigma2):
    """Normalised two-sided Gaussian: width sigma1 below mu and sigma2 above."""
    sigma = np.where(x <= mu, sigma1, sigma2)
    return (
        2
        / np.sqrt(2 * np.pi)
        / (abs(sigma1) + abs(sigma2))
        * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2)
    )


@nb.njit(**kwd)
def assymetric_double_gaussian(
    x, mu, sigma1, sigma2, mu_2, sigma1_2, sigma2_2, A, B, C
):
    return (
        A
        + B * assymetric_gaussian_pdf(x, mu, sigma1, sigma2)
        + C * assymetric_gaussian_pdf(x, mu_2, sigma1_2, sigma2_2)
    )


@nb.njit(**kwd)
def lorentz_pdf(x, mu, gamma):
    return 1 / (np.pi * gamma) * (gamma**2) / ((x - mu) ** 2 + gamma**2)


@nb.njit(**kwd)
def lorentzian(x, mu, gamma, A, B):
    return A + B / (np.pi * gamma) * (gamma**2) / ((x - mu) ** 2 + gamma**2)


@nb.njit(**kwd)
def double_lorentz(x, mu_1, gamma_1, mu_2, gamma_2, A, B, C):
    return A + B * lorentz_pdf(x, mu_1, gamma_1) + C * lorentz_pdf(x, mu_2, gamma_2)


#: Registry of the available models. For each model: the function, the name of
#: its parameters (in the order of the function signature), the name of the
#: parameter describing the constant background level and the number of peaks.
MODELS = {
    "gaussian": {
        "function": gaussian,
        "parnames": ["mu", "sigma", "A", "B"],
        "background": "A",
        "npeaks": 1,
    },
    "dgaussian": {
        "function": double_gaussian,
        "parnames": ["mu", "sigma", "mu_2", "sigma_2", "A", "B", "C"],
        "background": "A",
        "npeaks": 2,
    },
    "double_lorentz": {
        "function": double_lorentz,
        "parnames": ["mu_1", "gamma_1", "mu_2", "gamma_2", "A", "B", "C"],
        "background": "A",
        "npeaks": 2,
    },
    "asym_dgaussian": {
        "function": assymetric_double_gaussian,
        "parnames": [
            "mu",
            "sigma1",
            "sigma2",
            "mu_2",
            "sigma1_2",
            "sigma2_2",
            "A",
            "B",
            "C",
        ],
        "background": "A",
        "npeaks": 2,
    },
    "tgaussian": {
        "function": triple_gaussian,
        "parnames": [
            "A",
            "mu",
            "sigma",
            "mu_2",
            "sigma_2",
            "mu_3",
            "sigma_3",
            "B",
            "C",
            "D",
        ],
        "background": "A",
        "npeaks": 3,
    },
    "lorentzian": {
        "function": lorentzian,
        "parnames": ["mu_1", "gamma_1", "A", "B"],
        "background": "A",
        "npeaks": 1,
    },
}


def get_model_list():
    """Return the names of the models available to fit the pulse profile."""
    return list(MODELS)
