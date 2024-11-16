import astropy.units as u


def get_kwargs_points(sed_type="e2dnde", color="blue", label="Spectral points"):
    kwargs = {"sed_type": sed_type, "label": label, "color": color}
    return kwargs


def get_kwargs_line(
    sed_type="e2dnde",
    label="Reference Model",
    energy_bounds=[0.02 * u.TeV, 10 * u.TeV],
    yunits=u.Unit("erg cm-2 s-1"),
    color="blue",
    zorder=None,
):
    kwargs = {
        "sed_type": sed_type,
        "energy_bounds": energy_bounds,
        "label": label,
        "yunits": yunits,
        "color": color,
        "zorder": zorder,
    }
    return kwargs


def get_kwargs_region(
    sed_type="e2dnde",
    label="Reference Model",
    energy_bounds=[0.02 * u.TeV, 10 * u.TeV],
    yunits=u.Unit("erg cm-2 s-1"),
    color="blue",
    facecolor="blue",
    alpha=1,
    hatch=None,
):
    kwargs = {
        "sed_type": sed_type,
        "energy_bounds": energy_bounds,
        "label": label,
        "yunits": yunits,
        "color": color,
        "facecolor": facecolor,
        "hatch": hatch,
        "alpha": alpha,
    }
    return kwargs
