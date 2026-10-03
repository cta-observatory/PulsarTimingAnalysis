from numbers import Real

import numpy as np
import pandas as pd

__all__ = ["FilterPulsarAna"]


def _is_number(value):
    return isinstance(value, Real) and not isinstance(value, bool)


class FilterPulsarAna:
    """
    Event selection cuts applied to the DL2 data used in the pulsar analysis.

    Parameters
    ----------
    gammaness_cut : float or list of float, optional
        Minimum gammaness. A list defines one cut per bin of ``energy_binning_cut``.
    alpha_cut : float or list of float, optional
        Maximum alpha (deg), for source-dependent analysis.
    theta2_cut : float or list of float, optional
        Maximum theta2 (deg2), for source-independent analysis.
    zd_cut : list of float, optional
        Zenith distance range ``[zd_min, zd_max]`` in degrees.
    int_cut : float, optional
        Minimum intensity (p.e.).
    energy_cut : float or list of float, optional
        Energy range ``[e_min, e_max]`` in TeV. A single number is interpreted
        as the minimum energy.
    energy_binning_cut : list of float, optional
        Edges of the energy bins (TeV) used for the energy-dependent cuts.
    """

    def __init__(
        self,
        gammaness_cut=None,
        alpha_cut=None,
        theta2_cut=None,
        zd_cut=None,
        int_cut=None,
        energy_cut=None,
        energy_binning_cut=None,
    ):
        self.gammaness_cut = gammaness_cut
        self.alpha_cut = alpha_cut
        self.theta2_cut = theta2_cut
        self.int_cut = int_cut
        self.energy_binning_cut = energy_binning_cut
        self.zd_cut = zd_cut

        if _is_number(energy_cut):
            energy_cut = [energy_cut, None]
        self.energy_cut = energy_cut

        self.zd_cut_min = None
        self.zd_cut_max = None
        if self.zd_cut is not None:
            self.zd_cut_min = zd_cut[0]
            self.zd_cut_max = zd_cut[1]

        if self.energy_binning_cut is None:
            self.check_cuts()

    def apply_fixed_cut(self, events):
        dataframe = events.info

        if _is_number(self.gammaness_cut):
            dataframe = dataframe[dataframe["gammaness"] > self.gammaness_cut]

        if _is_number(self.alpha_cut):
            dataframe = dataframe[dataframe["alpha"] < self.alpha_cut]

        if _is_number(self.theta2_cut):
            dataframe = dataframe[dataframe["theta2"] < self.theta2_cut]

        if _is_number(self.zd_cut_min):
            dataframe = dataframe[
                (90 - np.rad2deg(dataframe["alt_tel"])) > self.zd_cut_min
            ]

        if _is_number(self.zd_cut_max):
            dataframe = dataframe[
                (90 - np.rad2deg(dataframe["alt_tel"])) < self.zd_cut_max
            ]

        if _is_number(self.int_cut):
            dataframe = dataframe[dataframe["intensity"] > self.int_cut]

        if self.energy_cut is not None:
            if _is_number(self.energy_cut[0]):
                dataframe = dataframe[dataframe["energy"] > self.energy_cut[0]]
            if _is_number(self.energy_cut[1]):
                dataframe = dataframe[dataframe["energy"] < self.energy_cut[1]]

        events.info = dataframe

    def check_cuts(self):
        if self.gammaness_cut is not None:
            if _is_number(self.gammaness_cut):
                if self.gammaness_cut >= 1 or self.gammaness_cut < 0:
                    raise ValueError("Gammaness cut no valid")
            elif isinstance(self.gammaness_cut, (list, np.ndarray)):
                for cut in self.gammaness_cut:
                    if cut >= 1 or cut < 0:
                        raise ValueError("Gammaness cut no valid")

        if self.alpha_cut is not None:
            if _is_number(self.alpha_cut):
                if self.alpha_cut < 0:
                    raise ValueError("alpha cut no valid")
            elif isinstance(self.alpha_cut, (list, np.ndarray)):
                for cut in self.alpha_cut:
                    if cut < 0:
                        raise ValueError("alpha cut no valid")

        if self.theta2_cut is not None:
            if _is_number(self.theta2_cut):
                if self.theta2_cut < 0:
                    raise ValueError("Theta2 cut no valid")
            elif isinstance(self.theta2_cut, (list, np.ndarray)):
                for cut in self.theta2_cut:
                    if cut < 0:
                        raise ValueError("Theta2 cut no valid")

        for zd_cut in (self.zd_cut_min, self.zd_cut_max):
            if _is_number(zd_cut) and (zd_cut > 90 or zd_cut < 0):
                raise ValueError("Zenith angle cut no valid")

    def use_fixed_cuts(self):
        for cut in [
            self.gammaness_cut,
            self.alpha_cut,
            self.theta2_cut,
            self.zd_cut_min,
            self.zd_cut_max,
        ]:
            if cut is not None and not _is_number(cut):
                return False
        return True

    def _check_energydep_cut(self, cut, name):
        nbins = len(self.energy_binning_cut) - 1
        if len(cut) != nbins:
            raise ValueError(
                f"The energy-dependent {name} cut needs one value per energy bin "
                f"({nbins}), but {len(cut)} were given"
            )

    def apply_energydep_cuts(self, events):
        if self.energy_binning_cut is None:
            raise ValueError("No energy binning defined for the energy-dependent cuts")

        dataframe_energy = []
        dataframe = events.info

        cuts = {}
        for name, cut in [
            ("gammaness", self.gammaness_cut),
            ("alpha", self.alpha_cut),
            ("theta2", self.theta2_cut),
        ]:
            if isinstance(cut, (list, tuple, np.ndarray)):
                self._check_energydep_cut(cut, name)
                cuts[name] = cut

        for i in range(len(self.energy_binning_cut) - 1):
            global_mask = (dataframe["energy"] > self.energy_binning_cut[i]) & (
                dataframe["energy"] < self.energy_binning_cut[i + 1]
            )

            if "gammaness" in cuts:
                global_mask = global_mask & (
                    dataframe["gammaness"] > cuts["gammaness"][i]
                )

            if "alpha" in cuts:
                global_mask = global_mask & (dataframe["alpha"] < cuts["alpha"][i])

            if "theta2" in cuts:
                global_mask = global_mask & (dataframe["theta2"] < cuts["theta2"][i])

            dataframe_energy.append(dataframe[global_mask])

        events.info = pd.concat(dataframe_energy)
        events.info = events.info.sort_values(by=["dragon_time"])
