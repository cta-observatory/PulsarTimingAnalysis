import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ["FilterPulsarAna"]


class FilterPulsarAna:
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
        self.energy_cut = energy_cut
        self.int_cut = int_cut
        self.energy_binning_cut = energy_binning_cut
        self.zd_cut = zd_cut

        if self.zd_cut is not None:
            self.zd_cut_min = zd_cut[0]
            self.zd_cut_max = zd_cut[1]

        if self.energy_binning_cut is None:
            self.check_cuts()

    def apply_fixed_cut(self, events, run_number=None):
        dataframe = events.info

        if self.gammaness_cut is not None:
            if isinstance(self.gammaness_cut, float):
                dataframe = dataframe[dataframe["gammaness"] > self.gammaness_cut]
        if self.alpha_cut is not None:
            if isinstance(self.alpha_cut, int) or isinstance(self.alpha_cut, float):
                dataframe = dataframe[dataframe["alpha"] < self.alpha_cut]
        if self.theta2_cut is not None:
            if isinstance(self.theta2_cut, float) or isinstance(self.theta2_cut, int):
                dataframe = dataframe[dataframe["theta2"] < self.theta2_cut]

        if self.zd_cut is not None:
            if isinstance(self.zd_cut_min, float) or isinstance(self.zd_cut_min, int):
                dataframe = dataframe[
                    (90 - np.rad2deg(dataframe["alt_tel"])) > self.zd_cut_min
                ]

            if isinstance(self.zd_cut_max, float) or isinstance(self.zd_cut_max, int):
                dataframe = dataframe[
                    (90 - np.rad2deg(dataframe["alt_tel"])) < self.zd_cut_max
                ]

        # dataframe = dataframe[(dataframe['r'] >= 0) & (dataframe['r'] <= 1)]
        if self.int_cut is not None:
            if isinstance(self.int_cut, float):
                if run_number is not None and run_number < 6045:
                    int_cut = 80.0
                    logger.info(f"Run {run_number} < 6045, using modified intensity cut: {int_cut}")
                else:
                    int_cut = self.int_cut
                dataframe = dataframe[dataframe["intensity"] > int_cut]

        if self.energy_cut is not None:
            if isinstance(self.energy_cut[0], float):
                dataframe = dataframe[dataframe["energy"] > self.energy_cut[0]]
            if isinstance(self.energy_cut[1], float):
                dataframe = dataframe[dataframe["energy"] < self.energy_cut[1]]

        events.info = dataframe

    def check_cuts(self):
        if self.gammaness_cut is not None:
            if isinstance(self.gammaness_cut, (float, int)):
                if self.gammaness_cut >= 1 or self.gammaness_cut < 0:
                    raise ValueError("Gammaness cut no valid")
            elif isinstance(self.gammaness_cut, (list, np.ndarray)):
                for cut in self.gammaness_cut:
                    if cut >= 1 or cut < 0:
                        raise ValueError("Gammaness cut no valid")

        if self.alpha_cut is not None:
            if isinstance(self.alpha_cut, (float, int)):
                if self.alpha_cut < 0:
                    raise ValueError("alpha cut no valid")
            elif isinstance(self.alpha_cut, (list, np.ndarray)):
                for cut in self.alpha_cut:
                    if cut < 0:
                        raise ValueError("alpha cut no valid")

        if self.theta2_cut is not None:
            if isinstance(self.theta2_cut, (float, int)):
                if self.theta2_cut < 0:
                    raise ValueError("Theta2 cut no valid")
            elif isinstance(self.theta2_cut, (list, np.ndarray)):
                for cut in self.theta2_cut:
                    if cut < 0:
                        raise ValueError("Theta2 cut no valid")

        if self.zd_cut is not None:
            if isinstance(self.zd_cut_min, (float, int)):
                if self.zd_cut_min > 90 or self.zd_cut_min < 0:
                    raise ValueError("Zenith angle cut no valid")

            if isinstance(self.zd_cut_max, (float, int)):
                if self.zd_cut_max > 90 or self.zd_cut_max < 0:
                    raise ValueError("Zenith angle cut no valid")

    def use_fixed_cuts(self):
        for cut in [
            self.gammaness_cut,
            self.alpha_cut,
            self.theta2_cut,
            self.zd_cut_min,
            self.zd_cut_max,
        ]:
            if (
                not isinstance(cut, float)
                and not isinstance(cut, int)
                and cut is not None
            ):
                return False
        return True

    def apply_energydep_cuts(self, events):
        dataframe_energy = []
        dataframe = events.info

        for i in range(0, len(self.energy_binning_cut) - 1):
            if isinstance(self.gammaness_cut, list):
                mask_gammaness = (
                    (dataframe["gammaness"] > self.gammaness_cut[i])
                    & (dataframe["energy"] > self.energy_binning_cut[i])
                    & (dataframe["energy"] < self.energy_binning_cut[i + 1])
                )

                global_mask = mask_gammaness

            if isinstance(self.alpha_cut, list):
                mask_alpha = (
                    (dataframe.alpha < self.alpha_cut[i])
                    & (dataframe["energy"] > self.energy_binning_cut[i])
                    & (dataframe["energy"] < self.energy_binning_cut[i + 1])
                )

                global_mask = global_mask & mask_alpha

            if isinstance(self.theta2_cut, list):
                mask_theta2 = (
                    (dataframe.theta2 < self.theta2_cut[i])
                    & (dataframe["energy"] > self.energy_binning_cut[i])
                    & (dataframe["energy"] < self.energy_binning_cut[i + 1])
                )

                global_mask = global_mask & mask_theta2

            dataframe_energy.append(dataframe[global_mask])

        events.info = pd.concat(dataframe_energy)
        events.info = events.info.sort_values(by=["dragon_time"])

    def apply_energydep_cuts_on_even_odd(self, events):
        dataframe_energy = []
        dataframe = events.info
        mask_even = dataframe["event_id"] % 2 == 0
        mask_odd  = dataframe["event_id"] % 2 != 0

        for i in range(0, len(self.energy_binning_cut) - 1):
            mask_energy_bin = (
                (dataframe["energy"] > self.energy_binning_cut[i])
                & (dataframe["energy"] < self.energy_binning_cut[i + 1])
            )
            for parity_idx, mask_parity in enumerate([mask_even, mask_odd]):
                global_mask = mask_energy_bin & mask_parity

                if isinstance(self.gammaness_cut, list):
                    global_mask = global_mask & (dataframe["gammaness"] > self.gammaness_cut[parity_idx][i])

                if isinstance(self.alpha_cut, list):
                    global_mask = global_mask & (dataframe["alpha"] < self.alpha_cut[parity_idx][i])

                if isinstance(self.theta2_cut, list):
                    global_mask = global_mask & (dataframe["theta2"] < self.theta2_cut[parity_idx][i])

                dataframe_energy.append(dataframe[global_mask])

        events.info = pd.concat(dataframe_energy)
        events.info = events.info.sort_values(by=["dragon_time"])