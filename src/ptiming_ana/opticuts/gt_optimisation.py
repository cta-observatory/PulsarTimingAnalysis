"""
gt_optimisation.py
----------------
Class to load 4D histograms from pickles and optimize gammaness + theta2 cuts
per energy bin using Li&Ma significance on pulsar phase regions.

Usage:
    optimizer = PulsarCutOptimizer(pickle_dir="/path/to/pickles/")
    results = optimizer.optimize(metric="P1P2")
    optimizer.plot_results(results)
"""

import os
import glob
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

LOG_FORMAT = "%(asctime)s %(levelname)-6s [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Phase region widths
# These must match what was used to build the histograms
DEFAULT_P1_WIDTH  = (0.026 - 0.0) + (1.0 - 0.983)   # [0, 0.026] + [0.983, 1]
DEFAULT_P2_WIDTH  = 0.422 - 0.377                     # [0.377, 0.422]
DEFAULT_OFF_WIDTH = 0.87 - 0.52                       # [0.52, 0.87]


def lima_significance(Non, Noff, alpha):
    """
    Li & Ma significance
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = Non  * np.log((1 + alpha) / alpha * Non  / (Non + Noff))
        term2 = Noff * np.log((1 + alpha)          * Noff / (Non + Noff))
        result = np.sqrt(2) * np.sqrt(np.where((term1 + term2) > 0, term1 + term2, 0))
    result = np.where((Non > 0) & (Noff > 0), result, 0.0)
    return result

def signal_to_noise_ratio(Non, Noff, alpha):
    """S/N ratio: excess / sqrt(background)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = Non - alpha * Noff
        bkg = alpha * Noff
        snr = np.where(bkg > 0, excess / np.sqrt(bkg), 0.0)
    return snr

class PulsarCutOptimizer:
    """
    Load 4D histograms from pickles and optimize gammaness + theta2 cuts
    per energy bin using Li&Ma significance on pulsar phase regions.

    Parameters
    ----------
    pickle_dir : str
        Directory containing the histogram pickle files.
    min_excess_fraction : float
        Minimum excess as fraction of OFF events (default 0.05 = 5%).
    min_signal_events : float
        Minimum number of excess events (default 10).
    p1_width : float
        Width of P1 phase region in phase units.
    p2_width : float
        Width of P2 phase region in phase units.
    off_width : float
        Width of OFF phase region in phase units.
    """

    def __init__(
        self,
        pickle_dir,
        min_excess_fraction=0.05,
        min_signal_events=10,
        min_on_events=3,
        min_off_events=3,
        peak_extractor="raw",
        p1_width=DEFAULT_P1_WIDTH,
        p2_width=DEFAULT_P2_WIDTH,
        off_width=DEFAULT_OFF_WIDTH,
    ):
        self.pickle_dir         = pickle_dir
        self.pkl_data = [] 
        self.min_excess_fraction = min_excess_fraction
        self.min_signal_events  = min_signal_events
        self.min_on_events      = min_on_events
        self.min_off_events     = min_off_events
        self.alpha_p1   = p1_width  / off_width
        self.alpha_p2   = p2_width  / off_width
        self.alpha_p1p2 = (p1_width + p2_width) / off_width
        self.peak_extractor = peak_extractor
        # Will be filled by _load_histograms
        self.cum_on_p1  = None
        self.cum_on_p2  = None
        self.cum_on_p3  = None
        self.cum_off    = None
        self.livetime   = None
        self.gammaness_bins = None
        self.theta2_bins    = None
        self.logenergy_bins = None
        self.energy_edges   = None   # in TeV


        self._load_histograms()

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def _load_histograms(self):
        """Load and sum all pickle files in pickle_dir."""
        pkl_files = sorted(glob.glob(os.path.join(self.pickle_dir, "*.pkl")))
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No pickle files found in {self.pickle_dir}")

        logger.info(f"Loading {len(pkl_files)} pickle files from {self.pickle_dir}")

        total_livetime = 0.0
        on_p1 = on_p2 = on_p3 = off = None
        on_p1_i = on_p2_i = on_p3_i = off_i = None

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                self.pkl_data.append(data)
            if on_p1 is None:
                on_p1 = data["on_p1"].astype("float64")
                on_p2 = data["on_p2"].astype("float64")
                on_p3 = data["on_p3"].astype("float64") if data["on_p3"] is not None else None
                off   = data["off"].astype("float64")
                self.gammaness_bins = data["bins"]["gammaness"]
                self.theta2_bins    = data["bins"]["theta2"]
                self.logenergy_bins = data["bins"]["logenergy"]
                self.intensity_bins = data["bins"]["intensity"]
            else:
                on_p1 += data["on_p1"]
                on_p2 += data["on_p2"]
                if on_p3 is not None and data["on_p3"] is not None:
                    on_p3 += data["on_p3"]
                off   += data["off"]
            if "on_p1_intensity" in data:
                if on_p1_i is None:
                    on_p1_i = data["on_p1_intensity"].astype("float64")
                    on_p2_i = data["on_p2_intensity"].astype("float64")
                    on_p3_i = data["on_p3_intensity"].astype("float64") if data["on_p3_intensity"] is not None else None
                    off_i   = data["off_intensity"].astype("float64")
                else:
                    on_p1_i += data["on_p1_intensity"]
                    on_p2_i += data["on_p2_intensity"]
                    off_i   += data["off_intensity"]
                    if on_p3_i is not None and data["on_p3_intensity"] is not None:
                        on_p3_i += data["on_p3_intensity"]
            total_livetime += data["livetime"]

        self.livetime = total_livetime
        self.energy_edges = 10 ** self.logenergy_bins   # TeV

        logger.info(f"Total livetime: {self.livetime:.2f} h")
        logger.info(f"Histogram shape: {on_p1.shape}  (evtid, energy, gammaness, theta2)")

        # Build cumulative histograms:
        # cum[g, :] = events with gammaness > g
        # cum[:, t] = events with theta2 < t
        assert np.all(np.diff(self.gammaness_bins) > 0)
        assert np.all(np.diff(self.theta2_bins) > 0)
        self.cum_on_p1 = self._make_cumulative(on_p1)
        self.cum_on_p2 = self._make_cumulative(on_p2)
        self.cum_on_p3 = self._make_cumulative(on_p3) if on_p3 is not None else None
        self.cum_off   = self._make_cumulative(off)

        self.cum_on_p1_intensity = self._make_cumulative(on_p1_i) if on_p1_i is not None else None
        self.cum_on_p2_intensity = self._make_cumulative(on_p2_i) if on_p2_i is not None else None
        self.cum_on_p3_intensity = self._make_cumulative(on_p3_i) if on_p3_i is not None else None
        self.cum_off_intensity   = self._make_cumulative(off_i)   if off_i is not None else None

    @staticmethod
    def _make_cumulative(hist):
        """
        Build cumulative histogram:
        - axis 2 (gammaness): integrate from each value to 1  → gammaness > cut
        - axis 3 (theta2):    integrate from 0 to each value  → theta2 < cut
        Shape stays (2, n_energy, n_gammaness, n_theta2).
        """
        cum = hist.copy()
        cum = np.cumsum(hist[:, :, ::-1, :], axis=2)[:, :, ::-1, :]
        cum = np.cumsum(cum, axis=3)

        return cum

   # ------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------

    def optimize(self, metric="P1P2", significance="lima", energy_mode="per_bin", energy_ranges=None, fixed_theta2=None, bins="energy"):
        """
        Parameters
        ----------
        metric : str
            "P1", "P2", "P1P2"
        significance : str
            "snr"  → excess / sqrt(alpha * Noff)
            "lima" → Li&Ma significance
        energy_mode : str
            "per_bin"    → un corte por bin de energía (comportamiento original)
            "integrated" → suma todos los bins, un solo corte
            "bands"      → suma bins dentro de rangos definidos en energy_ranges
        energy_ranges : list of (emin, emax) in TeV, solo para energy_mode="bands"
            e.g. [(0.025, 0.1), (0.1, 0.4), (0.4, 1.6)]
        """
        metric = metric.upper()
        self.fixed_theta2 = fixed_theta2
        if metric not in ("P1", "P2", "P1P2"):
            raise ValueError("metric must be one of 'P1', 'P2', 'P1P2'")

        logger.info(f"Optimizing cuts for metric: {metric}")

        if metric == "P1" and bins == "energy":
            cum_on = self.cum_on_p1
            alpha  = self.alpha_p1
        elif metric == "P1" and bins == "intensity":
            cum_on = self.cum_on_p1_intensity
            alpha  = self.alpha_p1
        elif metric == "P2" and bins == "energy":
            cum_on = self.cum_on_p2
            alpha  = self.alpha_p2
        elif metric == "P2" and bins == "intensity":
            cum_on = self.cum_on_p2_intensity
            alpha  = self.alpha_p2
        elif metric == "P1P2" and bins == "energy":
            cum_on = self.cum_on_p1 + self.cum_on_p2
            alpha  = self.alpha_p1p2
        elif metric == "P1P2" and bins == "intensity":
            cum_on = self.cum_on_p1_intensity + self.cum_on_p2_intensity
            alpha  = self.alpha_p1p2
        if bins == "energy":
            self.cum_off_used = self.cum_off
        else:
            self.cum_off_used = self.cum_off_intensity

        if bins == "energy":
            self.binning = self.energy_edges
            self.unit = "TeV"
        elif bins == "intensity":
            self.binning = self.intensity_bins
            self.unit = "pe"
        n_energy = len(self.binning) - 1
        n_g      = len(self.gammaness_bins) - 1
        n_t      = len(self.theta2_bins)    - 1

        if fixed_theta2 is not None:
            t_idx = np.argmin(np.abs(self.theta2_bins[:-1] - fixed_theta2))
            logger.info(
                f"Fixed theta2 cut: requested {fixed_theta2:.4f} deg², "
                f"using bin {t_idx} = {self.theta2_bins[t_idx]:.4f} deg²"
            )

        # Construir bandas de energía
        if energy_mode == "integrated":
            bands      = [("integrated", list(range(n_energy)))]
        elif energy_mode == "bands":
            if energy_ranges is None:
                raise ValueError("energy_ranges required for energy_mode='bands'")
            # Para cada bin de energía, calcular solapamiento con cada banda
            # y asignarlo a la banda con mayor solapamiento
            bin_to_band = {}
            for i in range(n_energy):
                ebin_min = self.binning[i]
                ebin_max = self.binning[i + 1]
                ebin_width = ebin_max - ebin_min

                best_band = None
                best_overlap = 0.0
                for band_idx, (emin_r, emax_r) in enumerate(energy_ranges):
                    overlap = max(0.0, min(ebin_max, emax_r) - max(ebin_min, emin_r))
                    fraction = overlap / ebin_width
                    if fraction > best_overlap:
                        best_overlap = fraction
                        best_band = band_idx

                if best_band is not None and best_overlap > 0:
                    bin_to_band.setdefault(best_band, []).append(i)

            bands = []
            for band_idx, (emin_r, emax_r) in enumerate(energy_ranges):
                idxs = bin_to_band.get(band_idx, [])
                if len(idxs) == 0:
                    logger.warning(f"No bins assigned to [{emin_r:.3f}, {emax_r:.3f}] {self.unit}, skipping")
                    continue
                emin_actual = self.binning[idxs[0]]
                emax_actual = self.binning[idxs[-1] + 1]
                label = f"{emin_actual:.3f}-{emax_actual:.3f} {self.unit}"
                logger.info(f"Band requested [{emin_r:.3f}, {emax_r:.3f}] {self.unit} → "
                            f"actual [{emin_actual:.3f}, {emax_actual:.3f}] {self.unit} "
                            f"({len(idxs)} bins: {idxs})")
                bands.append((label, idxs))
        else:  # per_bin
            bands = [(f"{self.binning[i]:.3f}-{self.binning[i+1]:.3f} {self.unit}", [i])
                    for i in range(n_energy)]

        n_bands = len(bands)

        gammaness_even = np.full(n_bands, np.nan)
        gammaness_odd  = np.full(n_bands, np.nan)
        theta2_even    = np.full(n_bands, np.nan)
        theta2_odd     = np.full(n_bands, np.nan)
        n_events_even  = np.zeros(n_bands)
        n_events_odd   = np.zeros(n_bands)

        if fixed_theta2 is not None:
            sigma_map_even = np.full((n_bands, n_g), np.nan)
            sigma_map_odd  = np.full((n_bands, n_g), np.nan)
        else:
            sigma_map_even = np.full((n_bands, n_g, n_t), np.nan)
            sigma_map_odd  = np.full((n_bands, n_g, n_t), np.nan)

        for bidx, (band_label, e_idxs) in enumerate(bands):

            for evtid, (sigma_map, g_out, t_out, n_out) in enumerate([
                (sigma_map_even, gammaness_even, theta2_even, n_events_even),
                (sigma_map_odd,  gammaness_odd,  theta2_odd,  n_events_odd),
            ]):
                # Sumar sobre los bins de energía de esta banda
                if fixed_theta2 is not None:
                    on  = cum_on[evtid, e_idxs, :, t_idx].sum(axis=0)   # (n_g,)
                    off = self.cum_off_used[evtid, e_idxs, :, t_idx].sum(axis=0)
                    n_out[bidx] = on[0] + off[0]
                else:
                    on  = cum_on[evtid, e_idxs, :, :].sum(axis=0)   # (n_g, n_t)
                    off = self.cum_off_used[evtid, e_idxs, :, :].sum(axis=0)
                    n_out[bidx] = on[0, -1] + off[0, -1]

                # on_2d  = cum_on[evtid, e_idxs, :, :].sum(axis=0)   # (n_g, n_t)
                # off_2d = self.cum_off[evtid, e_idxs, :, :].sum(axis=0)
                # n_out[bidx] = on_2d[0, -1] + off_2d[0, -1]
                

                # Significancia
                if significance == "lima":
                    sig = lima_significance(on, off, alpha)
                else:
                    sig = signal_to_noise_ratio(on, off, alpha)
                # Máscaras
                excess  = on - alpha * off
                bkg_est = alpha * off
                mask = (
                    (excess  > self.min_excess_fraction * bkg_est) &
                    (excess  > self.min_signal_events) &
                    (on   > self.min_on_events) &
                    (off  > self.min_off_events)
                )

                sig_masked = np.where(mask, sig, np.nan)
                sigma_map[bidx] = sig_masked

                if np.all(np.isnan(sig_masked)):
                    logger.warning(
                        f"[{'EVEN' if evtid==0 else 'ODD'}] "
                        f"{band_label}: no valid cuts found"
                    )
                    continue

                # # Suavizado gaussiano
                sigma_g_bins = 0.05 / (self.gammaness_bins[1] - self.gammaness_bins[0])
                sigma_t_bins = 0.02 / (self.theta2_bins[1]    - self.theta2_bins[0])
                sig_filled = np.nan_to_num(sig_masked, nan=0.0)
                sig_smooth = gaussian_filter(sig_filled, sigma=[sigma_g_bins, sigma_t_bins], mode='constant', cval=0.0)
                sig_smooth = np.where(mask, sig_smooth, np.nan)

                # Máximo global directo sobre el mapa suavizado
                if fixed_theta2 is not None:
                    best_idx = np.nanargmax(sig_masked)
                    best_g_idx = best_idx
                    best_t_idx = t_idx
                else:
                    best_idx   = np.nanargmax(sig_smooth)
                    best_g_idx, best_t_idx = np.unravel_index(best_idx, sig_smooth.shape)

                best_g   = self.gammaness_bins[best_g_idx]
                best_t   = self.theta2_bins[best_t_idx]
                if fixed_theta2 is not None:
                    best_sig = sig_masked[best_g_idx]
                else:
                    best_sig = sig_masked[best_g_idx, best_t_idx]

                evtlabel = "EVEN" if evtid == 0 else "ODD"
                logger.info(
                    f"[{evtlabel}] {band_label}: "
                    f"gammaness={best_g:.3f}, theta2={best_t:.4f} deg², "
                    f"{significance}={best_sig:.2f}"
                )

                g_out[bidx] = best_g
                t_out[bidx] = best_t

        return {
            "energy_edges":   self.energy_edges,
            "bands":          [b[0] for b in bands],
            "_bands_idxs":    [b[1] for b in bands],
            "gammaness_even": gammaness_even,
            "gammaness_odd":  gammaness_odd,
            "theta2_even":    theta2_even,
            "theta2_odd":     theta2_odd,
            "sigma_map_even": sigma_map_even,
            "sigma_map_odd":  sigma_map_odd,
            "n_events_even":  n_events_even,
            "n_events_odd":   n_events_odd,
            "metric":         metric,
            "mode":           energy_mode,
            "significance":   significance,
            "fixed_theta2":   fixed_theta2,
            "bins_type":    bins,
        }

    # =========================================================================
    # PLOTTING — helpers
    # =========================================================================

    @staticmethod
    def _grid(n):
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        return nrows, ncols

    def _get_cum_on_alpha(self, metric, bins_type="energy"):
        if bins_type == "intensity":
            if metric == "P1":
                return self.cum_on_p1_intensity, self.alpha_p1
            if metric == "P2":
                return self.cum_on_p2_intensity, self.alpha_p2
            return self.cum_on_p1_intensity + self.cum_on_p2_intensity, self.alpha_p1p2
        # energy (default)
        if metric == "P1":
            return self.cum_on_p1, self.alpha_p1
        if metric == "P2":
            return self.cum_on_p2, self.alpha_p2
        return self.cum_on_p1 + self.cum_on_p2, self.alpha_p1p2

    def _savefig(self, fig, output_dir, name):
        if output_dir:
            fig.savefig(os.path.join(output_dir, name), dpi=120, bbox_inches="tight")

    # =========================================================================
    # PLOTTING — plot_results
    # =========================================================================

    def plot_results(self, results, output_dir=None):
        """
        Diagnostic plots after optimize().

        Always produced (per evt parity)
        ---------------------------------
        A  – significance vs gammaness per energy band
             1-D if fixed_theta2, slice at optimal theta2 otherwise
        B  – excess map 2-D (full gammaness × theta2 space, always)
        C  – ON  counts per energy band, EVEN vs ODD
        D  – OFF counts per energy band, EVEN vs ODD
        E  – total events (ON+OFF) per band, EVEN vs ODD bars

        Only when fixed_theta2 is None
        --------------------------------
        A2 – 2-D significance map (smoothed) with optimal point marked
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        metric       = results["metric"]
        mode         = results["mode"]
        fixed_theta2 = results["fixed_theta2"]
        bands        = results["bands"]
        bands_idxs   = results["_bands_idxs"]
        n_bands      = len(bands)
        bins_type = results.get("bins_type", "energy")
        cum_on, alpha = self._get_cum_on_alpha(metric, bins_type)

        g_centers = (self.gammaness_bins[:-1] + self.gammaness_bins[1:]) / 2
        t_centers = (self.theta2_bins[:-1]    + self.theta2_bins[1:])    / 2

        for evtid, label in [(0, "EVEN"), (1, "ODD")]:
            g_opt   = results[f"gammaness_{label.lower()}"]
            t_opt   = results[f"theta2_{label.lower()}"]
            sig_map = results[f"sigma_map_{label.lower()}"]


            if fixed_theta2 is None:
                self._plot_A2_sig_map_2d(
                    sig_map, g_opt, t_opt, g_centers, t_centers,
                    bands, n_bands, metric, label, mode, output_dir,
                )
            else:
                self._plot_A_sig_vs_gammaness(
                    sig_map, g_opt, t_opt, g_centers, t_centers,
                    bands, n_bands, fixed_theta2, metric, label, mode, output_dir,
                )
            self._plot_B_excess_2d(
                cum_on, evtid, alpha, g_opt, t_opt, g_centers, t_centers,
                bands, bands_idxs, n_bands, metric, label, mode, output_dir,
            )

        self._plot_CD_counts_even_odd(
            cum_on, alpha, bands, bands_idxs, n_bands, metric, mode, output_dir,
        )
        self._plot_E_events_per_band(results, cum_on, alpha, bands, bands_idxs, n_bands, mode, output_dir)

    # ── plot_results sub-plots ────────────────────────────────────────────────

    def _plot_A_sig_vs_gammaness(
        self, sig_map, g_opt, t_opt, g_centers, t_centers,
        bands, n_bands, fixed_theta2, metric, label, mode, output_dir,
    ):
        nrows, ncols = self._grid(n_bands)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
        axes = axes.flatten()
        t_desc = (f"theta2 fixed = {fixed_theta2:.4f} deg²" if fixed_theta2 is not None
                  else "theta2 = optimal slice")
        fig.suptitle(f"significance vs gammaness  [{t_desc}] — {metric} — {label}",
                     fontsize=10)

        for bidx in range(n_bands):
            ax  = axes[bidx]
            sig = sig_map[bidx]
            if not np.all(np.isnan(sig)):
                if fixed_theta2 is not None:
                    ax.plot(g_centers, sig, lw=1.8, color="tab:blue")
                else:
                    t_slice = (int(np.argmin(np.abs(t_centers - t_opt[bidx])))
                               if not np.isnan(t_opt[bidx]) else sig.shape[1] // 2)
                    ax.plot(g_centers, sig[:, t_slice], lw=1.8, color="tab:blue",
                            label=f"theta2 = {t_centers[t_slice]:.3f}")
                    ax.legend(fontsize=7)
                ax.axhline(0, color="gray", lw=0.6, ls=":")
                if not np.isnan(g_opt[bidx]):
                    ax.axvline(g_opt[bidx], color="red", lw=1.2, ls="--",
                               label=f"g = {g_opt[bidx]:.3f}")
                    ax.legend(fontsize=7)
            ax.set_title(bands[bidx], fontsize=8)
            ax.set_xlabel("gammaness", fontsize=7)
            ax.set_ylabel("significance", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

        for i in range(n_bands, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        self._savefig(fig, output_dir, f"A_sig_gammaness_{metric}_{label}_{mode}.png")
        plt.show()

    def _plot_A2_sig_map_2d(
        self, sig_map, g_opt, t_opt, g_centers, t_centers,
        bands, n_bands, metric, label, mode, output_dir,
    ):
        sigma_g = 0.05 / (self.gammaness_bins[1] - self.gammaness_bins[0])
        sigma_t = 0.02 / (self.theta2_bins[1]    - self.theta2_bins[0])
        nrows, ncols = self._grid(n_bands)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f"significance map 2-D (gaussian smoothed) — {metric} — {label}", fontsize=10)

        for bidx in range(n_bands):
            ax  = axes[bidx]
            sig = sig_map[bidx]
            if not np.all(np.isnan(sig)):
                filled = np.nan_to_num(sig, nan=0.0)
                smooth = gaussian_filter(filled, sigma=[sigma_g, sigma_t],
                                         mode="constant", cval=0.0)
                smooth = np.where(np.isnan(sig), np.nan, smooth)
                vmax   = max(np.nanpercentile(smooth, 98), 0.1)
                im = ax.pcolormesh(t_centers, g_centers, smooth,
                                   cmap="viridis", vmin=0, vmax=vmax, shading="auto")
                plt.colorbar(im, ax=ax, label="significance", pad=0.02)
            if not np.isnan(g_opt[bidx]) and not np.isnan(t_opt[bidx]):
                ax.plot(t_opt[bidx], g_opt[bidx], "r+", ms=12, mew=2, label="optimum")
                ax.legend(fontsize=7)
            ax.set_title(bands[bidx], fontsize=8)
            ax.set_xlabel("theta2 (deg²)", fontsize=7)
            ax.set_ylabel("gammaness", fontsize=7)
            ax.tick_params(labelsize=7)

        for i in range(n_bands, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        self._savefig(fig, output_dir, f"A2_sigmap2d_{metric}_{label}_{mode}.png")
        plt.show()

    def _plot_B_excess_2d(
        self, cum_on, evtid, alpha, g_opt, t_opt, g_centers, t_centers,
        bands, bands_idxs, n_bands, metric, label, mode, output_dir,
    ):
        """Excess = ON - alpha*OFF in the full 2-D (gammaness x theta2) space."""
        nrows, ncols = self._grid(n_bands)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f"Excess map (ON − α·OFF) — {metric} — {label}", fontsize=10)

        for bidx, e_idxs in enumerate(bands_idxs):
            ax     = axes[bidx]
            on_2d  = cum_on[evtid, e_idxs, :, :].sum(axis=0)
            off_2d = self.cum_off_used[evtid, e_idxs, :, :].sum(axis=0)
            excess = on_2d - alpha * off_2d
            vmax   = max(np.nanpercentile(np.abs(excess), 98), 1.0)
            im = ax.pcolormesh(t_centers, g_centers, excess,
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            plt.colorbar(im, ax=ax, label="excess", pad=0.02)
            if not np.isnan(g_opt[bidx]) and not np.isnan(t_opt[bidx]):
                ax.plot(t_opt[bidx], g_opt[bidx], "r+", ms=12, mew=2, label="optimum")
                ax.legend(fontsize=7)
            ax.set_title(bands[bidx], fontsize=8)
            ax.set_xlabel("theta2 (deg²)", fontsize=7)
            ax.set_ylabel("gammaness", fontsize=7)
            ax.tick_params(labelsize=7)

        for i in range(n_bands, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        self._savefig(fig, output_dir, f"B_excess_{metric}_{label}_{mode}.png")
        plt.show()

    def _plot_CD_counts_even_odd(
        self, cum_on, alpha, bands, bands_idxs, n_bands, metric, mode, output_dir,
    ):
        """
        Plot C: ON  counts per energy band — EVEN vs ODD.
        Plot D: OFF counts per energy band — EVEN vs ODD.
        No gammaness or theta2 cut (loosest selection: g index 0, last t index).
        """
        t_last = len(self.theta2_bins) - 2

        for letter, what, cum_arr in [
            ("C", "ON",  cum_on),
            ("D", "OFF", self.cum_off_used),
        ]:
            nrows, ncols = self._grid(n_bands)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
            axes = axes.flatten()
            fig.suptitle(f"{what} counts per energy band — {metric} — EVEN vs ODD",
                         fontsize=10)

            for bidx, e_idxs in enumerate(bands_idxs):
                ax     = axes[bidx]
                n_even = cum_arr[0, e_idxs, 0, t_last].sum()
                n_odd  = cum_arr[1, e_idxs, 0, t_last].sum()
                bars   = ax.bar([0, 1], [n_even, n_odd],
                                color=["tab:blue", "tab:orange"], alpha=0.8, width=0.6)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["EVEN", "ODD"], fontsize=8)
                ax.set_title(bands[bidx], fontsize=8)
                ax.set_ylabel(f"N {what}", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3, axis="y")
                for bar, n in zip(bars, [n_even, n_odd]):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                            f"{n:.0f}", ha="center", va="bottom", fontsize=7)

            for i in range(n_bands, len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            self._savefig(fig, output_dir, f"{letter}_{what.lower()}_even_odd_{metric}_{mode}.png")
            plt.show()

    def _plot_E_events_per_band(self, results, cum_on, alpha, bands, bands_idxs, n_bands, mode, output_dir):
        metric       = results["metric"]
        t_idx_use    = len(self.theta2_bins) - 2

        fig, axes = plt.subplots(1, 3, figsize=(max(8, n_bands * 1.2) * 3 / 2, 4))
        x = np.arange(n_bands)

        for ax, evtid, label, color in [
            (axes[0], 0, "EVEN", "tab:blue"),
            (axes[1], 1, "ODD",  "tab:orange"),
        ]:
            on_vals  = np.array([cum_on[evtid, e_idxs, 0, t_idx_use].sum()
                                for _, e_idxs in zip(bands, bands_idxs)])
            off_vals = np.array([self.cum_off_used[evtid, e_idxs, 0, t_idx_use].sum()
                                for _, e_idxs in zip(bands, bands_idxs)])

            ax.bar(x - 0.2, on_vals,         width=0.4, color=color,   alpha=0.8, label="ON")
            ax.bar(x + 0.2, alpha * off_vals, width=0.4, color="gray",  alpha=0.5, label="α·OFF")
            ax.set_xticks(x); ax.set_xticklabels(bands, rotation=30, ha="right", fontsize=7)
            ax.set_xlabel("energy band"); ax.set_ylabel("counts")
            ax.set_title(f"ON vs α·OFF — {metric} — {label}")
            ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

        # exceso sin cortes
        ax = axes[2]
        for evtid, label, color in [(0, "EVEN", "tab:blue"), (1, "ODD", "tab:orange")]:
            on_vals  = np.array([cum_on[evtid, e_idxs, 0, t_idx_use].sum()
                                for _, e_idxs in zip(bands, bands_idxs)])
            off_vals = np.array([self.cum_off_used[evtid, e_idxs, 0, t_idx_use].sum()
                                for _, e_idxs in zip(bands, bands_idxs)])
            exc = on_vals - alpha * off_vals
            offset = -0.2 if evtid == 0 else 0.2
            ax.bar(x + offset, exc, width=0.4, color=color, alpha=0.8, label=label)

        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(self.min_signal_events, color="red", ls="--", lw=1.5, label="min_signal")
        ax.set_xticks(x); ax.set_xticklabels(bands, rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("energy band"); ax.set_ylabel("excess (ON − α·OFF)")
        ax.set_title(f"Excess per band — {metric} (no cuts)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        self._savefig(fig, output_dir, f"E_events_{metric}_{mode}.png")
        plt.show()

    def plot_phaseograms_with_cuts(
        self,
        results,
        pkl_data=None,
        p1_limits=[0.0, 0.026, 0.983, 1.0],
        p2_limits=[0.377, 0.422],
        off_limits=[0.52, 0.87],
        p3_limits=None,
        nbins=60,
        output_dir=None,
    ):
        if pkl_data is None:
            pkl_data = self.pkl_data

        metric       = results["metric"]
        bands        = results["bands"]
        bands_idxs   = results["_bands_idxs"]
        n_bands      = len(bands)
        bins_type    = results.get("bins_type", "energy")

        phase_bins  = np.linspace(0, 1, nbins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

        alpha_p3 = None
        if p3_limits is not None:
            p3_w  = sum(p3_limits[i+1] - p3_limits[i] for i in range(0, len(p3_limits), 2))
            off_w = sum(off_limits[i+1] - off_limits[i] for i in range(0, len(off_limits), 2))
            alpha_p3 = p3_w / off_w

        def _pm(phases, limits):
            mask = np.zeros(len(phases), dtype=bool)
            for i in range(0, len(limits), 2):
                mask |= (phases >= limits[i]) & (phases < limits[i + 1])
            return mask

        def _lima(phases, on_lim, off_lim, alpha):
            n_on  = _pm(phases, on_lim).sum()
            n_off = _pm(phases, off_lim).sum()
            if n_off == 0:
                return 0.0, int(n_on), 0
            return float(lima_significance(n_on, n_off, alpha)), int(n_on), int(n_off)

        def _select(e_idxs, g_cut, t_cut, evtid_target):
            """Select phases for evtid_target passing g_cut in e_idxs."""
            out = []
            for data in pkl_data:
                if "energy" not in data:
                    continue
                gammaness = data["gammaness"]
                theta2    = data["theta2"]
                phase     = data["pulsar_phase"]
                evtid     = data["event_id"] % 2

                in_ebin = np.zeros(len(phase), dtype=bool)
                if bins_type == "energy":
                    log_e = np.log10(data["energy"])
                    for idx in e_idxs:
                        in_ebin |= ((log_e >= self.logenergy_bins[idx]) &
                                    (log_e <  self.logenergy_bins[idx + 1]))
                else:
                    intensity = data["intensity"]
                    for idx in e_idxs:
                        in_ebin |= ((intensity >= self.intensity_bins[idx]) &
                                    (intensity <  self.intensity_bins[idx + 1]))

                sel = in_ebin & (theta2 <= t_cut) & (evtid == evtid_target)
                if not np.isnan(g_cut):
                    sel &= (gammaness >= g_cut)
                out.extend(phase[sel])
            return np.array(out)

        def _get_phases_band(e_idxs, bidx):
            """ODD with EVEN cuts + EVEN with ODD cuts concatenated."""
            g_e = results["gammaness_even"][bidx]
            g_o = results["gammaness_odd"][bidx]
            t_e = results["theta2_even"][bidx]
            t_o = results["theta2_odd"][bidx]
            ph_odd  = _select(e_idxs, g_e, t_e, evtid_target=1)  # ODD with EVEN cuts
            ph_even = _select(e_idxs, g_o, t_o, evtid_target=0)  # EVEN with ODD cuts
            return np.concatenate([ph_odd, ph_even]), g_e, g_o, t_e, t_o

        def _draw_panel(ax, phases, title, g_e, g_o, t_e, t_o):
            counts, _ = np.histogram(phases, bins=phase_bins)

            ax.bar(bin_centers,     counts, width=1/nbins, color="steelblue", alpha=0.7)
            ax.bar(bin_centers + 1, counts, width=1/nbins, color="steelblue", alpha=0.7)
            ax.errorbar(bin_centers,     counts, yerr=np.sqrt(np.maximum(counts, 1)),
                        fmt="none", color="black", lw=0.8)
            ax.errorbar(bin_centers + 1, counts, yerr=np.sqrt(np.maximum(counts, 1)),
                        fmt="none", color="black", lw=0.8)

            off_mask = _pm(bin_centers, off_limits)
            if off_mask.sum() > 0:
                ax.axhline(counts[off_mask].mean(), color="gray", ls="--", lw=1)

            for lims, color in [(p1_limits, "orange"), (p2_limits, "green"),
                                (off_limits, "gray")]:
                for i in range(0, len(lims), 2):
                    for offset in [0, 1]:
                        ax.axvspan(lims[i] + offset, lims[i+1] + offset,
                                alpha=0.15, color=color)
            if p3_limits is not None:
                for i in range(0, len(p3_limits), 2):
                    for offset in [0, 1]:
                        ax.axvspan(p3_limits[i] + offset, p3_limits[i+1] + offset,
                                alpha=0.15, color="purple")

            sig_p1,   n_p1,   _ = _lima(phases, p1_limits, off_limits, self.alpha_p1)
            sig_p2,   n_p2,   _ = _lima(phases, p2_limits, off_limits, self.alpha_p2)
            sig_p1p2, n_p1p2, _ = _lima(phases, p1_limits + p2_limits,
                                        off_limits, self.alpha_p1p2)
            g_e_str = f"{g_e:.3f}" if not np.isnan(g_e) else "nan"
            g_o_str = f"{g_o:.3f}" if not np.isnan(g_o) else "nan"
            t_e_str = f"{t_e:.3f}" if not np.isnan(t_e) else "nan"
            t_o_str = f"{t_o:.3f}" if not np.isnan(t_o) else "nan"
            lines = [
                f"P1:    {sig_p1:.2f}σ  (N={n_p1})",
                f"P2:    {sig_p2:.2f}σ  (N={n_p2})",
                f"P1+P2: {sig_p1p2:.2f}σ  (N={n_p1p2})",
            ]
            if p3_limits is not None and alpha_p3 is not None:
                sig_p3, n_p3, _ = _lima(phases, p3_limits, off_limits, alpha_p3)
                lines.append(f"P3:    {sig_p3:.2f}σ  (N={n_p3})")
            lines += [
                f"g_even>{g_e_str},  g_odd>{g_o_str}",
                f"th2_even>{t_e_str},  th2_odd<{t_o_str}",
            ]

            ax.text(0.98, 0.97, "\n".join(lines),
                    transform=ax.transAxes, va="top", ha="right", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Pulsar phase", fontsize=8)
            ax.set_ylabel("N events", fontsize=8)
            ax.set_xlim(0, 2)
            if counts.max() > 0:
                ax.set_ylim(max(0, counts.min() * 0.98), counts.max() * 1.02)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

        # ── layout ───────────────────────────────────────────────────────────
        if n_bands == 1:
            # single panel
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            fig.suptitle(f"Phaseogram (cross-cuts) — {metric}", fontsize=11)
            phases, g_e, g_o, t_e, t_o = _get_phases_band(bands_idxs[0], 0)
            _draw_panel(ax, phases, bands[0], g_e, g_o, t_e, t_o)
        else:
            # one row per band + one integrated row
            n_rows = n_bands + 1
            fig, axes = plt.subplots(n_rows, 1, figsize=(10, n_rows * 3.5), squeeze=False)
            fig.suptitle(f"Phaseogram (cross-cuts) — {metric}", fontsize=11)

            all_phases = []
            for bidx, (band_label, e_idxs) in enumerate(zip(bands, bands_idxs)):
                phases, g_e, g_o, t_e, t_o = _get_phases_band(e_idxs, bidx)
                all_phases.extend(phases.tolist())
                _draw_panel(axes[bidx, 0], phases, band_label, g_e, g_o, t_e, t_o)

            # integrated
            g_e_mean = float(np.nanmean(results["gammaness_even"]))
            g_o_mean = float(np.nanmean(results["gammaness_odd"]))
            _draw_panel(axes[n_bands, 0], np.array(all_phases),
                        "ALL BANDS (integrated)", g_e_mean, g_o_mean, t_e, t_o)

        plt.tight_layout()
        self._savefig(fig, output_dir, f"F_phaseogram_{metric}.png")
        plt.show()

    # =========================================================================
    # PLOTTING — plot_check_data
    # =========================================================================

    def plot_check_data(self, output_dir=None):
        """
        Diagnostic plots using the raw per-event arrays stored in the pickles.

        F1 – event count consistency: pkl raw vs histogram first/last bin, per energy bin
        F2 – N events distribution across runs, per energy bin (EVEN / ODD)
        F3 – 1-D histograms: intensity, length, width, gammaness, theta2  (log y)
        F4 – energy spectrum (log-log)
        F5 – phasogram (EVEN / ODD overlaid)
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # aggregate raw arrays from all pkl files
        fields = ["energy", "intensity", "length", "width",
                  "gammaness", "theta2", "pulsar_phase", "event_id"]
        agg = {f: [] for f in fields}
        for data in self.pkl_data:
            for f in fields:
                if f in data:
                    agg[f].append(data[f])
        arr       = {f: np.concatenate(agg[f]) for f in fields if agg[f]}
        even_mask = (arr["event_id"] % 2 == 0)
        odd_mask  = ~even_mask

        self._plot_F1_count_consistency(output_dir)
        # self._plot_F2_nevents_distribution(output_dir)
        self._plot_F3_1d_histograms(arr, even_mask, odd_mask, output_dir)
        self._plot_F4_energy_spectrum(arr, even_mask, odd_mask, output_dir)
        self._plot_F5_phasogram(arr, even_mask, odd_mask, output_dir)

    # ── plot_check_data sub-plots ─────────────────────────────────────────────

    def _plot_F1_count_consistency(self, output_dir):
        """
        Per energy bin: compare N events from pkl raw arrays vs histogram
        at loosest selection (g_min, t_max). Print totals on the plot.
        """
        n_energy = len(self.logenergy_bins) - 1

        # from histogram (g index 0 = all gammaness, last t index = all theta2)
        cum_all = self.cum_on_p1 + self.cum_on_p2 + self.cum_off
        h_even  = cum_all[0, :, 0, -1]
        h_odd   = cum_all[1, :, 0, -1]

        # from raw pkl event arrays
        raw_loge   = np.log10(np.concatenate([d["energy"]   for d in self.pkl_data if "energy"   in d]))
        raw_evtid  =          np.concatenate([d["event_id"] for d in self.pkl_data if "event_id" in d])
        raw_even   = np.histogram(raw_loge[raw_evtid % 2 == 0], bins=self.logenergy_bins)[0].astype(float)
        raw_odd    = np.histogram(raw_loge[raw_evtid % 2 == 1], bins=self.logenergy_bins)[0].astype(float)

        x      = np.arange(n_energy)
        xlabels = [f"{self.energy_edges[i]:.3f}" for i in range(n_energy)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Event count: pkl raw vs histogram (g 0.3, th2 0.2)", fontsize=11)

        for ax, h_e, h_o, r_e, r_o, title in [
            (axes[0], h_even, h_odd,  raw_even, raw_odd,  "EVEN vs ODD"),
            (axes[1], h_even + h_odd, None, raw_even + raw_odd, None, "Total"),
        ]:
            ax.bar(x - 0.2, h_e, width=0.4, color="tab:blue",   alpha=0.8, label="hist EVEN")
            if h_o is not None:
                ax.bar(x + 0.2, h_o, width=0.4, color="tab:orange", alpha=0.8, label="hist ODD")
            ax.step(x, r_e, where="mid", color="black", lw=1.5, ls="--", label="pkl EVEN")
            if r_o is not None:
                ax.step(x, r_o, where="mid", color="gray",  lw=1.5, ls=":",  label="pkl ODD")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=45, fontsize=6)
            ax.set_xlabel("E_min (TeV)", fontsize=8)
            ax.set_ylabel("N events", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis="y")
            total_h = h_e.sum() + (h_o.sum() if h_o is not None else 0)
            total_r = r_e.sum() + (r_o.sum() if r_o is not None else 0)
            ax.text(0.02, 0.97,
                    f"hist total: {total_h:.0f}\npkl  total: {total_r:.0f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        plt.tight_layout()
        self._savefig(fig, output_dir, "F1_count_consistency.png")
        plt.show()

    def _plot_F2_nevents_distribution(self, output_dir):
        """
        Per energy bin: histogram of N events per run for EVEN and ODD.
        """
        n_energy = len(self.logenergy_bins) - 1
        nrows, ncols = self._grid(n_energy)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
        axes = axes.flatten()
        fig.suptitle("N events per run distribution — EVEN vs ODD", fontsize=10)

        for i in range(n_energy):
            ax = axes[i]
            counts_even, counts_odd = [], []
            for data in self.pkl_data:
                if "energy" not in data or "event_id" not in data:
                    continue
                log_e  = np.log10(data["energy"])
                evtid  = data["event_id"] % 2
                in_bin = (log_e >= self.logenergy_bins[i]) & (log_e < self.logenergy_bins[i + 1])
                counts_even.append((in_bin & (evtid == 0)).sum())
                counts_odd.append( (in_bin & (evtid == 1)).sum())

            hi    = max(max(counts_even, default=1), max(counts_odd, default=1)) + 1
            bins  = np.linspace(0, hi, 20)
            ax.hist(counts_even, bins=bins, alpha=0.6, color="tab:blue",   label="EVEN")
            ax.hist(counts_odd,  bins=bins, alpha=0.6, color="tab:orange", label="ODD")
            ax.set_title(f"{self.energy_edges[i]:.3f}–{self.energy_edges[i+1]:.3f} TeV",
                         fontsize=8)
            ax.set_xlabel("N events / run", fontsize=7)
            ax.set_ylabel("N runs", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)

        for i in range(n_energy, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        self._savefig(fig, output_dir, "F2_nevents_distribution.png")
        plt.show()

    def _plot_F3_1d_histograms(self, arr, even_mask, odd_mask, output_dir):
        """
        1-D histograms of intensity, length, width, gammaness, theta2.
        EVEN and ODD overlaid, log y scale.
        """
        variables = [
            ("intensity", "Intensity (p.e.)", 50, None, None),
            ("length",    "Length (deg)",      50, None, None),
            ("width",     "Width (deg)",        50, None, None),
            ("gammaness", "Gammaness",          50, 0.0,  1.0),
            ("theta2",    "theta2 (deg²)",      50, 0.0,  0.4),
        ]
        fig, axes = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 4))
        fig.suptitle("1-D distributions — EVEN vs ODD", fontsize=10)

        for ax, (key, xlabel, nbins, xmin, xmax) in zip(axes, variables):
            if key not in arr:
                ax.set_visible(False)
                continue
            d_e = arr[key][even_mask]
            d_o = arr[key][odd_mask]
            lo  = xmin if xmin is not None else min(d_e.min(), d_o.min())
            hi  = xmax if xmax is not None else max(d_e.max(), d_o.max())
            bins = np.linspace(lo, hi, nbins + 1)
            ax.hist(d_e, bins=bins, alpha=0.6, color="tab:blue",   label="EVEN",
                    density=True, histtype="stepfilled")
            ax.hist(d_o, bins=bins, alpha=0.6, color="tab:orange", label="ODD",
                    density=True, histtype="stepfilled")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("density", fontsize=8)
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        self._savefig(fig, output_dir, "F3_1d_histograms.png")
        plt.show()

    def _plot_F4_energy_spectrum(self, arr, even_mask, odd_mask, output_dir):
        """Energy spectrum — log x, log y — EVEN vs ODD."""
        if "energy" not in arr:
            return
        bins = np.logspace(np.log10(arr["energy"].min() * 0.9),
                           np.log10(arr["energy"].max() * 1.1), 40)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(arr["energy"][even_mask], bins=bins, alpha=0.6,
                color="tab:blue",   label="EVEN", histtype="stepfilled")
        ax.hist(arr["energy"][odd_mask],  bins=bins, alpha=0.6,
                color="tab:orange", label="ODD",  histtype="stepfilled")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reconstructed energy (TeV)", fontsize=9)
        ax.set_ylabel("N events", fontsize=9)
        ax.set_title("Energy spectrum — EVEN vs ODD", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        self._savefig(fig, output_dir, "F4_energy_spectrum.png")
        plt.show()

    def _plot_F5_phasogram(self, arr, even_mask, odd_mask, output_dir):
        """Pulsar phasogram — EVEN and ODD overlaid."""
        if "pulsar_phase" not in arr:
            return
        bins = np.linspace(0, 1, 51)
        fig, ax = plt.subplots(figsize=(8, 4))
        counts_even, _ = np.histogram(arr["pulsar_phase"][even_mask], bins=bins)
        counts_odd, _  = np.histogram(arr["pulsar_phase"][odd_mask],  bins=bins)

        ymin = min(counts_even.min(), counts_odd.min())
        ymax = max(counts_even.max(), counts_odd.max())

        ax.hist(arr["pulsar_phase"][even_mask], bins=bins, alpha=0.6,
                color="tab:blue",   label="EVEN", histtype="stepfilled")
        ax.hist(arr["pulsar_phase"][odd_mask],  bins=bins, alpha=0.6,
                color="tab:orange", label="ODD",  histtype="stepfilled")
        ax.set_xlabel("Pulsar phase", fontsize=9)
        ax.set_ylabel("N events", fontsize=9)
        ax.set_ylim(0.95 * ymin, 1.05 * ymax)


        ax.set_title("Phasogram — EVEN vs ODD", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._savefig(fig, output_dir, "F5_phasogram.png")
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize gammaness (and theta2) cuts for pulsar analysis"
    )
    parser.add_argument("--pickle_dir",          required=True)
    parser.add_argument("--metric",              default="P1P2",
                        choices=["P1", "P2", "P1P2"])
    parser.add_argument("--significance",        default="snr",
                        choices=["snr", "lima"])
    parser.add_argument("--fixed_theta2",        type=float, default=None)
    parser.add_argument("--output_dir",          default=None)
    parser.add_argument("--min_excess_fraction", type=float, default=0.05)
    parser.add_argument("--min_signal_events",   type=float, default=10.0)
    parser.add_argument("--p1_width",            type=float, default=DEFAULT_P1_WIDTH)
    parser.add_argument("--p2_width",            type=float, default=DEFAULT_P2_WIDTH)
    parser.add_argument("--off_width",           type=float, default=DEFAULT_OFF_WIDTH)
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    optimizer = PulsarCutOptimizer(
        pickle_dir          = args.pickle_dir,
        min_excess_fraction = args.min_excess_fraction,
        min_signal_events   = args.min_signal_events,
        p1_width            = args.p1_width,
        p2_width            = args.p2_width,
        off_width           = args.off_width,
    )

    results = optimizer.optimize(
        metric       = args.metric,
        significance = args.significance,
        fixed_theta2 = args.fixed_theta2,
    )
    optimizer.plot_results(results, output_dir=args.output_dir)
    optimizer.plot_check_data(      output_dir=args.output_dir)

    print("\n=== RESULTS ===")
    for k in ("gammaness_even", "gammaness_odd", "theta2_even", "theta2_odd"):
        print(f"{k:20s}: {results[k]}")
