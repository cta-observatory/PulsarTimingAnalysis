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

# Phase region widths needed to compute alpha for Li&Ma
# These must match what was used to build the histograms
DEFAULT_P1_WIDTH  = (0.026 - 0.0) + (1.0 - 0.983)   # [0, 0.026] + [0.983, 1]
DEFAULT_P2_WIDTH  = 0.422 - 0.377                     # [0.377, 0.422]
DEFAULT_OFF_WIDTH = 0.87 - 0.52                       # [0.52, 0.87]


def lima_significance(Non, Noff, alpha):
    """
    Li & Ma significance
    Vectorized — works on numpy arrays.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = Non  * np.log((1 + alpha) / alpha * Non  / (Non + Noff))
        term2 = Noff * np.log((1 + alpha)          * Noff / (Non + Noff))
        result = np.sqrt(2) * np.sqrt(np.where((term1 + term2) > 0, term1 + term2, 0))
    result = np.where((Non > 0) & (Noff > 0), result, 0.0)
    return result

def signal_to_noise_ratio(Non, Noff, alpha):
    """Simple S/N ratio: excess / sqrt(background)"""
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

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            if on_p1 is None:
                on_p1 = data["on_p1"].astype("float64")
                on_p2 = data["on_p2"].astype("float64")
                on_p3 = data["on_p3"].astype("float64") if data["on_p3"] is not None else None
                off   = data["off"].astype("float64")
                self.gammaness_bins = data["bins"]["gammaness"]
                self.theta2_bins    = data["bins"]["theta2"]
                self.logenergy_bins = data["bins"]["logenergy"]
            else:
                on_p1 += data["on_p1"]
                on_p2 += data["on_p2"]
                if on_p3 is not None and data["on_p3"] is not None:
                    on_p3 += data["on_p3"]
                off   += data["off"]

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

    @staticmethod
    def _make_cumulative(hist):
        """
        Build cumulative histogram:
        - axis 2 (gammaness): integrate from each value to 1  → gammaness > cut
        - axis 3 (theta2):    integrate from 0 to each value  → theta2 < cut
        Shape stays (2, n_energy, n_gammaness, n_theta2).
        """
        cum = hist.copy()
        # n_evtid, n_energy, n_g, n_t = cum.shape
        # for evtid in range(n_evtid):
        #     for eidx in range(n_energy):
        #         # gammaness: cumsum from right
        #         for g in range(n_g - 2, -1, -1):
        #             cum[evtid, eidx, g, :] += cum[evtid, eidx, g + 1, :]
        #         # theta2: cumsum from left
        #         for t in range(1, n_t):
        #             cum[evtid, eidx, :, t] += cum[evtid, eidx, :, t - 1]
        cum = np.cumsum(hist[:, :, ::-1, :], axis=2)[:, :, ::-1, :]
        cum = np.cumsum(cum, axis=3)

        return cum

    def diagnose(self, metric="P1P2"):
        """
        Imprime estadísticas básicas para ver si los datos tienen sentido.
        """
        if metric == "P1":
            cum_on = self.cum_on_p1
            alpha  = self.alpha_p1
        elif metric == "P2":
            cum_on = self.cum_on_p2
            alpha  = self.alpha_p2
        else:
            cum_on = self.cum_on_p1 + self.cum_on_p2
            alpha  = self.alpha_p1p2

        print(f"\n{'='*60}")
        print(f"Alpha P1={self.alpha_p1:.4f}  P2={self.alpha_p2:.4f}  P1P2={self.alpha_p1p2:.4f}")
        print(f"{'='*60}")
        print(f"{'Bin (TeV)':<22} {'N_on':>8} {'N_off':>8} {'alpha*Noff':>11} {'excess':>9} {'ratio':>7}")
        print(f"{'-'*60}")

        n_energy = len(self.logenergy_bins) - 1
        for eidx in range(n_energy):
            emin = self.energy_edges[eidx]
            emax = self.energy_edges[eidx + 1]
            # sin cortes: gammaness mínimo, theta2 máximo
            non  = cum_on[0, eidx, 0, -1]
            noff = self.cum_off[0, eidx, 0, -1]
            bkg  = alpha * noff
            exc  = non - bkg
            ratio = non / noff if noff > 0 else float('nan')
            print(f"[{emin:.3f}, {emax:.3f}]       "
                f"{non:8.1f} {noff:8.1f} {bkg:11.1f} {exc:9.1f} {ratio:7.4f}")

        print(f"\nEsperado: ratio ~ alpha = {alpha:.4f}")
        print(f"Si ratio >> alpha en todos los bins -> alpha mal calculado")
        print(f"Si excess sistematicamente negativo -> ON y OFF intercambiados")
        print(f"Si excess ~ 0 en todos -> no hay señal o cortes demasiado duros")

    def diagnose_histograms(self):
        """
        Compara el histograma crudo con el acumulativo para detectar
        si el cumsum se está aplicando sobre datos ya acumulados.
        """
        import pickle, glob

        pkl_files = sorted(glob.glob(os.path.join(self.pickle_dir, "*.pkl")))
        with open(pkl_files[0], "rb") as f:
            data = pickle.load(f)

        raw_on_p1 = data["on_p1"].astype("float64")  # shape (2, n_e, n_g, n_t)

        print("=== HISTOGRAMA CRUDO (on_p1) ===")
        print(f"Shape: {raw_on_p1.shape}")
        print(f"Gammaness bins: {data['bins']['gammaness'][:5]} ...")
        print(f"Theta2 bins:    {data['bins']['theta2'][:5]} ...")

        # Toma el bin de energía con más estadística (eidx=2 ~ 0.025-0.040 TeV)
        eidx = 2
        raw_slice = raw_on_p1[0, eidx, :, :]   # (n_g, n_t)
        cum_slice = self.cum_on_p1[0, eidx, :, :]

        print(f"\n--- Bin energía {eidx} ---")
        print(f"Raw: suma total = {raw_slice.sum():.1f}")
        print(f"Raw: primeras celdas [g=0:3, t=0:3] =\n{raw_slice[:3, :3]}")
        print(f"Raw: últimas celdas  [g=-3:, t=-3:] =\n{raw_slice[-3:, -3:]}")

        print(f"\nCum: valor en [g=0, t=-1] (todo) = {cum_slice[0, -1]:.1f}")
        print(f"Cum: valor en [g=-1, t=-1] (g>max) = {cum_slice[-1, -1]:.1f}  ← debe ser ~0")
        print(f"Cum: valor en [g=0, t=0]  (t<min) = {cum_slice[0, 0]:.1f}   ← debe ser pequeño")

        print(f"\n¿Dónde está el exceso en el raw?")
        g_max_idx, t_max_idx = np.unravel_index(np.nanargmax(raw_slice), raw_slice.shape)
        print(f"  Celda con más eventos: g_idx={g_max_idx} "
            f"(g={data['bins']['gammaness'][g_max_idx]:.3f}), "
            f"t_idx={t_max_idx} "
            f"(t={data['bins']['theta2'][t_max_idx]:.4f})")
        print(f"  Valor: {raw_slice[g_max_idx, t_max_idx]:.1f}")

        print(f"\n¿Dónde está el exceso en el acumulativo?")
        # En el cum correcto, el máximo global debe estar en [g=0, t=-1]
        print(f"  cum[g=0,  t=-1] = {cum_slice[0,  -1]:.1f}  ← debe ser el máximo")
        print(f"  cum[g=-1, t=-1] = {cum_slice[-1, -1]:.1f}  ← debe ser ~0 o mínimo")
        print(f"  cum[g=0,  t=0]  = {cum_slice[0,   0]:.1f}  ← debe ser pequeño")

        # Plot rápido del raw y cum para inspección visual
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        g_bins = data['bins']['gammaness']
        t_bins = data['bins']['theta2']
        g_c = (g_bins[:-1] + g_bins[1:]) / 2
        t_c = (t_bins[:-1] + t_bins[1:]) / 2

        im0 = axes[0].pcolormesh(t_c, g_c, raw_slice, cmap="viridis", shading="auto")
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title(f"RAW on_p1 — bin E={eidx}")
        axes[0].set_xlabel("theta2"); axes[0].set_ylabel("gammaness")

        im1 = axes[1].pcolormesh(t_c, g_c, cum_slice, cmap="viridis", shading="auto")
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title("CUMULATIVE (gammaness>, theta2<)")
        axes[1].set_xlabel("theta2"); axes[1].set_ylabel("gammaness")

        # El cum correcto debe ser monótonamente decreciente en g y creciente en t
        axes[2].plot(g_c, cum_slice[:, -1], label="cum vs g (t=max)", color="tab:blue")
        axes[2].plot(g_c, raw_slice[:, -1], label="raw vs g (t=max)", color="tab:orange", ls="--")
        axes[2].set_xlabel("gammaness")
        axes[2].set_ylabel("counts")
        axes[2].set_title("¿Monótonamente decreciente en g?")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        # Añade esto al diagnose_histograms para verificar:
        print("\n=== CHECK ACUMULATIVO EN ENERGÍA ===")
        for eidx in range(min(5, raw_on_p1.shape[1])):
            total_raw = float(raw_on_p1[0, eidx, :, :].sum())
            total_cum = float(self.cum_on_p1[0, eidx, 0, -1])
            ratio = total_cum / total_raw if total_raw > 0 else float('inf')
            print(f"  E bin {eidx}: raw.sum()={total_raw:.0f}, cum[g=0,t=-1]={total_cum:.0f}, ratio={ratio:.1f}")
        print(f"Si ratio >> n_g ({raw_on_p1.shape[2]}): el histograma YA esta acumulado en gamma antes de entrar")

    def diagnose_signal(self, metric="P1P2"):
        """
        Comprueba el SNR en el punto conocido bueno (g=0.575, t=0.15)
        vs el máximo que encuentra el optimizador, para cada bin de energía.
        """
        if metric == "P1":
            cum_on = self.cum_on_p1
            alpha  = self.alpha_p1
        elif metric == "P2":
            cum_on = self.cum_on_p2
            alpha  = self.alpha_p2
        else:
            cum_on = self.cum_on_p1 + self.cum_on_p2
            alpha  = self.alpha_p1p2

        # índices del punto conocido bueno
        g_known = 0.575
        t_known = 0.15
        g_idx = np.argmin(np.abs(self.gammaness_bins[:-1] - g_known))
        t_idx = np.argmin(np.abs(self.theta2_bins[:-1]    - t_known))

        print(f"Punto de referencia: g>{self.gammaness_bins[g_idx]:.3f} "
            f"(idx={g_idx}), t<{self.theta2_bins[t_idx]:.3f} (idx={t_idx})")
        print(f"\n{'Bin (TeV)':<22} {'Non':>8} {'Noff':>8} "
            f"{'excess':>8} {'SNR_ref':>9} {'SNR_max':>9} {'g_max':>7} {'t_max':>7}")
        print("-"*75)

        n_e = len(self.logenergy_bins) - 1
        for eidx in range(n_e):
            emin = self.energy_edges[eidx]
            emax = self.energy_edges[eidx+1]

            on_2d  = cum_on[0, eidx, :, :]
            off_2d = self.cum_off[0, eidx, :, :]

            non_ref  = on_2d[g_idx, t_idx]
            noff_ref = off_2d[g_idx, t_idx]
            exc_ref  = non_ref - alpha * noff_ref
            bkg_ref  = alpha * noff_ref
            snr_ref  = exc_ref / np.sqrt(bkg_ref) if bkg_ref > 0 else 0.0

            # SNR máximo en todo el espacio 2D sin máscaras
            excess_2d = on_2d - alpha * off_2d
            bkg_2d    = alpha * off_2d
            with np.errstate(divide="ignore", invalid="ignore"):
                snr_2d = np.where(bkg_2d > 0, excess_2d / np.sqrt(bkg_2d), 0.0)

            best_idx = np.unravel_index(np.nanargmax(snr_2d), snr_2d.shape)
            snr_max  = snr_2d[best_idx]
            g_best   = self.gammaness_bins[best_idx[0]]
            t_best   = self.theta2_bins[best_idx[1]]

            print(f"[{emin:.3f},{emax:.3f}]   "
                f"{non_ref:8.0f} {noff_ref:8.0f} {exc_ref:8.1f} "
                f"{snr_ref:9.2f} {snr_max:9.2f} {g_best:7.3f} {t_best:7.4f}")
   # ------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------

    def optimize(self, metric="P1P2", significance="snr", energy_mode="per_bin", energy_ranges=None):
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
        if metric not in ("P1", "P2", "P1P2"):
            raise ValueError("metric must be one of 'P1', 'P2', 'P1P2'")

        logger.info(f"Optimizing cuts for metric: {metric}")

        if metric == "P1":
            cum_on = self.cum_on_p1
            alpha  = self.alpha_p1
        elif metric == "P2":
            cum_on = self.cum_on_p2
            alpha  = self.alpha_p2
        else:
            cum_on = self.cum_on_p1 + self.cum_on_p2
            alpha  = self.alpha_p1p2

        sigma_g_bins = 0.05 / (self.gammaness_bins[1] - self.gammaness_bins[0])
        sigma_t_bins = 0.02 / (self.theta2_bins[1]    - self.theta2_bins[0])

        theta2_fixed = 0.150
        t_idx_fixed  = np.argmin(np.abs(self.theta2_bins[:-1] - theta2_fixed))

        n_energy = len(self.logenergy_bins) - 1
        n_g      = len(self.gammaness_bins) - 1
        n_t      = len(self.theta2_bins)    - 1

        # Construir bandas de energía
        if energy_mode == "integrated":
            bands      = [("integrated", list(range(n_energy)))]

        elif energy_mode == "bands":
            if energy_ranges is None:
                raise ValueError("energy_ranges requerido para energy_mode='bands'")
            # Para cada bin de energía, calcular solapamiento con cada banda
            # y asignarlo a la banda con mayor solapamiento
            bin_to_band = {}
            for i in range(n_energy):
                ebin_min = self.energy_edges[i]
                ebin_max = self.energy_edges[i + 1]
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
                    logger.warning(f"No bins assigned to [{emin_r:.3f}, {emax_r:.3f}] TeV, skipping")
                    continue
                emin_actual = self.energy_edges[idxs[0]]
                emax_actual = self.energy_edges[idxs[-1] + 1]
                label = f"{emin_actual:.3f}-{emax_actual:.3f} TeV"
                logger.info(f"Band requested [{emin_r:.3f}, {emax_r:.3f}] TeV → "
                            f"actual [{emin_actual:.3f}, {emax_actual:.3f}] TeV "
                            f"({len(idxs)} bins: {idxs})")
                bands.append((label, idxs))
        else:  # per_bin
            bands = [(f"{self.energy_edges[i]:.3f}-{self.energy_edges[i+1]:.3f} TeV", [i])
                    for i in range(n_energy)]

        n_bands = len(bands)

        gammaness_even = np.full(n_bands, np.nan)
        gammaness_odd  = np.full(n_bands, np.nan)
        theta2_even    = np.full(n_bands, np.nan)
        theta2_odd     = np.full(n_bands, np.nan)
        sigma_map_even = np.full((n_bands, n_g, n_t), np.nan)
        sigma_map_odd  = np.full((n_bands, n_g, n_t), np.nan)
        n_events_even  = np.zeros(n_bands)
        n_events_odd   = np.zeros(n_bands)

        for bidx, (band_label, e_idxs) in enumerate(bands):

            for evtid, (sigma_map, g_out, t_out, n_out) in enumerate([
                (sigma_map_even, gammaness_even, theta2_even, n_events_even),
                (sigma_map_odd,  gammaness_odd,  theta2_odd,  n_events_odd),
            ]):
                # Sumar sobre los bins de energía de esta banda
                on_2d  = cum_on[evtid, e_idxs, :, :].sum(axis=0)   # (n_g, n_t)
                off_2d = self.cum_off[evtid, e_idxs, :, :].sum(axis=0)

                n_out[bidx] = on_2d[0, -1] + off_2d[0, -1]

                # Significancia
                if significance == "lima":
                    sig_2d = lima_significance(on_2d, off_2d, alpha)
                else:
                    sig_2d = signal_to_noise_ratio(on_2d, off_2d, alpha)
                # Máscaras
                excess  = on_2d - alpha * off_2d
                bkg_est = alpha * off_2d
                mask = (
                    (excess  > self.min_excess_fraction * bkg_est) &
                    (excess  > self.min_signal_events) &
                    (bkg_est > 2.0) &
                    (on_2d   > self.min_on_events) &
                    (off_2d  > self.min_off_events)
                )

                sig_2d_masked = np.where(mask, sig_2d, np.nan)
                sigma_map[bidx] = sig_2d_masked

                if np.all(np.isnan(sig_2d_masked)):
                    logger.warning(
                        f"[{'EVEN' if evtid==0 else 'ODD'}] "
                        f"{band_label}: no valid cuts found"
                    )
                    continue

                # Suavizado gaussiano
                sig_filled = np.nan_to_num(sig_2d_masked, nan=0.0)
                sig_smooth = gaussian_filter(sig_filled, sigma=[sigma_g_bins, sigma_t_bins], mode='constant', cval=0.0)
                sig_smooth = np.where(mask, sig_smooth, np.nan)

                # Máximo global directo sobre el mapa suavizado
                best_idx   = np.nanargmax(sig_smooth)
                best_g_idx, best_t_idx = np.unravel_index(best_idx, sig_smooth.shape)
                best_g   = self.gammaness_bins[best_g_idx]
                best_t   = self.theta2_bins[best_t_idx]
                best_sig = sig_smooth[best_g_idx, best_t_idx]

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
        }

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------

    def plot_results(self, results, output_dir=None):
        metric     = results["metric"]
        bands      = results["bands"]
        n_bands    = len(bands)
        mode       = results.get("mode", "per_bin")
        bands_idxs = results.get("_bands_idxs", [[i] for i in range(n_bands)])

        g_centers = (self.gammaness_bins[:-1] + self.gammaness_bins[1:]) / 2
        t_centers = (self.theta2_bins[:-1]    + self.theta2_bins[1:])    / 2

        theta2_fixed = 0.150
        t_idx_fixed  = np.argmin(np.abs(self.theta2_bins[:-1] - theta2_fixed))

        if metric == "P1":
            cum_on = self.cum_on_p1
            alpha  = self.alpha_p1
        elif metric == "P2":
            cum_on = self.cum_on_p2
            alpha  = self.alpha_p2
        else:
            cum_on = self.cum_on_p1 + self.cum_on_p2
            alpha  = self.alpha_p1p2

        ncols = min(4, n_bands)
        nrows = int(np.ceil(n_bands / ncols))

        for label, sigma_map, g_opt, t_opt in [
            ("EVEN", results["sigma_map_even"], results["gammaness_even"], results["theta2_even"]),
            ("ODD",  results["sigma_map_odd"],  results["gammaness_odd"],  results["theta2_odd"]),
        ]:
            evtid = 0 if label == "EVEN" else 1

            # --- Plot A: SNR vs gammaness (theta2 fijo) ---
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
            axes = axes.flatten()
            fig.suptitle(f"[A] SNR vs gammaness  (theta2={theta2_fixed}) — {metric} — {label}")
            for bidx in range(n_bands):
                ax  = axes[bidx]
                sig = sigma_map[bidx]
                if not np.all(np.isnan(sig)):
                    snr_g = sig[:, t_idx_fixed]
                    ax.plot(g_centers, snr_g, lw=1.8, color="tab:blue")
                    ax.axhline(0, color="gray", lw=0.6, ls=":")
                    if not np.isnan(g_opt[bidx]):
                        ax.axvline(g_opt[bidx], color="red", lw=1.2, ls="--",
                                label=f"g={g_opt[bidx]:.3f}")
                        ax.legend(fontsize=7)
                ax.set_title(bands[bidx], fontsize=8)
                ax.set_xlabel("gammaness", fontsize=7)
                ax.set_ylabel("SNR", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
            for bidx in range(n_bands, len(axes)):
                axes[bidx].set_visible(False)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir,
                    f"A_snr_gammaness_{metric}_{label}_{mode}.png"), dpi=120)
            plt.show()

            # --- Plot B: SNR vs theta2 (gammaness óptima) ---
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
            axes = axes.flatten()
            fig.suptitle(f"[B] SNR vs theta2  (g=best) — {metric} — {label}")
            for bidx in range(n_bands):
                ax  = axes[bidx]
                sig = sigma_map[bidx]
                if not np.all(np.isnan(sig)) and not np.isnan(g_opt[bidx]):
                    g_idx_opt = np.argmin(np.abs(g_centers - g_opt[bidx]))
                    snr_t = sig[g_idx_opt, :]
                    ax.plot(t_centers, snr_t, lw=1.8, color="tab:orange")
                    ax.axhline(0, color="gray", lw=0.6, ls=":")
                    if not np.isnan(t_opt[bidx]):
                        ax.axvline(t_opt[bidx], color="red", lw=1.2, ls="--",
                                label=f"t={t_opt[bidx]:.3f}")
                        ax.legend(fontsize=7)
                ax.set_title(bands[bidx], fontsize=8)
                ax.set_xlabel("theta2 (deg²)", fontsize=7)
                ax.set_ylabel("SNR", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
            for bidx in range(n_bands, len(axes)):
                axes[bidx].set_visible(False)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir,
                    f"B_snr_theta2_{metric}_{label}_{mode}.png"), dpi=120)
            plt.show()

            # --- Plot C: Excess map 2D ---
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5), squeeze=False)
            axes = axes.flatten()
            fig.suptitle(f"[C] Excess map — {metric} — {label}")
            for bidx, e_idxs in enumerate(bands_idxs):
                ax     = axes[bidx]
                on_2d  = cum_on[evtid, e_idxs, :, :].sum(axis=0)
                off_2d = self.cum_off[evtid, e_idxs, :, :].sum(axis=0)
                excess = on_2d - alpha * off_2d
                vmax   = max(np.nanpercentile(np.abs(excess), 98), 1.0)
                im = ax.pcolormesh(t_centers, g_centers, excess,
                                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
                plt.colorbar(im, ax=ax, label="excess", pad=0.02)
                if not np.isnan(g_opt[bidx]) and not np.isnan(t_opt[bidx]):
                    ax.plot(t_opt[bidx], g_opt[bidx], "r+", ms=12, mew=2.0, label="óptimo")
                    ax.legend(fontsize=7)
                ax.set_title(bands[bidx], fontsize=8)
                ax.set_xlabel("theta2 (deg²)", fontsize=7)
                ax.set_ylabel("gammaness", fontsize=7)
                ax.tick_params(labelsize=7)
            for bidx in range(n_bands, len(axes)):
                axes[bidx].set_visible(False)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir,
                    f"C_excess_{metric}_{label}_{mode}.png"), dpi=120)
            plt.show()

            # --- Plot C2: SNR map 2D suavizado con óptimo ---
            from scipy.ndimage import gaussian_filter
            sigma_g_bins = 0.05 / (self.gammaness_bins[1] - self.gammaness_bins[0])
            sigma_t_bins = 0.02 / (self.theta2_bins[1]    - self.theta2_bins[0])

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.5), squeeze=False)
            axes = axes.flatten()
            fig.suptitle(f"Li&Ma map 2D — {metric} — {label}")

            for bidx in range(n_bands):
                ax  = axes[bidx]
                sig = sigma_map[bidx]   # ya enmascarado con NaN fuera de máscara

                if not np.all(np.isnan(sig)):
                    # suavizar para visualización
                    sig_filled = np.nan_to_num(sig, nan=0.0)
                    sig_smooth = gaussian_filter(sig_filled,
                                                sigma=[sigma_g_bins, sigma_t_bins],
                                                mode='constant', cval=0.0)
                    sig_smooth = np.where(np.isnan(sig), np.nan, sig_smooth)

                    vmax = np.nanpercentile(sig_smooth, 98)
                    vmax = max(vmax, 0.1)
                    im = ax.pcolormesh(t_centers, g_centers, sig_smooth, #sig
                                    cmap="viridis", vmin=0, vmax=vmax, shading="auto")
                    plt.colorbar(im, ax=ax, label="Li&Ma", pad=0.02)

                # marcar óptimo
                if not np.isnan(g_opt[bidx]) and not np.isnan(t_opt[bidx]):
                    ax.plot(t_opt[bidx], g_opt[bidx], "r+", ms=12, mew=2.0, label="optimum")
                    ax.legend(fontsize=7)

                ax.set_title(bands[bidx], fontsize=8)
                ax.set_xlabel("theta2 (deg²)", fontsize=7)
                ax.set_ylabel("gammaness", fontsize=7)
                ax.tick_params(labelsize=7)

            for bidx in range(n_bands, len(axes)):
                axes[bidx].set_visible(False)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir,
                    f"C2_snrmap_{metric}_{label}_{mode}.png"), dpi=120)
            plt.show()

            # --- Plot D: cortes vs banda (solo si n_bands > 1) ---
            if n_bands > 1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"[D] Cortes óptimos por banda — {metric} — {label}")
                x = np.arange(n_bands)
                ax1.bar(x, np.where(np.isnan(g_opt), 0, g_opt),
                        color="tab:blue", alpha=0.8)
                ax1.set_xticks(x)
                ax1.set_xticklabels(bands, rotation=20, ha="right", fontsize=7)
                ax1.set_ylabel("gammaness cut")
                ax1.set_ylim(0, 1)
                ax1.grid(alpha=0.3, axis="y")
                ax2.bar(x, np.where(np.isnan(t_opt), 0, t_opt),
                        color="tab:orange", alpha=0.8)
                ax2.set_xticks(x)
                ax2.set_xticklabels(bands, rotation=20, ha="right", fontsize=7)
                ax2.set_ylabel("theta2 cut (deg²)")
                ax2.grid(alpha=0.3, axis="y")
                plt.tight_layout()
                if output_dir:
                    plt.savefig(os.path.join(output_dir,
                        f"D_cuts_{metric}_{label}_{mode}.png"), dpi=120)
                plt.show()

        # --- Plot E: eventos por banda (siempre) ---
        fig, ax = plt.subplots(figsize=(max(8, n_bands * 1.2), 4))
        x = np.arange(n_bands)
        ax.bar(x - 0.2, results["n_events_even"],
            width=0.4, color="tab:blue",   label="EVEN", alpha=0.8)
        ax.bar(x + 0.2, results["n_events_odd"],
            width=0.4, color="tab:orange", label="ODD",  alpha=0.8)
        ax.axhline(self.min_signal_events,
                color="red", ls="--", lw=1.5, label="min_signal_events")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("Banda de energía")
        ax.set_ylabel("Total eventos (ON + OFF)")
        ax.set_title(f"Eventos por banda — {metric}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir,
                f"E_events_{metric}_{mode}.png"), dpi=120)
        plt.show()

    def plot_even_odd_balance(self, metric="P1P2", output_dir=None):
        """
        Por cada bin de energía muestra:
        - N_on EVEN vs N_on ODD
        - N_off EVEN vs N_off ODD
        - Diferencia N_on(EVEN) - N_on(ODD)
        - Diferencia N_off(EVEN) - N_off(ODD)
        Sin ningún corte de gammaness ni theta2 (g mínimo, t máximo).
        """
        if metric == "P1":
            cum_on = self.cum_on_p1
            alpha  = self.alpha_p1
        elif metric == "P2":
            cum_on = self.cum_on_p2
            alpha  = self.alpha_p2
        else:
            cum_on = self.cum_on_p1 + self.cum_on_p2
            alpha  = self.alpha_p1p2

        n_energy = len(self.logenergy_bins) - 1
        e_centers = np.sqrt(self.energy_edges[:-1] * self.energy_edges[1:])
        e_labels  = [f"{self.energy_edges[i]:.3f}\n-{self.energy_edges[i+1]:.3f}"
                    for i in range(n_energy)]
        x = np.arange(n_energy)

        # Sin cortes: g=0 (mínimo gammaness), t=-1 (máximo theta2)
        on_even  = np.array([cum_on[0, i, 0, -1]       for i in range(n_energy)])
        on_odd   = np.array([cum_on[1, i, 0, -1]       for i in range(n_energy)])
        off_even = np.array([self.cum_off[0, i, 0, -1] for i in range(n_energy)])
        off_odd  = np.array([self.cum_off[1, i, 0, -1] for i in range(n_energy)])

        diff_on  = on_even  - on_odd
        diff_off = off_even - off_odd

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle(f"Even/Odd balance por bin de energía — {metric} (sin cortes)", fontsize=13)

        # --- ON: EVEN vs ODD ---
        ax = axes[0, 0]
        ax.bar(x - 0.2, on_even, width=0.4, color="tab:blue",   label="ON EVEN", alpha=0.8)
        ax.bar(x + 0.2, on_odd,  width=0.4, color="tab:orange", label="ON ODD",  alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(e_labels, fontsize=6, rotation=45)
        ax.set_ylabel("N eventos")
        ax.set_title("ON: EVEN vs ODD")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # --- OFF: EVEN vs ODD ---
        ax = axes[0, 1]
        ax.bar(x - 0.2, off_even, width=0.4, color="tab:blue",   label="OFF EVEN", alpha=0.8)
        ax.bar(x + 0.2, off_odd,  width=0.4, color="tab:orange", label="OFF ODD",  alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(e_labels, fontsize=6, rotation=45)
        ax.set_ylabel("N eventos")
        ax.set_title("OFF: EVEN vs ODD")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # --- Diferencia ON(EVEN) - ON(ODD) ---
        ax = axes[1, 0]
        colors_on = ["tab:blue" if d >= 0 else "tab:red" for d in diff_on]
        ax.bar(x, diff_on, color=colors_on, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        # línea de referencia: ±5% de la media
        mean_on = (on_even + on_odd).mean() / 2
        ax.axhline( 0.05 * mean_on, color="gray", lw=1, ls="--", label="±5% media")
        ax.axhline(-0.05 * mean_on, color="gray", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(e_labels, fontsize=6, rotation=45)
        ax.set_ylabel("N(EVEN) - N(ODD)")
        ax.set_title("Diferencia ON: EVEN − ODD")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # --- Diferencia OFF(EVEN) - OFF(ODD) ---
        ax = axes[1, 1]
        colors_off = ["tab:blue" if d >= 0 else "tab:red" for d in diff_off]
        ax.bar(x, diff_off, color=colors_off, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        mean_off = (off_even + off_odd).mean() / 2
        ax.axhline( 0.05 * mean_off, color="gray", lw=1, ls="--", label="±5% media")
        ax.axhline(-0.05 * mean_off, color="gray", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(e_labels, fontsize=6, rotation=45)
        ax.set_ylabel("N(EVEN) - N(ODD)")
        ax.set_title("Diferencia OFF: EVEN − ODD")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        if output_dir:
            fname = os.path.join(output_dir, f"even_odd_balance_{metric}.png")
            plt.savefig(fname, dpi=120)
            logger.info(f"Saved {fname}")
        plt.show()

        # Log numérico
        print(f"\n{'='*80}")
        print(f"{'Bin (TeV)':<22} {'ON_even':>10} {'ON_odd':>10} {'diff_ON':>10} "
            f"{'OFF_even':>10} {'OFF_odd':>10} {'diff_OFF':>10}")
        print(f"{'-'*80}")
        for i in range(n_energy):
            emin = self.energy_edges[i]
            emax = self.energy_edges[i+1]
            ratio_on  = diff_on[i]  / on_even[i]  * 100 if on_even[i]  > 0 else float('nan')
            ratio_off = diff_off[i] / off_even[i] * 100 if off_even[i] > 0 else float('nan')
            print(f"[{emin:.3f},{emax:.3f}]      "
                f"{on_even[i]:10.0f} {on_odd[i]:10.0f} "
                f"{diff_on[i]:+10.0f} ({ratio_on:+5.1f}%)  "
                f"{off_even[i]:10.0f} {off_odd[i]:10.0f} "
                f"{diff_off[i]:+10.0f} ({ratio_off:+5.1f}%)")
# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize gammaness and theta2 cuts for pulsar analysis"
    )
    parser.add_argument("--pickle_dir",  required=True,
                        help="Directory with histogram pickle files")
    parser.add_argument("--metric",      default="P1P2",
                        choices=["P1", "P2", "P1P2"],
                        help="Phase region to optimize (default: P1P2)")
    parser.add_argument("--output_dir",  default=None,
                        help="Directory to save plots (default: show only)")
    parser.add_argument("--min_excess_fraction", type=float, default=0.05,
                        help="Min excess as fraction of OFF (default: 0.05)")
    parser.add_argument("--min_signal_events",   type=float, default=10.0,
                        help="Min number of excess events (default: 10)")
    parser.add_argument("--p1_width",  type=float, default=DEFAULT_P1_WIDTH)
    parser.add_argument("--p2_width",  type=float, default=DEFAULT_P2_WIDTH)
    parser.add_argument("--off_width", type=float, default=DEFAULT_OFF_WIDTH)
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    optimizer = PulsarCutOptimizer(
        pickle_dir=args.pickle_dir,
        min_excess_fraction=args.min_excess_fraction,
        min_signal_events=args.min_signal_events,
        p1_width=args.p1_width,
        p2_width=args.p2_width,
        off_width=args.off_width,
    )

    results = optimizer.optimize(metric=args.metric)
    optimizer.plot_results(results, output_dir=args.output_dir)

    print("\n=== RESULTS ===")
    print(f"Energy edges (TeV): {results['energy_edges']}")
    print(f"Gammaness EVEN:     {results['gammaness_even']}")
    print(f"Gammaness ODD:      {results['gammaness_odd']}")
    print(f"Theta2 EVEN (deg²): {results['theta2_even']}")
    print(f"Theta2 ODD  (deg²): {results['theta2_odd']}")