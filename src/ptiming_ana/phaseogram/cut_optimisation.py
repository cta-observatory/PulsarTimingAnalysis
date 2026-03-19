from itertools import product
import numpy as np
import pandas as pd
import logging
from scipy.stats import norm, chi2 as chi2dist
from skopt import gp_minimize
from skopt.space import Real

logger = logging.getLogger(__name__)

METRIC_OPTIONS = ["sigma_P1", "sigma_P2", "sigma_P1P2", "Htest", "Ztest", "Chi2"]


def _compute_metric(dataframe, metric, P1_limits, P2_limits, OFF_limits):
    phases = dataframe["pulsar_phase"].values

    def count_in_region(limits):
        mask = np.zeros(len(phases), dtype=bool)
        for i in range(0, len(limits), 2):
            mask |= (phases >= limits[i]) & (phases < limits[i + 1])
        return mask.sum()

    off_width = sum(OFF_limits[i + 1] - OFF_limits[i] for i in range(0, len(OFF_limits), 2))
    Non_off   = count_in_region(OFF_limits)

    def lima_significance(Non, Noff, alpha):
        if Noff == 0 or Non == 0:
            return 0.0
        try:
            term1 = Non  * np.log((1 + alpha) / alpha * Non  / (Non + Noff))
            term2 = Noff * np.log((1 + alpha)          * Noff / (Non + Noff))
            return np.sqrt(2) * np.sqrt(term1 + term2)
        except Exception:
            return 0.0

    if metric == "sigma_P1" and P1_limits is not None:
        Non = count_in_region(P1_limits)
        sig_width = sum(P1_limits[i + 1] - P1_limits[i] for i in range(0, len(P1_limits), 2))
        return lima_significance(Non, Non_off, sig_width / off_width)

    elif metric == "sigma_P2" and P2_limits is not None:
        Non = count_in_region(P2_limits)
        sig_width = sum(P2_limits[i + 1] - P2_limits[i] for i in range(0, len(P2_limits), 2))
        return lima_significance(Non, Non_off, sig_width / off_width)

    elif metric == "sigma_P1P2" and P1_limits is not None and P2_limits is not None:
        p1p2_limits = P1_limits + P2_limits
        Non = count_in_region(p1p2_limits)
        sig_width = sum(p1p2_limits[i + 1] - p1p2_limits[i] for i in range(0, len(p1p2_limits), 2))
        return lima_significance(Non, Non_off, sig_width / off_width)

    elif metric in ("Htest", "Ztest", "Chi2"):
        n = len(phases)
        if n < 10:
            return 0.0
        if metric == "Htest":
            def z2m(m):
                return 2 / n * sum(
                    np.sum(np.cos(2 * np.pi * k * phases)) ** 2 +
                    np.sum(np.sin(2 * np.pi * k * phases)) ** 2
                    for k in range(1, m + 1)
                )
            H     = max(z2m(m) - 4 * (m - 1) for m in range(1, 21))
            p_val = max(np.exp(-0.4 * H), 1e-300)
            return norm.isf(p_val)
        elif metric == "Ztest":
            z2 = 2 / n * sum(
                np.sum(np.cos(2 * np.pi * k * phases)) ** 2 +
                np.sum(np.sin(2 * np.pi * k * phases)) ** 2
                for k in range(1, 3)
            )
            p_val = max(chi2dist.sf(z2, df=4), 1e-300)
            return norm.isf(p_val)
        elif metric == "Chi2":
            nbins_hist = 20
            counts, _  = np.histogram(phases, bins=nbins_hist, range=(0, 1))
            expected   = n / nbins_hist
            chi2_val   = np.sum((counts - expected) ** 2 / expected)
            p_val      = max(chi2dist.sf(chi2_val, df=nbins_hist - 1), 1e-300)
            return norm.isf(p_val)

    return 0.0


def _smooth_cuts(cuts, smoothness_weight=0.3):
    """
    Apply Gaussian smoothing to a list of cut values.
    smoothness_weight controls the strength: 0 = no smoothing, 1 = full smoothing.
    """
    cuts = np.array(cuts, dtype=float)
    if len(cuts) <= 2:
        return cuts
    from scipy.ndimage import gaussian_filter1d
    sigma   = max(1.0, smoothness_weight * len(cuts) / 5.0)
    smoothed = gaussian_filter1d(cuts, sigma=sigma)
    return smoothed


def _apply_energy_bin_cuts(df, energy_edges, gammaness_cuts, theta2_cuts):
    frames = []
    for i in range(len(energy_edges) - 1):
        emin_b, emax_b = energy_edges[i], energy_edges[i + 1]
        mask   = (df["energy"] >= emin_b) & (df["energy"] < emax_b)
        df_bin = df[mask].copy()
        if len(df_bin) == 0:
            continue
        df_bin = df_bin[df_bin["gammaness"] > gammaness_cuts[i]]
        df_bin = df_bin[df_bin["theta2"]    < theta2_cuts[i]]
        frames.append(df_bin)
    if len(frames) == 0:
        return df.iloc[0:0]
    return pd.concat(frames)


def optimize_cuts(
    df,
    metric="sigma_P1P2",
    P1_limits=None,
    P2_limits=None,
    OFF_limits=None,
    emin=0.02,
    emax=100.0,
    nbins=13,
    gammaness_range=(0.5, 0.95),
    theta2_range=(0.01, 0.15),
    min_events_per_bin=50,
    n_calls=50,               # Bayesian optimization calls per bin
    smoothness_weight=0.3,    # 0 = no smoothing, 1 = heavy smoothing
    smoothness_penalty=0.5,   # penalty weight for jumps between adjacent bins
    irf_gammaness=None,       # optional: array of IRF gammaness cuts (used as x0)
    irf_theta2=None,          # optional: array of IRF theta2 cuts (used as x0)
):
    """
    Optimize gammaness and theta2 cuts per energy bin using even/odd event splitting
    and Bayesian optimization (scikit-optimize) with smoothness regularization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: event_id, energy, gammaness, theta2, pulsar_phase.
    metric : str
        Metric to optimize. One of: sigma_P1, sigma_P2, sigma_P1P2, Htest, Ztest, Chi2.
    P1_limits, P2_limits, OFF_limits : list
        Phase region limits.
    emin, emax : float
        Energy range in TeV.
    nbins : int
        Number of energy bins (log-spaced). Set to None to auto-determine from min_events_per_bin.
    gammaness_range, theta2_range : tuple
        Search bounds for each parameter.
    min_events_per_bin : int
        Minimum events per bin to attempt optimization.
    n_calls : int
        Number of Bayesian optimization evaluations per bin.
    smoothness_weight : float
        Gaussian smoothing applied post-optimization (0=none, 1=heavy).
    smoothness_penalty : float
        Penalty added to objective for large jumps between adjacent bins.
    irf_gammaness, irf_theta2 : array-like, optional
        IRF cut values to use as initial points for the optimizer.
        If provided, must have length == nbins.

    Returns
    -------
    df_cuts_even, df_cuts_odd : pd.DataFrame
        Optimal cuts per energy bin derived from even/odd events.
    df_even_filtered, df_odd_filtered : pd.DataFrame
        Cross-filtered datasets.
    df_full_optimized : pd.DataFrame
        Full combined optimized dataset.
    """
    if metric not in METRIC_OPTIONS:
        raise ValueError(f"metric must be one of {METRIC_OPTIONS}")
    if OFF_limits is None:
        raise ValueError("OFF_limits must be provided")

    logger.info(f"Total events received: {len(df)}")

    # Split even/odd
    even_mask = df["event_id"] % 2 == 0
    df_even   = df[even_mask].reset_index(drop=True)
    df_odd    = df[~even_mask].reset_index(drop=True)
    logger.info(f"Even events: {len(df_even)}, Odd events: {len(df_odd)}")

    # Auto-determine nbins if not provided
    if nbins is None:
        # Use the half-dataset (even) to determine bins with enough events
        total_half  = len(df_even)
        nbins = max(5, int(total_half / (min_events_per_bin * 10)))
        logger.info(f"Auto-determined nbins={nbins}")

    energy_edges = np.geomspace(emin, emax, nbins + 1)
    g_default    = (gammaness_range[0] + gammaness_range[1]) / 2
    t_default    = (theta2_range[0]    + theta2_range[1])    / 2

    def optimize_on_dataset(df_train, label):
        gammaness_cuts = []
        theta2_cuts    = []
        elow_list      = []
        ehigh_list     = []
        prev_g         = None
        prev_t         = None

        for i in range(len(energy_edges) - 1):
            emin_b, emax_b = energy_edges[i], energy_edges[i + 1]
            elow_list.append(emin_b)
            ehigh_list.append(emax_b)

            mask   = (df_train["energy"] >= emin_b) & (df_train["energy"] < emax_b)
            df_bin = df_train[mask]

            if len(df_bin) < min_events_per_bin:
                logger.warning(
                    f"[{label}] bin [{emin_b:.3f}, {emax_b:.3f}] TeV: "
                    f"only {len(df_bin)} events, using default cuts"
                )
                g_best = prev_g if prev_g is not None else g_default
                t_best = prev_t if prev_t is not None else t_default
                gammaness_cuts.append(g_best)
                theta2_cuts.append(t_best)
                prev_g, prev_t = g_best, t_best
                continue

            # Initial point: IRF values if provided, else previous bin or midpoint
            if irf_gammaness is not None and i < len(irf_gammaness):
                x0_g = float(np.clip(irf_gammaness[i], *gammaness_range))
            else:
                x0_g = prev_g if prev_g is not None else g_default

            if irf_theta2 is not None and i < len(irf_theta2):
                x0_t = float(np.clip(irf_theta2[i], *theta2_range))
            else:
                x0_t = prev_t if prev_t is not None else t_default

            def objective(params):
                g_cut, t_cut = params
                df_filtered = df_bin[
                    (df_bin["gammaness"] > g_cut) &
                    (df_bin["theta2"]    < t_cut)
                ]
                if len(df_filtered) < 10:
                    return 0.0

                val = _compute_metric(df_filtered, metric, P1_limits, P2_limits, OFF_limits)

                # Smoothness penalty: penalize large jumps from previous bin
                penalty = 0.0
                if prev_g is not None:
                    dg = abs(g_cut - prev_g) / (gammaness_range[1] - gammaness_range[0])
                    dt = abs(t_cut - prev_t) / (theta2_range[1]    - theta2_range[0])
                    penalty = smoothness_penalty * (dg + dt)

                return -(val - penalty)   # minimize negative = maximize (metric - penalty)

            space = [
                Real(gammaness_range[0], gammaness_range[1], name="gammaness"),
                Real(theta2_range[0],    theta2_range[1],    name="theta2"),
            ]

            result = gp_minimize(
                objective,
                space,
                n_calls=n_calls,
                x0=[[x0_g, x0_t]],
                n_initial_points=max(5, n_calls // 5),
                noise=1e-6,
                random_state=42,
                verbose=False,
            )

            g_best = float(np.clip(result.x[0], *gammaness_range))
            t_best = float(np.clip(result.x[1], *theta2_range))

            logger.info(
                f"[{label}] bin [{emin_b:.3f}, {emax_b:.3f}] TeV: "
                f"gammaness={g_best:.3f}, theta2={t_best:.4f}, "
                f"{metric}={-result.fun:.2f}"
            )

            gammaness_cuts.append(g_best)
            theta2_cuts.append(t_best)
            prev_g, prev_t = g_best, t_best

        # Post-optimization smoothing
        if smoothness_weight > 0:
            gammaness_cuts = list(_smooth_cuts(gammaness_cuts, smoothness_weight))
            theta2_cuts    = list(_smooth_cuts(theta2_cuts,    smoothness_weight))
            logger.info(f"[{label}] Applied smoothing (weight={smoothness_weight})")

        return pd.DataFrame({
            "energy_low":  elow_list,
            "energy_high": ehigh_list,
            "gammaness":   gammaness_cuts,
            "theta2":      theta2_cuts,
        })

    logger.info(f"Optimizing cuts on EVEN events (Bayesian, {n_calls} calls/bin)...")
    df_cuts_even = optimize_on_dataset(df_even, label="EVEN")

    logger.info(f"Optimizing cuts on ODD events (Bayesian, {n_calls} calls/bin)...")
    df_cuts_odd  = optimize_on_dataset(df_odd,  label="ODD")

    # Cross-apply cuts
    logger.info("Applying EVEN cuts to ODD events...")
    df_odd_filtered = _apply_energy_bin_cuts(
        df_odd,  energy_edges,
        df_cuts_even["gammaness"].values,
        df_cuts_even["theta2"].values,
    )

    logger.info("Applying ODD cuts to EVEN events...")
    df_even_filtered = _apply_energy_bin_cuts(
        df_even, energy_edges,
        df_cuts_odd["gammaness"].values,
        df_cuts_odd["theta2"].values,
    )

    # Full optimized dataset
    df_full_optimized = (
        pd.concat([df_even_filtered, df_odd_filtered])
        .sort_values("dragon_time")
        .reset_index(drop=True)
    )

    logger.info(
        f"Final dataset: {len(df_full_optimized)} events "
        f"(even filtered: {len(df_even_filtered)}, odd filtered: {len(df_odd_filtered)})"
    )

    for lbl, dff in [
        ("EVEN (odd cuts)", df_even_filtered),
        ("ODD (even cuts)", df_odd_filtered),
        ("FULL",            df_full_optimized),
    ]:
        val = _compute_metric(dff, metric, P1_limits, P2_limits, OFF_limits)
        logger.info(f"Final {metric} on {lbl}: {val:.3f} sigma")

    return df_cuts_even, df_cuts_odd, df_even_filtered, df_odd_filtered, df_full_optimized