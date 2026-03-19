import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ptiming_ana.phaseogram import PulsarAnalysis, FilterPulsarAna
from ptiming_ana.phaseogram.phase_regions import PulsarPeak


def plot_histograms_by_theta2(df_full, phases_all, results, theta2_values, gammaness_values, off_limits, target_counts, target_is_mean=True, nbins=60):
    """
    Plotea histogramas con el gammaness óptimo para cada valor de theta2.
    
    Parameters
    ----------
    df_full : pd.DataFrame
        DataFrame con las columnas 'gammaness' y 'theta2'
    phases_all : np.ndarray
        Array de fases
    results : dict
        Diccionario {(g, t2): valor}
    theta2_values : array-like
        Valores de theta2 escaneados
    gammaness_values : array-like
        Valores de gammaness escaneados
    off_limits : list
        Límites de la región OFF [min, max]
    target_counts : float
        Valor objetivo
    target_is_mean : bool
        Si True, interpreta como media por bin
    nbins : int
        Número de bins para el histograma
    """
    n_theta2 = len(theta2_values)
    ncols = min(3, n_theta2)
    nrows = (n_theta2 + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, t2 in enumerate(theta2_values):
        ax = axes[idx]
        
        # Encontrar el gammaness óptimo para este theta2
        candidates = {g: results[(g, t2)] for g in gammaness_values if (g, t2) in results}
        best_g_for_t2 = min(candidates.keys(), key=lambda g: abs(candidates[g] - target_counts))
        
        # Extraer fases para esta combinación
        mask_t2 = df_full["theta2"] < t2 if "theta2" in df_full.columns else np.ones(len(df_full), dtype=bool)
        mask_g = df_full["gammaness"] > best_g_for_t2 if "gammaness" in df_full.columns else np.ones(len(df_full), dtype=bool)
        mask = mask_t2 & mask_g
        phases = phases_all[mask]
        phases = np.where(phases < 0, phases + 1, phases)
        
        # Histograma
        counts, bins = np.histogram(phases, bins=nbins, range=(0, 1))
        
        # Plot
        ax.step(bins[:-1], counts, where='mid', color='steelblue', linewidth=2, label='Data')
        ax.scatter((bins[:-1] + bins[1:]) / 2, counts, color='black', s=30, zorder=3, alpha=0.6)
        
        # OFF region (sombreado)
        ax.axvspan(off_limits[0], off_limits[1], alpha=0.2, color='gray', label='OFF Region')
        ax.axvspan(off_limits[0] + 1, off_limits[1] + 1, alpha=0.2, color='gray')
        
        # Media en OFF (línea horizontal)
        sel = (bins[:-1] > off_limits[0]) & (bins[1:] < off_limits[1])
        if sel.any():
            mean_off = np.mean(counts[sel])
            ax.axhline(mean_off, linestyle='--', color='red', linewidth=2, alpha=0.7, label=f'OFF Mean: {mean_off:.2f}')
        
        # Labels y títulos
        ax.set_xlabel('Pulsar Phase')
        ax.set_ylabel('Events')
        ax.set_title(f'θ² = {t2:.3f}, γ = {best_g_for_t2:.3f}\nValue: {candidates[best_g_for_t2]:.1f}')
        ax.set_xlim(0, 2)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Eliminar subplots vacíos
    for idx in range(n_theta2, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig


def find_gammaness_for_background(dirname, off_limits=[0.52,0.87], target_counts=123600, gammaness_values=None, theta2_values=None, src_dep=False, ctlearn=False, target_is_mean=True):
    """
    Escanea valores de gammaness y theta2 para encontrar la combinación que da
    un número de cuentas en la región OFF cercano al target.

    Parameters
    ----------
    dirname : str
        Path al directorio con archivos DL2 .h5
    off_limits : list
        Límites de la región OFF, e.g., [0.52, 0.87]
    target_counts : int or float
        Número objetivo de cuentas en OFF (total) o media por bin si
        ``target_is_mean=True``.
    gammaness_values : array-like, optional
        Array de valores de gammaness a probar.
        Si None, usa np.linspace(0.5, 0.9, 9)
    theta2_values : array-like, optional
        Array de valores de theta2 a probar.
        Si None, usa [0.2, 0.25, 0.3, 0.35, 0.4]
    src_dep : bool
        Si los datos son source-dependent
    ctlearn : bool
        Si usar ctlearn predictions
    target_is_mean : bool, optional
        Si True, interpreta ``target_counts`` como la media por bin que
        corresponde a la línea horizontal del faseograma en la región OFF.

    Returns
    -------
    best_g : float
        Valor de gammaness óptimo
    best_t2 : float
        Valor de theta2 óptimo
    best_counts : int or float
        Número de cuentas correspondiente
    results : dict
        Diccionario con claves (g, t2) y valores de cuentas
    fig : matplotlib.figure.Figure
        Figura con histogramas para cada theta2
    """
    
    # Default values
    if gammaness_values is None:
        gammaness_values = np.linspace(0.5, 0.9, 9)
    else:
        gammaness_values = np.asarray(gammaness_values)
    
    if theta2_values is None:
        theta2_values = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    else:
        theta2_values = np.asarray(theta2_values)
 
    # Crear PulsarAnalysis, cargando con los máximos de los arrays para ahorrar memoria
    h = PulsarAnalysis()
    h.setLSTInputFile(dirname=dirname, src_dep=src_dep, ctlearn=ctlearn)
    h.setBackgroundLimits(off_limits)
    h.setPeaklimits(P1_limits=[0,0.026,0.983,1], P2_limits=None, P3_limits=None)
    h.setBinning(60)

    # Cargar con cortes superiores: máximo gammaness y máximo theta2
    h.filter_data = True
    g_min = gammaness_values.min()
    t2_max = theta2_values.max()
    h.setParamCuts(gammaness_cut=g_min, theta2_cut=t2_max)
    h.initialize()
    
    print(f"Datos cargados con gammaness > {g_min:.3f} y theta2 < {t2_max:.3f}")

    # Dataframe completo ya puede estar parcialmente filtrado
    df_full = h.r.info

    # drop columnas innecesarias para reducir uso de RAM
    cols = ["pulsar_phase"]
    if "gammaness" in df_full.columns:
        cols.append("gammaness")
    if "theta2" in df_full.columns:
        cols.append("theta2")
    df_full = df_full[cols]

    # pre-calculamos máscaras fijas para acelerar el bucle
    mask_base = np.ones(len(df_full), dtype=bool)
    phases_all = df_full["pulsar_phase"].values

    results = {}

    for t2 in theta2_values:
        mask_t2 = (
            df_full["theta2"] < t2
            if "theta2" in df_full.columns
            else mask_base
        )
        
        for g in gammaness_values:
            # máscara de gammaness
            mask_g = (
                df_full["gammaness"] > g
                if "gammaness" in df_full.columns
                else mask_base
            )
            mask = mask_t2 & mask_g

            # seleccionamos fases y ajustamos
            phases = phases_all[mask]
            phases = np.where(phases < 0, phases + 1, phases)

            if target_is_mean:
                nbins = h.binning.nbins if hasattr(h, "binning") else 60
                counts, bins = np.histogram(phases, bins=nbins, range=(0, 1))
                sel = (bins[:-1] > off_limits[0]) & (bins[1:] < off_limits[1])
                off_value = np.mean(counts[sel]) if sel.any() else 0.0
            else:
                off_peak = PulsarPeak(peak_limits=off_limits, peaktype="background")
                off_peak.fillPeak(phases)
                off_value = off_peak.number

            results[(g, t2)] = off_value
            print(f"Gammaness: {g:.3f}, Theta2: {t2:.3f}, Value: {off_value:.1f}")

    # Encontrar el más cercano al target
    closest_key = min(results.keys(), key=lambda x: abs(results[x] - target_counts))
    best_g, best_t2 = closest_key
    best_counts = results[closest_key]

    print(f"\nMejor valor: gammaness={best_g:.3f}, theta2={best_t2:.3f}, valor={best_counts:.1f} (target: {target_counts})")

    # Plotear histogramas para cada theta2 con su gammaness óptimo
    nbins = h.binning.nbins if hasattr(h, "binning") else 60
    fig = plot_histograms_by_theta2(
        df_full, phases_all, results, theta2_values, gammaness_values, 
        off_limits, target_counts, target_is_mean=target_is_mean, nbins=nbins
    )
    plt.show()

    return best_g, best_t2, best_counts, results, fig

