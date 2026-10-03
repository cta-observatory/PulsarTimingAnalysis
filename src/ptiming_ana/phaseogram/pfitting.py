import logging

import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from .models import MODELS, get_model_list

logger = logging.getLogger(__name__)

__all__ = ["PeakFitting"]

# Number of points used to integrate the model over one period in the
# unbinned (extended likelihood) fit.
_N_INTEGRATION_POINTS = 4001


class PeakFitting:
    """
    Fit of the pulse profile to one of the models in ``ptiming_ana.phaseogram.models``.

    Parameters
    ----------
    binned : bool
        If True, a least-squares fit to the phaseogram is done. If False, an
        unbinned extended maximum likelihood fit to the list of phases is done.
    model : str
        Name of the model. See `~ptiming_ana.phaseogram.get_model_list`.
    peak : {"both", "P1", "P2"}
        Peak(s) to fit. Models with a single peak need "P1" or "P2", the rest
        of the models need "both".

    Attributes
    ----------
    parnames : list of str
        Name of the parameters of the model.
    params, errors : list of float
        Best-fit values and uncertainties, in the same order as ``parnames``.
        The amplitudes are given in counts per phase bin times phase and the
        background in counts per phase bin, for both the binned and the
        unbinned fit.
    df_result : `pandas.DataFrame`
        Table with the result of the fit.
    """

    def __init__(self, binned, model, peak="both"):
        # Define and check model
        self.model = model
        self.shift = 0
        self.peak_tofit = peak
        self.check_model()
        self.binned = binned

    ##############################################
    # EXECUTION
    #############################################

    def run(self, pulsar_phases):
        # Estimate initial values
        self.est_initial_values(pulsar_phases)

        # Do the fitting
        if self.binned:
            self.fit_Binned(pulsar_phases)
        else:
            self.fit_ULmodel(pulsar_phases)

    def check_model(self):
        if self.model not in get_model_list():
            raise ValueError(
                f"The model '{self.model}' is not in the available model list: "
                f"{get_model_list()}"
            )

        if self.peak_tofit not in ("both", "P1", "P2"):
            raise ValueError("The peak to fit must be one of 'both', 'P1' or 'P2'")

        npeaks = MODELS[self.model]["npeaks"]
        if npeaks == 1 and self.peak_tofit == "both":
            raise ValueError(
                f"The '{self.model}' model can only fit one peak (P1 or P2)"
            )

        if npeaks > 1 and self.peak_tofit != "both":
            raise ValueError(
                f"The '{self.model}' model needs {npeaks} peaks, "
                f"but only {self.peak_tofit} is set to be fitted"
            )

    def _region_names(self):
        npeaks = MODELS[self.model]["npeaks"]
        if npeaks == 1:
            return [self.peak_tofit]
        return ["P1", "P2", "P3"][:npeaks]

    def est_initial_values(self, pulsar_phases):
        self.check_model()
        self.init = []
        height = []

        # Mean and width of each peak estimated from the signal regions
        for name in self._region_names():
            P_info = pulsar_phases.regions.dic.get(name)
            if P_info is None:
                raise ValueError(
                    f"The '{self.model}' model needs the {name} region to be defined"
                )

            height.append(P_info.Nex)
            self.shift = (
                pulsar_phases.regions.OFF.limits[1]
                + pulsar_phases.regions.OFF.limits[0]
            ) / 2

            if len(P_info.limits) > 2:
                extension = (P_info.limits[0] + 1 + P_info.limits[3]) / 2
            else:
                extension = (P_info.limits[0] + P_info.limits[1]) / 2

            if extension < self.shift:
                extension = extension + 1

            self.init.extend([extension, P_info.deltaP / 2])

            if self.model == "asym_dgaussian":
                self.init.append(P_info.deltaP / 2)

        # Background level estimated from the OFF region
        lc = pulsar_phases.histogram.lc
        off_limits = pulsar_phases.regions.OFF.limits
        bkg = np.mean(lc[0][(lc[1][:-1] > off_limits[0]) & (lc[1][1:] < off_limits[1])])

        self.init.extend(height)
        self.init.append(bkg)

    def _model_with_fixed_background(self):
        """Model as a function of the free parameters (background is fixed)."""
        spec = MODELS[self.model]
        function = spec["function"]
        bkg_index = spec["parnames"].index(spec["background"])
        bkg = float(self.init[-1])

        def model(x, *free_params):
            params = list(free_params)
            params.insert(bkg_index, bkg)
            return function(x, *params)

        return model, bkg_index

    def _store_result(self, values, errors):
        self.parnames = list(MODELS[self.model]["parnames"])
        self.params = [float(value) for value in values]
        self.errors = [float(error) for error in errors]
        self.create_result_df()

    # Unbinned fitting
    def fit_ULmodel(self, pulsar_phases):
        """Unbinned extended maximum likelihood fit to the list of phases."""
        self.check_model()
        spec = MODELS[self.model]
        function = spec["function"]
        parnames = list(spec["parnames"])
        bkg_index = parnames.index(spec["background"])

        # Shift the phases if one of the peak is near the interval edge.
        # A copy is used so the phases of the analysis are not modified.
        phases = np.array(pulsar_phases.phases, dtype=float)
        if self.shift != 0:
            phases[phases < self.shift] += 1

        # The models are expressed in counts per phase bin, so the density
        # of events per unit phase is the model times the number of bins.
        nbins = len(pulsar_phases.histogram.lc[0])
        grid = np.linspace(self.shift, self.shift + 1, _N_INTEGRATION_POINTS)

        def scaled_pdf(x, *params):
            integral = nbins * trapezoid(function(grid, *params), grid)
            return integral, nbins * function(x, *params)

        # Initial values: the excess counts are converted to the units of the
        # amplitude parameters (counts per bin x phase).
        npeaks = spec["npeaks"]
        nshape = len(self.init) - npeaks - 1
        start = list(self.init[:nshape])
        start.extend(np.array(self.init[nshape:-1]) / nbins)
        start.insert(bkg_index, self.init[-1])

        minuit = Minuit(
            cost.ExtendedUnbinnedNLL(phases, scaled_pdf), *start, name=parnames
        )
        for name in parnames:
            if name.startswith("mu"):
                minuit.limits[name] = (self.shift, self.shift + 1)
            elif name.startswith(("sigma", "gamma")):
                minuit.limits[name] = (1e-4, 0.5)
            else:
                minuit.limits[name] = (0, None)

        minuit.migrad()
        minuit.hesse()
        if not minuit.valid:
            logger.warning("The unbinned fit did not converge to a valid minimum")

        # Store results as minuit object
        self.minuit = minuit
        self.unbinned_lk = minuit.fcn

        self._store_result(
            [minuit.values[name] for name in parnames],
            [minuit.errors[name] for name in parnames],
        )

    # Binned fitting
    def fit_Binned(self, pulsar_phases):
        """Least-squares fit to the phaseogram (background fixed to the OFF level)."""
        self.check_model()
        histogram = pulsar_phases.histogram

        # Shift the phases if one of the peak is near the interval edge
        left_edges = np.array(histogram.lc[1][:-1], dtype=float)
        bin_height = np.array(histogram.lc[0])

        if self.shift != 0:
            left_edges[left_edges < self.shift] += 1

        order = np.argsort(left_edges, kind="stable")
        bin_height = bin_height[order]
        edges = np.append(left_edges[order], left_edges[order][0] + 1)
        bin_centres = (edges[1:] + edges[:-1]) / 2

        # Poissonian uncertainties. Empty bins are given unit uncertainty to
        # keep the residuals finite.
        sigma = np.sqrt(bin_height)
        sigma[sigma == 0] = 1

        model, bkg_index = self._model_with_fixed_background()
        params, pcov = curve_fit(
            model,
            bin_centres,
            bin_height,
            sigma=sigma,
            p0=self.init[:-1],
        )

        # Store the result of params and errors (the background is not fitted)
        values = list(params)
        errors = list(np.sqrt(np.diag(pcov)))
        values.insert(bkg_index, self.init[-1])
        errors.insert(bkg_index, 0)

        self._store_result(values, errors)

    ##############################################
    # RESULTS
    #############################################

    def check_fit_result(self):
        return hasattr(self, "params")

    def create_result_df(self):
        d = {"Name": self.parnames, "Value": self.params, "Error": self.errors}
        self.df_result = pd.DataFrame(data=d)

    def show_result(self):
        if not hasattr(self, "df_result"):
            logger.warning("No fit has been done so far")
            return None
        return self.df_result
