import numpy as np


class PhaseBinning:
    """
    A class to define the binning used in the construction of the lightcurve. It allows to change, manipulate and find the phase binning of the histogram


    Parameters
    ----------
    bins : int, array
        If an integer is provided, it is used as the number fix-width bins in the lightcurve.
        If an array are provided, that list will be used as the bin edges of the lightcurve.
    xmin : float
        Lower edge of the lightcuve binning
    xmax : float
        Higher edge of the lightcuve binning


    Attributes
    ----------
    nbins : int
        Number of bins used in the lightcurve
    xmin : float
        Lower edge of the lightcuve binning
    xmax : float
        Higher edge of the lightcuve binning
    bins: list
        List of bin edges of the lightcuve
    """

    def __init__(self, bins, xmin=None, xmax=None):
        if isinstance(bins, int):
            self.nbins = bins
            if xmin is not None:
                self.xmin = xmin
            else:
                self.xmin = 0
            if xmax is not None:
                self.xmax = xmax
            else:
                self.xmax = 1
            self.set_edges()

        elif isinstance(bins, (list, np.ndarray)):
            self.edges = bins
            self.nbins = len(bins) - 1
            self.xmin = bins[0]
            self.xmax = bins[-1]

    def getNumEdges(self):
        return self.nbins + 1

    def set_edges(self, nbins=None, xmin=None, xmax=None):
        if nbins is not None:
            self.nbins = nbins
            if xmax is not None:
                self.xmax = xmax
            if xmin is not None:
                self.xmin = xmin
        else:
            if xmax is not None:
                self.xmax = xmax
            if xmin is not None:
                self.xmin = xmin
        self.edges = np.linspace(self.xmin, self.xmax, self.nbins + 1)

    def Find_LowHiEdge(self, value):
        for i in range(1, self.getNumEdges()):
            if self.edges[i] >= value:
                return (i - 1, i)
        return None

    def Find_CloseEdge(self, value):
        if value < self.xmin:
            return 0
        if value > self.xmax:
            return self.getNumEdges()
        for i in range(1, self.getNumEdges()):
            if self.edges[i] >= value:
                if (self.edges[i] - value) < (value - self.edges[i - 1]):
                    return i
                else:
                    return i - 1
