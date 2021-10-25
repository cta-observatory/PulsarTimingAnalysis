import math
import astropy as ast
import matplotlib.pyplot as plt
import numpy as np
import os
from gammapy.stats import WStatCountsStatistic
import warnings


def calculate_CountStats(on_file,off_file=None,factor=None):
    if off_file is None:
        raise ValueError('No off data given for the pulsar analysis')
        
    Non=len(on_file)
    Noff=len(off_file)
    stat = WStatCountsStatistic(n_on=Non, n_off=Noff, alpha=factor)
    yerr=np.sqrt(Non + ((factor** 2) * Noff))
    
    return(stat,yerr,Noff*factor)


class PulsarPeak():
    '''
    A class to manipulate and store information about the statistics of certain regions in a phase list object. 2 types of regions:
        1. Signal region: peaks present in the lightcurve
        2. Background region: background statistics
    
    
    Parameters
    ----------
    lc_data : PulsarPhases class object
        PulsarPhases object from which we extract the pulsar phases and the OFF region statistics
    peak_limits : list
        list of phase edges used to define the region.
        To define a non continuos region (for instance, the sum of two independent peaks), a list of four edges can be provided.
    peak_type : str, optional
        'Background' or 'Signal' type to calculate especific statistics.
    
    Note: To obtain peak statistics an OFF region must be defined in the phase list object first.
    
    
    Attributes
    ----------
    limits: str
        region phase edges
    type: str
        the type of region defined (signal or background)
    deltaP: float
        the total phase range of the region
    phases: numpy array
        list of phases that fall into the region
    number: int
        number of events in the region
    nregions: int
        number of phase intervals that define the region
    
    
    For 'signal' type only:

    sign: float
        Li&Ma Significance of the excess of events from the peak with respect to the background
    Nex: int
        Number of excess events in the peaks
    yerr: float
        Estimated error of the number of excess events
    sign_ratio: float
        Ratio of the significance and the square root of the time of observation.
    s_n_ratio: float
        Signal to noise ratio
    '''


    def __init__(self, lc_data, regions, peak_limits,peaktype='signal'):
        #Define the features of the region
        self.limits=peak_limits
        self.type=peaktype
        self.deltaP=0
        self.phases=np.array([])
        
        #Compute deltaP and phases for all the intervals that define the region
        if len(peak_limits) % 2 == 0:
            for i in range(1,len(self.limits),2):
                if self.limits[i-1]<self.limits[i]:
                    self.deltaP=self.deltaP+(self.limits[i]-self.limits[i-1])
                    self.phases=np.concatenate([self.phases,lc_data.phases[(lc_data.phases>self.limits[i-1]) & (lc_data.phases<self.limits[i])]])
                else:
                    self.deltaP=self.deltaP+(self.limits[i]-(self.limits[i-1]-1))
                    self.phases=np.concatenate([self.phases,lc_data.phases[(lc_data.phases>self.limits[i-1]) | (lc_data.phases<self.limits[i])]])
            self.number=len(self.phases)
            self.nregions=len(peak_limits)/2 
        else:
            raise ValueError('Wrong defined regions')
        
        #Make statistics if the region is signal type only
        if self.type=='signal':
                self.make_stats(lc_data,regions)
        
    def make_stats(self,lc_data,regions):
            stats,yerror,noff=calculate_CountStats(self.phases,off_file=regions.OFF.phases,factor=(self.deltaP)/regions.OFF.deltaP)
            self.sign=stats.sqrt_ts.item()
            self.Nex=stats.n_sig
            self.yerr=yerror
            self.sign_ratio=self.sign/np.sqrt(lc_data.tobs)
            self.s_n_ratio=self.Nex/np.sqrt(noff)
            
