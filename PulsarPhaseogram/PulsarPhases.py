import pandas as pd
import math
import astropy as ast
import numpy as np
from astropy.time import Time
from astropy import units as u
import warnings
from lstchain.reco.utils import get_effective_time,add_delta_t_key
from .PTimeEvolution import PulsarTimeEvolution
from .PhaseRegions import PhaseRegions
from .Lightcurve import Lightcurve
from .PeriodicityTest import PeriodicityTest
from .PFitting import PeakFitting


__all__ = ['PulsarPhases','FermiPulsarPhases']
pd.options.mode.chained_assignment = None

class PulsarPhases():

    '''
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.
    
    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set daraframe to False if it is not available and want to set the attributes manually.
    pdata : List of float
        List of phases (in case no dataframe is available).
    ptimes: List of float
        List of mjd times (in case no dataframe is available).
    tobservation: float
        Total effective time of observation
    peak_limits_1: tuple
        Edges of the first peak. Set to None if no P1 is present.
    peak_limits_2: tuple
        Edges of the second peak. Set to None if no P2 is present.
    off_limits: tuple
        Edges of the OFF region
    binned: boolean
        True for a binned fitting, False for an unbinned fitting.

        
    Attributes
    ----------
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    tobs : float
        Effective time of observation in hours
    regions: PhaseRegions object
        Information of the OFF/signal regions
    histogram: Lightcurve object
        Information of the phaseogram
    stats: PeriodicityTest object
        Information of Statistical Tests for searching Periodicity
    fitting: PeakFitting object
        Information abot the fitting used for the peaks
    '''
    
    def __init__(self, dataframe, pdata=None, ptimes=None, tobservation=None,nbins=50,peak_limits_1=[0.983-1,0.026],peak_limits_2=[0.377,0.422],off_limits=[-0.48,0.87-1],binned=True,model='dgaussian',tinterval=3600):
    
        #Create the dataframe from the list (or the FITS table)
        if dataframe is None:
            if pdata==None:
                raise ValueError('No dataframe or phase list is provided') 
            if ptimes==None:
                raise ValueError('No dataframe or time list provided')
            else:
                dataframe = pd.DataFrame({"dragon_time":ptimes,"pulsar_phase":pdata})   
                dataframe=add_delta_t_key(dataframe)
                self.phases=np.array(dataframe['pulsar_phase'].to_list())
                self.times=np.array(dataframe['dragon_time'].to_list())
                self.info=dataframe
        else:
            try:
                dataframe=add_delta_t_key(dataframe)
                self.phases=np.array(dataframe['pulsar_phase'].to_list())
                self.times=np.array(dataframe['dragon_time'].to_list())
                self.mjd_times=np.array(dataframe['mjd_time'].to_list())
                self.info=dataframe
            except:
                raise ValueError('Dataframe has no valid format') 

                
        #If the range of phases is not [0,1], shift the negative ones (the 0 phase is the same)
        self.shift_phases()
        
        
        #Update the information every 1 hour and store final values
        self.TimeEv=PulsarTimeEvolution(self,nbins,peak_limits_1,peak_limits_2,off_limits,tint=tinterval)
       
        if tobservation is not None:
            self.tobs=tobservation
                
        #Fit the histogram using PeakFitting class. If binned is False, an Unbinned Likelihood method is used for the fitting
        try:
            self.fitting=PeakFitting(self,binned,model)
        except:
            print('No fit could be done')
            
    def shift_phases(self):
        for i in range(0,len(self.phases)):
            if self.phases[i]<0:
                self.phases[i]+=1
                
        self.info['pulsar_phase']=self.phases
                            
    def calculate_tobs(self):
         return(get_effective_time(self.info)[1].value/3600)
            
    def update_info(self,nbins,peak_limits_1,peak_limits_2,off_limits):
        #Creates the regions using input limits and apply statistic using the PhaseRegion class.
        self.regions=PhaseRegions(self,off_limits,peak_limits_1,peak_limits_2) 
        
        #Create the phaseogram using the Lightcurve class
        self.histogram=Lightcurve(self,nbins)
        
        #Apply Periodicity stats and store them using the PeriodicityTest Class
        self.stats=PeriodicityTest(self)
        
        
        

class FermiPulsarPhases(PulsarPhases):
    
        def __init__(self, fits_table,nbins=50,tobservation=None,peak_limits_1=[0.983-1,0.026],peak_limits_2=[0.377,0.422],off_limits=[-0.48,0.87-1],binned=True,model='dgaussian',tinterval=3600*24*10):
            
            self.create_df_from_info(fits_table)
            self.phases=np.array(self.info['pulsar_phase'].to_list())
            self.mjd_times=np.array(self.info['mjd_time'].to_list())               
            
            #If the range of phases is not [0,1], shift the negative ones (the 0 phase is the same)
            self.shift_phases()
        
            #Update the information every 1 hour and store final values
            self.TimeEv=PulsarTimeEvolution(self,nbins,peak_limits_1,peak_limits_2,off_limits,tint=tinterval)
            
            if tobservation is not None:
                self.tobs=tobservation
            
            #Fit the histogram using PeakFitting class. If binned is False, an Unbinned Likelihood method is used for the fitting
            try:
                self.fitting=PeakFitting(self,binned,model)
            except:
                print('No fit could be done')

            
        def create_df_from_info(self,fits_table):
            time=fits_table['BARYCENTRIC_TIME'].byteswap().newbyteorder()
            phases=fits_table['PULSE_PHASE'].byteswap().newbyteorder()
            dataframe = pd.DataFrame({"mjd_time":time,"pulsar_phase":phases,"dragon_time":time*3600*24})
            self.info=dataframe
            return(self.info)
            
                
        def calculate_tobs(self):
            diff=self.mjd_times[1:]-self.mjd_times[0:-1]
            diff[diff>10/24]=0
            return(sum(diff)*24)
            

            
            