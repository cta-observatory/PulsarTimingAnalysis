import pandas as pd
import math
import astropy as ast
import numpy as np
import os
from astropy.time import Time
from astropy import units as u
import warnings
from lstchain.reco.utils import get_effective_time,add_delta_t_key
from .PulsarPhases import PulsarPhases, FermiPulsarPhases
import matplotlib.pyplot as plt
from astropy.io import fits

__all__ = ['PulsarPhasesArray','FermiPulsarPhasesArray']


class PulsarPhasesArray():
    
    '''
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.
    
    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set daraframe to False if it is not available and want to set the attributes manually.
    energy_edges: List of float
	Edges of the energy binning (in TeV)
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
    energy_edges: list of float
        List of edges for the energy binning
    energy_centres: list of float
        List of center of the energy bins
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    energies; list of float
        List of energies
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
    
    def __init__(self, dataframe, energy_edges,pdata=None, ptimes=None, penergies=None, tobservation=None,nbins=50,peak_limits_1=[0.983,0.026],peak_limits_2=[0.377,0.422],off_limits=[0.52,0.87],binned=True,model='dgaussian'):
        
        
        self.energy_edges=np.array(energy_edges)
        self.energy_centres=(self.energy_edges[1:]+self.energy_edges[:-1])/2
        
        #Create the dataframe from the list (or the FITS table)
        if dataframe is None:
            if pdata==None:
                raise ValueError('No dataframe or phase list is provided') 
            else:
                dataframe = pd.DataFrame({"dragon_time":ptimes,"pulsar_phase":pdata,"reco_energy":penergies})   
        try:
            self.phases=np.array(dataframe['pulsar_phase'].to_list())
            self.times=np.array(dataframe['dragon_time'].to_list())
            self.mjd_times=np.array(dataframe['mjd_time'].to_list()) 
            
            self.info=dataframe
        except:
            raise ValueError('Dataframe has no valid format')               
       
        try:                    
            dataframe=add_delta_t_key(dataframe)
            if tobservation is not None:
                self.tobs=tobservation
            else:
                self.tobs=self.calculate_tobs()
        except:
            print('No tobs estimates')
            
        #Create array of PulsarPhases objects binning in energy
        self.Parray=[]
        for i in range(0,len(energy_edges)-1):
            di=dataframe[(dataframe['reco_energy']>energy_edges[i]) & (dataframe['reco_energy']<energy_edges[i+1])]
            self.Parray.append(PulsarPhases(dataframe=di,nbins=nbins,tobservation=self.tobs,peak_limits_1=peak_limits_1,peak_limits_2=peak_limits_2,off_limits=off_limits,binned=binned,model=model))
          

    def calculate_tobs(self):
        return(get_effective_time(self.info)[1].value/3600)
    
    
    def PeaksVsEnergy(self):
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        
        #First figure:significance vs Energy
        y1=[]
        y2=[]
        y3=[]
        
        for i in range(0,len(self.energy_centres)):
            y1.append(self.Parray[i].regions.P1.sign)
            y2.append(self.Parray[i].regions.P2.sign)
            y3.append(self.Parray[i].regions.P1P2.sign)
            
        ax1.plot(self.energy_centres,y1,'o-',color='tab:orange')
        ax1.plot(self.energy_centres,y2,'o-',color='tab:green')
        ax1.plot(self.energy_centres,y3,'o-',color='tab:red')
            
        ax1.set_ylabel('Sign(sigmas)')
        ax1.set_xlabel('Enegy(TeV)')
        
        ax1.legend(['P1','P2','P1+P2'])
        ax1.grid()
        
        
        #Second figure: P1/P2 vs energy
        y4=[]
        
        for i in range(0,len(self.energy_centres)):
            y4.append(self.Parray[i].regions.P1P2_ratio)
            
        ax2.plot(self.energy_centres,y4,'o-',color='tab:blue')
                                                
        ax2.set_ylabel('P1/P2')
        ax2.set_xlabel('Enegy(TeV)')
        ax2.grid()
        
        
        # Third figure: FWHM vs Energy
        y5=[]
        y6=[]
        
        if self.Parray[0].fitting.model=='asym_dgaussian':
            prefactor=2.35482  
            for i in range(0,len(self.energy_centres)):
                y5.append(prefactor*self.Parray[i].fitting.params[1]/2+prefactor*self.Parray[i].fitting.params[2]/2)
                y6.append(prefactor*self.Parray[i].fitting.params[4]/2+prefactor*self.Parray[i].fitting.params[5]/2)
        
        else:
            if self.Parray[0].fitting.model=='dgaussian':
                prefactor=2.35482      

            elif self.Parray[0].fitting.model=='lorentzian':
                prefactor=2

            for i in range(0,len(self.energy_centres)):
                y5.append(prefactor*self.Parray[i].fitting.params[1])
                y6.append(prefactor*self.Parray[i].fitting.params[3])

        ax3.plot(self.energy_centres,y5,'o-',color='tab:orange')
        ax3.plot(self.energy_centres,y6,'o-',color='tab:green')
                                                
        ax3.set_ylabel('FWHM')
        ax3.set_xlabel('Enegy(TeV)')
        ax3.legend(['P1','P2'])
        ax3.grid()
        
        plt.show()
        
        
        
        
class FermiPulsarPhasesArray(PulsarPhasesArray):
    
    '''
    MAIN CLASS FOR THE PULSAR ANALYSIS.
    A class to store the pulsar phases and mjd_times to be used in the Pulsar analysis. This class allows to develop all the timing pular analysis using different supporting classes and subclasses.
    
    Parameters
    ----------
    dataframe : dataframe containing info
        DL2 LST file after the quality selection. Set daraframe to False if it is not available and want to set the attributes manually.
    energy_edges: List of float
	Edges of the energy binning (in TeV)
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
    energy_edges: list of float
        List of edges for the energy binning
    energy_centres: list of float
        List of center of the energy bins
    phases : list of float
        List of pulsar phases.
    times : list of float
        List of mjd times
    energies; list of float
        List of energies
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
    
    def __init__(self, fits_table, energy_edges,nbins=50,peak_limits_1=[0.983,0.026],peak_limits_2=[0.377,0.422],off_limits=[0.52,0.87],binned=True,model='dgaussian'):
            
            #Store global information (before binning)
            self.energy_edges=np.array(energy_edges)
            self.energy_centres=(self.energy_edges[1:]+self.energy_edges[:-1])/2
            
            self.create_df_from_info(fits_table)
            self.phases=np.array(self.info['pulsar_phase'].to_list())
            self.mjd_times=np.array(self.info['mjd_time'].to_list())            
        
            self.tobs=self.calculate_tobs()
            
            #Create array of PulsarPhases objects binning in energy
            self.Parray=[]
            for i in range(0,len(energy_edges)-1):
                self.Parray.append(FermiPulsarPhases(fits_table=fits_table[(self.info['energy']>energy_edges[i]) & (self.info['energy']<energy_edges[i+1])],nbins=nbins,tobservation=self.tobs,peak_limits_1=peak_limits_1,peak_limits_2=peak_limits_2,off_limits=off_limits,binned=binned,model=model))

    
    def create_df_from_info(self,fits_table):
            time=fits_table['MJD_BARYCENT_TIME'].byteswap().newbyteorder()
            phases=fits_table['PULSE_PHASE'].byteswap().newbyteorder()
            energy=fits_table['ENERGY'].byteswap().newbyteorder()
            dataframe = pd.DataFrame({"mjd_time":time,"pulsar_phase":phases,"dragon_time":time*3600*24,"energy":energy})
            self.info=dataframe
            return(self.info)
                 
            
    def calculate_tobs(self):
            diff=abs(self.mjd_times[1:]-self.mjd_times[0:-1])
            diff[diff>10/24]=0
            return(sum(diff)*24)
        
     