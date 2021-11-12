import pandas as pd
import math
import astropy as ast
import numpy as np
import os
from astropy.time import Time
from astropy import units as u
import warnings
import matplotlib.pyplot as plt
from astropy.io import fits
import copy


__all__=['PEnergyAnalysis']

class PEnergyAnalysis():
    
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
    
    def __init__(self, energy_edges):
        self.energy_edges=np.array(energy_edges)
        self.energy_centres=(self.energy_edges[1:]+self.energy_edges[:-1])/2

    
   ##############################################
                       #EXECUTION
   ############################################# 

    def run(self,pulsarana):
        self.energy_units=pulsarana.energy_units
        self.tobs=pulsarana.tobs
            
        #Create array of PulsarPhases objects binning in energy
        self.Parray=[]
        for i in range(0,len(self.energy_edges)-1):
            dataframe=pulsarana.info
            di=dataframe[(dataframe['energy']>self.energy_edges[i]) & (dataframe['energy']<self.energy_edges[i+1])]
            
            print('Creating object in '+f'energy range ('+self.energy_units+f'):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}')
            self.Parray.append(copy.copy(pulsarana))
            self.Parray[i].phases=np.array(di['pulsar_phase'].to_list())
            self.Parray[i].info=di
            self.Parray[i].init_regions()
            
            if self.Parray[i].do_fit==True:
                self.Parray[i].setFittingParams(self.Parray[i].fit_model,self.Parray[i].binned)
            
            #Update the information every 1 hour and store final values
            print('Calculating statistics...')
            self.Parray[i].execute_stats(self.tobs)

    
    
    
    
   ##############################################
                       #RESULTS
   ############################################# 
    
    def show_Energy_lightcurve(self):
        fig_array=[]
        for i in range(0,len(self.Parray)):
            #Plot histogram from 0 to 1 and from 1 to 2 (2 periods)
            fig=plt.figure(figsize=(15,5))
            self.Parray[i].histogram.show_phaseogram(self.Parray[i],[0,2],fit=True)
            plt.annotate(f'ENERGY RANGE ('+self.energy_units+f'):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}', xy=(0.1, 0.9), xytext=(0.31,0.9), fontsize=15, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='left', verticalalignment='top')
            fig_array.append(fig)
            plt.show()
  
        return fig_array

   
    
    def show_joined_Energy_fits(self,colorh=['tab:red','tab:purple','tab:blue','tab:brown','tab:cyan','tab:olive','tab:pink']):
        
        for i in range(0,len(self.Parray)):
            if self.Parray[i].fitting.check_fit_result()==True:
                fig=plt.figure(figsize=(17,8))
                break
            elif i==len(self.Parray)-1 and self.Parray[i].fitting.check_fit_result()==False:
                print('No fit available for any energy bin')
                return     
        for i in range(0,len(self.Parray)):
            if self.Parray[i].fitting.check_fit_result()==True:
                
                self.Parray[i].histogram.draw_fitting(self.Parray[i],color=colorh[i],density=True,label=f'Energies('+self.energy_units+f'):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}')                
        plt.xlim(2*self.Parray[0].fitting.shift,1+2*self.Parray[i].fitting.shift)
        plt.legend(fontsize=20)
        return(fig)   

  
    def show_joined_Energy_lightcurve(self,colorh=['tab:red','tab:purple','tab:blue','tab:brown','tab:cyan','tab:olive','tab:pink'],colorP=['orange','green','purple'],ylimits=None):
        
        fig=plt.figure(figsize=(17,8))
        
        for i in range(0,len(self.Parray)):
            self.Parray[i].histogram.draw_density_hist([0.7,1.7],colorhist=colorh[i],label=f'Energies('+self.energy_units+f'):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}',fill=False)
            
        self.Parray[i].histogram.draw_background(self.Parray[i],'grey',hline=False)
        
        signal=['P1','P2','P3']
        for j in range(0,len(signal)):
            self.Parray[0].histogram.draw_peakregion(self.Parray[0],signal[j],color=colorP[j])
            
        plt.legend(fontsize=15)
        
        if ylimits is not None:
            plt.ylim(ylimits)
            
        return(fig)
        
        
        
    def show_EnergyPresults(self):
        peak_stat=[0]*(len(self.energy_edges)-1)
        p_stat=[0]*(len(self.energy_edges)-1)
        
        for i in range(0,len(self.energy_edges)-1):
            print(f'ENERGY RANGE ('+self.energy_units+f'):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}'+'\n')
            peak_stat[i],p_stat[i]=self.Parray[i].show_Presults()
            print('\n \n')
            print('-------------------------------------------------------------------')
   
        return peak_stat,p_stat


    def show_Energy_fitresults(self):
        fit_results=[0]*(len(self.energy_edges)-1)
        
        for i in range(0,len(self.energy_edges)-1):
            print(f'ENERGY RANGE ('+self.energy_units+f'):{self.energy_edges[i]:.2f}-{self.energy_edges[i+1]:.2f}'+'\n')
            if self.Parray[i].fitting.check_fit_result()==True:
                fit_results[i]=self.Parray[i].show_fit_results()
            else:
                print('No fit available for this energy range')
            print('\n \n')
            print('-------------------------------------------------------------------')
   
        return fit_results



    def PSigVsEnergy(self):
        P1_s=[]
        P2_s=[]
        P1P2_s=[]
        
        for i in range(0,len(self.energy_centres)):
            if self.Parray[i].regions.dic['P1'] is not None:
                P1_s.append(self.Parray[i].regions.P1.sign)
            if self.Parray[i].regions.dic['P2'] is not None:
                P2_s.append(self.Parray[i].regions.P2.sign)
            if self.Parray[i].regions.dic['P1+P2'] is not None:
                P1P2_s.append(self.Parray[i].regions.P1P2.sign)
                
        if len(P1P2_s)>0:
            plt.plot(self.energy_centres,P1P2_s,'o-',color='tab:red',label='P1+P2')
            
        if len(P1_s)>0:
            plt.plot(self.energy_centres,P1_s,'o-',color='tab:orange',label='P1')
            
        if len(P2_s)>0:
            plt.plot(self.energy_centres,P2_s,'o-',color='tab:green',label='P2')
                
        plt.ylabel('Significance($\sigma$)')
        plt.xlabel('Enegy ('+ str(self.energy_units)+')')
        plt.legend()
        plt.tight_layout()
        plt.grid(which='both')
        plt.xscale('log')

    
    
    def P1P2VsEnergy(self):
        P1P2E=[]
        P1P2E_error=[]
        
        if self.Parray[0].regions.P1P2_ratio is not None:
            for i in range(0,len(self.energy_centres)):
                P1P2E.append(self.Parray[i].regions.P1P2_ratio)
                P1P2E_error.append(self.Parray[i].regions.P1P2_ratio_error)

            plt.errorbar(self.energy_centres,P1P2E,yerr=P1P2E_error,fmt='o-',color='tab:blue')                                 
            plt.ylabel('P1/P2')
            plt.xlabel('Enegy ('+ str(self.energy_units)+')')
            plt.tight_layout()
            plt.xscale('log')
            plt.grid(which='both')

        else:
            print('Cannot calculate P1/P2 since one of the peaks is not defined')
     
        
    
    def FWHMVsEnergy(self):
        FP1=[]
        FP2=[]
        FP1_err=[]
        FP2_err=[]
        energies_F1=[]
        energies_F2=[]

        if self.Parray[0].fitting.model=='asym_dgaussian':
            prefactor=2.35482  
            for i in range(0,len(self.energy_centres)):
                try:
                    FP1.append(prefactor*self.Parray[i].fitting.params[1]/2+prefactor*self.Parray[i].fitting.params[2]/2)
                    energies_F1.append(self.energy_centres[i])
                    try:
                        FP1_err.append(FP1*np.sqrt((self.Parray[i].errors.params[1]/self.Parray[i].fitting.params[1])**2+(self.Parray[i].fitting.errors[2]/self.Parray[i].fitting.params[2])**2))
                    except:
                        FP1_err.append(0) 
                except:
                    pass
                
                try:
                    FP2.append(prefactor*self.Parray[i].fitting.params[4]/2+prefactor*self.Parray[i].fitting.params[5]/2)
                    energies_F2.append(self.energy_centres[i])
                    try:
                        FP2_err.append(FP2*np.sqrt((self.Parray[i].errors.params[4]/self.Parray[i].fitting.params[4])**2+(self.Parray[i].fitting.errors[5]/self.Parray[i].fitting.params[5])**2))
                    except:
                        FP2_err.append(0) 
                except:
                    pass
           
        else:
            if self.Parray[0].fitting.model=='dgaussian':
                prefactor=2.35482      

            elif self.Parray[0].fitting.model=='lorentzian':
                prefactor=2
                
            else:
                prefactor=0

            for i in range(0,len(self.energy_centres)):
                try:
                    FP1.append(prefactor*self.Parray[i].fitting.params[1])
                    energies_F1.append(self.energy_centres[i])
                    try:
                        FP1_err.append(prefactor*self.Parray[i].fitting.errors[1])
                    except:
                        FP1_err.append(0) 
                except:
                    pass
                
                try:
                    FP2.append(prefactor*self.Parray[i].fitting.params[3])
                    energies_F2.append(self.energy_centres[i])
                    try:
                        FP2_err.append(prefactor*self.Parray[i].fitting.errors[3])
                    except:
                        FP2_err.append(0) 
                except:
                    pass
      
        
        if len(FP1)>0:
            plt.errorbar(energies_F1,FP1,yerr=FP1_err,fmt='o-',color='tab:orange',label='P1')
        if len(FP2)>0:
            plt.errorbar(energies_F2,FP2,yerr=FP2_err,fmt='o-',color='tab:green',label='P2')                                        
        if len(FP1)<=0 and len(FP2)<=0:
            plt.annotate('Plot not available', xy=(0.6,0.6), xytext=(0.6,0.6), fontsize=15, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')  
        else:
            plt.ylabel('FWHM')
            plt.xlabel('Enegy ('+ str(self.energy_units)+')')
            plt.legend()
            plt.tight_layout()
            plt.grid(which='both')
            plt.xscale('log')

        
    def MeanVsEnergy(self):
        M1=[]
        M2=[]
        M1_err=[]
        M2_err=[]
        energies_M1=[]
        energies_M2=[]
        
        if self.Parray[0].fitting.model=='asym_dgaussian':
            for i in range(0,len(self.energy_centres)):
                try:
                    M1.append(self.Parray[i].fitting.params[0])
                    energies_M1.append(self.energy_centres[i])
                    try:
                        M1_err.append(self.Parray[i].fitting.errors[0])
                    except:
                        M1_err.append(0) 
                except:
                    pass
                
                try:
                    M2.append(self.Parray[i].fitting.params[3])
                    energies_M2.append(self.energy_centres[i])
                    try:
                        M1_err.append(self.Parray[i].fitting.errors[3])
                    except:
                        M1_err.append(0) 
                except:
                    pass

        elif self.Parray[0].fitting.model=='dgaussian' or self.Parray[0].fitting.model=='lorentzian':
            for i in range(0,len(self.energy_centres)):
                try:
                    M1.append(self.Parray[i].fitting.params[0])
                    energies_M1.append(self.energy_centres[i])
                    try:
                        M1_err.append(self.Parray[i].fitting.errors[0])
                    except:
                        M1_err.append(0) 
                except:
                    pass
                
                try:
                    M2.append(self.Parray[i].fitting.params[2])
                    energies_M2.append(self.energy_centres[i])
                    try:
                        M2_err.append(self.Parray[i].fitting.errors[2])
                    except:
                        M2_err.append(0) 
                except:
                    pass
           
        if len(M1)==0 and len(M2)==0:
            print('No fit available for plotting')
            return
        elif len(M1)>0 and len(M2)>0:
            nplots=2
        else:
            nplots=1
                 
        fig = plt.figure(figsize=(10,5))
        if len(M1)>0:   
            plt.subplot(nplots,1,1)
            plt.errorbar(energies_M1,M1,yerr=M1_err,fmt='o-',color='tab:orange')
            plt.ylabel('Mean phase')
            plt.xlabel('Enegy ('+ str(self.energy_units)+')')
            plt.title('P1 mean phase')
            plt.tight_layout()
            plt.grid(which='both')
            plt.xscale('log')
            
        if len(M2)>0:
            plt.subplot(nplots,1,nplots)
            plt.errorbar(energies_M2,M2,yerr=M2_err,fmt='o-',color='tab:green')
            plt.title('P2 mean phase')
            plt.ylabel('Mean phase')
            plt.xlabel('Enegy ('+ str(self.energy_units)+')')
            plt.tight_layout()
            plt.grid(which='both')
            plt.xscale('log')

        return(fig)
    
    def PeaksVsEnergy(self):
   
        fig = plt.figure(figsize=(15,4))
    
        plt.subplot(1,3, 1)
        self.PSigVsEnergy()

        plt.subplot(1,3, 2)
        self.P1P2VsEnergy()
        
        plt.subplot(1,3, 3)
        self.FWHMVsEnergy()
        
        plt.tight_layout()
        plt.show()
        
        
        
        
        
