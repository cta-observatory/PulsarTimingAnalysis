import math
import astropy as ast
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import warnings
from scipy.stats import chi2,chisquare, norm
from .Phasebinning import PhaseBinning
from .models import *


class Lightcurve():

    '''
    A class to construct the lightcurve from the information provided by the rest of classes.
    
    Parameters
    ----------
    pulsar_phases: PulsarPhase object
        Object from which we extract the pulsar phases to construct the lightcurve
    nbins : int, array
        Number of bins used in the lightcurve if int
        Bin edges used in the lightcurve if array
    
    Attributes
    ----------
    binning : PhaseBinning object
        Binning information
    stats : PeriodicityTest object
        Information of the statistical tests.
    lc: numpy histogram object
        Information of the pular phase histogram
    '''
    
    def __init__(self, pulsar_phases,nbins=50):
        self.binning=PhaseBinning(nbins) #Create the PhaseBinning object
        self.create_histogram(pulsar_phases) #Create the histogram
     
    
    def create_histogram(self,pulsar_phases):
        self.lc=np.histogram(pulsar_phases.phases, bins=self.binning.edges)

        
    def chi_sqr_pulsar_test(self):
        
        #Calculate chi_sqr test
        mean_signal=np.mean(self.lc[0])
        chi_sqr,p_value=chisquare(self.lc[0],mean_signal)
        sigmas=norm.isf(p_value*10**(-7), loc=0, scale=1)
        
        return(chi_sqr,p_value,sigmas)
    
    def draw_periodstats(self,pulsar_phases,phase_limits):
        plt.annotate(f'$\chi^{2}$-test: $\chi^{2}$={pulsar_phases.stats.chisqr_res[0]:.2f} p_value={"{:.2e}".format(pulsar_phases.stats.chisqr_res[1])} sign={pulsar_phases.stats.chisqr_res[2]:.2f}$\sigma$ \n H-test: H={pulsar_phases.stats.Htest_res[0]:.2f} p_value={"{:.2e}".format(pulsar_phases.stats.Htest_res[1])} sign={pulsar_phases.stats.Htest_res[2]:.2f}$\sigma$ \n Z$_{{10}}$-test: Z$_{{10}}$={pulsar_phases.stats.Zntest_res[0]:.2f} p_value={"{:.2e}".format(pulsar_phases.stats.Zntest_res[1])} sign={pulsar_phases.stats.Zntest_res[2]:.2f}$\sigma$ ', xy=(0.52, 1.3), xytext=(0.52,1.3), fontsize=15,xycoords='axes fraction', textcoords='offset points', color='tab:brown',bbox=dict(facecolor='white', edgecolor='tab:brown'),horizontalalignment='left', verticalalignment='top')
    
    def draw_peakstats(self,pulsar_phases,phase_limits): 
        plt.annotate(f' P1+P2: Sig(Li&Ma):{pulsar_phases.regions.P1P2.sign:.2f}$\sigma$ \n P1: Sig(Li&Ma):{pulsar_phases.regions.P1.sign:.2f}$\sigma$ \n P2: Sig(Li&Ma):{pulsar_phases.regions.P2.sign:.2f}$\sigma$', xy=(0.25, 1.3), xytext=(0.25,1.3), fontsize=15,xycoords='axes fraction', textcoords='offset points', color='tab:red',bbox=dict(facecolor='white', edgecolor='tab:red'),horizontalalignment='left', verticalalignment='top')
    
    def draw_peakregion(self,pulsar_phases,number_peak,color):
        
        #Draw first peak region
        if number_peak==1:
            if pulsar_phases.regions.P1.limits[0]<pulsar_phases.regions.P1.limits[1]:
                plt.fill_between(np.linspace(pulsar_phases.regions.P1.limits[0],pulsar_phases.regions.P1.limits[1],150), 0, 1600500,facecolor=color,color=color,alpha=0.2,label='P1')
                plt.fill_between(np.linspace(pulsar_phases.regions.P1.limits[0]+1,pulsar_phases.regions.P1.limits[1]+1,150), 0, 1600500,facecolor=color,color=color,alpha=0.2)
            else:
                plt.fill_between(np.linspace(pulsar_phases.regions.P1.limits[0],pulsar_phases.regions.P1.limits[1]+1,150), 0, 1600500,facecolor=color,color=color,alpha=0.2,label='P1')
                plt.fill_between(np.linspace(pulsar_phases.regions.P1.limits[0]+1,pulsar_phases.regions.P1.limits[1]+2,150), 0, 1600500,facecolor=color,color=color,alpha=0.2)
                
         
        #Draw second peak region
        if number_peak==2:
            if pulsar_phases.regions.P2.limits[0]<pulsar_phases.regions.P2.limits[1]:
                plt.fill_between(np.linspace(pulsar_phases.regions.P2.limits[0],pulsar_phases.regions.P2.limits[1],150), 0,1600500,facecolor=color,color=color,alpha=0.2,label='P2')
                plt.fill_between(np.linspace(pulsar_phases.regions.P2.limits[0]+1,pulsar_phases.regions.P2.limits[1]+1,150), 0,1600500,facecolor=color,color=color,alpha=0.2)
            else:
                plt.fill_between(np.linspace(pulsar_phases.regions.P2.limits[0],pulsar_phases.regions.P2.limits[1]+1,150), 0,1600500,facecolor=color,color=color,alpha=0.2,label='P2')
                plt.fill_between(np.linspace(pulsar_phases.regions.P2.limits[0]+1,pulsar_phases.regions.P2.limits[1]+2,150), 0,1600500,facecolor=color,color=color,alpha=0.2)

        
        
    def draw_background(self,pulsar_phases,color,hline=True):
        
        #Draw the background region
        plt.fill_between(np.linspace(pulsar_phases.regions.OFF.limits[0],pulsar_phases.regions.OFF.limits[1],150), 0,1600500,facecolor=color,color=color,alpha=0.2,label='OFF')
        plt.fill_between(np.linspace(pulsar_phases.regions.OFF.limits[0]+1,pulsar_phases.regions.OFF.limits[1]+1,150), 0,1600500,facecolor=color,color=color,alpha=0.2)
        
        #Add hline for background level reference (default True)
        if hline==True:
            plt.hlines(y=np.mean((self.lc[0][(self.lc[1][:-1]>(pulsar_phases.regions.OFF.limits[0])) & (self.lc[1][1:]<(pulsar_phases.regions.OFF.limits[1]))])),xmin=0,xmax=2,linestyle='dashed',color=color) 
    
    
    
    def draw_fitting(self,pulsar_phases,color):
        if pulsar_phases.fitting.shift==0:
            x=np.linspace(0,1,100000)
        else:
            x=np.linspace(pulsar_phases.fitting.shift,1+pulsar_phases.fitting.shift,100000)
        
        #Calculate distribution y-points using a model and its fitted params
        hoff=np.mean(self.lc[0])
        if pulsar_phases.fitting.model=='dgaussian':
            y=hoff*double_gaussian(x, *pulsar_phases.fitting.params[0:7])
        elif pulsar_phases.fitting.model=='asym_dgaussian':
            y=hoff*assymetric_double_gaussian(x, *pulsar_phases.fitting.params)
        elif pulsar_phases.fitting.model=='lorentzian':
            y=hoff*double_lorentz(x, *pulsar_phases.fitting.params)
        
        #Plot
        plt.plot(x,y,color=color,label='Fitted distribution')
        plt.plot(x+1,y,color=color)
        plt.plot(x-1,y,color=color)
    
    
    def draw_histogram(self,phase_limits,colorhist):
        
        #Plot histogram from 0 to 1 and from 1 to 2 (2 periods)
        plt.figure(figsize=(15,5))
        plt.bar((self.lc[1][1:]+self.lc[1][:-1])/2,self.lc[0],width=1/self.binning.nbins,color=colorhist,alpha=0.5)
        plt.bar((self.lc[1][1:]+self.lc[1][:-1])/2+np.ones(len(self.lc[1][:-1])),self.lc[0],width=1/self.binning.nbins,color=colorhist,alpha=0.5)
        
        #Plot errorbars
        plt.errorbar((self.lc[1][1:]+self.lc[1][:-1])/2,self.lc[0],yerr=np.sqrt(self.lc[0]),color=colorhist,fmt='.')
        plt.errorbar((self.lc[1][1:]+self.lc[1][:-1])/2+np.ones(len(self.lc[1][:-1])),self.lc[0],yerr=np.sqrt(self.lc[0]),color=colorhist,fmt='.')
        
        #Add labels
        plt.xlabel('Pulsar phase')
        plt.ylabel('Events')
    
        #Set limits in axis
        plt.ylim(min(self.lc[0])-3*np.sqrt(min(self.lc[0])),max(self.lc[0])+2*np.sqrt(max(self.lc[0])))
        plt.xlim(phase_limits[0],phase_limits[1])
        
       
    
    def show_phaseogram(self,pulsar_phases,phase_limits=[0,1],peakstats=True,periodstats=True,background=True,signal='both',colorhist='blue',colorb='black',colorP1='orange',colorP2='green',colorfit='red',fit=True,hline=True):
        
        #Draw the histogram
        self.draw_histogram(phase_limits,colorhist)
     
        #Plot statistics (default True)
        if peakstats==True:
            self.draw_peakstats(pulsar_phases,phase_limits)
            
        if periodstats==True:
            self.draw_periodstats(pulsar_phases,phase_limits)
        
        #Plot regions (default True)
        if background==True:
            self.draw_background(pulsar_phases,colorb,hline)
            
        if signal=='both':
            self.draw_peakregion(pulsar_phases,number_peak=1,color=colorP1)
            self.draw_peakregion(pulsar_phases,number_peak=2,color=colorP2)
        elif signal=='P1':
            self.draw_peakregion(pulsar_phases,number_peak=1,color=colorP1)   
        elif signal=='P2':
            self.draw_peakregion(pulsar_phases,number_peak=2,color=colorP2)
            
        # Plot fitted distribution (default True)
        if fit==True:
            self.draw_fitting(pulsar_phases,color=colorfit)
            
        #Add Tobs label
        plt.annotate(f'Tobs={pulsar_phases.tobs:.1f} h', xy=(0.45, 0.1), xytext=(0.3,1.2), fontsize=15, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='left', verticalalignment='top')

        
        #Add legend
        plt.legend(loc=2,bbox_to_anchor=(-0.01, 1.37),fontsize=15)
          
        plt.show()
    
