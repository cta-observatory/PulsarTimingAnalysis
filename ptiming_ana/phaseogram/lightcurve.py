import math
import astropy as ast
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import warnings
from scipy.stats import chi2,chisquare, norm
from .phasebinning import PhaseBinning
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
    
    def __init__(self, pulsar_phases,binning):
        self.lc=np.histogram(pulsar_phases.phases, bins=binning.edges) #Create the histogram
    
    
   ##############################################
                       #EXECUTION
   ############################################# 
    
    
    def chi_sqr_pulsar_test(self):
        
        #Calculate chi_sqr test
        mean_signal=np.mean(self.lc[0])
        chi_sqr,p_value=chisquare(self.lc[0],mean_signal)
        sigmas=norm.isf(p_value*10**(-7), loc=0, scale=1)
        
        return(chi_sqr,p_value,sigmas)
    
    
    
   ##############################################
                       #RESULTS
   ############################################# 

    def draw_stats(self,pulsar_phases,phase_limits,stats='short'): 
        text_towrite=''
        count=0
        for key, value in pulsar_phases.regions.dic.items():
            if value is not None:
                if count==0:
                    text_towrite=text_towrite+key+f': Sig(Li&Ma):{value.sign:.2f}$\sigma$'
                    count+=1
                else:
                    text_towrite=text_towrite+'\n'+key+f': Sig(Li&Ma):{value.sign:.2f}$\sigma$'
                
        
        if stats=='long':
            plt.annotate(text_towrite +f'\n $\sigma$(P1+P2)/sqrt(T) ={pulsar_phases.regions.P1P2.sign/np.sqrt(pulsar_phases.tobs):.2f}'+r'h$^{-1/2}$', xy=(0.29, 1.4), xytext=(0.29,1.4), fontsize=13,xycoords='axes fraction', textcoords='offset points', color='black',bbox=dict(facecolor='white', edgecolor='black'),horizontalalignment='left', verticalalignment='top')
            
            plt.annotate(f'$\chi^{2}$-test: $\chi^{2}$={pulsar_phases.stats.chisqr_res[0]:.2f} p_value={"{:.2e}".format(pulsar_phases.stats.chisqr_res[1])} sign={pulsar_phases.stats.chisqr_res[2]:.2f}$\sigma$ \n H-test: H={pulsar_phases.stats.Htest_res[0]:.2f} p_value={"{:.2e}".format(pulsar_phases.stats.Htest_res[1])} sign={pulsar_phases.stats.Htest_res[2]:.2f}$\sigma$ \n Z$_{{10}}$-test: Z$_{{10}}$={pulsar_phases.stats.Zntest_res[0]:.2f} p_value={"{:.2e}".format(pulsar_phases.stats.Zntest_res[1])} sign={pulsar_phases.stats.Zntest_res[2]:.2f}$\sigma$ ', xy=(0.54, 1.3), xytext=(0.54,1.3), fontsize=15,xycoords='axes fraction', textcoords='offset points', color='black',bbox=dict(facecolor='white', edgecolor='black'),horizontalalignment='left', verticalalignment='top')
            
        elif stats=='short':
            plt.annotate(text_towrite +'\n'+f'$\chi^{2}$-test={pulsar_phases.stats.chisqr_res[2]:.2f}$\sigma$ \n H-test= {pulsar_phases.stats.Htest_res[2]:.2f}$\sigma$ \n Z$_{{10}}$-test={pulsar_phases.stats.Zntest_res[2]:.2f}$\sigma$ ', xy=(0.72, 1.3), xytext=(0.72,1.3), fontsize=15,xycoords='axes fraction', textcoords='offset points', color='black',bbox=dict(facecolor='white', edgecolor='black'),horizontalalignment='left', verticalalignment='top')
    
    
    
    def draw_peakregion(self,pulsar_phases,name,color):
        
        #Draw first peak region
        if pulsar_phases.regions.dic[name] is not None:
            plt.fill_between(np.linspace(pulsar_phases.regions.dic[name].limits[0],pulsar_phases.regions.dic[name].limits[1],150), 0, 1600500,facecolor=color,color=color,alpha=0.2,label=name,zorder=4)
            plt.fill_between(np.linspace(pulsar_phases.regions.dic[name].limits[0]+1,pulsar_phases.regions.dic[name].limits[1]+1,150), 0, 1600500,facecolor=color,color=color,alpha=0.2,zorder=4)
            
            
            if len(pulsar_phases.regions.dic[name].limits)>2:
                plt.fill_between(np.linspace(pulsar_phases.regions.dic[name].limits[2],pulsar_phases.regions.dic[name].limits[3],150), 0, 1600500,facecolor=color,color=color,alpha=0.2,zorder=4)
                plt.fill_between(np.linspace(pulsar_phases.regions.dic[name].limits[2]+1,pulsar_phases.regions.dic[name].limits[3]+1,150), 0, 1600500,facecolor=color,color=color,alpha=0.2,zorder=4)
            
 
    def draw_background(self,pulsar_phases,color,hline=True):
        
        #Draw the background region
        plt.fill_between(np.linspace(pulsar_phases.regions.OFF.limits[0],pulsar_phases.regions.OFF.limits[1],150), 0,1600500,facecolor='white',color=color,alpha=0.2, hatch='/',label='OFF')
        
        plt.fill_between(np.linspace(pulsar_phases.regions.OFF.limits[0]+1,pulsar_phases.regions.OFF.limits[1]+1,150), 0,1600500,facecolor='white',color=color,alpha=0.2, hatch='/')
        
        #Add hline for background level reference (default True)
        if hline==True:
            plt.hlines(y=np.mean((self.lc[0][(self.lc[1][:-1]>(pulsar_phases.regions.OFF.limits[0])) & (self.lc[1][1:]<(pulsar_phases.regions.OFF.limits[1]))])),xmin=0,xmax=2,linestyle='dashed',color=color,zorder=5) 
    
    
    
    def draw_fitting(self,pulsar_phases,color,density=False,label=None):

        if pulsar_phases.fitting.shift==0:
            x=np.linspace(0,1,100000)
        else:
            x=np.linspace(2*pulsar_phases.fitting.shift,1+2*pulsar_phases.fitting.shift,100000)
        
        #Calculate distribution y-points using a model and its fitted params
        hoff=np.mean(self.lc[0])
        if pulsar_phases.fitting.model=='dgaussian':
            y=hoff*double_gaussian(x, *pulsar_phases.fitting.params[0:7])
            
        elif pulsar_phases.fitting.model=='asym_dgaussian':
            assymetric_gaussian_pdf_vec=np.vectorize(assymetric_double_gaussian)
            y=hoff*assymetric_gaussian_pdf_vec(x, *pulsar_phases.fitting.params)
            
        elif pulsar_phases.fitting.model=='lorentzian':
            y=hoff*double_lorentz(x, *pulsar_phases.fitting.params)
         
        elif pulsar_phases.fitting.model=='gaussian':
            y=hoff*gaussian(x, *pulsar_phases.fitting.params)
            
        if density==True:
            width=1/len(self.lc[0])
            weight=np.ones(len(y))/(np.sum(self.lc[0])*width)
            y=y*weight
                
        #Plot
        if label is not None:
            plt.plot(x,y,color=color,label=label)
        else:
            plt.plot(x,y,color=color,label='Fit')
            
        plt.plot(x+1,y,color=color)
        plt.plot(x-1,y,color=color)
        
        #Add labels
        plt.xlabel('Pulsar phase',fontsize=10)
        
  
    
    def draw_density_hist(self,phase_limits,colorhist,label=None,fill=True):
        
        width=1/len(self.lc[0])
        height_norm=self.lc[0]*np.ones(len(self.lc[0]))/(np.sum(self.lc[0])*width)
        
        plt.bar((self.lc[1][1:]+self.lc[1][:-1])/2,height_norm,width=width,color=colorhist,alpha=0.7,label=label,fill=fill,edgecolor = colorhist,linewidth=2)
        plt.bar((self.lc[1][1:]+self.lc[1][:-1])/2+np.ones(len(self.lc[1][:-1])),height_norm,width=width,color=colorhist,alpha=0.7,fill=fill,edgecolor = colorhist,linewidth=2)
        
        #Add labels
        plt.xlabel('Pulsar phase',fontsize=10)
        plt.ylabel('Events',fontsize=10)
    
        #Set limits in axis
        plt.ylim(max(min(height_norm)-3*np.sqrt(min(height_norm)),0),max(height_norm)+np.sqrt(max(height_norm)))
        plt.xlim(phase_limits[0],phase_limits[1])

    
    
    def draw_histogram(self,phase_limits,colorhist):
        
        plt.bar((self.lc[1][1:]+self.lc[1][:-1])/2,self.lc[0],width=1/len(self.lc[0]),color=colorhist,alpha=1,zorder=0)
        plt.bar((self.lc[1][1:]+self.lc[1][:-1])/2+np.ones(len(self.lc[1][:-1])),self.lc[0],width=1/len(self.lc[0]),color=colorhist,alpha=1,zorder=0)
        
        #Plot errorbars
        plt.errorbar((self.lc[1][1:]+self.lc[1][:-1])/2,self.lc[0],yerr=np.sqrt(self.lc[0]),color='k',fmt='.',zorder=1)
        plt.errorbar((self.lc[1][1:]+self.lc[1][:-1])/2+np.ones(len(self.lc[1][:-1])),self.lc[0],yerr=np.sqrt(self.lc[0]),color='k',fmt='.',zorder=1)
        
        list_step=list(self.lc[0])
        list_step.insert(0,self.lc[0][0])
        
        plt.step((self.lc[1]),list_step,'k',linestyle='-',linewidth=1,zorder=1)
        plt.step((self.lc[1])+np.ones(len(self.lc[1])),list_step,'k',linestyle='-',linewidth=1,zorder=1)
    
        #Add labels
        plt.xlabel('Pulsar phase')
        plt.ylabel('Events')
    
        #Set limits in axis
        plt.ylim(max(min(self.lc[0])-3*np.sqrt(min(self.lc[0])),0),max(self.lc[0])+2*np.sqrt(max(self.lc[0])))
        plt.xlim(phase_limits[0],phase_limits[1])
        

    
    def show_phaseogram(self,pulsar_phases,phase_limits=[0,1],stats='short',background=True,signal=['P1','P2','P3'],colorhist='blue',colorb='grey',colorP=['orange','green','purple'],colorfit='red',fit=False,hline=True,stats_label=True,time_label=True,add_legend=True):
        
        #Draw the histogram
        self.draw_histogram(phase_limits,colorhist)
     
        #Plot statistics (default True)
        if stats_label:
            self.draw_stats(pulsar_phases,phase_limits,stats)
        
        #Plot regions (default True)
        if background==True:
            self.draw_background(pulsar_phases,colorb,hline)
        
        for i in range(0,len(signal)):
            self.draw_peakregion(pulsar_phases,signal[i],color=colorP[i])
            
        # Plot fitted distribution (default True)
        if fit==True:
            try:
                self.draw_fitting(pulsar_phases,color=colorfit)
            except:
                print('No good fit available')
            
        #Add Tobs label
        if time_label:
            plt.annotate(f'Tobs={pulsar_phases.tobs:.1f} h'+'\n'+f'Entries={len(pulsar_phases.phases)}', xy=(0.11, 1.3), xytext=(0.11,1.3), fontsize=15, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='left', verticalalignment='top')
            
        #Add legend
        if add_legend:
            plt.legend(loc=2,bbox_to_anchor=(-0.01, 1.37),fontsize=15)
    

