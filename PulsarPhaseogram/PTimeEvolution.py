import pandas as pd
import math
import astropy as ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
import os
from astropy.time import Time
from decimal import *
from pylab import *
from astropy import units as u
from gammapy.stats import WStatCountsStatistic
import warnings
from lstchain.reco.utils import get_effective_time,add_delta_t_key
from scipy.stats import chi2,chisquare, norm
from iminuit import Minuit
from probfit import UnbinnedLH, gaussian, doublegaussian
import numba as nb

def function_sqrt(x,A):
    return(A*np.sqrt(x))

def function_lin(x,A):
    return(A*x)

class PulsarTimeEvolution():
    
    def __init__(self, pulsar_phases,nbins=50,peak_limits_1=[0.983,0.026],peak_limits_2=[0.377,0.422],off_limits=[0.52,0.87],tint=3600):
        
        self.Tarray=[]
        self.HTime=[0]
        self.ZTime=[0]
        self.ChiTime=[0]
        self.P1sTime=[0]
        self.P1exTime=[0]
        self.P2sTime=[0]
        self.P2exTime=[0]
        self.P1P2sTime=[0]
        self.P1P2exTime=[0]
        self.sigvsTime(pulsar_phases,nbins,peak_limits_1,peak_limits_2,off_limits,tint=tint)
        
        
    def show_results(self):
        self.PsigVsTime()
        self.PexVsTime()
        self.StatsVsTime()
    
    def compare_Peaksig(self):
        plt.figure()
        if self.P1sTime is not None:
            plt.plot(np.array(self.t)/3600,self.P1sTime,'o-',color='tab:orange',label='P1')
            
        if self.P2sTime is not None:
            plt.plot(np.array(self.t)/3600,self.P2sTime,'o-',color='tab:green',label='P2')
            
        if self.P1P2sTime is not None:
            plt.plot(np.array(self.t)/3600,self.P1P2sTime,'o-',color='tab:red',label='P1+P2')
            
            
        plt.xlabel('Time of observation (h)')
        plt.ylabel('Significance (sigma)')
        plt.grid()  
        plt.legend()
        plt.tight_layout()
        plt.show()
        
 
    def PsigVsTime(self):
        xi=np.linspace(self.t[0]/3600,self.t[-1]/3600,1000)
        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)
    
        if self.P1sTime is not None:
            ax1.plot(np.array(self.t)/3600,self.P1sTime,'o-',color='tab:orange',label='P1')
            ax1.set_xlabel('Time of observation (h)')
            ax1.set_ylabel('Significance (sigma)')
            ax1.set_title('P1')
            ax1.grid()
           
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.P1sTime,p0=[0.1],maxfev=5000)
            ax1.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:orange')
            
            ax1.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')

            
        if self.P2sTime is not None:
            ax2.plot(np.array(self.t)/3600,self.P2sTime,'o-',color='tab:green',label='P2')
            ax2.set_xlabel('Time of observation (h)')
            ax2.set_ylabel('Significance (sigma)')
            ax2.set_title('P2')
            ax2.grid()
            
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.P2sTime,p0=[0.1],maxfev=5000)
            ax2.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:green')
            
            ax2.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
            
            
        if self.P1P2sTime is not None:
            ax3.plot(np.array(self.t)/3600,self.P1P2sTime,'o-',color='tab:red',label='P1+P2')
            ax3.set_xlabel('Time of observation (h)')
            ax3.set_ylabel('Significance (sigma)')
            ax3.grid()
            ax3.set_title('P1+P2')

            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.P1P2sTime,p0=[0.1],maxfev=5000)
            ax3.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:red')
            
            ax3.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')

        plt.tight_layout()
        plt.show()
       
    
    def PexVsTime(self):
        xi=np.linspace(self.t[0]/3600,self.t[-1]/3600,1000)
        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        if self.P1sTime is not None:
            ax1.plot(np.array(self.t)/3600,self.P1exTime,'o-',color='tab:orange',label='P1')
            ax1.set_xlabel('Time of observation (h)')
            ax1.set_ylabel('Number of excess events')
            ax1.grid()
            ax1.set_title('P1')
            
            ppot,pcov=curve_fit(function_lin,np.array(self.t)/3600,self.P1exTime,p0=[0.1],maxfev=5000)
            ax1.plot(xi,function_lin(xi,*ppot),'--',color='tab:orange')
            
            ax1.annotate(f'Nex=At'+ f' \n A=({ppot[0]:.0f}$\pm$ {np.sqrt(pcov[0][0]):.0f})'+r'h$^{-1}$',xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
            
        if self.P2sTime is not None:
            ax2.plot(np.array(self.t)/3600,self.P2exTime,'o-',color='tab:green',label='P2')
            ax2.set_xlabel('Time of observation (h)')
            ax2.set_ylabel('Number of excess events')
            ax2.set_title('P2')
            ax2.grid()
           
            ppot,pcov=curve_fit(function_lin,np.array(self.t)/3600,self.P2exTime,p0=[0.1],maxfev=5000)
            ax2.plot(xi,function_lin(xi,*ppot),'--',color='tab:green')
            
            ax2.annotate(f'Nex=At'+ f' \n A=({ppot[0]:.0f}$\pm$ {np.sqrt(pcov[0][0]):.0f})'+r'h$^{-1}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
            
        if self.P1P2sTime is not None:
            ax3.plot(np.array(self.t)/3600,self.P1P2exTime,'o-',color='tab:red',label='P1+P2')
            ax3.set_xlabel('Time of observation (h)')
            ax3.set_ylabel('Number of excess events')
            ax3.grid()
            ax3.set_title('P1+P2')
            
            ppot,pcov=curve_fit(function_lin,np.array(self.t)/3600,self.P1P2exTime,p0=[0.1],maxfev=5000)
            ax3.plot(xi,function_lin(xi,*ppot),'--',color='tab:red')
            
            ax3.annotate(f'Nex=At'+ f' \n A=({ppot[0]:.0f}$\pm$ {np.sqrt(pcov[0][0]):.0f})'+r'h$^{-1}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
            
        plt.tight_layout()
        plt.show()
            
    
    def StatsVsTime(self):
        xi=np.linspace(self.t[0]/3600,self.t[-1]/3600,1000)
        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        if self.P1sTime is not None:
            ax1.plot(np.array(self.t)/3600,self.HTime,'o-',color='tab:pink')
            ax1.set_xlabel('Time of observation (h)')
            ax1.set_ylabel('Significance (sigma)')
            ax1.set_title('H test')
            ax1.grid()
            
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.HTime,p0=[0.1],maxfev=5000)
            ax1.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:pink')
            
            ax1.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
            
        if self.P2sTime is not None:
            ax2.plot(np.array(self.t)/3600,self.ZTime,'o-',color='tab:cyan')
            ax2.set_xlabel('Time of observation (h)')
            ax2.set_ylabel('Significance (sigma)')
            ax2.set_title('Z test')
            ax2.grid()
            
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.ZTime,p0=[0.1],maxfev=5000)
            ax2.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:cyan')
            
            ax2.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')

            
        if self.P1P2sTime is not None:
            ax3.plot(np.array(self.t)/3600,self.ChiTime,'o-',color='tab:olive')
            ax3.set_xlabel('Time of observation (h)')
            ax3.set_ylabel('Significance (sigma)')
            ax3.set_title('Chi square test')
            ax3.grid()
        
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.ChiTime,p0=[0.1],maxfev=5000)
            ax3.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:olive')
            
            ax3.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
        plt.tight_layout()
        plt.show()
    
    
    def store_Tvalues(self,pulsar_phases):
        self.HTime.append(pulsar_phases.stats.Htest_res[2])
        self.ZTime.append(pulsar_phases.stats.Zntest_res[2])
        self.ChiTime.append(pulsar_phases.stats.chisqr_res[2])
        self.P1sTime.append(pulsar_phases.regions.P1.sign)
        self.P1exTime.append(pulsar_phases.regions.P1.Nex)
        self.P2sTime.append(pulsar_phases.regions.P2.sign)
        self.P2exTime.append(pulsar_phases.regions.P2.Nex)
        self.P1P2sTime.append(pulsar_phases.regions.P1P2.sign)
        self.P1P2exTime.append(pulsar_phases.regions.P1P2.Nex)
                    
                    
    def sigvsTime(self,pulsar_phases,nbins,peak_limits_1,peak_limits_2,off_limits,tint=3600):
        dataframe=pulsar_phases.info
        diff=abs(dataframe.dragon_time.values[1:]-dataframe.dragon_time.values[:-1])
        s=0
        t=[0]

        for i in range(0,len(diff)):
            if diff[i]<36000:
                s=s+diff[i]
                if s>tint:
                    pulse_df_partial=dataframe[dataframe['dragon_time']<dataframe.dragon_time.values[i+1]]
                    pulsar_phases.info=pulse_df_partial
                    pulsar_phases.phases=np.array(pulse_df_partial.pulsar_phase.to_list())
                    pulsar_phases.times=np.array(pulse_df_partial.dragon_time.to_list())
                    try:
                        pulsar_phases.mjd_times=np.array(pulse_df_partial.mjd_time.to_list())
                    except: 
                        pass
                    
                    pulsar_phases.tobs=pulsar_phases.calculate_tobs() 
                    
                    t.append(t[-1]+s)
                    if pulsar_phases.tobs<=0:
                        pulsar_phases.tobs=t[-1]/3600
                        
                    pulsar_phases.update_info(nbins,peak_limits_1,peak_limits_2,off_limits)
                    self.store_Tvalues(pulsar_phases)
                    s=0

                elif i==(len(diff)-1):
                    pulsar_phases.phases=np.array(dataframe.pulsar_phase.to_list())
                    pulsar_phases.times=np.array(dataframe.dragon_time.to_list())
                    try:
                        pulsar_phases.mjd_times=np.array(dataframe.mjd_time.to_list())
                    except: 
                        pass
                    
                    pulsar_phases.tobs=pulsar_phases.calculate_tobs()
                    
                    t.append(sum(diff))
                    if pulsar_phases.tobs<=0:
                        pulsar_phases.tobs=t[-1]/3600
                        
                    pulsar_phases.update_info(nbins,peak_limits_1,peak_limits_2,off_limits)
                    self.store_Tvalues(pulsar_phases)
                     
            else:
                diff[i]=0

        self.t=t
