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
import warnings




def function_sqrt(x,A):
    return(A*np.sqrt(x))

def function_lin(x,A):
    return(A*x)



class PulsarTimeAnalysis():
    
    def __init__(self,tint=3600):
        
        #Define the arrays to store information
        self.Tarray=[]
        self.HTime=[0]
        self.ZTime=[0]
        self.ChiTime=[0]
        self.P1sTime=[0]
        self.P1exTime=[0]
        self.P1exerror=[0]
        self.P2sTime=[0]
        self.P2exTime=[0]
        self.P2exerror=[0]
        self.P1P2sTime=[0]
        self.P1P2exTime=[0]
        self.P1P2exerror=[0]
        
        #Define the time interval
        self.tint=tint
        
            
        
    ##############################################
                       #EXECUTION
    #############################################
    
    
    
    def store_Tvalues(self,pulsar_phases):
        
        #Store periodicity tests stats
        self.HTime.append(pulsar_phases.stats.Htest_res[2])
        self.ZTime.append(pulsar_phases.stats.Zntest_res[2])
        self.ChiTime.append(pulsar_phases.stats.chisqr_res[2])
        
        #Store statistics of each existing signal region
        if pulsar_phases.regions.P1 is not None:
            self.P1sTime.append(pulsar_phases.regions.P1.sign)
            self.P1exTime.append(pulsar_phases.regions.P1.Nex)
            self.P1exerror.append(pulsar_phases.regions.P1.yerr)
        
        if pulsar_phases.regions.P2 is not None:
            self.P2sTime.append(pulsar_phases.regions.P2.sign)
            self.P2exTime.append(pulsar_phases.regions.P2.Nex)
            self.P2exerror.append(pulsar_phases.regions.P2.yerr)
        
        if pulsar_phases.regions.P1P2 is not None:
            self.P1P2sTime.append(pulsar_phases.regions.P1P2.sign)
            self.P1P2exTime.append(pulsar_phases.regions.P1P2.Nex)
            self.P1P2exerror.append(pulsar_phases.regions.P1P2.yerr)   
            
            
            
    def update_tinfo(self,pulsar_phases,dataframe):
        #Update info in the main object
        pulsar_phases.phases=np.array(dataframe.pulsar_phase.to_list())
        pulsar_phases.times=np.array(dataframe.dragon_time.to_list())
        pulsar_phases.info=dataframe

        try:
            pulsar_phases.mjd_times=np.array(dataframe.mjd_time.to_list())
        except: 
            pass
        
        pulsar_phases.tobs=self.t[-1]/3600
        
        #Calculate stats 
        pulsar_phases.update_info()
        
        #Store the results
        self.store_Tvalues(pulsar_phases)
            
                    
    def run(self,pulsar_phases):
        dataframe=pulsar_phases.info
        diff=abs(dataframe.dragon_time.values[1:]-dataframe.dragon_time.values[:-1])
        s=0
        self.t=[0]

        #Estimate from each telescope the interval of time at which we can ignore the differences of time
        if pulsar_phases.telescope=='fermi':
            diff_del=3600*5
        else:
            diff_del=3600
        
        #Update the information 
        for i in range(0,len(diff)):
            if diff[i]<diff_del:
                s=s+diff[i]
                
                #Update information when tobs>=tint
                if s>self.tint:
                    self.t.append(self.t[-1]+s)
                    pulse_df_partial=dataframe[dataframe['dragon_time']<dataframe.dragon_time.values[i+1]]
                    self.update_tinfo(pulsar_phases,pulse_df_partial)
                    s=0
                
                #Update last interval
                elif i==(len(diff)-1):
                    self.t.append(sum(diff))
                    self.update_tinfo(pulsar_phases,dataframe)  

            else:
                diff[i]=0

    
    
    ##############################################
                       #RESULTS
    ############################################# 

    
    def show_results(self):
        fig1=self.PsigVsTime()
        fig2=self.PexVsTime() 
        fig3=self.StatsVsTime()

        return fig1,fig2,fig3
   

    def compare_Peaksig(self):
        
        #Show all the signal regions vs Time if they exist
        
        fig=plt.figure()
        if len(self.P1P2sTime)>1:
            plt.plot(np.array(self.t)/3600,self.P1P2sTime,'o-',color='tab:red',label='P1+P2')
            
        if len(self.P1sTime)>1:
            plt.plot(np.array(self.t)/3600,self.P1sTime,'o-',color='tab:orange',label='P1')
            
        if len(self.P2sTime)>1:
            plt.plot(np.array(self.t)/3600,self.P2sTime,'o-',color='tab:green',label='P2')
               
            
        plt.xlabel('Time of observation (h)')
        plt.ylabel('Significance ($\sigma$)')
        plt.grid()  
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return(fig)


    def PsigVsTime(self):
        xi=np.linspace(self.t[0]/3600,self.t[-1]/3600,1000)
        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        #####################################
        #First plot: P1Sig vs Time
        if len(self.P1sTime)>1:
            ax1.plot(np.array(self.t)/3600,self.P1sTime,'o-',color='tab:orange',label='P1')
            ax1.grid()
           
            #Fit to the points
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.P1sTime,p0=[0.1],maxfev=5000)
            ax1.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:orange')
            
            ax1.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        else:
            
            # Plot not available if some signal is not valid
            ax1.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        ax1.set_xlabel('Time of observation (h)')
        ax1.set_ylabel('Significance (sigma)')
        ax1.set_title('P1')
        
        #####################################
        
        #Second plot: P2Sig vs Time
        
        if len(self.P2sTime)>1:
            ax2.plot(np.array(self.t)/3600,self.P2sTime,'o-',color='tab:green',label='P2')
            ax2.grid()
            
            #Fit to the points
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.P2sTime,p0=[0.1],maxfev=5000)
            ax2.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:green')
            
            ax2.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        else:
            
            # Plot not available if some signal is not valid
            ax2.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
         
        ax2.set_xlabel('Time of observation (h)')
        ax2.set_ylabel('Significance (sigma)')
        ax2.set_title('P2')
            
        #####################################
        
        #First plot: P1+P2 Sig vs Time
        
        if len(self.P1P2sTime)>1:
            ax3.plot(np.array(self.t)/3600,self.P1P2sTime,'o-',color='tab:red',label='P1+P2')
            ax3.grid()            

            #Fit to the points
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.P1P2sTime,p0=[0.1],maxfev=5000)
            ax3.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:red')
            
            ax3.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        else:
            
            # Plot not available if some signal is not valid
            ax3.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
        ax3.set_xlabel('Time of observation (h)')
        ax3.set_ylabel('Significance (sigma)')
        ax3.set_title('P1+P2')
            
        plt.tight_layout()
        plt.show()
        
        return(fig)
       
    
    
    
    
    def PexVsTime(self):
        xi=np.linspace(self.t[0]/3600,self.t[-1]/3600,1000)
        
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        #####################################
        
        #First plot: P1Ex vs Time
        
        if len(self.P1exTime)>1:
            
            ax1.errorbar(np.array(self.t)/3600,self.P1exTime,yerr=self.P1exerror,fmt='o-',color='tab:orange',label='P1')
            ax1.grid()
            
            #Fit to the points
            ppot,pcov=curve_fit(function_lin,np.array(self.t)/3600,self.P1exTime,p0=[0.1],maxfev=5000)
            ax1.plot(xi,function_lin(xi,*ppot),'--',color='tab:orange')
            
            ax1.annotate(f'Nex=At'+ f' \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})'+r'h$^{-1}$',xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        else:
            
            # Plot not available if some signal is not valid
            ax1.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        ax1.set_xlabel('Time of observation (h)')
        ax1.set_ylabel('Number of excess events')
        ax1.set_title('P1')
        #####################################
        
        #Second plot: P2Ex vs time.
        
        if len(self.P2exTime)>1:
            ax2.errorbar(np.array(self.t)/3600,self.P2exTime,yerr=self.P2exerror,fmt='o-',color='tab:green',label='P2')
            ax2.grid()
           
            #Fit to the points
            ppot,pcov=curve_fit(function_lin,np.array(self.t)/3600,self.P2exTime,p0=[0.1],maxfev=5000)
            ax2.plot(xi,function_lin(xi,*ppot),'--',color='tab:green')
            
            ax2.annotate(f'Nex=At'+ f' \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})'+r'h$^{-1}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
        else:
            
            # Plot not available if some signal is not valid
            ax2.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        ax2.set_xlabel('Time of observation (h)')
        ax2.set_ylabel('Number of excess events')
        ax2.set_title('P2')
        
        #####################################
        
        #Thirs plot: P1+P2 Ex vs time.
        
        if len(self.P1P2exTime)>1:
            ax3.errorbar(np.array(self.t)/3600,self.P1P2exTime,yerr=self.P1P2exerror,fmt='o-',color='tab:red',label='P1+P2')
            ax3.grid()
            
            #Fit to the points
            ppot,pcov=curve_fit(function_lin,np.array(self.t)/3600,self.P1P2exTime,p0=[0.1],maxfev=5000)
            ax3.plot(xi,function_lin(xi,*ppot),'--',color='tab:red')
            
            ax3.annotate(f'Nex=At'+ f' \n A=({ppot[0]:.2f}$\pm$ {np.sqrt(pcov[0][0]):.2f})'+r'h$^{-1}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
            
        else:
            
            # Plot not available if some signal is not valid
            ax3.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        ax3.set_xlabel('Time of observation (h)')
        ax3.set_ylabel('Number of excess events')
        ax3.set_title('P1+P2')
        
        plt.tight_layout()
        plt.show()
      
        return(fig)
    
    
    def StatsVsTime(self):

        xi=np.linspace(self.t[0]/3600,self.t[-1]/3600,1000)
        
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(12)

        #####################################
        
        #First plot: H-test vs time.

        if self.HTime is not None and isinstance(self.HTime[-1],float) and self.HTime[-1]!=float('inf'):
            ax1.plot(np.array(self.t)/3600,self.HTime,'o-',color='tab:pink')
            ax1.grid()
            
            #Fit to the points
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.HTime,p0=[0.1],maxfev=5000)
            ax1.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:pink')
            
            ax1.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
         
        else:
            
            # Plot not available if some signal is not valid
            ax1.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')

        #Set labels
        ax1.set_xlabel('Time of observation (h)')
        ax1.set_ylabel('Significance (sigma)')
        ax1.set_title('H test')
            
        #####################################
        
        #Second plot: Zn-test vs time.
        
        if self.ZTime is not None and isinstance(self.ZTime[-1],float) and self.ZTime[-1]!=float('inf') :
            ax2.plot(np.array(self.t)/3600,self.ZTime,'o-',color='tab:cyan')
            ax2.grid()
            
            #Fit to the points
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.ZTime,p0=[0.1],maxfev=5000)
            ax2.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:cyan')
            
            ax2.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')

        else:
            
            # Plot not available if some signal is not valid
            ax2.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        #Set labels
        ax2.set_xlabel('Time of observation (h)')
        ax2.set_ylabel('Significance (sigma)')
        ax2.set_title('Z test')
            
        #####################################
        
        #Third plot: CHi2 test vs time
        
        if self.ChiTime is not None and isinstance(self.ChiTime[-1],float) and self.ChiTime[-1]!=float('inf'):
            ax3.plot(np.array(self.t)/3600,self.ChiTime,'o-',color='tab:olive')
            ax3.set_xlabel('Time of observation (h)')
            ax3.set_ylabel('Significance (sigma)')
            ax3.set_title('Chi square test')
            ax3.grid()
        
            ppot,pcov=curve_fit(function_sqrt,np.array(self.t)/3600,self.ChiTime,p0=[0.1],maxfev=5000)
            ax3.plot(xi,function_sqrt(xi,*ppot),'--',color='tab:olive')
            
            ax3.annotate(f'Sig=A'+ r'$\sqrt{t}$'+ f' \n A=({ppot[0]:.3f}$\pm$ {np.sqrt(pcov[0][0]):.3f})'+r'h$^{-1/2}$', xy=(0.95,0.35), xytext=(0.95,0.35), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white', edgecolor='k',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        else:
            
            # Plot not available if some signal is not valid
            ax3.annotate('Plot not available', xy=(0.8,0.6), xytext=(0.8,0.6), fontsize=10, xycoords='axes fraction', textcoords='offset points', color='k',bbox=dict(facecolor='white',alpha=0.8),horizontalalignment='right', verticalalignment='top')
        
        #Set labels
        ax3.set_xlabel('Time of observation (h)')
        ax3.set_ylabel('Significance (sigma)')
        ax3.set_title('Chi square test')
        plt.tight_layout()
        plt.show()
        
        return(fig)

    
    