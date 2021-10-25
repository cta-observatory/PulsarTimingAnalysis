import math
import astropy as ast
import numpy as np
import os
from astropy import units as u
import warnings
from .Pulsarpeak import PulsarPeak

class PhaseRegions():
    
    def __init__(self, lc_data, off_limits,peak_limits_1,peak_limits_2):
        
        #Define background and null regions for both peaks and 
        self.OFF=PulsarPeak(lc_data,self,off_limits,peaktype='background')
        self.P1=None
        self.P2=None
        self.P1P2=None
        self.P3=None
        
        #Create the peaks in the previosly defined null regions
        self.create_peaks(lc_data,self,peak_limits_1,peak_limits_2)
        
        if self.npeaks>=2:
            self.P1P2_ratio=self.P1.Nex/self.P2.Nex
        
    def create_peaks(self,lc_data,regions,peak_limits_1,peak_limits_2):
            if peak_limits_1 is not None:
                self.P1=PulsarPeak(lc_data,regions,peak_limits_1)
                npeaks=1
            else:
                npeaks=0
                print('No P1 limits. Cant create P1 object')
                
            if peak_limits_2 is not None:
                self.P2=PulsarPeak(lc_data,regions,peak_limits_2)
                npeaks=npeaks+1
            else:
                print('No P2 limits. Cant create P2 object')
            
            if npeaks==2:
                self.P1P2=PulsarPeak(lc_data,regions,peak_limits_1+peak_limits_2)
            
            if self.P3 is not None:
                self.npeaks=npeaks+1 
                self.P1P2P3=PulsarPeak(lc_data,regions,peak_limits_1+peak_limits_2+regions.P3.limits)
            else:
                self.npeaks=npeaks
                        
    def change_limits(self,lc_data,new_off_limits=None,new_peak_limits1=None,new_peak_limits2=None):
            if new_off_limits is not None:
                self.OFF=PulsarPeak(lc_data,self,new_off_limits,peaktype='background')
             
            if new_peak_limits1 is not None:
                newlimits1=new_peak_limits1
            else:
                newlimits1=self.P1.limits
                    
            if new_peak_limits2 is not None:
                newlimits2=new_peak_limits2
            else:
                newlimits2=self.P2.limits
                        
            self.create_peaks(lc_data,self,newlimits1,newlimits2)    
            if self.npeaks==2:
                self.P1P2=PulsarPeak(lc_data,self,self.P1.limits+self.P2.limits)
        
    def remove_peak(self,peak_number):
            if peak_number=='P1':
                self.P1=None
                self.P1P2=None
                self.P1P2P3=None
                self.npeaks=self.npeaks-1
                
            elif peak_number=='P2':
                self.P2=None
                self.P1P2=None
                self.P1P2P3=None
                self.npeaks=self.npeaks-1
                
            elif peak_number=='P3':
                self.P3=None
                self.P1P2P3=None
                self.npeaks=self.npeaks-1
        
    def third_peak(self,lc_data,limits3):
            if self.P3 is None:
                self.npeaks=self.npeaks+1
                
            self.P3=PulsarPeak(lc_data,self,limits3)
            self.P1P2P3=PulsarPeak(lc_data,self,self.P1.limits+self.P2.limits+limits3)
            

    def show_peak_results(self):
        if self.npeaks<3:
            peak_results={'P1+P2':self.P1P2.sign,'P1':self.P1.sign,'P2':self.P2.sign}
        elif self.npeaks==3:
            peak_results={'P1+P2':self.P1P2.sign,'P1':self.P1.sign,'P2':self.P2.sign,'P3':self.P3.sign,'P1+P2+P3':self.P1P2P3.sign}
            
        return(peak_results)

