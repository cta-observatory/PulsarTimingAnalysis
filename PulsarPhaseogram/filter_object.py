import pandas as pd
import math
import numpy as np


class FilterPulsarAna():
    
    def __init__(self,gammaness_cut=None,alpha_cut=None,theta2_cut=None,zd_cut=None):
        self.gammaness_cut=gammaness_cut
        self.alpha_cut=alpha_cut
        self.theta2_cut=theta2_cut
        self.zd_cut=zd_cut
        
        self.check_cuts()
        
            
    def apply_fixed_cut(self,pulsarana):
        dataframe=pulsarana.info
        
        if self.gammaness_cut is not None:
            if isinstance(self.gammaness_cut, float):
                dataframe=dataframe[dataframe['gammaness']>self.gammaness_cut]
        if self.alpha_cut is not None:
            if isinstance(self.alpha_cut, int) or isinstance(self.alpha_cut, float):
                dataframe=dataframe[dataframe['alpha']<self.alpha_cut]
        if self.theta2_cut is not None:
            if isinstance(self.theta2_cut, float) or isinstance(self.theta2_cut, int):
                dataframe=dataframe[dataframe['theta2']>self.theta2_cut]
        if self.zd_cut is not None:
            if isinstance(self.zd_cut, float) or isinstance(self.zd_cut, int):
                dataframe=dataframe[(90-dataframe['alt_tel']*180/3.1416)<self.zd_cut]
        
        pulsarana.phases=np.array(dataframe['pulsar_phase'].to_list())
        pulsarana.info=dataframe
    
    
    def check_cuts(self):
        if self.gammaness_cut is not None:
            if isinstance(self.gammaness_cut, (float,int)):
                if self.gammaness_cut>=1 or self.gammaness_cut<0:
                    raise ValueError('Gammaness cut no valid')
            elif isinstance(self.gammaness_cut, (list,np.ndarray)):
                for cut in self.gammaness_cut:
                    if cut>=1 or cut<0:
                        raise ValueError('Gammaness cut no valid')
        
        if self.alpha_cut is not None:
            if isinstance(self.alpha_cut, (float,int)):
                if self.alpha_cut<0:
                    raise ValueError('alpha cut no valid')
            elif isinstance(self.alpha_cut, (list,np.ndarray)):
                for cut in self.alpha_cut:
                    if cut<0:
                        raise ValueError('alpha cut no valid')
                        
        if self.theta2_cut is not None:
            if isinstance(self.theta2_cut, (float,int)):
                if self.theta2_cut<0:
                    raise ValueError('Theta2 cut no valid')
            elif isinstance(self.theta2_cut, (list,np.ndarray)):
                for cut in self.theta2_cut:
                    if cut<0:
                        raise ValueError('Theta2 cut no valid')
                        
        if self.zd_cut is not None:
            if isinstance(self.zd_cut, (float,int)):
                if self.zd>90 or self.zd_cut<0:
                    raise ValueError('Zenith angle cut no valid')
            elif isinstance(self.zd_cut, (list,np.ndarray)):
                for cut in self.zd_cut:
                    if cut>90 or cut<0:
                        raise ValueError('Zenith angle cut no valid')
        
     
    def use_fixed_cuts(self):
        for cut in [self.gammaness_cut,self.alpha_cut,self.theta2_cut,self.zd_cut]:
            if not isinstance(cut, float) and not isinstance(cut, int) and cut is not None:
                return(False)
        return(True)
                
          
            
            
            
        