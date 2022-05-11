import math
import astropy as ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import warnings
from iminuit import Minuit
from probfit import UnbinnedLH, gaussian, doublegaussian
import numba as nb
import pandas as pd
from .models import *

class PeakFitting():

        def __init__(self,binned,model,peak='both'):
            
            #Define and check model
            self.model=model
            self.shift=0
            self.peak_tofit=peak
            self.check_model()
            self.binned=binned

       
    
    
                
        ##############################################
                           #EXECUTION
        #############################################
    
    
        def run(self,pulsar_phases):
            #Estimate initial values 
            self.est_initial_values(pulsar_phases)
            #Do the fitting 
            if self.binned==True:
                self.fit_Binned(pulsar_phases.histogram)
            else:
                self.fit_ULmodel(pulsar_phases)
         
            
        def check_model(self):
            model_list=get_model_list()
            if self.model not in model_list:
                raise ValueError('The model is not in the available model list')
        
            if self.peak_tofit=='both' and self.model=='gaussian':
                raise ValueError('Gaussian model can only fit one peak')
            
            if self.peak_tofit=='P1' and self.model=='dgaussian':
                raise ValueError('Dgaussian model needs two peaks')
            
            if self.peak_tofit=='P2' and self.model=='dgaussian':
                raise ValueError('Gaussian model needs two peaks')
        
                
                
        def est_initial_values(self,pulsar_phases):
                self.check_model()
                self.init=[]
                intensity=[]
                
                #Set different initial values for different models
                for name in ['P1','P2']:
                    
                    P_info=pulsar_phases.regions.dic[name]
                    if P_info is not None:
                        if name==self.peak_tofit or self.peak_tofit=='both':
                            intensity.append(P_info.Nex/P_info.noff)

                            if len(P_info.limits)>2:
                                self.init.extend([(P_info.limits[0]+1+P_info.limits[3])/2,P_info.deltaP/2])
                                self.shift=2*P_info.deltaP
                            else:
                                self.init.extend([(P_info.limits[0]+P_info.limits[1])/2,P_info.deltaP/2])

                            if self.model=='asym_dgaussian':
                                self.init.append(P_info.deltaP/2)
                    else:
                        if self.model!='gaussian':
                            raise ValueError('Double Gaussian model needs two peaks')
                
                self.init.append(1)      
                self.init.extend(intensity)

                
        #Unbinned fitting            
        def fit_ULmodel(self,pulsar_phases):  
            self.check_model()

            #Shift the phases if one of the peak is near the interval edge
            shift_phases=pulsar_phases.phases
            if self.shift!=0:
                for i in range(0,len(shift_phases)):
                    if shift_phases[i]<self.shift:
                        shift_phases[i]=shift_phases[i]+1

            if self.model=='dgaussian':
                unbinned_likelihood = UnbinnedLH(double_gaussian, np.array(shift_phases))
                minuit = Minuit(unbinned_likelihood,mu=self.init[0], sigma=self.init[1],mu_2=self.init[2],sigma_2=self.init[3],A=self.init[4],B=self.init[5],C=self.init[6])
                self.parnames=['mu', 'sigma','mu_2','sigma_2','A','B','C']
                
            elif self.model=='asym_dgaussian':
                unbinned_likelihood = UnbinnedLH(assymetric_double_gaussian, np.array(shift_phases))
                minuit = Minuit(unbinned_likelihood, mu=self.init[0], sigma1=self.init[1],sigma2=self.init[2],mu_2=self.init[3],sigma1_2=self.init[4],sigma2_2=self.init[5],A=self.init[6],B=self.init[7],C=self.init[8])
                self.parnames=['mu', 'sigma1','sigma2','mu_2','sigma1_2','sigma2_2','A','B','C']

                
            elif self.model=='lorentzian':
                unbinned_likelihood = UnbinnedLH(double_lorentz, np.array(shift_phases))
                minuit = Minuit(unbinned_likelihood, mu_1=self.init[0], gamma_1=self.init[1],mu_2=self.init[2],gamma_2=self.init[3],A=self.init[4],B=self.init[5],C=self.init[6])
                self.parnames=['mu_1', 'gamma_1','mu_2','gamma_2','A','B','C']

              
            elif self.model=='gaussian':
                unbinned_likelihood = UnbinnedLH(gaussian, np.array(shift_phases))
                minuit = Minuit(unbinned_likelihood, mu=self.init[0], sigma=self.init[1],A=self.init[2],B=self.init[3])
                self.parnames=['mu', 'sigma','A','B']

            minuit.errordef=0.5
            minuit.migrad()

            #Store results as minuit object
            self.minuit=minuit
            self.unbinned_lk=unbinned_likelihood
            
            #Store the result of params and errors
            self.params=[]
            self.errors=[]
            for name in self.parnames:
                self.params.append(self.minuit.values[name])
                self.errors.append(self.minuit.errors[name])
 
            self.create_result_df()
    
    

        #Binned fitting 
        def fit_Binned(self,histogram):
            self.check_model()
            
            #Shift the phases if one of the peak is near the interval edge
            bin_centres=(histogram.lc[1][1:]+histogram.lc[1][:-1])/2
            if self.shift!=0:
                for i in range(0,len(bin_centres)):
                    if bin_centres[i]<self.shift:
                        bin_centres[i]=bin_centres[i]+1
            
            
            if self.model=='dgaussian':
                self.params,pcov_l=curve_fit(double_gaussian,bin_centres,histogram.lc[0],p0=self.init)
                self.parnames=['mu', 'sigma','mu_2','sigma_2','A','B','C']
                
                   
            elif self.model=='asym_dgaussian':
                assymetric_gaussian_pdf_vec=np.vectorize(assymetric_double_gaussian)
                self.params,pcov_l=curve_fit(assymetric_gaussian_pdf_vec,bin_centres,histogram.lc[0],p0=self.init)
                self.parnames=['mu', 'sigma1','sigma2','mu_2','sigma1_2','sigma2_2','A','B','C']


            elif self.model=='lorentzian':
                self.params,pcov_l=curve_fit(double_lorentz,bin_centres,histogram.lc[0],p0=self.init)
                self.parnames=['mu_1', 'gamma_1','mu_2','gamma_2','A','B','C']

            
            elif self.model=='gaussian':
                self.params,pcov_l=curve_fit(gaussian,bin_centres,histogram.lc[0],p0=self.init)
                self.parnames=['mu', 'sigma','A','B']

                
            #Store the result of params and errors
            self.errors=np.sqrt(np.diag(pcov_l))
            self.create_result_df()

        
        
        ##############################################
                       #RESULTS
        #############################################
    
    
        def check_fit_result(self):
            try:
                self.params
            except:
                return(False)
            
            return(True)

        
        def create_result_df(self):
            d = {'Name': self.parnames, 'Value': self.params,'Error':self.errors}
            self.df_result=pd.DataFrame(data=d)
                
        def show_result(self):
            try:
                return(self.df_result)
            except:
                print('No fit has been done so far')
                
            

