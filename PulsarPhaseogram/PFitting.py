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

        def __init__(self, pulsar_phases,binned,model):
            
            #Define and check model
            self.model=model
            self.shift=0
            self.check_model()
            
            #Estimate initial values 
            self.est_initial_values(pulsar_phases)

            #Do the fitting 
            if binned==True:
                self.fit_Binned(pulsar_phases.histogram)
            else:
                self.fit_ULmodel(pulsar_phases)
         
            
        def check_model(self):
            model_list=['dgaussian','lorentzian','asym_dgaussian']
            if self.model not in model_list:
                raise ValueError('The model is not in the available model list')
        
            
        def est_initial_values(self,pulsar_phases):
                self.check_model()
                
                #Set different initial values for different models
                if self.model=='dgaussian' or self.model=='lorentzian':
                    if pulsar_phases.regions.P1.limits[0]>pulsar_phases.regions.P1.limits[1]:
                        self.init=[(pulsar_phases.regions.P1.limits[0]+pulsar_phases.regions.P1.limits[1]+1)/2,pulsar_phases.regions.P1.deltaP/2, np.sum(pulsar_phases.regions.P2.limits)/2,pulsar_phases.regions.P2.deltaP/2,100,0.1,0.1]
                        
                        self.shift=2*pulsar_phases.regions.P1.deltaP
                        
                    elif pulsar_phases.regions.P2.limits[0]>pulsar_phases.regions.P2.limits[1]:
                        self.init=[np.sum(pulsar_phases.regions.P1.limits)/2,pulsar_phases.regions.P1.deltaP/2, (pulsar_phases.regions.P2.limits[0]+pulsar_phases.regions.P2.limits[1]+1)/2,pulsar_phases.regions.P2.deltaP/2,100,0.1,0.1]
                    
                        self.shift=2*pulsar_phases.regions.P2.deltaP
                        
                    else:
                        self.init=[np.sum(pulsar_phases.regions.P1.limits)/2,pulsar_phases.regions.P1.deltaP/2, np.sum(pulsar_phases.regions.P2.limits)/2,pulsar_phases.regions.P2.deltaP/2,100,0.1,0.1]
                    
                        self.shift=0
                    
                elif self.model=='asym_dgaussian':
                    if pulsar_phases.regions.P1.limits[0]>pulsar_phases.regions.P1.limits[1]:
                        self.init=[(pulsar_phases.regions.P1.limits[0]+pulsar_phases.regions.P1.limits[1]+1)/2,pulsar_phases.regions.P1.deltaP/2,pulsar_phases.regions.P1.deltaP/2, np.sum(pulsar_phases.regions.P2.limits)/2,pulsar_phases.regions.P2.deltaP/2,pulsar_phases.regions.P2.deltaP/2,100,0.1,0.1]   
                    
                        self.shift=2*pulsar_phases.regions.P1.deltaP
                        
                    elif pulsar_phases.regions.P2.limits[0]>pulsar_phases.regions.P2.limits[1]:
                        self.init=[np.sum(pulsar_phases.regions.P1.limits)/2,pulsar_phases.regions.P1.deltaP/2,pulsar_phases.regions.P1.deltaP/2, (pulsar_phases.regions.P2.limits[0]+pulsar_phases.regions.P2.limits[1]+1)/2,pulsar_phases.regions.P2.deltaP/2,pulsar_phases.regions.P2.deltaP/2,100,0.1,0.1]   
                    
                        self.shift=2*pulsar_phases.regions.P2.deltaP
                        
                    else:
                        self.init=[np.sum(pulsar_phases.regions.P1.limits)/2,pulsar_phases.regions.P1.deltaP/2,pulsar_phases.regions.P1.deltaP/2, np.sum(pulsar_phases.regions.P2.limits)/2,pulsar_phases.regions.P2.deltaP/2,pulsar_phases.regions.P2.deltaP/2,100,0.1,0.1]   
                    
                        self.shift=0
                
                
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
                minuit.errordef=0.5
                minuit.migrad()
                
                #Store results as minuit object
                self.minuit=minuit
                self.unbinned_lk=unbinned_likelihood
                
            elif self.model=='asym_dgaussian':
                unbinned_likelihood_a = UnbinnedLH(assymetric_double_gaussian, np.array(shift_phases))
                minuit_a = Minuit(unbinned_likelihood_a, mu=self.init[0], sigma1=self.init[1],sigma2=self.init[2],mu_2=self.init[3],sigma1_2=self.init[4],sigma2_2=self.init[5],A=self.init[6],B=self.init[7],C=self.init[8])
                minuit_a.errordef=0.5
                minuit_a.migrad()
                
                #Store results as minuit object
                self.minuit=minuit_a
                self.unbinned_lk=unbinned_likelihood_a
                
            elif self.model=='lorentzian':
                unbinned_likelihood_l = UnbinnedLH(double_lorentz, np.array(shift_phases))
                minuit_l = Minuit(unbinned_likelihood_l, mu_1=self.init[0], gamma_1=self.init[1],mu_2=self.init[2],gamma_2=self.init[3],A=self.init[4],B=self.init[5],C=self.init[6])
                minuit_l.errordef=0.5
                minuit_l.migrad()
                
                #Store results as minuit object
                self.minuit=minuit_l
                self.unbinned_lk=unbinned_likelihood_l
                
            else:
                print('Input model not valid. Use dgaussian (double gaussian), adgaussian (assymetric double gaussian) or lorentzian')  
                
            #Store the result of params and errors
            self.parnames=self.minuit.parameters
            self.params=self.minuit.values.values()
            self.errors=self.minuit.errors.values()
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
                self.parnames=['mu', 'sigma','mu_2','sigma_2','A','B','C','Area1','Area2']
                self.errors=np.sqrt(np.diag(pcov_l))
                
                
                #Compute the areas
                self.params=np.append(self.params,[np.sqrt(2*np.pi)*self.params[-2]*abs(self.params[1]),np.sqrt(2*np.pi)*self.params[-1]*abs(self.params[4])])
                
                self.errors=np.append(self.errors,[self.params[-2]*np.sqrt((self.errors[1]/self.params[1])**2+(self.errors[5]/self.params[5])**2),self.params[-1]*np.sqrt((self.errors[3]/self.params[3])**2+(self.errors[6]/self.params[6])**2)])
                
                
            elif self.model=='asym_dgaussian':
                assymetric_gaussian_pdf_vec=np.vectorize(assymetric_double_gaussian)
                self.params,pcov_l=curve_fit(assymetric_gaussian_pdf_vec,bin_centres,histogram.lc[0],p0=self.init)
                self.parnames=['mu', 'sigma1','sigma2','mu_2','sigma1_2','sigma2_2','A','B','C']
                self.errors=np.sqrt(np.diag(pcov_l))
                
                
            elif self.model=='lorentzian':
                self.params,pcov_l=curve_fit(double_lorentz,bin_centres,histogram.lc[0],p0=self.init)
                self.parnames=['mu_1', 'gamma_1','mu_2','gamma_2','A','B','C']
                self.errors=np.sqrt(np.diag(pcov_l))
                
            #Store the result of params and errors
            self.create_result_df()

                
        
        def create_result_df(self):
            d = {'Name': self.parnames, 'Value': self.params,'Error':self.errors}
            self.df_result=pd.DataFrame(data=d)
                
        def show_result(self):
                return(self.df_result)
            

