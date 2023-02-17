import astropy.units as u
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import yaml
import logging 
import astropy.units as u
from .gammapy_utils import read_DL3_files, set_geometry, set_makers, execute_makers, set_model_to_fit, do_fitting, compute_spectral_points, read_DL4_files
from .config_reading import SpectralConfigSetting
from .plot_utils import get_kwargs_points, get_kwargs_line, get_kwargs_region
from IPython.display import display

from gammapy.modeling import Fit
from itertools import combinations

LOG_FORMAT="%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
logging.basicConfig(level = logging.INFO, format = LOG_FORMAT, datefmt = "%Y-%m-%d %H:%M:%S")

logger=logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled=True


class SpectralPulsarAnalysis():
    
    def __init__(self, config=None, ref_model=None):
        
        self.ref_model = ref_model
        
        if config is not None:
            if '.yaml' in config:
                self.init_settings(config)
                
        if ref_model is not None:
            self.spectral_model = ref_model
        else:
            self.spectral_model = None
            
                

    ############# SETTINGS #######################

    def init_settings(self, configuration_file):
        logger.info('Reading configuration file')
        self.config_params= SpectralConfigSetting(configuration_file)
        self.config_params.set_all()
        
        
    def prepare_analysis(self):
        edependent_theta, max_rad, zd_range = self.config_params.extract_detailed_reading_info()
        
        self.source_ra=self.config_params.target_info['ra']
        self.source_dec=self.config_params.target_info['dec']
        
        logger.info('Information about the source. Name: ' + str(self.config_params.target_info['name']) +', RA: ' + str(self.source_ra) + 'deg, DEC: ' + str(self.source_dec) +'deg')
        self.observation_list, self.id_list, reader = read_DL3_files(directory=self.config_params.directory,target_radec=[self.source_ra,self.source_dec],max_rad=max_rad,zd_cuts=zd_range,energy_dependent_theta=edependent_theta)
        self.reader = reader
        self.spectral_model, self.model = self.set_reference_model()
            
        return(reader)
        
        
    def prepare_geometry(self,reader):
        true_energy_axis, reco_energy_axis = self.config_params.extract_energy_geometry()
        self.on_region, self.geom, self.dataset_empty = set_geometry(reader, true_energy_axis, reco_energy_axis)
        
        
    def prepare_makers(self):
        self.dataset_maker, self.bkg_maker, safe_mask_maker = set_makers(self.on_phase_region, tuple(self.config_params.phase_region_dic['Bkg']))
        logger.info ('The Background phase region is: '+ str(tuple(self.config_params.phase_region_dic['Bkg'])))
        
        if self.config_params.extra_settings['use_safe_mask']:
            self.mask_maker = safe_mask_maker
        else:
            self.mask_maker = None
        
        
    def set_reference_model(self, reference_model = None):
        if reference_model is not None:
            self.spectral_model = reference_model
            
        if self.spectral_model is not None:
            logger.info('Setting given reference model')
        else:
            logger.info('Setting automatic predefined model')
        
        self.spectral_model, self.model = self.get_predefined_model(self.spectral_model)
        
        return(self.spectral_model, self.model)

    ############# EXECUTION #######################
    def get_predefined_model(self, spectral_model = None):
        spec_model, model= set_model_to_fit(spectral_model = spectral_model, predefined_model=self.config_params.model, model_name=self.config_params.target_info['name'])
        return(spec_model, model)
    
    
    def execute_analysis(self):
        
        if self.config_params.reader_info['use_DL4']:
            datasets = read_DL4_files(self.config_params.dl4_dir, self.id_list, stacked = self.config_params.extra_settings['stacked'])
        
        else:
            datasets = execute_makers(self.observation_list, self.id_list, self.dataset_empty, self.dataset_maker, self.bkg_maker, OGIP_dir = self.config_params.dl4_dir, save_DL4= self.config_params.reader_info['save_DL4'], name = self.config_params.target_info['name'], safe_mask_maker = self.mask_maker, stacked = self.config_params.extra_settings['stacked'])
            
        
        logger.info('Doing the fitting from ' +str(self.config_params.e_min_fitting) + ' to ' + str(self.config_params.e_max_fitting))
        
        self.datasets, self.model_best, self.fit_object, self.fitting_result = do_fitting(datasets, self.model, self.geom, emin_fit=self.config_params.e_min_fitting, emax_fit=self.config_params.e_max_fitting, stacked=self.config_params.extra_settings['stacked'])  
        
        if self.config_params.bins_per_decade:
            logger.info('Creating ' +str(self.config_params.bins_per_decade) +' spectral points per decade from ' +str(self.config_params.e_min_points) + ' to ' + str(self.config_params.e_max_points) + ' (minimun sqrt_ts = ' +str(self.config_params.min_ts) +')')
        else:
            logger.info('Creating ' +str(self.config_params.npoints) +' spectral points from ' +str(self.config_params.e_min_points) + ' to ' + str(self.config_params.e_max_points) + ' (minimun sqrt_ts = ' +str(self.config_params.min_ts) +')')
        
        self.flux_points, self.flux_points_dataset = compute_spectral_points(self.datasets, self.model_best, self.config_params.e_min_points, self.config_params.e_max_points, self.config_params.npoints, self.config_params.bins_per_decade, min_ts=self.config_params.min_ts, name=self.config_params.target_info['name'])

        
        
    def run(self, peak='P1'):
        if self.config_params is None:
            raise ValueError('No config parameters have been set yet')
        
        #Initialize
        reader = self.prepare_analysis()
        
        #Prepare geometry
        self.prepare_geometry(reader)
        
        if len(self.config_params.phase_region_dic[peak]) > 2:
            self.on_phase_region=[]
            for i in range(0,len(self.config_params.phase_region_dic[peak]),2):
                self.on_phase_region.append(tuple(self.config_params.phase_region_dic[peak][i:i+2]))
        else:
            self.on_phase_region = tuple(self.config_params.phase_region_dic[peak])
        
        logger.info('The Signal phase region is: ' +str(self.on_phase_region))
        #Prepare makers
        self.prepare_makers()
        logger.info('Gammapy makers prepared for analysis')
        
        #Execute he analysis
        logger.info('Executing analysis...')
        self.execute_analysis()
        
        logger.info('FINISHED. Showing results and final plots')
        self.show_fitting_results()
        self.get_flux_points()
        self.plot_SED_residuals()
        #self.plot_excess_counts()
        self.plot_fp_likelihood()
        self.create_contour_lines_params()
        self.get_covariance_matrix()
        
        
        
        
        ############# RESULTS ######################
        
    def plot_SED_residuals(self, include_reference=True, include_best_model = True, include_statistical = True, kwargs_SED=None,  kwargs_best=None, kwargs_best_error=None, kwargs_ref=None, label_fp='Spectral points', color_fp='black',color_model='blue', color_ref='red', ref_label='Reference model'):
        
        fig_sed = plt.figure(figsize=(8,8))

        gs2 = GridSpec(7, 1)
        gs2.update(hspace=0.1)

        args1 = [gs2[:4,:]]
        args2 = [gs2[5:,:]]

        fig_gs1 = fig_sed.add_subplot(*args1)
        fig_gs2 = fig_sed.add_subplot(*args2)
        
        if include_reference:
            self.plot_ref_model(ax = fig_gs1, ref_spec_model= self.ref_model, kwargs_ref = kwargs_ref, ref_label = ref_label, color_ref=color_ref)
            
        self.plot_SED(ax = fig_gs1 , include_best_model =include_best_model, include_statistical = include_statistical, label_fp=label_fp, color_fp=color_fp, color_model=color_model, kwargs_fp = kwargs_SED, kwargs_best = kwargs_best ,kwargs_best_error=kwargs_best_error)
        self.plot_residuals(ax = fig_gs2)
        
        fig_sed.savefig(self.config_params.output_dir + 'SED.png')

    
    def plot_residuals(self, ax = None, kwargs = None):
        
        if ax is None:
            fig_sed = plt.figure(figsize=(8,5))
            ax = fig_sed.add_subplot()
            
        self.flux_points_dataset.plot_residuals(ax=ax, method='diff/model')

        ax.legend()
        ax.grid(which='both')
        ax.set_title('SED residuals')
        
        orig_xticks = [0.02,0.05,0.08,0.1,0.2,0.3,0.4,0.5,0.7,1,5,10]
        new_xticks = [20,50,80,100,200,300,400,500,700,1000,5000,10000]

        ax.set_xticks(orig_xticks,labels=new_xticks)
        ax.set_xlabel('E (GeV)')
        
        ax.set_xlim([self.config_params.e_min_points,self.config_params.e_max_points])
        

    def plot_SED(self, ax = None, include_best_model = True, include_statistical = True, label_fp='Spectral points', color_fp='black', color_model='blue', kwargs_fp = None, kwargs_best=None, kwargs_best_error=None):
        
        if ax is None:
            fig_sed = plt.figure(figsize=(8,5))
            ax = fig_sed.add_subplot()
            
        if kwargs_fp is None:
            kwargs_fp = get_kwargs_points(label=label_fp,color=color_fp)
        self.flux_points.plot(ax=ax, **kwargs_fp)

        
        if include_best_model:
            if kwargs_best is None:
                kwargs_best = get_kwargs_line(energy_bounds=[self.config_params.e_min_points,self.config_params.e_max_points], color = color_model, label='Best fit model')        
            if kwargs_best_error is None:
                kwargs_best_error = get_kwargs_region(energy_bounds=[self.config_params.e_min_points,self.config_params.e_max_points], color = color_model ,facecolor=color_model,alpha=0.2,label='Statistical uncertainty') 
            
            self.model_best.spectral_model.plot(ax=ax, **kwargs_best)
            
            if include_statistical:
                self.model_best.spectral_model.plot_error(ax=ax, **kwargs_best_error)
            

        ax.legend()
        ax.grid(which='both')
        ax.set_title('SED')

        orig_xticks = [0.02,0.05,0.08,0.1,0.2,0.3,0.4,0.5,0.7,1,5,10]
        new_xticks = [20,50,80,100,200,300,400,500,700,1000,5000,10000]

        ax.set_xticks(orig_xticks,labels=new_xticks)
        ax.set_xlabel('E (GeV)')
        
        ax.set_xlim([self.config_params.e_min_points,self.config_params.e_max_points])
        
    def get_flux_points(self,sed_type='e2dnde'):
        print('\n'+'Flux points using ' + str(sed_type) +' format')
        table = self.flux_points.to_table(sed_type=sed_type)
        print(table)
        return(table)
    
    
    def plot_ref_model(self, ax= None, ref_spec_model= None, kwargs_ref=None, ref_label=None, color_ref='red'):
        if ref_spec_model is None:
            ref_spec_model, ref_model = self.get_predefined_model()
        
        if kwargs_ref is None:
            kwargs_ref = get_kwargs_line(energy_bounds=[self.config_params.e_min_points,self.config_params.e_max_points], color = color_ref ,label=ref_label)    
        
        ref_spec_model.plot(ax=ax, **kwargs_ref)
        
    
    def show_fitting_results(self):
        print('RESULTS of the fitting:' +'\n')
        print(self.fitting_result)
        display(self.fitting_result.models.to_parameters_table())
       
                
    def plot_excess_counts(self,index=0):
        fig, ax = plt.subplots(figsize=(8,5)) 

        ax_spectrum,ax_residuals = self.datasets.plot_fit()
        
        
        orig_xticks = [0.02,0.05,0.08,0.1,0.2,0.3,0.4,0.5,0.7,1,5,10]
        new_xticks = [20,50,80,100,200,300,400,500,700,1000,5000,10000]

        ax.set_xticks(orig_xticks,labels=new_xticks)
        ax.set_xlabel('E (GeV)')
        
        
        fig.savefig(self.config_params.output_dir + 'excess_counts.png')

                
    def plot_fp_likelihood(self, kwargs_fp = None, color_fp='orange'):
    
        fig, ax = plt.subplots() 
        if kwargs_fp is None:
            kwargs_fp = get_kwargs_points( color=color_fp)    
                
        self.flux_points.plot(ax=ax, **kwargs_fp)
        self.flux_points.plot_ts_profiles(ax =ax, sed_type= 'e2dnde')
        
        orig_xticks = [0.02,0.05,0.08,0.1,0.2,0.3,0.4,0.5,0.7,1,5,10]
        new_xticks = [20,50,80,100,200,300,400,500,700,1000,5000,10000]

        ax.set_xticks(orig_xticks,labels=new_xticks)
        ax.set_xlabel('E (GeV)')
        ax.set_xlim([self.config_params.e_min_points,self.config_params.e_max_points])
        
        
        fig.savefig(self.config_params.output_dir + 'likelihood_profile.png')
        
        
    def fit_statistic_profile_params(self):
        total_stat=self.fitting_result.total_stat
        
        fig, axes = plt.subplots(nrows=1, ncols=len(self.model.parameters.free_parameters), figsize=(10,4))
        for ax, par in zip(axes, self.model.parameters.free_parameters):
            par.scan_n_values = 17
            idx=self.model.parameters.index(par)
            name = self.model.parameters.names[idx]
            profile = self.fit_object.stat_profile(datasets = self.datasets, parameter = par)
            
            ax.plot(profile[f"{name}_scan"],profile["stat_scan"] - total_stat)
            ax.set_xlabel(f"{par.name} {par.unit}")
            ax.set_ylabel("Delta TS")
            ax.set_title(f"{name}:\n {par.value:.1e} +- {par.error:.1e}")
    
        
    def get_covariance_matrix(self):
        print('\n'+'Covariance matrix data:'+'\n')
        print(self.fitting_result.models.covariance)
        
        print('\n'+'Total Correlation matrix:' +'\n')
        fig, ax = plt.subplots(figsize=(6,6)) 
        self.fitting_result.models.covariance.plot_correlation(ax)
        
        
        fig.savefig(self.config_params.output_dir + 'correlation.png')
        
        
    def extract_parameters(self):
        name_list=[]
        idx_list=[]
        for  par in self.model.parameters.free_parameters:
            idx_list.append(self.model.parameters.index(par))
            name_list.append(self.model.parameters.names[idx_list[-1]])

        return(name_list,idx_list)
    
    
    def create_contour_lines_params(self):
        names,index=self.extract_parameters()
        panels=[]
        cts_sigma = []
        contours = dict()
        for par_1, par_2 in combinations(names, r=2):
            
            par1 = self.datasets.models.parameters[par_1]
            par2 = self.datasets.models.parameters[par_2]
            
            par1.scan_n_values=20
            par2.scan_n_values=20
            
            new_fit = Fit(backend = 'minuit', optimize_opts={"print_level":0})
            stat_surface = new_fit.stat_surface(datasets=self.datasets, x=par1, y=par2, reoptimize = False)
            
            TS= stat_surface["stat_scan"]-self.fitting_result.total_stat
            stat_surface = np.sqrt(TS.T)
                  
            fig, ax =plt.subplots(figsize =(8,6))
            x_values = par1.scan_values
            y_values = par2.scan_values
            
            im = ax.pcolormesh(x_values, y_values, stat_surface, shading='auto')
            fig.colorbar(im, label="sqrt(TS)")
            ax.set_xlabel(f"{par1.name} "+ f"{par1.unit}")
            ax.set_ylabel(f"{par2.name} "+f"{par2.unit}")
            
            
            levels=[1,2,3]
            contours=ax.contour(x_values,y_values, stat_surface,levels=levels,colors='white')
            ax.clabel(contours,fmt="%.0f$\,\sigma$",inline=3,fontsize=15)
        
            fig.savefig(self.config_params.output_dir + f'contour_line_{par_1}_{par_2}.png')