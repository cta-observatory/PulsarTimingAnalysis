import astropy.units as u
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import yaml
import logging 
import astropy.units as u
from .gammapy_utils import read_DL3_files, set_geometry, set_makers, execute_makers, set_model_to_fit, do_fitting, compute_spectral_points
from .config_reading import SpectralConfigSetting

LOG_FORMAT="%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
logging.basicConfig(level = logging.INFO, format = LOG_FORMAT, datefmt = "%Y-%m-%d %H:%M:%S")

logger=logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled=True


class SpectralPulsarAnalysis():
    
    def __init__(self, config=None):
        
        if config != None:
            if '.yaml' in config:
                self.init_settings(config)


    def init_settings(self, configuration_file):
        self.config_params= SpectralConfigSetting(configuration_file)
        self.config_params.set_all()
        
    
    def prepare_analysis(self):
        edependent_theta, max_rad, zd_range = self.config_params.extract_detailed_reading_info()
        
        self.source_ra=self.config_params.target_info['ra']
        self.source_dec=self.config_params.target_info['dec']
        
        logger.info('Information about the source. Name: ' + str(self.config_params.target_info['name']) +', RA: ' + str(self.source_ra) + 'deg, DEC: ' + str(self.source_dec) +'deg')
        self.observation_list, self.id_list, reader = read_DL3_files(directory=self.config_params.directory,target_radec=[self.source_ra,self.source_dec],max_rad=max_rad,zd_cuts=zd_range,energy_dependent_theta=edependent_theta)
        
        self.spectral_model, self.model = set_model_to_fit(predefined_model=self.config_params.model ,model_name=self.config_params.target_info['name'])

        return(reader)
        
        
    def prepare_geometry(self,reader):
        true_energy_axis, reco_energy_axis = self.config_params.extract_energy_geometry()
        self.on_region, self.geom, self.dataset_empty = set_geometry(reader, true_energy_axis, reco_energy_axis)
        
    def prepare_makers(self):
        self.dataset_maker, self.bkg_maker, safe_mask_maker = set_makers(tuple(self.config_params.phase_region_dic['Bkg']), self.on_phase_region)
        
        if self.config_params.extra_settings['use_safe_mask']:
            self.mask_maker = safe_mask_maker
        else:
            self.mask_maker = None
        
        
    def execute_analysis(self):
        datasets = execute_makers(self.observation_list, self.id_list, self.dataset_empty, self.dataset_maker, self.bkg_maker, name=self.config_params.target_info['name'], 
                                  safe_mask_maker=self.mask_maker, stacked=self.config_params.extra_settings['stacked'])
        
        logger.info('Doing the fitting from ' +str(self.config_params.e_min_fitting) + ' to ' + str(self.config_params.e_max_fitting))
        self.datasets, self.model_best, self.result = do_fitting(datasets, self.model, self.geom, emin_fit=self.config_params.e_min_fitting, emax_fit=self.config_params.e_max_fitting, stacked=self.config_params.extra_settings['stacked'])  
        
        logger.info('Creating ' +str(self.config_params.npoints) +' spectral points from ' +str(self.config_params.e_min_points) + ' to ' + str(self.config_params.e_max_points) + ' (minimun sqrt_ts = ' +str(self.config_params.min_ts) +')')
        self.flux_points, self.flux_points_dataset = compute_spectral_points(dataset, model, self.config_params.e_min_points, self.config_params.e_max_points, self.config_params.npoints, min_ts=self.config_params.min_ts, name=self.config_params.target_info['name'])

        
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
        
        logger.info('The on region phase range is: ' +str(self.on_phase_region))
        #Prepare makers
        self.prepare_makers()
        logger.info('Gammapy makers prepared for analysis')
        
        #Execute he analysis
        logger.info('Executing analysis...')
        self.execute_analysis()