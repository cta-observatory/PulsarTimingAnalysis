import astropy.units as u
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import yaml
import logging 
import astropy.units as u
from gammapy.maps import MapAxis

LOG_FORMAT="%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
logging.basicConfig(level = logging.INFO, format = LOG_FORMAT, datefmt = "%Y-%m-%d %H:%M:%S")

logger=logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled=True


class SpectralConfigSetting():
    def __init__(self, config):
        
        if config != None:
            if '.yaml' in config:
                self.read_configuration(config)

    
    def read_configuration(self,configuration_file):
        
        with open(configuration_file, "rb") as cfile:
                self.conf = yaml.safe_load(cfile)
                

    def set_general_from_config(self):
        logger.info('Reading general settings from configuration file')
        
        configuration_dict = self.conf

        # Read the directory
        self.directory = configuration_dict['pulsar_file_dir']
        
        # DL4 directory
        self.dl4_dir = configuration_dict['DL4_directory']
        if not os.path.exists(self.dl4_dir):
            logger.info('Creating directory: ' + self.dl4_dir)
            os.makedirs(self.dl4_dir)
            
        # Results Output directory
        self.output_dir = configuration_dict['results_output_directory']
        if not os.path.exists(self.output_dir):
            logger.info('Creating directory: ' + self.output_dir)
            os.makedirs(self.output_dir)

        #Energy dependent theta 
        self.reader_info = configuration_dict['reader']

        #Target info
        self.target_info = configuration_dict['target']

        #Regions
        self.phase_region_dic = configuration_dict['phase_regions']

        #Geometry
        self.energy_geometry = configuration_dict['energy_geometry']
        
        #Extra settings
        self.extra_settings = configuration_dict['analysis_extra_settings']
        
        
    def set_spectral_fitting_from_config(self):
        logger.info('Reading fitting parameters from configuration file')
        configuration_dict = self.conf

        spectral_fitting = configuration_dict['spectral_fitting']
        self.e_min_fitting = spectral_fitting['emin'] * u.Unit(spectral_fitting['units'])
        self.e_max_fitting = spectral_fitting['emax'] * u.Unit(spectral_fitting['units'])

        self.model = spectral_fitting['model']
            

        
    def set_spectral_points_from_config(self):
        logger.info('Reading spectral points parameters from configuration file')

        configuration_dict = self.conf
        
        spectral_points = configuration_dict['spectral_points']
        self.e_min_points = spectral_points['emin'] * u.Unit(spectral_points['units'])
        self.e_max_points = spectral_points['emax'] * u.Unit(spectral_points['units'])

        self.bins_per_decade = spectral_points['bins_per_decade']
        self.npoints = spectral_points['number_points']
        self.min_ts = spectral_points['min_ts']
            
            
    def set_all(self):
        self.set_general_from_config()
        self.set_spectral_fitting_from_config()
        self.set_spectral_points_from_config()
        
        
    def extract_energy_geometry(self):
        
        true_energies = self.energy_geometry['real']
        reco_energies = self.energy_geometry['reco']
        
        true_energy_axis = MapAxis.from_energy_bounds(true_energies['emin'], true_energies['emax'], true_energies['nbinning'], unit=true_energies['units'], name="energy_true")
        reco_energy_axis = MapAxis.from_energy_bounds(reco_energies['emin'], reco_energies['emax'], reco_energies['nbinning'], unit=reco_energies['units'], name="energy")

        return(true_energy_axis, reco_energy_axis)


    def extract_detailed_reading_info(self):
        zd_range = self.reader_info['zd_range']
        edependent_theta = self.reader_info['energy_dependent_theta']
        
        if not edependent_theta:
            max_rad = self.reader_info['max_rad']
        else:
            max_rad = None
        
        
        return(edependent_theta, max_rad, zd_range)