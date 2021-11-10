import pandas as pd
import math
import astropy as ast
import numpy as np
from astropy.time import Time
import matplotlib.pylab as plt
from astropy import units as u
from astropy.io import fits
import warnings
from lstchain.reco.utils import get_effective_time,add_delta_t_key
from lstchain.io.io import dl2_params_lstcam_key,dl2_params_src_dep_lstcam_key
import os

class ReadFermiFile():
    
        def __init__(self, file):
            if 'fits' not in file:
                raise ValueError('No FITS file provided for Fermi-LAT data')
            else:
                self.fname=file
                
        def read_file(self):
            f=fits.open(self.fname)
            fits_table=f[1].data
            return(fits_table)
                
        def create_df_from_info(self,fits_table):
            time=fits_table['BARYCENTRIC_TIME'].byteswap().newbyteorder()
            phases=fits_table['PULSE_PHASE'].byteswap().newbyteorder()
            energies=fits_table['ENERGY'].byteswap().newbyteorder()
            dataframe = pd.DataFrame({"mjd_time":time,"pulsar_phase":phases,"dragon_time":time*3600*24,"energy":energies/1000})
            dataframe=dataframe.sort_values(by=['mjd_time'])
            self.info=dataframe
            return(self.info)
                
        def calculate_tobs(self):
            diff=np.array(self.info['mjd_time'].to_list()[1:])-np.array(self.info['mjd_time'].to_list()[0:-1])
            diff[diff>5/24]=0
            return(sum(diff)*24)
        
        def run(self):
            ftable=self.read_file()
            self.create_df_from_info(ftable)
            self.tobs=self.calculate_tobs()
            

            
             
                
class ReadLSTFile():
    
        def __init__(self, file=None, directory=None,src_dependent=False):
            
            if file==None and directory==None:
                raise ValueError('No file provided')
            elif file is not None and directory is not None:
                raise ValueError('Can only provide file or directory, but not both')
            elif file is not None:
                if 'h5' not in file:
                    raise ValueError('No hdf5 file provided for LST data')
                else:
                    self.fname=file
                    
            elif directory is not None:
                self.direc=directory
                self.fname=[]
                for x in os.listdir(self.direc):
                    rel_dir = os.path.relpath(self.direc)
                    rel_file = os.path.join(rel_dir, x)
                    if 'h5' in rel_file:
                        self.fname.append(rel_file)
                        self.fname.sort()
                
            self.info=None
            self.src_dependent=src_dependent
            
            
        def read_LSTfile(self,fname):
            
            if self.src_dependent==False:
                df=pd.read_hdf(fname,key=dl2_params_lstcam_key)
            
            elif self.src_dependent==True:
                srcindep_df=pd.read_hdf(fname,key=dl2_params_lstcam_key,float_precision=20)
                srcindep_df=srcindep_df[["mjd_time","pulsar_phase", "dragon_time","alt_tel"]]
                
                srcdep_df=pd.read_hdf(fname,key=dl2_params_src_dep_lstcam_key)
                srcdep_df.columns = pd.MultiIndex.from_tuples([tuple(col[1:-1].replace('\'', '').replace(' ','').split(",")) for col in srcdep_df.columns])
                
                df= pd.concat([srcindep_df, srcdep_df['on']], axis=1)
            
            if 'alpha' in df and 'theta2' in df:
                df_filtered=df[["mjd_time","pulsar_phase", "dragon_time","gammaness","alpha","theta2","alt_tel"]]
            elif 'alpha' in df and 'theta2' not in df:
                df_filtered=df[["mjd_time","pulsar_phase", "dragon_time","gammaness","alpha","alt_tel"]]
            elif 'theta2' in df and 'alpha' not in df:
                df_filtered=df[["mjd_time","pulsar_phase", "dragon_time","gammaness","theta2","alt_tel"]]
            else:
                df_filtered=df[["mjd_time","pulsar_phase", "dragon_time","gammaness","alt_tel"]]
            
            try:
                df_filtered['energy']=df['reco_energy']
            except:
                df_filtered['energy']=df['energy']
            
            return(df_filtered)

                
             
            
        def calculate_tobs(self):
            dataframe=add_delta_t_key(self.info)
            return(get_effective_time(dataframe)[1].value/3600)
        
        
        def run(self,pulsarana):
            if isinstance(self.fname,list):
                info_list=[]
                for name in self.fname:
                    info_file=self.read_LSTfile(name)
                    self.info=info_file
                    pulsarana.cuts.apply_fixed_cut(self)
                    info_list.append(self.info)
                    
                self.info=pd.concat(info_list)
                self.tobs=self.calculate_tobs()
                
            else:
                self.info=self.read_LSTfile(self.fname)
                self.tobs=self.calculate_tobs()
                pulsarana.cuts.apply_fixed_cut(self)
                
            

            
            
            
class ReadtxtFile():
    
        def __init__(self, file,format_txt):
            self.fname=file
            self.format=format_txt
            
        def read_file(self):
            data = pd.read_csv(file, sep=" ", header=None)
            return(data)
        
        def check_format(self):
            for name in ['t','p']:
                if name not in self.format:
                    raise ValueError('No valid format')
                
                     
        def create_df_from_info(self,df):
            
            for i in range(0,len(self.format)):
                if self.format[i]=='t':
                    times=df.iloc[:, i]
                elif self.format[i]=='e':
                    energies=df.iloc[:, i]
                elif self.format[i]=='p':
                    phases=df.iloc[:, i]
                elif self.format[i]=='g':
                    gammaness=df.iloc[:, i]
                elif self.format[i]=='a':
                    alphas=df.iloc[:, i]
                elif self.format[i]=='t2':
                    theta2=df.iloc[:, i]
                elif self.format[i]=='at':
                    alt_tel=df.iloc[:, i]
            
            dataframe = pd.DataFrame({"mjd_time":times,"pulsar_phase":phases,"dragon_time":times*3600*24,"energy":energies})

            try:
                dataframe['gammaness']=gammaness
            except:
                pass
            
            try:
                dataframe['alpha']=alpha
            except:
                pass
            
            try:
                dataframe['theta2']=theta2
            except:
                pass
            
            try:
                dataframe['alt_tel']=alt_tel
            except:
                pass
            
            
            dataframe=dataframe.sort_values(by=['mjd_time'])
            self.info=dataframe
        
        
        def calculate_tobs(self):
            diff=np.array(self.info['mjd_time'].to_list()[1:])-np.array(self.info['mjd_time'].to_list()[0:-1])
            return(sum(diff)*24)
        
        
        def run(self):
            data=self.read_file()
            self.check_format()
            self.create_df_from_info(data)
            self.tobs=self.calculate_tobs()
                

                
class ReadList():
    
        def __init__(self, phases_list, time_list):
            self.plist=phases_list
            self.tlist=time_list
            
            
        def create_df_from_info(self):
            dataframe = pd.DataFrame({"mjd_time":self.tlist,"pulsar_phase":self.plist,"dragon_time":self.tlist*3600*24})
            dataframe=dataframe.sort_values(by=['mjd_time'])
            self.info=dataframe
        
        
        def calculate_tobs(self):
            diff=np.array(self.info['mjd_time'].to_list()[1:])-np.array(self.info['mjd_time'].to_list()[0:-1])
            return(sum(diff)*24)
        
        
        def run(self):
            self.create_df_from_info()
            self.tobs=self.calculate_tobs()
                
                
