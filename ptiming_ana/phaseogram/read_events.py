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
from lstchain.io.io import dl2_params_lstcam_key,dl2_params_src_dep_lstcam_key, get_srcdep_params
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
            time=fits_table['BARY_TIME'].byteswap().newbyteorder()
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
            print('    Reading Fermi-LAT data file')
            ftable=self.read_file()
            self.create_df_from_info(ftable)
            self.tobs=self.calculate_tobs()
            print('    Finishing reading. Total time is '+str(self.tobs)+' h'+'\n')
            

            
             
                
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
            
        def add_phases(self,pname):
            dphase=pd.read_hdf(pname,key=dl2_params_lstcam_key)
            self.info['pulsar_phase']=dphase['pulsar_phase']
            
        def read_LSTfile(self,fname,df_type='short'):
            
            if self.src_dependent==False:
                df=pd.read_hdf(fname,key=dl2_params_lstcam_key)
            
            elif self.src_dependent==True:
                srcindep_df=pd.read_hdf(fname,key=dl2_params_lstcam_key,float_precision=20)
                on_df_srcdep=get_srcdep_params(fname,'on')
                
                if 'reco_energy' in srcindep_df.keys():
                    srcindep_df.drop(['reco_energy'])
                    
                if 'gammaness' in srcindep_df.keys():
                    srcindep_df.drop(['gammaness'])
                    
                df = pd.concat([srcindep_df, on_df_srcdep], axis=1)

                
            if df_type=='short':
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
            else:
                df_filtered = df
                df_filtered['energy']=df['reco_energy']
             
            df_filtered=add_delta_t_key(df_filtered)
            return(df_filtered)

                
             
      
        def calculate_tobs(self):
            dataframe=add_delta_t_key(self.info)
            return(get_effective_time(dataframe)[1].value/3600)
           
        
        
        
        def run(self,pulsarana,df_type='long'):
            print('    Reading LST-1 data file')
            if isinstance(self.fname,list):
                info_list=[]
                for name in self.fname:
                    try:
                        info_file=self.read_LSTfile(name,df_type)
                        self.info=info_file    
                        self.tobs=self.calculate_tobs()
                        pulsarana.cuts.apply_fixed_cut(self)
                        if pulsarana.cuts.energy_binning_cut is not None:
                            pulsarana.cuts.apply_energydep_cuts(self)

                        info_list.append(self.info)
                    except:
                        raise ValueError('Failing when reading:'+ str(name))
                        
                self.info=pd.concat(info_list)
                self.tobs=self.calculate_tobs()
                
            else:
                self.info=self.read_LSTfile(self.fname,df_type)
                self.tobs=self.calculate_tobs()
                
                print('    Finishing reading. Total time is '+str(self.tobs)+' h')
                pulsarana.cuts.apply_fixed_cut(self)
            
                if pulsarana.cuts.energy_binning_cut is not None:
                    pulsarana.cuts.apply_energydep_cuts(self)
                
                print('    Finishing filtering events:')
                print('        gammaness cut:'+str(pulsarana.cuts.gammaness_cut))
                print('        alpha cut:'+str(pulsarana.cuts.alpha_cut))
                print('        theta2 cut:'+str(pulsarana.cuts.theta2_cut))
                print('        zd cut:'+str(pulsarana.cuts.zd_cut))
                print('        energy binning for the cuts:'+str(pulsarana.cuts.energy_binning_cut))
                print('\n')

            
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
                    raise ValueError('   No valid format')
                
                     
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
            
            print('    Finishing reading. Total time is '+str(self.tobs)+' s'+'\n')
                

                
class ReadList():
    
        def __init__(self, phases_list, time_list=None,energy_list=None,tel='LST'):
            self.plist=phases_list
            self.tlist=time_list
            self.elist=energy_list
            self.tel=tel
            
        def create_df_from_info(self):
            dataframe = pd.DataFrame({"mjd_time":self.tlist,"pulsar_phase":self.plist,"dragon_time":self.tlist*3600*24,"energy":self.elist})
            dataframe=dataframe.sort_values(by=['mjd_time'])
            self.info=dataframe
        
        
        def calculate_tobs(self):
            if self.tel=='LST' or self.tel=='MAGIC':
                dataframe=add_delta_t_key(self.info)
                return(get_effective_time(dataframe)[1].value/3600)
              
            elif self.tel=='fermi':
                diff=np.array(self.info['mjd_time'].to_list()[1:])-np.array(self.info['mjd_time'].to_list()[0:-1])
                diff[diff>5/24]=0
                return(sum(diff)*24)
        
        
        def run(self):
            self.create_df_from_info()
            self.tobs=self.calculate_tobs()
            print('    Finishing reading. Total time is '+str(self.tobs)+' s'+'\n')
                
                
