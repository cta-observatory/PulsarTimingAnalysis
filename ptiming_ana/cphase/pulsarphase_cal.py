
#####################
###Author: Alvaro Mas Aguilar (alvmas)
#mail: alvmas@ucm.es
#Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files.
#Modified by: Paula Molina Sanjurjo (Pau-mol)
#mail: p.molinasanjurjo@gmail.com
###################

import pandas as pd
import csv
import os
import numpy as np
from astropy.time import Time
import pint
import astropy.units as u
from astropy.io import fits
from pint.observatory.satellite_obs import get_satellite_observatory
from lstchain.reco import dl1_to_dl2
import pint.toa as toa
import time
from lstchain.io import global_metadata, write_metadata, standard_config,srcdep_config 
from lstchain.io.io import dl2_params_src_dep_lstcam_key, write_dataframe, write_dl2_dataframe
from pint.fits_utils import read_fits_event_mjds
from pint.fermi_toas import *
from pint.scripts import *
from ptiming.cphase.utils import  add_mjd,dl2time_totim, model_fromephem,read_ephemfile
import pint.models as models
from pint.models import parameter as p
from pint.models.timing_model import (TimingModel, Component,)

__all__=['fermi_calphase','calphase']


def update_fermi(timelist,ephem,t):
    model=model_fromephem(timelist,ephem)

    #Upload TOAs and model
    m=models.get_model(model)

    #Calculate the phases
    print('Calculating barycentric time and absolute phase')
    bt=m.get_barycentric_toas(t)
    p=m.phase(t,abs_phase=True)
    
    return(bt,p)


def fermi_calphase(file,ephem,output_dir,pickle,ft2_file=None):
    
    '''
    Calculates barycentered times and pulsar phases from the DL2 dile using ephemeris. 

    Parameters:
    -----------------
    dl2file: string
    DL2 input file with the arrival times
    
    ephem: string
    Ephemeris to be used (.txt file or similar)
    
    
    output_dir:string
    Directory of the the output file
    
    pickle: boolean
    True if want to save a pickle file with the loaded TOAs
    
    
    '''
    print('Input file:'+str(file))
    #Load observatory and TOAs
    get_satellite_observatory("Fermi", ft2_file)
    tl=load_Fermi_TOAs(file,fermiobs='fermi')
    
    #Extract the timelist in MJD
    timelist=[]
    for i in range(0,len(tl)):
        timelist.append(tl[i].mjd.value)

    #Create TOAs object
    t = toa.get_TOAs_list(tl)
   
        
    #Calculate the phases in intervals of 1000 events so that ephemeris are updated
    barycent_toas=[]
    phase=[]
    k=0
    for i in range(10000,len(t),1000):
        b,p=update_fermi(timelist[k:i],ephem,t[k:i])
        barycent_toas=barycent_toas+list(b.value)
        phase=phase+list(p.frac.value)
        k=i
            
    b,p=update_fermi(timelist[k:],ephem,t[k:])
    barycent_toas=barycent_toas+list(b.value)
    phase=phase+list(p.frac.value)

            
    #Write if dir given
    hdul=fits.open(file)
    orig_cols = hdul[1].columns
    new_col1 = fits.Column(name='BARYCENTRIC_TIME', format='D',array=barycent_toas)
    new_col2 = fits.Column(name='PULSE_PHASE', format='D',array=phase)
    hdu = fits.BinTableHDU.from_columns(orig_cols + new_col1+new_col2)
            
    dir_output=output_dir+str(os.path.basename(file).replace('.fits',''))+'_pulsar.fits'
            
    print('Writing outputfile in'+str(dir_output))
    hdu.writeto(dir_output)
            
    print('Finished')

    os.remove(str(os.getcwd())+'/'+timname)

def DL3_calphase(file,ephem,output_dir,pickle=False):
    data = fits.open(file)
    orig_table = data[1].data
    orig_cols = orig_table.columns

    #Extract the times from the dataframe and transform scales
    df = pd.DataFrame(data[1].data)
    df=df.sort_values('TIME')
    
    lst=Time("2018-10-01",scale='utc')
    time_orig=df['TIME']

    time=time_orig+lst.to_value(format='unix')
    timelist=list(Time(time,format='unix').to_value('mjd'))
    
    #Get the name of the files
    timname=str(os.path.basename(file).replace('.fits',''))+'.tim'
    parname=str(os.path.basename(file).replace('.fits',''))+'.par'
    
    #Compute the phases using the interpolation method
    phase,barycent_toas=compute_phase_interpolation(timelist,ephem,timname,parname,pickle)
    
    #Shift phases
    for i in range(0,len(phase)):
        if phase[i]<0:
            phase[i]=phase[i]+1

    #Generate needed columns
    cols_list=[]
    for j in range(0,len(orig_cols)):
        try:
            cols_list.append(fits.Column(name=orig_cols[j].name, format=orig_cols[j].format,unit=orig_cols[j].unit,array=df[orig_cols[j].name]))
            
        except:
            cols_list.append(fits.Column(name=orig_cols[j].name, format=orig_cols[j].format,array=df[orig_cols[j].name]))                 
    orig_cols_sorted=fits.ColDefs(cols_list)
    
    #New columns to add to the DL3 files
    new_cols = fits.ColDefs([fits.Column(name='PHASE', format='D',array=phase),fits.Column(name='BARYCENT_TIME',format='D',array=barycent_toas)])
    
    
    #Create the files
    hdu = fits.BinTableHDU.from_columns(orig_cols_sorted + new_cols,header=data[1].header)
    hdu_list=fits.HDUList([data[0],hdu,data[2],data[3],data[4],data[5]])
    
    output_file=output_dir+str(os.path.basename(file).replace('.fits',''))+'_pulsar.fits'
    print('Writing outputfile in'+str(output_file))
    
    hdu_list.writeto(output_file)

    #Removing tim file
    os.remove(str(os.getcwd())+'/'+timname)

    
    
    
    
    
    
    
def calphase(file,ephem,output_dir,pickle=False):
    '''
    Calculates barycentered times and pulsar phases from the DL2 dile using ephemeris. 

    Parameters:
    -----------------
    dl2file: string
    DL2 input file with the arrival times
    
    ephem: string
    Ephemeris to be used (.par or .txt file or similar)
    
    output_dir:string
    Directory of the the output file
    
    pickle: boolean
    True if want to save a pickle file with the loaded TOAs
    
    Returns:
    --------
    Returns same DL2 with two additional columns: 'mjd_barycenter_time' and 'pulsar_phase'
    The name of this new file is dl2.....run_number_ON_Crab_pulsar.h5
    
    '''

    dl2_params_lstcam_key='dl2/event/telescope/parameters/LST_LSTCam'

    #Read the file
    print('Input file:'+str(file))
    df_i=pd.read_hdf(file,key=dl2_params_lstcam_key,float_precision=20)
    add_mjd(df_i)

    try:     
        df_i_src=pd.read_hdf(file,key=dl2_params_src_dep_lstcam_key,float_precision=20)
        src_dep=True
    except:
        src_dep=False
        
    #Create the .tim file
    timelist=df_i.mjd_time.tolist()
    timname=str(os.path.basename(file).replace('.h5',''))+'.tim'
    
    print('Creating .tim file')
    dl2time_totim(timelist,name=timname)
    
    print('Setting the .par file')
    if ephem.endswith('.par'):
        model=ephem
    elif ephem.endswith('.gro'):
        print('No .par file given. Creating .par file from .gro file...')
        #Create model from ephemeris
        parname=str(os.path.basename(file).replace('.h5',''))+'.par'
        model=model_fromephem(timelist,ephem,parname)

    #Calculate phases
    barycent_toas,phase=get_phase_list_from_tim(timname,model,pickle)
    
    #Removing tim file
    os.remove(str(os.getcwd())+'/'+timname)
    
    #Removing .par file if it was created during execution
    if ephem.endswith('.gro'):
        os.remove(str(os.getcwd())+'/'+parname)  
    
    #Create new dataframe: 
    df_phase=pd.DataFrame({'obs_id':df_i['obs_id'],'event_id':df_i['event_id'],'mjd_barycenter_time':barycent_toas,'pulsar_phase':phase})
    df_phase.to_hdf(file,key='phase_info')
    print('Finished')



        
def compute_phase_interpolation(timelist,ephem,timname,parname,pickle):
    
    #Extraxting reference values of times for interpolation
    timelist_n=timelist[0::1000]
    if timelist_n[-1]!=timelist[-1]:
        timelist_n.append(timelist[-1])
        
    #Time in seconds:
    timelist_s = np.array(timelist)*86400
    timelist_ns = np.array(timelist_n)*86400   
    
    #Create the tim and par files
    dl2time_totim(timelist_n,name=timname)
    model=model_fromephem(timelist_n,ephem,parname)
    
    #Calculate the barycent times and phases for reference
    barycent_toas_sample,phase_sample=get_phase_list_from_tim(timname,model,pickle)
    os.remove(str(os.getcwd())+'/'+parname)
    
    #Time of barycent_toas in seconds:
    btime_sample_sec = np.array(barycent_toas_sample)*86400  
  
    #Getting the period:
    #Read the ephemeris txt file
    df_ephem=read_ephemfile(ephem)

    #Search the line of the ephemeris at which the interval time of arrivals given belongs
    for i in range(0,len(df_ephem['START'])):
    	if (timelist[0]>df_ephem['START'][i]) & (timelist[0]<df_ephem['FINISH'][i]):
        	break
    P = 1/(df_ephem['F0'][i])

    #Number of complete cicles(N):    
    phase_sam = np.array(phase_sample.frac) + 0.5
    N=(1/P)*(np.diff(btime_sample_sec)-P*(1+np.diff(phase_sam)))    
    N=np.round(N)
    
    #For having the same dimensions:
    N = np.append([0], N)
   
    #The cumulative sum of N:
    sN=np.cumsum(N)
    
    #Sum of phases:
    sp = np.cumsum(phase_sam)
    #Sum of complementary phases shifted by 1:
    spc= np.append([0], np.cumsum(1-phase_sam)[:-1])
    
    #Adding sN + sp+ spc:
    phase_s = sp+sN+spc
    
    #Interpolate to all values of times:    
    barycent_toas = interpolate_btoas(timelist,timelist_n,barycent_toas_sample)
    barycent_toas_sec = np.array(barycent_toas)*86400
    phase=interpolate_phase(barycent_toas_sec,btime_sample_sec,phase_s)
    phase = phase%1
    phase = phase - 0.5
    
    return(phase,barycent_toas)
        
        
def calphase_interpolated(file,ephem,pickle=False):
    '''
    Calculates barycentered times and pulsar phases from the DL2 dile using ephemeris. 

    Parameters:
    -----------------
    dl2file: string
    DL2 input file with the arrival times
    
    ephem: string
    Ephemeris to be used (.txt file or similar)
    
    
    output_dir:string
    Directory of the the output file
    
    pickle: boolean
    True if want to save a pickle file with the loaded TOAs
    
    Returns:
    --------
    Returns same DL2 with two additional columns: 'mjd_barycenter_time' and 'pulsar_phase'
    The name of this new file is dl2.....run_number_ON_Crab_pulsar.h5
    
    '''

    dl2_params_lstcam_key='dl2/event/telescope/parameters/LST_LSTCam'

    #Read the file
    print('Input file:'+str(file))
    df_i=pd.read_hdf(file,key=dl2_params_lstcam_key,float_precision=20)
    add_mjd(df_i)
    times=df_i.dragon_time.values

    try:
        df_i_src=pd.read_hdf(file,key=dl2_params_src_dep_lstcam_key,float_precision=20)
        src_dep=True
        print('Using source-dependent analysis files')
    except:
        src_dep=False    
    
    
    #Create the .tim file
    timelist=df_i.mjd_time.tolist()   

    timname=str(os.path.basename(file).replace('.h5',''))+'.tim'
    parname=str(os.path.basename(file).replace('.h5',''))+'.par'
    
    #Compute the phases using the interpolation method
    phase,barycent_toas=compute_phase_interpolation(timelist,ephem,timname,parname,pickle)
    
    #Create new dataframe: 
    df_phase=pd.DataFrame({'obs_id':df_i['obs_id'],'event_id':df_i['event_id'],'mjd_barycenter_time':barycent_toas,'pulsar_phase':phase})
    df_phase.to_hdf(file,key='phase_info')
    print('Finished')

    os.remove(str(os.getcwd())+'/'+timname)    
        
def interpolate_phase(timelist,time_sample,phase_s):
    from scipy.interpolate import interp1d

    print('Interpolating phase...')
    interp_function_phase=interp1d(time_sample, phase_s, fill_value="extrapolate")
    
    phase=interp_function_phase(timelist)

    return(phase)



def interpolate_btoas(timelist,time_sample,barycent_toas_sample):
    from scipy.interpolate import interp1d

    print('Interpolating btoas...')
    interp_function_mjd = interp1d(time_sample, barycent_toas_sample,fill_value="extrapolate")
    barycent_toas=interp_function_mjd(timelist)

    return(barycent_toas)    


def get_phase_list(timname,timelist,ephem,parname,pickle=False):
    print('Creating tim file')
    dl2time_totim(timelist,name=timname)
    print('creating TOA list')

    t= toa.get_TOAs(timname, usepickle=pickle)
    #Create model from ephemeris
    model=model_fromephem(timelist,ephem,parname)

    #Upload TOAs and model
    m=models.get_model(model)

    #Calculate the phases
    print('Calculating barycentric time and absolute phase')
    barycent_toas=m.get_barycentric_toas(t)
    phase=m.phase(t,abs_phase=True)
    
    os.remove(str(os.getcwd())+'/'+timname)
    
    return(barycent_toas,phase)


def get_phase_list_from_tim(timname,model,pickle=False):
    
    print('creating TOA list')
    t= toa.get_TOAs(timname, usepickle=pickle)
    
    #Upload TOAs and model
    m=models.get_model(model)
    print(m)
    
    #Calculate the phases
    print('Calculating barycentric time and absolute phase')
    barycent_toas=m.get_barycentric_toas(t)
    phase=m.phase(t,abs_phase=True)
    
    return(barycent_toas,phase)
