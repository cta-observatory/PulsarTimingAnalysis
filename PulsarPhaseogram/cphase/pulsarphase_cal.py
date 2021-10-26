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
from lstchain.io.io import dl2_params_src_dep_lstcam_key, write_dataframe
from pint.fits_utils import read_fits_event_mjds
from pint.fermi_toas import *
from pint.scripts import *
from .utils import  add_mjd,dl2time_totim, model_fromephem
import pint.models as models

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


    
def calphase(file,ephem,output_dir,pickle):
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
    df_i_src=pd.read_hdf(file,key=dl2_params_src_dep_lstcam_key,float_precision=20)
    add_mjd(df_i)

    #Create the .tim file
    timname=str(os.path.basename(file).replace('.h5',''))+'.tim'
    timelist=df_i.mjd_time.tolist()
    dl2time_totim(timelist,name=timname)
        
    t= toa.get_TOAs(timname, usepickle=pickle)
    #Create model from ephemeris
    model=model_fromephem(timelist,ephem)

    #Upload TOAs and model
    m=models.get_model(model)

    #Calculate the phases
    print('Calculating barycentric time and absolute phase')
    barycent_toas=m.get_barycentric_toas(t)
    phase=m.phase(t,abs_phase=True)

            
    #Write if dir given
    if output_dir is not None:
        print('Generating new columns in DL2 DataFrame')
        df_i['mjd_barycenter_time']=barycent_toas
        df_i['pulsar_phase']=phase.frac
        dir_output=output_dir+str(os.path.basename(file).replace('.h5',''))+'_pulsar.h5'
            
        print('Writing outputfile in'+str(dir_output))
        df_i.to_hdf(dir_output,key='dl2/event/telescope/parameters/LST_LSTCam')
        write_dataframe(df_i_src, dir_output, dl2_params_src_dep_lstcam_key)
        os.remove(str(os.getcwd())+'/'+timname)
        print('Finished')
 
    else:
        ('Finished. Not output directory given so the output is not saved')
    




