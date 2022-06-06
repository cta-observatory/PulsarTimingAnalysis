
#####################
###Author: Alvaro Mas Aguilar (alvmas)
#mail: alvmas@ucm.es
#Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files.
###################3




import pandas as pd
import csv
import numpy as np
from astropy.time import Time
import pint
import pint.models as models
from pint.models import (
        parameter as p,
)
from pint.models.timing_model import (
        TimingModel,
        Component,
)
import astropy.units as u
import pint.toa as toa
import decimal
import time
import os
import pandas as pd
from lstchain.io import global_metadata, write_metadata, standard_config,srcdep_config 
from lstchain.io.io import write_dataframe, dl2_params_src_dep_lstcam_key,write_dl2_dataframe,dl2_params_lstcam_key

__all__=[
        'read_ephemfile',
        'dl2time_totim',
        'model_fromepehm',
        'add_mjd',
        'merge_dl2_pulsar'
        ]


def merge_dl2_pulsar(directory,run_number,output_dir,src_dep=False):
        
    #Read the DL2 subrun files for the given run number
    filelist=[]
    for x in os.listdir(directory):
        p_dir = os.path.relpath(directory)
        p_file = os.path.join(directory, x)
        if run_number in p_file:
            filelist.append(p_file)
    filelist.sort()
    
    #Read the Dl2 events and concatenate the dataframes
    df_list=[]
    df_src_list=[]
    for file in filelist:
        df_i=pd.read_hdf(file,key=dl2_params_lstcam_key)
        df_list.append(df_i)
        
        #Include source dependent information if it is the case
        if src_dep==True:
            df_i_src=pd.read_hdf(file,key=dl2_params_src_dep_lstcam_key,float_precision=20)
            df_src_list.append(df_i_src)

    df = pd.concat(df_list)
    if src_dep==True:
        df_src=pd.concat(df_src_list)

    #Write the new merged dataframe into a file
    output_file=output_dir+str(os.path.basename(filelist[0]).replace('.0000_pulsar.h5',''))+'_pulsar.h5'
    metadata = global_metadata()
    write_metadata(metadata, output_file)

    if src_dep==False:
        write_dl2_dataframe(df, output_file, meta=metadata)

    else:
        write_dl2_dataframe(df, output_file,meta=metadata)
        write_dataframe(df_src, output_file, dl2_params_src_dep_lstcam_key,meta=metadata)




def read_ephemfile(ephem):

    '''
    Read the ephem file and store the values as a dataframe

    Parameters:
    -----------------
    ephem: string
    Name of the ephemeris file
    
    Returns:
    -----------------
    Ephem: Dataframe with the pulsar information
        
    '''

    colnames=['PSR', 'RAJ1','RAJ2','RAJ3', 'DECJ1','DECJ2','DECJ3', 'START', 'FINISH', 't0geo', 'F0', 'F1', 'F2',
              'RMS','Observatory', 'EPHEM', 'PSR2']
    Ephem = pd.read_csv(ephem, delimiter='\s+',names=colnames,header=None)
    
    return(Ephem)
    
    
    
def dl2time_totim(times, name='times.tim',obs='lst'):
    
    '''
    Creates a .tim file from the arrival times and the observatory (by default the lst)

    Parameters:
    -----------------
    times: string
    List of arrival times 
    
    name: string
    Name of the output .tim file
    
    obs: string
    Code of the observatory
    '''
    if obs=='lst':
        timFile=open(name,'w+')
        timFile.write('FORMAT 1 \n')
        if obs=='lap':
            timFile.write('lst '+'0.0 '+str(times[0])+' 0.0 '+ 'coe'+ ' \n')
            for i in range(1,len(times)):
                timFile.write('lst '+'0.0 '+str(times[i])+' 0.0 '+ str(obs)+' \n')
        else:
            for i in range(0,len(times)):
                timFile.write('lst '+'0.0 '+str(times[i])+' 0.0 '+ str(obs)+' \n')
        timFile.close()

    
def model_fromephem(times,ephem,model_name):

    '''
    Creates a .par file using the parameters from the ephemeris associated with the arrival time range from TOAs

    Parameters:
    -----------------
    time: string
    List of arrival times 
    
    ephem: string
    Ephemeris to be used (.txt file or similar)
      
    Returns:
    --------
    Name of the saved .par file
    
    '''

    #Read the ephemeris txt file
    df_ephem=read_ephemfile(ephem)
        
    #Search the line of the ephemeris at which the interval time of arrivals given belongs
    for i in range(0,len(df_ephem['START'])):
        if (times[0]>df_ephem['START'][i]) & (times[0]<df_ephem['FINISH'][i]):
            break
        elif (times[0]< df_ephem['START'][i]) & (i==0):
            print('No ephemeris available')
        elif (times[0]> df_ephem['START'][i]) & (times[0]> df_ephem['FINISH'][i])& (i==len(df_ephem['START'])):
            print('No ephemeris available')

    #Select componentes of the model
    all_components = Component.component_types
    selected_components = ["AbsPhase","AstrometryEquatorial", "Spindown","SolarSystemShapiro"]
    component_instances = []

    # Initiate the component instances (see PINT pulsar documentation)
    for cp_name in selected_components:
        component_class = all_components[cp_name]  # Get the component class
        component_instance = component_class()  # Instantiate a component object
        component_instances.append(component_instance)

    tm = TimingModel("pulsar_test", component_instances)

    #Add second derivative of the frequency
    f2 = p.prefixParameter(
        parameter_type="float",
        name="F2",
        value=0.0,
        units=u.Hz / (u.s) ** 2,
        longdouble=True,
    )

    tm.components["Spindown"].add_param(f2, setup=True)

    #Add START and FINISH parameters
    tm.add_param_from_top(
            p.MJDParameter(name="START", description="Start MJD for fitting"), ""
        )
    tm.add_param_from_top(
            p.MJDParameter(name="FINISH", description="End MJD for fitting"), ""
    )
    
        
    f1=float(str(df_ephem['F1'][i].replace('D','E')))
    f2=float(str(df_ephem['F2'][i].replace('D','E')))
    
    #Give values to the parameters
    params = {
            "PSR":(df_ephem['PSR'][i],) ,
            "RAJ": (str(df_ephem['RAJ1'][i])+':'+ str(df_ephem['RAJ2'][i])+':'+str(df_ephem['RAJ3'][i]),),
            "DECJ": (str(df_ephem['DECJ1'][i])+':'+ str(df_ephem['DECJ2'][i])+':'+str(df_ephem['DECJ3'][i]),),
            "START": (Time(df_ephem['START'][i], format="mjd", scale="tdb"),),
            "FINISH": (Time(df_ephem['FINISH'][i], format="mjd", scale="tdb"),),
            "EPHEM":(df_ephem['EPHEM'][i],),
            'PEPOCH':(Time(int(df_ephem['t0geo'][i]), format="mjd", scale="tdb"),),
            "F0": (df_ephem['F0'][i]*u.Hz,),
            "F1": (f1*u.Hz/u.s,),
            "F2":(f2*u.Hz/(u.s**2),),
            "TZRMJD":(Time(df_ephem['t0geo'][i], format="mjd", scale="tdb"),),
            "TZRFRQ":(0.0*u.Hz,),
            "TZRSITE":('coe',),
            }


    #Create the model using PINT
    for name, info in params.items():
        par = getattr(tm, name)  # Get parameter object from name
        par.quantity = info[0]  # set parameter value
        if len(info) > 1:
                if info[1] == 1:
                        par.frozen = False  # Frozen means not fit.
                        par.uncertainty = info[2]

    tm.validate()
    print('New model generated')
    print(tm)

    #Create the .par file
    tm.as_parfile()
    name=model_name
    f=open(name,"w+")
    f.write(tm.as_parfile())
    f.close()
    
    return(name)
    

def add_mjd(file_dataframe):
    times=file_dataframe.dragon_time.values
    t = Time(times,format='unix', scale='utc')
    mjd_time=t.mjd

    #Add time in MJD
    print('Adding MJD time')
    file_dataframe['mjd_time']=mjd_time.tolist()
    
    return(mjd_time.tolist())

