#####################
###Author: Alvaro Mas Aguilar (alvmas)
#mail: alvmas@ucm.es
#Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files in a new table called phase_info
###################3


'''
Script that adds the information of the phases and corrected times for the pulsar analysis to a DL2 file. It includes this information in a new table called 'phase_info' and optionally a new table with the ource position and theta2 values (called 'source_position')

Parameters:
----------------------
--dir: string 
  Directory where to find the standard DL3 files

--in-file: string
  DL3 standard file if want to analyze only one file

--ephem:string
  Path to the ephemeris file (.par or .gro)

--run-number: int 
  Run number to process (only if --dir is given)

--include-theta: boolean
  True if want to add the source position and theta2 values on the DL2 files (in a table called 'source_position')

--interpolation: boolean 
   Set to True if want to use the interpolation method (faster but loses some precision)


Usage:
------------------------
1. An example of usage for a given file is:
python add_DL2_phase_table.py 
       --in-file ./DL2_directory/dl2_LST-1.Run0000.fits.gz 
       --ephem model_test.par

2. Another way of calling (if we have a directory and the run number) is:
python add_DL2_phase_table.py 
       --dir ./DL2_directory/
       --run-number 0000
       --ephem model_test.par

3. If we want to use the interpolation method:
python add_DL2_phase_table.py 
       --in_file ./DL2_directory/dl3_LST-1.Run0000.fits.gz 
       --ephem model_test.par
       --interpolation

4. If we want to include the source position and the theta2 values:
python add_DL2_phase_table.py 
       --in_file ./DL2_directory/dl3_LST-1.Run0000.fits.gz 
       --ephem model_test.par
       --interpolation
       --include-theta

'''

import sys
import pandas as pd
import argparse
import numpy as np
import os
import warnings
from pulsarphase_cal import DL2_calphase
from utils import add_source_info_dl2

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', '-d', action='store',type=str,dest='directory',default=None)
	parser.add_argument('--in-file', '-f', action='store',type=str,dest='in_file',default=None)
	parser.add_argument('--ephem','-ephem',action='store',type=str,dest='ephem',default=None)
	parser.add_argument('--pickle','-pickle',action='store',type=bool,dest='pickle',default=False)
	parser.add_argument('--run-number','-r',action='store',type=str,dest='run',default=False)
	parser.add_argument('--include-theta','-t',action='store_true',dest='include_theta')
	parser.add_argument('--interpolation','-interp',action='store_true',dest='interpolation')
    
    
	args = parser.parse_args()

	ephem=args.ephem
	pickle=args.pickle
	in_file=args.in_file
	run=args.run
	include_theta=args.include_theta
	interpolation=args.interpolation
    
	dl2_params_lstcam_key='dl2/event/telescope/parameters/LST_LSTCam'
	pd.set_option("display.precision", 10)


	if ephem is None:
      		raise ValueError('No ephemeris provided')
                    
	if args.directory is not None:
		if in_file is not None:
			raise ValueError('Both directory and file were given, can only use one of them')
            
		filelist=[]
		for x in os.listdir(args.directory):
			rel_dir = os.path.relpath(args.directory)
			rel_file = os.path.join(rel_dir, x)
			if run in rel_file:
				filelist.append(rel_file)
                
		filelist.sort()
		for i in range(0,len(filelist)):
			#Calculate the phases
			DL2_calphase(filelist[i],ephem,'lst',interpolation,pickle)
			if include_theta:
				add_source_info_dl2(output_dir+str(os.path.basename(filelist[i]).replace('.h5',''))+'_pulsar.h5','Crab')
		
	else:
		if in_file is not None:
			#Calculate the phases
			DL2_calphase(in_file,ephem,'lst',interpolation,pickle)
			if include_theta:
				add_source_info_dl2(output_dir+str(os.path.basename(in_file).replace('.h5',''))+'_pulsar.h5','Crab')
		else:
			raise ValueError('No input file or directory given')

    
if __name__ == "__main__":
    main()


