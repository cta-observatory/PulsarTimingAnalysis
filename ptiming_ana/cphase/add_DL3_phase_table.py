#####################
###Author: Alvaro Mas Aguilar (alvmas)
#mail: alvmas@ucm.es
#Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files.
###################3

'''
Script that creates from a DL3 standard file a DL3_pulsar file with the information of the phases and corrected times for the pulsar analysis. 

Parameters:
----------------------
--dir: string 
  Directory where to find the standard DL3 files

--in-file: string
  DL3 standard file if want to analyze only one file

--ephem:string
  Path to the ephemeris file (.par or .gro)

--output:string
  Path where to store the DL3 file with the pulsar info

--run-number: int 
  Run number to process (only if --dir is given)

--interpolation: boolean 
   Set to True if want to use the interpolation method (faster but loses some precision)

Usage:
------------------------
1. An example of usage for a given file is:
python add_DL3_phase_table.py 
       --in-file ./DL3_directory/dl3_LST-1.Run0000.fits.gz 
       --output ./DL3_pulsar_directory/ 
       --ephem model_test.par

2. Another way of calling (if we have a directory and the run number) is:
python add_DL3_phase_table.py 
       --dir ./DL3_directory/
       --run-number 0000
       --output ./DL3_pulsar_directory/ 
       --ephem model_test.par

3. If we want to use the interpolation method:
python add_DL3_phase_table.py 
       --in_file ./DL3_directory/dl3_LST-1.Run0000.fits.gz 
       --output ./DL3_pulsar_directory/ 
       --ephem model_test.par
       --interpolation

'''

import sys
import pandas as pd
import argparse
import numpy as np
import os
import warnings
from ptiming_ana.cphase.pulsarphase_cal import DL3_calphase
    

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', '-d', action='store',type=str,dest='directory',default=None)
	parser.add_argument('--in-file', '-f', action='store',type=str,dest='in_file',default=None)
	parser.add_argument('--ephem','-ephem',action='store',type=str,dest='ephem',default=None)
	parser.add_argument('--output','-out',action='store',type=str,dest='dir_output',default=None)
	parser.add_argument('--pickle','-pickle',action='store',type=bool,dest='pickle',default=False)
	parser.add_argument('--run-number','-r',action='store',type=str,dest='run',default=False)
	parser.add_argument('--observatory','-obs',action='store',type=str,dest='observatory',default='lst')
	parser.add_argument('--interpolation','-interp',action='store_true',dest='interpolation')
	parser.add_argument('--create-tim','-tim',action='store_true',dest='create_tim')
    
    
	args = parser.parse_args()

	ephem=args.ephem
	output_dir=args.dir_output
	pickle=args.pickle
	in_file=args.in_file
	run=args.run
	interpolation=args.interpolation
	observatory=args.observatory
	create_tim=args.create_tim
    
	if output_dir is None:
		warnings.warn("WARNING: No output directory is given so the output will not be saved")

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
			DL3_calphase(filelist[i],ephem,output_dir,create_tim,observatory, interpolation,pickle)
		
	else:
		if in_file is not None:
			#Calculate the phases
			DL3_calphase(in_file,ephem,output_dir,create_tim,observatory,interpolation,pickle)
		else:
			raise ValueError('No input file or directory given')

    
if __name__ == "__main__":
    main()


