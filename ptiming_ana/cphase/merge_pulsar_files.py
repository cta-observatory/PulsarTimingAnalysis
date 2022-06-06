#####################
###Author: Alvaro Mas Aguilar (alvmas)
#mail: alvmas@ucm.es
#Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files.
###################3


import sys
import pandas as pd
import argparse
import numpy as np
import os
import warnings
from .utils import merge_dl2_pulsar

    

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', '-d', action='store',type=str,dest='directory',default=None)
	parser.add_argument('--output','-out',action='store',type=str,dest='dir_output',default=None)
	parser.add_argument('--srcdep','-srcdep',action='store',type=bool,dest='src_dep',default=False)
	parser.add_argument('--run-number','-r',action='store',type=str,dest='run',default=False)
    
    
	args = parser.parse_args()
	output_dir=args.dir_output
	src_dep=args.src_dep
	run=args.run
        directory=args.directory

	merge_dl2_pulsar(directory,run,output_dir,src_dep)

if __name__ == "__main__":
    main()


