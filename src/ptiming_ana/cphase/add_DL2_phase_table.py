"""
Script that adds the information of the phases and corrected times for the pulsar analysis to a DL2 file. It includes this information in a new table called 'phase_info' and optionally a new table with the source position and theta2 values (called 'source_position')

Parameters:
----------------------
--dir: string
  Directory where to find the DL2 files

--in-file: string
  DL2 file if want to analyze only one file

--ephem:string
  Path to the ephemeris file (.par or .gro)

--run-number: int
  Run number to process (only if --dir is given). If not given, all the files in the directory are processed.

--include-theta: boolean
  True if want to add the source position and theta2 values on the DL2 files (in a table called 'source_position')

--source-name: string
  Name of the source used to compute the source position (default: Crab)

--interpolation: boolean
  Set to True if want to use the interpolation method (faster but loses some precision)

--number-interpolation: int
  Number of events between two interpolation points.


Usage:
------------------------
1. An example of usage for a given file is:
add_DL2_phase_table
       --in-file ./DL2_directory/dl2_LST-1.Run0000.h5
       --ephem model_test.par

2. Another way of calling (if we have a directory and the run number) is:
add_DL2_phase_table
       --dir ./DL2_directory/
       --run-number 0000
       --ephem model_test.par

3. If we want to use the interpolation method:
add_DL2_phase_table
       --in-file ./DL2_directory/dl2_LST-1.Run0000.h5
       --ephem model_test.par
       --interpolation

4. If we want to include the source position and the theta2 values:
add_DL2_phase_table
       --in-file ./DL2_directory/dl2_LST-1.Run0000.h5
       --ephem model_test.par
       --interpolation
       --include-theta

Original author: Alvaro Mas Aguilar (alvmas)
"""

import argparse

import pandas as pd

from ptiming_ana import setup_logging
from ptiming_ana.cphase._cli import add_boolean_argument, find_files
from ptiming_ana.cphase.pulsarphase_cal import DL2_calphase
from ptiming_ana.cphase.utils import add_source_info_dl2


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add the pulsar phases and barycentric times to DL2 files"
    )
    parser.add_argument(
        "--dir",
        "-d",
        action="store",
        type=str,
        dest="directory",
        default=None,
        help="Directory where to find the DL2 files",
    )
    parser.add_argument(
        "--in-file",
        "-f",
        action="store",
        type=str,
        dest="in_file",
        default=None,
        help="DL2 file to process (if want to analyze only one file)",
    )
    parser.add_argument(
        "--ephem",
        "-ephem",
        action="store",
        type=str,
        dest="ephem",
        required=True,
        help="Path to the ephemeris file (.par or .gro)",
    )
    add_boolean_argument(
        parser,
        "--pickle",
        "-pickle",
        dest="pickle",
        help="Save a pickle file with the loaded TOAs",
    )
    parser.add_argument(
        "--run-number",
        "-r",
        action="store",
        type=str,
        dest="run",
        default=None,
        help="Run number to process (only if --dir is given)",
    )
    parser.add_argument(
        "--include-theta",
        "-t",
        action="store_true",
        dest="include_theta",
        help="True if want to add the source position and theta2 values on the DL2 files (in a table called source_position)",
    )
    parser.add_argument(
        "--source-name",
        "-s",
        action="store",
        type=str,
        dest="source_name",
        default="Crab",
        help="Name of the source, used to compute the source position if --include-theta is given",
    )
    parser.add_argument(
        "--observatory",
        "-obs",
        action="store",
        type=str,
        dest="observatory",
        default="lst",
        help="Observatory code",
    )
    parser.add_argument(
        "--interpolation",
        "-interp",
        action="store_true",
        dest="interpolation",
        help="Set to True if want to use the interpolation method (faster but loses some precision)",
    )
    parser.add_argument(
        "--number-interpolation",
        "-ninterp",
        action="store",
        type=int,
        dest="ninterp",
        default=1000,
        help="Number of events between two interpolation points",
    )
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    setup_logging()

    pd.set_option("display.precision", 10)

    if args.directory is not None and args.in_file is not None:
        parser.error("Both directory and file were given, can only use one of them")

    if args.directory is not None:
        filelist = find_files(args.directory, args.run)
    elif args.in_file is not None:
        filelist = [args.in_file]
    else:
        parser.error("No input file or directory given")

    for file in filelist:
        # Calculate the phases
        DL2_calphase(
            file,
            args.ephem,
            obs=args.observatory,
            use_interpolation=args.interpolation,
            n_interp=args.ninterp,
            pickle=args.pickle,
        )
        if args.include_theta:
            add_source_info_dl2(file, args.source_name)


if __name__ == "__main__":
    main()
