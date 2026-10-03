"""
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
  Run number to process (only if --dir is given). If not given, all the files in the directory are processed.

--interpolation: boolean
   Set to True if want to use the interpolation method (faster but loses some precision)

--number-interpolation: int
   Number of events between two interpolation points.

Usage:
------------------------
1. An example of usage for a given file is:
add_DL3_phase_table
       --in-file ./DL3_directory/dl3_LST-1.Run0000.fits.gz
       --output ./DL3_pulsar_directory/
       --ephem model_test.par

2. Another way of calling (if we have a directory and the run number) is:
add_DL3_phase_table
       --dir ./DL3_directory/
       --run-number 0000
       --output ./DL3_pulsar_directory/
       --ephem model_test.par

3. If we want to use the interpolation method:
add_DL3_phase_table
       --in-file ./DL3_directory/dl3_LST-1.Run0000.fits.gz
       --output ./DL3_pulsar_directory/
       --ephem model_test.par
       --interpolation

Original author: Alvaro Mas Aguilar (alvmas)
"""

import argparse

from ptiming_ana import setup_logging
from ptiming_ana.cphase._cli import add_boolean_argument, find_files
from ptiming_ana.cphase.pulsarphase_cal import DL3_calphase


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add the pulsar phases and barycentric times to DL3 files"
    )
    parser.add_argument(
        "--dir",
        "-d",
        action="store",
        type=str,
        dest="directory",
        default=None,
        help="Directory where to find the DL3 files",
    )
    parser.add_argument(
        "--in-file",
        "-f",
        action="store",
        type=str,
        dest="in_file",
        default=None,
        help="DL3 standard file if want to analyze only one file",
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
    parser.add_argument(
        "--output",
        "-out",
        action="store",
        type=str,
        dest="dir_output",
        required=True,
        help="Path where to store the DL3 file with the pulsar info",
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
        "--number-interpolation",
        "-ninterp",
        action="store",
        type=int,
        dest="ninterp",
        default=1000,
        help="Number of events between two interpolation points.",
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
        "--create-tim",
        "-tim",
        action="store_true",
        dest="create_tim",
        help="Set True to create and save a .tim file",
    )
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    setup_logging()

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
        DL3_calphase(
            file,
            args.ephem,
            args.dir_output,
            create_tim_file=args.create_tim,
            obs=args.observatory,
            use_interpolation=args.interpolation,
            n_interp=args.ninterp,
            pickle=args.pickle,
        )


if __name__ == "__main__":
    main()
