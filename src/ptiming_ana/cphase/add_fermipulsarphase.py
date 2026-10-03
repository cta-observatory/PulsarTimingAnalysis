"""
Script that adds the pulsar phases and barycentric times to Fermi-LAT event
files (FT1). A new file ``{name}_pulsar.fits`` with the columns
``BARYCENTRIC_TIME`` and ``PULSE_PHASE`` is written in the output directory.

Usage:
------------------------
add_fermi_phase_table
       --in-file ./fermi_events.fits
       --ft2 ./spacecraft.fits
       --ephem model_test.par
       --output ./fermi_pulsar_directory/

Original author: Alvaro Mas Aguilar (alvmas)
"""

import argparse

import pandas as pd

from ptiming_ana import setup_logging
from ptiming_ana.cphase._cli import add_boolean_argument, find_files
from ptiming_ana.cphase.pulsarphase_cal import fermi_calphase


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add the pulsar phases and barycentric times to Fermi-LAT files"
    )
    parser.add_argument(
        "--dir",
        "-d",
        action="store",
        type=str,
        dest="directory",
        default=None,
        help="Directory where to find the Fermi-LAT event files",
    )
    parser.add_argument(
        "--in-file",
        "--in_file",
        "-f",
        action="store",
        type=str,
        dest="in_file",
        default=None,
        help="Fermi-LAT event file if want to analyze only one file",
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
        help="Path where to store the file with the pulsar info",
    )
    add_boolean_argument(
        parser,
        "--pickle",
        "-pickle",
        dest="pickle",
        help="Save a pickle file with the loaded TOAs",
    )
    parser.add_argument(
        "--ft2",
        "-ft2",
        action="store",
        type=str,
        dest="ft2_file",
        required=True,
        help="Fermi-LAT spacecraft file (FT2)",
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
        filelist = find_files(args.directory)
    elif args.in_file is not None:
        filelist = [args.in_file]
    else:
        parser.error("No input file or directory given")

    for file in filelist:
        # Calculate the phases
        fermi_calphase(file, args.ephem, args.dir_output, args.pickle, args.ft2_file)


if __name__ == "__main__":
    main()
