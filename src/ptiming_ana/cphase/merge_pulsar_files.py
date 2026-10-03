"""
Script that merges the DL2 files (one per subrun) of a run, with the pulsar
phases already computed, into a single file.

Usage:
------------------------
merge_pulsar_files
       --dir ./DL2_pulsar_directory/
       --run-number 0000
       --output ./DL2_merged_directory/

Original author: Alvaro Mas Aguilar (alvmas)
"""

import argparse

from ptiming_ana import setup_logging
from ptiming_ana.cphase._cli import add_boolean_argument
from ptiming_ana.cphase.utils import merge_dl2_pulsar


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge the DL2 pulsar files of the subruns of a run"
    )
    parser.add_argument(
        "--dir",
        "-d",
        action="store",
        type=str,
        dest="directory",
        required=True,
        help="Directory where to find the DL2 files",
    )
    parser.add_argument(
        "--output",
        "-out",
        action="store",
        type=str,
        dest="dir_output",
        required=True,
        help="Directory where to store the merged file",
    )
    add_boolean_argument(
        parser,
        "--srcdep",
        "-srcdep",
        dest="src_dep",
        help="Include the source-dependent parameters",
    )
    parser.add_argument(
        "--run-number",
        "-r",
        action="store",
        type=str,
        dest="run",
        required=True,
        help="Run number to merge",
    )
    return parser


def main(args=None):
    args = get_parser().parse_args(args)
    setup_logging()

    merge_dl2_pulsar(args.directory, args.run, args.dir_output, args.src_dep)


if __name__ == "__main__":
    main()
