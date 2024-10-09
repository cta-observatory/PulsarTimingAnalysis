#####################
###Author: Alvaro Mas Aguilar (alvmas)
# mail: alvmas@ucm.es
# Using modules from PINT-pulsar and lstchain to calculate phases and add them to the input files.
###################3


import pandas as pd
import argparse
import os
from .pulsarphase_cal import DL2_calphase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", "-d", action="store", type=str, dest="directory", default=None
    )
    parser.add_argument(
        "--in-file", "-f", action="store", type=str, dest="in_file", default=None
    )
    parser.add_argument(
        "--ephem", "-ephem", action="store", type=str, dest="ephem", default=None
    )
    parser.add_argument(
        "--pickle", "-pickle", action="store", type=bool, dest="pickle", default=False
    )
    parser.add_argument(
        "--run-number", "-r", action="store", type=str, dest="run", default=False
    )

    args = parser.parse_args()

    ephem = args.ephem
    pickle = args.pickle
    in_file = args.in_file
    run = args.run

    pd.set_option("display.precision", 10)

    if ephem is None:
        raise ValueError("No ephemeris provided")

    if args.directory is not None:
        if in_file is not None:
            raise ValueError(
                "Both directory and file were given, can only use one of them"
            )

        filelist = []
        for x in os.listdir(args.directory):
            rel_dir = os.path.relpath(args.directory)
            rel_file = os.path.join(rel_dir, x)
            if run in rel_file:
                filelist.append(rel_file)

        filelist.sort()
        for i in range(0, len(filelist)):
            # Calculate the phases
            DL2_calphase(filelist[i], ephem, pickle)

    else:
        if in_file is not None:
            # Calculate the phases
            DL2_calphase(in_file, ephem, pickle)
        else:
            raise ValueError("No input file or directory given")


if __name__ == "__main__":
    main()
