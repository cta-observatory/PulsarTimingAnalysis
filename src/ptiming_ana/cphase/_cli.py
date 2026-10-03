"""Helpers shared by the command-line tools of ptiming_ana.cphase."""

import argparse
import os

__all__ = ["add_boolean_argument", "find_files", "str_to_bool"]


def str_to_bool(value):
    """Convert the usual string representations of a boolean to bool."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def add_boolean_argument(parser, *flags, dest, help):
    """
    Add a boolean option that can be used as a flag (``--pickle``) or with an
    explicit value (``--pickle True`` / ``--pickle False``).
    """
    parser.add_argument(
        *flags,
        dest=dest,
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=help,
    )


def find_files(directory, run=None):
    """
    Sorted list of the files in ``directory``. If ``run`` is given, only the
    files whose name contains the run number are returned.
    """
    filelist = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and (run is None or str(run) in name)
    ]
    filelist.sort()

    if len(filelist) == 0:
        selection = "" if run is None else f" for run number {run}"
        raise FileNotFoundError(f"No files found in {directory}{selection}")

    return filelist
