"""Logging helpers of ptiming_ana."""

import logging

__all__ = ["setup_logging"]

LOG_FORMAT = "%(asctime)2s %(levelname)-6s [%(name)3s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level=logging.INFO):
    """
    Show the log messages of ``ptiming_ana`` in the terminal or notebook.

    As recommended for libraries, ``ptiming_ana`` does not configure the
    logging system when it is imported. Call this function at the beginning of
    a script or notebook to see the progress and the results of the analysis
    (the command-line tools do it automatically).

    Parameters
    ----------
    level : int or str
        Minimum level of the messages to show (e.g. ``"INFO"`` or ``"DEBUG"``).

    Returns
    -------
    logging.Logger
        The ``ptiming_ana`` logger.
    """
    logger = logging.getLogger("ptiming_ana")
    logger.setLevel(level)

    if not any(getattr(h, "_ptiming_ana", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handler._ptiming_ana = True
        logger.addHandler(handler)
        logger.propagate = False

    return logger
