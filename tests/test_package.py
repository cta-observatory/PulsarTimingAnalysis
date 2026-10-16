import importlib
import logging
import re

import pytest

import ptiming_ana

MODULES = [
    "ptiming_ana",
    "ptiming_ana.cphase",
    "ptiming_ana.cphase.pulsarphase_cal",
    "ptiming_ana.cphase.utils",
    "ptiming_ana.io",
    "ptiming_ana.io.dl2",
    "ptiming_ana.phaseogram",
    "ptiming_ana.phaseogram.filter_object",
    "ptiming_ana.phaseogram.lightcurve",
    "ptiming_ana.phaseogram.models",
    "ptiming_ana.phaseogram.penergy_analysis",
    "ptiming_ana.phaseogram.periodicity_test",
    "ptiming_ana.phaseogram.pfitting",
    "ptiming_ana.phaseogram.phase_regions",
    "ptiming_ana.phaseogram.phasebinning",
    "ptiming_ana.phaseogram.ptime_analysis",
    "ptiming_ana.phaseogram.pulsar_analysis",
    "ptiming_ana.phaseogram.read_events",
    "ptiming_ana.spectral",
    "ptiming_ana.spectral.config_reading",
    "ptiming_ana.spectral.gammapy_utils",
    "ptiming_ana.spectral.plot_utils",
    "ptiming_ana.spectral.spectra",
]


def test_version():
    assert isinstance(ptiming_ana.__version__, str)
    assert re.match(r"^\d+\.\d+", ptiming_ana.__version__)
    assert ptiming_ana.__version__ != "0.0.0"


@pytest.mark.parametrize("module_name", MODULES)
def test_public_api_is_importable(module_name):
    """Every name listed in __all__ must exist in the module."""
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__")
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.{name} is not defined"


def test_star_import():
    namespace = {}
    exec("from ptiming_ana import *", namespace)
    for name in ("cphase", "phaseogram", "spectral"):
        assert name in namespace


@pytest.mark.parametrize(
    "module_name",
    [
        "ptiming_ana.cphase.add_DL2_phase_table",
        "ptiming_ana.cphase.add_DL3_phase_table",
        "ptiming_ana.cphase.add_fermipulsarphase",
        "ptiming_ana.cphase.merge_pulsar_files",
    ],
)
def test_command_line_tools_are_importable(module_name):
    module = importlib.import_module(module_name)
    assert callable(module.main)

    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0


def test_import_does_not_configure_logging():
    """A library must not silence or configure the loggers of other packages."""
    assert not logging.getLogger("gammapy").disabled
    assert not logging.getLogger("matplotlib.font_manager").disabled


def test_setup_logging():
    logger = logging.getLogger("ptiming_ana")
    handlers = list(logger.handlers)
    try:
        assert ptiming_ana.setup_logging("DEBUG") is logger
        assert logger.level == logging.DEBUG
        n_handlers = len(logger.handlers)

        # Calling it again must not duplicate the messages
        ptiming_ana.setup_logging(logging.INFO)
        assert len(logger.handlers) == n_handlers
        assert logger.level == logging.INFO
    finally:
        logger.handlers = handlers
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
