import unittest
from ptiming_ana.phaseogram import PulsarAnalysis
from astropy.io import fits
import pandas as pd
import numpy as np


class TutorialTest(unittest.TestCase):
    def test_create_analiys_object(self):
        h = PulsarAnalysis()
        self.assertIsInstance(h, PulsarAnalysis)

    def test_load_config_file(self):
        h_config_from_file = PulsarAnalysis()
        h_config_from_file.set_config("tests/files/config_test.yaml")

        h = PulsarAnalysis()
        h.setBackgroundLimits([0.52, 0.87])
        h.setPeaklimits(
            P1_limits=[0, 0.026, 0.983, 1], P2_limits=[0.377, 0.422], P3_limits=None
        )
        h.setBinning(50, xmin=0, xmax=1)
        h.setTimeInterval(tint=3600 * 24)
        h.setFittingParams(model="dgaussian", binned=True)
        h.setEnergybinning(
            np.geomspace(0.1 / 1e3, 1 / 1e3, 3), do_diff=True, do_integral=False
        )  # In TeV
        h.setFermiInputFile("tutorial/example_data/merged2_pulsar.fits")
        f = fits.open("tutorial/example_data/merged2_pulsar.fits")
        fits_table = f[1].data
        times = np.sort(fits_table["BARYCENTRIC_TIME"].byteswap().newbyteorder())
        phases = fits_table["PULSE_PHASE"].byteswap().newbyteorder()
        energies = fits_table["ENERGY"].byteswap().newbyteorder()
        h.setListsInput(phases, times, energies / 1e6, tel="fermi", energy_units="TeV")
        h.get_results = False

        self.assertEquals(h_config_from_file, h)

    def test_run_analysis_phaseogram(self):
        h = PulsarAnalysis()
        h.set_config("tests/files/config_test.yaml")
        h.run()
        results = h.show_Presults()

        expected_result = (
            pd.read_pickle("tests/files/peak_statistics_tutorial.pkl"),
            pd.read_pickle("tests/files/periodicity_statistics_tutorial.pkl"),
        )

        pd.testing.assert_frame_equal(results[0], expected_result[0])
        pd.testing.assert_frame_equal(results[1], expected_result[1])

    def test_run_analysis_fitting(self):
        h = PulsarAnalysis()
        h.set_config("tests/files/config_test.yaml")
        h.run()

        self.assertEquals(h.fit_model, "dgaussian")
        results = h.show_fit_results()
        expected_result = pd.read_pickle("tests/files/fit_result_tutorial.pkl")
        pd.testing.assert_frame_equal(results, expected_result)


if __name__ == "__main__":
    unittest.main()
