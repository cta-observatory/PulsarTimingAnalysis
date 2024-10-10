import unittest
from ptiming_ana.phaseogram import PulsarAnalysis
from astropy.io import fits
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

        self.assertEqual(h_config_from_file, h)

    def test_run_analysis_phaseogram(self):
        h = PulsarAnalysis()
        h.set_config("tests/files/config_test.yaml")
        h.run()
        results = h.show_Presults()

        peak_stadistics_data_expec = np.array(
            [
                [108.5913594, 58.00656799, 110.16571886, 14.3384333],
                [5959.23714286, 2677.27142857, 8636.50857143, 1126.74285714],
                [84.22028603, 62.20823982, 105.93665554, 81.25025086],
                [6969.0, 3734.0, 10703.0, 4884.0],
                [1009.76285714, 1056.72857143, 2066.49142857, 3757.25714286],
                [451.84108843, 241.36129211, 458.39188858, 59.661223],
                [187.53441252, 82.35892672, 189.98591475, 18.38185577],
            ]
        )

        periodicity_statistics_data_expec = np.array(
            [
                [26977.43375174, 25992.17369182, 28101.4067069],
                [0.0, 0.0, 0.0],
                [np.inf, np.inf, np.inf],
            ]
        )

        np.testing.assert_almost_equal(
            results[0].to_numpy(), peak_stadistics_data_expec, decimal=5
        )
        np.testing.assert_almost_equal(
            results[1].to_numpy(), periodicity_statistics_data_expec, decimal=5
        )

    def test_run_analysis_fitting(self):
        h = PulsarAnalysis()
        h.set_config("tests/files/config_test.yaml")
        h.run()

        self.assertEqual(h.fit_model, "dgaussian")
        results = h.show_fit_results()

        fit_result_expected = np.array(
            [
                [0.9924233191795764, 0.0015660661377770266],
                [0.026304542387059753, 0.0014698372893777872],
                [1.3763409922947634, 0.004402295450743174],
                [0.0509859742803803, 0.004192067589437329],
                [466.4375, 0.0],
                [200.23728941446313, 10.903033719301062],
                [145.13336608138428, 11.04167277916415],
            ]
        )

        np.testing.assert_almost_equal(
            results.to_numpy()[:, 1:], fit_result_expected, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
