import unittest
import numpy as np
from metrics import apply_minmax, fit_minmax, heatmap_metrics


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.h = np.zeros((8, 8))
        self.h[1:3, 1:3] = np.array([[.2, .4], [.6, 1.]])
        self.mask = np.indices((8, 8))[1] < 4

    def test_identical_maps_zero_difference(self):
        m = heatmap_metrics(self.h, self.h, self.mask)
        self.assertTrue(m["valid_cam"])
        for key in ["d1", "d2", "d3", "d4", "jsd", "foreground_shift"]:
            self.assertAlmostEqual(m[key], 0, places=12)

    def test_translation_and_bounds(self):
        m = heatmap_metrics(self.h, np.roll(self.h, 4, axis=1), self.mask)
        self.assertAlmostEqual(m["d1"], 4 / np.sqrt(128))
        self.assertGreater(m["d2"], 0)
        self.assertAlmostEqual(m["foreground_shift"], 1)
        self.assertAlmostEqual(m["background_increase"], 1)
        for key in ["d1", "d3", "jsd"]:
            self.assertTrue(0 <= m[key] <= 1)
        self.assertTrue(0 <= m["d4"] <= 2)

    def test_invalid_maps_explicit(self):
        for bad in [np.zeros_like(self.h), np.ones_like(self.h)]:
            m = heatmap_metrics(self.h, bad, self.mask)
            self.assertFalse(m["valid_cam"])
            self.assertTrue(np.isnan(m["d1"]))

    def test_jsd_symmetric(self):
        other = np.roll(self.h, 3, axis=1)
        self.assertAlmostEqual(heatmap_metrics(self.h, other, self.mask)["jsd"], heatmap_metrics(other, self.h, self.mask)["jsd"])

    def test_calibration_constant_and_extrapolation(self):
        x = np.array([[0., 0., 0., 1.], [1., 2., 1., 1.]])
        params = fit_minmax(x)
        values, clipped = apply_minmax([[2., -1., .5, 1.]], params)
        np.testing.assert_allclose(values, [[1., 0., .5, 0.]])
        self.assertEqual(clipped.sum(), 2)

    def test_reject_negative(self):
        with self.assertRaises(ValueError):
            heatmap_metrics(-self.h, self.h, self.mask)


if __name__ == "__main__":
    unittest.main()
