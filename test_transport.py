"""Independent numerical and analytical checks of the transport candidates."""
import unittest
import numpy as np
from scipy.stats import wasserstein_distance

from transport_metrics import cost_matrices, distances, exact_transport, pool_mass


class TransportTests(unittest.TestCase):
    def test_analytical_point_masses(self):
        a, b = np.zeros((8, 8)), np.zeros((8, 8))
        a[2, 2], b[2, 6] = 1, 1
        mask = np.zeros((8, 8), bool)
        mask[:, :4] = True
        result = distances(a, b, mask, grid=8)
        expected = .5 / np.sqrt(2)
        self.assertAlmostEqual(result["spatial_w1"], expected)
        self.assertAlmostEqual(result["BAT"], (expected+1)/2)
        self.assertEqual(result["foreground_mass_change"], -1)

    def test_matches_independent_1d_solver(self):
        rng = np.random.default_rng(37)
        p, q = rng.random(12), rng.random(12)
        p, q = p/p.sum(), q/q.sum()
        x = np.linspace(0, 1, 12)
        value, _ = exact_transport(p, q, abs(x[:, None]-x[None, :]), return_plan=True)
        self.assertAlmostEqual(value, wasserstein_distance(x, x, p, q), places=10)

    def test_symmetry_identity_triangle_and_bounds(self):
        rng = np.random.default_rng(41)
        a, b, c = rng.random((3, 8, 8))
        mask = rng.random((8, 8)) > .5
        ab, ba = distances(a, b, mask, 8), distances(b, a, mask, 8)
        ac, bc, aa = distances(a, c, mask, 8), distances(b, c, mask, 8), distances(a, a, mask, 8)
        for name in ["spatial_w1", "BAT"]:
            self.assertAlmostEqual(aa[name], 0, places=10)
            self.assertAlmostEqual(ab[name], ba[name], places=10)
            self.assertTrue(0 <= ab[name] <= 1)
            self.assertLessEqual(ac[name], ab[name]+bc[name]+1e-10)

    def test_pooling_preserves_mass(self):
        a = np.arange(64, dtype=float).reshape(8, 8)
        self.assertEqual(pool_mass(a, 4).sum(), a.sum())
        with self.assertRaises(ValueError):
            pool_mass(a, 3)

    def test_same_region_cost_has_no_boundary_penalty(self):
        a, b = np.eye(8), np.fliplr(np.eye(8))
        for mask in [np.zeros((8, 8), bool), np.ones((8, 8), bool)]:
            r = distances(a, b, mask, 8)
            self.assertAlmostEqual(r["BAT"], r["spatial_w1"]/2)

    def test_invalid_maps_and_masks_rejected(self):
        for a in [np.zeros((8, 8)), -np.ones((8, 8)), np.full((8, 8), np.nan)]:
            with self.assertRaises(ValueError):
                distances(a, np.eye(8), np.eye(8), 8)
        with self.assertRaises(ValueError):
            distances(np.eye(8), np.eye(8), np.full((8, 8), .5), 8)


if __name__ == "__main__":
    unittest.main()
