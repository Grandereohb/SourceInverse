import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from gaussian_puff_baseline import (
    profile_nonnegative_coefficients,
    profile_nonnegative_scale,
)


class GaussianPuffBaselineTests(unittest.TestCase):
    def test_profile_scale_recovers_known_amplitude(self):
        unit = np.array([[1.0, 2.0], [3.0, 4.0]])
        amplitude, rmse = profile_nonnegative_scale(unit, 7.5 * unit)
        self.assertAlmostEqual(amplitude, 7.5)
        self.assertAlmostEqual(rmse, 0.0)

    def test_profile_scale_enforces_nonnegative_amplitude(self):
        amplitude, rmse = profile_nonnegative_scale([1.0, 2.0], [-1.0, -2.0])
        self.assertEqual(amplitude, 0.0)
        self.assertGreater(rmse, 0.0)

    def test_search_can_return_auditable_candidate_surface(self):
        from gaussian_puff_baseline import search_constant_q_gaussian_puff

        result = search_constant_q_gaussian_puff(
            station_x_m=[0.0, 100.0],
            station_y_m=[0.0, 0.0],
            observation_times_h=[0.0, 1.0],
            wind_times_h=[0.0, 1.0],
            wind_u_mps=[1.0, 1.0],
            wind_v_mps=[0.0, 0.0],
            target_anomaly=np.ones((2, 2)),
            source_bounds_m=(-100.0, 100.0, -100.0, 100.0),
            diffusivity_m2s=2.0,
            initial_sigma_m=100.0,
            decay_per_hour=0.0,
            wind_factor=0.25,
            grid_size=3,
            refinement_levels=1,
            return_candidates=True,
        )
        self.assertEqual(len(result["candidates"]), 9)
        self.assertEqual(
            result["anomaly_rmse"],
            min(row["anomaly_rmse"] for row in result["candidates"]),
        )

    def test_nonnegative_coefficients_recover_linear_combination(self):
        design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        truth = np.array([2.0, 3.0])
        coefficients, rmse = profile_nonnegative_coefficients(design, design @ truth)
        np.testing.assert_allclose(coefficients, truth, rtol=1e-8, atol=1e-8)
        self.assertAlmostEqual(rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
