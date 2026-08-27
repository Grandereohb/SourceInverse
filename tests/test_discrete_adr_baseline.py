import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from discrete_adr_baseline import (
    _piecewise_linear_basis,
    discrete_adr_station_basis,
    search_discrete_adr_profiles,
)


class DiscreteAdrBaselineTests(unittest.TestCase):
    def test_piecewise_linear_basis_is_partition_of_unity(self):
        times = np.linspace(-1.0, 3.0, 17)
        basis = _piecewise_linear_basis(times, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(basis >= 0.0))

    def test_positive_x_wind_favors_downwind_station(self):
        prediction, audit = discrete_adr_station_basis(
            station_x_m=[500.0, -500.0],
            station_y_m=[0.0, 0.0],
            observation_times_h=[0.0, 1.0, 2.0],
            wind_times_h=[0.0, 1.0, 2.0],
            wind_u_mps=[1.0, 1.0, 1.0],
            wind_v_mps=[0.0, 0.0, 0.0],
            source_x_m=0.0,
            source_y_m=0.0,
            q_node_times_h=[0.0, 2.0],
            diffusivity_m2s=1.0,
            initial_sigma_m=100.0,
            decay_per_hour=0.0,
            wind_factor=0.25,
            transport_bounds_m=[-2000.0, 2000.0, -1500.0, 1500.0],
            grid_nx=40,
            grid_ny=32,
            pre_event_hours=0.0,
        )
        self.assertGreater(float(prediction[-1, 0].sum()), float(prediction[-1, 1].sum()))
        self.assertAlmostEqual(audit["source_kernel_mass"], 1.0, places=12)

    def test_profile_search_recovers_grid_aligned_source(self):
        shared = dict(
            station_x_m=[-800.0, 0.0, 800.0, -800.0, 0.0, 800.0],
            station_y_m=[-600.0, -600.0, -600.0, 600.0, 600.0, 600.0],
            observation_times_h=[0.0, 1.0, 2.0, 3.0],
            wind_times_h=[0.0, 1.0, 2.0, 3.0],
            wind_u_mps=[0.5, 0.5, -0.3, -0.3],
            wind_v_mps=[0.2, 0.2, 0.4, 0.4],
            diffusivity_m2s=2.0,
            initial_sigma_m=180.0,
            decay_per_hour=0.1,
            wind_factor=0.25,
            transport_bounds_m=[-2500.0, 2500.0, -2200.0, 2200.0],
            grid_nx=28,
            grid_ny=28,
            pre_event_hours=0.5,
        )
        basis, _ = discrete_adr_station_basis(
            **shared,
            source_x_m=0.0,
            source_y_m=0.0,
            q_node_times_h=[0.0, 1.5, 3.0],
        )
        target = basis @ np.array([1.0, 2.0, 0.5])
        results = search_discrete_adr_profiles(
            **{key: value for key, value in shared.items() if key not in {"grid_nx", "grid_ny"}},
            target_anomaly=target,
            source_bounds_m=[-800.0, 800.0, -600.0, 600.0],
            q_node_count=3,
            grid_size=5,
            refinement_levels=1,
            transport_grid_nx=28,
            transport_grid_ny=28,
        )
        dynamic = results[1]
        self.assertAlmostEqual(dynamic["source_x_m"], 0.0, places=12)
        self.assertAlmostEqual(dynamic["source_y_m"], 0.0, places=12)
        self.assertLess(dynamic["anomaly_rmse"], 1e-12)


if __name__ == "__main__":
    unittest.main()
