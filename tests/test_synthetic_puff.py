import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from synthetic_puff import gaussian_puff_station_concentrations  # noqa: E402


class SyntheticPuffTests(unittest.TestCase):
    def test_zero_wind_source_station_accumulates_positive_signal(self):
        result = gaussian_puff_station_concentrations(
            station_x_m=[0.0],
            station_y_m=[0.0],
            observation_times_h=[0.0, 0.5, 1.0],
            wind_times_h=[0.0, 1.0],
            wind_u_mps=[0.0, 0.0],
            wind_v_mps=[0.0, 0.0],
            source_x_m=0.0,
            source_y_m=0.0,
            q_at_time=lambda time: np.ones_like(time),
            diffusivity_m2s=0.0,
            initial_sigma_m=100.0,
            pre_event_hours=0.0,
            integration_dt_h=0.01,
        )
        self.assertGreater(result[1, 0], result[0, 0])
        self.assertGreater(result[2, 0], result[1, 0])

    def test_positive_x_wind_favors_downwind_station(self):
        result = gaussian_puff_station_concentrations(
            station_x_m=[900.0, -900.0],
            station_y_m=[0.0, 0.0],
            observation_times_h=[0.0, 1.0],
            wind_times_h=[0.0, 1.0],
            wind_u_mps=[0.5, 0.5],
            wind_v_mps=[0.0, 0.0],
            source_x_m=0.0,
            source_y_m=0.0,
            q_at_time=lambda time: np.ones_like(time),
            diffusivity_m2s=1.0,
            initial_sigma_m=80.0,
            pre_event_hours=0.0,
            integration_dt_h=0.01,
        )
        self.assertGreater(result[1, 0], 100.0 * result[1, 1])

    def test_negative_source_strength_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            gaussian_puff_station_concentrations(
                station_x_m=[0.0],
                station_y_m=[0.0],
                observation_times_h=[0.0],
                wind_times_h=[0.0],
                wind_u_mps=[0.0],
                wind_v_mps=[0.0],
                source_x_m=0.0,
                source_y_m=0.0,
                q_at_time=lambda time: -np.ones_like(time),
                diffusivity_m2s=0.0,
                initial_sigma_m=10.0,
            )


if __name__ == "__main__":
    unittest.main()
