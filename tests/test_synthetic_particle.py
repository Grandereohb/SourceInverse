import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from synthetic_particle import lagrangian_particle_station_concentrations


class SyntheticParticleTests(unittest.TestCase):
    def _run(self, **overrides):
        kwargs = dict(
            station_x_m=[0.0, 500.0], station_y_m=[0.0, 0.0],
            observation_times_h=[0.0, 1.0, 2.0], wind_times_h=[0.0, 1.0, 2.0],
            wind_u_mps=[1.0, 1.0, 1.0], wind_v_mps=[0.0, 0.0, 0.0],
            source_x_m=-500.0, source_y_m=0.0, q_at_time=lambda t: np.ones_like(t),
            diffusivity_along_m2s=2.0, diffusivity_cross_m2s=2.0,
            initial_sigma_along_m=80.0, initial_sigma_cross_m=80.0,
            sensor_kernel_sigma_m=50.0, wind_factor=0.2, decay_per_hour=0.1,
            release_dt_h=0.25, particles_per_release=16, random_seed=7,
        )
        kwargs.update(overrides)
        return lagrangian_particle_station_concentrations(**kwargs)

    def test_is_deterministic_nonnegative_and_has_expected_shape(self):
        first = self._run()
        second = self._run()
        self.assertEqual(first.shape, (3, 2))
        self.assertTrue(np.all(first >= 0.0))
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_anisotropy_and_meander_change_receptor_signal(self):
        reference = self._run()
        stressed = self._run(
            diffusivity_along_m2s=12.0, diffusivity_cross_m2s=0.5,
            initial_sigma_cross_m=30.0, meander_amplitude_m=300.0,
        )
        self.assertGreater(float(np.max(np.abs(reference - stressed))), 1e-12)


if __name__ == "__main__":
    unittest.main()
