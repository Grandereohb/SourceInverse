import sys
import unittest
from unittest.mock import patch
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pipeline import (  # noqa: E402
    _apply_source_initial_override,
    _runtime_physics_settings,
    _runtime_source_position_mode,
)


class SourceInitialOverrideTests(unittest.TestCase):
    def test_override_converts_physical_coordinates_to_normalized(self):
        x_norm, y_norm, metadata = _apply_source_initial_override(
            (300.0, -100.0),
            x0=100.0,
            y0=-200.0,
            length_m=1000.0,
            source_x_min_m=-500.0,
            source_x_max_m=500.0,
            source_y_min_m=-500.0,
            source_y_max_m=500.0,
        )
        self.assertAlmostEqual(x_norm, 0.2)
        self.assertAlmostEqual(y_norm, 0.1)
        self.assertEqual(metadata["mode"], "explicit_override")
        self.assertFalse(metadata["was_clipped"])

    def test_override_clips_to_declared_source_domain(self):
        x_norm, y_norm, metadata = _apply_source_initial_override(
            (900.0, -900.0),
            x0=0.0,
            y0=0.0,
            length_m=1000.0,
            source_x_min_m=-500.0,
            source_x_max_m=500.0,
            source_y_min_m=-400.0,
            source_y_max_m=400.0,
        )
        self.assertAlmostEqual(x_norm, 0.5)
        self.assertAlmostEqual(y_norm, -0.4)
        self.assertTrue(metadata["was_clipped"])

    def test_override_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _apply_source_initial_override(
                (float("nan"), 0.0),
                x0=0.0,
                y0=0.0,
                length_m=1.0,
                source_x_min_m=-1.0,
                source_x_max_m=1.0,
                source_y_min_m=-1.0,
                source_y_max_m=1.0,
            )

    def test_runtime_physics_overrides_are_explicit_and_validated(self):
        with patch.dict(
            "os.environ",
            {
                "PINN_WIND_SCALE": "0.35",
                "PINN_RECURRENT_DECAY": "0.08",
                "PINN_D_MIN_PHYS": "0.5",
                "PINN_SIGMA_SRC": "0.02",
            },
            clear=False,
        ):
            settings = _runtime_physics_settings()
        self.assertEqual(settings["wind_scale"], 0.35)
        self.assertEqual(settings["recurrent_decay_per_hour"], 0.08)
        self.assertEqual(settings["d_min_phys_m2s"], 0.5)
        self.assertEqual(settings["sigma_src_norm"], 0.02)

    def test_runtime_physics_rejects_negative_decay(self):
        with patch.dict("os.environ", {"PINN_RECURRENT_DECAY": "-0.1"}, clear=False):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                _runtime_physics_settings()

    def test_runtime_source_position_mode_accepts_fixed(self):
        with patch.dict(
            "os.environ", {"PINN_SOURCE_POSITION_MODE": "fixed"}, clear=False
        ):
            self.assertEqual(_runtime_source_position_mode(), "fixed")

    def test_runtime_source_position_mode_rejects_unknown_value(self):
        with patch.dict(
            "os.environ", {"PINN_SOURCE_POSITION_MODE": "profile"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "single.*fixed"):
                _runtime_source_position_mode()


if __name__ == "__main__":
    unittest.main()
