import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"pinn_source"))
from release_pulses import build_pulse_basis, build_release_aware_schedule


class ReleasePulseTests(unittest.TestCase):
    def test_release_schedule_keeps_nodes_wind_boundaries_and_pulse_area(self):
        times, indices = build_release_aware_schedule([0., 1., 2.], [.6, .75, .9], 2)
        np.testing.assert_allclose(times, [0., .5, .6, .75, .9, 1., 1.5, 2.])
        np.testing.assert_array_equal(indices, [0, 0, 0, 0, 0, 1, 1, 2])
        q = build_pulse_basis(times, [.6], [.9])[:, 0]
        self.assertAlmostEqual(np.trapezoid(q, times), .15)

    def test_release_schedule_validates_and_snaps_roundoff(self):
        original = torch.tensor([0., .5, 1.])
        times, _ = build_release_aware_schedule(original, [.5+1e-8, .75, .75], [2, 2])
        torch.testing.assert_close(times, torch.tensor([0., .25, .5, .75, 1.]))
        for t, knots, counts in (([0., 0.], [], 1), ([0., 1.], [1.1], 1),
                                 ([0., 1.], [float("nan")], 1), ([0., 1.], [], 1.5),
                                 ([0., 1., 2.], [], [2])):
            with self.assertRaises(ValueError):
                build_release_aware_schedule(t, knots, counts)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_release_schedule_cuda_dtype_and_indices(self):
        times, indices = build_release_aware_schedule(torch.tensor([0., 1., 2.], device="cuda"), [.75], 2)
        self.assertEqual(times.device.type, "cuda")
        self.assertEqual(times.dtype, torch.float32)
        self.assertEqual(indices.dtype, torch.long)
        torch.testing.assert_close(times.cpu(), torch.tensor([0., .5, .75, 1., 1.5, 2.]))

    def test_short_pulse_sampling_can_miss_release_without_renormalization(self):
        coarse = np.arange(0., 14.01, .5)
        fine = np.arange(0., 14.01, .25)
        missed = build_pulse_basis(coarse, [5.5], [6.])[:, 0]
        resolved = build_pulse_basis(fine, [5.5], [6.])[:, 0]
        self.assertEqual(float(np.trapezoid(missed, coarse)), 0.)
        self.assertAlmostEqual(float(np.trapezoid(resolved, fine)), .25)
        # A broader release on the same time grid retains its intended area.
        broad = build_pulse_basis(coarse, [5.25], [6.25])[:, 0]
        self.assertAlmostEqual(float(np.trapezoid(broad, coarse)), .5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_pulse_basis_cuda_matches_cpu(self):
        times = torch.linspace(0., 2., 17)
        expected = build_pulse_basis(times, [.5], [1.5])
        actual = build_pulse_basis(times.cuda(), [.5], [1.5])
        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual.cpu(), expected)

    def test_triangle_height_support_area_dtype_and_device(self):
        times = torch.linspace(0., 2., 201, dtype=torch.float64)
        basis = build_pulse_basis(times, [.5], [1.5])
        self.assertEqual(basis.dtype, times.dtype)
        self.assertEqual(basis.device, times.device)
        self.assertEqual(float(basis.max()), 1.)
        self.assertTrue(torch.all(basis[times <= .5] == 0.))
        self.assertAlmostEqual(float(torch.trapezoid(basis[:, 0], times)), .5)
        with self.assertRaises(ValueError):
            build_pulse_basis(times, [1.], [1.])



if __name__ == "__main__":
    unittest.main()
