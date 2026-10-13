import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from field import configure_recurrent_context  # noqa: E402
from models.pinn import PINN  # noqa: E402
from variable_projection import (  # noqa: E402
    build_q_basis,
    build_response_matrix,
    generate_candidate_grid,
    resolve_q_basis_count,
    solve_nonnegative_q,
    build_local_release_templates,
)


def configured_model(device="cpu"):
    model = PINN(hidden=8, depth=1).to(device)
    configure_recurrent_context(
        model, -1.0, 1.0, -1.0, 1.0,
        [0.0, 0.5, 1.0, 1.5], [0.12, 0.1, 0.06, 0.06],
        [0.0, 0.03, 0.05, 0.05], 0.0, 0.002,
        anisotropic_diffusion=True, d_parallel_min_norm=0.0005,
        d_perp_min_norm=0.0002, decay_norm=0.1, nx=21, ny=21,
    )
    return model


class VariableProjectionTests(unittest.TestCase):
    def test_unbounded_profile_does_not_penalize_response_amplitude(self):
        response = np.array([[1., .1], [.2, 1.], [.5, .4], [.6, .2]])
        target = np.array([1., 3., 2., 1.])
        base = solve_nonnegative_q(response, target)
        self.assertTrue(base.success)
        for scale in (1e-6, 1e6):
            fitted = solve_nonnegative_q(response * scale, target)
            self.assertTrue(fitted.success)
            np.testing.assert_allclose(fitted.prediction, base.prediction, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(fitted.coefficients * scale, base.coefficients, rtol=1e-8, atol=1e-10)
            self.assertAlmostEqual(fitted.weighted_sse, base.weighted_sse, places=10)

    def test_clock_only_local_release_templates(self):
        times = np.linspace(0., 1., 15)
        templates = build_local_release_templates(times)
        self.assertEqual(len(templates), 10)
        self.assertEqual(len(build_local_release_templates(np.linspace(0., 1., 13))), 8)
        self.assertEqual(next(item for item in templates if item["template_id"] == 6)["retained_indices"],
                         [0, 5, 6, 7, 8, 14])
        for item in templates:
            self.assertEqual(len(item["knots"]), 6)
            basis = build_q_basis(times, 6, knots=item["knots"])
            self.assertTrue(np.all(basis >= 0))
            np.testing.assert_allclose(basis.sum(axis=1), 1.)
        for invalid in ([0., 1.], [0., 1., 2., 3., 4., np.nan], [0., 1., 2., 3., 4., 4.]):
            with self.assertRaises(ValueError):
                build_local_release_templates(invalid)

    def test_independent_time_nodes_span_sampled_coarse_responses(self):
        model = configured_model()
        times = model.recurrent_integration_times
        knots = times.detach().cpu().numpy().astype(float)
        identity = build_q_basis(times, len(knots), knots=knots)
        torch.testing.assert_close(identity, torch.eye(len(knots)))
        xyt = torch.tensor([[.1, .1, t] for t in model.recurrent_times])
        response = build_response_matrix(model, .1, xyt, identity)
        target = np.array([.1, 1., .4, .2])
        full = solve_nonnegative_q(response, target)
        for count in (5, 6, 9):
            coarse = build_q_basis(times, count)
            actual = build_response_matrix(model, .1, xyt, coarse)
            torch.testing.assert_close(response @ coarse, actual, rtol=2e-5, atol=1e-7)
            restricted = solve_nonnegative_q(actual, target)
            self.assertTrue(full.success and restricted.success)
            self.assertLessEqual(full.weighted_sse, restricted.weighted_sse + 1e-7)

    def test_single_time_query_respects_explicit_knots(self):
        knots = [0., .3, .5, 1.]
        np.testing.assert_allclose(build_q_basis(np.array([.4]), 4, knots=knots),
                                   [[0., .5, .5, 0.]], atol=1e-12)
        torch.testing.assert_close(build_q_basis(torch.tensor([.4, .4]), 4, knots=knots),
                                   torch.tensor([[0., .5, .5, 0.], [0., .5, .5, 0.]]))
        np.testing.assert_allclose(build_q_basis(np.array([1.]), 4, knots=knots), [[0., 0., 0., 1.]])
        np.testing.assert_allclose(build_q_basis(np.array([.4]), 4), [[1., 0., 0., 0.]])

    def test_hat_basis_is_nonnegative_partition_of_unity(self):
        times = torch.linspace(0.0, 8.0, 33)
        basis = build_q_basis(times, 4)
        self.assertTrue(torch.all(basis >= 0.0))
        self.assertTrue(torch.allclose(basis.sum(1), torch.ones(33)))
        self.assertGreaterEqual(resolve_q_basis_count(24), 3)
        self.assertLessEqual(resolve_q_basis_count(24), 5)

    def test_ablation_counts_cover_same_time_horizon(self):
        times = torch.linspace(0., 1., 53)
        for count in (3, 5, 7):
            basis = build_q_basis(times, count)
            self.assertEqual(basis.shape, (53, count))
            self.assertTrue(torch.all(basis >= 0.))
            torch.testing.assert_close(basis.sum(1), torch.ones_like(times))
            self.assertEqual(float(basis[0, 0]), 1.)
            self.assertEqual(float(basis[-1, -1]), 1.)

    def test_nnls_recovers_known_coefficients(self):
        response = np.array([[1., 0., .2], [.1, 1., 0.], [0., .3, 1.], [.8, .4, .1], [.2, .7, .6]])
        expected = np.array([1.5, .4, 2.])
        result = solve_nonnegative_q(response, response @ expected)
        np.testing.assert_allclose(result.coefficients, expected, atol=1e-8)
        self.assertLess(result.weighted_sse, 1e-15)

    def test_nonuniform_hat_basis_and_uniform_backward_compatibility(self):
        times = torch.linspace(0., 12., 25)
        basis = build_q_basis(times, 5, knots=[0., 5., 6., 7., 12.])
        self.assertTrue(torch.all(basis >= 0.))
        torch.testing.assert_close(basis.sum(1), torch.ones_like(times))
        self.assertEqual(float(basis[12, 2]), 1.)
        torch.testing.assert_close(build_q_basis(times, 5),
                                  build_q_basis(times, 5, knots=[0., 3., 6., 9., 12.]))
        for knots in ([0., 6., 6., 7., 12.], [1., 5., 6., 7., 12.], [0., 5., 6., 7., 11.]):
            with self.assertRaises(ValueError):
                build_q_basis(times, 5, knots=knots)

    def test_bounded_coefficients_bound_partition_of_unity_q(self):
        response = np.eye(3)
        result = solve_nonnegative_q(response, [1., 20., 3.], coefficient_upper_bound=5.)
        np.testing.assert_allclose(result.coefficients, [1., 5., 3.], atol=1e-7)
        basis = build_q_basis(np.linspace(0., 2., 9), 3)
        q = basis @ result.coefficients
        self.assertTrue(np.all(q >= 0.))
        self.assertLessEqual(float(q.max()), 5. + 1e-8)
        for upper in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                solve_nonnegative_q(response, [1., 20., 3.], coefficient_upper_bound=upper)

    def test_response_matrix_matches_direct_linear_combination(self):
        model = configured_model()
        times = model.recurrent_times
        xyt = torch.column_stack([torch.full_like(times, .4), torch.zeros_like(times), times])
        basis = build_q_basis(model.recurrent_integration_times, 3)
        response = build_response_matrix(model, .08, xyt, basis)
        coefficients = torch.tensor([.4, 1.2, .7])
        from field import recurrent_plume_value_from_q
        direct = recurrent_plume_value_from_q(model, xyt, .08, basis @ coefficients).reshape(-1)
        self.assertTrue(torch.allclose(response @ coefficients, direct, rtol=2e-5, atol=2e-6))

    def test_weight_semantics_match_weighted_residual(self):
        response = np.array([[1., 0.], [0., 1.], [1., 1.]])
        observations = np.array([1., 2., 2.5])
        weights = np.array([1., 3., .5])
        result = solve_nonnegative_q(response, observations, weights)
        expected = np.mean((weights * result.residual) ** 2)
        self.assertAlmostEqual(result.weighted_sse / observations.size, expected)

    def test_candidate_grid_is_inclusive(self):
        self.assertEqual(len(generate_candidate_grid((-.5, .5), (-.5, .5), .5)), 9)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_response_to_cpu_solver(self):
        model = configured_model("cuda")
        times = model.recurrent_times
        xyt = torch.column_stack([torch.full_like(times, .4), torch.zeros_like(times), times])
        basis = build_q_basis(model.recurrent_integration_times, 2)
        response = build_response_matrix(model, .08, xyt, basis)
        result = solve_nonnegative_q(response, response @ torch.tensor([1., .5], device="cuda"))
        self.assertEqual(response.device.type, "cuda")
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
