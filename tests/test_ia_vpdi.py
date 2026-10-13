import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from config import INVERSION_MODE  # noqa: E402
from field import configure_recurrent_context  # noqa: E402
from ia_vpdi import (  # noqa: E402
    IAVPDIResult,
    compute_profile_information,
    build_ia_vpdi_diagnostics,
    sampled_profile_stability,
    model_anisotropic_diffusion_norm,
    recommend_sensor_locations,
    refine_profiled_candidate,
    run_ia_vpdi,
    score_all_data_candidate,
    select_refinement_starts,
    set_model_anisotropic_diffusion_norm,
)
from models.pinn import PINN  # noqa: E402
from pipeline import run  # noqa: E402
from variable_projection import build_q_basis, build_response_matrix, solve_nonnegative_q  # noqa: E402


def synthetic_case(single_station=False, device="cpu"):
    model = PINN(hidden=8, depth=1).to(device)
    configure_recurrent_context(
        model, -1., 1., -1., 1., [0., .5, 1., 1.5],
        [.15, .12, .08, .08], [0., .04, .07, .07],
        0., .002, anisotropic_diffusion=True,
        d_parallel_min_norm=.0002, d_perp_min_norm=.0001,
        decay_norm=.1, nx=17, ny=17,
    )
    set_model_anisotropic_diffusion_norm(model, .004, .002)
    with torch.no_grad():
        model.xs.fill_(.1)
        model.ys.fill_(-.1)
    stations = torch.tensor([[.45, 0.], [.2, .5], [-.4, .3], [-.55, -.4], [.55, -.55]], device=device)
    if single_station:
        stations = stations[:1]
    times = model.recurrent_times
    xyt = torch.cat([
        stations.repeat(times.numel(), 1),
        times.repeat_interleave(stations.shape[0])[:, None],
    ], dim=1)
    basis = build_q_basis(model.recurrent_integration_times, 3)
    coefficients = np.array([.5, 1.8, .7])
    response = build_response_matrix(model, .08, xyt, basis)
    observations = (response @ torch.tensor(coefficients, dtype=response.dtype, device=device)).detach()
    return model, xyt, basis, coefficients, observations, stations


class IAVPDITests(unittest.TestCase):
    def test_diverse_starts_keep_best_and_avoid_duplicate_seeds(self):
        from types import SimpleNamespace
        candidates = [SimpleNamespace(source_x=x, source_y=0., profile_score=i,
                                     solver_success=True) for i, x in enumerate([0., 0., .1, 1., 2.])]
        starts = select_refinement_starts(candidates, 3, "spatial_diverse", .9)
        self.assertEqual([c.source_x for c in starts], [0., 1., 2.])
        self.assertIs(starts[0], candidates[0])
        self.assertEqual(select_refinement_starts(candidates, 3), candidates[:3])
        self.assertEqual(select_refinement_starts(candidates, 3, "spatial_diverse", 10.),
                         [candidates[0], candidates[2], candidates[3]])
        self.assertEqual(select_refinement_starts(candidates[:2], 3, "spatial_diverse", 1.), candidates[:2])
        invalid = SimpleNamespace(source_x=np.nan, source_y=0., profile_score=-1., solver_success=True)
        self.assertEqual(select_refinement_starts([invalid] + candidates, 3), candidates[:3])

    def test_start_selection_validates_configuration(self):
        for kwargs in ({"top_k": 0}, {"top_k": 1.5}, {"strategy": "unknown"},
                       {"min_distance": -1.}, {"min_distance": np.inf}):
            with self.assertRaises(ValueError):
                select_refinement_starts([], **kwargs)

    def test_bound_applies_to_search_refinement_and_restored_state(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        unbounded = score_all_data_candidate(model, .08, xyt, observations, basis,
                                             .1, -.1, .004, .002)
        refined = refine_profiled_candidate(
            model, .08, xyt, observations, basis, unbounded, (-.8, .8), (-.8, .8),
            steps=2, q_upper_bound=.2, optimize_diffusion=False,
        )
        self.assertTrue(refined.success)
        self.assertLessEqual(refined.best.q_integration.max(), .2 + 1e-8)
        self.assertAlmostEqual(refined.best.diffusion_parallel_norm, .004, places=6)
        fresh = score_all_data_candidate(model, .08, xyt, observations, basis,
            refined.best.source_x, refined.best.source_y, .004, .002, q_upper_bound=.2)
        np.testing.assert_allclose(fresh.prediction, refined.best.prediction, atol=1e-8)
        result = run_ia_vpdi(model, .08, xyt, observations, (-.2, .2), (-.2, .2), .2,
            q_basis_count=3, top_k=1, diffusion_scale_seeds=(1.,),
            diffusion_ratio_seeds=(2.,), outer_steps=1, q_upper_bound=.2)
        self.assertTrue(all(c.q_integration.max() <= .2 + 1e-8 for c in result.coarse_candidates))
        self.assertLessEqual(result.best.q_integration.max(), .2 + 1e-8)

    def test_information_does_not_claim_ellipse_at_active_amplitude_bound(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        info = compute_profile_information(model, .08, xyt, observations, basis,
                                           coefficients, q_upper_bound=1.8)
        self.assertIsNone(info.ellipse_area_m2)
        self.assertIsNone(info.covariance_source_m2)
        self.assertTrue(any("upper bound is active" in w for w in info.warnings))

    def test_config_supports_current_modes_and_old_mode_is_rejected(self):
        self.assertIn(INVERSION_MODE, ("joint_adam", "ia_vpdi"))
        with self.assertRaisesRegex(ValueError, "joint_adam.*ia_vpdi"):
            run("missing", "missing", "missing", inversion_mode="profiled_crossfit")

    def test_anisotropic_setter_is_positive_and_ordered(self):
        model, *_ = synthetic_case()
        actual_parallel, actual_perp = set_model_anisotropic_diffusion_norm(model, .006, .0025)
        self.assertGreater(actual_perp, 0.)
        self.assertGreaterEqual(actual_parallel, actual_perp)
        self.assertAlmostEqual(actual_parallel, .006, places=6)
        self.assertAlmostEqual(actual_perp, .0025, places=6)

    def test_isotropic_diffusion_is_a_valid_boundary(self):
        model, *_ = synthetic_case()
        # Equal floors permit isotropy as the parallel delta approaches zero.
        model.recurrent_d_parallel_min_norm = model.recurrent_d_perp_min_norm
        parallel, perp = set_model_anisotropic_diffusion_norm(model, .004, .004)
        self.assertAlmostEqual(parallel, perp, places=7)

    def test_candidate_uses_all_observations_and_true_source_wins(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        true = score_all_data_candidate(model, .08, xyt, observations, basis, .1, -.1, .004, .002)
        wrong = score_all_data_candidate(model, .08, xyt, observations, basis, -.65, .65, .004, .002)
        self.assertEqual(true.prediction.size, observations.numel())
        self.assertLess(true.profile_score, 1e-10)
        self.assertGreater(wrong.profile_score, true.profile_score + 1e-7)

    def test_torch_outer_objective_matches_nnls_weighting(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        weights = np.linspace(.5, 2., observations.numel())
        response = build_response_matrix(model, .08, xyt, basis, track_gradients=True)
        solution = solve_nonnegative_q(response.detach(), observations, weights)
        coeff = torch.tensor(solution.coefficients, dtype=response.dtype)
        loss = torch.mean((torch.tensor(weights, dtype=response.dtype) * (response @ coeff - observations)) ** 2)
        self.assertAlmostEqual(float(loss.detach()), solution.weighted_sse / observations.numel(), places=7)

    def test_envelope_gradient_matches_centered_finite_difference(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        with torch.no_grad():
            model.xs.fill_(.02)
        response = build_response_matrix(model, .08, xyt, basis, track_gradients=True)
        solution = solve_nonnegative_q(response.detach(), observations)
        loss = torch.mean((response @ torch.tensor(solution.coefficients, dtype=response.dtype) - observations) ** 2)
        gradient = float(torch.autograd.grad(loss, model.xs)[0])
        eps = 2e-3
        d_parallel, d_perp = [float(v.detach()) for v in model_anisotropic_diffusion_norm(model)]
        plus = score_all_data_candidate(model, .08, xyt, observations, basis, .02 + eps, -.1, d_parallel, d_perp).profile_score
        minus = score_all_data_candidate(model, .08, xyt, observations, basis, .02 - eps, -.1, d_parallel, d_perp).profile_score
        finite = (plus - minus) / (2 * eps)
        self.assertAlmostEqual(gradient, finite, delta=max(2e-3, abs(finite) * .15))

    def test_continuous_refinement_lowers_profile_objective_and_location_error(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        start = score_all_data_candidate(model, .08, xyt, observations, basis, -.25, .25, .004, .002)
        run_result = refine_profiled_candidate(
            model, .08, xyt, observations, basis, start,
            (-.8, .8), (-.8, .8), steps=35, lr=.03, patience=12, run_id=1,
        )
        self.assertTrue(run_result.success)
        self.assertLess(run_result.best.profile_score, start.profile_score)
        start_error = np.hypot(start.source_x - .1, start.source_y + .1)
        final_error = np.hypot(run_result.best.source_x - .1, run_result.best.source_y + .1)
        self.assertLess(final_error, start_error)
        best = run_result.best
        fresh = score_all_data_candidate(
            model, .08, xyt, observations, basis, best.source_x, best.source_y,
            best.diffusion_parallel_norm, best.diffusion_perp_norm,
        )
        self.assertAlmostEqual(best.profile_score, fresh.profile_score, delta=1e-14)
        np.testing.assert_allclose(best.prediction, fresh.prediction, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(best.coefficients, fresh.coefficients, rtol=1e-7, atol=1e-9)
        self.assertAlmostEqual(float(model.xs.detach()), best.source_x, places=7)
        self.assertAlmostEqual(float(model.ys.detach()), best.source_y, places=7)

    def test_final_adam_update_is_scored(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        start = score_all_data_candidate(model, .08, xyt, observations, basis, -.25, .25, .004, .002)
        refined = refine_profiled_candidate(
            model, .08, xyt, observations, basis, start,
            (-.8, .8), (-.8, .8), steps=1, lr=.03,
        )
        self.assertTrue(refined.success)
        self.assertLess(refined.best.profile_score, start.profile_score)
        self.assertAlmostEqual(refined.trajectory[0]["source_x"], start.source_x, places=7)
        self.assertEqual(refined.forward_count, 3 * basis.shape[1])

    def test_zero_steps_does_not_move_source(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        start = score_all_data_candidate(model, .08, xyt, observations, basis, -.25, .25, .004, .002)
        refined = refine_profiled_candidate(
            model, .08, xyt, observations, basis, start, (-.8, .8), (-.8, .8), steps=0,
        )
        self.assertTrue(refined.success)
        self.assertEqual(refined.steps, 0)
        self.assertAlmostEqual(refined.best.source_x, start.source_x, places=7)

    def test_multistart_runs_and_selects_lowest_result(self):
        model, xyt, _, _, observations, _ = synthetic_case()
        result = run_ia_vpdi(
            model, .08, xyt, observations, (-.4, .4), (-.4, .4), .4,
            q_basis_count=3, top_k=3, diffusion_scale_seeds=(1.,),
            diffusion_ratio_seeds=(2.,), outer_steps=3, outer_patience=2,
        )
        self.assertEqual(len(result.refinement_runs), 3)
        self.assertAlmostEqual(
            result.best.profile_score,
            min(run.best.profile_score for run in result.refinement_runs if run.success),
        )

    def test_diverse_search_fixed_budget_and_diagnostics(self):
        model, xyt, _, _, observations, _ = synthetic_case()
        result = run_ia_vpdi(
            model, .08, xyt, observations, (-.4, .4), (-.4, .4), .4,
            q_basis_count=3, top_k=3, diffusion_scale_seeds=(1.,),
            diffusion_ratio_seeds=(1., 2.), outer_steps=1,
            start_strategy="spatial_diverse", start_min_distance=.4,
        )
        self.assertEqual(len(result.refinement_runs), 3)
        self.assertEqual(result.refinement_runs[0].start, result.coarse_candidates[0])
        starts = [(r.start.source_x, r.start.source_y) for r in result.refinement_runs]
        self.assertEqual(len(set(starts)), 3)
        self.assertTrue(all(np.hypot(x - u, y - v) >= .4
                            for i, (x, y) in enumerate(starts) for u, v in starts[:i]))
        self.assertEqual(result.forward_count, sum(c.forward_count for c in result.coarse_candidates) +
                         sum(r.forward_count for r in result.refinement_runs))
        with tempfile.TemporaryDirectory() as directory:
            report = build_ia_vpdi_diagnostics(result)["report"]
            self.assertEqual(report["multi_start_stability"]["start_strategy"], "spatial_diverse")
            self.assertEqual(report["multi_start_stability"]["initial_sources"], [list(p) for p in starts])

    def test_single_refinement_failure_does_not_abort_multistart(self):
        model, xyt, _, _, observations, _ = synthetic_case()
        import ia_vpdi
        original = ia_vpdi.refine_profiled_candidate
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                start = args[5]
                return ia_vpdi.IAVPDIRefinementRun(1, start, start, False, 0, "synthetic failure", [])
            return original(*args, **kwargs)

        with patch("ia_vpdi.refine_profiled_candidate", side_effect=flaky):
            result = run_ia_vpdi(
                model, .08, xyt, observations, (-.3, .3), (-.3, .3), .3,
                q_basis_count=3, top_k=2, diffusion_scale_seeds=(1.,),
                diffusion_ratio_seeds=(2.,), outer_steps=2,
            )
        self.assertEqual(len(result.refinement_runs), 2)
        self.assertTrue(any(run.success for run in result.refinement_runs))

    def test_information_is_symmetric_and_has_finite_ellipse(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        observations = observations + torch.tensor(
            np.random.default_rng(12).normal(0., 1e-4, observations.numel()), dtype=observations.dtype,
        )
        info = compute_profile_information(
            model, .08, xyt, observations, basis, coefficients,
            length_m=1000., source_bounds=((-1., 1.), (-1., 1.)),
        )
        self.assertEqual(info.profiled_source_information.shape, (2, 2))
        np.testing.assert_allclose(info.profiled_source_information, info.profiled_source_information.T, atol=1e-10)
        self.assertIn(info.status, {"identifiable", "weakly_identifiable"})
        self.assertTrue(np.isfinite(info.ellipse_area_m2))
        self.assertGreater(info.ellipse_area_m2, 0.)
        self.assertGreater(np.linalg.eigvalsh(info.profiled_source_information).min(), 0.)

    def test_information_rank_and_covariance_are_weight_scale_invariant(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        observations = observations + .0001
        base = compute_profile_information(model, .08, xyt, observations, basis, coefficients)
        for scale in (1e-5, 1e5):
            scaled = compute_profile_information(
                model, .08, xyt, observations, basis, coefficients,
                weights=np.full(observations.numel(), scale),
            )
            self.assertEqual(scaled.rank, base.rank)
            self.assertEqual(scaled.status, base.status)
            np.testing.assert_allclose(scaled.profiled_source_information / scale ** 2,
                                       base.profiled_source_information, rtol=1e-8, atol=1e-14)
            np.testing.assert_allclose(scaled.covariance_source_norm,
                                       base.covariance_source_norm, rtol=1e-8)

    def test_full_rank_does_not_imply_practical_precision(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        observations = observations + .0001
        base = compute_profile_information(model, .08, xyt, observations, basis, coefficients, length_m=1000.)
        limit = base.ellipse_major_m * .5
        info = compute_profile_information(model, .08, xyt, observations, basis, coefficients,
                                           length_m=1000., localization_radius_limit_m=limit)
        self.assertEqual(info.rank, 2)
        self.assertEqual(info.structural_status, base.structural_status)
        self.assertEqual(info.status, 'weakly_identifiable')
        self.assertEqual(info.precision_assessment, 'exceeds_reference_scale')
        self.assertAlmostEqual(info.ellipse_to_reference_ratio, 2.)
        np.testing.assert_allclose(info.covariance_source_m2, base.covariance_source_m2)

    def test_information_reference_uses_physical_domain_diagonal(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        info = compute_profile_information(model, .08, xyt, observations + .0001, basis, coefficients,
            length_m=1000., source_bounds=((-1., 1.), (-1., 1.)))
        self.assertAlmostEqual(info.localization_reference_m, np.sqrt(8.) * 1000.)
        self.assertAlmostEqual(info.ellipse_to_reference_ratio, info.ellipse_major_m / info.localization_reference_m)

    def test_information_rejects_invalid_precision_scales(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        for kwargs in ({'length_m': 0.}, {'length_m': float('nan')},
                       {'localization_radius_limit_m': -1.}, {'localization_radius_limit_m': float('inf')},
                       {'source_bounds': ((1., -1.), (-1., 1.))}):
            with self.assertRaises(ValueError):
                compute_profile_information(model, .08, xyt, observations, basis, coefficients, **kwargs)

    def test_projected_information_matches_direct_lstsq_reference(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        import ia_vpdi
        response = build_response_matrix(model, .08, xyt, basis, track_gradients=True)
        prediction = response @ torch.tensor(coefficients, dtype=response.dtype)
        jacobian = ia_vpdi._theta_jacobian_numpy(model, prediction, 1e-10)
        nuisance = np.concatenate([response.detach().numpy(), jacobian[:, 2:]], axis=1)
        projected = jacobian[:, :2] - nuisance @ np.linalg.lstsq(nuisance, jacobian[:, :2], rcond=1e-10)[0]
        info = compute_profile_information(model, .08, xyt, observations, basis, coefficients)
        np.testing.assert_allclose(info.profiled_source_information, projected.T @ projected, rtol=1e-7, atol=1e-13)

    def test_noiseless_ellipse_has_explicit_warning(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        info = compute_profile_information(model, .08, xyt, observations, basis, coefficients)
        self.assertTrue(any("zero estimated noise" in warning for warning in info.warnings))
        self.assertNotEqual(info.status, 'identifiable')
        self.assertEqual(info.precision_assessment, 'noise_not_established')

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_information_cuda_inputs_are_finite(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case(device="cuda")
        info = compute_profile_information(
            model, .08, xyt, observations + .0001, basis,
            torch.as_tensor(coefficients, device="cuda"),
            weights=torch.ones_like(observations),
        )
        self.assertTrue(np.isfinite(info.profiled_source_information).all())
        self.assertGreater(info.ellipse_area_m2, 0.)

    def test_single_station_reports_degenerate_information(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case(single_station=True)
        info = compute_profile_information(model, .08, xyt, observations, basis, coefficients)
        self.assertIn(info.status, {"rank_deficient", "weakly_identifiable", "boundary_sensitive"})

    def test_sensor_recommendation_does_not_increase_area_or_overlap(self):
        model, xyt, basis, coefficients, observations, stations = synthetic_case()
        observations = observations + .0001
        info = compute_profile_information(model, .08, xyt, observations, basis, coefficients, length_m=1000.)
        candidates = ((-.7, -.7), (.7, .7), (.45, 0.))
        recommendations = recommend_sensor_locations(
            model, .08, candidates, model.recurrent_integration_times,
            basis, coefficients, info, existing_station_xy=[tuple(row.tolist()) for row in stations],
            min_distance_norm=.05, top_k=3, length_m=1000.,
        )
        self.assertTrue(recommendations)
        self.assertGreater(info.ellipse_area_m2, 0.)
        self.assertTrue(all(item.ellipse_area_m2 > 0. for item in recommendations))
        self.assertTrue(any(item.area_reduction_fraction > 0. for item in recommendations))
        # Float32 propagation can perturb the ill-conditioned Gram reconstruction.
        self.assertTrue(all(item.ellipse_area_m2 <= info.ellipse_area_m2 * (1 + 1e-6) for item in recommendations))
        self.assertNotIn((.45, 0.), {(item.source_x, item.source_y) for item in recommendations})
        reversed_recommendations = recommend_sensor_locations(
            model, .08, tuple(reversed(candidates)), model.recurrent_integration_times,
            basis, coefficients, info, existing_station_xy=[tuple(row.tolist()) for row in stations],
            min_distance_norm=.05, top_k=3, length_m=1000.,
        )
        self.assertEqual([(r.source_x, r.source_y) for r in recommendations],
                         [(r.source_x, r.source_y) for r in reversed_recommendations])
        self.assertGreaterEqual(recommendations[0].logdet_gain, recommendations[-1].logdet_gain)

    def test_diagnostics_are_collected_without_duplicate_files(self):
        model, xyt, basis, coefficients, observations, _ = synthetic_case()
        best = score_all_data_candidate(model, .08, xyt, observations, basis, .1, -.1, .004, .002)
        result = IAVPDIResult(best, (best,), (), np.asarray(basis), "hat", None, (), 0., 3)
        with tempfile.TemporaryDirectory() as directory:
            exports = build_ia_vpdi_diagnostics(result)
            self.assertTrue(exports["tables"]["ia_vpdi_sensors"].empty)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(set(exports), {"tables", "report"})
            report = json.loads(json.dumps(exports["report"], allow_nan=False))
            self.assertEqual(report["mode"], "ia_vpdi")
            self.assertIn("timing", report)
            self.assertIn("sampled_profile_stability", report)

    def test_sampled_stability_reuses_candidates_without_forward_or_selection(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        template = score_all_data_candidate(model, .08, xyt, observations, basis, .1, -.1, .004, .002)
        best = replace(template, source_x=0., source_y=0., profile_score=1., q_integration=np.ones(4))
        neighbor = replace(best, source_x=.1, profile_score=1.04, q_integration=np.ones(4)*20.)
        duplicate = replace(neighbor, profile_score=1.1, q_integration=np.ones(4)*100.)
        failed = replace(best, source_x=.2, solver_success=False)
        result = IAVPDIResult(best, (best, neighbor, duplicate, failed), (), np.asarray(basis), "hat", None, (), 0., 99)
        with patch("ia_vpdi.build_response_matrix", side_effect=AssertionError("Report must not run forward")):
            report = sampled_profile_stability(result, length_m=10000., source_bounds=((0., 1.), (-1., 1.)))
            with tempfile.TemporaryDirectory() as directory:
                exports = build_ia_vpdi_diagnostics(result, length_m=10000., source_bounds=((0., 1.), (-1., 1.)))
                self.assertIn("ia_vpdi_stability", exports["tables"])
                self.assertIn("effective_response_gain", exports["tables"]["ia_vpdi_candidates"])
                self.assertIn("sampled_profile_stability", exports["report"])
        self.assertIs(result.best, best)
        self.assertEqual(result.forward_count, 99)
        self.assertEqual(report["unique_location_count"], 2)
        self.assertEqual(report["excluded_record_count"], 1)
        near = report["near_equivalent_sets"][1]
        self.assertEqual(near["candidate_location_count"], 2)
        self.assertEqual(near["x_span_m"], 1000.)
        self.assertEqual(near["q_peak_ratio"], 20.)
        self.assertTrue(near["touches_allowed_boundary"])
        self.assertEqual(report["extra_forward_count"], 0)

    def test_sampled_stability_zero_score_missing_scale_and_validation(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        template = score_all_data_candidate(model, .08, xyt, observations, basis, .1, -.1, .004, .002)
        best = replace(template, profile_score=0., q_integration=np.zeros(4))
        other = replace(best, source_x=.2, profile_score=1e-10)
        result = IAVPDIResult(best, (other,), (), np.asarray(basis), "hat", None, (), 0., 3)
        report = sampled_profile_stability(result)
        near = report["near_equivalent_sets"][1]
        self.assertEqual(near["candidate_location_count"], 1)
        self.assertIsNone(near["x_span_m"])
        self.assertIsNone(near["q_peak_ratio"])
        self.assertTrue(near["contains_zero_q_peak"])
        self.assertTrue(any("single candidate" in w for w in report["warnings"]))
        for kwargs in ({"length_m": 0}, {"score_tolerances": [-.1]}, {"source_bounds": ((1., 0.), (0., 1.))}):
            with self.assertRaises(ValueError):
                sampled_profile_stability(result, **kwargs)

    def test_sampled_stability_counts_location_once_but_retains_nuisance_q_range(self):
        model, xyt, basis, _, observations, _ = synthetic_case()
        template = score_all_data_candidate(model, .08, xyt, observations, basis, .1, -.1, .004, .002)
        best = replace(template, profile_score=1., q_integration=np.ones(4))
        other = replace(best, profile_score=1.02, q_integration=np.ones(4)*100., diffusion_parallel_norm=.008)
        result = IAVPDIResult(best, (best, other), (), np.asarray(basis), "hat", None, (), 0., 3)
        report = sampled_profile_stability(result)
        near = report["near_equivalent_sets"][1]
        self.assertEqual(report["evaluated_record_count"], 2)
        self.assertEqual(near["candidate_location_count"], 1)
        self.assertEqual(near["evaluated_fit_count"], 2)
        self.assertEqual(near["q_peak_ratio"], 100.)


if __name__ == "__main__":
    unittest.main()
