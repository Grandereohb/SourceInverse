import sys
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pinn_source'))
import field
import ia_vpdi
from variable_projection import build_release_layout_family, build_q_basis, build_response_matrix, solve_nonnegative_q


class ReleaseLayoutFamilyTests(unittest.TestCase):
    def family_case(self, device="cpu"):
        from test_ia_vpdi import synthetic_case
        model, _, _, _, _, stations = synthetic_case(device=device)
        event = torch.linspace(0., 1.5, 9, device=device)
        field.configure_recurrent_context(model, -1., 1., -1., 1., event,
            [.15]*9, [.04]*9, 0., .002, anisotropic_diffusion=True,
            d_parallel_min_norm=.0002, d_perp_min_norm=.0001,
            diffusion_solver="tensor_semigroup", source_decay_integrator="exponential_linear",
            nx=17, ny=17)
        ia_vpdi.set_model_anisotropic_diffusion_norm(model, .004, .002)
        family = build_release_layout_family(model.recurrent_integration_times, event)
        xyt = torch.cat((stations.repeat(len(event), 1), event.repeat_interleave(len(stations))[:, None]), dim=1)
        basis = family.layouts[-1].basis
        coefficients = torch.tensor([0., 0., .3, 1.5, .1, 0.], device=device)
        observations = build_response_matrix(model, .08, xyt, basis) @ coefficients
        return model, family, xyt, observations

    def test_candidate_family_reprofiles_q_and_restores_model(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            model, family, xyt, observations = self.family_case(device)
            saved = [p.detach().clone() for p in (model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta)]
            with patch("ia_vpdi.build_response_matrix", wraps=ia_vpdi.build_response_matrix) as calls:
                candidates = ia_vpdi.score_release_layouts(model, .08, xyt, observations, family,
                    .1, -.1, .004, .002)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(sum(c.forward_count for c in candidates), family.response_basis.shape[1])
            for c, layout in zip(candidates, family.layouts):
                direct = ia_vpdi.score_all_data_candidate(model, .08, xyt, observations,
                    layout.basis, .1, -.1, .004, .002)
                np.testing.assert_allclose(c.prediction, direct.prediction, rtol=2e-5, atol=1e-7)
                self.assertEqual(c.layout_id, layout.layout_id)
                self.assertEqual(c.q_knots, layout.knots)
            self.assertLess(min(c.profile_score for c in candidates), candidates[0].profile_score * .01)
            for p, old in zip((model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta), saved):
                torch.testing.assert_close(p, old, rtol=0., atol=0.)

    def test_search_refinement_exports_selected_layout_and_direct_replay(self):
        model, family, xyt, observations = self.family_case()
        result = ia_vpdi.run_ia_vpdi(model, .08, xyt, observations,
            (.1, .1), (-.1, -.1), .2, q_basis_count=5, top_k=1,
            diffusion_scale_seeds=(1.,), diffusion_ratio_seeds=(2.,), outer_steps=2,
            q_layout_strategy="local_templates")
        self.assertEqual(len(result.coarse_candidates), len(family.layouts))
        self.assertEqual(len(result.refinement_runs), 1)
        self.assertEqual(result.q_knots, result.best.q_knots)
        np.testing.assert_allclose(result.q_basis @ result.best.coefficients, result.best.q_integration)
        replay = ia_vpdi.score_all_data_candidate(model, .08, xyt, observations, result.q_basis,
            result.best.source_x, result.best.source_y,
            result.best.diffusion_parallel_norm, result.best.diffusion_perp_norm)
        np.testing.assert_allclose(replay.prediction, result.best.prediction, rtol=0., atol=0.)
        expected = (sum(c.forward_count for c in result.coarse_candidates)
            + sum(c.forward_count for c in result.final_layout_candidates)
            + sum(r.forward_count for r in result.refinement_runs) + result.q_basis.shape[1])
        self.assertEqual(result.forward_count, expected)
        self.assertEqual(result.inner_solve_count, len(result.coarse_candidates)
            + len(result.final_layout_candidates) + sum(r.inner_solve_count for r in result.refinement_runs) + 1)
        with tempfile.TemporaryDirectory() as directory:
            exported = ia_vpdi.build_ia_vpdi_diagnostics(result)
            report = exported['report']
            self.assertEqual(report['selected_layout_id'], result.best.layout_id)
            self.assertEqual(report['q_knots'], list(result.q_knots))
            self.assertEqual(report['q_layout_strategy'], 'local_templates')
            self.assertIn('conditional', report['layout_uncertainty_note'])
            self.assertIn('final_layout_reprofile', set(exported['tables']['ia_vpdi_candidates'].stage))

    def test_search_rejects_ambiguous_layout_configuration(self):
        model, _, xyt, observations = self.family_case()
        for kwargs in ({'q_layout_strategy': 'unknown'},
                       {'q_layout_strategy': 'local_templates', 'q_knots': [0., .75, 1.5]}):
            with self.assertRaises(ValueError):
                ia_vpdi.run_ia_vpdi(model, .08, xyt, observations, (-.1, .1), (-.1, .1), .2, **kwargs)

    def test_identity_and_union_represent_every_layout_and_keep_baseline(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            event = torch.linspace(0., 1., 13, device=device)
            for count in (13, 49):
                clock = torch.linspace(0., 1., count, device=device)
                family = build_release_layout_family(clock, event)
                self.assertEqual(len(family.layouts), 10)
                self.assertLessEqual(family.response_basis.shape[1], count)
                torch.testing.assert_close(family.layouts[0].basis, build_q_basis(clock, 5), rtol=0., atol=0.)
                for layout in family.layouts:
                    self.assertIn(layout.basis.shape[1], (5, 6))
                    self.assertTrue((layout.basis >= 0).all() & (layout.mapping >= 0).all())
                    torch.testing.assert_close(layout.basis.sum(1), torch.ones_like(clock))
                    torch.testing.assert_close(family.response_basis @ layout.mapping, layout.basis,
                                               rtol=3e-6, atol=2e-7)

    def test_actual_response_reuse_gradients_and_inner_q_recovery(self):
        from test_ia_vpdi import synthetic_case
        for steps in (1, 3):
            model, _, _, _, _, stations = synthetic_case()
            event = torch.linspace(0., 1.5, 6)
            with patch.multiple(field, RECURRENT_SUBSTEPS=steps, RECURRENT_ADAPTIVE_SUBSTEPS=False):
                field.configure_recurrent_context(model, -1., 1., -1., 1., event,
                    [.15]*6, [.04]*6, 0., .002, anisotropic_diffusion=True,
                    d_parallel_min_norm=.0002, d_perp_min_norm=.0001,
                    diffusion_solver='tensor_semigroup', source_decay_integrator='exponential_linear', nx=17, ny=17)
            family = build_release_layout_family(model.recurrent_integration_times, event)
            xyt = torch.cat((stations.repeat(len(event), 1), event.repeat_interleave(len(stations))[:, None]), dim=1)
            shared = build_response_matrix(model, .08, xyt, family.response_basis, track_gradients=True)
            parameters = (model.xs, model.ys, model.rawD_perp, model.rawD_parallel_delta)
            for layout in family.layouts:
                mapped = shared @ layout.mapping
                actual = build_response_matrix(model, .08, xyt, layout.basis, track_gradients=True)
                torch.testing.assert_close(mapped, actual, rtol=2e-5, atol=1e-7)
                first = torch.autograd.grad(mapped.square().sum(), parameters, retain_graph=True)
                second = torch.autograd.grad(actual.square().sum(), parameters)
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, rtol=3e-5, atol=1e-7)
                coefficients = torch.linspace(.2, 1., actual.shape[1])
                y = (actual.detach() @ coefficients).numpy()
                fitted = solve_nonnegative_q(mapped, y)
                self.assertTrue(fitted.success)
                np.testing.assert_allclose(fitted.coefficients, coefficients, rtol=3e-4, atol=1e-5)

    def test_layouts_do_not_depend_on_residuals_or_fitted_release(self):
        times = np.linspace(0., 1., 13)
        first = build_release_layout_family(times, times)
        second = build_release_layout_family(times.copy(), times.copy())
        self.assertEqual([x.knots for x in first.layouts], [x.knots for x in second.layouts])
        with self.assertRaises(TypeError):
            build_release_layout_family(times, times, observations=np.ones(13))

    def test_bad_clocks_and_counts_are_rejected(self):
        good = np.linspace(0., 1., 13)
        for event in (good[:5], good[::-1], [0., 0., .2, .3, .4, 1.], [0., .2, .3, .4, .5, np.nan], np.linspace(.1, 1., 13)):
            with self.assertRaises(ValueError):
                build_release_layout_family(good, event)
        for count in (0, 7, 2.5):
            with self.assertRaises(ValueError):
                build_release_layout_family(good, good, count)


if __name__ == '__main__':
    unittest.main()
