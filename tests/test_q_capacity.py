import unittest

import numpy as np
import torch

from tests.test_ia_vpdi import synthetic_case
from ia_vpdi import run_ia_vpdi, build_ia_vpdi_diagnostics
from q_capacity import build_capacity_family
from variable_projection import build_response_matrix


class ProductionCapacityTests(unittest.TestCase):
    def test_shared_span_is_exact_and_nonnegative(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            family = build_capacity_family(torch.linspace(0., 6., 19, device=device))
            for layout in family.layouts:
                self.assertEqual(layout.basis.device.type, device)
                self.assertTrue(bool((layout.basis >= 0).all()))
                torch.testing.assert_close(layout.basis.sum(1), torch.ones(19, device=device))
                torch.testing.assert_close(family.response_basis @ layout.mapping, layout.basis)
        with self.assertRaises(ValueError):
            build_capacity_family(torch.linspace(0., 6., 19), (1, 4, 5))

    def test_production_smoke_and_conditional_stability(self):
        torch.set_num_threads(1)
        model, xyt, _, _, _, _ = synthetic_case()
        family = build_capacity_family(model.recurrent_integration_times, (1, 2, 3))
        observed = build_response_matrix(model, .08, xyt, family.layouts[0].basis).detach().reshape(-1)
        observed = observed + 1e-5*torch.cos(torch.arange(observed.numel()))
        result = run_ia_vpdi(model, .08, xyt, observed, (-.2, .2), (-.2, .2), .2,
            q_basis_count=3, top_k=1, diffusion_scale_seeds=(1.,),
            diffusion_ratio_seeds=(2.,), outer_steps=25, outer_solver='trf',
            q_capacity_selection='bic', q_capacity_counts=(1, 2, 3))
        report = build_ia_vpdi_diagnostics(result)['report']
        self.assertEqual(report['q_capacity_selection']['mode'], 'bic')
        models = report['q_capacity_selection']['models']
        chosen = min(models, key=lambda m: m['best_bic'])['q_basis_count']
        self.assertEqual(report['q_capacity_selection']['selected_count'], chosen)
        self.assertEqual(result.q_basis.shape[1], chosen)
        self.assertEqual(len(result.refinement_runs), 3)
        self.assertEqual(sum(c.forward_count for c in result.coarse_candidates), 9*3)
        self.assertEqual(len(result.coarse_candidates), 9*3)
        replay = build_response_matrix(model, .08, xyt, result.q_basis).detach().numpy()@result.best.coefficients
        np.testing.assert_allclose(replay, result.best.prediction, rtol=1e-5, atol=1e-7)
        self.assertEqual(report['sampled_profile_stability']['conditioned_on_q_layout'],
                         result.best.layout_id)
        self.assertGreater(report['sampled_profile_stability']['unique_location_count'], 0)


if __name__ == '__main__':
    unittest.main()
