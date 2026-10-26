from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper_comparison_v4 as comparison
from paper_joint_baselines import JointBounds


class ComparisonProtocolTests(unittest.TestCase):
    def test_original_frozen_case_set_has_no_duplicate_or_winner_selection(self):
        cases = comparison.input_cases()
        self.assertEqual(len(cases), 216)
        self.assertEqual(len({spec['case_id'] for _, spec in cases}), 216)
        self.assertEqual(sum(parent == 'evaluation_v3_shared_forward' for parent, _ in cases), 120)

    def test_adam_projection_enforces_same_diffusion_and_amplitude_box(self):
        model, _, _ = comparison.formal.parcel.configure(dict(wind='steady'), [[.3, -.1]])
        model.double()
        bounds = JointBounds(np.array([-.75, -.75, .002, 0., 0.]),
                             np.array([.75, .75, .03, .02, 1.]))
        a = torch.nn.Parameter(torch.tensor([10.], dtype=torch.float64))
        comparison.formal.base.ia.set_model_anisotropic_diffusion_norm(model, .2, .1)
        original = torch.optim.Adam
        with comparison.projected_diffusion_box(model, bounds, coefficient_scale=2.):
            optimizer = torch.optim.Adam([model.xs, model.ys, model.rawD_perp,
                                           model.rawD_parallel_delta, a], lr=.01)
            optimizer.step()
        self.assertIs(torch.optim.Adam, original)
        dp, dq = comparison.formal.base.ia.model_anisotropic_diffusion_norm(model)
        self.assertLessEqual(float(dq.detach()), .03+1e-12)
        self.assertLessEqual(float((dp-dq).detach()), .02+1e-12)
        self.assertLessEqual(float(a.detach()), .5)


if __name__ == '__main__':
    unittest.main()
