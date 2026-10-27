from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_fast_parcels import CachedMeanResponse
from paper_joint_baselines import JointBounds, JointResidual
from paper_q_model_selection import fit_profiled_q_models
from run_paper_updated_evidence import physical_information, prior, control_cases
import run_paper_parcel_pilot as parcel


class UpdatedEvidenceTests(unittest.TestCase):
    def test_control_respects_fixed_supplementary_scenarios(self):
        entries = []
        for repeat in range(8):
            for network in ('ideal5', 'jjj6'):
                for wind in ('steady', 'veering'):
                    for release in ('constant', 'pulse', 'double_peak'):
                        entries.append({'case': dict(cohort='nominal', network=network,
                            wind=wind, release=release, repeat=repeat)})
        for network, wind, release in [('surround5_pulse', 'steady', 'pulse'),
                                       ('corridor6_double_peak', 'veering', 'double_peak')]:
            entries.extend({'case': dict(cohort='nominal', network=network, wind=wind,
                release=release, repeat=r)} for r in range(48))
        selected = control_cases(entries)
        self.assertEqual(len(selected), 24)
        for network in ('ideal5', 'jjj6'):
            rows = [e['case'] for e in selected if e['case']['network'] == network]
            self.assertEqual(len({(r['wind'], r['release']) for r in rows}), 6)
        for network in ('surround5_pulse', 'corridor6_double_peak'):
            rows = [e['case'] for e in selected if e['case']['network'] == network]
            self.assertEqual([r['repeat'] for r in rows], list(range(6)))

    def setup_objective(self):
        torch.set_num_threads(1)
        model, xyt, basis = parcel.configure(dict(wind='veering'), [[.3, -.1], [.5, 1.]])
        model.double()
        bounds = JointBounds(np.r_[[-.75, -.75, .00100005, 0], np.zeros(5)],
                             np.r_[[.75, .75, 1, 1], np.full(5, 10.)])
        obj = JointResidual(CachedMeanResponse(stations=2), model, .15, xyt,
                            basis, np.zeros(12), bounds)
        theta = np.r_[[.1, -.2, .02, .03], np.ones(5)]
        obj.observations = obj.tensor(obj.residual(theta)+1e-5*np.cos(np.arange(12)))
        return obj, theta

    def test_capacity_matched_joint_nls_uses_same_model_selection(self):
        obj, theta = self.setup_objective()
        fit = fit_profiled_q_models(obj, [theta], max_nfev=50, solver='joint_nls')
        self.assertEqual(fit['method'], 'joint_nls_release_bic')
        self.assertEqual(fit['q_basis_count'], 1)
        self.assertEqual(len(fit['release_models']), 3)

    def test_forward_information_matches_existing_projection_definition(self):
        obj, theta = self.setup_objective()
        predicted = obj.residual(theta)+obj.observations.numpy()
        fit = dict(source_norm=theta[:2], diffusion_norm=[.05, .02],
            q_basis_count=5, native_coefficients=theta[4:], prediction=predicted)
        fast = physical_information(obj, fit)
        ia = prior.formal.base.ia
        with patch.object(ia, 'build_response_matrix', obj.response):
            reference = asdict(ia.compute_profile_information(obj.model, obj.sigma, obj.xyt,
                obj.observations, obj.basis, theta[4:], q_upper_bound=10.))
        for key in ('full_information', 'profiled_theta_information', 'profiled_source_information'):
            np.testing.assert_allclose(fast[key], reference[key], rtol=1e-7, atol=1e-10)
        self.assertAlmostEqual(fast['sigma2'], reference['sigma2'])


if __name__ == '__main__':
    unittest.main()
