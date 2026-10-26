import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper_experiments as experiment


class PaperExperimentTests(unittest.TestCase):
    def test_hourly_mean_order_and_gradient(self):
        response = torch.arange(24., requires_grad=True)
        result = experiment.hourly_mean_response(response.reshape(12, 2), 2, 3, 2)
        torch.testing.assert_close(result, torch.tensor([[4., 5.], [6., 7.], [16., 17.], [18., 19.]]))
        result.sum().backward()
        torch.testing.assert_close(response.grad, torch.full_like(response, 1 / 3))
        with self.assertRaises(ValueError):
            experiment.hourly_mean_response(torch.zeros(4, 2))

    def test_design_is_complete_and_predeclared(self):
        cases = [experiment.case_definition(i) for i in range(12)]
        self.assertEqual(len({(c['wind'], c['network'], c['release']) for c in cases}), 12)
        with self.assertRaises(ValueError):
            experiment.case_definition(12)

    def test_zero_field_recovery_linearity_and_context_restoration(self):
        torch.set_num_threads(1)
        original = experiment.field.RECURRENT_INITIAL_RELEASE_FRACTION
        with experiment.experiment_context():
            model, xyt, basis, a, observed, clean, _, _ = experiment.make_case(0, grid=16, noise_fraction=0)
            response = experiment.averaged_response(model, experiment.SIGMA, xyt, basis)
            solution = experiment.vp.solve_nonnegative_q(response, clean)
            self.assertTrue(solution.success)
            np.testing.assert_allclose(solution.coefficients, a, rtol=1e-3, atol=1e-5)
            np.testing.assert_allclose(observed, clean)
            q = basis @ torch.as_tensor(a, dtype=basis.dtype)
            fields = experiment.field.recurrent_plume_fields_from_q(model, experiment.SIGMA, q)
            self.assertEqual(float(fields[0].detach().abs().max()), 0.0)
            direct = experiment.field.recurrent_plume_value_from_q(model, xyt, experiment.SIGMA, q)
            mean = experiment.hourly_mean_response(direct.reshape(-1, 1)).flatten()
            np.testing.assert_allclose(mean.detach().numpy(), clean, rtol=2e-5, atol=1e-6)
        self.assertEqual(experiment.field.RECURRENT_INITIAL_RELEASE_FRACTION, original)

    def test_all_pilot_noiseless_recovery_before_batch_execution(self):
        torch.set_num_threads(1)
        with experiment.experiment_context():
            for index in range(12):
                with self.subTest(index=index):
                    model, xyt, basis, a, _, clean, _, _ = experiment.make_case(index)
                    response = experiment.averaged_response(model, experiment.SIGMA, xyt, basis)
                    solution = experiment.vp.solve_nonnegative_q(response, clean)
                    self.assertTrue(solution.success)
                    self.assertLess(np.linalg.norm(solution.coefficients-a) / np.linalg.norm(a), 1e-3)
                    self.assertLess(np.linalg.norm(solution.prediction-clean) / np.linalg.norm(clean), 1e-4)


if __name__ == '__main__':
    unittest.main()
