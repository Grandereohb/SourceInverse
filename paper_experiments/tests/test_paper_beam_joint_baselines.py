from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import JointBounds
import run_paper_controlled_release as physical


class PhysicalJointTests(unittest.TestCase):
    def test_external_physical_parameters_and_forward_jacobian(self):
        torch.set_num_threads(1)
        event = physical.ROOT/'controlled_release_preparation_v2/source2_rel06'
        model, xyt, basis, response, observed, _ = physical.configure(.3, minutes=2,
            dtype=torch.float64, q_count=1, time_order=4, release_order=4,
            beam_solver='analytic', data_dir=event)
        theta = np.array([.55, .55, .018, .042, .0005])
        bounds = JointBounds(np.array([0., 0., .00100005, 0., 0.]),
                             np.array([1.1, 1.1, 1., 1., .01]))
        obj = BeamJointResidual(response, model, 0., xyt, basis, observed, bounds)
        np.testing.assert_allclose(obj.residual(theta),
            response(model, 0., xyt, basis).numpy().ravel()*theta[-1]-observed,
            rtol=1e-10, atol=1e-10)
        jac = obj.jacobian(theta)
        for j in range(len(theta)):
            delta = np.zeros_like(theta)
            delta[j] = 1e-7 if j == 4 else 1e-6
            numerical = (obj.residual(theta+delta)-obj.residual(theta-delta))/(2*delta[j])
            np.testing.assert_allclose(jac[:, j], numerical, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
