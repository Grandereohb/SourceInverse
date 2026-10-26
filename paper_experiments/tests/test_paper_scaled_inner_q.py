from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import lsq_linear
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_scaled_inner_q import solve_scaled_nonnegative_q


class ScaledInnerTests(unittest.TestCase):
    def test_same_unweighted_objective_as_well_scaled_reference(self):
        rng=np.random.default_rng(12718)
        a=rng.uniform(.01,1.,(20,5))
        y=rng.normal(.3,.4,20)
        result=solve_scaled_nonnegative_q(a,y)
        expected=lsq_linear(a,y,bounds=(0.,np.inf),tol=1e-12)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.prediction,a@expected.x,rtol=1e-8,atol=1e-8)

    def test_weak_nonzero_columns_are_not_discarded(self):
        rng=np.random.default_rng(12719)
        original=rng.uniform(.01,1.,(25,3))
        scale=np.array([1e-42,1e-28,1.])
        a=np.column_stack((original*scale,np.zeros((25,2))))
        y=original@np.array([.2,.8,.4])
        result=solve_scaled_nonnegative_q(a,y)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.coefficients[:3]*scale,[.2,.8,.4],rtol=1e-11)
        np.testing.assert_allclose(result.prediction,y,rtol=1e-12)
        np.testing.assert_array_equal(result.coefficients[3:],[0.,0.])

    def test_weights_and_finite_bounds_are_preserved(self):
        a=np.array([[.5,0.],[0.,2.],[.5,2.]])
        y=np.array([1.,2.,3.])
        weights=np.array([2.,.5,0.])
        result=solve_scaled_nonnegative_q(a,y,weights,coefficient_upper_bound=.2)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.coefficients,[.2,.2],atol=1e-12)
        self.assertAlmostEqual(result.weighted_sse,float(np.sum((weights*(a@result.coefficients-y))**2)))

    def test_zero_response_and_invalid_inputs(self):
        result=solve_scaled_nonnegative_q(np.zeros((4,3)),[1.,-1.,2.,0.])
        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.coefficients,np.zeros(3))
        self.assertAlmostEqual(result.weighted_sse,6.)
        with self.assertRaises(ValueError):
            solve_scaled_nonnegative_q([[-1.]],[1.])
        with self.assertRaises(ValueError):
            solve_scaled_nonnegative_q([[1.]],[1.],weights=[-1.])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_input_uses_same_cpu_inner_problem(self):
        a=np.array([[1.,.1],[.2,1.],[.5,.3]])
        y=np.array([.8,.3,.4])
        gpu=solve_scaled_nonnegative_q(torch.tensor(a,device='cuda'),torch.tensor(y,device='cuda'))
        cpu=solve_scaled_nonnegative_q(a,y)
        np.testing.assert_allclose(gpu.coefficients,cpu.coefficients,rtol=1e-12)


if __name__=='__main__':
    unittest.main()
