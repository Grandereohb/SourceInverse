import unittest
import numpy as np

from reduced_trust_region import ReducedResidual
from paper_linear_background import background_design


class LinearBackgroundTests(unittest.TestCase):
    def test_mixed_unit_bounds_recover_source_and_background(self):
        matrix=np.array([[1e4,1.],[2e4,1.],[1e4,1.],[0.,1.]])
        truth=np.array([.001,2.])
        problem=ReducedResidual(lambda theta,jac:matrix,matrix@truth,q_upper_bound=[.002,np.inf])
        np.testing.assert_allclose(problem.residual([0.]),0.,atol=1e-12)
        np.testing.assert_allclose(problem.solution['coefficients'],truth,rtol=1e-12)

    def test_background_rows_do_not_invent_missing_observations(self):
        pre=np.array([[2.,2.2],[2.,np.nan],[2.,2.2]])
        design=background_design((4,2),np.array([0,1,2,4,5,6,7]),pre)
        self.assertEqual(len(design['pre_values']),5)
        np.testing.assert_allclose(design['event_basis'].sum(1),1.)
        np.testing.assert_allclose(design['offsets'],[-.1,.1])
        self.assertTrue((design['calibration_basis'][:,0]==1).all())


if __name__=='__main__':
    unittest.main()
