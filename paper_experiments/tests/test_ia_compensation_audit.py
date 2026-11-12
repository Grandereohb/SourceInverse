import unittest
import numpy as np
from audit_ia_compensation import surviving_source_fraction


class CompensationAuditTests(unittest.TestCase):
    def test_source_direction_fully_explained_by_q(self):
        matrix=np.column_stack((np.ones(8),np.arange(8)))
        derivative=np.zeros((8,2,4))
        derivative[:,0,0]=matrix[:,0]
        derivative[:,0,1]=matrix[:,1]
        fraction=surviving_source_fraction(matrix,derivative,np.array([1.,0.]),np.array([True,True]))
        self.assertLess(fraction,1e-12)

    def test_orthogonal_source_directions_remain(self):
        q,_=np.linalg.qr(np.random.default_rng(81).normal(size=(12,5)))
        matrix=q[:,:1]
        derivative=q[:,1:5][:,None,:]
        fraction=surviving_source_fraction(matrix,derivative,np.ones(1),np.array([True]))
        self.assertAlmostEqual(fraction,1.,places=12)

    def test_zero_source_derivative_is_not_a_finite_information_claim(self):
        fraction=surviving_source_fraction(np.ones((8,1)),np.zeros((8,1,4)),
            np.ones(1),np.array([True]))
        self.assertIsNone(fraction)


if __name__=='__main__':
    unittest.main()
