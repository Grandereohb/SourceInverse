import unittest
import numpy as np

from paper_correlated_capacity import residual_energy_dof,correlated_capacity_score
from q_capacity import gaussian_bic


class CorrelatedCapacityTests(unittest.TestCase):
    def test_independent_noise_recovers_original_capacity_ranking(self):
        n=60
        nu=residual_energy_dof(np.repeat(np.arange(20),3),np.tile(np.arange(3),20),np.zeros(3))
        self.assertEqual(nu,n)
        for k in [5,7,9]:
            self.assertAlmostEqual(correlated_capacity_score(.5,n,nu,k),gaussian_bic(.5,n,k))

    def test_perfect_station_correlation_leaves_one_energy_degree_per_station(self):
        nu=residual_energy_dof(np.repeat(np.arange(20),3),np.tile(np.arange(3),20),np.ones(3))
        self.assertEqual(nu,3.)

    def test_missing_minutes_use_real_gaps(self):
        times=np.array([0,1,4,8])
        correlation=.8**np.abs(times[:,None]-times[None,:])
        expected=16/np.sum(correlation**2)
        self.assertAlmostEqual(residual_energy_dof(times,np.zeros(4,dtype=int),[.8]),expected)


if __name__=='__main__':
    unittest.main()
