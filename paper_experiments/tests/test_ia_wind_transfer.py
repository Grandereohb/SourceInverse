import unittest
import numpy as np
from ia_wind_transfer import fit_shared_wind_transfer


class WindTransferTests(unittest.TestCase):
    def setUp(self):
        self.positions=np.array([[-30.,-10.],[25.,-15.],[5.,25.]])
        self.mean=np.random.default_rng(81).normal(size=(20,2))
        self.bx=np.array([[.002,-.001],[.003,-.002]])
        self.by=np.array([[-.001,.002],[.004,.001]])
        self.gradient=self.mean[:,0,None,None]*self.bx+self.mean[:,1,None,None]*self.by
        self.values=self.mean[:,None,:]+np.einsum('tij,sj->tsi',self.gradient,self.positions)

    def test_recovers_shared_flow_without_source_information(self):
        result=fit_shared_wind_transfer(self.positions,self.values)
        self.assertEqual(result['shared_parameter_rank'],6)
        np.testing.assert_allclose(result['bx_per_m'],self.bx,atol=1e-14)
        np.testing.assert_allclose(result['by_per_m'],self.by,atol=1e-14)
        np.testing.assert_allclose(result['gradient'],self.gradient,atol=1e-14)
        np.testing.assert_allclose(np.trace(result['gradient'],axis1=1,axis2=2),0.,atol=1e-14)

    def test_uniform_wind_limit(self):
        values=np.tile(self.mean[:,None,:],(1,3,1))
        result=fit_shared_wind_transfer(self.positions,values)
        np.testing.assert_allclose(result['gradient'],0.,atol=1e-14)
        np.testing.assert_allclose(result['fitted'],values,atol=1e-14)

    def test_translation_and_rotation_equivariance(self):
        angle=.7
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        shifted=self.positions@rotation.T+np.array([400.,-100.])
        values=self.values@rotation.T
        result=fit_shared_wind_transfer(shifted,values)
        np.testing.assert_allclose(result['gradient'],rotation@self.gradient@rotation.T,atol=1e-14)
        np.testing.assert_allclose(result['fitted'],values,atol=1e-13)

    def test_constant_direction_exposes_rank_loss(self):
        mean=np.column_stack((np.linspace(1.,3.,12),np.zeros(12)))
        gradient=mean[:,0,None,None]*self.bx
        values=mean[:,None,:]+np.einsum('tij,sj->tsi',gradient,self.positions)
        result=fit_shared_wind_transfer(self.positions,values)
        self.assertEqual(result['shared_parameter_rank'],3)
        self.assertEqual(result['measured_wind_component_rank'],1)
        np.testing.assert_allclose(result['gradient'],gradient,atol=1e-14)

    def test_rejects_collinear_stations(self):
        with self.assertRaises(ValueError):
            fit_shared_wind_transfer(np.array([[0.,0.],[1.,0.],[2.,0.]]),self.values)


if __name__=='__main__':
    unittest.main()
