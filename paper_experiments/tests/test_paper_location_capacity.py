import unittest

import numpy as np
from types import SimpleNamespace
import torch

from paper_location_capacity import location_variance,choose_location_capacity,fit_location_capacity
from paper_joint_baselines import JointBounds,JointResidual


class LocationCapacityTests(unittest.TestCase):
    def test_nuisance_compensation_cannot_create_zero_uncertainty(self):
        source=np.column_stack((np.arange(8),np.arange(8)**2)).astype(float)
        jac=np.column_stack((source,source[:,0]))
        result=location_variance(jac,1.,100.)
        self.assertLess(result['source_rank'],2)
        self.assertIsNone(result['variance_trace_m2'])
        both=location_variance(np.column_stack((source,source)),1.,100.)
        self.assertEqual(both['source_rank'],0)

    def test_source_units_and_noise_scale_covariance(self):
        rng=np.random.default_rng(10)
        jac=rng.normal(size=(20,4))
        a=location_variance(jac,1.,1.)
        b=location_variance(jac,4.,100.)
        np.testing.assert_allclose(b['covariance_m2'],40000*a['covariance_m2'])

    def test_bad_fit_is_not_selected_for_its_small_variance(self):
        models=[dict(sse=100.,q_basis_count=1,location_information=dict(variance_trace_m2=.1)),
            dict(sse=1.,q_basis_count=3,location_information=dict(variance_trace_m2=2.)),
            dict(sse=.99,q_basis_count=5,location_information=dict(variance_trace_m2=4.))]
        selected,variance,fallback=choose_location_capacity(models,100)
        self.assertEqual(selected['q_basis_count'],3)
        self.assertFalse(models[0]['fit_compatible'])
        self.assertFalse(fallback)

    def test_no_finite_location_information_keeps_explicit_warning(self):
        models=[dict(sse=1.,q_basis_count=1,location_information=dict(variance_trace_m2=None)),
            dict(sse=.99,q_basis_count=5,location_information=dict(variance_trace_m2=None))]
        selected,variance,fallback=choose_location_capacity(models,100)
        self.assertTrue(fallback)
        self.assertEqual(selected['q_basis_count'],5)

    def test_small_fit_preserves_q_bounds_and_nested_objective(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(42)
        centers=torch.tensor(rng.normal(size=(25,5,4)),dtype=torch.float64)
        def response(model,sigma,xyt,basis,physical_parameters=None,**kwargs):
            return torch.exp(-.2*(physical_parameters-centers).square().sum(-1))
        model=SimpleNamespace(xs=torch.tensor(0.,dtype=torch.float64))
        basis=torch.eye(5,dtype=torch.float64)
        xyt=torch.zeros((25,3),dtype=torch.float64)
        truth=torch.tensor([.1,-.05,.1,.1],dtype=torch.float64)
        y=response(model,0.,xyt,basis,truth)@torch.full((5,),.5,dtype=torch.float64)
        y+=.001*torch.sin(torch.arange(25,dtype=torch.float64))
        bounds=JointBounds(np.r_[[-.3,-.3,.01,0.],np.zeros(5)],np.r_[np.full(4,.3),np.full(5,2.)])
        objective=JointResidual(response,model,0.,xyt,basis,y,bounds)
        starts=[np.r_[[0.,0.,.1,.05],np.zeros(5)],np.r_[[-.1,.1,.2,.2],np.zeros(5)]]
        result=fit_location_capacity(objective,starts,100.,max_nfev=30)
        self.assertEqual(len(result['prediction']),25)
        self.assertTrue(np.isfinite(result['prediction']).all())
        self.assertTrue((result['coefficients']>=0).all())
        self.assertTrue((result['coefficients']<=2.).all())
        scores=[model['sse'] for model in result['release_models']]
        self.assertTrue(np.all(np.diff(scores)<=1e-10))


if __name__=='__main__':
    unittest.main()
