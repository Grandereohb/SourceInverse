import unittest
from types import SimpleNamespace

import numpy as np
import torch

from ia_transport_capacity import fit_transport_capacity
from paper_location_capacity import capacity_response,fit_location_capacity
from paper_joint_baselines import JointBounds,JointResidual
from metec_point_adapter import configure


class IaTransportCapacityTests(unittest.TestCase):
    def test_isotropic_jacobian_matches_frozen_operator(self):
        data=dict(concentration=np.zeros((2,2)),wind_xy=np.array([[1.,.2],[.8,.4]]),
            stations_xy=np.array([[10.,0.],[20.,5.]]),station_heights=np.array([2.4,2.4]))
        model,xyt,basis,forward,y=configure(data,2.)
        bounds=JointBounds(np.r_[[-1.,-1.,.0006,0.],np.zeros(5)],
            np.r_[[1.,1.,.36,.36],np.ones(5)])
        objective=JointResidual(forward,model,0.,xyt,basis,y,bounds)
        response,mapping=capacity_response(objective,3,physical_dimension=3)
        theta=np.array([.01,-.02,.03])
        actual,jacobian=response(theta,True)
        def evaluate(t):
            full=torch.cat((t,t.new_zeros(1)))
            return forward(model,0.,xyt,basis,physical_parameters=full,track_gradients=True)@objective.tensor(mapping)
        expected=torch.autograd.functional.jacobian(evaluate,objective.tensor(theta),
            vectorize=True,strategy='forward-mode')
        np.testing.assert_allclose(actual,evaluate(objective.tensor(theta)).detach().numpy(),rtol=1e-10)
        np.testing.assert_allclose(jacobian,expected.detach().numpy(),rtol=1e-8,atol=1e-8)

    def test_nested_modes_keep_q_linear_and_correct_parameter_count(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(22)
        centers=torch.tensor(rng.normal(size=(25,5,4)),dtype=torch.float64)
        def response(model,sigma,xyt,basis,physical_parameters=None,**kwargs):
            return torch.exp(-.2*(physical_parameters-centers).square().sum(-1))
        model=SimpleNamespace(xs=torch.tensor(0.,dtype=torch.float64))
        basis=torch.eye(5,dtype=torch.float64)
        xyt=torch.zeros((25,3),dtype=torch.float64)
        truth=torch.tensor([.1,-.05,.1,0.],dtype=torch.float64)
        observed=response(model,0.,xyt,basis,truth)@torch.full((5,),.5,dtype=torch.float64)
        observed+=.001*torch.sin(torch.arange(25,dtype=torch.float64))
        bounds=JointBounds(np.r_[[-.3,-.3,.01,0.],np.zeros(5)],np.r_[np.full(4,.3),np.full(5,2.)])
        objective=JointResidual(response,model,0.,xyt,basis,observed,bounds)
        starts=[np.r_[[0.,0.,.1,.05],np.zeros(5)]]
        previous=fit_location_capacity(objective,starts,100.,counts=(1,),max_nfev=30)
        result=fit_transport_capacity(objective,starts,previous['release_models'],counts=(1,),max_nfev=30)
        self.assertEqual(len(result['models']),2)
        modes={m['diffusion_mode']:m for m in result['models']}
        self.assertEqual(modes['isotropic']['parameter_count'],4)
        self.assertEqual(modes['anisotropic']['parameter_count'],5)
        self.assertEqual(modes['isotropic']['diffusion_norm'][0],modes['isotropic']['diffusion_norm'][1])
        self.assertLessEqual(modes['anisotropic']['sse'],modes['isotropic']['sse']+1e-8)
        self.assertTrue(np.isfinite(result['prediction']).all())
        self.assertTrue((result['coefficients']>=0).all())


if __name__=='__main__':
    unittest.main()
