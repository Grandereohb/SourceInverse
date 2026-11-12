import unittest
import numpy as np
import torch
from scipy.integrate import quad
from ia_metec_point_adapter import configure


class VerticalDiffusionTests(unittest.TestCase):
    def evaluate(self,device):
        data=dict(concentration=np.zeros((2,2)),wind_xy=np.array([[1.,.2],[.5,.6]]),
            stations_xy=np.array([[10.,0.],[20.,4.]]),station_heights=np.array([2.4,3.]))
        model,xyt,basis,response,_=configure(data,2.,device=device)
        theta=model.xs.new_tensor([.01,-.02,.03,.02,.015])
        value,derivative=response.matrix_and_jacobian(model,0.,xyt,basis,theta)
        for j in range(5):
            step=torch.zeros_like(theta)
            step[j]=1e-6
            plus=response.matrix_and_jacobian(model,0.,xyt,basis,theta+step)[0]
            minus=response.matrix_and_jacobian(model,0.,xyt,basis,theta-step)[0]
            torch.testing.assert_close(derivative[...,j],(plus-minus)/2e-6,rtol=1e-6,atol=1e-7)
        tied=theta.clone()
        tied[4]=tied[2]
        a,da=response.matrix_and_jacobian(model,0.,xyt,basis,tied)
        b,db=response.matrix_and_jacobian(model,0.,xyt,basis,tied[:4])
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(da[...,2]+da[...,4],db[...,2])
        q=value.new_tensor([.001,.002,.003,.004,.005])
        torch.testing.assert_close(value@(q+q),value@q+value@q)
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        return value.cpu()

    def test_cpu_derivatives_tied_limit_and_q_linearity(self):
        self.evaluate('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_consistency(self):
        torch.testing.assert_close(self.evaluate('cuda'),self.evaluate('cpu'),rtol=1e-10,atol=1e-10)

    def test_separate_vertical_variance_independent_integral(self):
        data=dict(concentration=np.zeros((1,1)),wind_xy=np.array([[1.,0.]]),
            stations_xy=np.array([[20.,0.]]),station_heights=np.array([2.4]))
        model,xyt,basis,response,_=configure(data,2.,time_order=32,release_order=32)
        theta=model.xs.new_tensor([0.,0.,.03,0.,.018])
        matrix=response.matrix_and_jacobian(model,0.,xyt,basis,theta)[0]
        actual=float((matrix@model.xs.new_full((5,),.001))[0])
        def kernel(age):
            if age==0:
                return 0.
            horizontal=np.exp(-(20-age)**2/(20*age))/(20*np.pi*age)
            vertical=(np.exp(-.4**2/(12*age))+np.exp(-4.4**2/(12*age)))/np.sqrt(12*np.pi*age)
            return (60-age)/60*horizontal*vertical*.001*1.524e6
        expected=quad(kernel,0,60,epsabs=1e-10)[0]
        self.assertAlmostEqual(actual/expected,1.,places=6)


if __name__=='__main__':
    unittest.main()
