import unittest
import numpy as np
import torch

from metec_point_adapter import configure
from paper_wind_ensemble import LocalWindEnsembleResponse


class WindEnsembleTests(unittest.TestCase):
    def responses(self,device='cpu',identical=False):
        data=dict(concentration=np.array([[.2,np.nan],[-.1,.4],[.3,.5]]),
            wind_xy=np.array([[1.,.2]]*3),stations_xy=np.array([[10.,0.],[20.,4.]]),
            station_heights=np.array([2.4,2.4]))
        model,xyt,basis,first,observed=configure(data,2.,device=device)
        second=configure(dict(data,wind_xy=data['wind_xy'] if identical else np.array([[.5,.7]]*3)),
            2.,device=device)[3]
        ensemble=LocalWindEnsembleResponse([first,second],[[-20.,0.],[20.,20.]])
        return model,xyt,basis,first,ensemble

    def test_equal_wind_reproduces_original_response(self):
        model,xyt,basis,first,ensemble=self.responses(identical=True)
        theta=model.xs.new_tensor([.01,-.03,.03,.02])
        value,jacobian=ensemble.matrix_and_jacobian(model,0.,xyt,basis,theta)
        old,old_jacobian=first.matrix_and_jacobian(model,0.,xyt,basis,theta)
        torch.testing.assert_close(value,old,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(jacobian,old_jacobian,rtol=1e-10,atol=1e-10)

    def evaluate(self,device):
        model,xyt,basis,_,ensemble=self.responses(device)
        theta=model.xs.new_tensor([.01,-.03,.03,.02])
        weights,derivative=ensemble.weights_and_derivative(theta)
        self.assertTrue(torch.all(weights>0))
        self.assertAlmostEqual(float(weights.sum()),1.)
        torch.testing.assert_close(derivative.sum(0),torch.zeros_like(theta[:2]),atol=1e-12,rtol=0)
        value,jacobian=ensemble.matrix_and_jacobian(model,0.,xyt,basis,theta)
        def response(p):
            return ensemble(model,0.,xyt,basis,physical_parameters=p,track_gradients=True)
        expected=torch.autograd.functional.jacobian(response,theta,vectorize=True,strategy='forward-mode')
        torch.testing.assert_close(jacobian,expected,rtol=1e-8,atol=1e-8)
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.all(value>=0))
        q=value.new_tensor([.001,.002,.003,.004,.005])
        torch.testing.assert_close(value@(q+q),value@q+value@q)
        return value.cpu()

    def test_analytic_gradient_and_q_linearity(self):
        self.evaluate('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_matches_cpu(self):
        torch.testing.assert_close(self.evaluate('cuda'),self.evaluate('cpu'),rtol=1e-10,atol=1e-10)


if __name__=='__main__':
    unittest.main()
