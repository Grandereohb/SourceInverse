import unittest

import numpy as np
import torch

from paper_affine_wind import AffineWindHistory,affine_step
from paper_sqrt_affine_wind import SquareRootAffineWindHistory,step_factors,exact_step_factors
from paper_sqrt_gaussian import gaussian_from_factors,combine_diffusion_factors
from ia_metec_point_adapter import configure


class SquareRootWindTests(unittest.TestCase):
    def test_positive_step_quadrature_matches_exact_covariance(self):
        direction=np.array([.8,.6])
        for a in [np.zeros((2,2)),np.array([[.01,.005],[.002,-.01]])]:
            durations=np.array([.00001,1.,15.,60.])
            flow,shift,lp,lq=step_factors(a,[1.,.2],durations,direction)
            for axis,factor in [(direction,lp),(np.array([-.6,.8]),lq)]:
                expected=affine_step(a,[1.,.2],durations,np.outer(axis,axis))[2]
                np.testing.assert_allclose(factor@factor.swapaxes(-1,-2),expected,rtol=1e-10,atol=1e-11)

    def test_factor_transport_matches_regular_covariance_when_well_conditioned(self):
        edges=np.arange(5)*60.
        wind=np.array([[1.,.2]]*4)
        gradients=np.array([[[.003,.004],[.002,-.003]]]*4)
        regular=AffineWindHistory(edges,wind,gradients,wind)
        stable=SquareRootAffineWindHistory(edges,wind,gradients,wind)
        release=np.array([13.,71.,129.,167.])
        for actual,expected in zip(stable.relative(219.,release),regular.relative(219.,release)):
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)

    def test_exact_step_factor_matches_positive_quadrature(self):
        for a in [np.zeros((2,2)),np.array([[0.,.02],[0.,0.]]),
                  np.array([[0.,-.01],[.01,0.]]),np.array([[.03,.005],[.002,-.03]])]:
            for t in [0.,.00001,1.,15.,60.]:
                exact=exact_step_factors(a,[1.,.2],t,np.array([.8,.6]))
                reference=step_factors(a,[1.,.2],t,np.array([.8,.6]),32)
                for actual,expected in zip(exact[2:],reference[2:]):
                    np.testing.assert_allclose(actual@actual.T,expected@expected.T,rtol=1e-10,atol=1e-10)

    def test_stretched_rotated_covariance_has_finite_correct_density(self):
        rotation=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
        gradient=rotation@np.diag([.02,-.02])@rotation.T
        edges=np.arange(21)*60.
        stable=SquareRootAffineWindHistory(edges,np.zeros((20,2)),np.broadcast_to(gradient,(20,2,2)),
            np.broadcast_to([1.,0.],(20,2)))
        _,_,lp,lq=stable.relative_square_root(1200.,np.array([0.]))
        p,q=[torch.tensor(value[None],dtype=torch.float64) for value in (lp,lq)]
        zero=torch.zeros((1,1,1),dtype=torch.float64)
        density=gaussian_from_factors(zero,zero,p,q,torch.tensor(1.),torch.tensor(1.))
        variance1=np.expm1(48.)/.02
        variance2=-np.expm1(-48.)/.02
        expected=1/(2*np.pi*np.sqrt(variance1*variance2))
        self.assertTrue(torch.isfinite(density).all())
        self.assertAlmostEqual(float(density)/expected,1.,places=4)

    def test_covariance_scores_match_autodiff(self):
        lp=torch.tensor([[[[2.,0.],[.2,1.]]]],dtype=torch.float64)
        lq=torch.tensor([[[[.4,0.],[.1,3.]]]],dtype=torch.float64)
        dx=torch.tensor([[[1.]]],dtype=torch.float64)
        dy=torch.tensor([[[.5]]],dtype=torch.float64)
        theta=torch.tensor([1.,.5],dtype=torch.float64,requires_grad=True)
        def evaluate(p):
            return gaussian_from_factors(dx,dy,lp,lq,p[0]+p[1],p[0]).sum()
        derivative=torch.autograd.functional.jacobian(evaluate,theta)
        density,alpha,beta,iso,contrast=gaussian_from_factors(dx,dy,lp,lq,theta[0]+theta[1],theta[0],derivatives=True)
        torch.testing.assert_close(derivative,torch.stack([(density*iso).sum(),(density*contrast).sum()]),rtol=1e-10,atol=1e-10)

    def test_givens_reduction_matches_thin_qr(self):
        rng=np.random.default_rng(20261028)
        p=torch.tensor(np.tril(rng.normal(size=(200,2,2))),dtype=torch.float64)
        q=torch.tensor(np.tril(rng.normal(size=(200,2,2))),dtype=torch.float64)
        dp,dq=torch.tensor(3.,dtype=torch.float64),torch.tensor(.1,dtype=torch.float64)
        a,c,d=combine_diffusion_factors(p,q,dp,dq)
        lower=torch.stack((torch.stack((a,torch.zeros_like(a)),dim=-1),
            torch.stack((c,d),dim=-1)),dim=-2)
        columns=torch.cat((torch.sqrt(2*dp)*p,torch.sqrt(2*dq)*q),dim=-1)
        _,upper=torch.linalg.qr(columns.transpose(-1,-2),mode='reduced')
        expected=upper.transpose(-1,-2)@upper
        torch.testing.assert_close(lower@lower.transpose(-1,-2),expected,rtol=1e-11,atol=1e-11)

    def evaluate_response(self,device):
        data=dict(concentration=np.array([[0.,np.nan],[.1,.2],[.3,.2]]),
            wind_xy=np.array([[1.,.2],[.7,.5],[1.,-.1]]),
            stations_xy=np.array([[10.,0.],[20.,5.]]),station_heights=np.array([2.4,2.4]))
        gradient=np.array([[[.002,.003],[.001,-.002]]]*3)
        history=SquareRootAffineWindHistory(np.arange(4)*60.,data['wind_xy'],gradient,data['wind_xy'])
        m,x,b,r,y=configure(data,2.,device=device,transport_history=history)
        theta=m.xs.new_tensor([.01,-.02,.03,.02])
        matrix,jacobian=r.matrix_and_jacobian(m,0.,x,b,theta)
        def response(p):
            return r(m,0.,x,b,physical_parameters=p,track_gradients=True)
        expected=torch.autograd.functional.jacobian(response,theta,vectorize=True,strategy='forward-mode')
        torch.testing.assert_close(jacobian,expected,rtol=1e-8,atol=1e-8)
        regular=AffineWindHistory(np.arange(4)*60.,data['wind_xy'],gradient,data['wind_xy'])
        mo,xo,bo,ro,yo=configure(data,2.,device=device,transport_history=regular)
        previous,previous_jac=ro.matrix_and_jacobian(mo,0.,xo,bo,theta)
        torch.testing.assert_close(matrix,previous,rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(jacobian,previous_jac,rtol=1e-8,atol=1e-8)
        self.assertTrue(torch.isfinite(matrix).all())
        q=matrix.new_tensor([.001,.002,.003,.001,.002])
        torch.testing.assert_close(matrix@(q+q),matrix@q+matrix@q)
        return matrix.cpu()

    def test_response_equivalence_exact_jacobian_and_q_linearity(self):
        self.evaluate_response('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_response_cpu_cuda_agree(self):
        torch.testing.assert_close(self.evaluate_response('cuda'),self.evaluate_response('cpu'),rtol=1e-9,atol=1e-10)


if __name__=='__main__':
    unittest.main()
