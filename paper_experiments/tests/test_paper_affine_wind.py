import unittest

import numpy as np
from scipy.integrate import solve_ivp,quad_vec
from scipy.linalg import expm
import torch

from paper_affine_wind import affine_step,AffineWindHistory,fit_incompressible_wind
from ia_metec_point_adapter import configure


class AffineWindTests(unittest.TestCase):
    def test_exact_step_against_independent_matrix_exponential_and_integral(self):
        tensor=np.array([[.7,.2],[.2,.3]])
        velocity=np.array([1.,.2])
        for a in [np.zeros((2,2)),np.array([[0.,.02],[0.,0.]]),
                  np.array([[0.,-.01],[.01,0.]]),np.array([[.01,.003],[.002,-.01]])]:
            for t in [.0001,17.,60.]:
                flow,shift,cov=affine_step(a,velocity,t,tensor)
                augmented=np.zeros((3,3))
                augmented[:2,:2]=a
                augmented[:2,2]=velocity
                reference=expm(augmented*t)
                expected=quad_vec(lambda s:expm(a*s)@tensor@expm(a.T*s),0.,t,epsabs=1e-11)[0]
                np.testing.assert_allclose(flow,reference[:2,:2],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(shift,reference[:2,2],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(cov,expected,rtol=1e-9,atol=1e-11)
                self.assertAlmostEqual(np.linalg.det(flow),1.,places=10)
                self.assertTrue((np.linalg.eigvalsh(cov)>0).all())

    def test_relative_transport_matches_piecewise_moment_ode(self):
        edges=np.arange(4)*60.
        gradient=np.array([[[0.,.004],[.003,0.]],[[.002,0.],[0.,-.002]],[[0.,-.003],[.003,0.]]])
        velocity=np.array([[1.,.2],[.8,.4],[.7,-.2]])
        history=AffineWindHistory(edges,velocity,gradient,velocity)
        maps,shift,p,q=history.relative(171.,np.array([13.]))
        source=np.array([2.,-1.])
        state=np.r_[source,np.zeros(4)]
        for index,(left,right) in enumerate([(13.,60.),(60.,120.),(120.,171.)]):
            a,b=gradient[index],velocity[index]
            k=3*history.parallel[index]+(np.eye(2)-history.parallel[index])
            def ode(t,y):
                covariance=y[2:].reshape(2,2)
                return np.r_[a@y[:2]+b,(a@covariance+covariance@a.T+2*k).ravel()]
            state=solve_ivp(ode,[left,right],state,rtol=1e-11,atol=1e-11).y[:,-1]
        np.testing.assert_allclose(maps[0]@source+shift[0],state[:2],rtol=1e-9)
        expected=(2*(3*p+q))[0]
        np.testing.assert_allclose(expected,state[2:].reshape(2,2)[[0,0,1],[0,1,1]],rtol=1e-9)

    def data(self):
        return dict(concentration=np.array([[0.,np.nan],[.1,.2],[.3,.2]]),
            wind_xy=np.array([[1.,.2],[.7,.5],[1.,-.1]]),
            stations_xy=np.array([[10.,0.],[20.,5.]]),station_heights=np.array([2.4,2.4]))

    def test_constant_spatial_wind_reproduces_original_matrix_and_derivative(self):
        data=self.data()
        history=AffineWindHistory(np.arange(4)*60.,data['wind_xy'],np.zeros((3,2,2)),data['wind_xy'])
        old=configure(data,2.)
        new=configure(data,2.,transport_history=history)
        theta=old[0].xs.new_tensor([.01,-.02,.03,.02])
        for field in [0,1]:
            previous=old[3].matrix_and_jacobian(old[0],0.,old[1],old[2],theta)[field]
            actual=new[3].matrix_and_jacobian(new[0],0.,new[1],new[2],theta)[field]
            torch.testing.assert_close(actual,previous,rtol=1e-9,atol=1e-9)

    def evaluate(self,device):
        data=self.data()
        a=np.array([[[.002,.003],[.001,-.002]]]*3)
        history=AffineWindHistory(np.arange(4)*60.,data['wind_xy'],a,data['wind_xy'])
        m,x,b,r,y=configure(data,2.,device=device,transport_history=history)
        theta=m.xs.new_tensor([.01,-.02,.03,.02])
        matrix,jacobian=r.matrix_and_jacobian(m,0.,x,b,theta)
        def response(p):
            return r(m,0.,x,b,physical_parameters=p,track_gradients=True)
        expected=torch.autograd.functional.jacobian(response,theta,vectorize=True,strategy='forward-mode')
        torch.testing.assert_close(jacobian,expected,rtol=1e-8,atol=1e-8)
        self.assertTrue(torch.isfinite(matrix).all())
        self.assertTrue((matrix>=0).all())
        q=matrix.new_tensor([.001,.002,.003,.001,.002])
        torch.testing.assert_close(matrix@(q+q),matrix@q+matrix@q)
        return matrix.cpu()

    def test_affine_response_exact_jacobian_and_q_linearity(self):
        self.evaluate('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_affine_cpu_cuda_agree(self):
        torch.testing.assert_close(self.evaluate('cuda'),self.evaluate('cpu'),rtol=1e-9,atol=1e-10)

    def test_wind_calibration_uses_only_measurements(self):
        positions=np.array([[-20.,0.],[20.,10.],[0.,30.]])
        a=np.array([[.01,.02],[-.01,-.01]])
        b=np.array([1.,.2])
        velocities=np.broadcast_to(positions@a.T+b,(3,3,2)).copy()
        intercept,gradient,fitted=fit_incompressible_wind(positions,velocities)
        np.testing.assert_allclose(intercept,np.broadcast_to(b,(3,2)),atol=1e-12)
        np.testing.assert_allclose(gradient,np.broadcast_to(a,(3,2,2)),atol=1e-12)
        np.testing.assert_allclose(fitted,velocities,atol=1e-12)

    def test_rotation_preserves_isotropic_diffusion_and_gaussian_mass(self):
        a=np.array([[0.,-.01],[.01,0.]])
        flow,shift,integral=affine_step(a,[1.,.2],40.,np.eye(2))
        np.testing.assert_allclose(integral,40*np.eye(2),atol=1e-12)
        covariance=2*integral
        axis=np.linspace(-90.,90.,401)
        xx,yy=np.meshgrid(axis,axis)
        points=np.stack([xx,yy],axis=-1)
        exponent=np.einsum('...i,ij,...j->...',points,np.linalg.inv(covariance),points)
        density=np.exp(-.5*exponent)/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
        mass=np.trapezoid(np.trapezoid(density,axis,axis=0),axis)
        self.assertAlmostEqual(float(mass),1.,places=10)

    def test_population_evaluation_matches_individual_affine_response(self):
        data=self.data()
        history=AffineWindHistory(np.arange(4)*60.,data['wind_xy'],
            np.array([[[.002,.003],[.001,-.002]]]*3),data['wind_xy'])
        m,x,b,r,y=configure(data,2.,transport_history=history)
        theta=m.xs.new_tensor([[.01,-.02,.03,.02],[.02,.01,.04,.01]])
        actual=r.evaluate_population(m,0.,x,b,theta,batch_size=2)
        expected=torch.stack([r(m,0.,x,b,physical_parameters=p) for p in theta])
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


if __name__=='__main__':
    unittest.main()
