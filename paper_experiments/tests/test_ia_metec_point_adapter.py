import unittest
import numpy as np
import torch
from scipy.integrate import quad

from ia_metec_point_adapter import configure,hat_basis,xy_from_lon_lat,window_mean_release


class MetecPointTests(unittest.TestCase):
    def test_default_ia_response_matches_frozen_nls_operator(self):
        from metec_point_adapter import configure as frozen_configure
        current=configure(self.data(),2.)
        frozen=frozen_configure(self.data(),2.)
        theta=current[0].xs.new_tensor([.01,-.02,.03,.02])
        new=current[3].matrix_and_jacobian(current[0],0.,current[1],current[2],theta)
        old=frozen[3].matrix_and_jacobian(frozen[0],0.,frozen[1],frozen[2],theta)
        for actual,expected in zip(new,old):
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)

    def data(self):
        observations=np.array([[.2,np.nan],[-.1,.4],[.3,.5]])
        return dict(concentration=observations,wind_xy=np.array([[1.,.2]]*3),
            stations_xy=np.array([[10.,0.],[20.,4.]]),station_heights=np.array([2.4,2.4]))

    def evaluate(self,device):
        model,xyt,basis,response,observed=configure(self.data(),2.,device=device)
        theta=model.xs.new_tensor([0.,0.,.03,.02],requires_grad=True)
        matrix=response(model,0.,xyt,basis,physical_parameters=theta,track_gradients=True)
        self.assertEqual(matrix.shape,(5,5))
        self.assertEqual(observed[1],-.1)
        self.assertTrue(torch.isfinite(matrix).all())
        a=matrix.new_tensor([.001,.002,.003,.004,.002])
        torch.testing.assert_close(matrix@(a+a),matrix@a+matrix@a)
        matrix.sum().backward()
        self.assertTrue(torch.isfinite(theta.grad).all())
        return matrix.detach().cpu()

    def test_cpu_finite_mask_linearity_gradient(self):
        self.evaluate('cpu')

    def test_analytic_derivative_matches_autodiff(self):
        model,xyt,basis,response,_=configure(self.data(),2.)
        def evaluate(t):
            return response(model,0.,xyt,basis,physical_parameters=t,track_gradients=True)
        for parameters in [[.02,-.01,.03,.02],[.1,0.,.0006,.36],[0.,0.,.36,0.]]:
            theta=model.xs.new_tensor(parameters)
            value,derivative=response.matrix_and_jacobian(model,0.,xyt,basis,theta)
            torch.testing.assert_close(value,evaluate(theta),rtol=1e-10,atol=1e-10)
            expected=torch.autograd.functional.jacobian(evaluate,theta,vectorize=True,strategy='forward-mode')
            torch.testing.assert_close(derivative,expected,rtol=1e-8,atol=1e-8)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_matches_cpu(self):
        torch.testing.assert_close(self.evaluate('cuda'),self.evaluate('cpu'),rtol=1e-10,atol=1e-12)

    def test_hats_and_coordinates(self):
        values=hat_basis(np.linspace(0,10,51))
        np.testing.assert_allclose(values.sum(1),1.)
        self.assertTrue((values>=0).all())
        np.testing.assert_allclose(xy_from_lon_lat([100.],[40.],[100.,40.]),0.)
        self.assertAlmostEqual(window_mean_release([0.,1.,2.,3.,4.],60,[15,30]),1.5)
        self.assertAlmostEqual(window_mean_release([2.]*5,60,[1,59]),2.)

    def test_minute_batches_keep_full_release_and_wind_history(self):
        data=self.data()
        data['wind_xy']=np.array([[1.,.2],[.5,.7],[1.,-.3]])
        model,xyt,basis,response,observed=configure(data,2.)
        theta=model.xs.new_tensor([.01,-.02,.03,.02])
        full,derivative=response.matrix_and_jacobian(model,0.,xyt,basis,theta)
        matrices,jacobians,values=[],[],[]
        for query in [np.array([0]),np.array([1,2])]:
            m,x,b,r,y=configure(data,2.,observation_minutes=query)
            matrix,jacobian=r.matrix_and_jacobian(m,0.,x,b,theta)
            matrices.append(matrix)
            jacobians.append(jacobian)
            values.extend(y)
        torch.testing.assert_close(torch.cat(matrices),full,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(torch.cat(jacobians),derivative,rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(values,observed)

    def test_point_density_against_independent_time_integral(self):
        data=dict(concentration=np.zeros((1,1)),wind_xy=np.array([[1.,0.]]),
            stations_xy=np.array([[20.,0.]]),station_heights=np.array([2.4]))
        model,xyt,basis,response,_=configure(data,2.,time_order=32,release_order=32)
        theta=model.xs.new_tensor([0.,0.,.03,0.])
        actual=float((response(model,0.,xyt,basis,physical_parameters=theta) @ model.xs.new_full((5,),.001))[0])
        def density(age):
            if age==0:
                return 0.
            variance=10*age
            horizontal=np.exp(-(20-age)**2/(2*variance))/(2*np.pi*variance)
            vertical=(np.exp(-.4**2/(2*variance))+np.exp(-4.4**2/(2*variance)))/np.sqrt(2*np.pi*variance)
            return (60-age)/60*horizontal*vertical*.001*1.524e6
        reference=quad(density,0,60,epsabs=1e-9)[0]
        self.assertAlmostEqual(actual/reference,1.,places=6)
        data['release_window_s']=np.array([20.,40.])
        model,xyt,basis,response,_=configure(data,2.,time_order=64,release_order=32)
        actual=float((response(model,0.,xyt,basis,physical_parameters=theta) @ model.xs.new_full((5,),.001))[0])
        reference=quad(lambda age: density(age)*max(0.,min(40.,60-age)-20.)/max(60-age,1e-30),
            0,60,points=[20.,40.],epsabs=1e-9)[0]
        self.assertAlmostEqual(actual/reference,1.,places=3)


if __name__=='__main__':
    unittest.main()
