import unittest
import numpy as np
from scipy.optimize import nnls
from ia_robust_profile import solve_huber_amplitudes, profiled_value_gradient, huber_terms


class RobustProfileTests(unittest.TestCase):
    def test_gaussian_limit(self):
        rng=np.random.default_rng(731)
        matrix=rng.uniform(.1, 1., (30, 3))
        observed=matrix@np.array([1., 0., 3.])+rng.normal(0., .01, 30)
        fit=solve_huber_amplitudes(matrix, observed, .1, 100., delta=1e8)
        np.testing.assert_allclose(fit['coefficients'], nnls(matrix, observed)[0], atol=1e-7)
        self.assertTrue(fit['converged'])

    def test_envelope_gradient(self):
        x=np.linspace(-1., 1., 50)
        def response(theta, jac):
            value=np.column_stack((np.exp(theta[0]*x), 1.+x*x))
            derivative=np.zeros((len(x), 2, 1))
            derivative[:,0,0]=x*value[:,0]
            return (value, derivative) if jac else value
        observed=response([.8], False)@np.array([2., .3])
        observed[7]+=3.
        theta=np.array([.5])
        value,gradient,inner=profiled_value_gradient(response,theta,observed,.1,100.)
        step=1e-5
        finite=(profiled_value_gradient(response,theta+step,observed,.1,100.)[0]-
                profiled_value_gradient(response,theta-step,observed,.1,100.)[0])/(2*step)
        np.testing.assert_allclose(gradient[0], finite, rtol=2e-4, atol=1e-5)
        self.assertTrue(inner['converged'])

    def test_large_observations_are_not_discarded(self):
        rho,psi=huber_terms(np.array([1., 100., 1000.]))
        self.assertGreater(rho[2],rho[1])
        self.assertTrue(np.all(psi>0))

    def test_contaminated_amplitude(self):
        matrix=np.ones((31, 1))
        observed=np.full(31, 2.)
        observed[0]=100.
        fit=solve_huber_amplitudes(matrix,observed,.1,1000.)
        self.assertLess(abs(fit['coefficients'][0]-2.),.01)

    def test_envelope_with_active_amplitude_bound(self):
        x=np.linspace(.1, 1., 30)
        def response(theta,jac):
            value=np.exp(theta[0]*x)[:,None]
            derivative=(x*value[:,0])[:,None,None]
            return (value,derivative) if jac else value
        y=np.full(30,10.)
        theta=np.array([.1])
        value,gradient,inner=profiled_value_gradient(response,theta,y,.2,.5)
        self.assertAlmostEqual(inner['coefficients'][0],.5,places=8)
        h=1e-5
        finite=(profiled_value_gradient(response,theta+h,y,.2,.5)[0]-
                profiled_value_gradient(response,theta-h,y,.2,.5)[0])/(2*h)
        np.testing.assert_allclose(gradient[0],finite,rtol=1e-6)

    def test_zero_response_and_invalid_scale(self):
        matrix=np.zeros((10,2))
        result=solve_huber_amplitudes(matrix,np.ones(10),1.,10.)
        np.testing.assert_array_equal(result['coefficients'],[0.,0.])
        with self.assertRaises(ValueError):
            solve_huber_amplitudes(matrix,np.ones(10),0.,10.)


if __name__=='__main__':
    unittest.main()
