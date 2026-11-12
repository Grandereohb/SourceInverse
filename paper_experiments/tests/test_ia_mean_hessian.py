import unittest
import numpy as np
from ia_bias_reduction import separable_mean_derivatives,correct_source_estimate,station_ar1_covariance_operator


class MeanHessianTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(82).normal(size=(30,2,3))
        self.theta=np.array([.1,-.2,.3])
        self.a=np.array([.7,1.2])

    def response(self,theta,jacobian):
        value=np.exp(np.einsum('nkd,d->nk',self.x,theta))
        derivative=value[:,:,None]*self.x
        return (value,derivative) if jacobian else value

    def test_full_mean_hessian_and_cross_amplitude_terms(self):
        mean,jac,hessian=separable_mean_derivatives(self.response,self.theta,self.a,np.full(3,1e-5))
        value=self.response(self.theta,False)
        expected=np.einsum('nki,nkj,nk,k->nij',self.x,self.x,value,self.a)
        np.testing.assert_allclose(hessian[:,:3,:3],expected,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(hessian[:,:3,3:],(value[:,:,None]*self.x).swapaxes(1,2))
        np.testing.assert_allclose(hessian[:,3:,3:],0.)
        np.testing.assert_allclose(mean,value@self.a)

    def test_active_amplitude_is_not_silently_corrected(self):
        def noise(*args):
            raise AssertionError('No derivative/noise computation for a rejected boundary estimate')
        result=correct_source_estimate(self.response,np.ones(30),self.theta,self.a,
            np.full(3,-1.),np.ones(3),10.,noise,active_mask=[-1,0])
        self.assertFalse(result['applied'])

    def test_zero_noise_preserves_estimate_and_q(self):
        y=self.response(self.theta,False)@self.a
        result=correct_source_estimate(self.response,y,self.theta,self.a,
            np.full(3,-1.),np.ones(3),10.,lambda r,j:(np.zeros((30,30)),dict(kind='zero_noise')))
        self.assertTrue(result['applied'])
        np.testing.assert_allclose(result['physical'],self.theta)
        np.testing.assert_allclose(result['coefficients'],self.a,atol=1e-10)

    def test_irregular_station_covariance_matches_dense_matrix(self):
        times=np.array([0,0,1,2,3])
        stations=np.array([0,1,0,1,0])
        sigma=np.array([.2,.3])
        rho=np.array([.7,.4])
        dense=np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                if stations[i]==stations[j]:
                    dense[i,j]=sigma[stations[i]]**2*rho[stations[i]]**abs(times[i]-times[j])
        operator=station_ar1_covariance_operator(times,stations,sigma,rho)
        operand=np.random.default_rng(31).normal(size=(5,3))
        np.testing.assert_allclose(operator(operand),dense@operand)
        self.assertGreater(np.linalg.eigvalsh(dense).min(),0.)


if __name__=='__main__':
    unittest.main()
