import unittest
import numpy as np
from ia_bias_reduction import second_order_ls_bias


class BiasReductionTests(unittest.TestCase):
    def test_quadratic_mean_matches_analytic_iid_bias(self):
        n,theta,sigma=10,2.,.3
        jac=np.full((n,1),2*theta)
        hessian=np.full((n,1,1),2.)
        fit=second_order_ls_bias(jac,hessian,np.eye(n)*sigma**2)
        self.assertTrue(fit['identified'])
        self.assertAlmostEqual(fit['bias'][0],-sigma**2/(8*n*theta**3),places=14)

    def test_correlated_quadratic_noise_uses_mean_variance(self):
        n,theta=8,1.3
        covariance=.2**2*.7**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
        fit=second_order_ls_bias(np.full((n,1),2*theta),np.full((n,1,1),2.),covariance)
        expected=-covariance.sum()/(8*n*n*theta**3)
        self.assertAlmostEqual(fit['bias'][0],expected,places=14)

    def test_general_correlated_formula_includes_score_term(self):
        # For scalar mean f_i(theta)=exp(x_i theta), verify the complete score
        # expansion directly, rather than dropping the correlation component.
        x=np.array([.1,.3,.7,1.2])
        value=np.exp(.4*x)
        jac=(x*value)[:,None]
        hessian=(x*x*value)[:,None,None]
        covariance=np.diag([.01,.03,.02,.06])
        result=second_order_ls_bias(jac,hessian,covariance)
        inverse=np.linalg.pinv(jac)
        variance=inverse@covariance@inverse.T
        cross=inverse@covariance-variance@jac.T
        first=np.sum(hessian[:,0,0]*cross[0])/float((jac.T@jac)[0,0])
        expected=first-.5*(inverse@(hessian[:,0,0]*variance[0,0]))[0]
        self.assertAlmostEqual(result['bias'][0],expected,places=13)
        self.assertGreater(abs(result['bias_correlation_component'][0]),1e-5)

    def test_linear_mean_has_zero_bias_and_operator_matches_matrix(self):
        rng=np.random.default_rng(71)
        jac=rng.normal(size=(20,3))
        hessian=np.zeros((20,3,3))
        covariance=np.diag(np.linspace(.1,.2,20))
        direct=second_order_ls_bias(jac,hessian,covariance)
        operator=second_order_ls_bias(jac,hessian,lambda values:covariance@values)
        np.testing.assert_allclose(direct['bias'],0.,atol=1e-15)
        np.testing.assert_allclose(direct['parameter_covariance'],operator['parameter_covariance'])

    def test_parameter_units_do_not_change_physical_bias(self):
        x=np.linspace(.1,1.,12)
        jac=(x*np.exp(.5*x))[:,None]
        hessian=(x*x*np.exp(.5*x))[:,None,None]
        a=second_order_ls_bias(jac,hessian,np.eye(len(x))*.04)
        b=second_order_ls_bias(jac*1000,hessian*1000**2,np.eye(len(x))*.04)
        np.testing.assert_allclose(a['bias'],b['bias']*1000,rtol=1e-12)

    def test_rank_deficiency_is_not_a_fake_correction(self):
        jac=np.ones((12,2))
        result=second_order_ls_bias(jac,np.zeros((12,2,2)),np.eye(12))
        self.assertFalse(result['identified'])
        self.assertIsNone(result['bias'])

    def test_small_noise_quadratic_bias_reduction_in_closed_form(self):
        # A vectorized scalar Monte Carlo, not a source-inversion experiment.
        variance=.4**2/8
        means=1.+np.sqrt(variance)*np.random.default_rng(81).standard_normal(200000)
        self.assertTrue(np.all(means>0))
        estimate=np.sqrt(means)
        corrected=estimate+variance/(8*estimate**3)
        self.assertLess(abs(corrected.mean()-1.),.2*abs(estimate.mean()-1.))


if __name__=='__main__':
    unittest.main()
