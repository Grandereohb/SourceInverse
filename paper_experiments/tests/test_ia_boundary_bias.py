import unittest
import numpy as np
from ia_boundary_bias import linearized_box_bias,correct_boundary_source


class BoundaryBiasTests(unittest.TestCase):
    def test_scalar_half_normal_matches_exact_mean(self):
        result=linearized_box_bias(np.ones((1,1)),np.ones((1,1)),[0.],[0.],[100.],samples=4096)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['bias'][0],1/np.sqrt(2*np.pi),delta=5e-4)

    def test_source_coupling_matches_one_active_bound_formula(self):
        jac=np.array([[1.,1.],[1.,0.]])
        sigma=.2
        information_inverse=np.linalg.inv(jac.T@jac)
        expected=information_inverse[:,1]/information_inverse[1,1]*sigma*np.sqrt(information_inverse[1,1])/np.sqrt(2*np.pi)
        result=linearized_box_bias(jac,np.eye(2)*sigma**2,[0.,0.],[-100.,0.],[100.,100.],samples=4096)
        np.testing.assert_allclose(result['bias'],expected,rtol=.004,atol=1e-5)

    def test_interior_antithetic_mean_is_exactly_zero(self):
        result=linearized_box_bias(np.eye(2),np.eye(2)*.01,[0.,0.],[-100.,-100.],[100.,100.],samples=256)
        np.testing.assert_allclose(result['bias'],0.,atol=1e-15)
        self.assertEqual(result['projected_fraction'],0.)

    def test_parameter_unit_changes_preserve_physical_answer(self):
        jac=np.array([[1.,1.],[1.,0.]])
        first=linearized_box_bias(jac,np.eye(2)*.04,[0.,0.],[-100.,0.],[100.,100.],samples=1024)
        scale=np.array([1000.,.01])
        second=linearized_box_bias(jac*scale,np.eye(2)*.04,[0.,0.],np.array([-100.,0.])/scale,
            np.array([100.,100.])/scale,samples=1024)
        np.testing.assert_allclose(first['bias'],second['bias']*scale,rtol=1e-9,atol=1e-12)

    def test_rank_deficient_mean_is_not_a_unique_correction(self):
        result=linearized_box_bias(np.ones((5,2)),np.eye(5),[0.,0.],[-1.,0.],[1.,1.],samples=32)
        self.assertFalse(result['available'])

    def test_fixed_geometry_q_refit_preserves_nonnegativity(self):
        def response(theta,jac):
            x,y=theta
            matrix=np.column_stack(([1+x,2-x,1+y,2-y],[1.,.5,2.,.2]))
            derivative=np.zeros((4,2,2))
            derivative[:,0,0]=[1.,-1.,0.,0.]
            derivative[:,0,1]=[0.,0.,1.,-1.]
            return (matrix,derivative) if jac else matrix
        physical=np.zeros(2)
        coefficients=np.array([1.,0.])
        observed=response(physical,False)@coefficients
        result=correct_boundary_source(response,observed,physical,coefficients,
            np.full(2,-.5),np.full(2,.5),10.,
            lambda r,j:(np.eye(4)*1e-4,dict(kind='known_noise')),
            samples=4096,seed=20261028)
        self.assertTrue(result['applied'])
        self.assertTrue(np.all(result['coefficients']>=0))
        self.assertLess(result['relative_mean_linearization_remainder'],1e-10)


if __name__=='__main__':
    unittest.main()
