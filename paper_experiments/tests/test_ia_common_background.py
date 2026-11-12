import unittest
import numpy as np
from scipy.optimize import lsq_linear
from ia_common_background import TimeGroups,solve_q_background,CommonBackgroundResidual


class CommonBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(311)
        self.times=np.repeat(np.arange(8),5)
        self.groups=TimeGroups(self.times)
        self.matrix=self.rng.uniform(.1,1.,(40,3))

    def test_noiseless_nonnegative_q_and_arbitrary_common_background(self):
        a=np.array([.4,1.2,.8])
        bg=self.rng.uniform(1.,3.,8)
        y=self.matrix@a+bg[self.times]
        fit=solve_q_background(self.matrix,y,self.groups,5.)
        np.testing.assert_allclose(fit['coefficients'],a,atol=1e-10)
        np.testing.assert_allclose(fit['background'],bg,atol=1e-10)

    def test_matches_full_augmented_bounded_linear_solve(self):
        y=self.matrix@np.array([.4,1.2,0.])+np.array([0.,2.,0.,1.,0.,3.,0.,1.])[self.times]
        y+=self.rng.normal(0.,.01,40)
        fit=solve_q_background(self.matrix,y,self.groups,5.)
        design=np.column_stack((self.matrix,np.eye(8)[self.times]))
        full=lsq_linear(design,y,bounds=(0.,np.r_[np.full(3,5.),np.full(8,np.inf)]),method='bvls',tol=1e-12)
        np.testing.assert_allclose(fit['coefficients'],full.x[:3],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(fit['background'],full.x[3:],atol=1e-8)

    def test_positive_common_shift_does_not_change_source_strength(self):
        y=self.matrix@np.array([.4,1.2,.8])+2.
        first=solve_q_background(self.matrix,y,self.groups,5.)
        delta=np.linspace(.1,1.,8)
        second=solve_q_background(self.matrix,y+delta[self.times],self.groups,5.)
        np.testing.assert_allclose(first['coefficients'],second['coefficients'],atol=1e-10)

    def test_reduced_jacobian_with_background_elimination(self):
        features=self.rng.normal(size=(40,3,2))
        def response(theta,jac):
            value=np.exp(np.einsum('nkd,d->nk',features,theta))
            derivative=value[:,:,None]*features
            return (value,derivative) if jac else value
        y=response([.3,-.2],False)@np.array([.4,1.2,.8])+2.+self.rng.normal(0.,.01,40)
        objective=CommonBackgroundResidual(response,y,self.times,5.)
        theta=np.array([.2,-.15])
        jac=objective.jacobian(theta)
        finite=np.column_stack([(objective.residual(theta+np.eye(2)[j]*1e-6)-
            objective.residual(theta-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)])
        np.testing.assert_allclose(jac,finite,rtol=1e-5,atol=1e-7)

    def test_reduced_jacobian_with_zero_background_groups(self):
        features=self.rng.normal(size=(40,3,2))
        def response(theta,jac):
            value=np.exp(np.einsum('nkd,d->nk',features,theta))
            derivative=value[:,:,None]*features
            return (value,derivative) if jac else value
        background=np.array([0.,2.,0.,1.,0.,3.,0.,1.])
        y=response([.3,-.2],False)@np.array([.4,1.2,.8])+background[self.times]
        y+=self.rng.normal(0.,.01,40)
        objective=CommonBackgroundResidual(response,y,self.times,5.)
        theta=np.array([.2,-.15])
        jac=objective.jacobian(theta)
        self.assertTrue(np.any(~objective.solution['background_interior']))
        finite=np.column_stack([(objective.residual(theta+np.eye(2)[j]*1e-6)-
            objective.residual(theta-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)])
        np.testing.assert_allclose(jac,finite,rtol=1e-5,atol=1e-7)

    def test_missing_rows_do_not_invent_zero_observations(self):
        keep=np.arange(40)%7!=0
        a=np.array([.4,1.2,.8])
        y=self.matrix@a+2.
        fit=solve_q_background(self.matrix[keep],y[keep],TimeGroups(self.times[keep]),5.)
        np.testing.assert_allclose(fit['coefficients'],a,atol=1e-10)


if __name__=='__main__':
    unittest.main()
