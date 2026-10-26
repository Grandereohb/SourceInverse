import sys
from pathlib import Path
import unittest

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_probabilistic_demo import (ConstantPuffPlan,collapsed_rate,
    positive_normal_mean,rank_split_rhat,CollapsedPosterior,BatchedPosterior)
from paper_baseline_demos import mean_response


class ProbabilisticDemoTests(unittest.TestCase):
    def test_batched_target_matches_scalar_on_cpu_and_available_gpu(self):
        plan=ConstantPuffPlan([[.2,.1],[.8,.5],[1.,-.3]],'veering',.15)
        observed=plan.response([-.2,.1,np.log(.02),np.log(.04)])*1.2
        posterior=CollapsedPosterior(plan,observed)
        points=np.array([[-.2,.1,np.log(.02),np.log(.04)],
                         [.5,-.4,np.log(.1),np.log(.01)]])
        reference=np.asarray([posterior(v) for v in points])
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            batched=BatchedPosterior(posterior,device)
            np.testing.assert_allclose(batched(points),reference,rtol=1e-11,atol=1e-8)
            self.assertEqual(batched([[1.,0.,0.,0.]])[0][0],-np.inf)

    def test_cached_constant_response_matches_independent_integrator(self):
        stations=np.array([[.2,.1],[.8,.5],[1.,-.3]])
        theta=np.array([-.2,.1,np.log(.02),np.log(.04)])
        for wind in ('steady','veering'):
            plan=ConstantPuffPlan(stations,wind,.15)
            reference=mean_response(stations,theta[:2],.06100005,.02100005,
                                    wind,q_basis_count=1).ravel()
            np.testing.assert_allclose(plan.response(theta),reference,rtol=2e-12,atol=1e-14)

    def test_collapsed_q_integral_including_normalization(self):
        g=np.array([.3,.1,.2]);y=np.array([.1,.2,-.1]);sigma=.4;tau=2.
        value,m,s=collapsed_rate(g,y,sigma,tau)
        integral=quad(lambda q: np.exp(-.5*np.sum(((g*q-y)/sigma)**2))
            *np.sqrt(2/np.pi)/tau*np.exp(-.5*(q/tau)**2),0.,np.inf)[0]
        self.assertAlmostEqual(value-.5*float(y@y)/sigma**2,np.log(integral),places=10)
        for scale in (1.,5.,10.):
            self.assertAlmostEqual(collapsed_rate(np.zeros(3),y,sigma,scale)[0],0.,places=12)
        self.assertGreater(positive_normal_mean(m,s),0.)

    def test_rank_split_rhat_detects_shift_and_scale_mismatch(self):
        rng=np.random.default_rng(18271)
        iid=rng.normal(size=(4,3000,2))
        self.assertLess(rank_split_rhat(iid).max(),1.02)
        shifted=iid.copy();shifted[0,:,0]+=3.
        self.assertGreater(rank_split_rhat(shifted)[0],1.1)
        scaled=iid.copy();scaled[0,:,0]*=5.
        self.assertGreater(rank_split_rhat(scaled)[0],1.1)

    def test_supplement_plot_rejects_partial_cohort(self):
        import json
        import tempfile
        from plot_paper_constant_q import plot
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'analysis_status.json').write_text(json.dumps(
                dict(final=False,completed_cases=2)),encoding='utf-8')
            (root/'case_metrics.csv').write_text('case_id,method\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'complete eight-case'):
                plot(root)


if __name__=='__main__':
    unittest.main()
