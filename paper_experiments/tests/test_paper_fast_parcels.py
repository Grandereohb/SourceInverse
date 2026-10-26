import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
import unittest

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper_experiments'))
import run_paper_parcel_pilot as pilot
from paper_fast_parcels import ParcelResponsePlan, CachedMeanResponse
from gaussian_parcels import gaussian_parcel_response


class CachedParcelTests(unittest.TestCase):
    def exercise(self,device,dtype):
        # A short context keeps this a unit test rather than a whole experiment.
        model=pilot.experiment.PINN(hidden=8,depth=1).to(device=device,dtype=dtype)
        pilot.experiment.field.configure_recurrent_context(model,-1.,1.,-1.,1.,
            [0.,.5,1.], [.3,0.,0.], [0.,.3,.3],d_scale_norm=.05,
            anisotropic_diffusion=True,d_parallel_min_norm=.001,d_perp_min_norm=.001,
            decay_norm=.1,source_sigma_norm=.15,nx=12,ny=12)
        pilot.experiment.ia.set_model_anisotropic_diffusion_norm(model,.08,.03)
        # Unequal numbers of queries/time and permuted rows test padding/order.
        xyt=torch.tensor([[.2,.1,.75],[.1,.05,0.],[.15,.1,.5],[.3,-.1,.75]],device=device,dtype=dtype)
        basis=pilot.experiment.vp.build_q_basis(model.recurrent_integration_times,3).to(dtype=dtype)
        plan=ParcelResponsePlan(model,.15,xyt,model.recurrent_integration_times,basis,4)
        actual=plan.evaluate(model)
        expected=gaussian_parcel_response(model,.15,xyt,model.recurrent_integration_times,
            basis,subdivisions=4,initial_release_fraction=0.)
        tolerance=2e-5 if dtype==torch.float32 else 1e-10
        torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance*0.01)
        params=[model.xs,model.ys,model.rawD_perp,model.rawD_parallel_delta]
        grads_a=torch.autograd.grad(actual.square().sum(),params)
        grads_b=torch.autograd.grad(expected.square().sum(),params)
        for a,b in zip(grads_a,grads_b):
            torch.testing.assert_close(a,b,rtol=tolerance*4,atol=tolerance*0.1)
        self.assertTrue(bool((actual>=0).all()))
        torch.testing.assert_close(plan.evaluate(model,max_kernel_elements=1),actual)
        with torch.no_grad():
            model.xs.add_(.04)
        changed=plan.evaluate(model)
        reference=gaussian_parcel_response(model,.15,xyt,model.recurrent_integration_times,
            basis,subdivisions=4,initial_release_fraction=0.)
        torch.testing.assert_close(changed,reference,rtol=tolerance,atol=tolerance*.01)

    def test_cpu_float64_values_gradients_order_and_reuse(self):
        torch.set_num_threads(1)
        self.exercise('cpu',torch.float64)

    def test_cpu_float32_values_gradients_order_and_reuse(self):
        self.exercise('cpu',torch.float32)

    def test_cache_rebuilds_for_modified_wind_and_basis(self):
        model=pilot.experiment.PINN(hidden=8,depth=1)
        pilot.experiment.field.configure_recurrent_context(model,-1.,1.,-1.,1.,
            [0.,.5,1.],[.3,0.,0.],[0.,.3,.3],d_scale_norm=.05,anisotropic_diffusion=True,
            d_parallel_min_norm=.001,d_perp_min_norm=.001,source_sigma_norm=.15,nx=12,ny=12)
        xyt=torch.tensor([[.2,.1,.5],[.3,-.1,1.]])
        basis=pilot.experiment.vp.build_q_basis(model.recurrent_integration_times,3)
        callback=CachedMeanResponse(samples=1,hours=2,stations=1,subdivisions=4)
        first=callback(model,.15,xyt,basis)
        torch.testing.assert_close(callback(model,.15,xyt,basis),first)
        self.assertEqual(len(callback._plans),1)
        model.recurrent_u.add_(.1)
        changed=callback(model,.15,xyt,basis)
        reference=gaussian_parcel_response(model,.15,xyt,model.recurrent_integration_times,
            basis,subdivisions=4,initial_release_fraction=0.)
        torch.testing.assert_close(changed,reference)
        self.assertEqual(len(callback._plans),2)
        torch.testing.assert_close(callback(model,.15,xyt,2*basis),2*changed)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_values_and_gradients(self):
        self.exercise('cuda',torch.float32)


if __name__=='__main__':
    unittest.main()
