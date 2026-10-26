import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paper_industrial_transport as transport


def fixture():
    return dict(longitude=np.array([117.,117.002,117.001]),
        latitude=np.array([39.,39.001,38.999]),wind_speed_mps=np.array([.2,.4]),
        wind_direction_deg=np.array([0.,90.]),normalized_concentration=np.zeros((2,3)),
        observation_mask=np.ones((2,3),dtype=bool),
        labels=np.array(['2026-05-03 12:00:00','2026-05-03 13:00:00']))


class IndustrialTransportTests(unittest.TestCase):
    def test_physical_units_hour_clock_and_coordinate_roundtrip(self):
        arrays=fixture()
        arrays['wind_speed_mps'][:]=2.
        context=transport.configure(arrays,dtype=torch.float64,samples=4)
        model=context.model
        np.testing.assert_allclose(model.recurrent_times,[0.,1.,2.])
        self.assertAlmostEqual(float(model.recurrent_v[0])*context.length_m,-7200.)
        self.assertAlmostEqual(float(model.recurrent_u[1])*context.length_m,-7200.)
        self.assertAlmostEqual(transport.normalize_diffusivity_m2s(100.,1.,1000.),.36)
        self.assertAlmostEqual(context.sigma_norm*context.length_m,100.)
        self.assertGreaterEqual(float((.75-np.abs(context.stations_norm).max())*context.length_m),999.99999)
        for index,point in enumerate(context.stations_norm):
            np.testing.assert_allclose(transport.source_lonlat(context,point),
                [arrays['longitude'][index],arrays['latitude'][index]],atol=1e-10,rtol=0.)

    def check_response_and_gradient(self,device,dtype):
        arrays=fixture()
        arrays['normalized_concentration'][0,1]=np.nan
        arrays['observation_mask'][0,1]=False
        context=transport.configure(arrays,device=device,dtype=dtype,samples=4)
        model=context.model
        callback=transport.HourlyResponse(context,order=64)
        with torch.no_grad():
            model.xs.fill_(-.1)
            model.ys.fill_(.05)
        a=torch.arange(1.,6.,dtype=dtype,device=device)/5
        b=torch.flip(a,[0])*.3
        def response(basis,grad=False):
            return callback(model,context.sigma_norm,context.xyt,basis,track_gradients=grad)
        matrix=response(context.basis,True)
        self.assertEqual(matrix.shape,(5,5))
        self.assertTrue(bool(torch.isfinite(matrix).all() and (matrix>=0).all()))
        r1=response((context.basis@a)[:,None])
        r2=response((context.basis@b)[:,None])
        summed=response((context.basis@(a+b))[:,None])
        tolerance=4e-5 if dtype==torch.float32 else 1e-10
        torch.testing.assert_close(r1+r2,summed,rtol=tolerance,atol=tolerance*.01)
        torch.testing.assert_close((matrix@a)[:,None],r1,rtol=tolerance,atol=tolerance*.01)
        gradients=torch.autograd.grad(matrix.square().sum(),
            [model.xs,model.ys,model.rawD_perp,model.rawD_parallel_delta])
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in gradients))
        reference=transport.numpy_hourly_reference(context,[-.1,.05],150.,50.,np.linspace(0.,2.,5),order=64)
        np.testing.assert_allclose(matrix.detach().cpu(),reference,rtol=tolerance*5,atol=tolerance*.1)
        return matrix.detach().cpu()

    def test_cpu_float64_reference_linearity_mask_and_gradients(self):
        torch.set_num_threads(1)
        self.check_response_and_gradient('cpu',torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cpu_cuda_float32_consistency(self):
        cpu=self.check_response_and_gradient('cpu',torch.float32)
        cuda=self.check_response_and_gradient('cuda',torch.float32)
        torch.testing.assert_close(cpu,cuda,rtol=1e-4,atol=1e-6)

    def test_zero_wind_constant_q_center_has_analytic_integral(self):
        arrays=fixture()
        arrays['wind_speed_mps'][:]=0.
        context=transport.configure(arrays,dtype=torch.float64,samples=4)
        model=context.model
        d=transport.normalize_diffusivity_m2s(20.,1.,context.length_m)
        transport.ia.set_model_anisotropic_diffusion_norm(model,d,d)
        with torch.no_grad():
            model.xs.zero_()
            model.ys.zero_()
        plan=transport.GaussianReleasePlan(model,context.sigma_norm,
            torch.tensor([[0.,0.,1.]],dtype=torch.float64),model.recurrent_integration_times,
            torch.ones((len(model.recurrent_integration_times),1),dtype=torch.float64),order=64)
        dp,dq=transport.ia.model_anisotropic_diffusion_norm(model)
        effective=float(((dp+dq)/2).detach())
        expected=transport.RECURRENT_SOURCE_SCALE*np.log1p(2*effective/context.sigma_norm**2)/(4*np.pi*effective)
        self.assertAlmostEqual(float(plan.evaluate(model).detach()),expected,places=10)

    def test_nonhourly_or_missing_wind_is_rejected(self):
        arrays=fixture()
        arrays['wind_speed_mps'][1]=np.nan
        with self.assertRaisesRegex(ValueError,'interpolation'):
            transport.configure(arrays)

    def check_forward_mode_information_jacobian(self,device,dtype):
        from run_paper_industrial_cases import raw_prediction_jacobian
        context=transport.configure(fixture(),device=device,dtype=dtype,samples=4)
        callback=transport.HourlyResponse(context,32)
        coefficients=torch.arange(1.,6.,device=device,dtype=dtype)/5
        prediction=callback(context.model,context.sigma_norm,context.xyt,context.basis,True)@coefficients
        reverse=transport.ia._prediction_jacobian_raw(context.model,prediction,batch_size=1)
        forward=raw_prediction_jacobian(context,callback,coefficients)
        tolerance=1e-9 if dtype==torch.float64 else 3e-4
        torch.testing.assert_close(forward,reverse,rtol=tolerance,atol=tolerance*.01)
        self.assertIsInstance(context.model.xs,torch.nn.Parameter)
        torch.testing.assert_close(raw_prediction_jacobian(context,callback,torch.zeros_like(coefficients)),
                                  torch.zeros_like(reverse))

    def test_forward_reverse_information_jacobian_cpu(self):
        self.check_forward_mode_information_jacobian('cpu',torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_forward_reverse_information_jacobian_cuda(self):
        self.check_forward_mode_information_jacobian('cuda',torch.float32)
        self.check_forward_mode_information_jacobian('cuda',torch.float64)

    def test_execution_batch_preserves_cohort_and_never_claims_partial_completion(self):
        import run_paper_industrial_cases as runner
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            protocol=dict(case_ids=['unit_a','unit_b','unit_c'])
            def execute(case,preparation,output,settings):
                target=output/case
                target.mkdir()
                (target/'result.json').write_text('{}',encoding='utf-8')
                return dict(whole_case_seconds=0.)
            with patch.object(runner,'freeze',return_value=protocol), \
                 patch.object(runner,'execute_case',side_effect=execute) as worker:
                runner.run('full','cpu',root,root,root,max_new_cases=1)
                self.assertFalse((root/'RUN_FINISHED.json').exists())
                state=json.loads((root/'RUN_PROGRESS.json').read_text())
                self.assertEqual(state['planned_cases'],3)
                self.assertEqual(state['completed_cases'],1)
                self.assertEqual(state['pending'],['unit_b','unit_c'])
                runner.run('full','cpu',root,root,root,max_new_cases=1)
                runner.run('full','cpu',root,root,root,max_new_cases=1)
                self.assertEqual(worker.call_count,3)
                self.assertTrue((root/'RUN_FINISHED.json').exists())


if __name__=='__main__':
    unittest.main()
