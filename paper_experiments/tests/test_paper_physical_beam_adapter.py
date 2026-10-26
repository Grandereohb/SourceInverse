import os
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_physical_beam_adapter import (PhysicalBeamPlan, ppm_per_kg_m3,
    reflected_vertical_density, horizontal_gaussian_density,horizontal_gaussian_beam_mean)


class TinyModel(torch.nn.Module):
    def __init__(self,device='cpu',dtype=torch.float64):
        super().__init__()
        self.xs = torch.nn.Parameter(torch.tensor(.2,dtype=dtype,device=device))
        self.ys = torch.nn.Parameter(torch.tensor(.1,dtype=dtype,device=device))
        self.log_perp = torch.nn.Parameter(torch.tensor(np.log(.3),dtype=dtype,device=device))
        self.log_delta = torch.nn.Parameter(torch.tensor(np.log(.4),dtype=dtype,device=device))

    def anisotropic_D(self):
        transverse = torch.exp(self.log_perp)
        return transverse+torch.exp(self.log_delta), transverse


def small_plan(model, height=.3, order=8, basis=None,beam_solver='quadrature'):
    return PhysicalBeamPlan([0.,1.],[[.05,.02]],[[-1.,-.2,1.6]],[[2.,.2,1.6]],
        [288.15],[101325.],[0.,1.],np.eye(2) if basis is None else basis,height_m=height,
        beam_order=32,time_order=8,release_order=order,length_m=1.,time_unit_s=1.,
        device=model.xs.device,dtype=model.xs.dtype,beam_solver=beam_solver)


class PhysicalBeamAdapterTests(unittest.TestCase):
    def test_analytic_beam_matches_numerical_mean_and_tail_reversal(self):
        nodes,weights=np.polynomial.legendre.leggauss(512)
        clock=torch.tensor((nodes+1)*.5,dtype=torch.float64)
        weights=torch.tensor(weights*.5,dtype=torch.float64)
        for x,y in ((-.3,.1),(3.,.4),(-8.,.1)):
            dx=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            dy=torch.tensor(y,dtype=torch.float64)
            value=horizontal_gaussian_beam_mean(dx,dy,2.,.2,.7,.2,.4)
            numerical=(horizontal_gaussian_density(dx+2*clock,dy+.2*clock,.7,.2,.4)*weights).sum()
            torch.testing.assert_close(value,numerical,rtol=1e-10,atol=1e-16)
            reverse=horizontal_gaussian_beam_mean(dx+2.,dy+.2,-2.,-.2,.7,.2,.4)
            torch.testing.assert_close(value,reverse,rtol=1e-10,atol=1e-16)
            value.backward()
            self.assertTrue(torch.isfinite(dx.grad))
        model=TinyModel()
        analytic=small_plan(model,beam_solver='analytic').evaluate(model)
        numerical=small_plan(model).evaluate(model)
        torch.testing.assert_close(analytic,numerical,rtol=1e-9,atol=1e-10)
        left=torch.autograd.grad(analytic.sum(),tuple(model.parameters()))
        right=torch.autograd.grad(numerical.sum(),tuple(model.parameters()))
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-8)

    def test_reflected_halfspace_and_horizontal_mass(self):
        for height in (0.,.3,1.6):
            for age in (.1,1.,10.):
                actual = quad(lambda z: float(reflected_vertical_density(
                    torch.tensor(z,dtype=torch.float64),height,
                    torch.tensor(age,dtype=torch.float64),.3)),0.,np.inf,
                    epsabs=1e-10)[0]
                self.assertAlmostEqual(actual,1.,places=9)
        coordinate = torch.linspace(-8.,8.,801,dtype=torch.float64)
        x,y = torch.meshgrid(coordinate,coordinate,indexing='ij')
        density = horizontal_gaussian_density(x,y,.7,.2,.4)
        integral = torch.trapezoid(torch.trapezoid(density,coordinate,dim=1),coordinate)
        self.assertAlmostEqual(float(integral),1.,places=9)

    def test_ideal_gas_ppm_conversion(self):
        density = .01604246*101325./(8.31446261815324*288.15)
        self.assertAlmostEqual(float(ppm_per_kg_m3(288.15,101325.)*density),1e6,places=8)
        with self.assertRaises(ValueError):
            ppm_per_kg_m3(-1.,101325.)

    def check_values_gradients_and_linearity(self,device='cpu',dtype=torch.float64,beam_solver='quadrature'):
        model = TinyModel(device,dtype)
        response = small_plan(model,beam_solver=beam_solver).evaluate(model)
        self.assertTrue(bool(torch.isfinite(response).all()))
        self.assertTrue(bool((response>=0).all()))
        a = response.new_tensor([.001,.002])
        b = response.new_tensor([.003,.001])
        torch.testing.assert_close(response@(a+b),response@a+response@b)
        combined = torch.stack((a,b,a+b),dim=1)
        independently_integrated = small_plan(model,basis=combined.cpu().numpy(),beam_solver=beam_solver).evaluate(model)
        torch.testing.assert_close(independently_integrated,response@combined)
        response.sum().backward()
        for parameter in model.parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        if dtype==torch.float64:
            analytic = model.log_perp.grad.item()
            with torch.no_grad():
                model.log_perp.add_(1e-5)
                plus = small_plan(model,beam_solver=beam_solver).evaluate(model).sum().item()
                model.log_perp.sub_(2e-5)
                minus = small_plan(model,beam_solver=beam_solver).evaluate(model).sum().item()
            self.assertAlmostEqual((plus-minus)/2e-5/analytic,1.,places=7)

    def test_cpu_values_gradients_and_q_linearity(self):
        self.check_values_gradients_and_linearity()
        self.check_values_gradients_and_linearity(beam_solver='analytic')

    def test_sqrt_age_quadrature_at_equal_source_sensor_height(self):
        model = TinyModel()
        coarse = small_plan(model,height=1.6,order=8).evaluate(model)
        fine = small_plan(model,height=1.6,order=16).evaluate(model)
        torch.testing.assert_close(coarse,fine,rtol=1e-5,atol=1e-7)

    @unittest.skipUnless(os.environ.get('PAPER_TEST_PHYSICAL_CUDA')=='1' and torch.cuda.is_available(),
                         'CUDA test explicitly deferred until the timed evaluation is idle')
    def test_cuda_values_and_gradients(self):
        self.check_values_gradients_and_linearity('cuda',torch.float32)
        self.check_values_gradients_and_linearity('cuda',torch.float32,beam_solver='analytic')
        cpu = small_plan(TinyModel(dtype=torch.float32)).evaluate(TinyModel(dtype=torch.float32))
        gpu_model = TinyModel('cuda',torch.float32)
        gpu = small_plan(gpu_model).evaluate(gpu_model).cpu()
        torch.testing.assert_close(cpu,gpu,rtol=3e-5,atol=2e-6)


if __name__=='__main__':
    unittest.main()
