import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from inversion_validation_common import diffusion_impulse_moments, set_anisotropic_diffusion
import field
from models.pinn import PINN


class SelectedAnisotropyTests(unittest.TestCase):
    def test_physical_parameter_recovery(self):
        model = PINN(hidden=8, depth=1)
        field.configure_recurrent_context(model,-1.,1.,-1.,1.,[0.,1.],[1.,1.],[0.,0.],
            d_scale_norm=.002,d_min_norm=.002,anisotropic_diffusion=False,nx=8,ny=8)
        set_anisotropic_diffusion(model,150.,15.)
        parallel,perp = model.anisotropic_D()
        self.assertAlmostEqual(float(parallel.detach())+1.,150.,places=4)
        self.assertAlmostEqual(float(perp.detach())+1.,15.,places=4)
        with self.assertRaises(ValueError):
            set_anisotropic_diffusion(model,10.,20.)

    def test_rotated_resolved_moments(self):
        result = diffusion_impulse_moments(40.,50.,100.,20.,5.,1.,1.)
        self.assertAlmostEqual(result["retained_mass"],1.,places=6)
        self.assertLess(abs(result["parallel_variance_ratio"]-1),.015)
        self.assertLess(abs(result["perp_variance_ratio"]-1),.015)
        self.assertLess(abs(result["cross_covariance_m2"]),1.)

    def test_equal_axes_match_scalar_and_calm_fallback(self):
        scalar = diffusion_impulse_moments(40.,40.,100.,20.,20.,1.,1.,"gaussian")
        rotated = diffusion_impulse_moments(40.,40.,100.,20.,20.,1.,1.)
        self.assertAlmostEqual(scalar["effective_parallel_m2s"],rotated["effective_parallel_m2s"],places=9)
        calm = diffusion_impulse_moments(40.,40.,100.,30.,10.,0.,0.)
        self.assertTrue(calm["calm_isotropic_fallback"])
        self.assertAlmostEqual(calm["effective_parallel_m2s"],scalar["effective_parallel_m2s"],places=9)

    def test_undersampling_detected_not_mistaken_for_physics(self):
        sampled = diffusion_impulse_moments(300.,300.,100.,1.,1.,1.,0.,"gaussian")
        implicit = diffusion_impulse_moments(300.,300.,100.,1.,1.,1.,0.,"implicit")
        self.assertLess(sampled["perp_variance_ratio"],.01)
        self.assertAlmostEqual(implicit["perp_variance_ratio"],1.,places=6)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_anisotropic_response_cpu_cuda_finite_and_linear(self):
        def response(device):
            model = PINN(hidden=8,depth=1).to(device)
            with patch.multiple(field,RECURRENT_ADAPTIVE_SUBSTEPS=False,
                                RECURRENT_SUBSTEPS=2,RECURRENT_MAX_SUBSTEPS=2):
                field.configure_recurrent_context(model,-1.,1.,-1.,1.,[0.,.5,1.],
                    [.1]*3,[.1]*3,d_scale_norm=.002,anisotropic_diffusion=True,
                    source_mode="cell_gaussian",source_sigma_norm=.1,nx=16,ny=16)
            with torch.no_grad():
                model.xs.fill_(0.);model.ys.fill_(0.)
            set_anisotropic_diffusion(model,20.,5.)
            q = torch.ones_like(model.recurrent_integration_times)
            a = field.recurrent_plume_fields_from_q(model,.1,q)
            b = field.recurrent_plume_fields_from_q(model,.1,q*2)
            torch.testing.assert_close(b,2*a)
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue((a>=0).all())
            return a.detach().cpu()
        torch.testing.assert_close(response("cpu"),response("cuda"),rtol=2e-5,atol=2e-6)
