import sys
import unittest
from pathlib import Path

import torch
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pinn_source"))

from field import (  # noqa: E402
    configure_recurrent_context,
    recurrent_plume_fields,
    recurrent_plume_fields_from_q,
    recurrent_plume_value_from_q,
)
from config import INVERSION_MODE  # noqa: E402
from models.pinn import PINN  # noqa: E402
import field
from release_pulses import build_pulse_basis


class _LinearTransportModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.xs = torch.nn.Parameter(torch.tensor(-0.2, device=device))
        self.ys = torch.nn.Parameter(torch.tensor(0.1, device=device))
        self.raw_d = torch.nn.Parameter(torch.tensor(0.002, device=device))
        self.q_scale = torch.nn.Parameter(torch.tensor(1.25, device=device))
        self.q_call_count = 0

    def D(self):
        return self.raw_d

    def Q(self, t):
        self.q_call_count += 1
        return torch.ones_like(t) * self.q_scale

    def source_xy(self, t):
        return self.xs.expand_as(t), self.ys.expand_as(t)


def _configured_model(device="cpu"):
    model = _LinearTransportModel(device=device)
    configure_recurrent_context(
        model,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        t_values=[0.0, 0.5, 1.0, 1.5],
        u_values=[0.12, 0.08, -0.03, -0.03],
        v_values=[0.02, 0.06, 0.04, 0.04],
        d_min_norm=0.0,
        d_scale_norm=1.0,
        decay_norm=0.1,
        nx=25,
        ny=25,
    )
    return model


class ExternalQTransportTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_release_aware_transport_cpu_cuda_agree(self):
        results = []
        with patch.multiple(field, RECURRENT_ADAPTIVE_SUBSTEPS=False, RECURRENT_SUBSTEPS=2):
            for device in ("cpu", "cuda"):
                model = _LinearTransportModel(device)
                field.configure_recurrent_context(model, -1., 1., -1., 1.,
                    [0., 1., 2.], [.1, -.1, -.1], [.1, .2, .2],
                    anisotropic_diffusion=False, nx=21, ny=21,
                    release_breakpoints=[.6, .75, .9])
                q = build_pulse_basis(model.recurrent_integration_times, [.6], [.9])[:, 0]
                result = recurrent_plume_fields_from_q(model, .08, q)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(result.device.type, device)
                results.append(result.detach().cpu())
        torch.testing.assert_close(results[0], results[1], rtol=3e-5, atol=3e-6)

    def test_release_knots_preserve_wind_and_recover_source_mass(self):
        with patch.multiple(field, RECURRENT_ADAPTIVE_SUBSTEPS=False, RECURRENT_SUBSTEPS=2,
                            RECURRENT_INITIAL_RELEASE_FRACTION=0., RECURRENT_SOURCE_SCALE=1.):
            model = _LinearTransportModel()
            model.raw_d.data.zero_()
            field.configure_recurrent_context(model, -1., 1., -1., 1.,
                [0., 1., 2.], [0., .2], [0., -.1], anisotropic_diffusion=False,
                nx=21, ny=21, release_breakpoints=[.6, .75, .9])
            times = model.recurrent_times
            torch.testing.assert_close(model.recurrent_u[times < 1.], torch.zeros(5))
            torch.testing.assert_close(model.recurrent_u[times >= 1.], torch.full((3,), .2))
            self.assertTrue(all(s == 1 for s in model.recurrent_substeps_per_interval))
            self.assertAlmostEqual(model.recurrent_initial_release_dt, 1.)
            q = build_pulse_basis(model.recurrent_integration_times, [.6], [.9])[:, 0]
            fields = recurrent_plume_fields_from_q(model, .08, q)
            mass = fields[torch.searchsorted(times, torch.tensor(1.))].sum()*.1**2
            self.assertAlmostEqual(float(mass.detach()), .15, places=5)
            # Shared recurrence remains linear and differentiable in external Q.
            q.requires_grad_()
            result = recurrent_plume_fields_from_q(model, .08, q)
            result.sum().backward()
            self.assertTrue(torch.isfinite(q.grad).all())

    def test_already_aligned_release_knots_preserve_observation_predictions(self):
        with patch.multiple(field, RECURRENT_ADAPTIVE_SUBSTEPS=False, RECURRENT_SUBSTEPS=2):
            legacy = _LinearTransportModel()
            aware = _LinearTransportModel()
            for model, knots in ((legacy, None), (aware, [.25, .5, .75])):
                field.configure_recurrent_context(model, -1., 1., -1., 1.,
                    [0., .5, 1.], [.1, -.1, -.1], [0., .1, .1],
                    anisotropic_diffusion=False, nx=21, ny=21, release_breakpoints=knots)
            q = torch.linspace(.1, 1., len(legacy.recurrent_integration_times))
            torch.testing.assert_close(aware.recurrent_times, legacy.recurrent_times, rtol=0, atol=0)
            queries = torch.tensor([[0., 0., 0.], [0., 0., .5], [.1, .1, 1.]])
            a = recurrent_plume_value_from_q(legacy, queries, .08, q)
            b = recurrent_plume_value_from_q(aware, queries, .08, q)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_algorithm_selector_supports_user_config(self):
        self.assertIn(INVERSION_MODE, ("joint_adam", "ia_vpdi"))

    def test_external_q_matches_legacy_model_q_path(self):
        model = _configured_model()
        q_values = model.Q(model.recurrent_integration_times.view(-1, 1)).view(-1)
        model.q_call_count = 0

        legacy = recurrent_plume_fields(model, sigma_src=0.08)
        external = recurrent_plume_fields_from_q(model, 0.08, q_values)

        self.assertEqual(model.q_call_count, 1)
        self.assertTrue(torch.allclose(legacy, external, rtol=1e-6, atol=1e-7))

    def test_model_external_q_interpolates_without_trainable_q_values(self):
        model = PINN(hidden=8, depth=2)
        model.configure_external_q(
            t_values=[0.0, 0.5, 1.0],
            q_values=[0.0, 2.0, 4.0],
        )

        values = model.Q(torch.tensor([[0.25], [0.75]])).view(-1)

        self.assertTrue(torch.allclose(values, torch.tensor([1.0, 3.0])))
        self.assertEqual(model.q_mode, "external")
        self.assertNotIn("external_q_values", dict(model.named_parameters()))

    def test_transport_is_linear_in_external_q(self):
        model = _configured_model()
        n = model.recurrent_integration_times.numel()
        q1 = torch.linspace(0.1, 1.0, n)
        q2 = torch.linspace(0.8, 0.2, n)

        combined = recurrent_plume_fields_from_q(model, 0.08, q1 + q2)
        separate = recurrent_plume_fields_from_q(model, 0.08, q1)
        separate = separate + recurrent_plume_fields_from_q(model, 0.08, q2)

        self.assertTrue(torch.allclose(combined, separate, rtol=2e-5, atol=2e-6))

    def test_external_q_keeps_device_dtype_and_gradient(self):
        model = _configured_model()
        q_values = torch.ones(
            model.recurrent_integration_times.numel(), requires_grad=True
        )
        xyt = torch.tensor([[0.0, 0.0, 0.5], [0.4, 0.3, 1.5]])
        values = recurrent_plume_value_from_q(model, xyt, 0.08, q_values)
        values.sum().backward()

        self.assertEqual(values.dtype, model.xs.dtype)
        self.assertEqual(values.device, model.xs.device)
        self.assertTrue(torch.isfinite(values).all())
        self.assertIsNotNone(q_values.grad)
        self.assertTrue(torch.isfinite(q_values.grad).all())
        self.assertNotIn("q_integration", dict(model.named_parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_external_q_cuda_smoke(self):
        model = _configured_model(device="cuda")
        q_values = torch.ones(
            model.recurrent_integration_times.numel(), device="cuda"
        )
        fields = recurrent_plume_fields_from_q(model, 0.08, q_values)
        self.assertEqual(fields.device.type, "cuda")
        self.assertTrue(torch.isfinite(fields).all())


if __name__ == "__main__":
    unittest.main()
