from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = ROOT / "pinn_source"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from models.pinn import PINN  # noqa: E402
from q_parameterization import configure_model_q  # noqa: E402


class ConstantQTests(unittest.TestCase):
    def test_constant_q_is_independent_of_time_and_trainable(self):
        model = PINN()
        info = configure_model_q(
            model,
            "constant",
            np.array([0.0, 0.5, 1.0]),
            6,
            torch.device("cpu"),
        )
        values = model.Q(torch.tensor([[0.0], [0.5], [1.0]]))
        self.assertEqual(info["mode"], "constant")
        self.assertTrue(torch.allclose(values, values[0].expand_as(values)))
        values.sum().backward()
        self.assertIsNotNone(model.logQ.grad)
        self.assertGreater(float(model.logQ.grad), 0.0)

    def test_constant_q_has_zero_shape_regularization(self):
        model = PINN()
        configure_model_q(
            model,
            "constant",
            np.array([0.0, 1.0]),
            6,
            torch.device("cpu"),
        )
        smooth, l2 = model.q_regularization()
        self.assertEqual(float(smooth.detach()), 0.0)
        self.assertEqual(float(l2.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
