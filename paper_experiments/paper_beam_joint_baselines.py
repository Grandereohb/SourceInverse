"""Memory-bounded exact Jacobian for joint inversion of long beam records."""

import numpy as np
import torch

from paper_joint_baselines import JointResidual


class BeamJointResidual(JointResidual):
    def jacobian(self, theta):
        self.bounds.validate(theta)
        self.jacobian_calls += 1
        # Forward AD batches parameter directions (5 or9), not hundreds of data rows.
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(self.tensor_residual,
                self.tensor(theta), vectorize=True, strategy='forward-mode')
        result = jac.detach().cpu().numpy().astype(float)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite physical beam Jacobian')
        return result
