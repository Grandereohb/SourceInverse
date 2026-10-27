"""Memory-bounded exact Jacobian for joint inversion of long beam records."""

import numpy as np
import torch

from paper_joint_baselines import JointResidual


class BeamJointResidual(JointResidual):
    def jacobian(self, theta):
        self.bounds.validate(theta)
        self.jacobian_calls += 1
        if hasattr(self.response,'matrix_and_jacobian'):
            parameters=self.tensor(theta)
            matrix,derivative=self.response.matrix_and_jacobian(self.model,self.sigma,
                self.xyt,self.basis,physical_parameters=parameters[:4])
            self.matrix_calls+=1
            jac=torch.cat((torch.einsum('nkd,k->nd',derivative,parameters[4:]),matrix),dim=1)
            return jac.cpu().numpy().astype(float)
        # Forward AD batches parameter directions (5 or9), not hundreds of data rows.
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(self.tensor_residual,
                self.tensor(theta), vectorize=True, strategy='forward-mode')
        result = jac.detach().cpu().numpy().astype(float)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite physical beam Jacobian')
        return result
