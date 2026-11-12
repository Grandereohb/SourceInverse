import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

from paper_release_support import release_support_models,MappedReleaseResponse,fit_release_support
from paper_joint_baselines import JointBounds


class ReleaseSupportTests(unittest.TestCase):
    def test_all_nonempty_supports_and_original_capacities_are_present(self):
        models=release_support_models(5)
        self.assertEqual(len(models),33)
        self.assertEqual(len(set(name for name,_ in models)),33)
        mapping=dict(models)
        np.testing.assert_array_equal(mapping['support_2'],np.eye(5)[:,[2]])
        np.testing.assert_array_equal(mapping['support_0_1_2_3_4'],np.eye(5))
        for _,matrix in models:
            self.assertTrue((matrix>=0).all())
            self.assertEqual(np.linalg.matrix_rank(matrix),matrix.shape[1])

    def test_shared_dictionary_mapping_preserves_response_and_gradient(self):
        matrix=torch.arange(20,dtype=torch.float64).reshape(4,5)
        basis=torch.eye(5,dtype=torch.float64)
        def response(model,sigma,xyt,q_basis,physical_parameters=None,**kwargs):
            torch.testing.assert_close(q_basis,basis)
            return matrix*physical_parameters
        mapping=torch.tensor(dict(release_support_models())['support_1_3'])
        wrapped=MappedReleaseResponse(response,basis,mapping)
        parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        value=wrapped(None,0.,None,basis@mapping,physical_parameters=parameter)
        torch.testing.assert_close(value,(matrix@mapping)*parameter)
        value.sum().backward()
        self.assertEqual(parameter.grad.item(),float((matrix@mapping).sum()))

    def test_fit_casts_basis_to_physical_dtype(self):
        objective=SimpleNamespace(basis=torch.eye(5,dtype=torch.float32),
            model=SimpleNamespace(xs=torch.zeros((),dtype=torch.float64)),
            observations=torch.zeros(20,dtype=torch.float64),response=lambda *args,**kwargs:None,
            xyt=None,sigma=0.,tensor=lambda value:torch.as_tensor(value,dtype=torch.float64),
            bounds=JointBounds(np.r_[[-1.,-1.,.01,0.],np.zeros(5)],np.r_[np.ones(4),np.full(5,10.)]))
        def fit(small,starts,budget):
            if small is not objective:
                self.assertEqual(small.basis.dtype,torch.float64)
            return dict(sse=1.,coefficients=np.ones(small.basis.shape[1]),prediction=np.zeros(20),
                source_norm=[0.,0.],diffusion_norm=[.2,.1],converged=True,seconds=0.)
        with patch('paper_release_support.fit_profiled_trf',side_effect=fit):
            result=fit_release_support(objective,[np.r_[np.zeros(2),[.1,.1],np.ones(5)]],refine_all=True)
        self.assertEqual(len(result['release_models']),33)
        self.assertEqual(result['refined_layouts'],[item['layout_id'] for item in result['release_models']])


if __name__=='__main__':
    unittest.main()
