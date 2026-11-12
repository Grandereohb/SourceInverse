import unittest
import numpy as np
import torch

from paper_observation_covariance import StationAR1Whitening,estimate_background_noise,CovarianceResponse


class ObservationCovarianceTests(unittest.TestCase):
    def test_irregular_whitening_matches_full_gaussian_covariance(self):
        times=np.array([0.,0.,1.,2.,3.,5.])
        stations=np.array([0,1,0,1,0,1])
        sigma=np.array([2.,.5])
        rho=np.array([.8,.4])
        covariance=np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                if stations[i]==stations[j]:
                    s=stations[i]
                    covariance[i,j]=sigma[s]**2*rho[s]**abs(times[i]-times[j])
        whitening=StationAR1Whitening(times,stations,sigma,rho)
        matrix=whitening.numpy(np.eye(6))
        np.testing.assert_allclose(matrix.T@matrix,np.linalg.inv(covariance),rtol=1e-12,atol=1e-12)

    def test_independent_equal_noise_preserves_objective(self):
        whitening=StationAR1Whitening([0,0,1,1],[0,1,0,1],[1,1],[0,0])
        values=np.arange(12.).reshape(4,3)
        np.testing.assert_array_equal(whitening.numpy(values),values)
        torch.testing.assert_close(whitening.tensor(torch.tensor(values)),torch.tensor(values))

    def test_response_and_derivative_remain_linear(self):
        whitening=StationAR1Whitening([0,1,3],[0,0,0],[2.],[.8])
        matrix=torch.tensor([[1.,0.],[.5,1.],[0.,2.]],dtype=torch.float64)
        def response(parameter):
            return matrix*parameter
        wrapped=CovarianceResponse(response,whitening)
        parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        actual=wrapped(parameter)
        expected=whitening.tensor(matrix)*parameter
        torch.testing.assert_close(actual,expected)
        actual.sum().backward()
        self.assertAlmostEqual(parameter.grad.item(),whitening.numpy(matrix.numpy()).sum())

    def test_calibration_uses_only_supplied_background(self):
        background=np.array([[1.,3.],[1.2,3.2],[1.1,3.1],[1.4,3.4],[1.3,3.3]])
        fit=estimate_background_noise(background)
        self.assertEqual(fit['background_counts'],[5,5])
        self.assertTrue(np.all(np.array(fit['station_sigma'])>0))
        self.assertTrue(np.all(np.array(fit['station_rho'])<1))


if __name__=='__main__':
    unittest.main()
