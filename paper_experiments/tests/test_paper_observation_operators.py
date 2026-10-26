import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.special import erf
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_observation_operators import beam_quadrature, beam_mean_response
from audit_paper_controlled_release_data import parse_measurement, parse_measurement_start_label


class BeamMeanTests(unittest.TestCase):
    def test_original_timestamp_padding_is_parsed_without_changing_dates(self):
        for text in ('05/09/2017  14:23:09.0000','05/09/2017  14:23: 9.0000'):
            actual=parse_measurement_start_label(text)
            self.assertEqual(actual.isoformat(),'2017-05-09T14:23:09')
        with self.assertRaises(ValueError):
            parse_measurement_start_label('02/30/2017 14:23:09.0000')

    def test_measurement_parser_new_and_legacy_headers(self):
        columns='Elapsedtime_sec Temp_K Press_Pa Humidity_ppmv WindX_m/s WindY_m/s WindZ_m/s Speedofsound_m/s CH4vmr_ppmv CH4vmrerror_ppmv'
        row='0 283 101000 7000 -2 1 0 340 2.1 0.01'
        for header in (columns,'Elapsed time   Temp   Press\nSeconds Kelvin Pa'):
            _,values,names=parse_measurement('Metadata\r\r\n'+header+'\r\r\n'+row)
            self.assertEqual(values.shape,(1,10))
            self.assertEqual(names[-2],'CH4vmr_ppmv')
            self.assertEqual(values[0,8],2.1)

    def test_measurement_parser_rejects_missing_or_incomplete_table(self):
        for text in ('No table','Elapsedtime_sec Temp_K\n0 283'):
            with self.assertRaises(ValueError):
                parse_measurement(text)

    def test_constant_is_a_mean_not_a_length_integral(self):
        points,weights=beam_quadrature([[0.,0.,1.6],[0.,0.,1.6]],[[100.,0.,1.6],[0.,200.,1.6]])
        result=beam_mean_response(np.ones((weights.size,2))*3.,weights)
        np.testing.assert_allclose(result,np.full((2,2),3.),atol=1e-14)
        np.testing.assert_allclose(points[:,:,2],1.6)

    def test_linear_field_and_beam_reversal(self):
        start,end=np.array([[1.,2.,3.]]),np.array([[8.,10.,7.]])
        expected=((start+end)*0.5)@np.array([1.,2.,3.])
        for a,b in ((start,end),(end,start)):
            points,weights=beam_quadrature(a,b,order=4)
            field=(points@np.array([1.,2.,3.])).reshape(-1,1)
            np.testing.assert_allclose(beam_mean_response(field,weights).ravel(),expected,atol=1e-13)

    def test_gaussian_line_mean_closed_form(self):
        sigma=0.15
        points,weights=beam_quadrature([[-1.,0.]],[[1.,0.]],order=64)
        field=np.exp(-points[:,:,0]**2/(2*sigma*sigma)).reshape(-1,1)
        exact=0.5*np.sqrt(2*np.pi)*sigma*erf(1/(np.sqrt(2)*sigma))
        self.assertAlmostEqual(float(beam_mean_response(field,weights)[0,0]),exact,places=12)

    def test_torch_linearity_and_gradient(self):
        _,weights=beam_quadrature([[0.,0.]],[[1.,1.]],order=8)
        response=torch.arange(24.,dtype=torch.float64).reshape(8,3).requires_grad_(True)
        a=torch.tensor([1.,2.,3.],dtype=torch.float64)
        torch.testing.assert_close(beam_mean_response(response@a[:,None],weights),
                                   beam_mean_response(response,weights)@a[:,None])
        beam_mean_response(response,weights).sum().backward()
        torch.testing.assert_close(response.grad,torch.as_tensor(weights.T).expand(-1,3))


if __name__=='__main__':
    unittest.main()
