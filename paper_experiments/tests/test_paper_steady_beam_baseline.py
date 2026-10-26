import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_steady_beam_baseline import steady_beam_response
from paper_physical_beam_adapter import ppm_per_kg_m3


class SteadyPhysicalBeamTests(unittest.TestCase):
    def test_calm_isotropic_point_source_green_function_and_reflection(self):
        # Known3D Laplace Green function; quadrature along a finite test segment.
        actual=steady_beam_response([[0.,0.]],[[10.,0.,1.6]],[[10.1,0.,1.6]],
            [0.,0.],.3,1.,1.,[288.15],[101325.],order=128,source_variance=0.)[0]
        expected=quad(lambda t:(1/np.sqrt((10+.1*t)**2+1.3**2)+
                               1/np.sqrt((10+.1*t)**2+1.9**2))/(4*np.pi),0.,1.)[0]
        self.assertAlmostEqual(actual/float(ppm_per_kg_m3(288.15,101325.))/expected,1.,places=9)

    def test_beam_reversal_and_infinite_history_quadrature(self):
        arguments=([[1.,-.2],[.7,.3]],[[10.,-2.,1.6]],[[12.,2.,1.6]],
                   [0.,0.],.3,3.,1.,[285.,288.],[101000.,101325.])
        coarse=steady_beam_response(*arguments,order=128)
        fine=steady_beam_response(*arguments,order=256)
        reverse=steady_beam_response(arguments[0],arguments[2],arguments[1],*arguments[3:],order=256)
        np.testing.assert_allclose(coarse,fine,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(fine,reverse,rtol=1e-10,atol=1e-8)
        self.assertTrue(np.isfinite(fine).all() and (fine>=0).all())


if __name__=='__main__':
    unittest.main()
