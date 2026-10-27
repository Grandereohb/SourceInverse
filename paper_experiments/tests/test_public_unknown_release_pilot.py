import unittest

import numpy as np

from run_public_unknown_release_pilot import mean_release, make_bounds_and_starts


class UnknownReleasePilotTests(unittest.TestCase):
    def test_time_mean_uses_integral_not_coefficient_mean(self):
        self.assertEqual(mean_release([2.]), 2.)
        self.assertEqual(mean_release([0., 2., 0.]), 1.)
        self.assertEqual(mean_release([0., 0., 4., 0., 0.]), 1.)

    def test_shared_initialization_does_not_use_truth_fields(self):
        record = dict(bounds=dict(lower=[0.,0.,.001,0.,0.], upper=[1.,1.,1.,1.,10.]),
            common_starts=[dict(source_x=.2, source_y=.3, diffusion_perp_norm=.01,
                diffusion_parallel_norm=.02, coefficients=[2.])],
            methods={'truth_or_prior_results': 'must not enter fitting'})
        bounds, starts = make_bounds_and_starts(record)
        np.testing.assert_array_equal(bounds.upper[4:], [10.]*5)
        np.testing.assert_array_equal(starts[0][4:], [2.]*5)
        self.assertEqual(len(starts[0]), 9)


if __name__ == '__main__':
    unittest.main()
