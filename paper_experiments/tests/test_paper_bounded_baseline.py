import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_bounded_baseline import ORIGINAL_MEAN, roundoff_safe_mean_response


class BoundedBaselineTests(unittest.TestCase):
    def test_reproduced_one_ulp_violation_matches_isotropic_limit(self):
        dp = 0.041415426293130365
        dq = 0.04141542629313037
        args = ([[.3, .2]], [-.48044638, -.4475088])
        actual = roundoff_safe_mean_response(*args, dp, dq, 'steady')
        expected = ORIGINAL_MEAN(*args, dq, dq, 'steady')
        np.testing.assert_array_equal(actual, expected)

    def test_feasible_inputs_are_unchanged_and_real_violations_raise(self):
        args = ([[.3, .2]], [-.4, -.4])
        np.testing.assert_array_equal(roundoff_safe_mean_response(*args, .08, .03, 'steady'),
                                      ORIGINAL_MEAN(*args, .08, .03, 'steady'))
        with self.assertRaises(ValueError):
            roundoff_safe_mean_response(*args, .02, .03, 'steady')


if __name__ == '__main__':
    unittest.main()
